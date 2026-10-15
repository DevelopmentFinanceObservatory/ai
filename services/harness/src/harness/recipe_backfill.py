"""Stores in dfo-apis the recipe of every derived frame the materialisation store
holds. Run as `python -m harness.recipe_backfill`.

A derived frame published to dfo-apis carries its recipe (`common.recipe`) with
the publish. A frame published before that was the case has its recipe only in the
materialisation bucket, in its own trace and its ancestors' traces. This job
rebuilds each recipe from the bucket and writes it through
`PUT /api/datasets/frames/{spec_id}/{content_version}/recipe`, so the frame can be
replayed after the bucket's objects are deleted.

The walk is `MaterialisationStore.list_frames`, oldest first, keeping the derived
(`ops_`) frames. For each one:

1. `common.recipe.build_recipe` reads the frame's trace and every derived
   ancestor's, and each root's mint record. A root with no record gets one from
   `data_mcp.mint_recovery.recover_mint_record` over dfo-apis' versions route. A
   frame with a missing ancestor, an unreadable trace or an unrecoverable root has
   no recipe built, and nothing partial is sent. dfo-apis is asked whether it
   already holds a recipe for it.
2. With `--write`, the recipe is put to dfo-apis, which stores it, answers that
   the pair already holds it, refuses it as a conflict, or answers that the pair
   was never published. Without `--write` the job reads the stored recipe and
   compares the two hashes, and writes nothing to dfo-apis or to the store.

A recovered mint record is kept for the rest of the run, and with `--write` it is
written to the store, so later frames and later runs read it.

Every frame ends with one `Reason`. The report lists each frame with its reason
and closes with the counts an operator reads before restoring the bucket's age
rule. A run is safe to repeat: dfo-apis keeps a pair's first recipe, so a second
`--write` run finds every stored recipe already held.

The job reads traces and mint records. `get_trace` on the bucket-backed store
downloads the frame's whole object, at most its 64 MiB encoded-bytes ceiling,
checks its digest and reads the trace from the Parquet footer. No column is
decoded, so no rows are claimed from the frame row budget. Each derived frame's
trace is kept for the run once read, so an ancestor is downloaded once however
many frames were built on it. `FRAMES_IN_FLIGHT` frames are processed at a time.
A root with no mint record is the one frame decoded: recovery reads it in full to
count its StatVars, under the store's row ceiling, one root at a time and once per
run. The job writes no frame object and deletes nothing.

Settings come from the environment or `.env`, under the names the services use:
`DFO_APIS_URL`, `DATASETS_FIRST_PARTY_PRINCIPAL` and `FRAME_STORE_BUCKET`.

Exit status: 0 when every frame reached a settled reason, 1 when a frame ended
`apis_failure` or `unexpected_error` and a rerun may settle it, 2 when a setting
is missing or dfo-apis refused the first call.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

# Registers the codec a derived frame's stored trace decodes through.
import analysis_mcp.trace  # noqa: F401
from common.apis_auth import apis_auth_provider
from common.live_apis import PUBLISH_CONFLICT, LiveApisClient, LiveApisError
from common.materialisation import (
    AnchorRecord,
    FrameListing,
    FrameRef,
    MaterialisationStore,
    MintRecord,
    MintRecordUnrecoverableError,
    StoredFrameTrace,
    build_store,
)
from common.recipe import (
    RecipeAncestorMissingError,
    RecipeAnchorError,
    RecipeRootUnrecoverableError,
    RecipeTraceUnusableError,
    build_recipe,
)
from data_mcp.mint_recovery import recover_mint_record
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from common.materialisation import HeldFrame
    from dfo_frame_grammar import Frame

logger = logging.getLogger(__name__)

# Frames processed at once. Each holds at most one downloaded frame object, so
# this bounds the job's memory at four times the store's encoded-bytes ceiling.
FRAMES_IN_FLIGHT = 4

_DERIVED_SPEC_ID_PREFIX = "ops_"
_SNAPSHOT_NOT_FOUND = "snapshot_not_found"
_HASH_CONFLICT = "recipe_requires_matching_normal_form_hash"
_REPLAY_KEY_CONFLICT = "recipe_requires_matching_replay_key"

# A dataset id no dataset has. dfo-apis answers its versions with an empty list,
# so reading them tests the URL and the principal and nothing else.
_PREFLIGHT_SPEC_ID = "spec_" + "0" * 32


class Reason(StrEnum):
    """How one derived frame ended."""

    STORED = "stored"
    ALREADY_STORED = "already_stored"
    WOULD_STORE = "would_store"
    NOT_PUBLISHED = "not_published"
    MISSING_ANCESTOR = "missing_ancestor"
    ROOT_UNRECOVERABLE = "root_unrecoverable"
    TRACE_UNUSABLE = "trace_unusable"
    ANCHOR_UNRESOLVED = "anchor_unresolved"
    CONFLICT = "conflict"
    APIS_FAILURE = "apis_failure"
    UNEXPECTED_ERROR = "unexpected_error"


# dfo-apis holds a recipe for the frame.
_HELD = (Reason.STORED, Reason.ALREADY_STORED)
# A rerun may end the frame differently.
_UNSETTLED = (Reason.APIS_FAILURE, Reason.UNEXPECTED_ERROR)
# No recipe could be built, so dfo-apis was not asked whether the frame is published.
_UNBUILT = (
    Reason.MISSING_ANCESTOR,
    Reason.ROOT_UNRECOVERABLE,
    Reason.TRACE_UNUSABLE,
    Reason.ANCHOR_UNRESOLVED,
)


@dataclass(frozen=True, kw_only=True)
class FrameOutcome:
    """One derived frame's result. `detail` names what `reason` leaves open: the
    missing pair, the root and why it is unrecoverable, a conflict's constraint id,
    or a failed call's status and code."""

    ref: FrameRef
    created_at: datetime
    reason: Reason
    detail: str | None = None


@dataclass(frozen=True, kw_only=True)
class BackfillReport:
    """What one run found. `outcomes` is in walk order, oldest frame first."""

    write: bool
    at: datetime
    frames_held: int
    derived_held: int
    oldest_created_at: datetime | None
    outcomes: tuple[FrameOutcome, ...]

    def counts(self) -> Counter[Reason]:
        return Counter(outcome.reason for outcome in self.outcomes)

    @property
    def settled(self) -> bool:
        """Whether no frame ended on a failure a rerun may clear."""
        return not any(outcome.reason in _UNSETTLED for outcome in self.outcomes)

    def render(self) -> str:
        """The report as text: a line per frame, then the counts."""
        counts = self.counts()
        held = sum(counts[reason] for reason in _HELD)
        ready = counts[Reason.WOULD_STORE]
        unbuilt = sum(counts[reason] for reason in _UNBUILT)
        without = len(self.outcomes) - held - ready - counts[Reason.NOT_PUBLISHED]
        lines = [
            f"Recipe backfill, {'write' if self.write else 'dry run'}, {_stamp(self.at)}",
            f"Frames held: {self.frames_held}, of which derived: {self.derived_held}",
            f"Derived frames in this run: {len(self.outcomes)}",
            "",
            *(_frame_line(outcome) for outcome in self.outcomes),
            "",
            "Outcomes",
            *(f"  {reason.value:<19}{counts[reason]}" for reason in Reason if counts[reason]),
            "",
            f"Derived frames with a recipe stored in dfo-apis: {held}"
            f" (stored {counts[Reason.STORED]},"
            f" already stored {counts[Reason.ALREADY_STORED]})",
        ]
        if not self.write:
            lines.append(
                f"Recipes built and held back by the dry run: {ready}"
                " (--write stores each one whose frame is published)"
            )
        lines.append(f"Derived frames without a stored recipe: {without}")
        lines.extend(
            f"  {reason.value:<19}{counts[reason]}"
            for reason in Reason
            if counts[reason] and reason not in (*_HELD, Reason.WOULD_STORE, Reason.NOT_PUBLISHED)
        )
        if unbuilt:
            lines.append(
                f"  ({unbuilt} of these had no recipe to send, so dfo-apis was not asked"
                " whether they are published)"
            )
        if self.write:
            lines.append(
                f"Not published in dfo-apis, so no card draws on them:"
                f" {counts[Reason.NOT_PUBLISHED]}"
            )
        lines.append(self._oldest_line())
        return "\n".join(lines)

    def _oldest_line(self) -> str:
        if self.oldest_created_at is None:
            return "Oldest frame object: none held"
        age = self.at - self.oldest_created_at
        return (
            f"Oldest frame object: {age.total_seconds() / 86_400:.1f} days"
            f" (written {_stamp(self.oldest_created_at)})"
        )


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pair(ref: FrameRef) -> str:
    return f"{ref.spec_id} {ref.content_version}"


def _frame_line(outcome: FrameOutcome) -> str:
    line = f"{outcome.reason.value:<19}{_pair(outcome.ref)} {_stamp(outcome.created_at)}"
    return line if outcome.detail is None else f"{line} {outcome.detail}"


class _RunStore:
    """The store as one run reads it: a `MaterialisationStore` that keeps each derived
    frame's trace once read, and each recovered mint record. `recover` is the
    `build_recipe` recovery routine. Only with `write` does a recovered record reach
    the store underneath."""

    def __init__(self, store: MaterialisationStore, *, client: LiveApisClient, write: bool) -> None:
        self._store = store
        self._client = client
        self._write = write
        self._traces: dict[tuple[str, str], StoredFrameTrace] = {}
        self._recovered: dict[tuple[str, str], MintRecord] = {}
        self._unrecoverable: dict[tuple[str, str], MintRecordUnrecoverableError] = {}
        # One recovery at a time, so frames in flight over one root share its answer.
        self._recovering = asyncio.Lock()

    async def recover(self, ref: FrameRef) -> MintRecord:
        key = (ref.spec_id, ref.content_version)
        async with self._recovering:
            if key in self._recovered:
                return self._recovered[key]
            if key in self._unrecoverable:
                raise self._unrecoverable[key]
            try:
                record = await recover_mint_record(
                    ref, dataset_versions=self._client.list_versions, store=self._store
                )
            except MintRecordUnrecoverableError as exc:
                self._unrecoverable[key] = exc
                raise
            if self._write:
                await self._store.put_mint_record(record)
            self._recovered[key] = record
            return record

    async def get_trace(self, ref: FrameRef) -> StoredFrameTrace | None:
        key = (ref.spec_id, ref.content_version)
        if key not in self._traces:
            trace = await self._store.get_trace(ref)
            if trace is None:
                return None
            self._traces[key] = trace
        return self._traces[key]

    async def get_mint_record(self, ref: FrameRef) -> MintRecord | None:
        recovered = self._recovered.get((ref.spec_id, ref.content_version))
        return recovered if recovered is not None else await self._store.get_mint_record(ref)

    async def put_mint_record(self, record: MintRecord) -> None:
        await self._store.put_mint_record(record)

    async def get_anchor_record(self, anchor_key: str) -> AnchorRecord | None:
        return await self._store.get_anchor_record(anchor_key)

    async def put_anchor_record(self, record: AnchorRecord) -> None:
        await self._store.put_anchor_record(record)

    async def list_frames(self) -> list[FrameListing]:
        return await self._store.list_frames()

    async def put(
        self, ref: FrameRef, *, frame: Frame, trace: StoredFrameTrace, writer: str
    ) -> None:
        await self._store.put(ref, frame=frame, trace=trace, writer=writer)

    async def get(self, ref: FrameRef) -> tuple[Frame, StoredFrameTrace] | None:
        return await self._store.get(ref)

    async def open(self, ref: FrameRef) -> HeldFrame | None:
        return await self._store.open(ref)

    async def aclose(self) -> None:
        await self._store.aclose()


def _call_failure(exc: LiveApisError) -> tuple[Reason, str]:
    return Reason.APIS_FAILURE, f"status={exc.status} code={exc.code}"


async def _recipe_or_failure(
    ref: FrameRef, *, store: _RunStore
) -> dict[str, Any] | tuple[Reason, str]:
    """`ref`'s recipe, or why none can be built."""
    try:
        return await build_recipe(ref, store=store, recover=store.recover)
    except RecipeAncestorMissingError as exc:
        return Reason.MISSING_ANCESTOR, _pair(exc.ref)
    except RecipeRootUnrecoverableError as exc:
        return Reason.ROOT_UNRECOVERABLE, f"{_pair(exc.ref)} {exc.reason}"
    except RecipeTraceUnusableError as exc:
        return Reason.TRACE_UNUSABLE, f"{_pair(exc.ref)} {exc.reason}"
    except RecipeAnchorError as exc:
        return Reason.ANCHOR_UNRESOLVED, f"{_pair(exc.ref)} {exc.reason}"


async def _unbuilt(
    ref: FrameRef, failure: tuple[Reason, str], *, client: LiveApisClient
) -> tuple[Reason, str]:
    """The outcome of a frame no recipe was built for: `already_stored` when dfo-apis
    holds one from the frame's publish, else the build failure."""
    reason, detail = failure
    held = await client.read_recipe(spec_id=ref.spec_id, content_version=ref.content_version)
    if held is None:
        return failure
    return Reason.ALREADY_STORED, f"not rebuilt: {reason.value} {detail}"


async def _compare(
    ref: FrameRef, recipe: dict[str, Any], *, client: LiveApisClient
) -> tuple[Reason, str | None]:
    """What a write of `recipe` would meet, from the recipe dfo-apis holds now."""
    held = await client.read_recipe(spec_id=ref.spec_id, content_version=ref.content_version)
    if held is None:
        return Reason.WOULD_STORE, None
    if held.get("normal_form_hash") != recipe["normal_form_hash"]:
        return Reason.CONFLICT, _HASH_CONFLICT
    if held.get("replay_key") != recipe["replay_key"]:
        return Reason.CONFLICT, _REPLAY_KEY_CONFLICT
    return Reason.ALREADY_STORED, None


async def _store_recipe(
    ref: FrameRef, recipe: dict[str, Any], *, client: LiveApisClient
) -> tuple[Reason, str | None]:
    try:
        written = await client.put_recipe(
            spec_id=ref.spec_id, content_version=ref.content_version, recipe=recipe
        )
    except LiveApisError as exc:
        if exc.code == _SNAPSHOT_NOT_FOUND:
            return Reason.NOT_PUBLISHED, None
        if exc.code == PUBLISH_CONFLICT:
            return Reason.CONFLICT, exc.constraint_id
        raise
    return (Reason.STORED if written.created else Reason.ALREADY_STORED), None


async def _backfill_frame(
    listing: FrameListing, *, store: _RunStore, client: LiveApisClient, write: bool
) -> FrameOutcome:
    ref = listing.ref
    try:
        recipe = await _recipe_or_failure(ref, store=store)
        if isinstance(recipe, tuple):
            reason, detail = await _unbuilt(ref, recipe, client=client)
        elif write:
            reason, detail = await _store_recipe(ref, recipe, client=client)
        else:
            reason, detail = await _compare(ref, recipe, client=client)
    except LiveApisError as exc:
        reason, detail = _call_failure(exc)
    except Exception as exc:
        # One unreadable object must not end a walk of the whole bucket. The frame is
        # reported and the run's exit status says a rerun is needed.
        logger.exception("recipe backfill failed on %s", _pair(ref))
        reason, detail = Reason.UNEXPECTED_ERROR, type(exc).__name__
    return FrameOutcome(ref=ref, created_at=listing.created_at, reason=reason, detail=detail)


def _utcnow() -> datetime:
    return datetime.now(UTC)


async def run_backfill(
    *,
    store: MaterialisationStore,
    client: LiveApisClient,
    write: bool = False,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int | None = None,
    now: Callable[[], datetime] = _utcnow,
) -> BackfillReport:
    """Walk `store`'s derived frames oldest first and store each one's recipe.

    Args:
        store: The materialisation store to walk.
        client: dfo-apis, with the datasets principal set.
        write: Put recipes to dfo-apis and recovered mint records to `store`. Off,
            the run reads both and writes neither.
        since: Take only frames written at or after this instant.
        until: Take only frames written before this instant.
        limit: Take at most this many frames, the oldest of those in range.
        now: The clock the report's ages are measured from.

    Raises:
        LiveApisError: dfo-apis refused or failed the first call, which reads the
            versions of a dataset id no dataset has. No frame was read.
    """
    await client.list_versions(spec_id=_PREFLIGHT_SPEC_ID)
    held = await store.list_frames()
    derived = [
        listing for listing in held if listing.ref.spec_id.startswith(_DERIVED_SPEC_ID_PREFIX)
    ]
    selected = [
        listing
        for listing in derived
        if (since is None or listing.created_at >= since)
        and (until is None or listing.created_at < until)
    ][:limit]

    run_store = _RunStore(store, client=client, write=write)
    outcomes: dict[int, FrameOutcome] = {}
    pending = iter(enumerate(selected))

    async def work() -> None:
        for index, listing in pending:
            outcomes[index] = await _backfill_frame(
                listing, store=run_store, client=client, write=write
            )

    async with asyncio.TaskGroup() as group:
        for _ in range(FRAMES_IN_FLIGHT):
            group.create_task(work())

    return BackfillReport(
        write=write,
        at=now(),
        frames_held=len(held),
        derived_held=len(derived),
        oldest_created_at=held[0].created_at if held else None,
        outcomes=tuple(outcomes[index] for index in range(len(selected))),
    )


class BackfillSettings(BaseSettings):
    """The three settings the job needs, read as the services read them."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    dfo_apis_url: str | None = None
    datasets_first_party_principal: str | None = None
    frame_store_bucket: str | None = None

    def missing(self) -> list[str]:
        """The env var of every setting that is unset or blank."""
        return [
            name.upper()
            for name in type(self).model_fields
            if not (getattr(self, name) or "").strip()
        ]


def _instant(text: str) -> datetime:
    moment = datetime.fromisoformat(text)
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m harness.recipe_backfill",
        description="Store in dfo-apis the recipe of every derived frame in the "
        "materialisation bucket. Reads only, unless --write is given.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        dest="write",
        action="store_false",
        help="build and compare, write nothing (the default)",
    )
    mode.add_argument(
        "--write",
        dest="write",
        action="store_true",
        help="store recipes in dfo-apis and recovered mint records in the bucket",
    )
    parser.add_argument(
        "--since",
        type=_instant,
        metavar="ISO",
        help="only frames written at or after this instant (UTC when no offset is given)",
    )
    parser.add_argument(
        "--until", type=_instant, metavar="ISO", help="only frames written before this instant"
    )
    parser.add_argument(
        "--limit", type=int, metavar="N", help="at most N frames, the oldest in range"
    )
    parser.set_defaults(write=False)
    return parser.parse_args(argv)


async def _run(arguments: argparse.Namespace, settings: BackfillSettings) -> BackfillReport:
    assert settings.dfo_apis_url is not None  # `main` refuses a missing setting
    store = build_store(bucket_name=settings.frame_store_bucket)
    try:
        async with LiveApisClient(
            base_url=settings.dfo_apis_url,
            datasets_principal=settings.datasets_first_party_principal,
            auth=apis_auth_provider(base_url=settings.dfo_apis_url),
        ) as client:
            return await run_backfill(
                store=store,
                client=client,
                write=arguments.write,
                since=arguments.since,
                until=arguments.until,
                limit=arguments.limit,
            )
    finally:
        await store.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the job and print its report. Returns the exit status."""
    arguments = _arguments(argv)
    settings = BackfillSettings()
    if missing := settings.missing():
        print(f"recipe backfill: set {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        report = asyncio.run(_run(arguments, settings))
    except LiveApisError as exc:
        print(
            "recipe backfill: dfo-apis refused or failed the first call"
            f" (status={exc.status} code={exc.code}); no frame was read",
            file=sys.stderr,
        )
        return 2
    print(report.render())
    return 0 if report.settled else 1


if __name__ == "__main__":
    sys.exit(main())
