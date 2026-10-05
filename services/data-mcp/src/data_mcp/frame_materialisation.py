"""Fetches a `get_data` mint's artifact bytes, parses them into a `Frame`, and
registers the result in the shared materialisation store under its frame
reference (`frame_ref_for`) -- the step that closes the gap between a mint and
an analysis-mcp verb's own `frame_ref` resolution.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx2
import pyarrow as pa
import pyarrow.parquet as pq
import rfc8785
from common.materialisation import (
    FrameExpired,
    FrameIntegrityError,
    FrameRef,
    FrameTooLarge,
    MaterialisationStore,
    MaterialisationTrace,
    get_store,
)
from common.place_type import PLACE_ROLE, UNTYPED, type_column
from common.row_budget import RowBudgetExhausted, RowClaim, row_budget
from dfo_frame_grammar import (
    FACET_IDENTITY_COLUMNS,
    STAT_VAR_COLUMN,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    Frame,
    FrameError,
    array_kernel,
    build_frame_from_arrow,
    canonical_frame_schema,
    map_units,
    unit_codes,
)
from dfo_frame_grammar.grammar import validate_unit_request
from fastmcp.exceptions import ResourceError
from pydantic import ValidationError

from data_mcp.deadline import CallDeadline
from data_mcp.manifest import ArtifactEntry, Manifest
from data_mcp.settings import Settings
from data_mcp.tool_errors import ToolRefusal

__all__ = [
    "DegradedReason",
    "MaterialisationOutcome",
    "frame_ref_for",
    "materialise_and_register",
]

logger = logging.getLogger(__name__)

# The bare signed-URL GET has its own small retry budget (see `_fetch_artifact_bytes_uncapped`'s
# docstring): this hop is idempotent, so a handful of quick retries costs little and buys
# resilience against a transient GCS blip. `read_artifact` (the resign call) is never
# wrapped in a second retry loop of its own -- it already retries internally against
# dfo-apis' snapshot route.
_ARTIFACT_GET_ATTEMPTS = 3
_ARTIFACT_GET_BACKOFF_S = 0.5  # short: this hop is idempotent and cheap

# GCS's own signal that a signed URL it's asked to redeem is invalid or expired -- observed
# directly against a real signed URL from this deployment (a tampered signature reproduces
# this status with a `SignatureDoesNotMatch` body), not merely assumed.
_SIGNED_URL_EXPIRED_STATUS = 403

# Seconds of a caller's deadline the fetch leaves unspent, for the digest check,
# parse and store write in `_verify_parse_and_register` after it.
_STORE_WRITE_RESERVE_S = 10.0

# The least fetch budget worth starting a fetch with. The store read must end with
# this plus `_STORE_WRITE_RESERVE_S` still left, so the fetch after a read that runs
# out still gets at least this much.
_MIN_FETCH_BUDGET_S = 5.0

# Seconds of a caller's deadline the store write leaves unspent, for the caller's
# rendering of its result.
_AFTER_WRITE_RESERVE_S = 3.0

# Every reserved column name `build_frame_from_arrow` recognises -- everything else in a
# fetched artifact's own schema is a role key. Imported from `dfo_frame_grammar.grammar`'s
# own constants rather than hardcoded here, so this set can never drift from what
# `build_frame_from_arrow` enforces.
_RESERVED_COLUMNS = frozenset(
    {STAT_VAR_COLUMN, TIME_PERIOD_COLUMN, VALUE_COLUMN, UNIT_COLUMN, STATUS_COLUMN}
    | set(FACET_IDENTITY_COLUMNS)
)


# Why a `"degraded"` outcome holds no confirmed frame. `"fetch"`: the store answered
# with a miss and the artifact bytes were not fetched. `"store_unchecked"`: the store
# read ran out of deadline, so the frame may be held, and the bytes were not fetched.
# `"store_write"`: the bytes were fetched and parsed but the store write did not
# finish before the deadline.
DegradedReason = Literal["fetch", "store_unchecked", "store_write"]


@dataclass(frozen=True, kw_only=True)
class MaterialisationOutcome:
    """The result of one `materialise_and_register` call.

    `attempts` counts the bare signed-URL GET attempts actually made: `0` on the
    already-held no-op path, `1` on an ordinary successful fetch or a refusal reached
    before any GET. A `"degraded"` outcome's own count is however many GET attempts
    actually landed before the fetch gave up -- anywhere from `0` (a sign failure or a
    timeout before the first GET completed) up to `_ARTIFACT_GET_ATTEMPTS` (every attempt
    exhausted), never assumed to be the ceiling. `refusal` is set only when
    `status == "refused"`, and `degraded_reason` only when `status == "degraded"`. A
    `"store_write"` degrade's worker thread may still land the write. `role_keys` and
    `stat_var_count` are the frame's own role columns and distinct StatVars
    (`observed_shape`), both set whenever this call read the held frame or parsed the
    fetched one: on every `"registered"` outcome and on a `"store_write"` degrade.
    `frame_ref` is the reference the frame is held under (`frame_ref_for`), set on
    every outcome but a refusal.
    """

    status: Literal["registered", "degraded", "refused"]
    attempts: int
    refusal: ToolRefusal | None = None
    degraded_reason: DegradedReason | None = None
    role_keys: tuple[str, ...] | None = None
    stat_var_count: int | None = None
    frame_ref: FrameRef | None = None


async def materialise_and_register(
    *,
    manifest: Manifest,
    settings: Settings,
    writer: str,
    options: Mapping[str, Any],
    transport: httpx2.AsyncBaseTransport | None = None,
    deadline: CallDeadline | None = None,
) -> MaterialisationOutcome:
    """Fetch, parse, and register the frame `manifest` names, or report why it couldn't.

    `options` is the mint request's `options`. The frame registers under the
    reference `frame_ref_for` gives the manifest's pair and `options.columns`, and
    its bytes are the artifact whose `rendering` equals `options`
    (`_select_requested_artifact`).

    Called once per successful `get_data` mint, after the manifest itself has already
    validated. Checks the store first: a pair already held, read back under a row claim
    and passing every check that read makes, is a no-op with no fetch, since `get_data`
    may be called again for exactly this reason (see the degraded-fetch note
    `get_data_digest.py` renders). A held pair that fails the store's own integrity or
    size check on that read is refused rather than treated as a miss: the store is
    content-addressed and insert-only, so a held-but-broken object can't be healed by
    falling through to re-register (see `frame_integrity_failed`'s and `over_ceiling`'s
    own factories for what each case tells the caller). Otherwise fetches the
    requested artifact, bounds its row count from Parquet metadata alone
    (before any row is parsed), parses it into a `Frame`, and registers it under
    its frame reference.

    Returns a `"degraded"` outcome (never a refusal) when the artifact bytes could not be
    fetched within budget: the mint itself succeeded and named a real, durable result, and
    throwing that away would waste a call that can already run up to 125 seconds. A
    genuinely unusable manifest or artifact refuses instead -- see each refusal branch
    below for which.

    `deadline` bounds each step. The store read must end with `_MIN_FETCH_BUDGET_S`
    plus `_STORE_WRITE_RESERVE_S` left, and one that runs out is followed by a fetch
    whose failure degrades with `degraded_reason="store_unchecked"`. The
    fetch's wall-clock budget is `settings.artifact_fetch_budget_s`, cut to what
    `deadline` leaves once `_STORE_WRITE_RESERVE_S` is held back for the store write;
    under `_MIN_FETCH_BUDGET_S` the fetch is not started and the outcome is degraded.
    The store write must end with `_AFTER_WRITE_RESERVE_S` left, and one that runs out
    degrades with `degraded_reason="store_write"`.
    """
    ref = _ref_or_refusal(manifest, columns=tuple(options.get("columns", ())))
    if isinstance(ref, MaterialisationOutcome):
        return ref
    outcome = await _register(
        ref,
        manifest=manifest,
        options=options,
        settings=settings,
        writer=writer,
        transport=transport,
        deadline=deadline,
    )
    return outcome if outcome.status == "refused" else replace(outcome, frame_ref=ref)


async def _register(
    ref: FrameRef,
    *,
    manifest: Manifest,
    options: Mapping[str, Any],
    settings: Settings,
    writer: str,
    transport: httpx2.AsyncBaseTransport | None,
    deadline: CallDeadline | None,
) -> MaterialisationOutcome:
    """`materialise_and_register` for the frame held under `ref`."""
    store = get_store()
    checked = await _check_store(
        store, ref, max_frame_rows=settings.max_frame_rows, deadline=deadline
    )
    if isinstance(checked, MaterialisationOutcome):
        return checked
    fetch_failure_reason = checked

    artifact = _select_requested_artifact(manifest, options=options)
    if artifact is None:
        return MaterialisationOutcome(
            status="refused",
            attempts=0,
            refusal=ToolRefusal.upstream_response_unusable(
                detail="the mint's manifest lists no artifact with the requested rendering "
                f"{json.dumps(dict(options), sort_keys=True)}"
            ),
        )
    progress = _FetchProgress()
    budget_s = settings.artifact_fetch_budget_s
    if deadline is not None:
        budget_s = min(budget_s, deadline.remaining() - _STORE_WRITE_RESERVE_S)
        if budget_s < _MIN_FETCH_BUDGET_S:
            logger.warning(
                "materialise_and_register: only %.1fs of deadline left to fetch %s/%s; "
                "not starting the fetch",
                budget_s,
                ref.spec_id,
                ref.content_version,
            )
            return MaterialisationOutcome(
                status="degraded", attempts=0, degraded_reason=fetch_failure_reason
            )
    fetched = await _fetch_artifact_bytes(
        subject_id=ref.spec_id,
        # The artifact sits under dfo-apis' own pair, whatever the rendering.
        content_version=manifest.content_version,
        artifact_digest=artifact.artifact_digest,
        transport=transport,
        budget_s=budget_s,
        progress=progress,
    )
    if fetched is None:
        return MaterialisationOutcome(
            status="degraded", attempts=progress.attempts, degraded_reason=fetch_failure_reason
        )

    return await _verify_parse_and_register(
        fetched=fetched,
        artifact=artifact,
        ref=ref,
        manifest=manifest,
        settings=settings,
        writer=writer,
        store=store,
        write_deadline=deadline.reserving(_AFTER_WRITE_RESERVE_S) if deadline is not None else None,
    )


async def _check_store(
    store: MaterialisationStore,
    ref: FrameRef,
    *,
    max_frame_rows: int,
    deadline: CallDeadline | None,
) -> MaterialisationOutcome | DegradedReason:
    """The outcome the store read settles on its own, or the reason a failed fetch
    after it reports.

    A held pair that passes every check the store's read makes is a no-op carrying
    the held frame's role keys and StatVar count, and a held pair that fails the
    store's integrity or size check refuses. A miss or an expired pair returns
    `"fetch"`, and a read that runs out of deadline returns `"store_unchecked"`,
    since the pair may be held. A held pair's rows are claimed
    before it is read (`_read_held`), a claim that does not fit refuses
    `service_unavailable`, and a pair over `max_frame_rows` refuses `over_ceiling`.
    """
    read_deadline = (
        deadline.reserving(_STORE_WRITE_RESERVE_S + _MIN_FETCH_BUDGET_S)
        if deadline is not None
        else None
    )
    try:
        async with asyncio.timeout(_seconds_left(read_deadline)):
            held_shape = await _read_held(store, ref, max_frame_rows=max_frame_rows)
    except TimeoutError:
        logger.warning(
            "materialise_and_register: the store read for %s/%s ran out of deadline; "
            "fetching without knowing whether it holds the pair",
            ref.spec_id,
            ref.content_version,
        )
        return "store_unchecked"
    except FrameExpired:
        return "fetch"  # re-register fresh; `put` heals the expiry
    except RowBudgetExhausted as exc:
        return MaterialisationOutcome(
            status="refused",
            attempts=0,
            refusal=ToolRefusal.row_budget_full(
                requested=exc.requested, free=exc.free, max_rows=exc.max_rows
            ),
        )
    except FrameIntegrityError as exc:
        # Content-addressed and insert-only, so falling through to re-register would
        # just hit the identical corrupted object again -- see
        # `materialise_and_register`'s own docstring.
        return MaterialisationOutcome(
            status="refused",
            attempts=0,
            refusal=ToolRefusal.frame_integrity_failed(
                spec_id=ref.spec_id, content_version=ref.content_version, reason=exc.reason
            ),
        )
    except FrameTooLarge as exc:
        # Falls through for the same reason as `FrameIntegrityError` above, though
        # the held object is exactly what was written -- just over the store's own
        # serve-back ceiling or this service's row ceiling, so only a narrower scope
        # gets past it.
        return MaterialisationOutcome(
            status="refused",
            attempts=0,
            refusal=ToolRefusal.over_ceiling(detail=exc.reason, ceiling_kind="bytes"),
        )
    return (
        "fetch"
        if held_shape is None
        else MaterialisationOutcome(
            status="registered",
            attempts=0,
            role_keys=held_shape[0],
            stat_var_count=held_shape[1],
        )
    )


async def _read_held(
    store: MaterialisationStore, ref: FrameRef, *, max_frame_rows: int
) -> tuple[tuple[str, ...], int] | None:
    """The held frame's role keys and StatVar count (`observed_shape`), or `None` when
    the store does not hold `ref`. The frame is read in full under a claim on its rows
    so a cache hit passes every check a read makes, the store's content-version check
    included, and is then discarded.

    The claim is the row count the store recorded, or `max_frame_rows` for an object
    stored without one, and the read refuses an object holding more rows than that
    before decoding it. Raises `FrameTooLarge` for a recorded count over
    `max_frame_rows`, `RowBudgetExhausted` when the rows do not fit, and whatever the
    store's `open` or `HeldFrame.read` raises. The claim stays held until the read's
    worker thread ends, even when the deadline cuts it off."""
    held = await store.open(ref)
    if held is None:
        return None
    if held.num_rows is not None and held.num_rows > max_frame_rows:
        raise FrameTooLarge(
            ref,
            reason=(
                f"stored frame has {held.num_rows} rows, over this service's "
                f"{max_frame_rows}-row ceiling"
            ),
        )
    rows = max_frame_rows if held.num_rows is None else held.num_rows
    with row_budget().claim(rows) as claim:
        frame, _trace = await claim.hold_through(held.read(max_rows=rows))
    return observed_shape(frame)


def _seconds_left(deadline: CallDeadline | None) -> float | None:
    """What `asyncio.timeout` takes for `deadline`: `None` for no bound, never negative."""
    return max(deadline.remaining(), 0.0) if deadline is not None else None


async def _verify_parse_and_register(
    *,
    fetched: bytes,
    artifact: ArtifactEntry,
    ref: FrameRef,
    manifest: Manifest,
    settings: Settings,
    writer: str,
    store: MaterialisationStore,
    write_deadline: CallDeadline | None,
) -> MaterialisationOutcome:
    """Verify `fetched` against `artifact.artifact_digest`, bound and parse it into a
    `Frame`, and register the result under `ref` before `write_deadline` -- or the
    outcome for whichever step failed. Split out of `materialise_and_register` to keep
    that function's own return-statement count within this repo's lint bound.

    The artifact's rows are claimed from the process's frame row budget
    (`common.row_budget`) once its footer gives their count, and held through the parse
    and until the store write ends; a claim that does not fit refuses
    `service_unavailable`.
    """
    computed_digest = f"sha256:{hashlib.sha256(fetched).hexdigest()}"
    if computed_digest != artifact.artifact_digest:
        logger.warning(
            "materialise_and_register: fetched bytes for %s/%s do not hash to the "
            "advertised artifact_digest (advertised %s, computed %s)",
            ref.spec_id,
            ref.content_version,
            artifact.artifact_digest,
            computed_digest,
        )
        return MaterialisationOutcome(
            status="refused",
            attempts=1,
            refusal=ToolRefusal.upstream_response_unusable(
                detail="the fetched artifact bytes do not hash to the manifest's "
                f"advertised artifact_digest {artifact.artifact_digest!r}; "
                f"computed {computed_digest!r}"
            ),
        )

    try:
        row_count = _parquet_row_count(fetched)
    except pa.ArrowException as exc:
        return _unparseable(exc)
    if row_count > settings.max_frame_rows:
        return MaterialisationOutcome(
            status="refused",
            attempts=1,
            refusal=ToolRefusal.over_ceiling(
                detail=f"the fetched artifact parsed to {row_count} rows, over this "
                f"service's {settings.max_frame_rows}-row ceiling"
            ),
        )
    try:
        claim = row_budget().claim(row_count)
    except RowBudgetExhausted as exc:
        return MaterialisationOutcome(
            status="refused",
            attempts=1,
            refusal=ToolRefusal.row_budget_full(
                requested=exc.requested, free=exc.free, max_rows=exc.max_rows
            ),
        )
    with claim:
        return await _parse_and_register(
            fetched=fetched,
            ref=ref,
            manifest=manifest,
            writer=writer,
            store=store,
            claim=claim,
            write_deadline=write_deadline,
        )


def _unparseable(exc: Exception) -> MaterialisationOutcome:
    """The refusal for fetched bytes that do not parse into a valid frame."""
    return MaterialisationOutcome(
        status="refused",
        attempts=1,
        refusal=ToolRefusal.upstream_response_unusable(
            detail=f"the fetched artifact did not parse into a valid frame: {exc}"
        ),
    )


async def _parse_and_register(
    *,
    fetched: bytes,
    ref: FrameRef,
    manifest: Manifest,
    writer: str,
    store: MaterialisationStore,
    claim: RowClaim,
    write_deadline: CallDeadline | None,
) -> MaterialisationOutcome:
    """Parse `fetched` into a `Frame` and register it under `ref` before
    `write_deadline`, or the outcome for whichever step failed. `claim` holds the
    frame's rows until the store write ends, even when it outlives
    `write_deadline`."""
    # `pyarrow.ArrowException` (a 2xx response whose body isn't a valid/complete Parquet
    # file) and `KeyError` (a Parquet file missing its `time_period` or `value` column,
    # raised while decoding it) are neither one a `FrameError`, but a genuinely unusable
    # artifact either way -- all three are refused as a `FrameError` is.
    try:
        frame = _frame_from_parquet_bytes(fetched)
    except (FrameError, pa.ArrowException, KeyError) as exc:
        return _unparseable(exc)

    # No cv_ integrity gate here -- `_verify_parse_and_register`'s `artifact_digest`
    # comparison already gates the fetched bytes themselves; this is about the derived
    # `content_version` instead.
    # Recomputing `mint_content_version` from the freshly
    # parsed frame plus `manifest.trace`'s lineage fields does not reproduce this
    # deployment's own `content_version` bit-for-bit: `manifest.trace["facets"]` entries
    # carry bookkeeping members (`default_policy`, `selection_reason`,
    # `dispatch_facet_id`, ...) outside the closed `{axis, ref, basis}` shape
    # `mint_content_version` accepts, and their `basis` value observed on a real mint
    # (`"deployment_fallback"`) falls outside that function's own two-value vocabulary
    # (`bound`/`declared_default`) -- the recompute raises before it can even be compared.
    # Every frame this function parses is therefore registered under its frame reference
    # (`frame_ref_for`) unconditionally, never gated on a recompute this repo cannot yet
    # perform reliably.
    trace = MaterialisationTrace(
        upstream_trace=manifest.trace, spec_id=ref.spec_id, content_version=ref.content_version
    )
    try:
        async with asyncio.timeout(_seconds_left(write_deadline)):
            await claim.hold_through(store.put(ref, frame=frame, trace=trace, writer=writer))
    except TimeoutError:
        # The store is insert-only and content-addressed, so a write the worker
        # thread lands later is the same object a retry would write. The row claim
        # stays held until that thread's write ends.
        logger.warning(
            "materialise_and_register: the store write for %s/%s ran out of deadline unconfirmed",
            ref.spec_id,
            ref.content_version,
        )
        role_keys, stat_var_count = observed_shape(frame)
        return MaterialisationOutcome(
            status="degraded",
            attempts=1,
            degraded_reason="store_write",
            role_keys=role_keys,
            stat_var_count=stat_var_count,
        )
    except FrameTooLarge as exc:
        # `_verify_parse_and_register`'s row_count check bounds against this
        # service's own `settings.max_frame_rows`; the store's own `_max_frame_rows`
        # is a second, independently-owned constant (see `gcs_store.py`'s own
        # module docstring for why the two are never coupled), so a frame
        # that passed that check can still be refused here if that
        # constant is ever set lower. No object landed, so there is nothing
        # to register -- a refusal, not a raise.
        return MaterialisationOutcome(
            status="refused",
            attempts=1,
            refusal=ToolRefusal.over_ceiling(detail=exc.reason, ceiling_kind="rows"),
        )
    role_keys, stat_var_count = observed_shape(frame)
    return MaterialisationOutcome(
        status="registered", attempts=1, role_keys=role_keys, stat_var_count=stat_var_count
    )


def frame_ref_for(*, spec_id: str, content_version: str, columns: tuple[str, ...] = ()) -> FrameRef:
    """The frame reference of the frame a mint names as `{spec_id, content_version}`,
    minted with `options.columns` set to `columns`.

    dfo-apis keeps one `content_version` for every rendering of a materialisation,
    so a frame minted with extra columns (`get_data(place_types=True)`) and one
    minted with the default columns come back under one pair. The default
    rendering keeps the mint's own pair, and any other names its columns in its
    `content_version`: `cv_` and the first 32 hex digits of the SHA-256 of the
    RFC 8785 form of `{"columns": [...], "content_version": ...}`, with `columns`
    sorted. Two renderings of one pair are then two frames, and the store's
    insert-only entry for one is never served for the other.

    Raises:
        pydantic.ValidationError: When `spec_id` and `content_version` do not fit
            `FrameRef`'s pair shape.
    """
    ref = FrameRef(spec_id=spec_id, content_version=content_version)
    if not columns:
        return ref
    rendering = rfc8785.dumps({"columns": sorted(columns), "content_version": content_version})
    return FrameRef(
        spec_id=spec_id, content_version=f"cv_{hashlib.sha256(rendering).hexdigest()[:32]}"
    )


def _ref_or_refusal(
    manifest: Manifest, *, columns: tuple[str, ...]
) -> FrameRef | MaterialisationOutcome:
    """The reference this frame registers under (`frame_ref_for`), or the refused
    outcome for a manifest that can't produce one: either no `definition` section at
    all, or a `spec_id`/`content_version` combination that doesn't match `FrameRef`'s
    own pair shape.
    """
    if manifest.definition is None:
        # get_data always mints via `kind: "spec_id"`, which the mint route answers with a
        # definition manifest, never a collection one -- this would mean that contract
        # changed underneath this module (mirrors `get_data_digest.py`'s own identical
        # guard).
        return MaterialisationOutcome(
            status="refused",
            attempts=0,
            refusal=ToolRefusal.upstream_response_unusable(
                detail="get_data's manifest carries no 'definition'; a collection manifest "
                "here would mean the mint route's contract changed"
            ),
        )
    try:
        return frame_ref_for(
            spec_id=manifest.definition.spec_id,
            content_version=manifest.content_version,
            columns=columns,
        )
    except ValidationError as exc:
        return MaterialisationOutcome(
            status="refused",
            attempts=0,
            refusal=ToolRefusal.upstream_response_unusable(
                detail="the manifest's spec_id/content_version did not match the pair shape "
                f"this service expects: {exc}"
            ),
        )


def _select_requested_artifact(
    manifest: Manifest, *, options: Mapping[str, Any]
) -> ArtifactEntry | None:
    """The artifact whose `rendering` equals the mint request's `options`, or `None`.

    dfo-apis stores each artifact's `rendering` as the request's options with unset
    fields left out, and a manifest lists every rendering stored under its pair: other
    callers' groupby sums and column selections sit beside the requested one, in no
    set order. A manifest whose own artifact has expired lists none that matches.
    """
    return next((a for a in manifest.artifacts if a.rendering == options), None)


@dataclass
class _FetchProgress:
    """How many bare GETs a fetch has actually made so far.

    Mutable and passed by reference from `materialise_and_register` down through
    `_fetch_artifact_bytes` into `_fetch_artifact_bytes_uncapped`, rather than returned
    alongside the fetched bytes: `_fetch_artifact_bytes`'s own return shape (`bytes | None`)
    is a seam `data_mcp`'s and `harness`'s own test suites monkeypatch directly, and a
    `TimeoutError` cancelling the fetch coroutine mid-flight would lose a plain return
    value along with the cancelled frame anyway.
    """

    attempts: int = 0


async def _fetch_artifact_bytes(
    *,
    subject_id: str,
    content_version: str,
    artifact_digest: str,
    transport: httpx2.AsyncBaseTransport | None,
    budget_s: float,
    progress: _FetchProgress,
) -> bytes | None:
    """Fetch one artifact's bytes under one wall-clock budget, or return `None`.

    A crossed deadline mid-sequence returns `None` (degraded), the same outcome as
    exhausting the GET attempts inside `_fetch_artifact_bytes_uncapped` -- never a raised
    `TimeoutError`. `progress.attempts` records the number of bare GET attempts actually
    made, whichever way this returns.
    """
    try:
        async with asyncio.timeout(budget_s):
            return await _fetch_artifact_bytes_uncapped(
                subject_id=subject_id,
                content_version=content_version,
                artifact_digest=artifact_digest,
                transport=transport,
                progress=progress,
            )
    except TimeoutError:
        logger.warning(
            "materialise_and_register: artifact fetch for %s/%s exceeded the %.1fs budget "
            "after %d GET attempt(s)",
            subject_id,
            content_version,
            budget_s,
            progress.attempts,
        )
        return None


async def _fetch_artifact_bytes_uncapped(
    *,
    subject_id: str,
    content_version: str,
    artifact_digest: str,
    transport: httpx2.AsyncBaseTransport | None,
    progress: _FetchProgress,
) -> bytes | None:
    """One `read_artifact` call for a signed URL -- that function already retries
    internally against dfo-apis' snapshot route, so it is called once here, never wrapped
    in a second retry loop -- then up to `_ARTIFACT_GET_ATTEMPTS` bare GETs against the
    signed URL it returns. A GET response signalling the URL itself expired (a `403`,
    `_SIGNED_URL_EXPIRED_STATUS`) re-signs once via one further `read_artifact` call rather
    than retrying the dead URL; any other GET failure retries the same URL. `progress` is
    incremented once per bare GET actually attempted, whatever its outcome.

    `read_artifact` is imported here, not at module level: `data_mcp.artifact_resource`
    itself imports `_SPEC_ID_PATTERN` from `data_mcp.get_data_tool`, which imports
    `materialise_and_register` from this module -- a module-level import here would be
    circular. Deferred to first call, well after both modules have finished loading.
    """
    from data_mcp.artifact_resource import read_artifact  # noqa: PLC0415

    try:
        signed = await read_artifact(subject_id, content_version, artifact_digest)
    except ResourceError as exc:
        logger.warning(
            "materialise_and_register: could not sign artifact URL for %s/%s: %s",
            subject_id,
            content_version,
            exc,
        )
        return None

    logger.info(
        "materialise_and_register: fetching artifact bytes for %s/%s", subject_id, content_version
    )
    resigned = False
    async with httpx2.AsyncClient(transport=transport, timeout=httpx2.Timeout(30.0)) as client:
        for attempt in range(1, _ARTIFACT_GET_ATTEMPTS + 1):
            try:
                response = await client.get(signed["url"])
            except httpx2.TransportError as exc:
                progress.attempts += 1
                logger.warning(
                    "materialise_and_register: GET attempt %d/%d for %s/%s failed: %s",
                    attempt,
                    _ARTIFACT_GET_ATTEMPTS,
                    subject_id,
                    content_version,
                    exc,
                )
                if attempt < _ARTIFACT_GET_ATTEMPTS:
                    await asyncio.sleep(_ARTIFACT_GET_BACKOFF_S)
                continue
            progress.attempts += 1
            if response.is_success:
                return response.content
            if response.status_code == _SIGNED_URL_EXPIRED_STATUS and not resigned:
                try:
                    signed = await read_artifact(subject_id, content_version, artifact_digest)
                except ResourceError as exc:
                    logger.warning(
                        "materialise_and_register: could not re-sign artifact URL for "
                        "%s/%s after a %d response: %s",
                        subject_id,
                        content_version,
                        _SIGNED_URL_EXPIRED_STATUS,
                        exc,
                    )
                    return None
                resigned = True
                continue
            logger.warning(
                "materialise_and_register: GET attempt %d/%d for %s/%s returned status %d",
                attempt,
                _ARTIFACT_GET_ATTEMPTS,
                subject_id,
                content_version,
                response.status_code,
            )
            if attempt < _ARTIFACT_GET_ATTEMPTS:
                await asyncio.sleep(_ARTIFACT_GET_BACKOFF_S)
    logger.warning(
        "materialise_and_register: exhausted %d GET attempt(s) for %s/%s",
        _ARTIFACT_GET_ATTEMPTS,
        subject_id,
        content_version,
    )
    return None


def _parquet_row_count(data: bytes) -> int:
    """The artifact's row count, read from its own Parquet footer metadata only --
    never materialising row data, so an oversized artifact is bounded before any row is
    actually parsed. Raises `pyarrow.ArrowException` for bytes that aren't a valid or
    complete Parquet file; the caller (`materialise_and_register`) maps that to a refusal
    rather than letting it escape."""
    return pq.ParquetFile(io.BytesIO(data)).metadata.num_rows


_UTC_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MICROSECONDS_PER_DAY = 86_400_000_000

# How many of each temporal type's stored integers make up one day.
_TICKS_PER_DAY = {"s": 86_400, "ms": 86_400_000, "us": 86_400_000_000, "ns": 86_400_000_000_000}


def _time_period_text(column: pa.ChunkedArray) -> pa.ChunkedArray | pa.Array:
    """`time_period` as `build_frame_from_arrow` reads it: text. A `date32`/`date64`
    column becomes `YYYY-MM-DD`, and a `timestamp` column its date portion, taken in
    the column's own time zone when it has one. Any other column passes through
    unchanged, so a string column (the common case for a real fetched artifact) is
    untouched and a malformed one is left for `build_frame_from_arrow` to refuse.

    Each distinct value is converted once with Python's `datetime` and `zoneinfo`
    from its stored integer, so a date outside `0001-01-01` to `9999-12-31` refuses
    `invalid_time_period`. Arrow's own casts are not used: a timestamp to `date32`
    cast wraps a day count past 32 bits to a wrong date, and its time zone rules
    stop applying daylight saving after 2037.
    """
    data_type = column.type
    if pa.types.is_timestamp(data_type):
        ticks_per_day = _TICKS_PER_DAY[data_type.unit]
        zone = _zone(data_type.tz) if data_type.tz is not None else None
    elif pa.types.is_date32(data_type):
        ticks_per_day, zone = 1, None
    elif pa.types.is_date64(data_type):
        ticks_per_day, zone = _TICKS_PER_DAY["ms"], None
    else:
        return column
    encoded = array_kernel("dictionary_encode", column.combine_chunks())
    assert isinstance(encoded, pa.DictionaryArray)
    ticks = encoded.dictionary.cast(pa.int32() if pa.types.is_date32(data_type) else pa.int64())
    dates = [
        _date_text(tick * _MICROSECONDS_PER_DAY // ticks_per_day, zone)
        for tick in ticks.to_pylist()
    ]
    return array_kernel("take", pa.array(dates, type=pa.string()), encoded.indices)


def _zone(name: str) -> tzinfo:
    """The time zone an Arrow timestamp type names: a fixed `+HH:MM` offset or an
    IANA zone. One that names neither refuses `invalid_time_period`."""
    try:
        if name[:1] in "+-":
            zone = datetime.strptime(name, "%z").tzinfo
            assert zone is not None
            return zone
        return ZoneInfo(name)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise FrameError(
            "invalid_time_period", f"time_period's time zone {name!r} is not known"
        ) from exc


def _date_text(microseconds: int, zone: tzinfo | None) -> str:
    """The `YYYY-MM-DD` date of the instant `microseconds` after the Unix epoch, in
    `zone` or in UTC, or a refusal when it falls outside Python's `date` range."""
    try:
        moment = _UTC_EPOCH + timedelta(microseconds=microseconds)
        if zone is not None:
            moment = moment.astimezone(zone)
    except OverflowError as exc:
        raise FrameError(
            "invalid_time_period",
            f"time_period is outside 0001-01-01 to 9999-12-31: {microseconds} "
            "microseconds from the Unix epoch",
        ) from exc
    return moment.date().isoformat()


def _value_numbers(column: pa.ChunkedArray) -> pa.ChunkedArray | pa.Array:
    """`value` as `build_frame_from_arrow` reads it: an integer or floating column.
    A decimal column becomes `float64` through Python's `float(Decimal)`, applied once
    per distinct value, because Arrow's own decimal-to-double cast is not correctly
    rounded. A null stays null for `build_frame_from_arrow` to refuse. Any other column
    passes through unchanged.
    """
    if not pa.types.is_decimal(column.type):
        return column
    encoded = array_kernel("dictionary_encode", column.combine_chunks())
    assert isinstance(encoded, pa.DictionaryArray)
    floats = pa.array([float(cell) for cell in encoded.dictionary.to_pylist()], type=pa.float64())
    return array_kernel("take", floats, encoded.indices)


def _check_unit_requests(column: pa.ChunkedArray) -> None:
    """Refuse a struct `unit` column whose requests fail `validate_unit_request` when
    every struct field counts as a member, a null one included.

    A struct column carries `{currency, prices, base_year}` with every field on every
    row, so a current-price unit arrives with a null `base_year`; its null fields are
    dropped before the check, as `build_frame_from_arrow` drops them, so it passes. A struct
    in the frame's own tagged `ref`/`request` form, a current-price unit with a base
    year, or one with no currency is refused `invalid_unit`. Each distinct value is
    checked once. A text or null column passes.
    """
    if not pa.types.is_struct(column.type):
        return
    values = column.combine_chunks()
    for request in values.take(unit_codes(values).first_rows).to_pylist():
        if request is not None:
            validate_unit_request(
                {key: value for key, value in request.items() if value is not None}
            )


_UNIT_TYPE = canonical_frame_schema((UNIT_COLUMN,)).field(UNIT_COLUMN).type


def _tagged_unit_from_text(text: object) -> dict[str, object] | None:
    """One text `unit` cell as the frame's tagged `{ref, request}` unit value.

    dfo-apis writes a text `unit` column (`materialise.formats._cell_text`): a unit ref
    as its own text, and a converted unit as the JCS text of its
    `{currency, prices, base_year}` object. A unit ref never starts with `{`, so a cell
    that does is a converted unit. It must parse as JSON, which makes it an object, and
    pass `validate_unit_request`, checked here so Arrow never drops an unknown member or
    fails on a mistyped one while building the struct.
    """
    if text is None:
        return None
    cell = cast(str, text)
    if not cell.startswith("{"):
        return {"ref": cell, "request": None}
    try:
        request: dict[str, object] = json.loads(cell)
    except json.JSONDecodeError as exc:
        raise FrameError(
            "schema_violation", f"unit text {cell[:80]!r} is not a converted-unit object"
        ) from exc
    validate_unit_request(request)
    return {"ref": None, "request": request}


def _with_text_units_decoded(table: pa.Table) -> pa.Table:
    """`table` with a text `unit` column decoded into the frame's tagged unit struct
    (`_tagged_unit_from_text`), each distinct cell once. A table without a `unit`
    column, or with a struct or null one, is returned unchanged."""
    if UNIT_COLUMN not in table.schema.names:
        return table
    index = table.schema.get_field_index(UNIT_COLUMN)
    column_type = table.schema.field(index).type
    if not (pa.types.is_string(column_type) or pa.types.is_large_string(column_type)):
        return table
    decoded = map_units(table.column(index), _tagged_unit_from_text, result_type=_UNIT_TYPE)
    return table.set_column(index, UNIT_COLUMN, decoded)


def observed_shape(frame: Frame) -> tuple[tuple[str, ...], int]:
    """`frame`'s role columns and its distinct `stat_var` count.

    The role columns are every column outside `_RESERVED_COLUMNS`, in the order
    `grammar.derive_columns` gives them, which is `normalize_role_keys`' sorted order.
    An analysis-mcp verb declares the same pair as its result's `role_keys` and
    `stat_var_count` (`analysis_mcp.result.observed_shape`)."""
    role_keys = tuple(column for column in frame.columns if column not in _RESERVED_COLUMNS)
    return role_keys, len(frame.table.column(STAT_VAR_COLUMN).unique())


def _frame_from_parquet_bytes(data: bytes) -> Frame:
    """Parse `data` (a whole Parquet file's bytes) into a `Frame`.

    `role_keys` is every column name in the table's own schema minus the grammar's
    reserved set (`_RESERVED_COLUMNS`) -- whatever entity/geography roles this dataset's
    shape declares. `status` is reserved and `build_frame_from_arrow` refuses it if the
    table carries it at all, so it is dropped here. Raises `FrameError` for a genuinely
    malformed artifact, `pyarrow.ArrowException` for bytes that aren't a valid or
    complete Parquet file, and `KeyError` for a Parquet file missing `time_period` or
    `value`; the caller (`materialise_and_register`) maps all three to a refusal rather
    than letting any escape. A malformed artifact is refused whatever its row count: a
    zero-row file missing `stat_var` or `time_period`, or typing a role, `time_period` or
    `unit` column as an integer, is refused like a populated one.

    A text `unit` column holds unit refs and converted units as JCS object text, each
    decoded into the frame's own unit cell (`_with_text_units_decoded`).

    An `observationAbout_type` column, which `get_data(place_types=True)` asks for,
    is a role key too, with each empty cell written `UNTYPED` (`_with_untyped_places`).
    """
    table = _with_untyped_places(pq.read_table(io.BytesIO(data)))
    role_keys = [name for name in table.schema.names if name not in _RESERVED_COLUMNS]
    time_period = _time_period_text(table.column(TIME_PERIOD_COLUMN))
    value = _value_numbers(table.column(VALUE_COLUMN))
    table = table.set_column(
        table.schema.get_field_index(TIME_PERIOD_COLUMN), TIME_PERIOD_COLUMN, time_period
    )
    table = table.set_column(table.schema.get_field_index(VALUE_COLUMN), VALUE_COLUMN, value)
    if UNIT_COLUMN in table.schema.names:
        _check_unit_requests(table.column(UNIT_COLUMN))
    table = _with_text_units_decoded(_with_null_optionals_as_text(table))
    if STATUS_COLUMN in table.schema.names:
        table = table.drop_columns(STATUS_COLUMN)
    return build_frame_from_arrow(table, role_keys=role_keys)


def _with_untyped_places(table: pa.Table) -> pa.Table:
    """`table` with each empty or null `observationAbout_type` cell written as
    `common.place_type.UNTYPED`.

    dfo-apis leaves the cell empty for a place its graph holds no type for, and
    `build_frame_from_arrow` refuses an empty role cell. A column typed null holds no
    type at all. A table without the column, or one whose column is of another type,
    is returned unchanged, for `build_frame_from_arrow` to refuse.
    """
    name = type_column(PLACE_ROLE)
    if name not in table.schema.names:
        return table
    index = table.schema.get_field_index(name)
    column = table.column(index).combine_chunks()
    if pa.types.is_null(column.type):
        return table.set_column(index, name, pa.array([UNTYPED] * len(column), pa.string()))
    if not pa.types.is_string(column.type) and not pa.types.is_large_string(column.type):
        return table
    empty = array_kernel(
        "or_kleene",
        array_kernel("is_null", column),
        array_kernel("equal", column, pa.scalar("", type=column.type)),
    )
    filled = array_kernel("if_else", empty, pa.scalar(UNTYPED, type=column.type), column)
    return table.set_column(index, name, filled)


_OPTIONAL_COLUMNS = frozenset({UNIT_COLUMN, *FACET_IDENTITY_COLUMNS})


def _with_null_optionals_as_text(table: pa.Table) -> pa.Table:
    """`table` with each optional column (`unit` and the facet identity columns) whose
    rows are all null, in a populated table, retyped as a null string column.

    `build_frame_from_arrow` accepts a text or null-typed optional column but refuses
    any other type, even when every cell is null; a writer that emits such a column
    as `float64`, a list of nulls, or a struct means "absent". Only a struct `unit`
    column keeps its type (it is the converted-unit type), and a zero-row column keeps
    its type too so a malformed type still refuses.
    """
    for index, field in enumerate(table.schema):
        column = table.column(index)
        if (
            field.name in _OPTIONAL_COLUMNS
            and len(column) > 0
            and column.null_count == len(column)
            and not (field.name == UNIT_COLUMN and pa.types.is_struct(field.type))
        ):
            table = table.set_column(index, field.name, pa.nulls(len(column), pa.string()))
    return table
