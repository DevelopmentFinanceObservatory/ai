"""Seed intake for a seeded turn: reading, checking and describing what Studio hands over.

A seed is a Search result the user carries into a turn. This module reads each
one as a `harness.wire.Seed` (`seeds_slot`, `take_seeds`), checks Search's
current answer for its identity (`resolve_seed`, `answered_block`), and
words the seeds a turn keeps and drops for the orchestrator's agent
(`seeds_hand_back`). Nothing here touches the ADK `Context` or routing;
`harness.graph` calls it from its nodes and `harness.studio` calls
`seeds_slot` before the session exists.
"""

from __future__ import annotations

import json
from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import Any, TypedDict

from compiler.surface import ToolSurface
from data_mcp.definition_store import IdentityMismatch
from data_mcp.definition_writes import IdentityUnusable, caller_definition
from planner.workflow import ResolvedGroup
from pydantic import ValidationError
from translator.discovery import (
    DefinitionNotResolvedError,
    PartFault,
    PartNotResolvedError,
    resolve_definition,
    resolved_spec,
)

from harness.wire import (
    Seed,
    SpecBlock,
)

# The tool name `seed`'s steps carry on the Activity feed. `seed` resolves
# each seed's identity again through data-mcp's `resolve_definition`
# (`translator.discovery.resolve_definition`), a call the feed does not show,
# and records its own steps through the executor's `record_step`.
SEED_STEP_TOOL = "receive_seeds"


# The step id that counts the seeds `seed` kept, and fails when it kept none
# and data-mcp or Search could not be reached for one.
SEEDS_STEP_ID = "seeds"


# The prefix of the step id a seed `seed` cannot read or resolve fails under,
# followed by the seed's position (`seed-0`, `seed-1`, ...).
SEED_STEP_PREFIX = "seed"


# The code a seed's own step fails with when Search's answer for its identity
# is not the dataset the user handed over: Search refuses the identity, no
# longer resolves it to a definite part, or resolves it to another spec_id.
SEED_DROPPED = "seed_dropped"


# The code a seed's own step fails with when an earlier usable seed names its spec_id.
SEED_DUPLICATE = "seed_duplicate"


# The code the `SEEDS_STEP_ID` step fails with when `seed_spec` is one
# `{spec_id, content_version}` reference, the shape the member was reserved
# with: it carries no Spec block to start from, so the turn uses the question.
SEED_REFERENCE_UNUSED = "seed_reference_unused"


def seed_step(
    record_step: Callable[..., None] | None,
    *,
    step_id: str,
    args: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Record one of `seed`'s steps, when the executor records steps."""
    if record_step is not None:
        record_step(tool=SEED_STEP_TOOL, args=args, result=result, step_id=step_id)


def refuse_seed(
    record_step: Callable[..., None] | None,
    index: int,
    *,
    code: str,
    detail: str,
    retry_mode: str = "do_not_retry",
) -> None:
    """Record seed `index`'s own step as failed with `code`."""
    seed_step(
        record_step,
        step_id=f"{SEED_STEP_PREFIX}-{index}",
        args={"seed": index},
        result={
            "outcome": "refusal",
            "refusal": {"code": code, "retry_mode": retry_mode, "fix": None, "message": detail},
        },
    )


def refuse_seed_reference(record_step: Callable[..., None] | None) -> None:
    """Record the `SEEDS_STEP_ID` step of a turn whose `seed_spec` is one reference, as unused."""
    seed_step(
        record_step,
        step_id=SEEDS_STEP_ID,
        args={"seeds": 0},
        result={
            "outcome": "refusal",
            "refusal": {
                "code": SEED_REFERENCE_UNUSED,
                "retry_mode": "do_not_retry",
                "fix": None,
                "message": (
                    "seed_spec holds one {spec_id, content_version} reference, which "
                    "carries no Spec block to start from, so the turn uses the question; "
                    "send a list of seeds to start from Search results"
                ),
            },
        },
    )


def seeds_received(
    record_step: Callable[..., None] | None,
    *,
    posted: int,
    taken: int,
    kept: int,
    unreachable: Sequence[DefinitionNotResolvedError] = (),
) -> None:
    """Record the `SEEDS_STEP_ID` step of a turn that posted `posted` seeds and kept `kept`.

    It fails `service_unavailable`, with the first of `unreachable`, when no
    seed was kept and data-mcp or Search could not be reached for one; it
    counts the seeds kept otherwise. `taken` is how many could be read.
    """
    if not kept and unreachable:
        result: dict[str, Any] = {
            "outcome": "refusal",
            "refusal": {
                "code": "service_unavailable",
                "retry_mode": "retry_after_delay",
                "fix": None,
                "message": f"data-mcp could not resolve the datasets: {unreachable[0]}",
            },
        }
    else:
        result = {
            "outcome": "ok",
            "frame_refs": [],
            "summary": f"received {kept} of {posted} dataset(s) from Search",
        }
    seed_step(record_step, step_id=SEEDS_STEP_ID, args={"seeds": taken}, result=result)


@dataclass(frozen=True, slots=True)
class TakenSeed:
    """One seed as the turn takes it from the browser.

    `index` is the seed's position in the seeds slot, which names its own
    step. `seed` holds what the turn keeps of it: the spec_id, the identity,
    which hashes to the spec_id, and the operations (`harness.wire.SeedSpec`).
    """

    index: int
    seed: Seed


class StaleSeedError(Exception):
    """Search's answer for a seed's identity is not the dataset the user handed over."""


def stale_reason(error: PartNotResolvedError) -> str:
    """Why Search's answer for a seed is not the seed's dataset, in Search's own words when it states one."""
    if error.fault is PartFault.ANOTHER_DATASET:
        named = f" ({error.answered})" if error.answered else ""
        return f"Search now resolves it to another dataset{named}"
    if error.fault is PartFault.NO_DEFINITION:
        return "Search's answer for it carries no definition to retrieve"
    part = error.part
    status = part.get("status")
    no_data = part.get("no_data")
    unsupported = part.get("unsupported")
    if isinstance(no_data, Mapping) and isinstance(no_data.get("reason"), str):
        return f"Search finds no data for it ({no_data['reason']})"
    if isinstance(unsupported, Mapping) and isinstance(unsupported.get("detail"), str):
        return f"Search no longer supports it: {unsupported['detail']}"
    return f"Search no longer resolves it to one dataset (it answers {status!r})"


def answered_block(taken: TakenSeed, part: Mapping[str, Any]) -> SpecBlock:
    """The Spec block of `part`, Search's answer for `taken`'s identity.

    Raises:
        StaleSeedError: If the part is not the dataset the seed names
            (`translator.discovery.resolved_spec`).
        ValidationError: If the spec does not parse as a `harness.wire.SpecBlock`.
    """
    try:
        spec = resolved_spec(part, spec_id=taken.seed.spec.spec_id)
    except PartNotResolvedError as exc:
        raise StaleSeedError(stale_reason(exc)) from exc
    return SpecBlock.model_validate(spec)


@dataclass(frozen=True, slots=True)
class _SeedRefusal:
    """Why a seed cannot be taken: the code and detail its own step fails with."""

    code: str
    detail: str


# The one member of a seeds-slot entry that stands in for a seed `seeds_slot`
# could not take, holding its `_SeedRefusal` as `{"code", "detail"}`.
_SEED_REFUSED = "refused"


def _read_seed(index: int, item: object) -> Seed | _SeedRefusal:
    """Seed `index` read as a `harness.wire.Seed`, or why it cannot be.

    Its identity not hashing to its spec_id is `identity_mismatch`; anything
    else is `seed_invalid`, with the reason.
    """
    try:
        seed = Seed.model_validate(item)
        caller_definition(seed.spec.definition.model_dump(mode="json"))
    except ValidationError as exc:
        error = exc.errors()[0]
        where = "/".join(str(part) for part in error["loc"])
        detail = f"seed {index} cannot be used: {where + ': ' if where else ''}{error['msg']}"
        return _SeedRefusal(code="seed_invalid", detail=detail)
    except IdentityMismatch as exc:
        return _SeedRefusal(code="identity_mismatch", detail=str(exc))
    except IdentityUnusable as exc:
        return _SeedRefusal(code="seed_invalid", detail=f"seed {index} cannot be used: {exc}")
    return seed


def seeds_slot(posted: Sequence[object]) -> list[dict[str, Any]]:
    """The seeds slot (`SEEDS_STATE_KEY`) for a turn that posted `posted`, one entry per seed.

    Each seed is read as `take_seeds` reads it (`_read_seed`). One it can
    take is kept as its `harness.wire.Seed`: the spec_id, the definition
    and the operations, and nothing else of the Spec block. One it cannot
    take is kept as the refusal its step fails with,
    under `_SEED_REFUSED`. A seed's position is its index in the slot, so
    `seed` refuses each at the index it was posted at, and no posted block
    enters session state.
    """
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(posted):
        read = _read_seed(index, item)
        if isinstance(read, _SeedRefusal):
            entries.append({_SEED_REFUSED: {"code": read.code, "detail": read.detail}})
        else:
            entries.append(read.model_dump(mode="json"))
    return entries


def _slot_seed(index: int, entry: Any) -> Seed | _SeedRefusal:
    """Seeds-slot entry `index` as `_read_seed` reads it, or the refusal `seeds_slot` kept for it."""
    refused = entry.get(_SEED_REFUSED) if isinstance(entry, dict) else None
    if isinstance(refused, dict):
        return _SeedRefusal(code=refused["code"], detail=refused["detail"])
    return _read_seed(index, entry)


def take_seeds(raw: object, *, record_step: Callable[..., None] | None) -> list[TakenSeed]:
    """The usable seeds in the seeds slot `raw`, each read as a `harness.wire.Seed`.

    A seed that cannot be read fails its own `SEED_STEP_PREFIX-<n>` step
    with its `_SeedRefusal` and is left out; the others stand. A seed naming
    the spec_id of an earlier usable one fails as `SEED_DUPLICATE`, whatever
    else it carries, so one spec_id is one seed in the search state the turn
    stores (`harness.graph.state.store_search`).
    """
    taken: list[TakenSeed] = []
    first_at: dict[str, int] = {}
    for index, item in enumerate(raw if isinstance(raw, list) else []):
        seed = _slot_seed(index, item)
        if isinstance(seed, _SeedRefusal):
            refuse_seed(record_step, index, code=seed.code, detail=seed.detail)
            continue
        spec_id = seed.spec.spec_id
        if spec_id in first_at:
            refuse_seed(
                record_step,
                index,
                code=SEED_DUPLICATE,
                detail=(
                    f"seed {index} names the dataset seed {first_at[spec_id]} names "
                    f"({spec_id}); a dataset is handed over once, and the first stands"
                ),
            )
            continue
        first_at[spec_id] = index
        taken.append(TakenSeed(index=index, seed=seed))
    return taken


async def resolve_seed(surface: ToolSurface, taken: TakenSeed) -> Mapping[str, Any]:
    """The part Search resolves `taken`'s identity to now (`translator.discovery.resolve_definition`)."""
    spec = taken.seed.spec
    return await resolve_definition(
        surface,
        spec_id=spec.spec_id,
        identity=spec.definition.identity,
        groups=tuple(
            ResolvedGroup(
                role=role,
                label=group.label,
                ref=group.ref,
                total=group.total,
                whole_of=group.whole_of,
            )
            for role, group in (spec.entity_groups or {}).items()
        ),
    )


class DroppedSeed(TypedDict):
    """A seed the turn dropped, as the user is told of it.

    `name` is how the user is told of the seed. `reason` is why it was
    dropped, in Search's terms when Search gave one (`stale_reason`).
    """

    name: str
    reason: str


def seeds_hand_back(
    parts: Sequence[Mapping[str, Any]],
    drops: Sequence[DroppedSeed],
    *,
    budgets: Mapping[str, int],
) -> str:
    """What the orchestrator's agent reads of the seeds its turn starts from.

    `parts` are the seeds kept, as `search` returns its parts, each with the
    operations the user's question asked of it. `drops` are the seeds left
    out, each with why. With no seed kept, the turn goes on from the question.
    `budgets` are the turn's tool budgets (`harness.graph.state.turn_budgets`): the
    hand-back names the tools the turn can still call with a seed, and how
    many datasets it can fetch when that is fewer than the seeds kept.
    """
    lines = []
    if parts:
        if budgets["plan"]:
            lead = (
                "The user brought these datasets from Search: fetch or plan each with its "
                "spec_id. A part's operations are what the user's question asked of it; ask "
                "for them with plan."
            )
        else:
            lead = (
                "The user brought these datasets from Search: fetch each with its spec_id. "
                "A part's operations are what the user's question asked of it; they cannot "
                "run here, so tell the user the data comes as served."
            )
        fetches = budgets["fetch"]
        if fetches < len(parts):
            lead += (
                " fetch can run once this turn, so fetch the one dataset the question most needs."
                if fetches == 1
                else f" fetch can run {fetches} times this turn, so fetch the {fetches} datasets "
                "the question most needs."
            )
        lines += [lead, json.dumps({"parts": list(parts)})]
    lines += [
        f'"{drop["name"]}" was left out: {drop["reason"]}. Tell the user, and say that '
        "searching for it again may find it."
        for drop in drops
    ]
    if not parts:
        lines.append("No dataset from Search can be used, so answer from the question.")
    return "\n".join(lines)
