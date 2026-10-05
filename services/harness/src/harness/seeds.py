"""Seed intake for a seeded turn: reading, checking and describing what Studio hands over.

A seed is a Search result the user carries into a turn. This module reads each
one as a `harness.wire.Seed` (`seeds_slot`, `take_seeds`), checks Search's
current answer for its identity (`read_answer`, `resolve_seed`), and words the
seeds a turn drops (`dropped_seeds_note`). Nothing here touches the ADK
`Context` or routing; `harness.graph` calls it from its nodes and
`harness.studio` calls `seeds_slot` before the session exists.
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

from compiler.sources import source_from_spec_block
from compiler.surface import ToolSurface
from data_mcp.definition_store import IdentityMismatch
from data_mcp.definition_writes import IdentityUnusable, caller_definition
from executor.classify import (
    ToolFailure,
)
from executor.outcomes import (
    MethodChoiceNeeded,
    RediscoveryNeeded,
    Refused,
)
from planner.plan import (
    SeededSource,
    seed_of,
)
from planner.workflow import Workflow as PlanWorkflow
from pydantic import ValidationError
from translator.discovery import (
    DefinitionNotResolvedError,
    PartFault,
    PartNotResolvedError,
    compact_source,
    resolve_definition,
    resolved_spec,
)

from harness.tool_contracts import (
    graph_envelope,
)
from harness.wire import (
    MAX_SEED_RECORD_CHARS,
    MAX_SEED_SOURCE_CHARS,
    Seed,
    SeedOperation,
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


# The code a seed's own step fails with on the `download` surface when an
# earlier seed is the one the download retrieves.
ONE_DATASET_PER_DOWNLOAD = "one_dataset_per_download"


# The code a seed's own step fails with when an earlier usable seed names its spec_id.
SEED_DUPLICATE = "seed_duplicate"


# The code the `SEEDS_STEP_ID` step fails with when `seed_spec` is one
# `{spec_id, content_version}` reference, the shape the member was reserved
# with: it carries no Spec block to start from, so the turn uses the question.
SEED_REFERENCE_UNUSED = "seed_reference_unused"


# The code the `seed-<n>-operations` step fails with on the `download`
# surface: the seed is retrieved and the operations Search left over for it
# are not run.
NO_OPERATIONS_ON_DOWNLOAD = "no_operations_on_download"


def _seed_sentence(block: SpecBlock) -> str | None:
    """A Spec block's reading in Search's own words."""
    sentence = block.resolved.get("rendered_sentence")
    return sentence if isinstance(sentence, str) and sentence else None


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


def _seen_year(block: SpecBlock) -> int | None:
    """The year whose figures Search showed, when Search chose it and the question did not.

    With no year named, Search resolves over every year with data
    (`requested_window` kind `all`) and shows the latest year's figures
    (`resolved.headline`, on a role-bearing answer). `None` when the question
    named its years, or the block states no such headline.
    """
    headline = block.resolved.get("headline")
    requested = block.definition.identity.get("requested_window")
    if not isinstance(headline, dict) or headline.get("year_named") is not False:
        return None
    if not isinstance(requested, dict) or requested.get("kind") != "all":
        return None
    year = headline.get("year")
    return year if isinstance(year, int) and not isinstance(year, bool) else None


def _seen_rows(block: SpecBlock) -> int | None:
    """How many observations Search's headline counted in the year it showed."""
    headline = block.resolved.get("headline")
    rows = headline.get("observations") if isinstance(headline, dict) else None
    return rows if isinstance(rows, int) and not isinstance(rows, bool) else None


def _window_text(identity: Mapping[str, Any] | None) -> str:
    """The years an identity asks for, in words, for the planner."""
    window = (identity or {}).get("requested_window") or {}
    kind = window.get("kind")
    if kind == "latest":
        return "the latest observation"
    start, end = window.get("start_year"), window.get("end_year")
    if kind != "range" or (start is None and end is None):
        return "every year with data"
    if start == end:
        return str(start)
    if end is None:
        return f"{start} onwards"
    if start is None:
        return f"up to {end}"
    return f"{start} to {end}"


def _coverage_text(block: SpecBlock) -> str | None:
    """What a Spec block's coverage counts, in Search's own labels ("199 recipients, 1 sources")."""
    coverage = block.resolved.get("coverage")
    dimensions = coverage.get("dimensions") if isinstance(coverage, dict) else None
    counts = [
        f"{dimension['count']:,} {dimension['label']}"
        for dimension in dimensions or []
        if isinstance(dimension, dict)
        and isinstance(dimension.get("count"), int)
        and isinstance(dimension.get("label"), str)
    ]
    return ", ".join(counts) or None


# The members of a `graph_envelope` reading that describe one identity: its
# width, the facts that verdict rests on, and the entity axes it leaves open.
# A seeded envelope states them on each seed and never at its top level, where
# they would read as every seed's.
_SEED_FACT_KEYS = (
    "wide",
    "stat_var_shape_kind",
    "stat_var_item_count",
    "entity_roles",
    "stat_var_truncated",
    "open_entity_axes",
)


def _envelope_seed(
    block: SpecBlock, *, operations: Sequence[SeedOperation], reading: Mapping[str, Any]
) -> dict[str, Any]:
    """One seed as the planner reads it (`harness.planner_seam._render_seeds`).

    `block` is Search's own answer for the seed's identity and `operations`
    the seed's. `reading` is the block's `graph_envelope` reading, the source
    of its `_SEED_FACT_KEYS` and of the `choice_points` and `warnings` the
    envelope adds up (`seeded_envelope`). Its `open_entity_axes` name the
    entity axes the dataset leaves open: an operation over a party runs over
    one of them.
    """
    year, rows = _seen_year(block), _seen_rows(block)
    showed = None
    if year is not None:
        showed = f"{year}" + (f", {rows:,} rows" if rows is not None else "")
    return {
        "spec_id": block.spec_id,
        "sentence": _seed_sentence(block),
        "window": _window_text(block.definition.identity),
        "showed": showed,
        "coverage": _coverage_text(block),
        **{key: reading.get(key) for key in _SEED_FACT_KEYS},
        "choice_points": int(reading.get("choice_points", 0)),
        "warnings": list(reading.get("warnings") or []),
        "operations": [
            operation.model_dump(mode="json", exclude_none=True) for operation in operations
        ],
    }


def seeded_envelope(seeds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The resolve envelope of a turn that starts from `seeds`, each as `_envelope_seed` reads it.

    The top level holds what is true of the turn: `spec_id` is the first
    seed's, the one `fetch` retrieves when no plan runs; `choice_points` and
    `warnings` are every seed's together, and `wide` is whether any seed is,
    which is how `choose_depth` reads them. What describes one identity
    (`_SEED_FACT_KEYS`, and a reading's coverage `window`) is on its seed alone.
    `_drop_seeds_from_turn` builds the envelope again from the seeds left.
    """
    return {
        "status": "definite",
        "spec_id": seeds[0]["spec_id"],
        "choice_points": sum(int(seed.get("choice_points", 0)) for seed in seeds),
        "parts": len(seeds),
        "wide": any(bool(seed.get("wide", False)) for seed in seeds),
        "warnings": [warning for seed in seeds for warning in seed.get("warnings") or []],
        "candidates": [],
        "seeds": list(seeds),
    }


@dataclass(frozen=True, slots=True)
class TakenSeed:
    """One seed as the turn takes it from the browser.

    `index` is the seed's position in the seeds slot, which names its own
    step. `seed` holds what the turn keeps of it: the spec_id, the identity,
    which hashes to the spec_id, the dcids of the StatVars Search listed, and
    the operations (`harness.wire.SeedSpec`).
    """

    index: int
    seed: Seed


@dataclass(frozen=True, slots=True)
class ReadSeed:
    """One seed `seed` can use, read once from Search's own answer for its identity.

    `index` is the seed's position in the seeds slot, which names its own
    step. `source` is the dataset a plan pins, with the StatVars Search
    elected, which every `get_data` of the seed posts, `reading` what
    `graph_envelope` reads of the answer as a definite one, and `planned` the
    seed as the planner reads it (`_envelope_seed`). Nothing later reads the
    answer itself.
    """

    index: int
    source: SeededSource
    reading: dict[str, Any]
    planned: dict[str, Any]


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


def read_answer(taken: TakenSeed, part: Mapping[str, Any]) -> ReadSeed:
    """Read `part`, Search's answer for `taken`'s identity, into all `ReadSeed` holds.

    Raises:
        StaleSeedError: If the part is not the dataset the seed names
            (`translator.discovery.resolved_spec`).
        ValidationError: If the spec does not parse as a `harness.wire.SpecBlock`.
        ValueError: If what the seed puts into a tool result or into session
            state is past `MAX_SEED_SOURCE_CHARS` or `MAX_SEED_RECORD_CHARS`,
            or the block is not usable as a source
            (`compiler.sources.source_from_spec_block`).
    """
    try:
        spec = resolved_spec(part, spec_id=taken.seed.spec.spec_id)
    except PartNotResolvedError as exc:
        raise StaleSeedError(stale_reason(exc)) from exc
    block = SpecBlock.model_validate(spec)
    served = block.model_dump(mode="json")
    reading = graph_envelope({"parts": [{"status": "definite", "spec": served}]})
    source = SeededSource(source=source_from_spec_block(served), seen_year=_seen_year(block))
    described = len(json.dumps(compact_source(source.source)))
    if described > MAX_SEED_SOURCE_CHARS:
        raise ValueError(
            f"its sentence, StatVars and entities take {described:,} characters to "
            f"describe, past the cap of {MAX_SEED_SOURCE_CHARS:,}"
        )
    # As `planner.plan.save_seeded_sources` writes it.
    stored = len(json.dumps(source.model_dump(mode="json")))
    if stored > MAX_SEED_RECORD_CHARS:
        raise ValueError(
            f"its record takes {stored:,} characters, past the cap of {MAX_SEED_RECORD_CHARS:,}"
        )
    return ReadSeed(
        index=taken.index,
        source=source,
        reading=reading,
        planned=_envelope_seed(block, operations=taken.seed.operations, reading=reading),
    )


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
    take is kept as its `harness.wire.Seed`: the spec_id, the definition,
    the StatVar dcids and the operations, and nothing else of the Spec
    block. One it cannot take is kept as the refusal its step fails with,
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
    else it carries, so one spec_id is one seed for every reader that looks a
    seed up by it (`planner.plan.seeded_source`).
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
        stat_vars=spec.stat_vars,
    )


def candidate_stat_vars(candidates: list[dict[str, Any]], spec_id: str | None) -> str:
    """The StatVars `search` elected for the chosen candidate
    (`harness.tool_contracts.graph_envelope`'s `candidates` branch lists them
    per offered spec), carried forward the same way `_candidate_wide` carries
    `wide`: the re-entry resolve searches nothing, so `fetch` has no block to
    read them from. Returned as the JSON `GraphState.spec_id_stat_vars`
    holds, `{"spec_id": ..., "stat_vars": [...]}`: the list names the
    candidate it belongs to, so `_chosen_stat_vars` posts it for that
    `spec_id` and no other."""
    stat_vars = next(
        (
            [dcid for dcid in c.get("stat_vars") or [] if isinstance(dcid, str)]
            for c in candidates
            if c.get("spec_id") == spec_id
        ),
        [],
    )
    return json.dumps({"spec_id": spec_id, "stat_vars": stat_vars})


# The `fix` key of `get_data`'s `spec_not_exportable` refusal that names each StatVar
# the call posted in `resolved_stat_var_dcids` that is outside the identity's family
# (dfo-apis `member_not_resolvable`, `data_mcp.tool_errors._member_not_resolvable_refusal`).
_NOT_MEMBERS_FIX_KEY = "not_members"


def dropped_seed_shape(*, step_id: str | None, seed: str, failure: ToolFailure) -> dict[str, Any]:
    """A stop on a seed the turn drops, as the planner reads it on the replan.

    `spec_id` names the seed as the planner was shown it, where a
    `RediscoveryNeeded` names its source, so `harness.planner_seam` tells the
    planner not to propose it again: the seed has left the turn's seeded
    sources (`_drop_seeds_from_turn`), so nothing maps it back to a seed to
    keep pinned. The message says why and what to plan instead, and that the
    part of the question the seed answered is left out, which the planner's
    brief would otherwise have it plan as a part no seed covers.
    `propose_plan` and `discover_source` refuse the dataset for the rest of
    the turn (`planner.plan.dropped_seeds`).
    """
    named = failure.fix.get(_NOT_MEMBERS_FIX_KEY)
    not_members = ", ".join(str(dcid) for dcid in named) if isinstance(named, list) else ""
    return {
        "kind": "seed_dropped",
        "step_id": step_id,
        "spec_id": seed,
        "refusal": {
            "code": failure.code,
            "message": (
                f"the StatVars Search elected for the seeded dataset {seed!r} are no longer "
                f"members of its dataset ({not_members}). The user is told that this plan "
                "leaves it out and to search for it again, so it is dropped from this turn. "
                "Do not plan that dataset again in this turn: do not pin it, and do not add "
                "an unpinned requirement for the part of the question it answered, since a "
                "search for that part finds the same dataset and is refused. Plan from the "
                "other seeded datasets and leave that part out."
            ),
        },
    }


def stale_seeds(
    outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused,
    *,
    workflow: PlanWorkflow | None,
    seeded: Sequence[SeededSource],
) -> dict[str, str]:
    """The seeds `outcome`'s wave found stale: each seed's spec_id, by the id of the step that found it.

    A step finds its seed stale when `get_data` refused it naming
    `_NOT_MEMBERS_FIX_KEY`: StatVars Search elected for the seed, which every
    retrieval of the seed posts, have left its dataset. The step's evidence
    binding names the spec_id it retrieved, the seed's own or its copy over a
    plan's period (`SeededSource.windowed_from`), and either maps to the seed.
    """
    if workflow is None:
        return {}
    seeds = {item.source.spec_id: seed_of(item) for item in seeded}
    retrieved = {binding.step_id: binding.spec_id for binding in workflow.evidence_bindings}
    stale: dict[str, str] = {}
    for stop in (outcome, *outcome.also_stopped):
        if not isinstance(stop, Refused) or stop.payload is None or stop.step_id is None:
            continue
        if not stop.payload.fix.get(_NOT_MEMBERS_FIX_KEY):
            continue
        seed = seeds.get(retrieved.get(stop.step_id, ""))
        if seed is not None:
            stale[stop.step_id] = seed
    return stale


class DroppedSeed(TypedDict):
    """A seed the turn dropped, as the user is told of it.

    `name` is the seed as the user saw it named. `reason` is why it was
    dropped as it was taken (`read_answer`, in Search's terms); `None` for
    a seed whose StatVars left its dataset, which `_left_dataset_clause`
    words for any number of seeds.
    """

    name: str
    reason: str | None


def seed_name(seed: Mapping[str, Any]) -> str:
    """A seed of the resolve envelope's `seeds` as the user saw it named in Search."""
    return seed.get("sentence") or seed["spec_id"]


def _quoted_list(names: Sequence[str]) -> str:
    """`names`, each in quotes, as one English list."""
    quoted = [f'"{name}"' for name in names]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + ("," if len(quoted) > 2 else "") + f" and {quoted[-1]}"


def _left_dataset_clause(count: int) -> str:
    """Why `count` seeds whose StatVars left their datasets were dropped, with the pronoun their number takes."""
    if count == 1:
        return "the variables Search listed for it are no longer part of its dataset"
    return "the variables Search listed for them are no longer part of their datasets"


def _drop_reasons(drops: Sequence[DroppedSeed]) -> list[tuple[list[str], str]]:
    """`drops` as (names, why) pairs: every seed whose StatVars left its dataset
    under one clause, then each seed dropped as it was taken under its own."""
    stale = [drop["name"] for drop in drops if drop["reason"] is None]
    pairs = [(stale, _left_dataset_clause(len(stale)))] if stale else []
    return pairs + [([drop["name"]], drop["reason"]) for drop in drops if drop["reason"]]


def _again(drops: Sequence[DroppedSeed]) -> str:
    return "it" if len(drops) == 1 else "each"


def dropped_seeds_note(drops: Sequence[DroppedSeed]) -> str:
    """The sentence telling the user which seeds a plan leaves out, and why."""
    left_out = "; ".join(f"{_quoted_list(names)}: {why}" for names, why in _drop_reasons(drops))
    return f"This plan leaves out {left_out}. Search for {_again(drops)} again to add it."


def dropped_seeds_text(drops: Sequence[DroppedSeed]) -> str:
    """The sentences telling the user, when no plan is drafted, which seeds were dropped and why."""
    dropped = " ".join(
        f"{_quoted_list(names)} could not be retrieved: {why}."
        for names, why in _drop_reasons(drops)
    )
    return f"{dropped} Search for {_again(drops)} again."


def unplanned_seeds_text(drops: Sequence[DroppedSeed], kept: Sequence[str], reason: str) -> str:
    """The closing text of a turn that dropped seeds and then drafted no plan from the rest.

    Names the dropped seeds and why they could not be retrieved, then says
    that the `kept` ones were not planned, with the planner's own `reason`.
    """
    unplanned = (
        f"No plan was drafted for {_quoted_list(kept)}, so nothing is shown for "
        f"{'it' if len(kept) == 1 else 'them'}."
        if kept
        else "No plan was drafted for the rest of the question."
    )
    return " ".join(part for part in (dropped_seeds_text(drops), unplanned, reason) if part)
