"""The `switch` node: a scope choice on the plan card moved to another option, in code.

The `approve` node routes an answer carrying `switch` here. The node reads
the decision the user switched and every decision the card lists with it
(`harness.plan_display.listed_together`), and for each the source it pins
and that source's stored scope (`harness.turn_records.SearchReading.scope`).
It posts the scope's base identity with its entries and the switch
(`harness.scope_choices.switch_entries`) through data-mcp's
`resolve_definition` (`translator.discovery.resolve_definition`), which
registers the identity they compile to. That answer is the switched pin.
The plan pins it, and its reading and pin source state the dataset over
the years the question asked for. Admission finds it on record
(`planner.plan.definitions_on_record`), so where the plan reads it over a
window of its own, admission copies it over that window
(`submission.service._window_pins`) and the translation resolves the copy.
The turn keeps Search's answers
(`translator.discovery.resolving_once`), so a switch repeating one the
turn made posts nothing. The
plan is then revised by the planner seam with no model call
(`harness.graph.seams.PlannerModel.revise`): each requirement pinning a
switched source pins the compiled one, the decisions follow their pins,
and the plan compiles again. The node stores each new reading, so later
searches carry the switch, and offers the revised plan back to `approve`
with what else moved (`harness.scope_choices.moved_choices`).

The switch reads each requirement as the planner states it
(`submission.service.statement`): an input alignment moved onto another
pin's compile (`planner.plan.AlignedInput`) reads the pin it states. A switch
of that pin's row moves it with every other requirement stating the pin, and
admission aligns it again from the switched pin. The row of an axis alignment
moved for one input (`scope_<axis>_of_<requirement id>`) posts the compile's
scope (`planner.plan.compiled_pin`) and moves that input alone: the user
takes it over, so it pins the switched dataset as its own. Its words, roles
and every other choice stay those of the pin it states. Switched back to the
option its own pin counts, the input keeps that pin, since the user's pick
states the axis and alignment leaves a stated axis alone.

Each requirement the switch re-pins drops the source it stated
(`source`, `source_ref`, `source_exclude`, and `source_code` unless the new
dataset is exactly that series): the user chose the dataset, so its pin is
the requirement's identity, and text left behind would have discovery,
compile and the run's evidence check judge the new dataset against the old
one's name.

The card names a choice by its row's id (`harness.plan_display.choice_id`),
which every switch of the conversation keeps.

A switch Search refuses (`requested_option_unavailable`,
`conflicting_requested_options`), or one that fails anywhere else, offers
the unchanged plan again with `switch_failed` naming the option, the
refusal's code, one sentence saying why (`harness.sentences.
refusal_reason` for a refused revision) and whether the same switch can
succeed if asked again (`retryable`), and leaves the plan, the planner's
drafting state and the turn's readings as they were (`_RESTORED_KEYS`). A
switch the user stops (`SWITCH_STOP`, which the studio sets on a Stop
while the switch runs) ends the same way under `SWITCH_CANCELLED`, so the
card stays answerable.

A chip under an answer applies one option the same way, with no plan
(`apply_stored_choice`): the `seed` node posts the stored reading's base
identity with the switched entries, stores the reading Search compiles,
fetches it through the orchestrator's `fetch` tool and hands the result to
the agent, which answers from it with no search.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypedDict, cast

from compiler.sources import readable_wire_groups
from compiler.surface import ToolSurface
from data_mcp.scope import ScopeEntry
from google.adk import Context, Event
from google.adk.workflow._errors import NodeInterruptedError
from planner.agent import run_together
from planner.plan import (
    AlignedInput,
    DatasetRequirement,
    Decision,
    PinSource,
    SessionState,
    alignments,
    compiled_pin,
    definitions_on_record,
    earlier_block,
    exact_codes,
    identity_on_record,
    is_scope_choice,
    pin_sources,
    plan_revision,
    repinned,
    save_pin_sources,
    windowed_pins,
)
from planner.plan import Plan as PlannerPlan
from planner.tool_results import ProblemCode
from planner.workflow import ResolvedGroup
from pydantic import ValidationError
from submission.service import part_scope, statement
from translator.discovery import (
    RESOLVED_STATE_PREFIX,
    DefinitionNotResolvedError,
    resolve_definition,
    resolving_once,
)

from harness.decision_pick import picks_the_chosen_option
from harness.graph.seams import PlannerModel
from harness.graph.state import (
    ORCHESTRATOR_APPLIED_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPEC_WARNINGS_KEY,
    ORCHESTRATOR_SPECS_KEY,
    conversation_readings,
    readable_plan,
    store_search,
)
from harness.plan_display import choice_id as row_choice_id
from harness.plan_display import chosen_scope_decision, listed_together, pinned_choice_id
from harness.scope_choices import (
    PartScope,
    ScopeChoice,
    mark_planner_axes,
    moved_choices,
    planner_axes,
    switch_entries,
)
from harness.sentences import SeamCode
from harness.tool_contracts import resolver_digest
from harness.turn_records import CHIP_STALE_NOTE, SearchReading, stored_choice

logger = logging.getLogger(__name__)

# The codes Search refuses a switch it cannot apply with (422).
_OPTION_UNAVAILABLE = "requested_option_unavailable"
_SCOPE_REFUSALS = frozenset({_OPTION_UNAVAILABLE, "conflicting_requested_options"})

# The revision refusals a later switch to the same option can escape: the
# request ran out of wall-clock time, a search could not reach Search, or a
# compile could not reach the calculation service. Every other refusal of a
# revision, which calls no model, repeats.
_RETRYABLE_REVISION_CODES = frozenset(
    {
        SeamCode.TURN_WALL_TIME_SPENT.value,
        SeamCode.DISCOVERY_UNAVAILABLE.value,
        ProblemCode.SURFACE_UNAVAILABLE.value,
    }
)

# The state the planner seam and the translator keep while drafting, which a
# revision that fails leaves as it found. The parts Search answered this turn
# (`RESOLVED_STATE_PREFIX`) are not drafting state. Each holds whatever the
# switch's outcome, so a failed switch keeps them.
_DRAFTING_PREFIXES = ("planner:", "translator:")
# The other state a revision writes, which a switch that fails or is stopped
# puts back too: the plan, and the readings `store_search` stores for the
# switched sources and for the datasets the revision searches
# (`harness.planner_seam.store_searched_pin_readings`), with their blocks. Each
# maps to the value it is put back as when unset, since the studio reads the
# readings and blocks as a list and a mapping (`harness.studio`).
_RESTORED_KEYS: Mapping[str, Any] = {
    "plan": None,
    ORCHESTRATOR_SEARCHES_KEY: [],
    ORCHESTRATOR_SPECS_KEY: {},
    ORCHESTRATOR_SPEC_WARNINGS_KEY: {},
}


# The tool name and step id the Activity feed shows a chip's application under.
APPLY_CHOICE_STEP_TOOL = "apply_choice"
# The code that step fails with when the chip cannot be applied; its message
# is the note `SwitchFailedError` carries.
SCOPE_CHOICE_NOT_APPLIED = "scope_choice_not_applied"
# The code of a switch the user stopped while it ran.
SWITCH_CANCELLED = "switch_cancelled"

# Set by the studio when the user stops the stream a plan-card switch runs
# on (`harness.studio`): the `switch` node then abandons the revision and
# offers the plan unchanged. `None` outside such a stream.
SWITCH_STOP: ContextVar[asyncio.Event | None] = ContextVar("switch_stop", default=None)


class _Pick(TypedDict):
    """The option a chip picks: the choice and the option, as its step records them."""

    choice_id: str
    option_id: str


class SwitchFailedError(Exception):
    """A switch or chip that cannot be applied: `code` is the refusal's code,
    `reason`, the message, the one sentence saying why that the card shows
    under the choice, and `retryable` whether the same switch can succeed if
    asked again."""

    def __init__(self, reason: str, *, code: str, retryable: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class _Target:
    """What one decision of the switched row moves: the pin the planner states
    for it, the newest reading of that pin, the scope the switch posts from and
    its choice on the switched axis.

    `moved_for` is `None` for a pin's own scope decision, which every
    requirement stating `pin` follows. For an input alignment moved onto a
    compile (`planner.plan.AlignedPin`), the decision on an axis it moved is
    that requirement's alone: `moved_for` is its id, `scope` the compile's,
    and `block` the compile's spec block.
    """

    pin: str
    reading: SearchReading
    scope: PartScope
    choice: ScopeChoice
    moved_for: str | None = None
    block: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _Switched:
    """One source the switch moves: the pin it held, its reading and scope,
    the answer Search gave, and the spec_id and scope that answer states.
    `moved_for` names the one requirement it moves, `None` when it moves every
    requirement stating `pin`, and `block` is the compile's spec block for one
    it moves (`_Target`)."""

    pin: str
    reading: SearchReading
    scope: PartScope
    envelope: dict[str, Any]
    spec_id: str
    after: PartScope
    moved_for: str | None = None
    block: Mapping[str, Any] | None = None


def switch_node(
    *, planner: PlannerModel, definition_surface: ToolSurface
) -> Callable[[Context, Any], AsyncIterator[Any]]:
    """The `switch` node, revising with `planner` and posting through `definition_surface`."""

    async def switch(ctx: Context, node_input: Any) -> AsyncIterator[Any]:
        """Apply the switch `node_input` carries (`{"switch": {"choice_id", "option_id"}}`) and offer the plan again.

        One route out, `"offer"`, to the `approve` node: the revised plan
        with a new revision and `moved`, or the unchanged plan with a
        `switch_failed`, which the plan card shows on the row the user
        switched (`harness.studio_translate`), so no chat message repeats it.
        """
        pick = node_input.get("switch") if isinstance(node_input, dict) else None
        pick = pick if isinstance(pick, dict) else {}
        plan = readable_plan(ctx, ctx.state.get("plan"))
        assert plan is not None, "approve routes a switch only for a plan it read"
        choice_id, option_id = str(pick.get("choice_id")), str(pick.get("option_id"))
        # The state as it stood, which a switch that fails, is stopped or
        # ends on any other error leaves as it found.
        before = _restorable_state(ctx)
        try:
            # A `resolve_definition` call this turn already made is served its
            # answer (`translator.discovery.resolving_once`).
            with resolving_once(cast(SessionState, ctx.state)):
                output = await _unless_stopped(
                    ctx,
                    _switched_plan(
                        ctx,
                        plan,
                        choice_id=choice_id,
                        option_id=option_id,
                        planner=planner,
                        surface=definition_surface,
                    ),
                )
        except SwitchFailedError as exc:
            _restore(ctx, before)
            output = switch_failed_offer(plan, choice_id=choice_id, option_id=option_id, error=exc)
        except BaseException:
            _restore(ctx, before)
            raise
        yield Event(output=output, route="offer")

    return switch


def switch_failed_offer(
    plan: PlannerPlan, *, choice_id: str, option_id: str, error: SwitchFailedError
) -> dict[str, Any]:
    """The `approve` payload offering `plan` unchanged after the switch of row
    `choice_id` to option `option_id` failed with `error`: `switch_failed`
    names the row, the option, the code, the sentence saying why and
    whether the same switch can succeed if asked again."""
    logger.info(
        "plan switch not applied: choice_id=%s option_id=%s code=%s",
        choice_id,
        option_id,
        error.code,
    )
    return {
        **plan.model_dump(mode="json"),
        "summary": error.reason,
        "revision": plan_revision(plan),
        "switch_failed": {
            "choice_id": choice_id,
            "option_id": option_id,
            "code": error.code,
            "reason": error.reason,
            "retryable": error.retryable,
        },
    }


async def _unless_stopped(ctx: Context, revision: Awaitable[dict[str, Any]]) -> dict[str, Any]:
    """`revision`'s payload, unless the user stops the switch first (`SWITCH_STOP`).

    A stopped revision is cancelled and awaited without taking its outcome, so
    a revision that ends in any way once the Stop came, an error its cleanup
    raises included, ends stopped, as does one that finishes in the same wait
    as the Stop. The `switch` node puts the state it wrote back, so the plan
    stands unchanged. A cancellation of the node itself,
    while the revision unwinds, still cancels the node. A revision that
    paused on a question to the user (`NodeInterruptedError`) stays paused
    though the Stop came in the same wait: ADK has streamed the question, so
    the node run ends as that pause and offers no plan beside it.

    Raises:
        SwitchFailedError: Under `SWITCH_CANCELLED` when the user stopped
            the switch, else as `revision` raises it.
        NodeInterruptedError: When `revision` paused on a question.
    """
    stop = SWITCH_STOP.get()
    running = asyncio.ensure_future(revision)
    stopped = asyncio.ensure_future(stop.wait()) if stop is not None else None
    finished: set[asyncio.Future[Any]] = set()
    try:
        finished, _ = await asyncio.wait(
            {running, *([stopped] if stopped is not None else [])},
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        unfinished = {
            waiting for waiting in (running, stopped) if waiting is not None and not waiting.done()
        }
        for waiting in unfinished:
            waiting.cancel()
        if unfinished:
            # Raises nothing the tasks raise. A cancellation of the node raises.
            await asyncio.wait(unfinished)
    if running in finished and stopped not in finished and not running.cancelled():
        return running.result()
    error = None if running.cancelled() else running.exception()
    if isinstance(error, NodeInterruptedError):
        raise error
    if error is not None:
        logger.info("stopped plan switch raised: %r", error)
    raise SwitchFailedError(
        "You stopped this change, so the plan is as it was.",
        code=SWITCH_CANCELLED,
        retryable=True,
    )


async def _switched_plan(
    ctx: Context,
    plan: PlannerPlan,
    *,
    choice_id: str,
    option_id: str,
    planner: PlannerModel,
    surface: ToolSurface,
) -> dict[str, Any]:
    """The `approve` payload of `plan` revised for the switch.

    Raises:
        SwitchFailedError: If the switch cannot be applied, with the note to show.
    """
    if picks_the_chosen_option(plan, choice_id=choice_id, option_id=option_id):
        raise SwitchFailedError(
            "That choice is already on the plan.", code=SCOPE_CHOICE_NOT_APPLIED
        )
    clicked = chosen_scope_decision(plan, choice_id)
    if clicked is None or clicked.scope is None:
        raise SwitchFailedError(
            "That choice is no longer on the plan.", code=SCOPE_CHOICE_NOT_APPLIED
        )
    axis = clicked.scope.axis
    state = cast(SessionState, ctx.state)
    to_origin = {copy.spec_id: copy.windowed_from for copy in windowed_pins(state).values()}
    held = alignments(state)
    readings = _readings(ctx)
    targets = [
        _target(plan, decision, state=state, held=held, to_origin=to_origin, readings=readings)
        for decision in listed_together(plan, clicked.slot)
    ]
    label = next(
        (option.label for option in targets[0].choice.options if option.option_id == option_id),
        option_id,
    )
    if any(not any(o.option_id == option_id for o in t.choice.options) for t in targets):
        raise SwitchFailedError(
            f"Search does not offer {label} for this source.", code=_OPTION_UNAVAILABLE
        )
    specs = ctx.state.get(ORCHESTRATOR_SPECS_KEY) or {}
    sources = pin_sources(state)
    switched = await run_together(
        _resubmit(
            surface,
            target=target,
            entries=switch_entries(target.scope, choice=target.choice, option_id=option_id),
            groups=_groups(
                target.reading,
                target.block
                if target.moved_for is not None
                else specs.get(target.pin) or _source_block(sources.get(target.pin), target.pin),
                pin=target.pin,
            ),
            label=label,
        )
        for target in targets
    )
    digests = [_reading_digest(item, axis=axis) for item in switched]
    save_pin_sources(
        state,
        {
            **sources,
            **{
                item.spec_id: _pin_source(sources.get(item.pin), digest, item.reading)
                for item, digest in zip(switched, digests, strict=True)
            },
        },
    )
    shown = pin_sources(state)
    on_record = {
        item.spec_id: recorded
        for item in switched
        if (recorded := identity_on_record(item.spec_id, shown=shown, end_open=True)) is not None
    }
    codes = {
        item.spec_id: _exact_codes(digest) for item, digest in zip(switched, digests, strict=True)
    }
    # Each requirement reads as the planner states it: an input alignment moved
    # onto a compile reads its own pin, so a switch of that pin's row moves it
    # too, and admission aligns it again from the switched pin. A row of an axis
    # alignment moved is that input's alone: the user takes the input over.
    moved_to = {item.pin: item.spec_id for item in switched if item.moved_for is None}
    taken_over = {item.moved_for: item.spec_id for item in switched if item.moved_for is not None}
    requirements = [
        repinned(requirement, new, codes[new]).model_dump(mode="json")
        for requirement in _statement_of(plan, state).evidence_requirements
        if isinstance(requirement, DatasetRequirement)
        and (
            new := taken_over.get(requirement.id) or moved_to.get(requirement.source_spec_id or "")
        )
        is not None
    ]
    # Admission finds each switched dataset on record, so it copies the dataset
    # over the plan's years.
    with definitions_on_record(on_record):
        revised = await planner.revise(
            ctx=ctx,
            prior_plan=plan.model_dump(mode="json"),
            evidence_requirements=requirements,
            inherits={item.spec_id: item.pin for item in switched},
        )
    if "plan" not in revised:
        refused = revised.get("refused") or {}
        logger.warning("plan switch not revised: %s", refused)
        code = str(refused.get("code") or SCOPE_CHOICE_NOT_APPLIED)
        raise SwitchFailedError(
            str(refused.get("reason") or f"The plan could not be revised to count {label}."),
            code=code,
            retryable=code in _RETRYABLE_REVISION_CODES,
        )
    revised_plan = PlannerPlan.model_validate(revised["plan"])
    ctx.state["plan"] = revised_plan.model_dump_json()
    for item, digest in zip(switched, digests, strict=True):
        store_search(
            ctx.state,
            digest,
            asked=[item.reading.asked_roles],
            held_open=[item.reading.open_roles],
            share_sides=[item.reading.share_over],
        )
    logger.info(
        "plan switched: choice_id=%s option_id=%s sources=%d version=%d",
        choice_id,
        option_id,
        len(switched),
        revised_plan.version,
    )
    return {
        **revised_plan.model_dump(mode="json"),
        "summary": revised["summary"],
        "revision": plan_revision(revised_plan),
        "moved": _moved_entries(
            revised_plan,
            switched,
            axis=axis,
            stated=_statement_of(revised_plan, state),
        ),
    }


def _statement_of(plan: PlannerPlan, state: SessionState) -> PlannerPlan:
    """`plan` as the planner states it (`submission.service.statement`), read with
    the windowed copies and alignments `state` holds."""
    return statement(
        plan,
        to_origin={copy.spec_id: copy.windowed_from for copy in windowed_pins(state).values()},
        alignments=alignments(state),
    )


def _moved_entries(
    plan: PlannerPlan,
    switched: Sequence[_Switched],
    *,
    axis: str,
    stated: PlannerPlan,
) -> list[dict[str, str]]:
    """Each other choice the switch on `axis` moved (`moved_choices`), once per card row.

    A move is listed under its row's id (`_moved_row`), so one move across
    several switched sources sharing a row, or showing none, is listed once.
    `stated` is `plan` as the planner states it (`submission.service.statement`).
    """
    pins = {item.spec_id for item in switched}
    moved: dict[str, dict[str, str]] = {}
    for item in switched:
        for entry in moved_choices(item.scope, item.after, switched=axis):
            row = _moved_row(plan, item, entry["choice_id"], switched=pins, stated=stated)
            moved.setdefault(row, {**entry, "choice_id": row})
    return list(moved.values())


def _moved_row(
    plan: PlannerPlan,
    item: _Switched,
    moved: str,
    *,
    switched: Collection[str],
    stated: PlannerPlan,
) -> str:
    """The card's id for the choice of `item`'s compiled scope Search names `moved`.

    The id of `plan`'s row for it (`harness.plan_display.choice_id`); for a
    choice the plan holds no decision for, so the card shows no row, the id
    such a row would have over every switched source (`switched`,
    `pinned_choice_id` on `stated`, the plan as the planner states it), so
    the move is one entry of the card's `moved` list on each axis whichever
    source Search moved. Search's own id when no requirement pins a switched
    source either.
    """
    axis = next((choice.axis for choice in item.after.choices if choice.id == moved), None)
    decision = next(
        (
            decision
            for decision in plan.decisions
            if decision.scope is not None
            and decision.scope.axis == axis
            and decision.fixes is not None
            and decision.fixes.source_spec_id == item.spec_id
        ),
        None,
    )
    if decision is not None:
        return row_choice_id(plan, decision)
    if axis is None:
        return moved
    return pinned_choice_id(stated, switched, axis) or moved


def _exact_codes(digest: Mapping[str, Any]) -> frozenset[str]:
    """The publisher codes the dataset of `digest`'s one part is exactly, none when its block cannot be read."""
    (part,) = digest["part_digests"]
    return exact_codes(part["spec_block"])


def _readings(ctx: Context) -> list[SearchReading]:
    """The conversation's stored readings, newest first: this turn's, then earlier turns'."""
    readings: list[SearchReading] = []
    for item in reversed(conversation_readings(ctx.state)):
        try:
            readings.append(SearchReading.model_validate(item))
        except ValidationError:
            continue
    return readings


def _target(
    plan: PlannerPlan,
    decision: Decision,
    *,
    state: SessionState,
    held: Mapping[str, AlignedInput],
    to_origin: Mapping[str, str],
    readings: Sequence[SearchReading],
) -> _Target:
    """What switching `decision`, a decision of `plan`'s switched row, moves (`_Target`).

    A pin's own scope decision (`planner.plan.is_scope_choice`) moves the pin it
    fixes, a windowed copy read as the pin it copies (`to_origin`), from the
    newest reading of it with a scope holding the decision's axis. A decision
    on an axis alignment moved for one requirement (`held`), which only that
    requirement cites and which fixes the pin it states, moves that requirement
    alone, from the compile it is pinned to (`planner.plan.compiled_pin`): its scope
    and block are the compile's, and its reading the newest of the pin the
    alignment record says it states (`AlignedInput.stated_pin`), with a scope or
    none, whose words and roles the input keeps.

    Raises:
        SwitchFailedError: If `decision` is neither, no stored reading of its
            pin holds what the switch reads, or the compile is no longer held.
    """
    gone = SwitchFailedError("That choice is no longer on the plan.", code=SCOPE_CHOICE_NOT_APPLIED)
    if decision.fixes is None or decision.scope is None:
        raise gone
    axis = decision.scope.axis
    pinned = decision.fixes.source_spec_id or ""
    pin = to_origin.get(pinned, pinned)
    if is_scope_choice(decision):
        reading, choice = _newest_with(readings, pin=pin, axis=axis)
        assert reading.scope is not None
        return _Target(pin=pin, reading=reading, scope=reading.scope, choice=choice)
    if decision.caveat is not None:
        raise gone
    moved = next(
        (
            entry
            for requirement in plan.evidence_requirements
            if decision.slot in requirement.decision_ids
            and (entry := held.get(requirement.id)) is not None
        ),
        None,
    )
    if moved is None:
        raise gone
    compiled = compiled_pin(state, moved)
    scope = part_scope(compiled.spec_id, compiled.block) if compiled is not None else None
    choice = next((c for c in scope.choices if c.axis == axis), None) if scope else None
    if compiled is None or scope is None or choice is None:
        raise SwitchFailedError(
            "Search's record of this source is gone.", code=SCOPE_CHOICE_NOT_APPLIED
        )
    stated = moved.stated_pin
    reading = next((reading for reading in readings if reading.spec_id == stated), None)
    if stated is None or reading is None:
        raise SwitchFailedError(
            "Search's record of this source is gone.", code=SCOPE_CHOICE_NOT_APPLIED
        )
    return _Target(
        pin=stated,
        reading=reading,
        scope=scope,
        choice=choice,
        moved_for=moved.requirement_id,
        block=compiled.block,
    )


def _newest_with(
    readings: Sequence[SearchReading], *, pin: str, axis: str
) -> tuple[SearchReading, ScopeChoice]:
    """The newest of `readings` of `pin` whose scope holds a choice on `axis`, and that choice.

    Raises:
        SwitchFailedError: If no reading of `pin` holds one.
    """
    for reading in readings:
        if reading.spec_id != pin or reading.scope is None:
            continue
        choice = next((c for c in reading.scope.choices if c.axis == axis), None)
        if choice is not None:
            return reading, choice
    raise SwitchFailedError(
        "Search's record of this source is gone.", code=SCOPE_CHOICE_NOT_APPLIED
    )


def _source_block(source: PinSource | None, pin: str) -> Mapping[str, Any] | None:
    """The block the planner was shown for `pin`, when it was shown one."""
    return source.spec_block(pin) if source is not None else None


def _groups(
    reading: SearchReading, block: Mapping[str, Any] | None, *, pin: str
) -> tuple[ResolvedGroup, ...]:
    """The entity groups a switch of `reading`, pinned as `pin`, posts beside the part,
    as every resubmit posts them (`compiler.sources.wire_groups`), so Search names
    them again.

    They are `block`'s groups, read for what the resubmit posts alone
    (`compiler.sources.readable_wire_groups`), where `block` is at hand and names
    sets that read, else the groups `reading` was resolved with
    (`SearchReading.entity_groups`) where those read, else none where `block`
    reads and names none.

    Raises:
        SwitchFailedError: If neither reads: what `block`'s groups post is not
            readable or there is no `block`, and `reading`'s were not readable when
            it was stored (`SearchReading.entity_groups_unreadable`) or, beside a
            block naming sets, it holds none, since the switch would post the part
            without them.
    """
    groups = readable_wire_groups(block) if block is not None else None
    # A block naming no sets says nothing of the reading's, so readable stored sets
    # stand. A block whose sets do not read falls back only to stored sets that
    # do: an empty stored tuple beside it would post the part without them.
    stored_stands = not reading.entity_groups_unreadable and (
        groups is not None or block is None or bool(reading.entity_groups)
    )
    if not groups and stored_stands:
        groups = reading.entity_groups
    if groups is None:
        logger.warning("plan switch not compiled: pin=%s named sets not readable", pin)
        raise SwitchFailedError(
            "Search's record of this source could not be read.", code=SCOPE_CHOICE_NOT_APPLIED
        )
    return tuple(groups)


async def _resubmit(
    surface: ToolSurface,
    *,
    target: _Target,
    entries: Sequence[ScopeEntry],
    groups: Sequence[ResolvedGroup],
    label: str,
) -> _Switched:
    """Post `target`'s scope's base identity with `entries` through data-mcp's `resolve_definition`.

    The scope is the reading's, or the compile's for an axis alignment moved
    (`_Target`), and `entries` the switch's (`switch_entries`). The same
    resubmit a compile on another basis posts
    (`planner.agent.compile_on_basis`), through
    `translator.discovery.resolve_definition`: the part's entity groups
    beside it, as every resubmit posts them (`compiler.sources.wire_groups`),
    so Search names them again (`_groups`).

    Raises:
        SwitchFailedError: If the call does not complete, Search refuses it,
            or its answer is not one definite part with a scope.
    """
    pin, scope = target.pin, target.scope
    try:
        part = await resolve_definition(
            surface,
            spec_id=scope.base.spec_id,
            identity=scope.base.identity,
            groups=groups,
            scope_entries=[entry.model_dump(mode="json", exclude_none=True) for entry in entries],
        )
    except DefinitionNotResolvedError as exc:
        logger.info("plan switch not compiled: pin=%s code=%s: %s", pin, exc.code, exc)
        if exc.retryable or exc.not_issued:
            failure = "unanswered"
        elif exc.refused or exc.code in _SCOPE_REFUSALS:
            failure = "refused"
        else:
            failure = "empty"
        raise _compile_failed(
            failure, label=label, code=exc.code or SCOPE_CHOICE_NOT_APPLIED
        ) from exc
    spec = part.get("spec") if part.get("status") == "definite" else None
    if not isinstance(spec, dict) or not spec.get("spec_id"):
        raise _compile_failed("empty", label=label, code=SCOPE_CHOICE_NOT_APPLIED)
    resolved = spec.get("resolved")
    try:
        after = PartScope.from_search(
            resolved.get("scope_choices") if isinstance(resolved, dict) else None
        )
    except ValidationError as exc:
        raise SwitchFailedError(
            f"Search's answer for {label} could not be read.", code=SCOPE_CHOICE_NOT_APPLIED
        ) from exc
    return _Switched(
        pin=pin,
        reading=target.reading,
        scope=scope,
        envelope={"parts": [dict(part)]},
        spec_id=str(spec["spec_id"]),
        after=after,
        moved_for=target.moved_for,
        block=target.block,
    )


# How a switch's compile fails, by what Search did: the sentence the card
# shows, worded for `{label}`, and whether the same switch can succeed if
# asked again. Search not answering (unreachable, out of time) can;
# Search refusing the option, or answering with no data, repeats.
_COMPILE_FAILURES: dict[str, tuple[str, bool]] = {
    "unanswered": ("Search could not count {label} just now.", True),
    "refused": ("Search cannot count {label} for this source.", False),
    "empty": ("Search holds no data counted as {label} here.", False),
}


def _compile_failed(failure: str, *, label: str, code: str) -> SwitchFailedError:
    """The `SwitchFailedError` for a compile that failed as `failure` (`_COMPILE_FAILURES`)."""
    sentence, retryable = _COMPILE_FAILURES[failure]
    return SwitchFailedError(sentence.format(label=label), code=code, retryable=retryable)


def _reading_digest(item: _Switched, *, axis: str) -> dict[str, Any]:
    """The digest `store_search` stores for `item`'s answer, switched on `axis`, under the text of the reading it replaces.

    Search echoes the spec_id data-mcp posts as the part's text, so the
    reading keeps the words the user's search used. The axes the planner
    settled stay the planner's through the switch, less the switched axis,
    which is now the user's (`harness.scope_choices.PLANNER_AXES_BLOCK_KEY`):
    those its words settled on the reading (`SearchReading.planner_axes`), or,
    for an input alignment moved, those the compile's block marks, every axis
    alignment moved among them.
    """
    digest = resolver_digest(item.envelope)
    (part,) = digest.get("part_digests") or [{}]
    part["variable"] = item.reading.text
    settled = item.reading.planner_axes if item.block is None else planner_axes(item.block)
    mark_planner_axes(digest, set(settled) - {axis})
    return digest


def _pin_source(
    replaced: PinSource | None, digest: Mapping[str, Any], reading: SearchReading
) -> PinSource:
    """The pin source of a switched dataset: the one it replaces, holding the switched block."""
    (part,) = digest["part_digests"]
    block = part["spec_block"]
    if replaced is None:
        return PinSource(block=block, settled=True, asked_roles=reading.asked_roles)
    return replaced.model_copy(update={"block": block, "earlier_turn": False})


def _restorable_state(ctx: Context) -> dict[str, Any]:
    """A copy of the state a switch that fails, is stopped or raises puts back:
    the drafting state (`_DRAFTING_PREFIXES`, less `RESOLVED_STATE_PREFIX`) and
    `_RESTORED_KEYS`."""
    state = ctx.state.to_dict()
    kept = {key: value for key, value in state.items() if _drafting(key)}
    kept |= {key: state.get(key, unset) for key, unset in _RESTORED_KEYS.items()}
    return json.loads(json.dumps(kept))


def _restore(ctx: Context, before: Mapping[str, Any]) -> None:
    """Put the state back as `before` holds it, clearing any drafting key written since."""
    for key in ctx.state.to_dict():
        if _drafting(key) and key not in before:
            ctx.state[key] = None
    for key, value in before.items():
        ctx.state[key] = value


def _drafting(key: str) -> bool:
    """Whether `key` is drafting state a failed switch puts back (`_DRAFTING_PREFIXES`)."""
    return key.startswith(_DRAFTING_PREFIXES) and not key.startswith(RESOLVED_STATE_PREFIX)


async def apply_stored_choice(
    ctx: Context,
    *,
    choice_id: str,
    option_id: str,
    spec_id: str,
    surface: ToolSurface,
    fetch: Callable[..., Awaitable[dict[str, Any]]],
    record_step: Callable[..., None] | None,
) -> dict[str, str]:
    """Apply a chip: move choice `choice_id` of the conversation's newest
    reading holding it, the reading `spec_id` the chip was offered on, to
    `option_id`, fetch what Search compiles, and return the hand-back the
    agent answers from.

    The reading's base identity is posted with its entries and the switch
    (`switch_entries`) through `surface` (`_resubmit`), the reading Search
    compiles is stored as the plan-card switch stores it, and its spec_id is
    fetched with the orchestrator's `fetch` tool once for each way the answer
    fetched the reading, with that fetch's arguments (`FetchReading.args`:
    its places, period and reading of the family), so the chip changes the
    chosen option alone. Its frames, served scope and step reach the turn as
    any fetch's do. The chip applied is
    recorded under `ORCHESTRATOR_APPLIED_KEY`, with the option it moved from
    and the spec_id Search compiled.

    A chip that cannot be applied, because no reading holds the choice or
    the newest one is not `spec_id` (`CHIP_STALE_NOTE`, as the studio refuses
    it), the option is not one Search can serve, or Search refuses or fails
    the resubmit, fetches nothing: the hand-back carries the note a plan-card
    switch shows, and the `APPLY_CHOICE_STEP_TOOL` step fails with it under
    the code the switch reports (`SwitchFailedError.code`), to be retried
    after a delay where the same chip can succeed later
    (`SwitchFailedError.retryable`) and otherwise not to be retried.

    Returns:
        `{"status", "text"}` for the `orchestrate` node.
    """
    args = _Pick(choice_id=choice_id, option_id=option_id)
    reading, choice = stored_choice(conversation_readings(ctx.state), choice_id) or (None, None)
    option = next((o for o in choice.options if o.option_id == option_id), None) if choice else None
    label = option.label if option is not None else option_id
    try:
        if reading is None or choice is None:
            raise SwitchFailedError(
                "That choice is no longer on this answer.", code=SCOPE_CHOICE_NOT_APPLIED
            )
        if reading.spec_id != spec_id:
            raise SwitchFailedError(CHIP_STALE_NOTE, code=SCOPE_CHOICE_NOT_APPLIED)
        if (
            option is None
            or option_id == choice.active
            or option.availability.status == "unavailable"
        ):
            raise SwitchFailedError(
                f"Search does not offer {label} for this source.", code=_OPTION_UNAVAILABLE
            )
        assert reading.scope is not None and reading.spec_id is not None
        pin = reading.spec_id
        switched = await _resubmit(
            surface,
            target=_Target(pin=pin, reading=reading, scope=reading.scope, choice=choice),
            entries=switch_entries(reading.scope, choice=choice, option_id=option_id),
            groups=_groups(
                reading,
                (ctx.state.get(ORCHESTRATOR_SPECS_KEY) or {}).get(pin) or earlier_block(pin),
                pin=pin,
            ),
            label=label,
        )
    except SwitchFailedError as exc:
        note = str(exc)
        logger.info(
            "scope choice not applied: choice_id=%s option_id=%s code=%s",
            choice_id,
            option_id,
            exc.code,
        )
        _apply_step(
            record_step,
            args=args,
            result={
                "outcome": "refusal",
                "refusal": {
                    "code": exc.code,
                    "retry_mode": "retry_after_delay" if exc.retryable else "do_not_retry",
                    "fix": None,
                    "message": note,
                },
            },
        )
        return {
            "status": f"Counting {label}",
            "text": (
                f"The user asked to count {label}, and it was not applied: {note} Tell the "
                "user so in a sentence. The earlier answer stands."
            ),
        }
    assert reading.scope is not None
    store_search(
        ctx.state,
        _reading_digest(switched, axis=choice.axis),
        asked=[reading.asked_roles],
        held_open=[reading.open_roles],
        share_sides=[reading.share_over],
    )
    ctx.state[ORCHESTRATOR_APPLIED_KEY] = {
        **args,
        "from_option_id": choice.active,
        "spec_id": switched.spec_id,
    }
    logger.info(
        "scope choice applied: choice_id=%s option_id=%s from=%s",
        choice_id,
        option_id,
        choice.active,
    )
    _apply_step(
        record_step,
        args=args,
        result={"outcome": "ok", "frame_refs": [], "summary": f"counting {label}"},
    )
    # One fetch per way the answer fetched the reading, one after another, as
    # the agent's own calls run.
    fetched = [
        await fetch(spec_id=switched.spec_id, tool_context=ctx, **shaped)
        for shaped in (reading.fetch.args if reading.fetch is not None else ()) or ({},)
    ]
    held = choice.active_option
    moved = [
        f"Search also moved {entry['from_label']} to {entry['to_label']}: {entry['reason']}"
        for entry in moved_choices(reading.scope, switched.after, switched=choice.axis)
    ]
    lead = (
        f'The user chose to count "{reading.text or reading.sentence or pin}" as {label}'
        f"{f' in place of {held.label}' if held is not None else ''}. Search counted it again "
        f"as {switched.spec_id}, and its data was fetched for you below, once for each way the "
        "answer fetched it. Answer the user's last "
        "question again from these figures and say what the change did to them. Search and "
        "plan are not available this turn."
    )
    return {
        "status": f"Counting {label}",
        "text": "\n".join([lead, *moved, *(json.dumps(result) for result in fetched)]),
    }


def _apply_step(
    record_step: Callable[..., None] | None, *, args: _Pick, result: dict[str, Any]
) -> None:
    """Record the `APPLY_CHOICE_STEP_TOOL` step, when the executor records steps."""
    if record_step is not None:
        record_step(
            tool=APPLY_CHOICE_STEP_TOOL,
            args=dict(args),
            result=result,
            step_id=APPLY_CHOICE_STEP_TOOL,
        )
