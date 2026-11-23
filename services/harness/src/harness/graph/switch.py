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
the years the question asked for. Admission finds it on record through
its saved pin source (`planner.plan.identity_on_record`), so where the
plan reads it over a window of its own, admission copies it over that
window (`submission.service._window_pins`) and the translation resolves
the copy. The turn keeps Search's answers
(`translator.discovery.resolving_once`), so a switch repeating one the
turn made posts nothing. The
plan is then revised by the planner adapter with no model call
(`harness.graph.interfaces.PlannerModel.revise`): each requirement pinning a
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
states the axis and alignment leaves a stated axis alone. A row of a pin no
reading names, such as a compile a requirement pins directly, posts the scope
of the block the card read the row from (`planner.plan.held_sources`), under
the words and parties of the requirements stating the pin.

Each requirement the switch re-pins drops the source it stated
(`source`, `source_ref`, `source_exclude`, and `source_code` unless the new
dataset is exactly that series): the user chose the dataset, so its pin is
the requirement's identity, and text left behind would have discovery,
compile and the run's evidence check judge the new dataset against the old
one's name.

The card names a choice by its row's id (`harness.plan_display.choice_id`),
which every switch of the conversation keeps.

A switch of the row stating how pins are counted
(`harness.plan_display.count_choice_id`) moves every pin the row stands for
to its other count Search already named beside it, a dataset with a
definition on record, so it posts nothing to Search: every requirement
stating one of those pins is re-pinned to that pin's other count in one
revision, with the user's pick held (`_switched_count`).

A switch of the row stating the whole unit a unit question's answer holds
its parts to (`submission.prices_choice.unit_slot`) re-reads every part the
question covered in the unit picked (`harness.graph.orchestrate.
read_part_in_unit`), each from its own block, and re-pins every requirement
citing the row to the series Search read for its part, stating that unit,
in one revision (`_switched_unit`). A part Search cannot read in that unit
fails the whole switch. A row stating a part's whole unit under the part's
prices slot, on a plan card stored before unit questions had a row of their
own, and a question whose parts are recorded with no `group`
(`harness.graph.state.ServedUnit.group`) cannot be switched.

A switch Search refuses (`requested_option_unavailable`,
`conflicting_requested_options`), or one that fails anywhere else, offers
the unchanged plan again with `switch_failed` naming the option, the
refusal's code, one sentence saying why (`harness.sentences.
refusal_reason` for a refused revision) and whether the same switch can
succeed if asked again (`retryable`), and leaves the plan, the planner's
drafting state and the turn's readings as they were (`_RESTORED_KEYS`). A
switch the user stops ends the same way under `SWITCH_CANCELLED`, so the
card stays answerable: the revision runs under the turn's `StopScope`
(`harness.turn.node.NodeCtx.stop`), so the first Stop while it runs ends
the revision and the node returns its offer, which the loop writes with the
state put back. A second Stop ends the turn. A switch that raises closes
the turn with nothing it wrote stored.

A chip under an answer applies one option the same way, with no plan
(`apply_stored_choice`): the `seed` node posts the stored reading's base
identity with the switched entries, stores the reading Search compiles,
fetches it through the orchestrator's `fetch` tool and hands the result to
the agent, which answers from it with no search. A chip naming another
count of a part fetches that count, a dataset Search already named, with
no resubmit (`_apply_other_count`).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict, cast

from compiler.sources import readable_wire_groups, source_from_spec_block
from compiler.surface import ToolSurface
from data_mcp.scope import ScopeEntry
from orchestrator.call import OrchestratorCall
from planner.plan import (
    AlignedInput,
    DatasetRequirement,
    Decision,
    PinSource,
    SessionState,
    alignments,
    compiled_pin,
    earlier_block,
    exact_codes,
    held_sources,
    is_scope_choice,
    pin_sources,
    plan_revision,
    repinned,
    save_pin_sources,
    windowed_pins,
)
from planner.plan import Plan as PlannerPlan
from planner.source_reads import run_together
from planner.tool_results import ProblemCode
from planner.workflow import ResolvedGroup
from pydantic import ValidationError
from submission.prices_choice import PRICES_SLOT_PREFIX, UNIT_SLOT_PREFIX, measure_key
from submission.service import part_scope, statement
from translator.discovery import (
    RESOLVED_AT_STATE_PREFIX,
    RESOLVED_STATE_PREFIX,
    DefinitionNotResolvedError,
    requirement_parties,
    resolve_definition,
    resolving_once,
)

from harness.decision_pick import picks_the_chosen_option
from harness.graph.events import GraphEvent
from harness.graph.interfaces import PlannerModel
from harness.graph.orchestrate import read_part_in_unit, turn_id
from harness.graph.state import (
    ORCHESTRATOR_APPLIED_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPEC_WARNINGS_KEY,
    ORCHESTRATOR_SPECS_KEY,
    ORCHESTRATOR_UNIT_READ_KEY,
    ORCHESTRATOR_UNIT_SERVED_KEY,
    ORCHESTRATOR_UNIT_SETTLED_KEY,
    ServedUnit,
    conversation_readings,
    readable_plan,
    stated_unit,
    store_search,
    stored_mapping,
    unit_row_offers,
    units_served,
)
from harness.plan_display import (
    DECISION_CHOICE_PREFIX,
    chosen_count_decision,
    chosen_scope_decision,
    count_options,
    decision_option_id,
    listed_together,
    pinned_choice_id,
    switchable_decision,
)
from harness.plan_display import choice_id as row_choice_id
from harness.scope_choices import (
    PartScope,
    ScopeChoice,
    mark_planner_axes,
    moved_choices,
    planner_axes,
    switch_entries,
)
from harness.sentences import StopCode
from harness.tool_contracts import HarnessCode, resolver_digest
from harness.turn.node import Next, Node, NodeCtx
from harness.turn_records import (
    CHIP_STALE_NOTE,
    SearchReading,
    is_count_choice,
    stored_choice,
    stored_count,
)
from harness.unit_offer import offered_unit_refusal, unit_option_id, unit_words, whole_units
from harness.unit_switch import UnitRefused
from harness.wire import RetryMode, UnitChoice, UnitOffer

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
        StopCode.TURN_WALL_TIME_SPENT.value,
        StopCode.DISCOVERY_UNAVAILABLE.value,
        ProblemCode.SURFACE_UNAVAILABLE.value,
    }
)

# The state the planner adapter and the translator keep while drafting, which a
# revision that fails leaves as it found. The parts Search answered this turn
# (`RESOLVED_STATE_PREFIX`) and the times they were answered
# (`RESOLVED_AT_STATE_PREFIX`) are not drafting state. Each holds whatever the
# switch's outcome, so a failed switch keeps them.
_DRAFTING_PREFIXES = ("planner:", "translator:")
# The other state a revision writes, which a switch that fails or is stopped
# puts back too: the plan, and the readings `store_search` stores for the
# switched sources and for the datasets the revision searches
# (`harness.planner_adapter.store_searched_pin_readings`), with their blocks. Each
# maps to the value it is put back as when unset, since the readers take the
# readings and blocks as a list and a mapping. The blocks are the turn's whole
# spec mapping, which the turn store writes as given (`harness.turn.store`), so
# a block the revision added is deleted and one it rewrote is put back. A unit
# row's switch also writes the series each part is fetched as, the series it
# was re-read as and the spec_ids settled (`harness.graph.orchestrate.
# read_part_in_unit`), so a switch that fails leaves every part of the
# question in the unit it was in.
_RESTORED_KEYS: Mapping[str, Any] = {
    "plan": None,
    ORCHESTRATOR_SEARCHES_KEY: [],
    ORCHESTRATOR_SPECS_KEY: {},
    ORCHESTRATOR_SPEC_WARNINGS_KEY: {},
    ORCHESTRATOR_UNIT_READ_KEY: {},
    ORCHESTRATOR_UNIT_SERVED_KEY: {},
    ORCHESTRATOR_UNIT_SETTLED_KEY: [],
}


# The tool name and step id the Activity feed shows a chip's application under.
APPLY_CHOICE_STEP_TOOL = "apply_choice"
# The code that step fails with when the chip cannot be applied; its message
# is the note `SwitchFailedError` carries.
SCOPE_CHOICE_NOT_APPLIED = "scope_choice_not_applied"
# The code of a switch the user stopped while it ran.
SWITCH_CANCELLED = "switch_cancelled"


class _Pick(TypedDict):
    """The option a chip picks: the choice and the option, as its step records them."""

    choice_id: str
    option_id: str


class SwitchFailedError(Exception):
    """A switch or chip that cannot be applied: `code` is the refusal's code,
    `reason`, the message, the one sentence saying why that the card shows
    under the choice, and `retryable` whether the same switch can succeed if
    asked again. `answer` is a chip's fetch that raised inside DFO, as the
    `fetch` tool answers a raise (`harness.orchestrator_tools.raised_answer`),
    handed to the agent as its own fetch would be."""

    def __init__(
        self,
        reason: str,
        *,
        code: str,
        retryable: bool = False,
        answer: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code
        self.retryable = retryable
        self.answer = answer


@dataclass(frozen=True)
class _Target:
    """What one decision of the switched row moves: the pin the planner states
    for it, the reading the switch stores it under, the scope the switch posts
    from and its choice on the switched axis.

    `moved_for` is `None` for a pin's own scope decision, which every
    requirement stating `pin` follows. Its scope is that of the newest reading
    of `pin`, and `block` is `None`; with no such reading, the scope is that of
    the block on record for `pin` (`planner.plan.held_sources`), `block` that
    block, and `reading` the requirements' own (`_stated_reading`). For an
    input alignment moved onto a compile (`planner.plan.AlignedPin`), the
    decision on an axis it moved is that requirement's alone: `moved_for` is
    its id, `scope` the compile's, and `block` the compile's spec block.
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
    requirement stating `pin`, and `block` is the spec block the switch read
    the scope from when no reading held it (`_Target`)."""

    pin: str
    reading: SearchReading
    scope: PartScope
    envelope: dict[str, Any]
    spec_id: str
    after: PartScope
    moved_for: str | None = None
    block: Mapping[str, Any] | None = None


def switch_node(*, planner: PlannerModel, definition_surface: ToolSurface) -> Node[GraphEvent]:
    """The `switch` node, revising with `planner` and posting through `definition_surface`."""

    async def switch(ctx: NodeCtx[GraphEvent], node_input: Any) -> Next:
        """Apply the switch `node_input` carries (`{"switch": {"choice_id",
        "option_id"}}`, with `unit` on a unit row's switch) and offer the plan again.

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
        # The state as it stood, which a switch that fails or is stopped
        # leaves as it found.
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
                        unit=pick.get("unit"),
                        planner=planner,
                        surface=definition_surface,
                    ),
                )
        except SwitchFailedError as exc:
            _restore(ctx, before)
            output = switch_failed_offer(plan, choice_id=choice_id, option_id=option_id, error=exc)
        return Next("offer", output)

    # A stopped switch offers the unchanged plan, whose pause records the Stop.
    return Node(run=switch, stop_edges=frozenset({"offer"}))


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


async def _unless_stopped(
    ctx: NodeCtx[GraphEvent], revision: Awaitable[dict[str, Any]]
) -> dict[str, Any]:
    """`revision`'s payload, unless the user stops the switch first.

    The revision runs while the node holds the turn's `StopScope` open
    (`harness.turn.node.NodeCtx.stop`), so the first Stop sets the scope's
    event and ends the revision, not the turn. The scope closes as soon as
    the revision or the Stop ends the wait, before any cleanup, so a Stop
    the scope took always ends the switch stopped and a later one ends the
    turn. A stopped revision is cancelled and awaited without taking its
    outcome, so a revision that ends in any way once the Stop came, an error
    its cleanup raises included, ends stopped, as does one that finishes in
    the same wait as the Stop. The `switch` node puts the state it wrote
    back, so the plan stands unchanged. A cancellation of the node itself,
    while the revision unwinds, still cancels the node. A pick's replan
    that ended on the planner's question to the user (`{"asked": ...}`,
    `harness.graph.interfaces.PlannerModel`) stays paused though the Stop came in
    the same wait: the question is the user's to answer, so its payload is
    answered and the node pauses on it, offering no plan beside it.

    Raises:
        SwitchFailedError: Under `SWITCH_CANCELLED` when the user stopped
            the switch, else as `revision` raises it.
    """
    running = asyncio.ensure_future(revision)
    stopped: asyncio.Future[Any] | None = None
    finished: set[asyncio.Future[Any]] = set()
    was_stopped = False
    try:
        # The scope closes as the wait returns, before any cleanup awaits, so
        # a Stop either lands while the revision can still be stopped or
        # finds the scope closed and ends the turn.
        async with ctx.stop.open() as stop:
            stopped = asyncio.ensure_future(stop.wait())
            racing: set[asyncio.Future[Any]] = {running, stopped}
            try:
                finished, _ = await asyncio.wait(racing, return_when=asyncio.FIRST_COMPLETED)
            finally:
                was_stopped = stop.is_set()
    finally:
        unfinished = {
            waiting for waiting in (running, stopped) if waiting is not None and not waiting.done()
        }
        for waiting in unfinished:
            waiting.cancel()
        if unfinished:
            # Raises nothing the tasks raise. A cancellation of the node raises.
            await asyncio.wait(unfinished)
    if running in finished and not running.cancelled() and (not was_stopped or _asked(running)):
        return running.result()
    error = None if running.cancelled() else running.exception()
    if error is not None:
        logger.info("stopped plan switch raised: %r", error)
    raise SwitchFailedError(
        "You stopped this change, so the plan is as it was.",
        code=SWITCH_CANCELLED,
        retryable=True,
    )


def _asked(running: asyncio.Future[Any]) -> bool:
    """Whether `running`, done, answered a question to the user."""
    if running.exception() is not None:
        return False
    result = running.result()
    return isinstance(result, dict) and "asked" in result


async def _switched_plan(
    ctx: NodeCtx[GraphEvent],
    plan: PlannerPlan,
    *,
    choice_id: str,
    option_id: str,
    unit: object = None,
    planner: PlannerModel,
    surface: ToolSurface,
) -> dict[str, Any]:
    """The `approve` payload of `plan` revised for the switch, `unit` the
    unit a unit row's switch names beside `option_id`, when it names one.

    Raises:
        SwitchFailedError: If the switch cannot be applied, with the note to show.
    """
    if picks_the_chosen_option(plan, choice_id=choice_id, option_id=option_id):
        raise _already_chosen()
    slot = choice_id.removeprefix(DECISION_CHOICE_PREFIX)
    if choice_id.startswith(DECISION_CHOICE_PREFIX) and slot.startswith(UNIT_SLOT_PREFIX):
        return await _switched_unit(
            ctx,
            plan,
            slot,
            option_id=option_id,
            unit=unit,
            planner=planner,
            surface=surface,
        )
    priced = next((d for d in plan.decisions if d.slot == slot), None)
    if (
        choice_id.startswith(DECISION_CHOICE_PREFIX)
        and slot.startswith(PRICES_SLOT_PREFIX)
        and priced is not None
        and not switchable_decision(priced)
    ):
        # A part's prices row stating a whole unit, on a card stored before
        # unit questions had a row of their own. A switchable prices row
        # replans with the pick (`harness.decision_pick`).
        raise _unit_not_switchable()
    counted = chosen_count_decision(plan, choice_id)
    if counted is not None:
        return await _switched_count(
            ctx, plan, counted, choice_id=choice_id, option_id=option_id, planner=planner
        )
    clicked = chosen_scope_decision(plan, choice_id)
    if clicked is None or clicked.scope is None:
        raise _choice_gone()
    axis = clicked.scope.axis
    state = cast(SessionState, ctx.state)
    to_origin = {copy.spec_id: copy.windowed_from for copy in windowed_pins(state).values()}
    held = alignments(state)
    readings = _readings(ctx)
    sources = pin_sources(state)
    on_record = held_sources(state)
    stated = _statement_of(plan, state)
    targets = [
        _target(
            plan,
            decision,
            state=state,
            on_record=on_record,
            stated=stated,
            held=held,
            to_origin=to_origin,
            readings=readings,
        )
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
    specs = stored_mapping(ctx.state, ORCHESTRATOR_SPECS_KEY)
    switched = await run_together(
        _resubmit(
            surface,
            target=target,
            entries=switch_entries(target.scope, choice=target.choice, option_id=option_id),
            groups=_groups(
                target.reading,
                target.block
                if target.block is not None
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
        for requirement in stated.evidence_requirements
        if isinstance(requirement, DatasetRequirement)
        and (
            new := taken_over.get(requirement.id) or moved_to.get(requirement.source_spec_id or "")
        )
        is not None
    ]
    # Admission finds each switched dataset on record through its saved pin source
    # (`planner.plan.held_sources`), so it copies the dataset over the plan's years.
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


async def _switched_count(
    ctx: NodeCtx[GraphEvent],
    plan: PlannerPlan,
    decision: Decision,
    *,
    choice_id: str,
    option_id: str,
    planner: PlannerModel,
) -> dict[str, Any]:
    """The `approve` payload of `plan` revised for a switch of its count row
    (`harness.plan_display.count_choice_id`) to the count `option_id` names.

    The row states how the pins it stands for are counted
    (`harness.plan_display.listed_together`), each pin `decision` or one
    listed with it, a windowed copy read as the pin it copies. The row's
    counts are those every pin offers (`harness.plan_display.count_options`),
    and each pin's other count is a dataset Search already named beside it,
    with a definition on record (`_count_spec_ids`), so nothing is posted to
    Search: every requirement stating any of the pins
    (`submission.service.statement`) is re-pinned to its pin's other count
    (`planner.plan.repinned`) in one revision, and the plan is revised with
    no model call, the user's pick of each count held
    (`PlannerModel.revise`'s `picked`), so the measure decision admission
    writes for each new pin is the user's. Nothing else moves.

    Raises:
        SwitchFailedError: If the row offers no count under `option_id`, no
            record names that count's spec_id for one of its pins, no
            requirement states a pin, or the revision is refused.
    """
    state = cast(SessionState, ctx.state)
    to_origin = {copy.spec_id: copy.windowed_from for copy in windowed_pins(state).values()}
    label = next(
        (o for o in count_options(plan, decision) if decision_option_id(o) == option_id), None
    )
    if label is None:
        raise SwitchFailedError(
            f"Search does not offer {option_id} for this source.", code=_OPTION_UNAVAILABLE
        )
    moves: dict[str, str] = {}
    for member in listed_together(plan, decision.slot) or (decision,):
        assert member.fixes is not None and member.fixes.source_spec_id is not None
        pinned = member.fixes.source_spec_id
        pin = to_origin.get(pinned, pinned)
        new = _count_spec_ids(ctx, pin).get(label)
        if new is None or new == pin:
            raise SwitchFailedError(
                f"Search does not offer {label} for this source.", code=_OPTION_UNAVAILABLE
            )
        moves[pin] = new
    stated = _statement_of(plan, state)
    blocks = stored_mapping(ctx.state, ORCHESTRATOR_SPECS_KEY)
    requirements: list[dict[str, Any]] = []
    for requirement in stated.evidence_requirements:
        if not isinstance(requirement, DatasetRequirement):
            continue
        new = moves.get(requirement.source_spec_id or "")
        if new is None:
            continue
        block = blocks.get(new) or earlier_block(new)
        requirements.append(repinned(requirement, new, exact_codes(block)).model_dump(mode="json"))
    if {r["source_spec_id"] for r in requirements} != set(moves.values()):
        raise _choice_gone()
    revised = await planner.revise(
        ctx=ctx,
        prior_plan=plan.model_dump(mode="json"),
        evidence_requirements=requirements,
        inherits={new: pin for pin, new in moves.items()},
        picked=tuple(moves.values()),
    )
    if "plan" not in revised:
        refused = revised.get("refused") or {}
        logger.warning("plan count switch not revised: %s", refused)
        code = str(refused.get("code") or SCOPE_CHOICE_NOT_APPLIED)
        raise SwitchFailedError(
            str(refused.get("reason") or f"The plan could not be revised to count {label}."),
            code=code,
            retryable=code in _RETRYABLE_REVISION_CODES,
        )
    revised_plan = PlannerPlan.model_validate(revised["plan"])
    ctx.state["plan"] = revised_plan.model_dump_json()
    logger.info(
        "plan count switched: choice_id=%s moves=%s version=%d",
        choice_id,
        moves,
        revised_plan.version,
    )
    return {
        **revised_plan.model_dump(mode="json"),
        "summary": revised["summary"],
        "revision": plan_revision(revised_plan),
        "moved": [],
    }


async def _switched_unit(
    ctx: NodeCtx[GraphEvent],
    plan: PlannerPlan,
    slot: str,
    *,
    option_id: str,
    unit: object,
    planner: PlannerModel,
    surface: ToolSurface,
) -> dict[str, Any]:
    """The `approve` payload of `plan` revised for a switch of its unit row,
    the decision under `slot` (`submission.prices_choice.unit_slot`), to the
    unit picked (`_picked_unit`).

    The row's parts are those `units_served` records under the question's
    lead part (`ServedUnit.group`), and its offer the one the card showed,
    recomputed from state (`harness.graph.state.unit_row_offers`). Each part
    is read in the unit picked at once (`read_part_in_unit`), the count read
    in its place re-read again in its place, and every requirement citing the
    row, as the planner states it (`submission.service.statement`), is
    re-pinned to the series the part it reads (`_covering_members`) is
    fetched as now (`planner.plan.repinned`) and states the unit. The plan is
    revised with no model call, and admission writes the row again in the
    new unit (`submission.prices_choice.prices_decisions`).

    Raises:
        SwitchFailedError: Non-retryable when the plan holds no such row, the
            question's parts are recorded with no group or its lead part not
            at all, the row offers no unit, the unit picked lies outside its
            offer or is the one it holds, or a requirement citing the row
            reads no part of the question; as Search's refusal says
            (`harness.unit_switch.UnitRefused`) when it cannot read a part in
            the unit; and as the revision's refusal when it is refused.
    """
    state = cast(SessionState, ctx.state)
    decision = next((d for d in plan.decisions if d.slot == slot), None)
    if decision is None:
        raise _choice_gone()
    lead = slot.removeprefix(UNIT_SLOT_PREFIX)
    members = {part: served for part, served in units_served(state).items() if served.group == lead}
    offer = unit_row_offers(state).get(slot)
    if lead not in members or offer is None:
        raise _unit_not_switchable()
    picked = _picked_unit(offer, option_id=option_id, unit=unit)
    if picked == offer.start:
        raise _already_chosen()
    stated = stated_unit(picked)
    if stated is None:
        raise _unit_unavailable(picked, offer)
    covered = [
        requirement
        for requirement in _statement_of(plan, state).evidence_requirements
        if isinstance(requirement, DatasetRequirement) and slot in requirement.decision_ids
    ]
    if not covered:
        raise _choice_gone()
    member_of = _covering_members(state, covered, members)
    reads = await run_together(
        read_part_in_unit(state, surface, part, picked, group=lead, count=served.count)
        for part, served in members.items()
    )
    refused = next((read.result for read in reads if isinstance(read.result, UnitRefused)), None)
    if refused is not None:
        raise SwitchFailedError(
            refused.reason,
            code=refused.code or _OPTION_UNAVAILABLE,
            retryable=refused.retryable,
        )
    now = units_served(state)
    moves = {served.spec_id: now[part].spec_id for part, served in members.items()}
    blocks = stored_mapping(ctx.state, ORCHESTRATOR_SPECS_KEY)
    requirements: list[dict[str, Any]] = []
    for requirement in covered:
        pin = requirement.source_spec_id
        new = now[member_of[requirement.id]].spec_id
        moved = (
            repinned(requirement, new, exact_codes(blocks.get(new) or earlier_block(new)))
            if new != pin
            else requirement
        )
        requirements.append(moved.model_copy(update={"unit": stated}).model_dump(mode="json"))
    revised = await planner.revise(
        ctx=ctx,
        prior_plan=plan.model_dump(mode="json"),
        evidence_requirements=requirements,
        inherits={new: old for old, new in moves.items() if new != old},
    )
    words = _lowered(unit_words(picked, offer))
    if "plan" not in revised:
        refused_revision = revised.get("refused") or {}
        logger.warning("plan unit switch not revised: %s", refused_revision)
        code = str(refused_revision.get("code") or SCOPE_CHOICE_NOT_APPLIED)
        raise SwitchFailedError(
            str(refused_revision.get("reason") or f"The plan could not be revised to {words}."),
            code=code,
            retryable=code in _RETRYABLE_REVISION_CODES,
        )
    revised_plan = PlannerPlan.model_validate(revised["plan"])
    ctx.state["plan"] = revised_plan.model_dump_json()
    logger.info(
        "plan unit switched: slot=%s unit=%s moves=%s version=%d",
        slot,
        unit_option_id(picked),
        moves,
        revised_plan.version,
    )
    return {
        **revised_plan.model_dump(mode="json"),
        "summary": revised["summary"],
        "revision": plan_revision(revised_plan),
        "moved": [],
    }


def _covering_members(
    state: SessionState,
    covered: Sequence[DatasetRequirement],
    members: Mapping[str, ServedUnit],
) -> dict[str, str]:
    """The part of the unit question each of `covered` reads, by the
    requirement's id: the member whose series it pins, else the member whose
    series measures what its pin measures in the same currency, window and
    prices left out (`submission.prices_choice.measure_key`), such as a
    window of the measure the planner pinned itself, which the user's answer
    holds too. A requirement pinned to a member's series moves to the series
    that member is read as in the new unit, and admission copies it over the
    requirement's own years (`submission.service._window_pins`).

    Raises:
        SwitchFailedError: Non-retryable when a requirement pins a series
            whose block is not on record or that measures what no member's
            series measures, so no re-read can move it.
    """
    by_pin = {served.spec_id: part for part, served in members.items()}
    by_measure: dict[str, str] = {}
    for part, served in members.items():
        block = _block_on_record(state, served.spec_id)
        if block is not None and (key := measure_key(block)) is not None:
            by_measure.setdefault(key, part)
    member_of: dict[str, str] = {}
    for requirement in covered:
        pin = requirement.source_spec_id or ""
        block = _block_on_record(state, pin)
        key = measure_key(block) if block is not None else None
        member = by_pin.get(pin) or (by_measure.get(key) if key is not None else None)
        if member is None:
            logger.warning(
                "unit switch cannot move %s: pin %s matches no part of the question",
                requirement.id,
                pin,
            )
            raise _unit_not_switchable()
        member_of[requirement.id] = member
    return member_of


def _block_on_record(state: SessionState, spec_id: str) -> Mapping[str, Any] | None:
    """The spec block on record for `spec_id`: this turn's search's, the
    planner's held source's (`planner.plan.held_sources`), else an earlier
    turn's."""
    if (block := stored_mapping(state, ORCHESTRATOR_SPECS_KEY).get(spec_id)) is not None:
        return block
    if (source := held_sources(state).get(spec_id)) is not None and (
        block := source.spec_block(spec_id)
    ) is not None:
        return block
    return earlier_block(spec_id)


def _picked_unit(offer: UnitOffer, *, option_id: str, unit: object) -> UnitChoice:
    """The unit a switch of a unit row offering `offer` picks: `unit` when
    the answer states one, which `option_id` must name
    (`harness.unit_offer.unit_option_id`), else the whole unit the row lists
    under `option_id` (`harness.studio_translate`'s options: `whole_units`
    and the start).

    Raises:
        SwitchFailedError: Non-retryable when `unit` is no unit or not the
            one `option_id` names, `option_id` names no unit the row lists,
            or the unit lies outside `offer` (`offered_unit_refusal`).
    """
    if unit is not None:
        try:
            picked = UnitChoice.model_validate(unit)
        except ValidationError:
            raise SwitchFailedError(
                "That unit is not one the plan can be stated in.", code=_OPTION_UNAVAILABLE
            ) from None
        if unit_option_id(picked) != option_id:
            raise SwitchFailedError("That unit is not the option picked.", code=_OPTION_UNAVAILABLE)
    else:
        listed = {unit_option_id(offer.start): offer.start, **whole_units(offer)}
        if (found := listed.get(option_id)) is None:
            raise SwitchFailedError(
                "That unit is not offered for these figures.", code=_OPTION_UNAVAILABLE
            )
        picked = found
    if offered_unit_refusal(offer, picked) is not None:
        raise _unit_unavailable(picked, offer)
    return picked


def _unit_unavailable(unit: UnitChoice, offer: UnitOffer) -> SwitchFailedError:
    """The refusal of a switch to `unit`, which `offer` does not hold or no plan can state."""
    return SwitchFailedError(
        f"Search does not offer {_lowered(unit_words(unit, offer))} for these figures.",
        code=_OPTION_UNAVAILABLE,
    )


def _unit_not_switchable() -> SwitchFailedError:
    """The refusal of a switch of a unit row this release cannot re-read: one
    under a part's prices slot, or one whose question's parts are recorded
    with no group (`ServedUnit.group`)."""
    return SwitchFailedError(
        "This plan's unit cannot be changed here. Ask for the unit you want.",
        code=SCOPE_CHOICE_NOT_APPLIED,
    )


def _already_chosen() -> SwitchFailedError:
    """The refusal of a switch to the option its row already holds."""
    return SwitchFailedError("That choice is already on the plan.", code=SCOPE_CHOICE_NOT_APPLIED)


def _lowered(words: str) -> str:
    """`harness.unit_offer.unit_words` inside a sentence: "constant 2023 US dollars"."""
    return words[:1].lower() + words[1:]


def _count_spec_ids(ctx: NodeCtx[GraphEvent], pin: str) -> dict[str, str]:
    """Each count a pin of `pin` may be counted as, by label, with its spec_id.

    Read from the elected measure (`planner.workflow.ElectedMeasure`) of the
    block on record for `pin` (`planner.plan.held_sources`, else this turn's
    or an earlier turn's search): its own label for `pin` and each other
    count with the spec_id Search gave it. A pin that is itself one of those
    other counts may hold no block, as when the planner pinned it from the
    source that offered it; the counts are then read from the block of a
    source that offers `pin`. Empty when no block on record names `pin`'s
    counts.
    """
    state = cast(SessionState, ctx.state)
    blocks = {
        spec_id: block
        for spec_id, source in held_sources(state).items()
        if (block := source.spec_block(spec_id)) is not None
    }
    for spec_id, block in stored_mapping(ctx.state, ORCHESTRATOR_SPECS_KEY).items():
        blocks.setdefault(spec_id, block)
    if pin not in blocks and (earlier := earlier_block(pin)) is not None:
        blocks[pin] = earlier
    offers: list[tuple[str, Mapping[str, Any]]] = [(pin, blocks[pin])] if pin in blocks else []
    offers += [(spec_id, block) for spec_id, block in blocks.items() if spec_id != pin]
    for spec_id, block in offers:
        try:
            elected = source_from_spec_block(block).elected_measure
        except ValueError:
            continue
        if elected is None:
            continue
        counts = {elected.label: spec_id} | {
            other.label: other.spec_id for other in elected.other_counts()
        }
        if pin in counts.values():
            return counts
    return {}


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


def _readings(ctx: NodeCtx[GraphEvent]) -> list[SearchReading]:
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
    on_record: Mapping[str, PinSource],
    stated: PlannerPlan,
    held: Mapping[str, AlignedInput],
    to_origin: Mapping[str, str],
    readings: Sequence[SearchReading],
) -> _Target:
    """What switching `decision`, a decision of `plan`'s switched row, moves (`_Target`).

    A pin's own scope decision (`planner.plan.is_scope_choice`) moves the pin it
    fixes, a windowed copy read as the pin it copies (`to_origin`), from the
    newest reading of it with a scope holding the decision's axis. With no such
    reading, it moves from the block on record for the pin (`on_record`,
    `planner.plan.held_sources`), the one the card read the row from: a
    dataset the planner was shown, or a compile it was not. The switched
    dataset is then read as the requirements stating the pin (`stated`, the
    plan as the planner states it) state it (`_stated_reading`). A decision on an
    axis alignment moved for one requirement (`held`), which only that
    requirement cites and which fixes the pin it states, moves that requirement
    alone, from the compile it is pinned to (`planner.plan.compiled_pin`): its
    scope and block are the compile's, and its reading the newest of the pin the
    alignment record says it states (`AlignedInput.stated_pin`), with a scope or
    none, whose words and roles the input keeps.

    Raises:
        SwitchFailedError: If `decision` is neither, nothing on record for its
            pin holds what the switch reads, or the requirements stating the
            pin read it in different words or parties.
    """
    if decision.fixes is None or decision.scope is None:
        raise _choice_gone()
    axis = decision.scope.axis
    pinned = decision.fixes.source_spec_id or ""
    pin = to_origin.get(pinned, pinned)
    if is_scope_choice(decision):
        newest = _newest_with(readings, pin=pin, axis=axis)
        if newest is not None:
            reading, choice = newest
            assert reading.scope is not None
            return _Target(pin=pin, reading=reading, scope=reading.scope, choice=choice)
        source = on_record.get(pin)
        block = source.spec_block(pin) if source is not None else None
        scope, choice = _scope_on(pin, block, axis=axis)
        return _Target(
            pin=pin,
            reading=_stated_reading(stated, pin=pin),
            scope=scope,
            choice=choice,
            block=block,
        )
    if decision.caveat is not None:
        raise _choice_gone()
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
        raise _choice_gone()
    compiled = compiled_pin(state, moved)
    if compiled is None:
        raise _record_gone()
    scope, choice = _scope_on(compiled.spec_id, compiled.block, axis=axis)
    stated_pin = moved.stated_pin
    reading = next((reading for reading in readings if reading.spec_id == stated_pin), None)
    if stated_pin is None or reading is None:
        raise _record_gone()
    return _Target(
        pin=stated_pin,
        reading=reading,
        scope=scope,
        choice=choice,
        moved_for=moved.requirement_id,
        block=compiled.block,
    )


def _choice_gone() -> SwitchFailedError:
    """The refusal of a switch whose row the plan no longer holds."""
    return SwitchFailedError("That choice is no longer on the plan.", code=SCOPE_CHOICE_NOT_APPLIED)


def _record_gone() -> SwitchFailedError:
    """The refusal of a switch with no record of its source's scope to post from."""
    return SwitchFailedError(
        "Search's record of this source is gone.", code=SCOPE_CHOICE_NOT_APPLIED
    )


def _scope_on(
    spec_id: str, block: Mapping[str, Any] | None, *, axis: str
) -> tuple[PartScope, ScopeChoice]:
    """The scope `block`, the spec block of `spec_id`, carries and its choice on `axis`.

    Raises:
        SwitchFailedError: If there is no block, or it carries no scope or no
            choice on `axis`.
    """
    scope = part_scope(spec_id, block)
    choice = next((c for c in scope.choices if c.axis == axis), None) if scope else None
    if scope is None or choice is None:
        raise _record_gone()
    return scope, choice


def _stated_reading(stated: PlannerPlan, *, pin: str) -> SearchReading:
    """`pin`, which no reading names, read as each requirement of `stated` pinning it
    states it: under its measure, naming and holding open the parties a search for
    it would (`translator.discovery.requirement_parties`).

    A switch re-pins every one of those requirements and stores one reading of
    the switched dataset, so they must agree.

    Raises:
        SwitchFailedError: If no requirement of `stated` pins `pin`, or those
            that do name it in different words or parties.
    """
    worded = {
        (requirement.measure, *requirement_parties(requirement))
        for requirement in stated.evidence_requirements
        if isinstance(requirement, DatasetRequirement) and requirement.source_spec_id == pin
    }
    if not worded:
        raise _record_gone()
    if len(worded) > 1:
        raise SwitchFailedError(
            "This source is read in different words across the plan, so it cannot be "
            "switched for all of them at once.",
            code=SCOPE_CHOICE_NOT_APPLIED,
        )
    ((text, asked_roles, open_roles),) = worded
    return SearchReading(
        text=text,
        status="definite",
        spec_id=pin,
        asked_roles=asked_roles,
        open_roles=open_roles,
    )


def _newest_with(
    readings: Sequence[SearchReading], *, pin: str, axis: str
) -> tuple[SearchReading, ScopeChoice] | None:
    """The newest of `readings` of `pin` whose scope holds a choice on `axis`, and
    that choice; `None` when no reading of `pin` holds one."""
    for reading in readings:
        if reading.spec_id != pin or reading.scope is None:
            continue
        choice = next((c for c in reading.scope.choices if c.axis == axis), None)
        if choice is not None:
            return reading, choice
    return None


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

    The scope is the reading's, or, for an axis alignment moved or a pin no
    reading names, the block's the switch read it from (`_Target`), and
    `entries` the switch's (`switch_entries`). The same
    resubmit a compile on another basis posts
    (`planner.source_reads.compile_on_basis`), through
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
    for a source read from its block (`_Target.block`), those the block marks,
    every axis alignment moved among them.
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
    """The pin source of a switched dataset: the one it replaces, holding the switched block.

    It keeps none of the replaced source's warnings. A card lists rows of a
    pin from a search part that dropped part of the ask only once the
    translator's own search, asking the requirement's roles and years,
    answered with that pin and dropped nothing (`translator.discovery`), so
    the identity the switch resubmits from answers the requirement. The
    switched dataset is Search's answer for that identity on the option the
    user picked, which posts no text for Search to drop, and the run
    retrieves the dataset the card shows (`translator.discovery.narrowed_pin`).
    A reading of it in a later turn carries no warnings either
    (`_reading_digest`).
    """
    (part,) = digest["part_digests"]
    block = part["spec_block"]
    if replaced is None:
        return PinSource(block=block, settled=True, asked_roles=reading.asked_roles)
    return replaced.model_copy(update={"block": block, "earlier_turn": False, "warnings": ()})


def _restorable_state(ctx: NodeCtx[GraphEvent]) -> dict[str, Any]:
    """A copy of the state a switch that fails or is stopped puts back:
    the drafting state (`_DRAFTING_PREFIXES`, less the resolved parts and their times) and
    `_RESTORED_KEYS`."""
    state = ctx.state
    kept = {key: value for key, value in state.items() if _drafting(key)}
    kept |= {key: state.get(key, unset) for key, unset in _RESTORED_KEYS.items()}
    return json.loads(json.dumps(kept))


def _restore(ctx: NodeCtx[GraphEvent], before: Mapping[str, Any]) -> None:
    """Put the state back as `before` holds it, clearing any drafting key written since."""
    for key in list(ctx.state):
        if _drafting(key) and key not in before:
            ctx.state[key] = None
    for key, value in before.items():
        ctx.state[key] = value


def _drafting(key: str) -> bool:
    """Whether `key` is drafting state a failed switch puts back (`_DRAFTING_PREFIXES`)."""
    return key.startswith(_DRAFTING_PREFIXES) and not key.startswith(
        (RESOLVED_STATE_PREFIX, RESOLVED_AT_STATE_PREFIX)
    )


async def apply_stored_choice(
    ctx: NodeCtx[GraphEvent],
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
    any fetch's do. Once every fetch returned data, the chip applied is
    recorded under `ORCHESTRATOR_APPLIED_KEY`, with the option it moved from
    and the spec_id Search compiled, and its step succeeds.

    A chip that cannot be applied, because no reading holds the choice or
    the newest one is not `spec_id` (`CHIP_STALE_NOTE`, as the studio refuses
    it), the option is not one Search can serve, or Search refuses or fails
    the resubmit, fetches nothing: the hand-back carries the note a plan-card
    switch shows, and the `APPLY_CHOICE_STEP_TOOL` step fails with it under
    the code the switch reports (`SwitchFailedError.code`), to be retried
    after a delay where the same chip can succeed later
    (`SwitchFailedError.retryable`) and otherwise not to be retried. A fetch
    refused fails the chip the same way under the fetch's code (`_fetched`),
    with nothing recorded as applied.

    A count chip (`is_count_choice`) is applied by `_apply_other_count`.

    Returns:
        `{"status", "text"}` for the `orchestrate` node.
    """
    if is_count_choice(choice_id):
        return await _apply_other_count(
            ctx,
            choice_id=choice_id,
            option_id=option_id,
            spec_id=spec_id,
            fetch=fetch,
            record_step=record_step,
        )
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
                stored_mapping(ctx.state, ORCHESTRATOR_SPECS_KEY).get(pin) or earlier_block(pin),
                pin=pin,
            ),
            label=label,
        )
    except SwitchFailedError as exc:
        return _not_applied(record_step, args=args, label=label, error=exc)
    assert reading.scope is not None
    store_search(
        ctx.state,
        _reading_digest(switched, axis=choice.axis),
        asked=[reading.asked_roles],
        held_open=[reading.open_roles],
        share_sides=[reading.share_over],
    )
    try:
        fetched = await _fetched(ctx, fetch, spec_id=switched.spec_id, reading=reading, label=label)
    except SwitchFailedError as exc:
        return _not_applied(record_step, args=args, label=label, error=exc)
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


async def _apply_other_count(
    ctx: NodeCtx[GraphEvent],
    *,
    choice_id: str,
    option_id: str,
    spec_id: str,
    fetch: Callable[..., Awaitable[dict[str, Any]]],
    record_step: Callable[..., None] | None,
) -> dict[str, str]:
    """Apply a count chip: read the part the answer read as spec_id
    `spec_id` as `option_id`, one of the other counts its reading offers
    (`harness.turn_records.stored_count`), and return the hand-back the
    agent answers from.

    The other count is a dataset Search already named, with a definition on
    record from the turn that offered it
    (`harness.studio_state.ConversationState.recorded_identity`), so nothing
    is resubmitted. Its spec_id is fetched with the orchestrator's `fetch`
    tool once for each way the answer fetched the part
    (`FetchReading.args`), and the fetch reads it as the offering reading's
    other count (`SearchReading.sibling_of`). Once every fetch returned
    data, the chip applied is recorded under `ORCHESTRATOR_APPLIED_KEY`, with
    the count it moved from and the spec_id it moved to.

    A chip whose part reads a count other than `spec_id` (`CHIP_STALE_NOTE`), or
    whose option is not one of the other counts offered, fetches nothing and
    fails its step, as `apply_stored_choice` fails a scope chip. A fetch
    refused fails it the same way under the fetch's code (`_fetched`).
    """
    args = _Pick(choice_id=choice_id, option_id=option_id)
    counted = stored_count(conversation_readings(ctx.state), choice_id)
    reading = counted.reading if counted is not None else None
    other = (
        next((o for o in reading.alternatives if o.spec_id == option_id), None)
        if reading is not None
        else None
    )
    label = other.label if other is not None else option_id
    try:
        if counted is None or reading is None:
            raise SwitchFailedError(
                "That choice is no longer on this answer.", code=SCOPE_CHOICE_NOT_APPLIED
            )
        if reading.spec_id != spec_id or counted.current != spec_id:
            raise SwitchFailedError(CHIP_STALE_NOTE, code=SCOPE_CHOICE_NOT_APPLIED)
        if other is None:
            raise SwitchFailedError(
                f"Search does not offer {label} for this source.", code=_OPTION_UNAVAILABLE
            )
        fetched = await _fetched(ctx, fetch, spec_id=option_id, reading=reading, label=label)
    except SwitchFailedError as exc:
        return _not_applied(record_step, args=args, label=label, error=exc)
    ctx.state[ORCHESTRATOR_APPLIED_KEY] = {**args, "from_option_id": spec_id, "spec_id": option_id}
    logger.info(
        "count choice applied: choice_id=%s option_id=%s from=%s", choice_id, option_id, spec_id
    )
    _apply_step(
        record_step,
        args=args,
        result={"outcome": "ok", "frame_refs": [], "summary": f"counting {label}"},
    )
    held = f" in place of {reading.counted_as}" if reading.counted_as else ""
    lead = (
        f'The user chose to count "{reading.text or reading.sentence or spec_id}" as {label}'
        f"{held}. Search names that count as {option_id}, and its data was fetched for you "
        "below, once for each way the answer fetched it. Answer the user's last question "
        "again from these figures and say what the change did to them. Search and plan are "
        "not available this turn."
    )
    return {
        "status": f"Counting {label}",
        "text": "\n".join([lead, *(json.dumps(result) for result in fetched)]),
    }


# The codes of a fetch refusal that can pass if the same fetch is made later:
# `fetch` words these "Fetch it again" and states no retry mode of its own.
_RETRYABLE_FETCH_CODES = frozenset(
    {HarnessCode.TRANSIENT_UNAVAILABLE.value, HarnessCode.DEADLINE_EXCEEDED.value}
)


async def _fetched(
    ctx: NodeCtx[GraphEvent],
    fetch: Callable[..., Awaitable[dict[str, Any]]],
    *,
    spec_id: str,
    reading: SearchReading,
    label: str,
) -> list[dict[str, Any]]:
    """The results of fetching `spec_id` with the orchestrator's `fetch` tool
    once for each way the answer fetched `reading` (`FetchReading.args`),
    one after another, as the agent's own calls run.

    Raises:
        SwitchFailedError: At the first fetch refused (a result stating a
            `code`), under that code, retryable when the refusal's retry
            mode or its code says the same fetch can pass later. A fetch
            that raised (`HarnessCode.RAISED`) carries its answer
            (`SwitchFailedError.answer`). Later fetches are not made.
    """
    fetched: list[dict[str, Any]] = []
    for shaped in (reading.fetch.args if reading.fetch is not None else ()) or ({},):
        result = await fetch(
            spec_id=spec_id,
            call=OrchestratorCall(state=ctx.state, turn_id=turn_id(ctx.key)),
            **shaped,
        )
        code = result.get("code")
        if code:
            retryable = (
                result.get("retry_mode") == RetryMode.RETRY_AFTER_DELAY
                or code in _RETRYABLE_FETCH_CODES
            )
            raise SwitchFailedError(
                f"The data counted as {label} could not be fetched"
                f"{' just now' if retryable else ''}.",
                code=str(code),
                retryable=retryable,
                answer=result if code == HarnessCode.RAISED else None,
            )
        fetched.append(result)
    return fetched


def _not_applied(
    record_step: Callable[..., None] | None,
    *,
    args: _Pick,
    label: str,
    error: SwitchFailedError,
) -> dict[str, str]:
    """Fail the `APPLY_CHOICE_STEP_TOOL` step of a chip that cannot be applied
    with `error`, and return the hand-back telling the agent so, followed by
    the answer of a fetch that raised (`SwitchFailedError.answer`)."""
    note = str(error)
    logger.info(
        "scope choice not applied: choice_id=%s option_id=%s code=%s",
        args["choice_id"],
        args["option_id"],
        error.code,
    )
    _apply_step(
        record_step,
        args=args,
        result={
            "outcome": "refusal",
            "refusal": {
                "code": error.code,
                "retry_mode": "retry_after_delay" if error.retryable else "do_not_retry",
                "fix": None,
                "message": note,
            },
        },
    )
    lead = (
        f"The user asked to count {label}, and it was not applied: {note} Tell the "
        "user so in a sentence. The earlier answer stands."
    )
    raised = [json.dumps(error.answer)] if error.answer is not None else []
    return {"status": f"Counting {label}", "text": "\n".join([lead, *raised])}


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
