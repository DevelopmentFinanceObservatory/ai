"""The turn's state contract.

The keys and statuses the turn's nodes and the orchestrator's tools
(`harness.orchestrator_tools`) read and write, the inputs a turn opens with
(`TurnInput`), the one pending action the agent's hand-off records, the
tools' budgets, and the helpers more than one node reads: the stored plan
(`readable_plan`), the answers to the agent's question (`resume_answers`),
and the read-only reads of the large containers (`stored_mapping`,
`stored_sequence`).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import (
    Mapping,
    Sequence,
)
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, NamedTuple

from data_mcp.search_tool import MAX_PARTS
from planner.plan import (
    AskBreakdown,
    DatasetRequirement,
    SessionState,
    Unit,
    earlier_block,
    held_sources,
    scope_unread,
    stored_plan,
    with_blocks,
)
from planner.plan import Plan as PlannerPlan
from pydantic import BaseModel, BeforeValidator, Field, TypeAdapter, ValidationError
from submission.prices_choice import UNIT_SLOT_PREFIX, measure_key, plan_unit

from harness.tool_contracts import (
    reading_facts,
)
from harness.turn.events import APPROVE_UNIT_OFFERS_KEY
from harness.turn.store import SPECS_KEY, FrameKey, FrameKeys
from harness.turn_records import Origin, PricesForm, SearchReading, open_prices, prices_form
from harness.unit_offer import offered_unit_refusal, unit_offer
from harness.wire import UnitChoice, UnitOffer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from harness.turn.node import NodeCtx

# The kind of the pause the `approve` node puts the plan card to the user
# under, read by `harness.studio_translate` to tell the plan card from a
# question.
APPROVE_PAUSE: Final = "approve"

# The orchestrator's turn state, written by its tools
# (`harness.orchestrator_tools`) and read by the turn's nodes.
ORCHESTRATOR_FRAMES_KEY = "orchestrator:frames"  # frame id -> {"ref", "view", "store_ref"}
# spec_id -> its search spec block, kept by the turn store as rows of their own.
ORCHESTRATOR_SPECS_KEY = SPECS_KEY
# The action the agent's `plan` or `ask_user` call recorded, which the
# `orchestrate` node takes once the run hands off (`pending_action`).
ORCHESTRATOR_PENDING_KEY = "orchestrator:pending"
ORCHESTRATOR_CALLS_KEY = "orchestrator:calls"  # tool name -> calls this turn
ORCHESTRATOR_BRIEF_KEY = "orchestrator:brief"  # the question the planner drafts for
ORCHESTRATOR_READINGS_KEY = "orchestrator:readings"  # spec_id -> a plan step's stat_vars reading
# Every search part this turn read, oldest first, as `SearchReading` JSON
# (`harness.turn_records`), failed parts included; `fetch` adds its outcome
# to the readings of its spec_id.
ORCHESTRATOR_SEARCHES_KEY = "orchestrator:searches"
# Every plan this turn ran, oldest first, as `PlanRecord` JSON, written by
# `plan_done`; `harness.studio` keeps it with the turn's record.
ORCHESTRATOR_PLAN_RECORDS_KEY = "orchestrator:plan_records"
# Compile spec_id -> how a plan step's retrieval of it was served, as
# `harness.scope_choices.served_facts` JSON, for each compile alignment moved
# inputs onto (`planner.plan.aligned_pins`), which no search reading holds.
# Written by `execute` (`record_served_scopes`), read by `plan_done`
# (`record_plan`) into the plan record's `PlanSource.aligned_basis`.
ORCHESTRATOR_ALIGNED_SERVED_KEY = "orchestrator:aligned_served"
# spec_id -> the warnings of the search part that returned it, as `search`
# stores its block under `ORCHESTRATOR_SPECS_KEY`.
ORCHESTRATOR_SPEC_WARNINGS_KEY = "orchestrator:spec_warnings"
# The spec_ids whose unit the user settled this turn: each part a unit
# question covered and the count picked beside it, whether Search served,
# re-read or refused the chosen unit, and each spec_id it re-read them as.
# Their prices are closed (`unit_settled`, `settle_unit`).
ORCHESTRATOR_UNIT_SETTLED_KEY = "orchestrator:unit_settled"
# The spec_ids the `ask` node re-read each part in, by the part a unit question
# covered (`unit_reads`, `record_unit_read`), so the prices the user chose
# reach the re-read series (`planner.call.prices_chosen`).
ORCHESTRATOR_UNIT_READ_KEY = "orchestrator:unit_read"
# Each part a unit question covered -> the series the agent fetches for it,
# the unit that series is served in after the answer, the question's lead
# part and the unit it chose, as `ServedUnit` JSON (`units_served`,
# `record_unit_served`): the chosen unit when Search serves or re-reads it,
# the served unit when Search refuses the re-read. Written once, by the `ask`
# node: admission holds the part to that unit in the plan that first states
# it (`planner.call.prices_chosen`), and the plan holds it from then on
# (`submission.service._user_prices`).
ORCHESTRATOR_UNIT_SERVED_KEY = "orchestrator:unit_served"
# Each part a unit question put on a unit row that the plan does not read
# -> the series Search read it as in the unit a switch of the row picked,
# and that unit, as `ServedUnit` JSON (`units_reread`, `record_units_reread`).
# Written only by a unit row's switch (`harness.graph.switch`), for those
# parts alone, and read only for them (`submission.service._user_prices`):
# a part a later plan adds is pinned in the row's unit to that series.
ORCHESTRATOR_UNIT_REREAD_KEY = "orchestrator:unit_reread"
# The search readings of the conversation's earlier turns, with the warnings
# of the spec_ids they name and the spec_ids whose block an earlier turn
# stored, seeded by `harness.studio` from `ConversationState.
# prior_ledger_state`; unset when no earlier turn read any. The blocks
# themselves resolve through `planner.plan.earlier_block`.
ORCHESTRATOR_PRIOR_LEDGER_KEY = "orchestrator:prior_ledger"
# The orchestrator's conversation while its run waits on a hand-back, as
# `orchestrator.run.StoredHistory` JSON, written and taken by the
# `orchestrate` node.
ORCHESTRATOR_HISTORY_KEY = "orchestrator:history"
# Under `STUDIO_HISTORY=full`, the orchestrator's conversation once its run
# closed on the turn's reply, `{"history", "tokens"}`: the
# `orchestrator.run.StoredHistory` JSON as text, so a JSONB column cannot
# reorder its keys, and the tokens its final call reported
# (`harness.graph.orchestrate.carried_record`). Written by the `orchestrate`
# node and kept with the conversation for the next turn's opening
# (`harness.studio_store.keep_turn_records`).
ORCHESTRATOR_CARRIED_KEY = "orchestrator:carried"
# spec_id -> the refusal message of a fetch or a plan step that read it and
# was refused `capability_unavailable` this turn; `fetch` refuses it again
# without a call.
ORCHESTRATOR_UNAVAILABLE_KEY = "orchestrator:unavailable"
# The chip a turn applies, `{"choice_id", "option_id", "spec_id"}`
# (`harness.wire.ApplyChoice`), in the opening state of a turn requested
# with `apply_choice`; `seed` applies it
# (`harness.graph.switch.apply_stored_choice`) and `turn_budgets` reads it,
# the orchestrator's tools included, so it is state rather than an input.
APPLY_CHOICE_STATE_KEY = "apply_choice"
# The chip `seed` applied this turn: `{"choice_id", "option_id",
# "from_option_id", "spec_id"}`, the option it moved from and the spec_id
# Search compiled. For a count chip, they are the spec_ids of the count it
# moved from and of the count it moved to. `harness.studio` keeps it with the turn's record
# (`ConversationState.turn_applied_choice`), where the next turn reads whether
# the user kept it.
ORCHESTRATOR_APPLIED_KEY = "orchestrator:applied_choice"
# Ask number -> the terminal frames of the latest run of this plan call that
# computed that ask (`_kept_asks`), written by `execute` when a run stops and
# read by `plan_done`; cleared by `plan_start` and `plan_done`.
ORCHESTRATOR_KEPT_ASKS_KEY = "orchestrator:kept_asks"
# The `failed_step` (`executor.outcomes.failed_step`) of the latest run of this
# plan call that stopped and went back to the planner, written by `execute` on
# its `"replan"` edge. `plan_done` reads its remedy when the planner then
# refuses. `plan_start` and `plan_done` clear it.
ORCHESTRATOR_REPLANNED_STOP_KEY = "orchestrator:replanned_stop"
# The run the studio's interrupt route admitted with the approval of the
# plan `approve` last approved (`harness.studio`), or None when it admitted
# none: written by `approve.decide` on every approval and read by `execute`,
# which then finds the run and its stored plan already there.
APPROVED_RUN_KEY = "orchestrator:approved_run"
# The user's words for each amendment of this plan call's plan that the
# planner drafted, in order, each cut to `planner.plan.MAX_QUESTION_CHARS`.
# Written by the `planner` node, cleared by `plan_start`, and handed to the
# agent by `plan_done`, since the amendment reaches the planner alone.
ORCHESTRATOR_AMENDMENTS_KEY = "orchestrator:amendments"
# Why the turn's next plan card is a new attempt. It names the step the
# approved plan stopped at and what happened there, with no advice, since the
# new plan is the way past it. `plan_done` writes it when it gives the plan call
# back, and the `planner` node takes it onto the next first draft as its
# `refusal_note`, which Studio shows before the card.
ORCHESTRATOR_RETRY_NOTE_KEY = "orchestrator:retry_note"
# The asks of the plan that stopped when `plan_done` gave the plan call back,
# as the `PendingPlan` fields `asks`, `ask_breakdowns` and `asks_across_years`.
# The call given back may change the brief only, so the `plan` tool holds the
# new plan to these, whatever asks it is sent. The user's answer to a later
# question this turn (`harness.graph.orchestrate`) clears it, since the user
# may then ask for other outputs.
ORCHESTRATOR_GIVEN_BACK_ASKS_KEY = "orchestrator:given_back_asks"
# The user's words typed on the plan card that the planner read as a new
# question (`starts_new_plan`), written by the `planner` node as it ends the
# turn on its `new_question` edge. The stream's `done` frame hands them to
# the client to send as the next turn (`harness.studio_translate`).
NEW_QUESTION_KEY = "orchestrator:new_question"
# Set by the `planner` node as it ends the turn on its `new_question` edge
# when the planner read the plan card's amendment as a new question that no
# message of the user's states (`new_question_in` `none`). The turn closes
# asking the user to send their question, and hands nothing back
# (`harness.studio_translate`).
ASK_FOR_QUESTION_KEY = "orchestrator:ask_for_question"
# The state a planner entry replanning for a pick on the plan card puts back
# when the replan does not land (`harness.graph.planning`), written when the
# replan pauses on a planner-side ask and cleared when the entry ends, so it
# is set only while such a replan is paused. Outside the drafting prefixes
# the restore resets (`harness.graph.switch._DRAFTING_PREFIXES`).
PICK_REPLAN_KEY = "switch:pick_replan"
# The status a plan hand-back carries when the planner made no plan, and
# when the approved plan ran; `studio_translate._Translator` shows the first as
# a failed step and the second as nothing.
ORCHESTRATOR_PLAN_FAILED_STATUS = "Reading why no plan was made"
ORCHESTRATOR_PLAN_RAN_STATUS = "Reading the plan's results"


class TurnInput(BaseModel):
    """What a turn opens with, fixed for the turn: the user's `message`, the
    canvas and conversation digests, as the JSON text the conversation wrote
    and the agent reads, and the seeds slot, the Search results a "... in
    Studio" action handed over (each seed as `harness.seeds.seeds_slot`
    keeps it, or the one reference `harness.wire.TurnRequest.seed_spec` also
    takes). The rest of what a turn opens with, which the orchestrator's
    tools also read, is opening state (`turn_budgets`).

    Under `STUDIO_HISTORY=full`, a turn that extends the previous turn's
    conversation holds it as `carried_history` (`orchestrator.run.StoredHistory`
    JSON text) and the conversation digest's tail without its `turns` as
    `conversation_tail`, which the orchestrator reads in place of
    `conversation_digest`; the planner still reads the digest. Both are absent
    from the dumped inputs when unset."""

    message: str
    canvas_digest: str | None = None
    conversation_digest: str | None = None
    seeds: list[Any] | dict[str, Any] | None = None
    carried_history: str | None = Field(default=None, exclude_if=lambda value: value is None)
    conversation_tail: str | None = Field(default=None, exclude_if=lambda value: value is None)


class AskOption(BaseModel):
    """One answer `ask_user` offers: its wording and, for a part's candidate, its spec_id."""

    label: str = Field(description="the answer as the user reads it")
    spec_id: str | None = Field(
        default=None,
        description=(
            "for one of a part's candidates, the spec_id the search result gives it; "
            "leave it out for any other answer"
        ),
    )


def ask_options(options: list[Any]) -> list[dict[str, Any]]:
    """The `ask_user` options the model sent as `AskOption` dicts. Gemini
    does not hold a call's arguments to the declaration, so an option sent
    as its bare label reads here as an option without a spec_id."""
    return [{"label": option} if isinstance(option, str) else option for option in options]


class AskQuestion(BaseModel):
    """One question `ask_user` puts to the user, with the answers it offers."""

    question: str = Field(description="the question as the user reads it")
    options: Annotated[list[AskOption], BeforeValidator(ask_options)] = Field(
        default_factory=list,
        description=(
            "the likely answers, at least one; the user may also write their own. Leave it "
            "empty for a unit question, whose answers are filled in for you"
        ),
    )
    unit_of: str | None = Field(
        default=None,
        description="for the unit question of a part that lists `open_prices`, that part's spec_id",
    )
    prices: Literal["current", "constant"] | None = Field(
        default=None,
        description=(
            "for a unit question, the prices a skill prefers the user starts on; leave it out "
            "otherwise"
        ),
    )
    multi_select: bool = Field(
        default=False,
        description=(
            "true when several answers can hold together and the user may tick more than "
            "one; the options are then plain labels, with no spec_id, and no `unit_of`"
        ),
    )


class PendingAsk(BaseModel):
    """`ask_user` asked: the workflow pauses with its questions, in order."""

    kind: Literal["ask"] = "ask"
    questions: list[AskQuestion]


class PendingPlan(BaseModel):
    """`plan` asked: the workflow runs the plan path on the brief, holding the
    plan to `asks`, the outputs the user asked for, to `ask_breakdowns`, the
    parties those asks break down by, and to `asks_across_years`, the numbers
    of the asks put to one figure over several years."""

    kind: Literal["plan"] = "plan"
    brief: str
    asks: list[str] = []
    ask_breakdowns: list[AskBreakdown] = []
    asks_across_years: list[int] = []


def stored_mapping(state: SessionState, key: str) -> Mapping[str, Any]:
    """The dict `state` holds under `key` as a read-only view, empty when
    the key is unset.

    For the large containers a turn reads many times (frames, spec blocks,
    readings, the prior ledger). An assignment through the view raises
    `TypeError`, so every change is an assignment to the key, which the
    orchestrator's `harness.gemini_agent.RecordingState` records. Its
    entries are the stored objects, so a change is built on a copy
    (`dict(view)`, `{**entry, ...}`) and assigned to the key.
    """
    return MappingProxyType(state.get(key) or {})


def stored_sequence(state: SessionState, key: str) -> tuple[Any, ...]:
    """The list `state` holds under `key` as a tuple, empty when the key is
    unset. A tuple has no method that changes it in place, as
    `stored_mapping`'s view has none."""
    return tuple(state.get(key) or ())


def _turn_readings(state: SessionState) -> list[tuple[SearchReading, Mapping[str, Any]]]:
    """Each reading of the turn with a spec_id, with its stored spec block, oldest first."""
    specs = stored_mapping(state, ORCHESTRATOR_SPECS_KEY)
    found = []
    for stored in stored_sequence(state, ORCHESTRATOR_SEARCHES_KEY):
        reading = SearchReading.model_validate(stored)
        block = specs.get(reading.spec_id) if reading.spec_id is not None else None
        if block is not None:
            found.append((reading, block))
    return found


def unit_settled(state: SessionState) -> frozenset[str]:
    """The spec_ids whose unit the user settled this turn (`ORCHESTRATOR_UNIT_SETTLED_KEY`)."""
    return frozenset(stored_sequence(state, ORCHESTRATOR_UNIT_SETTLED_KEY))


def settle_unit(state: SessionState, *spec_ids: str) -> None:
    """Record `spec_ids` as settled in the unit the user chose: their prices are closed
    from here, so no reading lists `open_prices` for them and a second unit question
    on them is refused."""
    state[ORCHESTRATOR_UNIT_SETTLED_KEY] = sorted({*unit_settled(state), *spec_ids})


def unit_reads(state: SessionState) -> dict[str, list[str]]:
    """The spec_ids each part a unit question covered was re-read in this turn
    (`ORCHESTRATOR_UNIT_READ_KEY`), by the part."""
    return {
        str(part): [str(spec_id) for spec_id in spec_ids]
        for part, spec_ids in stored_mapping(state, ORCHESTRATOR_UNIT_READ_KEY).items()
        if isinstance(spec_ids, list)
    }


def record_unit_read(state: SessionState, part: str, spec_id: str) -> None:
    """Record that `part`, which a unit question covered, was re-read as `spec_id`."""
    reads = unit_reads(state)
    reads[part] = list(dict.fromkeys([*reads.get(part, ()), spec_id]))
    state[ORCHESTRATOR_UNIT_READ_KEY] = reads


class ServedUnit(BaseModel):
    """What a part a unit question covered is fetched as after the answer:
    `spec_id`, the series the agent was told to fetch, and `unit`, the unit
    that series is served in, `None` when its unit cannot be read as one a
    plan states (`stated_unit`). `group` is the lead part of the unit
    question that covered the part (`UnitAsk.parts[0]`), which every part
    that question covered shares, and `chosen` the unit that question's
    answer asked the part to be read in (`stated_unit`); each is `None` on a
    record that names none."""

    spec_id: str
    unit: Unit | None
    group: str | None = None
    chosen: Unit | None = None

    @property
    def row(self) -> str | None:
        """The lead part of the unit row the part sits on
        (`submission.prices_choice.unit_slot`): `group`, when the part is
        served in the unit `chosen` names. `None` for a part Search refused
        to read in that unit, which keeps a prices row of its own, and for a
        record naming no group or no chosen unit."""
        if self.group is None or self.chosen is None or self.unit != self.chosen:
            return None
        return self.group


def units_served(state: SessionState) -> dict[str, ServedUnit]:
    """The series and unit each part a unit question covered this turn is
    fetched as (`ORCHESTRATOR_UNIT_SERVED_KEY`), by the part."""
    return {
        str(part): ServedUnit.model_validate(served)
        for part, served in stored_mapping(state, ORCHESTRATOR_UNIT_SERVED_KEY).items()
    }


def units_reread(state: SessionState) -> dict[str, ServedUnit]:
    """The series and unit a switch of each part's unit row read each part
    the plan did not read as (`ORCHESTRATOR_UNIT_REREAD_KEY`), by the part."""
    return {
        str(part): ServedUnit.model_validate(read)
        for part, read in stored_mapping(state, ORCHESTRATOR_UNIT_REREAD_KEY).items()
    }


def record_units_reread(state: SessionState, reads: Mapping[str, tuple[str, Unit]]) -> None:
    """Record each part of `reads` as read as its series in its unit, by
    the part, in place of what an earlier switch read it as. Nothing to
    record writes nothing."""
    if not reads:
        return
    stored = {part: read.model_dump(mode="json") for part, read in units_reread(state).items()}
    for part, (spec_id, unit) in reads.items():
        stored[part] = ServedUnit(spec_id=spec_id, unit=unit).model_dump(mode="json")
    state[ORCHESTRATOR_UNIT_REREAD_KEY] = stored


def stated_unit(unit: UnitChoice) -> Unit | None:
    """`unit` as a plan states it (`planner.plan.Unit`), `None` when no plan can
    state it, such as a currency code that is not three capital letters."""
    try:
        return Unit.model_validate(unit.model_dump(mode="json"))
    except ValidationError:
        logger.warning("served unit %r is no unit a plan states", unit)
        return None


def record_unit_served(
    state: SessionState,
    part: str,
    spec_id: str,
    unit: UnitChoice | None,
    *,
    group: str,
    chosen: UnitChoice,
) -> None:
    """Record that `part`, which the unit question led by `group` covered, is
    fetched as `spec_id`, served in `unit`, the question's answer having
    chosen `chosen`. A unit no plan can state is recorded as none
    (`stated_unit`)."""
    stated = stated_unit(unit) if unit is not None else None
    served = {name: one.model_dump(mode="json") for name, one in units_served(state).items()}
    served[part] = ServedUnit(
        spec_id=spec_id, unit=stated, group=group, chosen=stated_unit(chosen)
    ).model_dump(mode="json")
    state[ORCHESTRATOR_UNIT_SERVED_KEY] = served


def count_block(block: Mapping[str, Any], count: str) -> dict[str, Any] | None:
    """`block` with the definition of `count`, another count it offers
    (`resolved.alternative_measures`), in place of its own; `None` when it offers none."""
    for alternative in (block.get("resolved") or {}).get("alternative_measures") or ():
        definition = alternative.get("definition") if isinstance(alternative, Mapping) else None
        if isinstance(definition, Mapping) and definition.get("spec_id") == count:
            return {**block, "definition": dict(definition)}
    return None


def _blocks_on_record(state: SessionState) -> dict[str, Mapping[str, Any]]:
    """Every spec block on record, by spec_id: this turn's searches'
    (`ORCHESTRATOR_SPECS_KEY`), then the planner's held sources'
    (`planner.plan.held_sources`)."""
    blocks: dict[str, Mapping[str, Any]] = dict(stored_mapping(state, ORCHESTRATOR_SPECS_KEY))
    for spec_id, source in held_sources(state).items():
        if spec_id not in blocks and (block := source.spec_block(spec_id)) is not None:
            blocks[spec_id] = block
    return blocks


def counted_from(state: SessionState, pin: str) -> str | None:
    """The series whose block on record offers `pin` as another count of it
    (`resolved.alternative_measures`), as a count switch of the plan card
    pins; `None` when no block on record does."""
    return next(
        (
            spec_id
            for spec_id, block in _blocks_on_record(state).items()
            if spec_id != pin and count_block(block, pin) is not None
        ),
        None,
    )


def own_block(state: SessionState, pin: str) -> Mapping[str, Any] | None:
    """The spec block on record for the series `pin` itself
    (`_blocks_on_record`, else an earlier turn's, `planner.plan.earlier_block`)."""
    return _blocks_on_record(state).get(pin) or earlier_block(pin)


def pin_block(state: SessionState, pin: str) -> Mapping[str, Any] | None:
    """The spec block of the series `pin`: its own (`own_block`), else, for
    another count a block on record offers (`counted_from`), which a count
    switch may pin with no block of its own, that block with the count's
    definition (`count_block`). `None` when nothing on record states it."""
    if (block := own_block(state, pin)) is not None:
        return block
    source = counted_from(state, pin)
    return count_block(_blocks_on_record(state)[source], pin) if source is not None else None


def unit_rows(plan: PlannerPlan) -> dict[str, list[str]]:
    """The series each unit row of `plan` covers, by the row's decision slot
    (`submission.prices_choice.unit_slot`): the distinct pins of the dataset
    requirements citing it, in plan order. `plan` is read as the planner
    states it (`submission.service.statement`)."""
    rows: dict[str, list[str]] = {}
    for decision in plan.decisions:
        if not decision.slot.startswith(UNIT_SLOT_PREFIX):
            continue
        rows[decision.slot] = list(
            dict.fromkeys(
                requirement.source_spec_id
                for requirement in plan.evidence_requirements
                if isinstance(requirement, DatasetRequirement)
                and decision.slot in requirement.decision_ids
                and requirement.source_spec_id is not None
            )
        )
    return rows


def unit_row_offers(state: SessionState, plan: PlannerPlan) -> dict[str, UnitOffer]:
    """The units the plan card's row for each unit question's whole unit offers,
    by the row's decision slot (`submission.prices_choice.unit_slot`).

    `plan` is read as the planner states it (`submission.service.statement`).
    A row's series are the pins of the requirements citing it (`unit_rows`),
    and its offer is what the block of each converts into (`pin_block`,
    `unit_offer`). It starts on the unit the plan holds the row to
    (`submission.prices_choice.plan_unit`). A row with a series whose block
    is not on record, whose requirements state no one unit, or whose blocks
    share no unit both bases serve or leave that unit out, offers none.
    """
    offers: dict[str, UnitOffer] = {}
    for slot, pins in unit_rows(plan).items():
        blocks = [pin_block(state, pin) for pin in pins]
        unit = plan_unit(plan, slot)
        if not blocks or unit is None:
            continue
        offer = unit_offer([block for block in blocks if block is not None])
        if offer is None or None in blocks:
            continue
        start = UnitChoice.model_validate(unit.model_dump(mode="json"))
        if (refusal := offered_unit_refusal(offer, start)) is not None:
            logger.warning("unit row %s offers no switch: %s", slot, refusal)
            continue
        offers[slot] = offer.model_copy(update={"start": start})
    return offers


def save_card_unit_offers(
    state: SessionState, revision: object, offers: Mapping[str, UnitOffer]
) -> None:
    """Record `offers` as the units the rows of the plan card of `revision`
    offer (`harness.turn.events.APPROVE_UNIT_OFFERS_KEY`), or clear the
    record when none does. They sit beside the pause, since a pause's
    payload is read back as a plan, and its report carries a copy
    (`harness.turn.events.Paused.state`)."""
    if not offers:
        if state.get(APPROVE_UNIT_OFFERS_KEY) is not None:
            state[APPROVE_UNIT_OFFERS_KEY] = None
        return
    state[APPROVE_UNIT_OFFERS_KEY] = {
        "revision": revision,
        "offers": {slot: offer.model_dump(mode="json") for slot, offer in offers.items()},
    }


def card_unit_offers(state: Mapping[str, Any], revision: object) -> dict[str, UnitOffer]:
    """The units the rows of the plan card of `revision` offer, by the row's
    slot, as the `approve` node recorded them in `state`; none for a card of
    another revision or a state that records none."""
    stored = state.get(APPROVE_UNIT_OFFERS_KEY)
    if not isinstance(stored, Mapping) or stored.get("revision") != revision:
        return {}
    return {
        str(slot): UnitOffer.model_validate(offer)
        for slot, offer in (stored.get("offers") or {}).items()
    }


def _reading_prices(reading: SearchReading) -> dict[str, Any]:
    """The arguments `harness.turn_records.open_prices` and `prices_form` read off `reading`."""
    return {
        "window": reading.window.model_dump() if reading.window is not None else None,
        "other_prices": [offer.model_dump() for offer in reading.other_prices],
    }


def open_parts(state: SessionState) -> dict[str, list[dict[str, Any]]]:
    """The `open_prices` entries of each reading of the turn whose prices are open,
    by its spec_id, each with its `prices`: the first the prices search read,
    current, and each after it constant (`harness.turn_records.open_prices`).

    Read over the readings of `ORCHESTRATOR_SEARCHES_KEY` and the blocks of
    `ORCHESTRATOR_SPECS_KEY`. Only code reads `prices`: the readings the model
    reads carry the entries without it (`store_search`).
    """
    parts: dict[str, list[dict[str, Any]]] = {}
    settled = unit_settled(state)
    for reading, block in _turn_readings(state):
        assert reading.spec_id is not None
        if reading.spec_id in settled:
            continue
        if entries := open_prices(block, **_reading_prices(reading)):
            parts[reading.spec_id] = [
                {**entry, "prices": "current" if index == 0 else "constant"}
                for index, entry in enumerate(entries)
            ]
    return parts


def prices_forms(state: SessionState) -> dict[str, PricesForm]:
    """How the prices question of each reading of the turn whose prices are open is
    asked, by its spec_id (`harness.turn_records.prices_form`): "unit" or "options"."""
    forms: dict[str, PricesForm] = {}
    settled = unit_settled(state)
    for reading, block in _turn_readings(state):
        assert reading.spec_id is not None
        if reading.spec_id in settled:
            continue
        if (form := prices_form(block, **_reading_prices(reading))) != "closed":
            forms[reading.spec_id] = form
    return forms


def prices_question(
    question: AskQuestion,
    parts: Mapping[str, list[dict[str, Any]]],
    forms: Mapping[str, PricesForm],
) -> AskQuestion:
    """`question` with the answers code gives it when it asks the unit of a part of
    `parts` (`AskQuestion.unit_of`): none for a part asked in the "unit" form
    (`forms`), whose answer is a unit, and the part's `open_prices` entries, each
    naming its spec_id but the reading's own, for one asked in the "options" form."""
    part = question.unit_of
    entries = parts.get(part) if part is not None else None
    if part is None or entries is None:
        return question
    if forms.get(part) == "unit":
        return question.model_copy(update={"options": []})
    return question.model_copy(
        update={
            "options": [
                AskOption(label=entry["label"], spec_id=entry.get("spec_id")) for entry in entries
            ]
        }
    )


class UnitAsk(NamedTuple):
    """A unit question's reach: `parts`, the spec_ids of the readings it covers, the
    one it names first leading, and `offer`, the units every one of them converts into."""

    parts: tuple[str, ...]
    offer: UnitOffer


def unit_asks(questions: Sequence[AskQuestion], state: SessionState) -> list[UnitAsk | None]:
    """For each of `questions`, the reach of a unit question, `None` for any other
    question or for one whose parts share no unit both bases serve.

    A unit question is one whose `unit_of` part is asked in the "unit" form
    (`prices_forms`). It covers that part and every other part asked in that
    form measuring the same thing (equal `submission.prices_choice.measure_key`),
    and offers what all of them
    convert into (`harness.unit_offer.unit_offer`), starting on the prices
    the question prefers (`AskQuestion.prices`).
    """
    forms = prices_forms(state)
    specs = stored_mapping(state, ORCHESTRATOR_SPECS_KEY)
    measures = {part: measure_key(specs[part]) for part in forms if part in specs}
    asks: list[UnitAsk | None] = []
    for question in questions:
        part = question.unit_of
        if part is None or forms.get(part) != "unit" or part not in specs:
            asks.append(None)
            continue
        key = measures.get(part)
        parts = (
            part,
            *(
                other
                for other, form in forms.items()
                if other != part
                and form == "unit"
                and other in specs
                and key is not None
                and measures.get(other) == key
            ),
        )
        offer = unit_offer([specs[p] for p in parts], prefer=question.prices)
        asks.append(UnitAsk(parts, offer) if offer is not None else None)
    return asks


def priced_options(
    questions: list[AskQuestion], state: SessionState
) -> list[dict[int, PricedOption]]:
    """For each of `questions`, the place of each option offering prices, with
    the part it prices and the prices: each option of a question pricing a
    part (`AskQuestion.unit_of`) that is one of the part's `open_prices`
    entries (`open_parts`), matched by spec_id, else by label among the
    entries carrying none. A question pricing no part has none."""
    parts = open_parts(state)
    priced: list[dict[int, PricedOption]] = []
    for question in questions:
        entries = parts.get(question.unit_of) if question.unit_of is not None else None
        marks: dict[int, PricedOption] = {}
        for index, option in enumerate(question.options):
            entry = next(
                (
                    entry
                    for entry in entries or ()
                    if entry.get("spec_id") == option.spec_id
                    and (option.spec_id is not None or entry["label"] == option.label)
                ),
                None,
            )
            if entry is not None and question.unit_of is not None:
                marks[index] = PricedOption(question.unit_of, entry["prices"])
        priced.append(marks)
    return priced


class PricedOption(NamedTuple):
    """The prices an `ask_user` option offers, as code reads them off the turn's
    search results (`priced_options`): `part`, the
    spec_id of the reading whose `open_prices` the option is one of, and
    `prices`, `current` or `constant`."""

    part: str
    prices: Literal["current", "constant"]


def ask_candidates(
    options: list[AskOption], priced: Mapping[int, PricedOption] | None = None
) -> list[dict[str, str]]:
    """One question's options as the clarify interrupt's candidates: a count
    under its own spec_id, so a pick returns it, and any other answer under
    `o{n}`, its place in the question's list. An option of `priced`, by
    place, carries its `part` and `prices`, which the planner reads off the
    answered interrupt (`planner.call.prices_chosen`)."""
    priced = priced or {}
    return [
        {
            "spec_id": option.spec_id or f"o{index + 1}",
            "label": option.label,
            **(priced[index]._asdict() if index in priced else {}),
        }
        for index, option in enumerate(options)
    ]


def ask_questions(
    questions: list[AskQuestion],
    priced: Sequence[Mapping[int, PricedOption]] = (),
    units: Sequence[UnitAsk | None] = (),
) -> list[dict[str, Any]]:
    """`PendingAsk.questions` as the clarify interrupt's payload entries: each
    under `q{n}`, its place in the call, with its candidates (`ask_candidates`),
    each question's priced options read from `priced` by place, and
    `multi_select` when the user may tick several. A unit
    question of `units`, by place, carries `unit`: the spec_ids it covers
    (`parts`) and its `offer` as `harness.wire.UnitOffer` JSON."""
    entries = []
    for index, question in enumerate(questions):
        entry: dict[str, Any] = {
            "id": f"q{index + 1}",
            "question": question.question,
            "candidates": ask_candidates(
                question.options, priced[index] if index < len(priced) else None
            ),
        }
        if question.multi_select:
            entry["multi_select"] = True
        if (unit := units[index] if index < len(units) else None) is not None:
            entry["unit"] = {
                "parts": list(unit.parts),
                "offer": unit.offer.model_dump(mode="json", exclude_none=True),
            }
        entries.append(entry)
    return entries


def resume_answers(result: Any) -> dict[str, dict[str, Any]] | None:
    """The clarify interrupt's resume value as one answer per question id, or
    None when the user dismissed the whole pause (`{"skip": True}`).

    The value `harness.studio`'s `/interrupt` route resumes a clarify with
    is `{"answers": {question id: answer}}`, each answer
    `{"kind": "option", "spec_id"}`, `{"kind": "options", "spec_ids"}`,
    `{"kind": "unit", "currency", "prices", "base_year"?}`,
    `{"kind": "text", "text"}` or `{"kind": "skipped"}`. Any other value,
    such as the plan card's answer, answers no question.
    """
    if not isinstance(result, dict):
        return {}
    if result.get("skip"):
        return None
    return dict(result.get("answers") or {})


_PENDING_ACTION = TypeAdapter(Annotated[PendingAsk | PendingPlan, Field(discriminator="kind")])


# Calls one turn may make of each of the orchestrator's tools
# (`harness.orchestrator_tools`). `search` counts every resolver call, `fetch`
# every dataset read; a plan is one per turn, and a plan that failed gives it
# back (`plan_done`) for a brief that avoids the failure. `make_charts` counts
# the calls that drew at least one chart, `make_charts_failed` the calls that
# drew none since a frame last reached the turn. A registration that adds a
# frame id clears it (`register_frames`). Each clear costs a spent `fetch` or
# a plan hand-back (at most 2 with the plan given back), so a turn makes at
# most 3 x (1 + 6 + 2) = 27 calls that draw nothing.
# `invalid_arguments` counts the calls of any tool refused for malformed
# arguments, which spend nothing of the tool's own budget.
BUDGETS = {
    # The first search, a corrected place, and one `text` search for each
    # part whose result carried `retry_with_text`.
    "search": MAX_PARTS + 2,
    "fetch": 6,
    "make_charts": 5,
    "make_charts_failed": 3,
    "ask_user": 2,
    "plan": 1,
    "get_skill": 2,
    # `get_skill` with a `path`: a file a skill links, or the next page of a long one.
    "get_skill_file": 4,
    "recall": 3,
    # recall(part="canvas") with no component: the canvas listing.
    "recall_canvas": 2,
    "remove_cards": 3,
    "invalid_arguments": 3,
}
# The download surface fetches one dataset and never plans.
DOWNLOAD_BUDGETS = {**BUDGETS, "fetch": 1, "plan": 0}
# A turn applying a chip answers from the data `seed` fetched for it, so it
# neither searches nor plans.
APPLY_CHOICE_BUDGETS = {**BUDGETS, "search": 0, "plan": 0}


def turn_budgets(state: Any) -> dict[str, int]:
    """The calls this turn may make of each orchestrator tool:
    `APPLY_CHOICE_BUDGETS` on a turn applying a chip
    (`APPLY_CHOICE_STATE_KEY`), `DOWNLOAD_BUDGETS` on the download surface,
    `BUDGETS` otherwise. `state` is the turn's state."""
    if state.get(APPLY_CHOICE_STATE_KEY):
        return APPLY_CHOICE_BUDGETS
    return DOWNLOAD_BUDGETS if state.get("surface") == "download" else BUDGETS


def count_call(state: Any, tool: str) -> None:
    """Count one more of `tool`'s calls this turn (`ORCHESTRATOR_CALLS_KEY`),
    reading the counts afresh. Synchronous, so no other tool call runs
    between a check of a budget and the change."""
    calls = dict(state.get(ORCHESTRATOR_CALLS_KEY) or {})
    calls[tool] = int(calls.get(tool, 0)) + 1
    state[ORCHESTRATOR_CALLS_KEY] = calls


def pending_action(state: Any) -> PendingAsk | PendingPlan | None:
    """The one action the agent's run recorded (`ORCHESTRATOR_PENDING_KEY`),
    or None. `state` is the turn's state."""
    raw = state.get(ORCHESTRATOR_PENDING_KEY)
    return _PENDING_ACTION.validate_python(raw) if raw else None


def frame_id_for(ref: dict[str, Any]) -> str:
    """The short id the orchestrator knows a frame by, from its reference, so
    the same frame gets the same id however many times it is registered."""
    digest = hashlib.sha1(json.dumps(ref, sort_keys=True).encode()).hexdigest()
    return f"f{digest[:6]}"


type DescribedFrame = tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]


def register_frames(
    state: SessionState,
    described: Sequence[DescribedFrame],
    *,
    fetch_args: dict[str, Any] | None = None,
    wholes: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Add each (frame ref, `describe_frame` view, store ref) to the frames in
    `state` (`ORCHESTRATOR_FRAMES_KEY`) under its id, and return the entries
    as the orchestrator reads them.

    The frame ref is the pair the frame is described and drawn by; the store
    ref is the pair the materialisation store holds it under, kept as the
    entry's `store_ref`. It is None for a frame whose executor states no
    store pair. The entry then carries no `store_ref` and is read by its
    frame ref (`frame_store_ref`). `fetch_args` are the arguments of the
    `fetch` that retrieved the frames, kept on each entry because
    `make_charts` asks the same retrieval again for the frame's members' kinds.
    `wholes` maps the id of a frame holding shares to the whole they divide
    by (`harness.plan_display.share_wholes`), kept as the entry's `whole`,
    which `make_charts` states in the card's title.

    A frame id the frames did not hold clears the turn's count of calls that
    drew nothing (`make_charts_failed`), so a chart of a frame that arrived
    after those calls is not refused for them. Synchronous, so no other tool
    call runs between the read of the frames and the writes.
    """
    frames = dict(stored_mapping(state, ORCHESTRATOR_FRAMES_KEY))
    added = False
    entries = []
    for ref, view, store_ref in described:
        frame_id = frame_id_for(ref)
        added = added or frame_id not in frames
        frames[frame_id] = {"ref": ref, "view": view}
        if store_ref is not None:
            frames[frame_id]["store_ref"] = store_ref
        if fetch_args is not None:
            frames[frame_id]["fetch_args"] = fetch_args
        if wholes and frame_id in wholes:
            frames[frame_id]["whole"] = wholes[frame_id]
        entries.append({"frame": frame_id, "view": view})
    state[ORCHESTRATOR_FRAMES_KEY] = frames
    calls = state.get(ORCHESTRATOR_CALLS_KEY) or {}
    if added and calls.get("make_charts_failed"):
        state[ORCHESTRATOR_CALLS_KEY] = {
            tool: count for tool, count in calls.items() if tool != "make_charts_failed"
        }
    return entries


def frame_store_ref(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The pair to open the materialisation store with for `entry`, an entry
    of `ORCHESTRATOR_FRAMES_KEY`: its `store_ref`, or its `ref` when the
    entry states no store pair or one with a member missing."""
    store_ref = entry.get("store_ref")
    if (
        isinstance(store_ref, Mapping)
        and store_ref.get("spec_id") is not None
        and store_ref.get("content_version") is not None
    ):
        return dict(store_ref)
    return entry["ref"]


def frame_store_keys(state: Mapping[str, Any]) -> FrameKeys:
    """The lineage store keys of the frames in `state`
    (`ORCHESTRATOR_FRAMES_KEY`): each entry's `ref` and `store_ref`, in order
    and each once, which the turn store keeps the lineage of
    (`harness.turn.store.FrameLineage`). A pair with a member missing names
    no key, and so does a frames value or entry that is not a mapping, which
    `FrameKeys.skipped` describes for the caller to report."""
    frames = state.get(ORCHESTRATOR_FRAMES_KEY) or {}
    if not isinstance(frames, Mapping):
        return FrameKeys(keys=[], skipped=[], container=type(frames).__name__)
    keys: dict[FrameKey, None] = {}
    skipped: list[str] = []
    for frame_id, entry in frames.items():
        if not isinstance(entry, Mapping):
            skipped.append(f"{frame_id}:{type(entry).__name__}")
            continue
        for ref in (entry.get("ref"), entry.get("store_ref")):
            if isinstance(ref, Mapping):
                spec_id, content_version = ref.get("spec_id"), ref.get("content_version")
                if isinstance(spec_id, str) and isinstance(content_version, str):
                    keys[(spec_id, content_version)] = None
    return FrameKeys(keys=list(keys), skipped=skipped)


def store_search(
    state: SessionState,
    digest: Mapping[str, Any],
    *,
    asked: Sequence[Sequence[str]] = (),
    held_open: Sequence[Sequence[str]] = (),
    share_sides: Sequence[Sequence[str]] = (),
    origin: Origin = "search",
) -> list[dict[str, Any]]:
    """Store what a `resolver_digest` read in the orchestrator's session state,
    and return its parts as the model reads them, without their blocks.

    Each part's spec blocks, its own and its candidates', go under
    `ORCHESTRATOR_SPECS_KEY`, each candidate's marked as one whose scope
    choices Search has not stated (`planner.plan.scope_unread`), and the
    part's warnings under `ORCHESTRATOR_SPEC_WARNINGS_KEY`, where `fetch`
    finds them. The part holds each of its spec_ids with its own record and
    warnings, and an earlier block of the spec_id, of an earlier part or
    already stored, that says more about its scope choices keeps its scope
    content standing (`planner.plan.with_blocks`). The parts' blocks are
    merged first and joined to the store once, which copies it once. Each
    part joins `ORCHESTRATOR_SEARCHES_KEY` as a `SearchReading` from `origin`.
    `asked` holds each part's asked flow parties by position, and
    `held_open` those it asked to hold open, and `share_sides` the sides
    its `share_of` operations run over; each is empty when no part's are
    known. A `definite` part whose prices are open over several years reads
    with `open_prices` (`harness.turn_records.open_prices`), which tells the
    model to ask its unit.

    A spec_id a part holds as its own keeps that part's block and warnings
    against a later part offering it as a candidate, since a pin of it is
    one of the search that settled on it (`harness.planner_adapter`, which
    reads both for the pin). Between blocks no search settled, or two
    searches that each settled it, `planner.plan.with_blocks` decides.
    """
    specs = stored_mapping(state, ORCHESTRATOR_SPECS_KEY)
    spec_warnings = dict(state.get(ORCHESTRATOR_SPEC_WARNINGS_KEY) or {})
    searches = list(stored_sequence(state, ORCHESTRATOR_SEARCHES_KEY))
    own = {stored["spec_id"] for stored in searches if stored.get("spec_id")}
    skill_ids = [skill["id"] for skill in digest.get("skills") or []]
    observed_at = datetime.now(UTC)
    readings = []
    found: dict[str, Mapping[str, Any]] = {}
    for index, part in enumerate(digest.get("part_digests") or []):
        reading = {
            key: value
            for key, value in part.items()
            if key not in ("spec_block", "candidate_blocks")
        }
        blocks = {
            spec_id: scope_unread(block)
            for spec_id, block in (part.get("candidate_blocks") or {}).items()
            if spec_id not in own
        }
        facts: dict[str, Any] = {}
        if part.get("spec_block") and part.get("spec_id"):
            blocks[part["spec_id"]] = part["spec_block"]
            own.add(part["spec_id"])
            facts = reading_facts(part["spec_block"])
            if part["spec_id"] not in unit_settled(state) and (
                prices := open_prices(
                    part["spec_block"],
                    window=part.get("window"),
                    other_prices=part.get("other_prices") or (),
                )
            ):
                reading["open_prices"] = prices
        if mentions := reading.get("resolved_sources"):
            blocks = {
                spec_id: {**block, "_resolved_sources": mentions}
                for spec_id, block in blocks.items()
            }
        found = with_blocks(found, blocks)
        spec_warnings.update(dict.fromkeys(blocks, reading.get("warnings") or []))
        if asked:
            facts["asked_roles"] = list(asked[index])
        if held_open:
            facts["open_roles"] = list(held_open[index])
        if share_sides:
            facts["share_over"] = list(share_sides[index])
        stored = SearchReading.from_part(
            {**reading, **facts}, skills=skill_ids, origin=origin, observed_at=observed_at
        )
        searches.append(stored.model_dump(mode="json"))
        readings.append(reading)
    state[ORCHESTRATOR_SPECS_KEY] = with_blocks(specs, found)
    state[ORCHESTRATOR_SPEC_WARNINGS_KEY] = spec_warnings
    state[ORCHESTRATOR_SEARCHES_KEY] = searches
    return readings


def conversation_readings(state: SessionState) -> list[dict[str, Any]]:
    """The conversation's stored search readings as JSON, oldest first: the
    earlier turns' (`ORCHESTRATOR_PRIOR_LEDGER_KEY`), then this turn's
    (`ORCHESTRATOR_SEARCHES_KEY`)."""
    return [
        *stored_mapping(state, ORCHESTRATOR_PRIOR_LEDGER_KEY).get("readings", []),
        *stored_sequence(state, ORCHESTRATOR_SEARCHES_KEY),
    ]


# What the plan card opens with when the plan the user held was stored in a
# shape this release does not read, and the question was planned again.
UNREADABLE_PLAN_NOTE = (
    "The plan you were shown could not be read after an update, so this is a new plan for "
    "the same question, with any change you typed."
)

# What the card of a plan drafted again opens with when the plan the user answered
# divided by a total an earlier release worked out (`planner.plan.rests_on_stale_whole`).
STALE_WHOLE_NOTE = (
    "The plan you were shown divided by a total worked out before an update, so this plan "
    "asks for that total again, with any change you typed."
)


# What the closing text opens with when the plan the user held was stored in
# a shape this release does not read, and the planner drafted no plan in its
# place. The planner's own refusal follows it.
UNREADABLE_PLAN_UNPLANNED_NOTE = (
    "The plan you were shown could not be read after an update, and the question could not "
    "be planned again."
)

# What the closing text opens with when `approve` dropped the plan the user held
# for resting on a stale whole (`STALE_WHOLE_NOTE`), and the planner drafted no
# plan in its place.
STALE_WHOLE_UNPLANNED_NOTE = (
    "The plan you were shown divided by a total worked out before an update, and the "
    "question could not be planned again."
)


def readable_plan(ctx: NodeCtx[Any], raw: str | None) -> PlannerPlan | None:
    """The plan `raw` holds, `ctx.state["plan"]` as a node read it; `None` when it holds none.

    A plan stored in a shape `planner.plan.Plan` rejects is none
    (`planner.plan.stored_plan`), and the slot is cleared here, by the node
    reading it, so no later node reads it again. `planner` then drafts the
    question from the start: its own entry, or `approve`'s and `execute`'s
    `{"unreadable_plan": True}` output, says so.
    """
    plan = stored_plan(raw, state_key="plan")
    if raw and plan is None:
        ctx.state["plan"] = None
    return plan
