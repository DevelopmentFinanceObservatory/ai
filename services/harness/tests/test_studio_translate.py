"""Unit tests for `harness.studio_translate`'s pure helpers.

The `_interrupt_frames` tests below drive the translator's elicitation
branch directly with plain `args` dicts -- what `_request_input_call` would
have already pulled off a real ADK event -- rather than building one, since
that branch's own logic (which shape's `candidates` to read, which option to
mark `recommended`) needs no real event to exercise.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from typing import Any, cast

import pytest
from compiler.surface import ProgressCallback, ToolCallResult, ToolSchema
from executor.run import current_step_id
from google.adk import Context
from google.adk.events import Event, EventActions
from google.adk.events.event import NodeInfo
from google.adk.flows.llm_flows.functions import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import START, Workflow, node
from google.genai import types
from harness import studio_translate as studio_translate_module
from harness.conversation import current_conversation_id, current_conversation_key
from harness.decision_pick import DecisionPick, decision_pick
from harness.graph.state import APPROVE_INTERRUPT, ORCHESTRATOR_PRIOR_LEDGER_KEY, STEP_PROGRESS_KEY
from harness.orchestrator_tools import _search_context
from harness.plan_display import decision_choice_id, decision_option_id
from harness.planner_seam import PlannerSeamError
from harness.scope_choices import PartScope, carried_entries
from harness.scripted_plan import scripted_plan
from harness.seeds import SEED_STEP_TOOL
from harness.sentences import DECLINE_REASONS, failure_sentence
from harness.studio_state import (
    ConversationState,
    StudioStore,
    _next_manifest,
    land_component,
    remove_cards,
)
from harness.studio_translate import (
    _ExecutorRecord,
    _RecordingExecutor,
    _RecordingSurface,
    _stream_fatal,
    _timeout_refusal,
    _Translator,
    plan_approved,
)
from harness.tool_contracts import HarnessCode, ResolverError
from harness.wire import (
    CompositionActor,
    MessageComplete,
    MessageDelta,
    PlanAction,
    PlanAsk,
    PlanDecision,
    PlanDecisionOption,
    PlanProposed,
    PlanStep,
    StepStatus,
)
from planner.agent import COMBINATION_OPTIONS
from planner.plan import (
    PLAN_STATE_KEY,
    DatasetRequirement,
    Decision,
    PinSource,
    Plan,
    ResolvedBy,
    SessionState,
    UnsupportedOperation,
    plan_revision,
)
from planner.workflow import ToolId
from submission.service import scope_choice_decisions

from .failure_text import classed_sentence, is_user_sentence
from .resolved_revision import resolved_revision
from .test_plan_display import _donor_rank_plan, _fixing_measure, _recorded_payload, _recorded_plan
from .test_planner_seam import ScriptedPlannerLlm, _seam
from .test_turn_records import (
    _EXAMPLE_LINE,
    _example_part,
    _fetched,
    _scoped_part,
    _served_identity,
    _switchable_scope,
)


async def _dummy_executor(*, tool: str, args: dict[str, object]) -> dict[str, object]:
    del tool, args
    return {"outcome": "ok"}


def _translator() -> _Translator:
    return _Translator(
        conversation=ConversationState(conversation_id="c1"),
        recorder=_RecordingExecutor(_dummy_executor, store=StudioStore(InMemorySessionService())),
    )


_OVER_CEILING_MESSAGE = (
    "Over the coverage ceiling: member_cardinality is 480 for "
    "spec_ed5b608a610cdc5a4bc703cf7d1bc2d0, over the 400 ceiling. This spec_id's definition "
    "has more member series than one request can serve. scope.entities and scope.window do "
    "not change that."
)


@pytest.mark.parametrize(
    "refusal",
    [
        {"code": "unsupported_component_type", "message": "no such component"},
        {"code": "internal_error", "message": "Internal error"},
        {
            "code": "over_ceiling",
            "retry_mode": "correct_and_retry",
            "fix": {"ceiling_kind": "member_cardinality"},
            "message": _OVER_CEILING_MESSAGE,
        },
        # A code no table names, with a message far past any cap.
        {"code": "refused", "retry_mode": "do_not_retry", "message": "word " * 200},
        {"code": "facet_not_available", "fix": {"issues": [{"axis": "gender"}]}},
    ],
    ids=["short", "trace-id", "member-cardinality", "long-message", "structured-fix"],
)
def test_a_failed_step_states_the_sentence_of_its_code_and_nothing_of_its_refusal(
    refusal: dict[str, Any],
) -> None:
    """A refusal's message, fix and trace id are written for a model or an
    operator. The failed line states the sentence its code has."""
    current_conversation_id.set("c1")
    translator = _translator()
    translator._recorder.record_step(
        tool="get_data",
        args={},
        result={"outcome": "refusal", "refusal": refusal},
        step_id="debt",
    )

    (frame,) = translator._drain_recorder()

    assert frame.status is StepStatus.FAILED
    assert frame.detail == failure_sentence(
        refusal["code"], refusal.get("retry_mode"), refusal.get("fix")
    )
    assert is_user_sentence(frame.detail)
    assert refusal.get("message", "gender").strip() not in frame.detail
    assert "41d94ed2" not in frame.detail


def test_two_codes_of_different_classes_read_differently_on_a_failed_step() -> None:
    current_conversation_id.set("c1")
    translator = _translator()
    for step_id, code in (("debt", "over_ceiling"), ("pop", "internal_error")):
        translator._recorder.record_step(
            tool="get_data",
            args={},
            result={"outcome": "refusal", "refusal": {"code": code, "message": "m"}},
            step_id=step_id,
        )

    first, second = translator._drain_recorder()

    assert first.detail != second.detail


def test_a_planner_ask_marks_the_recommended_option_and_carries_its_rationale() -> None:
    # `planner.agent`'s `adk_request_input` tool call has no `payload`
    # wrapper -- `candidates`, `recommended_spec_id` and `rationale` are its
    # own call args directly (`_interrupt_frames`'s own comment).
    args = {
        "message": "Which source did you mean?",
        "candidates": [
            {"spec_id": "spec_a", "label": "reading A"},
            {"spec_id": "spec_b", "label": "reading B"},
        ],
        "recommended_spec_id": "spec_b",
        "rationale": "broader coverage of the requested years",
    }

    (frame,) = _translator()._interrupt_frames("call_1", args)

    assert frame.event == "elicitation.request"
    assert frame.skippable is True
    (question,) = frame.questions
    assert (question.id, question.question) == ("q1", "Which source did you mean?")
    assert [(o.id, o.recommended) for o in question.options] == [
        ("spec_a", False),
        ("spec_b", True),
    ]
    assert question.rationale == "broader coverage of the requested years"
    assert (question.allow_free_text, question.skippable) == (True, False)


def test_a_combination_ask_offers_its_options_and_takes_no_free_text() -> None:
    # The planner's ask whether to add two datasets carries the refusal's
    # `options`; the user's pick of one comes back as its id.
    args = {
        "message": "ODA loans count in both aid and debt. Add them?",
        "options": [dict(option) for option in COMBINATION_OPTIONS],
    }

    (frame,) = _translator()._interrupt_frames("call_1", args)

    (question,) = frame.questions
    assert [(o.id, o.label) for o in question.options] == [
        (option["id"], option["label"]) for option in COMBINATION_OPTIONS
    ]
    assert question.allow_free_text is False
    assert frame.skippable is True


def _asked(number: int) -> dict[str, object]:
    """The `ask` node's payload entry for its `number`th question."""
    return {
        "id": f"q{number}",
        "question": f"Question {number}?",
        "candidates": [
            {"spec_id": f"spec_{number}", "label": f"reading {number}"},
            {"spec_id": "o2", "label": "something else"},
        ],
    }


@pytest.mark.parametrize(
    ("count", "pause_skippable", "question_skippable"),
    [
        pytest.param(1, True, False, id="one-question-dismissed-as-a-whole"),
        pytest.param(2, False, True, id="several-questions-each-left-to-the-agent"),
    ],
)
def test_an_agent_ask_puts_each_question_with_its_own_options(
    count: int, pause_skippable: bool, question_skippable: bool
) -> None:
    # The `ask` node raises a `RequestInput` whose `payload` carries the
    # agent's questions, each with its options as `candidates`, and nothing
    # about a recommendation.
    asked = [_asked(number) for number in range(1, count + 1)]
    translator = _translator()

    (frame,) = translator._interrupt_frames(
        "clarify", {"message": "Question 1?", "payload": {"questions": asked}}
    )

    assert frame.skippable is pause_skippable
    assert [
        (q.id, q.question, [(o.id, o.label, o.recommended) for o in q.options or ()])
        for q in frame.questions
    ] == [
        (
            entry["id"],
            entry["question"],
            [(c["spec_id"], c["label"], False) for c in entry["candidates"]],
        )
        for entry in asked
    ]
    assert {(q.allow_free_text, q.skippable, q.rationale) for q in frame.questions} == {
        (True, question_skippable, None)
    }
    pending = translator._conversation.pending[frame.interrupt_id]
    assert pending.skippable is pause_skippable
    assert [(q.id, q.options, q.skippable) for q in pending.questions] == [
        (q.id, {o.id: o.label for o in q.options or ()}, q.skippable) for q in frame.questions
    ]


def _plan_payload(**extra: object) -> dict[str, object]:
    """The planner node's `"ok"`-routed payload shape (`harness.graph`'s
    `planner` node), built from `harness.scripted_plan.scripted_plan`."""
    plan = scripted_plan("kenya life expectancy", spec_id=None)
    return {
        **plan.model_dump(mode="json"),
        "summary": "Kenya life expectancy, 1960-2023.",
        "revision": plan_revision(plan),
        **extra,
    }


def test_a_plan_proposed_ask_without_asks_leaves_them_unset() -> None:
    args = {"payload": _plan_payload()}

    (frame,) = _translator()._interrupt_frames(APPROVE_INTERRUPT, args)

    assert frame.asks is None


def test_a_scope_decision_on_the_card_carries_its_choice_and_every_option() -> None:
    """US health aid to Africa: each scope choice with another option to
    switch to is a card row naming the choice, its active option and every
    option, with why one out of reach is; a planner decision carries none."""
    plan = scripted_plan("kenya life expectancy", spec_id=None)
    dataset = next(r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement))
    pin = "spec_" + "e" * 32
    scope = _switchable_scope()
    grant = scope["choices"][2]["options"][2]
    grant["availability"] = {
        "status": "partial",
        "years": [[2011, 2024]],
        "code": "imputed_coverage",
        "detail": {},
    }
    block = {"spec_id": pin, "resolved": {"scope_choices": scope}}
    scoped = scope_choice_decisions({pin: PinSource(block=block, settled=True)})[pin]
    cited = dataset.model_copy(
        update={
            "decision_ids": (*dataset.decision_ids, *(d.slot for d in scoped)),
            "source_spec_id": pin,
        }
    )
    plan = Plan.model_validate(
        {
            **plan.model_dump(),
            "decisions": [*plan.decisions, *scoped],
            "evidence_requirements": [
                cited if r.id == dataset.id else r for r in plan.evidence_requirements
            ],
        }
    )
    payload = {**plan.model_dump(mode="json"), "revision": plan_revision(plan)}
    translator = _translator()

    (frame,) = translator._interrupt_frames(APPROVE_INTERRUPT, {"payload": payload})

    rows = {row.id: row for row in frame.decisions}
    planners = [rows[d.slot] for d in plan.decisions if d.scope is None and d.slot in rows]
    assert planners
    assert all(row.choice_id is row.options is row.active_option_id is None for row in planners)
    sector, region, basis = (rows[d.slot] for d in scoped)
    # The row's id is the requirement pinning the source and the axis, which
    # a switch keeps.
    assert (sector.choice_id, sector.active_option_id, sector.chosen, sector.resolved_by) == (
        f"{dataset.id}:sector",
        "sector:120",
        "health (CRS 120)",
        "user",
    )
    assert sector.reason == "The user's question asked for health (CRS 120)."
    assert (region.active_option_id, region.resolved_by) == ("region:members_only", "user")
    assert basis.options == [
        PlanDecisionOption(
            option_id="measure_basis:gross",
            label="gross disbursements, current USD",
            status="available",
        ),
        PlanDecisionOption(
            option_id="measure_basis:net",
            label="net disbursements",
            status="unavailable",
            reason="it conflicts with another choice",
        ),
        PlanDecisionOption(
            option_id="measure_basis:grant_equivalent",
            label="grant equivalents",
            status="partial",
            reason="imputed multilateral aid covers only some years",
            years=[(2011, 2024)],
        ),
    ]
    # A reason saying only that Search's default holds is left off the card.
    assert (basis.reason, basis.resolved_by) == ("", "default")
    assert frame.actions == ["approve", "amend", "reject", "switch"]
    assert frame.moved is None
    # What a switch on this card may name: every shown choice, with its
    # active option and each option's status.
    choices = translator._conversation.pending[frame.interrupt_id].choices
    assert set(choices) == {sector.choice_id, region.choice_id, basis.choice_id}
    assert choices[basis.choice_id].active == "measure_basis:gross"
    assert {k: o.status for k, o in choices[basis.choice_id].options.items()} == {
        "measure_basis:gross": "available",
        "measure_basis:net": "unavailable",
        "measure_basis:grant_equivalent": "partial",
    }


def test_a_planner_decision_settling_fields_is_a_row_the_card_can_switch() -> None:
    """Kenya life expectancy with its measure decision fixing the measure: the
    row names the decision by slot, offers each option as available with the
    chosen one active, and a switch may name it. A rationale-only decision
    stays a row with no choice."""
    plan = _fixing_measure(scripted_plan("kenya life expectancy", spec_id=None))
    (measure,) = plan.decisions
    rationale = Decision(
        slot="rounding",
        description="How figures are rounded.",
        options=("one decimal", "whole years"),
        chosen="one decimal",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="Life expectancy is reported to one decimal.",
    )
    dataset = next(r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement))
    cited = dataset.model_copy(update={"decision_ids": (*dataset.decision_ids, "rounding")})
    plan = Plan.model_validate(
        {
            **plan.model_dump(),
            "decisions": [measure.model_copy(update={"chosen": measure.options[1]}), rationale],
            "evidence_requirements": [
                cited if r.id == dataset.id else r for r in plan.evidence_requirements
            ],
        }
    )
    payload = {**plan.model_dump(mode="json"), "revision": plan_revision(plan)}
    translator = _translator()

    (frame,) = translator._interrupt_frames(APPROVE_INTERRUPT, {"payload": payload})

    rows = {row.id: row for row in frame.decisions}
    switched, plain = rows["measure"], rows["rounding"]
    birth, healthy = (decision_option_id(label) for label in measure.options)
    assert (switched.choice_id, switched.active_option_id, switched.chosen) == (
        "decision/measure",
        healthy,
        "healthy life expectancy",
    )
    assert switched.options == [
        PlanDecisionOption(option_id=birth, label="life expectancy at birth", status="available"),
        PlanDecisionOption(option_id=healthy, label="healthy life expectancy", status="available"),
    ]
    assert plain.choice_id is plain.options is plain.active_option_id is None
    assert PlanAction.SWITCH in frame.actions
    choices = translator._conversation.pending[frame.interrupt_id].choices
    assert set(choices) == {"decision/measure"}
    assert choices["decision/measure"].active == healthy
    assert {k: o.label for k, o in choices["decision/measure"].options.items()} == {
        birth: "life expectancy at birth",
        healthy: "healthy life expectancy",
    }


def test_a_planner_decision_option_keeps_its_id_when_a_replan_reorders_the_options() -> None:
    """The card's cross-revision marks compare option ids: a replan listing
    the measure decision's options in the other order gives each label the
    id it had."""
    plan = _fixing_measure(scripted_plan("kenya life expectancy", spec_id=None))
    (measure,) = plan.decisions
    reordered = Plan.model_validate(
        {
            **plan.model_dump(),
            "decisions": [measure.model_copy(update={"options": measure.options[::-1]})],
        }
    )

    def labels_by_id(shown: Plan) -> dict[str, str]:
        payload = {**shown.model_dump(mode="json"), "revision": plan_revision(shown)}
        (frame,) = _translator()._interrupt_frames(APPROVE_INTERRUPT, {"payload": payload})
        (row,) = frame.decisions
        return {option.option_id: option.label for option in row.options or ()}

    assert labels_by_id(reordered) == labels_by_id(plan)
    assert len(labels_by_id(plan)) == len(measure.options)


def test_a_switch_to_an_option_the_card_words_picks_the_option_as_the_planner_wrote_it(
    decision: Callable[..., Decision],
) -> None:
    """A rank decision the planner wrote as `among`'s values ("population",
    "named") shows the card's words for each, and a switch to the option
    the card words "only donors picked by name" picks "named", the option as the
    plan states it."""
    plan = _donor_rank_plan(decision)
    payload = {**plan.model_dump(mode="json"), "revision": plan_revision(plan)}

    (frame,) = _translator()._interrupt_frames(APPROVE_INTERRUPT, {"payload": payload})

    (row,) = [row for row in frame.decisions if row.choice_id is not None]
    labels = {option.label: option.option_id for option in row.options or ()}
    assert (row.chosen, list(labels)) == (
        "all 32 donors in DAC2a",
        ["all 32 donors in DAC2a", "only donors picked by name"],
    )
    assert decision_pick(
        plan,
        choice_id=row.choice_id,
        option_id=labels["only donors picked by name"],
        to_origin={},
        alignments={},
    ) == DecisionPick.on_card(plan, slot="measure", chosen="named")


def test_a_plan_a_switch_offers_says_what_else_moved_and_a_plain_plan_offers_no_switch() -> None:
    plan = scripted_plan("kenya life expectancy", spec_id=None)
    moved = {
        "choice_id": "spec_" + "e" * 32 + ":region",
        "from_label": "Africa incl. regional buckets",
        "to_label": "Africa's countries only",
        "reason": "Regional buckets apply only to a summed region.",
    }
    payload = {**plan.model_dump(mode="json"), "revision": plan_revision(plan), "moved": [moved]}
    translator = _translator()

    (frame,) = translator._interrupt_frames(APPROVE_INTERRUPT, {"payload": payload})

    assert frame.moved is not None
    assert [entry.model_dump() for entry in frame.moved] == [moved]
    assert frame.actions == ["approve", "amend", "reject"]
    assert translator._conversation.pending[frame.interrupt_id].choices == {}


def test_a_plan_offered_again_after_a_failed_switch_names_the_option_and_why() -> None:
    """The `switch` node's payload for a switch that did not apply: the card
    carries `switch_failed` so the site shows the note on the row the user
    switched, and no chat message repeats it."""
    plan = scripted_plan("kenya life expectancy", spec_id=None)
    note = (
        "Official development assistance would then count different providers' spending "
        "from health aid."
    )
    failed = {
        "choice_id": "spec_" + "e" * 32 + ":provider",
        "option_id": "provider:inclusive",
        "code": "operation_unsupported",
        "reason": note,
        "retryable": False,
    }
    payload = {
        **plan.model_dump(mode="json"),
        "revision": plan_revision(plan),
        "switch_failed": failed,
    }

    (frame,) = _translator()._interrupt_frames(APPROVE_INTERRUPT, {"payload": payload})

    assert frame.switch_failed is not None
    assert frame.switch_failed.model_dump() == failed
    assert frame.version == plan.version


def test_a_plan_card_states_the_decisions_that_qualify_its_answer_as_caveats() -> None:
    """Each decision the plan marks as a caveat (`Decision.caveat`) gives its
    reason on the card, before Approve, in plan order and once each; any
    other decision gives none."""
    plan = scripted_plan("kenya life expectancy", spec_id=None)
    differ = "The question asks for the inputs as their datasets count them."
    whole = "Denominator: the total over every sector from the CRS, the same dataset."

    def noted(slot: str, rationale: str, caveat: str | None) -> Decision:
        return Decision.model_validate(
            {
                "slot": slot,
                "description": "d",
                "options": ["a", "b"],
                "chosen": "a",
                "resolved_by": "planner_default",
                "rationale": rationale,
                "caveat": caveat,
            }
        )

    added = [
        noted("inputs_differ", differ, "providers_differ"),
        noted("other", "Not a caveat.", None),
        noted("whole", whole, "derived_whole"),
        # A second calculation stating the same sentence adds nothing.
        noted("inputs_differ_again", differ, "providers_differ"),
    ]
    dataset = next(r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement))
    cited = dataset.model_copy(
        update={"decision_ids": (*dataset.decision_ids, *(d.slot for d in added))}
    )
    plan = Plan.model_validate(
        {
            **plan.model_dump(),
            "decisions": [*plan.decisions, *added],
            "evidence_requirements": [
                cited if r.id == dataset.id else r for r in plan.evidence_requirements
            ],
        }
    )
    payload = {**plan.model_dump(mode="json"), "revision": plan_revision(plan)}
    translator = _translator()

    (frame,) = translator._interrupt_frames(APPROVE_INTERRUPT, {"payload": payload})

    assert frame.caveats == [differ, whole]


def _us_health_share(**updates: Any) -> Plan:
    """The recorded US health share of US ODA in 2023 (`_recorded_plan`), each
    field of `updates` set on it."""
    plan = _recorded_plan("long")
    return Plan.model_validate({**plan.model_dump(), **updates})


def _card_of(plan: Plan) -> PlanProposed:
    (frame,) = _translator()._interrupt_frames(
        APPROVE_INTERRUPT, {"payload": {**plan.model_dump(mode="json"), "revision": None}}
    )
    return frame


def test_a_caveat_reaches_the_card_only_as_a_caveat() -> None:
    """The US health share whose inputs count different providers: the decision
    saying so is listed under the card's caveats and is no row of its choices,
    and every decision a requirement on the card cites is one of its rows."""
    plan = _us_health_share()
    differ = Decision(
        slot="providers_differ_health_share",
        description="Whether the inputs count the same providers' spending.",
        options=("each input as its dataset counts it", "every input on one provider scope"),
        chosen="each input as its dataset counts it",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="The inputs count different providers' spending.",
        caveat="providers_differ",
    )
    plan = _us_health_share(
        decisions=[*plan.decisions, differ],
        evidence_requirements=[
            r.model_copy(update={"decision_ids": (*r.decision_ids, differ.slot)})
            if r.id == "health_share"
            else r
            for r in plan.evidence_requirements
        ],
    )

    card = _card_of(plan)

    assert card.caveats == [differ.rationale]
    rows = {row.id for row in card.decisions}
    assert differ.slot not in rows
    cited = {slot for requirement in card.requirements or () for slot in requirement.decision_ids}
    assert cited and cited <= rows


def test_a_planner_decisions_options_one_label_would_read_alike_reach_the_card_apart() -> None:
    """The US health share with a planner decision whose two alternatives open
    with one sector option id: each, and the chosen one, reads that option's
    label with the planner's words after it, since the label alone would read
    alike; the alternative no other reads alike with reads as written."""
    plan = _us_health_share()
    sector = "health incl. population and reproductive health (CRS 120 + 130)"
    which = Decision(
        slot="health_sector_side",
        description="Which side the health sectors count on.",
        options=(
            "sector:120+130 health for the numerator",
            "sector:120+130 health for the total",
            "every sector",
        ),
        chosen="sector:120+130 health for the total",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="The user asked for the health share.",
    )
    plan = _us_health_share(
        decisions=[*plan.decisions, which],
        evidence_requirements=[
            r.model_copy(update={"decision_ids": (*r.decision_ids, which.slot)})
            if r.id == "health_share"
            else r
            for r in plan.evidence_requirements
        ],
    )

    row = next(row for row in _card_of(plan).decisions if row.id == which.slot)

    assert (row.alternatives, row.chosen) == (
        [f"{sector} health for the numerator", f"{sector} health for the total", "every sector"],
        f"{sector} health for the total",
    )


def test_two_choices_on_one_axis_each_name_the_input_they_govern() -> None:
    """The US health share with the total's basis choice offering other options
    than the health figures' one, as after a switch of the health figures to a
    source served on gross alone: the card keeps a Basis row for each, and each
    row names its input in the card's words. A choice alone on its axis keeps
    its short name."""
    plan = _us_health_share()
    total_basis = "scope_measure_basis_spec_a85874f71d3d96e35c368b9454e051ca"

    def narrowed(decision: Decision) -> Decision:
        assert decision.scope is not None
        states = tuple(
            state
            if state.option_id == "measure_basis:gross"
            else state.model_copy(update={"status": "unavailable", "code": "basis_not_published"})
            for state in decision.scope.options
        )
        return decision.model_copy(
            update={"scope": decision.scope.model_copy(update={"options": states})}
        )

    plan = _us_health_share(
        decisions=[narrowed(d) if d.slot == total_basis else d for d in plan.decisions]
    )

    card = _card_of(plan)

    assert sorted(row.question for row in card.decisions if row.choice_id is not None) == [
        "Basis: health",
        "Basis: official development assistance",
        "Sector",
    ]


def test_a_reason_opening_with_an_option_id_reads_its_own_decisions_label() -> None:
    """The US health share whose two basis choices offer one option id under
    different labels, the total's reason opening with that id. The total's row
    reads the label its own decision gives that id."""
    plan = _us_health_share()
    total_basis = "scope_measure_basis_spec_a85874f71d3d96e35c368b9454e051ca"
    reason = " because the user asked for net flows"

    def relabelled(decision: Decision) -> Decision:
        return decision.model_copy(
            update={
                "options": tuple(f"total's {label}" for label in decision.options),
                "chosen": f"total's {decision.chosen}",
                "rationale": f"measure_basis:net{reason}",
            }
        )

    plan = _us_health_share(
        decisions=[relabelled(d) if d.slot == total_basis else d for d in plan.decisions]
    )

    reasons = {row.id: row.reason for row in _card_of(plan).decisions}

    assert reasons[total_basis] == f"total's net disbursements, current USD{reason}"


def test_an_approved_plan_states_the_card_the_user_saw() -> None:
    """Title, summary, step labels, caveats and, for each switchable choice,
    scope choice or planner decision, its `choice_id`, short name and the
    option it holds; a choice holding another label than before the turn's
    first switch of it says what it held, one switched back says nothing, and
    a decision without a choice id is no choice."""

    def scope(choice_id: str, question: str, held: str) -> PlanDecision:
        return PlanDecision(
            id=f"d_{question}",
            question=question,
            chosen=held,
            reason="because",
            resolved_by="user",
            choice_id=choice_id,
            active_option_id=f"{choice_id}:held",
            options=[
                PlanDecisionOption(option_id=f"{choice_id}:held", label=held, status="available"),
                PlanDecisionOption(option_id=f"{choice_id}:other", label="x", status="available"),
            ],
        )

    def planner(slot: str, question: str, held: str, other: str) -> PlanDecision:
        # The row the card gives a switchable planner decision, as `_choice_members` builds it.
        decision = Decision(
            slot=slot,
            description=question,
            options=(held, other),
            chosen=held,
            resolved_by="planner_default",
            rationale="because",
            fixes={"window_start": 2015},
        )
        return PlanDecision(
            id=f"d_{slot}",
            question=question,
            chosen=held,
            reason="because",
            resolved_by="default",
            choice_id=decision_choice_id(decision),
            active_option_id=decision_option_id(held),
            options=[
                PlanDecisionOption(
                    option_id=decision_option_id(label), label=label, status="available"
                )
                for label in (held, other)
            ],
        )

    caveat = "The share's two sides count different providers' spending, as asked."
    card = PlanProposed(
        event="plan.proposed",
        seq=3,
        interrupt_id="int_1",
        plan_version_id="plan_1",
        version=2,
        title="Research plan",
        summary="Gross aid to Kenya, 2020 to 2024.",
        steps=[
            PlanStep(id="s1", label="Fetch aid", description="detail"),
            PlanStep(id="s2", label="Sum it", gives_answer=True),
        ],
        actions=[PlanAction.APPROVE],
        decisions=[
            scope("c_basis", "Basis", "gross disbursements"),
            scope("c_region", "Region", "Africa's countries only"),
            planner("window", "Window", "since 2020", "since 2015"),
            PlanDecision(
                id="d_plan", question="Chart", chosen="line", reason="r", resolved_by="default"
            ),
        ],
        revision="rev_1",
        caveats=[caveat],
    )

    frame = plan_approved(
        card,
        seq=9,
        baseline={
            "c_basis": "gross disbursements",
            "c_region": "Africa incl. regional buckets",
            "decision/window": "since 2015",
        },
    )

    assert frame.model_dump(mode="json", exclude_none=True) == {
        "event": "plan.approved",
        "seq": 9,
        "plan_version_id": "plan_1",
        "version": 2,
        "revision": "rev_1",
        "title": "Research plan",
        "summary": "Gross aid to Kenya, 2020 to 2024.",
        "steps": [{"label": "Fetch aid"}, {"label": "Sum it", "gives_answer": True}],
        "choices": [
            {"choice_id": "c_basis", "question": "Basis", "chosen": "gross disbursements"},
            {
                "choice_id": "c_region",
                "question": "Region",
                "chosen": "Africa's countries only",
                "changed_from": "Africa incl. regional buckets",
            },
            {
                "choice_id": "decision/window",
                "question": "Window",
                "chosen": "since 2020",
                "changed_from": "since 2015",
            },
        ],
        "caveats": [caveat],
    }


def test_a_plan_proposed_card_shows_each_ask_with_its_steps_or_why_it_is_declined() -> None:
    plan = scripted_plan("kenya life expectancy", spec_id=None)
    final = plan.evidence_requirements[-1]
    assert plan.translation is not None
    (step_id,) = [op.id for op in plan.translation.operations if op.requirement_id == final.id]
    asks = [
        "life expectancy in Kenya each period",
        "a ranking of East African countries",
        "life expectancy in Kenya and its rank",
    ]
    payload = _plan_payload(
        asks=asks,
        answered=[{"ask": 0, "outputs": [final.id]}, {"ask": 2, "outputs": [final.id]}],
        declined=[
            {"ask": asks[1], "code": "unsupported_ask", "ask_ref": 1},
            {"ask": "its rank", "code": "unsupported_ask", "ask_ref": 2},
        ],
    )

    (frame,) = _translator()._interrupt_frames(APPROVE_INTERRUPT, {"payload": payload})

    assert frame.asks is not None
    answered, declined, in_part = frame.asks
    reason = DECLINE_REASONS[UnsupportedOperation.UNSUPPORTED_ASK]
    assert answered == PlanAsk(ask=asks[0], answered_by=[step_id])
    assert step_id in {step.id for step in frame.steps}
    assert declined == PlanAsk(ask=asks[1], declined=reason)
    assert in_part == PlanAsk(
        ask=asks[2], answered_by=[step_id], declined=reason, not_computed="its rank"
    )


def test_a_regressed_version_is_logged_and_does_not_move_latest_backwards(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Versions only rise within a turn once the amend/replan guard lands;
    a version-1 approve arriving after the conversation has already seen
    version 3 is the case that guard is meant to prevent, exercised here
    directly against `_interrupt_frames`."""
    conversation = ConversationState(conversation_id="c1")
    translator = _Translator(
        conversation=conversation,
        recorder=_RecordingExecutor(_dummy_executor, store=StudioStore(InMemorySessionService())),
    )
    (first,) = translator._interrupt_frames(
        APPROVE_INTERRUPT, {"payload": _plan_payload(version=1)}
    )
    translator._interrupt_frames(APPROVE_INTERRUPT, {"payload": _plan_payload(version=2)})
    translator._interrupt_frames(APPROVE_INTERRUPT, {"payload": _plan_payload(version=3)})

    with caplog.at_level(logging.ERROR, logger="harness.studio_translate"):
        (regressed,) = translator._interrupt_frames(
            APPROVE_INTERRUPT, {"payload": _plan_payload(version=1)}
        )

    assert any("regress" in record.message.lower() for record in caplog.records)
    assert conversation.latest_plan_version == 3
    assert regressed.event == "plan.proposed"
    assert regressed.plan_version_id != first.plan_version_id


@pytest.mark.parametrize("unreadable_version", [1, 2])
def test_a_restarted_plan_opens_a_new_line_of_plan_versions(
    unreadable_version: int, caplog: pytest.LogCaptureFixture
) -> None:
    """The planner drafts from the start over a plan it could not read, and
    flags the draft: its version 1 is no regression and no re-offer of the
    version 1 already shown. It supersedes the last card shown under an id of
    its own, and an amendment of it supersedes it in turn."""
    conversation = ConversationState(conversation_id="c1")
    translator = _Translator(
        conversation=conversation,
        recorder=_RecordingExecutor(_dummy_executor, store=StudioStore(InMemorySessionService())),
    )
    shown = [
        translator._interrupt_frames(
            APPROVE_INTERRUPT, {"payload": _plan_payload(version=version)}
        )[-1]
        for version in range(1, unreadable_version + 1)
    ]
    shown_ids = {card.plan_version_id for card in shown}

    with caplog.at_level(logging.ERROR, logger="harness.studio_translate"):
        versioned, redraft = translator._interrupt_frames(
            APPROVE_INTERRUPT, {"payload": _plan_payload(version=1, restarted=True)}
        )
        amended_versioned, amended = translator._interrupt_frames(
            APPROVE_INTERRUPT, {"payload": _plan_payload(version=2)}
        )

    assert caplog.records == []
    assert (redraft.event, redraft.version) == ("plan.proposed", 1)
    assert redraft.plan_version_id not in shown_ids
    assert (versioned.event, versioned.version) == ("plan.versioned", 1)
    assert versioned.plan_version_id == redraft.plan_version_id
    assert versioned.supersedes == shown[-1].plan_version_id
    assert (amended.event, amended.version) == ("plan.proposed", 2)
    assert amended.plan_version_id not in shown_ids | {redraft.plan_version_id}
    assert amended_versioned.event == "plan.versioned"
    assert amended_versioned.plan_version_id == amended.plan_version_id
    assert amended_versioned.supersedes == redraft.plan_version_id
    assert conversation.plan_version_ids == {
        1: redraft.plan_version_id,
        2: amended.plan_version_id,
    }


class _UnreachableSurface:
    """A `ToolSurface` `_RecordingSurface` never actually calls here: this
    test appends `_ExecutorRecord`s to it directly, the way a wave's own
    concurrent steps (`executor.run._Run.execute`) would have appended them
    out of the plan's own order."""

    async def list_tools(self) -> list[ToolSchema]:
        raise AssertionError("not called")

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        raise AssertionError("not called")


def test_out_of_order_surface_records_drain_in_the_plan_s_own_step_order() -> None:
    """A wave's independent steps finish, and get recorded, in whatever
    order the event loop schedules them -- here step B is recorded before
    step A even though the plan lists A first. `_drain_recorder` must still
    number them by the plan's own order: A as `s1`, B as `s2`."""
    current_conversation_id.set("c1")
    plan = scripted_plan("kenya life expectancy", spec_id=None)
    assert plan.workflow is not None
    step_a, step_b = plan.workflow.steps
    payload = {
        **plan.model_dump(mode="json"),
        "summary": "Kenya life expectancy, 1960-2023.",
        "revision": plan_revision(plan),
    }
    surface_recorder = _RecordingSurface(_UnreachableSurface())
    translator = _Translator(
        conversation=ConversationState(conversation_id="c1"),
        recorder=_RecordingExecutor(_dummy_executor, store=StudioStore(InMemorySessionService())),
        surface_recorder=surface_recorder,
    )

    translator._interrupt_frames(APPROVE_INTERRUPT, {"payload": payload})

    surface_recorder._records.append(
        _ExecutorRecord(
            tool="tool_b",
            args={},
            result={"outcome": "ok"},
            duration_ms=1,
            conversation="c1",
            step_id=step_b.id,
        )
    )
    surface_recorder._records.append(
        _ExecutorRecord(
            tool="tool_a",
            args={},
            result={"outcome": "ok"},
            duration_ms=1,
            conversation="c1",
            step_id=step_a.id,
        )
    )

    frames = translator._drain_recorder()

    activity_steps = [frame for frame in frames if frame.event == "activity.step"]
    assert [(step.step_id, step.tool) for step in activity_steps] == [
        ("s1", "tool_a"),
        ("s2", "tool_b"),
    ]


def _progress_event(step_id: str, message: str) -> Event:
    """What `harness.graph.execute._step_progress_event` yields for one relayed report."""
    return Event(
        custom_metadata={
            STEP_PROGRESS_KEY: {"step_id": step_id, "tool": "get_data", "message": message}
        }
    )


def _surface_record(step_id: str, result: dict[str, Any]) -> _ExecutorRecord:
    return _ExecutorRecord(
        tool="get_data", args={}, result=result, duration_ms=1, conversation="c1", step_id=step_id
    )


def _progress_translator() -> tuple[_Translator, _RecordingSurface]:
    current_conversation_id.set("c1")
    surface_recorder = _RecordingSurface(_UnreachableSurface())
    translator = _Translator(
        conversation=ConversationState(conversation_id="c1"),
        recorder=_RecordingExecutor(_dummy_executor, store=StudioStore(InMemorySessionService())),
        surface_recorder=surface_recorder,
    )
    return translator, surface_recorder


def test_a_step_keeps_one_wire_id_from_its_first_report_to_its_final_frame() -> None:
    """Studio keeps one Activity line per `step_id`, so a step's running
    reports, a failed attempt the executor retried, and its final frame all
    land on the one line its first report opened; a concurrent step gets a
    line of its own."""
    translator, surface_recorder = _progress_translator()

    (first,) = translator.translate(_progress_event("debt", "Preparing the dataset (queued, 3s)"))
    (other,) = translator.translate(_progress_event("pop", "Preparing the dataset (queued, 1s)"))
    (later,) = translator.translate(_progress_event("debt", "Retrying in 7s (attempt 2 of 3)"))
    surface_recorder._records.append(
        _surface_record(
            "debt",
            {"outcome": "refusal", "refusal": {"code": "service_unavailable", "message": "busy"}},
        )
    )
    surface_recorder._records.append(_surface_record("debt", {"outcome": "ok", "summary": "done"}))
    surface_recorder._records.append(_surface_record("pop", {"outcome": "ok", "summary": "done"}))
    final = translator._drain_recorder()

    assert [(f.status, f.detail) for f in (first, other, later)] == [
        (StepStatus.RUNNING, "Preparing the dataset (queued, 3s)"),
        (StepStatus.RUNNING, "Preparing the dataset (queued, 1s)"),
        (StepStatus.RUNNING, "Retrying in 7s (attempt 2 of 3)"),
    ]
    assert first.step_id == later.step_id != other.step_id
    assert first.label == final[0].label == "Fetching data"
    assert [(f.step_id, f.status) for f in final] == [
        (first.step_id, StepStatus.FAILED),
        (first.step_id, StepStatus.COMPLETE),
        (other.step_id, StepStatus.COMPLETE),
    ]
    # The Activity log replaces a line's detail with each frame, so the frame
    # that fails a line a report was showing on states a reason of its own.
    assert is_user_sentence(final[0].detail)


def test_a_mid_run_report_leaves_a_finished_steps_record_for_the_run_to_amend() -> None:
    """A step's record is written when its call returns, before the
    executor's post-call checks; `execute` amends it once the run is over
    (`_RecordingSurface.mark_step_failed`). A report from a sibling still
    running must not drain that record early, or the amendment finds
    nothing and the step reads complete forever."""
    translator, surface_recorder = _progress_translator()
    surface_recorder._records.append(_surface_record("debt", {"outcome": "ok", "summary": "done"}))

    (running,) = translator.translate(_progress_event("pop", "Preparing the dataset (queued, 1s)"))
    surface_recorder.mark_step_failed(
        step_id="debt", tool="get_data", code="source_drift", detail="drifted"
    )
    (final,) = translator._drain_recorder()

    assert running.status is StepStatus.RUNNING
    assert final.step_id != running.step_id
    assert final.status is StepStatus.FAILED
    assert final.detail == classed_sentence("source_drift", "do_not_retry")
    assert is_user_sentence(final.detail)


class _HangingSurface(_UnreachableSurface):
    """A `ToolSurface` whose call never returns, until cancelled."""

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def _cancel_mid_call(surface_recorder: _RecordingSurface, *, step_id: str) -> None:
    """Start `step_id`'s call through `surface_recorder`, then cancel it
    before it returns, as a step budget, a Stop or a disconnect does."""

    async def call() -> None:
        current_step_id.set(step_id)
        await surface_recorder.call_tool(
            tool=ToolId(server="data-mcp", name="get_data"), arguments={}
        )

    task = asyncio.create_task(call())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_step_cut_off_mid_call_closes_its_line_as_failed() -> None:
    """A step's budget cancels its call before it returns, so the call
    writes no record. `mark_step_failed` records the failure itself, and the
    step's final frame closes the line its running report opened."""
    translator, surface_recorder = _progress_translator()
    surface_recorder._surface = _HangingSurface()

    (running,) = translator.translate(_progress_event("debt", "Preparing the dataset (queued, 3s)"))
    await _cancel_mid_call(surface_recorder, step_id="debt")
    surface_recorder.mark_step_failed(
        step_id="debt", tool="get_data", code="transport_failed", detail="ran past its budget"
    )
    (final,) = translator._drain_recorder()

    assert (final.step_id, final.status, final.tool) == (
        running.step_id,
        StepStatus.FAILED,
        "get_data",
    )
    assert final.detail == classed_sentence("transport_failed", "do_not_retry")
    assert is_user_sentence(final.detail)
    assert final.duration_ms is not None


async def test_a_call_abandoned_mid_flight_leaves_nothing_once_its_segment_drains() -> None:
    """A Stop or a disconnect cancels the call and ends the segment, whose
    `finally` drains the recorder. Nothing of the abandoned call is left for
    a later segment to report."""
    surface_recorder = _RecordingSurface(_HangingSurface())
    current_conversation_id.set("c1")

    await _cancel_mid_call(surface_recorder, step_id="debt")

    assert surface_recorder.drain("c1") == []
    assert surface_recorder._in_flight == {}


def test_a_step_that_never_called_gets_a_failed_record_and_a_refused_one_keeps_its_own() -> None:
    """A step stopped before its first call still gets a line that closes
    as failed. A step whose last call already refused, with none in flight,
    is left alone: its own refusal is its final frame."""
    translator, surface_recorder = _progress_translator()
    refused = {"outcome": "refusal", "refusal": {"code": "service_unavailable", "message": "busy"}}
    surface_recorder._records.append(_surface_record("pop", refused))

    for step_id in ("debt", "pop"):
        surface_recorder.mark_step_failed(
            step_id=step_id, tool="get_data", code="transport_failed", detail="out of time"
        )
    frames = translator._drain_recorder()

    assert [(f.status, f.detail, f.duration_ms) for f in frames] == [
        (StepStatus.FAILED, classed_sentence("service_unavailable"), 1),
        (StepStatus.FAILED, classed_sentence("transport_failed", "do_not_retry"), 0),
    ]
    assert all(is_user_sentence(f.detail) for f in frames)


@pytest.mark.parametrize(
    ("step_id", "detail"),
    [
        (
            "seeds",
            "The datasets from Search could not be used, so the answer starts from your question",
        ),
        ("seed-1", "This Search result was left out"),
    ],
)
def test_a_failed_seed_step_says_in_plain_words_what_happened_to_the_datasets(
    step_id: str, detail: str
) -> None:
    """A seed is taken before the agent runs, so its step is where the user
    learns a dataset they brought was not used; its refusal stays off the wire."""
    current_conversation_id.set("c1")
    translator = _translator()
    refusal = {"code": "seed_duplicate", "message": "seed 1 names the dataset seed 0 names"}
    translator._recorder.record_step(
        tool=SEED_STEP_TOOL,
        args={"seed": 1},
        result={"outcome": "refusal", "refusal": refusal},
        step_id=step_id,
    )

    (frame,) = translator._drain_recorder()

    assert (frame.status, frame.detail) == (StepStatus.FAILED, detail)


# --- the recorder's components ------------------------------------------------


def _written_component(
    component_id: str = "cmp_test0001",
    *,
    revision: int = 1,
    title: str = "Life expectancy in Nigeria",
    labels: tuple[str, ...] = ("Nigeria",),
) -> dict[str, Any]:
    """A platform-shaped `write_component` result: the wire fields of
    `component.created` minus event/seq, one `FactSeries` per label."""
    series = [{"id": label.lower(), "label": label, "point_count": 64} for label in labels]
    return {
        "component_id": component_id,
        "revision": revision,
        "type": "line_chart",
        "catalog_version": "2026.8.0",
        "min_catalog_version": "2026.8.0",
        "payload": resolved_revision(
            "line_chart", component_id=component_id, revision=revision, title=title
        ),
        "fact_summary": {
            "component_id": component_id,
            "revision": revision,
            "series": series,
            "attribution": {
                "label": "WDI · 1 indicator(s)",
                "source_count": 1,
                "indicator_count": 1,
            },
            "window": {"start": "1960", "end": "2023"},
        },
        "provenance": {
            "inputs": [{"spec_id": "spec_" + "a" * 32, "content_version": "cv_" + "1" * 32}],
            "sources": [{"id": "src_test", "name": "Test source", "licence": "CC BY 4.0"}],
            "freshness": {
                "data_through": "2023",
                "last_refreshed": "2026-06-30",
                "partial_period": False,
            },
            "warnings": [],
            "generated_at": "2026-01-01T00:00:00Z",
            "citation": "Test source (2026).",
        },
    }


# --- `_component_frames`: the one place validation, landing and framing meet --


def test_component_frames_lands_and_appends() -> None:
    """`land=True`: an append, which also streams both frames."""
    current_conversation_id.set("c1")
    translator = _translator()
    translator._conversation.composition = _next_manifest(None, [], actor=CompositionActor.MODEL)

    frames = translator._component_frames(_written_component("cmp_two"), step_id="s1", land=True)

    events = [getattr(f, "event", None) for f in frames]
    assert events == ["component.created", "composition.updated"]
    assert translator._conversation.turn_components[-1]["action"] == "created"


async def test_a_step_that_removes_a_card_and_draws_one_streams_each_write_s_own_manifest() -> None:
    """`remove_cards` writes the composition as its tool runs and a drawn
    card lands when its record drains, so one model step holding both calls
    writes twice. Each `composition.updated` carries the manifest its own
    write produced, and revisions rise along the stream."""
    current_conversation_id.set("c1")
    drawn = _written_component("cmp_new")

    async def executor(*, tool: str, args: dict[str, object], on_progress=None) -> dict[str, Any]:
        del tool, args, on_progress
        return {"outcome": "ok", "component": drawn}

    conversation = ConversationState(conversation_id="c1")
    for held in ("cmp_01", "cmp_02"):
        land_component(conversation, _written_component(held))
    start = conversation.composition.revision
    recorder = _RecordingExecutor(executor, store=StudioStore(InMemorySessionService()))
    translator = _Translator(conversation=conversation, recorder=recorder)
    calls = {"remove_cards": {"removed": ["cmp_01"]}, "make_charts": {"charts": [{"title": "t"}]}}

    remove_cards(conversation, ["cmp_01"], all_cards=False)
    await recorder(tool="write_component", args={})
    recorder.land_component(drawn)
    frames = [
        *translator.translate(
            Event(
                author="dfo_orchestrator",
                content=types.Content(
                    parts=[
                        types.Part(function_call=types.FunctionCall(id=name, name=name, args={}))
                        for name in calls
                    ]
                ),
            )
        ),
        *translator.translate(
            Event(
                author="dfo_orchestrator",
                content=types.Content(
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=name, name=name, response=answer
                            )
                        )
                        for name, answer in calls.items()
                    ]
                ),
            )
        ),
    ]

    updates = [f.manifest for f in frames if getattr(f, "event", None) == "composition.updated"]
    assert [manifest.revision for manifest in updates] == [start + 1, start + 2]
    assert [[block.component_id for block in manifest.blocks] for manifest in updates] == [
        ["cmp_02"],
        ["cmp_02", "cmp_new"],
    ]
    assert conversation.composition == updates[-1]
    # The removal's manifest leads the drained card, and the card's own
    # manifest follows it. The `remove_cards` step closes after both, with
    # the other responses of its event.
    order = [
        f.event if f.event != "activity.step" else f"{f.tool}:{f.status.value}"
        for f in frames
        if f.event != "activity.step"
        or (f.tool == "remove_cards" and f.status is StepStatus.COMPLETE)
    ]
    assert order == [
        "composition.updated",
        "component.created",
        "composition.updated",
        "remove_cards:complete",
    ]


def test_component_frames_land_true_holds_when_auto_add_is_off() -> None:
    """`land=True` against a conversation with `auto_add` off resolves to
    `harness.studio_state.land_component`'s `"held"`: no block for
    `composition.updated` to report."""
    current_conversation_id.set("c1")
    translator = _translator()
    translator._conversation.auto_add = False

    frames = translator._component_frames(_written_component("cmp_two"), step_id="s1", land=True)

    events = [getattr(f, "event", None) for f in frames]
    assert events == ["component.created"]


def test_component_frames_land_false_emits_only_component_created() -> None:
    """`land=False` marks a write `make_charts` did not stamp to land. It
    skips `harness.studio_state.land_component` entirely. The component
    streams, no block lands for it, and the canvas is untouched."""
    current_conversation_id.set("c1")
    translator = _translator()
    before = translator._conversation.composition

    frames = translator._component_frames(_written_component("cmp_two"), step_id="s1", land=False)

    events = [getattr(f, "event", None) for f in frames]
    assert events == ["component.created"]
    assert translator._conversation.composition is before
    assert translator._conversation.turn_components == []


def test_component_frames_logs_error_for_a_stamped_component_that_fails_validation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`land=True` means `make_charts` already ran
    `harness.wire.usable_component` on this exact component and it passed,
    so a validation failure here means the wire check and this method's
    own contract have diverged, which is a bug and earns an ERROR record.
    Nothing streams and the canvas stays untouched. The method does not
    raise out of `translate` and fail the whole segment."""
    current_conversation_id.set("c1")
    translator = _translator()
    before = translator._conversation.composition
    invalid = {**_written_component("cmp_two")}
    del invalid["fact_summary"]

    with caplog.at_level(logging.ERROR, logger="harness.studio_translate"):
        frames = translator._component_frames(invalid, step_id="s1", land=True)

    assert frames == []
    assert translator._conversation.composition is before
    assert translator._conversation.turn_components == []
    assert any("validation" in record.message.lower() for record in caplog.records)


def test_component_frames_logs_nothing_for_an_unstamped_component_that_fails_validation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`land=False` means this write was never stamped usable.
    `make_charts` already reported it as not drawn because this same
    component failed `harness.wire.usable_component`. Failing validation
    here again is the expected outcome of that refusal, so no frames
    stream and nothing is logged at ERROR."""
    current_conversation_id.set("c1")
    translator = _translator()
    before = translator._conversation.composition
    invalid = {**_written_component("cmp_two")}
    del invalid["fact_summary"]

    with caplog.at_level(logging.DEBUG, logger="harness.studio_translate"):
        frames = translator._component_frames(invalid, step_id="s1", land=False)

    assert frames == []
    assert translator._conversation.composition is before
    assert translator._conversation.turn_components == []
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


# --- `_RecordingExecutor.land_component`: stamps the record, decides nothing -


def test_land_component_stamps_land_on_the_matching_record() -> None:
    """`land_component` only stamps the record whose result carries this
    exact component object (found by identity) -- it never resolves a
    conversation and never mutates a canvas; `_component_frames`, draining
    later, is what does the real work from what it stamps."""
    recorder = _RecordingExecutor(_dummy_executor, store=StudioStore(InMemorySessionService()))
    component = _written_component("cmp_two", labels=("Nigeria", "Guatemala"))
    recorder._records.append(
        _ExecutorRecord(
            tool="write_component",
            args={"intent": {}},
            result={"outcome": "ok", "component": component},
            duration_ms=1,
            conversation="c1",
        )
    )

    recorder.land_component(component)

    assert recorder._records[0].land is True


def test_land_component_is_a_silent_no_op_with_no_matching_record() -> None:
    """A component with no matching record (this capability called
    directly, not through `__call__` first) leaves nothing to stamp and
    raises nothing."""
    recorder = _RecordingExecutor(_dummy_executor, store=StudioStore(InMemorySessionService()))

    recorder.land_component(_written_component("cmp_two"))


# --- `_RecordingExecutor.__call__`: write tokens --------------------------


def _returning(result: dict[str, object]) -> Any:
    async def _executor(
        *, tool: str, args: dict[str, object], on_progress: ProgressCallback | None = None
    ) -> dict[str, object]:
        del tool, args, on_progress
        return dict(result)

    return _executor


async def test_a_create_that_returns_no_write_token_is_logged_at_error_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """That card is rebuilt as a new card on its next edit, so the missing
    token is a defect an operator must see, once per create."""
    store = StudioStore(InMemorySessionService())
    current_conversation_key.set(("alice", "c1"))
    current_conversation_id.set("alice/c1")
    await store.create("alice", "c1")
    component = _written_component("cmp_tokenless")
    recorder = _RecordingExecutor(
        _returning({"outcome": "ok", "component": component}), store=store
    )

    with caplog.at_level(logging.ERROR, logger="harness.studio_translate"):
        await recorder(tool="write_component", args={"intent": {"title": "x"}})

    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "cmp_tokenless" in errors[0].getMessage()
    assert "alice/c1" in errors[0].getMessage()


async def test_a_revise_returning_no_write_token_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only a create is issued a token; a revise's reply never carries one."""
    store = StudioStore(InMemorySessionService())
    current_conversation_key.set(("alice", "c1"))
    await store.create("alice", "c1")
    recorder = _RecordingExecutor(
        _returning({"outcome": "ok", "component": _written_component("cmp_one", revision=2)}),
        store=store,
    )

    with caplog.at_level(logging.ERROR, logger="harness.studio_translate"):
        await recorder(
            tool="write_component",
            args={"intent": {"title": "x"}, "component_id": "cmp_one", "write_token": "cmw_x"},
        )

    assert not [record for record in caplog.records if record.levelno == logging.ERROR]


async def test_the_record_of_a_revise_leaves_the_write_token_out() -> None:
    store = StudioStore(InMemorySessionService())
    current_conversation_key.set(("alice", "c1"))
    current_conversation_id.set("alice/c1")
    await store.create("alice", "c1")
    recorder = _RecordingExecutor(
        _returning({"outcome": "ok", "component": _written_component("cmp_one", revision=2)}),
        store=store,
    )
    args = {"intent": {"title": "x"}, "component_id": "cmp_one", "write_token": "cmw_secret"}

    await recorder(tool="write_component", args=args)

    [record] = recorder.drain("alice/c1")
    assert record.args == {"intent": {"title": "x"}, "component_id": "cmp_one"}
    assert args["write_token"] == "cmw_secret"


# --- `_drain_recorder`: performs the landing `land_component` only stamped -


# --- a refused write never reaches `_component_frames` -----------------------


def test_a_refused_write_never_reaches_component_frames() -> None:
    """A refusal carries no `component`, so `_drain_recorder` never calls
    `_component_frames` for it -- no `component.created` goes out, and the
    canvas is untouched."""
    current_conversation_id.set("c1")
    translator = _translator()
    before = translator._conversation.composition

    translator._recorder._records.append(
        _ExecutorRecord(
            tool="write_component",
            args={"intent": {}},
            result={
                "outcome": "refusal",
                "refusal": {"code": "constraint_violation", "message": "no commensurable unit"},
            },
            duration_ms=1,
            conversation="c1",
        )
    )
    frames = translator._drain_recorder()

    assert translator._conversation.composition is before
    assert not any(getattr(f, "event", None) == "component.created" for f in frames)
    activity = [f for f in frames if f.event == "activity.step"]
    assert activity and activity[0].status == StepStatus.FAILED


# --- the recorded action survives into the conversation digest ---------------


def _node_event(
    node: str, *, route: str | None = None, output: Any = None, **metadata: Any
) -> Event:
    return Event(
        author="dfo_analyst_graph",
        node_info=NodeInfo(path=f"dfo_analyst_graph/{node}@1"),
        actions=EventActions(route=route),
        output=output,
        custom_metadata=metadata or None,
    )


def _planner_call(name: str, **args: Any) -> Event:
    return Event(
        author="dfo_planner",
        content=types.Content(
            role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))]
        ),
    )


def _status_lines(frames: list[Any]) -> list[tuple[str, str, StepStatus]]:
    return [(f.step_id, f.label, f.status) for f in frames]


def test_each_planner_call_becomes_a_status_line_as_it_happens() -> None:
    """The planner's function-call events arrive as the model finishes a
    call, so each one opens its line then, and completes the line before it.
    `propose_plan` translates the plan before it answers, so its call opens
    the checking line and its response completes that line: the duration
    covers the translation. The completing frame's `detail` names the data of
    the plan the proposal recorded."""
    translator = _translator()
    scripted = scripted_plan("kenya life expectancy", spec_id=None)
    dataset = next(r for r in scripted.evidence_requirements if isinstance(r, DatasetRequirement))
    second = dataset.model_copy(update={"id": "second", "measure": "health spending"})
    plan = Plan(
        question="q",
        decisions=scripted.decisions,
        evidence_requirements=(dataset, second),
    )

    working = translator.translate(_node_event("planner", planner_started={"revising": False}))
    checking = translator.translate(_planner_call("propose_plan", question="q"))
    checked = translator.translate(
        _planner_answer(
            "propose_plan",
            {"outcome": "ok", "translation": {"outcome": "ok"}},
            **{PLAN_STATE_KEY: plan.model_dump_json()},
        )
    )
    done = translator.translate(_node_event("planner", route="ok", output={"version": 1}))

    assert _status_lines(working) == [("s0", "Working on a plan", StepStatus.RUNNING)]
    assert _status_lines(checking) == [
        ("s0", "Working on a plan", StepStatus.COMPLETE),
        ("s2", "Checking the plan against the data", StepStatus.RUNNING),
    ]
    assert _status_lines(checked) == [
        ("s2", "Checking the plan against the data", StepStatus.COMPLETE)
    ]
    # A compiled proposal declares the plan ready itself, so no line follows the check.
    assert _status_lines(done) == []
    assert [f.detail for f in checked] == ["life expectancy at birth for Kenya and 1 more"]
    # Studio shows a line as done only once a frame carries `duration_ms`.
    completed = [f for f in checking + checked if f.status is StepStatus.COMPLETE]
    assert all(f.duration_ms is not None for f in completed)
    assert all(f.duration_ms is None for f in working + checking if f.status is StepStatus.RUNNING)


def test_the_checking_line_is_open_before_the_proposal_is_answered() -> None:
    """Translation runs inside the `propose_plan` call, so the line has to be
    on the wire when the call event arrives, not when the response does; a
    line opened on the response would cover none of the translation."""
    translator = _translator()

    opened = translator.translate(_planner_call("propose_plan", question="q"))

    assert _status_lines(opened) == [
        ("s0", "Checking the plan against the data", StepStatus.RUNNING)
    ]
    assert translator.translate(_planner_call("propose_plan", amends_version=1)) == []


@pytest.mark.parametrize("recorded", ["none", "no_datasets"])
def test_an_accepted_proposal_without_a_translation_still_closes_its_line(recorded: str) -> None:
    """The standalone planner (no translator) answers an accepted proposal
    with no `translation`; the response opens nothing and ends the call's
    line. A response that carries no plan, or a plan with no dataset, leaves
    the closing frame without a `detail`."""
    state = (
        {PLAN_STATE_KEY: Plan(question="q").model_dump_json()} if recorded == "no_datasets" else {}
    )
    translator = _translator()
    translator.translate(_planner_call("propose_plan", question="q"))

    frames = translator.translate(_planner_answer("propose_plan", {"outcome": "ok"}, **state))

    assert _status_lines(frames) == [
        ("s0", "Checking the plan against the data", StepStatus.COMPLETE)
    ]
    assert frames[0].detail is None


def test_a_planner_call_other_than_a_proposing_one_adds_no_line() -> None:
    translator = _translator()

    assert translator.translate(_planner_call("reject_plan")) == []
    assert translator.translate(_planner_call("get_skill", skill_id="oda-measures")) == []


def test_a_replan_shows_revising_the_plan() -> None:
    """The planner node starts a revising draft on an amendment or a replan."""
    frames = _translator().translate(_node_event("planner", planner_started={"revising": True}))

    assert _status_lines(frames) == [("s0", "Revising the plan", StepStatus.RUNNING)]


@pytest.mark.parametrize("revising", [True, False], ids=["replan-or-amendment", "draft"])
def test_a_planner_draft_drops_the_labels_the_turns_switches_kept(revising: bool) -> None:
    """Whatever routes the turn to the planner, an amendment or the execute
    node's replan included, it drafts the plan again, possibly under the same
    requirement ids the choice ids are built from, so no earlier card's labels
    describe the new plan's choices."""
    translator = _translator()
    translator._conversation.turn_choice_baseline = {"aid:sector": "health (CRS 120)"}

    translator.translate(_node_event("planner", planner_started={"revising": revising}))

    assert translator._conversation.turn_choice_baseline == {}


_PICKED_ROW = "decision/donor_scope"


@pytest.mark.parametrize(
    ("ending", "kept"),
    [
        pytest.param(None, {"aid:sector", _PICKED_ROW}, id="raised"),
        pytest.param("pending_refusal", {"aid:sector", _PICKED_ROW}, id="stopped-or-refused"),
        pytest.param("ok", {_PICKED_ROW}, id="landed"),
    ],
)
def test_a_pick_replan_drops_the_scope_labels_the_turns_switches_kept_only_once_it_lands(
    ending: str | None, kept: set[str]
) -> None:
    """A replan for a pick on the card that raises, is stopped or is refused
    leaves the plan unchanged, so an earlier switch's label still names what
    the approved plan changed. A plan the replan drafts drops the scope
    choices' labels, whose ids it may reuse, and keeps the planner decision
    rows', named by slot."""
    translator = _translator()
    baseline = {"aid:sector": "health (CRS 120)", _PICKED_ROW: "the DAC total"}
    translator._conversation.turn_choice_baseline = dict(baseline)

    translator.translate(_node_event("planner", planner_started={"revising": True, "pick": True}))
    if ending is not None:
        translator.translate(_node_event("planner", route=ending, output={"version": 2}))

    assert translator._conversation.turn_choice_baseline == {
        choice_id: label for choice_id, label in baseline.items() if choice_id in kept
    }


def test_an_approved_plan_a_pick_replan_drafted_names_the_label_the_pick_was_made_from() -> None:
    """The user picks the top 10 donors on a card showing the DAC total, the
    replan lands, and the user approves it: the switched row says what it
    held before."""
    translator = _translator()
    translator._conversation.turn_choice_baseline = {_PICKED_ROW: "the DAC total"}
    translator.translate(_node_event("planner", planner_started={"revising": True, "pick": True}))
    translator.translate(_node_event("planner", route="ok", output={"version": 2}))
    labels = ("the DAC total", "the top 10 donors")
    card = PlanProposed(
        event="plan.proposed",
        seq=3,
        interrupt_id="int_1",
        plan_version_id="plan_1",
        version=2,
        title="Research plan",
        summary="The top 10 donors to Kenya.",
        steps=[PlanStep(id="s1", label="Rank donors", gives_answer=True)],
        actions=[PlanAction.APPROVE],
        decisions=[
            PlanDecision(
                id="d_donors",
                question="Donors",
                chosen=labels[1],
                reason="The user picked it.",
                resolved_by="user",
                choice_id=_PICKED_ROW,
                active_option_id=decision_option_id(labels[1]),
                options=[
                    PlanDecisionOption(
                        option_id=decision_option_id(label), label=label, status="available"
                    )
                    for label in labels
                ],
            )
        ],
        revision="rev_2",
    )

    frame = plan_approved(card, seq=9, baseline=translator._conversation.turn_choice_baseline)

    assert [choice.changed_from for choice in frame.choices] == ["the DAC total"]


def test_the_planner_starting_a_first_draft_opens_working_on_a_plan() -> None:
    """Whatever routes the turn to the planner, the line opens as the
    planner node starts its draft."""
    frames = _translator().translate(_node_event("planner", planner_started={"revising": False}))

    assert _status_lines(frames) == [("s0", "Working on a plan", StepStatus.RUNNING)]


def test_the_seams_late_progress_reports_are_not_shown() -> None:
    """The node relays the seam's reports only after the draft is done, so
    they would repeat the live lines out of order."""
    translator = _translator()
    report = {"agent": "dfo_planner", "tools": ["propose_plan"], "duration_ms": 5}

    assert translator.translate(_node_event("planner", progress=report)) == []


def test_a_pause_completes_the_running_line_before_its_card() -> None:
    translator = _translator()
    translator.translate(_planner_call("propose_plan", question="q"))
    ask = Event(
        author="dfo_planner",
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="ask-1",
                        name=REQUEST_INPUT_FUNCTION_CALL_NAME,
                        args={"message": "Which source?", "candidates": []},
                    )
                )
            ],
        ),
    )

    frames = translator.translate(ask)

    assert [f.event for f in frames] == ["activity.step", "elicitation.request"]
    assert frames[0].status is StepStatus.COMPLETE
    assert translator.finish() == []


def _planner_ask(message: str, *, partial: bool | None = None) -> Event:
    """The planner's `adk_request_input` call asking `message`, as a streamed
    chunk when `partial`."""
    call = types.FunctionCall(
        id="ask-1", name=REQUEST_INPUT_FUNCTION_CALL_NAME, args={"message": message}
    )
    return Event(
        author="dfo_planner",
        partial=partial,
        content=types.Content(role="model", parts=[types.Part(function_call=call)]),
    )


def test_a_streamed_ask_pauses_only_on_its_final_event() -> None:
    """A streamed model call reaches the translator as partial chunks and then
    one aggregated non-partial event, the only one the runner saves. Only that
    event may announce the call `/interrupt` will answer."""
    translator = _translator()

    chunk_frames = translator.translate(_planner_ask("Which donors?", partial=True))
    final_frames = translator.translate(_planner_ask("Which donors?", partial=False))

    assert [f.event for f in chunk_frames if f.event == "elicitation.request"] == []
    assert [f.event for f in final_frames if f.event == "elicitation.request"] == [
        "elicitation.request"
    ]


def test_a_question_after_a_draft_that_named_no_drop_is_asked_as_is() -> None:
    translator = _translator()
    translator.translate(_node_event("planner", planner_started={"revising": False}))

    (*_, asked) = translator.translate(_planner_ask("Which donors?"))

    assert asked.questions[0].question == "Which donors?"


def test_the_segment_end_completes_a_line_still_running() -> None:
    translator = _translator()
    translator.translate(_planner_call("propose_plan", question="q"))

    frames = translator.finish()

    assert (frames[0].label, frames[0].status) == (
        "Checking the plan against the data",
        StepStatus.COMPLETE,
    )


async def test_a_planner_call_becomes_its_line_while_the_draft_is_still_running() -> None:
    """A real `PlannerSeam`, driven through `ctx.run_node` the way the
    graph's planner node drives it, with its model held after its first
    call. The runner hands the translator the call and its response while
    the draft is still waiting, so their lines reach the wire before the
    draft ends."""
    seam = _seam()
    paused = asyncio.Event()
    release = asyncio.Event()
    seam._agent.model = ScriptedPlannerLlm(
        script=[[("propose_plan", {"question": "kenya debt"})]],
        pause_at=1,
        paused=paused,
        release=release,
    )

    async def planner(ctx: Context, node_input: Any) -> Any:
        del node_input
        return await seam(ctx=ctx, question="kenya debt", readings=[])

    workflow = Workflow(
        name="status_test_wf", edges=[(START, node(planner, name="planner", rerun_on_resume=True))]
    )
    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name="status_test", user_id="u")
    runner = Runner(node=workflow, app_name="status_test", session_service=sessions)
    translator = _translator()
    frames: list[Any] = []

    async def run() -> None:
        async for event in runner.run_async(
            user_id="u",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="probe")]),
        ):
            frames.extend(translator.translate(event))

    task = asyncio.create_task(run())
    await asyncio.wait_for(paused.wait(), timeout=10)
    mid_draft = _status_lines(frames)
    release.set()
    await task

    # The question-only proposal is refused, so its response swaps the line.
    assert mid_draft == [
        ("s0", "Checking the plan against the data", StepStatus.RUNNING),
        ("s0", "Checking the plan against the data", StepStatus.COMPLETE),
        ("s2", "Reworking the draft", StepStatus.RUNNING),
    ]


def _timed_out(tool: str) -> Exception:
    """The exception `mcp_client._call_tool` raises for a call past its timeout."""
    exc = TimeoutError(f"{tool} timed out")
    exc.timeout_tool = tool  # type: ignore[attr-defined]
    exc.timeout_key = f"{tool}:{{}}"  # type: ignore[attr-defined]
    return exc


@pytest.mark.parametrize(
    ("raised", "error", "retry", "code"),
    [
        (
            [
                PlannerSeamError(
                    "This plan still needs translating and compiling into a workflow before "
                    "it can execute. Propose it again to have it translated."
                )
            ],
            "capability_unavailable",
            "do_not_retry",
            "internal_error",
        ),
        (
            [
                ResolverError(
                    "search refused",
                    refusal={
                        "code": "rate_limited",
                        "retry_mode": "retry_after_delay",
                        "fix": {"detail": "resolve_parts over quota for spec_ab12"},
                    },
                )
            ],
            "rate_limited",
            "retry_after_delay",
            "rate_limited",
        ),
        # A code the wire grammar has no member for keeps its own sentence.
        (
            [ResolverError("gone", refusal={"code": "surface_changed"})],
            "transient_unavailable",
            "retry_after_delay",
            "surface_changed",
        ),
        ([RuntimeError("boom")], "transient_unavailable", "retry_after_delay", HarnessCode.RAISED),
        (
            [_timed_out("get_data")],
            "transient_unavailable",
            "retry_after_delay",
            "transient_unavailable",
        ),
        # The same call timing out again is advised not to retry.
        (
            [_timed_out("get_data"), _timed_out("get_data")],
            "transient_unavailable",
            "do_not_retry",
            "transient_unavailable",
        ),
        # A card write that timed out may have been stored.
        (
            [_timed_out("write_component")],
            "transient_unavailable",
            "do_not_retry",
            HarnessCode.WRITE_UNCONFIRMED,
        ),
    ],
    ids=[
        "planner-seam",
        "refusal",
        "code-off-the-wire",
        "bare-exception",
        "timeout",
        "timeout-repeated",
        "write-timeout",
    ],
)
def test_a_stream_fatal_error_reads_as_the_sentence_of_its_code(
    raised: list[Exception], error: str, retry: str, code: str | None
) -> None:
    """The `error` frame's detail is the chat reply. It states the sentence of
    the failure's code under the frame's retry mode, and nothing written for
    the model, no tool name and no trace id."""
    conversation = ConversationState(conversation_id="c1")
    conversation.begin_turn()

    *_, frame = [_stream_fatal(conversation, exc) for exc in raised]

    assert (frame.error.error.value, frame.error.retry.value) == (error, retry)
    assert frame.error.detail == classed_sentence(code, retry)
    assert is_user_sentence(frame.error.detail)
    assert conversation.trace_id not in frame.error.detail
    assert "Propose it again" not in frame.error.detail


def test_only_a_card_write_that_timed_out_is_unconfirmed() -> None:
    """A write that was sent may have been stored, so its timeout carries a
    code of its own. Any other tool's timeout, a repeated one included, is
    transient."""
    conversation = ConversationState(conversation_id="c1")
    conversation.begin_turn()

    write = _timeout_refusal(conversation, tool="write_component", key="w")
    fetches = [_timeout_refusal(conversation, tool="get_data", key="g") for _ in range(2)]

    assert (write["code"], write["retry_mode"]) == (
        HarnessCode.WRITE_UNCONFIRMED,
        "do_not_retry",
    )
    assert [(fetch["code"], fetch["retry_mode"]) for fetch in fetches] == [
        ("transient_unavailable", "retry_after_delay"),
        ("transient_unavailable", "do_not_retry"),
    ]
    frames = [
        _stream_fatal(conversation, _timed_out(tool)) for tool in ("write_component", "search")
    ]
    assert frames[0].error.detail != frames[1].error.detail


def _planner_answer(name: str, response: dict[str, Any], **state: Any) -> Event:
    """A planner tool's response event, with the state delta the tool wrote."""
    return Event(
        author="dfo_planner",
        content=types.Content(
            role="user",
            parts=[
                types.Part(function_response=types.FunctionResponse(name=name, response=response))
            ],
        ),
        actions=EventActions(state_delta=state),
    )


_REFUSED_DRAFT = {
    "outcome": "invalid_input",
    "code": "invalid_plan",
    "message": "Invalid plan data: evidence_requirements.2.filter.predicates.0: Input tag",
}


def test_a_refused_proposal_reads_reworking_the_draft_until_the_next_call() -> None:
    """A refused `propose_plan` ran no translation, so its response swaps the
    checking line for "Reworking the draft", which runs while the planner
    redrafts. The refusal's code shows in plain words and its message, which
    names plan fields, never reaches the wire."""
    translator = _translator()
    translator.translate(_planner_call("propose_plan", question="q"))

    refused = translator.translate(_planner_answer("propose_plan", _REFUSED_DRAFT))
    retry = translator.translate(_planner_call("propose_plan", question="q"))
    accepted = translator.translate(
        _planner_answer("propose_plan", {"outcome": "ok", "translation": {"outcome": "ok"}})
    )

    assert _status_lines(refused) == [
        ("s0", "Checking the plan against the data", StepStatus.COMPLETE),
        ("s2", "Reworking the draft", StepStatus.RUNNING),
    ]
    assert refused[1].detail == "a step didn't fit the plan's rules"
    assert _status_lines(retry) == [
        ("s2", "Reworking the draft", StepStatus.COMPLETE),
        ("s4", "Checking the plan against the data", StepStatus.RUNNING),
    ]
    assert _status_lines(accepted) == [
        ("s4", "Checking the plan against the data", StepStatus.COMPLETE)
    ]
    assert not any("evidence_requirements" in (f.detail or "") for f in refused + retry)


def test_a_refusal_with_an_unlisted_code_reworks_the_draft_without_a_reason() -> None:
    translator = _translator()

    frames = translator.translate(
        _planner_answer("propose_plan", {"outcome": "refused", "code": "x"})
    )

    assert _status_lines(frames) == [("s0", "Reworking the draft", StepStatus.RUNNING)]
    assert frames[0].detail is None


def test_an_approved_plan_shows_starting_the_plan_until_its_first_step() -> None:
    """The executor takes seconds to start after an approval; the line
    covers that, and completes before the first step's frame takes its seq."""
    translator = _translator()

    approved = translator.translate(_node_event("approve", route="approved", output={"v": 1}))
    first_step = translator.translate(_progress_event("fetch_1", "Reading the data"))

    assert _status_lines(approved) == [("s0", "Starting the plan", StepStatus.RUNNING)]
    assert [(f.step_id, f.label, f.status) for f in first_step] == [
        ("s0", "Starting the plan", StepStatus.COMPLETE),
        ("s2", "Fetching data", StepStatus.RUNNING),
    ]
    assert first_step[0].seq < first_step[1].seq


def test_starting_the_plan_completes_before_a_step_that_reported_no_progress() -> None:
    translator, surface_recorder = _progress_translator()
    translator.translate(_node_event("approve", route="approved", output={"v": 1}))
    surface_recorder._records.append(_surface_record("fetch_1", {"outcome": "ok"}))

    frames = translator.translate(_node_event("execute"))

    assert [(f.label, f.status) for f in frames][:2] == [
        ("Starting the plan", StepStatus.COMPLETE),
        ("Fetching data", StepStatus.COMPLETE),
    ]


def test_an_amendment_does_not_start_the_plan() -> None:
    frames = _translator().translate(_node_event("approve", route="amend", output={"plan": {}}))

    assert frames == []


def test_a_new_planner_run_after_a_refused_one_starts_its_own_line() -> None:
    """A planner run that ended on a refusal leaves "Reworking the draft"
    to the next run of the same segment, whose start opens its own line."""
    translator = _translator()
    translator.translate(_planner_answer("propose_plan", _REFUSED_DRAFT))

    frames = translator.translate(_node_event("planner", planner_started={"revising": False}))

    assert [(f.label, f.status) for f in frames] == [
        ("Reworking the draft", StepStatus.COMPLETE),
        ("Working on a plan", StepStatus.RUNNING),
    ]


def _answered(readings: list[dict[str, Any]]) -> tuple[_Translator, list[Any]]:
    """A translator that closed a turn answering with `readings` and the frames it sent."""
    current_conversation_id.set("c1")
    translator = _translator()
    translator._conversation.turn_user_message = "total health aid from the US to Africa"
    translator._conversation.turn_readings = readings
    translator._final_text = "The US gave $7.9bn."
    return translator, translator.finish()


@pytest.mark.parametrize("noted", [False, True], ids=["line-alone", "notes-and-line"])
def test_the_answer_closes_with_the_code_text_and_the_next_turns_history_leaves_it_out(
    monkeypatch: pytest.MonkeyPatch, noted: bool
) -> None:
    """The reply and the stored turn keep the loose-match notes and the basis
    line the code appends; the digest a later turn's model reads holds the
    answer alone, so the model has none of that text to restate before the
    code appends its own."""
    note = "The series used for health aid is a close match only."
    monkeypatch.setattr(
        studio_translate_module, "loose_match_notes", lambda *_, **__: [note] if noted else []
    )
    translator, frames = _answered([_fetched(_example_part())])
    conversation = translator._conversation

    deltas = [frame.text for frame in frames if isinstance(frame, MessageDelta)]
    complete = next(frame for frame in frames if isinstance(frame, MessageComplete))
    digest = conversation.conversation_digest_state()

    notes = f"\n\n{note}" if noted else ""
    assert complete.text == f"The US gave $7.9bn.{notes}\n\n{_EXAMPLE_LINE}"
    assert "The US gave $7.9bn." + "".join(deltas) == complete.text
    assert conversation.turn_final_text == complete.text
    stored = conversation.turns[-1]
    assert (stored["assistant"], stored.get("match_notes"), stored["basis_line"]) == (
        "The US gave $7.9bn.",
        note if noted else None,
        _EXAMPLE_LINE,
    )
    assert digest is not None
    assert digest["turns"][-1]["assistant"] == "The US gave $7.9bn."
    assert "Counted as" not in json.dumps(digest)
    assert note not in json.dumps(digest)


def test_an_answer_using_no_part_with_scope_choices_states_no_basis_line() -> None:
    unscoped = _fetched({**_example_part(), "scope": None})

    _, frames = _answered([unscoped])

    assert not any(isinstance(frame, MessageDelta) for frame in frames)
    assert next(f for f in frames if isinstance(f, MessageComplete)).text == "The US gave $7.9bn."


def test_a_later_search_carries_the_stored_scope_choices_and_not_the_basis_line() -> None:
    """A fetch that left out the regional buckets makes the line say so, but
    the region choice stored is the declared default, which carries nothing: the next search carries the entries of the stored
    choices, whatever the stored answer says."""
    part = _scoped_part()
    region = next(c for c in part["scope"]["choices"] if c["axis"] == "region")
    region |= {"active": "region:members_and_residuals", "provenance": "default"}
    reading = _fetched(part, _served_identity(residuals=False))
    translator, _ = _answered([reading])
    conversation = translator._conversation
    scope = PartScope.model_validate(reading["scope"])

    def carried() -> list[dict[str, Any]]:
        state = {ORCHESTRATOR_PRIOR_LEDGER_KEY: conversation.prior_ledger_state()}
        context = _search_context(cast("SessionState", state))
        return [
            entry.model_dump(mode="json", exclude_none=True) for entry in context.carried_entries
        ]

    stated = conversation.turns[-1]["basis_line"]
    with_line = carried()
    del conversation.turns[-1]["basis_line"]

    assert "Africa's countries only" in stated
    assert (
        with_line
        == carried()
        == [entry.model_dump(mode="json", exclude_none=True) for entry in carried_entries([scope])]
    )
    assert [entry["option_id"] for entry in with_line] == ["sector:120"]


def test_a_recorded_plan_after_a_switch_reaches_the_card_in_the_users_words() -> None:
    """A health share of US aid, switched to net: the card is titled by the
    ask and summed up in one sentence, its step labels name no basis, one
    step gives the answer, each scope choice goes by its short name, a
    default gives no reason, and the planner's slug reads as words."""
    payload = _recorded_payload("long_switched_to_net")

    frame = _translator()._interrupt_frames(APPROVE_INTERRUPT, {"payload": payload})[-1]

    assert isinstance(frame, PlanProposed)
    assert frame.title == "Health share of US ODA in 2023"
    assert frame.summary is not None
    assert frame.summary.startswith("Net disbursements, current USD, in 2023")
    assert [(step.label, step.gives_answer) for step in frame.steps] == [
        ("Fetch health for United States", False),
        ("Total health", False),
        ("Fetch official development assistance for United States", False),
        ("Total official development assistance", False),
        ("Health as a share of official development assistance", True),
    ]
    assert [(row.question, row.chosen, row.reason) for row in frame.decisions or ()] == [
        (
            "Denominator source for US total ODA to compute the sector share",
            "crs total",
            "Using the CRS total ensures consistent cash-flow accounting between numerator "
            "and denominator, avoiding grant-equivalent vs cash disbursement mismatches and "
            "CRS reporting gaps against DAC1.",
        ),
        ("Sector", "health incl. population and reproductive health (CRS 120 + 130)", ""),
        (
            "Basis",
            "net disbursements, current USD",
            "The user chose net disbursements, current USD on the plan.",
        ),
    ]
