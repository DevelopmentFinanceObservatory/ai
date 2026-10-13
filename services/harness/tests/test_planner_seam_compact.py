"""`PlannerSeam` over a planner that proposes through `submit_analysis`.

The seam reads a proposal's answer by its shape, a terminal `code` or a
`translation`, so a compact planner (`SchemaDelivery.COMPACT`) ends its turn
and has its referrals tracked as the eager one does, and the seam's own
messages name the tool the planner has.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from compiler.surface import FixtureSurface
from google.adk.tools.base_tool import BaseTool
from harness.planner_seam import PlannerSeam, _RunTracker
from harness.sentences import SeamCode, refusal_sentence
from harness.settings import Settings
from planner.plan import load_plan
from planner.settings import AgentSettings, SchemaDelivery
from submission.catalogue import catalogue_revision
from submission.tool import load_attempts

from .test_planner_seam import ScriptedPlannerLlm, _SeamHarness, _translated_plan


def _compact_seam() -> PlannerSeam:
    return PlannerSeam(
        settings=Settings(),
        surface=FixtureSurface(schemas=[]),
        agent_settings=AgentSettings(planner_schema_delivery=SchemaDelivery.COMPACT),
    )


def _track(tracker: _RunTracker, response: dict[str, Any]) -> None:
    PlannerSeam._track_tool_call(
        tracker,
        None,
        agent_name="dfo_planner",
        tool=cast(BaseTool, SimpleNamespace(name="submit_analysis")),
        tool_response=response,
    )


async def test_a_compact_restart_is_refused_and_ends_the_turn(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A `question_fixed` refusal from `submit_analysis` ends the turn as `propose_plan`'s does."""
    seeded = _translated_plan(decision, requirement, derived_requirement, version=3)
    seam = _compact_seam()
    seam._agent.model = ScriptedPlannerLlm(
        script=[
            [
                (
                    "submit_analysis",
                    {
                        "submission_stage": "complete",
                        "catalogue_revision": catalogue_revision(),
                        "starts_new_plan": True,
                        "question": "a different question",
                    },
                )
            ]
        ]
    )
    harness = _SeamHarness()

    result = await harness.run(
        seam,
        question="kenya debt",
        readings=[],
        prior_plan=seeded.model_dump(mode="json"),
        amendment="actually, tell me about Kenya's ODA instead",
    )

    assert result is not None and "refused" in result
    assert "own message" in result["refused"]["text"]
    plan = load_plan(await harness.state())
    assert plan is not None and plan.version == 3


async def test_an_intent_alone_records_nothing_and_ends_the_entry_with_no_plan_ready() -> None:
    seam = _compact_seam()
    llm = ScriptedPlannerLlm(
        script=[
            [
                (
                    "submit_analysis",
                    {
                        "submission_stage": "intent",
                        "catalogue_revision": catalogue_revision(),
                        "question": "kenya debt",
                        "answer_mode": "computed",
                        "evidence_requirements": [{"kind": "dataset", "id": "debt"}],
                    },
                )
            ]
        ]
    )
    seam._agent.model = llm
    harness = _SeamHarness()

    output = await harness.run(seam, question="kenya debt", readings=[])

    assert output is not None
    assert output["refused"] == {
        "text": refusal_sentence({"code": SeamCode.PLAN_NOT_READY}),
        "final": False,
        "code": SeamCode.PLAN_NOT_READY,
    }
    state = await harness.state()
    assert load_plan(state) is None
    assert [attempt.outcome for attempt in load_attempts(state)] == ["requirements"]


def test_an_amendment_asks_for_submit_analysis() -> None:
    message = _compact_seam()._message(
        fresh=False,
        question="kenya debt",
        prior_plan=None,
        refusal=None,
        amendment="use constant prices",
        canvas_digest=None,
        conversation_digest=None,
    )

    assert "Call submit_analysis again" in message


@pytest.mark.parametrize("outcome", ["recorded", "needs_decision"])
def test_a_closing_translation_from_submit_analysis_ends_the_entry(outcome: str) -> None:
    tracker = _RunTracker()

    _track(
        tracker,
        {
            "outcome": outcome,
            "status": "draft",
            "version": 1,
            "translation": {
                "kind": "internal_failure",
                "outcome": "internal",
                "reason": "cycle: loop",
                "reference": "0123456789ab",
            },
        },
    )

    assert tracker.ready_to_break
    assert tracker.refused is not None


def test_a_terminal_refusal_from_submit_analysis_ends_the_entry() -> None:
    tracker = _RunTracker()

    _track(tracker, {"outcome": "refused", "code": "budget_exhausted", "message": "cap"})

    assert tracker.ready_to_break
    assert tracker.refused == {"outcome": "refused", "code": "budget_exhausted", "message": "cap"}


def test_a_plan_submit_analysis_declared_ready_is_the_entry_s_submission() -> None:
    tracker = _RunTracker()
    answer = {
        "outcome": "ready_for_review",
        "status": "pending_approval",
        "version": 1,
        "translation": {"outcome": "ok"},
        "summary": {"version": 1, "revision": "r", "warnings": []},
    }

    _track(tracker, answer)

    assert tracker.submitted == answer
    assert not tracker.ready_to_break
