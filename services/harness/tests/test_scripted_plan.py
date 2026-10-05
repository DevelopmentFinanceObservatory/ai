"""`scripted_plan` and `ScriptedSurface`: the canned plan scripted mode runs.

The pins on `scripted_plan`'s workflow are computed from `ScriptedSurface`'s
own declared schemas at build time (`harness.scripted_plan`'s module
docstring), so the proof that they hold is to drive the plan through the real
comparison that would refuse them: `executor.run.execute_approved_plan`,
which calls `_surface_as_compiled` before running a single step. Perturbing
one declared schema and watching the same comparison refuse is what shows the
pins are load-bearing rather than merely present.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from compiler.compile import CompileIssue, CompileIssueCode, CompileResult
from compiler.surface import ProgressReport, ToolSchema
from executor.outcomes import Accepted, RefusalKind, Refused
from executor.run import execute_approved_plan
from harness.scripted import (
    SCRIPTED_MISNAMED_SPEC,
    SCRIPTED_SPEC,
    ScriptedConverse,
    ScriptedPlannerModel,
    ScriptedSurface,
)
from harness.scripted_plan import scripted_fetch_series_stats, scripted_plan
from planner.plan import CUT_MARKER, MAX_QUESTION_CHARS, Plan, PlanStatus, cut_user_text
from planner.workflow import ToolId
from pydantic import ValidationError


class _DriftedSurface(ScriptedSurface):
    """A `ScriptedSurface` whose `aggregate` schema has grown an argument.

    Everything else is unchanged, so a run against this surface exercises
    exactly one difference: `aggregate`'s `input_schema_hash` no longer
    matches what the plan was compiled against.
    """

    async def list_tools(self) -> list[ToolSchema]:
        schemas = await super().list_tools()
        drifted: list[ToolSchema] = []
        for schema in schemas:
            if schema.name != "aggregate":
                drifted.append(schema)
                continue
            parameters = {
                **schema.parameters,
                "properties": {
                    **schema.parameters["properties"],
                    "ceiling": {"type": "integer"},
                },
            }
            drifted.append(schema.model_copy(update={"parameters": parameters}))
        return drifted


def test_scripted_plan_validates_and_reaches_pending_approval() -> None:
    plan = scripted_plan("What is life expectancy in Kenya?", spec_id=SCRIPTED_SPEC, parts=1)

    assert isinstance(plan, Plan)
    assert plan.status is PlanStatus.PENDING_APPROVAL
    assert plan.workflow is not None
    assert [step.tool.name for step in plan.workflow.steps] == ["get_data", "aggregate"]


def test_an_extra_field_is_refused() -> None:
    plan = scripted_plan("What is life expectancy in Kenya?", spec_id=SCRIPTED_SPEC, parts=1)

    with pytest.raises(ValidationError):
        Plan.model_validate({**plan.model_dump(mode="json"), "not_a_real_field": True})


LONG_QUESTION = ("What is life expectancy in Kenya? " * 150)[:5_000]


def test_a_question_past_the_plan_limit_is_cut_and_the_plan_still_builds() -> None:
    assert len(LONG_QUESTION) == 5_000

    plan = scripted_plan(LONG_QUESTION, spec_id=SCRIPTED_SPEC, parts=1)

    assert plan.status is PlanStatus.PENDING_APPROVAL
    assert plan.question == cut_user_text(LONG_QUESTION, MAX_QUESTION_CHARS)
    assert len(plan.question) == MAX_QUESTION_CHARS
    assert plan.question.startswith("What is life expectancy in Kenya? What is")
    assert plan.question.endswith(CUT_MARKER)


async def test_the_scripted_planner_drafts_a_plan_for_a_question_past_the_limit() -> None:
    planner = ScriptedPlannerModel()

    drafted = await planner(
        ctx=cast(Any, None), question=LONG_QUESTION, envelope={"spec_id": SCRIPTED_SPEC}
    )

    assert len(drafted["plan"]["question"]) == MAX_QUESTION_CHARS


def test_a_question_within_the_plan_limit_is_carried_through_unchanged() -> None:
    question = "What is life expectancy in Kenya?"

    assert scripted_plan(question, spec_id=SCRIPTED_SPEC, parts=1).question == question


async def test_the_workflows_pins_match_the_surfaces_declared_schemas() -> None:
    plan = scripted_plan("What is life expectancy in Kenya?", spec_id=SCRIPTED_SPEC, parts=1)

    _, outcome = await execute_approved_plan(
        plan.approve(), surface=ScriptedSurface(), fetch_series_stats=scripted_fetch_series_stats
    )

    assert isinstance(outcome, Accepted), outcome


async def test_a_changed_declared_schema_makes_the_surface_check_refuse() -> None:
    plan = scripted_plan("What is life expectancy in Kenya?", spec_id=SCRIPTED_SPEC, parts=1)

    _, outcome = await execute_approved_plan(plan.approve(), surface=_DriftedSurface())

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.SURFACE_CHANGED
    assert "different arguments" in outcome.detail


def test_a_compile_that_yields_no_workflow_raises_instead_of_returning_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`compile_plan` failing on the canned plan's own fixed schemas is a fault in this
    module (the module docstring's premise), not a case `scripted_plan` should carry
    forward as a `PENDING_APPROVAL` plan with no workflow. Forcing `compile_plan` to
    return issues instead of a workflow proves the guard raises rather than silently
    returning `None`, which `Plan.workflow: Workflow | None` would otherwise accept.
    """

    def _broken_compile(**_: object) -> CompileResult:
        return CompileResult(
            issues=(CompileIssue(code=CompileIssueCode.CYCLE, reason="forced by test"),)
        )

    monkeypatch.setattr("harness.scripted_plan.compile_plan", _broken_compile)

    with pytest.raises(RuntimeError, match="did not compile"):
        scripted_plan("What is life expectancy in Kenya?", spec_id=SCRIPTED_SPEC, parts=1)


async def test_the_misnamed_questions_analysis_step_refuses_role_ambiguous() -> None:
    plan = scripted_plan(
        "What's driving the misnamed indicator in Kenya?",
        spec_id=SCRIPTED_MISNAMED_SPEC,
        parts=1,
    )

    _, outcome = await execute_approved_plan(
        plan.approve(), surface=ScriptedSurface(), fetch_series_stats=scripted_fetch_series_stats
    )

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.STEP_REFUSED
    assert outcome.payload is not None
    assert outcome.payload.code == "role_ambiguous"


async def test_scripted_converse_drafts_the_entity_the_question_names() -> None:
    """The ordinary scripted path draws one part about the country the
    question names, with no `revises`, so the turn draws a new card."""
    result = await ScriptedConverse()(question="life expectancy in Kenya")

    assert result["revises"] is None
    assert result["search_parts"] == [
        {"variable": "life expectancy", "entities": [{"text": "Kenya"}]}
    ]


async def test_scripted_converse_names_the_newest_card_and_both_entities_to_change() -> None:
    """An "add ... to the chart" question against a digest whose newest block
    shows Nigeria: the draft names that block, and the part carries its
    Nigeria plus the Guatemala the question adds."""
    digest = {
        "blocks": [
            {
                "component_id": "cmp_scripted0001",
                "type": "line_chart",
                "title": "Life expectancy in Nigeria",
                "entities": ["Nigeria"],
                "variable": ["Life Expectancy"],
            }
        ]
    }

    result = await ScriptedConverse()(question="add Guatemala to the chart", canvas_digest=digest)

    assert result["revises"] == "cmp_scripted0001"
    assert result["search_parts"] == [
        {"variable": "life expectancy", "entities": [{"text": "Nigeria"}, {"text": "Guatemala"}]}
    ]


async def test_scripted_converse_sets_no_revises_without_a_canvas() -> None:
    """A "to the chart" question names nothing when the conversation has no
    cards yet, so the draft leaves the field unset and the turn appends."""
    result = await ScriptedConverse()(question="add Guatemala to the chart")

    assert result["revises"] is None


_SCRIPTED_PROGRESS = (
    ProgressReport(progress=0, message="submitted"),
    ProgressReport(progress=10, message="running"),
)
_GET_DATA = ToolId(server="data-mcp", name="get_data")


async def test_a_scripted_get_data_plays_its_progress_before_it_answers() -> None:
    surface = ScriptedSurface(get_data_progress=_SCRIPTED_PROGRESS)
    reports: list[ProgressReport] = []

    result = await surface.call_tool(
        tool=_GET_DATA, arguments={"spec_id": SCRIPTED_SPEC}, on_progress=reports.append
    )

    assert reports == list(_SCRIPTED_PROGRESS)
    assert result.is_error is False


async def test_a_raising_callback_does_not_fail_a_scripted_get_data() -> None:
    surface = ScriptedSurface(get_data_progress=_SCRIPTED_PROGRESS)
    heard: list[ProgressReport] = []

    def _raises(report: ProgressReport) -> None:
        heard.append(report)
        raise RuntimeError("relay closed")

    result = await surface.call_tool(
        tool=_GET_DATA, arguments={"spec_id": SCRIPTED_SPEC}, on_progress=_raises
    )

    assert heard == list(_SCRIPTED_PROGRESS)
    assert result.is_error is False


async def test_the_default_scripted_surface_reports_no_progress() -> None:
    reports: list[ProgressReport] = []

    await ScriptedSurface().call_tool(
        tool=_GET_DATA, arguments={"spec_id": SCRIPTED_SPEC}, on_progress=reports.append
    )

    assert reports == []
