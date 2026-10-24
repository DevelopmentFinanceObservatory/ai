"""Tests for the newest years a translation reads before approval, and the run that follows it.

`translator.tool`'s `propose_operations` reads each group's newest year as the
run will (`executor.run.read_anchors`), over a surface scripted like the
executor suite's, and restates a plan whose rolling years count from a later
year than the data now holds. The run of the approved plan then resolves the
same year, which `executor.run._Run._check_anchor_held` lets through.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from compiler.compile import CompileIssueCode
from compiler.sources import DiscoveryStatus
from compiler.surface import ToolCallResult, ToolSchema
from executor.outcomes import Accepted
from executor.run import execute_approved_plan
from planner.plan import (
    PLAN_STATE_KEY,
    DatasetRequirement,
    Decision,
    FilterRequirement,
    FirstYear,
    GrowthRequirement,
    Plan,
    PlanStatus,
    plan_revision,
)
from planner.records import PeriodThreshold, RelativePeriod, RollingWindow
from planner.tool_results import ProblemCode
from planner.workflow import SelectedSource
from translator.discovery import DiscoveryRecord, load_log, requirement_revision, save_log
from translator.mechanical import build_translation
from translator.referral import ReferralKind
from translator.tool import build_translator_core

from .conftest import SPEC_KENYA, ScriptedSurface, ok, refusal

LAST_FIVE = RollingWindow(
    window_start=RelativePeriod(offset=-5), window_end=RelativePeriod(offset=0)
)


class _Context:
    """A stand-in for ADK's `ToolContext`: only `.state` is read."""

    def __init__(self, plan: Plan, source: SelectedSource) -> None:
        self.state: dict[str, Any] = {PLAN_STATE_KEY: plan.model_dump_json()}
        dataset = next(r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement))
        record = DiscoveryRecord(
            revision=plan_revision(plan),
            tool="search",
            requirement_id=dataset.id,
            requirement_key=requirement_revision(dataset),
            status=DiscoveryStatus.DEFINITE,
            spec_ids=(source.spec_id,),
        )
        save_log(self.state, load_log(self.state).with_record(record, (source,)))

    def plan(self) -> Plan:
        return Plan.model_validate_json(self.state[PLAN_STATE_KEY])


@pytest.fixture
def rolling_draft(
    decision: Callable[..., Decision], requirement: Callable[..., DatasetRequirement]
) -> Callable[..., Plan]:
    """Return a factory for a draft over Kenya's debt service, 2015 to 2024, with no end year.

    `growth` is its growth over `window` and `newest` keeps its newest year,
    both planned against 2024.
    """

    def _draft(*, window: RollingWindow = LAST_FIVE, start: int = 2019) -> Plan:
        dataset = requirement(period_end_open=True)
        return Plan(
            question="How has Kenya's debt service changed over the last five years?",
            decisions=[decision()],
            evidence_requirements=[
                dataset,
                GrowthRequirement(
                    id="growth",
                    decision_ids=["measure"],
                    input_id=dataset.id,
                    window_start=start,
                    window_end=2024,
                    rolling=window,
                ),
                FilterRequirement(
                    id="newest",
                    decision_ids=["measure"],
                    input_id=dataset.id,
                    predicates=(
                        PeriodThreshold(op="eq", year=2024, rolling=RelativePeriod(offset=0)),
                    ),
                ),
            ],
            first_years=[FirstYear(requirement_id=dataset.id, year=2015)],
            status=PlanStatus.DRAFT,
        )

    return _draft


def _anchor(period: str) -> ToolCallResult:
    return ok({"anchor_key": "rak_" + "7" * 32, "period": period, "left_out": []})


def _script_analysis(surface: ScriptedSurface, plan: Plan) -> None:
    """Script an answer for each analysis step of `plan`, declaring the shape it expects."""
    assert plan.workflow is not None
    for step in plan.workflow.steps:
        if step.tool.server == "analysis-mcp":
            surface.queues[step.tool.name] = deque(
                [
                    ok(
                        {
                            "spec_id": "ops_" + "2" * 32,
                            "content_version": "ocv_" + "3" * 32,
                            "rows": 1,
                            "role_keys": list(step.expected_role_keys),
                            "stat_var_count": 1,
                        }
                    )
                ]
            )


async def _propose(context: _Context, surface: ScriptedSurface) -> dict[str, Any]:
    plan = context.plan()
    dataset = next(r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement))
    translation = build_translation(plan, {dataset.id: SPEC_KENYA})
    core = build_translator_core(surface=surface)
    return await core.propose_operations(translation=translation, tool_context=context)


@pytest.mark.parametrize(("newest", "window"), [("2023", (2018, 2023)), ("2024", (2019, 2024))])
async def test_a_translation_states_its_rolling_years_from_the_newest_year_the_data_holds(
    rolling_draft: Callable[..., Plan],
    source: Callable[..., SelectedSource],
    manifest: Callable[..., dict[str, Any]],
    real_schemas: list[ToolSchema],
    newest: str,
    window: tuple[int, int],
) -> None:
    """Planned against 2024. Where fewer than half the series hold 2024 the
    plan reaching approval counts from 2023, and the run of it resolves 2023
    again and runs; where 2024 holds, the plan is recorded as written."""
    context = _Context(rolling_draft(), source())
    surface = ScriptedSurface(
        schemas=real_schemas,
        responses={
            "get_data": [ok(manifest())],
            "resolve_anchor": [_anchor(newest)],
        },
    )

    answer = await _propose(context, surface)

    assert answer["outcome"] == "ok", answer
    stored = context.plan()
    assert stored.status is PlanStatus.PENDING_APPROVAL
    growth = stored.requirement("growth")
    kept = stored.requirement("newest")
    assert isinstance(growth, GrowthRequirement)
    assert isinstance(kept, FilterRequirement)
    assert (growth.window_start, growth.window_end) == window
    assert [p.year for p in kept.predicates if isinstance(p, PeriodThreshold)] == [window[1]]
    assert [name for name, _ in surface.calls] == ["get_data", "resolve_anchor"]

    _script_analysis(surface, stored)
    _, outcome = await execute_approved_plan(stored.approve(at=datetime.now(UTC)), surface=surface)

    assert isinstance(outcome, Accepted), outcome
    assert [anchor.period for anchor in outcome.anchors] == [newest]
    ((_, sent),) = [call for call in surface.calls if call[0] == "growth"]
    assert (sent["window_start"], sent["window_end"]) == tuple(str(year) for year in window)


async def test_a_restatement_a_plan_rule_refuses_goes_back_as_a_compile_issue_naming_the_year(
    rolling_draft: Callable[..., Plan],
    source: Callable[..., SelectedSource],
    manifest: Callable[..., dict[str, Any]],
    real_schemas: list[ToolSchema],
) -> None:
    """Growth from a fixed 2023 to the newest year reads one year once the
    newest year is 2023, so the plan goes back to the planner unchanged."""
    draft = rolling_draft(window=RollingWindow(window_end=RelativePeriod(offset=0)), start=2023)
    context = _Context(draft, source())
    surface = ScriptedSurface(
        schemas=real_schemas,
        responses={"get_data": [ok(manifest())], "resolve_anchor": [_anchor("2023")]},
    )

    answer = await _propose(context, surface)

    assert answer["code"] == ProblemCode.COMPILE_FAILED.value, answer
    (issue,) = answer["detail"]["issues"]
    assert (issue["code"], issue["requirement_id"]) == (
        CompileIssueCode.ROLLING_ANCHOR_BEHIND.value,
        "growth",
    )
    assert "2023" in issue["reason"]
    assert answer["action"]["arguments"]["kind"] == ReferralKind.REQUIREMENT_NEEDS_REPAIR.value
    assert answer["action"]["arguments"]["constraint"] == CompileIssueCode.ROLLING_ANCHOR_BEHIND
    assert context.plan() == draft


async def test_a_refused_retrieval_stops_the_translation_with_the_code_the_run_gives(
    rolling_draft: Callable[..., Plan],
    source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
) -> None:
    draft = rolling_draft()
    context = _Context(draft, source())
    surface = ScriptedSurface(
        schemas=real_schemas,
        responses={"get_data": [refusal("no_data_at_source", "do_not_retry")]},
    )

    answer = await _propose(context, surface)

    assert answer["code"] == ProblemCode.RUN_WOULD_STOP.value, answer
    detail = answer["detail"]
    assert (detail["code"], detail["step_id"], detail["requirement_id"]) == (
        "no_data_at_source",
        "annual_debt_service",
        "annual_debt_service",
    )
    referral = answer["action"]["arguments"]
    assert (referral["kind"], referral["requirement_id"], referral["stop_code"]) == (
        ReferralKind.REQUIREMENT_UNMEETABLE.value,
        "annual_debt_service",
        "no_data_at_source",
    )
    assert [name for name, _ in surface.calls] == ["get_data"]
    assert context.plan() == draft
