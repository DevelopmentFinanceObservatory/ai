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
from data_mcp.series_statistics import SeriesStatisticsSchemaError
from executor.outcomes import Accepted
from executor.run import execute_approved_plan
from harness.plan_display import step_rows
from planner.plan import (
    PLAN_STATE_KEY,
    DatasetRequirement,
    Decision,
    FilterRequirement,
    FirstYear,
    GrowthRequirement,
    Plan,
    PlanStatus,
    current_discovery_time_left,
    plan_revision,
)
from planner.records import Period, PeriodThreshold, RelativePeriod, RollingWindow
from planner.tool_results import ProblemCode
from planner.workflow import SelectedSource
from translator import tool
from translator.discovery import DiscoveryRecord, load_log, requirement_revision, save_log
from translator.mechanical import build_translation
from translator.referral import ReferralKind
from translator.tool import build_translator_core

from .conftest import ScriptedSurface, ok, refusal

LAST_FIVE = RollingWindow(
    window_start=RelativePeriod(offset=-5), window_end=RelativePeriod(offset=0)
)


class _Context:
    """A stand-in for ADK's `ToolContext`: only `.state` is read."""

    def __init__(self, plan: Plan, *sources: SelectedSource) -> None:
        """`sources` settle `plan`'s dataset requirements, one each, in plan order."""
        self.state: dict[str, Any] = {PLAN_STATE_KEY: plan.model_dump_json()}
        datasets = [r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)]
        for dataset, source in zip(datasets, sources, strict=True):
            record = DiscoveryRecord(
                revision=plan_revision(plan),
                tool="search",
                requirement_id=dataset.id,
                requirement_key=requirement_revision(dataset),
                status=DiscoveryStatus.DEFINITE,
                spec_ids=(source.spec_id,),
            )
            save_log(self.state, load_log(self.state).with_record(record, (source,)))
        self.specs = {
            dataset.id: source.spec_id for dataset, source in zip(datasets, sources, strict=True)
        }

    def plan(self) -> Plan:
        return Plan.model_validate_json(self.state[PLAN_STATE_KEY])


@pytest.fixture
def rolling_draft(
    decision: Callable[..., Decision], requirement: Callable[..., DatasetRequirement]
) -> Callable[..., Plan]:
    """Return a factory for a draft over Kenya's debt service, 2015 to 2024, with no end year.

    `growth` is its growth over `window` and `newest` keeps its newest year,
    both planned against 2024. `period` states other years for the dataset,
    and `first_year` the first year it holds on record.
    """

    def _draft(
        *,
        window: RollingWindow = LAST_FIVE,
        start: int = 2019,
        period: Period | None = None,
        first_year: int = 2015,
    ) -> Plan:
        dataset = requirement(period_end_open=True, **({"period": period} if period else {}))
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
            first_years=[FirstYear(requirement_id=dataset.id, year=first_year)],
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
    translation = build_translation(context.plan(), context.specs)
    core = build_translator_core(surface=surface)
    return await core.propose_operations(translation=translation, tool_context=context)


@pytest.mark.parametrize(
    ("newest", "window", "stated"),
    [
        ("2023", (2018, 2023), None),
        ("2024", (2019, 2024), None),
        # Stated from 2019 with 2010 on record, the growth reads the whole history.
        ("2023", (2018, 2023), Period(start=2019, end=2024)),
    ],
    ids=["behind", "held", "behind-over-the-whole-history"],
)
async def test_a_translation_states_its_rolling_years_from_the_newest_year_the_data_holds(
    rolling_draft: Callable[..., Plan],
    source: Callable[..., SelectedSource],
    manifest: Callable[..., dict[str, Any]],
    real_schemas: list[ToolSchema],
    newest: str,
    window: tuple[int, int],
    stated: Period | None,
) -> None:
    """Planned against 2024. Where fewer than half the series hold 2024 the
    plan reaching approval counts from 2023, the answer names the years
    moved, and the run of it resolves 2023 again and runs; where 2024
    holds, the plan is recorded as written."""
    draft = rolling_draft(period=stated, first_year=2015 if stated is None else 2010)
    context = _Context(draft, source())
    surface = ScriptedSurface(
        schemas=real_schemas,
        responses={
            "get_data": [ok(manifest())],
            "resolve_anchor": [_anchor(newest)],
        },
    )

    answer = await _propose(context, surface)

    assert answer["outcome"] == "ok", answer
    moved = (
        []
        if newest == "2024"
        else [
            {
                "requirement_id": "growth",
                "anchor": 2023,
                "years_from": [2019, 2024],
                "years_to": [2018, 2023],
            },
            {"requirement_id": "newest", "anchor": 2023, "years_from": [2024], "years_to": [2023]},
        ]
    )
    assert answer.get("restated", []) == moved
    # The planner reads the moved years beside the verdict.
    assert tool._planner_answer(answer) == {"outcome": "ok"} | (
        {"restated": moved} if moved else {}
    )
    stored = context.plan()
    assert stored.status is PlanStatus.PENDING_APPROVAL
    growth = stored.requirement("growth")
    kept = stored.requirement("newest")
    assert isinstance(growth, GrowthRequirement)
    assert isinstance(kept, FilterRequirement)
    assert (growth.window_start, growth.window_end) == window
    assert [p.year for p in kept.predicates if isinstance(p, PeriodThreshold)] == [window[1]]
    # The approval card reads the plan as recorded, so it states the years the answer names.
    card = {row["id"]: row["description"] for row in step_rows(stored)}
    assert f"from {window[0]} to {window[1]}, following the newest year" in str(card["growth"])
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


@pytest.mark.parametrize(
    ("fix", "kind"),
    [
        ({"detail": "no_qualifying_period"}, ReferralKind.REQUIREMENT_UNMEETABLE),
        ({"detail": "populations_unmatched"}, ReferralKind.INTERNAL_FAILURE),
    ],
    ids=["data", "harness"],
)
async def test_a_resolution_refused_at_translation_goes_back_by_whose_stop_it_is(
    rolling_draft: Callable[..., Plan],
    source: Callable[..., SelectedSource],
    manifest: Callable[..., dict[str, Any]],
    real_schemas: list[ToolSchema],
    fix: dict[str, Any],
    kind: ReferralKind,
) -> None:
    """No year half the series hold is the data's, for the user to decide; populations the
    run sent that do not fit its frames are the harness's own fault."""
    draft = rolling_draft()
    context = _Context(draft, source())
    surface = ScriptedSurface(
        schemas=real_schemas,
        responses={
            "get_data": [ok(manifest())],
            "resolve_anchor": [refusal("rolling_anchor_unresolved", "do_not_retry", fix=fix)],
        },
    )

    answer = await _propose(context, surface)

    assert answer["code"] == ProblemCode.RUN_WOULD_STOP.value, answer
    referral = answer["action"]["arguments"]
    assert referral["kind"] == kind.value
    assert ("requirement_id" in referral) is (kind is ReferralKind.REQUIREMENT_UNMEETABLE)
    assert context.plan() == draft


async def _times_out(**_kwargs: Any) -> None:
    raise TimeoutError("signed artifact GET timed out")


async def _fails_its_digest(**_kwargs: Any) -> None:
    raise SeriesStatisticsSchemaError("fetched artifact bytes do not hash to the advertised digest")


@pytest.mark.parametrize(
    ("fetch", "held", "stop_code", "kind"),
    [
        (_times_out, "country/KEN", "transport_failed", ReferralKind.SURFACE_UNAVAILABLE),
        (_fails_its_digest, "country/KEN", "step_result_unusable", ReferralKind.INTERNAL_FAILURE),
        (None, "country/UGA", "entity_missing", ReferralKind.REQUIREMENT_UNMEETABLE),
    ],
    ids=["transport", "malformed", "shortfall"],
)
async def test_a_statistics_check_that_stops_at_translation_goes_back_by_whose_stop_it_is(
    rolling_draft: Callable[..., Plan],
    source: Callable[..., SelectedSource],
    manifest: Callable[..., dict[str, Any]],
    real_schemas: list[ToolSchema],
    monkeypatch: pytest.MonkeyPatch,
    fetch: Callable[..., Any] | None,
    held: str,
    stop_code: str,
    kind: ReferralKind,
) -> None:
    """A statistics artifact that cannot be fetched now is a later try's to get past, and
    one the run cannot trust is the harness's; only statistics that hold no series for
    Kenya are the data's, for the user to decide. The failures are injected at the download helper, so
    `executor.run.default_fetch_series_stats` reads them as production does."""
    if fetch is not None:
        monkeypatch.setattr("executor.run.fetch_series_statistics_table", fetch)
    draft = rolling_draft()
    context = _Context(draft, source())
    retrieved = manifest()
    retrieved["statistics"][0]["entities"][0]["ref"] = held
    surface = ScriptedSurface(schemas=real_schemas, responses={"get_data": [ok(retrieved)]})

    answer = await _propose(context, surface)

    assert answer["code"] == ProblemCode.RUN_WOULD_STOP.value, answer
    referral = answer["action"]["arguments"]
    assert (referral["kind"], referral["stop_code"]) == (kind.value, stop_code)
    assert ("requirement_id" in referral) is (kind is ReferralKind.REQUIREMENT_UNMEETABLE)
    assert [name for name, _ in surface.calls] == ["get_data"]
    assert context.plan() == draft


@pytest.mark.parametrize(
    "refused",
    [
        refusal("entity_ambiguous", "correct_and_retry", fix={"candidates": []}),
        refusal("upstream_version_changed", "refresh_then_retry"),
    ],
    ids=["method-choice", "rediscovery"],
)
async def test_a_read_meeting_a_choice_or_a_stale_source_records_the_plan_as_written(
    rolling_draft: Callable[..., Plan],
    source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
    refused: ToolCallResult,
) -> None:
    """The run asks for the choice, or discovers again, as it does for any plan."""
    context = _Context(rolling_draft(), source())
    surface = ScriptedSurface(schemas=real_schemas, responses={"get_data": [refused]})

    answer = await _propose(context, surface)

    assert answer["outcome"] == "ok", answer
    assert "restated" not in answer
    growth = context.plan().requirement("growth")
    assert isinstance(growth, GrowthRequirement)
    assert (growth.window_start, growth.window_end) == (2019, 2024)


@pytest.mark.parametrize(("time_left", "calls"), [(0.0, []), (0.05, ["get_data"])])
async def test_a_read_past_the_entrys_time_records_the_plan_as_written(
    rolling_draft: Callable[..., Plan],
    source: Callable[..., SelectedSource],
    manifest: Callable[..., dict[str, Any]],
    real_schemas: list[ToolSchema],
    monkeypatch: pytest.MonkeyPatch,
    time_left: float,
    calls: list[str],
) -> None:
    """With no time left the read is not started, and a read the entry's time cuts short is
    dropped; either is recorded on the trace, and the run's own check guards the years."""
    events: list[str] = []
    monkeypatch.setattr(tool, "record_event", lambda name, **_: events.append(name))
    context = _Context(rolling_draft(), source())
    surface = ScriptedSurface(
        schemas=real_schemas,
        responses={"get_data": [ok(manifest())], "resolve_anchor": [_anchor("2023")]},
        delays={"get_data": 5.0},
    )
    token = current_discovery_time_left.set(lambda: time_left)
    try:
        answer = await _propose(context, surface)
    finally:
        current_discovery_time_left.reset(token)

    assert answer["outcome"] == "ok", answer
    assert "restated" not in answer
    growth = context.plan().requirement("growth")
    assert isinstance(growth, GrowthRequirement)
    assert (growth.window_start, growth.window_end) == (2019, 2024)
    assert [name for name, _ in surface.calls] == calls
    assert events == ["translation.anchor_read_skipped"]


async def test_a_restatement_refused_names_the_requirement_whose_move_the_rule_refuses(
    decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    source: Callable[..., SelectedSource],
    gdp_source: Callable[..., SelectedSource],
    manifest_for: Callable[..., dict[str, Any]],
    real_schemas: list[ToolSchema],
) -> None:
    """Both groups count from 2023, GDP's read first. GDP's growth over the last five years
    moves cleanly; debt's from a fixed 2023 to the newest year reads one year, so the issue
    is debt's."""
    debt = requirement(period_end_open=True)
    gdp = requirement(id="annual_gdp", measure="gross domestic product", period_end_open=True)
    draft = Plan(
        question="How have Kenya's debt service and GDP changed?",
        decisions=[decision()],
        evidence_requirements=[
            debt,
            GrowthRequirement(
                id="growth",
                decision_ids=["measure"],
                input_id=debt.id,
                window_start=2023,
                window_end=2024,
                rolling=RollingWindow(window_end=RelativePeriod(offset=0)),
            ),
            gdp,
            GrowthRequirement(
                id="gdp_growth",
                decision_ids=["measure"],
                input_id=gdp.id,
                window_start=2019,
                window_end=2024,
                rolling=LAST_FIVE,
            ),
        ],
        first_years=[FirstYear(requirement_id=r.id, year=2015) for r in (debt, gdp)],
        status=PlanStatus.DRAFT,
    )
    debt_source, gdp_source_ = source(), gdp_source()
    context = _Context(draft, debt_source, gdp_source_)
    surface = ScriptedSurface(
        schemas=real_schemas,
        responses={
            "get_data": [ok(manifest_for(debt_source)), ok(manifest_for(gdp_source_))],
            "resolve_anchor": [_anchor("2023")],
        },
    )

    answer = await _propose(context, surface)

    assert answer["code"] == ProblemCode.COMPILE_FAILED.value, answer
    (issue,) = answer["detail"]["issues"]
    assert (issue["code"], issue["requirement_id"]) == (
        CompileIssueCode.ROLLING_ANCHOR_BEHIND.value,
        "growth",
    )
    assert context.plan() == draft
