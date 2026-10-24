"""Tests for rolling periods in `executor.run`: ordering, the one resolution, and its refusals.

Every case runs an approved plan holding a rolling growth and a rolling filter
over one retrieval against a scripted surface carrying the real servers'
schemas, so the arguments the run writes are checked against the tools as
served. `resolve_anchor` is scripted like any other tool, and a surface with no
answer for it fails the call, so a run that reaches it unasked does not pass.
`test_wide_frame.py` runs a rolling step on the real analysis-mcp server.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from analysis_mcp.tool_errors import ToolRefusal, to_tool_result
from compiler.capabilities import CAPABILITIES
from compiler.compile import compile_plan
from compiler.rolling import RollingBound
from compiler.surface import ToolCallResult, ToolSchema, ToolSurfaceUnavailableError
from executor.classify import (
    Disposition,
    ExecutorCode,
    classify_refusal,
)
from executor.outcomes import (
    Accepted,
    AnchorResult,
    LeftOutSeries,
    RefusalKind,
    Refused,
)
from executor.run import (
    _NAMED_YEARS_REMEDY,
    MAX_ATTEMPTS_PER_STEP,
    _first_year_kept,
    _waves,
    execute_approved_plan,
)
from planner.operations import (
    AggregateOperation,
    FilterOperation,
    GroupBy,
    GrowthOperation,
    Reduction,
    RetrieveOperation,
    Translation,
)
from planner.plan import (
    AggregateRequirement,
    AskAnswer,
    DatasetRequirement,
    Decision,
    Entity,
    FilterRequirement,
    FirstYear,
    GrowthRequirement,
    Period,
    Plan,
    PlanStatus,
    split_of,
)
from planner.records import PeriodThreshold, RelativePeriod, RollingWindow
from planner.workflow import EvidenceBinding, SelectedSource, ToolId, Workflow, WorkflowStep

from .conftest import SPEC_KENYA, SPEC_POPULATION, ScriptedSurface, ok, refusal

PlanFactory = Callable[..., Plan]
ManifestFactory = Callable[..., dict[str, Any]]

ANCHOR_KEY = "rak_" + "7" * 32
LAST_FIVE = RollingWindow(
    window_start=RelativePeriod(offset=-5), window_end=RelativePeriod(offset=0)
)
LEFT_OUT = {"root": 0, "coordinates": {"recipient": "country/UGA"}}


def _anchor(period: str = "2024") -> ToolCallResult:
    """`resolve_anchor`'s answer naming `period` as the newest."""
    return ok({"anchor_key": ANCHOR_KEY, "period": period, "left_out": [LEFT_OUT]})


@pytest.fixture
def rolling_plan(
    decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    source: Callable[..., SelectedSource],
    population_source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
) -> PlanFactory:
    """Return a factory for an approved plan whose growth and filter follow the newest year.

    `debt` is retrieved from `start` to 2024, `change` is its growth over the
    last five years and `newest` keeps its newest year, both planned against
    `newest`. `rolling=False` plans the same years fixed.
    `second_retrieval=True` adds a population retrieval and its total, which
    neither rolling step reads.
    """

    def _plan(
        *,
        rolling: bool = True,
        second_retrieval: bool = False,
        newest: int = 2024,
        start: int = 2015,
    ) -> Plan:
        window: dict[str, Any] = {"window_start": newest - 5, "window_end": newest}
        if rolling:
            window["rolling"] = LAST_FIVE
        kept = (
            PeriodThreshold(
                op="eq", year=newest, rolling=RelativePeriod(offset=0) if rolling else None
            ),
        )
        requirements: list[Any] = [
            requirement(period=Period(start=start, end=2024), period_end_open=rolling),
            GrowthRequirement(
                id="growth_of_debt",
                decision_ids=["measure"],
                input_id="annual_debt_service",
                **window,
            ),
            FilterRequirement(
                id="newest_year",
                decision_ids=["measure"],
                input_id="annual_debt_service",
                predicates=kept,
            ),
        ]
        operations: list[Any] = [
            RetrieveOperation(id="debt", requirement_id="annual_debt_service", spec_id=SPEC_KENYA),
            GrowthOperation(id="change", requirement_id="growth_of_debt", input="debt", **window),
            FilterOperation(
                id="newest", requirement_id="newest_year", input="debt", predicates=kept
            ),
        ]
        terminal = ["change", "newest"]
        discovered = {"annual_debt_service": source()}
        if second_retrieval:
            requirements += [
                DatasetRequirement(
                    id="population",
                    decision_ids=["measure"],
                    measure="total population",
                    entities=[Entity(name="Kenya")],
                    period=Period(start=2015, end=2024),
                    period_end_open=rolling,
                ),
                AggregateRequirement(
                    id="population_totals",
                    decision_ids=["measure"],
                    input_id="population",
                    reduction=Reduction.SUM,
                    group_by=(GroupBy.PERIOD,),
                ),
            ]
            operations += [
                RetrieveOperation(id="pop", requirement_id="population", spec_id=SPEC_POPULATION),
                AggregateOperation(
                    id="people",
                    requirement_id="population_totals",
                    input="pop",
                    reduction=Reduction.SUM,
                    group_by=(GroupBy.PERIOD,),
                ),
            ]
            terminal.append("people")
            discovered["population"] = population_source()
        draft = Plan(
            question="How has Kenya's debt service changed over the last five years?",
            decisions=[decision()],
            evidence_requirements=requirements,
            first_years=[
                FirstYear(requirement_id=r.id, year=r.period.start)
                for r in requirements
                if isinstance(r, DatasetRequirement) and r.period_end_open
            ],
            status=PlanStatus.DRAFT,
        )
        translation = Translation(operations=operations, terminal=terminal)
        compiled = compile_plan(
            plan=draft,
            translation=translation,
            schemas={schema.name: schema for schema in real_schemas},
            discovered=discovered,
        )
        assert compiled.workflow is not None, compiled.issues
        return draft.with_translation(translation=translation, workflow=compiled.workflow).approve(
            at=datetime.now(UTC)
        )

    return _plan


def _answers(plan: Plan) -> dict[str, list[ToolCallResult]]:
    """A scripted answer for each analysis tool of `plan`, declaring the shape its step expects."""
    assert plan.workflow is not None
    return {
        step.tool.name: [
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
        for step in plan.workflow.steps
        if step.tool.server == "analysis-mcp"
    }


def _surface(
    plan: Plan, real_schemas: list[ToolSchema], manifest: dict[str, Any], **overrides: Any
) -> ScriptedSurface:
    """A surface answering every tool of `plan`, and `resolve_anchor` with 2024 as the newest year."""
    responses: dict[str, Any] = {
        "get_data": [ok(manifest)],
        "resolve_anchor": [_anchor()],
        **_answers(plan),
    }
    delays = overrides.pop("delays", None)
    return ScriptedSurface(schemas=real_schemas, responses=responses | overrides, delays=delays)


async def _no_wait(seconds: float) -> None:
    del seconds


async def _run(plan: Plan, surface: ScriptedSurface, **kwargs: Any) -> Any:
    _, outcome = await execute_approved_plan(plan, surface=surface, sleep=_no_wait, **kwargs)
    return outcome


def _calls(surface: ScriptedSurface, tool: str) -> list[dict[str, Any]]:
    return [arguments for name, arguments in surface.calls if name == tool]


def _served(served: ToolRefusal) -> ToolCallResult:
    """`served` as the surface hands the executor a refusal analysis-mcp answered."""
    result = to_tool_result(served)
    return ToolCallResult(
        is_error=True, structured_content=result.structured_content, text=served.message
    )


# --- Ordering ------------------------------------------------------------------------


def _step(step_id: str, tool: str, *, reads: str | None = None, **args: Any) -> WorkflowStep:
    return WorkflowStep(
        id=step_id,
        operation_id=step_id,
        tool=ToolId(server="data-mcp" if tool == "get_data" else "analysis-mcp", name=tool),
        args=args,
        bindings=(
            {} if reads is None else {"frame_ref": {"from_step": reads, "output": "frame_ref"}}
        ),
        input_schema_hash="0" * 64,
    )


@pytest.mark.parametrize(
    ("rolling", "waves"),
    [
        (None, [("first",), ("total", "change"), ("second",), ("level",), ("both",)]),
        (
            {"population": []},
            [("first",), ("total",), ("second",), ("change", "level"), ("both",)],
        ),
    ],
    ids=["fixed", "rolling"],
)
def test_a_rolling_step_runs_after_every_retrieval_of_its_group_and_a_fixed_one_after_its_bindings(
    rolling: dict[str, Any] | None, waves: list[tuple[str, ...]]
) -> None:
    # `both` reads `change` and `level`, so the two stand on one period, taken
    # over both retrievals: `change` runs after `second`, a retrieval two levels
    # up that it binds no frame from.
    args = {} if rolling is None else {"rolling": rolling}
    both = WorkflowStep.model_validate(
        _step("both", "join", reads="change").model_dump()
        | {
            "bindings": {
                "frame_ref": {"from_step": "change", "output": "frame_ref"},
                "right_frame_ref": {"from_step": "level", "output": "frame_ref"},
            }
        }
    )
    workflow = Workflow(
        steps=(
            _step("first", "get_data"),
            _step("total", "aggregate", reads="first"),
            _step("second", "get_data", reads="total"),
            _step("change", "growth", reads="first", **args),
            _step("level", "filter", reads="second", **args),
            both,
        ),
        sources=(),
        evidence_bindings=(
            EvidenceBinding(requirement_id="one", step_id="first", spec_id=SPEC_KENYA),
            EvidenceBinding(requirement_id="two", step_id="second", spec_id=SPEC_POPULATION),
        ),
        terminal_steps=("both",),
    )

    assert [tuple(step.id for step in wave) for wave in _waves(workflow)] == waves


async def test_the_newest_period_is_resolved_once_over_the_retrievals_the_rolling_steps_read(
    rolling_plan: PlanFactory,
    manifest: Callable[..., dict[str, Any]],
    manifest_for: Callable[..., dict[str, Any]],
    population_source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
) -> None:
    plan = rolling_plan(second_retrieval=True)
    people = manifest_for(population_source())
    surface = _surface(
        plan,
        real_schemas,
        manifest(),
        get_data=[ok(manifest()), ok(people)],
        delays={"growth": 0.01, "filter": 0.01},
    )

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    names = [name for name, _ in surface.calls]
    # The growth and the filter each stand on a period over the debt retrieval
    # alone, so they share one resolution.
    assert names.count("resolve_anchor") == 1
    resolved = names.index("resolve_anchor")
    assert "get_data" in names[:resolved]
    assert not {"growth", "filter"} & set(names[:resolved])
    (sent,) = _calls(surface, "resolve_anchor")
    # The population retrieval no rolling step reads is no part of the period.
    assert sent["roots"] == [manifest()["frame_ref"]]
    assert {population["spec_id"] for population in sent["populations"]} == {SPEC_KENYA}
    assert len(outcome.anchors) == 1
    # The wave's two rolling steps were in flight together.
    assert surface.max_concurrency == 2


# --- What a rolling step is sent, and what the run records -----------------------------


@pytest.mark.parametrize(
    ("newest", "window"), [("2024", ("2019", "2024")), ("2025", ("2020", "2025"))]
)
async def test_a_rolling_steps_years_are_written_from_the_newest_period(
    rolling_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    newest: str,
    window: tuple[str, str],
) -> None:
    # Planned against 2024: a newest year on it or after it runs.
    plan = rolling_plan()
    surface = _surface(plan, real_schemas, manifest(), resolve_anchor=[_anchor(newest)])

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    (sent,) = _calls(surface, "resolve_anchor")
    (growth,) = _calls(surface, "growth")
    (kept,) = _calls(surface, "filter")
    assert (growth["window_start"], growth["window_end"]) == window
    assert [predicate["value"] for predicate in kept["predicates"]] == [newest]
    assert growth["anchor_key"] == kept["anchor_key"] == ANCHOR_KEY
    # One population value reaches the resolution and every rolling call.
    assert growth["rolling"]["population"] == kept["rolling"]["population"] == sent["populations"]
    assert growth["rolling"]["window_start"]["offset"] == -5
    assert outcome.anchors == (
        AnchorResult(
            anchor_key=ANCHOR_KEY,
            period=newest,
            left_out=(LeftOutSeries.model_validate(LEFT_OUT),),
        ),
    )


async def test_a_newest_period_before_the_plans_stops_the_run_before_any_rolling_step(
    rolling_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    # Planned against 2024; fewer than half the series now hold 2024, so the
    # newest year is 2023 and the years the user approved are not the ones read.
    plan = rolling_plan()
    surface = _surface(plan, real_schemas, manifest(), resolve_anchor=[_anchor("2023")])

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.ROLLING_ANCHOR_BEHIND_PLAN
    assert outcome.step_id == "change"
    assert outcome.payload is not None
    assert outcome.payload.code == ExecutorCode.ROLLING_ANCHOR_BEHIND_PLAN
    assert classify_refusal(outcome.payload) is Disposition.STOP
    assert outcome.payload.fix == {
        "remedy": outcome.payload.fix["remedy"],
        "anchor": "2023",
        "planned_anchor": "2024",
    }
    for named in ("2023", "2024"):
        assert named in outcome.payload.message
    assert [record.step_id for record in outcome.completed] == ["debt"]
    assert [anchor.period for anchor in outcome.anchors] == ["2023"]
    assert not _calls(surface, "growth") and not _calls(surface, "filter")


async def test_a_fixed_plan_resolves_no_period_and_its_records_are_unchanged(
    rolling_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    plan = rolling_plan(rolling=False)
    surface = _surface(plan, real_schemas, manifest())

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    assert [name for name, _ in surface.calls] == ["get_data", "growth", "filter"]
    (growth,) = _calls(surface, "growth")
    assert (growth["window_start"], growth["window_end"]) == ("2019", "2024")
    assert "anchor_key" not in growth
    assert outcome.anchors == ()
    for record in outcome.steps:
        assert set(record.model_dump()) == {
            "step_id",
            "tool",
            "attempts",
            "frame_ref",
            "manifest_ref",
            "digest",
            "row_count",
            "role_keys",
            "stat_var_count",
            "dropped_key_count",
            "ranked_before_limit",
        }


async def test_a_retried_rolling_step_reuses_the_runs_newest_period(
    rolling_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    plan = rolling_plan()
    answers = _answers(plan)
    surface = _surface(
        plan,
        real_schemas,
        manifest(),
        growth=[refusal("service_unavailable", "retry_after_delay"), *answers["growth"]],
    )

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    first, second = _calls(surface, "growth")
    assert first == second
    assert first["anchor_key"] == ANCHOR_KEY
    assert len(_calls(surface, "resolve_anchor")) == 1


async def test_the_resolution_is_retried_and_reported_like_a_steps_call(
    rolling_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    plan = rolling_plan()
    surface = _surface(
        plan,
        real_schemas,
        manifest(),
        resolve_anchor=[ToolSurfaceUnavailableError("connection reset"), _anchor()],
    )
    heard: list[tuple[str, float]] = []

    outcome = await _run(
        plan,
        surface,
        jitter=lambda low, high: low,
        on_progress=lambda step_id, report: heard.append((step_id, report.progress)),
    )

    assert isinstance(outcome, Accepted), outcome
    assert len(_calls(surface, "resolve_anchor")) == 2
    # The retry and the answer are reported under the first rolling step,
    # ahead of that step's own call.
    assert heard[:4] == [("debt", 1), ("change", 2), ("change", 2), ("change", 1)]


async def test_a_resolution_that_never_answers_stops_the_run_before_any_rolling_step(
    rolling_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    plan = rolling_plan()
    surface = _surface(
        plan,
        real_schemas,
        manifest(),
        resolve_anchor=[ToolSurfaceUnavailableError("connection reset")],
    )

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.TRANSPORT_FAILED
    assert outcome.step_id == "change"
    assert len(_calls(surface, "resolve_anchor")) == MAX_ATTEMPTS_PER_STEP
    assert [record.step_id for record in outcome.completed] == ["debt"]
    assert not _calls(surface, "growth") and not _calls(surface, "filter")


# --- The surface check -----------------------------------------------------------------


@pytest.mark.parametrize(
    "drift",
    [
        None,
        {"parameters": {"type": "object"}},
        {"output_schema": {"type": "object", "properties": {"x": {}}}},
        {"version": "2"},
    ],
    ids=["missing", "arguments", "output_schema", "version"],
)
async def test_a_changed_resolve_anchor_stops_a_rolling_plan_before_any_call(
    rolling_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    drift: dict[str, Any] | None,
) -> None:
    drifted = [
        schema.model_copy(update=drift) if drift and schema.name == "resolve_anchor" else schema
        for schema in real_schemas
        if drift or schema.name != "resolve_anchor"
    ]
    plan = rolling_plan()

    outcome = await _run(plan, _surface(plan, drifted, manifest()))
    fixed = rolling_plan(rolling=False)
    untouched = await _run(fixed, _surface(fixed, drifted, manifest()))

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.SURFACE_CHANGED
    assert outcome.step_id == "change"
    assert outcome.completed == ()
    # A plan with no rolling step never calls the tool, so it runs.
    assert isinstance(untouched, Accepted), untouched


async def test_a_rolling_plan_refuses_a_capability_table_that_pins_no_resolve_anchor(
    rolling_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    plan = rolling_plan()
    surface = _surface(plan, real_schemas, manifest())
    table = [row for row in CAPABILITIES if row.tool.name != "resolve_anchor"]

    outcome = await _run(plan, surface, capabilities=table)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.SURFACE_CHANGED
    assert surface.calls == []


# --- Refusals ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("served", "named_years"),
    [
        (
            ToolRefusal.rolling_anchor_unresolved(
                detail="no_qualifying_period",
                reason="no period is held by half of every population",
            ),
            True,
        ),
        (
            ToolRefusal.rolling_grain_unsupported(
                reason="a root frame holds periods that are no calendar year", periods=["2023-01"]
            ),
            True,
        ),
        # Named years cannot help two sides that name no entity in common.
        (
            ToolRefusal.rolling_anchor_unresolved(
                detail="pairing_shares_no_key",
                reason="the two sides of the pairing share no key",
                context={"keys": {"reporter": "reporter"}},
            ),
            False,
        ),
    ],
    ids=["no_qualifying_period", "grain", "pairing_shares_no_key"],
)
async def test_a_newest_period_the_data_cannot_give_stops_the_run_with_words_for_the_user(
    rolling_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    served: ToolRefusal,
    named_years: bool,
) -> None:
    plan = rolling_plan()
    surface = _surface(plan, real_schemas, manifest(), resolve_anchor=[_served(served)])

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.STEP_REFUSED
    assert outcome.step_id == "change"
    assert outcome.payload is not None
    assert outcome.payload.code == served.code
    assert classify_refusal(outcome.payload) is Disposition.STOP
    # The tool's message tells a caller which tool to call next; the user
    # reads the run's own sentence and what to ask instead.
    assert outcome.payload.message != served.message
    assert "resolve_anchor" not in outcome.payload.message
    assert outcome.payload.fix["remedy"] not in outcome.payload.message
    assert (outcome.payload.fix["remedy"] == _NAMED_YEARS_REMEDY) is named_years
    assert outcome.payload.fix.items() >= served.fix.items()
    assert [record.step_id for record in outcome.completed] == ["debt"]
    assert not _calls(surface, "growth") and not _calls(surface, "filter")


@pytest.mark.parametrize(
    ("tool", "detail"),
    [
        ("resolve_anchor", "root_without_population"),
        ("growth", "literal_differs"),
        ("filter", "anchor_record_missing"),
    ],
)
async def test_a_refusal_of_the_runs_own_newest_period_is_an_internal_error(
    rolling_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    tool: str,
    detail: str,
) -> None:
    served = ToolRefusal.rolling_anchor_unresolved(detail=detail, reason="the record differs")
    plan = rolling_plan()
    surface = _surface(plan, real_schemas, manifest(), **{tool: [_served(served)]})

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.STEP_REFUSED
    assert outcome.step_id == ("newest" if tool == "filter" else "change")
    assert outcome.payload is not None
    assert outcome.payload.code == ExecutorCode.INTERNAL_ERROR
    assert classify_refusal(outcome.payload) is Disposition.STOP
    assert served.message not in outcome.payload.message
    refused = outcome.payload.fix["refusal"]
    assert isinstance(refused, dict)
    assert (refused["code"], refused["fix"]) == (served.code, {"detail": detail})


async def test_a_rolled_start_before_the_datasets_first_year_is_refused_before_the_verb(
    rolling_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    # Search states the dataset from 2010, and the retrieved series start in
    # 2015. A newest year of 2018 puts the five-year growth's start in 2013;
    # the filter keeps 2018, which is held.
    plan = rolling_plan(newest=2018, start=2010)
    surface = _surface(plan, real_schemas, manifest(), resolve_anchor=[_anchor("2018")])

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.ROLLING_WINDOW_OUTSIDE_DATASET
    assert outcome.step_id == "change"
    assert outcome.requirement_id == "annual_debt_service"
    assert not _calls(surface, "growth")
    assert outcome.payload is not None
    assert outcome.payload.code == ExecutorCode.ROLLING_WINDOW_OUTSIDE_DATASET
    assert classify_refusal(outcome.payload) is Disposition.STOP
    assert outcome.payload.fix == {
        "remedy": outcome.payload.fix["remedy"],
        "spec_id": SPEC_KENYA,
        "first_year": 2015,
        "window_start": 2013,
        "anchor": "2018",
    }
    for named in ("external debt service paid", "2015", "2013"):
        assert named in outcome.payload.message
    # The wave's other rolling step ran, and the run keeps what it resolved.
    assert [record.step_id for record in outcome.completed] == ["debt", "newest"]
    assert [anchor.period for anchor in outcome.anchors] == ["2018"]


async def test_a_rolled_start_on_the_datasets_first_year_runs(
    rolling_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    plan = rolling_plan(newest=2020)
    surface = _surface(plan, real_schemas, manifest(), resolve_anchor=[_anchor("2020")])

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    (growth,) = _calls(surface, "growth")
    assert growth["window_start"] == "2015"


@pytest.fixture
def plan_of(
    decision: Callable[..., Decision],
    source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
) -> PlanFactory:
    """Return a factory for an approved plan over the requirements and operations it is given.

    Each dataset requirement is discovered as the Kenya `source`;
    `first_years` states the first year on record of the datasets it names.
    """

    def _plan(
        requirements: list[Any],
        operations: list[Any],
        *,
        terminal: list[str],
        first_years: dict[str, int],
        decisions: tuple[Decision, ...] = (),
        asks: tuple[str, ...] = (),
        answered: tuple[AskAnswer, ...] = (),
        across_years: tuple[int, ...] = (),
    ) -> Plan:
        draft = Plan(
            question="How has Kenya's debt service changed?",
            decisions=[decision(), *decisions],
            evidence_requirements=requirements,
            asks=list(asks),
            answered=list(answered),
            asks_across_years=list(across_years),
            first_years=[FirstYear(requirement_id=r, year=y) for r, y in first_years.items()],
            status=PlanStatus.DRAFT,
        )
        # The plan runs as admission records it, with each whole-history read split off.
        split = split_of(draft)
        draft = Plan.model_validate(
            {
                **draft.model_dump(),
                "evidence_requirements": split.requirements,
                "rolling_copies": split.copies,
            }
        )
        translation = Translation(operations=operations, terminal=terminal)
        compiled = compile_plan(
            plan=draft,
            translation=translation,
            schemas={schema.name: schema for schema in real_schemas},
            discovered={
                r.id: source() for r in split.requirements if isinstance(r, DatasetRequirement)
            },
        )
        assert compiled.workflow is not None, compiled.issues
        return draft.with_translation(translation=translation, workflow=compiled.workflow).approve(
            at=datetime.now(UTC)
        )

    return _plan


def _series(manifest: dict[str, Any], *, first: str, latest: dict[str, str]) -> dict[str, Any]:
    """`manifest` holding one series of Kenya's from `first` and one per entity of `latest`, each ending in its year."""
    template = manifest["statistics"][0]
    manifest["statistics"][:] = [
        {
            **template,
            "entities": [{"role": "recipient", "ref": ref}],
            "first": {"value": 1.0, "period": first},
            "latest": {"value": 1.0, "period": year},
            "observations": int(year) - int(first) + 1,
        }
        for ref, year in latest.items()
    ]
    return manifest


async def test_a_lagging_anchor_leaves_a_rolling_filter_five_years_of_a_dataset_read_from_its_first_year(
    plan_of: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    requirement: Callable[..., DatasetRequirement],
) -> None:
    """Two of three series end in 2023, so the anchor is 2023, the year the filter
    is planned from, and the dataset's stated 2020-2024 would leave the filter 2020
    on. It reads the dataset from 1990 instead, through its own retrieval, so it
    keeps 2019-2023, and the total beside it keeps the stated years and reports the
    two series short of 2024."""
    since = (PeriodThreshold(op="gte", year=2019, rolling=RelativePeriod(offset=-4)),)
    total = AggregateRequirement(
        id="total",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        reduction=Reduction.SUM,
        group_by=(GroupBy.ENTITY,),
    )
    plan = plan_of(
        [
            requirement(period=Period(start=2020, end=2024), period_end_open=True),
            total,
            FilterRequirement(
                id="recent_years",
                decision_ids=["measure"],
                input_id="annual_debt_service",
                predicates=since,
            ),
        ],
        [
            RetrieveOperation(id="debt", requirement_id="annual_debt_service", spec_id=SPEC_KENYA),
            RetrieveOperation(
                id="debt_history",
                requirement_id="annual_debt_service_history",
                spec_id=SPEC_KENYA,
            ),
            AggregateOperation(
                id="sum",
                requirement_id="total",
                input="debt",
                reduction=Reduction.SUM,
                group_by=(GroupBy.ENTITY,),
            ),
            FilterOperation(
                id="recent", requirement_id="recent_years", input="debt_history", predicates=since
            ),
        ],
        terminal=["sum", "recent"],
        first_years={"annual_debt_service": 1990},
        asks=("the total debt service", "debt service over the last five years"),
        answered=(
            AskAnswer(ask=0, outputs=("total",)),
            AskAnswer(ask=1, outputs=("recent_years",)),
        ),
        across_years=(0,),
    )
    assert [(c.copy_id, c.origin_id) for c in plan.rolling_copies] == [
        ("annual_debt_service_history", "annual_debt_service")
    ]
    held = _series(
        manifest(),
        first="1990",
        latest={"country/KEN": "2024", "country/GHA": "2023", "country/UGA": "2023"},
    )
    surface = _surface(plan, real_schemas, held, resolve_anchor=[_anchor("2023")])

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Accepted), outcome
    (recent,) = _calls(surface, "filter")
    assert [predicate["value"] for predicate in recent["predicates"]] == ["2019"]
    short = {
        record.requirement_id: [gap.code.value for gap in record.gaps]
        for record in outcome.evidence
    }
    assert short["annual_debt_service"].count("window_short") == 2


async def test_an_answer_naming_no_newest_period_stops_the_run(
    rolling_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    plan = rolling_plan()
    surface = _surface(plan, real_schemas, manifest(), resolve_anchor=[ok({"period": "2023"})])

    outcome = await _run(plan, surface)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.STEP_RESULT_UNUSABLE
    assert outcome.step_id == "change"
    assert not _calls(surface, "growth") and not _calls(surface, "filter")


@pytest.mark.parametrize(
    ("bound", "op", "first"),
    [
        ("window_start", None, 2018),
        ("window_end", None, None),
        ("predicates/0", "gte", 2018),
        ("predicates/0", "eq", 2018),
        ("predicates/0", "gt", 2019),
        ("predicates/0", "lte", None),
        ("predicates/0", "lt", None),
        ("predicates/0", "ne", None),
    ],
)
def test_only_a_bound_that_sets_where_the_rows_start_names_a_first_year(
    bound: str, op: str | None, first: int | None
) -> None:
    rolled = RollingBound(bound=bound, offset=-5, period="2018", op=op)

    assert _first_year_kept(rolled) == first
