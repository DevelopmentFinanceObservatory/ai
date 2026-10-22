"""Tests for rolling periods in the compiler: the steps it writes and what a run fills in.

Every case compiles against the real `data-mcp` and `analysis-mcp` surface and
the production capability table, since the `rolling` argument is the tools' own
and a compiled step has to fit it as served. Where a step's arguments are
completed with a resolved period, analysis-mcp's own op-spec models read them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import pytest
from analysis_mcp.op_spec import FilterOpSpec, GrowthOpSpec, RankOpSpec, with_anchor_key
from analysis_mcp.server import mcp as analysis_mcp
from compiler.capabilities import CAPABILITIES, schema_drift
from compiler.compile import CompileIssueCode, CompileResult, compile_plan
from compiler.frame_columns import (
    FrameColumns,
    UnboundColumnError,
    bound_arguments,
    bound_populations,
    frame_columns,
)
from compiler.rolling import (
    ANCHOR_KEY_PARAMETER,
    AnchorGroup,
    PopulationsDifferError,
    RollingBound,
    anchor_groups,
    is_rolling,
    resolved_arguments,
    rolling_bounds,
)
from compiler.shapes import shapes_for
from compiler.sources import source_from_spec_block
from compiler.surface import McpSurface, ToolSchema
from data_mcp.server import mcp as data_mcp
from planner.operations import (
    CompareOperation,
    FilterOperation,
    GrowthOperation,
    Operation,
    RankOperation,
    Reduction,
    RetrieveOperation,
    Side,
    Translation,
)
from planner.plan import (
    AnswerMode,
    ComparisonRequirement,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    EvidenceRequirement,
    FilterRequirement,
    GrowthRequirement,
    Plan,
    PlanStatus,
    RankRequirement,
    RequirementSide,
)
from planner.records import (
    EntityIn,
    Period,
    PeriodThreshold,
    PlaceLevel,
    PlaceType,
    RankAmong,
    RankDirection,
    RelativePeriod,
    RollingWindow,
    ValueThreshold,
)
from planner.shapes import FIXED, EndDrift, PeriodDrift, StartDrift
from planner.workflow import OpenEntityAxis, SelectedSource, ToolId, Workflow, WorkflowStep
from pydantic import JsonValue

SPEC_DEBT = "spec_" + "a" * 32
SPEC_GDP = "spec_" + "b" * 32
FRAME = {"spec_id": "ops_" + "1" * 32, "content_version": "cv_" + "2" * 32}
ANCHOR_KEY = "rak_" + "0" * 32
FILTER = ToolId(server="analysis-mcp", name="filter")

LATEST = {"anchor": "latest", "offset": 0, "rule": "half1"}
FIVE_BACK = {"anchor": "latest", "offset": -5, "rule": "half1"}
KENYA_ALONE = [{"spec_id": SPEC_DEBT, "bindings": {"destination": ["country/KEN"]}}]

LAST_FIVE = RollingWindow(
    window_start=RelativePeriod(offset=-5), window_end=RelativePeriod(offset=0)
)

DecisionFactory = Callable[..., Decision]
RequirementFactory = Callable[..., DatasetRequirement]
SourceFactory = Callable[..., SelectedSource]
Compile = Callable[..., CompileResult]


@pytest.fixture
async def live() -> dict[str, ToolSchema]:
    surface = McpSurface(sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp})
    return {schema.name: schema for schema in await surface.list_tools()}


@pytest.fixture
def build(live: dict[str, ToolSchema], source: SourceFactory) -> Compile:
    """Compile against the live surface, with the open copy of Kenya's source discovered."""

    def _compile(
        requirements: Sequence[EvidenceRequirement],
        operations: Sequence[Operation],
        *,
        terminal: Sequence[str],
        decisions: Sequence[Decision],
        discovered: dict[str, SelectedSource] | None = None,
        answer_mode: AnswerMode = AnswerMode.COMPUTED,
    ) -> CompileResult:
        plan = Plan(
            question="How has Kenya's debt service changed over the last five years?",
            answer_mode=answer_mode,
            decisions=decisions,
            evidence_requirements=requirements,
            status=PlanStatus.DRAFT,
        )
        return compile_plan(
            plan=plan,
            translation=Translation(operations=tuple(operations), terminal=tuple(terminal)),
            schemas=live,
            discovered={"annual_debt_service": source()} if discovered is None else discovered,
            capabilities=CAPABILITIES,
        )

    return _compile


def _retrieve(**overrides: Any) -> RetrieveOperation:
    fields: dict[str, Any] = {
        "id": "debt",
        "requirement_id": "annual_debt_service",
        "spec_id": SPEC_DEBT,
    }
    return RetrieveOperation(**(fields | overrides))


def _growth_pair(
    *, input_id: str = "annual_debt_service", reads: str = "debt", **overrides: Any
) -> tuple[GrowthRequirement, GrowthOperation]:
    """A growth over the last five years, as a requirement and the operation carrying it out."""
    fields: dict[str, Any] = {"window_start": 2019, "window_end": 2024, "rolling": LAST_FIVE}
    fields |= overrides
    return (
        GrowthRequirement(
            id="growth_of_debt", decision_ids=["measure"], input_id=input_id, **fields
        ),
        GrowthOperation(id="change", requirement_id="growth_of_debt", input=reads, **fields),
    )


def _rank_pair(**overrides: Any) -> tuple[RankRequirement, RankOperation]:
    """A rank of the last five years' totals, as a requirement and its operation."""
    fields: dict[str, Any] = {
        "direction": RankDirection.HIGHEST_FIRST,
        "among": RankAmong.POPULATION,
        "reduction": Reduction.SUM,
        "window_start": 2019,
        "window_end": 2024,
        "rolling": LAST_FIVE,
    } | overrides
    return (
        RankRequirement(
            id="ranked",
            decision_ids=["measure", "population"],
            input_id="annual_debt_service",
            **fields,
        ),
        RankOperation(id="ranked", requirement_id="ranked", input="debt", **fields),
    )


def _population_decision(decision: DecisionFactory) -> Decision:
    return decision(slot="population", fixes=DecisionFixes(among=RankAmong.POPULATION))


def _newest_year_pair(
    *, input_id: str = "annual_debt_service", reads: str = "debt", extra: tuple[Any, ...] = ()
) -> tuple[FilterRequirement, FilterOperation]:
    """A filter keeping the newest year, as a requirement and its operation."""
    predicates = (PeriodThreshold(op="eq", year=2024, rolling=RelativePeriod(offset=0)), *extra)
    return (
        FilterRequirement(
            id="newest", decision_ids=["measure"], input_id=input_id, predicates=predicates
        ),
        FilterOperation(id="newest", requirement_id="newest", input=reads, predicates=predicates),
    )


def _step(result: CompileResult, step_id: str) -> WorkflowStep:
    assert result.ok, result.issues
    assert result.workflow is not None
    return next(step for step in result.workflow.steps if step.id == step_id)


def _codes(result: CompileResult) -> list[CompileIssueCode]:
    return [issue.code for issue in result.issues]


# --- What a rolling step carries ----------------------------------------------------


def test_a_rolling_growth_keeps_its_planned_years_and_adds_the_overlay(
    build: Compile,
    decision: DecisionFactory,
    requirement: RequirementFactory,
    live: dict[str, ToolSchema],
) -> None:
    growth_req, change = _growth_pair()

    result = build(
        [requirement(period_end_open=True), growth_req],
        [_retrieve(), change],
        terminal=["change"],
        decisions=[decision()],
    )

    step = _step(result, "change")
    assert step.args == {
        "window_start": "2019",
        "window_end": "2024",
        "entity_by": ["destination"],
        "rolling": {"window_start": FIVE_BACK, "window_end": LATEST, "population": KENYA_ALONE},
    }
    # The retrieval is of the open copy as approved: no scope is written.
    assert _step(result, "debt").args == {"spec_id": SPEC_DEBT, "without_party_totals": True}
    # The key names a resolution that exists only once the retrievals have run.
    assert ANCHOR_KEY_PARAMETER not in step.args
    assert is_rolling(step)
    assert step.input_schema_hash == live["growth"].input_schema_hash
    assert schema_drift(CAPABILITIES, live) == []


def test_a_window_rolling_at_its_end_alone_writes_one_reference(
    build: Compile, decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    since_2015 = RollingWindow(window_end=RelativePeriod(offset=0))
    growth_req, change = _growth_pair(window_start=2015, rolling=since_2015)

    result = build(
        [requirement(period_end_open=True), growth_req],
        [_retrieve(), change],
        terminal=["change"],
        decisions=[decision()],
    )

    step = _step(result, "change")
    assert step.args["window_start"] == "2015"
    assert step.args["rolling"] == {"window_end": LATEST, "population": KENYA_ALONE}


def test_a_rolling_rank_keeps_its_planned_years_and_adds_the_overlay(
    build: Compile,
    decision: DecisionFactory,
    requirement: RequirementFactory,
    wide_source: SourceFactory,
) -> None:
    two = requirement(
        entities=[Entity(name="Kenya", role="debtor"), Entity(name="Ghana", role="debtor")],
        period_end_open=True,
    )
    ranked_req, ranked = _rank_pair()

    result = build(
        [two, ranked_req],
        [_retrieve(), ranked],
        terminal=["ranked"],
        decisions=[decision(), _population_decision(decision)],
        discovered={"annual_debt_service": wide_source()},
    )

    step = _step(result, "ranked")
    assert step.args["window_start"] == "2019"
    assert step.args["window_end"] == "2024"
    assert step.args["rolling"] == {
        "window_start": FIVE_BACK,
        "window_end": LATEST,
        "population": [
            {"spec_id": SPEC_DEBT, "bindings": {"destination": ["country/GHA", "country/KEN"]}}
        ],
    }
    assert ANCHOR_KEY_PARAMETER not in step.args


def test_a_rolling_filter_states_the_reference_on_its_period_threshold_alone(
    build: Compile, decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    newest_req, newest = _newest_year_pair(extra=(ValueThreshold(op="gt", value=0.0),))

    result = build(
        [requirement(period_end_open=True), newest_req],
        [_retrieve(), newest],
        terminal=["newest"],
        decisions=[decision()],
    )

    step = _step(result, "newest")
    assert step.args == {
        "predicates": [
            {
                "kind": "ordered_threshold",
                "column": "time_period",
                "op": "eq",
                "value": "2024",
                "rolling": LATEST,
            },
            {"kind": "ordered_threshold", "column": "value", "op": "gt", "value": 0.0},
        ],
        "rolling": {"population": KENYA_ALONE},
    }


def test_a_fixed_plan_names_no_rolling_step_and_no_ordering(
    build: Compile, decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    growth_req, change = _growth_pair(rolling=None)

    result = build(
        [requirement(), growth_req],
        [_retrieve(), change],
        terminal=["change"],
        decisions=[decision()],
    )

    step = _step(result, "change")
    assert step.args == {"window_start": "2019", "window_end": "2024", "entity_by": ["destination"]}
    assert not is_rolling(step)
    assert result.workflow is not None
    assert anchor_groups(result.workflow) == ()


def test_a_rolling_operation_resolves_the_shape_a_fixed_one_does(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    def resolved(rolling: RollingWindow | None) -> Any:
        _growth_req, change = _growth_pair(rolling=rolling)
        shapes, reasons = shapes_for(
            {"debt": _retrieve(), "change": change},
            requirements={"annual_debt_service": requirement(period_end_open=rolling is not None)},
            sources={"debt": source()},
            order=["debt", "change"],
        )
        assert reasons == []
        return shapes["change"].shape

    rolling, fixed = resolved(LAST_FIVE), resolved(None)

    assert replace(rolling, drift=FIXED) == fixed
    assert rolling.drift == PeriodDrift(
        start=StartDrift.ANCHOR, start_offset=-5, end=EndDrift.ANCHOR, end_offset=0
    )


# --- The population ------------------------------------------------------------------


def test_a_rolling_steps_population_and_roots_are_the_retrievals_it_reads(
    build: Compile,
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    wide_source: SourceFactory,
) -> None:
    # Debt service names Kenya and Ghana and is narrowed to Kenya before the
    # growth; GDP is retrieved beside it and read by no rolling step. No step
    # reads both rolling steps, so each stands on a period of its own.
    debt = requirement(
        entities=[Entity(name="Kenya", role="debtor"), Entity(name="Ghana", role="debtor")],
        period_end_open=True,
    )
    gdp = requirement(id="annual_gdp", measure="gross domestic product")
    kenya = (EntityIn(names=("Kenya",)),)
    kenya_req = FilterRequirement(
        id="kenya", decision_ids=["measure"], input_id="annual_debt_service", predicates=kenya
    )
    kenya_op = FilterOperation(id="kenya", requirement_id="kenya", input="debt", predicates=kenya)
    newest_req, newest = _newest_year_pair(input_id="kenya", reads="kenya")
    growth_req, change = _growth_pair(input_id="kenya", reads="kenya")

    result = build(
        [debt, gdp, kenya_req, newest_req, growth_req],
        [
            _retrieve(),
            _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_GDP),
            kenya_op,
            newest,
            change,
        ],
        terminal=["change", "newest", "gdp"],
        decisions=[decision()],
        discovered={
            "annual_debt_service": wide_source(),
            "annual_gdp": source(spec_id=SPEC_GDP),
        },
        answer_mode=AnswerMode.SERIES,
    )

    assert result.ok, result.issues
    workflow = result.workflow
    assert workflow is not None
    expected = [{"spec_id": SPEC_DEBT, "bindings": {"destination": ["country/KEN"]}}]
    assert _step(result, "change").args["rolling"]["population"] == expected
    assert _step(result, "newest").args["rolling"] == {"population": expected}
    assert sorted(anchor_groups(workflow), key=lambda group: group.steps) == [
        AnchorGroup(steps=("change",), roots=("debt",), population=tuple(expected)),
        AnchorGroup(steps=("newest",), roots=("debt",), population=tuple(expected)),
    ]


def test_rolling_steps_one_step_reads_together_stand_on_one_period_over_both_datasets(
    build: Compile,
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    # The newest year of debt service and of GDP, compared: the comparison
    # reads both rolling filters, so one period is taken over both datasets.
    debt = requirement(period_end_open=True)
    gdp = requirement(id="annual_gdp", measure="gross domestic product", period_end_open=True)
    debt_req, debt_newest = _newest_year_pair()
    gdp_req, gdp_newest = _newest_year_pair(input_id="annual_gdp", reads="gdp")
    gdp_req = gdp_req.model_copy(update={"id": "gdp_newest"})
    gdp_newest = gdp_newest.model_copy(update={"id": "gdp_newest", "requirement_id": "gdp_newest"})
    gap_req = ComparisonRequirement(
        id="gap",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="gdp_newest", entity="Kenya"),
        subtrahend=RequirementSide(input_id="newest", entity="Kenya"),
    )
    gap = CompareOperation(
        id="gap",
        requirement_id="gap",
        left=Side(input="newest", entity="Kenya"),
        right=Side(input="gdp_newest", entity="Kenya"),
    )

    result = build(
        [debt, gdp, debt_req, gdp_req, gap_req],
        [
            _retrieve(),
            _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_GDP),
            debt_newest,
            gdp_newest,
            gap,
        ],
        terminal=["gap"],
        decisions=[decision()],
        discovered={
            "annual_debt_service": source(),
            "annual_gdp": source(spec_id=SPEC_GDP),
        },
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    (group,) = anchor_groups(result.workflow)
    assert sorted(group.steps) == ["gdp_newest", "newest"]
    assert sorted(group.roots) == ["debt", "gdp"]
    assert {population["spec_id"] for population in group.population} == {SPEC_DEBT, SPEC_GDP}
    assert (
        _step(result, "newest").args["rolling"]
        == _step(result, "gdp_newest").args["rolling"]
        == {"population": list(group.population)}
    )


def test_a_later_verbs_entity_binding_narrows_the_side_it_reads(
    build: Compile,
    decision: DecisionFactory,
    requirement: RequirementFactory,
    wide_source: SourceFactory,
) -> None:
    two = requirement(
        entities=[Entity(name="Kenya", role="debtor"), Entity(name="Ghana", role="debtor")],
        period_end_open=True,
    )
    newest_req, newest = _newest_year_pair()
    gap_req = ComparisonRequirement(
        id="gap",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="newest", entity="Ghana"),
        subtrahend=RequirementSide(input_id="newest", entity="Kenya"),
    )
    gap = CompareOperation(
        id="gap",
        requirement_id="gap",
        left=Side(input="newest", entity="Kenya"),
        right=Side(input="newest", entity="Ghana"),
    )

    result = build(
        [two, newest_req, gap_req],
        [_retrieve(), newest, gap],
        terminal=["gap"],
        decisions=[decision()],
        discovered={"annual_debt_service": wide_source()},
    )

    # One population for each side of the comparison, over the one dataset.
    assert _step(result, "newest").args["rolling"] == {
        "population": [
            {"spec_id": SPEC_DEBT, "bindings": {"destination": ["country/GHA"]}},
            {"spec_id": SPEC_DEBT, "bindings": {"destination": ["country/KEN"]}},
        ]
    }


def test_a_place_level_filter_holds_the_population_to_that_level(
    build: Compile, decision: DecisionFactory, place_scope_block: dict[str, Any]
) -> None:
    # Health spending in every country: the source leaves its places open, so
    # the population is every series at the level the filter keeps.
    open_place = source_from_spec_block(place_scope_block)
    health = DatasetRequirement(
        id="health",
        decision_ids=["measure"],
        measure="health spending per person",
        across=("country",),
        period=Period(start=2015, end=2024),
        period_end_open=True,
    )
    newest_req, newest = _newest_year_pair(
        input_id="health", reads="fetch", extra=(PlaceLevel(level=PlaceType.COUNTRY),)
    )

    result = build(
        [health, newest_req],
        [
            RetrieveOperation(id="fetch", requirement_id="health", spec_id=open_place.spec_id),
            newest,
        ],
        terminal=["newest"],
        decisions=[decision()],
        discovered={"health": open_place},
    )

    assert _step(result, "fetch").args["place_types"] is True
    assert _step(result, "newest").args["rolling"] == {
        "population": [
            {
                "spec_id": open_place.spec_id,
                "bindings": {},
                "levels": {"observationAbout_type": "Country"},
            }
        ]
    }


def test_a_rolling_rank_over_a_strict_country_filter_holds_the_population_to_countries(
    build: Compile, decision: DecisionFactory, source: SourceFactory
) -> None:
    # Every recipient's debt service, kept to the countries, ranked over the
    # last five years. The strict level reaches the rank's population, which
    # analysis-mcp's op-spec rule reads at compile time
    # (`compiler.arguments.op_spec_errors`): a strict level names a level column.
    every_recipient = source(
        entities=(), open_entity_axes=(OpenEntityAxis(column="destination", member_count=134),)
    )
    debt = DatasetRequirement(
        id="annual_debt_service",
        decision_ids=["measure"],
        measure="external debt service",
        across=("recipient",),
        period=Period(start=2015, end=2024),
        period_end_open=True,
    )
    predicates = (PlaceLevel(level=PlaceType.COUNTRY, role="recipient", strict=True),)
    countries_req = FilterRequirement(
        id="countries",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        predicates=predicates,
    )
    countries = FilterOperation(
        id="countries", requirement_id="countries", input="debt", predicates=predicates
    )
    ranked_req, ranked = _rank_pair(axis="recipient")
    ranked_req = ranked_req.model_copy(update={"input_id": "countries"})
    ranked = ranked.model_copy(update={"input": "countries"})

    result = build(
        [debt, countries_req, ranked_req],
        [_retrieve(spec_id=every_recipient.spec_id), countries, ranked],
        terminal=["ranked"],
        decisions=[decision(), _population_decision(decision)],
        discovered={"annual_debt_service": every_recipient},
    )

    assert _step(result, "ranked").args["rolling"]["population"] == [
        {
            "spec_id": every_recipient.spec_id,
            "bindings": {},
            "levels": {"destination_type": "Country"},
            "strict_levels": ["destination_type"],
        }
    ]


# --- Issues --------------------------------------------------------------------------


def test_a_rolling_operation_over_a_dataset_with_a_fixed_end_is_refused(
    build: Compile, decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # The plan's own rule refuses a rolling requirement over a closed dataset,
    # so the operation rolls where its requirement does not.
    growth_req, _ = _growth_pair(rolling=None)
    _, change = _growth_pair()

    result = build(
        [requirement(), growth_req],
        [_retrieve(), change],
        terminal=["change"],
        decisions=[decision()],
    )

    (issue,) = [
        issue for issue in result.issues if issue.code is CompileIssueCode.ROLLING_NEEDS_OPEN_PERIOD
    ]
    assert (issue.operation_id, issue.requirement_id) == ("change", "annual_debt_service")


@pytest.mark.parametrize(
    ("window", "refused"),
    [
        ({"kind": "range", "start_year": 2015, "end_year": 2026}, True),
        ({"kind": "range", "start_year": 2015}, False),
        ({"kind": "all"}, False),
    ],
    ids=["searched-with-an-end-year", "searched-with-no-end-year", "every-year"],
)
def test_a_rolling_operation_over_a_source_searched_with_an_end_year_is_refused(
    build: Compile,
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    window: dict[str, Any],
    refused: bool,
) -> None:
    # The requirement is open from 2015; Search answered "since 2015" with the
    # source's identity as stated.
    growth_req, change = _growth_pair()
    searched = source()
    searched = searched.model_copy(
        update={"identity": {**(searched.identity or {}), "requested_window": window}}
    )

    result = build(
        [requirement(period_end_open=True), growth_req],
        [_retrieve(), change],
        terminal=["change"],
        decisions=[decision()],
        discovered={"annual_debt_service": searched},
    )

    found = [
        (issue.operation_id, issue.requirement_id)
        for issue in result.issues
        if issue.code is CompileIssueCode.ROLLING_NEEDS_OPEN_PERIOD
    ]
    assert found == ([("change", "annual_debt_service")] if refused else [])


def test_chains_that_keep_no_series_of_a_retrieval_leave_no_population(
    build: Compile,
    decision: DecisionFactory,
    requirement: RequirementFactory,
    wide_source: SourceFactory,
) -> None:
    # The filter keeps Kenya and the side reading it names Ghana.
    two = requirement(
        entities=[Entity(name="Kenya", role="debtor"), Entity(name="Ghana", role="debtor")],
        period_end_open=True,
    )
    newest_req, newest = _newest_year_pair(extra=(EntityIn(names=("Kenya",)),))
    gap_req = ComparisonRequirement(
        id="gap",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="annual_debt_service", entity="Kenya"),
        subtrahend=RequirementSide(input_id="newest", entity="Ghana"),
    )
    gap = CompareOperation(
        id="gap",
        requirement_id="gap",
        left=Side(input="newest", entity="Ghana"),
        right=Side(input="debt", entity="Kenya"),
    )

    result = build(
        [two, newest_req, gap_req],
        [_retrieve(), newest, gap],
        terminal=["gap"],
        decisions=[decision()],
        discovered={"annual_debt_service": wide_source()},
    )

    assert _codes(result) == [CompileIssueCode.ROLLING_POPULATION_UNUSABLE]
    assert result.issues[0].requirement_id == "annual_debt_service"


def test_rolling_steps_carrying_different_populations_are_a_fault(
    build: Compile, decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # The growth reads a rolling filter of the last six years, so the two stand
    # on one period.
    since = (PeriodThreshold(op="gte", year=2019, rolling=RelativePeriod(offset=-5)),)
    newest_req = FilterRequirement(
        id="newest", decision_ids=["measure"], input_id="annual_debt_service", predicates=since
    )
    newest = FilterOperation(id="newest", requirement_id="newest", input="debt", predicates=since)
    growth_req, change = _growth_pair(input_id="newest", reads="newest")
    result = build(
        [requirement(period_end_open=True), newest_req, growth_req],
        [_retrieve(), newest, change],
        terminal=["change"],
        decisions=[decision()],
    )
    assert result.workflow is not None
    (group,) = anchor_groups(result.workflow)
    assert sorted(group.steps) == ["change", "newest"]
    other = [{"spec_id": SPEC_GDP, "bindings": {}}]
    tampered = Workflow.model_validate(
        result.workflow.model_dump()
        | {
            "steps": [
                step.model_dump()
                | (
                    {"args": step.args | {"rolling": {"population": other}}}
                    if step.id == "newest"
                    else {}
                )
                for step in result.workflow.steps
            ]
        }
    )

    with pytest.raises(PopulationsDifferError):
        anchor_groups(tampered)


# --- What a run fills in ---------------------------------------------------------------


def _family_frame() -> FrameColumns:
    return frame_columns(
        role_keys=("financingRecipient", "financingProvider"),
        axis_by_column={"financingProvider": "origin", "financingRecipient": "destination"},
        compiled=("origin", "destination"),
    )


def _run_arguments(step: WorkflowStep, *, period: str) -> dict[str, JsonValue]:
    """`step`'s arguments as a run sends them: bound to the frames, then resolved at `period`."""
    bound = bound_arguments(
        step.tool,
        {**step.args, "frame_ref": FRAME},
        frames={"frame_ref": _family_frame()},
        roots={SPEC_DEBT: [_family_frame()]},
    )
    return resolved_arguments(bound, period=period, anchor_key=ANCHOR_KEY)


def test_a_resolved_growth_is_one_analysis_mcp_accepts(
    build: Compile, decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    growth_req, change = _growth_pair()
    result = build(
        [requirement(period_end_open=True), growth_req],
        [_retrieve(), change],
        terminal=["change"],
        decisions=[decision()],
    )

    arguments = _run_arguments(_step(result, "change"), period="2025")

    assert (arguments["window_start"], arguments["window_end"]) == ("2020", "2025")
    key = arguments.pop(ANCHOR_KEY_PARAMETER)
    assert key == ANCHOR_KEY
    op_spec = with_anchor_key(GrowthOpSpec.model_validate(arguments), key)
    assert op_spec.entity_by == ("financingRecipient",)
    assert op_spec.rolling is not None
    # The population names the retrieved frame's column, in the tool's own spelling.
    assert [p.model_dump() for p in op_spec.rolling.population] == [
        {"spec_id": SPEC_DEBT, "bindings": {"financingRecipient": ("country/KEN",)}}
    ]
    assert arguments["rolling"]["population"] == [
        {"spec_id": SPEC_DEBT, "bindings": {"financingRecipient": ["country/KEN"]}}
    ]


def test_a_resolved_rank_and_filter_are_ones_analysis_mcp_accepts(
    build: Compile,
    decision: DecisionFactory,
    requirement: RequirementFactory,
) -> None:
    ranked_req, ranked = _rank_pair()
    newest_req, newest = _newest_year_pair()
    result = build(
        [requirement(period_end_open=True), ranked_req, newest_req],
        [_retrieve(), ranked, newest],
        terminal=["ranked", "newest"],
        decisions=[decision(), _population_decision(decision)],
    )

    rank_arguments = _run_arguments(_step(result, "ranked"), period="2025")
    filter_arguments = _run_arguments(_step(result, "newest"), period="2025")

    assert rolling_bounds(rank_arguments) == (
        RollingBound(bound="window_start", offset=-5, period="2020", op=None),
        RollingBound(bound="window_end", offset=0, period="2025", op=None),
    )
    assert rolling_bounds(filter_arguments) == (
        RollingBound(bound="predicates/0", offset=0, period="2025", op="eq"),
    )
    # As compiled, the literals are the planned years.
    assert [b.period for b in rolling_bounds(_step(result, "ranked").args)] == ["2019", "2024"]
    assert rolling_bounds(_step(result, "debt").args) == ()
    rank_key = rank_arguments.pop(ANCHOR_KEY_PARAMETER)
    filter_key = filter_arguments.pop(ANCHOR_KEY_PARAMETER)
    rank_spec = with_anchor_key(RankOpSpec.model_validate(rank_arguments), rank_key)
    filter_spec = with_anchor_key(FilterOpSpec.model_validate(filter_arguments), filter_key)
    assert (rank_spec.window_start, rank_spec.window_end) == ("2020", "2025")
    assert [p.model_dump(exclude_none=True) for p in filter_spec.predicates] == [
        {
            "kind": "ordered_threshold",
            "column": "time_period",
            "op": "eq",
            "value": "2025",
            "rolling": LATEST,
        }
    ]
    assert rank_spec.rolling is not None and filter_spec.rolling is not None
    assert rank_spec.rolling.population == filter_spec.rolling.population


def test_a_rank_of_the_newest_year_alone_is_a_call_analysis_mcp_accepts(
    build: Compile,
    decision: DecisionFactory,
    requirement: RequirementFactory,
) -> None:
    # The input holds 2015-2024 and the window keeps one of them, so each
    # entity has one row and the call states no aggregation.
    newest = RollingWindow(
        window_start=RelativePeriod(offset=0), window_end=RelativePeriod(offset=0)
    )
    ranked_req, ranked = _rank_pair(
        reduction=None, window_start=2024, window_end=2024, rolling=newest
    )
    result = build(
        [requirement(period_end_open=True), ranked_req],
        [_retrieve(), ranked],
        terminal=["ranked"],
        decisions=[decision(), _population_decision(decision)],
    )

    step = _step(result, "ranked")
    assert (step.args["window_start"], step.args["window_end"]) == ("2024", "2024")
    assert "aggregation" not in step.args
    arguments = _run_arguments(step, period="2025")
    key = arguments.pop(ANCHOR_KEY_PARAMETER)
    spec = with_anchor_key(RankOpSpec.model_validate(arguments), key)
    assert (spec.window_start, spec.window_end, spec.aggregation) == ("2025", "2025", None)


def test_a_compiled_rolling_step_alone_is_not_a_call_analysis_mcp_accepts(
    build: Compile, decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # Without its key a rolling step is refused by the tool, so nothing the
    # compiler writes can run before the newest period is resolved.
    growth_req, change = _growth_pair()
    result = build(
        [requirement(period_end_open=True), growth_req],
        [_retrieve(), change],
        terminal=["change"],
        decisions=[decision()],
    )
    op_spec = GrowthOpSpec.model_validate({**_step(result, "change").args, "frame_ref": FRAME})

    with pytest.raises(ValueError, match="anchor_key"):
        with_anchor_key(op_spec, None)


# --- Spelling a population the way the retrieved frames do ------------------------------

GDP_ALONE = {"spec_id": SPEC_GDP, "bindings": {"observationAbout": ["country/KEN"]}}


def _plain_frame() -> FrameColumns:
    return frame_columns(
        role_keys=("observationAbout",),
        axis_by_column={"observationAbout": None},
        compiled=("observationAbout",),
    )


def _rolling_filter_arguments() -> dict[str, JsonValue]:
    """A rolling filter over the GDP frame, in a workflow that also retrieves debt service."""
    return {
        "frame_ref": FRAME,
        "predicates": [
            {"kind": "category_match", "column": "observationAbout", "values": ["country/KEN"]},
            {
                "kind": "ordered_threshold",
                "column": "time_period",
                "op": "eq",
                "value": "2024",
                "rolling": LATEST,
            },
        ],
        "rolling": {"population": [*KENYA_ALONE, GDP_ALONE]},
    }


def test_a_population_is_spelled_by_its_own_retrieval_not_by_the_frame_the_step_reads() -> None:
    # The step reads the GDP frame, which carries no column on the
    # `destination` axis; the debt entry is spelled by the debt frame.
    bound = bound_arguments(
        FILTER,
        _rolling_filter_arguments(),
        frames={"frame_ref": _plain_frame()},
        roots={SPEC_DEBT: [_family_frame()], SPEC_GDP: [_plain_frame()]},
    )

    assert bound["rolling"] == {
        "population": [
            {"spec_id": SPEC_DEBT, "bindings": {"financingRecipient": ["country/KEN"]}},
            GDP_ALONE,
        ]
    }
    # The predicates keep their references and are spelled by the frame read.
    assert bound["predicates"] == _rolling_filter_arguments()["predicates"]


def test_a_rolling_step_given_no_retrieved_frames_is_refused() -> None:
    with pytest.raises(UnboundColumnError, match="rolling"):
        bound_arguments(FILTER, _rolling_filter_arguments(), frames={"frame_ref": _plain_frame()})


def test_a_population_over_a_dataset_no_retrieval_fetched_is_refused() -> None:
    with pytest.raises(UnboundColumnError, match=SPEC_DEBT):
        bound_populations([*KENYA_ALONE, GDP_ALONE], roots={SPEC_GDP: [_plain_frame()]})


def test_a_population_naming_a_role_its_frame_does_not_carry_is_refused() -> None:
    unbound = frame_columns(
        role_keys=("financingProvider",),
        axis_by_column={"financingProvider": "origin"},
        compiled=("origin", "destination"),
    )

    with pytest.raises(UnboundColumnError, match="'destination' role"):
        bound_populations(KENYA_ALONE, roots={SPEC_DEBT: [unbound]})


def test_two_retrievals_of_one_dataset_spelling_a_role_differently_are_refused() -> None:
    with pytest.raises(UnboundColumnError, match="'destination' role"):
        bound_populations(KENYA_ALONE, roots={SPEC_DEBT: [_family_frame(), None]})


def test_a_respelled_population_is_put_back_in_canonical_order() -> None:
    # `origin` sorts after `destination` as compiled, and `financingProvider`
    # before `financingRecipient` as retrieved.
    both: list[JsonValue] = [
        {
            "spec_id": SPEC_DEBT,
            "bindings": {"destination": ["country/KEN"], "origin": ["country/FRA"]},
        }
    ]

    (bound,) = bound_populations(both, roots={SPEC_DEBT: [_family_frame()]})

    assert bound == {
        "spec_id": SPEC_DEBT,
        "bindings": {
            "financingProvider": ["country/FRA"],
            "financingRecipient": ["country/KEN"],
        },
    }
    assert isinstance(bound, dict) and isinstance(bindings := bound["bindings"], dict)
    assert list(bindings) == ["financingProvider", "financingRecipient"]
