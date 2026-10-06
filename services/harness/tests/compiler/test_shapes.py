"""Tests for `compiler.shapes`: the compiler's own driver for the frame-shape algebra.

`planner.shapes` is exercised directly in `tests/planner/test_shapes.py`; this
file covers what `compiler.shapes` adds on top of it -- wrapping each role
dimension with the literal source column behind it (`ResolvedRole`), and
`shapes_for`'s walk over a translation's own operations, sources and
requirements rather than over a plan's requirements. Each case drives
`shapes_for` directly, the module's one public entry point, the same way
`tests/compiler/test_compile.py` drives only `compile_plan`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest
from compiler.shapes import ResolvedShape, shapes_for
from compiler.sources import member_digest, source_from_spec_block
from planner.operations import (
    AggregateOperation,
    CompareOperation,
    FilterOperation,
    GroupBy,
    GrowthOperation,
    JoinOperation,
    Operation,
    RankOperation,
    Reduction,
    RetrieveOperation,
    ShareOfOperation,
    Side,
)
from planner.plan import AggregateRequirement, DatasetRequirement, Entity, EvidenceRequirement
from planner.records import (
    DenominatorScope,
    EntityIn,
    GrowthChange,
    PeriodThreshold,
    PlaceLevel,
    PlaceType,
    RankAmong,
    RankDirection,
)
from planner.workflow import OpenEntityAxis, Period, ResolvedEntity, SelectedSource

SPEC_KENYA = "spec_" + "a" * 32
SPEC_OTHER = "spec_" + "b" * 32

RequirementFactory = Callable[..., DatasetRequirement]
SourceFactory = Callable[..., SelectedSource]


def _retrieve(**overrides: object) -> RetrieveOperation:
    fields: dict[str, object] = {
        "id": "debt",
        "requirement_id": "annual_debt_service",
        "spec_id": SPEC_KENYA,
    }
    return RetrieveOperation(**(fields | overrides))  # type: ignore[arg-type]


def _resolve(
    *operations: Operation,
    requirements: Mapping[str, EvidenceRequirement],
    sources: Mapping[str, SelectedSource],
) -> tuple[dict[str, ResolvedShape], list[str]]:
    """`shapes_for` over `operations`, in the order given -- every fixture here already lists them upstream-first."""
    return shapes_for(
        {op.id: op for op in operations},
        requirements=requirements,
        sources=sources,
        order=[op.id for op in operations],
    )


# --- retrieve: `ResolvedRole` wraps each column with its column, phrase and surplus ---


def test_retrieve_wraps_the_requirement_entities_role_with_its_column_and_narrowed_false(
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    req = requirement()
    src = source()
    shapes, issues = _resolve(
        _retrieve(), requirements={"annual_debt_service": req}, sources={"debt": src}
    )

    assert issues == []
    resolved = shapes["debt"]
    assert len(resolved.columns) == 1
    (role,) = resolved.columns
    assert role.column == "destination"
    assert role.role == "debtor"  # the requirement's own phrase, not the source's column name
    assert role.requirement_id == "annual_debt_service"
    assert role.spec_id == src.spec_id
    assert role.narrowed is False
    assert role.surplus is False  # `source()` resolves only the named entity, Kenya
    assert resolved.shape.roles == (resolved.shape.roles[0],)  # one role dimension
    assert resolved.shape.roles[0].role == "debtor"
    assert resolved.shape.roles[0].requirement_id == "annual_debt_service"


def test_retrieve_marks_a_column_surplus_when_the_source_resolves_an_unnamed_label(
    requirement: RequirementFactory,
    wide_source: SourceFactory,
) -> None:
    req = requirement()  # names Kenya alone
    src = wide_source()  # resolves Kenya and Ghana under the same column

    shapes, issues = _resolve(
        _retrieve(), requirements={"annual_debt_service": req}, sources={"debt": src}
    )

    assert issues == []
    (role,) = shapes["debt"].columns
    assert role.surplus is True


def test_retrieve_with_no_selected_source_or_no_dataset_requirement_resolves_nothing(
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    req = requirement()
    src = source()

    # No source recorded for the retrieval.
    shapes, issues = _resolve(_retrieve(), requirements={"annual_debt_service": req}, sources={})
    assert shapes == {}
    assert issues == []

    # The requirement it names is not a `DatasetRequirement`.
    not_a_dataset = AggregateRequirement(
        id="not_a_dataset",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    shapes, issues = _resolve(
        _retrieve(requirement_id="not_a_dataset"),
        requirements={"not_a_dataset": not_a_dataset},
        sources={"debt": src},
    )
    assert shapes == {}
    assert issues == []


def test_retrieve_wraps_the_across_party_with_the_open_place_axis_column(
    requirement: RequirementFactory,
    place_scope_block: dict[str, Any],
) -> None:
    req = requirement(measure="health spending per person", entities=[], across="country")
    src = source_from_spec_block(place_scope_block)

    shapes, issues = _resolve(
        _retrieve(spec_id=src.spec_id),
        requirements={"annual_debt_service": req},
        sources={"debt": src},
    )

    assert issues == []
    (role,) = shapes["debt"].columns
    assert (role.column, role.role, role.surplus, role.narrowed) == (
        "observationAbout",
        "country",
        False,
        False,
    )
    assert [dim.role for dim in shapes["debt"].shape.roles] == ["country"]


def test_retrieve_keeps_the_named_entitys_column_then_the_open_axis_its_across_covers(
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    # Education aid to Kenya from every donor: Kenya binds `destination`,
    # and the donors are the source's open `origin` axis.
    req = requirement(entities=[Entity(name="Kenya", role="recipient")], across="donor")
    src = source(
        entities=(ResolvedEntity(role="destination", ref="country/KEN", label="Kenya"),),
        open_entity_axes=(OpenEntityAxis(column="origin", member_count=55),),
    )

    shapes, issues = _resolve(
        _retrieve(), requirements={"annual_debt_service": req}, sources={"debt": src}
    )

    assert issues == []
    assert [(r.column, r.role, r.surplus) for r in shapes["debt"].columns] == [
        ("destination", "recipient", False),
        ("origin", "donor", False),
    ]


def test_retrieve_builds_no_shape_when_no_open_axis_holds_the_across_party(
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    # Every donor, against a source whose one open axis is the receiving
    # side: no column holds the donors, and the source's own evidence
    # finding is what reports it.
    req = requirement(entities=[], across="donor")
    src = source(
        entities=(), open_entity_axes=(OpenEntityAxis(column="destination", member_count=30),)
    )

    shapes, issues = _resolve(
        _retrieve(), requirements={"annual_debt_service": req}, sources={"debt": src}
    )

    assert shapes == {}
    assert issues == []


# --- aggregate: `_carry_columns` keeps columns only when grouping by entity ---


def test_aggregate_carries_columns_forward_only_when_grouping_by_entity(
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    req = requirement()
    src = source()
    period_only = AggregateOperation(
        id="totals",
        requirement_id="annual_totals",
        input="debt",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    by_entity = AggregateOperation(
        id="by_entity",
        requirement_id="annual_totals",
        input="debt",
        reduction=Reduction.SUM,
        group_by=(GroupBy.ENTITY, GroupBy.PERIOD),
    )

    shapes, issues = _resolve(
        _retrieve(),
        period_only,
        by_entity,
        requirements={"annual_debt_service": req},
        sources={"debt": src},
    )

    assert issues == []
    assert shapes["totals"].columns == ()
    assert shapes["totals"].shape.roles == ()
    # Carried forward: the same `ResolvedRole` the retrieval built, re-keyed
    # but unchanged in value, since nothing here narrows or drops it.
    assert shapes["by_entity"].columns == shapes["debt"].columns


def test_an_aggregate_grouping_by_entity_with_no_role_dimension_is_refused(
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    req = requirement()
    src = source()
    period_only = AggregateOperation(
        id="totals",
        requirement_id="annual_totals",
        input="debt",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    by_entity_of_a_span = AggregateOperation(
        id="by_entity",
        requirement_id="annual_totals_2",
        input="totals",
        reduction=Reduction.SUM,
        group_by=(GroupBy.ENTITY,),
    )

    shapes, issues = _resolve(
        _retrieve(),
        period_only,
        by_entity_of_a_span,
        requirements={"annual_debt_service": req},
        sources={"debt": src},
    )

    assert "by_entity" not in shapes
    assert issues == [
        "by_entity: aggregate cannot group by entity over a frame with no role dimension; "
        "group by entity in the step that collapsed the roles"
    ]


# --- growth: columns and roles carry through unchanged, in either mode ---


def test_growth_carries_its_input_columns_forward_in_either_mode(
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    req = requirement()
    src = source()
    relative = GrowthOperation(
        id="relative",
        requirement_id="growth_of_debt",
        input="debt",
        window_start=2015,
        window_end=2020,
    )
    absolute = GrowthOperation(
        id="absolute",
        requirement_id="growth_of_debt",
        input="debt",
        window_start=2015,
        window_end=2020,
        change=GrowthChange.ABSOLUTE,
    )

    shapes, issues = _resolve(
        _retrieve(),
        relative,
        absolute,
        requirements={"annual_debt_service": req},
        sources={"debt": src},
    )

    assert issues == []
    assert shapes["relative"].columns == shapes["debt"].columns
    assert shapes["absolute"].columns == shapes["debt"].columns


# --- rank: columns narrow to the axis role alone, in either mode ---


def _two_role_requirement_and_source() -> tuple[DatasetRequirement, SelectedSource]:
    """A requirement naming an entity on each of two roles, and a source resolving both."""
    req = DatasetRequirement(
        id="annual_debt_service",
        decision_ids=["measure"],
        measure="external debt service paid",
        entities=[Entity(name="Kenya", role="debtor"), Entity(name="China", role="creditor")],
        period=Period(start=2015, end=2024),
    )
    src = SelectedSource(
        spec_id=SPEC_KENYA,
        sentence="external debt service paid by Kenya to China, 2015 to 2024",
        confidence="high",
        has_data=True,
        served_count=1,
        shape_member_digest=member_digest(["DebtServicePaid"]),
        entities=(
            ResolvedEntity(role="recipient", ref="country/KEN", label="Kenya"),
            ResolvedEntity(role="creditor", ref="country/CHN", label="China"),
        ),
        unit_ref="usd",
        unit_label="US dollars",
        coverage_window=Period(start=2010, end=2024),
        identity={"stat_var": "DebtServicePaid"},
    )
    return req, src


def test_rank_over_a_lone_role_narrows_columns_to_nothing_else(
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    req = requirement()
    src = source()
    rank_op = RankOperation(
        id="rank",
        requirement_id="rank_of_debt",
        input="debt",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.NAMED,
    )

    shapes, issues = _resolve(
        _retrieve(), rank_op, requirements={"annual_debt_service": req}, sources={"debt": src}
    )

    assert issues == []
    assert shapes["rank"].columns == shapes["debt"].columns


def test_rank_over_two_roles_keeps_only_the_axis_columns_own_role() -> None:
    req, src = _two_role_requirement_and_source()
    rank_op = RankOperation(
        id="rank",
        requirement_id="rank_of_debt",
        input="debt",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.NAMED,
        axis="creditor",
    )

    shapes, issues = _resolve(
        _retrieve(), rank_op, requirements={"annual_debt_service": req}, sources={"debt": src}
    )

    assert issues == []
    (role,) = shapes["rank"].columns
    assert role.column == "creditor"
    assert role.role == "creditor"
    # The debtor column is dropped, not merely reordered.
    assert {r.column for r in shapes["debt"].columns} == {"recipient", "creditor"}


def test_rank_per_period_carries_the_axis_roles_own_narrowed_flag_forward() -> None:
    req, src = _two_role_requirement_and_source()
    filter_op = FilterOperation(
        id="named",
        requirement_id="named_creditor",
        input="debt",
        predicates=(EntityIn(names=("China",), role="creditor"),),
    )
    rank_op = RankOperation(
        id="rank",
        requirement_id="rank_of_debt",
        input="named",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.NAMED,
        axis="creditor",
        per_period=True,
    )

    shapes, issues = _resolve(
        _retrieve(),
        filter_op,
        rank_op,
        requirements={"annual_debt_service": req},
        sources={"debt": src},
    )

    assert issues == []
    (role,) = shapes["rank"].columns
    assert role.column == "creditor"
    assert role.narrowed is True


def test_a_rank_with_no_axis_over_two_roles_is_refused() -> None:
    req, src = _two_role_requirement_and_source()
    rank_op = RankOperation(
        id="rank",
        requirement_id="rank_of_debt",
        input="debt",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.NAMED,
    )

    shapes, issues = _resolve(
        _retrieve(), rank_op, requirements={"annual_debt_service": req}, sources={"debt": src}
    )

    assert "rank" not in shapes
    assert len(issues) == 1
    assert issues[0].startswith("rank: ")


# --- filter: the matched role is narrowed, its column carried unchanged ---


def test_filter_with_an_unqualified_entity_in_narrows_the_lone_role_and_keeps_its_column(
    requirement: RequirementFactory,
    wide_source: SourceFactory,
) -> None:
    req = requirement()
    src = wide_source()
    kenya_only = (EntityIn(names=("Kenya",)),)
    filter_op = FilterOperation(
        id="kenya_only", requirement_id="kenya_only", input="debt", predicates=kenya_only
    )

    shapes, issues = _resolve(
        _retrieve(), filter_op, requirements={"annual_debt_service": req}, sources={"debt": src}
    )

    assert issues == []
    (before,) = shapes["debt"].columns
    (after,) = shapes["kenya_only"].columns
    assert after.column == before.column
    assert after.spec_id == before.spec_id
    assert after.surplus == before.surplus  # still surplus -- Ghana is still unnamed
    assert after.narrowed is True  # but no longer pooled, since this filter keeps Kenya alone


def test_filter_naming_a_role_explicitly_narrows_only_that_one() -> None:
    req, src = _two_role_requirement_and_source()
    filter_op = FilterOperation(
        id="named",
        requirement_id="named_creditor",
        input="debt",
        predicates=(EntityIn(names=("China",), role="creditor"),),
    )

    shapes, issues = _resolve(
        _retrieve(), filter_op, requirements={"annual_debt_service": req}, sources={"debt": src}
    )

    assert issues == []
    by_role = {role.role: role for role in shapes["named"].columns}
    assert by_role["creditor"].narrowed is True
    assert by_role["debtor"].narrowed is False


# --- share_of: output columns come from the numerator alone ---


def test_share_of_keeps_only_the_numerators_own_columns(
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    req = requirement()
    gdp_req = DatasetRequirement(
        id="annual_gdp",
        decision_ids=["measure"],
        measure="gross domestic product",
        entities=[Entity(name="Kenya", role="debtor")],
        period=Period(start=2015, end=2024),
    )
    gdp_src = SelectedSource(
        spec_id=SPEC_OTHER,
        sentence="Kenya's GDP, 2015 to 2024",
        confidence="high",
        has_data=True,
        served_count=1,
        shape_member_digest=member_digest(["Gdp"]),
        entities=(ResolvedEntity(role="recipient", ref="country/KEN", label="Kenya"),),
        unit_ref="usd",
        unit_label="US dollars",
        coverage_window=Period(start=2010, end=2024),
        identity={"stat_var": "Gdp"},
    )
    retrieve_gdp = _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER)
    share_op = ShareOfOperation(
        id="share",
        requirement_id="share",
        numerator=Side(input="debt"),
        denominator=Side(input="gdp"),
        axis="debtor",
        denominator_scope=DenominatorScope.PER_AXIS,
    )

    shapes, issues = _resolve(
        _retrieve(),
        retrieve_gdp,
        share_op,
        requirements={"annual_debt_service": req, "annual_gdp": gdp_req},
        sources={"debt": source(), "gdp": gdp_src},
    )

    assert issues == []
    assert shapes["share"].columns == shapes["debt"].columns
    assert shapes["share"].shape.roles == shapes["debt"].shape.roles


def test_an_unkeyed_share_of_carries_no_columns(
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    req = requirement()
    gdp_req = DatasetRequirement(
        id="annual_gdp",
        decision_ids=["measure"],
        measure="gross domestic product",
        entities=[Entity(name="Kenya", role="debtor")],
        period=Period(start=2015, end=2024),
    )
    gdp_src = SelectedSource(
        spec_id=SPEC_OTHER,
        sentence="Kenya's GDP, 2015 to 2024",
        confidence="high",
        has_data=True,
        served_count=1,
        shape_member_digest=member_digest(["Gdp"]),
        entities=(ResolvedEntity(role="recipient", ref="country/KEN", label="Kenya"),),
        unit_ref="usd",
        unit_label="US dollars",
        coverage_window=Period(start=2010, end=2024),
        identity={"stat_var": "Gdp"},
    )
    retrieve_gdp = _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER)
    share_op = ShareOfOperation(
        id="share",
        requirement_id="share",
        numerator=Side(input="debt"),
        denominator=Side(input="gdp"),
    )

    shapes, issues = _resolve(
        _retrieve(),
        retrieve_gdp,
        share_op,
        requirements={"annual_debt_service": req, "annual_gdp": gdp_req},
        sources={"debt": source(), "gdp": gdp_src},
    )

    assert issues == []
    assert shapes["share"].columns == ()


def _top_five_each_year(**overrides: object) -> RankOperation:
    fields: dict[str, object] = {
        "id": "top_five",
        "requirement_id": "top_five",
        "input": "debt",
        "direction": RankDirection.HIGHEST_FIRST,
        "among": RankAmong.POPULATION,
        "per_period": True,
        "limit": 5,
    }
    return RankOperation(**(fields | overrides))  # type: ignore[arg-type]


def test_a_share_of_per_entity_over_a_limited_per_period_rank_is_refused(
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    keyed = ShareOfOperation(
        id="keyed",
        requirement_id="keyed",
        numerator=Side(input="top_five"),
        denominator=Side(input="debt"),
        axis="debtor",
        denominator_scope=DenominatorScope.POOLED,
    )
    bound = ShareOfOperation(
        id="bound",
        requirement_id="bound",
        numerator=Side(input="debt"),
        denominator=Side(input="top_five", entity="Kenya"),
    )
    total = ShareOfOperation(
        id="total",
        requirement_id="total",
        numerator=Side(input="top_five"),
        denominator=Side(input="debt"),
    )

    shapes, issues = _resolve(
        _retrieve(),
        _top_five_each_year(),
        keyed,
        bound,
        total,
        requirements={"annual_debt_service": requirement()},
        sources={"debt": source()},
    )

    assert "keyed" not in shapes
    assert "bound" not in shapes
    assert "total" in shapes
    assert [issue.split(":", 1)[0] for issue in issues] == ["keyed", "bound"]
    keyed_issue, bound_issue = issues
    assert "compute the share_of first" in keyed_issue
    assert (
        "read the share_of's side from the frame that rank reads, keeping its entity" in bound_issue
    )


def test_no_rank_reads_a_limited_per_period_rank(
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    each_year = _top_five_each_year(id="each_year", requirement_id="each_year", input="top_five")
    overall = _top_five_each_year(
        id="overall", requirement_id="overall", input="top_five", per_period=False
    )

    shapes, issues = _resolve(
        _retrieve(),
        _top_five_each_year(),
        each_year,
        overall,
        requirements={"annual_debt_service": requirement()},
        sources={"debt": source()},
    )

    assert "each_year" not in shapes
    assert "overall" not in shapes
    assert [issue.split(":", 1)[0] for issue in issues] == ["each_year", "overall"]
    assert all("rank the source once with the limit the question needs" in i for i in issues)


def test_a_per_axis_share_of_over_a_limited_denominator_is_refused(
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    # One year leaves the limit uncut, so the refusal rests on the limit alone.
    one_year = requirement(period=Period(start=2022, end=2022))
    per_axis = ShareOfOperation(
        id="per_axis",
        requirement_id="per_axis",
        numerator=Side(input="debt"),
        denominator=Side(input="top_five"),
        axis="debtor",
        denominator_scope=DenominatorScope.PER_AXIS,
    )
    pooled = per_axis.model_copy(
        update={
            "id": "pooled",
            "requirement_id": "pooled",
            "denominator_scope": DenominatorScope.POOLED,
        }
    )

    shapes, issues = _resolve(
        _retrieve(),
        _top_five_each_year(),
        per_axis,
        pooled,
        requirements={"annual_debt_service": one_year},
        sources={"debt": source()},
    )

    assert "per_axis" not in shapes
    assert "pooled" in shapes
    assert [issue.split(":", 1)[0] for issue in issues] == ["per_axis"]
    assert "compute the share_of over the unlimited denominator" in issues[0]


@pytest.mark.parametrize(
    ("predicates", "refused"),
    [
        ((PeriodThreshold(op="gt", year=2023),), False),
        ((PeriodThreshold(op="lt", year=2016),), False),
        ((PeriodThreshold(op="gt", year=2017), PeriodThreshold(op="lt", year=2019)), False),
        ((PeriodThreshold(op="gt", year=2022),), True),
        ((PeriodThreshold(op="gt", year=2022), PeriodThreshold(op="ne", year=2024)), False),
        ((PeriodThreshold(op="ne", year=2024),), True),
    ],
)
def test_a_year_filter_that_leaves_one_year_of_a_limited_rank_clears_its_cut(
    requirement: RequirementFactory,
    source: SourceFactory,
    predicates: tuple[PeriodThreshold, ...],
    refused: bool,
) -> None:
    # Over 2015-2024, `year > 2023`, `year < 2016` and `2017 < year < 2019`
    # each leave one year, as does `year > 2022` without 2024; `year > 2022`
    # leaves two, and `year != 2024` leaves nine.
    recent = FilterOperation(
        id="recent", requirement_id="recent", input="top_five", predicates=predicates
    )
    totals = AggregateOperation(
        id="totals",
        requirement_id="totals",
        input="recent",
        reduction=Reduction.SUM,
        group_by=(GroupBy.ENTITY,),
    )

    shapes, issues = _resolve(
        _retrieve(),
        _top_five_each_year(),
        recent,
        totals,
        requirements={"annual_debt_service": requirement()},
        sources={"debt": source()},
    )

    assert ("totals" not in shapes) is refused
    assert [issue.split(":", 1)[0] for issue in issues] == (["totals"] if refused else [])


@pytest.mark.parametrize(
    "predicates",
    [
        (PeriodThreshold(op="gt", year=2024),),
        (PeriodThreshold(op="lt", year=2015),),
        (PeriodThreshold(op="gt", year=2017), PeriodThreshold(op="lt", year=2018)),
        (PeriodThreshold(op="gte", year=2025),),
        (PeriodThreshold(op="lte", year=2014),),
        (PeriodThreshold(op="eq", year=2030),),
        (PeriodThreshold(op="eq", year=2018), PeriodThreshold(op="eq", year=2019)),
        (PeriodThreshold(op="eq", year=2018), PeriodThreshold(op="gt", year=2020)),
        (PeriodThreshold(op="eq", year=2018), PeriodThreshold(op="ne", year=2018)),
    ],
)
def test_a_year_filter_that_leaves_no_year_is_refused_naming_the_filter(
    requirement: RequirementFactory,
    source: SourceFactory,
    predicates: tuple[PeriodThreshold, ...],
) -> None:
    recent = FilterOperation(
        id="recent", requirement_id="recent", input="debt", predicates=predicates
    )

    shapes, issues = _resolve(
        _retrieve(),
        recent,
        requirements={"annual_debt_service": requirement()},
        sources={"debt": source()},
    )

    assert "recent" not in shapes
    assert issues == ["recent: the year filter leaves no year of 2015-2024"]


def test_a_year_filter_excluding_the_one_year_an_earlier_filter_kept_is_refused_naming_it(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    one_year = FilterOperation(
        id="one_year",
        requirement_id="one_year",
        input="debt",
        predicates=(PeriodThreshold(op="eq", year=2020),),
    )
    others = FilterOperation(
        id="others",
        requirement_id="others",
        input="one_year",
        predicates=(PeriodThreshold(op="ne", year=2020),),
    )

    shapes, issues = _resolve(
        _retrieve(),
        one_year,
        others,
        requirements={"annual_debt_service": requirement()},
        sources={"debt": source()},
    )

    assert "others" not in shapes
    assert issues == ["others: the year filter leaves no year of 2020-2020"]


@pytest.mark.parametrize(
    "predicates",
    [
        (PeriodThreshold(op="eq", year=2018),),
        (PeriodThreshold(op="eq", year=2018), PeriodThreshold(op="eq", year=2018)),
        (PeriodThreshold(op="eq", year=2018), PeriodThreshold(op="gt", year=2017)),
        (PeriodThreshold(op="eq", year=2018), PeriodThreshold(op="ne", year=2019)),
    ],
)
def test_an_eq_year_filter_that_leaves_its_year_pins_that_year(
    requirement: RequirementFactory,
    source: SourceFactory,
    predicates: tuple[PeriodThreshold, ...],
) -> None:
    recent = FilterOperation(
        id="recent", requirement_id="recent", input="debt", predicates=predicates
    )

    shapes, issues = _resolve(
        _retrieve(),
        recent,
        requirements={"annual_debt_service": requirement()},
        sources={"debt": source()},
    )

    assert issues == []
    assert shapes["recent"].shape.period == Period(start=2018, end=2018)


# --- join: shared roles keep the left side's own `ResolvedRole` ---


def test_join_keeps_the_left_sides_column_for_a_shared_role() -> None:
    left_req = DatasetRequirement(
        id="annual_debt_service",
        decision_ids=["measure"],
        measure="external debt service paid",
        entities=[Entity(name="Kenya", role="debtor")],
        period=Period(start=2015, end=2024),
    )
    left_src = SelectedSource(
        spec_id=SPEC_KENYA,
        sentence="Kenya's debt service, 2015 to 2024",
        confidence="high",
        has_data=True,
        served_count=1,
        shape_member_digest=member_digest(["DebtServicePaid"]),
        entities=(ResolvedEntity(role="recipient", ref="country/KEN", label="Kenya"),),
        unit_ref="usd",
        unit_label="US dollars",
        coverage_window=Period(start=2010, end=2024),
        identity={"stat_var": "DebtServicePaid"},
    )
    right_req = DatasetRequirement(
        id="annual_debt_service_uganda",
        decision_ids=["measure"],
        measure="external debt service paid",
        entities=[Entity(name="Uganda", role="debtor")],
        period=Period(start=2015, end=2024),
    )
    right_src = SelectedSource(
        spec_id=SPEC_OTHER,
        sentence="Uganda's debt service, 2015 to 2024",
        confidence="high",
        has_data=True,
        served_count=1,
        shape_member_digest=member_digest(["DebtServicePaid"]),
        entities=(ResolvedEntity(role="debtor_col", ref="country/UGA", label="Uganda"),),
        unit_ref="usd",
        unit_label="US dollars",
        coverage_window=Period(start=2010, end=2024),
        identity={"stat_var": "DebtServicePaid"},
    )
    retrieve_right = _retrieve(
        id="uganda", requirement_id="annual_debt_service_uganda", spec_id=SPEC_OTHER
    )
    join_op = JoinOperation(id="joined", requirement_id="joined", left="debt", right="uganda")

    shapes, issues = _resolve(
        _retrieve(),
        retrieve_right,
        join_op,
        requirements={"annual_debt_service": left_req, "annual_debt_service_uganda": right_req},
        sources={"debt": left_src, "uganda": right_src},
    )

    assert issues == []
    (role,) = shapes["joined"].columns
    # Both sides' entities carry the phrase "debtor" (their own `Entity.role`),
    # so the shared role is real; the column kept is the left side's own.
    assert role.column == "recipient"
    assert role.requirement_id == "annual_debt_service"
    assert role.spec_id == SPEC_KENYA


# --- compare over an aggregate: a max or min keeps single years -----------


def _gdp() -> tuple[DatasetRequirement, SelectedSource, RetrieveOperation]:
    """Kenya's GDP, 2015 to 2024: a second retrieval for a two-sided verb."""
    gdp_req = DatasetRequirement(
        id="annual_gdp",
        decision_ids=["measure"],
        measure="gross domestic product",
        entities=[Entity(name="Kenya", role="debtor")],
        period=Period(start=2015, end=2024),
    )
    gdp_src = SelectedSource(
        spec_id=SPEC_OTHER,
        sentence="Kenya's GDP, 2015 to 2024",
        confidence="high",
        has_data=True,
        served_count=1,
        shape_member_digest=member_digest(["Gdp"]),
        entities=(ResolvedEntity(role="recipient", ref="country/KEN", label="Kenya"),),
        unit_ref="usd",
        unit_label="US dollars",
        coverage_window=Period(start=2010, end=2024),
        identity={"stat_var": "Gdp"},
    )
    retrieve_gdp = _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER)
    return gdp_req, gdp_src, retrieve_gdp


def _peak(
    reduction: Reduction = Reduction.MAX, *, name: str = "peak", reads: str = "debt"
) -> AggregateOperation:
    """The `reduction` by entity of `reads`, over its whole period."""
    return AggregateOperation(
        id=name, requirement_id=name, input=reads, reduction=reduction, group_by=(GroupBy.ENTITY,)
    )


@pytest.mark.parametrize(
    ("left", "right", "refused"),
    [
        pytest.param(_peak(Reduction.MAX), None, None, id="max-vs-series"),
        pytest.param(_peak(Reduction.MIN), None, None, id="min-vs-series"),
        pytest.param(_peak(Reduction.SUM), None, "carry single years", id="sum"),
        pytest.param(_peak(Reduction.MEAN), None, "carry single years", id="mean"),
        pytest.param(
            _peak(Reduction.MAX),
            _peak(Reduction.MAX, name="gdp_peak", reads="gdp"),
            "both sides hold",
            id="max-vs-max",
        ),
        pytest.param(
            _top_five_each_year(
                id="peak",
                requirement_id="peak",
                per_period=False,
                limit=None,
                reduction=Reduction.SUM,
                window_start=2010,
                window_end=2030,
            ),
            _peak(Reduction.SUM, name="debt_total"),
            None,
            id="rank-under-a-wider-window-vs-a-total-over-the-same-years",
        ),
    ],
)
def test_a_compare_matches_collapsed_sides_on_the_years_they_hold(
    requirement: RequirementFactory,
    source: SourceFactory,
    left: Operation,
    right: Operation | None,
    refused: str | None,
) -> None:
    # analysis-mcp labels a max or min row with the year its value came from;
    # a sum or mean over 2018-2022 is labelled 2018/2022, which matches no
    # year, and two peaks meet only where both fell in the same year. A rank
    # under a 2010-2030 window labels its rows with the years it ranked,
    # 2018/2022, which a total over 2018-2022 matches.
    req = requirement(period=Period(start=2018, end=2022))
    gdp_req, gdp_src, retrieve_gdp = _gdp()
    gap = CompareOperation(
        id="gap",
        requirement_id="gap",
        left=Side(input="peak"),
        right=Side(input=right.id if right is not None else "gdp"),
    )

    shapes, issues = _resolve(
        _retrieve(),
        retrieve_gdp,
        left,
        *([right] if right is not None else []),
        gap,
        requirements={"annual_debt_service": req, "annual_gdp": gdp_req},
        sources={"debt": source(), "gdp": gdp_src},
    )

    assert ("gap" not in shapes) is (refused is not None)
    assert len(issues) == int(refused is not None)
    assert all(refused in issue for issue in issues)


@pytest.mark.parametrize(
    ("reader", "refused"),
    [
        pytest.param(
            GrowthOperation(
                id="reader",
                requirement_id="reader",
                input="peak",
                window_start=2018,
                window_end=2022,
                change=GrowthChange.ABSOLUTE,
            ),
            "growth reads each entity's series",
            id="growth-after-max",
        ),
        pytest.param(
            _top_five_each_year(id="reader", requirement_id="reader", input="peak"),
            "rank counts every year",
            id="per-period-rank-after-max",
        ),
        pytest.param(
            _top_five_each_year(
                id="reader", requirement_id="reader", input="peak", per_period=False
            ),
            "rank counts every year",
            id="collapsing-rank-after-max",
        ),
    ],
)
def test_a_reader_of_an_extreme_by_entity_follows_its_picked_years(
    requirement: RequirementFactory,
    source: SourceFactory,
    reader: Operation,
    refused: str | None,
) -> None:
    # A max by entity holds one row per entity at the year its peak fell in:
    # no series to grow, no year whose entities a per-period rank could set
    # side by side, and one year of each entity's window for a collapsing
    # rank's coverage to count.
    req = requirement(period=Period(start=2018, end=2022))

    shapes, issues = _resolve(
        _retrieve(),
        _peak(),
        reader,
        requirements={"annual_debt_service": req},
        sources={"debt": source()},
    )

    assert ("reader" not in shapes) is (refused is not None)
    assert len(issues) == int(refused is not None)
    assert all(refused in issue for issue in issues)


# --- the declined read of a terminal compare/join --------------------------


def test_an_operation_reading_a_compare_is_declined_with_no_shape_and_no_issue(
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    req = requirement()
    gdp_req, gdp_src, retrieve_gdp = _gdp()
    gap = CompareOperation(
        id="gap", requirement_id="gap", left=Side(input="debt"), right=Side(input="gdp")
    )
    reader = AggregateOperation(
        id="over_gap",
        requirement_id="over_gap",
        input="gap",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )

    shapes, issues = _resolve(
        _retrieve(),
        retrieve_gdp,
        gap,
        reader,
        requirements={"annual_debt_service": req, "annual_gdp": gdp_req},
        sources={"debt": source(), "gdp": gdp_src},
    )

    # `gap` itself gets a shape, built for completeness; `compiler.compile`'s
    # own `INPUT_IS_TERMINAL` is the only issue a reader of it produces, so
    # this module leaves the reader out of both `shapes` and `issues`.
    assert "gap" in shapes
    assert "over_gap" not in shapes
    assert issues == []


def test_an_operation_reading_a_join_is_declined_with_no_shape_and_no_issue() -> None:
    left_req, left_src = _two_role_requirement_and_source()
    right_req = DatasetRequirement(
        id="annual_debt_service_uganda",
        decision_ids=["measure"],
        measure="external debt service paid",
        entities=[Entity(name="Uganda", role="debtor")],
        period=Period(start=2015, end=2024),
    )
    right_src = SelectedSource(
        spec_id=SPEC_OTHER,
        sentence="Uganda's debt service, 2015 to 2024",
        confidence="high",
        has_data=True,
        served_count=1,
        shape_member_digest=member_digest(["DebtServicePaid"]),
        entities=(ResolvedEntity(role="debtor_col", ref="country/UGA", label="Uganda"),),
        unit_ref="usd",
        unit_label="US dollars",
        coverage_window=Period(start=2010, end=2024),
        identity={"stat_var": "DebtServicePaid"},
    )
    retrieve_right = _retrieve(
        id="uganda", requirement_id="annual_debt_service_uganda", spec_id=SPEC_OTHER
    )
    join_op = JoinOperation(id="joined", requirement_id="joined", left="debt", right="uganda")
    reader = AggregateOperation(
        id="over_join",
        requirement_id="over_join",
        input="joined",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )

    shapes, issues = _resolve(
        _retrieve(),
        retrieve_right,
        join_op,
        reader,
        requirements={"annual_debt_service": left_req, "annual_debt_service_uganda": right_req},
        sources={"debt": left_src, "uganda": right_src},
    )

    assert "joined" in shapes
    assert "over_join" not in shapes
    assert issues == []


# --- place levels: the open place axis holds every level of place ---------------------


def _rank_every_place(*, reads: str) -> RankOperation:
    return RankOperation(
        id="ranked",
        requirement_id="ranked",
        input=reads,
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        reduction=Reduction.MEAN,
    )


def test_a_rank_over_the_open_place_axis_is_refused_until_a_filter_keeps_one_level(
    requirement: RequirementFactory,
    place_scope_block: dict[str, Any],
) -> None:
    # "Top countries by health spending": the recorded block leaves
    # `observationAbout` open, and a retrieval naming no place holds every
    # level of place on it.
    req = requirement(measure="health spending per person", entities=[], across="country")
    src = source_from_spec_block(place_scope_block)
    fetch = _retrieve(spec_id=src.spec_id)
    countries = FilterOperation(
        id="countries",
        requirement_id="countries",
        input="debt",
        predicates=(PlaceLevel(level=PlaceType.COUNTRY),),
    )

    shapes, issues = _resolve(
        fetch,
        _rank_every_place(reads="debt"),
        requirements={"annual_debt_service": req},
        sources={"debt": src},
    )

    assert shapes["debt"].shape.mixed_levels is True
    assert "ranked" not in shapes
    (issue,) = issues
    assert issue.startswith("ranked: the input names no place")
    assert "place_level" in issue

    shapes, issues = _resolve(
        fetch,
        countries,
        _rank_every_place(reads="countries"),
        requirements={"annual_debt_service": req},
        sources={"debt": src},
    )

    assert issues == []
    assert shapes["countries"].shape.mixed_levels is False
    assert [role.column for role in shapes["ranked"].columns] == ["observationAbout"]


def test_a_level_filter_over_named_places_is_refused(
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    countries = FilterOperation(
        id="countries",
        requirement_id="countries",
        input="debt",
        predicates=(PlaceLevel(level=PlaceType.COUNTRY),),
    )

    shapes, issues = _resolve(
        _retrieve(),
        countries,
        requirements={"annual_debt_service": requirement()},
        sources={"debt": source()},
    )

    assert "countries" not in shapes
    (issue,) = issues
    assert issue.startswith(
        "countries: a place_level filter reads a requirement that names no place"
    )
