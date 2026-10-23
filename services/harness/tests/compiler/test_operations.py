"""Tests for `planner.operations` (the closed set the translator may propose)
and `planner.proof.mismatches` (whether an operation carries out a requirement).
"""

from __future__ import annotations

import pytest
from planner.operations import (
    AggregateOperation,
    CompareOperation,
    FilterOperation,
    GroupBy,
    GrowthOperation,
    JoinOperation,
    Operation,
    QuotientOfOperation,
    RankOperation,
    Reduction,
    RetrieveOperation,
    ShareOfOperation,
    Side,
    Translation,
)
from planner.plan import (
    AggregateRequirement,
    ComparisonRequirement,
    DatasetRequirement,
    Entity,
    FilterRequirement,
    GrowthRequirement,
    JoinRequirement,
    Period,
    QuotientRequirement,
    RankRequirement,
    RequirementSide,
    ShareOfRequirement,
)
from planner.proof import mismatches
from planner.records import (
    MAX_RANK_LIMIT,
    EntityIn,
    RankAmong,
    RankDirection,
    TaggedModel,
    ValueThreshold,
)
from pydantic import TypeAdapter, ValidationError

OPERATIONS = TypeAdapter(list[Operation])


def test_operations_are_told_apart_by_kind() -> None:
    operations = OPERATIONS.validate_python(
        [
            {"kind": "retrieve", "id": "debt", "requirement_id": "annual_debt_service"},
            {
                "kind": "aggregate",
                "id": "totals",
                "requirement_id": "annual_totals",
                "input": "debt",
                "reduction": "sum",
                "group_by": ["period"],
            },
            {
                "kind": "growth",
                "id": "change",
                "requirement_id": "growth_of_debt",
                "input": "debt",
                "window_start": 2015,
                "window_end": 2023,
            },
            {
                "kind": "compare",
                "id": "gap",
                "requirement_id": "gap",
                "left": {"input": "debt", "entity": "Kenya"},
                "right": {"input": "debt", "entity": "Ghana"},
            },
            {
                "kind": "rank",
                "id": "ranked",
                "requirement_id": "ranked",
                "input": "debt",
                "direction": "highest_first",
                "among": "named",
            },
            {
                "kind": "share_of",
                "id": "share",
                "requirement_id": "share",
                "numerator": {"input": "debt"},
                "denominator": {"input": "gdp"},
            },
            {
                "kind": "quotient_of",
                "id": "per_head",
                "requirement_id": "per_head",
                "numerator": {"input": "debt"},
                "denominator": {"input": "gdp"},
                "axis": "debtor",
                "denominator_axis": "country",
            },
            {
                "kind": "join",
                "id": "joined",
                "requirement_id": "joined",
                "left": "debt",
                "right": "gdp",
            },
            {
                "kind": "filter",
                "id": "filtered",
                "requirement_id": "filtered",
                "input": "debt",
                "predicates": [{"kind": "value_threshold", "op": "gt", "value": 0}],
            },
        ]
    )

    assert [type(operation).__name__ for operation in operations] == [
        "RetrieveOperation",
        "AggregateOperation",
        "GrowthOperation",
        "CompareOperation",
        "RankOperation",
        "ShareOfOperation",
        "QuotientOfOperation",
        "JoinOperation",
        "FilterOperation",
    ]


def test_an_operation_of_an_unknown_kind_is_rejected() -> None:
    # The set is closed: what the compiler cannot map, the translator cannot
    # propose.
    with pytest.raises(ValidationError, match="kind"):
        OPERATIONS.validate_python(
            [{"kind": "deflate", "id": "real", "requirement_id": "r", "input": "debt"}]
        )


def test_an_operation_carries_no_tool_argument() -> None:
    # Column names, frame references and tool arguments are the compiler's to
    # write; a proposal that carries one is rejected, not passed through.
    with pytest.raises(ValidationError, match="frame_ref"):
        AggregateOperation(
            id="totals",
            requirement_id="annual_totals",
            input="debt",
            reduction=Reduction.SUM,
            group_by=(GroupBy.PERIOD,),
            frame_ref={},
        )


def test_an_aggregate_states_its_own_group_by() -> None:
    # Mirrors `AggregateRequirement`, which also has no default: the
    # translator has to say what it groups by, not inherit a guess.
    with pytest.raises(ValidationError, match="Field required"):
        AggregateOperation(
            id="totals", requirement_id="annual_totals", input="debt", reduction=Reduction.SUM
        )
    # An empty group is stated, and reduces every row to one figure.
    one_figure = AggregateOperation(
        id="totals",
        requirement_id="annual_totals",
        input="debt",
        reduction=Reduction.SUM,
        group_by=(),
    )
    assert one_figure.group_by == ()


def test_a_retrieval_may_wait_for_its_source() -> None:
    assert RetrieveOperation(id="debt", requirement_id="annual_debt_service").spec_id is None


def test_a_retrieval_names_a_source_by_its_minted_id() -> None:
    with pytest.raises(ValidationError, match="pattern"):
        RetrieveOperation(id="debt", requirement_id="annual_debt_service", spec_id="spec-abc")


def test_a_growth_window_spans_more_than_one_year() -> None:
    with pytest.raises(ValidationError, match="no later than it starts"):
        GrowthOperation(
            id="change",
            requirement_id="growth_of_debt",
            input="debt",
            window_start=2020,
            window_end=2020,
        )


def test_a_rank_window_needs_both_years_or_neither() -> None:
    with pytest.raises(ValidationError, match="needs both window_start and window_end, or neither"):
        RankOperation(
            id="ranked",
            requirement_id="ranked",
            input="debt",
            direction=RankDirection.HIGHEST_FIRST,
            among=RankAmong.NAMED,
            window_start=2020,
        )


@pytest.mark.parametrize("operation", [ShareOfOperation, QuotientOfOperation])
def test_a_share_of_or_quotient_of_sides_must_differ(
    operation: type[ShareOfOperation | QuotientOfOperation],
) -> None:
    with pytest.raises(ValidationError, match="numerator and denominator must differ"):
        operation(
            id="share",
            requirement_id="share",
            numerator=Side(input="debt"),
            denominator=Side(input="debt"),
        )


def test_a_quotient_of_states_a_denominator_axis_only_with_an_axis() -> None:
    with pytest.raises(ValidationError, match="stated only with axis"):
        QuotientOfOperation(
            id="per_head",
            requirement_id="per_head",
            numerator=Side(input="debt"),
            denominator=Side(input="gdp"),
            denominator_axis="country",
        )


def test_a_join_sides_must_differ() -> None:
    with pytest.raises(ValidationError, match="left and right must differ"):
        JoinOperation(id="joined", requirement_id="joined", left="debt", right="debt")


def test_a_filter_needs_at_least_one_predicate() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        FilterOperation(id="filtered", requirement_id="filtered", input="debt", predicates=())


def test_a_translation_names_at_least_one_operation_and_one_terminal() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        Translation(operations=[], terminal=["debt"])
    with pytest.raises(ValidationError, match="at least 1 item"):
        Translation(
            operations=[RetrieveOperation(id="debt", requirement_id="annual_debt_service")],
            terminal=[],
        )


@pytest.mark.parametrize(
    "tagged",
    [
        RetrieveOperation,
        AggregateOperation,
        GrowthOperation,
        CompareOperation,
        RankOperation,
        ShareOfOperation,
        QuotientOfOperation,
        JoinOperation,
        FilterOperation,
    ],
    ids=lambda cls: cls.__name__,
)
def test_the_json_schema_advertises_kind_as_required(tagged: type[TaggedModel]) -> None:
    # `kind` has a default so code constructing one need not state it, but the
    # translator proposing from the schema has to: the union picks a member
    # by it, and a proposal without one is `union_tag_not_found`.
    assert tagged.model_json_schema()["required"][0] == "kind"


def test_the_translation_schema_advertises_kind_as_required_for_every_tagged_part() -> None:
    # Ten operations, and the four predicates a filter carries.
    definitions = Translation.model_json_schema()["$defs"]
    tagged = {
        name: schema
        for name, schema in definitions.items()
        if "kind" in schema.get("properties", {})
    }

    assert len(tagged) == 14
    assert all(schema["required"][0] == "kind" for schema in tagged.values())


# --- planner.proof.mismatches --------------------------------------------------------
#
# `satisfied_by` maps a requirement id to the id of the operation
# `compiler.compile` established carries it out. Every case below assumes
# "annual_debt_service" is carried out by the retrieval "debt", so a side or
# input naming "debt" resolves.

SATISFIED_BY = {"annual_debt_service": "debt", "annual_gdp": "gdp"}

DATASET = DatasetRequirement(
    id="annual_debt_service",
    decision_ids=["measure"],
    measure="external debt service paid",
    entities=[Entity(name="Kenya", role="debtor")],
    period=Period(start=2015, end=2024),
)


def test_mismatches_reports_kind_alone_when_the_kind_does_not_correspond() -> None:
    retrieve = RetrieveOperation(id="debt", requirement_id="annual_debt_service")

    aggregate_requirement = AggregateRequirement(
        id="annual_totals",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )

    assert mismatches(aggregate_requirement, retrieve, SATISFIED_BY) == ["kind"]


def test_mismatches_is_empty_for_a_dataset_requirement_and_its_retrieval() -> None:
    retrieve = RetrieveOperation(id="debt", requirement_id="annual_debt_service")

    assert mismatches(DATASET, retrieve, SATISFIED_BY) == []


def test_mismatches_checks_an_aggregates_reduction_group_by_and_input() -> None:
    requirement = AggregateRequirement(
        id="annual_totals",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    matching = AggregateOperation(
        id="totals",
        requirement_id="annual_totals",
        input="debt",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    assert mismatches(requirement, matching, SATISFIED_BY) == []

    wrong_reduction = matching.model_copy(update={"reduction": Reduction.MEAN})
    assert mismatches(requirement, wrong_reduction, SATISFIED_BY) == ["reduction"]

    wrong_group_by = matching.model_copy(update={"group_by": (GroupBy.ENTITY,)})
    assert mismatches(requirement, wrong_group_by, SATISFIED_BY) == ["group_by"]

    wrong_input = matching.model_copy(update={"input": "ghost"})
    assert mismatches(requirement, wrong_input, SATISFIED_BY) == ["input"]


def test_mismatches_checks_a_growths_window_and_input() -> None:
    requirement = GrowthRequirement(
        id="growth_of_debt",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        window_start=2015,
        window_end=2023,
    )
    matching = GrowthOperation(
        id="change",
        requirement_id="growth_of_debt",
        input="debt",
        window_start=2015,
        window_end=2023,
    )
    assert mismatches(requirement, matching, SATISFIED_BY) == []

    wrong_start = matching.model_copy(update={"window_start": 2016})
    assert mismatches(requirement, wrong_start, SATISFIED_BY) == ["window_start"]

    wrong_end = matching.model_copy(update={"window_end": 2022})
    assert mismatches(requirement, wrong_end, SATISFIED_BY) == ["window_end"]


def test_mismatches_checks_a_ranks_direction_reduction_window_and_input() -> None:
    requirement = RankRequirement(
        id="ranked",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.NAMED,
        reduction=Reduction.SUM,
        window_start=2015,
        window_end=2023,
    )
    matching = RankOperation(
        id="ranked_op",
        requirement_id="ranked",
        input="debt",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.NAMED,
        reduction=Reduction.SUM,
        window_start=2015,
        window_end=2023,
    )
    assert mismatches(requirement, matching, SATISFIED_BY) == []

    wrong_direction = matching.model_copy(update={"direction": RankDirection.LOWEST_FIRST})
    assert mismatches(requirement, wrong_direction, SATISFIED_BY) == ["direction"]

    wrong_reduction = matching.model_copy(update={"reduction": Reduction.MEAN})
    assert mismatches(requirement, wrong_reduction, SATISFIED_BY) == ["reduction"]

    wrong_window = matching.model_copy(update={"window_start": 2016})
    assert mismatches(requirement, wrong_window, SATISFIED_BY) == ["window_start"]


def test_mismatches_checks_a_ranks_limit_set_unset_and_different() -> None:
    requirement = RankRequirement(
        id="ranked",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.NAMED,
        limit=10,
    )
    matching = RankOperation(
        id="ranked_op",
        requirement_id="ranked",
        input="debt",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.NAMED,
        limit=10,
    )
    assert mismatches(requirement, matching, SATISFIED_BY) == []

    unset = matching.model_copy(update={"limit": None})
    assert mismatches(requirement, unset, SATISFIED_BY) == ["limit"]

    different = matching.model_copy(update={"limit": 5})
    assert mismatches(requirement, different, SATISFIED_BY) == ["limit"]

    unlimited = requirement.model_copy(update={"limit": None})
    assert mismatches(unlimited, matching, SATISFIED_BY) == ["limit"]


@pytest.mark.parametrize("limit", [0, MAX_RANK_LIMIT + 1])
def test_a_rank_operations_limit_is_bounded(limit: int) -> None:
    with pytest.raises(ValidationError, match="limit"):
        RankOperation(
            id="ranked",
            requirement_id="ranked",
            input="debt",
            direction=RankDirection.HIGHEST_FIRST,
            among=RankAmong.NAMED,
            limit=limit,
        )


def test_mismatches_checks_a_comparisons_sides() -> None:
    requirement = ComparisonRequirement(
        id="gap",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="annual_gdp"),
        subtrahend=RequirementSide(input_id="annual_debt_service", entity="Kenya"),
    )
    matching = CompareOperation(
        id="gap_op",
        requirement_id="gap",
        left=Side(input="debt", entity="Kenya"),
        right=Side(input="gdp"),
    )
    assert mismatches(requirement, matching, SATISFIED_BY) == []

    wrong_left_entity = matching.model_copy(update={"left": Side(input="debt", entity="Ghana")})
    assert mismatches(requirement, wrong_left_entity, SATISFIED_BY) == ["subtrahend"]

    wrong_right_input = matching.model_copy(update={"right": Side(input="ghost")})
    assert mismatches(requirement, wrong_right_input, SATISFIED_BY) == ["minuend"]


def test_mismatches_checks_a_share_ofs_sides() -> None:
    requirement = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="annual_debt_service"),
        denominator=RequirementSide(input_id="annual_gdp"),
    )
    matching = ShareOfOperation(
        id="share_op",
        requirement_id="share",
        numerator=Side(input="debt"),
        denominator=Side(input="gdp"),
    )
    assert mismatches(requirement, matching, SATISFIED_BY) == []

    wrong_numerator = matching.model_copy(update={"numerator": Side(input="ghost")})
    assert mismatches(requirement, wrong_numerator, SATISFIED_BY) == ["numerator"]

    wrong_denominator = matching.model_copy(
        update={"denominator": Side(input="debt", entity="Kenya")}
    )
    assert mismatches(requirement, wrong_denominator, SATISFIED_BY) == ["denominator"]


@pytest.mark.parametrize(
    ("field", "proposed"),
    [
        pytest.param("numerator", Side(input="debt"), id="numerator"),
        pytest.param("denominator", Side(input="gdp", entity="Kenya"), id="denominator"),
        pytest.param("axis", "country", id="axis"),
        pytest.param("denominator_axis", None, id="denominator_axis-dropped"),
    ],
)
def test_mismatches_checks_each_field_of_a_quotient_of(field: str, proposed: object) -> None:
    requirement = QuotientRequirement(
        id="per_head",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="annual_debt_service", entity="Kenya"),
        denominator=RequirementSide(input_id="annual_gdp"),
        axis="debtor",
        denominator_axis="country",
    )
    matching = QuotientOfOperation(
        id="per_head_op",
        requirement_id="per_head",
        numerator=Side(input="debt", entity="Kenya"),
        denominator=Side(input="gdp"),
        axis="debtor",
        denominator_axis="country",
    )
    assert mismatches(requirement, matching, SATISFIED_BY) == []

    wrong = matching.model_copy(update={field: proposed})
    assert mismatches(requirement, wrong, SATISFIED_BY) == [field]


def test_mismatches_checks_a_joins_left_and_right() -> None:
    requirement = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id="annual_debt_service", right_id="annual_gdp"
    )
    matching = JoinOperation(id="joined_op", requirement_id="joined", left="debt", right="gdp")
    assert mismatches(requirement, matching, SATISFIED_BY) == []

    wrong_left = matching.model_copy(update={"left": "ghost"})
    assert mismatches(requirement, wrong_left, SATISFIED_BY) == ["left"]

    wrong_right = matching.model_copy(update={"right": "ghost"})
    assert mismatches(requirement, wrong_right, SATISFIED_BY) == ["right"]


def test_mismatches_checks_a_filters_predicates_and_input() -> None:
    requirement = FilterRequirement(
        id="filtered",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        predicates=(ValueThreshold(op="gt", value=0),),
    )
    matching = FilterOperation(
        id="filtered_op",
        requirement_id="filtered",
        input="debt",
        predicates=(ValueThreshold(op="gt", value=0),),
    )
    assert mismatches(requirement, matching, SATISFIED_BY) == []

    wrong_predicates = matching.model_copy(update={"predicates": (EntityIn(names=("Kenya",)),)})
    assert mismatches(requirement, wrong_predicates, SATISFIED_BY) == ["predicates"]

    wrong_input = matching.model_copy(update={"input": "ghost"})
    assert mismatches(requirement, wrong_input, SATISFIED_BY) == ["input"]
