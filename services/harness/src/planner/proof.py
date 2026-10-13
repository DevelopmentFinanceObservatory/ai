"""Does an operation actually carry out the requirement it names?

`planner.operations.Operation` and `planner.plan.EvidenceRequirement` share
one calculation vocabulary (`planner.records`) precisely so this module can
compare them field for field: `mismatches` is what `compiler.compile` and
`planner.plan._translation_and_workflow_agree` both call to prove that every
requirement a plan states is carried out, exactly once, by an operation that
repeats its calculation fields rather than merely naming it.

This lives in its own module, not in `planner.operations`, to break a cycle:
`planner.plan` already imports `Translation` from `planner.operations` at
module level, so `planner.operations` cannot import the requirement classes
from `planner.plan` at module level in return. `planner.proof` imports both
and is itself imported lazily, inside `_translation_and_workflow_agree`,
never at `planner.plan`'s own module level.
"""

from __future__ import annotations

from collections.abc import Mapping

from planner.operations import (
    AggregateOperation,
    CompareOperation,
    FilterOperation,
    GrowthOperation,
    JoinOperation,
    Operation,
    OperationKind,
    QuotientOfOperation,
    RankOperation,
    ShareOfOperation,
    Side,
    SumOfOperation,
)
from planner.plan import (
    AggregateRequirement,
    ComparisonRequirement,
    DatasetRequirement,
    EvidenceRequirement,
    FilterRequirement,
    GrowthRequirement,
    JoinRequirement,
    QuotientRequirement,
    RankRequirement,
    RequirementSide,
    ShareOfRequirement,
    SumOfRequirement,
)

# What kind of operation carries out each requirement kind. `compiler.compile`
# maps a translation's operations by this same table when it builds a step's
# tool call, so the two never drift apart from stating it twice.
KIND_OF_REQUIREMENT: Mapping[type, OperationKind] = {
    DatasetRequirement: OperationKind.RETRIEVE,
    AggregateRequirement: OperationKind.AGGREGATE,
    GrowthRequirement: OperationKind.GROWTH,
    ComparisonRequirement: OperationKind.COMPARE,
    SumOfRequirement: OperationKind.SUM_OF,
    RankRequirement: OperationKind.RANK,
    ShareOfRequirement: OperationKind.SHARE_OF,
    QuotientRequirement: OperationKind.QUOTIENT_OF,
    JoinRequirement: OperationKind.JOIN,
    FilterRequirement: OperationKind.FILTER,
}


def mismatches(
    requirement: EvidenceRequirement, operation: Operation, satisfied_by: Mapping[str, str]
) -> list[str]:
    """Return the names of the fields on which `operation` disagrees with `requirement`.

    `["kind"]` alone when `operation`'s kind does not correspond to
    `requirement`'s own kind under `KIND_OF_REQUIREMENT` -- nothing else is
    comparable once the shape itself is wrong. Otherwise, every calculation
    field the two share that carries a different value, and every input
    reference that does not resolve: an operation's `input` (or a side's
    `input`, or a join's `left`/`right`) has to equal
    `satisfied_by[<the requirement's corresponding input id>]`, the id of the
    operation that `compiler.compile` established carries out that upstream
    requirement, and a side's `entity` has to equal the requirement side's.
    Those report as `input`, `minuend`, `subtrahend`, `left`, `right`,
    `numerator` or `denominator`. A comparison's `minuend` is its operation's
    `right` and its `subtrahend` the operation's `left`, since analysis-mcp's
    `compare` takes the right side minus the left.

    Args:
        requirement: What the operation is meant to carry out.
        operation: What the translation proposes for it.
        satisfied_by: Requirement id to the id of the one operation
            `compiler.compile` established carries it out.

    Returns:
        The differing field names, empty when `operation` agrees with
        `requirement` in full.
    """
    expected_kind = KIND_OF_REQUIREMENT[type(requirement)]
    if operation.kind is not expected_kind:
        return ["kind"]

    found: list[str] = []

    def _bad_input(field: str, requirement_input_id: str, operation_input: str) -> None:
        if satisfied_by.get(requirement_input_id) != operation_input:
            found.append(field)

    def _bad_side(field: str, requirement_side: RequirementSide, operation_side: Side) -> None:
        resolved = satisfied_by.get(requirement_side.input_id)
        if resolved != operation_side.input or requirement_side.entity != operation_side.entity:
            found.append(field)

    if isinstance(requirement, DatasetRequirement):
        # A `RetrieveOperation` carries no calculation field of its own to
        # compare -- what a source can offer for it is `compiler.evidence`'s
        # concern, not this one.
        pass
    elif isinstance(requirement, AggregateRequirement):
        assert isinstance(operation, AggregateOperation)
        if requirement.reduction != operation.reduction:
            found.append("reduction")
        if requirement.group_by != operation.group_by:
            found.append("group_by")
        _bad_input("input", requirement.input_id, operation.input)
    elif isinstance(requirement, GrowthRequirement):
        assert isinstance(operation, GrowthOperation)
        if requirement.window_start != operation.window_start:
            found.append("window_start")
        if requirement.window_end != operation.window_end:
            found.append("window_end")
        if requirement.change != operation.change:
            found.append("change")
        if requirement.rolling != operation.rolling:
            found.append("rolling")
        _bad_input("input", requirement.input_id, operation.input)
    elif isinstance(requirement, RankRequirement):
        assert isinstance(operation, RankOperation)
        if requirement.direction != operation.direction:
            found.append("direction")
        if requirement.among != operation.among:
            found.append("among")
        if requirement.axis != operation.axis:
            found.append("axis")
        if requirement.reduction != operation.reduction:
            found.append("reduction")
        if requirement.window_start != operation.window_start:
            found.append("window_start")
        if requirement.window_end != operation.window_end:
            found.append("window_end")
        if requirement.per_period != operation.per_period:
            found.append("per_period")
        if requirement.limit != operation.limit:
            found.append("limit")
        if requirement.rolling != operation.rolling:
            found.append("rolling")
        _bad_input("input", requirement.input_id, operation.input)
    elif isinstance(requirement, FilterRequirement):
        assert isinstance(operation, FilterOperation)
        if requirement.predicates != operation.predicates:
            found.append("predicates")
        _bad_input("input", requirement.input_id, operation.input)
    elif isinstance(requirement, ComparisonRequirement):
        assert isinstance(operation, CompareOperation)
        _bad_side("minuend", requirement.minuend, operation.right)
        _bad_side("subtrahend", requirement.subtrahend, operation.left)
    elif isinstance(requirement, SumOfRequirement):
        assert isinstance(operation, SumOfOperation)
        _bad_side("left", requirement.left, operation.left)
        _bad_side("right", requirement.right, operation.right)
    elif isinstance(requirement, ShareOfRequirement):
        assert isinstance(operation, ShareOfOperation)
        _bad_side("numerator", requirement.numerator, operation.numerator)
        _bad_side("denominator", requirement.denominator, operation.denominator)
        if requirement.axis != operation.axis:
            found.append("axis")
        if requirement.denominator_scope != operation.denominator_scope:
            found.append("denominator_scope")
    elif isinstance(requirement, QuotientRequirement):
        assert isinstance(operation, QuotientOfOperation)
        _bad_side("numerator", requirement.numerator, operation.numerator)
        _bad_side("denominator", requirement.denominator, operation.denominator)
        if requirement.axis != operation.axis:
            found.append("axis")
        if requirement.denominator_axis != operation.denominator_axis:
            found.append("denominator_axis")
    else:
        assert isinstance(requirement, JoinRequirement)
        assert isinstance(operation, JoinOperation)
        _bad_input("left", requirement.left_id, operation.left)
        _bad_input("right", requirement.right_id, operation.right)

    return found
