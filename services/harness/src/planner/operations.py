"""The closed set of operations a plan's requirements resolve into.

The translator proposes a `Translation`: operations that retrieve each
requirement's evidence and work on it, and which of them answer the question.
`compiler.compile` turns those into tool calls. The set is closed on purpose,
and an operation carries only what the compiler needs to build a call --
which requirement it retrieves, which earlier operation it reads, a reduction,
a window, an entity by name. Tool names, column names and frame references are
the compiler's to write, from the requirement and the source discovery found,
so none of them can be proposed.

Every operation carries out exactly one requirement (`requirement_id`) and
repeats that requirement's calculation fields, so the compiler can compare
one against the other and prove they agree -- see `planner.proof.mismatches`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from common.hashed_fields import HASHED_WHEN_SET
from pydantic import Field, model_validator

from planner.records import (
    DenominatorScope,
    GroupBy,
    GrowthChange,
    Identifier,
    Predicate,
    RankAmong,
    RankDirection,
    RankLimit,
    Reduction,
    RollingWindow,
    SpecId,
    StrictModel,
    TaggedModel,
    Year,
    check_rolling_window,
)


class OperationKind(StrEnum):
    """What an operation does, and the discriminator of the `Operation` union."""

    RETRIEVE = "retrieve"
    AGGREGATE = "aggregate"
    GROWTH = "growth"
    COMPARE = "compare"
    SUM_OF = "sum_of"
    RANK = "rank"
    SHARE_OF = "share_of"
    QUOTIENT_OF = "quotient_of"
    JOIN = "join"
    FILTER = "filter"


class RetrieveOperation(TaggedModel):
    """Retrieve one requirement's evidence from the source discovery selected.

    `spec_id` is the definition discovery found for the requirement. The
    compiler accepts only an id that discovery recorded, so a retrieval names
    a source that exists and that the user can see.
    """

    kind: Literal[OperationKind.RETRIEVE] = Field(
        default=OperationKind.RETRIEVE, description="Tags this operation as a retrieval."
    )
    id: Identifier = Field(
        description="This operation's name; unique within a translation, so a later operation can cite it as input."
    )
    requirement_id: Identifier = Field(description="The requirement this operation carries out.")
    spec_id: SpecId | None = Field(
        default=None,
        description="The dataset definition discovery found for this requirement, once one has been selected.",
    )


class AggregateOperation(TaggedModel):
    """Reduce the frame an earlier operation produced, per group.

    `input` names a retrieval, an aggregate, a growth, a rank or a filter: an
    operation whose frame rests on one requirement. A comparison's result
    answers the question and feeds no further operation.
    """

    kind: Literal[OperationKind.AGGREGATE] = Field(
        default=OperationKind.AGGREGATE, description="Tags this operation as an aggregate."
    )
    id: Identifier = Field(
        description="This operation's name; unique within a translation, so a later operation can cite it as input."
    )
    requirement_id: Identifier = Field(description="The requirement this operation carries out.")
    input: Identifier = Field(description="The operation whose frame this aggregate reduces.")
    reduction: Reduction = Field(
        description="How to collapse each group's rows into one figure: sum, mean, max or min."
    )
    group_by: tuple[GroupBy, ...] = Field(
        min_length=1,
        description="Which columns to group by before reducing: period, entity, or both.",
    )


class GrowthOperation(TaggedModel):
    """The change across a window of years in the frame an earlier operation produced.

    `input` names a retrieval, an aggregate, a growth, a rank or a filter, as
    for an aggregate.
    """

    kind: Literal[OperationKind.GROWTH] = Field(
        default=OperationKind.GROWTH, description="Tags this operation as a growth."
    )
    id: Identifier = Field(
        description="This operation's name; unique within a translation, so a later operation can cite it as input."
    )
    requirement_id: Identifier = Field(description="The requirement this operation carries out.")
    input: Identifier = Field(
        description="The operation whose frame this growth reads its two endpoint values from."
    )
    window_start: Year = Field(description="The year growth is measured from.")
    window_end: Year = Field(description="The year growth is measured to.")
    change: GrowthChange = Field(
        default=GrowthChange.RELATIVE,
        description=(
            "How the two endpoint values are compared: 'relative' reads a dimensionless ratio, "
            "'absolute' reads a difference in the input's own unit."
        ),
    )
    rolling: Annotated[RollingWindow | None, HASHED_WHEN_SET] = Field(
        default=None,
        description="Which ends of the window follow the newest data, as the requirement states them.",
    )

    @model_validator(mode="after")
    def _rolling_agrees_with_the_years(self) -> GrowthOperation:
        check_rolling_window(
            self.rolling,
            window_start=self.window_start,
            window_end=self.window_end,
            error=ValueError,
        )
        return self

    @model_validator(mode="after")
    def _window_spans_more_than_one_year(self) -> GrowthOperation:
        if self.window_end <= self.window_start:
            raise ValueError(
                f"growth window ends ({self.window_end}) no later than it starts "
                f"({self.window_start})"
            )
        return self


class Side(StrictModel):
    """One side of a two-sided operation: an operation's frame, optionally one entity's slice of it.

    `entity` is an `Entity.name` from the requirement the input retrieves,
    so the compiler can bind it to the source's own column and value. Left
    unset, the side is the whole frame, which is only unambiguous when that
    requirement names one entity.
    """

    input: Identifier = Field(description="The operation this side reads its frame from.")
    entity: str | None = Field(
        default=None,
        min_length=1,
        description="The one entity from `input`'s requirement this side narrows to; unset to take the whole frame.",
    )


class CompareOperation(TaggedModel):
    """Set two sides against each other over the periods they share."""

    kind: Literal[OperationKind.COMPARE] = Field(
        default=OperationKind.COMPARE, description="Tags this operation as a comparison."
    )
    id: Identifier = Field(
        description="This operation's name; unique within a translation, so a later operation can cite it as input."
    )
    requirement_id: Identifier = Field(description="The requirement this operation carries out.")
    left: Side = Field(description="The side subtracted from `right`.")
    right: Side = Field(description="The side `left` is subtracted from.")


class SumOfOperation(TaggedModel):
    """Add two single-series sides together over the periods both report."""

    kind: Literal[OperationKind.SUM_OF] = Field(
        default=OperationKind.SUM_OF, description="Tags this operation as a sum of two series."
    )
    id: Identifier = Field(
        description="This operation's name; unique within a translation, so a later operation can cite it as input."
    )
    requirement_id: Identifier = Field(description="The requirement this operation carries out.")
    left: Side = Field(description="One of the two series added together.")
    right: Side = Field(description="The series added to `left`.")


class RankOperation(TaggedModel):
    """Rank `input`'s entities against each other.

    Mirrors `planner.plan.RankRequirement`'s own fields, so the compiler can
    compare them field for field.
    """

    kind: Literal[OperationKind.RANK] = Field(
        default=OperationKind.RANK, description="Tags this operation as a rank."
    )
    id: Identifier = Field(
        description="This operation's name; unique within a translation, so a later operation can cite it as input."
    )
    requirement_id: Identifier = Field(description="The requirement this operation carries out.")
    input: Identifier = Field(description="The operation whose frame this rank orders.")
    direction: RankDirection = Field(
        description="Which end of the ranking comes first: the highest value or the lowest."
    )
    among: RankAmong = Field(
        description=(
            "Who this rank compares an entity against: 'named' for exactly the entities a "
            "filter has narrowed to, 'population' for every entity the retrieved definition holds."
        )
    )
    axis: str | None = Field(
        default=None,
        min_length=1,
        description="Which role dimension of the input this rank keeps; required once the input carries more than one.",
    )
    reduction: Reduction | None = Field(
        default=None,
        description="How to collapse an entity's rows to one value first, when its window leaves more than one.",
    )
    window_start: Year | None = Field(
        default=None,
        description=(
            "The year this rank's window starts, when it is windowed. The same year as "
            "`window_end` ranks that one year."
        ),
    )
    window_end: Year | None = Field(
        default=None,
        description=(
            "The year this rank's window ends, when it is windowed. The same year as "
            "`window_start` ranks that one year."
        ),
    )
    per_period: bool = Field(
        default=False,
        description="Whether this rank orders each period on its own rather than collapsing the window into one value per entity.",
    )
    limit: Annotated[RankLimit | None, HASHED_WHEN_SET] = Field(
        default=None,
        description="Keep only the entities ranked at this place or better; unset keeps every entity ranked.",
    )
    rolling: Annotated[RollingWindow | None, HASHED_WHEN_SET] = Field(
        default=None,
        description="Which ends of the window follow the newest data, as the requirement states them.",
    )

    @model_validator(mode="after")
    def _rolling_agrees_with_the_years(self) -> RankOperation:
        check_rolling_window(
            self.rolling,
            window_start=self.window_start,
            window_end=self.window_end,
            error=ValueError,
        )
        return self

    @model_validator(mode="after")
    def _window_is_paired_and_ordered(self) -> RankOperation:
        if (self.window_start is None) != (self.window_end is None):
            raise ValueError("a rank window needs both window_start and window_end, or neither")
        if (
            self.window_start is not None
            and self.window_end is not None
            and self.window_end < self.window_start
        ):
            raise ValueError(
                f"rank window ends before it starts; got {self.window_start}-{self.window_end}"
            )
        return self


class ShareOfOperation(TaggedModel):
    """The share `numerator` holds of `denominator`, at each period."""

    kind: Literal[OperationKind.SHARE_OF] = Field(
        default=OperationKind.SHARE_OF, description="Tags this operation as a share_of."
    )
    id: Identifier = Field(
        description="This operation's name; unique within a translation, so a later operation can cite it as input."
    )
    requirement_id: Identifier = Field(description="The requirement this operation carries out.")
    numerator: Side = Field(description="The side this share is a share of.")
    denominator: Side = Field(description="The side `numerator` is divided by.")
    axis: str | None = Field(
        default=None,
        min_length=1,
        description="The role dimension to compute one share per entity on, rather than one figure overall.",
    )
    denominator_scope: DenominatorScope | None = Field(
        default=None,
        description=(
            "Whose total a keyed share divides by: 'per_axis' divides each entity by its own "
            "total, 'pooled' divides every entity by one total summed across the population."
        ),
    )

    @model_validator(mode="after")
    def _sides_differ(self) -> ShareOfOperation:
        if (self.numerator.input, self.numerator.entity) == (
            self.denominator.input,
            self.denominator.entity,
        ):
            raise ValueError("a share_of's numerator and denominator must differ")
        return self

    @model_validator(mode="after")
    def _denominator_scope_needs_an_axis(self) -> ShareOfOperation:
        if (self.axis is None) != (self.denominator_scope is None):
            raise ValueError(
                "denominator_scope is stated exactly when axis is: an un-keyed share has one "
                "denominator value and the question does not arise, a keyed share must state it"
            )
        return self


class QuotientOfOperation(TaggedModel):
    """`numerator` divided by `denominator`, at each period both report."""

    kind: Literal[OperationKind.QUOTIENT_OF] = Field(
        default=OperationKind.QUOTIENT_OF, description="Tags this operation as a quotient_of."
    )
    id: Identifier = Field(
        description="This operation's name; unique within a translation, so a later operation can cite it as input."
    )
    requirement_id: Identifier = Field(description="The requirement this operation carries out.")
    numerator: Side = Field(description="The side that is divided.")
    denominator: Side = Field(description="The side `numerator` is divided by.")
    axis: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The numerator's role dimension to compute one quotient per entity on, rather "
            "than one figure overall. Alone, every entity divides by the denominator's one "
            "total."
        ),
    )
    denominator_axis: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The role the denominator's frame carries the same entities under, so each "
            "divides by its own counterpart."
        ),
    )

    @model_validator(mode="after")
    def _sides_differ(self) -> QuotientOfOperation:
        if (self.numerator.input, self.numerator.entity) == (
            self.denominator.input,
            self.denominator.entity,
        ):
            raise ValueError("a quotient_of's numerator and denominator must differ")
        return self

    @model_validator(mode="after")
    def _denominator_axis_needs_an_axis(self) -> QuotientOfOperation:
        if self.denominator_axis is not None and self.axis is None:
            raise ValueError(
                "denominator_axis names the denominator's role for the entities axis keys the "
                "numerator by, so it is stated only with axis"
            )
        return self


class JoinOperation(TaggedModel):
    """Join two operations' frames into one."""

    kind: Literal[OperationKind.JOIN] = Field(
        default=OperationKind.JOIN, description="Tags this operation as a join."
    )
    id: Identifier = Field(
        description="This operation's name; unique within a translation, so a later operation can cite it as input."
    )
    requirement_id: Identifier = Field(description="The requirement this operation carries out.")
    left: Identifier = Field(description="One of the two operations this join combines.")
    right: Identifier = Field(description="The other operation this join combines.")

    @model_validator(mode="after")
    def _sides_are_distinct(self) -> JoinOperation:
        if self.left == self.right:
            raise ValueError(f"a join's left and right must differ; both are {self.left!r}")
        return self


class FilterOperation(TaggedModel):
    """Keep only `input`'s rows that satisfy every one of `predicates`."""

    kind: Literal[OperationKind.FILTER] = Field(
        default=OperationKind.FILTER, description="Tags this operation as a filter."
    )
    id: Identifier = Field(
        description="This operation's name; unique within a translation, so a later operation can cite it as input."
    )
    requirement_id: Identifier = Field(description="The requirement this operation carries out.")
    input: Identifier = Field(description="The operation whose rows this filter narrows.")
    predicates: tuple[Predicate, ...] = Field(
        min_length=1,
        description="The conditions a row has to satisfy, all of them, to survive this filter.",
    )


Operation = Annotated[
    RetrieveOperation
    | AggregateOperation
    | GrowthOperation
    | CompareOperation
    | SumOfOperation
    | RankOperation
    | ShareOfOperation
    | QuotientOfOperation
    | JoinOperation
    | FilterOperation,
    Field(discriminator="kind"),
]


class Translation(StrictModel):
    """The translator's proposal: the operations, and which of them answer the question."""

    operations: tuple[Operation, ...] = Field(
        min_length=1, description="Every operation this translation proposes."
    )
    terminal: tuple[Identifier, ...] = Field(
        min_length=1, description="The operation ids whose results answer the question."
    )
