"""The `compare` engine: value comparison between two slices, each a
`Frame`'s table narrowed to one entity's time series, over the intersection
of the periods each side offers. Periods are intersected, both realised
coverages are recorded, and the comparison refuses only on an empty
intersection -- the "proceed" side of `analysis_mcp.engine.common`'s
partial-overlap policy.

A "slice" is `frame_ref` (or `right_frame_ref`) plus an `entity` binding, a
`{role_column: value}` mapping narrowing the frame to one row per
`time_period` before unit/facets are resolved; empty when the frame already
names exactly one entity. The same `frame_ref` for both sides with two
different `entity` bindings is two slices of one frame; two different
`frame_ref`s is two frame_refs -- both reach this engine as
`(left_frame, right_frame)`, so there is one code path.

`delta` (right value minus left value) is a single subtraction, never a
reduction over rows, so `math.fsum` is never used here. Two finite doubles
of opposite sign and large magnitude can still subtract to an infinite
result under IEEE-754, so `compare` checks `delta` the same way `aggregate`
and `growth` check their own results.

Each side resolves its own unit independently; subtraction across two
different units is silently wrong, so `run_compare` refuses
(`CompareUnitMismatchError`) if the two sides' resolved units differ. One
side's rows leaving `unit` unbound entirely is a separate case
(`CompareUnitUnboundError`): there is no second unit to compare. Facets are
checked the same way (`CompareFacetMismatchError`): nothing else catches a
left slice scoped to `provenance=dcid:P1` paired against a right slice
scoped to `dcid:P2`. Neither check runs when either side's slice is empty;
`ComparisonPeriodsDisjointError` names that problem once the period
intersection is checked.

`slice_pair` (both slices and the unit check) and `align_periods` (the
period intersection) serve `analysis_mcp.engine.sum_of` as well, which
pairs its two sides the same way and differs in the value it computes and
in its facet policy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

import pyarrow as pa
from dfo_frame_grammar import Frame, array_kernel
from dfo_frame_grammar.grammar import (
    STAT_VAR_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    UnitRequest,
)

from analysis_mcp.engine.columnar import distinct, frame_from_columns, in_periods, keys_unique
from analysis_mcp.engine.common import (
    Exclusions,
    FacetSelection,
    available_entity_columns,
    combine_exclusions,
    cross_side_one_sided_facet_axes,
    facet_columns,
    first_cross_side_facet_mismatch,
    narrow_and_resolve,
    record_exclusion,
    validate_entity_columns,
)


class CompareAmbiguousSliceError(Exception):
    """More than one row shares one side's `time_period` after narrowing by
    its `entity` binding -- the binding doesn't uniquely identify one row
    per period, so there is no single value to compare for that period.
    """

    def __init__(self, *, side: str, entity: Mapping[str, str], available: Sequence[str]) -> None:
        self.side = side
        self.entity = dict(entity)
        self.available = tuple(available)
        super().__init__(
            f"{side} side's entity binding {self.entity!r} does not uniquely identify "
            "one row per time_period"
        )


class ComparisonPeriodsDisjointError(Exception):
    """The two sides share no overlapping `time_period` at all."""

    def __init__(self, *, left_periods: Sequence[str], right_periods: Sequence[str]) -> None:
        self.left_periods = tuple(left_periods)
        self.right_periods = tuple(right_periods)
        super().__init__("the two sides share no overlapping time_period")


class CompareUnitMismatchError(Exception):
    """Each side resolved its own single unit, but the two sides' resolved
    units differ from each other -- a USD-denominated left slice and a
    Percent-denominated right slice, say. Distinct from `UnitHeterogeneousError`,
    which fires when *one* frame carries more than one unit; here each side
    is internally uniform, and it's the pairing across sides that's unsafe.
    """

    def __init__(self, *, left_unit: str | UnitRequest, right_unit: str | UnitRequest) -> None:
        self.left_unit = left_unit
        self.right_unit = right_unit
        super().__init__(f"left resolved to unit {left_unit!r}, right to {right_unit!r}")


class CompareUnitUnboundError(Exception):
    """One side's own rows leave `unit` unbound entirely, while the other
    side resolved to a real declared unit. Distinct from
    `CompareUnitMismatchError`, which fires when both sides resolve to a
    real, differing unit -- here there is no second resolved unit to
    disagree with, only a fact about `unbound_side`'s own data: none of its
    own rows ever declared one. Silently inheriting `bound_side`'s own
    resolved unit would state something `unbound_side`'s own rows never
    declared.
    """

    def __init__(
        self, *, unbound_side: str, bound_side: str, bound_unit: str | UnitRequest
    ) -> None:
        self.unbound_side = unbound_side
        self.bound_side = bound_side
        self.bound_unit = bound_unit
        super().__init__(
            f"{unbound_side} side's own rows leave `unit` unbound; {bound_side} resolved "
            f"to {bound_unit!r}"
        )


class CompareFacetMismatchError(Exception):
    """Each side resolved its own facet selections independently, but they
    disagree on at least one axis -- a `provenance=dcid:P1` left slice
    compared against a `provenance=dcid:P2` right slice, say. Distinct from
    `analysis_mcp.engine.common.FacetHeterogeneousError`, which fires when
    *one* slice itself carries more than one declared value on an axis;
    here each side is internally uniform (or unbound) on its own, and it's
    the pairing across sides that's unsafe -- the same "each side resolves
    independently, the pairing is what's checked" shape
    `CompareUnitMismatchError` already applies to `unit`.
    """

    def __init__(self, *, axis: str, left_ref: str, right_ref: str) -> None:
        self.axis = axis
        self.left_ref = left_ref
        self.right_ref = right_ref
        super().__init__(f"left resolved facet {axis!r} to {left_ref!r}, right to {right_ref!r}")


class CompareNonFiniteResultError(Exception):
    """A period's combined value overflowed to a non-finite value.
    `quantity` names it: `delta` for `compare`, `sum` for `sum_of`."""

    def __init__(self, *, time_period: str, quantity: str = "delta") -> None:
        self.time_period = time_period
        self.quantity = quantity
        super().__init__(f"non-finite {quantity} for time_period {time_period!r}")


@dataclass(frozen=True, kw_only=True)
class PeriodComparison:
    """One intersected period's own two values and their delta, addressed
    by `time_period` rather than by position -- the same "never trust row
    order" discipline `analysis_mcp.engine.aggregate.GroupSummary` follows.
    `left_stat_var`/`right_stat_var` are each side's own StatVar at this
    period, kept alongside the delta since the output row's own `stat_var`
    (see `run_compare` below) always carries the left side's, even when the
    two differ.
    """

    time_period: str
    left_value: float
    right_value: float
    delta: float
    left_stat_var: str
    right_stat_var: str


@dataclass(frozen=True, kw_only=True)
class CompareResult:
    """Everything `compare_tool.py` needs to mint identity, write the store,
    and render the digest -- the frame is here for the store write and
    replay; the digest itself renders from `period_comparisons`, never
    `frame.rows`.
    """

    frame: Frame
    left_resolved_unit: str | UnitRequest | None
    right_resolved_unit: str | UnitRequest | None
    left_facet_selections: tuple[FacetSelection, ...]
    right_facet_selections: tuple[FacetSelection, ...]
    warnings: tuple[str, ...]
    left_periods: tuple[str, ...]
    right_periods: tuple[str, ...]
    intersected_periods: tuple[str, ...]
    left_coverage_fraction: float
    right_coverage_fraction: float
    period_comparisons: tuple[PeriodComparison, ...]
    exclusions: Exclusions = field(default_factory=Exclusions)


def _one_sided_facet_warning(axis: str, left_ref: str | None, right_ref: str | None) -> str:
    """One `cross_side_one_sided_facet_axes` finding, worded for compare's
    left/right vocabulary. The result carries only the left side's facet
    selections, so the warning states not just which side left the axis
    unbound, but whether the other side's value actually reaches the result.
    """
    if left_ref is not None:
        return (
            f"facet {axis!r} was left unbound on the right side's own rows; the left "
            f"side's own declared value {left_ref!r} is what the result carries."
        )
    return (
        f"facet {axis!r} was left unbound on the left side's own rows; the right side's "
        f"own declared value {right_ref!r} exists but the result carries no value for "
        "this axis (compare's own output always carries the left side's facet selections)."
    )


@dataclass(frozen=True, kw_only=True)
class SideSlice:
    """One side narrowed to its own entity's rows, with the unit/facets
    resolved over exactly that slice. `table` holds one row per period."""

    table: pa.Table
    unit: str | UnitRequest | None
    facet_selections: tuple[FacetSelection, ...]
    warnings: tuple[str, ...]
    exclusions: Exclusions

    def periods(self) -> tuple[str, ...]:
        """This side's periods, sorted."""
        return tuple(distinct(self.table, TIME_PERIOD_COLUMN))

    def matched(self, periods: Sequence[str]) -> tuple[pa.Table, pa.Table]:
        """This side's rows at `periods`, sorted by period, and its other rows."""
        matched = in_periods(self.table, periods)
        return (
            self.table.filter(matched).sort_by(TIME_PERIOD_COLUMN),
            self.table.filter(array_kernel("invert", matched)),
        )

    @property
    def populated(self) -> bool:
        """Whether any row survived narrowing."""
        return self.table.num_rows > 0


def _slice(
    frame: Frame,
    *,
    side: Literal["left", "right"],
    entity: Mapping[str, str],
    pinned_unit: str | UnitRequest | None,
) -> SideSlice:
    """Narrow `frame` to `entity`'s own rows and resolve unit/facets over
    exactly that slice.

    Refuses (`UnknownEntityColumnError`) if `entity` names a column this
    frame has no grain for. Refuses (`CompareAmbiguousSliceError`) if `entity`
    fails to narrow the frame to exactly one row per period.
    """
    validate_entity_columns(frame, tuple(entity))
    # `exclude_unitless=True`: a period whose row leaves `unit` unbound is
    # excluded from this pairing, shrinking this side's own
    # `left_periods`/`right_periods` list (and, if it empties the side
    # entirely, raising `ComparisonPeriodsDisjointError`). This side's own
    # coverage fraction can still read 1.0 after losing a period, if its
    # shorter list still equals the intersection; the other side's fraction
    # drops when the intersection it measures against shrinks.
    table, resolved, exclusions = narrow_and_resolve(
        frame, entity=entity, pinned_unit=pinned_unit, exclude_unitless=True, side=side
    )
    if not keys_unique(table, [TIME_PERIOD_COLUMN]):
        raise CompareAmbiguousSliceError(
            side=side, entity=entity, available=available_entity_columns(frame)
        )
    return SideSlice(
        table=table,
        unit=resolved.unit,
        facet_selections=resolved.facet_selections,
        warnings=tuple(resolved.warnings),
        exclusions=exclusions,
    )


def slice_pair(
    left_frame: Frame,
    right_frame: Frame,
    *,
    left_entity: Mapping[str, str],
    right_entity: Mapping[str, str],
    left_pinned_unit: str | UnitRequest | None,
    right_pinned_unit: str | UnitRequest | None,
) -> tuple[SideSlice, SideSlice]:
    """Both sides' slices, refused (`CompareUnitUnboundError`,
    `CompareUnitMismatchError`) unless their resolved units agree.

    The unit check runs only when both sides have rows. An empty slice
    resolves to `unit=None` with no facet selections
    (`narrow_and_resolve`'s documented behaviour), which would otherwise
    misread as a mismatch against whatever the other, populated side
    resolved to. The real problem an empty side has is no periods at all,
    which `align_periods`' `ComparisonPeriodsDisjointError` names.
    """
    left = _slice(left_frame, side="left", entity=left_entity, pinned_unit=left_pinned_unit)
    right = _slice(right_frame, side="right", entity=right_entity, pinned_unit=right_pinned_unit)
    if not (left.populated and right.populated):
        return left, right
    left_unit, right_unit = left.unit, right.unit
    if left_unit is None or right_unit is None:
        if right_unit is not None:
            raise CompareUnitUnboundError(
                unbound_side="left", bound_side="right", bound_unit=right_unit
            )
        if left_unit is not None:
            raise CompareUnitUnboundError(
                unbound_side="right", bound_side="left", bound_unit=left_unit
            )
    elif left_unit != right_unit:
        raise CompareUnitMismatchError(left_unit=left_unit, right_unit=right_unit)
    return left, right


@dataclass(frozen=True, kw_only=True)
class PeriodAlignment:
    """Two slices matched on the periods both offer. `left_matched` and
    `right_matched` hold one row per intersected period, in period order."""

    left_periods: tuple[str, ...]
    right_periods: tuple[str, ...]
    intersected_periods: tuple[str, ...]
    left_coverage_fraction: float
    right_coverage_fraction: float
    left_matched: pa.Table
    right_matched: pa.Table
    exclusions: Exclusions


def align_periods(
    left: SideSlice, right: SideSlice, *, left_frame: Frame, right_frame: Frame
) -> PeriodAlignment:
    """Intersect `left`'s and `right`'s periods, refusing
    (`ComparisonPeriodsDisjointError`) when they share none. Each side's
    coverage fraction is the intersection's share of the periods that side
    offered."""
    left_periods = left.periods()
    right_periods = right.periods()
    intersected_periods = tuple(sorted(set(left_periods) & set(right_periods)))
    if not intersected_periods:
        raise ComparisonPeriodsDisjointError(left_periods=left_periods, right_periods=right_periods)

    # Every period each side offered but the intersection dropped, recorded
    # as an exclusion. The coverage fractions state the ratio, not which
    # rows that ratio cost. `analysis_mcp.digest` renders only the coverage
    # line, not a second line repeating these records.
    left_matched, left_unmatched = left.matched(intersected_periods)
    right_matched, right_unmatched = right.matched(intersected_periods)
    exclusions = combine_exclusions(left.exclusions, right.exclusions)
    exclusions = record_exclusion(
        exclusions,
        frame=left_frame,
        reason="period_unmatched",
        dropped=left_unmatched,
        side="left",
    )
    exclusions = record_exclusion(
        exclusions,
        frame=right_frame,
        reason="period_unmatched",
        dropped=right_unmatched,
        side="right",
    )
    return PeriodAlignment(
        left_periods=left_periods,
        right_periods=right_periods,
        intersected_periods=intersected_periods,
        left_coverage_fraction=len(intersected_periods) / len(left_periods),
        right_coverage_fraction=len(intersected_periods) / len(right_periods),
        left_matched=left_matched,
        right_matched=right_matched,
        exclusions=exclusions,
    )


def run_compare(
    left_frame: Frame,
    right_frame: Frame,
    *,
    left_entity: Mapping[str, str],
    right_entity: Mapping[str, str],
    left_pinned_unit: str | UnitRequest | None = None,
    right_pinned_unit: str | UnitRequest | None = None,
) -> CompareResult:
    """Compare `left_frame`'s (narrowed by `left_entity`) and `right_frame`'s
    (narrowed by `right_entity`) own single-entity time series over their
    intersected periods. See the module docstring for what a "slice" and
    "the two sides" mean; see each raised exception's own docstring for its
    refusal condition.
    """
    left, right = slice_pair(
        left_frame,
        right_frame,
        left_entity=left_entity,
        right_entity=right_entity,
        left_pinned_unit=left_pinned_unit,
        right_pinned_unit=right_pinned_unit,
    )

    # Runs only when both sides have rows, for the reason `slice_pair`
    # gives for its unit check.
    cross_side_facet_warnings: tuple[str, ...] = ()
    if left.populated and right.populated:
        facet_mismatch = first_cross_side_facet_mismatch(
            left.facet_selections, right.facet_selections
        )
        if facet_mismatch is not None:
            axis, left_ref, right_ref = facet_mismatch
            raise CompareFacetMismatchError(axis=axis, left_ref=left_ref, right_ref=right_ref)

        # A one-sided axis (see `cross_side_one_sided_facet_axes`) doesn't
        # refuse, but still deserves the visibility a single slice's own
        # auto-selection gives -- worded here, not in the shared helper,
        # since only this verb knows its rule for which side's facets the
        # result carries (the left side's, always).
        cross_side_facet_warnings = tuple(
            _one_sided_facet_warning(axis, left_ref, right_ref)
            for axis, left_ref, right_ref in cross_side_one_sided_facet_axes(
                left.facet_selections, right.facet_selections
            )
        )

    alignment = align_periods(left, right, left_frame=left_frame, right_frame=right_frame)
    intersected_periods = alignment.intersected_periods
    left_matched, right_matched = alignment.left_matched, alignment.right_matched

    period_comparisons = []
    for period, left_stat_var, left_value, right_stat_var, right_value in zip(
        intersected_periods,
        cast("list[str]", left_matched.column(STAT_VAR_COLUMN).to_pylist()),
        cast("list[float]", left_matched.column(VALUE_COLUMN).to_pylist()),
        cast("list[str]", right_matched.column(STAT_VAR_COLUMN).to_pylist()),
        cast("list[float]", right_matched.column(VALUE_COLUMN).to_pylist()),
        strict=True,
    ):
        delta = right_value - left_value
        if not math.isfinite(delta):
            raise CompareNonFiniteResultError(time_period=period)
        # Signed zero is normalized at the engine boundary, the same
        # invariant every verb in this toolkit applies (see
        # `engine/growth.py`'s `growth_rate += 0.0`), so a `-0.0` cannot
        # persist into the stored output frame and re-chain to another verb.
        delta += 0.0
        period_comparisons.append(
            PeriodComparison(
                time_period=period,
                left_value=left_value,
                right_value=right_value,
                delta=delta,
                left_stat_var=left_stat_var,
                right_stat_var=right_stat_var,
            )
        )
    period_comparisons = tuple(period_comparisons)

    # The output frame carries the left side's `stat_var`, unit, and facet
    # selections -- the delta is labelled as a movement relative to the left
    # side, the same asymmetric choice `delta`'s own sign already makes. The
    # right side's unit/facets are still returned above, for the trace and
    # digest to record in full. Every row of the left slice carries its
    # resolved unit, so its `unit` column is the output's.
    output_frame = frame_from_columns(
        {
            STAT_VAR_COLUMN: left_matched.column(STAT_VAR_COLUMN),
            TIME_PERIOD_COLUMN: left_matched.column(TIME_PERIOD_COLUMN),
            VALUE_COLUMN: pa.array(
                [comparison.delta for comparison in period_comparisons], type=pa.float64()
            ),
            UNIT_COLUMN: left_matched.column(UNIT_COLUMN),
            **facet_columns(left.facet_selections, count=len(period_comparisons)),
        },
        role_keys=(),
    )

    return CompareResult(
        frame=output_frame,
        left_resolved_unit=left.unit,
        right_resolved_unit=right.unit,
        left_facet_selections=left.facet_selections,
        right_facet_selections=right.facet_selections,
        warnings=tuple(
            dict.fromkeys((*left.warnings, *right.warnings, *cross_side_facet_warnings))
        ),
        left_periods=alignment.left_periods,
        right_periods=alignment.right_periods,
        intersected_periods=intersected_periods,
        left_coverage_fraction=alignment.left_coverage_fraction,
        right_coverage_fraction=alignment.right_coverage_fraction,
        period_comparisons=period_comparisons,
        exclusions=alignment.exclusions,
    )
