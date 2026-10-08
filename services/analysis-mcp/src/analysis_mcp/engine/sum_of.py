"""The `sum_of` engine: two single-series slices added period by period,
over the intersection of the periods each side offers.

The sides are paired exactly as `analysis_mcp.engine.compare` pairs them
(`slice_pair`, `align_periods`): each is `frame_ref` (or `right_frame_ref`)
narrowed by an `entity` binding to one row per `time_period`, the two
resolved units must agree, and an empty period intersection refuses
(`ComparisonPeriodsDisjointError`). A period one side lacks is left out,
since its total is unknown, and recorded with the side that reports it and
that side's value there (`PeriodLeftOut`).

Facets are where `sum_of` departs from `compare`. A sum across two datasets
(reported bilateral flows plus modelled imputed shares, say) differs in
`provenance` and `measurementMethod` by construction, so a facet axis the
sides disagree on is a warning that names both values. The result carries a
facet axis only when both sides declare the same value on it, and leaves any
other axis unbound, since neither side's value describes the sum.

`stat_var` on every result row is the caller-supplied `stat_var`, the
operation's own ref (`dcid:<op_spec_hash>`, which `sum_of_tool` passes in),
since the sum of two variables is a third. `left_stat_vars` and
`right_stat_vars` keep what each side measured, for the digest's label.

Each period's value is one IEEE-754 addition, which is already correctly
rounded for two operands, so `math.fsum` would return the same double. Two
finite doubles of large magnitude can still add to an infinite result, so
`run_sum_of` checks each sum the way `compare` checks its delta.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

import pyarrow as pa
from dfo_frame_grammar import Frame
from dfo_frame_grammar.grammar import (
    STAT_VAR_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    ComposedUnit,
    UnitCell,
)

from analysis_mcp.engine.aggregate import ComposedUnitSumError
from analysis_mcp.engine.columnar import distinct, frame_from_columns, repeated
from analysis_mcp.engine.common import (
    Exclusions,
    FacetSelection,
    agreed_facets,
    facet_columns,
)
from analysis_mcp.engine.compare import (
    CompareNonFiniteResultError,
    SideSlice,
    align_periods,
    slice_pair,
)


@dataclass(frozen=True, kw_only=True)
class PeriodSum:
    """One intersected period's two values and their sum."""

    time_period: str
    left_value: float
    right_value: float
    value: float


@dataclass(frozen=True, kw_only=True)
class PeriodLeftOut:
    """A period only one side reports, which the sum leaves out: the side and its value there."""

    time_period: str
    side: Literal["left", "right"]
    value: float


@dataclass(frozen=True, kw_only=True)
class SumOfResult:
    """Everything `sum_of_tool.py` needs to write the store and render the
    digest. `facet_selections` are the axes both sides agree on, the ones
    the result carries; each side's own selections are kept for the trace."""

    frame: Frame
    stat_var: str
    left_stat_vars: tuple[str, ...]
    right_stat_vars: tuple[str, ...]
    unit: UnitCell | None
    facet_selections: tuple[FacetSelection, ...]
    left_facet_selections: tuple[FacetSelection, ...]
    right_facet_selections: tuple[FacetSelection, ...]
    warnings: tuple[str, ...]
    left_periods: tuple[str, ...]
    right_periods: tuple[str, ...]
    intersected_periods: tuple[str, ...]
    left_coverage_fraction: float
    right_coverage_fraction: float
    period_sums: tuple[PeriodSum, ...]
    left_out: tuple[PeriodLeftOut, ...] = ()
    exclusions: Exclusions = field(default_factory=Exclusions)


def run_sum_of(
    left_frame: Frame,
    right_frame: Frame,
    *,
    stat_var: str,
    left_entity: Mapping[str, str],
    right_entity: Mapping[str, str],
    left_pinned_unit: UnitCell | None = None,
    right_pinned_unit: UnitCell | None = None,
) -> SumOfResult:
    """Add `left_frame`'s (narrowed by `left_entity`) and `right_frame`'s
    (narrowed by `right_entity`) single-entity series over their
    intersected periods, labelling every result row `stat_var`. See the
    module docstring for the facet policy and `analysis_mcp.engine.compare`
    for the refusals the pairing raises. A side whose resolved unit is
    composed raises `ComposedUnitSumError`, after the pairing's own unit
    rules.
    """
    left, right = slice_pair(
        left_frame,
        right_frame,
        left_entity=left_entity,
        right_entity=right_entity,
        left_pinned_unit=left_pinned_unit,
        right_pinned_unit=right_pinned_unit,
    )
    for side, side_slice in (("left", left), ("right", right)):
        if isinstance(side_slice.unit, ComposedUnit):
            raise ComposedUnitSumError(side_slice.unit, side=side)
    shared_facets: tuple[FacetSelection, ...] = ()
    facet_warnings: tuple[str, ...] = ()
    if left.populated and right.populated:
        shared_facets, facet_warnings = agreed_facets(
            left.facet_selections, right.facet_selections, sides=("left", "right"), result="sum"
        )

    alignment = align_periods(left, right, left_frame=left_frame, right_frame=right_frame)
    left_matched, right_matched = alignment.left_matched, alignment.right_matched

    period_sums: list[PeriodSum] = []
    for period, left_value, right_value in zip(
        alignment.intersected_periods,
        cast("list[float]", left_matched.column(VALUE_COLUMN).to_pylist()),
        cast("list[float]", right_matched.column(VALUE_COLUMN).to_pylist()),
        strict=True,
    ):
        total = left_value + right_value
        if not math.isfinite(total):
            raise CompareNonFiniteResultError(time_period=period, quantity="sum")
        period_sums.append(
            PeriodSum(
                time_period=period,
                left_value=left_value,
                right_value=right_value,
                # Signed zero normalised, as every verb's output is.
                value=total + 0.0,
            )
        )

    count = len(period_sums)
    output_frame = frame_from_columns(
        {
            STAT_VAR_COLUMN: repeated(stat_var, count),
            TIME_PERIOD_COLUMN: left_matched.column(TIME_PERIOD_COLUMN),
            VALUE_COLUMN: pa.array([entry.value for entry in period_sums], type=pa.float64()),
            UNIT_COLUMN: left_matched.column(UNIT_COLUMN),
            **facet_columns(shared_facets, count=count),
        },
        role_keys=(),
    )

    return SumOfResult(
        frame=output_frame,
        stat_var=stat_var,
        left_stat_vars=tuple(distinct(left_matched, STAT_VAR_COLUMN)),
        right_stat_vars=tuple(distinct(right_matched, STAT_VAR_COLUMN)),
        unit=left.unit,
        facet_selections=shared_facets,
        left_facet_selections=left.facet_selections,
        right_facet_selections=right.facet_selections,
        warnings=tuple(dict.fromkeys((*left.warnings, *right.warnings, *facet_warnings))),
        left_periods=alignment.left_periods,
        right_periods=alignment.right_periods,
        intersected_periods=alignment.intersected_periods,
        left_coverage_fraction=alignment.left_coverage_fraction,
        right_coverage_fraction=alignment.right_coverage_fraction,
        period_sums=tuple(period_sums),
        left_out=_left_out(left, right, alignment.intersected_periods),
        exclusions=alignment.exclusions,
    )


def _left_out(
    left: SideSlice, right: SideSlice, intersected: Sequence[str]
) -> tuple[PeriodLeftOut, ...]:
    """Every period one side alone reports, by period, with that side's own value there."""
    left_out: list[PeriodLeftOut] = []
    for side, slice_ in (("left", left), ("right", right)):
        _, unmatched = slice_.matched(intersected)
        periods = cast("list[str]", unmatched.column(TIME_PERIOD_COLUMN).to_pylist())
        values = cast("list[float]", unmatched.column(VALUE_COLUMN).to_pylist())
        left_out.extend(
            PeriodLeftOut(time_period=period, side=side, value=value)
            for period, value in zip(periods, values, strict=True)
        )
    return tuple(sorted(left_out, key=lambda entry: (entry.time_period, entry.side)))
