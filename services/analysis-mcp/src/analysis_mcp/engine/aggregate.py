"""The `aggregate` engine: group-by reduction over a frame's Arrow table.

Rows are grouped with `analysis_mcp.engine.columnar.group_rows`, and each
group's values are reduced in Python. `math.fsum` is used
for every additive reduction -- a plain running `+=` accumulates IEEE-754
rounding error across a group's rows, and `fsum` is exact (correctly-rounded
to the nearest representable double) regardless of accumulation order, which
is what makes the input-order-perturbation test in `tests/test_aggregate_tool.py`
meaningful: shuffle the input rows before `Frame.build` and the result must
still be bit-identical.

Coverage is over ABSENT GRAIN CELLS, not null values. `Frame.build` has no
way to represent an in-frame null observation -- `value` must be a finite
number and `status` is reserved, always `None` (see
`analysis_mcp.fixtures`'s module docstring) -- so "skip nulls" for this verb
can only mean "skip grid cells the frame never materialised a row for."
`expected_cell_count` is the cartesian product of each `group_by` column's
own distinct values across the (unit-resolved) input rows; `present_cell_count`
is the number of group keys that actually appear; the difference is
`skipped_cell_count`. This is exactly `sparse_coverage_frame`'s shape: some
`(recipient, year)` cells are simply missing rows.

The reduction runs after `analysis_mcp.engine.common.narrow_and_resolve`.
On the pinned path, narrowing to `pinned_unit`'s own rows already excludes
a row leaving `unit` unbound for free (`None` never matches a real pinned
unit). On the unpinned path, `narrow_and_resolve` resolves once first to
learn whether this slice has a real unit at all; only if it does, it drops
rows leaving `unit` unbound and resolves a second time over what remains.
A frame where no row declares any unit at all is left alone either way:
there is no declared unit for "unitless" to mean anything relative to, and
every such row is already part of one consistent, dimensionless whole.
Either path's own exclusion, when it drops anything, is reported as a
warning -- unit safety is load-bearing, so it is checked first, not folded
into the reduction loop.

Every function here raises a typed exception on refusal rather than
returning one; only the tool boundary (`analysis_mcp.aggregate_tool`) catches
these and converts them to a `ToolRefusal`, matching `data_mcp`'s own
discipline of raising deep and converting once, at the top.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import pyarrow as pa
from common.time_grain import span_label
from dfo_frame_grammar import Frame, array_kernel
from dfo_frame_grammar.grammar import (
    FACET_IDENTITY_COLUMNS,
    STAT_VAR_COLUMN,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    ComposedUnit,
    UnitCell,
)

from analysis_mcp.engine.columnar import (
    distinct,
    entity_counts,
    group_members,
    group_rows,
    repeated,
    unit_column,
)
from analysis_mcp.engine.common import (
    Exclusions,
    FacetSelection,
    PooledParties,
    combine_exclusions,
    entity_roles,
    facet_columns,
    frame_keeping_role_types,
    narrow_and_resolve,
    pooled_parties,
    pooled_roles,
    require_one_place_level,
    role_columns,
    table_before_unitless_exclusion,
    without_party_totals,
)

# The closed aggregation vocabulary. `AggregateOpSpec.aggregation` is
# a plain `str`, not a pydantic `Literal`, precisely so a value outside this
# tuple reaches `UnsupportedAggregationError`, the tool's own
# `operation_uncovered` refusal, rather than a generic schema error.
AGGREGATIONS: tuple[str, ...] = ("sum", "mean", "max", "min")

# Never a valid `group_by` dimension. `value` is what's being reduced;
# `status` is reserved and always `None` at this grammar version (grouping
# by it would always yield exactly one, uninformative group); `unit` and the
# facet axes are resolved to a single value (or refused) before this engine
# ever groups anything, so grouping by them too would be redundant at best
# -- and, since a `unit` cell can be `None` or carry a `grammar.UnitRequest`
# rather than a plain string, allowing it here would break the "every
# group_by column is string-valued" guarantee the rest of this function
# relies on (stat_var, time_period, and every role column are always plain,
# non-null strings by construction -- see `Frame.build`'s own normalisation).
_DISALLOWED_GROUP_BY_COLUMNS = frozenset(
    {VALUE_COLUMN, STATUS_COLUMN, UNIT_COLUMN, *FACET_IDENTITY_COLUMNS}
)


class UnsupportedAggregationError(Exception):
    """`aggregation` is not one of `AGGREGATIONS`."""

    def __init__(self, aggregation: str) -> None:
        self.aggregation = aggregation
        super().__init__(f"unsupported aggregation {aggregation!r}; supported: {AGGREGATIONS!r}")


class ComposedUnitSumError(Exception):
    """A sum was asked over values in `unit`, a composed unit. Two figures
    of the form "X per Y" added together are not a figure per Y, so no verb
    writes their total under that unit. `aggregate` and `rank` raise it for
    the `sum` aggregation and `sum_of` for either side, where `side` is
    `left` or `right`; a mean, max or min over a composed unit computes.
    `two_sided.sum_side` raises it for a `numerator` or `denominator` side
    whose (key, period) cell holds more than one row.
    """

    def __init__(self, unit: ComposedUnit, *, side: str | None = None) -> None:
        self.unit = unit
        self.side = side
        super().__init__(f"a sum over the composed unit {unit!r}")


def refuse_sum_over_composed_unit(aggregation: str | None, unit: UnitCell | None) -> None:
    """Raise `ComposedUnitSumError` when `aggregation` is `sum` and `unit`,
    the one unit the reduced rows resolved to, is composed."""
    if aggregation == "sum" and isinstance(unit, ComposedUnit):
        raise ComposedUnitSumError(unit)


class UnknownGroupByColumnError(Exception):
    """One or more `group_by` entries name a column this frame has no grain
    for -- absent from `frame.columns`, or one of the disallowed reduction/
    reserved columns (`value`, `status`)."""

    def __init__(self, columns: Sequence[str], *, available: Sequence[str]) -> None:
        self.columns = tuple(columns)
        self.available = tuple(available)
        super().__init__(f"group_by names unknown/disallowed column(s): {self.columns!r}")


class StatVarUnresolvedError(Exception):
    """The input frame carries no rows, or carries more than one distinct
    `stat_var` with no way to tell them apart in the result.

    `distinct_stat_vars` is empty for the "no rows at all" case.
    """

    def __init__(self, *, reason: str, distinct_stat_vars: Sequence[str] = ()) -> None:
        self.reason = reason
        self.distinct_stat_vars = tuple(distinct_stat_vars)
        super().__init__(reason)


class NonFiniteAggregateResultError(Exception):
    """A group's reduction produced a non-finite (`inf`/`nan`) result."""

    def __init__(self, *, group_key: dict[str, str], aggregation: str) -> None:
        self.group_key = dict(group_key)
        self.aggregation = aggregation
        super().__init__(f"non-finite {aggregation} for group {group_key!r}")


def _reduce[Period: (str, int)](
    values: Sequence[float], periods: Sequence[Period], *, aggregation: str
) -> tuple[float, Period | None]:
    """The reduction `run_aggregate` and `run_rank` both apply: `sum` (via
    `math.fsum`), `mean` (that same sum divided by the member count), or the
    `max`/`min` member's own value, each value paired by position with
    `periods`. Returns the reduced value paired with a period. For
    `max`/`min` that period is the member that supplied the value, the
    earliest period among any tie, so the result holds under any row order.
    For `sum` and `mean` the period is `None`, since no single member
    supplies the answer.

    `periods` holds either the periods themselves or their `GroupMembers`
    codes, which order as the periods do. `values` and `periods` are the
    same length and never empty. Both
    callers build them from one group's own member rows. This function may
    raise `OverflowError`, the same as a bare `math.fsum` call, for a sum
    whose exact value exceeds the finite double range. The caller converts
    that, and any non-finite result, to its own typed refusal carrying its
    own group or entity context.
    """
    if aggregation in ("sum", "mean"):
        result = math.fsum(values)
        if aggregation == "mean":
            result = result / len(values)
        return result, None

    picker = max if aggregation == "max" else min
    winning_value = picker(values)
    winning_period = min(
        period for value, period in zip(values, periods, strict=True) if value == winning_value
    )
    return winning_value, winning_period


@dataclass(frozen=True, kw_only=True)
class GroupSummary:
    """One realised group's computed result, addressed by its own
    `group_by`-keyed identity rather than by position in `AggregateResult.frame`.
    Frame construction sorts rows by the frame grammar's own total ordering, so
    a caller matching a summary back to a group by list position would be
    matching against an order this module never chose.

    `value_period` names the period of the member row that supplied `value`
    under `max` and `min`. It is `None` under `sum` and `mean`, where no
    single member supplies the answer.
    """

    group_key: tuple[tuple[str, str], ...]
    value: float
    row_count: int
    time_period_min: str
    time_period_max: str
    value_period: str | None = None


@dataclass(frozen=True, kw_only=True)
class AggregateResult:
    """Everything `aggregate_tool.py` needs to mint identity, write the store,
    and render the digest -- the frame is here for the store write and replay;
    the digest itself renders from `group_summaries`, never `frame.rows`.

    `collapsed_periods` is every distinct `time_period` this call's narrowed
    input rows carried, across every group, sorted. `period_label` is the
    one `time_period` every row of `frame` carries under sum or mean with
    `time_period` outside `group_by`, `span_label(collapsed_periods)`, and
    `None` otherwise. `aggregate_tool.py` passes `collapsed_periods` to
    `frame_ref_resolution.build_interval` as `observed` when that label
    collapses more than one point, to say whether the span is contiguous.

    `entity_count` is how many output entities hold a total: the distinct
    value combinations across the entity roles (`common.entity_roles`) in
    `group_by`, and `None` when `group_by` names none. `period_entities`
    maps each of `collapsed_periods` to how many input entities its totals
    pool: the distinct value combinations, across all groups, of the entity
    roles outside `group_by`, or of the grouped ones when every one is
    grouped, since each total then pools only its own entity. A place's
    type or a party's kind, sector or whole cell describes an entity and
    counts as none.

    `pooled_parties` holds, under `sum`, for each flow party role the call pooled,
    how many parties it summed and how many of them are in the private sector,
    with their value. It is empty under any other aggregation.
    """

    frame: Frame
    resolved_unit: UnitCell | None
    facet_selections: tuple[FacetSelection, ...]
    warnings: tuple[str, ...]
    realised_coverage_fraction: float
    skipped_cell_count: int
    expected_cell_count: int
    present_cell_count: int
    group_summaries: tuple[GroupSummary, ...]
    collapsed_periods: tuple[str, ...] = ()
    period_label: str | None = None
    entity_count: int | None = None
    period_entities: tuple[tuple[str, int], ...] = ()
    exclusions: Exclusions = field(default_factory=Exclusions)
    pooled_parties: tuple[PooledParties, ...] = ()


def run_aggregate(
    frame: Frame,
    *,
    group_by: Sequence[str],
    aggregation: str,
    pinned_unit: UnitCell | None = None,
) -> AggregateResult:
    """Group `frame`'s rows by `group_by` and reduce each group's `value` with
    `aggregation`. See the module docstring for the coverage and unit/facet
    resolution rules. See each raised exception's own docstring for its
    refusal condition. Never raises a bare exception type. Every refusal
    condition below has its own named class.
    """
    if aggregation not in AGGREGATIONS:
        raise UnsupportedAggregationError(aggregation)

    group_by = tuple(group_by)
    disallowed = [
        column
        for column in group_by
        if column not in frame.columns or column in _DISALLOWED_GROUP_BY_COLUMNS
    ]
    if disallowed:
        available = [c for c in frame.columns if c not in _DISALLOWED_GROUP_BY_COLUMNS]
        raise UnknownGroupByColumnError(disallowed, available=available)

    if not frame.num_rows:
        raise StatVarUnresolvedError(reason="the input frame carries no rows")
    pooled = pooled_roles(frame, group_by)
    require_one_place_level(frame, pooled)
    frame, party_exclusions = without_party_totals(frame, pooled)

    distinct_stat_vars = distinct(frame.table, STAT_VAR_COLUMN)
    if len(distinct_stat_vars) > 1 and STAT_VAR_COLUMN not in group_by:
        raise StatVarUnresolvedError(
            reason=(
                "the input frame carries more than one StatVar; include 'stat_var' in "
                "group_by, or narrow the input to a frame_ref that carries only one"
            ),
            distinct_stat_vars=distinct_stat_vars,
        )

    # `exclude_unitless=True` excludes a row leaving `unit` unbound from the
    # arithmetic on both the pinned and the unpinned path alike. The pinned
    # path already gets this for free (`None` never matches a real
    # `pinned_unit`), so the unpinned path opts in explicitly to agree with
    # it, rather than silently summing an unbound row in. Either path warns
    # when this actually drops a row.
    table, resolved, exclusions = narrow_and_resolve(
        frame, pinned_unit=pinned_unit, exclude_unitless=True
    )
    exclusions = combine_exclusions(party_exclusions, exclusions)
    refuse_sum_over_composed_unit(aggregation, resolved.unit)
    # Only a sum's digest reports the parties it pooled.
    summed_parties = pooled_parties(table, pooled) if aggregation == "sum" else ()

    # Every `group_by` column left after `_DISALLOWED_GROUP_BY_COLUMNS` is
    # stat_var, time_period, or a role, always a plain, non-null string by
    # the frame grammar's own normalisation. A column named twice groups
    # once.
    key_columns = tuple(dict.fromkeys(group_by))
    groups = group_rows(table, key_columns)
    members = group_members(table, groups)
    keys = {
        column: cast("list[str]", groups.keys.column(column).to_pylist()) for column in key_columns
    }

    # Under sum or mean with `time_period` outside `group_by`, every output
    # row carries the one label covering every period the call collapsed,
    # across all groups. A single-span input keeps its span and a
    # one-period input keeps its point. Under max or min a row carries the
    # period its value came from. A grouped `time_period` is overwritten by
    # the group's own key below.
    period_label = (
        span_label(members.periods)
        if aggregation in ("sum", "mean") and TIME_PERIOD_COLUMN not in key_columns
        else None
    )
    output_periods: list[str] = []
    output_values: list[float] = []
    summaries: list[GroupSummary] = []
    for group in range(groups.count):
        group_key_map = {column: values[group] for column, values in keys.items()}
        values, period_codes = members.of(group)
        try:
            result_value, value_code = _reduce(values, period_codes, aggregation=aggregation)
        except OverflowError as exc:
            raise NonFiniteAggregateResultError(
                group_key=group_key_map, aggregation=aggregation
            ) from exc
        if not math.isfinite(result_value):
            raise NonFiniteAggregateResultError(group_key=group_key_map, aggregation=aggregation)

        time_period_max = members.periods[max(period_codes)]
        value_period = None if value_code is None else members.periods[value_code]
        output_periods.append(value_period or period_label or time_period_max)
        output_values.append(result_value)
        summaries.append(
            GroupSummary(
                group_key=tuple(group_key_map.items()),
                value=result_value,
                row_count=len(values),
                time_period_min=members.periods[min(period_codes)],
                time_period_max=time_period_max,
                value_period=value_period,
            )
        )

    role_in_group_by = [column for column in group_by if column in role_columns(frame)]
    columns: dict[str, pa.Array | pa.ChunkedArray] = {
        STAT_VAR_COLUMN: repeated(distinct_stat_vars[0], groups.count),
        TIME_PERIOD_COLUMN: pa.array(output_periods, type=pa.string()),
        VALUE_COLUMN: pa.array(output_values, type=pa.float64()),
        UNIT_COLUMN: unit_column(table, count=groups.count),
        **facet_columns(resolved.facet_selections, count=groups.count),
    }
    # A grouped stat_var, time_period or role is the group's own key.
    for column in key_columns:
        columns[column] = groups.keys.column(column)
    output_frame = frame_keeping_role_types(columns, role_keys=role_in_group_by, source=frame)

    # `expected_cell_count`'s own cartesian product is computed over the
    # rows before `exclude_unitless` drops any, not `table` above, which
    # has already had them removed. A row `exclude_unitless` excludes still
    # names a real group_by cell this grid expected to see. If excluding it
    # empties that cell's only row, the cell must count as skipped rather
    # than vanish from the grid's own denominator along with the row that
    # would have populated it (coverage accounting is this module's honesty
    # mechanism, see the module docstring). This uses
    # `table_before_unitless_exclusion` rather than a second
    # `narrow_and_resolve` call, because on the pinned path,
    # `pinned_unit`'s own comparison already drops an unbound row before
    # `exclude_unitless` gets the chance to, so a plain unpinned re-narrow
    # would silently agree with the very undercount this fix exists to close
    # on that path.
    pre_exclusion_table = table_before_unitless_exclusion(frame, pinned_unit=pinned_unit)
    expected_cell_count = math.prod(
        len(array_kernel("unique", pre_exclusion_table.column(column))) for column in group_by
    )
    present_cell_count = groups.count
    skipped_cell_count = max(expected_cell_count - present_cell_count, 0)
    realised_coverage_fraction = (
        present_cell_count / expected_cell_count if expected_cell_count else 1.0
    )

    roles = entity_roles(frame)
    grouped = tuple(column for column in roles if column in key_columns)
    pooled = tuple(column for column in roles if column not in key_columns)
    per_period = entity_counts(table, keys=(TIME_PERIOD_COLUMN,), entity_columns=pooled or grouped)
    return AggregateResult(
        frame=output_frame,
        resolved_unit=resolved.unit,
        facet_selections=resolved.facet_selections,
        warnings=resolved.warnings,
        realised_coverage_fraction=realised_coverage_fraction,
        skipped_cell_count=skipped_cell_count,
        expected_cell_count=expected_cell_count,
        present_cell_count=present_cell_count,
        group_summaries=tuple(sorted(summaries, key=lambda summary: summary.group_key)),
        collapsed_periods=tuple(members.periods),
        period_label=period_label,
        entity_count=(
            entity_counts(table, keys=(), entity_columns=grouped).get((), 0) if grouped else None
        ),
        period_entities=tuple((period, per_period[(period,)]) for period in members.periods),
        exclusions=exclusions,
        pooled_parties=summed_parties,
    )
