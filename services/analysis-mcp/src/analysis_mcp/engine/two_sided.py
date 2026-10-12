"""What every numerator-over-denominator verb does before and after its own
pairing rules: sum one side per (key, period), map `denominator_by` onto the
denominator's own columns, match each paired key's periods with its
denominator group's and record the periods left out, and divide one matched
cell.

Which units, facets and keys may pair across the two sides, and what a key
sharing no period means, is each verb's own rule and stays in its own engine module
(`analysis_mcp.engine.share_of`, `analysis_mcp.engine.quotient_of`). The
exceptions here carry no verb's name, so each verb surfaces them as its own
refusal.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, cast

from dfo_frame_grammar import Frame
from dfo_frame_grammar.grammar import TIME_PERIOD_COLUMN, VALUE_COLUMN, ComposedUnit, UnitCell

from analysis_mcp.engine.aggregate import ComposedUnitSumError
from analysis_mcp.engine.columnar import group_rows
from analysis_mcp.engine.common import (
    ExclusionReason,
    Exclusions,
    FacetSelection,
    combine_exclusions,
    narrow_and_resolve,
    record_counted_exclusion,
    require_one_place_level,
    require_single_stat_var,
    role_columns,
    validate_entity_columns,
    without_party_totals,
)

type Side = Literal["numerator", "denominator"]
# One key's own `by` values, in `numerator_by`'s order; `()` for an un-keyed
# call or a pooled denominator.
type Key = tuple[str, ...]
# One key as its `(column, value)` pairs, in `numerator_by`'s order; `()` for
# an un-keyed call or a pooled denominator.
type KeyPairs = tuple[tuple[str, str], ...]
# One key and the sorted periods it holds on one side alone.
type KeyPeriods = tuple[KeyPairs, tuple[str, ...]]


class NonFiniteCellError(Exception):
    """A period's denominator summed to exactly zero (`reason="zero
    denominator"`), or a side's sum or the ratio overflowed
    (`reason="overflow"`)."""

    def __init__(self, *, time_period: str, reason: str) -> None:
        self.time_period = time_period
        self.reason = reason
        super().__init__(f"non-finite result for period {time_period!r}: {reason}")


class DenominatorByPartialError(Exception):
    """`denominator_by` is non-empty but does not cover every `numerator_by`
    column -- a partial mapping has no defined pairing for the column(s) it
    leaves out. `column` is the first `numerator_by` column with no
    `denominator_by` entry (sorted, for a deterministic message);
    `denominator_by_keys` are the keys `denominator_by` does carry.
    """

    def __init__(
        self, *, column: str, numerator_by: Sequence[str], denominator_by_keys: Sequence[str]
    ) -> None:
        self.column = column
        self.numerator_by = tuple(numerator_by)
        self.denominator_by_keys = tuple(denominator_by_keys)
        super().__init__(
            f"denominator_by does not cover numerator_by column {column!r}; "
            f"denominator_by names: {self.denominator_by_keys!r}"
        )


@dataclass(frozen=True, kw_only=True)
class SideSums:
    """One side's own summed values and what its slice resolved to.

    `by_key` maps `by`'s own key tuple to that key's own
    `{time_period: summed_value}` series -- a key with no rows at all is
    simply absent, never present with an empty series; the caller decides
    what an absent key means for its own side. `row_counts` has the same
    shape and holds how many rows each sum read. `stat_var` is the whole
    frame's sole `stat_var`.
    """

    stat_var: str
    by_key: dict[tuple[str, ...], dict[str, float]]
    row_counts: dict[tuple[str, ...], dict[str, int]]
    unit: UnitCell | None
    facet_selections: tuple[FacetSelection, ...]
    warnings: tuple[str, ...]
    exclusions: Exclusions


def sum_side(
    frame: Frame,
    *,
    side: Literal["numerator", "denominator"],
    entity: Mapping[str, str],
    by: Sequence[str],
    pinned_unit: UnitCell | None,
) -> SideSums:
    """Narrow `frame` to `entity`'s own rows, resolve unit/facets over
    exactly that slice, then sum (`math.fsum`) every remaining row sharing
    both `by`'s own key values and a `time_period` -- one summed value per
    (key, period), never a per-row value, so more than one matching row in
    one cell is a deliberate total, not an ambiguity. `by=()` computes one
    key, the empty tuple, covering every remaining row. Refuses
    (`ComposedUnitSumError`, naming `side`) when the slice resolves to a
    composed unit and some cell holds more than one row: figures per head
    have no total, and a cell of one row adds nothing. Refuses
    (`UnknownEntityColumnError`) if `entity` or `by` names a column this
    frame has no grain for, and (`PlaceLevelsMixedError`) if `frame`'s
    places share no level, whether this side sums across them or keys by them,
    as the harness's `planner.shapes` refuses either side of a share over such a frame.
    A flow party that totals other parties is left out of the side, whether
    the side sums across its role or keys by it, unless `entity` binds that
    role (`without_party_totals`).
    """
    stat_var = require_single_stat_var(frame)
    validate_entity_columns(frame, (*entity, *by))
    require_one_place_level(frame, role_columns(frame))
    frame, party_exclusions = without_party_totals(
        frame, (column for column in role_columns(frame) if column not in entity), side=side
    )
    # `exclude_unitless=True`: a row leaving `unit` unbound is excluded from
    # this side's sum, consistent with the policy `aggregate` applies.
    table, resolved, exclusions = narrow_and_resolve(
        frame, entity=entity, pinned_unit=pinned_unit, exclude_unitless=True, side=side
    )
    exclusions = combine_exclusions(party_exclusions, exclusions)

    # Groups come in order of their first row, so keys land in this dict in
    # order of their first row and each key's periods in order of theirs:
    # the order an overflowing sum below is found in.
    groups = group_rows(table, (*by, TIME_PERIOD_COLUMN))
    values = cast("list[float]", table.column(VALUE_COLUMN).take(groups.indices).to_pylist())
    key_columns = [cast("list[str]", groups.keys.column(column).to_pylist()) for column in by]
    periods = cast("list[str]", groups.keys.column(TIME_PERIOD_COLUMN).to_pylist())
    by_key_and_period: dict[tuple[str, ...], dict[str, list[float]]] = {}
    for group, (start, end) in enumerate(pairwise(groups.offsets)):
        key = tuple(column[group] for column in key_columns)
        by_key_and_period.setdefault(key, {})[periods[group]] = values[start:end]

    summed: dict[tuple[str, ...], dict[str, float]] = {}
    row_counts = {
        key: {period: len(values) for period, values in by_period.items()}
        for key, by_period in by_key_and_period.items()
    }
    if isinstance(resolved.unit, ComposedUnit) and any(
        count > 1 for by_period in row_counts.values() for count in by_period.values()
    ):
        raise ComposedUnitSumError(resolved.unit, side=side)
    for key, by_period in by_key_and_period.items():
        summed_periods: dict[str, float] = {}
        for period, values in by_period.items():
            try:
                # `math.fsum` raises `OverflowError` for a sum whose exact
                # value exceeds the finite double range, rather than
                # returning `inf` the way naive `+` accumulation would --
                # the same guard `aggregate.run_aggregate` and
                # `rank.run_rank` already apply to their own `math.fsum`
                # reduction.
                summed_periods[period] = math.fsum(values)
            except OverflowError as exc:
                raise NonFiniteCellError(time_period=period, reason="overflow") from exc
        summed[key] = summed_periods

    return SideSums(
        stat_var=stat_var,
        by_key=summed,
        row_counts=row_counts,
        unit=resolved.unit,
        facet_selections=resolved.facet_selections,
        warnings=tuple(resolved.warnings),
        exclusions=exclusions,
    )


def denominator_key_columns(
    *, numerator_by: Sequence[str], denominator_by: Mapping[str, str]
) -> tuple[str, ...]:
    """The denominator's own column for each `numerator_by` column, in
    `numerator_by`'s order, so a numerator key tuple and its matching
    denominator key tuple carry identical values. An empty `denominator_by`
    means a pooled denominator and maps to `()`. Refuses
    (`DenominatorByPartialError`) when a non-empty `denominator_by` leaves a
    `numerator_by` column out.
    """
    if not denominator_by:
        return ()
    missing = sorted(set(numerator_by) - set(denominator_by))
    if missing:
        raise DenominatorByPartialError(
            column=missing[0],
            numerator_by=numerator_by,
            denominator_by_keys=tuple(sorted(denominator_by)),
        )
    return tuple(denominator_by[column] for column in numerator_by)


def divide_cell(*, numerator_value: float, denominator_value: float, time_period: str) -> float:
    """One matched (key, period)'s own numerator over its denominator.
    Refuses (`NonFiniteCellError`) on a denominator of exactly zero, before
    any division is attempted, and on a ratio that overflows.
    """
    if denominator_value == 0.0:
        raise NonFiniteCellError(time_period=time_period, reason="zero denominator")
    quotient = numerator_value / denominator_value
    if not math.isfinite(quotient):
        raise NonFiniteCellError(time_period=time_period, reason="overflow")
    # Signed zero is normalized at the engine boundary. This is an
    # engine-level invariant this toolkit states for every verb
    # (see `engine/growth.py`'s own `growth_rate += 0.0`), applied
    # so a `-0.0` cannot persist into the stored output frame where
    # it might be re-chained to another verb.
    return quotient + 0.0


@dataclass(frozen=True, kw_only=True)
class PeriodPairing:
    """How each numerator key that pairs with a denominator group meets it
    in time.

    `shared` maps each paired key, in the order given, to the sorted
    periods it shares with its group, empty when it shares none.
    `numerator_periods` and `denominator_periods` are every period the
    paired keys and their groups hold, and `matched_periods` every period
    some key shares with its group, all sorted. A side's coverage fraction
    counts (key, period) cells: those a key shares with its group, over
    those the side holds for the paired keys. A pooled group counts each of
    its periods once, and an un-keyed call has one cell per period.

    `numerator_unmatched` and `denominator_unmatched` are the (key or group,
    period) cells a side holds outside every shared period, which a verb
    records as `period_unmatched` (`record_unmatched_periods`).
    `numerator_only_periods` and `denominator_only_periods` are, for each
    paired key of a keyed call sharing at least one period, the periods it
    holds on that side alone; a key holding none is absent, and a pooled
    denominator has no periods of one key's own.
    """

    shared: dict[Key, tuple[str, ...]]
    numerator_periods: tuple[str, ...]
    denominator_periods: tuple[str, ...]
    matched_periods: tuple[str, ...]
    numerator_coverage_fraction: float
    denominator_coverage_fraction: float
    numerator_unmatched: tuple[tuple[Key, str], ...]
    denominator_unmatched: tuple[tuple[Key, str], ...]
    numerator_only_periods: tuple[KeyPeriods, ...]
    denominator_only_periods: tuple[KeyPeriods, ...]


def pair_periods(
    numerator: SideSums,
    denominator: SideSums,
    *,
    numerator_by: Sequence[str],
    paired_keys: Sequence[Key],
    pooled: bool,
) -> PeriodPairing:
    """Match each of `paired_keys` (numerator keys holding a denominator
    group, or every key when `pooled`) with its group over the periods both
    report. A key absent from a side's sums holds no period there.
    """
    paired_groups: list[Key] = [()] if pooled else list(paired_keys)
    shared: dict[Key, tuple[str, ...]] = {}
    denominator_shared: set[tuple[Key, str]] = set()
    numerator_unmatched: list[tuple[Key, str]] = []
    numerator_only: list[KeyPeriods] = []
    denominator_only: list[KeyPeriods] = []
    for key in paired_keys:
        group = () if pooled else key
        own = numerator.by_key.get(key, {})
        other = denominator.by_key.get(group, {})
        periods = tuple(sorted(own.keys() & other.keys()))
        shared[key] = periods
        denominator_shared.update((group, period) for period in periods)
        numerator_unmatched.extend((key, period) for period in own if period not in other)
        if periods and numerator_by:
            key_pairs = tuple(zip(numerator_by, key, strict=True))
            one_sided = [(numerator_only, own, other)]
            if not pooled:
                one_sided.append((denominator_only, other, own))
            for collected, held, against in one_sided:
                alone = tuple(sorted(held.keys() - against.keys()))
                if alone:
                    collected.append((key_pairs, alone))

    numerator_cells = sum(len(numerator.by_key.get(key, {})) for key in paired_keys)
    denominator_cells = sum(len(denominator.by_key.get(group, {})) for group in paired_groups)
    matched_cells = sum(len(periods) for periods in shared.values())
    return PeriodPairing(
        shared=shared,
        numerator_periods=tuple(
            sorted({period for key in paired_keys for period in numerator.by_key.get(key, {})})
        ),
        denominator_periods=tuple(
            sorted(
                {period for group in paired_groups for period in denominator.by_key.get(group, {})}
            )
        ),
        matched_periods=tuple(sorted({period for _, period in denominator_shared})),
        numerator_coverage_fraction=matched_cells / numerator_cells if numerator_cells else 0.0,
        denominator_coverage_fraction=(
            len(denominator_shared) / denominator_cells if denominator_cells else 0.0
        ),
        numerator_unmatched=tuple(numerator_unmatched),
        denominator_unmatched=tuple(
            (group, period)
            for group in paired_groups
            for period in denominator.by_key.get(group, {})
            if (group, period) not in denominator_shared
        ),
        numerator_only_periods=tuple(numerator_only),
        denominator_only_periods=tuple(denominator_only),
    )


def record_cells(
    exclusions: Exclusions,
    *,
    reason: ExclusionReason,
    side: Side,
    sums: SideSums,
    columns: Sequence[str],
    cells: Iterable[tuple[Key, str]],
) -> Exclusions:
    """`exclusions`, plus one record for `side`'s (key, period) `cells` under
    `reason`: the rows `sums` read for them, and each of `columns`' values
    across their keys. Unchanged when `cells` is empty."""
    cells = list(cells)
    return record_counted_exclusion(
        exclusions,
        reason=reason,
        row_count=sum(sums.row_counts[key][period] for key, period in cells),
        entities={column: [key[index] for key, _ in cells] for index, column in enumerate(columns)},
        side=side,
    )


def record_unmatched_periods(
    exclusions: Exclusions,
    pairing: PeriodPairing,
    *,
    numerator: SideSums,
    denominator: SideSums,
    numerator_by: Sequence[str],
    denominator_columns: Sequence[str],
) -> Exclusions:
    """`exclusions`, plus a `period_unmatched` record for each side's cells
    outside every shared period, the numerator's first."""
    exclusions = record_cells(
        exclusions,
        reason="period_unmatched",
        side="numerator",
        sums=numerator,
        columns=numerator_by,
        cells=pairing.numerator_unmatched,
    )
    return record_cells(
        exclusions,
        reason="period_unmatched",
        side="denominator",
        sums=denominator,
        columns=denominator_columns,
        cells=pairing.denominator_unmatched,
    )
