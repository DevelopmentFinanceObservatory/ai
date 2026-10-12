"""The `rank` engine: competition ranking of entities by a value, optionally
reduced over a calendar window first, over a frame's Arrow table. `rank` ranks what's
present, reports per-entity coverage in the trace with a digest warning
when it's unequal, and refuses below the coverage floor.

Reuses `analysis_mcp.engine.aggregate`'s own reduction internals directly --
`AGGREGATIONS` (the closed vocabulary, one source of truth), `_reduce` (the
reduction itself), `UnsupportedAggregationError`, and
`NonFiniteAggregateResultError` -- rather than redefining them, since
`rank`'s optional aggregation step *is* `aggregate`'s own reduction, just
windowed and followed by a sort. Ranking itself (a sort and a rank-number
assignment) is not a reduction and calls `_reduce` for none of it.

Coverage is per entity: the fraction of the window's own "period calendar"
(every distinct `time_period` any ranked entity contributes) that this
entity actually has a row for. This engine reports the *overall* grid's own
coverage -- every entity times every calendar period -- for the tool
wrapper to check against `settings.aggregate_coverage_floor`, the same
floor `aggregate` itself uses, and refuse (reusing `aggregate`'s own
`coverage_insufficient` shape) when it falls short.

Tie policy: competition ranking ("1224") -- entities sharing a value share a
rank number, and the next distinct value's rank skips ahead by the count of
entities that shared the previous one. The rank *number* never breaks a tie;
the *order* two tied entities are listed in does, deterministically, via
`dfo_frame_grammar.grammar.cell_sort_key` applied to the entity's own
`rank_by` values -- the frame grammar's own total ordering, not an
accident of dict iteration or insertion order. Ties, and this ordering, are
resolved per period under `per_period=True` below, independently of every
other period.

What the two modes are: `per_period=False` (the default) collapses the
window into one ranking, one row per entity, whose own `value` is the
measured value. `per_period=True` keeps the window's
periods apart: each period is ranked on its own, and the output carries one
row per entity per period whose own `value` is that entity's rank
*position* in that period, not the measured value -- so a rank series can
be grown ("how has Kenya's rank changed since 2010"). A `limit` keeps the
measured value instead; see below. `aggregation` still
means "reduce to one value before ranking" in either mode; the grouping key
it reduces to changes from (entity) to (entity, period).

`limit` keeps every entity whose rank number is at most `limit`, within
each period under `per_period=True`. The rank number decides membership
and the tie-break order never does: entities sharing the boundary value
share its rank number, so all of them stay and a ranking can keep more
than `limit` entities. The cut runs after ranking and before the output frame
is built, because the frame grammar sorts the frame's rows on its grain
columns and the rank order exists only here. A limited ranking's output
`value` is the measured value in both modes: "the top 5 donors each year"
asks for their amounts, and the kept rows already say who placed. Coverage
and `RankResult.entity_ranks` describe the whole ranked population; the
input rows behind the cells the limit leaves out are recorded as a
`beyond_limit` exclusion, which names only the entities left out of every
period, since an entity kept in one period is still in the frame.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import cast

import pyarrow as pa
from common.time_grain import span_label
from dfo_frame_grammar import Frame, array_kernel
from dfo_frame_grammar import grammar as frame_grammar
from dfo_frame_grammar.grammar import (
    STAT_VAR_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    UnitCell,
)

from analysis_mcp.engine.aggregate import (
    AGGREGATIONS,
    NonFiniteAggregateResultError,
    UnsupportedAggregationError,
    _reduce,
    refuse_sum_over_composed_unit,
)
from analysis_mcp.engine.columnar import (
    GroupMembers,
    Groups,
    distinct,
    group_members,
    group_rows,
    in_periods,
    repeated,
    unit_column,
)
from analysis_mcp.engine.common import (
    Exclusions,
    FacetSelection,
    available_entity_columns,
    combine_exclusions,
    facet_columns,
    frame_keeping_role_types,
    narrow_and_resolve,
    record_counted_exclusion,
    record_exclusion,
    require_one_place_level,
    require_single_stat_var,
    role_columns,
    table_before_unitless_exclusion,
    validate_entity_columns,
    without_party_totals,
)

# One entity's or one cell's member rows: their values and `GroupMembers`
# period codes, in row order.
type _Members = tuple[list[float], list[int]]


class RankNoRowsInWindowError(Exception):
    """No row falls inside `[window_start, window_end]` (or, with no window,
    the frame carries no rows at all) -- there is nothing to rank."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class RankAmbiguousEntityError(Exception):
    """`aggregation` was left `None` (no reduction requested), but at least
    one entity's own rows within the window number more than one -- there is
    no single, unambiguous value to rank that entity by without reducing.
    """

    def __init__(self, *, rank_by: Sequence[str], available: Sequence[str]) -> None:
        self.rank_by = tuple(rank_by)
        self.available = tuple(available)
        super().__init__(
            f"rank_by {self.rank_by!r} does not uniquely identify one row per entity; "
            "supply `aggregation` to reduce first"
        )


@dataclass(frozen=True, kw_only=True)
class EntityRank:
    """One entity's own ranked result, in *ranked order* -- unlike
    `analysis_mcp.engine.aggregate.GroupSummary`, position here is load-
    bearing: `RankResult.entity_ranks` is the tie-broken order itself, not a
    canonical-but-arbitrary ordering a caller must re-sort.

    `time_period` is the label this entity's own output row carries. It is
    the period it was ranked in under `per_period=True`, else the window's
    observed extent (`span_label` over `RankResult.periods_in_window`),
    shared by every entity. It defaults to the empty string only so a
    hand-built `EntityRank` in a test that doesn't care about it (e.g.
    `digest_corpus.py`) need not set it.
    """

    entity_key: tuple[tuple[str, str], ...]
    value: float
    rank: int
    row_count: int
    coverage_fraction: float
    time_period: str = ""


@dataclass(frozen=True, kw_only=True)
class RankResult:
    """Everything `rank_tool.py` needs to mint identity, write the store,
    and render the digest.

    `entity_ranks` is the whole ranking; `kept_ranks` is the part of it
    the output frame holds, every entity within `limit`. `ordinal` is
    whether the output frame's `value` is the rank position.
    """

    frame: Frame
    resolved_unit: UnitCell | None
    facet_selections: tuple[FacetSelection, ...]
    warnings: tuple[str, ...]
    aggregation: str | None
    window_start: str | None
    window_end: str | None
    periods_in_window: tuple[str, ...]
    expected_cell_count: int
    present_cell_count: int
    overall_coverage_fraction: float
    entity_ranks: tuple[EntityRank, ...]
    per_period: bool = False
    limit: int | None = None
    exclusions: Exclusions = field(default_factory=Exclusions)

    @property
    def kept_ranks(self) -> tuple[EntityRank, ...]:
        """The `entity_ranks` the output frame holds."""
        return _within_limit(self.entity_ranks, self.limit)

    @property
    def ordinal(self) -> bool:
        """Whether the output frame's `value` is the rank position."""
        return _writes_positions(per_period=self.per_period, limit=self.limit)


def _writes_positions(*, per_period: bool, limit: int | None) -> bool:
    """Whether a rank's output `value` is the rank position: per period
    with no `limit`. Every other rank writes the measured value."""
    return per_period and limit is None


def _within_limit(entity_ranks: Sequence[EntityRank], limit: int | None) -> tuple[EntityRank, ...]:
    """Every entry of `entity_ranks` whose rank number is at most `limit`,
    in ranked order; all of them when `limit` is `None`."""
    return tuple(er for er in entity_ranks if limit is None or er.rank <= limit)


def _beyond_limit_exclusion(
    exclusions: Exclusions,
    *,
    frame: Frame,
    entities: Groups,
    members: GroupMembers,
    entity_keys: Sequence[tuple[str, ...]],
    rank_by: Sequence[str],
    kept: Sequence[EntityRank],
    per_period: bool,
) -> Exclusions:
    """`exclusions`, plus the `beyond_limit` record for the input rows that
    fed a ranked cell `kept` leaves out. A cell is an entity, or an
    (entity, period) pair under `per_period`, so an entity kept in one
    period still loses its rows in the periods where it ranked beyond the
    limit, while the frame still holds it. The record names, on each
    `rank_by` column, only the entities `kept` leaves out whole.
    `entities` partitions the ranked rows, `members` holds them in group
    order, and `entity_keys[g]` is group `g`'s key.

    The count comes from the grouping and copies no row: an entity left
    out whole counts its group's size, and only a kept entity's period
    codes are read, to count the cells it lost under `per_period`."""
    kept_entities = {tuple(value for _, value in er.entity_key) for er in kept}
    kept_cells = {(tuple(value for _, value in er.entity_key), er.time_period) for er in kept}
    row_count = 0
    for group, key in enumerate(entity_keys):
        start, end = entities.offsets[group], entities.offsets[group + 1]
        if key not in kept_entities:
            row_count += end - start
        elif per_period:
            row_count += sum(
                (key, members.periods[code]) not in kept_cells
                for code in members.period_codes[start:end]
            )

    def left_out(position: int) -> Iterator[str]:
        return (key[position] for key in entity_keys if key not in kept_entities)

    return record_counted_exclusion(
        exclusions,
        reason="beyond_limit",
        row_count=row_count,
        entities={
            column: left_out(rank_by.index(column))
            for column in role_columns(frame)
            if column in rank_by
        },
    )


def _tie_break_key(entity_key: tuple[str, ...]) -> tuple[object, ...]:
    """The frame grammar's own total ordering over an entity's `rank_by`
    values -- what breaks a tie in *order*, never in rank number."""
    return tuple(frame_grammar.cell_sort_key(value) for value in entity_key)


def _year_in_window(time_period: str, *, start_year: int, end_year: int) -> bool:
    """Whether every calendar year `time_period` covers falls inside
    `[start_year, end_year]`, compared as numbers.

    `window_start`/`window_end` are whole calendar years (`RankOpSpec`'s
    `^[0-9]{4}$` pattern). `time_period` is a point at any grain or an
    interval, read by `dfo_frame_grammar.grammar.parse_time_period`. A
    monthly or daily point counts in the year it falls in, so `2020-03` is
    inside a 2020 window. An interval is inside only when both its start
    and end years are, so `2018/2022` is inside a 2018-2022 window and
    outside a 2020-2024 one, since a row labelled with a span the window
    cuts would carry years the caller left out.
    """
    cell = frame_grammar.parse_time_period(time_period)
    years = (int(cell.start[: len("YYYY")]), int(cell.end[: len("YYYY")]))
    return all(start_year <= year <= end_year for year in years)


def _window_mask(table: pa.Table, *, start_year: int, end_year: int) -> pa.Array:
    """Whether each row of `table` falls inside the window, by
    `_year_in_window` over each distinct `time_period`."""
    return in_periods(
        table,
        (
            period
            for period in distinct(table, TIME_PERIOD_COLUMN)
            if _year_in_window(period, start_year=start_year, end_year=end_year)
        ),
    )


def _competition_rank(
    values: dict[tuple[str, ...], float], *, ascending: bool
) -> list[tuple[tuple[str, ...], float, int]]:
    """`values` (a mapping from entity key to the value it's ranked by),
    ordered by competition rank ("1224"): the primary order is each key's
    own value (`ascending` picks the direction), the *order* two tied
    entities are listed in is `_tie_break_key`'s frame-grammar total
    ordering, and the rank *number* never breaks a tie -- see the module
    docstring's own tie policy.

    Returns `(key, value, rank)` triples in ranked order; `value` is
    normalised (`+ 0.0`) to collapse a signed zero, the same engine-boundary
    invariant `run_rank` states for the collapsing path this factors out of.
    Shared by that path and `_rank_per_period` below, one period's own
    `values` at a time, so both run the identical sort and rank-assignment
    logic.
    """
    ordered_keys = sorted(
        values,
        key=lambda key: (
            values[key] if ascending else -values[key],
            _tie_break_key(key),
        ),
    )
    ranked: list[tuple[tuple[str, ...], float, int]] = []
    current_rank = 1
    previous_value: float | None = None
    for position, key in enumerate(ordered_keys, start=1):
        value = values[key] + 0.0
        if previous_value is not None and value != previous_value:
            current_rank = position
        ranked.append((key, value, current_rank))
        previous_value = value
    return ranked


def _reduced_cell_value(
    *,
    members: _Members,
    entity_key_map: dict[str, str],
    rank_by: Sequence[str],
    available: Sequence[str],
    aggregation: str | None,
) -> float:
    """One cell's own ranked value: its one member's raw value when
    `aggregation` is `None` and there's exactly one member, `_reduce`'s
    output otherwise. Raises `RankAmbiguousEntityError` for more than one row with
    no `aggregation`, and `NonFiniteAggregateResultError` for a reduction
    that overflows or isn't finite -- the identical two refusals `run_rank`'s
    own collapsing path raises, now shared with `_rank_per_period`'s own
    per-(entity, period) cells.
    """
    values, period_codes = members
    if aggregation is None:
        if len(values) > 1:
            raise RankAmbiguousEntityError(rank_by=rank_by, available=available)
        return values[0]
    try:
        reduced, _ = _reduce(values, period_codes, aggregation=aggregation)
    except OverflowError as exc:
        raise NonFiniteAggregateResultError(
            group_key=entity_key_map, aggregation=aggregation
        ) from exc
    if not math.isfinite(reduced):
        raise NonFiniteAggregateResultError(group_key=entity_key_map, aggregation=aggregation)
    return reduced


def _rank_per_period(
    *,
    frame: Frame,
    entity_members: dict[tuple[str, ...], _Members],
    periods: Sequence[str],
    entity_coverage: dict[tuple[str, ...], float],
    rank_by: Sequence[str],
    aggregation: str | None,
    ascending: bool,
) -> list[EntityRank]:
    """Rank every period apart, rather than collapsing the window into one
    ranking -- see the module docstring's "What the two modes are" and
    `run_rank`'s own per-period paragraph. The grouping key becomes
    (entity, period): each period's own cell is optionally reduced by
    `aggregation` (needed only when a cell carries more than one row --
    with none, `_reduced_cell_value` raises `RankAmbiguousEntityError`
    exactly as the collapsing path does), and competition ranking
    (`_competition_rank`) runs independently per period, so a tie inside
    one period cannot affect another's. Row counts and `entity_coverage`
    stay entity-wide -- unaffected by this per-period grouping, since
    coverage is still measured against the whole window, not one period at
    a time. `periods` names the period codes in `entity_members`.

    Returns one `EntityRank` per (entity, period), whose `time_period` is
    the period it was ranked in.
    """
    cell_members: dict[int, dict[tuple[str, ...], _Members]] = {}
    for key, (values, period_codes) in entity_members.items():
        for value, code in zip(values, period_codes, strict=True):
            cell_values, cell_codes = cell_members.setdefault(code, {}).setdefault(key, ([], []))
            cell_values.append(value)
            cell_codes.append(code)

    available = available_entity_columns(frame)
    entity_ranks: list[EntityRank] = []
    for code in sorted(cell_members):
        period = periods[code]
        ranked_values: dict[tuple[str, ...], float] = {}
        for key, members in cell_members[code].items():
            ranked_values[key] = _reduced_cell_value(
                members=members,
                entity_key_map=dict(zip(rank_by, key, strict=True)),
                rank_by=rank_by,
                available=available,
                aggregation=aggregation,
            )
        for key, value, rank_number in _competition_rank(ranked_values, ascending=ascending):
            entity_ranks.append(
                EntityRank(
                    entity_key=tuple(zip(rank_by, key, strict=True)),
                    value=value,
                    rank=rank_number,
                    row_count=len(entity_members[key][0]),
                    coverage_fraction=entity_coverage[key],
                    time_period=period,
                )
            )
    return entity_ranks


def run_rank(
    frame: Frame,
    *,
    rank_by: Sequence[str],
    aggregation: str | None,
    window_start: str | None,
    window_end: str | None,
    ascending: bool,
    pinned_unit: UnitCell | None = None,
    per_period: bool = False,
    limit: int | None = None,
) -> RankResult:
    """Rank entities identified by `rank_by`, optionally reducing each
    entity's own rows with `aggregation` first. See the module docstring for
    the coverage and tie policies; see each raised exception's own docstring
    for its refusal condition.

    `per_period=False` (the default) collapses the window's rows into one
    ranking, one row per entity, whose own `value` is the measured value.
    `per_period=True` keeps the window's
    periods apart -- each is ranked on its own (`_rank_per_period`), and the
    output frame's own `value` column carries the rank *position* instead,
    below, where the two modes fork. `EntityRank.value` itself is always
    the measured value, in either mode -- only the output frame's own
    column changes meaning, which is why the ordinal marker `rank_tool.py`
    writes travels with the frame, not with `EntityRank`.

    `limit` cuts the output frame to `RankResult.kept_ranks`, and its
    `value` column carries the measured value in both modes; see the module
    docstring.
    """
    if aggregation is not None and aggregation not in AGGREGATIONS:
        raise UnsupportedAggregationError(aggregation)

    stat_var = require_single_stat_var(frame)
    rank_by = validate_entity_columns(frame, rank_by)
    require_one_place_level(frame, rank_by)
    # A total of other parties is neither ranked beside them nor summed into
    # an entity's value, whether its role is ranked or pooled. A ranked role
    # holding only totals ("which regions receive the most?") ranks them.
    frame, party_exclusions = without_party_totals(frame, role_columns(frame), ranked=rank_by)
    # `exclude_unitless=True`: `resolved.unit` labels every ranked entity's
    # output row alike. An entity whose rows leave `unit` unbound is excluded;
    # it must not be ranked alongside entities denominated in a real unit.
    table, resolved, exclusions = narrow_and_resolve(
        frame, pinned_unit=pinned_unit, exclude_unitless=True
    )
    exclusions = combine_exclusions(party_exclusions, exclusions)
    refuse_sum_over_composed_unit(aggregation, resolved.unit)
    pre_exclusion_table = table_before_unitless_exclusion(frame, pinned_unit=pinned_unit)

    if window_start is not None and window_end is not None:
        start_year, end_year = int(window_start), int(window_end)
        in_window = _window_mask(table, start_year=start_year, end_year=end_year)
        exclusions = record_exclusion(
            exclusions,
            frame=frame,
            reason="outside_window",
            dropped=table.filter(array_kernel("invert", in_window)),
        )
        table = table.filter(in_window)
        pre_exclusion_table = pre_exclusion_table.filter(
            _window_mask(pre_exclusion_table, start_year=start_year, end_year=end_year)
        )

    if not table.num_rows:
        reason = (
            "no rows fall inside the requested window"
            if window_start is not None
            else "the input frame carries no rows"
        )
        raise RankNoRowsInWindowError(reason=reason)

    # A column named twice in `rank_by` groups once.
    entity_columns = tuple(dict.fromkeys(rank_by))
    entities = group_rows(table, entity_columns)
    members = group_members(table, entities)
    periods_in_window = tuple(members.periods)
    key_columns = [
        cast("list[str]", entities.keys.column(column).to_pylist()) for column in rank_by
    ]
    entity_keys = [
        tuple(values[entity] for values in key_columns) for entity in range(entities.count)
    ]
    entity_members = {key: members.of(entity) for entity, key in enumerate(entity_keys)}

    # `pre_exclusion_period_count` is computed before the per-entity coverage
    # loop, using the pre-exclusion row set. This ensures all entities are
    # measured against the same denominator (the same grid), rather than each
    # entity's per-entity `coverage_fraction` shrinking to its own excluded
    # subset's denominator.
    pre_exclusion_period_count = len(
        array_kernel("unique", pre_exclusion_table.column(TIME_PERIOD_COLUMN))
    )

    # `entity_coverage`/`entity_present_periods` are entity-wide regardless
    # of `per_period` -- coverage is always measured against the whole
    # window, never one period at a time (see the module docstring's own
    # coverage paragraph).
    entity_coverage: dict[tuple[str, ...], float] = {}
    entity_present_periods: dict[tuple[str, ...], tuple[str, ...]] = {}
    for key, (_, period_codes) in entity_members.items():
        present_periods = tuple(members.periods[code] for code in sorted(set(period_codes)))
        entity_present_periods[key] = present_periods
        entity_coverage[key] = len(present_periods) / pre_exclusion_period_count

    # `expected_cell_count`'s own grid is computed over the rows before
    # `exclude_unitless` drops any -- not `entity_members`/
    # `periods_in_window` above, both derived from the already-excluded
    # `table` -- the same "an excluded row must still count as an
    # expected-but-missing cell" fix `aggregate.py` applies to its own
    # coverage grid, and for the identical reason: a wholly-unitless entity
    # would otherwise vanish from both the ranking *and* the coverage
    # denominator together, reading as 100% coverage of a grid that quietly
    # shrank to fit. `pre_exclusion_period_count` was already computed
    # above, before the per-entity coverage loop, which needs the identical
    # denominator.
    pre_exclusion_entity_count = group_rows(pre_exclusion_table, entity_columns).count
    expected_cell_count = pre_exclusion_entity_count * pre_exclusion_period_count
    present_cell_count = sum(len(periods) for periods in entity_present_periods.values())
    overall_coverage_fraction = (
        present_cell_count / expected_cell_count if expected_cell_count else 1.0
    )

    entity_ranks: list[EntityRank]
    if per_period:
        entity_ranks = _rank_per_period(
            frame=frame,
            entity_members=entity_members,
            periods=members.periods,
            entity_coverage=entity_coverage,
            rank_by=rank_by,
            aggregation=aggregation,
            ascending=ascending,
        )
    else:
        entity_values: dict[tuple[str, ...], float] = {}
        available = available_entity_columns(frame)
        for key, entity_member_values in entity_members.items():
            entity_values[key] = _reduced_cell_value(
                members=entity_member_values,
                entity_key_map=dict(zip(rank_by, key, strict=True)),
                rank_by=rank_by,
                available=available,
                aggregation=aggregation,
            )

        # Every row carries the observed extent of the window
        # (`span_label`), such as `2018/2022` over points 2018 to 2022, the
        # point itself over one period, and the input's own span over a span
        # frame.
        window_label = span_label(periods_in_window)
        entity_ranks = []
        for key, value, rank_number in _competition_rank(entity_values, ascending=ascending):
            entity_ranks.append(
                EntityRank(
                    entity_key=tuple(zip(rank_by, key, strict=True)),
                    value=value,
                    rank=rank_number,
                    row_count=len(entity_members[key][0]),
                    coverage_fraction=entity_coverage[key],
                    time_period=window_label,
                )
            )

    warnings = list(resolved.warnings)
    if len({round(er.coverage_fraction, 9) for er in entity_ranks}) > 1:
        warnings.append(
            "ranked entities have unequal coverage of the window's own periods -- some "
            "entities' values rest on sparser data than others."
        )

    kept = _within_limit(entity_ranks, limit)
    if limit is not None:
        exclusions = _beyond_limit_exclusion(
            exclusions,
            frame=frame,
            entities=entities,
            members=members,
            entity_keys=entity_keys,
            rank_by=rank_by,
            kept=kept,
            per_period=per_period,
        )

    # The frame's own `value` is the rank position in per-period mode with
    # no `limit` and the measured value otherwise -- `EntityRank.value` is
    # always the measured value; only the frame's own column changes
    # meaning, which is why the ordinal marker (`rank_tool.py`,
    # `DerivedMarker.ordinal`) travels with the frame rather than with
    # `EntityRank`. The frame's own `unit` follows the identical split: a
    # rank position has no unit.
    ordinal = _writes_positions(per_period=per_period, limit=limit)
    count = len(kept)
    columns: dict[str, pa.Array | pa.ChunkedArray] = {
        STAT_VAR_COLUMN: repeated(stat_var, count),
        TIME_PERIOD_COLUMN: pa.array([er.time_period for er in kept], type=pa.string()),
        VALUE_COLUMN: pa.array(
            [float(er.rank) if ordinal else er.value for er in kept],
            type=pa.float64(),
        ),
        **facet_columns(resolved.facet_selections, count=count),
    }
    if not ordinal:
        columns[UNIT_COLUMN] = unit_column(table, count=count)
    for position, column in enumerate(rank_by):
        columns[column] = pa.array([er.entity_key[position][1] for er in kept], type=pa.string())
    output_frame = frame_keeping_role_types(columns, role_keys=rank_by, source=frame)

    return RankResult(
        frame=output_frame,
        resolved_unit=resolved.unit,
        facet_selections=resolved.facet_selections,
        warnings=tuple(warnings),
        aggregation=aggregation,
        per_period=per_period,
        limit=limit,
        window_start=window_start,
        window_end=window_end,
        periods_in_window=periods_in_window,
        expected_cell_count=expected_cell_count,
        present_cell_count=present_cell_count,
        overall_coverage_fraction=overall_coverage_fraction,
        entity_ranks=tuple(entity_ranks),
        exclusions=exclusions,
    )
