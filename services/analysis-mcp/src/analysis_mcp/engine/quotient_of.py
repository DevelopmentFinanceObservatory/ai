"""The `quotient_of` engine: one slice divided by another per key and
period, in the unit the two sides compose.

Each side is summed as `share_of` sums it (`analysis_mcp.engine.two_sided.
sum_side`): `frame_ref` (or `denominator_frame_ref`) narrowed by an `entity`
binding, unit and facets resolved over that slice, then one `math.fsum` per
(key, period). `numerator_by` and `denominator_by` key the sum and pair the
sides by column name as they do for `share_of`; an empty `denominator_by` is
one pooled denominator every numerator key divides by.

Units: any pair of sides divides. Every output row carries
`ComposedUnit(numerator part, denominator part)`, where a side's part is its
resolved unit, or `StatVarPart(<the side's StatVar>)` when none of its rows
declares one (GDP in `dcid:USDollar` over `dcid:Count_Person` reads "US
dollars per person"). A composed unit has one level, so a side whose
resolved unit is itself composed refuses (`QuotientOfSideUnitUnsupportedError`).

Facets: a quotient across two datasets differs in `provenance` and
`measurementMethod` by construction, so an axis the sides disagree on, or
that one side alone declares, is a warning. The result carries an axis only
where both sides declare the same value (`analysis_mcp.engine.common.
agreed_facets`, `sum_of`'s policy).

Periods and keys follow the proceeding side of the partial-overlap policy in
`analysis_mcp.engine.common`. Each numerator key divides over the periods it
shares with its denominator group, and every (key, period) a paired key or
its group holds outside that intersection is recorded as a
`period_unmatched` exclusion with its side. A numerator key with no
denominator group is left out and recorded as `unmatched_key`, the reason
`join` records for a key one side lacks. A matched cell whose denominator
sums to exactly zero has no quotient: it is left out and recorded as
`zero_denominator`, and nothing is divided by zero.

A dropped key is a numerator key the result holds no row for. There are
three kinds, and `QuotientOfResult` names each: a key with no denominator
group, a key sharing no period with its group, and a key whose every
matched cell has a zero denominator.

Three calls have no row to return and refuse: both sides hold keys and none
pairs (`QuotientOfNoKeyMatchedError`); no key shares a period with its
group, or a side's slice holds no row
(`QuotientOfPeriodsDisjointError`); and every matched cell has a zero
denominator (`QuotientOfZeroDenominatorError`).

`stat_var` on every result row is the caller-supplied `stat_var`, the
operation's own ref, since one variable over another is a third.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import pyarrow as pa
from dfo_frame_grammar import Frame, unit_cell_array
from dfo_frame_grammar.grammar import (
    STAT_VAR_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    ComposedUnit,
    StatVarPart,
    UnitCell,
    UnitPart,
)

from analysis_mcp.engine.aggregate import ComposedUnitSumError
from analysis_mcp.engine.columnar import repeated
from analysis_mcp.engine.common import (
    Exclusions,
    FacetSelection,
    PooledParties,
    agreed_facets,
    combine_exclusions,
    facet_columns,
    frame_keeping_role_types,
    require_single_stat_var,
)
from analysis_mcp.engine.two_sided import (
    KeyPairs,
    KeyPeriods,
    ResultCoverage,
    Side,
    SideSums,
    denominator_key_columns,
    divide_cell,
    pair_periods,
    record_cells,
    record_unmatched_periods,
    result_coverage,
    sum_side,
)

# How many keys per side a `QuotientOfNoKeyMatchedError` samples.
KEY_SAMPLE_CAP = 20


class QuotientOfSideUnitUnsupportedError(Exception):
    """`side` cannot be one part of a composed unit. `reason` is `composed`:
    the side's resolved unit is itself a `ComposedUnit` (`unit`), a quotient
    already, and a composed unit holds no composed part.
    """

    def __init__(self, *, side: Side, reason: Literal["composed"], unit: UnitCell) -> None:
        self.side = side
        self.reason = reason
        self.unit = unit
        super().__init__(f"{side} side resolved to the composed unit {unit!r}")


class QuotientOfNoKeyMatchedError(Exception):
    """Both sides hold keys and no numerator key has a denominator group
    under `denominator_by`'s mapping. `numerator_keys`/`denominator_keys` are each side's `KEY_SAMPLE_CAP`
    smallest keys, each as `(column, value)` pairs under that side's own
    column names; `numerator_key_count`/`denominator_key_count` are how many
    keys each side holds.
    """

    def __init__(
        self,
        *,
        numerator_keys: Sequence[KeyPairs],
        denominator_keys: Sequence[KeyPairs],
        numerator_key_count: int,
        denominator_key_count: int,
    ) -> None:
        self.numerator_keys = tuple(numerator_keys)
        self.denominator_keys = tuple(denominator_keys)
        self.numerator_key_count = numerator_key_count
        self.denominator_key_count = denominator_key_count
        super().__init__(
            f"none of the numerator's {numerator_key_count} key(s) matches any of the "
            f"denominator's {denominator_key_count}"
        )


class QuotientOfPeriodsDisjointError(Exception):
    """No numerator key shares a `time_period` with its denominator group.
    `numerator_periods`/`denominator_periods` are every period the paired
    keys and their groups hold, sorted. An empty tuple is a side whose
    `entity` binding matched no rows; when a keyed side has no rows, the
    other side's tuple is every period its slice holds.
    """

    def __init__(
        self, *, numerator_periods: Sequence[str], denominator_periods: Sequence[str]
    ) -> None:
        self.numerator_periods = tuple(numerator_periods)
        self.denominator_periods = tuple(denominator_periods)
        super().__init__("the numerator and the denominator share no time_period for any key")


class QuotientOfZeroDenominatorError(Exception):
    """Every matched (key, period) has a denominator that sums to exactly
    zero, so the call has no quotient to return. `time_periods` are those
    cells' periods, sorted; `cell_count` is how many cells there are.
    """

    def __init__(self, *, time_periods: Sequence[str], cell_count: int) -> None:
        self.time_periods = tuple(time_periods)
        self.cell_count = cell_count
        super().__init__(
            f"the denominator sums to zero in all {cell_count} matched cell(s), over "
            f"period(s) {self.time_periods!r}"
        )


@dataclass(frozen=True, kw_only=True)
class PeriodQuotient:
    """One matched (key, period)'s numerator and denominator sums and their
    quotient, addressed by `time_period` and `key`. `key` is `()` for an
    un-keyed call."""

    time_period: str
    numerator_value: float
    denominator_value: float
    quotient: float
    key: KeyPairs = ()


@dataclass(frozen=True, kw_only=True)
class ZeroDenominatorCell:
    """One matched (key, period) left out because its denominator sums to
    exactly zero, with the numerator's sum there."""

    time_period: str
    numerator_value: float
    key: KeyPairs = ()


@dataclass(frozen=True, kw_only=True)
class QuotientOfResult:
    """Everything `quotient_of_tool.py` needs to mint identity, write the
    store and render the digest.

    `facet_selections` are the axes both sides agree on, the ones the result
    carries; each side's own selections are kept for the trace. Periods and
    coverage read the keys that pair: `numerator_periods` and
    `denominator_periods` are every period those keys and their denominator
    groups hold, and `matched_periods` every period some key shares with its
    group. A side's coverage fraction counts (key, period) cells: those a
    key shares with its group, over those the side holds for the keys that
    pair. A pooled denominator group counts each of its periods once, and an
    un-keyed call has one cell per period.

    Three sorted tuples name the dropped keys, the numerator keys the frame
    holds no row for: `unmatched_keys`, with no denominator group;
    `keys_without_shared_period`, which pair and share no period with their
    group; and `keys_with_zero_denominator`, whose every shared period has a
    zero denominator. `dropped_key_count` is their total, and with the keys
    in `period_quotients` they are every key the numerator holds.
    `zero_denominator_cells` are the matched cells left out, whether or not
    their key has another row. `coverage` counts the keys and periods the
    result holds and the entities each period's denominator sums
    (`analysis_mcp.engine.two_sided.ResultCoverage`).

    `numerator_only_periods` and `denominator_only_periods` are, for each
    key of a keyed call that has a row, the periods it holds on that side
    and not on the other, sorted by key under `numerator_by`'s names. A key
    holding none is absent, and a pooled denominator has no periods of one
    key's own, so `denominator_only_periods` is empty for it.

    `numerator_pooled` and `denominator_pooled` hold what each side's sums
    pooled in each flow party role (`analysis_mcp.engine.common.PooledParties`).
    """

    frame: Frame
    stat_var: str
    numerator_stat_var: str
    denominator_stat_var: str
    numerator_resolved_unit: UnitCell | None
    denominator_resolved_unit: UnitCell | None
    unit: ComposedUnit
    facet_selections: tuple[FacetSelection, ...]
    numerator_facet_selections: tuple[FacetSelection, ...]
    denominator_facet_selections: tuple[FacetSelection, ...]
    warnings: tuple[str, ...]
    numerator_periods: tuple[str, ...]
    denominator_periods: tuple[str, ...]
    matched_periods: tuple[str, ...]
    numerator_coverage_fraction: float
    denominator_coverage_fraction: float
    period_quotients: tuple[PeriodQuotient, ...]
    unmatched_keys: tuple[KeyPairs, ...] = ()
    keys_without_shared_period: tuple[KeyPairs, ...] = ()
    keys_with_zero_denominator: tuple[KeyPairs, ...] = ()
    zero_denominator_cells: tuple[ZeroDenominatorCell, ...] = ()
    numerator_only_periods: tuple[KeyPeriods, ...] = ()
    denominator_only_periods: tuple[KeyPeriods, ...] = ()
    coverage: ResultCoverage = field(default_factory=ResultCoverage)
    exclusions: Exclusions = field(default_factory=Exclusions)
    numerator_pooled: tuple[PooledParties, ...] = ()
    denominator_pooled: tuple[PooledParties, ...] = ()

    @property
    def dropped_key_count(self) -> int:
        return (
            len(self.unmatched_keys)
            + len(self.keys_without_shared_period)
            + len(self.keys_with_zero_denominator)
        )


def _unit_part(side: Side, unit: UnitCell | None, *, stat_var: str) -> UnitPart:
    """`side`'s part of the composed unit: its resolved unit, or the
    StatVar it counts when no row declares one. Refuses
    (`QuotientOfSideUnitUnsupportedError`) a unit that is itself composed."""
    if unit is None:
        return StatVarPart(stat_var)
    if isinstance(unit, ComposedUnit):
        raise QuotientOfSideUnitUnsupportedError(side=side, reason="composed", unit=unit)
    return unit


def _sum_side(
    frame: Frame,
    *,
    side: Side,
    entity: Mapping[str, str],
    by: Sequence[str],
    pinned_unit: UnitCell | None,
    numerator_by: Sequence[str] = (),
) -> SideSums:
    """`sum_side` for `side`. A composed side refuses as `_unit_part`
    refuses it (`QuotientOfSideUnitUnsupportedError`) whether its cells hold
    one row or several."""
    try:
        return sum_side(
            frame,
            side=side,
            entity=entity,
            by=by,
            pinned_unit=pinned_unit,
            numerator_by=numerator_by,
        )
    except ComposedUnitSumError as exc:
        raise QuotientOfSideUnitUnsupportedError(
            side=side, reason="composed", unit=exc.unit
        ) from exc


def _key_sample(sums: SideSums, columns: Sequence[str]) -> tuple[KeyPairs, ...]:
    return tuple(
        tuple(zip(columns, key, strict=True))
        for key in heapq.nsmallest(KEY_SAMPLE_CAP, sums.by_key)
    )


def _all_periods(sums: SideSums) -> list[str]:
    """Every period any of `sums`' keys holds, sorted."""
    return sorted({period for by_period in sums.by_key.values() for period in by_period})


def run_quotient_of(
    numerator_frame: Frame,
    denominator_frame: Frame,
    *,
    stat_var: str,
    numerator_entity: Mapping[str, str],
    denominator_entity: Mapping[str, str],
    numerator_by: Sequence[str] = (),
    denominator_by: Mapping[str, str] | None = None,
    numerator_pinned_unit: UnitCell | None = None,
    denominator_pinned_unit: UnitCell | None = None,
) -> QuotientOfResult:
    """Divide `numerator_frame`'s (narrowed by `numerator_entity`) sum by
    `denominator_frame`'s (narrowed by `denominator_entity`) per matched
    (key, period), labelling every result row `stat_var`. See the module
    docstring for the unit, facet, period, key and zero-denominator
    policies, and each raised exception's docstring for its refusal
    condition. A side's sum or a quotient that overflows raises
    `analysis_mcp.engine.two_sided.NonFiniteCellError` with
    `reason="overflow"`, and a `denominator_by` that leaves a `numerator_by`
    column out raises `DenominatorByPartialError`, which
    `QuotientOfOpSpec` refuses first.
    """
    numerator_by = tuple(numerator_by)
    denominator_columns = denominator_key_columns(
        numerator_by=numerator_by, denominator_by=dict(denominator_by or {})
    )
    numerator = _sum_side(
        numerator_frame,
        side="numerator",
        entity=numerator_entity,
        by=numerator_by,
        pinned_unit=numerator_pinned_unit,
    )
    denominator = _sum_side(
        denominator_frame,
        side="denominator",
        entity=denominator_entity,
        by=denominator_columns,
        pinned_unit=denominator_pinned_unit,
        numerator_by=numerator_by,
    )
    numerator_stat_var = require_single_stat_var(numerator_frame)
    denominator_stat_var = require_single_stat_var(denominator_frame)
    unit = ComposedUnit(
        numerator=_unit_part("numerator", numerator.unit, stat_var=numerator_stat_var),
        denominator=_unit_part("denominator", denominator.unit, stat_var=denominator_stat_var),
    )

    # An un-keyed side has one implicit key, `()`, whatever it summed to.
    # `sum_side` records only a key some row produced, so a slice with no
    # rows is read here as that one key holding no periods.
    pooled = not denominator_columns
    numerator_keys = sorted(numerator.by_key) if numerator_by else [()]
    # A keyed side whose slice holds no row has no key to pair, which no
    # `denominator_by` mapping changes, so it refuses as an un-keyed side
    # with no rows does.
    if not numerator_keys or not (pooled or denominator.by_key):
        raise QuotientOfPeriodsDisjointError(
            numerator_periods=_all_periods(numerator),
            denominator_periods=_all_periods(denominator),
        )

    unmatched_keys: list[tuple[str, ...]] = []
    paired_keys: list[tuple[str, ...]] = []
    for key in numerator_keys:
        if pooled or key in denominator.by_key:
            paired_keys.append(key)
        else:
            unmatched_keys.append(key)
    if not paired_keys:
        raise QuotientOfNoKeyMatchedError(
            numerator_keys=_key_sample(numerator, numerator_by),
            denominator_keys=_key_sample(denominator, denominator_columns),
            numerator_key_count=len(numerator.by_key),
            denominator_key_count=len(denominator.by_key),
        )
    pairing = pair_periods(
        numerator, denominator, numerator_by=numerator_by, paired_keys=paired_keys, pooled=pooled
    )
    if not pairing.matched_periods:
        raise QuotientOfPeriodsDisjointError(
            numerator_periods=pairing.numerator_periods,
            denominator_periods=pairing.denominator_periods,
        )

    period_quotients: list[PeriodQuotient] = []
    kept_cells: list[tuple[tuple[str, ...], str]] = []
    zero_denominator_cells: list[ZeroDenominatorCell] = []
    keys_without_shared_period: list[tuple[str, ...]] = []
    keys_with_zero_denominator: list[tuple[str, ...]] = []
    # Every denominator (group, period) some numerator key shares that sums
    # to zero. A pooled group serves every key, so each cell counts once.
    denominator_zero: set[tuple[tuple[str, ...], str]] = set()
    for key, shared in pairing.shared.items():
        group = () if pooled else key
        numerator_by_period = numerator.by_key.get(key, {})
        denominator_by_period = denominator.by_key.get(group, {})
        key_pairs = tuple(zip(numerator_by, key, strict=True))
        if not shared:
            keys_without_shared_period.append(key)
        elif all(denominator_by_period[period] == 0.0 for period in shared):
            keys_with_zero_denominator.append(key)
        for period in shared:
            numerator_value = numerator_by_period[period]
            denominator_value = denominator_by_period[period]
            if denominator_value == 0.0:
                denominator_zero.add((group, period))
                zero_denominator_cells.append(
                    ZeroDenominatorCell(
                        time_period=period, numerator_value=numerator_value, key=key_pairs
                    )
                )
                continue
            kept_cells.append((key, period))
            period_quotients.append(
                PeriodQuotient(
                    time_period=period,
                    numerator_value=numerator_value,
                    denominator_value=denominator_value,
                    quotient=divide_cell(
                        numerator_value=numerator_value,
                        denominator_value=denominator_value,
                        time_period=period,
                    ),
                    key=key_pairs,
                )
            )
    if not period_quotients:
        raise QuotientOfZeroDenominatorError(
            time_periods=sorted({cell.time_period for cell in zero_denominator_cells}),
            cell_count=len(zero_denominator_cells),
        )
    # A key whose every matched cell divides by zero has no row, so the
    # periods it holds on one side alone are not named against it.
    zero_keys = {tuple(zip(numerator_by, key, strict=True)) for key in keys_with_zero_denominator}

    # Both sides have rows here, so each one's facet selections describe a
    # real slice.
    shared_facets, facet_warnings = agreed_facets(
        numerator.facet_selections,
        denominator.facet_selections,
        sides=("numerator", "denominator"),
        result="quotient",
    )

    exclusions = combine_exclusions(numerator.exclusions, denominator.exclusions)
    exclusions = record_cells(
        exclusions,
        reason="unmatched_key",
        side="numerator",
        sums=numerator,
        columns=numerator_by,
        cells=((key, period) for key in unmatched_keys for period in numerator.by_key[key]),
    )
    exclusions = record_unmatched_periods(
        exclusions,
        pairing,
        numerator=numerator,
        denominator=denominator,
        numerator_by=numerator_by,
        denominator_columns=denominator_columns,
    )
    exclusions = record_cells(
        exclusions,
        reason="zero_denominator",
        side="numerator",
        sums=numerator,
        columns=numerator_by,
        cells=(
            (tuple(value for _, value in cell.key), cell.time_period)
            for cell in zero_denominator_cells
        ),
    )
    exclusions = record_cells(
        exclusions,
        reason="zero_denominator",
        side="denominator",
        sums=denominator,
        columns=denominator_columns,
        cells=sorted(denominator_zero),
    )

    count = len(period_quotients)
    output_frame = frame_keeping_role_types(
        {
            STAT_VAR_COLUMN: repeated(stat_var, count),
            TIME_PERIOD_COLUMN: pa.array(
                [entry.time_period for entry in period_quotients], type=pa.string()
            ),
            VALUE_COLUMN: pa.array(
                [entry.quotient for entry in period_quotients], type=pa.float64()
            ),
            UNIT_COLUMN: unit_cell_array(unit, count),
            **facet_columns(shared_facets, count=count),
            **{
                column: pa.array(
                    [entry.key[index][1] for entry in period_quotients], type=pa.string()
                )
                for index, column in enumerate(numerator_by)
            },
        },
        role_keys=numerator_by,
        source=numerator_frame,
    )

    return QuotientOfResult(
        frame=output_frame,
        stat_var=stat_var,
        numerator_stat_var=numerator_stat_var,
        denominator_stat_var=denominator_stat_var,
        numerator_resolved_unit=numerator.unit,
        denominator_resolved_unit=denominator.unit,
        unit=unit,
        facet_selections=shared_facets,
        numerator_facet_selections=numerator.facet_selections,
        denominator_facet_selections=denominator.facet_selections,
        warnings=tuple(
            dict.fromkeys((*numerator.warnings, *denominator.warnings, *facet_warnings))
        ),
        numerator_periods=pairing.numerator_periods,
        denominator_periods=pairing.denominator_periods,
        matched_periods=pairing.matched_periods,
        numerator_coverage_fraction=pairing.numerator_coverage_fraction,
        denominator_coverage_fraction=pairing.denominator_coverage_fraction,
        period_quotients=tuple(period_quotients),
        unmatched_keys=tuple(tuple(zip(numerator_by, key, strict=True)) for key in unmatched_keys),
        keys_without_shared_period=tuple(
            tuple(zip(numerator_by, key, strict=True)) for key in keys_without_shared_period
        ),
        keys_with_zero_denominator=tuple(
            tuple(zip(numerator_by, key, strict=True)) for key in keys_with_zero_denominator
        ),
        zero_denominator_cells=tuple(zero_denominator_cells),
        numerator_only_periods=tuple(
            entry for entry in pairing.numerator_only_periods if entry[0] not in zero_keys
        ),
        denominator_only_periods=tuple(
            entry for entry in pairing.denominator_only_periods if entry[0] not in zero_keys
        ),
        coverage=result_coverage(
            kept_cells, keyed=bool(numerator_by), denominator=denominator, pooled=pooled
        ),
        exclusions=exclusions,
        numerator_pooled=numerator.pooled_parties,
        denominator_pooled=denominator.pooled_parties,
    )
