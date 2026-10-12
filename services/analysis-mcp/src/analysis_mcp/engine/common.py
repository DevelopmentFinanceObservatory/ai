"""Shared engine helpers: a frame's role columns, the exclusions a
narrowing step records, and narrowing then resolving unit/facet
heterogeneity before any verb computes over a frame.

Unit safety is load-bearing: every verb refuses on unit/facet heterogeneity,
because summing dollars and a percentage together fails silently.
`narrow_and_resolve` narrows a frame's table to the caller's scope (an
`entity` binding, a `pinned_unit`) and resolves unit and facets over
exactly that slice, so a row outside the caller's scope never participates
in deciding what unit or facet value the computation runs under. Facet
heterogeneity (`measurementMethod`, `provenance`) gets a softer rule (see
`narrow_and_resolve`) because a facet axis left unbound on some rows is
common and usually resolvable without a caller's help, unlike a unit
mismatch, which always is.

Every exception raised here is caught only at its own verb's tool boundary
(`aggregate_tool`, `compare_tool`, `growth_tool`, `join_tool`, `rank_tool`,
`share_of_tool`, `sum_of_tool`, `quotient_of_tool`), never inside the engine
itself.

Partial-overlap policy: `compare`, `sum_of`, `share_of`, `quotient_of` and
`join` each combine two independently-sourced inputs and eventually meet a
period or key where the two sides differ; `growth` takes one input but meets
the analogous gap inside its own single series. One framework produces the
answer for all six:

    Proceed when each output row is independently complete. Refuse when
    nothing is left to return, or, for `share_of`, when a key or a period
    the result would carry has no share.

`compare` intersects periods and proceeds, reporting both sides' coverage:
one output row is one period's delta, complete from the two values actually
present, and `delta` claims no total. `sum_of` proceeds the same way: one
output row is one period's sum of two values both present, a total over
exactly two series for that period, and a period one side lacks is left
out rather than summed with a zero. `share_of` and `quotient_of` intersect
periods per key and proceed, recording every period they leave out: one
output row is one key's sum over its denominator's sum for one period,
complete from the two sums present. A period one side lacks leaves the
other side's sums for the remaining periods unchanged, so dropping it
changes no total that is kept. When any key shares no period with its
denominator group, `share_of` refuses the whole call, even if every other
key has rows; `quotient_of` leaves such a key out, records it, and refuses
only when no key shares a period. A matched cell whose denominator sums to
exactly zero refuses the whole `share_of` call; `quotient_of` leaves it out
and records it, and refuses only when every matched cell is one. `growth` proceeds for every entity
holding both endpoint years. A growth row reads only `value_start` and
`value_end`, so it is complete whatever interior years the entity lacks. An
entity missing an endpoint has no complete row and is excluded, and the call
refuses only when no entity holds both.
`join` proceeds, dropping unmatched keys with counts reported: a joined row
is one matched key, complete on its own, claiming no total over either side.
"""

from __future__ import annotations

import bisect
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Literal, cast

import pyarrow as pa
from common.place_type import (
    PARTY_KINDS,
    PARTY_ROLES,
    TOTAL_KIND,
    UNTYPED,
    at_one_level,
    enrichment_columns,
    kind_column,
    type_column,
)
from dfo_frame_grammar import Frame, array_kernel, unit_cells, unit_codes
from dfo_frame_grammar.grammar import (
    FACET_IDENTITY_COLUMNS,
    STAT_VAR_COLUMN,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    UnitCell,
    cell_sort_key,
)

from analysis_mcp.engine.columnar import (
    count_distinct,
    distinct,
    frame_from_columns,
    group_rows,
    repeated,
)

# The closed set of reasons a narrowing step drops rows: `narrow_and_resolve`'s
# unit narrowing (`unit_unbound`, `unit_not_selected`) and entity narrowing
# (`entity_not_selected`), `engine.growth`'s and `engine.rank`'s window
# filters (`outside_window`), `engine.growth`'s entities lacking a row in
# either endpoint year (`endpoint_missing`), with a zero or sign-crossing base
# under a relative change (`zero_or_sign_crossing_base`), or overflowing to a
# non-finite value (`non_finite`), `engine.filter`'s predicate loop
# (`predicate`), `engine.join`'s unmatched-key sets (`unmatched_key`),
# `engine.compare`'s unintersected periods (`period_unmatched`),
# `engine.rank`'s limit (`beyond_limit`), `engine.quotient_of`'s matched
# cells whose denominator sums to zero (`zero_denominator`), and the flow
# parties `without_party_totals` leaves out of a sum or a rank because they
# total other parties (`party_total`). A heterogeneous facet always refuses
# with `FacetHeterogeneousError`, so this set carries no facet reason.
ExclusionReason = Literal[
    "unit_unbound",
    "unit_not_selected",
    "entity_not_selected",
    "outside_window",
    "endpoint_missing",
    "zero_or_sign_crossing_base",
    "non_finite",
    "predicate",
    "unmatched_key",
    "period_unmatched",
    "beyond_limit",
    "zero_denominator",
    "party_total",
]

# Which of a two-input verb's two sides a dropped row came from. `None` for
# a single-input verb (`aggregate`, `growth`, `rank`, `filter`). `compare`/
# `join` tag `left`/`right`; `share_of` and `quotient_of` tag
# `numerator`/`denominator` for the identical two-slice shape. For `share_of`,
# knowing the side tells a reader whether the reported share is inflated or
# deflated.
ExclusionSide = Literal["numerator", "denominator", "left", "right"]

# Per-role-column cap `smallest_distinct` applies, mirroring
# `analysis_mcp.artifact.ENTITY_CAP` (the engine layer does not import the
# pydantic model layer).
_ENTITY_CAP = 20

_FIXED_COLUMNS = frozenset(
    {STAT_VAR_COLUMN, TIME_PERIOD_COLUMN, VALUE_COLUMN, UNIT_COLUMN, STATUS_COLUMN}
) | frozenset(FACET_IDENTITY_COLUMNS)


# How many of a place-type column's distinct cells a `PlaceLevelsMixedError` names.
_PLACE_TYPE_CAP = 10


class PlaceLevelsMixedError(Exception):
    """A verb would set places that share no level side by side: a role it
    ranks or pools carries a `{role}_type` column and holds several places
    with no level of `common.place_type.PLACE_LEVELS` among every one's
    levels. For a flow party role (`without_party_totals`), several parties
    of which one has no kind, every one a total on a pooled role, or
    several parties with no `{role}_kind` column at all. `column` is the type or
    kind column, and `types` its distinct cells, sorted and capped at
    `_PLACE_TYPE_CAP`, empty when the frame carries no such column.
    Maps to the tool boundary's `place_levels_mixed` refusal.
    """

    def __init__(self, *, column: str, types: Sequence[str]) -> None:
        self.column = column
        self.types = tuple(types)
        super().__init__(
            f"the places typed in {column!r} share no level a filter can keep: {self.types!r}"
        )


def pooled_roles(frame: Frame, keys: Iterable[str]) -> tuple[str, ...]:
    """The role columns a verb keyed by `keys` sums or averages across: every
    role column outside `keys`, less a place role whose type column is in
    `keys`, since grouping by a place's type keeps its levels apart. A flow
    party role (`common.place_type.PARTY_ROLES`) outside `keys` is always
    pooled, whatever its type or kind column: the parties of one kind or
    type can total one another, and `without_party_totals` reads the role."""
    kept = set(keys)
    return tuple(
        column
        for column in role_columns(frame)
        if column not in kept and (column in PARTY_ROLES or type_column(column) not in kept)
    )


def require_one_place_level(frame: Frame, columns: Iterable[str]) -> None:
    """Refuse when a role in `columns` carries a place-type column and its
    places are not at one level (`common.place_type.at_one_level`,
    `PlaceLevelsMixedError`). A role with no such column, a role holding one
    place, and a frame with no rows pass, and so does a flow party role,
    which `without_party_totals` reads instead."""
    for column in columns:
        types = type_column(column)
        if column in PARTY_ROLES or types not in frame.columns:
            continue
        cells = distinct(frame.table, types)
        if not at_one_level(cells, place_count=partial(count_distinct, frame.table, column)):
            raise PlaceLevelsMixedError(column=types, types=cells[:_PLACE_TYPE_CAP])


def without_party_totals(
    frame: Frame,
    columns: Iterable[str],
    *,
    ranked: Collection[str] = (),
    side: ExclusionSide | None = None,
) -> tuple[Frame, Exclusions]:
    """`frame` less the rows of every flow party in `columns` whose
    `{role}_kind` cell is `total`, with one `party_total` exclusion per role
    that lost rows, naming the totals left out and tagged `side`. Parties of
    kind `party` and `residual` stay.

    A role outside `common.place_type.PARTY_ROLES` and a role holding one
    party keep every row, and so does a role in `ranked` whose parties are
    all totals: ranking totals beside each other sums nothing across them.
    Refuses (`PlaceLevelsMixedError`) when a role's several parties carry no
    kind column or include one with no kind, since no sum or rank across
    them is known to keep a total apart from the parties it holds, and when
    a role outside `ranked` holds only totals.
    """
    table = frame.table
    exclusions = Exclusions()
    for column in dict.fromkeys(columns):
        if column not in PARTY_ROLES or count_distinct(table, column) < 2:
            continue
        kinds = kind_column(column)
        if kinds not in frame.columns:
            raise PlaceLevelsMixedError(column=kinds, types=())
        cells = distinct(table, kinds)
        if not PARTY_KINDS.issuperset(cells):
            raise PlaceLevelsMixedError(column=kinds, types=cells[:_PLACE_TYPE_CAP])
        if cells == [TOTAL_KIND]:
            if column in ranked:
                continue
            raise PlaceLevelsMixedError(column=kinds, types=cells)
        if TOTAL_KIND not in cells:
            continue
        is_total = array_kernel(
            "equal", table.column(kinds).combine_chunks(), pa.scalar(TOTAL_KIND, pa.string())
        )
        dropped = table.filter(is_total)
        exclusions = record_counted_exclusion(
            exclusions,
            reason="party_total",
            row_count=dropped.num_rows,
            entities={column: distinct(dropped, column)},
            side=side,
        )
        table = table.filter(array_kernel("invert", is_total))
    if table is frame.table:
        return frame, exclusions
    return Frame.from_table(columns=frame.columns, table=table), exclusions


def role_columns(frame: Frame) -> tuple[str, ...]:
    """Every column `frame` declares outside the fixed positions
    `grammar.derive_columns` always emits: the columns an
    `analysis_mcp.artifact.Exclusion`'s `entities` mapping reports one entry
    for."""
    return tuple(column for column in frame.columns if column not in _FIXED_COLUMNS)


def frame_keeping_role_types(
    columns: Mapping[str, pa.Array | pa.ChunkedArray],
    *,
    role_keys: Sequence[str],
    source: Frame,
) -> Frame:
    """An output frame of `columns` declaring `role_keys`, plus the
    `{role}_type` and `{role}_kind` columns of every role in `role_keys` that
    `source`, the verb's `frame_ref` input, carries and `role_keys` leaves
    out (`common.place_type.enrichment_columns`).

    A place's type and a party's kind belong to it, so a verb that keeps a
    role column keeps them, and a later sum or rank over the output still
    finds the parties that total others (`without_party_totals`) and the
    places at different levels (`require_one_place_level`). The harness
    predicts the same columns from the input's role keys
    (`compiler.frame_columns.bound_role_keys`). Each output row's cell is
    read off `source` for its place. A place `source` describes twice takes
    the cell that sorts first, and a place it lacks takes `UNTYPED`, which
    both checks refuse to pool beside other places."""
    described: dict[str, pa.Array] = {}
    for role in role_keys:
        for enrichment in enrichment_columns(role):
            if enrichment in role_keys or enrichment not in source.columns:
                continue
            cells_by_place = group_rows(source.table, (role, enrichment)).keys.sort_by(
                [(role, "ascending"), (enrichment, "ascending")]
            )
            cells = columns[role]
            positions = array_kernel(
                "index_in",
                cells.combine_chunks() if isinstance(cells, pa.ChunkedArray) else cells,
                value_set=cells_by_place.column(role).combine_chunks(),
            )
            described[enrichment] = (
                cells_by_place.column(enrichment)
                .combine_chunks()
                .take(positions)
                .fill_null(UNTYPED)
            )
    return frame_from_columns({**columns, **described}, role_keys=(*role_keys, *described))


@dataclass(frozen=True, kw_only=True)
class ExclusionRecord:
    """Rows one narrowing step dropped, and why. The raw material an
    `analysis_mcp.artifact.Exclusion` is built from, once the tool layer
    wraps it in that pydantic model
    (`analysis_mcp.tool_pipeline.exclusions_for_trace`). A plain dataclass,
    not the pydantic model itself, keeping every engine function free of a
    pydantic dependency.

    `side` names which of a two-input verb's two sides these rows came from
    (see `ExclusionSide`), or `None` for a single-input verb.
    """

    reason: ExclusionReason
    row_count: int
    entities: dict[str, tuple[str, ...]]
    entities_truncated: bool
    side: ExclusionSide | None = None


@dataclass(frozen=True, kw_only=True)
class Exclusions:
    """The exclusion records a chain of narrowing steps produced, in the
    order each step ran. Every engine function that narrows rows returns one
    of these alongside its result; `record_exclusion`,
    `record_counted_exclusion` and `combine_exclusions` are the only ways
    to build one."""

    records: tuple[ExclusionRecord, ...] = ()


def combine_exclusions(*groups: Exclusions) -> Exclusions:
    """Every `ExclusionRecord` from `groups`, concatenated in order. How a
    two-input verb (`compare`, `share_of`, `join`) merges each side's
    accumulator into the one `Exclusions` its result carries."""
    records: list[ExclusionRecord] = []
    for group in groups:
        records.extend(group.records)
    return Exclusions(records=tuple(records))


@dataclass(frozen=True, kw_only=True)
class FacetSelection:
    """One facet axis selected during resolution: the sole declared value
    found across the narrowed slice `narrow_and_resolve` resolved over."""

    axis: str
    ref: str


@dataclass(frozen=True, kw_only=True)
class ResolvedUnitAndFacets:
    """What `narrow_and_resolve` decided, before any reduction runs."""

    unit: UnitCell | None
    facet_selections: tuple[FacetSelection, ...]
    warnings: tuple[str, ...]


class UnitHeterogeneousError(Exception):
    """The frame carries more than one declared `unit`, and no `pinned_unit`
    (a unit ref or converted unit, such as an `AggregateOpSpec.unit`
    disambiguation retry) narrowed it to one.

    `units` is every distinct declared unit found, sorted. It is the candidate
    set a `unit_heterogeneous` refusal's disambiguation payload offers.
    """

    def __init__(self, units: Sequence[UnitCell]) -> None:
        self.units = tuple(units)
        super().__init__(f"heterogeneous units: {self.units!r}")


class UnitNotPresentError(Exception):
    """`pinned_unit` was supplied, but no row in the narrowed slice (see
    `narrow_and_resolve`) declares it.

    Distinct from `UnitHeterogeneousError`: this is a caller mistake (a
    disambiguation retry naming a unit that was never a candidate for this
    slice), and the slice itself is unambiguous. `units` is what the slice
    offers, for the refusal's `fix` payload to list.
    `narrow_and_resolve` checks `pinned_unit` against the slice's own rows
    before narrowing to the pinned unit's rows, so `units` lists the slice's
    candidates. A check after that narrowing would always see an empty list.
    """

    def __init__(self, pinned_unit: UnitCell, units: Sequence[UnitCell]) -> None:
        self.pinned_unit = pinned_unit
        self.units = tuple(units)
        super().__init__(f"unit {pinned_unit!r} not present; frame declares {self.units!r}")


class FacetHeterogeneousError(Exception):
    """One facet axis carries more than one distinct declared value.

    Unlike unit heterogeneity, there is no op-spec field to pin a facet
    choice -- the fix is a different, already facet-scoped `frame_ref`,
    named in the refusal's `fix` payload.
    """

    def __init__(self, axis: str, values: Sequence[str]) -> None:
        self.axis = axis
        self.values = tuple(values)
        super().__init__(f"heterogeneous facet {axis!r}: {self.values!r}")


def smallest_distinct(values: Iterable[str]) -> tuple[tuple[str, ...], bool]:
    """The `_ENTITY_CAP` smallest distinct `values`, sorted, and whether
    `values` holds more. Holds at most `_ENTITY_CAP + 1` values at a time,
    so it reads an iterable as long as a ranked population in bounded
    memory."""
    smallest: list[str] = []
    for value in values:
        if len(smallest) > _ENTITY_CAP and value >= smallest[-1]:
            continue
        position = bisect.bisect_left(smallest, value)
        if position < len(smallest) and smallest[position] == value:
            continue
        smallest.insert(position, value)
        if len(smallest) > _ENTITY_CAP + 1:
            smallest.pop()
    return tuple(smallest[:_ENTITY_CAP]), len(smallest) > _ENTITY_CAP


def record_exclusion(
    exclusions: Exclusions,
    *,
    frame: Frame,
    reason: ExclusionReason,
    dropped: pa.Table,
    side: ExclusionSide | None = None,
) -> Exclusions:
    """`exclusions`, plus one more `ExclusionRecord` for `dropped`, rows
    taken from `frame.table`, under `reason` (tagged `side`, when this
    narrowing step belongs to one side of a two-input verb), naming each
    role column's distinct values across `dropped`. `exclusions` unchanged
    when `dropped` is empty."""
    if not dropped.num_rows:
        return exclusions
    return record_counted_exclusion(
        exclusions,
        reason=reason,
        row_count=dropped.num_rows,
        entities={column: distinct(dropped, column) for column in role_columns(frame)},
        side=side,
    )


def record_counted_exclusion(
    exclusions: Exclusions,
    *,
    reason: ExclusionReason,
    row_count: int,
    entities: Mapping[str, Iterable[str]],
    side: ExclusionSide | None = None,
) -> Exclusions:
    """`exclusions`, plus one more `ExclusionRecord` for `row_count`
    dropped rows under `reason`, for a step that counts its dropped rows
    without holding them. `entities` maps each role column the record
    names to the values its dropped rows carry there, in any order and
    with repeats; each is cut to `smallest_distinct`.
    `exclusions` unchanged when `row_count` is zero."""
    if not row_count:
        return exclusions
    named = {column: smallest_distinct(values) for column, values in entities.items()}
    record = ExclusionRecord(
        reason=reason,
        row_count=row_count,
        entities={column: values for column, (values, _) in named.items()},
        entities_truncated=any(truncated for _, truncated in named.values()),
        side=side,
    )
    return Exclusions(records=(*exclusions.records, record))


@dataclass(frozen=True, slots=True)
class Units:
    """A table's `unit` column as one code per row and one cell per code."""

    codes: pa.Array
    cells: list[UnitCell | None]

    @classmethod
    def of(cls, table: pa.Table) -> Units:
        column = table.column(UNIT_COLUMN)
        codes = unit_codes(column)
        return cls(codes=codes.codes, cells=unit_cells(column, codes))

    def declared(self) -> list[UnitCell]:
        """Every distinct non-null unit in grammar collation order. The order
        is total, so it sorts a ref, a `UnitRequest` and a `ComposedUnit`
        together where Python's own `sorted` raises `TypeError`."""
        return sorted((cell for cell in self.cells if cell is not None), key=cell_sort_key)

    def mask(self, *units: UnitCell | None) -> pa.Array:
        """Whether each row's unit is one of `units`."""
        wanted = [code for code, cell in enumerate(self.cells) if cell in units]
        return array_kernel("is_in", self.codes, value_set=pa.array(wanted, type=pa.int64()))


def _resolve_facets(table: pa.Table, *, unit: UnitCell | None) -> ResolvedUnitAndFacets:
    """The facet resolution `narrow_and_resolve` describes, over `table`,
    whose unit is already resolved to `unit`."""
    warnings: list[str] = []
    facet_selections: list[FacetSelection] = []
    for axis in FACET_IDENTITY_COLUMNS:
        column = table.column(axis)
        values = array_kernel("unique", column).to_pylist()
        declared = sorted(cast(str, value) for value in values if value is not None)
        if len(declared) > 1:
            raise FacetHeterogeneousError(axis, declared)
        if not declared:
            continue
        selected = declared[0]
        if column.null_count:
            warnings.append(
                f"facet {axis!r} was left unbound on some rows; the sole declared value "
                f"{selected!r} was selected for the whole aggregate."
            )
        facet_selections.append(FacetSelection(axis=axis, ref=selected))
    return ResolvedUnitAndFacets(
        unit=unit, facet_selections=tuple(facet_selections), warnings=tuple(warnings)
    )


def narrow_and_resolve(
    frame: Frame,
    *,
    entity: Mapping[str, str] | None = None,
    pinned_unit: UnitCell | None = None,
    exclude_unitless: bool = False,
    side: ExclusionSide | None = None,
) -> tuple[pa.Table, ResolvedUnitAndFacets, Exclusions]:
    """Narrow `frame.table` to the caller's scope, then resolve unit and
    facets over that narrowed slice, never the other way around: a row
    outside the caller's scope (a different entity, a different unit) never
    takes part in deciding what unit or facet value the computation runs
    under. Returns the narrowed rows as a filter of `frame.table` in its
    canonical order, the resolution, and the exclusions.

    Narrowing happens in two steps, both optional: to `entity`'s rows (a
    `{role_column: value}` binding), then to `pinned_unit`'s rows (a row
    that leaves `unit` unbound is excluded rather than assumed to match).
    `pinned_unit` is a unit ref or a converted unit, matched against each
    row's `unit` cell by equality.

    A slice with no rows resolves to `unit=None` with no facet selections
    and no warnings, without checking `pinned_unit`: checking a pin against
    no rows would misreport that the frame declares nothing. The caller's
    own refusal (`ComparisonPeriodsDisjointError`,
    `ShareOfGrainMismatchError`) names that problem once it sees zero rows.

    Unit: a row leaving `unit` unbound takes no part in the check. Units are
    read once per distinct value (`unit_codes`), and candidate units are
    ordered by grammar collation, so a ref, a converted `UnitRequest` and a
    `ComposedUnit` order like any two refs. `pinned_unit` must be one of the
    entity-narrowed slice's declared units, or this raises
    `UnitNotPresentError` listing them, and the slice narrows to its rows.
    With no pin, more than one declared unit raises
    `UnitHeterogeneousError`, and zero or one resolves. `exclude_unitless`,
    when set and the slice resolves to a real unit, then drops the rows
    leaving `unit` unbound, as a pin already does. A slice where no row
    declares a unit resolves to `unit=None` and keeps every row.

    Facets (`measurementMethod`, `provenance`): more than one distinct
    declared value on one axis raises `FacetHeterogeneousError`; there is no
    pin for this, only a differently scoped `frame_ref`. Exactly one
    declared value is selected as a `FacetSelection`, with a warning when
    some rows leave the axis unbound. An axis every row leaves unbound
    resolves to nothing. With no pin, facets resolve over the slice before
    `exclude_unitless` drops rows, and again after.

    The exclusions name every row this call dropped: on an entity basis
    first (`entity_not_selected`, one record per `entity` column), then on
    a unit basis (`unit_unbound` for a row leaving `unit` unbound,
    `unit_not_selected` for a row declaring another unit), each tagged
    `side`: which of a two-input verb's sides this call narrows
    (`compare`/`join`'s `left`/`right`, `share_of`'s
    `numerator`/`denominator`), or `None` for a single-input verb.
    """
    exclusions = Exclusions()
    table = frame.table
    for column, value in (entity or {}).items():
        matched = array_kernel(
            "fill_null",
            array_kernel(
                "equal",
                table.column(column).combine_chunks(),
                pa.scalar(value, type=pa.string()),
            ),
            False,
        )
        exclusions = record_exclusion(
            exclusions,
            frame=frame,
            reason="entity_not_selected",
            dropped=table.filter(array_kernel("invert", matched)),
            side=side,
        )
        table = table.filter(matched)

    if not table.num_rows:
        return (
            table,
            ResolvedUnitAndFacets(unit=None, facet_selections=(), warnings=()),
            exclusions,
        )

    units = Units.of(table)
    declared = units.declared()

    if pinned_unit is not None:
        if pinned_unit not in declared:
            raise UnitNotPresentError(pinned_unit, declared)
        unbound = units.mask(None)
        selected = units.mask(pinned_unit)
        exclusions = record_exclusion(
            exclusions, frame=frame, reason="unit_unbound", dropped=table.filter(unbound), side=side
        )
        exclusions = record_exclusion(
            exclusions,
            frame=frame,
            reason="unit_not_selected",
            dropped=table.filter(array_kernel("invert", array_kernel("or", unbound, selected))),
            side=side,
        )
        table = table.filter(selected)
        return table, _resolve_facets(table, unit=pinned_unit), exclusions

    if len(declared) > 1:
        raise UnitHeterogeneousError(declared)
    unit = declared[0] if declared else None
    resolved = _resolve_facets(table, unit=unit)
    if exclude_unitless and unit is not None and None in units.cells:
        unbound = units.mask(None)
        exclusions = record_exclusion(
            exclusions, frame=frame, reason="unit_unbound", dropped=table.filter(unbound), side=side
        )
        table = table.filter(array_kernel("invert", unbound))
        resolved = _resolve_facets(table, unit=unit)

    return table, resolved, exclusions


def table_before_unitless_exclusion(frame: Frame, *, pinned_unit: UnitCell | None) -> pa.Table:
    """`frame.table` narrowed by `pinned_unit` as `narrow_and_resolve`
    narrows it, except that a row leaving `unit` unbound is kept: the rows
    declaring `pinned_unit` or no unit, or every row when `pinned_unit` is
    `None`.

    For a coverage grid (`aggregate`'s and `rank`'s `expected_cell_count`):
    a row `exclude_unitless` drops still counts as an expected but missing
    cell in the grid, and stays in the grid's denominator.
    """
    if pinned_unit is None:
        return frame.table
    return frame.table.filter(Units.of(frame.table).mask(pinned_unit, None))


def facet_columns(facet_selections: Sequence[FacetSelection], *, count: int) -> dict[str, pa.Array]:
    """Each selected facet's ref on `count` output rows, by axis."""
    return {selection.axis: repeated(selection.ref, count) for selection in facet_selections}


def agreed_facets(
    first: Sequence[FacetSelection],
    second: Sequence[FacetSelection],
    *,
    sides: tuple[str, str],
    result: str,
) -> tuple[tuple[FacetSelection, ...], tuple[str, ...]]:
    """The facet selections both sides declare with the same value, and a
    warning for every axis the result leaves unbound because the sides
    disagree on it or only one side declares it. `sides` names `first` and
    `second` in those warnings, and `result` names what the verb computes
    (`sum`, `quotient`)."""
    first_side, second_side = sides
    first_map = {selection.axis: selection.ref for selection in first}
    second_map = {selection.axis: selection.ref for selection in second}
    shared: list[FacetSelection] = []
    warnings: list[str] = []
    for axis in dict.fromkeys((*first_map, *second_map)):
        first_ref, second_ref = first_map.get(axis), second_map.get(axis)
        if first_ref == second_ref:
            shared.append(FacetSelection(axis=axis, ref=cast("str", first_ref)))
        elif first_ref is not None and second_ref is not None:
            warnings.append(
                f"facet {axis!r} differs between the sides ({first_side} {first_ref!r}, "
                f"{second_side} {second_ref!r}); the {result} carries no value for this axis."
            )
        else:
            side, ref = (
                (first_side, first_ref) if first_ref is not None else (second_side, second_ref)
            )
            warnings.append(
                f"facet {axis!r} is declared only on the {side} side ({ref!r}); the {result} "
                "carries no value for this axis."
            )
    return tuple(shared), tuple(warnings)


def cross_side_differing_facet_axes(
    left: Sequence[FacetSelection], right: Sequence[FacetSelection]
) -> tuple[tuple[str, str, str], ...]:
    """Every facet axis, in `FACET_IDENTITY_COLUMNS`'s own fixed order,
    that both sides declare to different values, as `(axis, left_ref,
    right_ref)`. An axis one side left wholly unbound is not here.
    `cross_side_one_sided_facet_axes` below reports it.
    """
    left_map = {selection.axis: selection.ref for selection in left}
    right_map = {selection.axis: selection.ref for selection in right}
    found: list[tuple[str, str, str]] = []
    for axis in FACET_IDENTITY_COLUMNS:
        left_ref = left_map.get(axis)
        right_ref = right_map.get(axis)
        if left_ref is not None and right_ref is not None and left_ref != right_ref:
            found.append((axis, left_ref, right_ref))
    return tuple(found)


def first_cross_side_facet_mismatch(
    left: Sequence[FacetSelection],
    right: Sequence[FacetSelection],
    *,
    left_stat_vars: Iterable[str],
    right_stat_vars: Iterable[str],
) -> tuple[str, str, str] | None:
    """The first of `cross_side_differing_facet_axes` when the two sides
    share a `stat_var`; `None` when they share none or agree on every axis.

    `compare` and `share_of` each resolve their own two sides' facets
    independently through `narrow_and_resolve`. `FacetHeterogeneousError`
    fires only within one side's own slice. Nothing else catches one measure
    resolved to `provenance=dcid:P1` on the left paired against the same
    measure resolved to `provenance=dcid:P2` on the right, even though each
    side is internally uniform and resolves cleanly on its own. Two
    different measures, such as debt service over GDP, usually come from
    different sources and methods, and each side's own selection is the
    right one for its own measure, so a differing axis there refuses
    nothing and each verb words it as a warning. One concept published
    under two `stat_var` dcids passes that way too, with the warning naming
    both selections.
    """
    if set(left_stat_vars).isdisjoint(right_stat_vars):
        return None
    differing = cross_side_differing_facet_axes(left, right)
    return differing[0] if differing else None


def cross_side_one_sided_facet_axes(
    left: Sequence[FacetSelection], right: Sequence[FacetSelection]
) -> tuple[tuple[str, str | None, str | None], ...]:
    """Every facet axis where exactly one side declares a value and the
    other left it wholly unbound, as `(axis, left_ref, right_ref)` -- exactly
    one of the two always `None`. One-sided axes do not refuse (unlike axes
    both sides declare to different values). Each caller (`compare`, `share_of`)
    turns findings here into its own warning, using its own vocabulary
    (left/right or numerator/denominator) and naming which side's value the
    result carries.
    """
    left_map = {selection.axis: selection.ref for selection in left}
    right_map = {selection.axis: selection.ref for selection in right}
    found: list[tuple[str, str | None, str | None]] = []
    for axis in FACET_IDENTITY_COLUMNS:
        left_ref = left_map.get(axis)
        right_ref = right_map.get(axis)
        if (left_ref is None) != (right_ref is None):
            found.append((axis, left_ref, right_ref))
    return tuple(found)


# Backs `validate_entity_columns`, called at four sites: `growth`'s
# `entity_by`, `rank`'s `rank_by`, `compare`'s `left_entity`/`right_entity`,
# and `share_of`'s `numerator_entity`/`denominator_entity`. `growth`/`rank`
# pass these columns into `frame_from_columns(role_keys=...)`, where `stat_var`/
# `time_period` would collide with reserved column names; `compare`/
# `share_of` use their `entity` binding only as a row selector, where
# neither is a legitimate entity-identifying column at all.
#
# `join` is not among the four: `join_keys` is checked against its own
# private `_DISALLOWED_JOIN_KEY_COLUMNS` (same membership, same
# role-key-collision reason). `join` does reuse `available_entity_columns`
# below to shape its refusal's retry candidates.
_DISALLOWED_ENTITY_COLUMNS = frozenset(
    {VALUE_COLUMN, STATUS_COLUMN, UNIT_COLUMN, STAT_VAR_COLUMN, TIME_PERIOD_COLUMN}
    | set(FACET_IDENTITY_COLUMNS)
)


def available_entity_columns(frame: Frame) -> list[str]:
    """The columns a retry could actually use for an entity-identifying
    parameter (`entity_by`, `rank_by`, `join_keys`) against `frame`: every
    declared column outside `_DISALLOWED_ENTITY_COLUMNS`. Every such
    refusal's `fix["available"]` should carry this filtered list, not the
    raw `list(frame.columns)`, which would offer many columns that would
    fail if retried.
    """
    return [column for column in frame.columns if column not in _DISALLOWED_ENTITY_COLUMNS]


class UnknownEntityColumnError(Exception):
    """One or more entity-identifying columns (a verb's `entity_by`,
    `rank_by`, or similar) name a column this frame lacks the grain for --
    either the column does not appear in `frame.columns`, or it is a
    disallowed reduction/reserved column (`value`, `status`, `unit`,
    `stat_var`, `time_period`, or a facet axis). Same shape as
    `analysis_mcp.engine.aggregate.UnknownGroupByColumnError` -- both map to
    the tool boundary's `role_ambiguous` refusal.
    """

    def __init__(self, columns: Sequence[str], *, available: Sequence[str]) -> None:
        self.columns = tuple(columns)
        self.available = tuple(available)
        super().__init__(f"names unknown/disallowed column(s): {self.columns!r}")


def validate_entity_columns(frame: Frame, columns: Sequence[str]) -> tuple[str, ...]:
    """Validate `columns` are legitimate entity-identifying columns for
    `frame`: each must appear in `frame.columns` and not be a member of
    `_DISALLOWED_ENTITY_COLUMNS`. Returns `columns` as a tuple; raises
    `UnknownEntityColumnError` naming every offending column at once,
    together with the columns a retry could actually use.
    """
    columns = tuple(columns)
    disallowed = [
        column
        for column in columns
        if column not in frame.columns or column in _DISALLOWED_ENTITY_COLUMNS
    ]
    if disallowed:
        raise UnknownEntityColumnError(disallowed, available=available_entity_columns(frame))
    return columns


class StatVarUnresolvedError(Exception):
    """The input frame carries no rows, or more than one distinct
    `stat_var` with nothing telling them apart. Same shape as
    `analysis_mcp.engine.aggregate.StatVarUnresolvedError` -- both map to
    the tool boundary's `stat_var_unresolved` refusal.
    """

    def __init__(self, *, reason: str, distinct_stat_vars: Sequence[str] = ()) -> None:
        self.reason = reason
        self.distinct_stat_vars = tuple(distinct_stat_vars)
        super().__init__(reason)


def require_single_stat_var(frame: Frame) -> str:
    """The frame's sole `stat_var`, or raise `StatVarUnresolvedError`.

    Unlike `run_aggregate`'s own resolution, there is no "unless `stat_var`
    is itself a grouping column" escape hatch here -- no verb using this
    helper groups by `stat_var`, so a frame carrying more than one always
    refuses rather than silently picking one arbitrarily.
    """
    if not frame.num_rows:
        raise StatVarUnresolvedError(reason="the input frame carries no rows")
    stat_vars = distinct(frame.table, STAT_VAR_COLUMN)
    if len(stat_vars) > 1:
        raise StatVarUnresolvedError(
            reason=(
                "the input frame carries more than one StatVar; narrow the input to a "
                "frame_ref that carries only one"
            ),
            distinct_stat_vars=stat_vars,
        )
    return stat_vars[0]
