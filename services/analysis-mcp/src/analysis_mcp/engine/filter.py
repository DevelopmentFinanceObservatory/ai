"""The `filter` engine applies row-value predicate filtering over one
frame's Arrow table, using `analysis_mcp.op_spec.FilterOpSpec`'s own closed
predicate vocabulary (`OrderedThresholdPredicate`, `CategoryMatchPredicate`,
`PlaceLevelPredicate`), each evaluated as a mask over its column. It
keeps a row only if the row satisfies every predicate (conjunction).

Each mask reproduces the Python comparison it stands for. An
`ordered_threshold` compares a float64 `value` with a float operand, or a
`time_period` string with a string operand, and Arrow's comparison kernels
follow IEEE 754 for floats and compare strings by UTF-8 bytes, which is the
code-point order of Python's `str`. Neither column holds a null, and the
op-spec and frame grammar keep every number finite. A `category_match` is
`is_in`, under which a null cell matches none of the predicate's strings,
as `None in values` does. A `place_level` reads each distinct cell of a
place-type column with `common.place_type.levels_of` and keeps the rows
whose place sits at its level.

Filter is architecturally distinct from every other verb this toolkit
implements. It never combines two rows' own values into one number, so
there is nothing for `analysis_mcp.engine.common.narrow_and_resolve`
to protect against. Every reducing verb refuses on unit/facet
heterogeneity because mixing a dollar figure with a percentage into one
sum would be silently wrong. Filter inspects each row in isolation, so
a `CategoryMatchPredicate` on the `unit` column itself is a perfectly
ordinary way to narrow a heterogeneous-unit frame down to one unit (see
`tests/test_engine_filter.py`'s own multi-unit case), and this engine
never resolves or refuses on unit/facet heterogeneity at all. This is a
deliberate exemption for filter.

A `unit` cell that is not a ref holds an object: a `grammar.UnitRequest` in
a *converted* frame, a `grammar.ComposedUnit` in the output of a verb that
divides one frame by another (see `_unit_object_types`). A `category_match`
on `unit` refuses (`FilterUnitConversionUnsupportedError`) over a frame
holding either, since the predicate's values are unit strings. Every other
predicate filters such a frame like any other, and a kept row carries its
unit object into the output unchanged.

An empty result after filtering is valid. `dfo_frame_grammar.frame.
Frame.build`'s own docstring says plainly that "an empty `rows` iterable
produces a valid, columns-only empty frame," so `run_filter` returns a
valid zero-row result rather than inventing a `filter_empty_result` refusal
the grammar gives no reason to need.

A frame whose `time_period` cells are one span (`"2018/2022"`, the label an
`aggregate` or `rank` over several years writes) refuses an `ordered_threshold`
on `time_period`, since a lexical comparison against the span's string says
nothing about the periods it covers. The refusal is a type mismatch with `actual_type`
`"interval"`. A `category_match` on the span string keeps its rows. For the
same reason an `ordered_threshold` on `time_period` takes a point operand, and
an interval operand refuses as a value type mismatch with `actual_type`
`"interval"` and `expected_type` `"point"`.

A predicate's own `kind` is checked against its column's cell type once,
over every predicate, before any row is inspected for matching. For every
column except `unit`, this is a *declared*, schema-level check: `value` is
numeric, `time_period` is an ordered string (`"ordered_str"`:
`YYYY`/`YYYY-MM`/`YYYY-MM-DD`, zero-padded, comparing lexically across every
grain), `stat_var`/every role column is a non-null plain `str`, a facet axis
is `str | None`, and `status` is always `None`. The frame grammar pins
each of these domains at construction (`Frame.build`'s own normalisation),
so the type is knowable without looking at a single cell, the same way
`AGGREGATIONS` is a closed set checked without looking at data.

`unit` is the one exception. Its own declared type is
`dfo_frame_grammar.grammar.UnitCell | None` (a `unit` cell can carry a
`UnitRequest` or `ComposedUnit` object, per the
`_DISALLOWED_GROUP_BY_COLUMNS` comment in `engine/aggregate.py`), so a
`category_match` predicate on `unit` is checked against what this
particular frame's rows actually carry rather than against a schema-level
type (see `_unit_object_types`).

Checking once, up front, means a `None` cell on a nullable column (`unit`,
a facet axis) is treated as a valid value at match time rather than as a
type error. It simply fails to match a `category_match`'s closed set of
string values, the same as any other non-matching value.

An `ordered_threshold` predicate also carries an operand-pairing check. Its
`value` must be a number against `value` and a period string against
`time_period`, checked once alongside the column-type check
(`FilterPredicateValueTypeMismatchError`), so a swapped operand type refuses
before any row is inspected, the same as a swapped column or kind does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pyarrow as pa
from common.place_type import levels_of
from common.time_grain import is_interval
from dfo_frame_grammar import Frame, array_kernel, scalar_kernel
from dfo_frame_grammar.grammar import (
    FACET_IDENTITY_COLUMNS,
    STAT_VAR_COLUMN,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    UnitCell,
)

from analysis_mcp.engine.columnar import frame_from_columns
from analysis_mcp.engine.common import Exclusions, Units, record_exclusion, role_columns

if TYPE_CHECKING:
    from analysis_mcp.op_spec import Predicate


class FilterUnknownColumnError(Exception):
    """One or more predicates name a column this frame does not declare."""

    def __init__(self, columns: Sequence[str], *, available: Sequence[str]) -> None:
        self.columns = tuple(columns)
        self.available = tuple(available)
        super().__init__(f"predicates name unknown column(s): {self.columns!r}")


class FilterPredicateTypeMismatchError(Exception):
    """A predicate's target column exists but its own declared cell type
    does not match that predicate kind -- an `ordered_threshold` against a
    plain-string-typed column, or either kind against `status` (always
    `None`). A retry naming a different predicate kind or column can fix
    this; see `FilterPredicateValueTypeMismatchError` for the case where the
    column is right and only the operand's own type is wrong, and
    `FilterUnitConversionUnsupportedError` for the one `unit` case where no
    retry succeeds.

    Checking every predicate's declared type once, before any row inspection,
    prevents silent failures: a mistyped predicate would either silently drop
    every row (indistinguishable from "this predicate matched zero rows
    legitimately") or fail inconsistently depending on predicate order or row
    presence. `_validate_predicate_types` catches these cases upfront.
    """

    def __init__(self, *, column: str, predicate_kind: str, actual_type: str) -> None:
        self.column = column
        self.predicate_kind = predicate_kind
        self.actual_type = actual_type
        super().__init__(
            f"predicate kind {predicate_kind!r} does not match column {column!r}'s own "
            f"declared cell type ({actual_type})"
        )


class FilterPredicateValueTypeMismatchError(Exception):
    """An `ordered_threshold` predicate's own `column` is a legal target for
    that kind, but `value`'s own type is wrong for it: a string operand
    against `value` (numeric), or a numeric operand against `time_period`
    (an ordered string). Distinct from `FilterPredicateTypeMismatchError`,
    which fires when the column cannot take this predicate kind at all.
    Here the column is right, so a retry with the operand type this column
    expects can fix it.
    """

    def __init__(self, *, column: str, actual_type: str, expected_type: str) -> None:
        self.column = column
        self.actual_type = actual_type
        self.expected_type = expected_type
        super().__init__(
            f"an ordered_threshold predicate's value on column {column!r} must be "
            f"{expected_type}, not {actual_type}"
        )


class FilterPlaceLevelUnknownError(Exception):
    """A strict place-level predicate found entities with unknown typing."""

    def __init__(self, *, column: str) -> None:
        self.column = column
        super().__init__(
            f"strict place-level predicate on {column!r} cannot classify every entity; "
            "request or retain its type enrichment"
        )


class FilterUnitConversionUnsupportedError(Exception):
    """A `category_match` predicate targets `unit`, and this particular
    frame's own `unit` cells carry a `grammar.UnitRequest` or
    `grammar.ComposedUnit` object rather than a plain string or `None` (see
    the module docstring's own note on this limit, and
    `_unit_object_types`). `actual_type` names the object types the cells
    carry, joined by `" | "`.

    Distinct from `FilterPredicateTypeMismatchError`, which names the wrong
    predicate kind or column. Here the kind and column are legal, but
    `category_match`'s `values` is a closed set of plain strings, which never
    matches a unit object. Predicates on every other column still apply to
    this frame.
    """

    def __init__(self, *, column: str, predicate_kind: str, actual_type: str) -> None:
        self.column = column
        self.predicate_kind = predicate_kind
        self.actual_type = actual_type
        super().__init__(
            f"predicate kind {predicate_kind!r} on column {column!r} cannot match: this "
            f"frame's own unit cells carry a {actual_type} object, not a plain string"
        )


@dataclass(frozen=True, kw_only=True)
class FilterRowSummary:
    """One surviving row's own identity and value, for the digest's capped
    preview -- addressed by its own grain, never by position in
    `FilterResult.frame` (the same "never trust row order" discipline every
    other engine result here follows).
    """

    row_key: tuple[tuple[str, str], ...]
    value: float
    unit: UnitCell | None


@dataclass(frozen=True, kw_only=True)
class FilterResult:
    """Everything `filter_tool.py` needs to mint identity, write the store,
    and render the digest -- the frame is here for the store write and
    replay; the digest itself renders from `kept_row_summaries`, never
    `frame.rows`. `units` is every distinct unit the kept rows declare, with
    `None` for rows that declare none."""

    frame: Frame
    total_row_count: int
    kept_row_count: int
    dropped_row_count: int
    kept_row_summaries: tuple[FilterRowSummary, ...]
    exclusions: Exclusions = field(default_factory=Exclusions)
    units: tuple[UnitCell | None, ...] = ()


def _predicate_columns(predicates: Sequence[Predicate]) -> tuple[str, ...]:
    return tuple(predicate.column for predicate in predicates)


# The Arrow comparison kernel for each `ordered_threshold` op.
_THRESHOLD_KERNELS = {
    "gt": "greater",
    "gte": "greater_equal",
    "lt": "less",
    "lte": "less_equal",
    "eq": "equal",
    "ne": "not_equal",
}


def _declared_cell_type(column: str) -> str:
    """`column`'s own declared cell type. Accurate for every column
    including `unit` (`str | UnitRequest | ComposedUnit | None`), but for
    `unit` this is only a *legality* check. Whether a `category_match`
    predicate on `unit` is actually safe against this frame depends on its
    rows' own cells rather than this declared type (see
    `_unit_object_types`)."""
    if column == VALUE_COLUMN:
        return "float"
    if column == STATUS_COLUMN:
        return "NoneType"
    if column == UNIT_COLUMN:
        return "str | UnitRequest | ComposedUnit | None"
    if column in FACET_IDENTITY_COLUMNS:
        return "str | None"
    if column == TIME_PERIOD_COLUMN:
        return "ordered_str"
    return "str"


def _predicate_kind_matches_column_type(*, column: str, kind: str) -> bool:
    """Whether `kind` is a legal predicate kind for `column`'s own declared
    cell type: `ordered_threshold` only for `value` (numeric) and
    `time_period` (ordered string); `category_match` for every string-typed
    column, nullable or not, `time_period` included; `place_level` for a
    non-null plain-text column (`stat_var` or a role); no kind for
    `status` (always `None`, so no predicate can legitimately touch it). For
    `unit`, this is necessary but not sufficient. See
    `_validate_predicate_types`, which additionally checks `unit`'s actual
    cells before admitting a `category_match` predicate on it."""
    if column == STATUS_COLUMN:
        return False
    if kind == "ordered_threshold":
        return column in (VALUE_COLUMN, TIME_PERIOD_COLUMN)
    if kind == "place_level":
        return _declared_cell_type(column) == "str"
    return column != VALUE_COLUMN


def _unit_object_types(table: pa.Table) -> str | None:
    """The object types `table`'s `unit` cells carry beside refs and nulls,
    joined by `" | "`: `"UnitRequest"` for a cell whose struct holds a
    `request`, `"ComposedUnit"` for any other cell with no `ref`. `None`
    when every cell is a ref or null. A `category_match` predicate's own
    `values` is a closed set of plain strings, so it can never legitimately
    match a unit object; without this check, such a predicate would silently
    drop every row carrying one instead of refusing
    (`FilterUnitConversionUnsupportedError`'s own docstring). Every row is
    inspected -- the verdict is "does any row carry one", not "what does the
    first row carry" -- so it does not depend on row order.
    """
    column = table.column(UNIT_COLUMN).combine_chunks()
    is_request = array_kernel("is_valid", array_kernel("struct_field", column, "request"))
    is_object = array_kernel(
        "and",
        array_kernel("is_valid", column),
        array_kernel("is_null", array_kernel("struct_field", column, "ref")),
    )
    is_composed = array_kernel("and_not", is_object, is_request)
    held = [
        name
        for name, mask in (("UnitRequest", is_request), ("ComposedUnit", is_composed))
        if scalar_kernel("any", mask).as_py()
    ]
    return " | ".join(held) or None


def _validate_predicate_types(
    predicates: Sequence[Predicate], frame: Frame, *, span: str | None
) -> None:
    """Validate every predicate's own `kind` against its column's cell
    type, once, before any row is inspected for matching. See the module
    docstring for why this replaces a per-cell check inside the row loop,
    and for why `unit` alone is checked against its actual cells rather
    than its declared type. A `category_match` on `unit` that is legal at
    the schema level can still refuse (`FilterUnitConversionUnsupportedError`,
    not `FilterPredicateTypeMismatchError`) once this frame's own cells are
    checked. See that exception's own docstring for why it is a distinct
    condition rather than another declared-type mismatch. An
    `ordered_threshold` predicate whose column is legal for that kind still
    needs its own operand checked against that column
    (`FilterPredicateValueTypeMismatchError`). `value` takes a numeric
    operand and `time_period` takes a period-string operand, and the two
    cannot be swapped. A `time_period` operand must be a point: an interval
    such as `"2018/2022"` would compare as a string, so `gte "2018/2022"`
    would drop 2018 and keep every later year. When `span` is set, an
    `ordered_threshold` on `time_period` refuses as an `"interval"` type
    mismatch.
    """
    unit_checked = False
    for predicate in predicates:
        if (
            span is not None
            and predicate.kind == "ordered_threshold"
            and predicate.column == TIME_PERIOD_COLUMN
        ):
            raise FilterPredicateTypeMismatchError(
                column=predicate.column, predicate_kind=predicate.kind, actual_type="interval"
            )
        if not _predicate_kind_matches_column_type(column=predicate.column, kind=predicate.kind):
            raise FilterPredicateTypeMismatchError(
                column=predicate.column,
                predicate_kind=predicate.kind,
                actual_type=_declared_cell_type(predicate.column),
            )
        if predicate.kind == "ordered_threshold":
            operand_type = "str" if isinstance(predicate.value, str) else "float"
            expected_type = "str" if predicate.column == TIME_PERIOD_COLUMN else "float"
            if operand_type != expected_type:
                raise FilterPredicateValueTypeMismatchError(
                    column=predicate.column,
                    actual_type=operand_type,
                    expected_type=expected_type,
                )
            if isinstance(predicate.value, str) and is_interval(predicate.value):
                raise FilterPredicateValueTypeMismatchError(
                    column=predicate.column, actual_type="interval", expected_type="point"
                )
        if predicate.column == UNIT_COLUMN and predicate.kind == "category_match":
            unit_object_types = None if unit_checked else _unit_object_types(frame.table)
            unit_checked = True
            if unit_object_types is not None:
                raise FilterUnitConversionUnsupportedError(
                    column=UNIT_COLUMN,
                    predicate_kind=predicate.kind,
                    actual_type=unit_object_types,
                )


def _predicate_mask(table: pa.Table, predicate: Predicate) -> pa.Array:
    """Whether each row of `table` satisfies `predicate`, with no nulls.
    Types are already validated by `_validate_predicate_types` before any
    row is reached, so an `ordered_threshold` predicate's own operand
    already matches its column's own type (`float` against `value`, `str`
    against `time_period`). A `category_match` on `unit` reads the ref of
    each cell, since every cell is a ref or null by then.
    """
    column = table.column(predicate.column).combine_chunks()
    if predicate.kind == "ordered_threshold":
        return array_kernel(
            _THRESHOLD_KERNELS[predicate.op], column, pa.scalar(predicate.value, type=column.type)
        )
    if predicate.kind == "place_level":
        return _level_mask(column, predicate.column, predicate.level, strict=predicate.strict)
    if predicate.column == UNIT_COLUMN:
        column = array_kernel("struct_field", column, "ref")
    return array_kernel("is_in", column, value_set=pa.array(predicate.values, type=pa.string()))


def _level_mask(column: pa.Array, name: str, level: str, *, strict: bool) -> pa.Array:
    """Whether each place-type cell of `column` describes a place at `level`
    (`common.place_type.levels_of`), decided once per distinct cell."""
    encoded = array_kernel("dictionary_encode", column)
    assert isinstance(encoded, pa.DictionaryArray)
    cells = cast("list[str]", encoded.dictionary.to_pylist())
    if strict and any(not levels_of(cell) for cell in cells):
        raise FilterPlaceLevelUnknownError(column=name)
    matches = pa.array([level in levels_of(cell) for cell in cells], type=pa.bool_())
    return array_kernel("take", matches, encoded.indices)


def run_filter(
    frame: Frame, *, predicates: Sequence[Predicate], span: str | None = None
) -> FilterResult:
    """Keep only the rows of `frame` satisfying every predicate in
    `predicates` (conjunction; an empty `predicates` keeps every row). `span`
    is the one span `frame`'s periods form (`AdmittedFrame.span`), or `None` for
    points, and makes an `ordered_threshold` on `time_period` refuse. See
    the module docstring for why this engine resolves no unit/facet and
    refuses on no empty-result condition; see `FilterUnknownColumnError`'s,
    `FilterPredicateTypeMismatchError`'s, `FilterPredicateValueTypeMismatchError`'s,
    and `FilterUnitConversionUnsupportedError`'s own docstrings for their
    refusal conditions.
    """
    invalid = [column for column in _predicate_columns(predicates) if column not in frame.columns]
    if invalid:
        raise FilterUnknownColumnError(invalid, available=list(frame.columns))

    _validate_predicate_types(predicates, frame, span=span)

    table = frame.table
    keep = pa.repeat(pa.scalar(True), table.num_rows)
    for predicate in predicates:
        keep = array_kernel("and", keep, _predicate_mask(table, predicate))
    kept = table.filter(keep)
    exclusions = record_exclusion(
        Exclusions(),
        frame=frame,
        reason="predicate",
        dropped=table.filter(array_kernel("invert", keep)),
    )

    roles = role_columns(frame)
    output_frame = frame_from_columns(
        {column: kept.column(column) for column in frame.columns if column != STATUS_COLUMN},
        role_keys=roles,
    )

    output = output_frame.table
    role_cells = [cast("list[str]", output.column(column).to_pylist()) for column in roles]
    # One cell per distinct unit, taken back to one per row.
    units = Units.of(output)
    unit_per_row = [units.cells[code] for code in cast("list[int]", units.codes.to_pylist())]
    kept_row_summaries = tuple(
        FilterRowSummary(
            row_key=(
                (STAT_VAR_COLUMN, stat_var),
                *zip(roles, role_values, strict=True),
                (TIME_PERIOD_COLUMN, time_period),
            ),
            value=value,
            unit=unit,
        )
        for stat_var, *role_values, time_period, value, unit in zip(
            cast("list[str]", output.column(STAT_VAR_COLUMN).to_pylist()),
            *role_cells,
            cast("list[str]", output.column(TIME_PERIOD_COLUMN).to_pylist()),
            cast("list[float]", output.column(VALUE_COLUMN).to_pylist()),
            unit_per_row,
            strict=True,
        )
    )

    return FilterResult(
        frame=output_frame,
        total_row_count=table.num_rows,
        kept_row_count=kept.num_rows,
        dropped_row_count=table.num_rows - kept.num_rows,
        kept_row_summaries=kept_row_summaries,
        exclusions=exclusions,
        units=tuple(units.cells),
    )
