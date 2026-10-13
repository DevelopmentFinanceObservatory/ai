"""Arrow kernels the verb engines share: grouping a frame's table by key
columns, reading each group's members, masking and reading columns, and
building an output `Frame` from columns.

Group order is deterministic: groups appear in order of their first row, and
each group's rows keep their order in the input table.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import pyarrow as pa
from dfo_frame_grammar import (
    Frame,
    UnitCodes,
    array_kernel,
    build_frame_from_arrow,
    scalar_kernel,
    unit_codes,
)
from dfo_frame_grammar.grammar import TIME_PERIOD_COLUMN, UNIT_COLUMN, VALUE_COLUMN


@dataclass(frozen=True, slots=True)
class Groups:
    """A table's rows partitioned by the values of some key columns.

    Attributes:
        keys: one row per group, holding the key columns' values, in order
            of each group's first row in the input table. With no key
            columns it has no columns, and Arrow reports it as zero rows;
            `count` still counts the one group.
        indices: `int64` row indices into the input table, grouped: group
            `g`'s rows are `indices[offsets[g]:offsets[g + 1]]`, ascending.
        offsets: `count + 1` boundaries into `indices`, starting at 0.
    """

    keys: pa.Table
    indices: pa.Array
    offsets: tuple[int, ...]

    @property
    def count(self) -> int:
        """The number of groups."""
        return len(self.offsets) - 1


def group_rows(table: pa.Table, keys: Sequence[str]) -> Groups:
    """`table`'s rows grouped by equal values across `keys`, a null its own
    value. No keys puts every row in one group, and an empty table has no
    groups."""
    if keys:
        codes = unit_codes(
            pa.StructArray.from_arrays(
                [table.column(key).combine_chunks() for key in keys], names=list(keys)
            )
        )
    else:
        codes = UnitCodes(
            codes=pa.array([0] * table.num_rows, type=pa.int64()),
            first_rows=pa.array([0] if table.num_rows else [], type=pa.int64()),
        )
    # `unit_codes` numbers values by first occurrence, and `sort_indices` is
    # stable, so sorting by code yields groups in first-row order with each
    # group's rows ascending.
    indices = array_kernel("sort_indices", codes.codes)
    run_ends = array_kernel("run_end_encode", array_kernel("take", codes.codes, indices))
    assert isinstance(run_ends, pa.RunEndEncodedArray)
    return Groups(
        keys=table.select(list(keys)).take(codes.first_rows),
        indices=indices,
        offsets=(0, *run_ends.run_ends.to_pylist()),
    )


@dataclass(frozen=True, slots=True)
class GroupMembers:
    """The `value` and `time_period` of every row a `Groups` partitions,
    read once, in group order.

    Attributes:
        periods: the table's distinct periods, sorted.
        values: each member row's value.
        period_codes: each member row's period as its position in
            `periods`, so comparing two codes compares their periods.
        offsets: the `Groups` boundaries into `values` and `period_codes`.
    """

    periods: list[str]
    values: list[float]
    period_codes: list[int]
    offsets: tuple[int, ...]

    def of(self, group: int) -> tuple[list[float], list[int]]:
        """Group `group`'s member values and period codes, in row order."""
        start, end = self.offsets[group], self.offsets[group + 1]
        return self.values[start:end], self.period_codes[start:end]


def entity_counts(
    table: pa.Table, *, keys: Sequence[str], entity_columns: Iterable[str]
) -> dict[tuple[str, ...], int]:
    """How many entities each group of `table` under `keys` holds: the
    distinct value combinations across `entity_columns` among its rows, one
    when `entity_columns` names no column outside `keys`. A group with no row
    is absent."""
    combined = group_rows(table, tuple(dict.fromkeys((*keys, *entity_columns))))
    if not combined.count:
        return {}
    if not keys:
        return {(): combined.count}
    groups = zip(
        *(cast("list[str]", combined.keys.column(key).to_pylist()) for key in keys), strict=True
    )
    return dict(Counter(groups))


def keys_unique(table: pa.Table, keys: Sequence[str]) -> bool:
    """Whether no two rows of `table` share their values across `keys`."""
    return group_rows(table, keys).count == table.num_rows


def group_members(table: pa.Table, groups: Groups) -> GroupMembers:
    """The `GroupMembers` of `groups`, a partition of `table`."""
    periods = distinct(table, TIME_PERIOD_COLUMN)
    codes = array_kernel(
        "index_in",
        table.column(TIME_PERIOD_COLUMN).combine_chunks(),
        value_set=pa.array(periods, type=pa.string()),
    )
    return GroupMembers(
        periods=periods,
        values=table.column(VALUE_COLUMN).take(groups.indices).to_pylist(),
        period_codes=codes.take(groups.indices).to_pylist(),
        offsets=groups.offsets,
    )


def distinct(table: pa.Table, column: str) -> list[str]:
    """`column`'s distinct values in `table`, sorted. `column` is a non-null
    string column: `stat_var`, `time_period` or a role."""
    return sorted(cast("list[str]", array_kernel("unique", table.column(column)).to_pylist()))


def count_distinct(table: pa.Table, column: str) -> int:
    """How many distinct values `column` holds in `table`."""
    return cast("int", scalar_kernel("count_distinct", table.column(column)).as_py())


def in_periods(table: pa.Table, periods: Iterable[str]) -> pa.Array:
    """Whether each row of `table` has its `time_period` in `periods`."""
    return array_kernel(
        "is_in",
        table.column(TIME_PERIOD_COLUMN).combine_chunks(),
        value_set=pa.array(list(periods), type=pa.string()),
    )


def repeated(value: str | None, count: int) -> pa.Array:
    """`value` as a string column of `count` rows."""
    return pa.repeat(pa.scalar(value, type=pa.string()), count)


def unit_column(table: pa.Table, *, count: int) -> pa.ChunkedArray:
    """`count` copies of the unit of `table`, a narrowed table whose rows all
    carry its resolved unit, as `table` holds it: a ref, a converted unit, or
    null."""
    return table.column(UNIT_COLUMN).take(pa.array([0] * count, type=pa.int64()))


def frame_from_columns(
    columns: Mapping[str, pa.Array | pa.ChunkedArray], *, role_keys: Iterable[str]
) -> Frame:
    """An output `Frame` from named columns, validated, canonically sorted and
    checked for duplicate grain by `build_frame_from_arrow`, as `Frame.build`
    does for rows."""
    return build_frame_from_arrow(pa.table(dict(columns)), role_keys=role_keys)
