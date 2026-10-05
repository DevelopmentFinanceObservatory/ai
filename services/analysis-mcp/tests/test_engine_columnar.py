"""`engine.columnar`: deterministic grouping and output frames built from
columns."""

from __future__ import annotations

from itertools import pairwise

import pyarrow as pa
import pytest
from analysis_mcp.engine.columnar import frame_from_columns, group_rows
from dfo_frame_grammar import Frame, FrameError, frame_sha256


def _groups_as_lists(table: pa.Table, keys: list[str]) -> tuple[list[dict], list[list[int]]]:
    groups = group_rows(table, keys)
    indices = groups.indices.to_pylist()
    members = [indices[start:end] for start, end in pairwise(groups.offsets)]
    return groups.keys.to_pylist(), members


def test_groups_come_in_first_row_order_with_rows_ascending() -> None:
    table = pa.table(
        {
            "donor": ["B", "A", "B", "C", "A", "B"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    keys, members = _groups_as_lists(table, ["donor"])
    assert keys == [{"donor": "B"}, {"donor": "A"}, {"donor": "C"}]
    assert members == [[0, 2, 5], [1, 4], [3]]


def test_group_order_is_the_same_on_every_call() -> None:
    table = pa.table({"donor": [f"d{index % 97}" for index in range(5_000)]})
    first = group_rows(table, ["donor"])
    for _ in range(3):
        again = group_rows(table, ["donor"])
        assert again.keys.equals(first.keys)
        assert again.indices.equals(first.indices)
        assert again.offsets == first.offsets
    assert first.keys.column("donor").to_pylist() == [f"d{index}" for index in range(97)]


def test_several_keys_group_by_their_combination_with_null_as_a_value() -> None:
    table = pa.table(
        {
            "donor": ["A", "A", "B", "A", "A"],
            "method": [None, "m", None, None, "m"],
        }
    )
    keys, members = _groups_as_lists(table, ["donor", "method"])
    assert keys == [
        {"donor": "A", "method": None},
        {"donor": "A", "method": "m"},
        {"donor": "B", "method": None},
    ]
    assert members == [[0, 3], [1, 4], [2]]


def test_unit_struct_column_groups_by_value() -> None:
    frame = Frame.build(
        role_keys=["donor"],
        rows=[
            {"stat_var": "dcid:X", "donor": "A", "time_period": "2020", "value": 1.0},
            {
                "stat_var": "dcid:X",
                "donor": "B",
                "time_period": "2020",
                "value": 2.0,
                "unit": {"currency": "USD", "prices": "current"},
            },
            {
                "stat_var": "dcid:X",
                "donor": "C",
                "time_period": "2020",
                "value": 3.0,
                "unit": {"currency": "USD", "prices": "current"},
            },
        ],
    )
    groups = group_rows(frame.table, ["unit"])
    assert groups.count == 2
    assert groups.indices.to_pylist() == [0, 1, 2]
    assert groups.offsets == (0, 1, 3)


def test_no_keys_is_one_group_and_an_empty_table_has_none() -> None:
    table = pa.table({"donor": ["B", "A"], "value": [1.0, 2.0]})
    groups = group_rows(table, [])
    assert (groups.count, groups.indices.to_pylist(), groups.offsets) == (1, [0, 1], (0, 2))
    assert group_rows(table.slice(0, 0), []).count == 0
    assert group_rows(table.slice(0, 0), ["donor"]).count == 0


def test_frame_from_columns_matches_frame_build() -> None:
    rows = [
        {"stat_var": "dcid:X", "donor": "B", "time_period": "2021", "value": 2.5},
        {"stat_var": "dcid:X", "donor": "A", "time_period": "2020", "value": -0.0},
    ]
    frame = frame_from_columns(
        {
            "stat_var": pa.array([row["stat_var"] for row in rows]),
            "donor": pa.array([row["donor"] for row in rows]),
            "time_period": pa.array([row["time_period"] for row in rows]),
            "value": pa.array([row["value"] for row in rows], type=pa.float64()),
        },
        role_keys=["donor"],
    )
    expected = Frame.build(role_keys=["donor"], rows=rows)
    assert frame == expected
    assert frame_sha256(frame) == frame_sha256(expected)


def test_frame_from_columns_refuses_a_duplicate_grain() -> None:
    with pytest.raises(FrameError) as caught:
        frame_from_columns(
            {
                "stat_var": pa.array(["dcid:X", "dcid:X"]),
                "donor": pa.array(["A", "A"]),
                "time_period": pa.array(["2020", "2020"]),
                "value": pa.array([1.0, 2.0]),
            },
            role_keys=["donor"],
        )
    assert caught.value.category == "duplicate_row_key"
