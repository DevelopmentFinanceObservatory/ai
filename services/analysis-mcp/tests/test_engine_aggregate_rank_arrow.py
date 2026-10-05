"""`run_aggregate` and `run_rank` compute on a frame's Arrow table: neither
reads `Frame.rows`, a key column named twice groups as if named once, and a
converted unit reaches the output unchanged.
"""

from __future__ import annotations

import pytest
from analysis_mcp.engine.aggregate import run_aggregate
from analysis_mcp.engine.rank import run_rank
from dfo_frame_grammar import Frame
from dfo_frame_grammar.grammar import UnitRequest

_CONVERTED = UnitRequest(currency="EUR", prices="constant", base_year=2020)


def _frame() -> Frame:
    return Frame.build(
        role_keys=["donor"],
        rows=[
            {
                "stat_var": "dcid:X",
                "donor": donor,
                "time_period": period,
                "value": value,
                "unit": unit,
            }
            for donor, period, value, unit in (
                ("dcid:A", "2020", 0.1, "dcid:USDollar"),
                ("dcid:A", "2021", 0.2, "dcid:USDollar"),
                ("dcid:B", "2020", 0.3, "dcid:USDollar"),
                ("dcid:B", "2021", 0.3, None),
                ("dcid:C", "2019", 1e16, "dcid:USDollar"),
            )
        ],
    )


def test_aggregate_and_rank_never_read_frame_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame()

    def refuse(_: Frame) -> None:
        raise AssertionError("Frame.rows was read")

    monkeypatch.setattr(Frame, "rows", property(refuse))

    run_aggregate(frame, group_by=["donor"], aggregation="max")
    run_aggregate(frame, group_by=[], aggregation="sum", pinned_unit="dcid:USDollar")
    for per_period in (False, True):
        for limit in (None, 1):
            run_rank(
                frame,
                rank_by=["donor"],
                aggregation="mean",
                window_start="2019",
                window_end="2020",
                ascending=False,
                per_period=per_period,
                limit=limit,
            )


def test_a_group_by_column_named_twice_groups_once() -> None:
    once = run_aggregate(_frame(), group_by=["donor"], aggregation="sum")
    twice = run_aggregate(_frame(), group_by=["donor", "donor"], aggregation="sum")

    assert twice.frame == once.frame
    assert twice.group_summaries == once.group_summaries


def test_a_rank_by_column_named_twice_ranks_once() -> None:
    once = run_rank(
        _frame(),
        rank_by=["donor"],
        aggregation="sum",
        window_start=None,
        window_end=None,
        ascending=True,
    )
    twice = run_rank(
        _frame(),
        rank_by=["donor", "donor"],
        aggregation="sum",
        window_start=None,
        window_end=None,
        ascending=True,
    )

    assert twice.frame == once.frame
    assert [er.rank for er in twice.entity_ranks] == [er.rank for er in once.entity_ranks]


def _converted_frame() -> Frame:
    return Frame.build(
        role_keys=["donor"],
        rows=[
            {
                "stat_var": "dcid:X",
                "donor": donor,
                "time_period": period,
                "value": value,
                "unit": unit,
            }
            for donor, period, value, unit in (
                ("dcid:A", "2020", 1.5, _CONVERTED.as_json()),
                ("dcid:A", "2021", 2.5, _CONVERTED.as_json()),
                ("dcid:B", "2020", 0.25, _CONVERTED.as_json()),
                ("dcid:B", "2021", 4.0, None),
            )
        ],
    )


def _units(frame: Frame) -> set[object]:
    index = frame.columns.index("unit")
    return {row[index] for row in frame.rows}


def test_a_converted_unit_passes_through_aggregate_unchanged() -> None:
    result = run_aggregate(_converted_frame(), group_by=["donor"], aggregation="sum")

    assert result.resolved_unit == _CONVERTED
    assert result.frame.num_rows == 2
    assert _units(result.frame) == {_CONVERTED}


def test_a_converted_unit_passes_through_rank_unchanged_where_rank_writes_a_unit() -> None:
    def ranked(*, per_period: bool) -> Frame:
        return run_rank(
            _converted_frame(),
            rank_by=["donor"],
            aggregation="sum",
            window_start=None,
            window_end=None,
            ascending=False,
            per_period=per_period,
        ).frame

    assert _units(ranked(per_period=False)) == {_CONVERTED}
    assert _units(ranked(per_period=True)) == {None}
