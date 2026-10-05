"""`run_filter` evaluates predicates as Arrow masks: each mask keeps exactly
the rows Python's `operator` comparison or `in` membership keeps, at the
float and string edges where the two could part.
"""

from __future__ import annotations

import operator
import sys

import pytest
from analysis_mcp.engine.filter import run_filter
from analysis_mcp.op_spec import CategoryMatchPredicate, OrderedThresholdPredicate
from dfo_frame_grammar import Frame
from dfo_frame_grammar.grammar import UnitRequest

_OPS = {
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
}

_VALUES = [
    0.0,
    5e-324,
    -5e-324,
    sys.float_info.min,
    sys.float_info.max,
    -sys.float_info.max,
    0.1 + 0.2,
    0.30000000000000004,
    0.3,
    9007199254740992.0,
    9007199254740994.0,
    -1.5,
]

_PERIODS = ["2019", "2020", "2020-01", "2020-01-01", "2020-12", "2021"]


def _frame(values: list[float], periods: list[str]) -> Frame:
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": f"dcid:R{index}",
                "time_period": period,
                "value": value,
                "unit": "dcid:USDollar",
            }
            for index, (value, period) in enumerate(zip(values, periods, strict=False))
        ],
    )


def _kept(frame: Frame, **predicate: object) -> list[tuple[str, ...]]:
    result = run_filter(frame, predicates=(OrderedThresholdPredicate.model_validate(predicate),))
    return sorted(summary.row_key for summary in result.kept_row_summaries)


@pytest.mark.parametrize("op", sorted(_OPS))
@pytest.mark.parametrize("operand", [0.0, -0.0, 5e-324, 0.3, 0.1 + 0.2, 9007199254740993.0])
def test_value_threshold_keeps_the_rows_python_comparison_keeps(op: str, operand: float) -> None:
    frame = _frame(_VALUES, ["2020"] * len(_VALUES))
    expected = sorted(
        (("stat_var", "dcid:X"), ("reporter", f"dcid:R{index}"), ("time_period", "2020"))
        for index, value in enumerate(_VALUES)
        if _OPS[op](value, operand)
    )

    assert _kept(frame, column="value", op=op, value=operand) == expected


@pytest.mark.parametrize("op", sorted(_OPS))
@pytest.mark.parametrize("operand", ["2020", "2020-01", "2020-06-15"])
def test_period_threshold_keeps_the_rows_python_comparison_keeps(op: str, operand: str) -> None:
    frame = _frame([1.0] * len(_PERIODS), _PERIODS)
    expected = sorted(
        (("stat_var", "dcid:X"), ("reporter", f"dcid:R{index}"), ("time_period", period))
        for index, period in enumerate(_PERIODS)
        if _OPS[op](period, operand)
    )

    assert _kept(frame, column="time_period", op=op, value=operand) == expected


def test_category_match_is_exact_membership_and_a_null_cell_matches_nothing() -> None:
    reporters = ["dcid:a", "dcid:A", "dcid:é", "dcid:\U0001d538", "dcid:ab"]
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": reporter,
                "time_period": "2020",
                "value": 1.0,
                "measurementMethod": None if index % 2 else "dcid:M",
            }
            for index, reporter in enumerate(reporters)
        ],
    )

    by_reporter = run_filter(
        frame,
        predicates=(CategoryMatchPredicate(column="reporter", values=("dcid:a", "dcid:é")),),
    )
    by_facet = run_filter(
        frame, predicates=(CategoryMatchPredicate(column="measurementMethod", values=("dcid:M",)),)
    )

    assert [dict(summary.row_key)["reporter"] for summary in by_reporter.kept_row_summaries] == [
        "dcid:a",
        "dcid:é",
    ]
    assert by_facet.kept_row_count == 3
    assert by_facet.dropped_row_count == 2


def test_category_match_on_unit_reads_each_cell_ref() -> None:
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": reporter,
                "time_period": "2020",
                "value": 1.0,
                "unit": unit,
            }
            for reporter, unit in (
                ("dcid:A", "dcid:USDollar"),
                ("dcid:B", "dcid:Euro"),
                ("dcid:C", None),
            )
        ],
    )

    result = run_filter(
        frame, predicates=(CategoryMatchPredicate(column="unit", values=("dcid:Euro",)),)
    )

    assert [(dict(s.row_key)["reporter"], s.unit) for s in result.kept_row_summaries] == [
        ("dcid:B", "dcid:Euro")
    ]


def _converted_frame() -> Frame:
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": reporter,
                "time_period": "2020",
                "value": value,
                "unit": {"currency": "USD", "prices": "current"},
            }
            for reporter, value in (("dcid:A", 1.0), ("dcid:B", 3.0))
        ],
    )


def test_a_kept_converted_unit_is_carried_into_the_output_unchanged() -> None:
    frame = _converted_frame()

    result = run_filter(
        frame, predicates=(OrderedThresholdPredicate(column="value", op="gt", value=2.0),)
    )

    assert result.frame.rows == tuple(row for row in frame.rows if row[3] == 3.0)
    assert [summary.unit for summary in result.kept_row_summaries] == [
        UnitRequest(currency="USD", prices="current", base_year=None)
    ]


def test_a_converted_unit_filtered_away_leaves_a_valid_result() -> None:
    result = run_filter(
        _converted_frame(),
        predicates=(OrderedThresholdPredicate(column="value", op="gt", value=5.0),),
    )

    assert result.kept_row_count == 0
    assert result.frame.num_rows == 0
