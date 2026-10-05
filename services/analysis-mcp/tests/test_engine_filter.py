"""`run_filter`: row-value predicate filtering, the no-unit-resolution
policy, the valid-empty-result policy, determinism (order perturbation), and
every refusal condition.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from analysis_mcp.engine.filter import (
    FilterPredicateTypeMismatchError,
    FilterPredicateValueTypeMismatchError,
    FilterUnitConversionUnsupportedError,
    FilterUnknownColumnError,
    run_filter,
)
from analysis_mcp.fixtures import (
    multi_unit_frame,
    plain_single_unit_frame,
    unit_partially_declared_frame,
)
from analysis_mcp.op_spec import CategoryMatchPredicate, OrderedThresholdPredicate
from dfo_frame_grammar import Frame

_FIXTURES = Path(__file__).parent / "fixtures"


# Golden file -----------------------------------------------------------------


def test_golden_worked_filter_matches_the_committed_fixture() -> None:
    golden = json.loads((_FIXTURES / "filter_golden.json").read_text())
    frame = plain_single_unit_frame()

    predicates = tuple(
        OrderedThresholdPredicate(column=p["column"], op=p["op"], value=p["value"])
        for p in golden["predicates"]
    )
    result = run_filter(frame, predicates=predicates)

    assert result.total_row_count == golden["expected_total_row_count"]
    assert result.kept_row_count == golden["expected_kept_row_count"]
    assert result.dropped_row_count == golden["expected_dropped_row_count"]
    actual = [
        {"row_key": [list(pair) for pair in s.row_key], "value": s.value, "unit": s.unit}
        for s in result.kept_row_summaries
    ]
    assert actual == golden["expected_kept_row_summaries"]


# Declared policy: row-value predicates only, no unit/facet resolution ----------


def test_policy_predicates_are_applied_as_a_conjunction() -> None:
    frame = plain_single_unit_frame()
    predicates = (
        OrderedThresholdPredicate(column="value", op="gte", value=50_000_000_000),
        CategoryMatchPredicate(column="reporter", values=("dcid:USA",)),
    )
    result = run_filter(frame, predicates=predicates)
    assert result.kept_row_count == 2  # USA 2022 and 2023 only


def test_policy_no_predicates_keeps_every_row() -> None:
    frame = plain_single_unit_frame()
    result = run_filter(frame, predicates=())
    assert result.kept_row_count == result.total_row_count
    assert result.dropped_row_count == 0


def test_dropped_rows_are_reported_as_a_predicate_exclusion() -> None:
    """`plain_single_unit_frame`'s own five donors at 2021: a
    `time_period >= "2022"` predicate drops exactly those five rows, one per
    donor, reported as one `predicate` exclusion."""
    frame = plain_single_unit_frame()
    predicates = (OrderedThresholdPredicate(column="time_period", op="gte", value="2022"),)
    result = run_filter(frame, predicates=predicates)
    assert result.dropped_row_count == 5
    assert len(result.exclusions.records) == 1
    record = result.exclusions.records[0]
    assert record.reason == "predicate"
    assert record.row_count == 5
    assert record.entities["reporter"] == (
        "dcid:DEU",
        "dcid:FRA",
        "dcid:GBR",
        "dcid:JPN",
        "dcid:USA",
    )


def test_policy_category_match_on_the_unit_column_narrows_a_heterogeneous_frame() -> None:
    """The declared boundary: filter is exempt from unit/facet resolution.
    Predicates apply directly to the frame's raw column values, so a
    heterogeneous-unit frame is filterable without refusing -- narrowing by
    `unit` itself is an ordinary category_match, not scope narrowing."""
    frame = multi_unit_frame()
    result = run_filter(
        frame, predicates=(CategoryMatchPredicate(column="unit", values=("dcid:USDollar",)),)
    )
    assert result.kept_row_count > 0
    assert all(summary.unit == "dcid:USDollar" for summary in result.kept_row_summaries)


def test_policy_empty_result_is_not_an_error() -> None:
    """The declared policy: an empty result after
    filtering is not a refusal -- `Frame.build` accepts a valid, columns-only
    empty frame, so this test fails if a `filter_empty_result` refusal is
    ever introduced instead."""
    frame = plain_single_unit_frame()
    result = run_filter(
        frame, predicates=(OrderedThresholdPredicate(column="value", op="gt", value=1e20),)
    )
    assert result.kept_row_count == 0
    assert result.frame.rows == ()


# Determinism -------------------------------------------------------------------


def _row_mapping_with_reversed_key_order(
    frame: Frame, row: tuple[object, ...]
) -> dict[str, object]:
    """`row`'s own cells, keyed by `frame.columns` but inserted in the
    opposite order (`status` dropped -- reserved, never caller-supplied).
    `Frame.build` sorts the *list* of rows it's given, which absorbs any
    perturbation to row order; it says nothing about a perturbation to one
    row *mapping's* own key insertion order, which this builds instead."""
    pairs = [(key, value) for key, value in zip(frame.columns, row, strict=True) if key != "status"]
    return dict(reversed(pairs))


def test_row_key_order_and_predicate_order_perturbation_produce_an_identical_result() -> None:
    """The two perturbations that actually reach this engine (row order
    itself is absorbed by `Frame.build`'s own sort, so shuffling it proves
    nothing): each row mapping's own key insertion order, and `predicates`'
    own tuple order. Two predicates, not one, make the second dimension
    non-trivial. Unlike `group_by`/`rank_by`/`entity_by`/`join_keys`,
    `predicates` is a conjunction over every row independently -- keeping a
    row is order-independent by construction, and `FilterRowSummary.row_key`
    is built from `frame.columns`, never from `predicates`' own order -- so
    (on this valid, non-conflicting pair) this is a direct equality, not the
    order-insensitive comparison the column-sequence verbs need."""
    frame = plain_single_unit_frame()
    predicates = (
        OrderedThresholdPredicate(column="value", op="gte", value=20_000_000_000),
        CategoryMatchPredicate(
            column="reporter", values=("dcid:USA", "dcid:GBR", "dcid:FRA", "dcid:JPN", "dcid:DEU")
        ),
    )
    original = run_filter(frame, predicates=predicates)

    reordered_rows = [_row_mapping_with_reversed_key_order(frame, row) for row in frame.rows]
    reordered_frame = Frame.build(role_keys=["reporter"], rows=reordered_rows)
    perturbed = run_filter(reordered_frame, predicates=tuple(reversed(predicates)))

    assert perturbed.kept_row_summaries == original.kept_row_summaries
    assert perturbed.frame == original.frame


# Predicate operators -------------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "threshold", "expected_kept"),
    [
        ("gt", 55_303_000_000, 1),
        ("gte", 55_303_000_000, 2),
        ("lt", 15_476_000_000, 1),
        ("lte", 15_476_000_000, 2),
        ("eq", 15_476_000_000, 1),
        ("ne", 15_476_000_000, 14),
    ],
)
def test_ordered_threshold_operators_over_value(
    op: str, threshold: float, expected_kept: int
) -> None:
    frame = plain_single_unit_frame()
    result = run_filter(
        frame, predicates=(OrderedThresholdPredicate(column="value", op=op, value=threshold),)
    )
    assert result.kept_row_count == expected_kept


# Ordered threshold over time_period -----------------------------------------------


def _mixed_grain_frame() -> Frame:
    """One reporter's own series across three grains, so `time_period`'s
    lexical ordering is exercised across a year, a year-month, and another
    year, not just within one grain."""
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:oda_disbursed",
                "reporter": "dcid:USA",
                "time_period": period,
                "value": value,
                "unit": "dcid:USDollar",
            }
            for period, value in (
                ("2014", 1.0),
                ("2015", 2.0),
                ("2015-06", 3.0),
                ("2016", 4.0),
            )
        ],
    )


def test_ordered_threshold_over_time_period_filters_a_mixed_grain_frame() -> None:
    frame = _mixed_grain_frame()
    result = run_filter(
        frame, predicates=(OrderedThresholdPredicate(column="time_period", op="gte", value="2015"),)
    )
    kept_periods = {summary.row_key[-1][1] for summary in result.kept_row_summaries}
    assert kept_periods == {"2015", "2015-06", "2016"}


def test_ordered_threshold_over_time_period_keeps_a_coarse_period_at_its_own_start() -> None:
    """`gt "2014"` keeps `"2015"`: a coarse period compares at its own
    start, so the year `"2015"` sorts after the year `"2014"` even though
    neither carries a month or day."""
    frame = _mixed_grain_frame()
    result = run_filter(
        frame, predicates=(OrderedThresholdPredicate(column="time_period", op="gt", value="2014"),)
    )
    kept_periods = {summary.row_key[-1][1] for summary in result.kept_row_summaries}
    assert "2015" in kept_periods


def test_ordered_threshold_over_time_period_drops_a_coarse_period_below_a_finer_bound() -> None:
    """`gte "2015-06"` drops `"2015"`: the year `"2015"` sorts before the
    finer `"2015-06"` bound, the same one semantic edge `gt "2014"` shows
    from the other side."""
    frame = _mixed_grain_frame()
    result = run_filter(
        frame,
        predicates=(OrderedThresholdPredicate(column="time_period", op="gte", value="2015-06"),),
    )
    kept_periods = {summary.row_key[-1][1] for summary in result.kept_row_summaries}
    assert "2015" not in kept_periods
    assert kept_periods == {"2015-06", "2016"}


@pytest.mark.parametrize(
    ("column", "operand", "actual_type", "expected_type"),
    [
        pytest.param("time_period", 2015.0, "float", "str", id="number-against-time-period"),
        pytest.param("value", "2015", "str", "float", id="period-against-value"),
        pytest.param(
            "time_period", "2015/2016", "interval", "point", id="year-interval-against-time-period"
        ),
        pytest.param(
            "time_period",
            "2015-01/2016-12",
            "interval",
            "point",
            id="month-interval-against-time-period",
        ),
        pytest.param(
            "time_period",
            "2015-01-01/2016-12-31",
            "interval",
            "point",
            id="day-interval-against-time-period",
        ),
    ],
)
def test_ordered_threshold_refuses_an_operand_of_the_wrong_type(
    column: str, operand: float | str, actual_type: str, expected_type: str
) -> None:
    """An interval operand against `time_period` would compare as a string,
    so `gte "2015/2016"` would drop 2015 and keep every later period."""
    frame = _mixed_grain_frame()
    with pytest.raises(FilterPredicateValueTypeMismatchError) as exc_info:
        run_filter(
            frame,
            predicates=(OrderedThresholdPredicate(column=column, op="gte", value=operand),),
        )
    assert exc_info.value.column == column
    assert exc_info.value.actual_type == actual_type
    assert exc_info.value.expected_type == expected_type


# Refusals ------------------------------------------------------------------------


def test_unknown_predicate_column_refuses() -> None:
    frame = plain_single_unit_frame()
    with pytest.raises(FilterUnknownColumnError) as exc_info:
        run_filter(
            frame, predicates=(OrderedThresholdPredicate(column="donor", op="gte", value=1),)
        )
    assert exc_info.value.columns == ("donor",)


# Schema-level type check -----------------------------------------------------


def test_category_match_on_unit_over_a_null_unit_row_keeps_declared_and_drops_null() -> None:
    """The documented workflow: `category_match` on `unit` narrows a
    heterogeneous frame -- including one row whose `unit` was never
    declared at all. The null row never matches (it isn't a type error),
    so it is dropped like any other non-matching row, never refused."""
    frame = unit_partially_declared_frame()
    result = run_filter(
        frame, predicates=(CategoryMatchPredicate(column="unit", values=("dcid:USDollar",)),)
    )
    assert result.kept_row_count == 2
    assert result.dropped_row_count == 1
    assert all(summary.unit == "dcid:USDollar" for summary in result.kept_row_summaries)


def test_predicate_order_does_not_change_the_outcome() -> None:
    """A mistyped predicate and a legitimate one, in both orders, refuse
    identically -- the type check runs over every predicate before any row
    is inspected, so the outcome never depends on which predicate a
    short-circuiting per-row check would have reached first."""
    frame = plain_single_unit_frame()
    mistyped = OrderedThresholdPredicate(column="reporter", op="gte", value=1)
    legitimate = CategoryMatchPredicate(column="reporter", values=("dcid:USA",))

    with pytest.raises(FilterPredicateTypeMismatchError) as first_order:
        run_filter(frame, predicates=(mistyped, legitimate))
    with pytest.raises(FilterPredicateTypeMismatchError) as second_order:
        run_filter(frame, predicates=(legitimate, mistyped))

    assert first_order.value.column == second_order.value.column == "reporter"


def test_mistyped_predicate_against_an_empty_frame_still_refuses() -> None:
    """An empty frame type-checks nothing about any actual cell -- the
    check is schema-level, so it fires regardless of row count."""
    empty_frame = Frame.build(role_keys=["reporter"], rows=[])
    with pytest.raises(FilterPredicateTypeMismatchError):
        run_filter(
            empty_frame,
            predicates=(OrderedThresholdPredicate(column="reporter", op="gte", value=1),),
        )


def test_category_match_on_unit_over_a_unit_request_cell_refuses() -> None:
    """A converted frame's `unit` cell can carry a `grammar.UnitRequest`
    object rather than a plain string (`Frame.build` normalises a dict
    `unit` this way). `unit`'s own declared type admits this, so a
    `category_match` predicate on `unit` passes the schema-level check --
    but `UnitRequest in (...)` against a closed set of plain strings is
    always `False`, so before this refusal existed, every row would be
    silently dropped rather than matched or refused. Refuses with
    `FilterUnitConversionUnsupportedError`, not
    `FilterPredicateTypeMismatchError`: the predicate kind and column are
    legal (see that exception's own docstring)."""
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:oda_disbursed",
                "reporter": "dcid:USA",
                "time_period": "2022",
                "value": 1.0,
                "unit": {"currency": "USD", "prices": "current"},
            },
        ],
    )
    with pytest.raises(FilterUnitConversionUnsupportedError) as exc_info:
        run_filter(frame, predicates=(CategoryMatchPredicate(column="unit", values=("USD",)),))
    assert exc_info.value.column == "unit"
    assert exc_info.value.actual_type == "UnitRequest"


def test_predicate_against_status_always_refuses() -> None:
    """`status` is always `None` at this grammar version -- neither
    predicate kind can legitimately touch it."""
    frame = plain_single_unit_frame()
    with pytest.raises(FilterPredicateTypeMismatchError) as exc_info:
        run_filter(frame, predicates=(CategoryMatchPredicate(column="status", values=("x",)),))
    assert exc_info.value.column == "status"
