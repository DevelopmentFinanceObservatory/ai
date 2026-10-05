"""`run_aggregate`: group-by reduction, coverage over absent grain cells,
determinism (order/identity perturbation), and every refusal condition.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from analysis_mcp.engine.aggregate import (
    AGGREGATIONS,
    GroupSummary,
    NonFiniteAggregateResultError,
    StatVarUnresolvedError,
    UnknownGroupByColumnError,
    UnsupportedAggregationError,
    _reduce,
    run_aggregate,
)
from analysis_mcp.engine.common import (
    FacetHeterogeneousError,
    UnitHeterogeneousError,
    UnitNotPresentError,
)
from analysis_mcp.fixtures import (
    multi_unit_frame,
    plain_single_unit_frame,
    sparse_coverage_frame,
    unit_and_unitless_row_frame,
)
from dfo_frame_grammar import Frame

_FIXTURES = Path(__file__).parent / "fixtures"


def _frame(rows: list[dict[str, object]], *, role_keys: list[str] | None = None) -> Frame:
    return Frame.build(role_keys=role_keys or ["reporter"], rows=rows)


def _row(
    *, reporter: str = "dcid:USA", year: str = "2020", value: float = 1.0, **extra: object
) -> dict:
    return {
        "stat_var": "dcid:X",
        "reporter": reporter,
        "time_period": year,
        "value": value,
        **extra,
    }


# Golden file -----------------------------------------------------------------


def test_golden_worked_aggregate_matches_the_committed_fixture() -> None:
    golden = json.loads((_FIXTURES / "aggregate_golden.json").read_text())
    frame = plain_single_unit_frame()

    result = run_aggregate(
        frame, group_by=tuple(golden["group_by"]), aggregation=golden["aggregation"]
    )

    assert result.resolved_unit == golden["expected_resolved_unit"]
    coverage = golden["expected_coverage"]
    assert result.realised_coverage_fraction == coverage["realised_coverage_fraction"]
    assert result.present_cell_count == coverage["present_cell_count"]
    assert result.expected_cell_count == coverage["expected_cell_count"]
    assert result.skipped_cell_count == coverage["skipped_cell_count"]

    actual_groups = [
        {
            "group_key": [list(pair) for pair in summary.group_key],
            "value": summary.value,
            "row_count": summary.row_count,
            "time_period_min": summary.time_period_min,
            "time_period_max": summary.time_period_max,
        }
        for summary in result.group_summaries
    ]
    assert actual_groups == golden["expected_groups"]


# Determinism -------------------------------------------------------------------


def test_replay_from_raw_parameters_alone_is_bit_identical() -> None:
    """No `AggregateOpSpec` in the loop -- exactly `execution_record.content`'s
    own `parameters` shape, parsed as plain JSON."""
    frame = plain_single_unit_frame()
    original = run_aggregate(frame, group_by=("reporter",), aggregation="sum")

    raw_parameters = json.loads('{"aggregation": "sum", "group_by": ["reporter"], "unit": null}')
    replayed = run_aggregate(
        frame,
        group_by=tuple(raw_parameters["group_by"]),
        aggregation=raw_parameters["aggregation"],
        pinned_unit=raw_parameters["unit"],
    )

    assert replayed.frame == original.frame
    assert replayed.group_summaries == original.group_summaries


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


def _summary_key(summary: GroupSummary) -> tuple[object, ...]:
    """`GroupSummary` equality with `group_key` order-insensitive --
    `group_key`'s own tuple order mirrors whatever order `group_by` was
    called with (see `run_aggregate`'s `group_key_map = dict(zip(group_by,
    key, ...))`), so two calls differing only in `group_by`'s column order
    produce genuinely different `group_key` tuples despite describing the
    identical group."""
    return (
        frozenset(summary.group_key),
        summary.value,
        summary.row_count,
        summary.time_period_min,
        summary.time_period_max,
    )


def test_row_key_order_and_group_by_order_perturbation_produce_an_equivalent_result() -> None:
    """The two perturbations that actually reach this engine (row order
    itself is absorbed by `Frame.build`'s own sort, so shuffling it proves
    nothing): each row mapping's own key insertion order, and `group_by`'s
    own column order. `group_by=("reporter", "recipient")` -- two
    same-typed role columns, not a role column and `time_period` -- makes
    the second dimension non-trivial without also making a mis-association
    unobservable: `time_period` is regex-bound
    (`^[0-9]{4}(-[0-9]{2}(-[0-9]{2})?)?$`) while a role value like
    `dcid:USA` never matches it, so pairing `group_by` with the wrong
    `key` value (`run_aggregate`'s own `group_key_map = dict(zip(group_by,
    key, strict=True))`) would crash inside this function's own
    `Frame.build(role_keys=role_in_group_by, rows=output_rows)` call before
    either of this test's own assertions ever runs -- proving the crash
    happened, not that the mis-association was ever actually caught by the
    invariance check below. Two role columns are both plain `dcid:`
    strings, so the identical mis-association instead produces a
    well-formed (but wrong) frame, which is what makes this test's own
    assertions the thing that actually catches it. `run_aggregate` never
    canonicalises `group_by` itself (see `frame_ref_resolution.canonical_
    columns_or_refusal`, which the tool boundary calls before this function
    ever runs), so `group_key`'s own tuple order does move with the
    perturbation; `_summary_key` is what makes the comparison honest about
    that instead of hiding it."""
    frame = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=10.0, recipient="dcid:KEN"),
            _row(reporter="dcid:GBR", year="2020", value=20.0, recipient="dcid:TZA"),
        ],
        role_keys=["reporter", "recipient"],
    )
    original = run_aggregate(frame, group_by=("reporter", "recipient"), aggregation="sum")

    reordered_rows = [_row_mapping_with_reversed_key_order(frame, row) for row in frame.rows]
    reordered_frame = Frame.build(role_keys=["reporter", "recipient"], rows=reordered_rows)
    perturbed = run_aggregate(
        reordered_frame, group_by=("recipient", "reporter"), aggregation="sum"
    )

    assert perturbed.frame == original.frame
    assert {_summary_key(s) for s in perturbed.group_summaries} == {
        _summary_key(s) for s in original.group_summaries
    }


def test_fsum_is_used_not_a_naive_running_sum() -> None:
    """At this magnitude a naive `+=` loses precision that `math.fsum` doesn't --
    proof the engine actually calls `fsum` rather than an accumulator loop."""
    values = [1e16, 1.0, 1.0, 1.0]
    frame = _frame([_row(year=str(2017 + i), value=v) for i, v in enumerate(values)])

    result = run_aggregate(frame, group_by=(), aggregation="sum")

    naive_sum = 0.0
    for value in values:
        naive_sum += value

    assert result.group_summaries[0].value == math.fsum(values)
    assert result.group_summaries[0].value != naive_sum


def test_mean_divides_the_fsum_by_the_group_row_count() -> None:
    frame = _frame([_row(value=1.0), _row(reporter="dcid:GBR", value=3.0)])
    result = run_aggregate(frame, group_by=(), aggregation="mean")
    assert result.group_summaries[0].value == 2.0


# max / min -----------------------------------------------------------------------


def test_max_returns_the_value_and_the_period_that_supplied_it() -> None:
    frame = _frame(
        [
            _row(year="2020", value=10.0),
            _row(year="2021", value=30.0),
            _row(year="2022", value=20.0),
        ]
    )
    result = run_aggregate(frame, group_by=(), aggregation="max")
    summary = result.group_summaries[0]
    assert summary.value == 30.0
    assert summary.value_period == "2021"


def test_min_returns_the_value_and_the_period_that_supplied_it() -> None:
    frame = _frame(
        [
            _row(year="2020", value=10.0),
            _row(year="2021", value=30.0),
            _row(year="2022", value=20.0),
        ]
    )
    result = run_aggregate(frame, group_by=(), aggregation="min")
    summary = result.group_summaries[0]
    assert summary.value == 10.0
    assert summary.value_period == "2020"


def test_sum_and_mean_leave_value_period_none() -> None:
    frame = _frame([_row(year="2020", value=10.0), _row(year="2021", value=30.0)])
    for aggregation in ("sum", "mean"):
        result = run_aggregate(frame, group_by=(), aggregation=aggregation)
        assert result.group_summaries[0].value_period is None


_USA_2018_TO_2022_GBR_TO_2020 = [
    *(_row(year=str(year), value=year - 2017.0) for year in range(2018, 2023)),
    *(_row(reporter="dcid:GBR", year=str(year), value=year - 2017.0) for year in range(2018, 2021)),
]


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
@pytest.mark.parametrize(
    ("rows", "label"),
    [
        pytest.param(_USA_2018_TO_2022_GBR_TO_2020, "2018/2022", id="call-wide-extent"),
        pytest.param(
            [_row(year="2018"), _row(reporter="dcid:GBR", year="2022")],
            "2018/2022",
            id="one-period-per-group",
        ),
        pytest.param(
            [_row(year="2020"), _row(reporter="dcid:GBR", year="2020")], "2020", id="one-point"
        ),
        pytest.param(
            [_row(year="2018/2022"), _row(reporter="dcid:GBR", year="2018/2022")],
            "2018/2022",
            id="one-span",
        ),
    ],
)
def test_sum_and_mean_stamp_every_row_with_the_call_wide_span(
    rows: list[dict], label: str, aggregation: str
) -> None:
    """Every output row carries the one label covering every period the
    call reduced, including a group whose own rows stop earlier."""
    result = run_aggregate(_frame(rows), group_by=("reporter",), aggregation=aggregation)
    time_idx = result.frame.columns.index("time_period")
    assert [row[time_idx] for row in result.frame.rows] == [label, label]


def test_max_keeps_each_groups_winning_period_beside_the_call_wide_extent() -> None:
    frame = _frame(_USA_2018_TO_2022_GBR_TO_2020)
    result = run_aggregate(frame, group_by=("reporter",), aggregation="max")
    time_idx = result.frame.columns.index("time_period")
    reporter_idx = result.frame.columns.index("reporter")
    periods = {row[reporter_idx]: row[time_idx] for row in result.frame.rows}
    assert periods == {"dcid:USA": "2022", "dcid:GBR": "2020"}


def test_grouped_time_period_keeps_each_groups_key_under_sum() -> None:
    frame = _frame(_USA_2018_TO_2022_GBR_TO_2020)
    result = run_aggregate(frame, group_by=("time_period",), aggregation="sum")
    time_idx = result.frame.columns.index("time_period")
    assert [row[time_idx] for row in result.frame.rows] == [str(y) for y in range(2018, 2023)]


def test_single_row_group_under_max_carries_that_rows_own_period() -> None:
    frame = _frame([_row(year="2020", value=5.0)])
    result = run_aggregate(frame, group_by=(), aggregation="max")
    summary = result.group_summaries[0]
    assert summary.value == 5.0
    assert summary.value_period == "2020"


def test_output_row_time_period_carries_the_winning_period_under_max() -> None:
    """With `time_period` outside `group_by`, the output row's own
    `time_period` cell names the period that actually supplied the max
    value, not just the span's own latest year."""
    frame = _frame(
        [
            _row(year="2020", value=10.0),
            _row(year="2021", value=30.0),
            _row(year="2022", value=20.0),
        ]
    )
    result = run_aggregate(frame, group_by=(), aggregation="max")
    time_idx = result.frame.columns.index("time_period")
    assert result.frame.rows[0][time_idx] == "2021"


@pytest.mark.parametrize(
    "periods", [("2021", "2020"), ("2020", "2021")], ids=["latest_first", "earliest_first"]
)
def test_reduce_breaks_a_max_tie_by_the_earliest_period_regardless_of_order(
    periods: tuple[str, str],
) -> None:
    value, value_period = _reduce([10.0, 10.0], list(periods), aggregation="max")
    assert value == 10.0
    assert value_period == "2020"


@pytest.mark.parametrize(
    "periods", [("2021", "2020"), ("2020", "2021")], ids=["latest_first", "earliest_first"]
)
def test_reduce_breaks_a_min_tie_by_the_earliest_period_regardless_of_order(
    periods: tuple[str, str],
) -> None:
    value, value_period = _reduce([10.0, 10.0], list(periods), aggregation="min")
    assert value == 10.0
    assert value_period == "2020"


# Coverage over absent grain cells -----------------------------------------------


def test_sparse_frame_coverage_counts_absent_grid_cells() -> None:
    frame = sparse_coverage_frame()
    result = run_aggregate(frame, group_by=("recipient", "time_period"), aggregation="sum")
    assert result.expected_cell_count == 12  # 4 recipients x 3 years
    assert result.present_cell_count == 9
    assert result.skipped_cell_count == 3
    assert result.realised_coverage_fraction == pytest.approx(0.75)


def test_full_coverage_frame_has_no_skipped_cells() -> None:
    frame = plain_single_unit_frame()
    result = run_aggregate(frame, group_by=("reporter", "time_period"), aggregation="sum")
    assert result.skipped_cell_count == 0
    assert result.realised_coverage_fraction == 1.0


def test_empty_group_by_reduces_the_whole_frame_to_one_group() -> None:
    frame = plain_single_unit_frame()
    result = run_aggregate(frame, group_by=(), aggregation="sum")
    assert len(result.group_summaries) == 1
    assert result.expected_cell_count == 1
    assert result.present_cell_count == 1


# Unit/facet resolution wired through to the engine -----------------------------


def test_heterogeneous_unit_frame_refuses() -> None:
    frame = multi_unit_frame()
    with pytest.raises(UnitHeterogeneousError):
        run_aggregate(frame, group_by=("reporter",), aggregation="sum")


def test_pinned_unit_disambiguates_and_filters_the_computed_rows() -> None:
    frame = multi_unit_frame()
    result = run_aggregate(
        frame, group_by=("reporter",), aggregation="sum", pinned_unit="dcid:USDollar"
    )
    assert result.resolved_unit == "dcid:USDollar"
    # Only the USD-denominated rows (2 years x 3 donors) contribute.
    assert {
        key: value for summary in result.group_summaries for key, value in summary.group_key
    } != {}


def test_pinned_unit_not_present_refuses() -> None:
    frame = multi_unit_frame()
    with pytest.raises(UnitNotPresentError):
        run_aggregate(frame, group_by=("reporter",), aggregation="sum", pinned_unit="dcid:EUR")


def test_unpinned_path_excludes_unitless_rows_like_the_pinned_path_does() -> None:
    """The declared policy: a row leaving `unit` unbound is excluded from
    the arithmetic on the unpinned path exactly as the pinned path already
    excludes it (a pinned filter can never match `None`) -- not silently
    summed in, and reported structurally as a `unit_unbound` exclusion (see
    `test_unpinned_path_reports_the_excluded_unitless_row_as_an_exclusion`),
    not restated as a separate warning."""
    frame = unit_and_unitless_row_frame()
    result = run_aggregate(frame, group_by=(), aggregation="sum")
    assert result.group_summaries[0].value == 100.0
    assert result.resolved_unit == "dcid:USDollar"
    assert len(result.exclusions.records) == 1
    assert result.exclusions.records[0].reason == "unit_unbound"


def test_unpinned_path_reports_the_excluded_unitless_row_as_an_exclusion() -> None:
    """The warning above names the excluded count in prose; the structured
    fact a consumer can act on lives in `result.exclusions`: one
    `unit_unbound` record naming ETH's own row and its `recipient`."""
    frame = unit_and_unitless_row_frame()
    result = run_aggregate(frame, group_by=(), aggregation="sum")
    assert len(result.exclusions.records) == 1
    record = result.exclusions.records[0]
    assert record.reason == "unit_unbound"
    assert record.row_count == 1
    assert record.entities == {"recipient": ("dcid:ETH",)}
    assert record.entities_truncated is False


def test_excluded_unitless_row_still_counts_as_a_skipped_cell_in_a_grouped_aggregate() -> None:
    """An excluded row must not silently delete its own whole group from
    the grid: `expected_cell_count` is derived from the rows before
    `exclude_unitless` drops any, so ETH's own group counts as an expected-
    but-missing cell -- coverage accounting is this module's honesty
    mechanism, and a 100%-coverage report here would be false."""
    frame = unit_and_unitless_row_frame()
    result = run_aggregate(frame, group_by=("recipient",), aggregation="sum")
    assert len(result.group_summaries) == 1
    assert result.expected_cell_count == 2
    assert result.present_cell_count == 1
    assert result.skipped_cell_count == 1
    assert result.realised_coverage_fraction == pytest.approx(0.5)


def test_pinned_path_reports_the_same_coverage_as_the_unpinned_path() -> None:
    """Pinning `pinned_unit` already excludes ETH's own unitless row for
    free (`None` never matches a real pin) -- that must not make the
    pinned path's own `expected_cell_count` blind to the same excluded
    group the unpinned path already counts as skipped. Both paths warn
    on the identical unitless row; only one of them should not get to
    call it 100% coverage."""
    frame = unit_and_unitless_row_frame()
    unpinned = run_aggregate(frame, group_by=("recipient",), aggregation="sum")
    pinned = run_aggregate(
        frame, group_by=("recipient",), aggregation="sum", pinned_unit="dcid:USDollar"
    )
    assert pinned.expected_cell_count == unpinned.expected_cell_count == 2
    assert pinned.present_cell_count == unpinned.present_cell_count == 1
    assert pinned.skipped_cell_count == unpinned.skipped_cell_count == 1
    assert (
        pinned.realised_coverage_fraction
        == unpinned.realised_coverage_fraction
        == pytest.approx(0.5)
    )


def test_pinned_path_reports_excluded_unitless_rows_same_as_the_unpinned_path() -> None:
    """Same policy, same observability: pinning `pinned_unit` already
    excludes the unitless row from the arithmetic for free (`None` never
    matches a real pin), but that must still surface, structurally, as the
    same `unit_unbound` exclusion the unpinned path already reports, not
    silence."""
    frame = unit_and_unitless_row_frame()
    result = run_aggregate(frame, group_by=(), aggregation="sum", pinned_unit="dcid:USDollar")
    assert result.group_summaries[0].value == 100.0
    assert any(record.reason == "unit_unbound" for record in result.exclusions.records)


def test_facet_heterogeneous_frame_refuses() -> None:
    frame = _frame(
        [
            _row(provenance="dcid:OECD_DAC2A"),
            _row(reporter="dcid:GBR", provenance="dcid:OECD_DAC1"),
        ]
    )
    with pytest.raises(FacetHeterogeneousError):
        run_aggregate(frame, group_by=("reporter",), aggregation="sum")


def test_facet_auto_selection_is_recorded_on_the_result() -> None:
    frame = _frame([_row(provenance="dcid:OECD_DAC2A"), _row(reporter="dcid:GBR")])
    result = run_aggregate(frame, group_by=("reporter",), aggregation="sum")
    assert result.facet_selections[0].axis == "provenance"
    assert result.warnings


# Refusals owned by the engine itself --------------------------------------------


def test_unsupported_aggregation_refuses() -> None:
    frame = plain_single_unit_frame()
    with pytest.raises(UnsupportedAggregationError) as exc_info:
        run_aggregate(frame, group_by=("reporter",), aggregation="median")
    assert exc_info.value.aggregation == "median"


def test_every_supported_aggregation_actually_runs() -> None:
    frame = plain_single_unit_frame()
    for aggregation in AGGREGATIONS:
        result = run_aggregate(frame, group_by=("reporter",), aggregation=aggregation)
        assert result.group_summaries


def test_unknown_group_by_column_refuses() -> None:
    frame = plain_single_unit_frame()
    with pytest.raises(UnknownGroupByColumnError) as exc_info:
        run_aggregate(frame, group_by=("donor",), aggregation="sum")
    assert exc_info.value.columns == ("donor",)


@pytest.mark.parametrize("disallowed_column", ["value", "status"])
def test_grouping_by_a_disallowed_column_refuses(disallowed_column: str) -> None:
    frame = plain_single_unit_frame()
    with pytest.raises(UnknownGroupByColumnError):
        run_aggregate(frame, group_by=(disallowed_column,), aggregation="sum")


def test_empty_frame_refuses_with_stat_var_unresolved() -> None:
    frame = Frame.build(role_keys=["reporter"], rows=[])
    with pytest.raises(StatVarUnresolvedError) as exc_info:
        run_aggregate(frame, group_by=(), aggregation="sum")
    assert exc_info.value.distinct_stat_vars == ()


def test_multiple_stat_vars_without_stat_var_in_group_by_refuses() -> None:
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:A", "reporter": "dcid:USA", "time_period": "2020", "value": 1.0},
            {"stat_var": "dcid:B", "reporter": "dcid:USA", "time_period": "2020", "value": 2.0},
        ],
    )
    with pytest.raises(StatVarUnresolvedError) as exc_info:
        run_aggregate(frame, group_by=("reporter",), aggregation="sum")
    assert set(exc_info.value.distinct_stat_vars) == {"dcid:A", "dcid:B"}


def test_multiple_stat_vars_with_stat_var_in_group_by_succeeds() -> None:
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:A", "reporter": "dcid:USA", "time_period": "2020", "value": 1.0},
            {"stat_var": "dcid:B", "reporter": "dcid:USA", "time_period": "2020", "value": 2.0},
        ],
    )
    result = run_aggregate(frame, group_by=("stat_var",), aggregation="sum")
    assert len(result.group_summaries) == 2


def test_non_finite_result_refuses() -> None:
    near_max = 1.0e308
    frame = _frame([_row(year="2020", value=near_max), _row(year="2021", value=near_max)])
    with pytest.raises(NonFiniteAggregateResultError) as exc_info:
        run_aggregate(frame, group_by=(), aggregation="sum")
    assert exc_info.value.aggregation == "sum"
    assert not math.isfinite(near_max + near_max)  # sanity: this really does overflow
