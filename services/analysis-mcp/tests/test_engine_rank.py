"""`run_rank`: competition ranking, tie-break-by-total-ordering, the
coverage grid, and every refusal condition.
"""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path

import pyarrow as pa
import pytest
from analysis_mcp.engine import rank as rank_module
from analysis_mcp.engine.aggregate import (
    NonFiniteAggregateResultError,
    UnsupportedAggregationError,
    run_aggregate,
)
from analysis_mcp.engine.common import (
    Exclusions,
    StatVarUnresolvedError,
    UnitHeterogeneousError,
    UnknownEntityColumnError,
)
from analysis_mcp.engine.rank import (
    RankAmbiguousEntityError,
    RankNoRowsInWindowError,
    RankResult,
    run_rank,
)
from analysis_mcp.fixtures import (
    multi_unit_frame,
    plain_single_unit_frame,
    unit_and_unitless_row_frame,
)
from dfo_frame_grammar import Frame, build_frame_from_arrow

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


def test_golden_worked_rank_matches_the_committed_fixture() -> None:
    golden = json.loads((_FIXTURES / "rank_golden.json").read_text())
    frame = plain_single_unit_frame()

    result = run_rank(
        frame,
        rank_by=tuple(golden["rank_by"]),
        aggregation=golden["aggregation"],
        window_start=golden["window_start"],
        window_end=golden["window_end"],
        ascending=False,
    )

    assert result.overall_coverage_fraction == golden["expected_overall_coverage_fraction"]
    actual = [
        {
            "entity_key": [list(pair) for pair in er.entity_key],
            "value": er.value,
            "rank": er.rank,
            "row_count": er.row_count,
            "coverage_fraction": er.coverage_fraction,
        }
        for er in result.entity_ranks
    ]
    assert actual == golden["expected_entity_ranks"]


def test_collapse_mode_output_frame_is_unchanged_byte_for_byte() -> None:
    """`per_period`'s own default (`False`) must build the identical output
    frame the collapsing path always has -- `value` the measured value,
    `unit` the resolved unit, one row per entity labelled with the window's
    observed extent. This is the regression bar
    the per-period grouping added in this PR must clear, the same discipline
    `test_numerator_by_empty_reproduces_the_output_frame_byte_for_byte`
    applies to keyed share_of's own `numerator_by=()` default."""
    golden = json.loads((_FIXTURES / "rank_golden.json").read_text())
    frame = plain_single_unit_frame()

    result = run_rank(
        frame,
        rank_by=tuple(golden["rank_by"]),
        aggregation=golden["aggregation"],
        window_start=golden["window_start"],
        window_end=golden["window_end"],
        ascending=False,
    )

    assert result.per_period is False
    assert result.frame.columns == (
        "stat_var",
        "reporter",
        "time_period",
        "value",
        "unit",
        "measurementMethod",
        "provenance",
        "status",
    )
    expected_rows = [
        {
            "stat_var": "dcid:OfficialDevelopmentAssistanceNet_Disbursement",
            "reporter": entry["entity_key"][0][1],
            "time_period": "2021/2023",
            "value": entry["value"],
            "unit": "dcid:USDollar",
            "measurementMethod": None,
            "provenance": None,
            "status": None,
        }
        for entry in golden["expected_entity_ranks"]
    ]
    # `Frame.build` sorts its own rows (by role key, not by rank), so the
    # comparison is order-insensitive -- the row *content*, not the row
    # *order*, is this test's own regression bar; `entity_ranks`' own order
    # is already covered by the golden test above.
    actual_rows = [dict(zip(result.frame.columns, row, strict=True)) for row in result.frame.rows]
    assert sorted(actual_rows, key=lambda row: row["reporter"]) == sorted(
        expected_rows, key=lambda row: row["reporter"]
    )


# Declared tie policy: competition ranking, tie-break by total ordering ---------


def test_policy_tied_values_share_a_competition_rank_number() -> None:
    """The declared policy: competition ranking ("1224") --
    tied entities share a rank number, and the next distinct value's rank
    skips ahead by the tie's own size. Pinned so a switch to dense ("1223")
    or ordinal ("1234") ranking fails this test."""
    frame = _frame(
        [
            _row(reporter="dcid:AAA", year="2020", value=100.0),
            _row(reporter="dcid:BBB", year="2020", value=100.0),
            _row(reporter="dcid:CCC", year="2020", value=50.0),
        ]
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
    )
    ranks = {dict(er.entity_key)["reporter"]: er.rank for er in result.entity_ranks}
    assert ranks == {"dcid:AAA": 1, "dcid:BBB": 1, "dcid:CCC": 3}


def test_policy_tie_order_is_broken_by_the_frame_grammars_total_ordering() -> None:
    """A genuine tie: two entities share a value. The rank NUMBER ties, but
    the ORDER they're listed in is deterministic -- the frame grammar's own
    total ordering over `rank_by`'s values (Unicode code point of the NFC
    form), not insertion order or a hash."""
    frame = _frame(
        [
            _row(reporter="dcid:ZZZ", year="2020", value=100.0),
            _row(reporter="dcid:AAA", year="2020", value=100.0),
        ]
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
    )
    ordered_reporters = [dict(er.entity_key)["reporter"] for er in result.entity_ranks]
    assert ordered_reporters == ["dcid:AAA", "dcid:ZZZ"]
    assert result.entity_ranks[0].rank == result.entity_ranks[1].rank == 1


def test_ascending_reverses_the_primary_order_but_not_the_tie_break() -> None:
    frame = _frame(
        [
            _row(reporter="dcid:AAA", year="2020", value=10.0),
            _row(reporter="dcid:BBB", year="2020", value=20.0),
        ]
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=True,
    )
    assert [dict(er.entity_key)["reporter"] for er in result.entity_ranks] == [
        "dcid:AAA",
        "dcid:BBB",
    ]


# Optional aggregation over a window (reuses aggregate's own reduction) ---------


def test_aggregation_reduces_each_entitys_rows_before_ranking() -> None:
    frame = plain_single_unit_frame()
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation="sum",
        window_start="2021",
        window_end="2023",
        ascending=False,
    )
    assert result.entity_ranks[0].value == pytest.approx(166_683_000_000.0)  # USA


def test_no_aggregation_requires_exactly_one_row_per_entity() -> None:
    frame = _frame([_row(reporter="dcid:USA", year="2020", value=42.0)])
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
    )
    assert result.entity_ranks[0].value == 42.0


def test_entity_value_normalises_negative_zero() -> None:
    """A raw single-row value of IEEE-754 negative zero reaches `run_rank`
    unnormalised (`aggregation=None` takes it as-is) -- `value = entity_
    values[key] + 0.0` normalises it at the engine boundary, not left to
    the digest's own render-time `_format_number`, so `EntityRank.value`
    and `run_rank`'s own output frame store positive zero, not just what
    the digest renders."""
    frame = _frame([_row(reporter="dcid:USA", year="2020", value=-0.0)])
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
    )
    assert repr(result.entity_ranks[0].value) == "0.0"


# max / min -----------------------------------------------------------------------


def test_max_aggregation_reduces_to_the_largest_value_before_ranking() -> None:
    frame = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=10.0),
            _row(reporter="dcid:USA", year="2021", value=30.0),
            _row(reporter="dcid:GBR", year="2020", value=50.0),
            _row(reporter="dcid:GBR", year="2021", value=5.0),
        ]
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation="max",
        window_start="2020",
        window_end="2021",
        ascending=False,
    )
    by_reporter = {dict(er.entity_key)["reporter"]: er.value for er in result.entity_ranks}
    assert by_reporter == {"dcid:USA": 30.0, "dcid:GBR": 50.0}


def test_max_aggregation_labels_the_entity_row_by_the_window_not_the_winning_period() -> None:
    """`run_rank` discards the winning member's own period under `max`/`min`
    -- the output row carries the window's extent, the same as it does
    under `sum`/`mean`, not the period that supplied the extremum -- the
    deliberate asymmetry with `aggregate`."""
    frame = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=30.0),
            _row(reporter="dcid:USA", year="2021", value=10.0),
        ]
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation="max",
        window_start="2020",
        window_end="2021",
        ascending=False,
    )
    entity = result.entity_ranks[0]
    assert entity.value == 30.0
    assert entity.time_period == "2020/2021"


_GBR_2020_ONLY = (("dcid:GBR", "2020"),)
_USA_2018_TO_2022_GBR_2020 = [
    *(("dcid:USA", str(year)) for year in range(2018, 2023)),
    *_GBR_2020_ONLY,
]


@pytest.mark.parametrize(
    ("observations", "window", "limit", "label"),
    [
        pytest.param(
            _USA_2018_TO_2022_GBR_2020,
            ("2018", "2022"),
            None,
            "2018/2022",
            id="points-collapse-to-their-extent",
        ),
        pytest.param(
            _USA_2018_TO_2022_GBR_2020,
            ("2018", "2024"),
            None,
            "2018/2022",
            id="window-past-the-data-stops-at-the-data",
        ),
        pytest.param(_USA_2018_TO_2022_GBR_2020, None, None, "2018/2022", id="no-window"),
        pytest.param(
            _USA_2018_TO_2022_GBR_2020,
            ("2020", "2020"),
            None,
            "2020",
            id="one-period-window-gives-the-point",
        ),
        pytest.param(
            [("dcid:USA", "2018/2022"), ("dcid:GBR", "2018/2022")],
            ("2018", "2022"),
            None,
            "2018/2022",
            id="span-input-passes-through",
        ),
        pytest.param(
            [("dcid:USA", "2018/2022"), ("dcid:GBR", "2018/2022")],
            ("2015", "2024"),
            None,
            "2018/2022",
            id="span-input-inside-a-wider-window-passes-through",
        ),
        pytest.param(
            # USA's four rows outrank GBR's one, so the limit keeps USA, whose
            # own rows span 2019/2022; the label is the population's extent.
            [("dcid:GBR", "2018"), *(("dcid:USA", str(year)) for year in range(2019, 2023))],
            ("2018", "2022"),
            1,
            "2018/2022",
            id="limited-labels-like-unlimited",
        ),
    ],
)
def test_collapsing_labels_every_row_with_the_windows_observed_extent(
    observations: list[tuple[str, str]],
    window: tuple[str, str] | None,
    limit: int | None,
    label: str,
) -> None:
    """GBR's one 2020 row carries the same label as USA's five years: the
    label is the window's observed extent across every ranked entity,
    shared by every kept row, including when a limit keeps fewer."""
    frame = _frame(
        [_row(reporter=reporter, year=year, value=1.0) for reporter, year in observations]
    )
    window_start, window_end = window or (None, None)
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation="sum",
        window_start=window_start,
        window_end=window_end,
        ascending=False,
        limit=limit,
    )
    time_periods = result.frame.columns.index("time_period")
    assert [row[time_periods] for row in result.frame.rows] == [label] * len(result.kept_ranks)
    expected_kept = 1 if limit else len({reporter for reporter, _ in observations})
    assert len(result.kept_ranks) == expected_kept


def test_run_rank_under_max_agrees_with_run_aggregate() -> None:
    frame = plain_single_unit_frame()
    rank_result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation="max",
        window_start=None,
        window_end=None,
        ascending=False,
    )
    aggregate_result = run_aggregate(frame, group_by=("reporter",), aggregation="max")
    rank_values = {dict(er.entity_key)["reporter"]: er.value for er in rank_result.entity_ranks}
    aggregate_values = {
        dict(summary.group_key)["reporter"]: summary.value
        for summary in aggregate_result.group_summaries
    }
    assert rank_values == aggregate_values


# Coverage ------------------------------------------------------------------------


def test_unequal_coverage_across_entities_warns_in_the_digest_facing_warnings() -> None:
    frame = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=1.0),
            _row(reporter="dcid:USA", year="2021", value=2.0),
            _row(reporter="dcid:GBR", year="2020", value=3.0),
        ]
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation="sum",
        window_start="2020",
        window_end="2021",
        ascending=False,
    )
    assert any("unequal coverage" in warning for warning in result.warnings)


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


def _entity_key(entity) -> tuple[object, ...]:
    """`EntityRank` equality with `entity_key` order-insensitive --
    `entity_key`'s own tuple order mirrors whatever order `rank_by` was
    called with (`run_rank`'s `entity_key_map = dict(zip(rank_by, key,
    ...))`), so two calls differing only in `rank_by`'s column order
    produce genuinely different `entity_key` tuples describing the
    identical entity. `rank` (this entity's own assigned position) is kept
    outside the frozenset since it stays an ordinary int either way."""
    return (
        frozenset(entity.entity_key),
        entity.value,
        entity.rank,
        entity.row_count,
        entity.coverage_fraction,
        entity.time_period,
    )


def test_row_key_order_and_rank_by_order_perturbation_produce_an_equivalent_result() -> None:
    """The two perturbations that actually reach this engine (row order
    itself is absorbed by `Frame.build`'s own sort, so shuffling it proves
    nothing): each row mapping's own key insertion order, and `rank_by`'s
    own column order. `rank_by` needs two role columns -- `time_period` is
    disallowed as an entity column (`engine.common._DISALLOWED_ENTITY_
    COLUMNS`), so unlike `aggregate`'s `group_by` this can't reuse
    `plain_single_unit_frame`'s single role and borrow `time_period` as the
    second dimension; a fresh two-role frame is built here instead.
    `run_rank` never canonicalises `rank_by` itself (the tool boundary
    does, before this function ever runs), so `entity_key`'s own tuple
    order does move with the perturbation; `_entity_key` is what makes the
    comparison honest about that instead of hiding it."""
    frame = _frame(
        [
            _row(reporter="dcid:USA", recipient="dcid:KEN", year=year, value=value)
            for year, value in (("2021", 10.0), ("2022", 20.0), ("2023", 30.0))
        ]
        + [
            _row(reporter="dcid:GBR", recipient="dcid:TZA", year=year, value=value)
            for year, value in (("2021", 5.0), ("2022", 8.0), ("2023", 9.0))
        ],
        role_keys=["reporter", "recipient"],
    )
    original = run_rank(
        frame,
        rank_by=("reporter", "recipient"),
        aggregation="sum",
        window_start="2021",
        window_end="2023",
        ascending=False,
    )

    reordered_rows = [_row_mapping_with_reversed_key_order(frame, row) for row in frame.rows]
    reordered_frame = Frame.build(role_keys=["reporter", "recipient"], rows=reordered_rows)
    perturbed = run_rank(
        reordered_frame,
        rank_by=("recipient", "reporter"),
        aggregation="sum",
        window_start="2021",
        window_end="2023",
        ascending=False,
    )

    assert perturbed.frame == original.frame
    assert {_entity_key(e) for e in perturbed.entity_ranks} == {
        _entity_key(e) for e in original.entity_ranks
    }


# Refusals ------------------------------------------------------------------------


def test_unsupported_aggregation_refuses() -> None:
    frame = plain_single_unit_frame()
    with pytest.raises(UnsupportedAggregationError):
        run_rank(
            frame,
            rank_by=("reporter",),
            aggregation="median",
            window_start=None,
            window_end=None,
            ascending=False,
        )


def test_unknown_rank_by_column_refuses() -> None:
    frame = plain_single_unit_frame()
    with pytest.raises(UnknownEntityColumnError):
        run_rank(
            frame,
            rank_by=("donor",),
            aggregation=None,
            window_start=None,
            window_end=None,
            ascending=False,
        )


def test_ambiguous_entity_without_aggregation_refuses() -> None:
    frame = plain_single_unit_frame()  # 5 reporters x 3 years, each reporter has 3 rows
    with pytest.raises(RankAmbiguousEntityError):
        run_rank(
            frame,
            rank_by=("reporter",),
            aggregation=None,
            window_start=None,
            window_end=None,
            ascending=False,
        )


def test_no_rows_in_window_refuses() -> None:
    frame = plain_single_unit_frame()
    with pytest.raises(RankNoRowsInWindowError):
        run_rank(
            frame,
            rank_by=("reporter",),
            aggregation="sum",
            window_start="2099",
            window_end="2100",
            ascending=False,
        )


def test_multiple_stat_vars_refuse_stat_var_unresolved() -> None:
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:A", "reporter": "dcid:USA", "time_period": "2020", "value": 1.0},
            {"stat_var": "dcid:B", "reporter": "dcid:GBR", "time_period": "2020", "value": 2.0},
        ],
    )
    with pytest.raises(StatVarUnresolvedError):
        run_rank(
            frame,
            rank_by=("reporter",),
            aggregation=None,
            window_start=None,
            window_end=None,
            ascending=False,
        )


def test_heterogeneous_unit_frame_refuses() -> None:
    frame = multi_unit_frame()
    with pytest.raises(UnitHeterogeneousError):
        run_rank(
            frame,
            rank_by=("reporter",),
            aggregation="sum",
            window_start=None,
            window_end=None,
            ascending=False,
        )


def test_aggregated_ranking_excludes_unitless_rows_like_aggregate_does() -> None:
    """The same `math.fsum` reduction `aggregate` already applies this
    policy to: ETH's own unitless row must not be silently ranked inside a
    result the output frame labels `dcid:USDollar`."""
    frame = unit_and_unitless_row_frame()
    result = run_rank(
        frame,
        rank_by=("recipient",),
        aggregation="sum",
        window_start=None,
        window_end=None,
        ascending=False,
    )
    assert len(result.entity_ranks) == 1
    assert dict(result.entity_ranks[0].entity_key) == {"recipient": "dcid:KEN"}
    assert result.entity_ranks[0].value == 100.0
    assert result.resolved_unit == "dcid:USDollar"
    assert any(record.reason == "unit_unbound" for record in result.exclusions.records)


def test_coverage_counts_a_wholly_unitless_entity_as_missing_not_absent() -> None:
    """The same coverage invariant `aggregate.py`'s own `expected_cell_count`
    enforces: USA declares USD across both window years, GBR leaves `unit`
    unbound entirely across both -- GBR is excluded from the ranking, but
    it must still count as two expected-but-missing cells, not vanish from
    the coverage denominator right along with the rows that populated it
    (which would otherwise read as 100% coverage of a grid that quietly
    shrank to fit)."""
    frame = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=10.0, unit="dcid:USDollar"),
            _row(reporter="dcid:USA", year="2021", value=20.0, unit="dcid:USDollar"),
            _row(reporter="dcid:GBR", year="2020", value=5.0),
            _row(reporter="dcid:GBR", year="2021", value=6.0),
        ]
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation="sum",
        window_start="2020",
        window_end="2021",
        ascending=False,
    )
    assert len(result.entity_ranks) == 1
    assert dict(result.entity_ranks[0].entity_key) == {"reporter": "dcid:USA"}
    assert result.expected_cell_count == 4
    assert result.present_cell_count == 2
    assert result.overall_coverage_fraction == pytest.approx(0.5)


def test_per_entity_coverage_fraction_uses_the_same_denominator_as_the_overall_figure() -> None:
    """The same shrunken-grid risk as the previous test, but for
    `EntityRank.coverage_fraction` specifically: USA has a row in both of
    the window's two years, GBR's own rows are both excluded (wholly
    unitless), and GBR's exclusion must not silently narrow the grid USA's
    own coverage is measured against. USA's own two present periods over
    the *pre-exclusion* two-period grid is 1.0 either way here, so this
    fixture adds a third window year with no rows for anyone at all --
    present only in the frame's own pinned_unit-free declared-unit
    universe -- so that a denominator still built from `periods_in_window`
    (post-exclusion) would read a different, wrong fraction than one built
    from the full pre-exclusion grid."""
    frame = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=10.0, unit="dcid:USDollar"),
            _row(reporter="dcid:USA", year="2021", value=20.0, unit="dcid:USDollar"),
            _row(reporter="dcid:GBR", year="2020", value=5.0),
            _row(reporter="dcid:GBR", year="2021", value=6.0),
            _row(reporter="dcid:GBR", year="2022", value=7.0),
        ]
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation="sum",
        window_start="2020",
        window_end="2022",
        ascending=False,
    )
    assert len(result.entity_ranks) == 1
    assert result.entity_ranks[0].coverage_fraction == pytest.approx(2 / 3)


def test_non_finite_aggregated_result_refuses() -> None:
    near_max = 1.0e308
    frame = _frame(
        [
            _row(year="2020", value=near_max),
            _row(year="2021", value=near_max),
        ]
    )
    with pytest.raises(NonFiniteAggregateResultError):
        run_rank(
            frame,
            rank_by=("reporter",),
            aggregation="sum",
            window_start=None,
            window_end=None,
            ascending=False,
        )


# Monthly-grain rows inside an annual window ------------------------------------


def test_monthly_grain_rows_inside_an_annual_window_are_counted_not_dropped() -> None:
    """A monthly row counts in the calendar year it falls in, so an annual
    2020-2021 window keeps `2020-03` and `2021-06` and drops `2022-01`. A
    comparison on exact 4-digit strings would drop every monthly row.

    The coverage grid moves with the same comparison: three distinct months
    fall inside the window across two entities, so `expected_cell_count` is
    `2 entities * 3 periods = 6`, and the row label is the months' extent.
    """
    frame = _frame(
        [
            _row(reporter="dcid:USA", year="2020-03", value=10.0),
            _row(reporter="dcid:USA", year="2021-06", value=20.0),
            _row(reporter="dcid:USA", year="2022-01", value=100.0),
            _row(reporter="dcid:GBR", year="2020-01", value=5.0),
            _row(reporter="dcid:GBR", year="2021-06", value=7.0),
        ]
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation="sum",
        window_start="2020",
        window_end="2021",
        ascending=False,
    )
    by_reporter = {dict(er.entity_key)["reporter"]: er for er in result.entity_ranks}
    assert len(result.entity_ranks) == 2
    assert by_reporter["dcid:USA"].value == pytest.approx(30.0)
    assert by_reporter["dcid:USA"].row_count == 2
    assert result.expected_cell_count == 6
    assert result.present_cell_count == 4
    assert {er.time_period for er in result.entity_ranks} == {"2020-01/2021-06"}
    assert result.overall_coverage_fraction == pytest.approx(4 / 6)


@pytest.mark.parametrize(
    ("time_period", "start_year", "end_year", "inside"),
    [
        ("2020", 2020, 2021, True),
        ("2019", 2020, 2021, False),
        ("2020-03", 2020, 2020, True),
        ("2021-12-31", 2020, 2020, False),
        ("2018/2022", 2018, 2022, True),
        ("2018/2022", 2015, 2024, True),
        ("2018/2022", 2022, 2022, False),
        ("2018/2022", 2020, 2024, False),
        ("2018/2022", 2015, 2021, False),
        ("2018-01/2022-12", 2018, 2022, True),
        ("2018-01/2022-12", 2020, 2022, False),
    ],
)
def test_a_period_is_in_a_year_window_when_every_year_it_covers_is(
    time_period: str, start_year: int, end_year: int, inside: bool
) -> None:
    """A span is inside only a window holding both its start and end years,
    so a growth result labelled `2018/2022` survives a 2018-2022 rank
    window and a window that cuts it drops it."""
    assert (
        rank_module._year_in_window(time_period, start_year=start_year, end_year=end_year) is inside
    )


# Per-period mode -----------------------------------------------------------------


def test_per_period_emits_one_row_per_entity_per_period() -> None:
    """Three entities across three periods, every cell present: 9 rows, one
    per (entity, period)."""
    frame = _frame(
        [
            _row(reporter=reporter, year=year, value=value)
            for reporter, values in (
                ("dcid:AAA", (10.0, 20.0, 30.0)),
                ("dcid:BBB", (5.0, 25.0, 15.0)),
                ("dcid:CCC", (1.0, 2.0, 3.0)),
            )
            for year, value in zip(("2020", "2021", "2022"), values, strict=True)
        ]
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
        per_period=True,
    )
    assert result.per_period is True
    assert len(result.entity_ranks) == 9
    assert {er.time_period for er in result.entity_ranks} == {"2020", "2021", "2022"}


def test_per_period_frame_carries_the_position_as_its_value() -> None:
    """The output *frame*'s own `value` column holds the rank position, with
    `unit=None` -- `EntityRank.value` itself stays the measured value (read
    off `result.entity_ranks`), only the frame's own column changes
    meaning."""
    frame = _frame(
        [
            _row(reporter="dcid:AAA", year="2020", value=100.0, unit="dcid:USDollar"),
            _row(reporter="dcid:BBB", year="2020", value=50.0, unit="dcid:USDollar"),
        ]
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
        per_period=True,
    )
    by_reporter = {dict(er.entity_key)["reporter"]: er for er in result.entity_ranks}
    assert by_reporter["dcid:AAA"].value == 100.0  # EntityRank.value: the measured value
    assert by_reporter["dcid:AAA"].rank == 1

    frame_rows = {
        dict(zip(result.frame.columns, row, strict=True))["reporter"]: dict(
            zip(result.frame.columns, row, strict=True)
        )
        for row in result.frame.rows
    }
    assert frame_rows["dcid:AAA"]["value"] == 1.0  # the frame: the rank position
    assert frame_rows["dcid:AAA"]["unit"] is None
    assert frame_rows["dcid:BBB"]["value"] == 2.0
    assert frame_rows["dcid:BBB"]["unit"] is None


def test_per_period_ties_are_independent_per_period() -> None:
    """Two periods with different tie structures: 2020 ties AAA/BBB at the
    top; 2021 has no tie at all. Neither period's own tie contaminates the
    other's rank numbers."""
    frame = _frame(
        [
            _row(reporter="dcid:AAA", year="2020", value=100.0),
            _row(reporter="dcid:BBB", year="2020", value=100.0),
            _row(reporter="dcid:CCC", year="2020", value=50.0),
            _row(reporter="dcid:AAA", year="2021", value=10.0),
            _row(reporter="dcid:BBB", year="2021", value=20.0),
            _row(reporter="dcid:CCC", year="2021", value=30.0),
        ]
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
        per_period=True,
    )
    ranks_2020 = {
        dict(er.entity_key)["reporter"]: er.rank
        for er in result.entity_ranks
        if er.time_period == "2020"
    }
    ranks_2021 = {
        dict(er.entity_key)["reporter"]: er.rank
        for er in result.entity_ranks
        if er.time_period == "2021"
    }
    assert ranks_2020 == {"dcid:AAA": 1, "dcid:BBB": 1, "dcid:CCC": 3}
    assert ranks_2021 == {"dcid:AAA": 3, "dcid:BBB": 2, "dcid:CCC": 1}


def test_per_period_entity_absent_in_one_period_is_ranked_in_the_others_only() -> None:
    """USA has rows in both periods; GBR only in 2021 -- GBR is ranked
    alongside USA in 2021 and simply has no row at all in 2020, not a
    refusal and not a zero."""
    frame = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=10.0),
            _row(reporter="dcid:USA", year="2021", value=20.0),
            _row(reporter="dcid:GBR", year="2021", value=30.0),
        ]
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
        per_period=True,
    )
    reporters_2020 = {
        dict(er.entity_key)["reporter"] for er in result.entity_ranks if er.time_period == "2020"
    }
    reporters_2021 = {
        dict(er.entity_key)["reporter"] for er in result.entity_ranks if er.time_period == "2021"
    }
    assert reporters_2020 == {"dcid:USA"}
    assert reporters_2021 == {"dcid:USA", "dcid:GBR"}


def test_per_period_no_aggregation_and_two_rows_in_one_cell_raises_ambiguous() -> None:
    """`aggregation=None` and more than one row for the identical (entity,
    period) cell -- the identical `RankAmbiguousEntityError` the collapsing
    path raises, now checked per cell."""
    frame = _frame(
        [
            _row(reporter="dcid:USA", recipient="dcid:KEN", year="2020", value=10.0),
            _row(reporter="dcid:USA", recipient="dcid:TZA", year="2020", value=20.0),
        ],
        role_keys=["reporter", "recipient"],
    )
    with pytest.raises(RankAmbiguousEntityError):
        run_rank(
            frame,
            rank_by=("reporter",),
            aggregation=None,
            window_start=None,
            window_end=None,
            ascending=False,
            per_period=True,
        )


def test_per_period_aggregation_reduces_the_cell_and_periods_stay_apart() -> None:
    """A frame with a second role (creditor x debtor, ranked by creditor)
    carries more than one row per (entity, period) cell -- `aggregation`
    reduces each cell, and the periods themselves stay separate rankings."""
    frame = _frame(
        [
            _row(reporter="dcid:USA", recipient="dcid:KEN", year="2020", value=10.0),
            _row(reporter="dcid:USA", recipient="dcid:TZA", year="2020", value=20.0),
            _row(reporter="dcid:USA", recipient="dcid:KEN", year="2021", value=5.0),
            _row(reporter="dcid:USA", recipient="dcid:TZA", year="2021", value=5.0),
            _row(reporter="dcid:GBR", recipient="dcid:KEN", year="2020", value=100.0),
            _row(reporter="dcid:GBR", recipient="dcid:KEN", year="2021", value=1.0),
        ],
        role_keys=["reporter", "recipient"],
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation="sum",
        window_start=None,
        window_end=None,
        ascending=False,
        per_period=True,
    )
    by_period = {
        (dict(er.entity_key)["reporter"], er.time_period): er.value for er in result.entity_ranks
    }
    assert by_period[("dcid:USA", "2020")] == 30.0
    assert by_period[("dcid:USA", "2021")] == 10.0
    assert by_period[("dcid:GBR", "2020")] == 100.0
    assert by_period[("dcid:GBR", "2021")] == 1.0


@pytest.mark.parametrize(
    ("years", "labels"),
    [
        (("2020", "2021"), {"2020", "2021"}),
        (("2018/2022",), {"2018/2022"}),
    ],
)
def test_per_period_labels_each_row_with_the_period_it_was_ranked_in(
    years: tuple[str, ...], labels: set[str]
) -> None:
    """Every row's own `time_period` is the one period its rows came from;
    over a span frame that is the span itself."""
    frame = _frame(
        [
            _row(reporter=reporter, year=year, value=value)
            for year in years
            for reporter, value in (("dcid:USA", 10.0), ("dcid:GBR", 20.0))
        ]
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
        per_period=True,
    )
    time_periods = result.frame.columns.index("time_period")
    assert {row[time_periods] for row in result.frame.rows} == labels
    assert len(result.frame.rows) == 2 * len(years)


def test_per_period_overall_coverage_fraction_still_measures_the_entities_by_periods_grid() -> None:
    """The coverage grid is unaffected by `per_period` -- built from the
    same pre-exclusion entities x periods count either mode uses."""
    frame = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=10.0, unit="dcid:USDollar"),
            _row(reporter="dcid:USA", year="2021", value=20.0, unit="dcid:USDollar"),
            _row(reporter="dcid:GBR", year="2020", value=5.0, unit="dcid:USDollar"),
        ]
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
        per_period=True,
    )
    assert result.expected_cell_count == 4
    assert result.present_cell_count == 3
    assert result.overall_coverage_fraction == pytest.approx(0.75)


# Limit ---------------------------------------------------------------------------


def _six_reporter_frame() -> Frame:
    """Six reporters in one period; CCC and DDD share third place."""
    return _frame(
        [
            _row(reporter=reporter, year="2020", value=value, unit="dcid:USDollar")
            for reporter, value in (
                ("dcid:AAA", 60.0),
                ("dcid:BBB", 50.0),
                ("dcid:CCC", 40.0),
                ("dcid:DDD", 40.0),
                ("dcid:EEE", 20.0),
                ("dcid:FFF", 10.0),
            )
        ]
    )


def _frame_reporters(result: RankResult) -> list[str]:
    column = result.frame.columns.index("reporter")
    return sorted(row[column] for row in result.frame.rows)


def _rank_six(*, limit: int | None = None, ascending: bool = False) -> RankResult:
    return run_rank(
        _six_reporter_frame(),
        rank_by=("reporter",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=ascending,
        limit=limit,
    )


def test_limit_keeps_every_entity_ranked_within_it() -> None:
    result = _rank_six(limit=2)

    assert _frame_reporters(result) == ["dcid:AAA", "dcid:BBB"]
    assert [dict(er.entity_key)["reporter"] for er in result.kept_ranks] == [
        "dcid:AAA",
        "dcid:BBB",
    ]
    assert result.limit == 2


def test_limit_keeps_every_entity_sharing_the_boundary_value() -> None:
    """CCC and DDD share rank 3, so a limit of 3 keeps four entities: the
    tie-break decides their listing order, never their membership."""
    result = _rank_six(limit=3)

    assert _frame_reporters(result) == ["dcid:AAA", "dcid:BBB", "dcid:CCC", "dcid:DDD"]


def test_limit_inside_a_shared_rank_keeps_only_the_better_ranks() -> None:
    """No entity holds rank 4 when two share rank 3, so a limit of 4 keeps
    the same four entities as a limit of 3."""
    assert _frame_reporters(_rank_six(limit=4)) == _frame_reporters(_rank_six(limit=3))


def test_ascending_with_a_limit_keeps_the_lowest() -> None:
    result = _rank_six(limit=2, ascending=True)

    assert _frame_reporters(result) == ["dcid:EEE", "dcid:FFF"]


def test_limit_larger_than_the_population_keeps_everyone_and_records_nothing() -> None:
    limited = _rank_six(limit=50)
    unlimited = _rank_six()

    assert limited.frame == unlimited.frame
    assert not any(record.reason == "beyond_limit" for record in limited.exclusions.records)


def test_no_limit_records_no_beyond_limit_exclusion() -> None:
    result = _rank_six()

    assert result.limit is None
    assert result.kept_ranks == result.entity_ranks
    assert not any(record.reason == "beyond_limit" for record in result.exclusions.records)


def test_limit_leaves_coverage_and_the_full_ranking_unchanged() -> None:
    frame = _frame(
        [
            _row(reporter="dcid:AAA", year="2020", value=30.0),
            _row(reporter="dcid:AAA", year="2021", value=30.0),
            _row(reporter="dcid:BBB", year="2020", value=20.0),
            _row(reporter="dcid:CCC", year="2021", value=10.0),
        ]
    )

    def ranked(limit: int | None) -> RankResult:
        return run_rank(
            frame,
            rank_by=("reporter",),
            aggregation="sum",
            window_start=None,
            window_end=None,
            ascending=False,
            limit=limit,
        )

    limited, unlimited = ranked(1), ranked(None)

    assert limited.expected_cell_count == unlimited.expected_cell_count == 6
    assert limited.present_cell_count == unlimited.present_cell_count == 4
    assert limited.overall_coverage_fraction == unlimited.overall_coverage_fraction
    assert limited.entity_ranks == unlimited.entity_ranks
    assert limited.warnings == unlimited.warnings


def test_limit_records_the_dropped_entities_input_rows_as_beyond_limit() -> None:
    frame = _frame(
        [
            _row(reporter="dcid:AAA", year="2020", value=30.0),
            _row(reporter="dcid:AAA", year="2021", value=30.0),
            _row(reporter="dcid:BBB", year="2020", value=20.0),
            _row(reporter="dcid:BBB", year="2021", value=5.0),
            _row(reporter="dcid:CCC", year="2021", value=10.0),
        ]
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation="sum",
        window_start=None,
        window_end=None,
        ascending=False,
        limit=1,
    )

    (record,) = [record for record in result.exclusions.records if record.reason == "beyond_limit"]
    assert record.row_count == 3
    assert record.entities == {"reporter": ("dcid:BBB", "dcid:CCC")}
    assert len(result.entity_ranks) == 3


def test_per_period_limit_cuts_each_period_on_its_own() -> None:
    """AAA leads 2020 and trails 2021: a limit of 1 keeps AAA in 2020 only
    and CCC in 2021 only, at their measured values and in their unit. The
    exclusion counts every other cell's row and names only BBB, the one
    entity the frame holds in no period."""
    frame = _frame(
        [
            _row(reporter="dcid:AAA", year="2020", value=100.0, unit="dcid:USDollar"),
            _row(reporter="dcid:BBB", year="2020", value=50.0, unit="dcid:USDollar"),
            _row(reporter="dcid:CCC", year="2020", value=10.0, unit="dcid:USDollar"),
            _row(reporter="dcid:AAA", year="2021", value=10.0, unit="dcid:USDollar"),
            _row(reporter="dcid:BBB", year="2021", value=20.0, unit="dcid:USDollar"),
            _row(reporter="dcid:CCC", year="2021", value=30.0, unit="dcid:USDollar"),
        ]
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
        per_period=True,
        limit=1,
    )

    columns = result.frame.columns
    kept_cells = sorted(
        (
            row[columns.index("reporter")],
            row[columns.index("time_period")],
            row[columns.index("value")],
        )
        for row in result.frame.rows
    )
    assert kept_cells == [("dcid:AAA", "2020", 100.0), ("dcid:CCC", "2021", 30.0)]
    assert {row[columns.index("unit")] for row in result.frame.rows} == {"dcid:USDollar"}
    assert result.ordinal is False
    (record,) = [record for record in result.exclusions.records if record.reason == "beyond_limit"]
    assert record.row_count == 4
    assert record.entities == {"reporter": ("dcid:BBB",)}
    assert len(result.entity_ranks) == 6


def test_beyond_limit_names_only_the_rank_by_columns() -> None:
    """The output frame declares `reporter` alone, so the record names the
    reporters cut and leaves the sectors their rows were summed over out."""
    frame = _frame(
        [
            _row(reporter="dcid:AAA", value=30.0, sector="dcid:Health"),
            _row(reporter="dcid:AAA", value=30.0, sector="dcid:Water"),
            _row(reporter="dcid:BBB", value=20.0, sector="dcid:Health"),
            _row(reporter="dcid:CCC", value=10.0, sector="dcid:Water"),
        ],
        role_keys=["reporter", "sector"],
    )
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation="sum",
        window_start=None,
        window_end=None,
        ascending=False,
        limit=1,
    )

    (record,) = [record for record in result.exclusions.records if record.reason == "beyond_limit"]
    assert record.row_count == 2
    assert record.entities == {"reporter": ("dcid:BBB", "dcid:CCC")}


def test_beyond_limit_names_the_twenty_smallest_entities_cut() -> None:
    frame = _frame([_row(reporter=f"dcid:R{i:02d}", value=float(i)) for i in range(25)])
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
        limit=1,
    )

    (record,) = [record for record in result.exclusions.records if record.reason == "beyond_limit"]
    assert record.row_count == 24
    assert record.entities == {"reporter": tuple(f"dcid:R{i:02d}" for i in range(20))}
    assert record.entities_truncated is True


def test_beyond_limit_accounting_copies_no_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """A top-1 rank over 20,000 entities counts the 19,999 it cuts with no
    Arrow allocation and a Python heap peak far below one machine word per
    row, so no copy or index list of the cut rows is ever built."""
    entities = 20_000
    frame = build_frame_from_arrow(
        pa.table(
            {
                "stat_var": pa.repeat(pa.scalar("dcid:X"), entities),
                "reporter": pa.array([f"dcid:E{i:05d}" for i in range(entities)]),
                "time_period": pa.repeat(pa.scalar("2020"), entities),
                "value": pa.array([float(i) for i in range(entities)]),
            }
        ),
        role_keys=["reporter"],
    )
    account = rank_module._beyond_limit_exclusion
    proxy = pa.proxy_memory_pool(pa.default_memory_pool())
    peaks: dict[str, int] = {}

    def measured(*args: object, **kwargs: object) -> Exclusions:
        default = pa.default_memory_pool()
        pa.set_memory_pool(proxy)
        tracemalloc.start()
        try:
            return account(*args, **kwargs)
        finally:
            peaks["python"] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            pa.set_memory_pool(default)

    monkeypatch.setattr(rank_module, "_beyond_limit_exclusion", measured)
    result = run_rank(
        frame,
        rank_by=("reporter",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
        limit=1,
    )

    (record,) = [record for record in result.exclusions.records if record.reason == "beyond_limit"]
    assert record.row_count == entities - 1
    assert proxy.max_memory() == 0
    assert peaks["python"] < entities
