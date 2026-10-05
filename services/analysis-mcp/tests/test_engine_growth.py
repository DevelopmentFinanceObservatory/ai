"""`run_growth`: the gap-year policy, the growth-rate formula, determinism
(order perturbation), and every refusal condition.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from analysis_mcp.engine.common import (
    FacetSelection,
    StatVarUnresolvedError,
    UnitHeterogeneousError,
    UnknownEntityColumnError,
)
from analysis_mcp.engine.growth import (
    GrowthAmbiguousEntityError,
    GrowthGapYearError,
    GrowthNonFiniteResultError,
    run_growth,
)
from analysis_mcp.fixtures import (
    multi_unit_frame,
    time_series_with_gap_year_frame,
    unit_disjoint_facet_leak_frame,
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


def test_golden_worked_growth_matches_the_committed_fixture() -> None:
    golden = json.loads((_FIXTURES / "growth_golden.json").read_text())
    frame = time_series_with_gap_year_frame()

    result = run_growth(
        frame,
        window_start=golden["window_start"],
        window_end=golden["window_end"],
        entity_by=tuple(golden["entity_by"]),
    )

    actual = [
        {
            "entity_key": [list(pair) for pair in eg.entity_key],
            "value_start": eg.value_start,
            "value_end": eg.value_end,
            "growth_rate": eg.growth_rate,
            "realised_coverage_fraction": eg.realised_coverage_fraction,
        }
        for eg in result.entity_growths
    ]
    assert actual == golden["expected_entity_growths"]


# Declared policy: refuse across a gap year ---------------------------------


def test_policy_gap_year_inside_the_window_refuses() -> None:
    """The declared policy: a gap year inside the window is a break,
    refused -- not skipped, not silently computed across. This test
    fails if that policy ever changes to "skip" or "compute across"."""
    frame = time_series_with_gap_year_frame()

    with pytest.raises(GrowthGapYearError) as exc_info:
        run_growth(frame, window_start="2017", window_end="2022", entity_by=("reporter",))

    assert exc_info.value.gap_years == ("2021",)
    assert exc_info.value.longest_gap_free_window == ("2017", "2020")
    assert exc_info.value.entity_key == {"reporter": "dcid:USA"}


def test_policy_growth_rate_is_total_percent_change_not_cagr() -> None:
    """The declared formula: `(end - start) / start`, not a compound annual
    rate -- pinned so a switch to CAGR would fail this test."""
    frame = _frame(
        [
            _row(year="2020", value=100.0),
            _row(year="2021", value=150.0),
            _row(year="2022", value=200.0),
        ]
    )
    result = run_growth(frame, window_start="2020", window_end="2022", entity_by=("reporter",))
    assert result.entity_growths[0].growth_rate == pytest.approx(1.0)  # (200-100)/100


def test_every_output_row_carries_the_window_as_a_span() -> None:
    frame = _frame(
        [
            _row(reporter=reporter, year=str(year), value=float(year))
            for reporter in ("dcid:USA", "dcid:GBR")
            for year in range(2018, 2023)
        ]
    )
    result = run_growth(frame, window_start="2018", window_end="2022", entity_by=("reporter",))
    periods = {
        dict(zip(result.frame.columns, row, strict=True))["time_period"]
        for row in result.frame.rows
    }
    assert periods == {"2018/2022"}
    assert result.frame.num_rows == 2


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
    """`EntityGrowth` equality with `entity_key` order-insensitive --
    `entity_key`'s own tuple order mirrors whatever order `entity_by` was
    called with (`run_growth`'s `entity_key_map = dict(zip(entity_by, key,
    ...))`), so two calls differing only in `entity_by`'s column order
    produce genuinely different `entity_key` tuples describing the
    identical entity."""
    return (
        frozenset(entity.entity_key),
        entity.value_start,
        entity.value_end,
        entity.growth_rate,
        entity.realised_coverage_fraction,
    )


def test_row_key_order_and_entity_by_order_perturbation_produce_an_equivalent_result() -> None:
    """The two perturbations that actually reach this engine (row order
    itself is absorbed by `Frame.build`'s own sort, so shuffling it proves
    nothing): each row mapping's own key insertion order, and `entity_by`'s
    own column order. `entity_by` needs two role columns -- facet-identity
    columns are disallowed as entity columns (`engine.common._DISALLOWED_
    ENTITY_COLUMNS`) -- so a fresh two-role, two-entity frame is built here
    instead of reusing `time_series_with_gap_year_frame`'s single reporter.
    `run_growth` never canonicalises `entity_by` itself (the tool boundary
    does, before this function ever runs), so `entity_key`'s own tuple
    order does move with the perturbation; `_entity_key` is what makes the
    comparison honest about that instead of hiding it."""
    frame = _frame(
        [
            _row(reporter="dcid:USA", recipient="dcid:KEN", year=year, value=value)
            for year, value in (("2020", 10.0), ("2021", 20.0), ("2022", 30.0))
        ]
        + [
            _row(reporter="dcid:GBR", recipient="dcid:TZA", year=year, value=value)
            for year, value in (("2020", 5.0), ("2021", 8.0), ("2022", 9.0))
        ],
        role_keys=["reporter", "recipient"],
    )
    original = run_growth(
        frame, window_start="2020", window_end="2022", entity_by=("reporter", "recipient")
    )

    reordered_rows = [_row_mapping_with_reversed_key_order(frame, row) for row in frame.rows]
    reordered_frame = Frame.build(role_keys=["reporter", "recipient"], rows=reordered_rows)
    perturbed = run_growth(
        reordered_frame,
        window_start="2020",
        window_end="2022",
        entity_by=("recipient", "reporter"),
    )

    assert perturbed.frame == original.frame
    assert {_entity_key(e) for e in perturbed.entity_growths} == {
        _entity_key(e) for e in original.entity_growths
    }


# Multi-entity ------------------------------------------------------------------


def test_multiple_entities_each_get_their_own_growth_rate() -> None:
    frame = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=100.0),
            _row(reporter="dcid:USA", year="2021", value=200.0),
            _row(reporter="dcid:GBR", year="2020", value=50.0),
            _row(reporter="dcid:GBR", year="2021", value=25.0),
        ]
    )
    result = run_growth(frame, window_start="2020", window_end="2021", entity_by=("reporter",))
    rates = {dict(eg.entity_key)["reporter"]: eg.growth_rate for eg in result.entity_growths}
    assert rates["dcid:USA"] == pytest.approx(1.0)
    assert rates["dcid:GBR"] == pytest.approx(-0.5)


def test_empty_entity_by_treats_the_whole_frame_as_one_entity() -> None:
    frame = time_series_with_gap_year_frame()
    result = run_growth(frame, window_start="2017", window_end="2020", entity_by=())
    assert len(result.entity_growths) == 1
    assert result.entity_growths[0].entity_key == ()


def test_rows_outside_the_window_are_reported_as_an_outside_window_exclusion() -> None:
    """`time_series_with_gap_year_frame`'s own seven years (2017-2020,
    2022-2024, 2021 missing): a window of 2022-2024 is gap-free on its own,
    so the call succeeds, and the four years outside it (2017-2020) are
    dropped before the gap-year check ever runs -- reported here, not just
    silently narrowed away."""
    frame = time_series_with_gap_year_frame()
    result = run_growth(frame, window_start="2022", window_end="2024", entity_by=("reporter",))
    assert len(result.entity_growths) == 1
    assert len(result.exclusions.records) == 1
    record = result.exclusions.records[0]
    assert record.reason == "outside_window"
    assert record.row_count == 4
    assert record.entities == {"reporter": ("dcid:USA",)}


# Refusals ------------------------------------------------------------------------


def test_unknown_entity_by_column_refuses() -> None:
    frame = time_series_with_gap_year_frame()
    with pytest.raises(UnknownEntityColumnError) as exc_info:
        run_growth(frame, window_start="2017", window_end="2020", entity_by=("donor",))
    assert exc_info.value.columns == ("donor",)


def test_ambiguous_entity_refuses_when_entity_by_does_not_disambiguate() -> None:
    """`plain_single_unit_frame`-shaped data (5 reporters) with an empty
    `entity_by` collides more than one row onto the same period."""
    frame = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=1.0),
            _row(reporter="dcid:GBR", year="2020", value=2.0),
        ]
    )
    with pytest.raises(GrowthAmbiguousEntityError):
        run_growth(frame, window_start="2020", window_end="2020", entity_by=())


def test_zero_base_refuses_non_finite() -> None:
    frame = _frame([_row(year="2020", value=0.0), _row(year="2021", value=100.0)])
    with pytest.raises(GrowthNonFiniteResultError) as exc_info:
        run_growth(frame, window_start="2020", window_end="2021", entity_by=("reporter",))
    assert exc_info.value.reason == "zero or sign-crossing base"


def test_sign_crossing_base_refuses_non_finite() -> None:
    frame = _frame([_row(year="2020", value=-100.0), _row(year="2021", value=100.0)])
    with pytest.raises(GrowthNonFiniteResultError) as exc_info:
        run_growth(frame, window_start="2020", window_end="2021", entity_by=("reporter",))
    assert exc_info.value.reason == "zero or sign-crossing base"


def test_no_rows_at_all_in_window_refuses_gap_year() -> None:
    frame = time_series_with_gap_year_frame()
    with pytest.raises(GrowthGapYearError) as exc_info:
        run_growth(frame, window_start="2099", window_end="2100", entity_by=("reporter",))
    assert exc_info.value.gap_years == ("2099", "2100")


def test_multiple_stat_vars_refuse_stat_var_unresolved() -> None:
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:A", "reporter": "dcid:USA", "time_period": "2020", "value": 1.0},
            {"stat_var": "dcid:B", "reporter": "dcid:USA", "time_period": "2021", "value": 2.0},
        ],
    )
    with pytest.raises(StatVarUnresolvedError):
        run_growth(frame, window_start="2020", window_end="2021", entity_by=("reporter",))


def test_heterogeneous_unit_frame_refuses() -> None:
    frame = multi_unit_frame()
    with pytest.raises(UnitHeterogeneousError):
        run_growth(frame, window_start="2022", window_end="2023", entity_by=("reporter",))


# Absolute change ---------------------------------------------------------------


def test_an_absolute_change_over_a_share_reads_in_points() -> None:
    """The e2e recipe's own worked example: a dimensionless series (as a
    keyed `share_of` result would produce, `unit=None` on every row) moving
    0.10 -> 0.15 reads `absolute` as `0.05` (five percentage points, with
    `unit=None` since there was no unit to keep) and `relative` as `0.5`
    (fifty percent) -- the two readings a user could mean by "how did the
    share change" and the reason this slice exists."""
    frame = Frame.build(
        role_keys=[],
        rows=[
            {"stat_var": "dcid:X", "time_period": "2020", "value": 0.10},
            {"stat_var": "dcid:X", "time_period": "2021", "value": 0.15},
        ],
    )
    absolute = run_growth(
        frame, window_start="2020", window_end="2021", entity_by=(), change="absolute"
    )
    relative = run_growth(frame, window_start="2020", window_end="2021", entity_by=())
    assert absolute.entity_growths[0].growth_rate == pytest.approx(0.05)
    assert relative.entity_growths[0].growth_rate == pytest.approx(0.5)
    unit_idx = absolute.frame.columns.index("unit")
    assert all(row[unit_idx] is None for row in absolute.frame.rows)


def test_ten_to_fifteen_reads_plus_five_absolute_and_plus_half_relative() -> None:
    frame = _frame([_row(year="2020", value=10.0), _row(year="2021", value=15.0)])
    relative = run_growth(frame, window_start="2020", window_end="2021", entity_by=("reporter",))
    absolute = run_growth(
        frame, window_start="2020", window_end="2021", entity_by=("reporter",), change="absolute"
    )
    assert relative.entity_growths[0].growth_rate == pytest.approx(0.5)
    assert absolute.entity_growths[0].growth_rate == pytest.approx(5.0)


def test_zero_base_refuses_relative_and_succeeds_absolute() -> None:
    frame = _frame([_row(year="2020", value=0.0), _row(year="2021", value=100.0)])
    with pytest.raises(GrowthNonFiniteResultError):
        run_growth(frame, window_start="2020", window_end="2021", entity_by=("reporter",))
    absolute = run_growth(
        frame, window_start="2020", window_end="2021", entity_by=("reporter",), change="absolute"
    )
    assert absolute.entity_growths[0].growth_rate == pytest.approx(100.0)


def test_sign_crossing_base_refuses_relative_and_succeeds_absolute() -> None:
    frame = _frame([_row(year="2020", value=-100.0), _row(year="2021", value=100.0)])
    with pytest.raises(GrowthNonFiniteResultError):
        run_growth(frame, window_start="2020", window_end="2021", entity_by=("reporter",))
    absolute = run_growth(
        frame, window_start="2020", window_end="2021", entity_by=("reporter",), change="absolute"
    )
    assert absolute.entity_growths[0].growth_rate == pytest.approx(200.0)


def test_absolute_output_row_keeps_the_input_unit_and_the_relative_one_does_not() -> None:
    rows = [
        _row(year="2020", value=10.0, unit="dcid:USDollar"),
        _row(year="2021", value=15.0, unit="dcid:USDollar"),
    ]
    frame = _frame(rows)
    relative = run_growth(frame, window_start="2020", window_end="2021", entity_by=("reporter",))
    absolute = run_growth(
        frame, window_start="2020", window_end="2021", entity_by=("reporter",), change="absolute"
    )
    unit_idx = relative.frame.columns.index("unit")
    assert all(row[unit_idx] is None for row in relative.frame.rows)
    assert all(row[unit_idx] == "dcid:USDollar" for row in absolute.frame.rows)
    assert relative.resolved_unit == absolute.resolved_unit == "dcid:USDollar"


def test_absolute_change_result_records_its_own_mode() -> None:
    frame = _frame([_row(year="2020", value=10.0), _row(year="2021", value=15.0)])
    absolute = run_growth(
        frame, window_start="2020", window_end="2021", entity_by=("reporter",), change="absolute"
    )
    relative = run_growth(frame, window_start="2020", window_end="2021", entity_by=("reporter",))
    assert absolute.change == "absolute"
    assert relative.change == "relative"


def test_absolute_overflow_still_refuses_non_finite() -> None:
    """The overflow guard applies to both modes -- an absolute difference
    between two extreme-magnitude opposite-signed floats can itself
    overflow to a non-finite value, even though the zero/sign-crossing
    refusal does not apply to absolute change."""
    huge = 1.7e308
    frame = _frame([_row(year="2020", value=-huge), _row(year="2021", value=huge)])
    with pytest.raises(GrowthNonFiniteResultError) as exc_info:
        run_growth(
            frame,
            window_start="2020",
            window_end="2021",
            entity_by=("reporter",),
            change="absolute",
        )
    assert exc_info.value.reason == "overflow"


def test_pinned_unit_narrows_before_resolving_facets() -> None:
    """The declared narrow-before-resolve policy: pinning `pinned_unit` to
    USD already excludes every GBR/Percent row from the computation -- it
    must exclude those rows from facet resolution too, so GBR's own
    `measurementMethod=dcid:M2` never gets stamped onto (or refused
    against) a USD-scoped result whose own rows only ever declared M1.
    `unit_disjoint_facet_leak_frame` is genuinely facet-heterogeneous at
    the whole-frame level (M1 vs M2) -- a resolver that lets GBR's own
    rows leak into the pinned computation's own facet check would refuse
    `FacetHeterogeneousError` here, not merely mislabel the result, which
    a fixture with only one declared facet value could never catch."""
    frame = unit_disjoint_facet_leak_frame()
    result = run_growth(
        frame,
        window_start="2021",
        window_end="2022",
        entity_by=("reporter",),
        pinned_unit="dcid:USDollar",
    )
    assert result.facet_selections == (FacetSelection(axis="measurementMethod", ref="dcid:M1"),)
    assert not any("measurementMethod" in warning for warning in result.warnings)
    measurement_method_idx = result.frame.columns.index("measurementMethod")
    assert all(row[measurement_method_idx] == "dcid:M1" for row in result.frame.rows)
