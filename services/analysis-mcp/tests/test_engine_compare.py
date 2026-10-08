"""`run_compare`: period intersection, the delta formula, determinism (order
perturbation), and every refusal condition.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from analysis_mcp.engine.common import FacetSelection, UnitHeterogeneousError
from analysis_mcp.engine.compare import (
    CompareAmbiguousSliceError,
    CompareFacetMismatchError,
    CompareUnitUnboundError,
    ComparisonPeriodsDisjointError,
    run_compare,
)
from analysis_mcp.fixtures import (
    facet_disjoint_by_reporter_frame,
    join_compatible_frame_pair,
    multi_unit_frame,
    plain_single_unit_frame,
    unit_disjoint_by_reporter_frame,
    unit_partially_declared_frame,
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


def test_golden_worked_compare_matches_the_committed_fixture() -> None:
    golden = json.loads((_FIXTURES / "compare_golden.json").read_text())
    oda, gni = join_compatible_frame_pair()

    result = run_compare(
        oda, gni, left_entity=golden["left_entity"], right_entity=golden["right_entity"]
    )

    assert list(result.left_periods) == golden["expected_left_periods"]
    assert list(result.right_periods) == golden["expected_right_periods"]
    assert list(result.intersected_periods) == golden["expected_intersected_periods"]
    assert result.left_coverage_fraction == golden["expected_left_coverage_fraction"]
    assert result.right_coverage_fraction == golden["expected_right_coverage_fraction"]

    actual = [
        {
            "time_period": pc.time_period,
            "left_value": pc.left_value,
            "right_value": pc.right_value,
            "delta": pc.delta,
        }
        for pc in result.period_comparisons
    ]
    assert actual == golden["expected_period_comparisons"]


# Declared policy: intersect periods, refuse only on empty intersection ---------


def test_policy_periods_are_intersected_not_unioned() -> None:
    """The declared policy: compare over the intersection of each side's
    periods, not the union -- pinned so a switch to union (or to
    left-only/right-only) fails this test."""
    left = _frame([_row(year="2020"), _row(year="2021"), _row(year="2022")])
    right = _frame([_row(year="2021"), _row(year="2022"), _row(year="2023")])

    result = run_compare(left, right, left_entity={}, right_entity={})

    assert result.intersected_periods == ("2021", "2022")


def test_policy_disjoint_periods_refuse() -> None:
    left = _frame([_row(year="2020")])
    right = _frame([_row(year="2023")])

    with pytest.raises(ComparisonPeriodsDisjointError) as exc_info:
        run_compare(left, right, left_entity={}, right_entity={})

    assert exc_info.value.left_periods == ("2020",)
    assert exc_info.value.right_periods == ("2023",)


def test_delta_is_right_minus_left() -> None:
    left = _frame([_row(year="2020", value=10.0)])
    right = _frame([_row(year="2020", value=25.0)])
    result = run_compare(left, right, left_entity={}, right_entity={})
    assert result.period_comparisons[0].delta == pytest.approx(15.0)


def test_delta_normalises_negative_zero() -> None:
    """`right_value - left_value` genuinely lands on IEEE-754 negative zero
    when `right_value=-0.0` and `left_value=0.0` (`-0.0 - 0.0 == -0.0`) --
    `run_compare`'s own `delta += 0.0` normalises that at the engine
    boundary, not left to the digest's own render-time `_format_number`, so
    `delta` is what the output frame itself stores, not just what the
    digest renders."""
    left = _frame([_row(year="2020", value=0.0)])
    right = _frame([_row(year="2020", value=-0.0)])
    result = run_compare(left, right, left_entity={}, right_entity={})
    assert repr(result.period_comparisons[0].delta) == "0.0"


# Determinism -------------------------------------------------------------------


def test_entity_binding_key_order_perturbation_produces_an_identical_result() -> None:
    """Varying `left_entity`/`right_entity` key insertion order does not
    change the result. `Frame.build` absorbs row-mapping key order, but an
    `entity` binding is passed straight to `narrow_and_resolve`, which walks
    `entity.items()` directly -- so two dicts with the same pairs in
    different order reach the engine as different objects.

    Two role columns make key order variation meaningful. The expected
    `delta` is pinned by hand (`60.0 - 100.0`) to verify the formula
    itself, not just compared against `original`, which could hide a
    uniformly wrong formula applied to both sides."""
    frame = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=100.0, recipient="dcid:KEN"),
            _row(reporter="dcid:GBR", year="2020", value=60.0, recipient="dcid:KEN"),
        ],
        role_keys=["reporter", "recipient"],
    )
    original = run_compare(
        frame,
        frame,
        left_entity={"reporter": "dcid:USA", "recipient": "dcid:KEN"},
        right_entity={"reporter": "dcid:GBR", "recipient": "dcid:KEN"},
    )
    perturbed = run_compare(
        frame,
        frame,
        left_entity={"recipient": "dcid:KEN", "reporter": "dcid:USA"},
        right_entity={"recipient": "dcid:KEN", "reporter": "dcid:GBR"},
    )

    assert perturbed.period_comparisons == original.period_comparisons
    assert original.period_comparisons[0].delta == pytest.approx(-40.0)


# Two slices of one frame --------------------------------------------------------


def test_two_slices_of_one_frame_compares_two_entities() -> None:
    frame = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=100.0),
            _row(reporter="dcid:GBR", year="2020", value=60.0),
        ]
    )
    result = run_compare(
        frame, frame, left_entity={"reporter": "dcid:USA"}, right_entity={"reporter": "dcid:GBR"}
    )
    assert result.period_comparisons[0].left_value == 100.0
    assert result.period_comparisons[0].right_value == 60.0


# Refusals ------------------------------------------------------------------------


def test_ambiguous_entity_binding_refuses() -> None:
    frame = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=1.0),
            _row(reporter="dcid:GBR", year="2020", value=2.0),
        ]
    )
    with pytest.raises(CompareAmbiguousSliceError) as exc_info:
        run_compare(frame, frame, left_entity={}, right_entity={})
    assert exc_info.value.side == "left"


def test_heterogeneous_unit_frame_refuses() -> None:
    frame = multi_unit_frame()
    with pytest.raises(UnitHeterogeneousError):
        run_compare(frame, frame, left_entity={"reporter": "dcid:USA"}, right_entity={})


def test_a_reporter_narrowed_out_of_a_heterogeneous_frame_succeeds() -> None:
    """The declared narrow-before-resolve policy: a frame heterogeneous at
    large (USA in USD, GBR in Percent) must not refuse `UnitHeterogeneousError`
    when both sides are narrowed to the one reporter (USA) that is, on its
    own, unit-homogeneous -- compare's own primary documented use case."""
    frame = unit_disjoint_by_reporter_frame()
    result = run_compare(
        frame, frame, left_entity={"reporter": "dcid:USA"}, right_entity={"reporter": "dcid:USA"}
    )
    assert result.intersected_periods == ("2020", "2021")
    assert result.left_resolved_unit == "dcid:USDollar"
    assert result.right_resolved_unit == "dcid:USDollar"


# Cross-side facet and unit-unbound checks -----------------------------------


def _sourced_frame(reporter: str, provenance: str, stat_vars: Sequence[str]) -> Frame:
    """One reporter in USD under `provenance`, one row per StatVar, the
    StatVars on consecutive years from 2020."""
    return _frame(
        [
            _row(
                reporter=reporter,
                year=str(2020 + index),
                stat_var=stat_var,
                unit="dcid:USDollar",
                provenance=provenance,
            )
            for index, stat_var in enumerate(stat_vars)
        ]
    )


@pytest.mark.parametrize(
    ("frames", "stat_vars"),
    [
        pytest.param(
            lambda: (facet_disjoint_by_reporter_frame(),) * 2,
            ("dcid:OfficialDevelopmentAssistanceNet_Disbursement",),
            id="one-frame",
        ),
        pytest.param(
            lambda: (
                _sourced_frame("dcid:USA", "dcid:P1", ["dcid:X", "dcid:X"]),
                _sourced_frame("dcid:GBR", "dcid:P2", ["dcid:X", "dcid:X"]),
            ),
            ("dcid:X",),
            id="two-frames-one-stat-var",
        ),
        pytest.param(
            lambda: (
                _sourced_frame("dcid:USA", "dcid:P1", ["dcid:A", "dcid:B"]),
                _sourced_frame("dcid:GBR", "dcid:P2", ["dcid:B", "dcid:C"]),
            ),
            ("dcid:B",),
            id="two-frames-overlapping-stat-vars",
        ),
    ],
)
def test_cross_side_facet_mismatch_refuses_even_though_each_side_resolves_cleanly(
    frames: Callable[[], tuple[Frame, Frame]], stat_vars: tuple[str, ...]
) -> None:
    """Each side (USA in `provenance=dcid:P1`, GBR in `dcid:P2`) resolves
    cleanly on its own -- nothing about resolving either side alone ever
    sees the other's own facet. Where the sides share a StatVar the pairing
    itself must still refuse, naming the StatVars both hold."""
    left, right = frames()
    with pytest.raises(CompareFacetMismatchError) as exc_info:
        run_compare(
            left,
            right,
            left_entity={"reporter": "dcid:USA"},
            right_entity={"reporter": "dcid:GBR"},
        )
    assert exc_info.value.axis == "provenance"
    assert exc_info.value.left_ref == "dcid:P1"
    assert exc_info.value.right_ref == "dcid:P2"
    assert exc_info.value.stat_vars == stat_vars


def test_cross_side_facet_difference_between_disjoint_stat_vars_warns_and_compares() -> None:
    """Two slices sharing no StatVar each keep their own provenance; the
    comparison runs, the result leaves provenance unbound, and the warning
    names both selections."""
    left = _sourced_frame("dcid:USA", "dcid:P1", ["dcid:A", "dcid:B"])
    right = _sourced_frame("dcid:GBR", "dcid:P2", ["dcid:C", "dcid:D"])

    result = run_compare(left, right, left_entity={}, right_entity={})

    assert result.intersected_periods == ("2020", "2021")
    assert result.frame.table.column("provenance").to_pylist() == [None, None]
    assert result.left_facet_selections == (FacetSelection(axis="provenance", ref="dcid:P1"),)
    assert result.right_facet_selections == (FacetSelection(axis="provenance", ref="dcid:P2"),)
    assert len(result.warnings) == 1
    assert "'dcid:P1'" in result.warnings[0]
    assert "'dcid:P2'" in result.warnings[0]


def test_facet_left_wholly_unbound_on_one_side_succeeds_rather_than_refusing() -> None:
    """The softer facet policy (`engine.common`'s own module docstring): an
    axis one side never declared is not a disagreement between two declared
    values, so it resolves via auto-selection on the side that has it
    rather than refusing the pairing -- but it must still warn, naming the
    axis, which side left it unbound, and whether the result carries a
    value for it, regardless of which side is passed as `left`.
    """
    frame = _frame([_row(reporter="dcid:USA", provenance="dcid:P1"), _row(reporter="dcid:GBR")])

    # USA (declares P1) as left: the result carries P1 -- compare's output
    # always carries the left side's own facet selections.
    usa_left = run_compare(
        frame, frame, left_entity={"reporter": "dcid:USA"}, right_entity={"reporter": "dcid:GBR"}
    )
    assert usa_left.left_facet_selections == (FacetSelection(axis="provenance", ref="dcid:P1"),)
    assert usa_left.right_facet_selections == ()
    assert len(usa_left.warnings) == 1
    assert "provenance" in usa_left.warnings[0]
    assert "right" in usa_left.warnings[0]
    assert "dcid:P1" in usa_left.warnings[0]

    # GBR (declares nothing) as left: the result carries *no* value for
    # this axis at all, even though the right side (USA) does declare one
    # -- an order-dependent output label, surfaced by a warning instead of
    # passing silently.
    gbr_left = run_compare(
        frame, frame, left_entity={"reporter": "dcid:GBR"}, right_entity={"reporter": "dcid:USA"}
    )
    assert gbr_left.left_facet_selections == ()
    assert gbr_left.right_facet_selections == (FacetSelection(axis="provenance", ref="dcid:P1"),)
    assert len(gbr_left.warnings) == 1
    assert "provenance" in gbr_left.warnings[0]
    assert "left" in gbr_left.warnings[0]
    assert "no value" in gbr_left.warnings[0]


def test_entity_matching_no_rows_with_a_real_pinned_unit_refuses_periods_disjoint_not_unit_not_present() -> (
    None
):
    """An `entity` binding matching zero rows must not be checked against
    `pinned_unit` at all -- `plain_single_unit_frame` genuinely declares
    `dcid:USDollar`, just not for a reporter that doesn't exist in it. The
    real problem is the empty right side, which `ComparisonPeriodsDisjointError`
    names; `UnitNotPresentError` would misreport it as declaring nothing."""
    frame = plain_single_unit_frame()
    with pytest.raises(ComparisonPeriodsDisjointError) as exc_info:
        run_compare(
            frame,
            frame,
            left_entity={"reporter": "dcid:USA"},
            right_entity={"reporter": "dcid:NOPE"},
            right_pinned_unit="dcid:USDollar",
        )
    assert exc_info.value.right_periods == ()


def test_unit_unbound_side_refuses_distinctly_from_a_genuine_unit_mismatch() -> None:
    """`unit_partially_declared_frame` declares nothing but dollars --
    comparing the USD-declaring USA against the FRA row that leaves `unit`
    unbound entirely must say that plainly, not report it as two resolved
    units disagreeing (`CompareUnitMismatchError`)."""
    frame = unit_partially_declared_frame()
    with pytest.raises(CompareUnitUnboundError) as exc_info:
        run_compare(
            frame,
            frame,
            left_entity={"reporter": "dcid:USA"},
            right_entity={"reporter": "dcid:FRA"},
        )
    assert exc_info.value.unbound_side == "right"
    assert exc_info.value.bound_side == "left"
    assert exc_info.value.bound_unit == "dcid:USDollar"


def test_unit_unbound_exclusion_is_tagged_by_which_side_dropped_it() -> None:
    """Only the left side's own row leaves `unit` unbound here -- the
    dropped row's own exclusion must say `left`, not `right`, so a reader
    can tell which side of the delta actually lost data."""
    left = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=50.0, unit="dcid:USDollar"),
            _row(reporter="dcid:USA", year="2021", value=25.0),  # unit left unbound
        ]
    )
    right = _frame([_row(reporter="dcid:USA", year="2020", value=200.0, unit="dcid:USDollar")])

    result = run_compare(left, right, left_entity={}, right_entity={})

    unit_unbound = [
        record for record in result.exclusions.records if record.reason == "unit_unbound"
    ]
    assert len(unit_unbound) == 1
    assert unit_unbound[0].side == "left"
    assert unit_unbound[0].row_count == 1


def test_unintersected_periods_are_recorded_as_exclusions_per_side() -> None:
    """Each side's own periods outside the intersection are a real,
    permanent loss -- `left_coverage_fraction`/`right_coverage_fraction`
    state the ratio, but only an exclusion names which periods (and how
    many rows) that ratio cost."""
    left = _frame([_row(year="2021"), _row(year="2022"), _row(year="2023")])
    right = _frame([_row(year="2022"), _row(year="2023"), _row(year="2024")])

    result = run_compare(left, right, left_entity={}, right_entity={})

    period_records = {
        record.side: record
        for record in result.exclusions.records
        if record.reason == "period_unmatched"
    }
    assert set(period_records) == {"left", "right"}
    assert period_records["left"].row_count == 1
    assert period_records["right"].row_count == 1


def test_full_period_overlap_records_no_period_unmatched_exclusion() -> None:
    left = _frame([_row(year="2021"), _row(year="2022")])
    right = _frame([_row(year="2021"), _row(year="2022")])

    result = run_compare(left, right, left_entity={}, right_entity={})

    assert not any(record.reason == "period_unmatched" for record in result.exclusions.records)


def test_entity_narrowed_compare_records_the_dropped_entities_per_side() -> None:
    """`left_entity`/`right_entity` each narrow `frame` down to one
    reporter -- every other reporter's own rows must be attributed to the
    side that dropped them."""
    frame = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=1.0),
            _row(reporter="dcid:GBR", year="2020", value=2.0),
            _row(reporter="dcid:DEU", year="2020", value=3.0),
        ]
    )
    result = run_compare(
        frame, frame, left_entity={"reporter": "dcid:USA"}, right_entity={"reporter": "dcid:GBR"}
    )
    entity_records = {
        record.side: record
        for record in result.exclusions.records
        if record.reason == "entity_not_selected"
    }
    assert entity_records["left"].row_count == 2
    assert entity_records["left"].entities["reporter"] == ("dcid:DEU", "dcid:GBR")
    assert entity_records["right"].row_count == 2
    assert entity_records["right"].entities["reporter"] == ("dcid:DEU", "dcid:USA")


def test_default_entity_binding_over_an_already_empty_frame_with_a_pin_refuses_periods_disjoint_not_unit_not_present() -> (
    None
):
    """`entity={}` (the default -- "frame already names exactly one
    entity") over an already-empty frame must not be checked against
    `pinned_unit` either: `entity`'s own truthiness has nothing to do with
    whether there are rows to check a pin against, and the real problem
    here is that the right side has no rows at all."""
    left = plain_single_unit_frame()
    right = Frame.build(role_keys=["reporter"], rows=[])
    with pytest.raises(ComparisonPeriodsDisjointError) as exc_info:
        run_compare(
            left,
            right,
            left_entity={"reporter": "dcid:USA"},
            right_entity={},
            right_pinned_unit="dcid:USDollar",
        )
    assert exc_info.value.right_periods == ()
