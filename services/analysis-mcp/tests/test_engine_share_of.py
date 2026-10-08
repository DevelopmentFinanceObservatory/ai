"""`run_share_of`: the grain-match policy (refuse, never report-both-and-
proceed), fsum-for-any-summed-side, determinism (order perturbation), keyed
share_of's per-key/per-axis grouping, and every refusal condition.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import cast

import pytest
from analysis_mcp.engine.aggregate import ComposedUnitSumError
from analysis_mcp.engine.common import FacetSelection, UnitHeterogeneousError
from analysis_mcp.engine.share_of import (
    ShareOfFacetMismatchError,
    ShareOfGrainMismatchError,
    ShareOfKeyUnknownError,
    ShareOfKeyUnmatchedError,
    ShareOfNonFiniteResultError,
    ShareOfResult,
    ShareOfUnitUnboundError,
)
from analysis_mcp.engine.share_of import run_share_of as run_share_of_labelled
from analysis_mcp.fixtures import (
    facet_disjoint_by_reporter_frame,
    join_compatible_frame_pair,
    multi_unit_frame,
    plain_single_unit_frame,
    unit_and_unitless_row_frame,
    unit_partially_declared_frame,
)
from analysis_mcp.op_spec import ShareOfOpSpec
from dfo_frame_grammar import Frame, unit_cell_to_json
from dfo_frame_grammar.grammar import ComposedUnit, StatVarPart

_FIXTURES = Path(__file__).parent / "fixtures"

# The minted ref the tool passes as `stat_var`; every call here labels its
# rows with it.
_SHARE_STAT_VAR = "dcid:ops_" + "c" * 32
run_share_of = partial(run_share_of_labelled, stat_var=_SHARE_STAT_VAR)


def _frame(rows: list[dict[str, object]], *, role_keys: list[str] | None = None) -> Frame:
    return Frame.build(role_keys=role_keys or ["reporter"], rows=rows)


def _row(
    *,
    reporter: str = "dcid:USA",
    year: str = "2020",
    value: float = 1.0,
    stat_var: str = "dcid:X",
    **extra: object,
) -> dict:
    return {
        "stat_var": stat_var,
        "reporter": reporter,
        "time_period": year,
        "value": value,
        **extra,
    }


# Golden file -----------------------------------------------------------------


def test_golden_worked_share_of_matches_the_committed_fixture() -> None:
    golden = json.loads((_FIXTURES / "share_of_golden.json").read_text())
    oda, gni = join_compatible_frame_pair()

    result = run_share_of(
        oda,
        gni,
        numerator_entity=golden["numerator_entity"],
        denominator_entity=golden["denominator_entity"],
    )

    assert list(result.numerator_periods) == golden["expected_numerator_periods"]
    assert list(result.denominator_periods) == golden["expected_denominator_periods"]
    actual = [
        {
            "time_period": ps.time_period,
            "numerator_value": ps.numerator_value,
            "denominator_value": ps.denominator_value,
            "percent": ps.percent,
        }
        for ps in result.period_shares
    ]
    assert actual == golden["expected_period_shares"]


def test_numerator_by_empty_reproduces_the_output_frame_byte_for_byte() -> None:
    """`numerator_by=()` (the default) builds one row per matched period and
    no role columns, each row stating what it holds: the share in per cent,
    the `Percent` unit a published share carries, and the caller's minted
    `stat_var`. The result names the StatVar each side resolved to."""
    golden = json.loads((_FIXTURES / "share_of_golden.json").read_text())
    oda, gni = join_compatible_frame_pair()

    result = run_share_of(
        oda,
        gni,
        numerator_entity=golden["numerator_entity"],
        denominator_entity=golden["denominator_entity"],
    )

    assert result.frame.columns == (
        "stat_var",
        "time_period",
        "value",
        "unit",
        "measurementMethod",
        "provenance",
        "status",
    )
    expected_rows = [
        {
            "stat_var": _SHARE_STAT_VAR,
            "time_period": share["time_period"],
            "value": share["percent"],
            "unit": "Percent",
            "measurementMethod": None,
            "provenance": None,
            "status": None,
        }
        for share in golden["expected_period_shares"]
    ]
    actual_rows = [dict(zip(result.frame.columns, row, strict=True)) for row in result.frame.rows]
    assert actual_rows == expected_rows
    assert result.stat_var == _SHARE_STAT_VAR
    assert (result.numerator_stat_var, result.denominator_stat_var) == (
        "dcid:OfficialDevelopmentAssistanceNet_Disbursement",
        "dcid:GrossNationalIncome",
    )


# Declared policy: divide over the shared periods, record the rest -----------


def _period_unmatched(result: ShareOfResult) -> dict[str, int]:
    """Each side's `period_unmatched` row count."""
    return {
        cast("str", record.side): record.row_count
        for record in result.exclusions.records
        if record.reason == "period_unmatched"
    }


def test_policy_unmatched_periods_are_left_out_and_recorded() -> None:
    """The declared policy, `quotient_of`'s: the share divides over the
    periods both sides report, and a period one side holds alone is left
    out and recorded as `period_unmatched` on that side."""
    numerator = _frame(
        [
            _row(year="2020", value=1.0, stat_var="dcid:X"),
            _row(year="2021", value=2.0, stat_var="dcid:X"),
        ]
    )
    denominator = _frame(
        [
            _row(year="2019", value=8.0, stat_var="dcid:Y"),
            _row(year="2020", value=4.0, stat_var="dcid:Y"),
        ]
    )

    result = run_share_of(numerator, denominator, numerator_entity={}, denominator_entity={})

    assert [(ps.time_period, ps.percent) for ps in result.period_shares] == [("2020", 25.0)]
    assert result.numerator_periods == ("2020", "2021")
    assert result.denominator_periods == ("2019", "2020")
    assert result.matched_periods == ("2020",)
    assert result.numerator_coverage_fraction == pytest.approx(0.5)
    assert result.denominator_coverage_fraction == pytest.approx(0.5)
    assert _period_unmatched(result) == {"numerator": 1, "denominator": 1}


def test_policy_no_shared_period_refuses() -> None:
    numerator = _frame([_row(year="2020", stat_var="dcid:X")])
    denominator = _frame([_row(year="2021", stat_var="dcid:Y")])

    with pytest.raises(ShareOfGrainMismatchError) as exc_info:
        run_share_of(numerator, denominator, numerator_entity={}, denominator_entity={})

    assert exc_info.value.numerator_periods == ("2020",)
    assert exc_info.value.denominator_periods == ("2021",)
    assert exc_info.value.key == ()


def test_policy_more_than_one_row_per_period_is_summed_not_ambiguous() -> None:
    """The declared policy: fsum for any summed side. Unlike `compare`'s
    ambiguous-slice refusal, more than one matching row per period on a
    share_of side is a deliberate total."""
    numerator = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=10.0, stat_var="dcid:X"),
            _row(reporter="dcid:GBR", year="2020", value=20.0, stat_var="dcid:X"),
        ]
    )
    denominator = _frame([_row(year="2020", value=100.0, stat_var="dcid:Y")])

    result = run_share_of(numerator, denominator, numerator_entity={}, denominator_entity={})

    assert result.period_shares[0].numerator_value == pytest.approx(30.0)
    assert result.period_shares[0].percent == pytest.approx(30.0)


def test_zero_denominator_refuses_non_finite_before_dividing() -> None:
    numerator = _frame([_row(year="2020", value=5.0, stat_var="dcid:X")])
    denominator = _frame([_row(year="2020", value=0.0, stat_var="dcid:Y")])

    with pytest.raises(ShareOfNonFiniteResultError) as exc_info:
        run_share_of(numerator, denominator, numerator_entity={}, denominator_entity={})

    assert exc_info.value.time_period == "2020"
    assert exc_info.value.reason == "zero denominator"


@pytest.mark.parametrize(
    ("numerator_values", "denominator_value"),
    [
        # `math.fsum` raises `OverflowError` for a sum whose exact value
        # exceeds the finite double range -- the same guard
        # `aggregate.run_aggregate` and `rank.run_rank` apply to their own
        # `math.fsum` reduction.
        pytest.param((1.0e308, 1.0e308), 100.0, id="summed-numerator"),
        # A finite ratio of 2e307 overflows once stated in per cent.
        pytest.param((1.0e307,), 0.5, id="ratio-in-per-cent"),
    ],
)
def test_overflow_refuses_non_finite(
    numerator_values: tuple[float, ...], denominator_value: float
) -> None:
    numerator = _frame(
        [
            _row(reporter=f"dcid:R{index}", year="2020", value=value, stat_var="dcid:X")
            for index, value in enumerate(numerator_values)
        ]
    )
    denominator = _frame([_row(year="2020", value=denominator_value, stat_var="dcid:Y")])

    with pytest.raises(ShareOfNonFiniteResultError) as exc_info:
        run_share_of(numerator, denominator, numerator_entity={}, denominator_entity={})

    assert exc_info.value.time_period == "2020"
    assert exc_info.value.reason == "overflow"
    assert not math.isfinite(sum(numerator_values) / denominator_value * 100.0)


def test_share_normalises_negative_zero() -> None:
    """`numerator_value / denominator_value` genuinely lands on IEEE-754
    negative zero when `numerator_value=0.0` and `denominator_value` is
    negative (`0.0 / -5.0 == -0.0`) -- `two_sided.divide_cell`'s own
    `quotient + 0.0` normalises that at the engine boundary, not left to the
    digest's own render-time formatting, so `percent` is what the output
    frame itself stores, not just what the digest renders."""
    numerator = _frame([_row(year="2020", value=0.0, stat_var="dcid:X")])
    denominator = _frame([_row(year="2020", value=-5.0, stat_var="dcid:Y")])

    result = run_share_of(numerator, denominator, numerator_entity={}, denominator_entity={})

    assert repr(result.period_shares[0].percent) == "0.0"


# Determinism -------------------------------------------------------------------


def test_entity_binding_key_order_perturbation_produces_an_identical_result() -> None:
    """Varying `numerator_entity`/`denominator_entity` key insertion order
    does not change the result. `Frame.build` absorbs row-mapping key order,
    but an `entity` binding is passed straight to `narrow_and_resolve`,
    which walks `entity.items()` directly -- so two dicts with the same
    pairs in different order reach the engine as different objects.

    Two role columns make key order variation meaningful. The expected
    `percent` is pinned by hand (`30.0 / 100.0`, 30%) to verify the formula
    itself, not just compared against `original`, which could hide a
    uniformly wrong formula applied to both sides."""
    frame = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=30.0, recipient="dcid:KEN"),
            _row(reporter="dcid:GBR", year="2020", value=100.0, recipient="dcid:KEN"),
        ],
        role_keys=["reporter", "recipient"],
    )
    original = run_share_of(
        frame,
        frame,
        numerator_entity={"reporter": "dcid:USA", "recipient": "dcid:KEN"},
        denominator_entity={"reporter": "dcid:GBR", "recipient": "dcid:KEN"},
    )
    perturbed = run_share_of(
        frame,
        frame,
        numerator_entity={"recipient": "dcid:KEN", "reporter": "dcid:USA"},
        denominator_entity={"recipient": "dcid:KEN", "reporter": "dcid:GBR"},
    )

    assert perturbed.period_shares == original.period_shares
    assert original.period_shares[0].percent == pytest.approx(30.0)


# Refusals ------------------------------------------------------------------------


def test_heterogeneous_unit_frame_refuses() -> None:
    frame = multi_unit_frame()
    with pytest.raises(UnitHeterogeneousError):
        run_share_of(frame, frame, numerator_entity={"reporter": "dcid:USA"}, denominator_entity={})


# Cross-side checks and the unitless-row policy -------------------------------


def _one_measure_from_two_sources() -> tuple[Frame, Frame]:
    """Two frames holding the same StatVar, USA in `provenance=dcid:P1` and
    GBR in `dcid:P2`."""
    return (
        _frame([_row(reporter="dcid:USA", unit="dcid:USDollar", provenance="dcid:P1")]),
        _frame([_row(reporter="dcid:GBR", unit="dcid:USDollar", provenance="dcid:P2")]),
    )


@pytest.mark.parametrize(
    "frames",
    [
        pytest.param(lambda: (facet_disjoint_by_reporter_frame(),) * 2, id="one-frame"),
        pytest.param(_one_measure_from_two_sources, id="two-frames"),
    ],
)
def test_cross_side_facet_mismatch_refuses_even_though_each_side_resolves_cleanly(
    frames: Callable[[], tuple[Frame, Frame]],
) -> None:
    """Each side (USA in `provenance=dcid:P1`, GBR in `dcid:P2`) resolves
    cleanly on its own, and both hold the same StatVar; the pairing itself
    must still refuse, whether the sides are two slices of one frame or two
    frames."""
    numerator, denominator = frames()
    with pytest.raises(ShareOfFacetMismatchError) as exc_info:
        run_share_of(
            numerator,
            denominator,
            numerator_entity={"reporter": "dcid:USA"},
            denominator_entity={"reporter": "dcid:GBR"},
        )
    assert exc_info.value.axis == "provenance"
    assert exc_info.value.numerator_ref == "dcid:P1"
    assert exc_info.value.denominator_ref == "dcid:P2"


def test_cross_side_facet_difference_between_two_stat_vars_warns_and_divides() -> None:
    """Debt service from one source over GDP from another: two measures,
    each under its own provenance, divide, and the warning names both
    selections."""
    numerator = _frame(
        [_row(stat_var="dcid:DebtService", value=2.0, unit="dcid:USDollar", provenance="dcid:P1")]
    )
    denominator = _frame(
        [_row(stat_var="dcid:Gdp", value=8.0, unit="dcid:USDollar", provenance="dcid:P2")]
    )

    result = run_share_of(numerator, denominator, numerator_entity={}, denominator_entity={})

    assert [ps.percent for ps in result.period_shares] == [25.0]
    assert result.numerator_facet_selections == (FacetSelection(axis="provenance", ref="dcid:P1"),)
    assert result.denominator_facet_selections == (
        FacetSelection(axis="provenance", ref="dcid:P2"),
    )
    assert len(result.warnings) == 1
    assert "'dcid:P1'" in result.warnings[0]
    assert "'dcid:P2'" in result.warnings[0]


def test_facet_left_wholly_unbound_on_one_side_succeeds_rather_than_refusing() -> None:
    """The softer facet policy: an axis one side never declared is not a
    disagreement between two declared values, so it resolves via
    auto-selection on the side that has it rather than refusing the
    pairing -- but it must still warn, naming the axis, which side left it
    unbound, and whether the resolved facets actually carry a value for it,
    regardless of which side is passed as the numerator.
    """
    frame = _frame([_row(reporter="dcid:USA", provenance="dcid:P1"), _row(reporter="dcid:GBR")])

    # USA (declares P1) as numerator: the trace's own resolved_facets
    # record carries P1 -- share_of always records the numerator's own
    # facet selections.
    usa_numerator = run_share_of(
        frame,
        frame,
        numerator_entity={"reporter": "dcid:USA"},
        denominator_entity={"reporter": "dcid:GBR"},
    )
    assert usa_numerator.numerator_facet_selections == (
        FacetSelection(axis="provenance", ref="dcid:P1"),
    )
    assert usa_numerator.denominator_facet_selections == ()
    assert len(usa_numerator.warnings) == 1
    assert "provenance" in usa_numerator.warnings[0]
    assert "denominator" in usa_numerator.warnings[0]
    assert "dcid:P1" in usa_numerator.warnings[0]

    # GBR (declares nothing) as numerator: the trace's own resolved_facets
    # record carries *no* value for this axis at all, even though the
    # denominator (USA) does declare one.
    gbr_numerator = run_share_of(
        frame,
        frame,
        numerator_entity={"reporter": "dcid:GBR"},
        denominator_entity={"reporter": "dcid:USA"},
    )
    assert gbr_numerator.numerator_facet_selections == ()
    assert gbr_numerator.denominator_facet_selections == (
        FacetSelection(axis="provenance", ref="dcid:P1"),
    )
    assert len(gbr_numerator.warnings) == 1
    assert "provenance" in gbr_numerator.warnings[0]
    assert "numerator" in gbr_numerator.warnings[0]
    assert "no value" in gbr_numerator.warnings[0]


def test_entity_matching_no_rows_with_a_real_pinned_unit_refuses_grain_mismatch_not_unit_not_present() -> (
    None
):
    """An `entity` binding matching zero rows must not be checked against
    `pinned_unit` at all -- `plain_single_unit_frame` genuinely declares
    `dcid:USDollar`, just not for a reporter that doesn't exist in it. The
    real problem is the empty denominator, which `ShareOfGrainMismatchError`
    names; `UnitNotPresentError` would misreport it as declaring nothing."""
    frame = plain_single_unit_frame()
    with pytest.raises(ShareOfGrainMismatchError) as exc_info:
        run_share_of(
            frame,
            frame,
            numerator_entity={"reporter": "dcid:USA"},
            denominator_entity={"reporter": "dcid:NOPE"},
            denominator_pinned_unit="dcid:USDollar",
        )
    assert exc_info.value.denominator_periods == ()


def test_both_sides_empty_refuses_instead_of_a_vacuous_zero_period_success() -> None:
    """A typo'd role value on *both* `numerator_entity` and
    `denominator_entity` makes both sides' own period sets the empty tuple
    -- and `set() != set()` is `False`, two empty sets are equal, so the
    grain-mismatch check alone would never fire. Left unguarded, this falls
    through the empty `period_shares` loop and returns clean:
    `is_error=False`, a registered pair, "0 matched period(s)" -- a vacuous
    success a caller could easily mistake for a genuine (if unlucky) empty
    overlap, rather than the row-selection mistake it actually is. This
    must refuse the same as any other grain mismatch, not merely return
    with nothing shared."""
    frame = plain_single_unit_frame()
    with pytest.raises(ShareOfGrainMismatchError) as exc_info:
        run_share_of(
            frame,
            frame,
            numerator_entity={"reporter": "dcid:TYPO_NUMERATOR"},
            denominator_entity={"reporter": "dcid:TYPO_DENOMINATOR"},
        )
    assert exc_info.value.numerator_periods == ()
    assert exc_info.value.denominator_periods == ()


def test_unit_unbound_side_refuses_distinctly_from_a_genuine_unit_mismatch() -> None:
    """`unit_partially_declared_frame` declares nothing but dollars --
    a USD-declaring numerator (USA) paired against a denominator (FRA)
    that leaves `unit` unbound entirely must say that plainly, not report
    it as two resolved units disagreeing (`ShareOfUnitMismatchError`)."""
    frame = unit_partially_declared_frame()
    with pytest.raises(ShareOfUnitUnboundError) as exc_info:
        run_share_of(
            frame,
            frame,
            numerator_entity={"reporter": "dcid:USA"},
            denominator_entity={"reporter": "dcid:FRA"},
        )
    assert exc_info.value.unbound_side == "denominator"
    assert exc_info.value.bound_side == "numerator"
    assert exc_info.value.bound_unit == "dcid:USDollar"


def test_unpinned_summed_side_excludes_unitless_rows_like_aggregate_does() -> None:
    """The same `math.fsum` reduction `aggregate` already applies this
    policy to: ETH's own unitless row must not be silently summed into a
    total the output frame labels `dcid:USDollar` -- reported structurally,
    per side, as a `unit_unbound` exclusion tagged `numerator`/`denominator`
    (the identical frame supplies both sides here, so both tags appear)."""
    frame = unit_and_unitless_row_frame()
    result = run_share_of(frame, frame, numerator_entity={}, denominator_entity={})
    assert result.period_shares[0].numerator_value == 100.0
    assert result.period_shares[0].denominator_value == 100.0
    sides = {record.side for record in result.exclusions.records if record.reason == "unit_unbound"}
    assert sides == {"numerator", "denominator"}


def test_unit_unbound_exclusion_is_tagged_by_which_side_dropped_it() -> None:
    """Only the numerator's own rows leave `unit` unbound here -- the
    dropped row's own exclusion must say `numerator`, not `denominator`,
    so a reader can tell which side of the ratio actually lost data (a
    dropped numerator row deflates the reported share; a dropped
    denominator row inflates it)."""
    numerator = _frame(
        [
            _row(reporter="dcid:USA", value=50.0, unit="dcid:USDollar"),
            _row(reporter="dcid:USA", value=25.0),  # unit left unbound
        ]
    )
    denominator = _frame([_row(reporter="dcid:USA", value=200.0, unit="dcid:USDollar")])

    result = run_share_of(numerator, denominator, numerator_entity={}, denominator_entity={})

    unit_unbound = [
        record for record in result.exclusions.records if record.reason == "unit_unbound"
    ]
    assert len(unit_unbound) == 1
    assert unit_unbound[0].side == "numerator"
    assert unit_unbound[0].row_count == 1


def test_entity_narrowed_share_of_records_the_dropped_entities_per_side() -> None:
    """`numerator_entity`/`denominator_entity` each narrow `frame` down to
    one reporter -- the same `narrow_and_resolve` entity-narrowing
    accounting `compare` gets, tagged `numerator`/`denominator` here."""
    frame = _frame(
        [
            _row(reporter="dcid:USA", value=10.0),
            _row(reporter="dcid:GBR", value=20.0),
            _row(reporter="dcid:DEU", value=30.0),
        ]
    )
    result = run_share_of(
        frame,
        frame,
        numerator_entity={"reporter": "dcid:USA"},
        denominator_entity={"reporter": "dcid:GBR"},
    )
    entity_records = {
        record.side: record
        for record in result.exclusions.records
        if record.reason == "entity_not_selected"
    }
    assert entity_records["numerator"].row_count == 2
    assert entity_records["numerator"].entities["reporter"] == ("dcid:DEU", "dcid:GBR")
    assert entity_records["denominator"].row_count == 2
    assert entity_records["denominator"].entities["reporter"] == ("dcid:DEU", "dcid:USA")


# Composed units ------------------------------------------------------------------

_USD_PER_PERSON = unit_cell_to_json(
    ComposedUnit(numerator="dcid:USDollar", denominator=StatVarPart("dcid:Count_Person"))
)


def _aid_per_head() -> Frame:
    """Aid per head for two recipients in one period, in dollars per person."""
    return _frame(
        [
            _row(reporter="dcid:KEN", value=10.0, unit=_USD_PER_PERSON),
            _row(reporter="dcid:GTM", value=1.0, unit=_USD_PER_PERSON),
        ]
    )


@pytest.mark.parametrize(
    ("side", "numerator_entity", "denominator_entity"),
    [
        ("denominator", {"reporter": "dcid:KEN"}, {}),
        ("numerator", {}, {"reporter": "dcid:KEN"}),
    ],
)
def test_a_composed_unit_side_summing_several_rows_in_a_cell_refuses(
    side: str, numerator_entity: dict[str, str], denominator_entity: dict[str, str]
) -> None:
    frame = _aid_per_head()

    with pytest.raises(ComposedUnitSumError) as exc_info:
        run_share_of(
            frame,
            frame,
            numerator_entity=numerator_entity,
            denominator_entity=denominator_entity,
        )

    assert exc_info.value.side == side
    assert unit_cell_to_json(exc_info.value.unit) == _USD_PER_PERSON


def test_a_keyed_share_over_a_pooled_composed_unit_denominator_refuses() -> None:
    frame = _aid_per_head()

    with pytest.raises(ComposedUnitSumError) as exc_info:
        run_share_of(
            frame,
            frame,
            numerator_entity={},
            denominator_entity={},
            numerator_by=("reporter",),
        )

    assert exc_info.value.side == "denominator"


@pytest.mark.parametrize("denominator_by", [{}, {"reporter": "reporter"}])
def test_composed_unit_sides_holding_one_row_per_cell_compute(
    denominator_by: dict[str, str],
) -> None:
    """One per-head figure over another is a ratio of two figures: a keyed
    denominator, or a pooled one bound to a single recipient, adds nothing."""
    frame = _aid_per_head()

    result = run_share_of(
        frame,
        frame,
        numerator_entity={},
        denominator_entity={} if denominator_by else {"reporter": "dcid:KEN"},
        numerator_by=("reporter",),
        denominator_by=denominator_by,
    )

    shares = {ps.key: ps.percent for ps in result.period_shares}
    expected_gtm = 100.0 if denominator_by else 10.0
    assert shares == {
        (("reporter", "dcid:KEN"),): 100.0,
        (("reporter", "dcid:GTM"),): expected_gtm,
    }


# Keyed share_of ------------------------------------------------------------------


def _keyed_frame(rows: list[dict[str, object]], *, role_keys: list[str]) -> Frame:
    return Frame.build(role_keys=role_keys, rows=rows)


def test_per_axis_keyed_share_of_yields_one_share_per_donor_per_period() -> None:
    """A two-donor numerator over a per-axis denominator: each donor's own
    share of its own denominator total, one row per (donor, period)."""
    numerator = _keyed_frame(
        [
            {
                "stat_var": "dcid:X",
                "donor": "dcid:USA",
                "time_period": "2020",
                "value": 10.0,
            },
            {
                "stat_var": "dcid:X",
                "donor": "dcid:GBR",
                "time_period": "2020",
                "value": 20.0,
            },
        ],
        role_keys=["donor"],
    )
    denominator = _keyed_frame(
        [
            {
                "stat_var": "dcid:Y",
                "provider": "dcid:USA",
                "time_period": "2020",
                "value": 100.0,
            },
            {
                "stat_var": "dcid:Y",
                "provider": "dcid:GBR",
                "time_period": "2020",
                "value": 200.0,
            },
        ],
        role_keys=["provider"],
    )

    result = run_share_of(
        numerator,
        denominator,
        numerator_entity={},
        denominator_entity={},
        numerator_by=("donor",),
        denominator_by={"donor": "provider"},
    )

    shares = {ps.key: ps.percent for ps in result.period_shares}
    assert shares[(("donor", "dcid:USA"),)] == pytest.approx(10.0)
    assert shares[(("donor", "dcid:GBR"),)] == pytest.approx(10.0)
    assert result.frame.rows == tuple(
        sorted(result.frame.rows, key=lambda row: row[result.frame.columns.index("donor")])
    )


def test_pooled_keyed_share_of_divides_by_the_one_period_total_and_differs_from_per_axis() -> None:
    """The same numerator, pooled (no `denominator_by`): each donor's own
    value over the *combined* denominator total, not its own counterpart --
    a different number from the per-axis reading above."""
    numerator = _keyed_frame(
        [
            {
                "stat_var": "dcid:X",
                "donor": "dcid:USA",
                "time_period": "2020",
                "value": 10.0,
            },
            {
                "stat_var": "dcid:X",
                "donor": "dcid:GBR",
                "time_period": "2020",
                "value": 20.0,
            },
        ],
        role_keys=["donor"],
    )
    denominator = _keyed_frame(
        [{"stat_var": "dcid:Y", "time_period": "2020", "value": 300.0}],
        role_keys=[],
    )

    result = run_share_of(
        numerator, denominator, numerator_entity={}, denominator_entity={}, numerator_by=("donor",)
    )

    shares = {ps.key: ps.percent for ps in result.period_shares}
    assert shares[(("donor", "dcid:USA"),)] == pytest.approx(100 * 10.0 / 300.0)
    assert shares[(("donor", "dcid:GBR"),)] == pytest.approx(100 * 20.0 / 300.0)
    # Distinct from the per-axis reading above (10% for both donors): pooled
    # divides by the combined total, not each donor's own counterpart.
    assert shares[(("donor", "dcid:USA"),)] != pytest.approx(10.0)


def test_two_keys_pair_by_name_not_by_sorted_order() -> None:
    """The regression test named in the plan's verification table: the
    positional bug revision 1 carried, and the test that would fail
    against `run_share_of`'s pre-keying signature with a bare `TypeError`.
    `numerator_by=("donor","sector")` sorts to itself alphabetically, and a
    positionally-paired `denominator_by=["provider","area"]` list would
    `numerator_by=("donor","sector")` sorts to itself alphabetically, and a
    positionally-paired `denominator_by=["provider","area"]` list would
    independently sort to `["area","provider"]` -- donor paired with area,
    sector paired with provider. This fixture is adversarial: a denominator
    row exists under the *wrong* (swapped) pairing for every numerator row,
    so a positional bug would not crash, it would silently divide by the
    wrong total. The name-keyed mapping `{"donor": "provider", "sector":
    "area"}` must resolve every key correctly regardless.
    """
    numerator = _keyed_frame(
        [
            {
                "stat_var": "dcid:X",
                "donor": "dcid:USA",
                "sector": "dcid:Health",
                "time_period": "2020",
                "value": 10.0,
            },
            {
                "stat_var": "dcid:X",
                "donor": "dcid:GBR",
                "sector": "dcid:Education",
                "time_period": "2020",
                "value": 20.0,
            },
        ],
        role_keys=["donor", "sector"],
    )
    denominator = _keyed_frame(
        [
            # The correct (name-keyed) denominator group for each numerator row.
            {
                "stat_var": "dcid:Y",
                "provider": "dcid:USA",
                "area": "dcid:Health",
                "time_period": "2020",
                "value": 1000.0,
            },
            {
                "stat_var": "dcid:Y",
                "provider": "dcid:GBR",
                "area": "dcid:Education",
                "time_period": "2020",
                "value": 2000.0,
            },
            # A group that only a *swapped* (donor<->area, sector<->provider)
            # pairing would find, sharing the identical value tuple a
            # positional bug would read off the first numerator row.
            {
                "stat_var": "dcid:Y",
                "provider": "dcid:Health",
                "area": "dcid:USA",
                "time_period": "2020",
                "value": 9999.0,
            },
            {
                "stat_var": "dcid:Y",
                "provider": "dcid:Education",
                "area": "dcid:GBR",
                "time_period": "2020",
                "value": 8888.0,
            },
        ],
        role_keys=["provider", "area"],
    )

    result = run_share_of(
        numerator,
        denominator,
        numerator_entity={},
        denominator_entity={},
        numerator_by=("donor", "sector"),
        denominator_by={"donor": "provider", "sector": "area"},
    )

    shares = {ps.key: ps.percent for ps in result.period_shares}
    usa_key = (("donor", "dcid:USA"), ("sector", "dcid:Health"))
    gbr_key = (("donor", "dcid:GBR"), ("sector", "dcid:Education"))
    assert shares[usa_key] == pytest.approx(100 * 10.0 / 1000.0)
    assert shares[gbr_key] == pytest.approx(100 * 20.0 / 2000.0)
    # The swapped-pairing reading would have divided by 9999.0/8888.0 instead.
    assert shares[usa_key] != pytest.approx(100 * 10.0 / 9999.0)
    assert shares[gbr_key] != pytest.approx(100 * 20.0 / 8888.0)


def test_denominator_by_key_not_in_numerator_by_refuses_at_the_op_spec() -> None:
    """A `denominator_by` key `numerator_by` itself does not carry refuses
    structurally -- checked purely from the op-spec's own two fields, no
    frame data needed. `ShareOfOpSpec`'s own `model_validator` raises this,
    not `run_share_of`."""
    with pytest.raises(ValueError, match="denominator_by names column"):
        ShareOfOpSpec(
            frame_ref={"spec_id": "spec_" + "a" * 32, "content_version": "cv_" + "b" * 32},
            numerator_by=("donor",),
            denominator_by={"sector": "area"},
        )


def test_denominator_by_partial_mapping_refuses_at_the_engine() -> None:
    """`denominator_by` non-empty but leaving a `numerator_by` column out --
    a partial mapping has no defined pairing for the column it omits. Every
    `denominator_by` key here *is* one `numerator_by` names (so the
    op-spec's own membership check would pass); the mapping is simply
    incomplete, which only `run_share_of` itself can tell."""
    numerator = _keyed_frame(
        [
            {
                "stat_var": "dcid:X",
                "donor": "dcid:USA",
                "sector": "dcid:Health",
                "time_period": "2020",
                "value": 10.0,
            }
        ],
        role_keys=["donor", "sector"],
    )
    denominator = _keyed_frame(
        [
            {
                "stat_var": "dcid:Y",
                "provider": "dcid:USA",
                "time_period": "2020",
                "value": 100.0,
            }
        ],
        role_keys=["provider"],
    )

    with pytest.raises(ShareOfKeyUnknownError) as exc_info:
        run_share_of(
            numerator,
            denominator,
            numerator_entity={},
            denominator_entity={},
            numerator_by=("donor", "sector"),
            denominator_by={"donor": "provider"},
        )
    assert exc_info.value.column == "sector"


def test_numerator_key_absent_from_denominator_refuses_in_per_axis_mode() -> None:
    """A numerator key with no matching denominator group under
    `denominator_by`'s own mapping refuses distinctly from a grain
    mismatch -- there is no denominator series to compare periods against
    at all."""
    numerator = _keyed_frame(
        [
            {
                "stat_var": "dcid:X",
                "donor": "dcid:USA",
                "time_period": "2020",
                "value": 10.0,
            },
            {
                "stat_var": "dcid:X",
                "donor": "dcid:FRA",
                "time_period": "2020",
                "value": 5.0,
            },
        ],
        role_keys=["donor"],
    )
    denominator = _keyed_frame(
        [
            {
                "stat_var": "dcid:Y",
                "provider": "dcid:USA",
                "time_period": "2020",
                "value": 100.0,
            }
        ],
        role_keys=["provider"],
    )

    with pytest.raises(ShareOfKeyUnmatchedError) as exc_info:
        run_share_of(
            numerator,
            denominator,
            numerator_entity={},
            denominator_entity={},
            numerator_by=("donor",),
            denominator_by={"donor": "provider"},
        )
    assert exc_info.value.key == (("donor", "dcid:FRA"),)
    assert exc_info.value.available == ((("donor", "dcid:USA"),),)


def test_per_key_unmatched_period_is_left_out_for_that_key_only() -> None:
    """Periods pair per key, not pooled across every key: FRA's 2021, which
    its own denominator group lacks, is left out and recorded against FRA,
    and every other share stands."""
    numerator = _keyed_frame(
        [
            {
                "stat_var": "dcid:X",
                "donor": "dcid:USA",
                "time_period": "2020",
                "value": 10.0,
            },
            {
                "stat_var": "dcid:X",
                "donor": "dcid:FRA",
                "time_period": "2020",
                "value": 5.0,
            },
            {
                "stat_var": "dcid:X",
                "donor": "dcid:FRA",
                "time_period": "2021",
                "value": 6.0,
            },
        ],
        role_keys=["donor"],
    )
    denominator = _keyed_frame(
        [
            {
                "stat_var": "dcid:Y",
                "provider": "dcid:USA",
                "time_period": "2020",
                "value": 100.0,
            },
            {
                "stat_var": "dcid:Y",
                "provider": "dcid:FRA",
                "time_period": "2020",
                "value": 50.0,
            },
            # No 2021 row for FRA: FRA's own periods (2020, 2021) disagree
            # with its own denominator counterpart's periods (2020 only).
        ],
        role_keys=["provider"],
    )

    result = run_share_of(
        numerator,
        denominator,
        numerator_entity={},
        denominator_entity={},
        numerator_by=("donor",),
        denominator_by={"donor": "provider"},
    )

    assert [(ps.key, ps.time_period, ps.percent) for ps in result.period_shares] == [
        ((("donor", "dcid:FRA"),), "2020", 10.0),
        ((("donor", "dcid:USA"),), "2020", 10.0),
    ]
    assert result.numerator_only_periods == (((("donor", "dcid:FRA"),), ("2021",)),)
    assert result.denominator_only_periods == ()
    assert result.numerator_coverage_fraction == pytest.approx(2 / 3)
    assert result.denominator_coverage_fraction == pytest.approx(1.0)
    unmatched = [
        record for record in result.exclusions.records if record.reason == "period_unmatched"
    ]
    assert [(record.side, record.row_count, record.entities) for record in unmatched] == [
        ("numerator", 1, {"donor": ("dcid:FRA",)})
    ]


def test_per_key_with_no_shared_period_refuses_naming_the_key() -> None:
    """A key sharing no period with its own denominator group refuses
    naming that key, even when every other key divides."""
    numerator = _keyed_frame(
        [
            {"stat_var": "dcid:X", "donor": "dcid:USA", "time_period": "2020", "value": 10.0},
            {"stat_var": "dcid:X", "donor": "dcid:FRA", "time_period": "2021", "value": 6.0},
        ],
        role_keys=["donor"],
    )
    denominator = _keyed_frame(
        [
            {"stat_var": "dcid:Y", "provider": "dcid:USA", "time_period": "2020", "value": 100.0},
            {"stat_var": "dcid:Y", "provider": "dcid:FRA", "time_period": "2020", "value": 50.0},
        ],
        role_keys=["provider"],
    )

    with pytest.raises(ShareOfGrainMismatchError) as exc_info:
        run_share_of(
            numerator,
            denominator,
            numerator_entity={},
            denominator_entity={},
            numerator_by=("donor",),
            denominator_by={"donor": "provider"},
        )
    assert exc_info.value.key == (("donor", "dcid:FRA"),)
    assert exc_info.value.numerator_periods == ("2021",)
    assert exc_info.value.denominator_periods == ("2020",)


def test_zero_denominator_for_one_key_refuses() -> None:
    """A zero-summed denominator for one key refuses the same as an
    un-keyed call's zero denominator, naming that key's own period."""
    numerator = _keyed_frame(
        [
            {
                "stat_var": "dcid:X",
                "donor": "dcid:USA",
                "time_period": "2020",
                "value": 10.0,
            },
            {
                "stat_var": "dcid:X",
                "donor": "dcid:FRA",
                "time_period": "2020",
                "value": 5.0,
            },
        ],
        role_keys=["donor"],
    )
    denominator = _keyed_frame(
        [
            {
                "stat_var": "dcid:Y",
                "provider": "dcid:USA",
                "time_period": "2020",
                "value": 100.0,
            },
            {
                "stat_var": "dcid:Y",
                "provider": "dcid:FRA",
                "time_period": "2020",
                "value": 0.0,
            },
        ],
        role_keys=["provider"],
    )

    with pytest.raises(ShareOfNonFiniteResultError) as exc_info:
        run_share_of(
            numerator,
            denominator,
            numerator_entity={},
            denominator_entity={},
            numerator_by=("donor",),
            denominator_by={"donor": "provider"},
        )
    assert exc_info.value.time_period == "2020"
    assert exc_info.value.reason == "zero denominator"


def test_role_keys_output_frame_carries_numerator_by_as_its_own_role_columns() -> None:
    """The output frame's own role columns are `numerator_by`, not the
    denominator's own column names -- `Frame.build` sorts role_keys, so the
    frame carries both `donor` and `sector`, not `provider`/`area`."""
    numerator = _keyed_frame(
        [
            {
                "stat_var": "dcid:X",
                "donor": "dcid:USA",
                "sector": "dcid:Health",
                "time_period": "2020",
                "value": 10.0,
            }
        ],
        role_keys=["donor", "sector"],
    )
    denominator = _keyed_frame(
        [
            {
                "stat_var": "dcid:Y",
                "provider": "dcid:USA",
                "area": "dcid:Health",
                "time_period": "2020",
                "value": 100.0,
            }
        ],
        role_keys=["provider", "area"],
    )

    result = run_share_of(
        numerator,
        denominator,
        numerator_entity={},
        denominator_entity={},
        numerator_by=("donor", "sector"),
        denominator_by={"donor": "provider", "sector": "area"},
    )

    assert "donor" in result.frame.columns
    assert "sector" in result.frame.columns
    assert "provider" not in result.frame.columns
    assert "area" not in result.frame.columns
