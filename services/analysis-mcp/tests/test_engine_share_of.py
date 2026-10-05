"""`run_share_of`: the grain-match policy (refuse, never report-both-and-
proceed), fsum-for-any-summed-side, determinism (order perturbation), keyed
share_of's per-key/per-axis grouping, and every refusal condition.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from analysis_mcp.engine.common import FacetSelection, UnitHeterogeneousError
from analysis_mcp.engine.share_of import (
    ShareOfFacetMismatchError,
    ShareOfGrainMismatchError,
    ShareOfKeyUnknownError,
    ShareOfKeyUnmatchedError,
    ShareOfNonFiniteResultError,
    ShareOfUnitUnboundError,
    run_share_of,
)
from analysis_mcp.fixtures import (
    facet_disjoint_by_reporter_frame,
    join_compatible_frame_pair,
    multi_unit_frame,
    plain_single_unit_frame,
    unit_and_unitless_row_frame,
    unit_partially_declared_frame,
)
from analysis_mcp.op_spec import ShareOfOpSpec
from dfo_frame_grammar import Frame

_FIXTURES = Path(__file__).parent / "fixtures"


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
            "share": ps.share,
        }
        for ps in result.period_shares
    ]
    assert actual == golden["expected_period_shares"]


def test_numerator_by_empty_reproduces_the_output_frame_byte_for_byte() -> None:
    """`numerator_by=()` (the default) must build the identical output frame
    keyed share_of's per-key grouping was layered on top of -- no role
    columns, one row per matched period, `stat_var` inherited from the
    numerator. This is the regression bar the per-key grouping added in
    this PR must clear."""
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
            "stat_var": "dcid:OfficialDevelopmentAssistanceNet_Disbursement",
            "time_period": share["time_period"],
            "value": share["share"],
            "unit": None,
            "measurementMethod": None,
            "provenance": None,
            "status": None,
        }
        for share in golden["expected_period_shares"]
    ]
    actual_rows = [dict(zip(result.frame.columns, row, strict=True)) for row in result.frame.rows]
    assert actual_rows == expected_rows


# Declared policy: refuse on grain mismatch, never report-both-and-proceed ------


def test_policy_grain_mismatch_refuses() -> None:
    """The declared policy: a denominator whose periods
    don't match the numerator's own refuses, rather than reporting both
    sides' numbers and proceeding regardless. This test fails if that
    policy ever changes to "intersect and proceed" (compare's own looser
    policy)."""
    numerator = _frame([_row(year="2020", stat_var="dcid:X"), _row(year="2021", stat_var="dcid:X")])
    denominator = _frame([_row(year="2020", stat_var="dcid:Y")])

    with pytest.raises(ShareOfGrainMismatchError) as exc_info:
        run_share_of(numerator, denominator, numerator_entity={}, denominator_entity={})

    assert exc_info.value.numerator_periods == ("2020", "2021")
    assert exc_info.value.denominator_periods == ("2020",)
    assert exc_info.value.numerator_coverage_fraction == pytest.approx(0.5)
    assert exc_info.value.denominator_coverage_fraction == pytest.approx(1.0)


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
    assert result.period_shares[0].share == pytest.approx(0.3)


def test_zero_denominator_refuses_non_finite_before_dividing() -> None:
    numerator = _frame([_row(year="2020", value=5.0, stat_var="dcid:X")])
    denominator = _frame([_row(year="2020", value=0.0, stat_var="dcid:Y")])

    with pytest.raises(ShareOfNonFiniteResultError) as exc_info:
        run_share_of(numerator, denominator, numerator_entity={}, denominator_entity={})

    assert exc_info.value.time_period == "2020"
    assert exc_info.value.reason == "zero denominator"


def test_numerator_overflow_refuses_non_finite() -> None:
    """`math.fsum` raises `OverflowError` for a sum whose exact value
    exceeds the finite double range -- the same guard `aggregate.run_
    aggregate` and `rank.run_rank` already apply to their own `math.fsum`
    reduction; this mirrors `test_engine_aggregate.py`'s own
    `test_non_finite_result_refuses`."""
    near_max = 1.0e308
    numerator = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=near_max, stat_var="dcid:X"),
            _row(reporter="dcid:GBR", year="2020", value=near_max, stat_var="dcid:X"),
        ]
    )
    denominator = _frame([_row(year="2020", value=100.0, stat_var="dcid:Y")])

    with pytest.raises(ShareOfNonFiniteResultError) as exc_info:
        run_share_of(numerator, denominator, numerator_entity={}, denominator_entity={})

    assert exc_info.value.time_period == "2020"
    assert exc_info.value.reason == "overflow"
    assert not math.isfinite(near_max + near_max)  # sanity: this really does overflow


def test_share_normalises_negative_zero() -> None:
    """`numerator_value / denominator_value` genuinely lands on IEEE-754
    negative zero when `numerator_value=0.0` and `denominator_value` is
    negative (`0.0 / -5.0 == -0.0`) -- `run_share_of`'s own `share += 0.0`
    normalises that at the engine boundary, not left to the digest's own
    render-time `_format_number`, so `share` is what the output frame
    itself stores, not just what the digest renders."""
    numerator = _frame([_row(year="2020", value=0.0, stat_var="dcid:X")])
    denominator = _frame([_row(year="2020", value=-5.0, stat_var="dcid:Y")])

    result = run_share_of(numerator, denominator, numerator_entity={}, denominator_entity={})

    assert repr(result.period_shares[0].share) == "0.0"


# Determinism -------------------------------------------------------------------


def test_entity_binding_key_order_perturbation_produces_an_identical_result() -> None:
    """Varying `numerator_entity`/`denominator_entity` key insertion order
    does not change the result. `Frame.build` absorbs row-mapping key order,
    but an `entity` binding is passed straight to `narrow_and_resolve`,
    which walks `entity.items()` directly -- so two dicts with the same
    pairs in different order reach the engine as different objects.

    Two role columns make key order variation meaningful. The expected
    `share` is pinned by hand (`30.0 / 100.0`) to verify the formula
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
    assert original.period_shares[0].share == pytest.approx(0.3)


# Refusals ------------------------------------------------------------------------


def test_heterogeneous_unit_frame_refuses() -> None:
    frame = multi_unit_frame()
    with pytest.raises(UnitHeterogeneousError):
        run_share_of(frame, frame, numerator_entity={"reporter": "dcid:USA"}, denominator_entity={})


# Cross-side checks and the unitless-row policy -------------------------------


def test_cross_side_facet_mismatch_refuses_even_though_each_side_resolves_cleanly() -> None:
    """Each side (USA in `provenance=dcid:P1`, GBR in `dcid:P2`) resolves
    cleanly on its own; the pairing itself must still refuse."""
    frame = facet_disjoint_by_reporter_frame()
    with pytest.raises(ShareOfFacetMismatchError) as exc_info:
        run_share_of(
            frame,
            frame,
            numerator_entity={"reporter": "dcid:USA"},
            denominator_entity={"reporter": "dcid:GBR"},
        )
    assert exc_info.value.axis == "provenance"
    assert exc_info.value.numerator_ref == "dcid:P1"
    assert exc_info.value.denominator_ref == "dcid:P2"


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
    assert exc_info.value.matched_periods == ()


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

    shares = {ps.key: ps.share for ps in result.period_shares}
    assert shares[(("donor", "dcid:USA"),)] == pytest.approx(0.1)
    assert shares[(("donor", "dcid:GBR"),)] == pytest.approx(0.1)
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

    shares = {ps.key: ps.share for ps in result.period_shares}
    assert shares[(("donor", "dcid:USA"),)] == pytest.approx(10.0 / 300.0)
    assert shares[(("donor", "dcid:GBR"),)] == pytest.approx(20.0 / 300.0)
    # Distinct from the per-axis reading above (0.1 for both donors): pooled
    # divides by the combined total, not each donor's own counterpart.
    assert shares[(("donor", "dcid:USA"),)] != pytest.approx(0.1)


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

    shares = {ps.key: ps.share for ps in result.period_shares}
    usa_key = (("donor", "dcid:USA"), ("sector", "dcid:Health"))
    gbr_key = (("donor", "dcid:GBR"), ("sector", "dcid:Education"))
    assert shares[usa_key] == pytest.approx(10.0 / 1000.0)
    assert shares[gbr_key] == pytest.approx(20.0 / 2000.0)
    # The swapped-pairing reading would have divided by 9999.0/8888.0 instead.
    assert shares[usa_key] != pytest.approx(10.0 / 9999.0)
    assert shares[gbr_key] != pytest.approx(20.0 / 8888.0)


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


def test_per_key_period_set_mismatch_refuses_naming_the_key() -> None:
    """The identical-period-set rule applies per key, not pooled across
    every key -- one key's own mismatch must name that key, not the whole
    call."""
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
    assert exc_info.value.numerator_periods == ("2020", "2021")
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
