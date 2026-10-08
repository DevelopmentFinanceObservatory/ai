"""`run_sum_of`: period intersection, the per-period sum, the facet policy,
the result's own `stat_var`, and every refusal condition.
"""

from __future__ import annotations

import pytest
from analysis_mcp.engine.aggregate import ComposedUnitSumError
from analysis_mcp.engine.compare import (
    CompareAmbiguousSliceError,
    CompareNonFiniteResultError,
    CompareUnitMismatchError,
    CompareUnitUnboundError,
    ComparisonPeriodsDisjointError,
)
from analysis_mcp.engine.sum_of import run_sum_of
from analysis_mcp.fixtures import join_compatible_frame_pair, two_source_sum_frame_pair
from analysis_mcp.result import observed_shape
from dfo_frame_grammar import Frame
from dfo_frame_grammar.grammar import ComposedUnit, StatVarPart

_OWN_REF = "dcid:ops_" + "e" * 32


def _frame(rows: list[dict[str, object]]) -> Frame:
    return Frame.build(role_keys=["reporter"], rows=rows)


def _row(
    *, year: str = "2020", value: float = 1.0, stat_var: str = "dcid:X", **extra: object
) -> dict[str, object]:
    return {
        "stat_var": stat_var,
        "reporter": "dcid:USA",
        "time_period": year,
        "value": value,
        **extra,
    }


def test_two_datasets_sum_over_the_years_both_report() -> None:
    """Reported bilateral flows (2010-2024) plus modelled imputed shares
    (2011-2023). The sum covers 2011-2023, each side's coverage is stated,
    and the differing facets are warned about and left unbound."""
    bilateral, imputed = two_source_sum_frame_pair()

    result = run_sum_of(bilateral, imputed, stat_var=_OWN_REF, left_entity={}, right_entity={})

    years = tuple(str(year) for year in range(2011, 2024))
    assert result.intersected_periods == years
    assert result.left_periods == tuple(str(year) for year in range(2010, 2025))
    assert result.right_periods == years
    assert result.left_coverage_fraction == 13 / 15
    assert result.right_coverage_fraction == 1.0
    first, last = result.period_sums[0], result.period_sums[-1]
    assert (first.time_period, first.left_value, first.right_value, first.value) == (
        "2011",
        4_250_000.0,
        1_500_000.0,
        5_750_000.0,
    )
    assert (last.time_period, last.value) == ("2023", 7_250_000.0 + 2_700_000.0)

    assert observed_shape(result.frame) == ((), 1)
    columns = result.frame.table.to_pydict()
    assert set(columns["stat_var"]) == {_OWN_REF}
    assert columns["value"] == [entry.value for entry in result.period_sums]
    assert result.unit == "dcid:USDollar"
    assert set(columns["provenance"]) == {None}
    assert set(columns["measurementMethod"]) == {None}
    assert result.facet_selections == ()
    assert len(result.warnings) == 2
    assert all("differs between the sides" in warning for warning in result.warnings)
    assert result.left_stat_vars == ("dcid:OdaBilateralGrossDisbursement",)
    assert result.right_stat_vars == ("dcid:OdaImputedMultilateralGrossDisbursement",)
    excluded = {(record.side, record.row_count) for record in result.exclusions.records}
    assert ("left", 2) in excluded


@pytest.mark.parametrize(
    ("left_extra", "right_extra", "carried", "warning_count"),
    [
        ({"provenance": "dcid:P1"}, {"provenance": "dcid:P1"}, ("dcid:P1",), 0),
        ({"provenance": "dcid:P1"}, {"provenance": "dcid:P2"}, (), 1),
        ({"provenance": "dcid:P1"}, {}, (), 1),
        ({}, {"provenance": "dcid:P2"}, (), 1),
    ],
    ids=["agree", "differ", "left only", "right only"],
)
def test_a_facet_is_carried_only_when_both_sides_declare_the_same_value(
    left_extra: dict[str, str],
    right_extra: dict[str, str],
    carried: tuple[str, ...],
    warning_count: int,
) -> None:
    unit = {"unit": "dcid:USDollar"}
    left = _frame([_row(**unit, **left_extra)])
    right = _frame([_row(**unit, **right_extra)])

    result = run_sum_of(left, right, stat_var=_OWN_REF, left_entity={}, right_entity={})

    assert tuple(selection.ref for selection in result.facet_selections) == carried
    assert result.frame.table.to_pydict()["provenance"] == [carried[0] if carried else None]
    assert len(result.warnings) == warning_count


def test_two_slices_of_one_frame() -> None:
    oda, _gni = join_compatible_frame_pair()

    result = run_sum_of(
        oda,
        oda,
        stat_var=_OWN_REF,
        left_entity={"reporter": "dcid:USA"},
        right_entity={"reporter": "dcid:GBR"},
    )

    assert [entry.value for entry in result.period_sums] == [
        55_303_000_000.0 + 15_694_000_000.0,
        63_575_000_000.0 + 19_073_000_000.0,
    ]
    assert result.facet_selections == ()
    assert result.warnings == ()


@pytest.mark.parametrize(
    ("left_rows", "right_rows", "error", "attributes"),
    [
        (
            [_row(year="2020", unit="dcid:USDollar")],
            [_row(year="2023", unit="dcid:USDollar")],
            ComparisonPeriodsDisjointError,
            {"left_periods": ("2020",), "right_periods": ("2023",)},
        ),
        (
            [_row(unit="dcid:USDollar")],
            [_row(unit="dcid:Percent")],
            CompareUnitMismatchError,
            {"left_unit": "dcid:USDollar", "right_unit": "dcid:Percent"},
        ),
        (
            [_row()],
            [_row(unit="dcid:USDollar")],
            CompareUnitUnboundError,
            {"unbound_side": "left", "bound_side": "right"},
        ),
        (
            [_row(value=1.7e308, unit="dcid:USDollar")],
            [_row(value=1.7e308, unit="dcid:USDollar")],
            CompareNonFiniteResultError,
            {"time_period": "2020", "quantity": "sum"},
        ),
        (
            [_row(unit="dcid:USDollar"), _row(stat_var="dcid:Y", unit="dcid:USDollar")],
            [_row(unit="dcid:USDollar")],
            CompareAmbiguousSliceError,
            {"side": "left"},
        ),
    ],
    ids=["periods disjoint", "unit mismatch", "unit unbound", "non-finite sum", "ambiguous"],
)
def test_refusals(
    left_rows: list[dict[str, object]],
    right_rows: list[dict[str, object]],
    error: type[Exception],
    attributes: dict[str, object],
) -> None:
    with pytest.raises(error) as raised:
        run_sum_of(
            _frame(left_rows),
            _frame(right_rows),
            stat_var=_OWN_REF,
            left_entity={},
            right_entity={},
        )
    for name, expected in attributes.items():
        assert getattr(raised.value, name) == expected


def test_two_sides_in_a_composed_unit_refuse() -> None:
    """Two per-head series agree on their unit, so the pairing's own unit
    rules pass and the sum is refused for what it would add."""
    per_head = {"numerator": "dcid:USDollar", "denominator": {"stat_var": "dcid:Count_Person"}}
    left = _frame([_row(value=2.0, unit=per_head)])
    right = _frame([_row(value=6.0, stat_var="dcid:Y", unit=per_head)])

    with pytest.raises(ComposedUnitSumError) as exc_info:
        run_sum_of(left, right, stat_var=_OWN_REF, left_entity={}, right_entity={})

    assert exc_info.value.side == "left"
    assert exc_info.value.unit == ComposedUnit(
        numerator="dcid:USDollar", denominator=StatVarPart("dcid:Count_Person")
    )
