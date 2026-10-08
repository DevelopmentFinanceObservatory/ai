"""`run_quotient_of`: the composed unit each pair of sides writes, the
intersect-and-record period policy, keyed pairing, the zero-denominator
exclusion, determinism, and every refusal condition.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import pytest
from analysis_mcp.engine.common import ExclusionRecord
from analysis_mcp.engine.quotient_of import (
    QuotientOfNoKeyMatchedError,
    QuotientOfPeriodsDisjointError,
    QuotientOfResult,
    QuotientOfSideUnitUnsupportedError,
    QuotientOfZeroDenominatorError,
    run_quotient_of,
)
from analysis_mcp.engine.two_sided import NonFiniteCellError
from dfo_frame_grammar import Frame, unit_cell_to_json
from dfo_frame_grammar.grammar import ComposedUnit, StatVarPart, UnitCell, UnitRequest

STAT_VAR = "dcid:ops_" + "a" * 32
GDP = "dcid:Amount_EconomicActivity_GrossDomesticProduction_Nominal"
POPULATION = "dcid:Count_Person"
AID = "dcid:OfficialDevelopmentAssistanceNet_Disbursement"
USD = "dcid:USDollar"
EUR_2020 = UnitRequest(currency="EUR", prices="constant", base_year=2020)


def _series(
    stat_var: str,
    values: dict[str, float],
    *,
    unit: UnitCell | None = None,
    place: str = "dcid:country/KEN",
    **extra: object,
) -> Frame:
    """One place's series: one row per period of `values`."""
    return Frame.build(
        role_keys=["observationAbout"],
        rows=[
            {
                "stat_var": stat_var,
                "observationAbout": place,
                "time_period": period,
                "value": value,
                "unit": unit_cell_to_json(unit),
                **extra,
            }
            for period, value in values.items()
        ],
    )


def _keyed(
    stat_var: str,
    column: str,
    values: dict[str, dict[str, float]],
    *,
    unit: UnitCell | None = None,
) -> Frame:
    """One series per value of `column`: `values` maps each to its periods."""
    return Frame.build(
        role_keys=[column],
        rows=[
            {
                "stat_var": stat_var,
                column: key,
                "time_period": period,
                "value": value,
                "unit": unit_cell_to_json(unit),
            }
            for key, by_period in values.items()
            for period, value in by_period.items()
        ],
    )


def _run(
    numerator: Frame,
    denominator: Frame,
    *,
    numerator_by: Sequence[str] = (),
    denominator_by: Mapping[str, str] | None = None,
) -> QuotientOfResult:
    return run_quotient_of(
        numerator,
        denominator,
        stat_var=STAT_VAR,
        numerator_entity={},
        denominator_entity={},
        numerator_by=numerator_by,
        denominator_by=denominator_by,
    )


def _rows(result: QuotientOfResult) -> list[dict[str, object]]:
    return [dict(zip(result.frame.columns, row, strict=True)) for row in result.frame.rows]


def _cells(frame: Frame, column: str) -> set[tuple[object, object]]:
    """The (key, period) cells `frame` holds under `column`."""
    key, period = frame.columns.index(column), frame.columns.index("time_period")
    return {(row[key], row[period]) for row in frame.rows}


def _records(result: QuotientOfResult, reason: str) -> dict[str | None, ExclusionRecord]:
    """The result's exclusion records under `reason`, by side."""
    records = [record for record in result.exclusions.records if record.reason == reason]
    by_side = {record.side: record for record in records}
    assert len(by_side) == len(records)
    return by_side


# Units -----------------------------------------------------------------------


def test_a_unit_over_a_unitless_count_writes_the_composed_unit_on_every_row() -> None:
    gdp = _series(GDP, {"2020": 100.0, "2021": 120.0, "2022": 150.0}, unit=USD)
    population = _series(POPULATION, {"2020": 50.0, "2021": 60.0, "2022": 50.0})

    result = _run(gdp, population)

    unit = ComposedUnit(numerator=USD, denominator=StatVarPart(POPULATION))
    assert result.unit == unit
    assert result.numerator_resolved_unit == USD
    assert result.denominator_resolved_unit is None
    assert (result.numerator_stat_var, result.denominator_stat_var) == (GDP, POPULATION)
    assert "observationAbout" not in result.frame.columns
    assert [
        (row["stat_var"], row["time_period"], row["value"], row["unit"]) for row in _rows(result)
    ] == [
        (STAT_VAR, "2020", 2.0, unit),
        (STAT_VAR, "2021", 2.0, unit),
        (STAT_VAR, "2022", 3.0, unit),
    ]
    assert [
        (entry.time_period, entry.numerator_value, entry.denominator_value, entry.quotient)
        for entry in result.period_quotients
    ] == [("2020", 100.0, 50.0, 2.0), ("2021", 120.0, 60.0, 2.0), ("2022", 150.0, 50.0, 3.0)]
    assert result.exclusions.records == ()


@pytest.mark.parametrize(
    ("numerator_unit", "denominator_unit", "denominator_stat_var", "expected"),
    [
        pytest.param(
            "dcid:CurrencyUSD",
            "dcid:CurrencyUSD",
            "dcid:Exports",
            ComposedUnit(numerator="dcid:CurrencyUSD", denominator="dcid:CurrencyUSD"),
            id="the-same-unit-on-both-sides",
        ),
        pytest.param(
            EUR_2020,
            None,
            POPULATION,
            ComposedUnit(numerator=EUR_2020, denominator=StatVarPart(POPULATION)),
            id="a-converted-numerator",
        ),
        pytest.param(
            None,
            "dcid:SquareKilometer",
            "dcid:Area_Land",
            ComposedUnit(numerator=StatVarPart(AID), denominator="dcid:SquareKilometer"),
            id="a-unitless-numerator",
        ),
        pytest.param(
            USD,
            None,
            "dcid:spec_" + "c" * 32,
            ComposedUnit(numerator=USD, denominator=StatVarPart("dcid:spec_" + "c" * 32)),
            id="a-unitless-summed-family-denominator",
        ),
    ],
)
def test_each_side_contributes_its_unit_or_the_stat_var_it_counts(
    numerator_unit: UnitCell | None,
    denominator_unit: UnitCell | None,
    denominator_stat_var: str,
    expected: ComposedUnit,
) -> None:
    result = _run(
        _series(AID, {"2020": 10.0}, unit=numerator_unit),
        _series(denominator_stat_var, {"2020": 4.0}, unit=denominator_unit),
    )

    assert result.unit == expected
    assert [(row["value"], row["unit"]) for row in _rows(result)] == [(2.5, expected)]


@pytest.mark.parametrize("side", ["numerator", "denominator"])
@pytest.mark.parametrize("places", [1, 2], ids=["one_row_per_cell", "pooled_rows"])
def test_a_side_whose_unit_is_already_composed_refuses(side: str, places: int) -> None:
    composed = ComposedUnit(numerator=USD, denominator=StatVarPart(POPULATION))
    quotient = _keyed(
        STAT_VAR,
        "observationAbout",
        {f"dcid:country/{index}": {"2020": 2.0} for index in range(places)},
        unit=composed,
    )
    plain = _series(POPULATION, {"2020": 4.0})
    numerator, denominator = (quotient, plain) if side == "numerator" else (plain, quotient)

    with pytest.raises(QuotientOfSideUnitUnsupportedError) as exc_info:
        _run(numerator, denominator)

    assert exc_info.value.side == side
    assert exc_info.value.reason == "composed"
    assert exc_info.value.unit == composed


def test_unitless_rows_on_a_unit_bearing_side_are_excluded_and_recorded() -> None:
    numerator = Frame.build(
        role_keys=["donor"],
        rows=[
            {"stat_var": AID, "donor": "dcid:A", "time_period": "2020", "value": 30.0, "unit": USD},
            {"stat_var": AID, "donor": "dcid:B", "time_period": "2020", "value": 70.0},
        ],
    )

    result = _run(numerator, _series(POPULATION, {"2020": 10.0}))

    assert result.unit == ComposedUnit(numerator=USD, denominator=StatVarPart(POPULATION))
    assert result.period_quotients[0].numerator_value == 30.0
    record = _records(result, "unit_unbound")["numerator"]
    assert record.row_count == 1
    assert record.entities == {"donor": ("dcid:B",)}


# Facets ----------------------------------------------------------------------


def test_sides_from_two_datasets_over_unequal_spans_divide_the_shared_periods() -> None:
    """The shape measured on dev: debt service 1970 to 2024 from one
    provenance over population 1960 to 2025 from another."""
    debt_service = _series(
        "dcid:DebtService",
        {str(year): float(year) for year in range(1970, 2025)},
        unit=USD,
        provenance="dcid:P1",
    )
    population = _series(
        POPULATION,
        {str(year): 10.0 for year in range(1960, 2026)},
        provenance="dcid:P2",
    )

    result = _run(debt_service, population)

    periods = tuple(str(year) for year in range(1970, 2025))
    assert tuple(row["time_period"] for row in _rows(result)) == periods
    assert result.matched_periods == periods
    assert result.numerator_periods == periods
    assert len(result.denominator_periods) == 66
    assert result.numerator_coverage_fraction == 1.0
    assert result.denominator_coverage_fraction == 55 / 66

    unmatched = _records(result, "period_unmatched")
    assert set(unmatched) == {"denominator"}
    assert unmatched["denominator"].row_count == 11

    assert result.facet_selections == ()
    assert [selection.ref for selection in result.numerator_facet_selections] == ["dcid:P1"]
    assert [selection.ref for selection in result.denominator_facet_selections] == ["dcid:P2"]
    assert {row["provenance"] for row in _rows(result)} == {None}
    (warning,) = result.warnings
    assert "'provenance'" in warning
    assert "dcid:P1" in warning
    assert "dcid:P2" in warning


def test_a_facet_both_sides_agree_on_is_carried_and_a_one_sided_one_warns() -> None:
    numerator = _series(AID, {"2020": 10.0}, provenance="dcid:P1", measurementMethod="dcid:M1")
    denominator = _series(POPULATION, {"2020": 5.0}, provenance="dcid:P1")

    result = _run(numerator, denominator)

    assert [(s.axis, s.ref) for s in result.facet_selections] == [("provenance", "dcid:P1")]
    (row,) = _rows(result)
    assert (row["provenance"], row["measurementMethod"]) == ("dcid:P1", None)
    (warning,) = result.warnings
    assert "'measurementMethod'" in warning
    assert "numerator" in warning


# Periods ---------------------------------------------------------------------


def test_sides_sharing_no_period_refuse_with_both_sides_periods() -> None:
    with pytest.raises(QuotientOfPeriodsDisjointError) as exc_info:
        _run(
            _series(GDP, {"2020": 1.0, "2021": 2.0}),
            _series(POPULATION, {"2022": 1.0, "2023": 2.0}),
        )

    assert exc_info.value.numerator_periods == ("2020", "2021")
    assert exc_info.value.denominator_periods == ("2022", "2023")


def test_an_entity_binding_matching_no_rows_refuses_as_disjoint_periods() -> None:
    with pytest.raises(QuotientOfPeriodsDisjointError) as exc_info:
        run_quotient_of(
            _series(GDP, {"2020": 1.0}),
            _series(POPULATION, {"2020": 1.0}),
            stat_var=STAT_VAR,
            numerator_entity={"observationAbout": "dcid:country/NOWHERE"},
            denominator_entity={},
        )

    assert exc_info.value.numerator_periods == ()
    assert exc_info.value.denominator_periods == ("2020",)


def test_a_keyed_numerator_with_no_rows_refuses_as_disjoint_periods() -> None:
    aid = Frame.build(
        role_keys=["donor", "recipient"],
        rows=[
            {
                "stat_var": AID,
                "donor": "dcid:country/FRA",
                "recipient": "dcid:country/KEN",
                "time_period": "2020",
                "value": 10.0,
                "unit": USD,
            }
        ],
    )
    population = _keyed(
        POPULATION,
        "observationAbout",
        {"dcid:country/FRA": {"2020": 1.0}, "dcid:country/KEN": {"2021": 2.0}},
    )

    with pytest.raises(QuotientOfPeriodsDisjointError) as exc_info:
        run_quotient_of(
            aid,
            population,
            stat_var=STAT_VAR,
            numerator_entity={"recipient": "dcid:country/NOWHERE"},
            denominator_entity={},
            numerator_by=["donor"],
            denominator_by={"donor": "observationAbout"},
        )

    assert exc_info.value.numerator_periods == ()
    assert exc_info.value.denominator_periods == ("2020", "2021")


def test_a_keyed_denominator_with_no_rows_refuses_as_disjoint_periods() -> None:
    aid = _keyed(AID, "recipient", {"dcid:country/KEN": {"2020": 1.0, "2021": 2.0}})
    population = _keyed(POPULATION, "observationAbout", {"dcid:country/KEN": {"2020": 1.0}})

    with pytest.raises(QuotientOfPeriodsDisjointError) as exc_info:
        run_quotient_of(
            aid,
            population,
            stat_var=STAT_VAR,
            numerator_entity={},
            denominator_entity={"observationAbout": "dcid:country/NOWHERE"},
            numerator_by=["recipient"],
            denominator_by={"recipient": "observationAbout"},
        )

    assert exc_info.value.numerator_periods == ("2020", "2021")
    assert exc_info.value.denominator_periods == ()


def test_a_period_one_side_lacks_is_recorded_with_its_side() -> None:
    result = _run(
        _series(GDP, {"2019": 1.0, "2020": 6.0}),
        _series(POPULATION, {"2020": 3.0, "2021": 1.0, "2022": 1.0}),
    )

    assert [(e.time_period, e.quotient) for e in result.period_quotients] == [("2020", 2.0)]
    unmatched = _records(result, "period_unmatched")
    assert unmatched["numerator"].row_count == 1
    assert unmatched["denominator"].row_count == 2
    assert result.numerator_coverage_fraction == 0.5
    assert result.denominator_coverage_fraction == 1 / 3


# Zero denominator ------------------------------------------------------------


def test_a_zero_denominator_cell_is_excluded_and_recorded() -> None:
    result = _run(
        _series(GDP, {"2020": 10.0, "2021": 20.0, "2022": 30.0}),
        _series(POPULATION, {"2020": 5.0, "2021": 0.0, "2022": 5.0}),
    )

    assert [(e.time_period, e.quotient) for e in result.period_quotients] == [
        ("2020", 2.0),
        ("2022", 6.0),
    ]
    assert [(cell.time_period, cell.numerator_value) for cell in result.zero_denominator_cells] == [
        ("2021", 20.0)
    ]
    zero = _records(result, "zero_denominator")
    assert zero["numerator"].row_count == 1
    assert zero["denominator"].row_count == 1
    # A zero-denominator period is matched, so it is no `period_unmatched`.
    assert result.matched_periods == ("2020", "2021", "2022")
    assert _records(result, "period_unmatched") == {}


def test_denominator_rows_that_sum_to_zero_are_excluded_as_zero() -> None:
    denominator = _keyed(
        POPULATION, "donor", {"dcid:A": {"2020": 5.0, "2021": 4.0}, "dcid:B": {"2020": -5.0}}
    )

    result = _run(_series(GDP, {"2020": 10.0, "2021": 8.0}), denominator)

    assert [(e.time_period, e.quotient) for e in result.period_quotients] == [("2021", 2.0)]
    assert _records(result, "zero_denominator")["denominator"].row_count == 2


def test_every_matched_cell_having_a_zero_denominator_refuses() -> None:
    with pytest.raises(QuotientOfZeroDenominatorError) as exc_info:
        _run(
            _series(GDP, {"2019": 1.0, "2020": 10.0, "2021": 20.0}),
            _series(POPULATION, {"2020": 0.0, "2021": 0.0}),
        )

    assert exc_info.value.time_periods == ("2020", "2021")
    assert exc_info.value.cell_count == 2


# Keys ------------------------------------------------------------------------


def _aid_by_recipient() -> Frame:
    return _keyed(
        AID,
        "recipient",
        {
            "dcid:country/KEN": {"2020": 100.0, "2021": 150.0},
            "dcid:country/UGA": {"2020": 60.0, "2021": 90.0},
        },
        unit=USD,
    )


def test_a_keyed_call_divides_each_key_by_its_own_denominator_group() -> None:
    population = _keyed(
        POPULATION,
        "observationAbout",
        {
            "dcid:country/UGA": {"2020": 30.0, "2021": 30.0},
            "dcid:country/KEN": {"2020": 50.0, "2021": 50.0},
        },
    )

    result = _run(
        _aid_by_recipient(),
        population,
        numerator_by=["recipient"],
        denominator_by={"recipient": "observationAbout"},
    )

    assert "recipient" in result.frame.columns
    assert "observationAbout" not in result.frame.columns
    assert [(row["recipient"], row["time_period"], row["value"]) for row in _rows(result)] == [
        ("dcid:country/KEN", "2020", 2.0),
        ("dcid:country/KEN", "2021", 3.0),
        ("dcid:country/UGA", "2020", 2.0),
        ("dcid:country/UGA", "2021", 3.0),
    ]
    assert result.period_quotients[0].key == (("recipient", "dcid:country/KEN"),)
    assert result.unmatched_keys == ()
    assert result.exclusions.records == ()


def test_a_numerator_key_with_no_denominator_group_is_dropped_and_recorded() -> None:
    population = _keyed(POPULATION, "observationAbout", {"dcid:country/KEN": {"2020": 50.0}})
    aid = _keyed(
        AID,
        "recipient",
        {"dcid:country/KEN": {"2020": 100.0}, "dcid:country/UGA": {"2020": 60.0, "2021": 90.0}},
    )

    result = _run(
        aid,
        population,
        numerator_by=["recipient"],
        denominator_by={"recipient": "observationAbout"},
    )

    assert [(row["recipient"], row["value"]) for row in _rows(result)] == [
        ("dcid:country/KEN", 2.0)
    ]
    assert result.unmatched_keys == ((("recipient", "dcid:country/UGA"),),)
    record = _records(result, "unmatched_key")["numerator"]
    assert record.row_count == 2
    assert record.entities == {"recipient": ("dcid:country/UGA",)}
    # The dropped key's periods are no part of either side's coverage.
    assert result.numerator_periods == ("2020",)
    assert _records(result, "period_unmatched") == {}


def test_no_numerator_key_having_a_denominator_group_refuses_with_sample_keys() -> None:
    population = _keyed(
        POPULATION,
        "observationAbout",
        {f"country/{index:02d}": {"2020": 1.0} for index in range(25)},
    )

    with pytest.raises(QuotientOfNoKeyMatchedError) as exc_info:
        _run(
            _aid_by_recipient(),
            population,
            numerator_by=["recipient"],
            denominator_by={"recipient": "observationAbout"},
        )

    assert exc_info.value.numerator_keys == (
        (("recipient", "dcid:country/KEN"),),
        (("recipient", "dcid:country/UGA"),),
    )
    assert exc_info.value.numerator_key_count == 2
    assert exc_info.value.denominator_keys == tuple(
        (("observationAbout", f"country/{index:02d}"),) for index in range(20)
    )
    assert exc_info.value.denominator_key_count == 25


def test_a_keyed_numerator_with_no_denominator_by_divides_every_key_by_the_one_total() -> None:
    aid = _keyed(
        AID,
        "donor",
        {"dcid:A": {"2020": 100.0, "2021": 30.0}, "dcid:B": {"2020": 50.0}},
        unit=USD,
    )

    result = _run(aid, _series(POPULATION, {"2020": 50.0, "2021": 60.0}), numerator_by=["donor"])

    assert [(row["donor"], row["time_period"], row["value"]) for row in _rows(result)] == [
        ("dcid:A", "2020", 2.0),
        ("dcid:A", "2021", 0.5),
        ("dcid:B", "2020", 1.0),
    ]
    assert {entry.denominator_value for entry in result.period_quotients} == {50.0, 60.0}
    # The pooled 2021 total serves donor A, so donor B lacking 2021 leaves
    # nothing unmatched on the denominator's side.
    assert _records(result, "period_unmatched") == {}


def test_a_key_sharing_no_period_with_its_group_is_dropped_while_others_compute() -> None:
    aid = _keyed(
        AID,
        "recipient",
        {"dcid:country/KEN": {"2020": 100.0}, "dcid:country/UGA": {"2018": 60.0, "2019": 90.0}},
    )
    population = _keyed(
        POPULATION,
        "observationAbout",
        {"dcid:country/KEN": {"2020": 50.0}, "dcid:country/UGA": {"2020": 30.0}},
    )

    result = _run(
        aid,
        population,
        numerator_by=["recipient"],
        denominator_by={"recipient": "observationAbout"},
    )

    assert [(row["recipient"], row["value"]) for row in _rows(result)] == [
        ("dcid:country/KEN", 2.0)
    ]
    assert result.keys_without_shared_period == ((("recipient", "dcid:country/UGA"),),)
    assert result.unmatched_keys == ()
    unmatched = _records(result, "period_unmatched")
    assert unmatched["numerator"].row_count == 2
    assert unmatched["numerator"].entities == {"recipient": ("dcid:country/UGA",)}
    assert unmatched["denominator"].row_count == 1
    assert unmatched["denominator"].entities == {"observationAbout": ("dcid:country/UGA",)}


def test_a_key_whose_every_matched_cell_has_a_zero_denominator_is_a_dropped_key() -> None:
    """Tanzania pairs and shares its period, and its one cell has a zero
    denominator, so it has no row. Kenya loses one cell of two and keeps its
    key."""
    aid = _keyed(
        AID,
        "recipient",
        {
            "dcid:country/KEN": {"2021": 80.0, "2022": 100.0},
            "dcid:country/TCD": {"2019": 30.0},
            "dcid:country/TZA": {"2022": 9.0},
            "dcid:country/UGA": {"2022": 5.0},
        },
    )
    population = _keyed(
        POPULATION,
        "observationAbout",
        {
            "dcid:country/KEN": {"2021": 0.0, "2022": 50.0},
            "dcid:country/TCD": {"2022": 16.0},
            "dcid:country/TZA": {"2022": 0.0},
        },
    )

    result = _run(
        aid,
        population,
        numerator_by=["recipient"],
        denominator_by={"recipient": "observationAbout"},
    )

    assert [(row["recipient"], row["value"]) for row in _rows(result)] == [
        ("dcid:country/KEN", 2.0)
    ]
    assert result.unmatched_keys == ((("recipient", "dcid:country/UGA"),),)
    assert result.keys_without_shared_period == ((("recipient", "dcid:country/TCD"),),)
    assert result.keys_with_zero_denominator == ((("recipient", "dcid:country/TZA"),),)
    assert len(result.zero_denominator_cells) == 2
    numerator_keys = {key for key, _period in _cells(aid, "recipient")}
    keys_with_a_row = {row["recipient"] for row in _rows(result)}
    assert result.dropped_key_count == len(numerator_keys) - len(keys_with_a_row) == 3


def test_every_key_having_a_zero_denominator_refuses() -> None:
    aid = _keyed(
        AID, "recipient", {"dcid:country/KEN": {"2020": 1.0}, "dcid:country/TZA": {"2021": 2.0}}
    )
    population = _keyed(
        POPULATION,
        "observationAbout",
        {"dcid:country/KEN": {"2020": 0.0}, "dcid:country/TZA": {"2021": 0.0}},
    )

    with pytest.raises(QuotientOfZeroDenominatorError) as exc_info:
        _run(
            aid,
            population,
            numerator_by=["recipient"],
            denominator_by={"recipient": "observationAbout"},
        )

    assert exc_info.value.time_periods == ("2020", "2021")
    assert exc_info.value.cell_count == 2


@pytest.mark.parametrize(
    ("population", "denominator_by", "expected"),
    [
        pytest.param(
            _keyed(POPULATION, "observationAbout", {"A": {"2020": 4.0}, "B": {"2021": 5.0}}),
            {"recipient": "observationAbout"},
            (0.5, 1.0),
            id="paired",
        ),
        pytest.param(
            _keyed(
                POPULATION,
                "observationAbout",
                {"A": {"2020": 4.0, "2022": 4.0}, "B": {"2021": 5.0, "2023": 5.0}},
            ),
            {"recipient": "observationAbout"},
            (0.5, 0.5),
            id="paired_both_sides_short",
        ),
        pytest.param(
            _series(POPULATION, {"2020": 4.0, "2019": 4.0}),
            None,
            (0.5, 0.5),
            id="pooled",
        ),
    ],
)
def test_keyed_coverage_is_the_share_of_cells_each_side_matched(
    population: Frame, denominator_by: dict[str, str] | None, expected: tuple[float, float]
) -> None:
    """Both numerator keys hold 2020 and 2021. A period one key lacks a
    denominator for is uncovered there, whichever other key matches it, and
    a pooled group's period counts once however many keys divide by it."""
    aid = _keyed(
        AID, "recipient", {"A": {"2020": 1.0, "2021": 2.0}, "B": {"2020": 3.0, "2021": 4.0}}
    )

    result = _run(aid, population, numerator_by=["recipient"], denominator_by=denominator_by)

    assert len(result.period_quotients) == 2
    assert (
        result.numerator_coverage_fraction,
        result.denominator_coverage_fraction,
    ) == expected


def test_periods_one_side_alone_holds_are_named_under_each_key_that_has_rows() -> None:
    """El Salvador's GDP starts two years after its population and Honduras
    has a GDP year its population lacks. Guatemala matches throughout, and
    Chad, which shares no period, is a dropped key and is named as one."""
    gdp = _keyed(
        GDP,
        "recipient",
        {
            "GTM": {"1960": 1.0, "1961": 1.0, "1962": 1.0},
            "HND": {"1960": 1.0, "1961": 1.0, "1962": 1.0},
            "SLV": {"1962": 1.0},
            "TCD": {"1950": 1.0},
        },
    )
    population = _keyed(
        POPULATION,
        "observationAbout",
        {
            "GTM": {"1960": 2.0, "1961": 2.0, "1962": 2.0},
            "HND": {"1960": 2.0, "1962": 2.0},
            "SLV": {"1960": 2.0, "1961": 2.0, "1962": 2.0},
            "TCD": {"1960": 2.0},
        },
    )
    keyed = {"numerator_by": ["recipient"], "denominator_by": {"recipient": "observationAbout"}}

    result = _run(gdp, population, **keyed)

    assert result.numerator_only_periods == (((("recipient", "HND"),), ("1961",)),)
    assert result.denominator_only_periods == (((("recipient", "SLV"),), ("1960", "1961")),)
    assert result.keys_without_shared_period == ((("recipient", "TCD"),),)

    pooled = _run(gdp, _series(POPULATION, {"1962": 2.0, "1963": 2.0}), numerator_by=["recipient"])

    assert [key for key, _periods in pooled.numerator_only_periods] == [
        (("recipient", "GTM"),),
        (("recipient", "HND"),),
    ]
    assert pooled.denominator_only_periods == ()

    un_keyed = _run(_series(GDP, {"1960": 1.0, "1961": 1.0}), _series(POPULATION, {"1960": 2.0}))

    assert un_keyed.numerator_only_periods == ()


# Determinism and non-finite results --------------------------------------------


def test_the_result_is_independent_of_input_row_order() -> None:
    def rows(stat_var: str, column: str, scale: float) -> list[dict[str, object]]:
        return [
            {
                "stat_var": stat_var,
                column: f"dcid:country/{place}",
                "time_period": str(year),
                "value": scale * (index + 1) * (year - 2000),
            }
            for index, place in enumerate(("KEN", "UGA", "TZA"))
            for year in range(2015, 2021)
        ]

    def run(seed: int | None) -> QuotientOfResult:
        def side(stat_var: str, column: str, scale: float) -> Frame:
            side_rows = rows(stat_var, column, scale)
            if seed is not None:
                random.Random(seed).shuffle(side_rows)
            return Frame.build(role_keys=[column], rows=side_rows)

        return _run(
            side(AID, "recipient", 6.0),
            side(POPULATION, "observationAbout", 3.0),
            numerator_by=["recipient"],
            denominator_by={"recipient": "observationAbout"},
        )

    baseline = run(None)
    shuffled = run(7)

    assert shuffled.frame.rows == baseline.frame.rows
    assert shuffled.period_quotients == baseline.period_quotients
    assert [(e.key[0][1], e.time_period) for e in baseline.period_quotients] == sorted(
        (e.key[0][1], e.time_period) for e in baseline.period_quotients
    )


def test_a_quotient_that_overflows_refuses_non_finite() -> None:
    with pytest.raises(NonFiniteCellError) as exc_info:
        _run(_series(GDP, {"2020": 1.0e308}), _series(POPULATION, {"2020": 1.0e-10}))

    assert exc_info.value.time_period == "2020"
    assert exc_info.value.reason == "overflow"


def test_a_side_whose_sum_overflows_refuses_non_finite() -> None:
    numerator = _keyed(GDP, "donor", {"dcid:A": {"2020": 1.0e308}, "dcid:B": {"2020": 1.0e308}})

    with pytest.raises(NonFiniteCellError) as exc_info:
        _run(numerator, _series(POPULATION, {"2020": 100.0}))

    assert exc_info.value.reason == "overflow"


def test_a_negative_zero_quotient_is_normalised() -> None:
    result = _run(_series(GDP, {"2020": 0.0}), _series(POPULATION, {"2020": -5.0}))

    assert str(result.period_quotients[0].quotient) == "0.0"
