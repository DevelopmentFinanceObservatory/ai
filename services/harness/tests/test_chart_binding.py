"""The ranking that cuts a select over the series cap
(`harness.chart_binding.top_members`) and the members it compares
(`comparable_members`)."""

from __future__ import annotations

from typing import Any

import pytest
from dfo_frame_grammar.grammar import UnitRequest
from dfo_frame_grammar.rolling import RollingError
from harness.chart_binding import TopMembers, comparable_members, top_members

KENYA = {"financingRecipient": "country/KEN"}


def _provider_rows(
    values: dict[str, dict[str, float | None]], **extra: Any
) -> list[dict[str, Any]]:
    """Frame rows of aid to Kenya, one per provider and period."""
    return [
        {
            "stat_var": "dcid:aid",
            "financingProvider": provider,
            "financingRecipient": "country/KEN",
            "time_period": period,
            "value": value,
            "unit": "USDollar",
            "provenance": "crs",
            **extra,
        }
        for provider, by_period in values.items()
        for period, value in by_period.items()
    ]


# Ten providers. A and B are the largest in 2023, the one year at least half
# of them hold. C and D are the largest in 2024, and E is above D in 2025.
TEN_PROVIDERS: dict[str, dict[str, float | None]] = {
    "A": {"2023": 100.0, "2024": 1.0},
    "B": {"2023": 90.0, "2024": 2.0},
    "C": {"2023": 30.0, "2024": 80.0},
    "D": {"2023": 20.0, "2024": 70.0, "2025": 50.0},
    "E": {"2023": 10.0, "2025": 60.0},
    **{member: {"2023": float(value)} for value, member in enumerate("FGHIJ", start=1)},
}


def test_top_members_ranks_at_the_period_half_the_members_hold() -> None:
    """The members are ranked at the anchor of the ten, 2023, where the
    newest period any holds (2025) and the newest the two largest share
    (2024) each name other members."""
    top = top_members(
        rows=_provider_rows(TEN_PROVIDERS), select=KENYA, dimension="financingProvider", limit=2
    )

    assert top == TopMembers(
        members=["A", "B"],
        period="2023",
        holding=10,
        unvalued=[],
        other_kinds=0,
    )


@pytest.mark.parametrize(
    ("select", "cells"),
    [
        ({"financingProvider": ["A", "B", "C", "D"]}, None),
        (
            {},
            {
                **dict.fromkeys("ABCD", "party"),
                **dict.fromkeys(("T1", "T2", "T3", "T4", "T5", "T6"), "total"),
            },
        ),
    ],
    ids=["outside_the_select", "group_totals"],
)
def test_top_members_anchors_on_the_compared_members_alone(
    select: dict[str, Any], cells: dict[str, str] | None
) -> None:
    """One of the four compared members reports 2025, so they are ranked at
    2024. The frame's six other members all hold 2025: counted, seven of ten
    would hold it and the anchor would move there."""
    rows = _provider_rows(
        {
            "A": {"2024": 4.0, "2025": 9.0},
            "B": {"2024": 3.0},
            "C": {"2024": 2.0},
            "D": {"2023": 8.0},
            **{f"T{number}": {"2024": 50.0, "2025": 60.0} for number in range(1, 7)},
        }
    )

    top = top_members(rows=rows, select=select, dimension="financingProvider", limit=2, cells=cells)

    assert top is not None
    assert (top.members, top.period) == (["A", "B"], "2024")
    # D holds no 2024 value, so it is left out of the ranking and of the count of three.
    assert (top.holding, top.unvalued) == (3, ["D"])
    assert top.other_kinds == (0 if cells is None else 6)


def test_top_members_counts_a_row_with_no_value_as_not_held() -> None:
    rows = _provider_rows({"A": {"2024": 1.0}, "B": {"2024": 2.0}, "C": {"2024": None}})

    top = top_members(rows=rows, select=KENYA, dimension="financingProvider", limit=None)

    assert top == TopMembers(
        members=["B", "A"],
        period="2024",
        holding=2,
        unvalued=["C"],
        other_kinds=0,
    )


@pytest.mark.parametrize(
    "values",
    [
        {"A": {"2022": 1.0}, "B": {"2023": 2.0}, "C": {"2024": 3.0}},
        {"A": {"2024": 1.0}, "B": {"2024-06": 2.0}},
        {"A": {"2015/2019": 1.0}, "B": {"2020/2024": 2.0}},
    ],
    ids=["no_period_half_hold", "mixed_grains", "two_spans"],
)
def test_top_members_raises_when_the_members_have_no_anchor(
    values: dict[str, dict[str, float | None]],
) -> None:
    with pytest.raises(RollingError):
        top_members(
            rows=_provider_rows(values), select=KENYA, dimension="financingProvider", limit=2
        )


@pytest.mark.parametrize("period", ["2020/2024", "2024", "2024-06"])
def test_top_members_ranks_at_the_one_period_the_members_hold(period: str) -> None:
    """Members holding one period between them, a span included, are ranked
    at it, the only period they hold."""
    rows = _provider_rows({"A": {period: 1.0}, "B": {period: 2.0}, "C": {period: 3.0}})

    top = top_members(rows=rows, select=KENYA, dimension="financingProvider", limit=2)

    assert top is not None
    assert (top.members, top.period, top.holding) == (["C", "B"], period, 3)


def test_top_members_ranks_no_row_without_a_member() -> None:
    """A row with no member names nothing a select could list."""
    rows = _provider_rows({"A": {"2024": 1.0}, "B": {"2024": 2.0}})
    rows[1]["financingProvider"] = None

    top = top_members(rows=rows, select=KENYA, dimension="financingProvider", limit=5)

    assert top is not None
    assert (top.members, top.holding) == (["A"], 1)


def test_top_members_matches_a_converted_unit_as_the_select_names_it() -> None:
    """A stored frame holds a converted unit as a `UnitRequest`. A select
    names it as the apis service's view does, `currency/prices/base_year`."""
    unit = UnitRequest(currency="USD", prices="constant", base_year=2023)
    rows = _provider_rows({"A": {"2024": 1.0}, "B": {"2024": 2.0}}, unit=unit)

    top = top_members(
        rows=rows, select={"unit": "USD/constant/2023"}, dimension="financingProvider", limit=1
    )

    assert top is not None
    assert (top.members, top.holding) == (["B"], 2)


@pytest.mark.parametrize(
    "other",
    [
        {"unit": "EUR"},
        {"unit": UnitRequest(currency="USD", prices="current")},
        {"provenance": "other"},
        {"measurementMethod": "imputed"},
        {"stat_var": "dcid:other"},
    ],
)
def test_top_members_ranks_nothing_across_unlike_values(other: dict[str, Any]) -> None:
    """Rows that differ on any other coordinate, a unit, a source, a method
    or a measure, are not one ranking, so nothing is picked, even when each
    provider appears under one of them only."""
    rows = _provider_rows({"A": {"2024": 1.0}, "B": {"2024": 2.0}})
    rows += _provider_rows({"C": {"2024": 3.0}}, **other)

    assert top_members(rows=rows, select=KENYA, dimension="financingProvider", limit=2) is None


def test_comparable_members_keeps_the_parties_of_a_flow_role() -> None:
    """A group total such as the G7, a residual and a party with no kind are
    left out of a ranking of donors. A private donor, of kind `party` with
    its class in a column of its own, ranks beside the others."""
    cells = {
        "DACOfficialDonors": "total",
        "GroupOfSeven": "total",
        "AfricaRegional": "residual",
        "Unknown": "untyped",
        "Germany": "party",
        "Japan": "party",
        "GatesFoundation": "party",
    }

    assert comparable_members(cells, dimension="financingProvider") == {
        "Germany",
        "Japan",
        "GatesFoundation",
    }


def test_comparable_members_keeps_the_places_at_the_level_most_share() -> None:
    """Among countries, a World Bank aggregate, a UN region, a continent and
    an untyped place are left out. Hong Kong, typed Country among others,
    stays."""
    cells = {
        "country/KEN": "Country",
        "country/NGA": "Country",
        "country/HKG": "AdministrativeArea1,Country",
        "wb/LMY": "ProvisionalNode",
        "undata-geo/G00201000": "UNGeoRegion",
        "africa": "Continent",
        "ida/total": "untyped",
    }

    assert comparable_members(cells, dimension="observationAbout") == {
        "country/KEN",
        "country/NGA",
        "country/HKG",
    }
    assert comparable_members({"wb/LMY": "ProvisionalNode"}, dimension="observationAbout") == set()
