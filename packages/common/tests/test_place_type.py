"""`common.place_type`: the levels of every place typed Country beside another type, and
whether several places sit at one level.

The cells are the `typeOf` values the development mixer holds for the places GDP
(`Amount_EconomicActivity_GrossDomesticProduction_Nominal`, 1,996 places) and female life
expectancy (`LifeExpectancy_Person_Female`, 1,885) have data at.
"""

from __future__ import annotations

import pytest
from common.place_type import (
    UNTYPED,
    at_one_level,
    enrichment_columns,
    enrichment_role,
    levels_of,
)

# Territories each sovereign state's own figures leave out: countries.
_TERRITORIES = {
    "country/ASM": "AdministrativeArea1,Country",
    "country/BMU": "AdministrativeArea1,Country",
    "country/CYM": "AdministrativeArea1,Country",
    "country/FRO": "AdministrativeArea1,Country",  # codespell:ignore
    "country/GIB": "AdministrativeArea1,Country",
    "country/GRL": "AdministrativeArea1,Country",
    "country/GUM": "AdministrativeArea1,Country",
    "country/MNP": "AdministrativeArea1,Country",
    "country/NCL": "AdministrativeArea1,Country",
    "country/TCA": "AdministrativeArea1,Country",
    "country/VGB": "AdministrativeArea1,Country",
    "country/HKG": "AdministrativeArea1,City,Country,SpecialAdministrativeRegion",
    "country/MAC": "AdministrativeArea1,City,Country,SpecialAdministrativeRegion",
    "country/MAF": "AdministrativeArea,Country",
    "country/PYF": "AdministrativeArea,Country",
    "country/VIR": "AdministrativeArea,Country",
    "country/IND": "Country,Place",
    "country/USA": "Country,Place",
}


@pytest.mark.parametrize("cell", sorted(set(_TERRITORIES.values())))
def test_a_place_typed_country_with_no_nuts_level_is_a_country(cell: str) -> None:
    assert "Country" in levels_of(cell)


def test_a_nuts_region_typed_country_is_not_one() -> None:
    # Åland: Finland's figures include it.
    assert levels_of("AdministrativeArea1,Country,EurostatNUTS1,Region") == {
        "AdministrativeArea1",
        "EurostatNUTS1",
        "Region",
    }


def test_a_place_sits_at_each_of_its_other_types() -> None:
    assert levels_of("AdministrativeArea1,AdministrativeArea2,City,District,EurostatNUTS2") == {
        "AdministrativeArea1",
        "AdministrativeArea2",
        "City",
        "District",
        "EurostatNUTS2",
    }
    assert levels_of(UNTYPED) == frozenset()


def _uncounted() -> int:
    raise AssertionError("the places need no counting")


def test_several_places_sit_at_one_level_when_they_share_a_place_level() -> None:
    assert at_one_level(["Country", *_TERRITORIES.values()], place_count=_uncounted)
    assert at_one_level(["Country,Place"], place_count=_uncounted)
    assert at_one_level([], place_count=_uncounted)
    assert not at_one_level(["Place", "Country,Place"], place_count=_uncounted)
    assert not at_one_level(
        ["Country", "AdministrativeArea1,Country,EurostatNUTS1,Region"], place_count=_uncounted
    )


@pytest.mark.parametrize("cell", ["Place", "ProvisionalNode", "Region", UNTYPED])
def test_one_place_sits_at_one_level_and_several_sharing_no_place_level_do_not(cell: str) -> None:
    assert at_one_level([cell], place_count=lambda: 1)
    assert not at_one_level([cell], place_count=lambda: 2)


def test_places_of_a_type_whose_places_nest_do_not_sit_at_one_level() -> None:
    # Data Commons types both Sub-Saharan Africa and the Eastern Africa it contains UNGeoRegion.
    assert not at_one_level(["UNGeoRegion"], place_count=lambda: 2)


@pytest.mark.parametrize(
    ("column", "role"),
    [
        ("observationAbout_type", "observationAbout"),
        ("financingRecipient_kind", "financingRecipient"),
        ("financingProvider_type", "financingProvider"),
        ("financingRecipient", None),
        ("flow_kind", None),
        ("financingRecipient_label", None),
    ],
)
def test_an_enrichment_column_names_the_role_it_describes(column: str, role: str | None) -> None:
    roles = ("observationAbout", "financingProvider", "financingRecipient")
    assert enrichment_role(column, roles=roles) == role
    if role is not None:
        assert column in enrichment_columns(role)
