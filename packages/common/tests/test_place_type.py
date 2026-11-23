"""`common.place_type`: the levels of every place typed Country beside another type.

The cells are the `typeOf` values the development mixer holds for the places GDP
(`Amount_EconomicActivity_GrossDomesticProduction_Nominal`, 1,996 places) and female life
expectancy (`LifeExpectancy_Person_Female`, 1,885) have data at.
"""

from __future__ import annotations

import pytest
from common.place_type import (
    PARTY_KINDS,
    UNTYPED,
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


@pytest.mark.parametrize(
    ("column", "role"),
    [
        ("observationAbout_type", "observationAbout"),
        ("financingRecipient_party_kind", "financingRecipient"),
        ("financingProvider_party_class", "financingProvider"),
        ("financingProvider_party_whole", "financingProvider"),
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


@pytest.mark.parametrize(
    ("cell", "accepted"),
    [
        ("party", True),
        ("residual", True),
        ("total", True),
        ("private", False),
        ("whole_total", False),
        (UNTYPED, False),
    ],
)
def test_a_party_kind_cell_names_one_of_three_kinds(cell: str, accepted: bool) -> None:
    assert (cell in PARTY_KINDS) is accepted
