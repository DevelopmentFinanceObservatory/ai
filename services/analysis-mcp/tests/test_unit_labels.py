"""Labels and series coordinates of a converted unit that states the prices which deflated it."""

from __future__ import annotations

import pytest
from analysis_mcp.rolling_tool import coordinate_text
from analysis_mcp.units import converted_unit_label, unit_name
from dfo_frame_grammar.grammar import ComposedUnit, DeflationBasis, StatVarPart, UnitRequest


def _naira(*, deflation_basis: DeflationBasis | None) -> UnitRequest:
    return UnitRequest(
        currency="NGN",
        prices="constant",
        base_year=2024,
        deflation_basis=deflation_basis,
    )


@pytest.mark.parametrize(
    ("deflation_basis", "label"),
    [
        (None, "converted to constant 2024 prices, NGN"),
        ("donor", "converted to constant 2024 prices, NGN, donor prices"),
        ("recipient", "converted to constant 2024 prices, NGN, recipient prices"),
    ],
)
def test_a_converted_unit_label_names_the_prices_that_deflated_it(
    deflation_basis: DeflationBasis | None, label: str
) -> None:
    """A DAC2a amount (donor prices) and an IDS amount (recipient prices) in
    constant 2024 naira read differently."""
    assert converted_unit_label(_naira(deflation_basis=deflation_basis)) == label


def test_the_converted_side_of_a_composed_unit_names_its_basis() -> None:
    unit = ComposedUnit(
        numerator=_naira(deflation_basis="donor"),
        denominator=StatVarPart(stat_var="dcid:Count_Person"),
    )

    assert unit_name(unit) == "constant 2024 prices, NGN, donor prices per Count_Person"


@pytest.mark.parametrize(
    ("unit", "text"),
    [
        (UnitRequest(currency="USD", prices="current"), "USD/current"),
        (UnitRequest(currency="NGN", prices="constant", base_year=2024), "NGN/constant/2024"),
        (_naira(deflation_basis="donor"), "NGN/constant/2024/donor"),
        (_naira(deflation_basis="recipient"), "NGN/constant/2024/recipient"),
    ],
)
def test_a_series_coordinate_names_a_converted_units_basis(unit: UnitRequest, text: str) -> None:
    """The spelling dfo-apis' assembler groups a converted unit under."""
    assert coordinate_text(unit) == text
