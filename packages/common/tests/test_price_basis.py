"""`price_basis_reading`: the price basis and currency name of a unit."""

from __future__ import annotations

import pytest
from common.price_basis import price_basis_reading


@pytest.mark.parametrize(
    ("currency", "prices", "base_year", "deflation_basis", "reading"),
    [
        ("USD", "current", None, None, "current prices, US dollars"),
        ("USD", "current", None, "donor", "current prices, US dollars"),
        ("EUR", "constant", 2020, None, "constant 2020 prices, euros"),
        ("GBP", "constant", "1999", None, "constant 1999 prices, pounds sterling"),
        ("JPY", "constant", None, None, "constant prices, JPY"),
        ("NGN", "constant", 2024, "donor", "constant 2024 prices, NGN, donor prices"),
        ("NGN", "constant", 2024, "recipient", "constant 2024 prices, NGN, recipient prices"),
        ("EUR", "constant", 2020, "national", "constant 2020 prices, euros, national prices"),
    ],
)
def test_price_basis_reading(
    currency: str,
    prices: str,
    base_year: int | str | None,
    deflation_basis: str | None,
    reading: str,
) -> None:
    assert (
        price_basis_reading(
            currency=currency,
            prices=prices,
            base_year=base_year,
            deflation_basis=deflation_basis,
        )
        == reading
    )
