"""`price_basis_reading`: the price basis and currency name of a unit."""

from __future__ import annotations

import pytest
from common.price_basis import price_basis_reading


@pytest.mark.parametrize(
    ("currency", "prices", "base_year", "reading"),
    [
        ("USD", "current", None, "current prices, US dollars"),
        ("EUR", "constant", 2020, "constant 2020 prices, euros"),
        ("GBP", "constant", "1999", "constant 1999 prices, pounds sterling"),
        ("JPY", "constant", None, "constant prices, JPY"),
    ],
)
def test_price_basis_reading(
    currency: str, prices: str, base_year: int | str | None, reading: str
) -> None:
    assert price_basis_reading(currency=currency, prices=prices, base_year=base_year) == reading
