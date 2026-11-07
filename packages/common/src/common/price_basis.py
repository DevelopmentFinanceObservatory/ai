"""The plain-language price basis of a currency unit, such as "current prices,
US dollars" or "constant 2020 prices, euros".

data-mcp's `get_data` digest reads a recognised unit code this way, and
analysis-mcp's unit refusals label a converted unit the same way, so a model
meets one vocabulary for units across both services.
"""

from __future__ import annotations

# Currencies named in words. Any other currency is named by its code.
_CURRENCY_NAMES = {"USD": "US dollars", "EUR": "euros", "GBP": "pounds sterling"}


def price_basis_reading(
    *,
    currency: str,
    prices: str,
    base_year: int | str | None,
    deflation_basis: str | None = None,
) -> str:
    """The price basis, then `currency`'s name, joined by a comma. `prices`
    is `current` or `constant`, and a constant basis names `base_year` when
    it has one, then the `deflation_basis` (`donor`, `recipient` or
    `national`) whose prices deflated it. The frame grammar states a basis
    only at constant prices, so a current basis names none:
    "current prices, US dollars", "constant 2020 prices, euros",
    "constant prices, JPY", "constant 2024 prices, NGN, donor prices"."""
    if prices == "current":
        return f"current prices, {_CURRENCY_NAMES.get(currency, currency)}"
    basis = "constant prices" if base_year is None else f"constant {base_year} prices"
    reading = f"{basis}, {_CURRENCY_NAMES.get(currency, currency)}"
    return reading if deflation_basis is None else f"{reading}, {deflation_basis} prices"
