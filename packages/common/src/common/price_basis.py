"""The plain-language price basis of a currency unit, such as "current prices,
US dollars" or "constant 2020 prices, euros".

data-mcp's `get_data` digest reads a recognised unit code this way, and
analysis-mcp's unit refusals label a converted unit the same way, so a model
meets one vocabulary for units across both services.
"""

from __future__ import annotations

# Currencies named in words. Any other currency is named by its code.
_CURRENCY_NAMES = {"USD": "US dollars", "EUR": "euros", "GBP": "pounds sterling"}


def price_basis_reading(*, currency: str, prices: str, base_year: int | str | None) -> str:
    """The price basis, then `currency`'s name, joined by a comma. `prices`
    is `current` or `constant`, and a constant basis names `base_year` when
    it has one: "current prices, US dollars", "constant 2020 prices, euros",
    "constant prices, JPY"."""
    if prices == "current":
        basis = "current prices"
    elif base_year is not None:
        basis = f"constant {base_year} prices"
    else:
        basis = "constant prices"
    return f"{basis}, {_CURRENCY_NAMES.get(currency, currency)}"
