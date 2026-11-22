"""Tests for `harness.unit_offer`: the units a unit question offers, read off Search's blocks."""

from __future__ import annotations

from typing import Any

import pytest
from harness.unit_offer import served_unit, unit_offer, unit_words, whole_units
from harness.wire import UnitChoice, UnitCurrency, UnitOffer, UnitPrices


def _currency(code: str, noun: str, *, low: int = 2000, high: int = 2023) -> dict[str, Any]:
    years = {"start_year": 2000, "end_year": 2023}
    return {
        "code": code,
        "noun": noun,
        "current_years": years,
        "constant_years": years,
        "base_year_min": low,
        "base_year_max": high,
    }


def _block(*currencies: dict[str, Any], **conversion: Any) -> dict[str, Any]:
    return {"resolved": {"unit_conversion": {"currencies": list(currencies), **conversion}}}


def test_the_offer_lists_us_dollars_then_each_role_s_currency_and_starts_on_constant() -> None:
    block = _block(
        _currency("EUR", "euros"),
        _currency("KES", "Kenyan shillings"),
        _currency("USD", "US dollars"),
        default_base_year=2030,
        role_currencies={"origin": "EUR", "destination": "KES"},
    )

    offer = unit_offer([block])

    assert offer is not None
    assert offer.presets == ["USD", "EUR", "KES"]
    # The declared base year lies past USD's range, so the start clamps to its end.
    assert offer.start == UnitChoice(currency="USD", prices=UnitPrices.CONSTANT, base_year=2023)


def test_a_preferred_basis_the_currency_serves_is_the_start() -> None:
    offer = unit_offer([_block(_currency("USD", "US dollars"))], prefer="current")

    assert offer is not None
    assert offer.start == UnitChoice(currency="USD", prices=UnitPrices.CURRENT)


@pytest.mark.parametrize(
    ("other", "shared"),
    [
        (_currency("USD", "US dollars", low=2010, high=2020), (2010, 2020)),
        (_currency("USD", "US dollars", low=2024, high=2025), None),
        (_currency("EUR", "euros"), None),
    ],
    ids=["narrower-range", "disjoint-range", "other-currency"],
)
def test_several_blocks_offer_what_every_one_serves(
    other: dict[str, Any], shared: tuple[int, int] | None
) -> None:
    offer = unit_offer([_block(_currency("USD", "US dollars")), _block(other)])

    if shared is None:
        assert offer is None
        return
    assert offer is not None
    ((currency),) = offer.currencies
    assert (currency.base_year_min, currency.base_year_max) == shared


@pytest.mark.parametrize(
    ("unit", "words"),
    [
        (
            UnitChoice(currency="USD", prices=UnitPrices.CONSTANT, base_year=2023),
            "Constant 2023 US dollars",
        ),
        (UnitChoice(currency="EUR", prices=UnitPrices.CURRENT), "Current EUR"),
    ],
)
def test_a_unit_reads_in_the_offer_s_words_or_by_its_code(unit: UnitChoice, words: str) -> None:
    offer = unit_offer([_block(_currency("USD", "US dollars"))])

    assert unit_words(unit, offer) == words


def test_a_currency_with_an_inverted_base_year_range_is_left_out() -> None:
    block = _block(_currency("EUR", "euros", low=2023, high=2000), _currency("USD", "US dollars"))

    offer = unit_offer([block])

    assert offer is not None
    assert [currency.code for currency in offer.currencies] == ["USD"]


def test_a_constant_request_without_a_base_year_reads_as_no_served_unit() -> None:
    block = {
        "definition": {
            "identity": {"facets": {"unit": {"request": {"currency": "USD", "prices": "constant"}}}}
        }
    }

    assert served_unit(block) is None


_EUR_CURRENT_ONLY = UnitCurrency(code="EUR", name="euros", prices=[UnitPrices.CURRENT])
_USD_BOTH = UnitCurrency(
    code="USD",
    name="US dollars",
    prices=[UnitPrices.CURRENT, UnitPrices.CONSTANT],
    base_year_min=2000,
    base_year_max=2023,
)
_CURRENT_EUR = UnitChoice(currency="EUR", prices=UnitPrices.CURRENT)
_CONSTANT_2015_USD = UnitChoice(currency="USD", prices=UnitPrices.CONSTANT, base_year=2015)


@pytest.mark.parametrize(
    ("currencies", "start", "options"),
    [
        (
            [_USD_BOTH, _EUR_CURRENT_ONLY],
            _CONSTANT_2015_USD,
            ["USD-constant-2015", "USD-current"],
        ),
        (
            [_EUR_CURRENT_ONLY, _USD_BOTH],
            _CURRENT_EUR,
            ["USD-current", "USD-constant-2023"],
        ),
        ([_EUR_CURRENT_ONLY], _CURRENT_EUR, ["EUR-current"]),
    ],
    ids=["start-currency-serves-both", "first-currency-serving-both", "no-currency-serves-both"],
)
def test_whole_units_offer_both_bases_the_start_s_basis_first(
    currencies: list[UnitCurrency], start: UnitChoice, options: list[str]
) -> None:
    offer = UnitOffer(currencies=currencies, presets=[c.code for c in currencies], start=start)

    assert list(whole_units(offer)) == options
