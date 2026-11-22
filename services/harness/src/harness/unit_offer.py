"""The units a question or a card offers for money DFO converts, read off Search's spec blocks.

Search states what it can convert a part into on `resolved.unit_conversion`:
each currency with the years it serves in current and in constant prices,
the base-year range of its constant prices, the base year an unnamed one
completes to (`default_base_year`) and the national currency of each role's
one place (`role_currencies`). `unit_offer` turns one or more blocks into a
`harness.wire.UnitOffer`, keeping only what every block serves, so a unit the
user picks is one Search serves for each part the offer covers.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from compiler.evidence import read_unit_dcid
from pydantic import ValidationError

from harness.wire import UnitChoice, UnitCurrency, UnitOffer, UnitPrices

logger = logging.getLogger(__name__)

# The currency the chips start on when it is offered, listed first.
_FIRST_CURRENCY = "USD"


def _conversion(block: Mapping[str, Any]) -> Mapping[str, Any]:
    resolved = block.get("resolved")
    conversion = resolved.get("unit_conversion") if isinstance(resolved, Mapping) else None
    return conversion if isinstance(conversion, Mapping) else {}


def _currencies(block: Mapping[str, Any]) -> dict[str, UnitCurrency]:
    """The currencies `block` converts into, by code, each with the bases it serves.

    Constant prices count only with a base-year range. A currency whose
    entry does not make a `UnitCurrency` (an inverted or non-integer range)
    is left out and logged, so a Search defect narrows the offer.
    """
    offered: dict[str, UnitCurrency] = {}
    for currency in _conversion(block).get("currencies") or ():
        if not isinstance(currency, Mapping) or not currency.get("code"):
            continue
        low, high = currency.get("base_year_min"), currency.get("base_year_max")
        constant = currency.get("constant_years") is not None and None not in (low, high)
        prices = [
            *([UnitPrices.CURRENT] if currency.get("current_years") is not None else []),
            *([UnitPrices.CONSTANT] if constant else []),
        ]
        if not prices:
            continue
        code = str(currency["code"])
        try:
            offered[code] = UnitCurrency(
                code=code,
                name=str(currency.get("noun") or code),
                prices=prices,
                base_year_min=low if constant else None,
                base_year_max=high if constant else None,
            )
        except ValidationError as error:
            logger.warning("unit_conversion currency %s left out: %s", code, error)
    return offered


def _shared(first: UnitCurrency, other: UnitCurrency) -> UnitCurrency | None:
    """What `first` and `other`, one currency of two blocks, both serve; `None` when nothing."""
    prices = [basis for basis in first.prices if basis in other.prices]
    low = high = None
    if UnitPrices.CONSTANT in prices:
        assert first.base_year_min is not None and other.base_year_min is not None
        assert first.base_year_max is not None and other.base_year_max is not None
        low = max(first.base_year_min, other.base_year_min)
        high = min(first.base_year_max, other.base_year_max)
        if low > high:
            prices.remove(UnitPrices.CONSTANT)
            low = high = None
    if not prices:
        return None
    return first.model_copy(update={"prices": prices, "base_year_min": low, "base_year_max": high})


def _start(
    currency: UnitCurrency, *, prefer: Literal["current", "constant"] | None, base_year: int | None
) -> UnitChoice:
    """Where the chips start in `currency`: `prefer` when served, else constant when served, else
    current; a constant start takes `base_year` clamped into the currency's range."""
    if prefer is not None and UnitPrices(prefer) in currency.prices:
        prices = UnitPrices(prefer)
    elif UnitPrices.CONSTANT in currency.prices:
        prices = UnitPrices.CONSTANT
    else:
        prices = UnitPrices.CURRENT
    if prices is UnitPrices.CURRENT:
        return UnitChoice(currency=currency.code, prices=prices)
    low, high = currency.base_year_min, currency.base_year_max
    assert low is not None and high is not None
    year = high if base_year is None else min(max(base_year, low), high)
    return UnitChoice(currency=currency.code, prices=prices, base_year=year)


def unit_offer(
    blocks: Sequence[Mapping[str, Any]],
    *,
    prefer: Literal["current", "constant"] | None = None,
) -> UnitOffer | None:
    """The units every one of `blocks` converts into, or `None` when they share no currency
    serving both current and constant prices.

    A currency is offered when every block offers it, with the bases every
    block serves in it and the base years every block's range holds.
    `presets` lists US dollars first, then each role's currency, each
    offered and once. The chips start on US dollars when offered, else the
    first preset, else the first currency; on `prefer`, the prices a skill
    prefers, when that currency serves it, else constant prices when it
    does, else current; and, for constant, on the first block's
    `default_base_year` clamped into the currency's range.
    """
    if not blocks:
        return None
    shared = _currencies(blocks[0])
    for block in blocks[1:]:
        theirs = _currencies(block)
        shared = {
            code: kept
            for code, currency in shared.items()
            if code in theirs and (kept := _shared(currency, theirs[code])) is not None
        }
    if not any(len(currency.prices) == 2 for currency in shared.values()):
        return None
    roles = [
        str(code)
        for block in blocks
        for code in (_conversion(block).get("role_currencies") or {}).values()
    ]
    presets = list(dict.fromkeys(c for c in (_FIRST_CURRENCY, *roles) if c in shared))
    start_code = presets[0] if presets else next(iter(shared))
    default = _conversion(blocks[0]).get("default_base_year")
    return UnitOffer(
        currencies=list(shared.values()),
        presets=presets,
        start=_start(
            shared[start_code],
            prefer=prefer,
            base_year=default if isinstance(default, int) else None,
        ),
    )


def unit_on_basis(
    block: Mapping[str, Any], offer: UnitOffer, *, currency: str, prices: UnitPrices
) -> UnitChoice | None:
    """`prices` in `currency`, as `offer`, the units `block` converts into,
    serves them: for constant prices, at the base year `block` completes an
    unnamed one to (`default_base_year`), clamped into the currency's range.
    `None` when `offer` does not serve `currency` in `prices`."""
    served = next((c for c in offer.currencies if c.code == currency), None)
    if served is None or prices not in served.prices:
        return None
    default = _conversion(block).get("default_base_year")
    return _start(
        served, prefer=prices.value, base_year=default if isinstance(default, int) else None
    )


def served_unit(block: Mapping[str, Any]) -> UnitChoice | None:
    """The unit `block`'s series is served in, `None` when its currency cannot be read.

    The unit its identity requests, else the one its unit facet's id names. A
    request that does not make a `UnitChoice` reads as `None`.
    """
    identity = (block.get("definition") or {}).get("identity") or {}
    request = ((identity.get("facets") or {}).get("unit") or {}).get("request")
    if isinstance(request, Mapping) and request.get("currency") and request.get("prices"):
        try:
            return UnitChoice.model_validate(
                {key: request.get(key) for key in ("currency", "prices", "base_year")}
            )
        except ValidationError:
            logger.warning("unit request %r is not a unit", dict(request))
            return None
    unit = ((block.get("resolved") or {}).get("facets") or {}).get("unit") or {}
    ref = unit.get("ref")
    reading = read_unit_dcid(ref if isinstance(ref, str) else None)
    currency = reading.currency or unit.get("currency")
    if not isinstance(currency, str) or currency == "PPP":
        return None
    if reading.prices == "constant" and reading.base_year is not None:
        return UnitChoice(
            currency=currency, prices=UnitPrices.CONSTANT, base_year=reading.base_year
        )
    return UnitChoice(currency=currency, prices=UnitPrices.CURRENT)


def unit_words(unit: UnitChoice, offer: UnitOffer | None = None) -> str:
    """`unit` in words, "Constant 2023 US dollars" or "Current US dollars", the currency
    named as `offer` names it, else by its code."""
    names = {currency.code: currency.name for currency in offer.currencies} if offer else {}
    name = names.get(unit.currency, unit.currency)
    if unit.prices is UnitPrices.CONSTANT:
        return f"Constant {unit.base_year} {name}"
    return f"Current {name}"


def offered_unit_refusal(offer: UnitOffer, unit: UnitChoice) -> str | None:
    """Why `unit` lies outside `offer`, or `None` when the offer holds it."""
    currency = next((c for c in offer.currencies if c.code == unit.currency), None)
    if currency is None:
        return f"currency {unit.currency!r} is not offered"
    if unit.prices not in currency.prices:
        return f"{unit.currency} is not offered in {unit.prices.value} prices"
    if unit.base_year is not None:
        assert currency.base_year_min is not None and currency.base_year_max is not None
        if not currency.base_year_min <= unit.base_year <= currency.base_year_max:
            return (
                f"base year {unit.base_year} is outside {currency.base_year_min} to "
                f"{currency.base_year_max} for {unit.currency}"
            )
    return None


def whole_units(offer: UnitOffer) -> dict[str, UnitChoice]:
    """The whole units a client that renders no unit question picks among, by option id.

    Each basis of one currency: the start's when it serves both bases, else
    the first offered currency that does, else the start's alone. The unit on
    the start's basis comes first, the start itself when its currency is the
    one. A constant unit takes the start's base year when the currency's
    range holds it, else the currency's latest. An id names the unit,
    "USD-current" or "USD-constant-2023".
    """
    start = offer.start
    own = next(c for c in offer.currencies if c.code == start.currency)
    both = next((c for c in (own, *offer.currencies) if set(c.prices) == set(UnitPrices)), own)
    ordered = sorted(both.prices, key=lambda prices: prices is not start.prices)
    units = []
    for prices in ordered:
        if prices is UnitPrices.CURRENT:
            units.append(UnitChoice(currency=both.code, prices=prices))
            continue
        assert both.base_year_min is not None and both.base_year_max is not None
        year = start.base_year
        if year is None or not both.base_year_min <= year <= both.base_year_max:
            year = both.base_year_max
        units.append(UnitChoice(currency=both.code, prices=prices, base_year=year))
    return {
        "-".join(
            str(part)
            for part in (unit.currency, unit.prices.value, unit.base_year)
            if part is not None
        ): unit
        for unit in units
    }
