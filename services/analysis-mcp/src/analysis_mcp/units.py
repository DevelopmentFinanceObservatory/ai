"""How a unit reads in model-facing text and in stored JSON.

A frame's `unit` cell is a unit ref (a plain string), a converted unit (a
`dfo_frame_grammar.grammar.UnitRequest`), or `None`. A ref reads as itself
in both forms. A converted unit reads in text as the price-basis words
`get_data`'s digest uses, and in JSON as its request object. The digest,
each verb's trace and each unit refusal all render units through here.
"""

from __future__ import annotations

from common.price_basis import price_basis_reading
from dfo_frame_grammar.grammar import UnitRequest


def converted_unit_label(unit: UnitRequest) -> str:
    """`unit` in the price-basis words `get_data`'s digest uses, such as
    "converted to constant 2020 prices, euros"."""
    reading = price_basis_reading(
        currency=unit.currency, prices=unit.prices, base_year=unit.base_year
    )
    return f"converted to {reading}"


def unit_name(unit: str | UnitRequest) -> str:
    """`unit` as text names it: a ref as itself, a converted unit by its
    label."""
    return unit if isinstance(unit, str) else converted_unit_label(unit)


def unit_json(unit: str | UnitRequest | None) -> str | dict[str, object] | None:
    """`unit` as JSON carries it: a ref as itself, a converted unit as its
    request object (the `unit_request` a refusal's candidate carries)."""
    return unit.as_json() if isinstance(unit, UnitRequest) else unit
