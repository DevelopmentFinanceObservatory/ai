"""How a unit reads in model-facing text and in stored JSON.

A frame's `unit` cell is a unit ref (a plain string), a converted unit (a
`dfo_frame_grammar.grammar.UnitRequest`), a composed unit (a
`dfo_frame_grammar.grammar.ComposedUnit`, the unit of one measure divided by
another), or `None`. A ref reads as itself in both forms. A converted unit
reads in text as the price-basis words `get_data`'s digest uses, and in JSON
as its request object. A composed unit reads in text as
`<numerator> per <denominator>` and in JSON as its
`{numerator, denominator}` object. The digest, each verb's trace and each
unit refusal all render units through here.
"""

from __future__ import annotations

from common.price_basis import price_basis_reading
from dfo_frame_grammar.grammar import (
    ComposedUnit,
    UnitCell,
    UnitPart,
    UnitRequest,
    unit_cell_to_json,
)


def _price_basis(unit: UnitRequest) -> str:
    return price_basis_reading(currency=unit.currency, prices=unit.prices, base_year=unit.base_year)


def converted_unit_label(unit: UnitRequest) -> str:
    """`unit` in the price-basis words `get_data`'s digest uses, such as
    "converted to constant 2020 prices, euros"."""
    return f"converted to {_price_basis(unit)}"


def _part_name(part: UnitPart) -> str:
    """One side of a composed unit as text names it: a ref as itself, a
    converted unit by its price basis alone ("constant 2020 prices, euros"),
    and a side that declares no unit by the id of the StatVar it counts,
    without the `dcid:` prefix."""
    if isinstance(part, str):
        return part
    if isinstance(part, UnitRequest):
        return _price_basis(part)
    return part.stat_var.removeprefix("dcid:")


def unit_name(unit: UnitCell) -> str:
    """`unit` as text names it: a ref as itself, a converted unit by its
    label, a composed unit as "<numerator> per <denominator>"."""
    if isinstance(unit, str):
        return unit
    if isinstance(unit, ComposedUnit):
        return f"{_part_name(unit.numerator)} per {_part_name(unit.denominator)}"
    return converted_unit_label(unit)


def unit_json(unit: UnitCell | None) -> str | dict[str, object] | None:
    """`unit` as JSON carries it, in the frame grammar's own form: a ref as
    itself, a converted unit as its request object (the `unit_request` a
    refusal's candidate carries), a composed unit as its
    `{numerator, denominator}` object."""
    return unit_cell_to_json(unit)
