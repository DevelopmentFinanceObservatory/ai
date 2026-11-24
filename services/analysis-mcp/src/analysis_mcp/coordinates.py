"""A series' coordinates as text, as dfo-apis' assembler groups a frame's rows."""

from __future__ import annotations

from dfo_frame_grammar.grammar import ComposedUnit, UnitCell, UnitRequest
from dfo_spec_kit.identity import canonicalize


def coordinate_text(cell: str | UnitCell | None) -> str | None:
    """One cell of a series' coordinates as text, in the form `apis`' assembler
    groups it under: a converted unit as `currency/prices[/base_year][/deflation_basis]`, a composed
    unit as the RFC 8785 text of its JSON object, and any other cell as itself."""
    if isinstance(cell, UnitRequest):
        parts = (cell.currency, cell.prices, cell.base_year, cell.deflation_basis)
        return "/".join(str(part) for part in parts if part is not None)
    if isinstance(cell, ComposedUnit):
        return canonicalize(cell.as_json()).decode()
    return cell
