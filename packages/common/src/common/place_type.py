"""Place types: the column dfo-apis enriches a place column with, and the level a row sits at.

A mint asked for `{role}_type` in its `options.columns` fills it from the place's Data
Commons `typeOf`, every value sorted and comma-joined: `Country`, `Place` for Earth,
`ProvisionalNode` for a World Bank aggregate, `AdministrativeArea1,Country,EurostatNUTS1,Region`
for `nuts/FI2`. A place the graph holds no type for gets an empty cell, and data-mcp writes
that cell as `UNTYPED`, since a frame's role column holds no empty string.

A place sits at each of its types (`levels_of`), with one exception: a place carrying a
Eurostat NUTS type is not at the `Country` level. NUTS divides the territory of an EU or EFTA
state, so such a place's figures sit inside that state's. The graph types Åland (`nuts/FI2`)
both `Country` and `EurostatNUTS1`, and Finland's figures include it. A territory the graph
types `Country` and `AdministrativeArea1` with no NUTS type, such as Hong Kong, Guam or
Greenland, stays a country: its sovereign state's figures leave it out.

Several places sit at one level when some level in `PLACE_LEVELS` is one of every place's
levels (`at_one_level`): the countries a level filter kept, or the countries dfo-apis serves
when it falls back from every place. `PLACE_LEVELS` holds the types whose places do not contain
one another, so a rank or a sum across places at one level counts each figure once. A type
outside `PLACE_LEVELS` names no level. `UNGeoRegion` is such a type: the graph types both
Sub-Saharan Africa and the Eastern Africa it contains `UNGeoRegion`. Earth (`Place`) and the
countries typed `Country,Place` share `Place` and no level, and World Bank aggregates sharing
`ProvisionalNode` overlap, as "Low & middle income" contains "Low income".
One place alone sits at one level whatever its types.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Literal, get_args

# The subject role a place-less request leaves open, and the only role data-mcp asks types for.
PLACE_ROLE = "observationAbout"

# What a place with no type is written as in a `{role}_type` column.
UNTYPED = "untyped"

# The levels a place-level filter keeps, by Data Commons type: types whose places do not contain
# one another.
PlaceLevel = Literal[
    "Country",
    "Continent",
    "AdministrativeArea1",
    "AdministrativeArea2",
    "City",
    "EurostatNUTS1",
    "EurostatNUTS2",
    "EurostatNUTS3",
]
PLACE_LEVELS: frozenset[str] = frozenset(get_args(PlaceLevel))

_COUNTRY = "Country"
_NUTS = frozenset({"EurostatNUTS1", "EurostatNUTS2", "EurostatNUTS3"})
_SEPARATOR = ","


def type_column(role: str) -> str:
    """The enrichment column naming each `role` place's types: `observationAbout_type`."""
    return f"{role}_type"


def levels_of(cell: str) -> frozenset[str]:
    """The levels the place one `{role}_type` cell describes sits at: its types, less `Country`
    when a Eurostat NUTS type is among them. None for `UNTYPED`."""
    if cell == UNTYPED:
        return frozenset()
    types = frozenset(part for part in cell.split(_SEPARATOR) if part)
    return types - {_COUNTRY} if types & _NUTS else types


def at_one_level(cells: Collection[str], *, place_count: Callable[[], int]) -> bool:
    """Whether the places a `{role}_type` column describes sit at one level.

    `cells` are the column's distinct cells and `place_count` counts the distinct places in
    its role column. Places whose cells share a level in `PLACE_LEVELS` sit at one level, and
    so does one place alone, whatever its cell. A place has one cell, so `place_count` is
    called only when a single cell with no shared level leaves the number of places open.
    """
    if not cells or frozenset.intersection(*(levels_of(cell) & PLACE_LEVELS for cell in cells)):
        return True
    return len(cells) == 1 and place_count() == 1
