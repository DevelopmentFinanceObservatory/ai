"""Place types: the column apis enriches a place column with, and the level a row sits at.

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
levels (`at_one_level`): the countries a level filter kept, or the countries apis serves
when it falls back from every place. `PLACE_LEVELS` holds the types whose places do not contain
one another, so a rank or a sum across places at one level counts each figure once. A type
outside `PLACE_LEVELS` names no level. `UNGeoRegion` is such a type: the graph types both
Sub-Saharan Africa and the Eastern Africa it contains `UNGeoRegion`. Earth (`Place`) and the
countries typed `Country,Place` share `Place` and no level, and World Bank aggregates sharing
`ProvisionalNode` overlap, as "Low & middle income" contains "Low income".
One place alone sits at one level whatever its types.

A flow's parties (`financingProvider`, `financingRecipient`) carry a kind instead, in a
`{role}_party_kind` column (`kind_column`) apis fills from its code index: `party`, `total` or
`residual`. DAC tables publish group totals as parties of their own: "Africa, Total", an
income group, the G7. A total holds other parties' figures, so a sum across a flow role leaves
it out. A residual ("Africa, regional") holds figures no other party does, so it adds to the
parties. apis leaves the cell empty for a party it declares no kind for, and data-mcp
writes that cell as `UNTYPED` too, which no kind names.

Two attribute columns sit beside the kind. `{role}_party_sector` (`sector_column`) holds a
party's `Sector`, `official` or `private` (a foundation, a national lottery), and a total's
when every party it holds shares one; it is empty, so `UNTYPED`, for a residual and for a
total holding both. `{role}_party_whole` (`whole_column`) is `WHOLE_CELL` on a total that holds
a whole population in the row's dataset and year, such as "Official Donors, Total", and
`false` on every other row. Two whole totals on one axis hold disjoint classes of party
(`MemberClass`), so a sum across a role of whole totals alone adds them.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from types import MappingProxyType
from typing import Literal, get_args

# The subject role a place-less request leaves open.
PLACE_ROLE = "observationAbout"

# The frame column each flow axis of a spec's identity is served in. DAC1, DAC2A, DAC2B, the CRS
# and IDS all spell their giving and receiving sides so.
FLOW_ROLE_COLUMNS: Mapping[str, str] = MappingProxyType(
    {"origin": "financingProvider", "destination": "financingRecipient"}
)
PARTY_ROLES: frozenset[str] = frozenset(FLOW_ROLE_COLUMNS.values())

# The population a search names for the total across each flow party role's whole side.
WHOLE_SIDE_POPULATIONS: Mapping[str, str] = MappingProxyType(
    {"financingProvider": "all donors", "financingRecipient": "all recipients"}
)

# Every role a `{role}_type` column can describe, in sorted order.
TYPED_ROLES: tuple[str, ...] = tuple(sorted({PLACE_ROLE, *PARTY_ROLES}))

# What a place with no type, or a party with no kind, is written as in its enrichment column.
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

# The kinds apis serves in a `{role}_party_kind` cell. A total holds other parties' figures.
PartyKind = Literal["party", "residual", "total"]
PARTY_KINDS: frozenset[str] = frozenset(get_args(PartyKind))
TOTAL_KIND: PartyKind = "total"

# The sectors apis serves in a `{role}_party_sector` cell.
Sector = Literal["official", "private"]
PRIVATE_SECTOR: Sector = "private"

# The classes of party a whole total can hold in full: a party of either sector, or a residual.
MemberClass = Literal["official", "private", "residual"]

# The `{role}_party_whole` cell of a total holding a whole population; every other row's is
# `false`.
WHOLE_CELL = "true"

_ENRICHMENT_SUFFIXES = ("_type", "_party_kind", "_party_sector", "_party_whole")


def type_column(role: str) -> str:
    """The enrichment column naming each `role` place's types: `observationAbout_type`."""
    return f"{role}_type"


def kind_column(role: str) -> str:
    """The enrichment column naming each `role` party's kind: `financingRecipient_party_kind`."""
    return f"{role}_party_kind"


def sector_column(role: str) -> str:
    """The enrichment column naming each `role` party's sector: `financingProvider_party_sector`."""
    return f"{role}_party_sector"


def whole_column(role: str) -> str:
    """The enrichment column marking each `role` total that holds a whole population:
    `financingProvider_party_whole`."""
    return f"{role}_party_whole"


def enrichment_columns(role: str) -> tuple[str, str, str, str]:
    """The columns that can describe `role`'s places: its type, kind, sector and whole column."""
    return type_column(role), kind_column(role), sector_column(role), whole_column(role)


def enrichment_role(column: str, *, roles: Collection[str]) -> str | None:
    """The role among `roles` one of whose enrichment columns (`enrichment_columns`)
    `column` is, or None when it is none of them."""
    for suffix in _ENRICHMENT_SUFFIXES:
        role = column.removesuffix(suffix)
        if role != column and role in roles:
            return role
    return None


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
