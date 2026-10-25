"""Rolling periods: the populations a latest period is taken over, and the key
its stored resolution is filed under.

A rolling call states its period relative to the latest period of the data.
That period is the newest one at least half of the series hold
(`dfo_frame_grammar.latest_anchor`), counted over populations: one for each root
dataset a rolling period of the recipe reads, narrowed to the series that count.

Which roots those are is decided by path. A path from the top of a recipe's tree
down to a root counts when it passes through a rolling op-spec, one whose
`rolling` parameter is an object. `rolling_roots` gives the roots such paths end
at and `anchor_populations` the populations along them, both from the one walk
of the tree, so a dataset joined beside or above a rolling step, with no rolling
step on its own path, is in neither: its coverage does not move the period.

A population is `{"spec_id": <dataset id>, "bindings": {column: [values]}}`,
with `"levels": {column: level}` beside them when a place-type column is held to
one level of place. `bindings` keeps the series whose column holds one of the
values. `levels` keeps the series whose place-type column, such as
`observationAbout_type`, describes a place at the level, such as `Country`
(`common.place_type.levels_of`): a rule over the cell, so it names no cell and
holds however the data spells a place's types. A population carries no
`content_version`, so it holds on newer data. `canonical_populations` gives a
list of them its one spelling, the one `analysis_mcp.op_spec.Populations` hashes
into a rolling op-spec:

- every column and value NFC-normalised, each column's values in code point
  order with repeats dropped, and the columns in name order;
- `levels` with its columns NFC-normalised and in name order, and absent when no
  column is held to a level, so a population without one is `{spec_id,
  bindings}` alone;
- the entries ordered by `spec_id`, then by each entry's RFC 8785 bytes, with an
  entry stated twice kept once.

A pairing population, `{"pair": [left, right], "keys": {left column: right
column}}`, stands for the series a verb pairs key by key: `left` and `right` are
each a list of populations, flat or pairing, for the verb's two sides, and
`keys` maps each key column of the left side's frame to the right side's column
holding the same entities. Its canonical form has both sides canonical, the
keys NFC-normalised in left-name order, and at least one key. Pairing entries
sort after every flat one, by their RFC 8785 bytes, so the flat rules alone
spell and order a list without one.

`anchor_populations` builds that list for a recipe from its tree. Each counted
path gives one population over its root's dataset, bound by:

- every `category_match` predicate of a `filter` on the path, and every
  `place_level` predicate as a level;
- the entity binding of each two-sided verb on the path, for the side the path
  goes down: `left_entity` and `numerator_entity` for `frame_ref`,
  `right_entity` for `right_frame_ref`, `denominator_entity` for
  `denominator_frame_ref`. A verb whose second frame slot is null reads
  `frame_ref` twice, so the paths under it are walked once for each side;
- the card's selection.

A binding counts only where the root carries its column, and never on
`time_period`, `value` or `status`, which are no part of a series. Two bindings
on one column keep the values both allow. A level differs. A `place_level`
predicate on a path whose root lacks the place-type column raises
`PopulationLevelColumnMissingError`, since the population would otherwise count
places the filter drops.

A keyed pairing verb whose two sides both hold a counted path gives one pairing
population in place of its sides' populations, with the key map read from its
own arguments: `denominator_by` for a `share_of` or `quotient_of`, and
`join_keys`, each paired through `right_keys` or else with itself, for a
`join`. A `join` with no `join_keys` leaves its keys to analysis-mcp's
auto-detection, which no tree records, so it raises `PairingKeysUnstatedError`.
A verb that pairs no keys (`compare`, `sum_of`, and a `share_of` or
`quotient_of` with no `denominator_by`, whose keys share one pooled
denominator), and a keyed verb with one side counted, pass each side's
populations through flat. A side that is itself a keyed pairing nests.

`anchor_key` is `rak_` and 32 hex digits of the SHA-256 of the RFC 8785 form of
`{"populations", "roots", "rule"}`: the canonical populations, the
`{spec_id, content_version}` pairs of the root frames the period is taken over
(`rolling_roots`) sorted with repeats dropped, and the rule's name. Nothing in
the preimage says which root a population belongs to, so a population is
counted over every root frame of its dataset. A pairing population's roots are
the roots of the populations on its sides.

This module reads plain JSON values and imports no store, service or planner.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import rfc8785

__all__ = [
    "ANCHOR_KEY_PATTERN",
    "NON_SERIES_COLUMNS",
    "PairingKeysUnstatedError",
    "PopulationEmptyError",
    "PopulationLevelColumnMissingError",
    "anchor_key",
    "anchor_populations",
    "canonical_bindings",
    "canonical_keys",
    "canonical_levels",
    "canonical_populations",
    "canonical_roots",
    "canonical_strict_levels",
    "flat_populations",
    "population_order",
    "rolling_roots",
]

ANCHOR_KEY_PREFIX = "rak_"
ANCHOR_KEY_PATTERN = r"^rak_[0-9a-f]{32}$"

_DIGEST_HEX_LENGTH = 32
_ROLLING = "rolling"
_PAIR = "pair"
_PRIMARY_SLOT = "frame_ref"
_SLOT_SUFFIX = "_frame_ref"
# The columns that are no part of a series: `apis`' assembler groups rows
# into series by every other column.
NON_SERIES_COLUMNS = frozenset({"time_period", "value", "status"})
# Each entity binding of a two-sided verb, and the frame slot it narrows.
_ENTITY_SLOTS = {
    "left_entity": _PRIMARY_SLOT,
    "numerator_entity": _PRIMARY_SLOT,
    "right_entity": "right_frame_ref",
    "denominator_entity": "denominator_frame_ref",
}


class PopulationEmptyError(ValueError):
    """The bindings on one path to a root share no value for `column`, or hold
    that place-type column to two levels, so no series of that root counts."""

    def __init__(self, *, spec_id: str, column: str) -> None:
        self.spec_id = spec_id
        self.column = column
        super().__init__(
            f"the bindings on {column!r} leave no series of {spec_id} to take a latest period over"
        )


class PairingKeysUnstatedError(ValueError):
    """A `join` whose two sides both hold a counted path states no `join_keys`,
    so which keys it pairs is left to analysis-mcp and no population can say."""

    def __init__(self) -> None:
        super().__init__(
            "a join over two rolling sides states no join_keys, so the keys it pairs "
            "cannot be stated in its population"
        )


class PopulationLevelColumnMissingError(ValueError):
    """A `place_level` predicate on a path to a root tests `column`, which that
    root's frame does not carry, so the population cannot be held to the level."""

    def __init__(self, *, spec_id: str, column: str) -> None:
        self.spec_id = spec_id
        self.column = column
        super().__init__(
            f"the root frame of {spec_id} carries no {column!r} column to hold its "
            "population to a level of place"
        )


def canonical_levels(levels: Mapping[str, str]) -> dict[str, str]:
    """`levels` in its one spelling: columns NFC-normalised and in name order.

    Raises:
        ValueError: Two columns normalise to one name.
    """
    canonical: dict[str, str] = {}
    for column, level in levels.items():
        name = unicodedata.normalize("NFC", column)
        if name in canonical:
            raise ValueError(f"levels name the column {name!r} twice")
        canonical[name] = level
    return dict(sorted(canonical.items()))


def canonical_strict_levels(columns: Iterable[str]) -> tuple[str, ...]:
    """The strict level columns in their one spelling."""
    return tuple(sorted({unicodedata.normalize("NFC", column) for column in columns}))


def canonical_bindings(bindings: Mapping[str, Iterable[str]]) -> dict[str, tuple[str, ...]]:
    """`bindings` in its one spelling. See the module docstring.

    Raises:
        ValueError: Two columns normalise to one name.
    """
    canonical: dict[str, tuple[str, ...]] = {}
    for column, values in bindings.items():
        name = unicodedata.normalize("NFC", column)
        if name in canonical:
            raise ValueError(f"bindings name the column {name!r} twice")
        canonical[name] = tuple(sorted({unicodedata.normalize("NFC", value) for value in values}))
    return dict(sorted(canonical.items()))


def canonical_keys(keys: Mapping[str, str]) -> dict[str, str]:
    """A pairing's `keys` in their one spelling: both columns of each pair
    NFC-normalised, in left-name order.

    Raises:
        ValueError: `keys` is empty, or two left columns normalise to one name.
    """
    if not keys:
        raise ValueError("a pairing population pairs no keys")
    canonical: dict[str, str] = {}
    for left, right in keys.items():
        name = unicodedata.normalize("NFC", left)
        if name in canonical:
            raise ValueError(f"keys name the column {name!r} twice")
        canonical[name] = unicodedata.normalize("NFC", right)
    return dict(sorted(canonical.items()))


def population_order(population: Mapping[str, Any]) -> tuple[bool, str, bytes]:
    """The sort key of one population already in its canonical spelling:
    whether it pairs, its `spec_id` (empty for a pairing), then its RFC 8785
    bytes."""
    pairs = _PAIR in population
    return pairs, "" if pairs else population["spec_id"], rfc8785.dumps(population)


def canonical_populations(populations: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """`populations` in its one spelling, as JSON values. See the module docstring.

    Raises:
        ValueError: Two columns of one entry normalise to one name, or a
            pairing has an empty side or no keys.
    """
    keyed: dict[tuple[bool, str, bytes], dict[str, Any]] = {}
    for population in populations:
        if _PAIR in population:
            left, right = population[_PAIR]
            sides = [canonical_populations(left), canonical_populations(right)]
            if not all(sides):
                raise ValueError("a side of a pairing population holds no population")
            pairing = {_PAIR: sides, "keys": canonical_keys(population["keys"])}
            keyed[population_order(pairing)] = pairing
            continue
        bindings = canonical_bindings(population.get("bindings", {}))
        levels = canonical_levels(population.get("levels") or {})
        entry: dict[str, Any] = {
            "spec_id": population["spec_id"],
            "bindings": {column: list(values) for column, values in bindings.items()},
        }
        if levels:
            entry["levels"] = levels
        if strict_levels := canonical_strict_levels(population.get("strict_levels") or ()):
            unknown = sorted(set(strict_levels) - set(levels))
            if unknown:
                raise ValueError(f"strict_levels name columns without a level: {unknown!r}")
            entry["strict_levels"] = list(strict_levels)
        keyed[population_order(entry)] = entry
    return [keyed[key] for key in sorted(keyed)]


def flat_populations(populations: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    """Every flat population of `populations`, those on each side of a pairing
    included, depth first, left side before right."""
    for population in populations:
        if _PAIR in population:
            for side in population[_PAIR]:
                yield from flat_populations(side)
        else:
            yield population


def canonical_roots(roots: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """The `{spec_id, content_version}` pairs of `roots`, sorted, each once."""
    pairs = {(root["spec_id"], root["content_version"]) for root in roots}
    return [
        {"spec_id": spec_id, "content_version": content_version}
        for spec_id, content_version in sorted(pairs)
    ]


def anchor_key(
    *, populations: Iterable[Mapping[str, Any]], roots: Iterable[Mapping[str, str]], rule: str
) -> str:
    """`rak_` and 32 hex digits naming one resolution of the latest period.

    Args:
        populations: The populations, in any spelling.
        roots: The `{spec_id, content_version}` pair of each root frame, as the
            materialisation store holds it. Order and repeats do not matter.
        rule: The name of the rule choosing the period, `dfo_frame_grammar.ANCHOR_RULE`.
            The name versions the computation: a change to how the latest period
            is chosen, or to which rows make one series, takes a new rule name,
            so a record stored under the old one is never read as the new. A
            form of population added to the preimage leaves the key of every
            list without it unchanged, so it takes no new rule name.
    """
    preimage = {
        "populations": canonical_populations(populations),
        "roots": canonical_roots(roots),
        "rule": rule,
    }
    digest = hashlib.sha256(rfc8785.dumps(preimage)).hexdigest()[:_DIGEST_HEX_LENGTH]
    return f"{ANCHOR_KEY_PREFIX}{digest}"


class _Narrowing:
    """What the verbs on one path keep of a root's series: `bindings`, each
    column to the values kept, and `levels`, each place-type column to the
    levels it is held to."""

    def __init__(self, above: _Narrowing | None = None) -> None:
        self.bindings: dict[str, set[str]] = (
            {column: set(values) for column, values in above.bindings.items()} if above else {}
        )
        self.levels: dict[str, set[str]] = (
            {column: set(levels) for column, levels in above.levels.items()} if above else {}
        )
        self.strict_levels: set[str] = set(above.strict_levels) if above else set()

    def bind(self, column: str, values: Iterable[str]) -> None:
        """Bind `column` to `values`, keeping what an earlier binding on it also allows."""
        kept = set(values)
        self.bindings[column] = self.bindings[column] & kept if column in self.bindings else kept

    def hold(self, column: str, level: str, *, strict: bool = False) -> None:
        """Hold the place-type `column` to `level`, beside any level it is already held to."""
        self.levels.setdefault(column, set()).add(level)
        if strict:
            self.strict_levels.add(column)


def _sides(
    node: Mapping[str, Any], above: _Narrowing
) -> Iterator[tuple[str, Mapping[str, Any], _Narrowing]]:
    """Each side of the derived `node`: its frame slot, the node of the frame it
    reads, and what `above` and the verb itself keep of that side."""
    parameters = node["op_spec"].get("parameters", {})
    inputs = node["inputs"]
    slots = [
        _PRIMARY_SLOT,
        *sorted(name for name in parameters if name.endswith(_SLOT_SUFFIX)),
    ]
    for slot in slots:
        kept = _Narrowing(above)
        for predicate in parameters.get("predicates", ()):
            if predicate.get("kind") == "category_match":
                kept.bind(predicate["column"], predicate["values"])
            elif predicate.get("kind") == "place_level":
                kept.hold(
                    predicate["column"], predicate["level"], strict=predicate.get("strict", False)
                )
        for parameter, entity_slot in _ENTITY_SLOTS.items():
            if entity_slot == slot:
                for column, value in (parameters.get(parameter) or {}).items():
                    kept.bind(column, [value])
        # A null second slot reads the first slot's frame again.
        yield slot, inputs.get(slot, inputs[_PRIMARY_SLOT]), kept


def _pairing(op_spec: Mapping[str, Any]) -> tuple[str, dict[str, str]] | None:
    """The second frame slot of a verb that pairs keys, and the key map its
    arguments state, empty for a `join` that leaves its keys to auto-detection.
    `None` for a verb that pairs no keys."""
    parameters = op_spec.get("parameters", {})
    if denominator_by := parameters.get("denominator_by"):
        return "denominator_frame_ref", dict(denominator_by)
    # A stored recipe's op-spec names its verb. A tree the compiler writes from
    # a plan carries parameters alone, and writes `join_keys` on every join.
    if op_spec.get("verb") == "join" or "join_keys" in parameters:
        right_keys = parameters.get("right_keys") or {}
        join_keys = parameters.get("join_keys") or ()
        return "right_frame_ref", {key: right_keys.get(key, key) for key in join_keys}
    return None


@dataclass(frozen=True, slots=True)
class _Pair:
    """The counted paths below each side of a verb that pairs `keys`."""

    left: list[_Counted]
    right: list[_Counted]
    keys: dict[str, str]


# A counted path, as its root's node and what the verbs on the way keep of it,
# or the counted paths of a keyed pairing's two sides.
type _Counted = tuple[Mapping[str, Any], _Narrowing] | _Pair


def _counted(node: Mapping[str, Any], above: _Narrowing, *, rolled: bool = False) -> list[_Counted]:
    """The paths from `node` down to a root that pass through a rolling
    op-spec, `above` applied, with the paths below a keyed pairing whose sides
    both hold one gathered into a `_Pair`. `rolled` says a rolling op-spec is
    already above `node`."""
    if "root" in node:
        return [(node, _Narrowing(above))] if rolled else []
    rolled = rolled or isinstance(node["op_spec"].get("parameters", {}).get(_ROLLING), Mapping)
    sides = [
        (slot, _counted(child, kept, rolled=rolled)) for slot, child, kept in _sides(node, above)
    ]
    if (pairing := _pairing(node["op_spec"])) is not None:
        right_slot, keys = pairing
        left = [path for slot, paths in sides if slot == _PRIMARY_SLOT for path in paths]
        right = [path for slot, paths in sides if slot == right_slot for path in paths]
        if left and right:
            return [_Pair(left=left, right=right, keys=keys)]
    return [path for _, paths in sides for path in paths]


def _roots_of(counted: Iterable[_Counted]) -> Iterator[Mapping[str, Any]]:
    for path in counted:
        if isinstance(path, _Pair):
            yield from _roots_of(path.left)
            yield from _roots_of(path.right)
        else:
            yield path[0]


def rolling_roots(tree: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The root nodes of `tree` its rolling periods are taken over: each root
    some path through a rolling op-spec ends at, once, in `root` index order.
    Empty for a tree with no rolling op-spec. See the module docstring."""
    reached = {node["root"]: node for node in _roots_of(_counted(tree, _Narrowing()))}
    return [reached[index] for index in sorted(reached)]


def anchor_populations(
    tree: Mapping[str, Any],
    roots: Sequence[Mapping[str, Any]],
    *,
    root_columns: Sequence[Collection[str]],
    selection: Mapping[str, Iterable[str]] | None = None,
) -> list[dict[str, Any]]:
    """The canonical populations of the recipe `tree`. See the module docstring.

    Args:
        tree: The recipe's tree as `common.recipe.build_recipe` writes it. A
            derived node is read for `op_spec.parameters` and `inputs`, a root
            node for `root`, so a tree written from a plan before any frame
            exists serves as well: a node needs no `frame`, and a rolling
            node's `rolling` parameter needs no `population` yet.
        roots: The recipe's `roots`. Each is read for `dataset.spec_id`.
        root_columns: The column names of each root's frame, in `roots` order.
        selection: The card's selection, each dimension to the values it keeps.

    Returns:
        One `{"spec_id", "bindings"}` for each distinct population, with `levels`
        when it has any, and one `{"pair", "keys"}` for each keyed pairing whose
        sides both count, in the order and spelling a rolling op-spec hashes.
        Empty for a tree with no rolling op-spec.

    Raises:
        PopulationEmptyError: The bindings on one path share no value for a
            column, or hold one place-type column to two levels.
        PopulationLevelColumnMissingError: A `place_level` predicate on a path
            tests a column that path's root does not carry.
        PairingKeysUnstatedError: A `join` whose sides both hold a counted
            path states no `join_keys`.
    """

    def population(path: _Counted) -> dict[str, Any]:
        if isinstance(path, _Pair):
            if not path.keys:
                raise PairingKeysUnstatedError
            return {
                _PAIR: [
                    [population(side) for side in path.left],
                    [population(side) for side in path.right],
                ],
                "keys": path.keys,
            }
        node, kept = path
        root = node["root"]
        for dimension, values in (selection or {}).items():
            kept.bind(dimension, values)
        spec_id = roots[root]["dataset"]["spec_id"]
        carried = set(root_columns[root]) - NON_SERIES_COLUMNS
        bindings = {column: values for column, values in kept.bindings.items() if column in carried}
        for column, values in bindings.items():
            if not values:
                raise PopulationEmptyError(spec_id=spec_id, column=column)
        levels: dict[str, str] = {}
        for column, held in kept.levels.items():
            if column not in carried:
                raise PopulationLevelColumnMissingError(spec_id=spec_id, column=column)
            if len(held) > 1:
                # No place sits at two of the levels a filter keeps.
                raise PopulationEmptyError(spec_id=spec_id, column=column)
            (levels[column],) = held
        return {
            "spec_id": spec_id,
            "bindings": bindings,
            "levels": levels,
            "strict_levels": sorted(set(levels) & kept.strict_levels),
        }

    return canonical_populations(population(path) for path in _counted(tree, _Narrowing()))
