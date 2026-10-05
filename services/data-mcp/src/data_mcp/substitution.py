"""The single-series rule for a described-arm identity.

A described-arm identity (`shape.kind == "statvar_schema"`) names a StatVar family by its
slot bindings, and the spec_id mint resolves the whole family. `root_member` names the one
elected StatVar that is the family's root, when `search` elected exactly one and it is;
the harness's `compiler.sources.pinned_stat_var` reads it to judge a resolved identity one
series. `MAX_RESOLVED_STAT_VAR_DCIDS` is how many elected StatVars a mint takes.

This module is a leaf: standard library only, no import of the upstream client, the
definition store, fastmcp, or mcp, so the harness reaching `root_member`
never drags data-mcp's own transport stack into its own import chain.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# The most `resolved_stat_var_dcids` entries dfo-apis' mint takes
# (`SpecIdInput.resolved_stat_var_dcids`, capped at
# `statvar_members.enumerate.DEFAULT_MEMBER_CAP`), the most members its graph walk
# returns. `get_data` bounds its argument by it and the harness refuses a source that
# elects more.
MAX_RESOLVED_STAT_VAR_DCIDS = 700

# A described-arm identity whose entity roles are anything but exactly
# `{"observationAbout"}` -- the CRS `FinancialFlow` case's `origin`/
# `destination`, an aid flow's `provider`/`recipient`/`donor`, or any other
# named-family shape this predicate hasn't been taught -- reconstructs its own
# StatVars from slot bindings and legitimately enumerates many members a user may
# want, so `root_member` declines for every role set but this one. This is a
# whitelist, not a denylist of known role-bearing shapes: an entity role this
# predicate has never seen gets the same caution as a known one.
_SINGLE_SERIES_ENTITY_ROLES = frozenset({"observationAbout"})

# Slot binding states whose property every member of the family carries as a
# constraint key. dfo-apis maps `exact` to a `value` binding.
_BOUND_STATES = frozenset({"value", "exact"})


def _is_described_arm(identity: Mapping[str, Any]) -> bool:
    shape = identity.get("shape")
    return isinstance(shape, Mapping) and shape.get("kind") == "statvar_schema"


def _is_role_bearing(identity: Mapping[str, Any]) -> bool:
    """Whether `identity`'s entity roles are anything but exactly `{"observationAbout"}`.

    A whitelist, not a denylist: missing `entities`, an empty one, or a role set
    naming `observationAbout` plus (or instead of) anything else -- `origin`/
    `destination`, `provider`/`recipient`/`donor`, or a role this predicate has never
    seen -- all count as role-bearing. Only the one role set the single-series rule is
    proven safe for gets through.
    """
    entities = identity.get("entities")
    if not isinstance(entities, Mapping):
        return True
    return set(entities) != _SINGLE_SERIES_ENTITY_ROLES


def _bound_slots(identity: Mapping[str, Any]) -> dict[str, tuple[str, str]] | None:
    """Each `value`- or `exact`-bound slot of `identity`: its binding state and ref, by
    property. `None` when no member can be the root.

    A missing `slots` binds nothing, as dfo-apis' mint reads it. `None` for a `set`-bound
    slot (its refs yield members with identical keys, which tie under subset domination),
    a binding state other than `value`, `exact` or `unbound`, a bound slot without a
    string `ref`, and a malformed `slots`.
    """
    slots = identity.get("slots", [])
    if not isinstance(slots, list):
        return None
    bound: dict[str, tuple[str, str]] = {}
    for slot in slots:
        binding = slot.get("binding") if isinstance(slot, Mapping) else None
        state = binding.get("state") if isinstance(binding, Mapping) else None
        prop = slot.get("property") if isinstance(slot, Mapping) else None
        if not isinstance(prop, str) or state not in (*_BOUND_STATES, "unbound"):
            return None
        if state in _BOUND_STATES:
            ref = binding.get("ref") if isinstance(binding, Mapping) else None
            if not isinstance(ref, str) or not ref:
                return None
            bound[prop] = (state, ref)
    return bound


def _dcid(value: object) -> str | None:
    """The `dcid` of a Search API graph ref, or `None` when `value` is not one."""
    dcid = value.get("dcid") if isinstance(value, Mapping) else None
    return dcid if isinstance(dcid, str) and dcid else None


def _slot_property(entry: object) -> str | None:
    """The property dcid under `key.property` of a block's slot or an item's slot value."""
    key = entry.get("key") if isinstance(entry, Mapping) else None
    return _dcid(key.get("property")) if isinstance(key, Mapping) else None


def _slot_values(item: Mapping[str, Any]) -> dict[str, str] | None:
    """An elected item's `slot_values` as value dcid by property, or `None` when the field
    is missing or malformed. Search lists a bound slot's value there only when the
    StatVar's own constraint on that property is that value."""
    entries = item.get("slot_values")
    if not isinstance(entries, list):
        return None
    values: dict[str, str] = {}
    for entry in entries:
        prop = _slot_property(entry)
        value = entry.get("value") if isinstance(entry, Mapping) else None
        ref = _dcid(value.get("ref")) if isinstance(value, Mapping) else None
        if prop is None or ref is None or prop in values:
            return None
        values[prop] = ref
    return values


def _widens(prop: str, slots: object) -> bool:
    """Whether the block shows no proof that a `value` binding on `prop` denotes one code.

    `slots` is the block's `resolved.slots`. dfo-apis' mint widens a `value` ref to its
    code-index descendants, and Search reports the same walk as the slot's `expansion`,
    which it serves only when the walk added a code. So the binding denotes one code
    when the block lists the slot for `prop` with no `expansion`. A missing or malformed
    `slots`, and a block without the slot, prove nothing and count as widening.
    """
    if not isinstance(slots, list):
        return True
    matching = [
        slot for slot in slots if isinstance(slot, Mapping) and _slot_property(slot) == prop
    ]
    return len(matching) != 1 or matching[0].get("expansion") is not None


def _constraint_properties(item: Mapping[str, Any]) -> frozenset[str] | None:
    """An elected item's `constraint_properties`, or `None` when the field is missing or
    malformed. Search lists every constraint property the StatVar asserts, bound or not,
    and an empty list for a StatVar that asserts none."""
    props = item.get("constraint_properties")
    if not isinstance(props, list):
        return None
    strings = [prop for prop in props if isinstance(prop, str)]
    if len(strings) != len(props):
        return None
    return frozenset(strings)


def root_member(
    identity: Mapping[str, Any], items: Sequence[Mapping[str, Any]], *, slots: object
) -> str | None:
    """The one elected dcid that is `identity`'s family root, or `None`.

    `items` is the block's `resolved.stat_vars.items` and `slots` its `resolved.slots`.
    A root is a member whose
    constraint properties equal the identity's `value`- and `exact`-bound slot
    properties. Every member carries those properties, so a root dominates every member
    with extra ones and the family reduces to that one series. dfo-apis runs the same
    test on its confirmed members to prove a truncated member walk complete
    (`statvar_members/enumerate.py`, `_antichain_proven_complete`).

    Returns the dcid when `identity` is described-arm with entity roles exactly
    `{"observationAbout"}`, no slot is `set`-bound, `search` elected exactly one item,
    that item's `constraint_properties` equal the bound slot properties, its
    `slot_values` equal the bound slots' refs, and no `value`-bound slot widens to other
    codes (`_widens`). "Population
    of Guatemala" (no bound slot, `Count_Person` with `constraint_properties: []`) and
    "female population in Mexico" (`gender` bound, `Count_Person_Female` with
    `["gender"]`) both qualify. `Count_Person_Female_Urban` under the Mexico identity
    (`["gender", "placeOfResidenceClassification"]`) declines: its `slot_values` show
    only the bound `gender`, so the extra constraint is visible in
    `constraint_properties` alone. An item whose `constraint_properties` is missing or
    `null` (Search does not know its constraints) and two or more elected items also
    decline, so the identity routes wide.

    A slot bound on a hierarchical code family (`crsPurpose`, `crsTypeOfFinance`) is the
    reason for the last two tests. dfo-apis' mint widens a `value` ref with code-index
    descendants into a `set` binding and serves the whole subtree, so a slot whose block
    entry carries an `expansion` declines. And Search can elect a member constrained to
    a descendant code: it carries the bound properties, so its `constraint_properties`
    pass, and its `slot_values` omit the slot whose value it does not hold. An `exact`
    binding denotes its code alone and never widens.

    The block cannot show one case in which a root still mints as several series: an
    unbound property whose declaration reads `not_declared` keeps same-population twins
    apart.
    """
    if not _is_described_arm(identity) or _is_role_bearing(identity) or len(items) != 1:
        return None
    bound = _bound_slots(identity)
    item = items[0]
    if bound is None or not isinstance(item, Mapping):
        return None
    ref = item.get("ref")
    is_root = (
        _constraint_properties(item) == frozenset(bound)
        and _slot_values(item) == {prop: bound_ref for prop, (_, bound_ref) in bound.items()}
        and not any(state == "value" and _widens(prop, slots) for prop, (state, _) in bound.items())
    )
    return ref if isinstance(ref, str) and ref and is_root else None
