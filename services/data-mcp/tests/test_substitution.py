"""Tests for `data_mcp.substitution`: the single-series rule.

This module is meant to stay a leaf -- standard library only, no import of
the upstream client, the definition store, fastmcp, or mcp -- so `harness.tool_contracts`
can reach `root_member` without dragging data-mcp's own transport stack into the harness
import chain. `services/harness/tests/test_tool_contracts.py` carries the assertion that
the harness side actually stays a leaf; this file covers the rule itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_mcp.substitution import root_member

_FIXTURES = Path(__file__).parent / "fixtures"


def _slot_value(prop: str, ref: str) -> dict[str, Any]:
    return {
        "key": {"axis": "how", "property": {"dcid": prop, "label": prop}, "label": prop},
        "value": {"ref": {"dcid": ref, "label": ref}, "value_kind": "enum_value"},
    }


def _unbound(prop: str) -> dict[str, Any]:
    return {"property": prop, "binding": {"state": "unbound"}}


def _block_slot(prop: str, ref: str, *, added: list[str] | None = None) -> dict[str, Any]:
    """A `value`-bound entry of a block's `resolved.slots`, with the `expansion` Search
    serves when the code index walk from `ref` added descendant codes."""
    slot: dict[str, Any] = {
        "key": {"axis": "how", "property": {"dcid": prop, "label": prop}, "label": prop},
        "binding": {
            "kind": "value",
            "value": {"ref": {"dcid": ref, "label": ref}, "value_kind": "enum_value"},
        },
    }
    if added is not None:
        slot["expansion"] = {"added": [{"dcid": dcid, "label": dcid} for dcid in added]}
    return slot


# The Mexico block's `resolved.slots`, as far as the root test reads them: `Female` is a
# leaf of a flat code family, so its slot carries no `expansion`.
_MEXICO_SLOTS = [_block_slot("gender", "Female")]


# "female population in Mexico" as dev search resolves it: `gender` bound to `Female`,
# five demographic slots unbound. Search fills an item's `slot_values` from the bound
# slots alone and lists every constraint the StatVar asserts in `constraint_properties`.
def _mexico_identity(**overrides: Any) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "Person",
            "measured_property": "count",
            "stat_type": "measuredValue",
        },
        "entities": {"observationAbout": ["country/MEX"]},
        "requested_window": {"kind": "all"},
        "slots": [
            _unbound("age"),
            {"property": "gender", "binding": {"state": "value", "ref": "Female"}},
            _unbound("literacyStatus"),
            _unbound("maritalStatus"),
            _unbound("nativity"),
            _unbound("placeOfResidenceClassification"),
        ],
    }
    identity.update(overrides)
    return identity


_FEMALE_ITEM = {
    "ref": "Count_Person_Female",
    "label": "Female population",
    "slot_values": [_slot_value("gender", "Female")],
    "constraint_properties": ["gender"],
}

# A member Search can elect under the Mexico identity. Its urbanisation constraint sits
# on an unbound slot, so `slot_values` shows `gender` alone.
_FEMALE_URBAN_ITEM = {
    "ref": "Count_Person_Female_Urban",
    "label": "Female urban population",
    "slot_values": [_slot_value("gender", "Female")],
    "constraint_properties": ["gender", "placeOfResidenceClassification"],
}


def _guatemala_spec() -> dict[str, Any]:
    envelope = json.loads((_FIXTURES / "resolve_population_guatemala.json").read_text())
    return envelope["parts"][0]["spec"]


def test_the_unconstrained_item_under_an_unbound_identity_is_the_root() -> None:
    """The recorded "population of Guatemala" envelope: twelve unbound slots and one
    elected item with `slot_values: []` and `constraint_properties: []`."""
    spec = _guatemala_spec()
    items = spec["resolved"]["stat_vars"]["items"]
    assert (
        root_member(spec["definition"]["identity"], items, slots=spec["resolved"]["slots"])
        == "Count_Person"
    )


def test_an_item_constrained_by_the_bound_slots_alone_is_the_root() -> None:
    assert (
        root_member(_mexico_identity(), [_FEMALE_ITEM], slots=_MEXICO_SLOTS)
        == "Count_Person_Female"
    )


def test_an_exact_binding_counts_as_bound() -> None:
    identity = _mexico_identity()
    identity["slots"][1]["binding"] = {"state": "exact", "ref": "Female"}
    assert root_member(identity, [_FEMALE_ITEM], slots=_MEXICO_SLOTS) == "Count_Person_Female"


def test_an_item_constrained_beyond_the_bound_slots_is_not_the_root() -> None:
    """`Count_Person_Female` dominates `Count_Person_Female_Urban`, so the mint resolves
    more than the elected series. The item's `slot_values` match the root's, and only
    `constraint_properties` shows the extra constraint."""
    assert _FEMALE_URBAN_ITEM["slot_values"] == _FEMALE_ITEM["slot_values"]
    assert root_member(_mexico_identity(), [_FEMALE_URBAN_ITEM], slots=_MEXICO_SLOTS) is None


def test_a_constrained_item_under_an_unbound_identity_is_not_the_root() -> None:
    """Under the Guatemala identity nothing is bound, so Search serves `slot_values: []`
    for `Count_Person_Female` too."""
    identity = _guatemala_spec()["definition"]["identity"]
    item = {"ref": "Count_Person_Female", "slot_values": [], "constraint_properties": ["gender"]}
    assert root_member(identity, [item], slots=_MEXICO_SLOTS) is None


def test_an_item_without_constraint_properties_is_not_the_root() -> None:
    """A dfo-apis Search that predates the field serves `slot_values` alone, which cannot
    tell the root from a dominated member, so the identity routes wide."""
    item = {key: value for key, value in _FEMALE_ITEM.items() if key != "constraint_properties"}
    assert root_member(_mexico_identity(), [item], slots=_MEXICO_SLOTS) is None


def test_a_set_bound_slot_has_no_root() -> None:
    """Each ref of a `set` binding yields members with identical keys, which tie."""
    identity = _mexico_identity()
    identity["slots"][1]["binding"] = {"state": "set", "refs": ["Female", "Male"]}
    assert root_member(identity, [_FEMALE_ITEM], slots=_MEXICO_SLOTS) is None


def test_an_unknown_binding_state_has_no_root() -> None:
    identity = _mexico_identity()
    identity["slots"][0]["binding"] = {"state": "range"}
    assert root_member(identity, [_FEMALE_ITEM], slots=_MEXICO_SLOTS) is None


def test_two_elected_items_have_no_root() -> None:
    """Even when one of them would pass the root test on its own."""
    assert (
        root_member(_mexico_identity(), [_FEMALE_ITEM, _FEMALE_URBAN_ITEM], slots=_MEXICO_SLOTS)
        is None
    )


def test_nothing_elected_has_no_root() -> None:
    assert root_member(_mexico_identity(), [], slots=_MEXICO_SLOTS) is None


def test_a_pinned_arm_identity_has_no_root() -> None:
    identity = _mexico_identity(shape={"id": "Count_Person_Female", "kind": "stat_var"})
    assert root_member(identity, [_FEMALE_ITEM], slots=_MEXICO_SLOTS) is None


def test_a_role_bearing_identity_has_no_root() -> None:
    identity = _mexico_identity(entities={"origin": ["country/FRA"]})
    assert root_member(identity, [_FEMALE_ITEM], slots=_MEXICO_SLOTS) is None


def test_roles_beyond_the_whitelist_have_no_root() -> None:
    """`_is_role_bearing` is a whitelist of exactly `{"observationAbout"}`, not a
    denylist of the two roles this module happened to name first -- an aid flow's
    `provider`/`recipient`/`donor` roles must decline exactly like `origin`/
    `destination` does, with no need to enumerate every named-family shape."""
    identity = _mexico_identity(
        entities={"observationAbout": ["country/KEN"], "donor": ["country/FRA"]}
    )
    assert root_member(identity, [_FEMALE_ITEM], slots=_MEXICO_SLOTS) is None


def test_a_sole_non_observation_about_role_has_no_root() -> None:
    identity = _mexico_identity(entities={"recipient": ["country/KEN"]})
    assert root_member(identity, [_FEMALE_ITEM], slots=_MEXICO_SLOTS) is None


def test_missing_entities_have_no_root() -> None:
    """No `entities` at all is not the same as `{"observationAbout"}` -- the
    whitelist declines it rather than treating an absent key as harmless."""
    identity = _mexico_identity()
    del identity["entities"]
    assert root_member(identity, [_FEMALE_ITEM], slots=_MEXICO_SLOTS) is None


def test_missing_slots_bind_nothing() -> None:
    """dfo-apis' mint reads a missing `slots` as no slots, so the unconstrained item is
    the root and a gendered one is not."""
    identity = _mexico_identity()
    del identity["slots"]
    unconstrained = {"ref": "Count_Person", "slot_values": [], "constraint_properties": []}
    female = {"ref": "Count_Person_Female", "slot_values": [], "constraint_properties": ["gender"]}
    assert root_member(identity, [unconstrained], slots=_MEXICO_SLOTS) == "Count_Person"
    assert root_member(identity, [female], slots=_MEXICO_SLOTS) is None


def test_malformed_slots_have_no_root() -> None:
    identity = _mexico_identity(slots={"gender": "Female"})
    assert root_member(identity, [_FEMALE_ITEM], slots=_MEXICO_SLOTS) is None


def test_an_item_whose_constraints_search_does_not_know_is_not_the_root() -> None:
    """Search serves `constraint_properties: null` for a StatVar whose constraints it
    did not read, under a bound identity and an unbound one alike."""
    female = {**_FEMALE_ITEM, "constraint_properties": None}
    assert root_member(_mexico_identity(), [female], slots=_MEXICO_SLOTS) is None
    spec = _guatemala_spec()
    unknown = {**spec["resolved"]["stat_vars"]["items"][0], "constraint_properties": None}
    identity = spec["definition"]["identity"]
    assert root_member(identity, [unknown], slots=spec["resolved"]["slots"]) is None


# "Education aid" binds `crsPurpose` to `CRSPurposeCode110`, a hierarchical code whose
# subtree the mint serves, with `observationAbout` as its one entity role here.
def _education_identity(state: str = "value") -> dict[str, Any]:
    return _mexico_identity(
        slots=[
            {"property": "crsPurpose", "binding": {"state": state, "ref": "CRSPurposeCode110"}},
            _unbound("crsTypeOfFinance"),
        ]
    )


_EDUCATION_ITEM = {
    "ref": "Amount_FinancialFlow_CRSPurposeCode110",
    "slot_values": [_slot_value("crsPurpose", "CRSPurposeCode110")],
    "constraint_properties": ["crsPurpose"],
}
_EDUCATION_SLOTS = [_block_slot("crsPurpose", "CRSPurposeCode110", added=["CRSPurposeCode11110"])]


def test_a_bound_value_that_widens_to_descendants_has_no_root() -> None:
    """The item holds the bound code itself, and the mint still serves every descendant
    code's member, which the slot's `expansion` shows."""
    assert root_member(_education_identity(), [_EDUCATION_ITEM], slots=_EDUCATION_SLOTS) is None
    leaf = [_block_slot("crsPurpose", "CRSPurposeCode110")]
    assert root_member(_education_identity(), [_EDUCATION_ITEM], slots=leaf) == (
        "Amount_FinancialFlow_CRSPurposeCode110"
    )


def test_an_exact_binding_does_not_widen() -> None:
    """`exact` denotes its code alone, whatever the block's slot shows."""
    assert root_member(_education_identity("exact"), [_EDUCATION_ITEM], slots=_EDUCATION_SLOTS) == (
        "Amount_FinancialFlow_CRSPurposeCode110"
    )


def test_an_item_constrained_to_a_descendant_code_is_not_the_root() -> None:
    """Search elected the one confirmed member, which holds a descendant of the bound
    code. It asserts the bound property, so `constraint_properties` match, and Search
    leaves the slot out of `slot_values` because the value differs."""
    descendant = {
        "ref": "Amount_FinancialFlow_CRSPurposeCode11110",
        "slot_values": [],
        "constraint_properties": ["crsPurpose"],
    }
    leaf = [_block_slot("crsPurpose", "CRSPurposeCode110")]
    assert root_member(_education_identity(), [descendant], slots=leaf) is None


def test_slot_values_naming_another_value_than_the_binding_are_not_the_root() -> None:
    item = {**_FEMALE_ITEM, "slot_values": [_slot_value("gender", "Male")]}
    assert root_member(_mexico_identity(), [item], slots=_MEXICO_SLOTS) is None


def test_missing_or_malformed_slot_values_have_no_root() -> None:
    missing = {key: value for key, value in _FEMALE_ITEM.items() if key != "slot_values"}
    assert root_member(_mexico_identity(), [missing], slots=_MEXICO_SLOTS) is None
    malformed = {**_FEMALE_ITEM, "slot_values": [{"key": "gender", "value": "Female"}]}
    assert root_member(_mexico_identity(), [malformed], slots=_MEXICO_SLOTS) is None


def test_a_bound_slot_the_block_does_not_list_has_no_root() -> None:
    """Without the block's slot nothing shows whether the binding widens."""
    assert root_member(_mexico_identity(), [_FEMALE_ITEM], slots=None) is None
    assert root_member(_mexico_identity(), [_FEMALE_ITEM], slots=[]) is None
    assert root_member(_mexico_identity(), [_FEMALE_ITEM], slots=[_block_slot("age", "Y0")]) is None
