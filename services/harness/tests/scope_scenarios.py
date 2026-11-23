"""US health aid to Africa's scope choices and Search's answer to a switch of
them, which the plan-card switch and a chip both resubmit, and the sets of
Africa a reading's block may name."""

from __future__ import annotations

import copy
from typing import Any

from data_mcp.definition_store.identity import canonicalize, spec_id_for

from .test_turn_records import _switchable_scope


def _hashed_scope() -> dict[str, Any]:
    """US health aid to Africa's scope (`_switchable_scope`), its base under the
    spec_id its identity hashes to, so data-mcp takes it as posted: health as
    CRS 120 alone from the user's words, Africa's countries only carried,
    gross disbursements by default."""
    scope = _switchable_scope()
    base = spec_id_for(canonicalize(scope["base"]["identity"]))
    scope["id"] = scope["base"]["spec_id"] = base
    for choice in scope["choices"]:
        choice["id"] = f"{base}:{choice['axis']}"
    return scope


def _switched_answer(scope: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The spec_id and envelope Search answers for `scope` switched to CRS
    120 and 130: grant equivalents become the basis, since gross is out of
    reach beside the switch."""
    switched = copy.deepcopy(scope)
    identity = copy.deepcopy(scope["base"]["identity"])
    identity["slots"] = [s for s in identity["slots"] if s["property"] != "nettingBasis"]
    identity["shape"]["measured_property"] = "grantEquivalent"
    spec_id = spec_id_for(canonicalize(identity))
    sector, _, basis = switched["choices"]
    sector["active"] = "sector:120+130"
    basis["active"] = "measure_basis:grant_equivalent"
    basis["options"][0]["availability"] = {
        "status": "unavailable",
        "code": "conflicts_with",
        "detail": {"entry": "sector:120+130"},
    }
    switched["entries"] = [
        switched["entries"][1],
        {
            "axis": "sector",
            "option_id": "sector:120+130",
            "provenance": "explicit",
            "concept": "health",
            "origin": {"kind": "choice", "choice_id": sector["id"]},
        },
    ]
    switched["compiled"] = {"spec_id": spec_id, "identity": identity, "components": []}
    block = {
        "spec_id": spec_id,
        "sentence": "US health aid to Africa",
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": identity},
        "resolved": {"scope_choices": switched},
    }
    envelope = {
        "schema_version": 1,
        "status": "definite",
        "parts": [{"status": "definite", "variable_text": scope["id"], "spec": block}],
    }
    return spec_id, envelope


_AFRICA_TOTAL = {"label": "Africa", "ref": "africa", "total": "DACAfricaTotal"}
# Africa as a set naming a class of party outside `common.place_type.PartyClass`: the
# set does not read, but what a resubmit posts of it does.
_AFRICA_UNKNOWN_CLASS = {"label": "Africa", "ref": "africa", "whole_of": ["sovereign_fund"]}
# Africa as a set stating a registry id that is not text, so what a resubmit posts of
# it does not read.
_AFRICA_UNPOSTABLE = {"label": "Africa", "ref": "africa", "group_id": 7}
