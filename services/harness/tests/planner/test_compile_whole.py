"""The resubmits the planner posts to compile a pin on another basis or as a declared
whole (`planner.agent.compile_on_basis`, `planner.agent.compile_whole`), where what a
resubmit posts of the part's named sets cannot be read, or reads though the rest of
the set does not."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from compiler.surface import ToolSurface
from planner import agent as planner_agent
from planner.plan import PinSource, SessionState, save_pin_sources
from planner.tool_results import ProblemCode
from submission.service import Realignment

from ..basis_alignment import TOTAL, BasisCompiles, WholeCompiles, basis_block, us_2023_block

_WHOLE = "whole"

# A named set stating no label, so what a resubmit posts of it does not read.
_UNREADABLE = {"ref": "africa", "total": "DACAfricaTotal"}
# A named set stating a class of party outside `common.place_type.PartyClass`: the
# set does not read, but what a resubmit posts of it does.
_UNKNOWN_CLASS = {"label": "Africa", "ref": "africa", "whole_of": ["sovereign_fund"]}
_POSTED = {"label": "Africa", "ref": "africa"}


def _with_groups(block: dict[str, Any], groups: dict[str, Any]) -> dict[str, Any]:
    """A copy of `block` naming `groups` as its `resolved.entity_groups`."""
    copied = json.loads(json.dumps(block))
    copied["resolved"]["entity_groups"] = groups
    return copied


@pytest.mark.parametrize(
    ("axis", "role", "group", "posted"),
    [
        # The set sits on a role the whole keeps, so the whole would lose it.
        pytest.param("sector", "destination", _UNREADABLE, None, id="kept-role"),
        # The set sits on the role the whole opens, which posts no set there.
        pytest.param("donor", "origin", _UNREADABLE, {}, id="opened-role"),
        # The set's class is unknown, but the whole posts all it needs of it.
        pytest.param(
            "sector", "destination", _UNKNOWN_CLASS, {"destination": _POSTED}, id="unknown-class"
        ),
    ],
)
async def test_a_whole_whose_kept_sets_cannot_be_read_is_refused_unposted(
    axis: str, role: str, group: dict[str, Any], posted: dict[str, Any] | None
) -> None:
    """US health aid whose block names a set whose posted identity this release cannot
    read. A whole keeping the set's role is refused under `groups_unreadable` with no
    call to Search, since posting the identity without the set lets Search count
    another population. A whole opening that role posts no set there, so it compiles.
    A set naming a class of party this release does not know is posted by its label
    and ref, which is all the resubmit sends."""
    surface = WholeCompiles()
    state = cast(SessionState, {})
    block = _with_groups(us_2023_block("health"), {role: group})
    save_pin_sources(state, {block["spec_id"]: PinSource(block=block, settled=True)})

    refused = await planner_agent.compile_whole(
        cast(ToolSurface, surface),
        Realignment(block["spec_id"], (), _WHOLE, _WHOLE, whole_axis=axis),
        state=state,
    )

    if posted is not None:
        (call,) = surface.calls
        assert call.get("entity_groups", {}) == posted
        return
    assert surface.calls == []
    assert refused is not None
    assert refused["code"] == ProblemCode.SHARE_WHOLE_UNDERIVABLE.value
    assert cast(dict[str, Any], refused["detail"])["whole"]["cause"] == "groups_unreadable"


@pytest.mark.parametrize(
    ("group", "posted"),
    [
        pytest.param(_UNREADABLE, None, id="no-label"),
        pytest.param(_UNKNOWN_CLASS, {"origin": _POSTED}, id="unknown-class"),
    ],
)
async def test_a_basis_compile_whose_sets_cannot_be_read_is_refused_unposted(
    group: dict[str, Any], posted: dict[str, Any] | None
) -> None:
    """The US donor total, counted in grant equivalents, is wanted gross for a share,
    and its block names a set whose posted identity this release cannot read. The
    compile is refused as an unsupported ask with no call to Search, since posting
    the identity without the set lets Search count another population. A set naming
    a class of party this release does not know is posted by its label and ref."""
    surface = BasisCompiles()
    state = cast(SessionState, {})
    block = _with_groups(
        basis_block(TOTAL, narrow=False, active="measure_basis:grant_equivalent"),
        {"origin": group},
    )
    save_pin_sources(state, {TOTAL: PinSource(block=block, settled=True)})

    refused = await planner_agent.compile_on_basis(
        cast(ToolSurface, surface),
        Realignment(TOTAL, ("measure_basis:gross",), "total", "share"),
        state=state,
    )

    if posted is not None:
        (call,) = surface.calls
        assert call["entity_groups"] == posted
        return
    assert surface.calls == []
    assert refused is not None
    assert refused["code"] == ProblemCode.OPERATION_UNSUPPORTED.value
    assert cast(dict[str, Any], refused["detail"])["basis"]["requirement_id"] == "total"
