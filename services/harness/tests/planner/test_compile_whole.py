"""The resubmits the planner posts to compile a pin on another basis or as a declared
whole (`planner.agent.compile_on_basis`, `planner.agent.compile_whole`), where the
part's named sets cannot be read."""

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

# A named set stating a class of party outside `common.place_type.PartyClass`.
_UNREADABLE = {"label": "Africa", "ref": "africa", "whole_of": ["sovereign_fund"]}


def _with_groups(block: dict[str, Any], groups: dict[str, Any]) -> dict[str, Any]:
    """A copy of `block` naming `groups` as its `resolved.entity_groups`."""
    copied = json.loads(json.dumps(block))
    copied["resolved"]["entity_groups"] = groups
    return copied


@pytest.mark.parametrize(
    ("axis", "role", "calls"),
    [
        # The set sits on a role the whole keeps, so the whole would lose it.
        pytest.param("sector", "destination", 0, id="kept-role"),
        # The set sits on the role the whole opens, which posts no set there.
        pytest.param("donor", "origin", 1, id="opened-role"),
    ],
)
async def test_a_whole_whose_kept_sets_cannot_be_read_is_refused_unposted(
    axis: str, role: str, calls: int
) -> None:
    """US health aid whose block names a set this release cannot read. A whole
    keeping the set's role is refused under `groups_unreadable` with no call to
    Search, since posting the identity without the set lets Search count another
    population. A whole opening that role posts no set there, so it compiles."""
    surface = WholeCompiles()
    state = cast(SessionState, {})
    block = _with_groups(us_2023_block("health"), {role: _UNREADABLE})
    save_pin_sources(state, {block["spec_id"]: PinSource(block=block, settled=True)})

    refused = await planner_agent.compile_whole(
        cast(ToolSurface, surface),
        Realignment(block["spec_id"], (), _WHOLE, _WHOLE, whole_axis=axis),
        state=state,
    )

    assert len(surface.calls) == calls
    if calls:
        assert "entity_groups" not in surface.calls[0]
        return
    assert refused is not None
    assert refused["code"] == ProblemCode.SHARE_WHOLE_UNDERIVABLE.value
    assert cast(dict[str, Any], refused["detail"])["whole"]["cause"] == "groups_unreadable"


async def test_a_basis_compile_whose_sets_cannot_be_read_is_refused_unposted() -> None:
    """The US donor total, counted in grant equivalents, is wanted gross for a share,
    and its block names a set this release cannot read. The compile is refused as
    an unsupported ask with no call to Search, since posting the identity without
    the set lets Search count another population."""
    surface = BasisCompiles()
    state = cast(SessionState, {})
    block = _with_groups(
        basis_block(TOTAL, narrow=False, active="measure_basis:grant_equivalent"),
        {"origin": _UNREADABLE},
    )
    save_pin_sources(state, {TOTAL: PinSource(block=block, settled=True)})

    refused = await planner_agent.compile_on_basis(
        cast(ToolSurface, surface),
        Realignment(TOTAL, ("measure_basis:gross",), "total", "share"),
        state=state,
    )

    assert surface.calls == []
    assert refused is not None
    assert refused["code"] == ProblemCode.OPERATION_UNSUPPORTED.value
    assert cast(dict[str, Any], refused["detail"])["basis"]["requirement_id"] == "total"
