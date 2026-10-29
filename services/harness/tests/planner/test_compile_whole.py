"""The resubmits the planner posts to compile a pin on another basis or as a declared
whole (`planner.agent.compile_on_basis`, `planner.agent.compile_whole`), where what a
resubmit posts of the part's named sets cannot be read, or reads though the rest of
the set does not."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from compiler.surface import ToolSurface
from harness.scope_choices import mark_planner_axes, planner_axes
from planner import agent as planner_agent
from planner.plan import (
    AlignedPin,
    HeldSources,
    PinSource,
    SessionState,
    aligned_pins,
    pin_sources,
    save_aligned_pin,
    save_pin_sources,
)
from planner.tool_results import ProblemCode
from submission.service import Realignment

from ..basis_alignment import (
    TOTAL,
    TOTAL_GROSS,
    BasisCompiles,
    WholeCompiles,
    basis_block,
    us_2023_block,
)

_WHOLE = "whole"

# A named set stating a registry id that is not text, so what a resubmit posts of it
# does not read.
_UNREADABLE = {"label": "Africa", "ref": "africa", "group_id": 7}
# A named set stating no label, which names nothing a resubmit can post.
_UNLABELLED = {"ref": "africa", "total": "DACAfricaTotal"}
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
        # The set states no label, so the whole is posted without it.
        pytest.param("sector", "destination", _UNLABELLED, {}, id="no-label"),
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
    another population. A whole opening that role posts no set there, so it compiles,
    as does one keeping a set stating no label, which names nothing to post.
    A set naming a class of party this release does not know is posted by its label
    and ref, which is all the resubmit sends."""
    surface = WholeCompiles()
    state = cast(SessionState, {})
    block = _with_groups(us_2023_block("health"), {role: group})
    save_pin_sources(state, {block["spec_id"]: PinSource(block=block, settled=True)})

    read = await planner_agent.compile_whole(
        cast(ToolSurface, surface),
        Realignment(block["spec_id"], (), _WHOLE, _WHOLE, whole_axis=axis),
        state=state,
    )
    refused = read(state)

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
        pytest.param(_UNREADABLE, None, id="group-id-not-text"),
        pytest.param(_UNLABELLED, {}, id="no-label"),
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
    a class of party this release does not know is posted by its label and ref, and
    one stating no label, which names nothing to post, is left out of the post."""
    surface = BasisCompiles()
    state = cast(SessionState, {})
    block = _with_groups(
        basis_block(TOTAL, narrow=False, active="measure_basis:grant_equivalent"),
        {"origin": group},
    )
    save_pin_sources(state, {TOTAL: PinSource(block=block, settled=True)})

    read = await planner_agent.compile_on_basis(
        cast(ToolSurface, surface),
        Realignment(TOTAL, ("measure_basis:gross",), "total", "share"),
        state=state,
    )
    refused = read(state)

    if posted is not None:
        (call,) = surface.calls
        assert call.get("entity_groups", {}) == posted
        return
    assert surface.calls == []
    assert refused is not None
    assert refused["code"] == ProblemCode.OPERATION_UNSUPPORTED.value
    assert cast(dict[str, Any], refused["detail"])["basis"]["requirement_id"] == "total"


@pytest.mark.parametrize("settled", [True, False], ids=["settled", "candidate"])
@pytest.mark.parametrize("whole", [False, True], ids=["basis", "whole"])
async def test_a_compile_answered_with_a_shown_spec_id_leaves_it_as_shown(
    whole: bool, settled: bool
) -> None:
    """Search answers a compile, the US donor total on gross or US health aid over
    every sector, with a spec_id the planner was shown, settled or only offered as a
    candidate. The shown source stands as shown either way, since a pin of the plan's
    may rest on it, and the compile is held by what it was compiled for, so the
    requirement it was made for reads it (`planner.plan.HeldSources`)."""
    state = cast(SessionState, {})
    if whole:
        surface: Any = WholeCompiles()
        origin = us_2023_block("health")
        compiled = us_2023_block("whole")["spec_id"]
        realignment = Realignment(origin["spec_id"], (), _WHOLE, _WHOLE, whole_axis="sector")
    else:
        surface = BasisCompiles()
        origin = basis_block(TOTAL, narrow=False, active="measure_basis:grant_equivalent")
        compiled = TOTAL_GROSS
        realignment = Realignment(TOTAL, ("measure_basis:gross",), "total", "share")
    shown = PinSource(
        block=basis_block(compiled, narrow=False, active="measure_basis:net"), settled=settled
    )
    save_pin_sources(
        state, {origin["spec_id"]: PinSource(block=origin, settled=True), compiled: shown}
    )

    read = await planner_agent.compile_on_basis(
        cast(ToolSurface, surface), realignment, state=state
    )
    refused = read(state)

    assert refused is None
    (held,) = aligned_pins(state)
    assert held.spec_id == compiled
    (part,) = held.digest["part_digests"]
    assert pin_sources(state)[compiled] == shown
    read = HeldSources.of(state).compiled_source(held.key, compiled)
    assert read is not None and read.block == part["spec_block"]


async def test_a_whole_of_an_aligned_part_reads_the_part_from_its_compile() -> None:
    """US health aid, moved by alignment onto a spec_id the planner was shown as a
    search's answer, is declared the part of its total over every sector. The whole is
    compiled from the part's compile, which marks the provider scope as the planner's,
    so the whole marks it too; the shown source marks no axis."""
    state = cast(SessionState, {})
    part = us_2023_block("health")
    origin = us_2023_block("health_inclusive")
    save_pin_sources(
        state,
        {
            part["spec_id"]: PinSource(block=part, settled=True),
            origin["spec_id"]: PinSource(block=origin, settled=True),
        },
    )
    digest: dict[str, Any] = {"part_digests": [{"spec_block": part}]}
    mark_planner_axes(digest, {"provider"})
    save_aligned_pin(
        state,
        AlignedPin(
            origin=origin["spec_id"],
            option_ids=("provider:bilateral",),
            spec_id=part["spec_id"],
            digest=digest,
            records_origin=True,
        ),
    )

    read = await planner_agent.compile_whole(
        cast(ToolSurface, WholeCompiles()),
        Realignment(
            part["spec_id"],
            (),
            _WHOLE,
            _WHOLE,
            whole_axis="sector",
            part_origin=origin["spec_id"],
            part_option_ids=("provider:bilateral",),
        ),
        state=state,
    )
    refused = read(state)

    assert refused is None
    whole = next(pin for pin in aligned_pins(state) if pin.whole_axis == "sector")
    (answered,) = whole.digest["part_digests"]
    assert planner_axes(answered["spec_block"]) == {"provider"}
