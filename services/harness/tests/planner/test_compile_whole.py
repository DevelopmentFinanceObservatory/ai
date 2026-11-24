"""The resubmits the planner posts to compile a pin on another basis or to ask for a
declared whole (`planner.source_reads.compile_on_basis`,
`planner.source_reads.compile_whole`), where what a resubmit posts of the part's named
sets cannot be read, or reads though the rest of the set does not, or where the part
has nothing on record to post."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from compiler.surface import ToolSurface
from planner import source_reads as planner_source_reads
from planner.plan import PinSource, SessionState, save_pin_sources
from planner.tool_results import ProblemCode
from submission.service import Realignment

from ..basis_alignment import TOTAL, BasisCompiles, WholeCompiles, basis_block, us_2023_block

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
    ("group", "posted"),
    [
        # The set's registry id is not text, so the whole would be posted without it.
        pytest.param(_UNREADABLE, None, id="group-id-not-text"),
        # The set states no label, so it names nothing to post.
        pytest.param(_UNLABELLED, {}, id="no-label"),
        # The set's class is unknown, but the whole posts all it needs of it.
        pytest.param(_UNKNOWN_CLASS, {"destination": _POSTED}, id="unknown-class"),
    ],
)
async def test_a_whole_whose_sets_cannot_be_read_is_refused_unposted(
    group: dict[str, Any], posted: dict[str, Any] | None
) -> None:
    """US health aid whose block names a set whose posted identity this release cannot
    read. Its whole is refused under `groups_unreadable` with no call to Search, since
    posting the part without the set lets Search count another population. A set
    stating no label names nothing to post, and one naming a class of party this
    release does not know is posted by its label and ref, which is all the resubmit
    sends, so both are posted with the whole request."""
    surface = WholeCompiles()
    state = cast(SessionState, {})
    block = _with_groups(us_2023_block("health"), {"destination": group})
    save_pin_sources(state, {block["spec_id"]: PinSource(block=block, settled=True)})

    read = await planner_source_reads.compile_whole(
        cast(ToolSurface, surface),
        Realignment(block["spec_id"], (), _WHOLE, _WHOLE, whole_axis="provider"),
        state=state,
    )
    refused = read(state)

    if posted is not None:
        (call,) = surface.calls
        assert (call.get("entity_groups", {}), call["whole"]) == (posted, {"axis": "provider"})
        return
    assert surface.calls == []
    assert refused is not None
    assert refused["code"] == ProblemCode.SHARE_WHOLE_UNDERIVABLE.value
    assert cast(dict[str, Any], refused["detail"])["whole"]["cause"] == "groups_unreadable"


async def test_a_whole_whose_part_has_no_definition_on_record_is_refused_unposted() -> None:
    """A whole whose part's pin has no block on record has nothing to post: it is
    refused under `part_unread` with no call to Search."""
    surface = WholeCompiles()
    state = cast(SessionState, {})

    read = await planner_source_reads.compile_whole(
        cast(ToolSurface, surface),
        Realignment(us_2023_block("health")["spec_id"], (), _WHOLE, _WHOLE, whole_axis="provider"),
        state=state,
    )
    refused = read(state)

    assert surface.calls == []
    assert refused is not None
    assert cast(dict[str, Any], refused["detail"])["whole"]["cause"] == "part_unread"


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

    read = await planner_source_reads.compile_on_basis(
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
