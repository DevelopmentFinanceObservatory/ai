"""The resubmits the planner posts to compile a pin on another basis or to ask for a
declared whole (`planner.source_reads.compile_on_basis`,
`planner.source_reads.compile_whole`), where what a resubmit posts of the part's named
sets cannot be read, or reads though the rest of the set does not, or where the part
has nothing on record to post."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from compiler.surface import ToolCallResult, ToolSurface
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from harness.scope_choices import PLANNER_AXES_BLOCK_KEY, planner_axes
from planner import source_reads as planner_source_reads
from planner.plan import PinSource, SessionState, aligned_pins, save_pin_sources
from planner.tool_results import ProblemCode
from submission.service import Realignment

from ..basis_alignment import (
    TOTAL,
    BasisCompiles,
    WholeCompiles,
    basis_block,
    derived_answer,
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
        # The set's registry id is not text, so the whole would be posted without it.
        pytest.param("provider", "destination", _UNREADABLE, None, id="group-id-not-text"),
        # The set sits on the side the whole widens, which posts no set there.
        pytest.param("donor", "origin", _UNREADABLE, {}, id="widened-side"),
        # The set states no label, so it names nothing to post.
        pytest.param("provider", "destination", _UNLABELLED, {}, id="no-label"),
        # The set's class is unknown, but the whole posts all it needs of it.
        pytest.param(
            "provider", "destination", _UNKNOWN_CLASS, {"destination": _POSTED}, id="unknown-class"
        ),
    ],
)
async def test_a_whole_whose_sets_cannot_be_read_is_refused_unposted(
    axis: str, role: str, group: dict[str, Any], posted: dict[str, Any] | None
) -> None:
    """US health aid whose block names a set whose posted identity this release cannot
    read. Its whole is refused under `groups_unreadable` with no call to Search, since
    posting the part without the set lets Search count another population. A whole
    widening the side the set sits on posts no set there, so it is asked for, as is
    one beside a set stating no label, which names nothing to post. A set naming a
    class of party this release does not know is posted by its label and ref, which
    is all the resubmit sends."""
    surface = WholeCompiles()
    state = cast(SessionState, {})
    block = _with_groups(us_2023_block("health"), {role: group})
    save_pin_sources(state, {block["spec_id"]: PinSource(block=block, settled=True)})

    read = await planner_source_reads.compile_whole(
        cast(ToolSurface, surface),
        Realignment(block["spec_id"], (), _WHOLE, _WHOLE, whole_axis=axis),
        state=state,
    )
    refused = read(state)

    if posted is not None:
        (call,) = surface.calls
        assert (call.get("entity_groups", {}), call["whole"]) == (posted, {"axis": axis})
        assert refused is None
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


# A recipient region choice as Search states it on a DAC2a part naming Africa
# (`evals/planner_entry/recordings/answers/c6468b49689420a2.json`).
_REGION_OPTIONS = [
    {
        "option_id": "region:members_and_residuals",
        "label": "Africa incl. regional and unspecified amounts",
        "binding": {"kind": "entities", "role": "destination", "add": ["DACAfricaUnallocated"]},
        "availability": {"status": "available", "detail": {}},
    },
    {
        "option_id": "region:members_only",
        "label": "Africa's countries only",
        "binding": {"kind": "entities", "role": "destination", "add": []},
        "availability": {"status": "available", "detail": {}},
    },
]


def _with_region(block: dict[str, Any], active: str) -> dict[str, Any]:
    """A copy of `block` holding a region choice on `active`."""
    copied = json.loads(json.dumps(block))
    scope = copied["resolved"]["scope_choices"]
    scope["choices"].append(
        {
            "id": f"{copied['spec_id']}:region",
            "axis": "region",
            "rule": "region_residuals",
            "provenance": "explicit",
            "active": active,
            "options": _REGION_OPTIONS,
        }
    )
    return copied


class _RegionWhole:
    """A tool surface whose `resolve_definition` answers a whole over every recipient of
    `part` as Search does: the part's figure under another spec_id, holding its
    other choices and the region on Search's default, `region:members_only`."""

    def __init__(self, part: dict[str, Any]) -> None:
        self.part = part

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(self, *, tool: Any, arguments: Any, on_progress: Any = None) -> Any:
        del tool, on_progress
        whole = _with_region(us_2023_block("health"), "region:members_only")
        whole["spec_id"] = f"spec_{'e' * 32}"
        whole["resolved"]["scope_choices"]["choices"][-1]["provenance"] = "default"
        answer = derived_answer(self.part, whole, axis=arguments["whole"]["axis"])
        envelope = {"parts": [{"status": "definite", "spec": answer}]}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: envelope})


async def test_a_whole_reads_as_the_planners_only_where_it_holds_the_parts_option() -> None:
    """US health aid to Africa and countries' regional aid, the planner having settled
    its sector and region, as a share of its whole over every recipient: Search's
    whole holds the sector as the part does and the region on its own default, so
    the whole reads as settled by the planner on the sector alone."""
    part = _with_region(us_2023_block("health"), "region:members_and_residuals")
    part[PLANNER_AXES_BLOCK_KEY] = ["region", "sector"]
    state = cast(SessionState, {})
    save_pin_sources(state, {part["spec_id"]: PinSource(block=part, settled=True)})

    read = await planner_source_reads.compile_whole(
        cast(ToolSurface, _RegionWhole(part)),
        Realignment(part["spec_id"], (), _WHOLE, _WHOLE, whole_axis="recipient"),
        state=state,
    )

    assert read(state) is None
    (held,) = aligned_pins(state)
    assert planner_axes(held.digest["spec_block"]) == {"sector"}
