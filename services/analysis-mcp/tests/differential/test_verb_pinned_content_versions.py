"""One real output `cv_` pinned per verb: each tool runs end to end on stored
frames of non-round values (resolve, engine, identity, trace, store), and the
output frame's `frame_content_version` must equal the pinned string.

The `cv_` is minted from the output frame's `frame_sha256` alone, so it moves
when any output bit moves. The result's own
`content_version` (`ocv_`) is left unpinned: it hashes the op-spec, the
input `cv_`s and the `ExecutionIdentity`, none of which an engine rewrite
touches.
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.compare_tool import compare
from analysis_mcp.filter_tool import filter_verb
from analysis_mcp.fixtures import _fixture_ref, _fixture_trace
from analysis_mcp.growth_tool import growth
from analysis_mcp.join_tool import join
from analysis_mcp.op_spec import OrderedThresholdPredicate
from analysis_mcp.rank_tool import rank
from analysis_mcp.share_of_tool import share_of
from common.materialisation import InMemoryMaterialisationStore, set_store
from differential.cases import DONORS, SV_A, SV_B, YEARS, frame_from_spec, grid
from fastmcp.tools import ToolResult

PINNED_FRAME_CONTENT_VERSIONS = {
    "aggregate": "cv_be5edeb0270097c62f272a8ff1e18996",
    "compare": "cv_2a4a3d3492aa710ce332b50a4b293330",
    "growth": "cv_efdffe291ba60fbc9c971a845a362aec",
    "rank": "cv_0b10a81e235a65529ef030eb8d43716e",
    "share_of": "cv_8cd749dd9481a54c5f8ac37607b2df83",
    "join": "cv_b4842efedf2cb9c80cb241fc75ca9a22",
    "filter": "cv_27bbde40bac36ae5e332a842eea3f420",
}

_WINDOW = (YEARS[2], YEARS[6])


def _non_round_frame_spec(stat_var: str, seed: int) -> dict[str, Any]:
    return grid(
        random.Random(seed),
        roles={"donor": DONORS[:4]},
        periods=YEARS[2:7],
        stat_vars=(stat_var,),
        values="positive",
    )


@pytest.fixture
async def refs():
    store = InMemoryMaterialisationStore()
    handles: dict[str, dict[str, str]] = {}
    for name, stat_var, seed in (("pinned_left", SV_A, 1), ("pinned_right", SV_B, 2)):
        frame = frame_from_spec(_non_round_frame_spec(stat_var, seed))
        ref = _fixture_ref(name)
        await store.put(ref, frame=frame, trace=_fixture_trace(name, frame), writer="fixtures")
        handles[name] = {"spec_id": ref.spec_id, "content_version": ref.content_version}
    set_store(store)
    yield handles
    set_store(None)


_CALLS: dict[str, Callable[[dict[str, dict[str, str]]], Awaitable[ToolResult]]] = {
    "aggregate": lambda refs: aggregate(
        frame_ref=refs["pinned_left"], aggregation="sum", group_by=("donor",)
    ),
    "compare": lambda refs: compare(
        frame_ref=refs["pinned_left"],
        left_entity={"donor": DONORS[0]},
        right_entity={"donor": DONORS[1]},
    ),
    "growth": lambda refs: growth(
        frame_ref=refs["pinned_left"],
        window_start=_WINDOW[0],
        window_end=_WINDOW[1],
        entity_by=("donor",),
    ),
    "rank": lambda refs: rank(
        frame_ref=refs["pinned_left"],
        rank_by=("donor",),
        aggregation="mean",
        window_start=_WINDOW[0],
        window_end=_WINDOW[1],
    ),
    "share_of": lambda refs: share_of(
        frame_ref=refs["pinned_left"], numerator_entity={"donor": DONORS[2]}
    ),
    "join": lambda refs: join(frame_ref=refs["pinned_left"], right_frame_ref=refs["pinned_right"]),
    "filter": lambda refs: filter_verb(
        frame_ref=refs["pinned_left"],
        predicates=(OrderedThresholdPredicate(column="value", op="gt", value=1e6),),
    ),
}


@pytest.mark.parametrize("verb", list(PINNED_FRAME_CONTENT_VERSIONS))
async def test_verb_output_content_version_is_pinned(refs, verb: str) -> None:
    result = await _CALLS[verb](refs)
    assert result.is_error is False, result.structured_content
    assert result.structured_content is not None
    assert result.structured_content["rows"] > 0
    assert result.structured_content["frame_content_version"] == PINNED_FRAME_CONTENT_VERSIONS[verb]
