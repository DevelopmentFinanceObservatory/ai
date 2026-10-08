"""Stored anchors for the tests of rolling `growth`, `rank` and `filter` calls.

A rolling call runs only against an anchor record that covers it, so each such
test resolves one over the fixture frame it reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from analysis_mcp.rolling_tool import resolve_anchor
from common.materialisation import FrameRef, MaterialisationStore

LATEST = {"anchor": "latest", "offset": 0, "rule": "half1"}


class Anchored(NamedTuple):
    """A resolved anchor: the populations it was taken over, its key, and the
    latest period."""

    populations: list[dict[str, Any]]
    anchor_key: str
    period: str


async def anchored(
    *roots: Mapping[str, str], bindings: Mapping[str, Sequence[str]] | None = None
) -> Anchored:
    """`resolve_anchor` over `roots`, frames of one dataset, narrowed by `bindings`."""
    populations = [{"spec_id": roots[0]["spec_id"], "bindings": dict(bindings or {})}]
    result = await resolve_anchor(populations=populations, roots=[dict(root) for root in roots])
    assert result.is_error is False, result.content
    content = result.structured_content
    return Anchored(populations, content["anchor_key"], content["period"])


async def second_version(store: MaterialisationStore, root: Mapping[str, str]) -> dict[str, str]:
    """The frame held under `root`, stored again as another version of its dataset."""
    held = await store.get(FrameRef(**root))
    assert held is not None
    frame, trace = held
    other = FrameRef(spec_id=root["spec_id"], content_version="cv_" + "9" * 32)
    await store.put(other, frame=frame, trace=trace, writer="test")
    return other.model_dump()
