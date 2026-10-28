"""`harness.derived_frames`: whether a frame dfo-apis' runner stored holds rank positions."""

from __future__ import annotations

from typing import Any

import pytest
from harness.derived_frames import holds_rank_positions

_ROOT = {"root": 0}


def _step(verb: str, child: dict[str, Any] = _ROOT, **parameters: Any) -> dict[str, Any]:
    return {"op_spec": {"verb": verb, "parameters": parameters}, "inputs": {"frame_ref": child}}


@pytest.mark.parametrize(
    ("tree", "positions"),
    [
        (_step("rank", per_period=True), True),
        (_step("rank", per_period=True, limit=3), False),
        (_step("rank", per_period=False), False),
        (_step("growth", _step("rank", per_period=True), change="absolute"), True),
        (_step("aggregate", group_by=["donor"]), False),
    ],
    ids=["per-period", "limited", "collapsed", "under-a-growth", "no-rank"],
)
def test_a_per_period_rank_with_no_limit_anywhere_in_the_tree_holds_positions(
    tree: dict[str, Any], positions: bool
) -> None:
    assert holds_rank_positions(tree) is positions
