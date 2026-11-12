"""The marking of the axes a search in the planner's words settled, shared by the
planner's own search and discovery's (`harness.scope_choices.mark_searched_axes`)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest
from compiler.sources import outcome_from_envelope
from compiler.surface import ToolCallResult
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from harness.scope_choices import PLANNER_AXES_BLOCK_KEY
from planner import source_reads as planner_source_reads
from planner.plan import pin_sources
from submission.service import Unpinned
from translator.discovery import found_blocks

_RECORDED = json.loads(
    (Path(__file__).parent / "fixtures" / "prices_decision_uk_oda.json").read_text()
)["translator"]


def _envelope(*, provenance: str | None) -> dict[str, Any]:
    """The recorded search answer for UK ODA, its scope choices all on `provenance`,
    or unreadable (a declared field missing) for `None`."""
    envelope = copy.deepcopy(_RECORDED)
    scope = envelope["parts"][0]["spec"]["resolved"]["scope_choices"]
    if provenance is None:
        del scope["rule_version"]
    for choice in scope["choices"]:
        choice["provenance"] = provenance
    return envelope


@pytest.mark.parametrize(
    ("provenance", "marked"),
    [
        pytest.param("default", None, id="settled-by-default"),
        pytest.param("explicit", ["measure_basis", "provider"], id="settled-by-an-entry"),
        pytest.param(None, None, id="unreadable"),
    ],
)
async def test_both_searches_mark_the_same_axes_for_the_same_part(
    provenance: str | None, marked: list[str] | None
) -> None:
    """Discovery's answer (`found_blocks`) and the planner's own search
    (`pin_by_search`) mark the same axes on the same part, and a part whose
    scope choices this release cannot read marks none in both."""
    envelope = _envelope(provenance=provenance)
    spec_id = envelope["parts"][0]["spec"]["spec_id"]

    class _Searches:
        async def call_tool(self, **_: Any) -> ToolCallResult:
            return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: envelope})

    read = await planner_source_reads.pin_by_search(
        cast(Any, _Searches()),
        Unpinned(
            requirement_id="r",
            step_id="s",
            requirement_key="k",
            part={},
            asked_roles=(),
            open_roles=(),
        ),
    )
    state: dict[str, Any] = {}
    assert read is not None
    assert read(state) is None
    searched = pin_sources(state)[spec_id].block
    found = found_blocks(outcome_from_envelope(envelope))[spec_id]

    assert searched.get(PLANNER_AXES_BLOCK_KEY) == found.get(PLANNER_AXES_BLOCK_KEY) == marked
