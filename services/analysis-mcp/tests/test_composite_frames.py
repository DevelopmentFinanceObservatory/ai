"""Verbs over a composite frame read with one series per StatVar.

dfo-apis mints a composite definition (bilateral plus imputed multilateral aid) as one
frame whose rows each belong to one component, and lists the components under the
manifest's `trace.components`. The fixture's rows and components are those its
`tests/datasets_api/test_mint_composite.py` client mints for CRS education aid from France
to Kenya, 2020-2021, each component's StatVars cut to the ones holding rows.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.growth_tool import growth
from analysis_mcp.quotient_of_tool import quotient_of
from analysis_mcp.rank_tool import rank
from common.materialisation import (
    FrameRef,
    InMemoryMaterialisationStore,
    MaterialisationTrace,
    set_store,
)
from dfo_frame_grammar import Frame
from fastmcp.tools import ToolResult

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "composite_kenya_education.json").read_text()
)


@pytest.fixture
async def store():
    held = InMemoryMaterialisationStore()
    set_store(held)
    yield held
    set_store(None)


async def _held(store: InMemoryMaterialisationStore, *, composite: bool) -> dict[str, str]:
    """The fixture's frame, held with its components on its trace or with none."""
    frame = Frame.build(
        role_keys=["financingProvider", "financingRecipient"], rows=_FIXTURE["rows"]
    )
    digest = secrets.token_hex(16)
    ref = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    trace = MaterialisationTrace(
        upstream_trace={"components": _FIXTURE["components"]} if composite else {},
        spec_id=ref.spec_id,
        content_version=ref.content_version,
    )
    await store.put(ref, frame=frame, trace=trace, writer="test")
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


_RECIPIENT = {"financingRecipient": "country/KEN"}

_VERBS: dict[str, Callable[[dict[str, str]], Awaitable[ToolResult]]] = {
    "aggregate": lambda ref: aggregate(
        frame_ref=ref, aggregation="sum", group_by=("financingRecipient",)
    ),
    "rank": lambda ref: rank(frame_ref=ref, rank_by=("financingRecipient",)),
    "growth": lambda ref: growth(frame_ref=ref, window_start="2020", window_end="2021"),
    "quotient_of": lambda ref: quotient_of(
        frame_ref=ref,
        denominator_frame_ref=ref,
        numerator_entity=_RECIPIENT,
        denominator_entity=_RECIPIENT,
    ),
}


@pytest.mark.parametrize("verb", list(_VERBS))
async def test_a_verb_refusing_a_composite_s_statvars_says_to_read_its_components_summed(
    store: InMemoryMaterialisationStore, verb: str
) -> None:
    composite = await _VERBS[verb](await _held(store, composite=True))
    plain = await _VERBS[verb](await _held(store, composite=False))

    refused: dict[str, Any] = composite.structured_content or {}
    assert refused["code"] == "stat_var_unresolved"
    assert refused["fix"]["components"] == [
        {"role": "bilateral", "stat_vars": 2},
        {"role": "imputed", "stat_vars": 16},
    ]
    assert refused["fix"]["get_data"] == {"stat_vars": "sum"}
    assert (plain.structured_content or {})["code"] == "stat_var_unresolved"
    assert "components" not in (plain.structured_content or {})["fix"]
