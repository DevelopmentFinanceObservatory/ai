"""No signed URL survives a full search -> get_data -> resource-read sequence.

One `Client(mcp)` session, one `httpx2.MockTransport` handler routing by path,
walking every step a real caller would: resolve a spec_id through `search`,
mint it through `get_data`, then read the resource link `get_data`'s own
result carries. Two distinct sentinel URLs prove the only one a caller can
reach is the one minted at resource-read time.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest
from data_mcp import client_holder, get_data_tool
from data_mcp.definition_store import (
    InMemoryDefinitionStore,
    canonicalize,
    set_store,
    spec_id_for,
)
from data_mcp.frame_materialisation import MaterialisationOutcome
from data_mcp.manifest import Manifest
from data_mcp.server import mcp
from fastmcp import Client
from mcp.types import ResourceLink

from .net_guard import no_network  # noqa: F401 (imported for its autouse fixture)
from .test_client import _client, text_of
from .test_get_data_tool import _identity_doc, _manifest_response, mint_pair

Handler = Callable[[httpx2.Request], httpx2.Response]


@pytest.fixture(autouse=True)
def _stub_materialise_and_register(monkeypatch: pytest.MonkeyPatch) -> None:
    """This module's own concern is signed-URL leakage through the digest/resource-link
    pipeline. `_CONTENT_VERSION` here (`"cv_1"`) predates `FrameRef`'s pair pattern,
    so this stubs `materialise_and_register` to a no-op success; the real function
    (exercised directly in `test_frame_materialisation.py`) would otherwise refuse
    this sequence on an unrelated shape mismatch.
    """

    async def _stub(*, manifest: Manifest, **_: Any) -> MaterialisationOutcome:
        return MaterialisationOutcome(
            status="registered", attempts=0, frame_ref=mint_pair(manifest)
        )

    monkeypatch.setattr(get_data_tool, "materialise_and_register", _stub)


# `search`'s own write path (`_register_definitions`) hash-checks a spec
# node's identity against the outer spec_id it claims (`verify_write`), so
# this has to be the real hash of `_identity_doc()`, not an arbitrary id.
_SPEC_ID = spec_id_for(canonicalize(_identity_doc()))
_CONTENT_VERSION = "cv_1"
_MINT_SENTINEL = "https://signed.example/never-in-a-tool-result"
_RESOURCE_READ_SENTINEL = "https://signed.example/only-on-the-resource-read"


def _search_envelope() -> dict[str, Any]:
    """A `definite` envelope minting `_SPEC_ID`, with a `definition` sidecar so
    `search`'s own write path (`_register_definitions`) registers it. That is
    what lets the later `get_data` call find it in the store rather than
    falling back to the registry.
    """
    return {
        "schema_version": 1,
        "query_echo": {"entry_path": "parsed", "extract_skipped": True},
        "diagnostics": {"engine_build": "test", "trace_id": "trace-1", "warnings": []},
        "status": "definite",
        "parts": [
            {
                "status": "definite",
                "variable_text": "malaria grants",
                "warnings": [],
                "spec": {
                    "spec_id": _SPEC_ID,
                    "resolved": {
                        "rendered_sentence": "malaria grants from France to Togo, 2015 to 2024",
                        "confidence": "high",
                        "coverage": {"kind": "bare", "has_data": True},
                        "stat_vars": [],
                        "entities": [],
                        "facets": {},
                    },
                    "definition": {
                        "schema_version": 1,
                        "spec_id": _SPEC_ID,
                        "identity": _identity_doc(),
                    },
                },
            }
        ],
    }


def _routing_handler() -> Handler:
    mint_payload = _manifest_response(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)
    mint_payload["artifacts"][0]["url"] = _MINT_SENTINEL

    snapshot_payload = _manifest_response(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)
    snapshot_payload["artifacts"][0]["url"] = _RESOURCE_READ_SENTINEL
    snapshot_payload["artifacts"][0]["url_expires_at"] = (
        datetime.now(UTC) + timedelta(seconds=900)
    ).isoformat()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST" and request.url.path == "/api/search/resolve":
            return httpx2.Response(200, json=_search_envelope())
        if request.method == "POST" and request.url.path == "/api/datasets":
            return httpx2.Response(200, json=mint_payload)
        if request.method == "GET" and request.url.path.startswith("/api/datasets/"):
            return httpx2.Response(200, json=snapshot_payload)
        raise AssertionError(f"unexpected call to {request.method} {request.url.path}")

    return handler


def _no_sentinel_anywhere(dumped: str) -> bool:
    return _MINT_SENTINEL not in dumped and _RESOURCE_READ_SENTINEL not in dumped


async def test_no_signed_url_survives_the_full_search_to_pull_sequence() -> None:
    async with _client(_routing_handler()) as upstream:
        client_holder.set_client(upstream)
        async with Client(mcp) as client:
            set_store(InMemoryDefinitionStore())

            search_result = await client.call_tool(
                "search", {"parts": [{"variable": "malaria grants"}]}
            )
            assert search_result.is_error is False
            dumped_search_content = json.dumps(
                [block.model_dump(mode="json") for block in search_result.content]
            )
            dumped_search_structured = json.dumps(search_result.structured_content)
            dumped_search_meta = json.dumps(search_result.meta)
            assert _no_sentinel_anywhere(dumped_search_content)
            assert _no_sentinel_anywhere(dumped_search_structured)
            assert _no_sentinel_anywhere(dumped_search_meta)

            get_data_result = await client.call_tool("get_data", {"spec_id": _SPEC_ID})
            assert get_data_result.is_error is False
            dumped_get_data_content = json.dumps(
                [block.model_dump(mode="json") for block in get_data_result.content]
            )
            dumped_get_data_structured = json.dumps(get_data_result.structured_content)
            assert _no_sentinel_anywhere(dumped_get_data_content)
            assert _no_sentinel_anywhere(dumped_get_data_structured)

            resource_link = next(
                block for block in get_data_result.content if isinstance(block, ResourceLink)
            )
            read_contents = await client.read_resource(str(resource_link.uri))

    doc = json.loads(read_contents[0].text)
    assert doc["url"] == _RESOURCE_READ_SENTINEL
    assert doc["url"] != _MINT_SENTINEL

    every_tool_result_seen = [
        dumped_search_content,
        dumped_search_structured,
        dumped_search_meta,
        dumped_get_data_content,
        dumped_get_data_structured,
    ]
    for dumped in every_tool_result_seen:
        assert _MINT_SENTINEL not in dumped
        assert _RESOURCE_READ_SENTINEL not in dumped


async def test_a_stale_selection_refuses_and_the_next_search_lists_what_to_pass() -> None:
    """StatVars the graph has moved past refuse naming the outsider at this tool's own
    argument; the next `search` lists the StatVars to pass now, and `get_data` with
    those succeeds. The refusal is the recorded dfo-apis answer."""
    recorded = json.loads(
        (Path(__file__).parent / "fixtures" / "mint_refused_selection.json").read_text()
    )
    definition = recorded["request"]["definition"]
    stale = recorded["request"]["resolved_stat_var_dcids"]
    envelope = _search_envelope()
    spec = envelope["parts"][0]["spec"]
    spec["spec_id"] = definition["spec_id"]
    spec["definition"] = definition
    spec["resolved"]["stat_vars"] = {
        "items": [{"ref": "Count_Person", "label": "Population", "slot_values": []}],
        "member_count": 1,
        "truncated": False,
    }
    posted: list[list[str] | None] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/search/resolve":
            return httpx2.Response(200, json=envelope)
        assert request.url.path == "/api/datasets"
        selection = json.loads(request.content)["input"].get("resolved_stat_var_dcids")
        posted.append(selection)
        if "Count_Household" in (selection or []):
            return httpx2.Response(
                recorded["response"]["status"], json=recorded["response"]["body"]
            )
        return httpx2.Response(200, json=_manifest_response(spec_id=definition["spec_id"]))

    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        async with Client(mcp) as client:
            set_store(InMemoryDefinitionStore())
            await client.call_tool("search", {"parts": [{"variable": "population"}]})
            refused = await client.call_tool(
                "get_data",
                {"spec_id": definition["spec_id"], "resolved_stat_var_dcids": stale},
                raise_on_error=False,
            )
            searched = await client.call_tool("search", {"parts": [{"variable": "population"}]})
            listed = searched.structured_content["parts"][0]["specs"][0]["resolved_stat_var_dcids"]
            minted = await client.call_tool(
                "get_data",
                {"spec_id": definition["spec_id"], "resolved_stat_var_dcids": listed},
                raise_on_error=False,
            )

    assert refused.is_error is True
    assert refused.structured_content is not None
    assert refused.structured_content["fix"]["not_members"] == ["Count_Household"]
    assert refused.structured_content["fix"]["pointers"] == ["/resolved_stat_var_dcids/1"]
    assert "Count_Person (Population)" in text_of(searched)
    assert minted.is_error is False
    assert posted == [stale, ["Count_Person"]]
