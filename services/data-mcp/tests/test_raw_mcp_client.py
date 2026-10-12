"""A raw, low-level MCP client reads the templated artifact resource end to end, no ADK.

`fastmcp.Client` is a convenience wrapper this suite otherwise drives everything
through, but it discovers resources through `resources/list` alone -- the same
path ADK's harness uses (see `artifact_resource.py`'s module docstring) -- so it
cannot prove the templated URI is reachable by anything at all. This
module drives a raw `mcp.ClientSession` over streamable HTTP instead, the way a
client that already holds a URI (from a `get_data` result's resource link) would.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx2
import pytest
from common.auth import apply_gate
from data_mcp import client_holder, get_data_tool
from data_mcp.client import UpstreamClient
from data_mcp.definition_store import InMemoryDefinitionStore, set_store
from data_mcp.frame_materialisation import MaterialisationOutcome
from data_mcp.manifest import ARTIFACT_URI_TEMPLATE, Manifest
from data_mcp.server import mcp
from data_mcp.settings import Settings
from data_mcp.skills_resources import SKILL_FILE_TEMPLATE
from mcp import ClientSession, MCPError
from mcp.client.streamable_http import streamable_http_client
from mcp.types import ResourceLink, TextContent

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .test_get_data_tool import _manifest_response, _stored_definition, mint_pair

_SENTINEL_URL = "/links/only-on-the-raw-session-read"
_TEST_TOKEN = "test-token"


@pytest.fixture(autouse=True)
def _stub_materialise_and_register(monkeypatch: pytest.MonkeyPatch) -> None:
    """This module's own concern is the raw MCP session reaching the templated artifact
    resource, not frame materialisation -- `_manifest_response`'s default `content_version`
    (`"cv_1"`) predates `FrameRef`'s pair pattern. Stubbed to a no-op success so the real
    `materialise_and_register` (exercised directly in `test_frame_materialisation.py`)
    doesn't refuse this sequence on an unrelated shape mismatch.
    """

    async def _stub(*, manifest: Manifest, **_: Any) -> MaterialisationOutcome:
        return MaterialisationOutcome(
            status="registered", attempts=0, frame_ref=mint_pair(manifest)
        )

    monkeypatch.setattr(get_data_tool, "materialise_and_register", _stub)


def _routing_handler(*, mint_payload: dict[str, Any], snapshot_payload: dict[str, Any]) -> Any:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST" and request.url.path == "/api/datasets":
            return httpx2.Response(200, json=mint_payload)
        if request.method == "GET" and request.url.path.startswith("/api/datasets/"):
            return httpx2.Response(200, json=snapshot_payload)
        if request.method == "GET" and request.url.path == "/api/skills":
            return httpx2.Response(200, json={"skills": [], "warnings": []})
        raise AssertionError(f"unexpected call to {request.method} {request.url.path}")

    return handler


@asynccontextmanager
async def _raw_session() -> AsyncIterator[ClientSession]:
    """A raw `ClientSession` over streamable HTTP, ASGI-mounted with no sockets --
    `net_guard`'s `no_network` fixture applies here too, and is what proves the "no ADK, no
    network" claim. `app.router.lifespan_context(app)` is required: the mounted FastMCP
    app's `StreamableHTTPSessionManager` only starts inside its own lifespan, the same
    contract `harness/server.py` documents for `combine_lifespans`.
    """
    # `mcp` carries a `BearerTokenVerifier` set at import time (`common.auth`),
    # not resolved to an actual token until something calls `apply_gate`.
    # `data_mcp.server.main()` and `harness.server.create_app` both do that.
    # This raw session builds `.http_app()` directly, so it has to as well,
    # sending the same token back as a raw `Authorization` header (no
    # fastmcp `Client` here to do it for us).
    apply_gate(mcp, token=_TEST_TOKEN, disabled=False)
    app = mcp.http_app(transport="streamable-http")

    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url="http://testserver",
            headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
        ) as http_client,
        streamable_http_client("http://testserver/mcp", http_client=http_client) as (
            read,
            write,
        ),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


async def test_raw_client_session_walks_the_whole_artifact_surface() -> None:
    stored = _stored_definition()
    mint_payload = _manifest_response(spec_id=stored.spec_id)
    digest = mint_payload["artifacts"][0]["artifact_digest"]

    snapshot_payload = _manifest_response(spec_id=stored.spec_id)
    snapshot_payload["artifacts"][0]["url"] = _SENTINEL_URL

    upstream = UpstreamClient(
        settings=Settings(),
        transport=httpx2.MockTransport(
            _routing_handler(mint_payload=mint_payload, snapshot_payload=snapshot_payload)
        ),
    )
    client_holder.set_client(upstream)
    set_store(InMemoryDefinitionStore())

    async with _raw_session() as session:
        resources = await session.list_resources()
        assert resources.resources == []

        templates = await session.list_resource_templates()
        assert [t.uri_template for t in templates.resource_templates] == [
            ARTIFACT_URI_TEMPLATE,
            SKILL_FILE_TEMPLATE,
        ]

        tool_result = await session.call_tool("get_data", {"spec_id": stored.spec_id})
        assert isinstance(tool_result.content[0], TextContent)
        resource_link = next(
            block for block in tool_result.content if isinstance(block, ResourceLink)
        )

        read_result = await session.read_resource(resource_link.uri)

    assert len(read_result.contents) == 1
    content = read_result.contents[0]
    assert content.mime_type == "application/json"
    doc = json.loads(content.text)
    assert doc["url"].endswith(_SENTINEL_URL)
    assert doc["artifact_digest"] == digest
    assert "url_expires_at" not in doc


async def test_raw_client_session_reading_an_aged_out_artifact_gets_the_marker_prefix() -> None:
    """`error.code` is fastmcp-assigned and carries no meaning -- asserted on the message
    prefix only, deliberately, so a later reader does not "improve" this assertion onto
    `code`.
    """
    subject_id = "spec_" + "a" * 32
    uri = f"dfo://artifact/{subject_id}/cv_1/sha256:" + "1" * 64

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            410,
            json={
                "code": "artifact_expired",
                "title": "gone",
                "detail": "aged out",
                "retryable": False,
            },
        )

    upstream = UpstreamClient(settings=Settings(), transport=httpx2.MockTransport(handler))
    client_holder.set_client(upstream)
    set_store(InMemoryDefinitionStore())

    async with _raw_session() as session:
        with pytest.raises(MCPError) as exc_info:
            await session.read_resource(uri)  # type: ignore[arg-type]

    assert exc_info.value.error.message.startswith("artifact_expired: ")
