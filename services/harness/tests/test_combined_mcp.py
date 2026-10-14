"""The combined MCP server at `/mcp/`, served beside the per-server mounts.

Each MCP test runs once per protocol revision: `mode="legacy"` negotiates
2025-11-25 through the initialize handshake, and `"2026-07-28"` adopts the
sessionless revision directly.

Every test runs presentation-mcp against `_StubPlatform` in place of dfo-apis,
so its three tools mount whatever listens on the platform's address.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, MutableMapping
from contextlib import asynccontextmanager, contextmanager, nullcontext
from contextvars import ContextVar
from itertools import pairwise
from types import SimpleNamespace
from typing import Any, cast, override

import httpx2
import langfuse
import pytest
from analysis_mcp import server as analysis_server
from analysis_mcp import tool_errors as analysis_tool_errors
from analysis_mcp.fixtures import seed_fixture_store
from common import auth as mcp_auth
from common import refusal as refusal_module
from common import telemetry
from common.logging import configure_logging
from common.materialisation import get_store as get_frame_store
from common.refusal import internal_error_message, refusal_meta
from common.uri_redaction import UriRedactingExporter
from data_mcp import artifact_resource, client_holder, get_data_tool, skills_resources
from data_mcp import server as data_server
from data_mcp import tool_errors as data_tool_errors
from data_mcp.client import UpstreamClient
from data_mcp.deadline import (
    DEADLINE_RETRY_AFTER_S,
    DIRECT_CALL_BUDGET,
    EXTERNAL_CALL_BUDGET,
    EXTERNAL_CALL_LIMIT_S,
    CallBudget,
)
from data_mcp.definition_store import InMemoryDefinitionStore, set_store
from data_mcp.mint_ledger import MintLedger
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.settings import Settings as DataMcpSettings
from data_mcp.skills_extension import SKILLS_EXTENSION_ID, SkillsGetParams, SkillsListParams
from fastmcp import Client, FastMCP
from fastmcp.apps import UI_EXTENSION_ID, UI_MIME_TYPE
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.server.providers import FastMCPProvider, LocalProvider
from fastmcp.tools.base import ToolResult
from fastmcp.utilities.lifespan import combine_lifespans
from harness import combined_mcp
from harness.combined_mcp import (
    COMBINED_MCP_INSTRUCTIONS,
    HARNESS_ONLY_TOOLS,
    build_combined_mcp,
    children_start_once,
    combined_http_app,
)
from harness.server import create_app
from harness.settings import (
    ANALYSIS_MCP_MOUNT_PATH,
    DATA_MCP_MOUNT_PATH,
    PRESENTATION_MCP_MOUNT_PATH,
    Settings,
)
from mcp import MCPError
from mcp import types as mt
from mcp.client import advertise
from mcp.shared._httpx_utils import McpHttpClientFactory
from mcp_types import INTERNAL_ERROR, INVALID_PARAMS
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode, format_trace_id
from opentelemetry.util._once import Once
from presentation_mcp import server as presentation_server
from presentation_mcp.client import WRITE_TOKEN_HEADER
from presentation_mcp.platform_contract import (
    RESOLVED_PAYLOAD_META_KEY,
    VIEW_STATE_META_KEY,
    Caller,
    Refusal,
    RetryMode,
    WriteReply,
)
from presentation_mcp.projection import ROWLESS_NOTICE, UNKNOWN_HOST_ROWLESS_NOTICE
from pydantic import BaseModel, ConfigDict
from starlette.applications import Starlette
from starlette.types import ASGIApp, Receive, Scope, Send

from .resolved_revision import resolved_revision

TOKEN = "test-token"
COMBINED = "/mcp/"
DIRECT = (
    f"{DATA_MCP_MOUNT_PATH}/mcp",
    f"{ANALYSIS_MCP_MOUNT_PATH}/mcp",
    f"{PRESENTATION_MCP_MOUNT_PATH}/mcp",
)
PRESENTATION_TOOLS = ("write_component", "describe_frame", "describe_component")

REVISIONS = pytest.mark.parametrize(
    ("mode", "revision"),
    [("legacy", "2025-11-25"), ("2026-07-28", "2026-07-28")],
    ids=["2025-11-25", "2026-07-28"],
)

# A call `get_data` refuses in its own body, before it reads the store or reaches dfo-apis:
# "sum" totals the whole family, which leaves no code to break it down by.
CONTRADICTORY_GET_DATA = {
    "spec_id": "spec_" + "0" * 32,
    "stat_vars": "sum",
    "breakdown": {"code_system": "crsPurpose", "level": 0},
}
UNHELD_FRAME = {"spec_id": "spec_" + "0" * 32, "content_version": "cv_" + "0" * 32}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from any of these vars set in the ambient environment."""
    for var in (
        "DATA_MCP_URL",
        "ANALYSIS_MCP_URL",
        "PRESENTATION_MCP_URL",
        "DATABASE_URL",
        "SESSION_STORE_INSTANCE_CONNECTION_NAME",
        "SESSION_STORE_DATABASE",
        "SESSION_STORE_RUNTIME_DB_USER",
        "PORT",
    ):
        monkeypatch.delenv(var, raising=False)


# The least release document presentation-mcp's lifespan admits and assembles
# its tool schemas from: one member type whose shared `$defs` carries a
# `frame_ref`.
_RELEASE: dict[str, Any] = {
    "schema_version": 1,
    "catalog_version": "test",
    "intent_schemas_tolerant": {
        "line_chart": {
            "$defs": {
                "frame_ref": {
                    "type": "object",
                    "properties": {
                        "spec_id": {"type": "string"},
                        "content_version": {"type": "string"},
                    },
                }
            },
            "type": "object",
            "properties": {
                "type": {"const": "line_chart"},
                "frame_ref": {"$ref": "#/$defs/frame_ref"},
            },
        }
    },
}

LINE_CHART_INTENT = {
    "type": "line_chart",
    "frame_ref": {"spec_id": "spec_1", "content_version": "cv_1"},
}


class _StubPlatform:
    """Stands in for `presentation_mcp.server.PlatformAssembler`: serves `_RELEASE`,
    mints `cmp_1` and its write token for any create, refuses `not_found` for
    any revise, and answers any describe `transient_unavailable`, as the
    assembler does when the platform cannot be reached."""

    def __init__(self, *, settings: object) -> None:
        del settings

    async def release(self) -> dict[str, Any]:
        return _RELEASE

    async def create_component(self, *, intent: dict[str, Any], caller: Caller) -> WriteReply:
        del intent, caller
        return WriteReply(
            component_id="cmp_1",
            revision=1,
            fact_summary={},
            resolved_payload=resolved_revision("line_chart", component_id="cmp_1", revision=1),
            write_token="cmw_stub1",
        )

    async def revise_component(
        self, *, component_id: str, patch: dict[str, Any], write_token: str, caller: Caller
    ) -> Refusal:
        del patch, write_token, caller
        return Refusal(
            error="not_found",
            retry=RetryMode.CORRECT_AND_RETRY,
            detail=f"Component {component_id!r} is not held by this assembler.",
            fix="Use an id a previous reply returned.",
        )

    async def get_component(self, *, component_id: str, revision: int | None) -> Refusal:
        del component_id, revision
        return Refusal(
            error="transient_unavailable",
            retry=RetryMode.RETRY_AFTER_DELAY,
            detail="The platform call failed before a response arrived.",
        )

    async def aclose(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _stub_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(presentation_server, "PlatformAssembler", _StubPlatform)


@pytest.fixture
def logging_configured() -> None:
    """Run `create_app`'s `configure_logging` before the test body. It replaces
    the root logger's handlers the first time it runs in a process; run during
    setup, it leaves the handler `caplog` adds for the test body in place."""
    configure_logging()


def _described(tools: list[mt.Tool]) -> dict[str, tuple[str | None, dict[str, Any]]]:
    return {tool.name: (tool.description, tool.input_schema) for tool in tools}


@REVISIONS
async def test_the_combined_server_lists_what_the_direct_mounts_list(
    mode: str,
    revision: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    async with running_lifespan(app):
        direct_tools: list[mt.Tool] = []
        direct_templates: list[mt.ResourceTemplate] = []
        for path in DIRECT:
            async with Client(mounted_mcp_transport(app, path, token=TOKEN), mode=mode) as client:
                direct_tools += await client.list_tools()
                direct_templates += await client.list_resource_templates()
        async with Client(mounted_mcp_transport(app, COMBINED, token=TOKEN), mode=mode) as client:
            assert client.protocol_version == revision
            combined_tools = await client.list_tools()
            combined_templates = await client.list_resource_templates()

    assert {tool.name for tool in direct_tools} >= HARNESS_ONLY_TOOLS
    direct_tools = [tool for tool in direct_tools if tool.name not in HARNESS_ONLY_TOOLS]
    assert len(combined_tools) == len(direct_tools)
    assert {"search", "get_data", "aggregate", *PRESENTATION_TOOLS} <= _described(
        combined_tools
    ).keys()
    assert _described(combined_tools) == _described(direct_tools)
    assert any(tool.output_schema is not None for tool in direct_tools)
    assert [tool.output_schema for tool in combined_tools] == [None] * len(combined_tools)
    assert combined_templates == direct_templates
    assert [template.uri_template for template in combined_templates] == [
        "dfo://artifact/{subject_id}/{content_version}/{artifact_digest}",
        "skill://{skill_id}/{path*}",
        "ui://presentation-mcp/renderer-{version}.html",
    ]


async def test_the_parent_mounts_the_three_children_and_serves_only_their_tools() -> None:
    combined = build_combined_mcp(settings=Settings(mcp_api_token=TOKEN))
    mounted = [p.server for p in combined.providers if isinstance(p, FastMCPProvider)]
    others = [type(p) for p in combined.providers if not isinstance(p, FastMCPProvider)]
    served = {tool.name for tool in await combined.list_tools(run_middleware=False)}
    children = {
        tool.name
        for server in (data_server.mcp, analysis_server.mcp, presentation_server.mcp)
        for tool in await server.list_tools(run_middleware=False)
    }

    assert mounted == [data_server.mcp, analysis_server.mcp, presentation_server.mcp]
    assert others == [LocalProvider]
    assert children >= HARNESS_ONLY_TOOLS
    assert served == children - HARNESS_ONLY_TOOLS


@REVISIONS
async def test_the_combined_server_refuses_as_the_direct_mounts_do(
    mode: str,
    revision: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """One refusal from each child, neither of which reaches an upstream."""
    app = create_app(settings=Settings(mcp_api_token=TOKEN))
    calls = (
        (DIRECT[0], "get_data", CONTRADICTORY_GET_DATA),
        (DIRECT[1], "aggregate", {"frame_ref": UNHELD_FRAME, "aggregation": "sum"}),
    )

    async with running_lifespan(app):
        pairs = []
        for direct_path, tool, arguments in calls:
            results = []
            for path in (direct_path, COMBINED):
                transport = mounted_mcp_transport(app, path, token=TOKEN)
                async with Client(transport, mode=mode) as client:
                    assert client.protocol_version == revision
                    results.append(await client.call_tool(tool, arguments, raise_on_error=False))
            pairs.append(results)

    for direct, combined in pairs:
        assert direct.is_error is True
        assert combined.is_error is True
        assert combined.content == direct.content
    assert '"code":"request_invalid"' in pairs[0][1].content[0].text


def _envelope(result: Any) -> dict[str, Any]:
    """The refusal envelope a model reads: the last line of the result's text."""
    return json.loads(result.content[0].text.splitlines()[-1])


@REVISIONS
async def test_a_success_through_the_combined_server_carries_only_the_digest(
    mode: str,
    revision: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    async with running_lifespan(app):
        handles = await seed_fixture_store(get_frame_store())
        arguments = {
            "frame_ref": handles.plain_single_unit,
            "aggregation": "sum",
            "group_by": ["reporter"],
        }
        results = []
        for path in (DIRECT[1], COMBINED):
            async with Client(mounted_mcp_transport(app, path, token=TOKEN), mode=mode) as client:
                assert client.protocol_version == revision
                results.append(await client.call_tool("aggregate", arguments))
    direct, combined = results

    assert direct.structured_content is not None
    assert combined.is_error is False
    assert combined.structured_content is None
    assert combined.content == direct.content


async def test_a_tool_a_child_adds_is_shaped_without_being_named(
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """Every tool the parent serves is shaped, so a new data-mcp tool loses
    its output schema and structured content with no change here."""

    def tally() -> dict[str, int]:
        return {"count": 1}

    data_server.mcp.tool(tally)
    try:
        app = create_app(settings=Settings(mcp_api_token=TOKEN))
        async with (
            running_lifespan(app),
            Client(mounted_mcp_transport(app, COMBINED, token=TOKEN)) as client,
        ):
            listed = {tool.name: tool for tool in await client.list_tools()}
            result = await client.call_tool("tally", {})
    finally:
        data_server.mcp.local_provider.remove_tool("tally")

    assert listed["tally"].output_schema is None
    assert result.is_error is False
    assert result.structured_content is None
    assert json.loads(result.content[0].text) == {"count": 1}


async def test_the_search_envelope_in_meta_stays_behind_the_combined_server(
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """The harness reads `search`'s envelope from the direct mount; an
    outside client gets the digest and the rest of `_meta`."""

    def enveloped() -> ToolResult:
        return ToolResult(
            content="digest",
            meta={SEARCH_ENVELOPE_META_KEY: {"parts": []}, "org.one.dfo/other": 1},
        )

    data_server.mcp.tool(enveloped)
    try:
        app = create_app(settings=Settings(mcp_api_token=TOKEN))
        results = []
        async with running_lifespan(app):
            for path in (DIRECT[0], COMBINED):
                async with Client(mounted_mcp_transport(app, path, token=TOKEN)) as client:
                    results.append(await client.call_tool("enveloped", {}))
    finally:
        data_server.mcp.local_provider.remove_tool("enveloped")
    direct, combined = results

    assert direct.meta is not None
    assert direct.meta[SEARCH_ENVELOPE_META_KEY] == {"parts": []}
    assert combined.meta is not None
    assert SEARCH_ENVELOPE_META_KEY not in combined.meta
    assert combined.meta["org.one.dfo/other"] == 1


# An MCP Apps host declares the UI extension among its client capabilities.
_APPS_HOST = [advertise(UI_EXTENSION_ID, {"mimeTypes": [UI_MIME_TYPE]})]


@REVISIONS
async def test_the_presentation_tools_are_listed_with_their_view(
    mode: str,
    revision: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, COMBINED, token=TOKEN), mode=mode) as client,
    ):
        assert client.protocol_version == revision
        listed = {tool.name: tool for tool in await client.list_tools()}

    view = {"resourceUri": presentation_server._RENDERER_URI}
    assert {name: (listed[name].meta or {}).get("ui") for name in PRESENTATION_TOOLS} == {
        "write_component": view,
        "describe_frame": None,
        "describe_component": view,
    }


@REVISIONS
async def test_every_listed_tool_has_a_title_and_explicit_annotations(
    mode: str,
    revision: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, COMBINED, token=TOKEN), mode=mode) as client,
    ):
        assert client.protocol_version == revision
        tools = await client.list_tools()

    for tool in tools:
        assert tool.title, tool.name
        assert tool.annotations is not None, tool.name
        for hint in ("read_only_hint", "destructive_hint", "open_world_hint"):
            assert isinstance(getattr(tool.annotations, hint), bool), (tool.name, hint)
    assert {t.name for t in tools if t.annotations.read_only_hint is False} == {"write_component"}
    assert {t.name for t in tools if t.annotations.open_world_hint} == {"write_component"}


@REVISIONS
async def test_the_combined_server_announces_its_instructions_and_the_harness_version(
    mode: str,
    revision: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, COMBINED, token=TOKEN), mode=mode) as client,
    ):
        assert client.protocol_version == revision
        if mode == "legacy":
            # The initialize result carries both.
            instructions = client.instructions
            server_info = client.server_info
            assert server_info is not None
            version = server_info.version
        else:
            # The 2026-07-28 revision has no initialize. `server/discover` carries the
            # instructions, and every result's `_meta` carries the server info.
            discovered = await client.session.send_discover(revision)
            instructions = discovered["instructions"]
            version = discovered["_meta"][_SERVER_INFO]["version"]
        listed = {tool.name for tool in await client.list_tools()}

    assert instructions == COMBINED_MCP_INSTRUCTIONS
    assert version == importlib.metadata.version("harness")
    named = set(re.findall(r"`(\w+)`", instructions))
    assert named == listed, (
        f"named but not listed: {sorted(named - listed)}; "
        f"listed but not named: {sorted(listed - named)}"
    )


async def _through_each(
    app: Any,
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
    *,
    mode: str,
    arguments: dict[str, Any],
) -> list[Any]:
    """write_component called with `arguments` by an MCP Apps host, on the
    presentation-mcp mount and then on the combined server."""
    results = []
    for path in (DIRECT[2], COMBINED):
        transport = mounted_mcp_transport(app, path, token=TOKEN)
        async with Client(transport, mode=mode, extensions=_APPS_HOST) as client:
            results.append(
                await client.call_tool("write_component", arguments, raise_on_error=False)
            )
    return results


# The `_meta` key naming the server that answered, which differs by mount.
_SERVER_INFO = "io.modelcontextprotocol/serverInfo"


def _without_server_info(meta: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (meta or {}).items() if key != _SERVER_INFO}


@REVISIONS
async def test_a_view_passes_the_combined_server_whole_within_its_call_limit(
    mode: str,
    revision: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """write_component's result keeps the view's layout in `structuredContent`
    and the values it binds in `_meta`, and finishes inside the
    `EXTERNAL_CALL_LIMIT_S` the combined server allows a call.

    The content differs in one line on 2025-11-25: the combined server
    answers that revision's requests without a session, so presentation-mcp
    cannot see the host's capabilities and says so in its rowless notice.
    """
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    async with running_lifespan(app):
        started = time.monotonic()
        direct, combined = await _through_each(
            app, mounted_mcp_transport, mode=mode, arguments={"intent": LINE_CHART_INTENT}
        )
        elapsed = time.monotonic() - started

    assert elapsed < EXTERNAL_CALL_LIMIT_S
    assert combined.is_error is False
    assert combined.structured_content is not None
    assert combined.structured_content == direct.structured_content
    assert combined.meta is not None
    assert combined.meta[VIEW_STATE_META_KEY]
    assert combined.meta[RESOLVED_PAYLOAD_META_KEY]["component_id"] == "cmp_1"
    assert _without_server_info(combined.meta) == _without_server_info(direct.meta)
    notice = UNKNOWN_HOST_ROWLESS_NOTICE if revision == "2025-11-25" else ROWLESS_NOTICE
    direct_text = [block.text for block in direct.content]
    assert ROWLESS_NOTICE in direct_text[0]
    assert [block.text for block in combined.content] == [
        direct_text[0].replace(ROWLESS_NOTICE, notice),
        *direct_text[1:],
    ]


@REVISIONS
async def test_a_view_tools_refusal_passes_the_combined_server_whole(
    mode: str,
    revision: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """The refusal's notice view stays in `structuredContent` for the host to draw."""
    app = create_app(settings=Settings(mcp_api_token=TOKEN))
    arguments = {"intent": {"title": "Aid"}, "component_id": "cmp_9"}

    async with running_lifespan(app):
        direct, combined = await _through_each(
            app, mounted_mcp_transport, mode=mode, arguments=arguments
        )

    assert combined.is_error is True
    assert combined.structured_content is not None
    assert "cmp_9" in json.dumps(combined.structured_content)
    assert (combined.content, combined.structured_content) == (
        direct.content,
        direct.structured_content,
    )
    assert _without_server_info(combined.meta) == _without_server_info(direct.meta)


@REVISIONS
async def test_the_renderer_page_reads_through_the_combined_server(
    mode: str,
    revision: str,
    logging_configured: None,
    caplog: pytest.LogCaptureFixture,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """The listed URI and any other version's, each through the request log."""
    app = create_app(settings=Settings(mcp_api_token=TOKEN))
    uris = (presentation_server._RENDERER_URI, "ui://presentation-mcp/renderer-0.html")

    async with running_lifespan(app):
        async with Client(mounted_mcp_transport(app, DIRECT[2], token=TOKEN), mode=mode) as client:
            [direct] = await client.read_resource(uris[0])
        with caplog.at_level(logging.INFO, logger="harness.combined_mcp"):
            async with Client(
                mounted_mcp_transport(app, COMBINED, token=TOKEN), mode=mode
            ) as client:
                listed = [str(resource.uri) for resource in await client.list_resources()]
                read = [content for uri in uris for content in await client.read_resource(uri)]

    assert uris[0] in listed
    assert [content.mime_type for content in read] == [UI_MIME_TYPE] * 2
    assert direct.meta is not None
    assert direct.meta["ui"]["csp"]
    assert [content.meta for content in read] == [direct.meta] * 2
    logged = [
        r.getMessage()
        for r in caplog.records
        if r.name == "harness.combined_mcp" and r.getMessage().startswith("combined mcp call")
    ]
    assert [message.split(" revision=")[0] for message in logged] == [
        "combined mcp call resource_list outcome=ok",
        *["combined mcp call resource outcome=ok"] * 2,
    ]
    assert " uri=" not in logged[0]
    assert [message.split(" uri=")[1] for message in logged[1:]] == [
        json.dumps(uri) for uri in uris
    ]


@REVISIONS
async def test_a_resource_reads_query_and_fragment_stay_out_of_the_request_log(
    mode: str,
    revision: str,
    logging_configured: None,
    caplog: pytest.LogCaptureFixture,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """The client parses the URI, which gives an `https` host an empty path (`/`) before
    the query; a `dfo` host keeps its query and fragment against the bare host."""
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, COMBINED, token=TOKEN), mode=mode) as client,
    ):
        with (
            caplog.at_level(logging.INFO, logger="harness.combined_mcp"),
            pytest.raises(MCPError),
        ):
            await client.read_resource("dfo://artifact?token=secret#frag")

    [logged] = [
        r.getMessage()
        for r in caplog.records
        if r.name == "harness.combined_mcp" and r.getMessage().startswith("combined mcp call")
    ]
    assert logged.endswith(" uri=" + json.dumps("dfo://artifact"))
    assert "secret" not in logged


@REVISIONS
async def test_the_rate_limit_counts_tool_calls_alone(
    mode: str,
    revision: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """A bucket of one admits one tool call after any number of other requests.
    A session DELETE never reaches middleware."""
    app = create_app(
        settings=Settings(
            mcp_api_token=TOKEN, combined_mcp_requests_per_s=0.001, combined_mcp_burst=1
        )
    )
    arguments = {"frame_ref": UNHELD_FRAME, "aggregation": "sum"}

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, COMBINED, token=TOKEN), mode=mode) as client,
    ):
        assert client.protocol_version == revision
        for _ in range(3):
            await client.list_tools()
            await client.list_resources()
            await client.read_resource(presentation_server._RENDERER_URI)
            if mode == "legacy":  # the 2026-07-28 revision has no ping
                await client.ping()
        admitted = await client.call_tool("aggregate", arguments, raise_on_error=False)
        refused = await client.call_tool("aggregate", arguments, raise_on_error=False)

    assert _envelope(admitted)["code"] == "frame_not_held"
    assert _envelope(refused)["code"] == "rate_limited"


@REVISIONS
async def test_an_unknown_tool_refuses_with_the_tools_the_server_serves(
    mode: str,
    revision: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """No child sees a call the parent cannot route, so analysis-mcp's own
    unknown-verb message never runs."""
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, COMBINED, token=TOKEN), mode=mode) as client,
    ):
        assert client.protocol_version == revision
        tools = sorted(tool.name for tool in await client.list_tools())
        result = await client.call_tool("summarise", {}, raise_on_error=False)

    assert result.is_error is True
    assert result.content[0].text.startswith(
        f"No tool is named 'summarise'. Call one of: {tools[0]}"
    )
    assert _envelope(result) == {
        "code": "tool_not_found",
        "retry_mode": "correct_and_retry",
        "fix": {"tools": tools},
    }


@REVISIONS
async def test_an_outside_client_cannot_call_a_harness_only_tool(
    mode: str,
    revision: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """data-mcp's own mount serves `resolve_definition`, refusing an identity that
    does not hash to its spec_id before it asks Search; the combined server refuses
    the same call as a tool it does not serve."""
    app = create_app(settings=Settings(mcp_api_token=TOKEN))
    arguments = {
        "spec_id": "spec_" + "0" * 32,
        "identity": {"identity_version": "spec1", "seed": "outside"},
    }

    async with running_lifespan(app):
        results = []
        for path in (DIRECT[0], COMBINED):
            async with Client(mounted_mcp_transport(app, path, token=TOKEN), mode=mode) as client:
                assert client.protocol_version == revision
                results.append(
                    await client.call_tool("resolve_definition", arguments, raise_on_error=False)
                )
    direct, combined = results

    assert _envelope(direct)["code"] == "request_invalid"
    assert _envelope(combined)["code"] == "tool_not_found"
    assert "resolve_definition" not in _envelope(combined)["fix"]["tools"]


@REVISIONS
async def test_arguments_that_fail_the_input_schema_refuse_in_the_envelope(
    mode: str,
    revision: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """The direct mount keeps fastmcp's bare validation text."""
    app = create_app(settings=Settings(mcp_api_token=TOKEN))
    arguments = {"frame_ref": UNHELD_FRAME}  # `aggregation` is required

    async with running_lifespan(app):
        results = []
        for path in (DIRECT[1], COMBINED):
            async with Client(mounted_mcp_transport(app, path, token=TOKEN), mode=mode) as client:
                assert client.protocol_version == revision
                results.append(await client.call_tool("aggregate", arguments, raise_on_error=False))
    direct, combined = results

    assert direct.is_error is True
    assert '"code"' not in direct.content[0].text
    assert combined.is_error is True
    assert combined.content[0].text.startswith(
        "The arguments to aggregate do not match its input schema: "
    )
    envelope = _envelope(combined)
    assert (envelope["code"], envelope["retry_mode"]) == ("request_invalid", "correct_and_retry")
    assert envelope["fix"]["tool"] == "aggregate"
    assert "aggregation" in envelope["fix"]["detail"]


_KNOWN_TRACE_ID = "0af7651916cd43dd8448eb211c80319c"

# An exception text naming what an outside caller must never see.
_SECRET_FAILURE = (
    "403 reading gs://dfo-prod-frames/private/frame.parquet as "
    "serviceAccount:runtime@dfo-prod.iam.gserviceaccount.com"
)


def _assert_no_secret(replied: str) -> None:
    assert "gs://" not in replied
    assert "dfo-prod" not in replied
    assert "iam.gserviceaccount.com" not in replied


def _assert_logged_under(
    caplog: pytest.LogCaptureFixture, trace_id: str, *, logger: str = "harness.combined_mcp"
) -> None:
    """One ERROR record from `logger` names `trace_id` and carries the
    exception, whose chain holds the secret text."""
    [record] = [r for r in caplog.records if r.name == logger and r.levelno >= logging.ERROR]
    assert f"trace_id={trace_id}" in record.getMessage()
    assert record.exc_info is not None
    raised = record.exc_info[1]
    chain = [raised, raised.__cause__ if raised is not None else None]
    assert any(_SECRET_FAILURE in str(exc) for exc in chain if exc is not None)


async def test_a_tool_that_raises_refuses_with_no_trace_id_and_none_of_its_text(
    logging_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """The exception's text can name a bucket or a service account, so the
    refusal carries neither, and the log line carries the trace id and the
    exception. The direct mount, which answers loopback clients alone, keeps
    fastmcp's own text."""

    def explode() -> str:
        raise RuntimeError(_SECRET_FAILURE)

    data_server.mcp.tool(explode)
    monkeypatch.setattr(refusal_module, "current_trace_id", lambda: _KNOWN_TRACE_ID)
    try:
        app = create_app(settings=Settings(mcp_api_token=TOKEN))
        async with running_lifespan(app):
            results = []
            with caplog.at_level(logging.ERROR, logger="harness.combined_mcp"):
                for path in (DIRECT[0], COMBINED):
                    async with Client(mounted_mcp_transport(app, path, token=TOKEN)) as client:
                        results.append(await client.call_tool("explode", {}, raise_on_error=False))
    finally:
        data_server.mcp.local_provider.remove_tool("explode")
    direct, combined = results

    assert direct.is_error is True
    assert direct.content[0].text == f"Error calling tool 'explode': {_SECRET_FAILURE}"
    assert combined.is_error is True
    envelope = _envelope(combined)
    assert envelope == {
        "code": "internal_error",
        "retry_mode": "do_not_retry",
        "fix": {"tool": "explode"},
    }
    message = internal_error_message(subject="explode")
    assert combined.content[0].text.startswith(f"{message}\n\n")
    assert combined.structured_content is not None
    assert combined.structured_content["message"] == message
    replied = combined.content[0].text + json.dumps(combined.structured_content)
    assert _KNOWN_TRACE_ID not in replied
    _assert_no_secret(replied)
    _assert_logged_under(caplog, _KNOWN_TRACE_ID)


async def test_a_resource_read_that_raises_fails_with_no_trace_id_and_none_of_its_text(
    logging_configured: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """An unexpected exception inside the artifact resource fails with fixed
    text, logged by data-mcp with its trace id, while the resource's
    own refusals, written for the caller, pass through unchanged."""

    def unreachable() -> UpstreamClient:
        raise RuntimeError(_SECRET_FAILURE)

    monkeypatch.setattr(artifact_resource, "get_client", unreachable)
    monkeypatch.setattr(refusal_module, "current_trace_id", lambda: _KNOWN_TRACE_ID)
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, COMBINED, token=TOKEN)) as client,
    ):
        with (
            caplog.at_level(logging.ERROR, logger="data_mcp.artifact_resource"),
            pytest.raises(MCPError) as raised,
        ):
            await client.read_resource(_ARTIFACT_URI)
        with pytest.raises(MCPError) as refused:
            await client.read_resource(_ARTIFACT_URI.replace("/cv_1/", "/not-a-content-version/"))

    message = str(raised.value)
    assert message == "internal_error: " + internal_error_message(subject="Reading this resource")
    assert _KNOWN_TRACE_ID not in message
    _assert_no_secret(message)
    _assert_logged_under(caplog, _KNOWN_TRACE_ID, logger="data_mcp.artifact_resource")
    assert str(refused.value).startswith(
        "request_invalid: content_version 'not-a-content-version' does not look like a cv_ id"
    )


def test_every_server_writes_the_same_refusal_text_for_the_same_fields() -> None:
    """data-mcp, analysis-mcp and the combined server render the envelope
    through one helper, so one set of fields gives one text."""
    message = "The frame failed its integrity check."
    fix = {"frame_ref": UNHELD_FRAME}
    data = data_tool_errors.to_tool_result(
        data_tool_errors.ToolRefusal(
            code=data_tool_errors.ToolErrorCode.FRAME_INTEGRITY_FAILED, message=message, fix=fix
        )
    )
    analysis = analysis_tool_errors.to_tool_result(
        analysis_tool_errors.ToolRefusal(
            code=analysis_tool_errors.ToolErrorCode.FRAME_INTEGRITY_FAILED,
            message=message,
            fix=fix,
        )
    )
    combined = combined_mcp._refusal(
        code="frame_integrity_failed", retry_mode="do_not_retry", message=message, fix=fix
    )

    texts = [result.content[0].text for result in (data, analysis, combined)]
    assert texts[0] == (
        f"{message}\n\n"
        '{"code":"frame_integrity_failed","retry_mode":"do_not_retry",'
        f'"fix":{{"frame_ref":{json.dumps(UNHELD_FRAME, separators=(",", ":"))}}}}}'
    )
    assert texts[1] == texts[0]
    assert texts[2] == texts[0]


@REVISIONS
async def test_get_data_through_the_combined_server_runs_under_the_external_budget(
    mode: str,
    revision: str,
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    read: dict[str, list[CallBudget]] = {}
    current_call_budget = get_data_tool.current_call_budget
    path_now = ""

    def recording() -> CallBudget:
        budget = current_call_budget()
        read.setdefault(path_now, []).append(budget)
        return budget

    monkeypatch.setattr(get_data_tool, "current_call_budget", recording)
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    async with running_lifespan(app):
        for path_now in (COMBINED, DIRECT[0]):
            async with Client(
                mounted_mcp_transport(app, path_now, token=TOKEN), mode=mode
            ) as client:
                assert client.protocol_version == revision
                await client.call_tool("get_data", CONTRADICTORY_GET_DATA, raise_on_error=False)

    assert set(read) == {COMBINED, DIRECT[0]}
    assert set(read[COMBINED]) == {EXTERNAL_CALL_BUDGET}
    assert set(read[DIRECT[0]]) == {DIRECT_CALL_BUDGET}


@REVISIONS
async def test_a_tool_call_over_the_rate_limit_refuses_in_the_envelope(
    mode: str,
    revision: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """A bucket of two that refills once in 1000 s admits two tool calls. The
    direct mount is not limited."""
    app = create_app(
        settings=Settings(
            mcp_api_token=TOKEN, combined_mcp_requests_per_s=0.001, combined_mcp_burst=2
        )
    )
    arguments = {"frame_ref": UNHELD_FRAME, "aggregation": "sum"}

    async with running_lifespan(app):
        async with Client(mounted_mcp_transport(app, COMBINED, token=TOKEN), mode=mode) as client:
            assert client.protocol_version == revision
            results = [
                await client.call_tool("aggregate", arguments, raise_on_error=False)
                for _ in range(3)
            ]
        async with Client(mounted_mcp_transport(app, DIRECT[1], token=TOKEN), mode=mode) as client:
            direct = await client.call_tool("aggregate", arguments, raise_on_error=False)

    assert _envelope(results[0])["code"] == "frame_not_held"
    assert (
        results[-1]
        .content[0]
        .text.startswith(
            "This server is receiving more calls than it accepts across all of its callers."
        )
    )
    assert _envelope(results[-1]) == {
        "code": "rate_limited",
        "retry_mode": "retry_after_delay",
        "fix": {},
        "retry_after_s": 1000.0,
    }
    assert _envelope(direct)["code"] == "frame_not_held"


# A call limit short enough for a test to reach, in place of `EXTERNAL_CALL_LIMIT_S`.
LIMIT_S = 0.5

# How long a test's client waits on a call the limit should stop, so a server that
# fails to stop it fails the test instead of hanging it.
_CLIENT_TIMEOUT_S = 10

_ARTIFACT_URI = f"dfo://artifact/spec_{'5' * 32}/cv_1/sha256:{'1' * 64}"
_UNSTORED_SPEC = "spec_" + "5" * 32


class _StalledUpstream:
    """dfo-apis holding every request until `release` is set, recording the path of
    each request whose wait was cancelled. Once released, it answers 400."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.cancelled: list[str] = []

    async def handler(self, request: httpx2.Request) -> httpx2.Response:
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.append(request.url.path)
            raise
        return httpx2.Response(400)


@pytest.fixture
def stalled_upstream() -> Iterator[_StalledUpstream]:
    """data-mcp's upstream client, answering through a `_StalledUpstream`."""
    upstream = _StalledUpstream()
    client_holder.set_client(
        UpstreamClient(settings=DataMcpSettings(), transport=httpx2.MockTransport(upstream.handler))
    )
    try:
        yield upstream
    finally:
        upstream.release.set()
        client_holder.set_client(None)


@pytest.fixture
def empty_mint_ledger(monkeypatch: pytest.MonkeyPatch) -> MintLedger:
    """A fresh mint ledger, so a mint one test holds is not held in the next."""
    ledger = MintLedger()
    monkeypatch.setattr(get_data_tool, "_mint_ledger", ledger)
    return ledger


def _limited_app(*, call_limit_s: float = LIMIT_S, **settings: Any) -> Any:
    """The combined server's app, alone, with a short call limit."""
    combined = build_combined_mcp(
        settings=Settings(mcp_api_token=TOKEN, **settings), call_limit_s=call_limit_s
    )
    mcp_auth.apply_gate(combined, token=TOKEN, disabled=False)
    return combined_http_app(combined)


def _stopped_envelope(tool: str, limit_s: float = LIMIT_S) -> dict[str, Any]:
    return {
        "code": "service_unavailable",
        "retry_mode": "retry_after_delay",
        "fix": {"tool": tool, "limit_s": limit_s},
        "retry_after_s": 5.0,
    }


def test_the_combined_server_stops_calls_at_the_external_call_limit_by_default() -> None:
    """`get_data`'s own deadline is sized to end inside this limit
    (`data_mcp.deadline.EXTERNAL_CALL_MARGIN_S`)."""
    combined = build_combined_mcp(settings=Settings(mcp_api_token=TOKEN))
    [limit] = [m for m in combined.middleware if isinstance(m, combined_mcp._CallLimit)]

    assert limit._limit_s == EXTERNAL_CALL_LIMIT_S
    assert EXTERNAL_CALL_BUDGET.deadline_s is not None
    assert EXTERNAL_CALL_BUDGET.deadline_s < EXTERNAL_CALL_LIMIT_S


@REVISIONS
async def test_a_stalled_search_is_stopped_at_the_limit_on_the_combined_server_alone(
    mode: str,
    revision: str,
    stalled_upstream: _StalledUpstream,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """`search` runs under the upstream client's own retry budget, which is longer
    than the limit. Through the combined server the call is cancelled at the limit,
    its upstream request with it; through data-mcp directly it runs on."""
    app = _limited_app()
    arguments = {"parts": [{"variable": "aid to Kenya"}]}

    async with running_lifespan(app):
        async with Client(
            mounted_mcp_transport(app, "/", token=TOKEN), mode=mode, timeout=_CLIENT_TIMEOUT_S
        ) as client:
            assert client.protocol_version == revision
            started = time.monotonic()
            stopped = await client.call_tool("search", arguments, raise_on_error=False)
            stopped_after = time.monotonic() - started
        cancelled = list(stalled_upstream.cancelled)

        async with Client(data_server.mcp) as client:
            direct = asyncio.create_task(
                client.call_tool("search", arguments, raise_on_error=False)
            )
            await asyncio.sleep(3 * LIMIT_S)
            still_running = not direct.done()
            stalled_upstream.release.set()
            answered = await direct

    assert stopped.is_error is True
    assert stopped.content[0].text.startswith(
        f"search did not finish within the {LIMIT_S:g}s this server allows a call, so the "
        "call was stopped. Repeat it unchanged after 5s."
    )
    assert _envelope(stopped) == _stopped_envelope("search")
    assert LIMIT_S <= stopped_after < LIMIT_S + 1.0
    assert cancelled == ["/api/search/resolve"]
    assert still_running is True
    assert answered.structured_content is not None
    assert answered.structured_content["code"] == "upstream_error"


async def test_a_stalled_resource_read_is_stopped_at_the_limit(
    stalled_upstream: _StalledUpstream,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    app = _limited_app()

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, "/", token=TOKEN), timeout=_CLIENT_TIMEOUT_S) as client,
    ):
        with pytest.raises(MCPError) as raised:
            await client.read_resource(_ARTIFACT_URI)

    assert str(raised.value) == (
        f"service_unavailable: reading this resource did not finish within the {LIMIT_S:g}s "
        "this server allows a request, so it was stopped. Read it again after 5s."
    )
    assert stalled_upstream.cancelled == [f"/api/datasets/spec_{'5' * 32}/cv_1"]


async def test_a_get_data_mint_stopped_at_the_limit_keeps_its_mint_ledger_slot(
    stalled_upstream: _StalledUpstream,
    empty_mint_ledger: MintLedger,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """dfo-apis keeps running a mint whose client went away, so the slot the
    cancelled POST took stays held, and a repeat of the call refuses rather than
    starting the same mint twice."""
    app = _limited_app()

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, "/", token=TOKEN), timeout=_CLIENT_TIMEOUT_S) as client,
    ):
        stopped = await client.call_tool(
            "get_data", {"spec_id": _UNSTORED_SPEC}, raise_on_error=False
        )
        repeated = await client.call_tool(
            "get_data", {"spec_id": _UNSTORED_SPEC}, raise_on_error=False
        )

    assert _envelope(stopped) == _stopped_envelope("get_data")
    assert stalled_upstream.cancelled == ["/api/datasets"]
    assert len(empty_mint_ledger._held) == 1
    assert repeated.structured_content is not None
    assert _envelope(repeated)["code"] == "service_unavailable"
    assert _envelope(repeated)["fix"]["spec_id"] == _UNSTORED_SPEC


class _JumpingClock:
    """Real monotonic time plus an offset a test moves forward."""

    def __init__(self) -> None:
        self.offset = 0.0

    def __call__(self) -> float:
        return time.monotonic() + self.offset


@pytest.mark.parametrize(
    ("poll_time_left_s", "outcome"),
    [(0.2, "operation_still_processing"), (3.0, "stopped")],
    ids=["deadline inside the limit", "deadline past the limit"],
)
async def test_get_data_refuses_on_its_own_deadline_before_the_limit_stops_it(
    poll_time_left_s: float,
    outcome: str,
    monkeypatch: pytest.MonkeyPatch,
    empty_mint_ledger: MintLedger,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """dfo-apis accepts the mint as a background operation that never finishes.
    When the POST is answered, the call's deadline clock jumps so its poll has
    `poll_time_left_s` of real time left. With that inside the limit, `get_data`
    answers its own refusal; past it, the limit stops the call."""
    limit_s = 1.5
    clock = _JumpingClock()
    monkeypatch.setattr(get_data_tool, "_deadline_clock", clock)
    operation = {
        "operation_id": "op_" + "0" * 64,
        "status": "running",
        "poll": "/api/datasets/operations/op_" + "0" * 64,
        "retry_after_ms": 50,
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            assert EXTERNAL_CALL_BUDGET.deadline_s is not None
            clock.offset += (
                EXTERNAL_CALL_BUDGET.deadline_s
                - EXTERNAL_CALL_BUDGET.post_mint_reserve_s
                - poll_time_left_s
            )
            return httpx2.Response(202, json={"operation": operation})
        assert request.url.path == operation["poll"]
        return httpx2.Response(200, json={"operation": operation})

    client_holder.set_client(
        UpstreamClient(settings=DataMcpSettings(), transport=httpx2.MockTransport(handler))
    )
    app = _limited_app(call_limit_s=limit_s)
    try:
        async with (
            running_lifespan(app),
            Client(
                mounted_mcp_transport(app, "/", token=TOKEN), timeout=_CLIENT_TIMEOUT_S
            ) as client,
        ):
            started = time.monotonic()
            result = await client.call_tool(
                "get_data", {"spec_id": _UNSTORED_SPEC}, raise_on_error=False
            )
            answered_after = time.monotonic() - started
    finally:
        client_holder.set_client(None)

    if outcome == "stopped":
        assert _envelope(result) == _stopped_envelope("get_data", limit_s)
        assert answered_after >= limit_s
    else:
        assert _envelope(result)["code"] == "operation_still_processing"
        assert answered_after < limit_s


async def _until(condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not condition():
            await asyncio.sleep(0.01)


async def test_the_in_flight_cap_refuses_the_next_call_at_once_and_admits_after_one_ends(
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    probe = FastMCP(name="probe")
    started: list[str] = []
    releases = {key: asyncio.Event() for key in ("first", "second", "third")}

    @probe.tool
    async def hold(key: str) -> str:
        started.append(key)
        await releases[key].wait()
        return key

    combined = build_combined_mcp(
        settings=Settings(mcp_api_token=TOKEN, combined_mcp_max_in_flight=2)
    )
    combined.mount(probe)
    mcp_auth.apply_gate(combined, token=TOKEN, disabled=False)
    app = combined_http_app(combined)

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, "/", token=TOKEN)) as client,
    ):
        first = asyncio.create_task(client.call_tool("hold", {"key": "first"}))
        second = asyncio.create_task(client.call_tool("hold", {"key": "second"}))
        await _until(lambda: len(started) == 2)
        refused = await client.call_tool("hold", {"key": "third"}, raise_on_error=False)
        started_when_refused = sorted(started)
        releases["first"].set()
        await first
        third = asyncio.create_task(client.call_tool("hold", {"key": "third"}))
        await _until(lambda: "third" in started)
        releases["second"].set()
        releases["third"].set()
        answers = [(await task).content[0].text for task in (second, third)]

    assert started_when_refused == ["first", "second"]
    assert refused.content[0].text.startswith(
        "This server is already running as many calls as it runs at once (2). Repeat the "
        "call unchanged after 5s."
    )
    assert _envelope(refused) == {
        "code": "rate_limited",
        "retry_mode": "retry_after_delay",
        "fix": {"max_in_flight": 2},
        "retry_after_s": 5.0,
    }
    assert answers == ["second", "third"]


def _probe_app(*, call_limit_s: float = EXTERNAL_CALL_LIMIT_S, max_in_flight: int = 16) -> Any:
    """The combined server with `_PROBE` mounted beside its children, and its
    in-flight limit, for tests that need tools with a known behaviour."""
    combined = build_combined_mcp(
        settings=Settings(mcp_api_token=TOKEN, combined_mcp_max_in_flight=max_in_flight),
        call_limit_s=call_limit_s,
    )
    combined.mount(_PROBE)
    mcp_auth.apply_gate(combined, token=TOKEN, disabled=False)
    [in_flight] = [m for m in combined.middleware if isinstance(m, combined_mcp._InFlightLimit)]
    return combined_http_app(combined), in_flight


_PROBE = FastMCP(name="probe")
_PROBE_STARTED: list[str] = []


@_PROBE.tool
async def hold() -> str:
    """Runs until cancelled."""
    _PROBE_STARTED.append("hold")
    await asyncio.Event().wait()
    return "unreachable"


@_PROBE.tool
def quick() -> str:
    return "done"


@_PROBE.tool
def explode() -> str:
    raise RuntimeError(_SECRET_FAILURE)


@_PROBE.tool
def times_out() -> str:
    raise TimeoutError("the upstream read timed out")


def _data_refusal(code: data_tool_errors.ToolErrorCode, **fields: Any) -> ToolResult:
    """A refusal as data-mcp's `to_tool_result` writes it."""
    return data_tool_errors.to_tool_result(
        data_tool_errors.ToolRefusal(code=code, message=code.value, fix={}, **fields)
    )


@_PROBE.tool
def refuses_upstream() -> ToolResult:
    """Refuses as data-mcp does when dfo-apis fails the call."""
    return _data_refusal(data_tool_errors.ToolErrorCode.UPSTREAM_ERROR, upstream_code="bad_shape")


@_PROBE.tool
def refuses_release_mismatch() -> ToolResult:
    """Refuses as presentation-mcp does for a release the platform does not serve."""
    return ToolResult(
        content="release_mismatch", meta=refusal_meta(code="release_mismatch"), is_error=True
    )


@_PROBE.tool
def refuses_caller() -> ToolResult:
    """Refuses as data-mcp does for a definition this deployment does not hold."""
    return _data_refusal(data_tool_errors.ToolErrorCode.DEFINITION_NOT_HELD)


@_PROBE.tool
def still_processing() -> ToolResult:
    """Refuses as `get_data` does when its deadline passes with the mint still running."""
    return _data_refusal(data_tool_errors.ToolErrorCode.OPERATION_STILL_PROCESSING)


@pytest.fixture(autouse=True)
def _clear_probe_starts() -> None:
    _PROBE_STARTED.clear()


async def test_the_in_flight_count_returns_to_zero_after_a_stopped_and_a_raised_call(
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """With a cap of one, each call below is admitted only if the one before it
    released its slot, whichever way it ended."""
    app, in_flight = _probe_app(call_limit_s=LIMIT_S, max_in_flight=1)

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, "/", token=TOKEN), timeout=_CLIENT_TIMEOUT_S) as client,
    ):
        stopped = await client.call_tool("hold", {}, raise_on_error=False)
        after_stopped = in_flight._in_flight
        raised = await client.call_tool("explode", {}, raise_on_error=False)
        after_raised = in_flight._in_flight
        admitted = await client.call_tool("quick", {}, raise_on_error=False)

    assert _envelope(stopped) == _stopped_envelope("hold")
    assert after_stopped == 0
    assert _envelope(raised)["code"] == "internal_error"
    assert after_raised == 0
    assert admitted.is_error is False
    assert in_flight._in_flight == 0


async def test_a_client_that_disconnects_mid_call_frees_its_in_flight_slot(
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """The limit is the default 45 s, so only the disconnect itself can free the slot
    within the 5 s this test waits."""
    app, in_flight = _probe_app(max_in_flight=1)

    async with running_lifespan(app):
        async with Client(mounted_mcp_transport(app, "/", token=TOKEN)) as client:
            call = asyncio.create_task(client.call_tool("hold", {}))
            await _until(lambda: _PROBE_STARTED == ["hold"])
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
        await _until(lambda: in_flight._in_flight == 0)
        async with Client(mounted_mcp_transport(app, "/", token=TOKEN)) as client:
            admitted = await client.call_tool("quick", {}, raise_on_error=False)

    assert admitted.is_error is False


async def test_a_resource_read_over_the_in_flight_cap_fails_at_once(
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    app, _ = _probe_app(call_limit_s=LIMIT_S * 4, max_in_flight=1)

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, "/", token=TOKEN), timeout=_CLIENT_TIMEOUT_S) as client,
    ):
        held = asyncio.create_task(client.call_tool("hold", {}, raise_on_error=False))
        await _until(lambda: _PROBE_STARTED == ["hold"])
        started = time.monotonic()
        with pytest.raises(MCPError) as refused:
            await client.read_resource(_ARTIFACT_URI)
        refused_after = time.monotonic() - started
        await held

    assert str(refused.value) == (
        "rate_limited: This server is already running as many calls as it runs at once (1). "
        "Read the resource again after 5s."
    )
    assert refused_after < LIMIT_S


async def test_a_timeout_error_a_tool_raises_itself_is_an_internal_error(
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """Only the limit's own expiry answers the limit's refusal."""
    app, _ = _probe_app()

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, "/", token=TOKEN)) as client,
    ):
        result = await client.call_tool("times_out", {}, raise_on_error=False)

    assert _envelope(result)["code"] == "internal_error"
    assert _envelope(result)["fix"]["tool"] == "times_out"


def _logged_trace_id(caplog: pytest.LogCaptureFixture) -> str:
    """The trace id on the one ERROR record `harness.combined_mcp` logged."""
    [record] = [
        r for r in caplog.records if r.name == "harness.combined_mcp" and r.levelno >= logging.ERROR
    ]
    match = re.search(r"trace_id=([0-9a-f]{32})", record.getMessage())
    assert match is not None
    return match[1]


async def test_an_internal_errors_logged_trace_id_is_the_calls_recorded_trace(
    exported_spans: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """With a tracer provider set, the logged exception names the trace of
    fastmcp's one SERVER span for the call, and the combined server adds no
    second one."""
    app, _ = _probe_app()

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, "/", token=TOKEN)) as client,
    ):
        with caplog.at_level(logging.ERROR, logger="harness.combined_mcp"):
            result = await client.call_tool("explode", {}, raise_on_error=False)

    assert "trace_id" not in _envelope(result)["fix"]
    trace_id = _logged_trace_id(caplog)
    [server_span] = [
        span
        for span in exported_spans.get_finished_spans()
        if span.kind is SpanKind.SERVER
        and span.attributes is not None
        and span.attributes.get("fastmcp.server.name") == combined_mcp.COMBINED_MCP_NAME
        and span.attributes.get("mcp.method.name") == "tools/call"
    ]
    assert server_span.context is not None
    assert trace_id == format_trace_id(server_span.context.trace_id)


@pytest.fixture
def no_tracer_provider() -> Iterator[None]:
    """No global tracer provider, as in a process with telemetry off."""
    saved_provider = otel_trace._TRACER_PROVIDER
    saved_once = otel_trace._TRACER_PROVIDER_SET_ONCE
    otel_trace._TRACER_PROVIDER = None
    otel_trace._TRACER_PROVIDER_SET_ONCE = Once()
    try:
        yield
    finally:
        otel_trace._TRACER_PROVIDER = saved_provider
        otel_trace._TRACER_PROVIDER_SET_ONCE = saved_once


async def test_without_a_tracer_provider_the_logged_exception_and_the_request_line_share_one_trace_id(
    no_tracer_provider: None,
    caplog: pytest.LogCaptureFixture,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    app, _ = _probe_app()

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, "/", token=TOKEN)) as client,
    ):
        with caplog.at_level(logging.INFO, logger="harness.combined_mcp"):
            result = await client.call_tool("explode", {}, raise_on_error=False)

    assert "trace_id" not in _envelope(result)["fix"]
    trace_id = _logged_trace_id(caplog)
    assert re.fullmatch(r"[0-9a-f]{32}", trace_id)
    [request_line] = [
        r.getMessage()
        for r in caplog.records
        if r.name == "harness.combined_mcp"
        and r.levelno == logging.INFO
        and r.getMessage().startswith("combined mcp call")
    ]
    assert "tool=explode outcome=internal_error " in request_line
    assert f" trace_id={trace_id} " in request_line


@pytest.fixture
def langfuse_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """The Langfuse backend, with a stub client whose observations record their
    updates: one `(name, updates)` entry per observation opened, in order."""
    opened: list[tuple[str, list[dict[str, Any]]]] = []

    def start_as_current_observation(*, name: str, as_type: str) -> Any:
        updates: list[dict[str, Any]] = []
        opened.append((name, updates))
        return nullcontext(SimpleNamespace(update=lambda **fields: updates.append(fields)))

    client = SimpleNamespace(start_as_current_observation=start_as_current_observation)
    monkeypatch.setattr(langfuse, "get_client", lambda: client)
    monkeypatch.setattr(telemetry, "_backend", "langfuse")
    return opened


async def test_each_calls_observation_records_its_outcome_under_langfuse(
    langfuse_observations: list[tuple[str, list[dict[str, Any]]]],
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    app, _ = _probe_app()

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, "/", token=TOKEN)) as client,
    ):
        for tool in (
            "quick",
            "no_such_tool",
            "still_processing",
            "refuses_upstream",
            "refuses_release_mismatch",
            "explode",
        ):
            await client.call_tool(tool, {}, raise_on_error=False)
        with pytest.raises(MCPError):
            await client.read_resource("ui://no-such-view")

    assert langfuse_observations == [
        ("combined mcp tools/call quick", []),
        (
            "combined mcp tools/call no_such_tool",
            [{"level": "WARNING", "status_message": "tool_not_found"}],
        ),
        (
            "combined mcp tools/call still_processing",
            [{"level": "DEFAULT", "status_message": "operation_still_processing"}],
        ),
        (
            "combined mcp tools/call refuses_upstream",
            [{"level": "ERROR", "status_message": "upstream_error: bad_shape"}],
        ),
        (
            "combined mcp tools/call refuses_release_mismatch",
            [{"level": "ERROR", "status_message": "release_mismatch"}],
        ),
        (
            "combined mcp tools/call explode",
            [{"level": "ERROR", "status_message": "internal_error"}],
        ),
        (
            "combined mcp resources/read",
            [{"level": "ERROR", "status_message": "ResourceError"}],
        ),
    ]


async def test_a_refusal_the_server_caused_marks_the_calls_server_span_as_an_error(
    exported_spans: InMemorySpanExporter,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """Under plain OTel the call's span is fastmcp's SERVER span. A refusal the
    caller caused leaves its status unset."""
    app, _ = _probe_app()

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, "/", token=TOKEN)) as client,
    ):
        for tool in ("refuses_upstream", "refuses_release_mismatch", "refuses_caller"):
            await client.call_tool(tool, {}, raise_on_error=False)

    statuses = {
        span.attributes.get("gen_ai.tool.name"): (span.status.status_code, span.status.description)
        for span in exported_spans.get_finished_spans()
        if span.kind is SpanKind.SERVER
        and span.attributes is not None
        and span.attributes.get("fastmcp.server.name") == combined_mcp.COMBINED_MCP_NAME
        and span.attributes.get("mcp.method.name") == "tools/call"
    }
    assert statuses == {
        "refuses_upstream": (StatusCode.ERROR, "upstream_error: bad_shape"),
        "refuses_release_mismatch": (StatusCode.ERROR, "release_mismatch"),
        "refuses_caller": (StatusCode.UNSET, None),
    }


@pytest.mark.parametrize("apps_host", [True, False], ids=["apps-host", "plain-host"])
async def test_a_presentation_refusal_records_its_code_with_or_without_a_view(
    apps_host: bool,
    langfuse_observations: list[tuple[str, list[dict[str, Any]]]],
    logging_configured: None,
    caplog: pytest.LogCaptureFixture,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """presentation-mcp names its code `error` in `structuredContent`, and an MCP
    Apps host gets a notice view there instead. The span and the log line name
    the code either way: the platform's `not_found` is the caller's to correct,
    and `transient_unavailable`, a platform the assembler could not reach, is
    the server's failure."""
    app = create_app(settings=Settings(mcp_api_token=TOKEN))
    transport = mounted_mcp_transport(app, COMBINED, token=TOKEN)

    async with (
        running_lifespan(app),
        Client(
            transport, mode="2026-07-28", extensions=_APPS_HOST if apps_host else None
        ) as client,
    ):
        with caplog.at_level(logging.INFO, logger="harness.combined_mcp"):
            revised = await client.call_tool(
                "write_component",
                {"intent": {"title": "Aid"}, "component_id": "cmp_9", "write_token": "cmw_1"},
                raise_on_error=False,
            )
            described = await client.call_tool(
                "describe_component", {"component_id": "cmp_9"}, raise_on_error=False
            )

    for result in (revised, described):
        assert result.is_error is True
        assert ("error" in (result.structured_content or {})) is not apps_host
    assert langfuse_observations == [
        (
            "combined mcp tools/call write_component",
            [{"level": "WARNING", "status_message": "not_found"}],
        ),
        (
            "combined mcp tools/call describe_component",
            [{"level": "ERROR", "status_message": "transient_unavailable"}],
        ),
    ]
    outcomes = [
        match.group(1)
        for record in caplog.records
        if record.name == "harness.combined_mcp"
        and (match := re.match(r"combined mcp call tool=\S+ outcome=(\S+) ", record.getMessage()))
    ]
    assert outcomes == ["not_found", "transient_unavailable"]


@pytest.mark.parametrize(
    ("code", "level"),
    [
        ("upstream_error", "ERROR"),
        ("service_unavailable", "ERROR"),
        ("internal_error", "ERROR"),
        ("frame_integrity_failed", "ERROR"),
        ("transient_unavailable", "ERROR"),
        ("timeout", "ERROR"),
        ("publish_not_configured", "ERROR"),
        ("publish_not_permitted", "ERROR"),
        ("publish_conflict", "ERROR"),
        ("release_mismatch", "ERROR"),
        ("capability_unavailable", "ERROR"),
        ("upstream_version_changed", "WARNING"),
        (None, "ERROR"),
        ("operation_still_processing", "DEFAULT"),
        ("rate_limited", "WARNING"),
        ("request_invalid", "WARNING"),
        ("definition_not_held", "WARNING"),
        ("spec_not_exportable", "WARNING"),
        ("binding_invalid", "WARNING"),
        ("write_refused", "WARNING"),
    ],
)
def test_a_refusals_code_sets_its_level(code: str | None, level: str) -> None:
    assert combined_mcp._refusal_level(code) == level


@REVISIONS
async def test_each_call_through_the_combined_server_is_logged(
    mode: str,
    revision: str,
    logging_configured: None,
    caplog: pytest.LogCaptureFixture,
    running_lifespan: Callable[..., Any],
) -> None:
    app = create_app(settings=Settings(mcp_api_token=TOKEN))
    transport = StreamableHttpTransport(
        url=f"http://testserver{COMBINED}",
        auth=TOKEN,
        headers={"x-dfo-caller": "ip:test"},
        httpx_client_factory=cast(
            McpHttpClientFactory,
            lambda **kwargs: httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app), base_url="http://testserver", **kwargs
            ),
        ),
    )

    async with running_lifespan(app):
        handles = await seed_fixture_store(get_frame_store())
        with caplog.at_level(logging.INFO, logger="harness.combined_mcp"):
            async with Client(transport, mode=mode) as client:
                await client.call_tool(
                    "aggregate", {"frame_ref": handles.plain_single_unit, "aggregation": "sum"}
                )
                await client.call_tool(
                    "aggregate",
                    {"frame_ref": UNHELD_FRAME, "aggregation": "sum"},
                    raise_on_error=False,
                )

    messages = [
        record.getMessage() for record in caplog.records if record.name == "harness.combined_mcp"
    ]
    # fastmcp's own client names itself `mcp` and declares no MCP Apps extension.
    # A 2025-11-25 client says so on initialize alone.
    host = 'client="mcp" client_version="0.1.0" views=undeclared view_mime_types=None'
    initialize = [f"combined mcp initialize outcome=ok revision={revision} caller=ip:test {host}"]
    unseen = "client=None client_version=None views=unknown view_mime_types=None"
    call_host = unseen if revision == "2025-11-25" else host
    assert [re.sub(r"trace_id=[0-9a-f]{32}", "trace_id=<id>", message) for message in messages] == [
        *(initialize if revision == "2025-11-25" else []),
        *(
            f"combined mcp call tool=aggregate outcome={outcome} revision={revision} "
            f"caller=ip:test trace_id=<id> {call_host}"
            for outcome in ("ok", "frame_not_held")
        ),
    ]
    trace_ids = [re.findall(r"trace_id=([0-9a-f]{32})", message) for message in messages]
    assert trace_ids[-2] != trace_ids[-1]


_HEADER_WRITE_TOKEN = "cmw_header" + "7" * 24
_ARGUMENT_WRITE_TOKEN = "cmw_argument" + "8" * 24


@REVISIONS
@pytest.mark.parametrize("apps_host", [True, False], ids=["apps-host", "plain-host"])
async def test_the_request_log_names_the_host_and_what_it_declared_about_views(
    mode: str,
    revision: str,
    apps_host: bool,
    logging_configured: None,
    caplog: pytest.LogCaptureFixture,
    running_lifespan: Callable[..., Any],
) -> None:
    """Each line carries the client's name and version and its MCP Apps
    declaration where the revision delivers them, a resource read's URI with an
    artifact URI cut to its scheme and host, and no header, argument or token.

    A 2025-11-25 client declares itself on initialize alone, so its calls to
    this stateless server log `views=unknown`, the reading presentation-mcp
    answers them by. A client name is capped and written as a JSON string.
    """
    app = create_app(settings=Settings(mcp_api_token=TOKEN))
    name = 'host" views=declared\n' + "n" * 300
    transport = StreamableHttpTransport(
        url=f"http://testserver{COMBINED}",
        auth=TOKEN,
        headers={"x-dfo-caller": "ip:test", WRITE_TOKEN_HEADER: _HEADER_WRITE_TOKEN},
        httpx_client_factory=cast(
            McpHttpClientFactory,
            lambda **kwargs: httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app), base_url="http://testserver", **kwargs
            ),
        ),
    )
    artifact = f"dfo://artifact/spec_{'1' * 32}/cv_{'2' * 32}/sha256_{'3' * 64}"

    async with running_lifespan(app):
        with caplog.at_level(logging.INFO):
            async with Client(
                transport,
                mode=mode,
                client_info=mt.Implementation(name=name, version="2.5.0"),
                extensions=_APPS_HOST if apps_host else None,
            ) as client:
                await client.call_tool(
                    "write_component",
                    {
                        "intent": {"title": "Aid"},
                        "component_id": "cmp_9",
                        "write_token": _ARGUMENT_WRITE_TOKEN,
                    },
                    raise_on_error=False,
                )
                await client.read_resource(presentation_server._RENDERER_URI)
                with pytest.raises(MCPError):
                    await client.read_resource(artifact)

    messages = [r.getMessage() for r in caplog.records if r.name == "harness.combined_mcp"]
    views = "declared" if apps_host else "undeclared"
    mime_types = json.dumps([UI_MIME_TYPE]) if apps_host else "None"
    declared = f'client={json.dumps(name[:128])} client_version="2.5.0" views={views} ' + (
        f"view_mime_types={mime_types}"
    )
    unseen = "client=None client_version=None views=unknown view_mime_types=None"
    initialize = [
        f"combined mcp initialize outcome=ok revision={revision} caller=ip:test {declared}"
    ]
    call_host = unseen if revision == "2025-11-25" else declared
    assert [re.sub(r" trace_id=[0-9a-f]{32}", "", message) for message in messages] == [
        *(initialize if revision == "2025-11-25" else []),
        f"combined mcp call tool=write_component outcome=not_found revision={revision} "
        f"caller=ip:test {call_host}",
        f"combined mcp call resource outcome=ok revision={revision} caller=ip:test {call_host} "
        f"uri={json.dumps(presentation_server._RENDERER_URI)}",
        f"combined mcp call resource outcome=ArtifactRefusalError revision={revision} caller=ip:test "
        f'{call_host} uri="dfo://artifact"',
    ]
    logged = "\n".join(
        f"{record.getMessage()} {record.args!r} {record.exc_text or ''}"
        for record in caplog.records
    )
    for secret in (TOKEN, _HEADER_WRITE_TOKEN, _ARGUMENT_WRITE_TOKEN):
        assert secret not in logged
    assert all(artifact.split("/", 3)[3] not in message for message in messages)


@REVISIONS
async def test_a_client_supplied_tool_name_cannot_forge_a_log_field(
    mode: str,
    revision: str,
    logging_configured: None,
    caplog: pytest.LogCaptureFixture,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """A plain tool name is logged bare, and any other name, which the server
    may not know, as a capped JSON string."""
    app = create_app(settings=Settings(mcp_api_token=TOKEN))
    odd = "x y=z\nw"

    async with running_lifespan(app):
        with caplog.at_level(logging.INFO, logger="harness.combined_mcp"):
            async with Client(
                mounted_mcp_transport(app, COMBINED, token=TOKEN), mode=mode
            ) as client:
                await client.call_tool(odd, {}, raise_on_error=False)
                await client.call_tool("summarise", {}, raise_on_error=False)
                await client.call_tool("n" * 300, {}, raise_on_error=False)

    lines = [
        r.getMessage()
        for r in caplog.records
        if r.name == "harness.combined_mcp" and r.getMessage().startswith("combined mcp call")
    ]
    assert [line.split(" outcome=")[0] for line in lines] == [
        f"combined mcp call tool={json.dumps(odd)}",
        "combined mcp call tool=summarise",
        f"combined mcp call tool={json.dumps('n' * 128)}",
    ]
    assert all(line.count(" outcome=") == 1 and "\n" not in line for line in lines)


@pytest.mark.parametrize(
    ("uri", "logged"),
    [
        ("dfo://[x/y", '"dfo://[x"'),
        ("dfo://user:pw@artifact/a/b", '"dfo://artifact"'),
        ("artifact/spec_1", '""'),
    ],
)
async def test_a_malformed_resource_uri_is_logged_and_the_read_fails_as_usual(
    uri: str,
    logged: str,
    logging_configured: None,
    caplog: pytest.LogCaptureFixture,
    running_lifespan: Callable[..., Any],
) -> None:
    """The line names the URI's scheme and host without any userinfo, and the
    read fails with the error it would have without the log. The request is
    raw, since the MCP client refuses a malformed URI before sending it."""
    app = create_app(settings=Settings(mcp_api_token=TOKEN))
    request = {"jsonrpc": "2.0", "id": 1, "method": "resources/read", "params": {"uri": uri}}

    async with (
        running_lifespan(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://testserver"
        ) as http,
    ):
        with caplog.at_level(logging.INFO, logger="harness.combined_mcp"):
            response = await http.post(
                COMBINED,
                json=request,
                headers={
                    "Authorization": f"Bearer {TOKEN}",
                    "Accept": "application/json, text/event-stream",
                },
            )

    assert "error" in response.text
    [line] = [
        r.getMessage()
        for r in caplog.records
        if r.name == "harness.combined_mcp" and r.getMessage().startswith("combined mcp call")
    ]
    assert line.startswith("combined mcp call resource outcome=ResourceError ")
    assert line.endswith(f" uri={logged}")
    assert "pw" not in line


# A servable artifact, and one whose snapshot dfo-apis refuses.
_SERVED_PATH = f"spec_{'6' * 32}/cv_served/sha256:{'6' * 64}"
_REFUSED_PATH = f"spec_{'6' * 32}/cv_refused/sha256:{'7' * 64}"


def _snapshot(request: httpx2.Request) -> httpx2.Response:
    """dfo-apis' snapshot route: `cv_served` lists the served artifact
    under its stable link, and any other content version is not found."""
    subject_id, content_version, digest = _SERVED_PATH.split("/")
    if not request.url.path.endswith(f"/{content_version}"):
        return httpx2.Response(
            404,
            json={"code": "snapshot_not_found", "title": "refused", "retryable": False},
            headers={"content-type": "application/problem+json"},
        )
    return httpx2.Response(
        200,
        json={
            "schema_version": 1,
            "definition": {"schema_version": 1, "spec_id": subject_id, "identity": {}},
            "content_version": content_version,
            "coverage": {"rows": 1, "dimensions": []},
            "citation": {"text": "OECD", "sources": [{"ref": "OECD", "label": "OECD"}]},
            "trace": {},
            "warnings": [],
            "artifacts": [
                {
                    "artifact_digest": digest,
                    "rendering": {"bundle": False, "format": "parquet", "layout": "frame"},
                    "bytes": 100,
                    "media_type": "application/octet-stream",
                    "created_at": "2026-08-11T00:00:00+00:00",
                    "tier": "working",
                    "url": f"/api/datasets/{subject_id}/{content_version}/files/{digest}",
                }
            ],
        },
    )


class _Records(logging.Handler):
    """Keeps every record a logger emits, whatever its propagation."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    @override
    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def every_record() -> Iterator[list[logging.LogRecord]]:
    """Every record from fastmcp's logger, which does not propagate, and from the
    root logger, at every level."""
    handler = _Records()
    loggers = [logging.getLogger("fastmcp"), logging.getLogger()]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        for logger, level in zip(loggers, levels, strict=True):
            logger.removeHandler(handler)
            logger.setLevel(level)


async def _read_artifacts(app: Any, *, mode: str, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Read the served artifact, the refused one, and one whose read raises,
    through the combined server, and return the text of each failure."""
    transport = StreamableHttpTransport(
        url=f"http://testserver{COMBINED}",
        auth=TOKEN,
        httpx_client_factory=cast(
            McpHttpClientFactory,
            lambda **kwargs: httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app), base_url="http://testserver", **kwargs
            ),
        ),
    )

    def unreachable() -> UpstreamClient:
        raise RuntimeError(_SECRET_FAILURE)

    failures: list[str] = []
    async with Client(transport, mode=mode) as client:
        await client.read_resource(f"dfo://artifact/{_SERVED_PATH}")
        with pytest.raises(MCPError) as refused:
            await client.read_resource(f"dfo://artifact/{_REFUSED_PATH}")
        failures.append(str(refused.value))
        monkeypatch.setattr(artifact_resource, "get_client", unreachable)
        with pytest.raises(MCPError) as raised:
            await client.read_resource(f"dfo://artifact/{_SERVED_PATH}")
        failures.append(str(raised.value))
    return failures


@pytest.fixture
def snapshot_upstream() -> Iterator[None]:
    """data-mcp's upstream client, answering the snapshot route through `_snapshot`."""
    client_holder.set_client(
        UpstreamClient(settings=DataMcpSettings(), transport=httpx2.MockTransport(_snapshot))
    )
    try:
        yield
    finally:
        client_holder.set_client(None)


def _holds_an_artifact_path(text: str) -> bool:
    return any(
        part in text
        for path in (_SERVED_PATH, _REFUSED_PATH)
        for part in (path, path.rsplit("/", 1)[1], path.rsplit(":", 1)[1])
    )


@REVISIONS
async def test_an_artifact_uri_reaches_no_log_record_or_failure_text(
    mode: str,
    revision: str,
    logging_configured: None,
    every_record: list[logging.LogRecord],
    snapshot_upstream: None,
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
) -> None:
    """A `dfo://artifact` URI is a bearer credential. fastmcp logs a failed
    read's whole URI with its traceback, from its own logger, in both the
    mounted data-mcp server and the combined server. Both lines stay at DEBUG,
    data-mcp logs each failure without the URI, and no record at INFO or above,
    nor any failure text, holds an artifact's path."""
    del revision
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    async with running_lifespan(app):
        failures = await _read_artifacts(app, mode=mode, monkeypatch=monkeypatch)

    assert failures[0].startswith("snapshot_not_found: ")
    assert failures[1].startswith("internal_error: Reading this resource failed")
    fastmcp_lines = [r for r in every_record if r.getMessage().startswith("Error reading resource")]
    assert len(fastmcp_lines) == 4
    assert {r.levelno for r in fastmcp_lines} == {logging.DEBUG}
    assert all(_holds_an_artifact_path(r.getMessage()) for r in fastmcp_lines)
    [refused, raised] = [r for r in every_record if r.name == "data_mcp.artifact_resource"]
    assert refused.levelno == logging.WARNING
    assert refused.getMessage().startswith("artifact read refused code=snapshot_not_found ")
    assert raised.levelno == logging.ERROR
    assert raised.exc_info is not None
    assert _SECRET_FAILURE in str(raised.exc_info[1])
    formatter = logging.Formatter()
    for record in every_record:
        if record.levelno < logging.INFO:
            continue
        logged = record.getMessage()
        if record.exc_info:
            logged += formatter.formatException(record.exc_info)
        assert not _holds_an_artifact_path(logged), record.name
    assert not any(_holds_an_artifact_path(failure) for failure in failures)


async def test_an_artifact_uri_reaches_no_exported_span(
    exported_spans: InMemorySpanExporter,
    snapshot_upstream: None,
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
) -> None:
    """fastmcp names a read's whole URI on its spans; `UriRedactingExporter`,
    which the OTLP backend exports through, cuts each to `dfo://artifact`."""
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    async with running_lifespan(app):
        await _read_artifacts(app, mode="2026-07-28", monkeypatch=monkeypatch)

    recorded = exported_spans.get_finished_spans()
    assert any(_holds_an_artifact_path(span.to_json()) for span in recorded)
    exported = InMemorySpanExporter()
    UriRedactingExporter(exported).export(recorded)
    spans = exported.get_finished_spans()
    assert len(spans) == len(recorded)
    assert not any(_holds_an_artifact_path(span.to_json()) for span in spans)
    uris = {
        span.attributes["mcp.resource.uri"]
        for span in spans
        if span.attributes and "mcp.resource.uri" in span.attributes
    }
    assert uris == {"dfo://artifact"}


def _timing_bodies(app: ASGIApp, sent: list[tuple[float, bytes]]) -> ASGIApp:
    """`app`, recording when each response body chunk is sent and what it holds."""

    async def timed(scope: Scope, receive: Receive, send: Send) -> None:
        async def recording(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.body":
                sent.append((time.monotonic(), message.get("body", b"")))
            await send(message)

        await app(scope, receive, recording)

    return timed


@REVISIONS
async def test_a_slow_call_is_kept_alive_with_pings_under_20s_apart(
    mode: str,
    revision: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """A proxy on the way cuts a response after 100 s without a byte. Both
    revisions send an SSE ping comment every 15 s while a call runs."""
    probe = FastMCP(name="probe")

    @probe.tool
    async def slow() -> str:
        await asyncio.sleep(21)
        return "done"

    combined = build_combined_mcp(settings=Settings(mcp_api_token=TOKEN))
    combined.mount(probe)
    mcp_auth.apply_gate(combined, token=TOKEN, disabled=False)
    app = combined_http_app(combined)
    sent: list[tuple[float, bytes]] = []

    async with running_lifespan(app):
        transport = mounted_mcp_transport(_timing_bodies(app, sent), "/", token=TOKEN)
        async with Client(transport, mode=mode, timeout=60) as client:
            assert client.protocol_version == revision
            sent.clear()
            started = time.monotonic()
            result = await client.call_tool("slow", {})
            ended = time.monotonic()

    pings = [at for at, body in sent if body.startswith(b": ping")]
    marks = [started, *pings, ended]
    assert result.content[0].text == "done"  # shaped: the text alone
    assert ended - started > 20
    assert pings
    assert max(later - earlier for earlier, later in pairwise(marks)) < 20


async def test_the_combined_server_refuses_an_unauthenticated_request(
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    async with running_lifespan(app):
        statuses: list[int] = []
        transport = mounted_mcp_transport(app, COMBINED, statuses=statuses)
        with pytest.raises(MCPError):
            async with Client(transport) as client:
                await client.list_tools()

    assert set(statuses) == {401}


async def test_healthz_reports_each_mounted_server(running_lifespan: Callable[..., Any]) -> None:
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    async with (
        running_lifespan(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://testserver"
        ) as client,
    ):
        response = await client.get("/mcp/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "servers": {
            "data-mcp": {"status": "ok"},
            "analysis-mcp": {"status": "ok"},
            "presentation-mcp": {"status": "ok"},
        },
    }


async def test_healthz_is_degraded_when_a_child_reports_degraded_with_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """presentation-mcp answers its own `/healthz` 200 with a degraded body
    when it cannot reach the platform; the combined check reads the body."""
    degraded = {"status": "degraded", "reason": "could not reach the platform: boom"}
    monkeypatch.setattr(presentation_server, "_health", degraded)
    app = combined_http_app(build_combined_mcp(settings=Settings(mcp_api_token=TOKEN)))

    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get("/healthz")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "servers": {
            "data-mcp": {"status": "ok"},
            "analysis-mcp": {"status": "ok"},
            "presentation-mcp": degraded,
        },
    }


def test_no_child_custom_route_reaches_the_combined_app() -> None:
    """Every child registers `/healthz`, and presentation-mcp its theme
    stylesheet. FastMCP would copy each onto the parent's app, outside the
    auth gate; the parent serves only its MCP endpoint and its own `/healthz`."""
    combined = build_combined_mcp(settings=Settings(mcp_api_token=TOKEN))
    app = combined_http_app(combined)

    assert [getattr(route, "path", None) for route in app.routes] == ["/", "/healthz"]


class _Alternating(httpx2.AsyncBaseTransport):
    """Sends each request to whichever app `target` names at that moment,
    recording every request's headers."""

    def __init__(self, apps: dict[str, Any]) -> None:
        self._transports = {name: httpx2.ASGITransport(app=app) for name, app in apps.items()}
        self.target = next(iter(apps))
        self.served: list[tuple[str, httpx2.Headers]] = []

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        self.served.append((self.target, request.headers))
        return await self._transports[self.target].handle_async_request(request)


async def test_a_2025_client_is_served_by_any_instance_without_a_session(
    running_lifespan: Callable[..., Any],
) -> None:
    """Two app instances share the process's frame store, as Cloud Run
    instances share one bucket. A 2025-11-25 client initializes against
    one and chains a call to the other, and neither issues a session."""
    first = create_app(settings=Settings(mcp_api_token=TOKEN))
    second = create_app(settings=Settings(mcp_api_token=TOKEN))
    alternating = _Alternating({"first": first, "second": second})
    response_headers: list[httpx2.Headers] = []

    async def record(response: httpx2.Response) -> None:
        response_headers.append(response.headers)

    def factory(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=alternating,
            base_url="http://testserver",
            event_hooks={"response": [record]},
            **kwargs,
        )

    transport = StreamableHttpTransport(
        url=f"http://testserver{COMBINED}",
        auth=TOKEN,
        httpx_client_factory=cast(McpHttpClientFactory, factory),
    )

    async with running_lifespan(first), running_lifespan(second):
        handles = await seed_fixture_store(get_frame_store())
        async with Client(transport, mode="legacy") as client:
            assert client.protocol_version == "2025-11-25"
            aggregated = await client.call_tool(
                "aggregate",
                {
                    "frame_ref": handles.plain_single_unit,
                    "aggregation": "sum",
                    "group_by": ["reporter"],
                },
            )
            alternating.target = "second"
            # The combined server returns only the digest, which names the frame.
            named = re.search(r"spec_id=(\w+), content_version=(\w+)", aggregated.content[0].text)
            assert named is not None
            frame_ref = {"spec_id": named[1], "content_version": named[2]}
            chained = await client.call_tool(
                "aggregate", {"frame_ref": frame_ref, "aggregation": "sum"}
            )

    assert chained.is_error is False
    assert {name for name, _ in alternating.served} == {"first", "second"}
    assert not any("mcp-session-id" in headers for _, headers in alternating.served)
    assert not any("mcp-session-id" in headers for headers in response_headers)


_ENTRY: ContextVar[str] = ContextVar("entry", default="direct")


class _MarkEntry(Middleware):
    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        token = _ENTRY.set("combined")
        try:
            return await call_next(context)
        finally:
            _ENTRY.reset(token)


@REVISIONS
async def test_a_contextvar_set_in_parent_middleware_reaches_a_mounted_childs_tool(
    mode: str,
    revision: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """A parent `on_call_tool` is how a call through the combined server
    tells a child's tool which entry point it came in by. The child reached
    directly never runs the parent's middleware."""
    probe = FastMCP(name="probe")

    @probe.tool
    def entry() -> str:
        return _ENTRY.get()

    combined = build_combined_mcp(settings=Settings(mcp_api_token=TOKEN))
    combined.mount(probe)
    combined.add_middleware(_MarkEntry())
    mcp_auth.apply_gate(combined, token=TOKEN, disabled=False)
    app = combined_http_app(combined)

    async with running_lifespan(app):
        transport = mounted_mcp_transport(app, "/", token=TOKEN)
        async with Client(transport, mode=mode) as client:
            assert client.protocol_version == revision
            through_parent = await client.call_tool("entry", {})
    async with Client(probe) as client:
        direct = await client.call_tool("entry", {})

    assert through_parent.content[0].text == "combined"  # shaped: the text alone
    assert direct.data == "direct"


async def test_startup_refuses_a_combined_server_that_replaces_a_childs_store() -> None:
    """The check itself, against a lifespan that swaps the definition store."""

    @asynccontextmanager
    async def replacing(_app: object) -> AsyncIterator[None]:
        set_store(InMemoryDefinitionStore())
        yield

    set_store(InMemoryDefinitionStore())
    try:
        with pytest.raises(RuntimeError, match="replaced the definition store"):
            async with children_start_once(replacing)(object()):
                pass
    finally:
        set_store(None)


async def test_startup_refuses_a_parent_that_mounts_a_second_copy_of_a_child() -> None:
    """A second `FastMCP` built with data-mcp's lifespan keeps its own
    reference count, so the parent runs that lifespan again."""
    direct_app = data_server.mcp.http_app(transport="streamable-http")
    parent = FastMCP(name="parent")
    parent.mount(FastMCP(name="data-mcp", lifespan=data_server._lifespan))
    parent_app = combined_http_app(parent)
    lifespan = combine_lifespans(direct_app.lifespan, children_start_once(parent_app.lifespan))

    with pytest.RaisesGroup(
        pytest.RaisesExc(RuntimeError, match="replaced the definition store"),
        flatten_subgroups=True,
    ):
        async with lifespan(Starlette()):
            pass


# Skills: the `skill://` resources data-mcp serves, and the Skills over MCP
# extension the parent registers.

_SKILL_MD = "---\nname: oda-concepts\ndescription: Core ODA concepts.\n---\n\n# ODA concepts\n"
_SKILL_MAX_AGE_S = 300


def _skill_entry(skill_id: str, *, attached: bool = True, excluded: bool = False) -> dict[str, Any]:
    """One Skills API list entry, whose manifest holds `SKILL.md` alone."""
    return {
        "id": skill_id,
        "frontmatter": {"name": skill_id, "description": "Core ODA concepts."},
        "attached": attached,
        "excluded": excluded,
        "resources": [
            {
                "path": "SKILL.md",
                "size": len(_SKILL_MD.encode()),
                "digest": "sha256:" + "a" * 64,
                "media_type": "text/markdown",
            }
        ],
    }


_SKILL_CATALOGUE = [
    _skill_entry("dfo-faq", attached=False),
    _skill_entry("oda-channels", excluded=True),
    _skill_entry("oda-concepts"),
]


def _skills_api(request: httpx2.Request) -> httpx2.Response:
    """A Skills API serving `_SKILL_CATALOGUE`, whose `oda-concepts` holds `SKILL.md` alone."""
    cache = {"cache-control": f"public, max-age={_SKILL_MAX_AGE_S}"}
    path = request.url.path
    if path == "/api/skills":
        return httpx2.Response(
            200, json={"skills": _SKILL_CATALOGUE, "warnings": []}, headers=cache
        )
    if path == "/api/skills/oda-concepts":
        return httpx2.Response(
            200, json={**_skill_entry("oda-concepts"), "content_text": _SKILL_MD}, headers=cache
        )
    if path == "/api/skills/oda-concepts/resources/SKILL.md":
        body = {
            "skill_id": "oda-concepts",
            **_skill_entry("oda-concepts")["resources"][0],
            "content_text": _SKILL_MD,
        }
        return httpx2.Response(200, json=body, headers=cache)
    if path == "/api/skills/healthz":
        return httpx2.Response(200, json={"status": "ok"})
    if path.startswith("/api/skills/oda-concepts/resources/"):
        return _skills_refusal(404, "skill_resource_not_found")
    return _skills_refusal(404, "skill_not_found")


def _skills_refusal(status: int, code: str) -> httpx2.Response:
    return httpx2.Response(
        status,
        json={"code": code, "title": "refused", "detail": "because", "retryable": status >= 500},
        headers={"content-type": "application/problem+json"},
    )


def _skills_unmounted(request: httpx2.Request) -> httpx2.Response:
    del request
    return _skills_refusal(404, "not_found")


def _skills_building(request: httpx2.Request) -> httpx2.Response:
    del request
    return _skills_refusal(503, "service_unavailable")


@contextmanager
def _skills_upstream(handler: Callable[[httpx2.Request], httpx2.Response]) -> Iterator[None]:
    """data-mcp's upstream client, answering through `handler`."""
    client_holder.set_client(
        UpstreamClient(settings=DataMcpSettings(), transport=httpx2.MockTransport(handler))
    )
    try:
        yield
    finally:
        client_holder.set_client(None)


class _AnyResult(BaseModel):
    """Any result, every field kept as sent."""

    model_config = ConfigDict(extra="allow")


async def _extension_request(client: Client, method: str, **params: Any) -> dict[str, Any]:
    """`method` sent as its SEP-2640 params model, without the result's `_meta`."""
    params_type = SkillsGetParams if method == "skills/get" else SkillsListParams
    result = await client.session.send_request(
        mt.Request(method=method, params=params_type.model_validate(params)),
        _AnyResult,
    )
    return {key: value for key, value in (result.model_extra or {}).items() if key != "_meta"}


@REVISIONS
async def test_the_combined_server_lists_the_skills_search_can_name(
    mode: str,
    revision: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    with _skills_upstream(_skills_api):
        async with (
            running_lifespan(app),
            Client(mounted_mcp_transport(app, COMBINED, token=TOKEN), mode=mode) as client,
        ):
            assert client.protocol_version == revision
            listed = [str(r.uri) for r in await client.list_resources()]
            [skill_md] = await client.read_resource("skill://oda-concepts/SKILL.md")

    assert [uri for uri in listed if uri.startswith("skill://")] == [
        "skill://oda-concepts/SKILL.md",
        "skill://oda-concepts/_manifest",
    ]
    assert isinstance(skill_md, mt.TextResourceContents)
    assert (skill_md.text, skill_md.mime_type) == (_SKILL_MD, "text/markdown")


@REVISIONS
@pytest.mark.parametrize(
    ("handler", "uri", "message"),
    [
        (
            _skills_api,
            "skill://no-such-skill/SKILL.md",
            "skill_not_found: No skill has the id 'no-such-skill'.",
        ),
        (
            _skills_api,
            "skill://oda-concepts/references/missing.md",
            "skill_resource_not_found: Skill 'oda-concepts' has no file "
            "'references/missing.md'; get_skill lists its files.",
        ),
        (
            _skills_unmounted,
            "skill://oda-concepts/_manifest",
            "capability_unavailable: This deployment serves no skills.",
        ),
        (
            _skills_building,
            "skill://oda-concepts/SKILL.md",
            "service_unavailable: The skill catalogue is still loading; retry shortly.",
        ),
    ],
    ids=["unknown-skill", "unknown-file", "unmounted", "building"],
)
async def test_each_skills_api_refusal_reaches_an_outside_client_as_written(
    mode: str,
    revision: str,
    handler: Callable[[httpx2.Request], httpx2.Response],
    uri: str,
    message: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    with _skills_upstream(handler):
        async with (
            running_lifespan(app),
            Client(mounted_mcp_transport(app, COMBINED, token=TOKEN), mode=mode) as client,
        ):
            assert client.protocol_version == revision
            with pytest.raises(MCPError) as refused:
                await client.read_resource(uri)

    assert str(refused.value) == message


async def test_a_2025_11_25_client_sees_resources_and_no_extensions(
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, COMBINED, token=TOKEN), mode="legacy") as client,
    ):
        initialized = client.initialize_result

    assert initialized is not None
    assert initialized.protocol_version == "2025-11-25"
    assert initialized.capabilities.resources is not None
    assert initialized.capabilities.extensions is None


async def test_a_2026_07_28_clients_discovery_declares_the_skills_extension(
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, COMBINED, token=TOKEN), mode="2026-07-28") as client,
    ):
        discovered = await client.session.send_discover("2026-07-28")

    capabilities = discovered["capabilities"]
    assert "resources" in capabilities
    assert capabilities["extensions"][SKILLS_EXTENSION_ID] == {}


async def test_skills_list_and_skills_get_answer_through_the_combined_server(
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    app = create_app(settings=Settings(mcp_api_token=TOKEN))
    entry = {
        "uri": "skill://oda-concepts/SKILL.md",
        "frontmatter": {"name": "oda-concepts", "description": "Core ODA concepts."},
        "resources": [
            {
                "uri": "skill://oda-concepts/SKILL.md",
                "digest": "sha256:" + "a" * 64,
                "size": len(_SKILL_MD.encode()),
            }
        ],
    }
    cache = {"ttlMs": _SKILL_MAX_AGE_S * 1000, "cacheScope": "public"}

    with _skills_upstream(_skills_api):
        async with (
            running_lifespan(app),
            Client(mounted_mcp_transport(app, COMBINED, token=TOKEN), mode="2026-07-28") as client,
        ):
            listed = await _extension_request(client, "skills/list")
            got = await _extension_request(
                client, "skills/get", uri="skill://oda-concepts/SKILL.md"
            )

    assert listed == {"resultType": "complete", "skills": [entry], **cache}
    assert got == {"resultType": "complete", "skill": entry, **cache}


@pytest.mark.parametrize(
    ("handler", "method", "uri", "code", "message"),
    [
        (
            _skills_api,
            "skills/get",
            "skill://oda-concepts/_manifest",
            INVALID_PARAMS,
            "request_invalid: 'skill://oda-concepts/_manifest' is not a skill URI; skills/get takes "
            "skill://{id}/SKILL.md, as skills/list names it.",
        ),
        (
            _skills_api,
            "skills/get",
            "skill://no-such-skill/SKILL.md",
            INVALID_PARAMS,
            "skill_not_found: No skill has the id 'no-such-skill'.",
        ),
        (
            _skills_unmounted,
            "skills/list",
            None,
            INTERNAL_ERROR,
            "capability_unavailable: This deployment serves no skills.",
        ),
        (
            _skills_building,
            "skills/get",
            "skill://oda-concepts/SKILL.md",
            INTERNAL_ERROR,
            "service_unavailable: The skill catalogue is still loading; retry shortly.",
        ),
    ],
    ids=["malformed-uri", "unknown-skill", "unmounted", "building"],
)
async def test_the_extension_fails_with_the_json_rpc_code_sep_2640_names(
    handler: Callable[[httpx2.Request], httpx2.Response],
    method: str,
    uri: str | None,
    code: int,
    message: str,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    with _skills_upstream(handler):
        async with (
            running_lifespan(app),
            Client(mounted_mcp_transport(app, COMBINED, token=TOKEN), mode="2026-07-28") as client,
        ):
            with pytest.raises(MCPError) as refused:
                await _extension_request(client, method, **({"uri": uri} if uri else {}))

    assert (refused.value.code, refused.value.message) == (code, message)


async def test_an_extension_method_over_the_in_flight_cap_fails_at_once(
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """The extension shares the in-flight cap tool calls and resource reads count against."""
    app, _ = _probe_app(max_in_flight=1)

    with _skills_upstream(_skills_api):
        async with (
            running_lifespan(app),
            Client(
                mounted_mcp_transport(app, "/", token=TOKEN),
                mode="2026-07-28",
                timeout=_CLIENT_TIMEOUT_S,
            ) as client,
        ):
            held = asyncio.create_task(client.call_tool("hold", {}, raise_on_error=False))
            await _until(lambda: _PROBE_STARTED == ["hold"])
            with pytest.raises(MCPError) as refused:
                await _extension_request(client, "skills/list")
            held.cancel()

    assert (refused.value.code, refused.value.message) == (
        INTERNAL_ERROR,
        "rate_limited: This server is already running as many calls as it runs at once (1). "
        "Repeat the request after 5s.",
    )


async def test_stalled_concurrent_resource_listings_take_slots_one_at_a_time_and_stop_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., StreamableHttpTransport],
) -> None:
    """`resources/list` reads the skill catalogue upstream, so it takes an in-flight
    slot and runs within the skills read limit. Of two concurrent listings whose
    upstream read stalls, with a cap of one, one is refused at once and the other
    is stopped at the limit, and only one upstream read starts."""
    started: list[str] = []

    async def stalled_list() -> Any:
        started.append("list")
        await asyncio.Event().wait()

    monkeypatch.setattr(skills_resources, "fetch_skill_list", stalled_list)
    monkeypatch.setattr(combined_mcp, "_SKILLS_READ_LIMIT_S", LIMIT_S)
    app, in_flight = _probe_app(max_in_flight=1)

    async with (
        running_lifespan(app),
        Client(mounted_mcp_transport(app, "/", token=TOKEN), timeout=_CLIENT_TIMEOUT_S) as client,
    ):
        began = time.monotonic()
        outcomes = await asyncio.gather(
            client.list_resources(), client.list_resources(), return_exceptions=True
        )
        answered_after = time.monotonic() - began
        after = in_flight._in_flight

    assert all(isinstance(outcome, MCPError) for outcome in outcomes)
    assert sorted(str(outcome) for outcome in outcomes) == [
        "rate_limited: This server is already running as many calls as it runs at once (1). "
        "List the resources again after 5s.",
        f"service_unavailable: listing resources did not finish within the {LIMIT_S:g}s this "
        f"server allows it, so it was stopped. List them again after {DEADLINE_RETRY_AFTER_S:g}s.",
    ]
    assert started == ["list"]
    assert LIMIT_S <= answered_after < LIMIT_S * 4
    assert after == 0


# `_guard_extension_method`, called directly.


def _guarded(
    handler: Callable[..., Awaitable[Any]], in_flight: combined_mcp._InFlightLimit
) -> Callable[..., Awaitable[Any]]:
    return combined_mcp._guard_extension_method(handler, method="skills/list", in_flight=in_flight)


async def test_the_guard_runs_the_handler_in_a_request_span_and_an_in_flight_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spans: list[str] = []
    real_span = combined_mcp.use_request_span

    @contextmanager
    def recording_span(name: str) -> Iterator[Any]:
        spans.append(name)
        with real_span(name) as span:
            yield span

    monkeypatch.setattr(combined_mcp, "use_request_span", recording_span)
    in_flight = combined_mcp._InFlightLimit(max_in_flight=2)
    seen: list[tuple[list[str], bool]] = []

    async def handler(ctx: Any, params: Any) -> dict[str, Any]:
        del ctx, params
        seen.append((list(spans), in_flight._in_flight == 1))
        return {"resultType": "complete"}

    result = await _guarded(handler, in_flight)(None, None)

    assert result == {"resultType": "complete"}
    assert seen == [(["combined mcp skills/list"], True)]
    assert in_flight._in_flight == 0


async def test_the_guard_refuses_at_once_over_the_in_flight_cap() -> None:
    in_flight = combined_mcp._InFlightLimit(max_in_flight=1)
    called: list[bool] = []

    async def handler(ctx: Any, params: Any) -> None:
        del ctx, params
        called.append(True)

    with in_flight.slot(), pytest.raises(MCPError) as refused:
        await _guarded(handler, in_flight)(None, None)

    assert called == []
    assert refused.value.code == INTERNAL_ERROR
    assert refused.value.message.startswith("rate_limited: ")


async def test_the_guard_stops_a_handler_at_its_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(combined_mcp, "_SKILLS_READ_LIMIT_S", 0.05)
    in_flight = combined_mcp._InFlightLimit(max_in_flight=1)

    async def handler(ctx: Any, params: Any) -> None:
        del ctx, params
        await asyncio.Event().wait()

    with pytest.raises(MCPError) as stopped:
        await _guarded(handler, in_flight)(None, None)

    assert (stopped.value.code, stopped.value.message) == (
        INTERNAL_ERROR,
        "service_unavailable: skills/list did not finish within the 0.05s this server allows "
        f"it, so it was stopped. Repeat it after {DEADLINE_RETRY_AFTER_S:g}s.",
    )
    assert in_flight._in_flight == 0


def test_the_skills_read_deadline_is_ten_seconds() -> None:
    assert combined_mcp._SKILLS_READ_LIMIT_S == 10.0


async def test_the_guard_passes_an_mcp_error_and_hides_any_other_exception() -> None:
    in_flight = combined_mcp._InFlightLimit(max_in_flight=1)

    async def refuses(ctx: Any, params: Any) -> None:
        del ctx, params
        raise MCPError(code=INVALID_PARAMS, message="written for the caller")

    async def explodes(ctx: Any, params: Any) -> None:
        del ctx, params
        raise TimeoutError(_SECRET_FAILURE)

    with pytest.raises(MCPError) as refused:
        await _guarded(refuses, in_flight)(None, None)
    with pytest.raises(MCPError) as raised:
        await _guarded(explodes, in_flight)(None, None)

    assert (refused.value.code, refused.value.message) == (INVALID_PARAMS, "written for the caller")
    assert raised.value.code == INTERNAL_ERROR
    assert raised.value.message.startswith("internal_error: skills/list failed")
    _assert_no_secret(raised.value.message)
