"""The caller reaches the MCP servers on both of the harness's client paths.

`harness.mcp_client.McpExecutor` (the direct `get_data`) and the surface
`harness.agents.build_tool_surface` builds (the executor's verbs) each hold
clients built by the harness's own construction code. Only the network
layer is replaced: fastmcp's default httpx factory is patched to route
through `httpx2.ASGITransport` into mounted probe servers, so the header
travels the same streamable-HTTP path it does in production, on the
2026-07-28 revision the harness negotiates. An in-memory client would
carry no headers at all.

The turn test runs the composed app's real mounts with dfo-apis stubbed
behind each, so the caller and turn travel both hops: harness to mount, and
mount to dfo-apis.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any

import compiler.surface as surface_module
import httpx2
import pytest
from analysis_mcp.runner_client import RunnerClient, runner_holder
from common.caller import (
    CALLER_HEADER,
    CALLER_PRINCIPAL_HEADER,
    SURFACE_HEADER,
    TURN_HEADER,
    conversation_caller,
    current_caller,
)
from compiler.surface import current_spec_uses
from data_mcp import client_holder, definition_registration
from data_mcp.client import UpstreamClient
from data_mcp.definition_store import (
    InMemoryDefinitionStore,
    StoredDefinition,
    get_store,
    set_store,
)
from data_mcp.definition_store.identity import canonicalize, spec_id_for
from data_mcp.settings import Settings as DataMcpSettings
from fastmcp import Context, FastMCP
from fastmcp.client.transports import http as fastmcp_http
from fastmcp.tools import ToolResult
from harness.agents import build_tool_surface
from harness.conversation import current_conversation_id, current_turn_id, shared_work_context
from harness.mcp_client import McpExecutor
from harness.server import create_app
from harness.settings import (
    ANALYSIS_MCP_MOUNT_PATH,
    DATA_MCP_MOUNT_PATH,
    PRESENTATION_MCP_MOUNT_PATH,
    Settings,
)
from planner.workflow import ToolId
from presentation_mcp import server as presentation_server
from presentation_mcp.client import WRITE_TOKEN_HEADER, PlatformAssembler
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import ASGIApp, Receive, Scope, Send

from .resolved_revision import resolved_revision

_SPEC_ID = "spec_" + "1" * 32
_MODERN = "2026-07-28"
_AGGREGATE = ToolId(server="analysis-mcp", name="aggregate")

# (tool, caller, protocol revision), one row per call the probes served.
type Seen = list[tuple[str, str | None, str]]


def _revision(ctx: Context) -> str:
    assert ctx.request_context is not None
    return ctx.request_context.protocol_version


def _probes(seen: Seen) -> tuple[FastMCP, FastMCP]:
    data = FastMCP(name="data-mcp")
    analysis = FastMCP(name="analysis-mcp")

    @data.tool(name="get_data")
    async def get_data(spec_id: str, ctx: Context) -> ToolResult:
        seen.append(("get_data", current_caller(), _revision(ctx)))
        return ToolResult(
            content="fetched",
            structured_content={"definition": {"spec_id": spec_id}, "content_version": "cv_1"},
        )

    @analysis.tool(name="aggregate")
    async def aggregate(ctx: Context) -> ToolResult:
        seen.append(("aggregate", current_caller(), _revision(ctx)))
        return ToolResult(content="aggregated", structured_content={"ok": True})

    return data, analysis


def _isolated(app: ASGIApp) -> ASGIApp:
    """`app`, each request run on a task with an empty context, as a server in
    its own process sees it: `httpx2.ASGITransport` would otherwise run the
    mounts in the client's context, where the harness's contextvars live."""

    async def isolated(scope: Scope, receive: Receive, send: Send) -> None:
        await asyncio.create_task(app(scope, receive, send), context=contextvars.Context())

    return isolated


async def _no_token() -> dict[str, str]:
    return {}


def _asgi_http_client(app: Starlette) -> Callable[..., httpx2.AsyncClient]:
    """fastmcp's default httpx factory, routed into `app` in-process."""

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx2.Timeout | None = None,
        auth: httpx2.Auth | None = None,
    ) -> httpx2.AsyncClient:
        kwargs: dict[str, Any] = {"headers": headers, "auth": auth}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), **kwargs)

    return factory


async def test_each_conversation_names_itself_on_every_call_on_both_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: Seen = []
    data, analysis = _probes(seen)
    data_app = data.http_app(path="/mcp")
    analysis_app = analysis.http_app(path="/mcp")
    app = Starlette(routes=[Mount("/data", app=data_app), Mount("/analysis", app=analysis_app)])
    monkeypatch.setattr(fastmcp_http, "create_mcp_http_client", _asgi_http_client(app))
    settings = Settings(
        data_mcp_url="http://127.0.0.1/data/mcp",
        analysis_mcp_url="http://127.0.0.1/analysis/mcp",
    )
    executor = McpExecutor(settings=settings)
    surface = build_tool_surface(settings=settings)

    async def _step() -> None:
        fetched = await executor(tool="get_data", args={"spec_id": _SPEC_ID})
        assert fetched["outcome"] == "ok"
        computed = await surface.call_tool(tool=_AGGREGATE, arguments={})
        assert computed.is_error is False

    async with AsyncExitStack() as stack:
        await stack.enter_async_context(data_app.lifespan(data_app))
        await stack.enter_async_context(analysis_app.lifespan(analysis_app))
        stack.push_async_callback(surface.aclose)
        stack.push_async_callback(executor.aclose)

        await _step()  # outside any conversation
        current_conversation_id.set("user-1/conv-a")
        await _step()
        await _step()
        current_conversation_id.set("user-1/conv-b")
        await _step()

    a = conversation_caller("user-1/conv-a")
    b = conversation_caller("user-1/conv-b")
    assert seen == [
        ("get_data", None, _MODERN),
        ("aggregate", None, _MODERN),
        ("get_data", a, _MODERN),
        ("aggregate", a, _MODERN),
        ("get_data", a, _MODERN),
        ("aggregate", a, _MODERN),
        ("get_data", b, _MODERN),
        ("aggregate", b, _MODERN),
    ]


async def test_each_call_on_both_paths_leaves_one_tool_result_and_a_fetch_names_its_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct `get_data` and a verb on the surface each add one
    `dfo.tool_result` observation naming what the call produced, and only
    the `get_data` tells the turn it fetched a spec."""
    data = FastMCP(name="data-mcp")
    analysis = FastMCP(name="analysis-mcp")

    @data.tool(name="get_data")
    async def get_data(spec_id: str) -> ToolResult:
        return ToolResult(
            content="fetched",
            structured_content={
                "definition": {"spec_id": spec_id},
                "content_version": "cv_1",
                "coverage": {"rows": 16},
                "stat_var_count": 2,
            },
        )

    @analysis.tool(name="aggregate")
    async def aggregate() -> ToolResult:
        return ToolResult(
            content="aggregated",
            structured_content={
                "spec_id": "ops_1",
                "content_version": "ocv_1",
                "frame_content_version": "cv_2",
                "rows": 9,
                "stat_var_count": 1,
            },
        )

    data_app = data.http_app(path="/mcp")
    analysis_app = analysis.http_app(path="/mcp")
    app = Starlette(routes=[Mount("/data", app=data_app), Mount("/analysis", app=analysis_app)])
    monkeypatch.setattr(fastmcp_http, "create_mcp_http_client", _asgi_http_client(app))
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        surface_module,
        "record_event",
        lambda name, *, level, metadata: events.append((name, dict(metadata))),
    )
    uses: list[tuple[str, str]] = []
    settings = Settings(
        data_mcp_url="http://127.0.0.1/data/mcp",
        analysis_mcp_url="http://127.0.0.1/analysis/mcp",
    )
    executor = McpExecutor(settings=settings)
    surface = build_tool_surface(settings=settings)

    async with AsyncExitStack() as stack:
        await stack.enter_async_context(data_app.lifespan(data_app))
        await stack.enter_async_context(analysis_app.lifespan(analysis_app))
        stack.push_async_callback(surface.aclose)
        stack.push_async_callback(executor.aclose)
        token = current_spec_uses.set(lambda use, spec_id: uses.append((use, spec_id)))
        stack.callback(current_spec_uses.reset, token)

        assert (await executor(tool="get_data", args={"spec_id": _SPEC_ID}))["outcome"] == "ok"
        assert (await surface.call_tool(tool=_AGGREGATE, arguments={})).is_error is False

    assert events == [
        (
            "dfo.tool_result",
            {
                "tool": "get_data",
                "outcome": "ok",
                "spec_id": _SPEC_ID,
                "content_version": "cv_1",
                "stat_var_count": 2,
                "rows": 16,
            },
        ),
        (
            "dfo.tool_result",
            {
                "tool": "aggregate",
                "outcome": "ok",
                "spec_id": "ops_1",
                "content_version": "ocv_1",
                "frame_content_version": "cv_2",
                "stat_var_count": 1,
                "rows": 9,
            },
        ),
    ]
    assert uses == [("fetched", _SPEC_ID)]


_TOKEN = "test-token"
# The headers that name a dfo-apis call's caller, in the order a shared call's
# row lists them.
_IDENTITY = (CALLER_PRINCIPAL_HEADER, SURFACE_HEADER, CALLER_HEADER, TURN_HEADER)
_CALLER_PRINCIPAL = "caller-principal-secret"
_SEARCHED_IDENTITY: dict[str, Any] = {"dataset": "DAC1", "fixed": {"measure": "ODA"}}
_SEARCHED_SPEC = spec_id_for(canonicalize(_SEARCHED_IDENTITY))
_FRAME = {"spec_id": _SEARCHED_SPEC}
_WRITE_TOKEN = "cmw_stub1"
# The least release document presentation-mcp's lifespan admits.
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


class _Apis:
    """dfo-apis behind all three mounts, recording each request under the turn
    the test is in. It stores one component, `cmp_1`, and revises it only with
    its write token."""

    def __init__(self) -> None:
        self.turn = ""
        # Each request as its turn, upstream, method, path and headers.
        self.seen: list[tuple[str, str, str, str, httpx2.Headers]] = []

    def transport(self, upstream: str) -> httpx2.MockTransport:
        def handle(request: httpx2.Request) -> httpx2.Response:
            path = request.url.path
            if path == "/api/presentation/release":
                return httpx2.Response(200, json=_RELEASE)
            self.seen.append((self.turn, upstream, request.method, path, request.headers))
            if path == "/api/presentation/components":
                return self._written(201, revision=1)
            if path == "/api/presentation/components/cmp_1" and request.method == "PATCH":
                if request.headers.get(WRITE_TOKEN_HEADER) != _WRITE_TOKEN:
                    return httpx2.Response(403, json={"code": "forbidden", "retryable": False})
                return self._written(200, revision=2)
            return httpx2.Response(200, json={})

        return httpx2.MockTransport(handle)

    @staticmethod
    def _written(status: int, *, revision: int) -> httpx2.Response:
        envelope = resolved_revision("line_chart", component_id="cmp_1", revision=revision)
        body = {
            "component_id": "cmp_1",
            "revision": revision,
            "fact_summary": {},
            "resolved_payload": envelope,
            **({"write_token": _WRITE_TOKEN} if revision == 1 else {}),
        }
        return httpx2.Response(status, json=body)


async def test_each_turn_reaches_dfo_apis_under_its_own_id_through_every_mount(
    monkeypatch: pytest.MonkeyPatch, running_lifespan: Callable[..., Any]
) -> None:
    """Two turns of one conversation, each calling data-mcp, analysis-mcp and
    presentation-mcp through the harness's own clients over loopback HTTP. Each
    dfo-apis call data-mcp and analysis-mcp make names the turn that caused it,
    so a client held into the next turn would fail this. presentation-mcp names
    no caller for Studio's first-party writes, so it names no turn, and the card
    turn 1 creates is revised on turn 2 through that turn's own client. The same
    calls made for shared work inside turn 2 name no caller and no turn, and
    their reads reach dfo-apis as the Studio surface's."""
    apis = _Apis()
    app = create_app(settings=Settings(mcp_api_token=_TOKEN))
    connects: list[str] = []

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx2.Timeout | None = None,
        auth: httpx2.Auth | None = None,
    ) -> httpx2.AsyncClient:
        connects.append(apis.turn)
        kwargs: dict[str, Any] = {"headers": headers, "auth": auth}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=_isolated(app)), **kwargs)

    monkeypatch.setattr(fastmcp_http, "create_mcp_http_client", factory)
    monkeypatch.setattr(
        presentation_server,
        "PlatformAssembler",
        lambda *, settings, auth=None: PlatformAssembler(
            settings=settings, auth=auth, transport=apis.transport("presentation")
        ),
    )
    store = InMemoryDefinitionStore()
    await store.put_many(
        [
            StoredDefinition(
                spec_id=_SEARCHED_SPEC,
                schema_version=1,
                canonical_bytes=canonicalize(_SEARCHED_IDENTITY),
            )
        ]
    )
    previous_store = get_store()
    set_store(store)
    definition_registration._registered.clear()
    client_holder.set_client(
        UpstreamClient(
            settings=DataMcpSettings(dfo_caller_principal=_CALLER_PRINCIPAL),
            transport=apis.transport("data"),
        )
    )
    settings = Settings(
        mcp_api_token=_TOKEN,
        data_mcp_url=f"http://127.0.0.1{DATA_MCP_MOUNT_PATH}/mcp",
        analysis_mcp_url=f"http://127.0.0.1{ANALYSIS_MCP_MOUNT_PATH}/mcp",
        presentation_mcp_url=f"http://127.0.0.1{PRESENTATION_MCP_MOUNT_PATH}/mcp",
    )
    turns = ("tr_" + "1" * 16, "tr_" + "2" * 16)
    written: list[dict[str, Any]] = []
    try:
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(running_lifespan(app))
            runner_holder.set(
                RunnerClient(
                    base_url="http://runner.test",
                    timeout_s=5.0,
                    read_timeout_s=5.0,
                    caller_principal=_CALLER_PRINCIPAL,
                    auth=_no_token,
                    transport=apis.transport("analysis"),
                )
            )
            executor = McpExecutor(settings=settings)
            surface = build_tool_surface(settings=settings)
            stack.push_async_callback(surface.aclose)
            stack.push_async_callback(executor.aclose)

            current_conversation_id.set("user-1/conv-a")
            for turn, write in zip(
                turns,
                (
                    {"intent": {"type": "line_chart", "frame_ref": _FRAME}},
                    {
                        "component_id": "cmp_1",
                        "write_token": _WRITE_TOKEN,
                        "intent": {"frame_ref": _FRAME},
                    },
                ),
                strict=True,
            ):
                apis.turn = turn
                current_turn_id.set(turn)
                await executor(tool="get_data", args={"spec_id": _SEARCHED_SPEC})
                await surface.call_tool(
                    tool=_AGGREGATE, arguments={"frame_ref": _FRAME, "aggregation": "sum"}
                )
                written.append(await executor(tool="write_component", args=write))

            async def shared_calls() -> None:
                await executor(tool="get_data", args={"spec_id": _SEARCHED_SPEC})
                # A breakdown reads apis' capabilities first, which the stub
                # answers with none, so the call ends at that read.
                await executor(
                    tool="get_data",
                    args={
                        "spec_id": _SEARCHED_SPEC,
                        "breakdown": {"code_system": "crsPurpose", "level": 1},
                    },
                )
                await surface.call_tool(
                    tool=_AGGREGATE, arguments={"frame_ref": _FRAME, "aggregation": "sum"}
                )

            apis.turn = "shared"
            await asyncio.create_task(shared_calls(), context=shared_work_context())
    finally:
        client_holder.set_client(None)
        set_store(previous_store)

    caller = conversation_caller("user-1/conv-a")
    by_turn = {
        turn: {
            upstream: [
                (headers.get(CALLER_HEADER), headers.get(TURN_HEADER)) for *_, headers in rows
            ]
            for upstream in ("data", "analysis", "presentation")
            if (rows := [row for row in apis.seen if row[0] == turn and row[1] == upstream])
        }
        for turn in turns
    }
    for turn in turns:
        assert set(by_turn[turn]) == {"data", "analysis", "presentation"}, turn
        assert set(by_turn[turn]["data"]) == {(caller, turn)}
        assert set(by_turn[turn]["analysis"]) == {(caller, turn)}
        assert set(by_turn[turn]["presentation"]) == {(None, None)}
    shared = {
        (upstream, method, *(headers.get(name) for name in _IDENTITY))
        for turn, upstream, method, _, headers in apis.seen
        if turn == "shared"
    }
    # The reads name the principal and Studio, and a write names nothing, since
    # dfo-apis refuses a vouched POST that names no caller on its pin and export
    # doors.
    assert shared == {
        ("data", "GET", _CALLER_PRINCIPAL, "studio", None, None),
        ("data", "POST", None, None, None, None),
        ("analysis", "POST", None, None, None, None),
    }
    # One canvas: turn 1 creates the card and turn 2 revises it with turn 1's
    # write token, which dfo-apis accepts.
    writes = [
        (turn, method, path)
        for turn, upstream, method, path, _ in apis.seen
        if upstream == "presentation" and path.startswith("/api/presentation/components")
    ]
    assert writes == [
        (turns[0], "POST", "/api/presentation/components"),
        (turns[1], "PATCH", "/api/presentation/components/cmp_1"),
    ]
    # presentation-mcp answered both writes; the harness withholds each card for
    # the source lineage this test links none of. A refused revise would come
    # back as presentation-mcp's own refusal.
    assert [outcome["refusal"]["code"] for outcome in written] == ["provenance_unavailable"] * 2
    # A turn connects once to each mount it calls: data-mcp's `get_data`,
    # analysis-mcp's verbs and presentation-mcp's tools each hold a client per turn.
    assert [connects.count(turn) for turn in turns] == [3, 3]
