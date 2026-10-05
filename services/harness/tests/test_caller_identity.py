"""The caller reaches the MCP servers on both of the harness's client paths.

`harness.mcp_client.McpExecutor` (the direct `get_data`) and the surface
`harness.agents.build_tool_surface` builds (the executor's verbs) each hold
clients built by the harness's own construction code. Only the network
layer is replaced: fastmcp's default httpx factory is patched to route
through `httpx2.ASGITransport` into mounted probe servers, so the header
travels the same streamable-HTTP path it does in production, on the
2026-07-28 revision the harness negotiates. An in-memory client would
carry no headers at all.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any

import httpx2
import pytest
from common.caller import conversation_caller, current_caller
from fastmcp import Context, FastMCP
from fastmcp.client.transports import http as fastmcp_http
from fastmcp.tools import ToolResult
from harness.agents import build_tool_surface
from harness.conversation import current_conversation_id
from harness.mcp_client import McpExecutor
from harness.settings import Settings
from planner.workflow import ToolId
from starlette.applications import Starlette
from starlette.routing import Mount

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
