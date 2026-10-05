"""The Langfuse session of a harness turn, and each call's input and output, on
the spans its MCP calls open.

The real Langfuse SDK, tracer provider, span filter, composed harness app and
the harness's own MCP clients (`McpExecutor`, `McpSurface`) run; only the
Langfuse server is fake (`fake_langfuse`). The clients reach the mounts over
`httpx2.ASGITransport`, which gives the mounts a loopback peer as a real
request from `127.0.0.1` would.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import urllib.parse
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

import httpx2
import pytest
from common.caller import CALLER_HEADER
from common.mcp_telemetry import OBSERVATION_INPUT, OBSERVATION_OUTPUT
from common.telemetry import (
    SESSION_META_KEY,
    end_turn_span,
    flush_telemetry,
    start_turn_span,
    use_turn_span,
)
from compiler.surface import McpSurface
from fastapi import FastAPI
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from harness import mcp_deadlines
from harness.mcp_client import McpExecutor
from harness.server import create_app
from harness.settings import (
    ANALYSIS_MCP_MOUNT_PATH,
    COMBINED_MCP_MOUNT_PATH,
    DATA_MCP_MOUNT_PATH,
    PRESENTATION_MCP_MOUNT_PATH,
    Settings,
)
from mcp.shared._httpx_utils import McpHttpClientFactory
from opentelemetry.proto.trace.v1.trace_pb2 import Span
from planner.workflow import ToolId
from starlette.types import ASGIApp, Receive, Scope, Send

from .conftest import FakeLangfuse

TOKEN = "test-token"
CONVERSATION = "conv-session-1"
TURN = "tr_mcp_session_01"
_SERVER_KIND = Span.SpanKind.SPAN_KIND_SERVER


def _attribute(span: Span, key: str) -> str | int | float | bool | None:
    for attribute in span.attributes:
        if attribute.key == key:
            return getattr(attribute.value, attribute.value.WhichOneof("value"))
    return None


def _in_a_fresh_context(app: ASGIApp) -> ASGIApp:
    """`app`, each request run on a task with an empty context.

    `httpx2.ASGITransport` runs the app in the client's own context, so the
    server would see the turn's OpenTelemetry context and Langfuse attributes
    that a server in its own process never does. An empty context leaves
    `_meta` and the trace headers as the only way the turn reaches the server."""

    async def isolated(scope: Scope, receive: Receive, send: Send) -> None:
        await asyncio.create_task(app(scope, receive, send), context=contextvars.Context())

    return isolated


def _asgi_transport(
    app: FastAPI, path: str, *, caller: str | None = None, host: str = "127.0.0.1"
) -> StreamableHttpTransport:
    """A streamable-HTTP transport to `path` on `app`, in process, whose URL is
    on `host` (loopback by default, as the harness's own mounts are), carrying
    the bearer and `caller`. The app sees a request from a loopback peer
    whatever `host` is."""

    def factory(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=_in_a_fresh_context(app)),
            base_url="http://testserver",
            **kwargs,
        )

    return StreamableHttpTransport(
        f"http://{host}{path}",
        headers={CALLER_HEADER: caller} if caller is not None else {},
        auth=TOKEN,
        httpx_client_factory=cast(McpHttpClientFactory, factory),
    )


def _tool_call_spans(fake: FakeLangfuse, tool: str) -> list[Span]:
    """The SERVER spans exported for `tools/call` of `tool`."""
    return [
        span
        for span in fake.spans()
        if span.kind == _SERVER_KIND and _attribute(span, "gen_ai.tool.name") == tool
    ]


@pytest.fixture
async def composed_app(
    fake_langfuse: FakeLangfuse,
    running_lifespan: Callable[[FastAPI], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[FastAPI]:
    """The composed harness app with its lifespan running and the harness's
    own clients' transports pointed at it in process."""
    app = create_app(settings=Settings(mcp_api_token=TOKEN))

    def transport(url: str, *, caller: str | None, auth: str | None) -> StreamableHttpTransport:
        parts = urllib.parse.urlparse(url)
        return _asgi_transport(app, parts.path, caller=caller, host=parts.hostname or "")

    monkeypatch.setattr(mcp_deadlines, "caller_transport", transport)
    async with running_lifespan(app):
        yield app


# A dcid `describe` refuses on its grammar, before it reads any backend.
MALFORMED_DCID = "dcid:not a dcid"

# One call per harness client path: `McpExecutor` reaches presentation-mcp
# (`describe_frame`) and data-mcp (`describe`), `McpSurface` reaches data-mcp
# (`sources`) and analysis-mcp (`filter`). Each is answered without a backend,
# by a refusal or a catalogue read.
CALLED_TOOLS = ("describe_frame", "describe", "sources", "filter")
ABSENT_FRAME = {"spec_id": "absent", "content_version": "absent"}


async def _call_through_the_harness_clients(app: FastAPI) -> None:
    surface = McpSurface(
        sources={
            "data-mcp": _asgi_transport(app, f"{DATA_MCP_MOUNT_PATH}/mcp"),
            "analysis-mcp": _asgi_transport(app, f"{ANALYSIS_MCP_MOUNT_PATH}/mcp"),
        }
    )
    executor = McpExecutor(
        settings=Settings(
            mcp_api_token=TOKEN,
            data_mcp_url=f"http://127.0.0.1{DATA_MCP_MOUNT_PATH}/mcp",
            presentation_mcp_url=f"http://127.0.0.1{PRESENTATION_MCP_MOUNT_PATH}/mcp",
        )
    )
    frame_ref = ABSENT_FRAME
    try:
        await executor(tool="describe_frame", args={"frame_ref": frame_ref})
        await executor._describe_into_cache([MALFORMED_DCID])
        await surface.call_tool(tool=ToolId(server="data-mcp", name="sources"), arguments={})
        await surface.call_tool(
            tool=ToolId(server="analysis-mcp", name="filter"), arguments={"frame_ref": frame_ref}
        )
    finally:
        await executor.aclose()
        await surface.aclose()


async def test_a_turns_mcp_server_spans_carry_the_session_and_trace_name_its_turn_span_does(
    composed_app: FastAPI, fake_langfuse: FakeLangfuse
) -> None:
    handle = start_turn_span(TURN, session_id=CONVERSATION)
    with use_turn_span(handle):
        await _call_through_the_harness_clients(composed_app)
    end_turn_span(handle)
    await flush_telemetry()

    (turn,) = [span for span in fake_langfuse.spans() if span.name == "turn"]
    turn_session = _attribute(turn, "session.id")
    assert turn_session is not None
    assert _attribute(turn, "langfuse.trace.name") == "turn"
    for tool in CALLED_TOOLS:
        # Control: the call's server span exists, once, in the turn's trace.
        (span,) = _tool_call_spans(fake_langfuse, tool)
        assert span.trace_id == turn.trace_id, tool
        span_session = _attribute(span, "session.id")
        assert span_session == turn_session, tool
        assert _attribute(span, "langfuse.trace.name") == _attribute(turn, "langfuse.trace.name")


async def test_an_mcp_call_outside_a_turn_has_no_session_on_its_server_span(
    composed_app: FastAPI, fake_langfuse: FakeLangfuse
) -> None:
    await _call_through_the_harness_clients(composed_app)
    await flush_telemetry()

    for tool in CALLED_TOOLS:
        (span,) = _tool_call_spans(fake_langfuse, tool)
        span_session = _attribute(span, "session.id")
        assert span_session is None, tool


async def test_a_harness_mcp_call_records_its_arguments_and_result_on_its_server_span(
    composed_app: FastAPI, fake_langfuse: FakeLangfuse
) -> None:
    await _call_through_the_harness_clients(composed_app)
    await flush_telemetry()

    outputs: dict[str, str] = {}
    for tool in CALLED_TOOLS:
        (span,) = _tool_call_spans(fake_langfuse, tool)
        output = _attribute(span, OBSERVATION_OUTPUT)
        assert isinstance(output, str), tool
        assert output, tool
        outputs[tool] = output
    (filter_span,) = _tool_call_spans(fake_langfuse, "filter")
    filter_input = _attribute(filter_span, OBSERVATION_INPUT)
    assert isinstance(filter_input, str)
    assert json.loads(filter_input) == {"frame_ref": ABSENT_FRAME}
    (describe_span,) = _tool_call_spans(fake_langfuse, "describe")
    describe_input = _attribute(describe_span, OBSERVATION_INPUT)
    assert isinstance(describe_input, str)
    assert MALFORMED_DCID.removeprefix("dcid:") in describe_input
    # `describe` refuses the malformed dcid, and the span's output is that refusal.
    assert json.loads(outputs["describe"])["code"] == "request_invalid"


async def test_a_call_to_a_server_off_loopback_sends_no_session(
    composed_app: FastAPI, fake_langfuse: FakeLangfuse
) -> None:
    """The mounts stamp a loopback peer's session, and `ASGITransport` makes
    every request a loopback peer's, so a session on the wire would show on the
    server span. A client whose URL is a remote host sends none."""
    surface = McpSurface(
        sources={
            "data-mcp": _asgi_transport(
                composed_app, f"{DATA_MCP_MOUNT_PATH}/mcp", host="dfo-data.run.app"
            )
        }
    )
    handle = start_turn_span(TURN, session_id=CONVERSATION)
    try:
        with use_turn_span(handle):
            await surface.call_tool(tool=ToolId(server="data-mcp", name="sources"), arguments={})
    finally:
        await surface.aclose()
    end_turn_span(handle)
    await flush_telemetry()

    # Control: the call reached the server and its span is in the turn's trace.
    (turn,) = [span for span in fake_langfuse.spans() if span.name == "turn"]
    (span,) = _tool_call_spans(fake_langfuse, "sources")
    assert span.trace_id == turn.trace_id
    assert _attribute(span, "session.id") is None


async def test_a_call_through_the_public_combined_mount_has_no_session_input_or_output(
    composed_app: FastAPI, fake_langfuse: FakeLangfuse
) -> None:
    """The combined `/mcp/` mount never passes `_loopback_only`, so even a
    client at loopback cannot set a session in `_meta`, and its arguments and
    result stay off the span."""
    async with Client(_asgi_transport(composed_app, f"{COMBINED_MCP_MOUNT_PATH}/")) as client:
        await client.call_tool(
            "describe",
            {"ids": ["not a dcid"]},
            meta={SESSION_META_KEY: "forged"},
            raise_on_error=False,
        )
    await flush_telemetry()

    spans = _tool_call_spans(fake_langfuse, "describe")
    assert spans
    for span in spans:
        span_session = _attribute(span, "session.id")
        assert span_session is None
        assert _attribute(span, OBSERVATION_INPUT) is None
        assert _attribute(span, OBSERVATION_OUTPUT) is None
