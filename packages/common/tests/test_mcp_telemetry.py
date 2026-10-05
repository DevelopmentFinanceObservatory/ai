"""`LangfuseSessionMiddleware` reads the session from the request context fastmcp
binds and enters `use_call_session` only for a request marked as loopback.
`LangfuseToolIoMiddleware` sets a loopback call's input and output on the
current span. Their behaviour through a served app is tested where the servers
are composed."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import mcp.types as mt
import pytest
from common import mcp_telemetry
from common.mcp_telemetry import (
    IO_ATTRIBUTE_LIMIT,
    LOOPBACK_SCOPE_KEY,
    OBSERVATION_INPUT,
    OBSERVATION_OUTPUT,
    TRUNCATION_MARKER,
    LangfuseSessionMiddleware,
    LangfuseToolIoMiddleware,
    is_loopback_url,
)
from common.telemetry import SESSION_META_KEY
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import fastmcp_request_ctx
from fastmcp.tools import ToolResult
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from starlette.requests import Request


@pytest.fixture
def entered(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """The session id of every `use_call_session` the middleware enters."""
    sessions: list[str | None] = []

    @contextmanager
    def record(session_id: str | None) -> Iterator[None]:
        sessions.append(session_id)
        yield

    monkeypatch.setattr(mcp_telemetry, "use_call_session", record)
    return sessions


@contextmanager
def _request(*, scope: dict[str, Any] | None, meta: Any) -> Iterator[None]:
    """Bind a fastmcp request context for an HTTP request with `scope` (no HTTP
    request when `None`) and the wire `_meta` `meta`."""
    request = Request({"type": "http", "headers": [], **scope}) if scope is not None else None
    token = fastmcp_request_ctx.set(SimpleNamespace(request=request, meta=meta))
    try:
        yield
    finally:
        fastmcp_request_ctx.reset(token)


async def _call(middleware: LangfuseSessionMiddleware) -> list[str]:
    ran: list[str] = []

    async def call_next(_context: Any) -> str:
        ran.append("tool")
        return "result"

    assert await middleware.on_call_tool(SimpleNamespace(), call_next) == "result"  # ty: ignore[invalid-argument-type]
    return ran


async def test_a_loopback_request_with_a_session_runs_the_call_inside_that_session(
    entered: list[str | None],
) -> None:
    with _request(scope={LOOPBACK_SCOPE_KEY: True}, meta={SESSION_META_KEY: "conv-1"}):
        assert await _call(LangfuseSessionMiddleware()) == ["tool"]

    assert entered == ["conv-1"]


@pytest.mark.parametrize(
    ("scope", "meta"),
    [
        pytest.param(None, {SESSION_META_KEY: "conv-1"}, id="no-http-request"),
        pytest.param({}, {SESSION_META_KEY: "conv-1"}, id="no-loopback-mark"),
        pytest.param({LOOPBACK_SCOPE_KEY: "yes"}, {SESSION_META_KEY: "conv-1"}, id="mark-not-true"),
        pytest.param({LOOPBACK_SCOPE_KEY: True}, None, id="no-meta"),
        pytest.param({LOOPBACK_SCOPE_KEY: True}, {}, id="no-key"),
        pytest.param({LOOPBACK_SCOPE_KEY: True}, {SESSION_META_KEY: 7}, id="key-not-a-string"),
    ],
)
async def test_a_call_that_names_no_trusted_session_runs_untouched(
    entered: list[str | None], scope: dict[str, Any] | None, meta: Any
) -> None:
    with _request(scope=scope, meta=meta):
        assert await _call(LangfuseSessionMiddleware()) == ["tool"]

    assert entered == [None]


async def test_a_call_with_no_request_context_runs_untouched(entered: list[str | None]) -> None:
    assert await _call(LangfuseSessionMiddleware()) == ["tool"]

    assert entered == [None]


async def test_an_unreadable_request_context_runs_the_call_untouched(
    entered: list[str | None],
) -> None:
    marked = Request({"type": "http", LOOPBACK_SCOPE_KEY: True})
    token = fastmcp_request_ctx.set(SimpleNamespace(request=marked, meta=3))
    try:
        assert await _call(LangfuseSessionMiddleware()) == ["tool"]
    finally:
        fastmcp_request_ctx.reset(token)

    assert entered == [None]


@pytest.mark.parametrize(
    ("url", "loopback"),
    [
        ("http://127.0.0.1:8080/mcp/data", True),
        ("http://localhost/mcp", True),
        ("http://[::1]:8080/mcp", True),
        ("https://dfo-data-abc.a.run.app/mcp", False),
        ("http://127.0.0.1.example.com/mcp", False),
        ("not a url", False),
    ],
)
def test_is_loopback_url_reads_the_host(url: str, loopback: bool) -> None:
    assert is_loopback_url(url) is loopback


LOOPBACK = {LOOPBACK_SCOPE_KEY: True}


async def _call_io(
    *, scope: dict[str, Any] | None, arguments: Any, outcome: ToolResult | Exception
) -> dict[str, Any]:
    """Run `LangfuseToolIoMiddleware` around a call with `arguments` that returns
    or raises `outcome`, inside a recording span, and return the span's attributes."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    async def call_next(_context: Any) -> ToolResult:
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    context = SimpleNamespace(message=SimpleNamespace(arguments=arguments))
    with (
        _request(scope=scope, meta=None),
        provider.get_tracer("test").start_as_current_span("tools/call"),
    ):
        try:
            result = await LangfuseToolIoMiddleware().on_call_tool(context, call_next)  # ty: ignore[invalid-argument-type]
        except Exception as exc:
            assert exc is outcome
        else:
            assert result is outcome
    (span,) = exporter.get_finished_spans()
    return dict(span.attributes or {})


@pytest.mark.parametrize(
    ("outcome", "output"),
    [
        pytest.param(
            ToolResult(structured_content={"rows": [1, 2]}), {"rows": [1, 2]}, id="structured"
        ),
        pytest.param(
            ToolResult(structured_content={"code": "binding_invalid"}, is_error=True),
            {"code": "binding_invalid"},
            id="refusal-result",
        ),
        pytest.param(
            ToolError("no such verb"),
            {"error": "ToolError", "message": "no such verb"},
            id="raised",
        ),
    ],
)
async def test_a_loopback_call_records_its_arguments_and_its_result_or_refusal(
    outcome: ToolResult | Exception, output: dict[str, Any]
) -> None:
    attributes = await _call_io(scope=LOOPBACK, arguments={"ids": ["dc/1"]}, outcome=outcome)

    assert json.loads(attributes[OBSERVATION_INPUT]) == {"ids": ["dc/1"]}
    assert json.loads(attributes[OBSERVATION_OUTPUT]) == output


async def test_a_result_with_no_structured_content_records_its_text() -> None:
    result = ToolResult(
        content=[mt.TextContent(type="text", text="a"), mt.TextContent(type="text", text="b")]
    )

    attributes = await _call_io(scope=LOOPBACK, arguments=None, outcome=result)

    assert attributes[OBSERVATION_INPUT] == "{}"
    assert attributes[OBSERVATION_OUTPUT] == "a\nb"


@pytest.mark.parametrize(
    "scope",
    [
        pytest.param(None, id="no-http-request"),
        pytest.param({}, id="no-loopback-mark"),
        pytest.param({LOOPBACK_SCOPE_KEY: "yes"}, id="mark-not-true"),
    ],
)
async def test_a_call_off_loopback_records_nothing(scope: dict[str, Any] | None) -> None:
    attributes = await _call_io(
        scope=scope, arguments={"ids": ["dc/1"]}, outcome=ToolResult(structured_content={"a": 1})
    )

    assert OBSERVATION_INPUT not in attributes
    assert OBSERVATION_OUTPUT not in attributes


async def test_a_long_value_is_redacted_then_cut_to_the_limit_with_a_marker() -> None:
    signed = "https://storage.googleapis.com/b/o.parquet?X-Goog-Signature=" + "f" * 64
    result = ToolResult(structured_content={"url": signed, "pad": "x" * (2 * IO_ATTRIBUTE_LIMIT)})

    attributes = await _call_io(scope=LOOPBACK, arguments={"url": signed}, outcome=result)

    assert "X-Goog-Signature" not in attributes[OBSERVATION_INPUT]
    assert "storage.googleapis.com/b/o.parquet" in attributes[OBSERVATION_INPUT]
    output = attributes[OBSERVATION_OUTPUT]
    assert len(output) == IO_ATTRIBUTE_LIMIT
    assert output.endswith(TRUNCATION_MARKER)
    assert "X-Goog-Signature" not in output


async def test_arguments_that_cannot_be_serialised_leave_the_call_running() -> None:
    circular: dict[str, Any] = {}
    circular["self"] = circular

    attributes = await _call_io(
        scope=LOOPBACK, arguments=circular, outcome=ToolResult(structured_content={"a": 1})
    )

    assert OBSERVATION_INPUT not in attributes
    assert json.loads(attributes[OBSERVATION_OUTPUT]) == {"a": 1}


TOKEN = "cmw_Abc-123_xyz"
# presentation-mcp's create text block: the token is its last line.
TOKEN_TEXT = (
    "Keep this write_token with the component_id. Revising the component needs both.\n"
    f"write_token: {TOKEN}"
)


@pytest.mark.parametrize(
    ("arguments", "result", "input_", "output"),
    [
        pytest.param(
            {"intent": {"type": "bar_chart"}},
            ToolResult(
                content=[mt.TextContent(type="text", text="{}")],
                structured_content={"component_id": "cmp_1", "write_token": TOKEN},
            ),
            {"intent": {"type": "bar_chart"}},
            {"component_id": "cmp_1"},
            id="create-structured",
        ),
        pytest.param(
            {"intent": {"type": "bar_chart"}},
            ToolResult(
                content=[
                    mt.TextContent(type="text", text='{"component_id":"cmp_1"}'),
                    mt.TextContent(type="text", text=TOKEN_TEXT),
                ]
            ),
            {"intent": {"type": "bar_chart"}},
            '{"component_id":"cmp_1"}\n'
            "Keep this write_token with the component_id. Revising the component needs both.",
            id="create-text",
        ),
        pytest.param(
            {"intent": {"title": "t"}, "component_id": "cmp_1", "write_token": TOKEN},
            ToolResult(
                structured_content={"component_id": "cmp_1", "nested": [{"write_token": TOKEN}]}
            ),
            {"intent": {"title": "t"}, "component_id": "cmp_1"},
            {"component_id": "cmp_1", "nested": [{}]},
            id="edit-arguments",
        ),
    ],
)
async def test_a_write_token_is_exported_in_no_form(
    arguments: dict[str, Any], result: ToolResult, input_: Any, output: Any
) -> None:
    attributes = await _call_io(scope=LOOPBACK, arguments=arguments, outcome=result)

    assert TOKEN not in attributes[OBSERVATION_INPUT]
    assert TOKEN not in attributes[OBSERVATION_OUTPUT]
    assert json.loads(attributes[OBSERVATION_INPUT]) == input_
    recorded = attributes[OBSERVATION_OUTPUT]
    assert (recorded if isinstance(output, str) else json.loads(recorded)) == output
