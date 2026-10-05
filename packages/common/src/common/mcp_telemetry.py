"""FastMCP middleware that puts a harness turn's Langfuse session on an MCP
server's spans.

The harness calls its own MCP servers over loopback HTTP, and each call's spans
start in the server's own context, so they carry none of the turn's attributes.
`common.telemetry.turn_call_meta` sends the session in `_meta`. The harness
marks every request that arrived over loopback with `LOOPBACK_SCOPE_KEY`, and
`LangfuseSessionMiddleware` honours the session in `_meta` only on a marked
request. A call through the public combined `/mcp/` server never has the mark,
so an outside client cannot stamp a session on spans. The client sends the
session only to a URL `is_loopback_url` accepts.

`LangfuseToolIoMiddleware` puts a loopback call's arguments and result on its
SERVER span as the Langfuse observation's input and output, on the same trust
boundary: a public caller's arguments are not exported. The exported values
carry no component write token: `WRITE_TOKEN_KEY` is dropped at any depth of
the arguments and structured content, and the line naming the token is cut
from text.

This module imports fastmcp. `import common` and `common.telemetry` do not.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import mcp.types as mt
from fastmcp.server.dependencies import fastmcp_request_ctx, get_http_request
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult
from opentelemetry import trace

from common.telemetry import SESSION_META_KEY, use_call_session
from common.uri_redaction import redact_uris

logger = logging.getLogger(__name__)

# The ASGI scope key the harness's loopback guard sets to `True` on a request
# whose peer is the loopback interface.
LOOPBACK_SCOPE_KEY = "dfo.loopback_peer"

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# The span attributes Langfuse reads as an observation's input and output, each
# a JSON string: https://langfuse.com/integrations/native/opentelemetry
OBSERVATION_INPUT = "langfuse.observation.input"
OBSERVATION_OUTPUT = "langfuse.observation.output"

# The most characters `LangfuseToolIoMiddleware` puts in one attribute, the
# truncation marker included.
IO_ATTRIBUTE_LIMIT = 16 * 1024
TRUNCATION_MARKER = "...[truncated]"

# The field presentation-mcp returns a component's write token under, top level
# in a create's structured content, and the argument a revise sends it back as.
# The token is a bearer secret granting revises of that component.
WRITE_TOKEN_KEY = "write_token"

# The line presentation-mcp ends a create's token text block with.
_WRITE_TOKEN_LINE = re.compile(rf"\n?^{WRITE_TOKEN_KEY}: \S+[ \t]*$", re.MULTILINE)


def is_loopback_url(url: str) -> bool:
    """Whether `url`'s host is a loopback host: `127.0.0.1`, `localhost` or `::1`."""
    return urlsplit(url).hostname in LOOPBACK_HOSTS


def _is_loopback_request() -> bool:
    """Whether the current HTTP request carries the `LOOPBACK_SCOPE_KEY` mark the
    harness sets. Raises `RuntimeError` when no HTTP request is in flight."""
    return get_http_request().scope.get(LOOPBACK_SCOPE_KEY) is True


def _session_from_loopback_request() -> str | None:
    """The session the current request's `_meta` carries, or `None` when the
    request has no HTTP request behind it, did not arrive over loopback, or
    carries no string under `SESSION_META_KEY`. Never raises."""
    try:
        if not _is_loopback_request():
            return None
        request_context = fastmcp_request_ctx.get()
        meta = request_context.meta if request_context is not None else None
        session = meta.get(SESSION_META_KEY) if meta else None
    except RuntimeError:  # no HTTP request in flight
        return None
    except Exception:
        logger.debug("failed to read the call session -- continuing", exc_info=True)
        return None
    return session if isinstance(session, str) else None


class LangfuseSessionMiddleware(Middleware):
    """Runs each tool call inside `use_call_session` for the session its
    loopback client sent in `_meta`, so the call's SERVER span and everything
    it opens carry `session.id`.

    The session is read from `fastmcp_request_ctx`, whose `meta` is the wire
    `_meta`. `context.message.meta` is rebuilt by fastmcp without the wire keys.
    Add it before any middleware that can refuse a call, so a refused call's
    span carries the session too. A call with no session proceeds untouched.
    """

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        with use_call_session(_session_from_loopback_request()):
            return await call_next(context)


def _bounded(text: str) -> str:
    """`text` with the credentials in its URIs cut out, then cut to
    `IO_ATTRIBUTE_LIMIT` characters ending in `TRUNCATION_MARKER` when longer.

    Redacting first spends the limit on what the exporter keeps, and keeps the
    marker from being glued to a URL query the exporter would drop with it."""
    text = redact_uris(text)
    if len(text) <= IO_ATTRIBUTE_LIMIT:
        return text
    return text[: IO_ATTRIBUTE_LIMIT - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER


def _without_write_token(value: Any) -> Any:
    """`value` with every `WRITE_TOKEN_KEY` entry dropped, at any depth of its
    dicts and lists."""
    if isinstance(value, dict):
        return {
            key: _without_write_token(item) for key, item in value.items() if key != WRITE_TOKEN_KEY
        }
    if isinstance(value, list):
        return [_without_write_token(item) for item in value]
    return value


def _arguments_text(arguments: dict[str, Any] | None) -> str:
    """The call's arguments as JSON without the write token, `{}` when it sent none."""
    return json.dumps(_without_write_token(arguments or {}), ensure_ascii=False, default=str)


def _result_text(result: ToolResult) -> str:
    """The result's structured content as JSON, else its text blocks joined by
    newlines, either without the write token."""
    if result.structured_content is not None:
        structured = _without_write_token(result.structured_content)
        return json.dumps(structured, ensure_ascii=False, default=str)
    return "\n".join(
        _WRITE_TOKEN_LINE.sub("", block.text)
        for block in result.content
        if isinstance(block, mt.TextContent)
    )


def _exception_text(exc: Exception) -> str:
    """A raised call's output: the exception's type and message as JSON."""
    return json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False)


def _set_bounded[T](span: trace.Span, key: str, render: Callable[[T], str], value: T) -> None:
    """Set `key` on `span` to `render(value)`, bounded. A failure logs at debug."""
    try:
        span.set_attribute(key, _bounded(render(value)))
    except Exception:
        logger.debug("failed to record %s -- continuing", key, exc_info=True)


def _recording_loopback_span() -> trace.Span | None:
    """The current span when it records and the request arrived over loopback,
    else `None`. Never raises."""
    span = trace.get_current_span()
    if not span.is_recording():
        return None
    try:
        return span if _is_loopback_request() else None
    except RuntimeError:  # no HTTP request in flight
        return None
    except Exception:
        logger.debug("failed to read the loopback mark -- continuing", exc_info=True)
        return None


class LangfuseToolIoMiddleware(Middleware):
    """Sets `OBSERVATION_INPUT` (the call's arguments as JSON) and
    `OBSERVATION_OUTPUT` (the result's structured content as JSON, else its
    text) on a loopback tool call's SERVER span, which fastmcp has made current
    by the time the middleware chain runs. Neither carries the write token.

    A refusal result records its refusal, and a call that raises records the
    exception's type and message before the exception propagates. Each value
    goes through `_bounded`. A call off loopback, or with no recording span,
    proceeds untouched. Add it before any middleware that can refuse a call, so
    a refused call records its input.
    """

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        span = _recording_loopback_span()
        if span is None:
            return await call_next(context)
        _set_bounded(span, OBSERVATION_INPUT, _arguments_text, context.message.arguments)
        try:
            result = await call_next(context)
        except Exception as exc:
            _set_bounded(span, OBSERVATION_OUTPUT, _exception_text, exc)
            raise
        _set_bounded(span, OBSERVATION_OUTPUT, _result_text, result)
        return result
