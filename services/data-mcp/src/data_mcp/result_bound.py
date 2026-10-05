"""The bound on one tool result's size, and the middleware that refuses a result past it.

An MCP client reads a streamable-HTTP result as one server-sent event. The MCP
SDK's client parses it with httpx2's `EventSource`, which drops an event over
`DEFAULT_MAX_EVENT_SIZE_BYTES` (1,048,576 bytes) and logs that only at debug
level, so the caller sees the stream end with no response and this server logs
nothing. `ResultBoundMiddleware` measures each result as JSON and replaces one
past `MAX_RESULT_BYTES` with a logged `over_ceiling` refusal that does reach
the caller.
"""

from __future__ import annotations

import logging

import mcp.types as mt
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult

from data_mcp.tool_errors import ToolRefusal, to_tool_result

logger = logging.getLogger(__name__)

# httpx2's `DEFAULT_MAX_EVENT_SIZE_BYTES`, less 16,384 bytes for the JSON-RPC
# envelope and the event's own `event:`/`id:`/`data:` lines.
MAX_RESULT_BYTES = 1_048_576 - 16_384


def result_bytes(result: ToolResult) -> int:
    """The size of `result` as the JSON its `CallToolResult` serialises to."""
    message = mt.CallToolResult(
        content=result.content,
        structuredContent=result.structured_content,
        isError=bool(result.is_error),
        _meta=result.meta,  # the alias of `meta`
    )
    return len(message.model_dump_json(by_alias=True, exclude_none=True).encode())


class ResultBoundMiddleware(Middleware):
    """Refuses a tool result over `MAX_RESULT_BYTES` as `over_ceiling`, logging its size."""

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        result = await call_next(context)
        size = result_bytes(result)
        if size <= MAX_RESULT_BYTES:
            return result
        logger.warning(
            "%s returned %d bytes of JSON, over the %d-byte bound on one result; refusing it",
            context.message.name,
            size,
            MAX_RESULT_BYTES,
        )
        return to_tool_result(
            ToolRefusal.over_ceiling(
                detail=f"the {context.message.name} result is {size} bytes of JSON, over "
                f"the {MAX_RESULT_BYTES}-byte bound on one result",
                ceiling_kind=None,
            )
        )
