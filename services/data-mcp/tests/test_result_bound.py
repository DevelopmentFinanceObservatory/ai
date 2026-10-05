"""`ResultBoundMiddleware` refuses a tool result too large for one server-sent event."""

from __future__ import annotations

import pytest
from data_mcp.result_bound import MAX_RESULT_BYTES, ResultBoundMiddleware
from fastmcp import Client, FastMCP
from fastmcp.tools import ToolResult
from mcp.types import TextContent


@pytest.mark.parametrize(("padding", "refused"), [(-1_024, False), (1, True)])
async def test_a_result_past_the_bound_is_refused_as_over_ceiling(
    padding: int, refused: bool
) -> None:
    server = FastMCP(name="bound")
    server.add_middleware(ResultBoundMiddleware())

    @server.tool
    def big() -> ToolResult:
        return ToolResult(
            content=[TextContent(type="text", text="ok")],
            structured_content={"blob": "x" * (MAX_RESULT_BYTES + padding)},
        )

    async with Client(server) as client:
        result = await client.call_tool("big", {}, raise_on_error=False)

    assert result.is_error is refused
    assert result.structured_content is not None
    if refused:
        assert result.structured_content["code"] == "over_ceiling"
    else:
        assert "blob" in result.structured_content
