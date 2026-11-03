"""A data-mcp that cannot be reached, as `McpSurface` meets it."""

from compiler.surface import McpSurface

# Nothing listens on the discard port, so the connection is refused at once.
UNREACHABLE_URL = "http://127.0.0.1:9/mcp"


def unreachable_surface() -> McpSurface:
    """A real `McpSurface` over a server that is down. Its `list_tools` and `call_tool`
    raise the `ToolSurfaceUnavailableError` a restarting data-mcp gives: fastmcp's
    `RuntimeError('Client failed to connect: ...')` raised from the transport's
    `ConnectError`, marked retryable because a later attempt may find the server up."""
    return McpSurface(sources={"data-mcp": UNREACHABLE_URL})
