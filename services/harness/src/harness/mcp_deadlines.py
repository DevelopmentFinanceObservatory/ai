"""Centralised MCP client timeout policy.

`McpTimeoutPolicy` is the one place that turns `Settings`'s three MCP
timeout fields, plus `compiler.capabilities.call_timeouts()`'s per-tool
overrides, into the deadlines every FastMCP `Client` this service
constructs actually carries. Two call sites read it: `harness.agents.
build_tool_surface` hands its pieces to `compiler.surface.McpSurface` (the
translator/compiler's discovery surface), and `harness.mcp_client`'s
`McpResolver` / `McpExecutor` (the served request path) build `Client`s
through `.client()` directly. Both carry the same call and init deadlines,
so a stalled data-mcp, analysis-mcp, or presentation-mcp call cannot hold a
served request open indefinitely (`fastmcp.Client`'s own `timeout` and
`init_timeout` both default to `None`, unbounded).

`compiler.capabilities.call_timeouts()` only covers `data-mcp.search`,
`data-mcp.describe`, and `data-mcp.get_data` -- the three tools the
compiler's capability table knows about. Every analysis-mcp verb and all
three presentation-mcp tools (`describe_frame`, `write_component`,
`describe_component`) get `settings.mcp_call_timeout_s` (60s by default):
none declares a budget of its own the way `get_data`'s mint does (180s,
against its own 125s budget), and presentation-mcp's calls are synchronous
CRUD against the ONE platform -- the same shape as an analysis verb -- so a
bespoke deadline for it would be inventing a number rather than reading one
off a real constraint.
"""

from __future__ import annotations

from compiler.caller_clients import caller_transport
from compiler.capabilities import call_timeouts
from fastmcp import Client
from planner.workflow import ToolId

from harness.settings import Settings


class McpTimeoutPolicy:
    """Deadlines for every MCP `Client` the harness constructs."""

    def __init__(self, *, settings: Settings) -> None:
        self.call_timeouts: dict[ToolId, float] = call_timeouts()
        self.default_call_timeout_s = settings.mcp_call_timeout_s
        self.list_timeout_s = settings.mcp_list_timeout_s
        self.init_timeout_s = settings.mcp_init_timeout_s
        # `call_timeouts()` keys by `ToolId` (server + name); every call site
        # below already resolved one URL to one server, so it names a tool by
        # its bare name alone. Two servers serving the same tool name would
        # collide here, but `McpSurface.list_tools` already refuses that
        # (`DuplicateToolNameError`) before a surface built off this same
        # table would ever compile a plan against it.
        self._by_name = {tool.name: seconds for tool, seconds in self.call_timeouts.items()}

    def call_timeout_s(self, tool: str) -> float:
        """The deadline `tool` gets: its pinned row, or the general default."""
        return self._by_name.get(tool, self.default_call_timeout_s)

    def client(
        self, url: str, *, auth: str | None, caller: str | None, tool: str | None = None
    ) -> Client:
        """A `Client` for `caller`, bound by this policy's call and init deadlines.

        `caller` goes out as the `x-dfo-caller` header on every request
        (`compiler.caller_clients.caller_transport`); `None` sends none.

        `tool=None` is for a client not scoped to one tool ahead of time --
        the long-lived presentation client (`harness.mcp_client.McpExecutor.
        _presentation_client`), which serves several tools across a
        conversation -- and gets the general default, same as any tool
        absent from the pinned table.
        """
        timeout = self.default_call_timeout_s if tool is None else self.call_timeout_s(tool)
        return Client(
            caller_transport(url, caller=caller, auth=auth),
            timeout=timeout,
            init_timeout=self.init_timeout_s,
        )


__all__ = ["McpTimeoutPolicy"]
