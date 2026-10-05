"""Tests for `harness.mcp_deadlines.McpTimeoutPolicy`.

The one place that turns `Settings`'s three MCP timeout fields, plus
`compiler.capabilities.call_timeouts()`'s per-tool overrides, into the
deadlines a constructed FastMCP `Client` actually carries. Every assertion
reads the deadline off the constructed `Client` itself (`_session_kwargs`,
`_init_timeout`) rather than off the argument passed in, so a policy that
built the timeout but forgot to thread it into the `Client` call would still
fail here.
"""

from __future__ import annotations

import pytest
from common.caller import CALLER_HEADER
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from harness.mcp_deadlines import McpTimeoutPolicy
from harness.settings import Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("MCP_CALL_TIMEOUT_S", "MCP_LIST_TIMEOUT_S", "MCP_INIT_TIMEOUT_S"):
        monkeypatch.delenv(var, raising=False)


def _call_deadline_s(client: Client) -> float | None:
    return client._session_kwargs["read_timeout_seconds"]


def test_get_data_carries_its_180s_capability_row() -> None:
    policy = McpTimeoutPolicy(settings=Settings())

    client = policy.client("http://data-mcp.internal/mcp", auth=None, caller=None, tool="get_data")

    assert _call_deadline_s(client) == 180.0


def test_search_and_describe_carry_their_30s_capability_row() -> None:
    policy = McpTimeoutPolicy(settings=Settings())

    for tool in ("search", "describe"):
        client = policy.client("http://data-mcp.internal/mcp", auth=None, caller=None, tool=tool)
        assert _call_deadline_s(client) == 30.0, tool


def test_a_tool_with_no_capability_row_gets_the_general_default() -> None:
    """Every analysis-mcp verb and every presentation-mcp tool: neither is in
    `compiler.capabilities.CAPABILITIES`, so `settings.mcp_call_timeout_s`
    covers them -- the general default, not an invented per-tool number."""
    settings = Settings(mcp_call_timeout_s=42.0)
    policy = McpTimeoutPolicy(settings=settings)

    for tool in ("aggregate", "growth", "describe_frame", "write_component"):
        client = policy.client("http://x.internal/mcp", auth=None, caller=None, tool=tool)
        assert _call_deadline_s(client) == 42.0, tool


def test_no_tool_named_gets_the_general_default() -> None:
    """The long-lived presentation client (`McpExecutor._presentation_client`)
    isn't scoped to one tool ahead of time."""
    settings = Settings(mcp_call_timeout_s=42.0)
    policy = McpTimeoutPolicy(settings=settings)

    client = policy.client("http://presentation-mcp.internal/mcp", auth=None, caller=None)

    assert _call_deadline_s(client) == 42.0


def test_every_client_carries_the_configured_init_deadline() -> None:
    settings = Settings(mcp_init_timeout_s=3.5)
    policy = McpTimeoutPolicy(settings=settings)

    client = policy.client("http://x.internal/mcp", auth=None, caller=None, tool="search")

    assert client._init_timeout == 3.5


def test_call_timeout_s_reads_the_same_mapping_client_construction_uses() -> None:
    policy = McpTimeoutPolicy(settings=Settings())

    assert policy.call_timeout_s("get_data") == 180.0
    assert policy.call_timeout_s("aggregate") == policy.default_call_timeout_s


def test_every_client_names_its_caller_in_the_header() -> None:
    policy = McpTimeoutPolicy(settings=Settings())

    named = policy.client("http://x.internal/mcp", auth="t", caller="conversation:a", tool="search")
    anonymous = policy.client("http://x.internal/mcp", auth="t", caller=None, tool="search")

    assert isinstance(named.transport, StreamableHttpTransport)
    assert isinstance(anonymous.transport, StreamableHttpTransport)
    assert named.transport.headers == {CALLER_HEADER: "conversation:a"}
    assert anonymous.transport.headers == {}
