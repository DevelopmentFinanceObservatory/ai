"""`current_caller`: the `x-dfo-caller` header of the current MCP request,
`None` without one, and `supplied_caller` in its place for in-memory clients."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx2
import pytest
from common.caller import (
    ANONYMOUS_WRITER,
    CALLER_HEADER,
    MAX_CALLER_CHARS,
    conversation_caller,
    current_caller,
    current_writer,
    supplied_caller,
)
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server.http import set_http_request
from mcp.shared._httpx_utils import McpHttpClientFactory
from starlette.requests import Request


def _probe_server(seen: list[str | None]) -> FastMCP:
    server = FastMCP(name="_caller_probe")

    @server.tool
    async def probe() -> str:
        """Record the caller this call arrived with."""
        seen.append(current_caller())
        return "ok"

    return server


@asynccontextmanager
async def _http_client(server: FastMCP, *, headers: dict[str, str]) -> AsyncIterator[Client]:
    """A client reaching `server` over streamable HTTP in-process, sending `headers`."""
    app = server.http_app(path="/mcp")

    def factory(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://testserver", **kwargs
        )

    transport = StreamableHttpTransport(
        "http://testserver/mcp",
        headers=headers,
        httpx_client_factory=cast(McpHttpClientFactory, factory),
    )
    async with app.lifespan(app), Client(transport) as client:
        yield client


def test_no_caller_outside_a_request() -> None:
    assert current_caller() is None
    assert current_writer() == ANONYMOUS_WRITER


async def test_the_header_names_the_caller_on_every_call() -> None:
    seen: list[str | None] = []

    async with _http_client(
        _probe_server(seen), headers={CALLER_HEADER: "conversation:u/c1"}
    ) as client:
        await client.call_tool("probe", {})
        await client.call_tool("probe", {})
        revision = client.protocol_version

    # The sessionless revision: each call is its own connection, so only the
    # header ties the two together.
    assert revision == "2026-07-28"
    assert seen == ["conversation:u/c1", "conversation:u/c1"]


async def test_a_request_without_the_header_has_no_caller() -> None:
    seen: list[str | None] = []

    async with _http_client(_probe_server(seen), headers={}) as client:
        await client.call_tool("probe", {})

    assert seen == [None]


async def test_a_blank_header_is_no_caller() -> None:
    seen: list[str | None] = []

    async with _http_client(_probe_server(seen), headers={CALLER_HEADER: "  "}) as client:
        await client.call_tool("probe", {})

    assert seen == [None]


async def test_an_in_memory_client_has_no_caller_unless_one_is_supplied() -> None:
    seen: list[str | None] = []
    server = _probe_server(seen)

    async with Client(server) as client:
        await client.call_tool("probe", {})
    with supplied_caller("conversation:u/c2"):
        async with Client(server) as client:
            await client.call_tool("probe", {})

    assert seen == [None, "conversation:u/c2"]
    assert current_caller() is None  # the block restored the default


def _caller_of(headers: list[tuple[str, str]]) -> str | None:
    """`current_caller` inside a request carrying `headers`, in order and with repeats."""
    scope = {
        "type": "http",
        "headers": [(name.encode(), value.encode()) for name, value in headers],
    }
    with set_http_request(Request(scope)):
        return current_caller()


def test_a_repeated_header_is_no_caller() -> None:
    """A proxy's value must not lose to a copy the client appended beside it."""
    assert _caller_of([(CALLER_HEADER, "user:edge")]) == "user:edge"
    assert _caller_of([(CALLER_HEADER, "user:edge"), (CALLER_HEADER, "user:forged")]) is None
    assert _caller_of([(CALLER_HEADER, "user:edge"), (CALLER_HEADER, "user:edge")]) is None


@pytest.mark.parametrize(
    ("length", "accepted"), [(MAX_CALLER_CHARS, True), (MAX_CALLER_CHARS + 1, False)]
)
def test_a_caller_past_the_length_cap_is_no_caller(length: int, accepted: bool) -> None:
    caller = "user:" + "x" * (length - len("user:"))

    assert _caller_of([(CALLER_HEADER, caller)]) == (caller if accepted else None)


def test_a_conversation_caller_is_a_fixed_length_digest_of_the_key() -> None:
    """Neither the principal nor the conversation id leaves the process, and
    the caller is a valid header value whatever the key holds."""
    key = "u_0123456789abcdef/café\nnext"
    caller = conversation_caller(key)

    assert caller == "conversation:" + hashlib.sha256(key.encode()).hexdigest()[:32]
    assert len(conversation_caller("u/a")) == len(caller) == len("conversation:") + 32
    assert "u_0123456789abcdef" not in caller
    assert conversation_caller(key) == caller
    assert conversation_caller("u/a") != conversation_caller("u/b")
