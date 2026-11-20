"""`current_caller`: the `x-dfo-caller` header of the current MCP request,
`None` without one, and `supplied_caller` in its place for in-memory clients."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import copy_context
from typing import Any, cast

import httpx2
import pytest
from common.caller import (
    ANONYMOUS_WRITER,
    CALLER_HEADER,
    CALLER_PRINCIPAL_HEADER,
    MAX_CALLER_CHARS,
    MAX_USER_AGENT_CHARS,
    SURFACE_HEADER,
    TURN_HEADER,
    USER_AGENT_HEADER,
    conversation_caller,
    current_caller,
    current_turn,
    current_user_agent,
    current_writer,
    set_supplied_caller,
    set_supplied_turn,
    shared_work_context,
    supplied_caller,
    vouched_caller_headers,
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


def _user_agent_of(user_agent: str) -> str | None:
    scope = {"type": "http", "headers": [(b"user-agent", user_agent.encode("latin-1"))]}
    with set_http_request(Request(scope)):
        return current_user_agent()


@pytest.mark.parametrize(
    ("user_agent", "expected"),
    [
        ("claude-user/1.0 (+https://claude.ai)", "claude-user/1.0 (+https://claude.ai)"),
        ("  openai-mcp/1.0.0 ", "openai-mcp/1.0.0"),
        ("a/" + "x" * 300, ("a/" + "x" * 300)[:MAX_USER_AGENT_CHARS]),
        ("caf\xe9/1.0", "caf/1.0"),  # codespell:ignore caf
        ("a/1.0\t(x)", "a/1.0(x)"),
        ("   ", None),
    ],
    ids=["whole", "stripped", "clipped", "non-ascii", "tab", "blank"],
)
def test_the_user_agent_is_kept_whole_and_clipped(user_agent: str, expected: str | None) -> None:
    assert _user_agent_of(user_agent) == expected


def test_no_user_agent_outside_a_request() -> None:
    assert current_user_agent() is None


_ADDRESS = "ip:" + "0" * 32
_CONVERSATION = conversation_caller("u/c1")


@pytest.mark.parametrize(
    ("caller", "user_agent", "principal", "sent"),
    [
        (
            _ADDRESS,
            "claude-user/1.0 (+https://claude.ai)",
            "p",
            {
                CALLER_PRINCIPAL_HEADER: "p",
                CALLER_HEADER: _ADDRESS,
                SURFACE_HEADER: "mcp",
                USER_AGENT_HEADER: "claude-user/1.0 (+https://claude.ai)",
            },
        ),
        (
            _CONVERSATION,
            "python-httpx/0.28",
            "p",
            {CALLER_PRINCIPAL_HEADER: "p", CALLER_HEADER: _CONVERSATION},
        ),
        (
            _ADDRESS,
            None,
            "p",
            {CALLER_PRINCIPAL_HEADER: "p", CALLER_HEADER: _ADDRESS, SURFACE_HEADER: "mcp"},
        ),
        (_ADDRESS, "claude-user/1.0", None, {}),
        (_CONVERSATION, "claude-user/1.0", None, {}),
        ("conversation:u/c1", "claude-user/1.0", "p", {}),
        ("user:edge", "claude-user/1.0", "p", {}),
        (None, "claude-user/1.0", "p", {}),
    ],
    ids=[
        "mcp-client",
        "studio",
        "no-user-agent",
        "address-unset",
        "conversation-unset",
        "malformed-conversation",
        "other-kind",
        "no-caller",
    ],
)
def test_the_vouched_headers_name_an_address_or_conversation_caller(
    caller: str | None, user_agent: str | None, principal: str | None, sent: dict[str, str]
) -> None:
    """dfo-apis reads Studio from a `conversation:` prefix, so that caller sends
    no surface, and it refuses any other caller form beside the principal."""
    headers = [(CALLER_HEADER, caller), ("user-agent", user_agent)]
    scope = {
        "type": "http",
        "headers": [(k.encode(), v.encode()) for k, v in headers if v is not None],
    }
    with set_http_request(Request(scope)):
        assert vouched_caller_headers(principal) == sent


_TURN = "tr_" + "0123456789abcdef"


@pytest.mark.parametrize(
    ("caller", "turns", "sent_turn"),
    [
        (_CONVERSATION, [_TURN], _TURN),
        (_ADDRESS, [_TURN], None),
        (_CONVERSATION, [], None),
        (_CONVERSATION, ["tr_" + "0123456789ABCDEF"], None),
        (_CONVERSATION, [_TURN, _TURN], None),
    ],
    ids=["studio-turn", "address-caller", "no-turn", "malformed", "repeated"],
)
def test_the_vouched_headers_name_a_conversations_turn(
    caller: str, turns: list[str], sent_turn: str | None
) -> None:
    """A Studio turn reaches dfo-apis beside its `conversation:` caller only,
    and only in the form the harness mints."""
    headers = [(CALLER_HEADER, caller), *((TURN_HEADER, turn) for turn in turns)]
    scope = {"type": "http", "headers": [(k.encode(), v.encode()) for k, v in headers]}
    with set_http_request(Request(scope)):
        assert vouched_caller_headers("p").get(TURN_HEADER) == sent_turn


async def test_a_supplied_turn_names_an_in_process_call() -> None:
    """Outside any MCP request the supplied caller and turn stand in for the
    headers, as the harness's graph calls dfo-apis in process."""
    assert current_turn() is None
    set_supplied_caller(_CONVERSATION)
    set_supplied_turn(_TURN)

    assert vouched_caller_headers("p") == {
        CALLER_PRINCIPAL_HEADER: "p",
        CALLER_HEADER: _CONVERSATION,
        TURN_HEADER: _TURN,
    }


_STUDIO_READ = {CALLER_PRINCIPAL_HEADER: "p", SURFACE_HEADER: "studio"}


@pytest.mark.parametrize(
    ("shared", "inbound", "principal", "method", "sent"),
    [
        (True, [], "p", "GET", _STUDIO_READ),
        (True, [], "p", "head", _STUDIO_READ),
        (True, [(CALLER_HEADER, _CONVERSATION), (TURN_HEADER, _TURN)], "p", "GET", _STUDIO_READ),
        (True, [], "p", "POST", {}),
        (True, [], "p", None, {}),
        (True, [], None, "GET", {}),
        (False, [(SURFACE_HEADER, "studio")], "p", "GET", _STUDIO_READ),
        (False, [(SURFACE_HEADER, "studio")], "p", "POST", {}),
        (False, [(SURFACE_HEADER, "mcp")], "p", "GET", {}),
        (False, [(SURFACE_HEADER, "studio"), (SURFACE_HEADER, "studio")], "p", "GET", {}),
        (
            False,
            [(CALLER_HEADER, _CONVERSATION), (SURFACE_HEADER, "studio")],
            "p",
            "GET",
            {CALLER_PRINCIPAL_HEADER: "p", CALLER_HEADER: _CONVERSATION},
        ),
        (False, [], "p", "GET", {}),
    ],
    ids=[
        "shared-get",
        "shared-head",
        "shared-inside-a-turns-request",
        "shared-post",
        "shared-method-unknown",
        "shared-unset",
        "header-get",
        "header-post",
        "header-unshared-surface",
        "header-repeated",
        "header-beside-a-caller",
        "no-caller",
    ],
)
def test_shared_work_reads_name_their_surface_and_no_caller(
    shared: bool,
    inbound: list[tuple[str, str]],
    principal: str | None,
    method: str | None,
    sent: dict[str, str],
) -> None:
    """Shared work, in this process or reaching it from the harness in the
    surface header of a caller-less request, names the principal and its surface
    on a read and nothing on a write: dfo-apis refuses a vouched request with no
    caller on the doors that read the caller, which take POST."""
    scope = {"type": "http", "headers": [(k.encode(), v.encode()) for k, v in inbound]}
    with set_http_request(Request(scope)):
        context = shared_work_context(surface="studio") if shared else copy_context()
        assert context.run(vouched_caller_headers, principal, method=method) == sent


def test_shared_work_serves_only_a_first_party_surface() -> None:
    """A shared read counts against its surface, so the surface of a public
    client, whose reads count against that client, is refused."""
    with pytest.raises(ValueError, match="mcp"):
        shared_work_context(surface="mcp")
