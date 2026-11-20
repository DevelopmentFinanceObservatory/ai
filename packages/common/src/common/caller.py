"""The caller of the current MCP request.

A caller is an opaque string with a kind prefix, carried in the
`x-dfo-caller` request header: `conversation:<digest>` from the harness
(`conversation_caller`), and
`ip:`, `key:` or `user:` from the website's edge. A request without the
header has no caller, and `current_caller` returns `None` for it. Nothing
here generates an identity in its place.

The MCP transport session is not a caller. Under the 2026-07-28 protocol
revision a server builds a connection per request, so FastMCP's
`Context.session_id` is a fresh id on every call.

`current_caller` reads the header from `fastmcp.server.dependencies.
get_http_request`, and finds no caller outside an HTTP request. An in-memory
client carries no headers at all, so `supplied_caller` stands in for the
header in tests that drive a server that way. `set_supplied_caller` names
the caller for code that calls dfo-apis outside any MCP request.

A Studio turn also names itself, in the `x-dfo-turn` header: the turn's wire
trace id, `tr_` and 16 hex characters. `current_turn` reads it as
`current_caller` reads the caller, and `set_supplied_turn` stands in for it
outside any MCP request. `vouched_caller_headers` forwards it to dfo-apis
beside a `conversation:` caller, so dfo-apis logs each delivery with the turn
that caused it.

Work several requests join, such as a replay two Studio conversations wait on,
runs in `shared_work_context` and names no caller and no turn. It names the
surface it serves instead (`current_shared_surface`), and
`vouched_caller_headers` sends dfo-apis the principal and that surface alone on
a read, so dfo-apis logs the read under the surface with no caller. The
harness carries the surface to its MCP servers in `SURFACE_HEADER` on a request
that names no caller.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import Context, ContextVar, copy_context

CALLER_HEADER = "x-dfo-caller"

# Sent beside `CALLER_HEADER` to dfo-apis (`vouched_caller_headers`): the secret
# dfo-apis shares with its first-party callers, which vouches for the caller it
# names (`dfo_platform.caller.resolve_caller`).
CALLER_PRINCIPAL_HEADER = "x-dfo-caller-principal"

# Sent beside `CALLER_HEADER` for a `CONVERSATION_CALLER`: the Studio turn the
# call serves, by its wire trace id (`current_turn`).
TURN_HEADER = "x-dfo-turn"

# The form of a turn id: the trace id `harness.studio_state` mints per turn.
TURN = re.compile(r"tr_[0-9a-f]{16}")

# The form of a caller the website named by its client's address.
ADDRESS_CALLER = re.compile(r"ip:[0-9a-f]{32}")

# The form of the caller `conversation_caller` builds for a Studio conversation.
CONVERSATION_CALLER = re.compile(r"conversation:[0-9a-f]{32}")

# Sent beside `CALLER_PRINCIPAL_HEADER` for an `ADDRESS_CALLER`: the surface the
# caller reached dfo-apis through. A `conversation:` caller sends none, since
# dfo-apis reads Studio from the caller's prefix. Shared work sends its own
# surface with no caller (`current_shared_surface`).
SURFACE_HEADER = "x-dfo-surface"
MCP_SURFACE = "mcp"
STUDIO_SURFACE = "studio"

# The surfaces shared work may name: those whose every consumer is first-party,
# so a read several of them join counts against that surface and no client.
SHARED_SURFACES = frozenset({STUDIO_SURFACE})

# The methods a caller-less vouched request may use. dfo-apis refuses a vouched
# request without a caller (`CallerHeaderInvalid`) on the doors that read the
# caller, which take POST: pins and new exports.
_READ_METHODS = frozenset({"GET", "HEAD"})

# Sent beside `CALLER_PRINCIPAL_HEADER` for an `ADDRESS_CALLER`: the MCP client's
# `User-Agent` (`current_user_agent`). Distinct from x-dfo-client, which carries a
# dfo SDK client's label.
USER_AGENT_HEADER = "x-dfo-user-agent"

# The longest `User-Agent` `current_user_agent` keeps.
MAX_USER_AGENT_CHARS = 256

_USER_AGENT_HEADER = "user-agent"

# The longest caller `current_caller` accepts. The frame store writes the
# caller into GCS object metadata, which holds 8 KiB in all.
MAX_CALLER_CHARS = 256

# Hex characters of SHA-256 in a conversation caller (128 bits). Among a
# billion conversations the chance that any two share a caller is about
# 1.5e-21 (n^2 / 2^129).
_CONVERSATION_DIGEST_CHARS = 32

# What the frame store records as `writer` for a frame written with no caller.
ANONYMOUS_WRITER = "anonymous"

_supplied: ContextVar[str | None] = ContextVar("supplied_caller", default=None)
_supplied_turn: ContextVar[str | None] = ContextVar("supplied_turn", default=None)
# Set by `shared_work_context` to the surface the work serves: the context runs
# work several requests join, so the current MCP request's headers name none of
# them.
_shared_work: ContextVar[str | None] = ContextVar("shared_work", default=None)


def _request_headers(name: str) -> list[str] | None:
    """Every value of the current MCP request's `name` header, or `None` outside
    an HTTP request or inside `shared_work_context`."""
    if _shared_work.get() is not None:
        return None
    # Imported here so that `import common` stays free of fastmcp: the
    # package's value modules, such as `common.hashed_fields`, are imported
    # by code that serves no MCP request.
    from fastmcp.server.dependencies import get_http_request  # noqa: PLC0415

    try:
        return get_http_request().headers.getlist(name)
    except RuntimeError:  # no HTTP request in flight
        return None


def current_caller() -> str | None:
    """The caller of the current MCP request, or `None` when it names none.

    A caller set with `supplied_caller` wins over the header. A blank header,
    a repeated one, and one longer than `MAX_CALLER_CHARS` all count as no
    header. A proxy that sets the header must not lose to a copy the client
    sent beside it.
    """
    supplied = _supplied.get()
    if supplied is not None:
        return supplied
    values = _request_headers(CALLER_HEADER)
    if values is None or len(values) != 1:
        return None
    caller = values[0].strip()
    return caller if 0 < len(caller) <= MAX_CALLER_CHARS else None


def current_turn() -> str | None:
    """The Studio turn the current MCP request serves, or `None` when it names none.

    A turn set with `set_supplied_turn` wins over the header. Either counts only
    in the `TURN` form, and a repeated header counts as none, so the value is
    always a valid header value.
    """
    turn = _supplied_turn.get()
    if turn is None:
        values = _request_headers(TURN_HEADER)
        if values is None or len(values) != 1:
            return None
        turn = values[0].strip()
    return turn if TURN.fullmatch(turn) else None


def current_user_agent() -> str | None:
    """The current MCP request's `User-Agent`, whole, or `None`.

    Characters that are not printable ASCII (controls, tabs, accented letters)
    are dropped, then the value is stripped and clipped to `MAX_USER_AGENT_CHARS`
    (256). A header that is blank after that gives `None`, as does a call outside
    an HTTP request. The value is a valid header value and cannot forge a log
    field.
    """
    values = _request_headers(_USER_AGENT_HEADER)
    if not values:
        return None
    user_agent = values[0]
    cleaned = "".join(ch for ch in user_agent if ch.isascii() and ch.isprintable())
    return cleaned.strip()[:MAX_USER_AGENT_CHARS].strip() or None


def current_shared_surface() -> str | None:
    """The surface the current shared work serves, or `None` outside shared work.

    Inside `shared_work_context`, the surface it was built with. Otherwise, on an
    MCP request that names no caller, its `SURFACE_HEADER` when that is a single
    value in `SHARED_SURFACES`: the harness's shared work reaching an MCP server
    (`compiler.caller_clients.caller_transport`). A request that names a caller
    is that caller's, whatever surface header it carries.
    """
    if (surface := _shared_work.get()) is not None:
        return surface
    if current_caller() is not None:
        return None
    values = _request_headers(SURFACE_HEADER)
    if values is None or len(values) != 1:
        return None
    surface = values[0].strip().lower()
    return surface if surface in SHARED_SURFACES else None


def vouched_caller_headers(principal: str | None, *, method: str | None = None) -> dict[str, str]:
    """The headers that name the current request's caller to dfo-apis.

    Sent when this process holds `principal`, the secret dfo-apis shares with
    its first-party callers, and the caller has a form dfo-apis accepts beside
    it. For an `ADDRESS_CALLER`, an MCP client the website named by address:
    `CALLER_PRINCIPAL_HEADER`, `CALLER_HEADER`, `SURFACE_HEADER` as
    `MCP_SURFACE`, and `USER_AGENT_HEADER` when `current_user_agent` finds one.
    dfo-apis counts that client's allowance. For a `CONVERSATION_CALLER` from the
    harness: `CALLER_PRINCIPAL_HEADER` and `CALLER_HEADER`, and `TURN_HEADER`
    when `current_turn` names the turn, with no surface and no user agent, since
    dfo-apis reads Studio from the caller's prefix and the User-Agent is the
    harness's own. For shared work (`current_shared_surface`) read with a GET or
    HEAD `method`: `CALLER_PRINCIPAL_HEADER` and `SURFACE_HEADER` as the shared
    surface, with no caller and no turn, which dfo-apis logs under that surface
    with no caller. Empty otherwise, a shared write or a `method=None` call
    included, and dfo-apis counts the call against this service.
    """
    if principal is None:
        return {}
    caller = current_caller()
    if caller is None:
        surface = current_shared_surface()
        if surface is None or method is None or method.upper() not in _READ_METHODS:
            return {}
        return {CALLER_PRINCIPAL_HEADER: principal, SURFACE_HEADER: surface}
    headers = {CALLER_PRINCIPAL_HEADER: principal, CALLER_HEADER: caller}
    if ADDRESS_CALLER.fullmatch(caller):
        headers[SURFACE_HEADER] = MCP_SURFACE
        if (user_agent := current_user_agent()) is not None:
            headers[USER_AGENT_HEADER] = user_agent
    elif CONVERSATION_CALLER.fullmatch(caller):
        if (turn := current_turn()) is not None:
            headers[TURN_HEADER] = turn
    else:
        return {}
    return headers


def current_writer() -> str:
    """The frame store's `writer` for a frame written now: the caller, or
    `ANONYMOUS_WRITER` when there is none."""
    return current_caller() or ANONYMOUS_WRITER


def conversation_caller(conversation: str) -> str:
    """The caller the harness sends for `conversation`: `conversation:` and
    the first 32 hex characters of the key's SHA-256.

    The key holds the principal and a client-minted conversation id, and the
    caller reaches GCS metadata, apis' lease records and logs, so only
    the digest leaves the process. It has a fixed length and is always a
    valid header value.
    """
    digest = hashlib.sha256(conversation.encode()).hexdigest()
    return f"conversation:{digest[:_CONVERSATION_DIGEST_CHARS]}"


def set_supplied_caller(caller: str | None) -> None:
    """Make `current_caller` return `caller` for the rest of the current context,
    header or not. `None` hands the caller back to the header.

    For a process that calls dfo-apis for a caller outside any MCP request: the
    harness names a Studio turn's conversation this way, so the dfo-apis calls
    its graph makes in process (`vouched_caller_headers`) carry the caller its
    MCP calls name. Set it in a request's own context, which every task the
    request spawns copies.
    """
    _supplied.set(caller)


def set_supplied_turn(turn: str | None) -> None:
    """Make `current_turn` return `turn` for the rest of the current context,
    header or not. `None` hands the turn back to the header.

    The turn's counterpart of `set_supplied_caller`: the harness names a Studio
    turn this way beside its conversation, so the dfo-apis calls its graph makes
    in process carry the turn its MCP calls name.
    """
    _supplied_turn.set(turn)


def shared_work_context(*, surface: str) -> Context:
    """A copy of the current context that names no caller and no turn, and
    serves `surface`, one of `SHARED_SURFACES`.

    For a task several requests join, such as a replay two conversations both
    wait on or a fetch several MCP requests share: run it in this context
    (`asyncio.create_task(..., context=...)`) so the dfo-apis calls it makes name
    neither the caller nor the turn that happened to start it. The supplied
    caller and turn are cleared, and `current_caller`, `current_turn` and
    `current_user_agent` read no header of the MCP request in flight
    (`in_shared_work`). Its reads name `surface` to dfo-apis
    (`vouched_caller_headers`), so name only a surface every request that can
    join the work belongs to.

    Raises:
        ValueError: `surface` is not in `SHARED_SURFACES`.
    """
    if surface not in SHARED_SURFACES:
        raise ValueError(f"{surface!r} is not a surface shared work can serve")
    context = copy_context()
    context.run(_supplied.set, None)
    context.run(_supplied_turn.set, None)
    context.run(_shared_work.set, surface)
    return context


def in_shared_work() -> bool:
    """Whether the current context is one `shared_work_context` built, so the
    work it runs belongs to no single caller."""
    return _shared_work.get() is not None


@contextmanager
def supplied_caller(caller: str) -> Iterator[None]:
    """Make `current_caller` return `caller` inside the block, header or not.

    For a server driven through an in-memory client, which sends no
    headers. Open the client inside the block. The server's tasks copy the
    context when the client connects.
    """
    token = _supplied.set(caller)
    try:
        yield
    finally:
        _supplied.reset(token)
