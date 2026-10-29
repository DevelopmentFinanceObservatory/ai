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
header in tests that drive a server that way.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

CALLER_HEADER = "x-dfo-caller"

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
    # Imported here so that `import common` stays free of fastmcp: the
    # package's value modules, such as `common.hashed_fields`, are imported
    # by code that serves no MCP request.
    from fastmcp.server.dependencies import get_http_request  # noqa: PLC0415

    try:
        values = get_http_request().headers.getlist(CALLER_HEADER)
    except RuntimeError:  # no HTTP request in flight
        return None
    if len(values) != 1:
        return None
    caller = values[0].strip()
    return caller if 0 < len(caller) <= MAX_CALLER_CHARS else None


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
