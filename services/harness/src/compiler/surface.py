"""The live MCP tool surface, as the compiler reads it.

`ToolSurface` is the interface the compiler depends on. `McpSurface` serves it
by asking each MCP server for its own tools, and `FixtureSurface` serves a
surface a test pins itself.

Reading over MCP keeps the compiler and the executor looking at one surface.
Tools are called over streamable HTTP at the URLs in `harness.settings`, and
the surface reads from those same URLs, so a workflow is compiled against the
schemas that will run it. data-mcp and analysis-mcp are standalone MCP
services; that they deploy inside the harness's own process today is a
deployment detail, and `DATA_MCP_URL` / `ANALYSIS_MCP_URL` exist to point at
them wherever they run.

Each source is keyed by its server's name, which is how a `ToolId` names the
server a tool belongs to.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import socket
import ssl
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Protocol, Self

import aiohttp
import anyio
import asyncpg
import httpx2
import rfc8785
from common.auth import bearer_token
from fastmcp import Client, FastMCP
from fastmcp.client.progress import ProgressHandler
from fastmcp.client.transports import ClientTransport
from google.auth import exceptions as google_auth_errors
from google.genai import errors as genai_errors
from mcp import MCPError
from mcp.types import CONNECTION_CLOSED, REQUEST_TIMEOUT, TextContent, Tool
from planner.workflow import ToolId
from pydantic import BaseModel, Field
from sqlalchemy import exc as sa_exc

from compiler.caller_clients import CallerClients, caller_transport, session_call_meta

logger = logging.getLogger(__name__)


class UnhashableSchemaError(ValueError):
    """Raised when a tool's JSON Schema holds a value RFC 8785 cannot encode.

    An integer outside IEEE 754 double precision (`rfc8785.IntegerDomainError`)
    or a non-finite float (`rfc8785.FloatDomainError`) is valid JSON Schema but
    has no canonical JCS encoding, so `schema_hash` cannot pin it.
    """


def schema_hash(schema: Mapping[str, object]) -> str:
    """Return the SHA-256 of `schema` as canonical JSON (RFC 8785).

    Canonical, so the hash sees the schema's content and not the order a
    server happened to serialise it in.

    Raises:
        UnhashableSchemaError: If `schema` holds a value RFC 8785 cannot
            encode, e.g. an out-of-range integer bound.
    """
    try:
        encoded = rfc8785.dumps(dict(schema))
    except ValueError as exc:
        raise UnhashableSchemaError(str(exc)) from exc
    return hashlib.sha256(encoded).hexdigest()


def schema_hash_error(schema: Mapping[str, object] | None) -> str | None:
    """Return why `schema` cannot be hashed, or `None` when it can (and for `None`).

    The total predicate `schema_hash` itself is not: mirrors
    `compiler.arguments.unusable_schema_reason`'s shape so the compiler reads
    both checks the same way.
    """
    if schema is None:
        return None
    try:
        schema_hash(schema)
    except UnhashableSchemaError as exc:
        return str(exc)
    return None


class ToolSchema(BaseModel):
    """One tool as the compiler sees it.

    `parameters` is the tool's own `inputSchema`, carried verbatim off the
    wire: arguments are validated against it directly, so any narrowing here
    would be a second, drifting description of the same contract.
    `output_schema` and `version` are carried when the server declares them.
    """

    server: str
    name: str
    description: str = ""
    parameters: dict[str, object] = Field(default_factory=dict)
    output_schema: dict[str, object] | None = None
    version: str | None = None

    @property
    def input_schema_hash(self) -> str:
        """The hash a capability pins this tool's argument schema by."""
        return schema_hash(self.parameters)

    @property
    def output_schema_hash(self) -> str | None:
        """The hash a capability pins this tool's output schema by, when it declares one."""
        return schema_hash(self.output_schema) if self.output_schema is not None else None


class ToolCallResult(BaseModel):
    """What one tool call returned, in the forms a caller reads.

    `structured_content` is the machine-readable result. `text` is the digest
    the tool renders for a model. `meta` is the result's `_meta`, where a tool
    puts what code reads and a model does not (`search`'s envelope, under
    `data_mcp.search_result.SEARCH_ENVELOPE_META_KEY`). A refusal arrives with
    `is_error` set and its typed detail in `structured_content`.
    """

    is_error: bool
    structured_content: dict[str, object] | None = None
    text: str = ""
    meta: dict[str, object] | None = None


@dataclass(frozen=True)
class ProgressReport:
    """One MCP progress notification a server sent while a tool call ran.

    The fields are the notification's own: `progress` rises with each
    report, `total` is `None` when the server cannot say how much work is
    left, and `message` is the server's prose for the current state.
    """

    progress: float
    total: float | None = None
    message: str | None = None


# What a caller hands `ToolSurface.call_tool` to hear a call's progress. It runs
# on the client's message loop, so it must return promptly; a slow callback
# delays every message on that session.
type ProgressCallback = Callable[[ProgressReport], None]


def deliver_progress(on_progress: ProgressCallback, report: ProgressReport) -> None:
    """Hand `report` to `on_progress`, logging and dropping it if the callback raises.

    Progress describes a call, and the call carries on without it, so a
    failing callback never fails the call it reports on.
    """
    try:
        on_progress(report)
    except Exception:
        logger.warning("a progress callback raised; dropping its report", exc_info=True)


def mcp_progress_handler(on_progress: ProgressCallback | None) -> ProgressHandler | None:
    """Adapt `on_progress` to the handler `fastmcp.Client.call_tool` takes.

    `Client.call_tool` attaches a progress token to the request, so every
    `ctx.report_progress` the tool makes reaches the handler. Returns `None`
    for no callback, which leaves the client's own default handler in place.
    """
    if on_progress is None:
        return None

    async def handle(progress: float, total: float | None, message: str | None) -> None:
        deliver_progress(
            on_progress, ProgressReport(progress=progress, total=total, message=message)
        )

    return handle


class ToolSurface(Protocol):
    """Reads the tool surface a workflow is compiled against, and calls tools on it."""

    async def list_tools(self) -> list[ToolSchema]:
        """Return every tool the covered servers serve."""
        ...

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, object],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        """Call `tool` with `arguments` and return what it answered, refusal included.

        `on_progress`, when given, receives each progress report the server
        sends for this call, before the call returns. A surface whose tools
        report no progress never calls it.
        """
        ...


class DuplicateToolNameError(ValueError):
    """Raised when two covered servers serve the same tool name.

    Capabilities and steps name a tool by server and name, but the compiler
    also has to look a tool up by name alone from a proposal, and two tools
    sharing one would leave that ambiguous.
    """


class ToolSurfaceUnavailableError(RuntimeError):
    """Raised when a covered server cannot be read.

    Separate from `DuplicateToolNameError` because the two call for different
    responses. A duplicate name is a misconfiguration that persists until
    someone fixes it, so retrying is pointless. An unreachable server is worth
    another attempt: the tools are fine and the attempt simply did not
    complete.

    `retryable` is `True` exactly when a later attempt can succeed, and an
    explicit `False` is final. A raise site that wraps an exception it did not
    classify itself sets it to `retryable_failure(cause)`: retryable unless
    the cause chain holds a fault a retry meets again. A site that knows the
    fault is permanent, such as a server this surface does not cover, sets it
    `False`, and one whose call was cut off or never issued for lack of time
    sets it `True`.
    """

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[object, ...], dict[str, object]]:
        # `BaseException.__reduce__` rebuilds as `cls(*args)`, which fails for
        # the required keyword-only `retryable` and for a subclass with its own
        # signature. This rebuilds without `__init__` and restores `__dict__`.
        return _rebuild, (type(self), *self.args), self.__dict__


def _rebuild[E: BaseException](cls: type[E], *args: object) -> E:
    """A `cls` holding `args`, made without calling its `__init__`."""
    return cls.__new__(cls, *args)


# The MCP error codes of a call whose connection closed or that timed out:
# the server may answer the same call later.
TRANSIENT_MCP_CODES = frozenset({CONNECTION_CLOSED, REQUEST_TIMEOUT})

# Raised by `fastmcp.client.client.Client._await_with_session_monitoring`
# (fastmcp 4) as a bare `RuntimeError` when the MCP session's task ends
# without an exception while a call waits on it: the server closed the
# stream. fastmcp offers no other signal: the error has no type of its own,
# `Client.is_connected()` stays true until the client exits, and the session
# task is private state. The message is all that tells it from any other
# `RuntimeError`, so this module is the one place that reads it, and
# `test_surface.py` pins it against the real client.
_FASTMCP_SESSION_ENDED = "Session task completed unexpectedly"


# The Postgres errors that end the connection itself: the server shutting
# down or crashing, starting up, or closing an idle session. A later attempt
# opens a new connection.
_POSTGRES_CONNECTION_ENDED = (
    asyncpg.AdminShutdownError,
    asyncpg.CrashShutdownError,
    asyncpg.CannotConnectNowError,
    asyncpg.IdleSessionTimeoutError,
)


def transient_status(status: int | None) -> bool:
    """Whether an HTTP `status` says to try again later: a rate limit (429) or
    a server error (5xx). Any other 4xx repeats for the same request."""
    return status is not None and (status == httpx2.codes.TOO_MANY_REQUESTS or status >= 500)


# The faults a retry meets again: a certificate that failed verification, a
# URL no client can send to (a scheme httpx2 does not speak, or a malformed
# one), and a request httpx2 itself refused to write.
_PERMANENT_FAULTS = (
    ssl.SSLCertVerificationError,
    httpx2.UnsupportedProtocol,
    httpx2.LocalProtocolError,
    httpx2.InvalidURL,
)


def permanent_fault(exc: BaseException) -> bool:
    """Whether `exc`, an exception it was raised from (`__cause__`), or a
    member of an exception group among them, is a fault the next attempt
    meets again (`_PERMANENT_FAULTS`). A transport wraps one, as httpx2's
    `ConnectError` and aiohttp's `ClientConnectorCertificateError` wrap a
    failed certificate, in an error that reads as a dropped connection. An
    exception merely raised while another was handled is its own fault."""
    seen: set[int] = set()
    link: BaseException | None = exc
    while link is not None and id(link) not in seen:
        if isinstance(link, _PERMANENT_FAULTS):
            return True
        if isinstance(link, BaseExceptionGroup) and any(
            permanent_fault(member) for member in link.exceptions
        ):
            return True
        seen.add(id(link))
        link = link.__cause__
    return False


def retryable_failure(exc: BaseException) -> bool:
    """Whether a call that failed with `exc`, caught by a site that does not
    classify it further, can succeed later. It can unless a known permanent
    fault is in the cause chain (`permanent_fault`): an answer the server gave
    in error, such as the `MCPError` the mcp client raises for a 5xx or a
    terminated session, a refused connection, a failed name lookup, a timeout
    and a closed connection all read as a fault a later attempt may not meet."""
    return not permanent_fault(exc)


def transient(exc: BaseException) -> bool:
    """Whether `exc`, or an exception it was raised from (`__cause__`), is a
    fault a later attempt may not meet. A wrapper such as
    `executor.run.RunAbandonedError`, or fastmcp's `RuntimeError('Client
    failed to connect: ...')` over an httpx2 `ConnectError`, keeps the class
    of the transport error under it. An exception merely raised while another
    was handled is its own fault, such as a bug in an `except` block. An
    exception group, as anyio's task groups raise, is transient when every
    exception in it is, so one bug among dropped connections still reads as a
    fault. A permanent fault anywhere in the chain makes it not transient
    (`permanent_fault`), whatever wraps it. A `ToolSurfaceUnavailableError`
    states its own answer, `retryable`, which is final."""
    if permanent_fault(exc):
        return False
    seen: set[int] = set()
    link: BaseException | None = exc
    while link is not None and id(link) not in seen:
        if isinstance(link, ToolSurfaceUnavailableError):
            return link.retryable
        if isinstance(link, BaseExceptionGroup):
            return all(transient(leaf) for leaf in link.exceptions)
        if _transient_link(link):
            return True
        seen.add(id(link))
        link = link.__cause__
    return False


def _transient_link(exc: BaseException) -> bool:
    """Whether `exc` alone is transient: a timeout, the session store's pool
    timeout among them (`sqlalchemy.exc.TimeoutError`, every pooled
    connection busy under load); a connection that failed or dropped, a
    Postgres one included (`PostgresConnectionError`, or a server ending
    the connection, `_POSTGRES_CONNECTION_ENDED`, or refusing one when its
    connection slots are full, `TooManyConnectionsError`), or one SQLAlchemy
    invalidated as a disconnect (`DBAPIError.connection_invalidated`); a
    name lookup or TLS failure; a transport error, httpx2's, aiohttp's
    connection or payload error (the Gemini SDK's transport, under which a
    peer's reset reads as `ClientPayloadError`), or google-auth's while
    fetching a token (a `RefreshError` raised from one is read by its
    cause); an HTTP or model answer of 429 or 5xx after the client's
    retries, httpx2's included (`transient_status`); an MCP call whose
    connection closed or that timed out (`TRANSIENT_MCP_CODES`), fastmcp's
    session ending under a call included (`_FASTMCP_SESSION_ENDED`; anyio's
    closed or broken stream, under its "Server session was closed
    unexpectedly"). A TLS failure is transient except a certificate that
    failed verification (`ssl.SSLCertVerificationError`), and a transport
    error that is a permanent fault (`permanent_fault`) is not transient
    either; `transient` refuses these and any wrapper of one before this reads
    the link. Any other `OSError`, such as `FileNotFoundError`, and any other
    aiohttp error, such as `InvalidURL`, is a fault in this code, and any
    other MCP error an answer the server or this client got wrong, which a
    later attempt meets again. Any other `DBAPIError` is read by the asyncpg error it was
    raised from (`transient` walks `__cause__`), so a connection error under
    it is transient and a SQL error a fault."""
    match exc:
        case (
            genai_errors.APIError(code=status)
            | aiohttp.ClientResponseError(status=status)
            | httpx2.HTTPStatusError(response=httpx2.Response(status_code=status))
        ):
            return transient_status(status)
    if isinstance(exc, sa_exc.DBAPIError):
        return exc.connection_invalidated
    if isinstance(exc, MCPError):
        return exc.error.code in TRANSIENT_MCP_CODES
    if type(exc) is RuntimeError and exc.args == (_FASTMCP_SESSION_ENDED,):
        return True
    return isinstance(
        exc,
        (
            TimeoutError,
            ConnectionError,
            socket.gaierror,
            ssl.SSLError,
            asyncpg.PostgresConnectionError,
            asyncpg.TooManyConnectionsError,
            *_POSTGRES_CONNECTION_ENDED,
            sa_exc.TimeoutError,
            httpx2.TransportError,
            aiohttp.ClientConnectionError,
            aiohttp.ClientPayloadError,
            google_auth_errors.TransportError,
            anyio.ClosedResourceError,
            anyio.BrokenResourceError,
        ),
    )


# What a server is reached through: its URL, the `FastMCP` instance itself over
# the in-memory transport, or a transport a test built to reach a mounted app.
Source = str | FastMCP | ClientTransport

# Deadlines a surface falls back to when a caller states no per-tool override.
# The default is the short one deliberately: a surface built without the
# per-tool mapping (`compiler.capabilities.call_timeouts`, wired in through
# `harness.agents.build_tool_surface`) fails loudly on a slow mint rather than
# silently stretching every call to cover it.
DEFAULT_CALL_TIMEOUT_S = 60.0
DEFAULT_LIST_TIMEOUT_S = 15.0
DEFAULT_INIT_TIMEOUT_S = 10.0


def _source_label(source: Source) -> str:
    """Name a source for an error message: its URL, the server's name, or the transport."""
    if isinstance(source, str):
        return source
    if isinstance(source, FastMCP):
        return source.name
    return repr(source)


def _auth_for(source: Source) -> str | None:
    """The bearer `Client` should carry to reach `source`, or `None`.

    A URL source is the mount over real streamable HTTP -- `common.auth`'s
    gate applies, so it carries the same `bearer_token()` the harness's own
    `mcp_client` clients do (`None` when the gate is disabled). A `FastMCP`
    source is the in-memory transport, which the gate never reaches
    (`fastmcp.Client(auth=...)` on that transport raises), so it must stay
    unauthenticated. A `ClientTransport` a test built is left to that test to
    authenticate, since it may or may not be gated.
    """
    return bearer_token() if isinstance(source, str) else None


def _no_caller() -> None:
    """The caller of a surface built without one: none."""
    return None


def _call_client(source: Source, caller: str | None, *, init_timeout_s: float) -> Client:
    """The `call_tool` client for `source` on behalf of `caller`.

    A URL source is the mount over streamable HTTP, so its client names
    `caller` in the `x-dfo-caller` header and carries the bearer token. An
    in-memory `FastMCP` or a test-built transport has no header to set, and
    its calls carry no caller.
    """
    if isinstance(source, str):
        transport = caller_transport(source, caller=caller, auth=_auth_for(source))
        return Client(transport, init_timeout=init_timeout_s)
    return Client(source, init_timeout=init_timeout_s)


class McpSurface:
    """A `ToolSurface` that asks each MCP server for its own tools.

    A source is the URL of an MCP server -- what the harness passes, taken
    from the same settings tools are called on -- a `FastMCP` instance, which
    `fastmcp.Client` serves over its in-memory transport, or a
    `ClientTransport` a test built to reach a server mounted in a composed
    app. The last two are for tests and eval runs: each still speaks the MCP
    protocol, and needs no server on a port to do it.

    `list_tools` reads every source fresh, on every call: a workflow compiled
    against a stale surface is the failure this class exists to prevent,
    which outweighs the loopback round trip each read costs. `call_tool` gets
    no such benefit from a fresh client, since the tool it invokes was
    already resolved by an earlier `list_tools`. It holds one `Client` per
    source per caller (`compiler.caller_clients.CallerClients`), the same
    registry `harness.mcp_client` holds its clients in. `caller` returns the
    current caller; a URL source's client names it in the `x-dfo-caller`
    header, which data-mcp and analysis-mcp record as a stored frame's
    `writer`. The harness passes the current conversation's caller
    (`harness.agents.build_tool_surface`); a surface built without one holds
    one client per source and its calls carry no caller.

    `call_timeouts` is an opaque `{ToolId: seconds}` mapping supplied at
    construction. This module knows no tool by name and must not import
    `compiler.capabilities` (`ToolCapability.call_timeout_s`, gathered by
    `compiler.capabilities.call_timeouts()` and wired in by
    `harness.agents.build_tool_surface`, the one module that already imports
    both), since capabilities imports `ToolSchema` from here and that would be
    a cycle. A tool absent from the mapping, or a surface built with none,
    gets `default_call_timeout_s`. `list_timeout_s` and `init_timeout_s` bound
    `list_tools`'s own reads the same way.
    """

    def __init__(
        self,
        *,
        sources: Mapping[str, Source],
        call_timeouts: Mapping[ToolId, float] | None = None,
        default_call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
        list_timeout_s: float = DEFAULT_LIST_TIMEOUT_S,
        init_timeout_s: float = DEFAULT_INIT_TIMEOUT_S,
        caller: Callable[[], str | None] = _no_caller,
    ) -> None:
        """Keep `sources`, keyed by server name.

        Raises:
            ValueError: If a `FastMCP` source is keyed by a name other than
                its own, since the key is what a `ToolId` names the server by.
        """
        for name, source in sources.items():
            if isinstance(source, FastMCP) and source.name != name:
                raise ValueError(f"source {name!r} is the server named {source.name!r}")
        self._sources = dict(sources)
        self._call_timeouts = dict(call_timeouts or {})
        self._default_call_timeout_s = default_call_timeout_s
        self._list_timeout_s = list_timeout_s
        self._init_timeout_s = init_timeout_s
        # `call_tool`'s held clients, one registry per source. `list_tools`
        # and `_read` open a fresh client every time by design and never
        # touch these.
        self._call_clients = {
            server: CallerClients(
                partial(_call_client, source, init_timeout_s=init_timeout_s), caller=caller
            )
            for server, source in self._sources.items()
        }

    async def aclose(self) -> None:
        """Disconnect every client `call_tool` is holding.

        Nothing on the served path calls this: `harness.agents.
        build_tool_surface` builds the one `McpSurface` the process runs for
        its whole life, and no shutdown hook holds a reference to it today.
        A caller that builds its own `McpSurface` and wants its held clients
        released must call this itself.
        """
        for clients in self._call_clients.values():
            await clients.aclose()

    async def list_tools(self) -> list[ToolSchema]:
        """Return every tool the covered servers serve, server by server.

        Raises:
            ToolSurfaceUnavailableError: If a source cannot be read.
            DuplicateToolNameError: If two servers serve the same tool name.
        """
        results = await asyncio.gather(
            *(self._read(source) for source in self._sources.values()), return_exceptions=True
        )
        tool_lists: list[list[Tool]] = []
        for result in results:
            if isinstance(result, BaseException):
                raise result
            tool_lists.append(result)
        schemas: dict[str, ToolSchema] = {}
        for server, tools in zip(self._sources.keys(), tool_lists, strict=True):
            for tool in tools:
                if tool.name in schemas:
                    raise DuplicateToolNameError(
                        f"tool {tool.name!r} is served by more than one covered server"
                    )
                schemas[tool.name] = ToolSchema(
                    server=server,
                    name=tool.name,
                    description=tool.description or "",
                    parameters=dict(tool.input_schema),
                    output_schema=dict(tool.output_schema) if tool.output_schema else None,
                    version=_declared_version(tool),
                )
        return list(schemas.values())

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, object],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        """Call `tool` on the server that serves it.

        A refusal is an answer, not a failure: it comes back as a
        `ToolCallResult` with `is_error` set, for the caller to read.
        `on_progress` hears the server's progress notifications for this
        call (`mcp_progress_handler`).

        Raises:
            ToolSurfaceUnavailableError: If the server is not covered, or the
                call itself did not complete -- including a call that ran
                past its deadline. A timeout on its own does not discard the
                held client: it says this one call was slow, not that the
                connection is broken, so a slow `get_data` call leaves the
                same client free to serve `search` right after. Any other
                exception discards it, so the next call to the same source
                reconnects; a sibling call still in flight on it keeps it
                until that call is done (`CallerClients.discard`).
        """
        source = self._sources.get(tool.server)
        if source is None:
            raise ToolSurfaceUnavailableError(
                f"no server named {tool.server!r} is covered; covered: {sorted(self._sources)!r}",
                retryable=False,
            )
        timeout = self._call_timeouts.get(tool, self._default_call_timeout_s)
        clients = self._call_clients[tool.server]
        # `client` stays `None` when `get()` itself fails to connect: nothing
        # was acquired, so there is nothing to discard or release below. The
        # outer `except` still reports it the same way.
        client: Client | None = None
        try:
            client = await clients.get()
            result = await client.call_tool(
                tool.name,
                dict(arguments),
                timeout=timeout,
                meta=session_call_meta(client),
                progress_handler=mcp_progress_handler(on_progress),
                raise_on_error=False,
            )
        except MCPError as exc:
            if client is not None and exc.error.code != REQUEST_TIMEOUT:
                logger.warning(
                    "%r on %r failed for a reason other than its own timeout; "
                    "discarding %r's held client so the next call reconnects",
                    tool.name,
                    tool.server,
                    tool.server,
                    exc_info=True,
                )
                await clients.discard(client)
            raise ToolSurfaceUnavailableError(
                f"could not call {tool.name!r} at {_source_label(source)!r}: {exc}",
                retryable=retryable_failure(exc),
            ) from exc
        except Exception as exc:
            logger.warning(
                "%r on %r failed; discarding %r's held client so the next call reconnects",
                tool.name,
                tool.server,
                tool.server,
                exc_info=True,
            )
            if client is not None:
                await clients.discard(client)
            raise ToolSurfaceUnavailableError(
                f"could not call {tool.name!r} at {_source_label(source)!r}: {exc}",
                retryable=retryable_failure(exc),
            ) from exc
        finally:
            if client is not None:
                await clients.release(client)
        structured = result.structured_content
        return ToolCallResult(
            is_error=result.is_error,
            structured_content=dict(structured) if isinstance(structured, Mapping) else None,
            meta=dict(result.meta) if result.meta else None,
            text="\n".join(
                block.text for block in result.content if isinstance(block, TextContent)
            ),
        )

    async def _read(self, source: Source) -> list[Tool]:
        """List one source's tools, or raise `ToolSurfaceUnavailableError`.

        The broad `except` is the point: fastmcp reports a server it cannot
        reach, or one that ran past `list_timeout_s` / `init_timeout_s`, as a
        bare `RuntimeError`, so there is no narrower type to catch that would
        still cover both. Only the client read sits inside it, which leaves
        `DuplicateToolNameError` free to escape on its own from `list_tools`.
        """
        try:
            async with Client(
                source,
                auth=_auth_for(source),
                timeout=self._list_timeout_s,
                init_timeout=self._init_timeout_s,
            ) as client:
                return await client.list_tools()
        except Exception as exc:
            raise ToolSurfaceUnavailableError(
                f"could not read the tool surface at {_source_label(source)!r}: {exc}",
                retryable=retryable_failure(exc),
            ) from exc


def _declared_version(tool: Tool) -> str | None:
    """The version a FastMCP server declares for `tool`, if it declares one."""
    fastmcp_meta = (tool.meta or {}).get("fastmcp")
    if not isinstance(fastmcp_meta, Mapping):
        return None
    version = fastmcp_meta.get("version")
    return version if isinstance(version, str) else None


class FixtureSurface:
    """A `ToolSurface` over fixed schemas and canned answers, for tests that
    pin the surface they want to state a case against.

    `responses` maps a tool name to what a call to it answers. A tool with
    no answer is treated as one whose server cannot be reached.
    """

    def __init__(
        self,
        *,
        schemas: Sequence[ToolSchema],
        responses: Mapping[str, ToolCallResult] | None = None,
    ) -> None:
        self._schemas = list(schemas)
        self._responses = dict(responses or {})

    async def list_tools(self) -> list[ToolSchema]:
        return list(self._schemas)

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, object],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        del arguments, on_progress  # a canned answer depends on neither
        if tool.name not in self._responses:
            raise ToolSurfaceUnavailableError(
                f"no canned answer for {tool.name!r}", retryable=False
            )
        return self._responses[tool.name]
