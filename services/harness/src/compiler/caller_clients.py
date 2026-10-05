"""One held MCP `Client` per caller, and the transport that names the caller.

Every MCP call the harness makes carries its caller in the `x-dfo-caller`
header (`common.caller`): `conversation:<digest>` for a call made inside a
studio conversation, none outside one. A header is fixed when the transport
is built, so a client belongs to one caller. `CallerClients` holds one
client per caller and is the one registry both call paths use:
`harness.mcp_client` (`search`, `get_data`, `describe` and the presentation
tools) and `compiler.surface.McpSurface` (the analysis verbs and the
translator's discovery calls). This module lives in `compiler` because
`harness` imports `compiler` and not the reverse; which caller is current is
a function the harness passes in.

A held client for a whole conversation is the reuse `fastmcp.Client` is
built for (its docstring: "efficient session reuse in any scenario with
nested or concurrent client usage"), and it saves the connect round trips a
fresh client pays on every call.

Nothing in this process observes a conversation ending, so the registry is
an LRU bounded at `MAX_CALLERS` held clients, the same bound
`harness.provenance.ProvenanceStore` puts on its conversation buckets.

ADK can dispatch a turn's tool calls concurrently, so two calls can be in
flight on one caller's client at once. Every `get()` is paired with a
`release()`, and a client that `discard()` or the LRU evicts is
disconnected only once its last in-flight call releases it.

A caller's first `get()` connects in a task of its own, outside the
registry lock, so a slow connect delays only that caller's calls. Its other
calls wait on the same task.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Callable

from common.caller import CALLER_HEADER
from common.mcp_telemetry import is_loopback_url
from common.telemetry import turn_call_meta
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

logger = logging.getLogger(__name__)

MAX_CALLERS = 256


def caller_transport(url: str, *, caller: str | None, auth: str | None) -> StreamableHttpTransport:
    """A streamable-HTTP transport to `url` that sends `caller` on every request.

    `caller=None` sends no `x-dfo-caller` header, which the server reads as
    no caller. `auth` is the bearer token, `None` when the gate is disabled.
    """
    headers = {CALLER_HEADER: caller} if caller is not None else {}
    return StreamableHttpTransport(url, headers=headers, auth=auth)


def session_call_meta(client: Client) -> dict[str, str]:
    """The `_meta` for a `call_tool` on `client`: the turn's Langfuse session
    (`common.telemetry.turn_call_meta`) when `client` reaches a loopback HTTP
    URL, `{}` otherwise.

    Only a mount in this process honours the session
    (`common.mcp_telemetry.LangfuseSessionMiddleware`). A client at another
    host, or one with no HTTP transport (in-memory), sends none, so the
    conversation id does not leave the process.
    """
    transport = getattr(client, "transport", None)
    if isinstance(transport, StreamableHttpTransport) and is_loopback_url(transport.url):
        return turn_call_meta()
    return {}


class _HeldClient:
    """One caller's held client and the bookkeeping that guards its teardown.

    `in_flight` counts the calls between `CallerClients.get` and
    `CallerClients.release` on this client; while it is above zero nothing
    may disconnect it. `discarded` marks that `discard` or the LRU wants it
    gone, and the last `release` then disconnects it.
    """

    __slots__ = ("caller", "client", "discarded", "in_flight")

    def __init__(self, client: Client, *, caller: str | None) -> None:
        self.client = client
        self.caller = caller
        self.in_flight = 0
        self.discarded = False


class CallerClients:
    """One connected `Client` per caller, opened on first use and held until
    `aclose` or LRU eviction.

    `factory` builds the client for a caller; it runs at most once per
    caller while that caller's client is held. `caller` returns the current
    caller, and is read on every `get()`.
    """

    def __init__(
        self, factory: Callable[[str | None], Client], *, caller: Callable[[], str | None]
    ) -> None:
        self._factory = factory
        self._caller = caller
        # Least recently used first; `get()` moves a caller to the end.
        self._clients: OrderedDict[str | None, _HeldClient] = OrderedDict()
        # The connect in progress for each caller that has no held client yet.
        self._connecting: dict[str | None, asyncio.Task[None]] = {}
        # A held client stays here, keyed by `id(client)`, after `discard()`
        # or eviction removes it from `_clients`, so a straggling `release()`
        # still finds it even when its caller already holds a fresh client.
        self._by_client_id: dict[int, _HeldClient] = {}
        self._lock = asyncio.Lock()

    async def get(self) -> Client:
        """The current caller's client, connecting it on the caller's first
        call, with this call counted as one of its in-flight users.

        Concurrent calls from one caller share the returned client;
        `fastmcp.Client` supports concurrent `call_tool`s on one connection.
        A failed connect raises in every call waiting on it, and the next
        `get()` tries again.
        """
        caller = self._caller()
        while True:
            async with self._lock:
                held = self._clients.get(caller)
                if held is not None:
                    self._clients.move_to_end(caller)
                    held.in_flight += 1
                    return held.client
                connecting = self._connecting.get(caller)
                if connecting is None:
                    connecting = asyncio.create_task(self._connect(caller))
                    self._connecting[caller] = connecting
            # Shielded: one waiter's cancellation must not cancel the connect
            # the caller's other calls wait on. The loop then takes the held
            # client under the lock, or connects again if it was evicted or
            # discarded in between.
            await asyncio.shield(connecting)

    async def _connect(self, caller: str | None) -> None:
        """Connect `caller`'s client and hold it, evicting past `MAX_CALLERS`.

        The client becomes held in the same step that retires the connect,
        so a waiter always finds one or the other.
        """
        try:
            client = self._factory(caller)
            await client.__aenter__()
        except BaseException:
            del self._connecting[caller]
            raise
        to_disconnect: list[_HeldClient] = []
        async with self._lock:
            del self._connecting[caller]
            held = _HeldClient(client, caller=caller)
            self._clients[caller] = held
            self._by_client_id[id(client)] = held
            while len(self._clients) > MAX_CALLERS:
                evicted_caller, evicted = self._clients.popitem(last=False)
                logger.warning(
                    "caller_clients: dropped caller %r's held client past the %d-caller bound",
                    evicted_caller,
                    MAX_CALLERS,
                )
                if self._mark_discarded_locked(evicted):
                    to_disconnect.append(evicted)
        for evicted in to_disconnect:
            await evicted.client.__aexit__(None, None, None)

    async def release(self, client: Client) -> None:
        """Release one hold on `client` taken by `get()`, disconnecting it
        when it was discarded and this was its last in-flight user."""
        disconnect = False
        async with self._lock:
            held = self._by_client_id.get(id(client))
            if held is not None:
                held.in_flight -= 1
                if held.discarded and held.in_flight == 0:
                    disconnect = True
                    del self._by_client_id[id(client)]
        if disconnect:
            await client.__aexit__(None, None, None)

    async def discard(self, client: Client) -> None:
        """Stop holding `client`, so its caller's next `get()` reconnects.

        For a call that failed for any reason but its own read timeout: the
        connection itself is broken, and every later call would repeat the
        failure. Identity-checked, so a fresh client a concurrent call
        already reconnected survives. `client` is disconnected once its
        last in-flight user releases it, never under a sibling call.
        """
        disconnect = False
        async with self._lock:
            held = self._by_client_id.get(id(client))
            if held is not None and not held.discarded:
                if self._clients.get(held.caller) is held:
                    del self._clients[held.caller]
                disconnect = self._mark_discarded_locked(held)
        if disconnect:
            await client.__aexit__(None, None, None)

    def _mark_discarded_locked(self, held: _HeldClient) -> bool:
        """Mark `held` discarded and return whether it has no in-flight
        user, in which case the caller disconnects it now; otherwise the
        last `release()` does. Runs with `_lock` held."""
        held.discarded = True
        if held.in_flight == 0:
            del self._by_client_id[id(held.client)]
            return True
        return False

    async def aclose(self) -> None:
        """Disconnect every held client, at shutdown.

        Does not wait for in-flight calls: shutdown expects none. A connect
        still in progress is cancelled.
        """
        connecting = list(self._connecting.values())
        for task in connecting:
            task.cancel()
        await asyncio.gather(*connecting, return_exceptions=True)
        held_clients, self._clients = self._clients, OrderedDict()
        self._by_client_id = {}
        for held in held_clients.values():
            await held.client.__aexit__(None, None, None)
