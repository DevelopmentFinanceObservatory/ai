"""One held MCP `Client` per caller and turn, and the transport that names them.

Every MCP call the harness makes carries its caller in the `x-dfo-caller`
header (`common.caller`): `conversation:<digest>` for a call made inside a
studio conversation, none outside one. A call made inside a Studio turn also
carries the turn's wire trace id in the `x-dfo-turn` header, which the MCP
servers forward to dfo-apis (`common.caller.vouched_caller_headers`). A header
is fixed when the transport is built, so a client belongs to one caller and
one turn (`CallerTurn`): a client kept into the next turn would name the
turn before it on every call. `CallerClients` holds one client per caller
and turn and is the one registry both call paths use:
`harness.mcp_client` (`search`, `get_data`, `describe` and the presentation
tools) and `compiler.surface.McpSurface` (the analysis verbs and the
translator's discovery calls). This module lives in `compiler` because
`harness` imports `compiler` and not the reverse; which caller is current is
a function the harness passes in.

A held client for a whole turn is the reuse `fastmcp.Client` is built for
(its docstring: "efficient session reuse in any scenario with nested or
concurrent client usage"), and it saves the connect round trips a fresh
client pays on every call. A turn pays one connect per registry it calls on,
and each registry reconnects on every turn: a turn that searches, fetches and
aggregates pays about four, since data-mcp alone has three registries
(`harness.mcp_client.McpResolver` for `search`, `harness.mcp_client.McpExecutor`
for `get_data`, and `McpSurface`'s data-mcp source) and analysis-mcp adds one.

The registry tracks each caller's latest turn: the newest turn `get()` has
seen for it, among the last `RECENT_TURNS`. A turn id is random, so the order
`get()` first sees turns in is the only order there is. The latest turn's
connect retires the caller's clients of older turns and disconnects them in a
task of its own (`CallerClients._retire`), so the new turn's call neither
waits on nor fails with that disconnect. A connect for an older
turn, such as an early mint the turn before queued and sent late, retires
nothing, so it never cuts the latest turn's client. Nothing in this process
observes a conversation ending, so the registry is also an LRU bounded at
`MAX_CALLERS` held clients, the same bound
`harness.provenance.ProvenanceStore` puts on its conversation buckets.

Pydantic AI dispatches a response's tool calls concurrently, so two calls can
be in flight on one caller's client at once. Every `get()` is paired with a
`release()`, and a client that `discard()` or the LRU evicts is disconnected
only once its last in-flight call releases it.

A caller's first `get()` connects in a task of its own, outside the
registry lock, so a slow connect delays only that caller's calls. Its other
calls wait on the same task.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import Any, NamedTuple

from common.caller import CALLER_HEADER, TURN_HEADER
from common.mcp_telemetry import is_loopback_url
from common.telemetry import turn_call_meta
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

logger = logging.getLogger(__name__)

MAX_CALLERS = 256

# The turns a caller's order is kept for. A call from a turn older than these
# counts as a new turn. An early mint is withdrawn once its card is approved or
# replaced, so none outlives one later turn, let alone this many.
RECENT_TURNS = 8


class CallerTurn(NamedTuple):
    """Whom an MCP call is made for: the caller, and the Studio turn it serves.

    `caller=None` is a call outside any conversation. `turn=None` is a call
    outside any Studio turn.
    """

    caller: str | None
    turn: str | None = None


def caller_transport(
    url: str, *, caller: str | None, auth: str | None, turn: str | None = None
) -> StreamableHttpTransport:
    """A streamable-HTTP transport to `url` that sends `caller` and `turn` on
    every request.

    `caller=None` sends no `x-dfo-caller` header, which the server reads as
    no caller, and `turn=None` no `x-dfo-turn` header. `auth` is the bearer
    token, `None` when the gate is disabled.
    """
    headers = {CALLER_HEADER: caller} if caller is not None else {}
    if turn is not None:
        headers[TURN_HEADER] = turn
    return StreamableHttpTransport(url, headers=headers, auth=auth)


def session_call_meta(client: Client) -> dict[str, Any]:
    """The `_meta` for a `call_tool` on `client`: the turn's Langfuse session,
    user and tags (`common.telemetry.turn_call_meta`) when `client` reaches a
    loopback HTTP URL, `{}` otherwise.

    Only a mount in this process honours them
    (`common.mcp_telemetry.LangfuseSessionMiddleware`). A client at another
    host, or one with no HTTP transport (in-memory), sends none, so none of
    the three leaves the process.
    """
    transport = getattr(client, "transport", None)
    if isinstance(transport, StreamableHttpTransport) and is_loopback_url(transport.url):
        return turn_call_meta()
    return {}


class _HeldClient:
    """One caller and turn's held client and the bookkeeping that guards its
    teardown.

    `in_flight` counts the calls between `CallerClients.get` and
    `CallerClients.release` on this client; while it is above zero nothing
    may disconnect it. `discarded` marks that `discard` or the LRU wants it
    gone, and the last `release` then disconnects it.
    """

    __slots__ = ("caller", "client", "discarded", "in_flight")

    def __init__(self, client: Client, *, caller: CallerTurn) -> None:
        self.client = client
        self.caller = caller
        self.in_flight = 0
        self.discarded = False


class CallerClients:
    """One connected `Client` per caller and turn, opened on first use and held
    until a later turn of the caller connects, `aclose`, or LRU eviction.

    `factory` builds the client for a `CallerTurn`; it runs at most once per
    caller and turn while that client is held. `caller` returns the current
    `CallerTurn`, and is read on every `get()`.
    """

    def __init__(
        self, factory: Callable[[CallerTurn], Client], *, caller: Callable[[], CallerTurn]
    ) -> None:
        self._factory = factory
        self._caller = caller
        # Least recently used first; `get()` moves a caller to the end.
        self._clients: OrderedDict[CallerTurn, _HeldClient] = OrderedDict()
        # The connect in progress for each caller and turn that has no held
        # client yet.
        self._connecting: dict[CallerTurn, asyncio.Task[None]] = {}
        # A held client stays here, keyed by `id(client)`, after `discard()`
        # or eviction removes it from `_clients`, so a straggling `release()`
        # still finds it even when its caller already holds a fresh client.
        self._by_client_id: dict[int, _HeldClient] = {}
        # Each caller's last `RECENT_TURNS` turns in the order `get()` first saw
        # them, the latest last; least recently used caller first.
        self._turns: OrderedDict[str | None, deque[str | None]] = OrderedDict()
        self._lock = asyncio.Lock()
        # The disconnects `_retire` runs in the background, held until each ends.
        self._retiring: set[asyncio.Task[None]] = set()

    async def get(self) -> Client:
        """The current `CallerTurn`'s client, connecting it on that caller and
        turn's first call, with this call counted as one of its in-flight users.

        Concurrent calls from one caller share the returned client;
        `fastmcp.Client` supports concurrent `call_tool`s on one connection.
        A failed connect raises in every call waiting on it, and the next
        `get()` tries again.
        """
        caller = self._caller()
        async with self._lock:
            self._note_turn_locked(caller)
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

    def _note_turn_locked(self, caller: CallerTurn) -> None:
        """Record `caller.turn` as its caller's latest turn when it is one the
        caller's recent turns do not hold. Runs with `_lock` held."""
        turns = self._turns.get(caller.caller)
        if turns is None:
            turns = self._turns[caller.caller] = deque(maxlen=RECENT_TURNS)
        self._turns.move_to_end(caller.caller)
        if caller.turn not in turns:
            turns.append(caller.turn)
        while len(self._turns) > MAX_CALLERS:
            self._turns.popitem(last=False)

    def _is_latest_locked(self, caller: CallerTurn) -> bool:
        """Whether `caller.turn` is its caller's latest turn. A caller whose turns
        were dropped past `MAX_CALLERS` counts its turn as latest. Runs with
        `_lock` held."""
        turns = self._turns.get(caller.caller)
        return not turns or turns[-1] == caller.turn

    async def _connect(self, caller: CallerTurn) -> None:
        """Connect `caller`'s client and hold it, evicting past `MAX_CALLERS`.
        When `caller.turn` is its caller's latest turn, the caller's clients of
        older turns are retired.

        The client becomes held in the same step that retires the connect,
        so a waiter always finds one or the other. A retired client with no
        in-flight call is disconnected in the background (`_retire`); one with
        a call in flight is disconnected once its last call releases it.
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
            superseded = (
                [key for key in self._clients if key.caller == caller.caller and key != caller]
                if self._is_latest_locked(caller)
                else []
            )
            for key in superseded:
                retired = self._clients.pop(key)
                if self._mark_discarded_locked(retired):
                    to_disconnect.append(retired)
            while len(self._clients) > MAX_CALLERS:
                evicted_caller, evicted = self._clients.popitem(last=False)
                logger.warning(
                    "caller_clients: dropped caller %r's held client past the %d-caller bound",
                    evicted_caller,
                    MAX_CALLERS,
                )
                if self._mark_discarded_locked(evicted):
                    to_disconnect.append(evicted)
        for retired in to_disconnect:
            self._retire(retired.client)

    def _retire(self, client: Client) -> None:
        """Disconnect `client` in a task of its own, which logs and swallows a
        failed disconnect. fastmcp can wait seconds on a session's end, and
        the connect that retires `client` serves another turn's call."""
        task = asyncio.create_task(_disconnect_quietly(client))
        self._retiring.add(task)
        task.add_done_callback(self._retiring.discard)

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
        still in progress is cancelled, and the background disconnects of
        retired clients are awaited.
        """
        connecting = list(self._connecting.values())
        for task in connecting:
            task.cancel()
        await asyncio.gather(*connecting, return_exceptions=True)
        held_clients, self._clients = self._clients, OrderedDict()
        self._by_client_id = {}
        for held in held_clients.values():
            await held.client.__aexit__(None, None, None)
        await asyncio.gather(*self._retiring, return_exceptions=True)


async def _disconnect_quietly(client: Client) -> None:
    """Disconnect `client`, logging rather than raising a failure."""
    try:
        await client.__aexit__(None, None, None)
    except Exception:
        logger.warning("caller_clients: a retired client failed to disconnect", exc_info=True)
