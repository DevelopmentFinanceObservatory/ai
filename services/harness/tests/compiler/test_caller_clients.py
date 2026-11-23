"""Tests for `compiler.caller_clients`: one held client per caller and turn,
bounded by an LRU, and the transport that names the caller and turn in its
headers."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from common.caller import CALLER_HEADER, TURN_HEADER
from common.telemetry import SESSION_META_KEY
from compiler import caller_clients
from compiler.caller_clients import (
    MAX_CALLERS,
    CallerClients,
    CallerTurn,
    caller_transport,
    session_call_meta,
)
from fastmcp import Client, FastMCP


class _CountingClient:
    def __init__(self, caller: str | None, turn: str | None = None) -> None:
        self.caller = caller
        self.turn = turn
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> _CountingClient:
        self.entered += 1
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.exited += 1


class _GatedClient(_CountingClient):
    """A client whose connect waits until the test opens `gate`, or fails
    when `fail` is set."""

    def __init__(self, caller: str | None, *, gate: asyncio.Event, fail: bool) -> None:
        super().__init__(caller)
        self._gate = gate
        self._fail = fail

    async def __aenter__(self) -> _GatedClient:
        await self._gate.wait()
        if self._fail:
            raise ConnectionError("connect failed")
        return await super().__aenter__()


class _Registry:
    """A `CallerClients` over `_CountingClient`s, with a caller and turn the test sets.

    A caller in `gates` connects only once its gate opens; a caller in
    `failing` fails its connect.
    """

    def __init__(self) -> None:
        self.caller: str | None = None
        self.turn: str | None = None
        self.surface: str | None = None
        self.built: list[_CountingClient] = []
        self.gates: dict[str | None, asyncio.Event] = {}
        self.failing: set[str | None] = set()
        self.clients = CallerClients(
            self._build, caller=lambda: CallerTurn(self.caller, self.turn, self.surface)
        )

    def _build(self, who: CallerTurn) -> Any:
        caller = who.caller
        gate = self.gates.get(caller)
        client = (
            _GatedClient(caller, gate=gate, fail=caller in self.failing)
            if gate is not None
            else _CountingClient(caller, who.turn)
        )
        self.built.append(client)
        return client

    async def get_as(self, caller: str | None) -> Any:
        """`get()` with `caller` current, read before the first suspension."""
        self.caller = caller
        return await self.clients.get()


async def test_one_caller_reuses_one_client() -> None:
    registry = _Registry()
    registry.caller = "conversation:a"

    first = await registry.clients.get()
    second = await registry.clients.get()

    assert first is second
    assert len(registry.built) == 1
    assert registry.built[0].entered == 1


async def test_each_caller_gets_its_own_client_built_for_it() -> None:
    registry = _Registry()

    registry.caller = "conversation:a"
    a = await registry.clients.get()
    registry.caller = "conversation:b"
    b = await registry.clients.get()
    registry.caller = None
    anonymous = await registry.clients.get()

    assert len({id(a), id(b), id(anonymous)}) == 3
    assert [client.caller for client in registry.built] == [
        "conversation:a",
        "conversation:b",
        None,
    ]


async def test_a_callers_next_turn_gets_a_client_built_for_it() -> None:
    """A transport's headers are fixed when it is built, so a client kept into
    the next turn would name the turn before it. The new turn's client retires
    the old one, which disconnects once its in-flight call releases it."""
    registry = _Registry()
    registry.caller = "conversation:a"
    registry.turn = "tr_" + "1" * 16
    first = await registry.clients.get()  # still in flight below
    again = await registry.clients.get()
    await registry.clients.release(again)

    registry.turn = "tr_" + "2" * 16
    second = await registry.clients.get()

    assert again is first
    assert second is not first
    assert [(client.caller, client.turn) for client in registry.built] == [
        ("conversation:a", "tr_" + "1" * 16),
        ("conversation:a", "tr_" + "2" * 16),
    ]
    assert registry.built[0].exited == 0
    await registry.clients.release(first)
    assert registry.built[0].exited == 1
    assert registry.built[1].exited == 0


async def test_caller_less_clients_of_two_surfaces_never_retire_each_other() -> None:
    """A Studio replay's client and a caller-less client outside shared work
    name no caller and no turn, and differ in their surface header alone. Each
    is its own party's latest turn, so neither connect retires the other."""
    registry = _Registry()
    clients = []
    for surface in ["studio", None, "studio", None]:
        registry.surface = surface
        clients.append(await registry.clients.get())
        await registry.clients.release(clients[-1])

    assert clients[0] is clients[2]
    assert clients[1] is clients[3]
    assert len(registry.built) == 2
    assert [client.exited for client in registry.built] == [0, 0]


class _StuckExitClient(_CountingClient):
    """A client whose disconnect waits until the test opens `gate`."""

    def __init__(self, caller: str | None, turn: str | None, *, gate: asyncio.Event) -> None:
        super().__init__(caller, turn)
        self._gate = gate

    async def __aexit__(self, *exc_info: object) -> None:
        await self._gate.wait()
        await super().__aexit__(*exc_info)


class _FailingExitClient(_CountingClient):
    async def __aexit__(self, *exc_info: object) -> None:
        raise RuntimeError("session runner failed")


async def test_retiring_the_turn_befores_client_neither_delays_nor_fails_the_new_turn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """fastmcp can wait seconds on a session's end, or raise from it. The new
    turn's first call returns while the old client's disconnect still waits, and
    a disconnect that raises is logged. `aclose` waits for the stuck one."""
    gate = asyncio.Event()
    built: list[_CountingClient] = []

    def build(who: CallerTurn) -> Any:
        if who.turn == "tr_" + "1" * 16:
            client: _CountingClient = _StuckExitClient(who.caller, who.turn, gate=gate)
        elif who.turn == "tr_" + "2" * 16:
            client = _FailingExitClient(who.caller, who.turn)
        else:
            client = _CountingClient(who.caller, who.turn)
        built.append(client)
        return client

    turn = ["tr_" + "1" * 16]
    clients = CallerClients(build, caller=lambda: CallerTurn("conversation:a", turn[0]))
    for digit in "123":
        turn[0] = "tr_" + digit * 16
        client = await asyncio.wait_for(clients.get(), timeout=1)
        await clients.release(client)
    await asyncio.sleep(0)

    assert [client.turn for client in built] == ["tr_" + d * 16 for d in "123"]
    assert built[0].exited == 0  # still disconnecting
    assert "failed to disconnect" in caplog.text
    closing = asyncio.create_task(clients.aclose())
    await asyncio.sleep(0)
    assert not closing.done()
    gate.set()
    await asyncio.wait_for(closing, timeout=1)
    assert [built[0].exited, built[2].exited] == [1, 1]


async def test_a_late_call_from_an_earlier_turn_leaves_the_latest_turns_client_held() -> None:
    """An early mint turn 1 queued goes out after turn 2 began. It reconnects
    under turn 1's id and retires nothing; turn 3 retires both."""
    registry = _Registry()
    registry.caller = "conversation:a"
    turns = ["tr_" + digit * 16 for digit in "123"]

    async def call_in(turn: str) -> Any:
        registry.turn = turn
        client = await registry.clients.get()
        await registry.clients.release(client)
        return client

    first = await call_in(turns[0])
    second = await call_in(turns[1])
    late = await call_in(turns[0])
    again = await call_in(turns[1])
    third = await call_in(turns[2])

    assert first.exited == 1  # retired by turn 2's connect
    assert late is not first
    assert late.turn == turns[0]
    assert again is second
    assert [late.exited, second.exited, third.exited] == [1, 1, 0]


async def test_aclose_disconnects_and_forgets_every_client() -> None:
    registry = _Registry()
    client = await registry.clients.get()

    await registry.clients.aclose()

    assert registry.built[0].exited == 1
    again = await registry.clients.get()  # asking again after aclose reconnects
    assert again is not client


async def test_discard_reconnects_the_caller_without_touching_another_callers_client() -> None:
    registry = _Registry()
    registry.caller = "conversation:a"
    a = await registry.clients.get()
    await registry.clients.release(a)
    registry.caller = "conversation:b"
    b = await registry.clients.get()
    await registry.clients.release(b)

    await registry.clients.discard(a)
    registry.caller = "conversation:a"
    fresh = await registry.clients.get()
    registry.caller = "conversation:b"
    same_b = await registry.clients.get()

    assert registry.built[0].exited == 1
    assert fresh is not a
    assert same_b is b


async def test_the_least_recently_used_caller_is_evicted_past_the_cap() -> None:
    """Nothing tells the registry when a conversation ends, so the
    `MAX_CALLERS`-plus-first caller evicts the least recently used one's
    client rather than let the registry grow for the process life."""
    registry = _Registry()
    for i in range(MAX_CALLERS):
        registry.caller = f"conversation:{i}"
        client = await registry.clients.get()
        await registry.clients.release(client)

    oldest = registry.built[0]
    assert oldest.exited == 0

    registry.caller = "conversation:overflow"
    overflow = await registry.clients.get()
    await registry.clients.release(overflow)

    assert oldest.exited == 1
    assert len(registry.built) == MAX_CALLERS + 1

    registry.caller = "conversation:0"  # the evicted caller
    reconnected = await registry.clients.get()
    assert reconnected is not oldest


async def test_eviction_waits_for_the_evicted_clients_in_flight_call() -> None:
    """An evicted caller's client may still have a call in flight; it is
    disconnected by that call's own `release`, never out from under it."""
    registry = _Registry()
    registry.caller = "conversation:0"
    held = await registry.clients.get()  # never released below: still in flight

    for i in range(1, MAX_CALLERS + 1):
        registry.caller = f"conversation:{i}"
        client = await registry.clients.get()
        await registry.clients.release(client)

    assert registry.built[0].exited == 0

    await registry.clients.release(held)

    assert registry.built[0].exited == 1


def test_the_transport_names_the_caller_in_its_header() -> None:
    transport = caller_transport("http://mcp.internal/mcp", caller="conversation:a", auth="t")

    assert transport.headers == {CALLER_HEADER: "conversation:a"}
    assert transport.auth is not None


def test_the_transport_names_the_turn_beside_the_caller() -> None:
    transport = caller_transport(
        "http://mcp.internal/mcp", caller="conversation:a", turn="tr_" + "1" * 16, auth=None
    )

    assert transport.headers == {CALLER_HEADER: "conversation:a", TURN_HEADER: "tr_" + "1" * 16}


def test_no_caller_sends_no_header() -> None:
    transport = caller_transport("http://mcp.internal/mcp", caller=None, auth=None)

    assert transport.headers == {}
    assert transport.auth is None


@pytest.mark.parametrize(
    ("source", "sends_session"),
    [
        (caller_transport("http://127.0.0.1:8080/mcp/data", caller=None, auth=None), True),
        (caller_transport("http://localhost:8080/mcp/data", caller=None, auth=None), True),
        (caller_transport("http://[::1]:8080/mcp/data", caller=None, auth=None), True),
        (caller_transport("https://dfo-data-abc.a.run.app/mcp", caller=None, auth=None), False),
        (FastMCP("in-memory"), False),
    ],
    ids=["127.0.0.1", "localhost", "::1", "cloud-run", "in-memory"],
)
def test_the_session_goes_only_to_a_loopback_http_client(
    monkeypatch: pytest.MonkeyPatch, source: Any, sends_session: bool
) -> None:
    monkeypatch.setattr(caller_clients, "turn_call_meta", lambda: {SESSION_META_KEY: "conv-1"})

    meta = session_call_meta(Client(source))

    assert meta == ({SESSION_META_KEY: "conv-1"} if sends_session else {})


async def test_a_slow_connect_does_not_hold_up_another_callers_calls() -> None:
    """One conversation's connect hanging must not stall every other
    conversation's first call."""
    registry = _Registry()
    registry.gates["conversation:slow"] = asyncio.Event()
    slow = asyncio.create_task(registry.get_as("conversation:slow"))
    await asyncio.sleep(0)

    fast = await asyncio.wait_for(registry.get_as("conversation:fast"), timeout=1)

    assert fast.caller == "conversation:fast"
    assert not slow.done()
    registry.gates["conversation:slow"].set()
    assert (await slow).caller == "conversation:slow"


async def test_concurrent_first_calls_from_one_caller_share_one_connect() -> None:
    registry = _Registry()
    gate = registry.gates["conversation:a"] = asyncio.Event()
    waiters = [asyncio.create_task(registry.get_as("conversation:a")) for _ in range(3)]
    await asyncio.sleep(0)

    gate.set()
    clients = await asyncio.gather(*waiters)

    assert len(registry.built) == 1
    assert all(client is clients[0] for client in clients)
    assert registry.built[0].entered == 1


async def test_a_failed_connect_raises_in_every_waiter_and_the_next_call_retries() -> None:
    registry = _Registry()
    gate = registry.gates["conversation:a"] = asyncio.Event()
    registry.failing.add("conversation:a")
    waiters = [asyncio.create_task(registry.get_as("conversation:a")) for _ in range(2)]
    await asyncio.sleep(0)

    gate.set()
    for waiter in waiters:
        with pytest.raises(ConnectionError):
            await waiter

    registry.failing.clear()
    retried = await registry.get_as("conversation:a")
    assert retried is registry.built[1]
    assert len(registry.built) == 2


async def test_a_cancelled_waiter_leaves_the_connect_running_for_the_others() -> None:
    registry = _Registry()
    gate = registry.gates["conversation:a"] = asyncio.Event()
    cancelled = asyncio.create_task(registry.get_as("conversation:a"))
    other = asyncio.create_task(registry.get_as("conversation:a"))
    await asyncio.sleep(0)

    cancelled.cancel()
    gate.set()

    assert (await other).entered == 1
    assert cancelled.cancelled()
    assert len(registry.built) == 1
