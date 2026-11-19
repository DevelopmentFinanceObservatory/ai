"""How a stream's response class ends on the harness's own server.

A Studio stream has one owner: a task, created by the route as its last step,
that runs the segment, writes its close and puts the stream's frames on a
channel. The response class relays the channel to the client and, in a
`finally` around its ASGI `__call__`, sets the client-gone flag and awaits the
owner under a shield. These tests run that response class
(`harness.studio_relay.StudioStream`) with a scripted owner on uvicorn over a real socket, with the server configuration
`harness.server.main` uses, and check every way a stream can end: the client
closes before the response starts, while the relay waits for a frame, while
the relay is blocked sending, or during a slow owner write; the server shuts
down mid-stream, gracefully or past a graceful-shutdown timeout; and a
keep-alive connection carries a second request after a stream.

In each shape the `finally` runs, the owner finishes uncancelled, and it
finishes before the ASGI call returns. Past a graceful-shutdown timeout uvicorn
stops waiting for the response, so the app's lifespan shutdown also awaits
every live owner (`harness.studio_stream.LiveOwners`); without that wait the event loop's teardown cancels an owner
still in its closing write.

uvicorn's HTTP protocols report ASGI spec version 2.3, so Starlette's
`StreamingResponse` runs a disconnect listener beside the body and cancels the
body when the client leaves; the parent `__call__` then returns normally. The
tests record which path each shape took (`Probe.exit`), so a server that moves
to spec 2.4, where a send to a gone client raises `OSError` instead, shows up
here as a changed exit.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import override

import uvicorn
from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from harness.server import GRACEFUL_SHUTDOWN_S
from harness.studio_relay import Channel, StudioStream
from harness.studio_stream import LiveOwners
from pydantic import BaseModel
from starlette.types import Message, Receive, Scope, Send

# Every wait in this file is bounded by this, so a shape that hangs fails.
_DEADLINE_S = 5.0


@dataclass(frozen=True)
class Script:
    """What the scripted owner does.

    The owner sends `frames` frames, `interval_s` apart, then makes its
    closing write (`close_s`) and sends `done`. Before frame
    `slow_write_at` it makes a mid-stream write of `slow_write_s`.
    `frames=None` runs until the client leaves. `frame_bytes` pads each frame.
    After its last frame the owner waits for `hold` to be set, when given.
    """

    frames: int | None
    interval_s: float = 0.02
    close_s: float = 0.2
    slow_write_at: int | None = None
    slow_write_s: float = 0.0
    frame_bytes: int = 0
    hold: asyncio.Event | None = None


class _Frame(BaseModel):
    event: str
    n: int = 0
    pad: str = ""


@dataclass
class Probe:
    """What one stream did, as the relay and the owner recorded it."""

    owner_log: list[str] = field(default_factory=list)
    exit: str | None = None
    finally_ran: bool = False
    owner_done_at_return: bool = False
    spec_version: str | None = None
    sending: bool = False
    sends: int = 0
    returned: asyncio.Event = field(default_factory=asyncio.Event)
    relay_waiting: asyncio.Event = field(default_factory=asyncio.Event)


async def _owner(script: Script, *, channel: Channel, gone: asyncio.Event, probe: Probe) -> None:
    """A scripted owner: reads the gone flag only while it waits for the work."""
    log = probe.owner_log
    try:
        outcome = "done"
        n = 0
        while script.frames is None or n < script.frames:
            if n == script.slow_write_at:
                log.append("slow-write")
                await asyncio.sleep(script.slow_write_s)
                log.append("slow-write-committed")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(gone.wait(), timeout=script.interval_s)
            if gone.is_set():
                outcome = "left"
                break
            channel.put_nowait(_Frame(event="frame", n=n, pad="x" * script.frame_bytes))
            n += 1
        if script.hold is not None and outcome == "done":
            await script.hold.wait()
        log.append(f"closing:{outcome}")
        await asyncio.sleep(script.close_s)
        log.append("closed")
        if outcome == "done":
            channel.put_nowait(_Frame(event="done"))
        channel.put_nowait(None)
        log.append("finished")
    except asyncio.CancelledError:
        log.append("cancelled")
        raise


class _Watched(asyncio.Queue[BaseModel | None]):
    """The owner's channel, noting when the response waits on it."""

    def __init__(self, probe: Probe) -> None:
        super().__init__()
        self._probe = probe

    @override
    async def get(self) -> BaseModel | None:
        self._probe.relay_waiting.set()
        return await super().get()


class _Probed(StudioStream):
    """The response class, with what its ASGI call did recorded on `probe`."""

    def __init__(
        self, *, channel: Channel, owner: asyncio.Task[None], gone: asyncio.Event, probe: Probe
    ) -> None:
        super().__init__(channel=channel, owner=owner, gone=gone, keepalive_s=1.0)
        self._watched_owner = owner
        self._watched_gone = gone
        self._probe = probe

    @override
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self._probe.spec_version = scope.get("asgi", {}).get("spec_version")

        async def watched(message: Message) -> None:
            self._probe.sending = True
            try:
                await send(message)
            finally:
                self._probe.sending = False
                self._probe.sends += 1

        try:
            await super().__call__(scope, receive, watched)
            self._probe.exit = "returned"
        except BaseException as exc:
            self._probe.exit = type(exc).__name__
            raise
        finally:
            self._probe.finally_ran = self._watched_gone.is_set()
            self._probe.owner_done_at_return = self._watched_owner.done()
            self._probe.returned.set()


@dataclass
class Stage:
    """The route's inputs for the next stream, set by each test."""

    script: Script = field(default_factory=lambda: Script(frames=3))
    probe: Probe = field(default_factory=Probe)
    wait_for_disconnect: bool = False
    owners: LiveOwners = field(default_factory=LiveOwners)


def _app(stage: Stage) -> FastAPI:
    """A FastAPI app routed as the Studio routes are: a prefixed router whose
    POST takes a JSON body and an `authorization` header."""
    router = APIRouter(prefix="/api/studio")

    @router.post("/turn")
    async def turn(
        request: Request, body: dict[str, str], authorization: str | None = Header(default=None)
    ) -> _Probed:
        del body, authorization
        if stage.wait_for_disconnect:
            while not await request.is_disconnected():
                await asyncio.sleep(0.01)
        channel = _Watched(stage.probe)
        gone = asyncio.Event()
        owner = stage.owners.start(
            _owner(stage.script, channel=channel, gone=gone, probe=stage.probe)
        )
        return _Probed(channel=channel, owner=owner, gone=gone, probe=stage.probe)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        # Past a graceful-shutdown timeout uvicorn cancels each request task
        # and moves on to this shutdown; once `serve` returns, the event
        # loop's teardown would cancel every owner still running.
        await stage.owners.wait(timeout=_DEADLINE_S)

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"ok": True})

    return app


class _Server(uvicorn.Server):
    """uvicorn's server, leaving the test process's signal handlers alone."""

    @override
    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


@contextlib.asynccontextmanager
async def _serving(
    stage: Stage, *, timeout_graceful_shutdown: int = GRACEFUL_SHUTDOWN_S
) -> AsyncIterator[tuple[uvicorn.Server, int]]:
    """uvicorn on a free loopback port with `harness.server.main`'s
    configuration: its graceful-shutdown timeout and every other option at its
    default."""
    config = uvicorn.Config(
        _app(stage),
        host="127.0.0.1",
        port=0,
        log_level="warning",
        timeout_graceful_shutdown=timeout_graceful_shutdown,
    )
    server = _Server(config)
    task = asyncio.create_task(server.serve())
    async with asyncio.timeout(_DEADLINE_S):
        while not server.started:
            await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield server, port
    finally:
        server.should_exit = True
        async with asyncio.timeout(_DEADLINE_S):
            await task


class _Client:
    """An HTTP/1.1 client on a raw socket, so a test controls when it stops
    reading and when it closes."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer

    @classmethod
    async def connect(cls, port: int) -> _Client:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        return cls(reader, writer)

    def post_turn(self) -> None:
        body = json.dumps({"conversation_id": "c"}).encode()
        head = (
            "POST /api/studio/turn HTTP/1.1\r\nHost: test\r\nAuthorization: Bearer t\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
        )
        self.writer.write(head.encode() + body)

    async def headers(self) -> bytes:
        return await self.reader.readuntil(b"\r\n\r\n")

    async def chunk(self) -> bytes:
        """The next chunk of a chunked body; empty at the body's end."""
        size = int((await self.reader.readuntil(b"\r\n")).strip(), 16)
        data = await self.reader.readexactly(size + 2)
        return data[:-2]

    async def body(self) -> list[bytes]:
        chunks: list[bytes] = []
        while chunk := await self.chunk():
            chunks.append(chunk)
        return chunks

    async def close(self) -> None:
        self.writer.close()
        with contextlib.suppress(ConnectionError):
            await self.writer.wait_closed()


async def _returned(probe: Probe) -> None:
    async with asyncio.timeout(_DEADLINE_S):
        await probe.returned.wait()


async def _still_sending(probe: Probe) -> bool:
    """Whether one send stays pending for 100 ms, so the relay is blocked in it."""
    sends = probe.sends
    await asyncio.sleep(0.1)
    return probe.sending and probe.sends == sends


def _assert_owner_finished(probe: Probe, *, outcome: str) -> None:
    assert probe.finally_ran
    assert probe.owner_done_at_return
    assert "cancelled" not in probe.owner_log
    assert probe.owner_log[-3:] == [f"closing:{outcome}", "closed", "finished"]


async def test_the_client_closes_before_the_response_starts() -> None:
    stage = Stage(script=Script(frames=None), wait_for_disconnect=True)
    async with _serving(stage) as (_, port):
        client = await _Client.connect(port)
        client.post_turn()
        await client.writer.drain()
        await client.close()
        await _returned(stage.probe)
    assert stage.probe.spec_version == "2.3"
    assert stage.probe.exit == "returned"
    _assert_owner_finished(stage.probe, outcome="left")


async def test_the_client_closes_while_the_relay_waits_for_a_frame() -> None:
    # The owner sends nothing for a second: the relay sits at its wait.
    stage = Stage(script=Script(frames=None, interval_s=1.0))
    async with _serving(stage) as (_, port):
        client = await _Client.connect(port)
        client.post_turn()
        await client.headers()
        async with asyncio.timeout(_DEADLINE_S):
            await stage.probe.relay_waiting.wait()
        await client.close()
        await _returned(stage.probe)
    assert stage.probe.exit == "returned"
    _assert_owner_finished(stage.probe, outcome="left")


async def test_the_client_closes_mid_body() -> None:
    stage = Stage(script=Script(frames=None))
    async with _serving(stage) as (_, port):
        client = await _Client.connect(port)
        client.post_turn()
        await client.headers()
        assert (await client.chunk()).startswith(b"data: ")
        await client.close()
        await _returned(stage.probe)
    assert stage.probe.exit == "returned"
    _assert_owner_finished(stage.probe, outcome="left")


async def test_the_client_closes_while_the_relay_is_blocked_sending() -> None:
    # 256 KiB frames with no gap fill the socket buffers once the client
    # stops reading, so the relay's send waits on uvicorn's flow control.
    stage = Stage(script=Script(frames=None, interval_s=0.0, frame_bytes=256 * 1024))
    async with _serving(stage) as (_, port):
        client = await _Client.connect(port)
        client.post_turn()
        await client.headers()
        async with asyncio.timeout(_DEADLINE_S):
            while not (stage.probe.sending and await _still_sending(stage.probe)):
                await asyncio.sleep(0.01)
        await client.close()
        await _returned(stage.probe)
    assert stage.probe.exit == "returned"
    _assert_owner_finished(stage.probe, outcome="left")


async def test_the_client_closes_during_a_slow_owner_write() -> None:
    stage = Stage(script=Script(frames=None, slow_write_at=1, slow_write_s=0.5))
    async with _serving(stage) as (_, port):
        client = await _Client.connect(port)
        client.post_turn()
        await client.headers()
        await client.chunk()
        async with asyncio.timeout(_DEADLINE_S):
            while "slow-write" not in stage.probe.owner_log:
                await asyncio.sleep(0.01)
        await client.close()
        await _returned(stage.probe)
    assert stage.probe.exit == "returned"
    _assert_owner_finished(stage.probe, outcome="left")
    # The write the client left during ran to its commit.
    assert "slow-write-committed" in stage.probe.owner_log


async def test_a_graceful_shutdown_lets_the_stream_end() -> None:
    # A stream that ends inside the graceful-shutdown timeout, as deployed
    # (`GRACEFUL_SHUTDOWN_S`), is waited for, so the client receives every
    # frame and `done`.
    # The owner ends on an event the test sets right after `should_exit`, so
    # the end has no timing of its own to race the bound.
    hold = asyncio.Event()
    stage = Stage(script=Script(frames=1, hold=hold))
    async with _serving(stage) as (server, port):
        client = await _Client.connect(port)
        client.post_turn()
        await client.headers()
        await client.chunk()
        server.should_exit = True
        hold.set()
        chunks = await client.body()
        await _returned(stage.probe)
        await client.close()
    assert json.loads(chunks[-1].removeprefix(b"data: "))["event"] == "done"
    assert stage.probe.exit == "returned"
    _assert_owner_finished(stage.probe, outcome="done")


async def test_a_stream_that_outlasts_the_deployed_timeout_is_cut_and_closed_as_left() -> None:
    # The stream has no end of its own, so only uvicorn's bound after SIGTERM
    # (`GRACEFUL_SHUTDOWN_S`) ends it: the response is cancelled, the owner
    # cuts its turn and finishes its closing write, all before `serve` returns.
    stage = Stage(script=Script(frames=None, interval_s=0.05))
    async with _serving(stage) as (server, port):
        client = await _Client.connect(port)
        client.post_turn()
        await client.headers()
        await client.chunk()
        server.should_exit = True
    await _returned(stage.probe)
    await client.close()
    _assert_owner_finished(stage.probe, outcome="left")


async def test_a_shutdown_past_its_timeout_cancels_the_response_and_not_the_owner() -> None:
    # The owner is still in its closing write when uvicorn cancels the
    # response, so only the lifespan's wait keeps it alive until `serve`
    # returns.
    stage = Stage(script=Script(frames=None, interval_s=0.05, close_s=0.5))
    async with _serving(stage, timeout_graceful_shutdown=0) as (server, port):
        client = await _Client.connect(port)
        client.post_turn()
        await client.headers()
        await client.chunk()
        server.should_exit = True
    assert stage.probe.owner_log[-1] == "finished"
    await _returned(stage.probe)
    await client.close()
    assert stage.probe.exit == "CancelledError"
    _assert_owner_finished(stage.probe, outcome="left")


async def test_a_keep_alive_connection_serves_the_next_request_after_a_stream() -> None:
    stage = Stage(script=Script(frames=2))
    async with _serving(stage) as (_, port):
        client = await _Client.connect(port)
        client.post_turn()
        await client.headers()
        chunks = await client.body()
        await _returned(stage.probe)
        client.writer.write(b"GET /healthz HTTP/1.1\r\nHost: test\r\n\r\n")
        head = await client.headers()
        await client.close()
    assert json.loads(chunks[-1].removeprefix(b"data: "))["event"] == "done"
    assert head.startswith(b"HTTP/1.1 200")
    assert stage.probe.exit == "returned"
    _assert_owner_finished(stage.probe, outcome="done")
