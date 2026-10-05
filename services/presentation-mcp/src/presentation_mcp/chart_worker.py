"""The worker process that draws chart images, off the server's event loop.

`chart_image.chart_png` is CPU-bound and resvg holds the GIL while it
rasterises, so run on the event loop, or on a thread, it would stall every
MCP server and Studio turn the harness process serves. `ChartWorker` runs it
in one worker process instead, which it owns outright, so that a render
past its timeout can be ended by killing the process. The server's lifespan
starts the worker and stops it (`server._lifespan`).

The worker process is started with `spawn`, a fresh interpreter, since the
server's process runs threads, asyncio, and OpenTelemetry, which a `fork`
would copy mid-state. `spawn` runs the parent's `__main__` module again in
the worker, which is why `harness.cli`, the `harness-server` script's
module, imports the harness only when it runs. Once warm, the worker reports
its resident memory and whether `harness.server` got imported, and the
server logs both at INFO, so a launch path that drags the harness into the
worker shows in the logs.

The worker serves one render at a time over a socket pair. A request is a
function of this package and its argument, and the reply is the result or
the exception it raised. Only the envelope, a JSON-shaped dict, goes to
the worker for a render, and only PNG bytes or None come back.

Each message crosses the socket as a frame: the length of its pickle in an
8-byte header, then the pickle. The server's end of the pair is a pair of
asyncio streams, so every write and read of a frame is awaited on the event
loop, and a timeout or a cancellation ends a transfer at whichever byte it
has reached. The worker's end is a blocking socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing
import pickle
import resource
import socket
import struct
import sys
from asyncio import StreamReader, StreamWriter
from collections.abc import Callable, Mapping
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

from presentation_mcp import chart_image

logger = logging.getLogger(__name__)

# Seconds a warm-up may take, from the spawn to the fonts loaded,
# before the worker is killed and the next render starts a new one.
_WARM_TIMEOUT_S = 30.0
# Seconds `stop` waits for the worker to finish its render and exit before
# it kills the worker.
_STOP_WAIT_S = 5.0
# The header of a frame: the length of the pickle that follows it, in bytes.
_FRAME_SIZE = struct.Struct("!Q")


class WorkerDiedError(Exception):
    """The worker process exited before it replied."""


class WorkerStoppedError(Exception):
    """`stop` ran before the worker process could start."""


class ChartWorker:
    """One worker process that renders `chart_png`, and its lifecycle.

    Renders wait for a warm-up in progress, then take turns on the worker
    under `_lock`. A render that overruns its timeout or is cancelled while
    the worker draws it, or that finds the worker dead, kills the process.
    `_call` does the killing and, where `warm` asks for it, starts a
    background warm-up of a replacement in the same step; without `warm`, the
    next render starts the replacement. A worker killed during its own
    warm-up gets no warm-up of its own, since that one is still in progress,
    so the next render starts its replacement. A render that times out or is
    cancelled before its turn kills nothing and starts nothing. Nothing
    starts a worker once `stop` has run: `_spawn` checks for it in the same
    step as it starts the process.
    """

    def __init__(self, *, timeout_s: float, warm: bool) -> None:
        self._timeout_s = timeout_s
        self._warm = warm
        self._lock: asyncio.Lock | None = None
        self._process: BaseProcess | None = None
        self._streams: tuple[StreamReader, StreamWriter] | None = None
        self._warming: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the worker process and warm it, without waiting for either.
        Without `warm`, the process starts on the first render.

        Raises:
            RuntimeError: The worker is already running.
        """
        if self._lock is not None:
            raise RuntimeError("the chart worker is already running")
        self._lock = asyncio.Lock()
        self._start_warming()

    async def stop(self) -> None:
        """Stop the worker: let it finish the render in hand and exit, for up
        to `_STOP_WAIT_S`, then kill it. A render still waiting its turn gets
        None."""
        lock, self._lock = self._lock, None
        if lock is None:
            return
        process, streams = self._process, self._streams
        if process is not None and streams is not None:
            # The worker reads the end of the stream after the render in
            # hand, and exits.
            with contextlib.suppress(OSError):
                streams[1].write_eof()
            await asyncio.to_thread(process.join, _STOP_WAIT_S)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join)
        async with lock:
            self._discard()
        self._warming = None

    async def render(self, envelope: Mapping[str, Any], *, component_id: object) -> bytes | None:
        """The envelope's chart as PNG bytes, drawn in the worker, or None.

        None is `chart_png`'s own None, a render `stop` overtook before its
        worker started, a render not done within the timeout, counted from
        the call, or a worker that died. The last two log a WARNING naming
        `component_id`. `_call` kills the worker where the render overran in
        it or was cancelled in it, and starts its replacement. An exception
        `chart_png` raised is raised here.

        Raises:
            RuntimeError: The worker is not running.
        """
        lock = self._lock
        if lock is None:
            raise RuntimeError("the chart worker is not running")
        try:
            async with asyncio.timeout(self._timeout_s):
                if self._warming is not None:
                    await asyncio.shield(self._warming)
                async with lock:
                    if self._lock is not lock:
                        return None
                    return await self._call(chart_image.chart_png, envelope)
        except TimeoutError:
            logger.warning(
                "the chart of %s took over %gs to draw, so it goes without its image",
                component_id,
                self._timeout_s,
            )
        except WorkerStoppedError:
            pass
        except WorkerDiedError:
            logger.warning(
                "the chart worker died drawing the chart of %s, so it goes without its "
                "image; starting a new worker",
                component_id,
                exc_info=True,
            )
        return None

    async def _call[T, R](self, function: Callable[[T], R], argument: T) -> R:
        """`function(argument)` run in the worker, which starts if it is not
        running. Hold `_lock` to call it. A cancelled call kills the worker,
        since the process may still be running it, and so does one that finds
        the worker dead. Either starts the replacement's warm-up, which takes
        `_lock` once the caller lets go of it. Called from a warm-up, it
        starts none, since that warm-up is still the one in progress.

        Raises:
            WorkerDiedError: The worker exited before it replied.
            WorkerStoppedError: `stop` ran before the worker could start.
        """
        if self._process is None:
            await self._spawn()
        streams = self._streams
        assert streams is not None
        try:
            done, value = await _exchange(*streams, (function, argument))
        except asyncio.CancelledError:
            self._discard()
            self._start_warming()
            raise
        except (EOFError, OSError) as error:
            self._discard()
            self._start_warming()
            raise WorkerDiedError from error
        if not done:
            raise value
        return value

    async def _spawn(self) -> None:
        """Start the worker process, with the server's end of its socket pair
        as streams. The streams connect first, since that is the one await:
        the process starts in the same step as the check that `stop` has not
        run meanwhile. A spawn that fails, is cancelled, or finds the worker
        stopped closes both sockets and leaves no process.

        Raises:
            WorkerStoppedError: `stop` ran while the streams connected.
        """
        lock = self._lock
        context = multiprocessing.get_context("spawn")
        sock, child_sock = socket.socketpair()
        with child_sock:
            try:
                streams = await asyncio.open_connection(sock=sock)
            except BaseException:
                sock.close()
                raise
            try:
                if self._lock is not lock:
                    raise WorkerStoppedError
                process = context.Process(
                    target=_serve, args=(child_sock,), name="chart-worker", daemon=True
                )
                process.start()
            except BaseException:
                streams[1].transport.abort()
                raise
        self._process, self._streams = process, streams

    def _discard(self) -> None:
        """Kill the worker, if there is one, and forget it. Its streams are
        aborted, which drops what they had left to write and closes the
        socket."""
        process, streams = self._process, self._streams
        self._process = self._streams = None
        if process is not None:
            process.kill()
            process.join()
        if streams is not None:
            streams[1].transport.abort()

    def _start_warming(self) -> None:
        """Start and warm a worker in the background, where `warm` asks for
        it, the worker is running, and none is warming already."""
        if not self._warm or self._lock is None:
            return
        if self._warming is not None and not self._warming.done():
            return
        self._warming = asyncio.get_running_loop().create_task(self._warm_up(self._lock))

    async def _warm_up(self, lock: asyncio.Lock) -> None:
        async with lock:
            if self._lock is not lock or self._process is not None:
                return
            try:
                async with asyncio.timeout(_WARM_TIMEOUT_S):
                    resident_mb, harness_loaded = await self._call(_warm, None)
            except WorkerStoppedError:
                return
            except Exception:
                logger.warning("the chart worker could not warm up", exc_info=True)
                return
        logger.info(
            "the chart worker is warm: %.0f MB resident, harness.server %s",
            resident_mb,
            "imported" if harness_loaded else "not imported",
        )


def _frame(message: object) -> bytes:
    """`message` as a frame: its pickle, behind the pickle's length."""
    body = pickle.dumps(message)
    return _FRAME_SIZE.pack(len(body)) + body


async def _exchange(reader: StreamReader, writer: StreamWriter, request: object) -> Any:
    """Send `request` to the worker and return its reply. Each step awaits
    the socket, so the loop runs on while the worker is slow to read the
    request or to write the reply.

    Raises:
        EOFError: The worker closed the socket before the reply was whole.
        OSError: The socket failed.
    """
    writer.write(_frame(request))
    await writer.drain()
    (size,) = _FRAME_SIZE.unpack(await reader.readexactly(_FRAME_SIZE.size))
    return pickle.loads(await reader.readexactly(size))


def _serve(sock: socket.socket) -> None:
    """The worker's loop: run each request and send back `(True, result)`,
    or `(False, error)` for an exception, until the stream of requests
    ends."""
    with sock, sock.makefile("rwb") as stream:
        while True:
            header = stream.read(_FRAME_SIZE.size)
            if len(header) < _FRAME_SIZE.size:
                return
            (size,) = _FRAME_SIZE.unpack(header)
            function, argument = pickle.loads(stream.read(size))
            try:
                reply = (True, function(argument))
            except Exception as error:
                reply = (False, error)
            stream.write(_frame(reply))
            stream.flush()


def _warm(_: None) -> tuple[float, bool]:
    """Warm `chart_image` in the worker, and report the worker's resident
    memory in MB and whether `harness.server` is imported."""
    chart_image.warm()
    return _resident_mb(), "harness.server" in sys.modules


def _resident_mb() -> float:
    """The resident memory of this process, in MB.

    Linux reports the current figure from `/proc/self/status`. `ru_maxrss` is
    no use there, since it survives the fork before exec and so carries the
    parent's peak into a spawned worker. On macOS the spawned process starts
    its own `ru_maxrss` count, in bytes, which is its peak rather than its
    current size.
    """
    if sys.platform == "linux":
        return _vmrss_mb(Path("/proc/self/status").read_text())
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _vmrss_mb(status: str) -> float:
    """The `VmRSS` line of a `/proc/<pid>/status` text, which is in kB, in MB.

    Raises:
        ValueError: The text has no `VmRSS` line.
    """
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 2**10
    raise ValueError("no VmRSS line in the process status")
