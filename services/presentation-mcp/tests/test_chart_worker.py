"""Chart worker tests: the render runs in a worker process, a slow render
and a dead worker leave the result without an image, a render that
overruns or is cancelled kills its worker, a warm worker follows every
one killed, the event loop runs on while a worker is slow to read a
request or to write a reply, and no worker starts once `stop` has run."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing
import os
import signal
import socket
import sys
import threading
import time
from collections.abc import AsyncIterator
from multiprocessing.context import SpawnProcess
from typing import Any, cast

import pytest
import resvg_py
from presentation_mcp import chart_image, chart_worker
from presentation_mcp.chart_worker import ChartWorker

from .stub_assembler import resolved_revision

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
async def worker() -> AsyncIterator[ChartWorker]:
    running = ChartWorker(timeout_s=5.0, warm=True)
    running.start()
    yield running
    await running.stop()


async def _worker_pid(worker: ChartWorker) -> int:
    """The pid of the worker, once it has drawn a chart."""
    assert await worker.render(resolved_revision("line_chart"), component_id="cmp_0")
    process = worker._process
    assert process is not None
    assert process.pid is not None
    return process.pid


async def _warmed_pid(worker: ChartWorker) -> int:
    """The pid of the worker the last warm-up left running, with no render
    asked of it."""
    warming = worker._warming
    assert warming is not None
    await warming
    process = worker._process
    assert process is not None
    assert process.is_alive()
    assert process.pid is not None
    return process.pid


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@contextlib.asynccontextmanager
async def _heartbeat() -> AsyncIterator[list[float]]:
    """A task that ticks on the event loop every 10 ms while the block runs,
    and the time of each tick."""
    ticks: list[float] = []

    async def _tick() -> None:
        while True:
            await asyncio.sleep(0.01)
            ticks.append(time.monotonic())

    beating = asyncio.create_task(_tick())
    try:
        yield ticks
    finally:
        beating.cancel()


def _chart_worker_records(
    caplog: pytest.LogCaptureFixture, *, level: int
) -> list[logging.LogRecord]:
    """The chart worker's records at `level`. The warm-up logs at INFO from a
    background task, and the root logger's level, which the server's lifespan
    sets to INFO in the same pytest session, decides whether `caplog` sees
    those, so a test names the level it means."""
    return [r for r in caplog.records if r.name == chart_worker.__name__ and r.levelno == level]


async def test_the_render_runs_in_the_worker_process(
    worker: ChartWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _refuse(**_: Any) -> bytes:
        raise AssertionError("the chart was rasterised on the event loop's process")

    monkeypatch.setattr(resvg_py, "svg_to_bytes", _refuse)

    png = await worker.render(resolved_revision("line_chart"), component_id="cmp_1")

    assert png is not None
    assert png.startswith(_PNG_SIGNATURE)
    assert await _worker_pid(worker) != os.getpid()


async def test_warming_logs_the_worker_s_memory_and_no_harness(
    caplog: pytest.LogCaptureFixture,
) -> None:
    running = ChartWorker(timeout_s=5.0, warm=True)
    with caplog.at_level(logging.INFO, logger=chart_worker.__name__):
        running.start()
        await _worker_pid(running)
        await running.stop()

    (record,) = _chart_worker_records(caplog, level=logging.INFO)
    assert "MB resident" in record.getMessage()
    assert "harness.server not imported" in record.getMessage()


def test_vmrss_is_read_from_the_process_status_in_megabytes() -> None:
    status = "Name:\tpython\nVmPeak:\t 9000000 kB\nVmRSS:\t  204800 kB\nThreads:\t3\n"

    assert chart_worker._vmrss_mb(status) == 200.0


def test_a_status_without_vmrss_is_refused() -> None:
    with pytest.raises(ValueError, match="VmRSS"):
        chart_worker._vmrss_mb("Name:\tpython\nVmPeak:\t 9000000 kB\n")


def test_the_resident_memory_is_this_process_s_own() -> None:
    assert 10 < chart_worker._resident_mb() < 100_000


async def test_a_chart_with_no_image_comes_back_as_none(worker: ChartWorker) -> None:
    assert await worker.render(resolved_revision("table"), component_id="cmp_1") is None


async def test_an_error_in_the_render_is_raised_and_the_worker_serves_on(
    worker: ChartWorker,
) -> None:
    with pytest.raises(TypeError):
        await worker.render(cast(Any, None), component_id="cmp_1")

    assert await worker.render(resolved_revision("line_chart"), component_id="cmp_1")


async def test_a_render_waiting_past_the_timeout_goes_without_its_image(
    worker: ChartWorker, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pid = await _worker_pid(worker)
    lock = worker._lock
    assert lock is not None
    monkeypatch.setattr(worker, "_timeout_s", 0.1)
    envelope = resolved_revision("line_chart")

    async with lock:
        with caplog.at_level(logging.WARNING, logger=chart_worker.__name__):
            png = await worker.render(envelope, component_id="cmp_1")

    assert png is None
    (record,) = _chart_worker_records(caplog, level=logging.WARNING)
    assert "cmp_1" in record.getMessage()
    assert envelope["component"]["title"]["text"] not in record.getMessage()
    assert await _worker_pid(worker) == pid


async def test_a_render_past_the_timeout_kills_its_worker_and_a_new_one_draws_the_next(
    worker: ChartWorker, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    overrun = await _worker_pid(worker)

    with monkeypatch.context() as patch:
        patch.setattr(worker, "_timeout_s", 0.5)
        patch.setattr(chart_image, "chart_png", time.sleep)
        with caplog.at_level(logging.WARNING, logger=chart_worker.__name__):
            png = await worker.render(cast(Any, 30.0), component_id="cmp_1")
    warmed = await _warmed_pid(worker)
    after = await worker.render(resolved_revision("line_chart"), component_id="cmp_2")

    assert png is None
    (record,) = _chart_worker_records(caplog, level=logging.WARNING)
    assert "cmp_1" in record.getMessage()
    assert after is not None
    assert after.startswith(_PNG_SIGNATURE)
    assert warmed != overrun
    assert await _worker_pid(worker) == warmed
    assert _gone(overrun)


async def test_a_cancelled_render_kills_its_worker_and_a_warm_one_follows(
    worker: ChartWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancelled = await _worker_pid(worker)
    lock = worker._lock
    assert lock is not None

    with monkeypatch.context() as patch:
        patch.setattr(chart_image, "chart_png", time.sleep)
        hung = asyncio.create_task(worker.render(cast(Any, 30.0), component_id="cmp_1"))
        while not lock.locked():
            await asyncio.sleep(0)
        hung.cancel()
        with pytest.raises(asyncio.CancelledError):
            await hung

    assert _gone(cancelled)
    warmed = await _warmed_pid(worker)
    assert warmed != cancelled
    assert await _worker_pid(worker) == warmed


async def test_a_reply_whose_body_comes_late_ends_at_the_timeout_and_the_loop_runs_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The worker here is a process that serves nothing, and the test holds the
    # far end of its socket: the header of a reply is there to read from the
    # start, and its body follows 20 timeouts later.
    sock, far_end = socket.socketpair()
    process = multiprocessing.get_context("spawn").Process(
        target=time.sleep, args=(60.0,), daemon=True
    )
    process.start()
    running = ChartWorker(timeout_s=0.5, warm=False)
    running.start()
    running._process = process
    running._streams = await asyncio.open_connection(sock=sock)
    reply = chart_worker._frame((True, _PNG_SIGNATURE))
    header, body = reply[:8], reply[8:]
    far_end.sendall(header)
    late_body = threading.Timer(10.0, far_end.sendall, (body,))
    late_body.start()

    try:
        with caplog.at_level(logging.WARNING, logger=chart_worker.__name__):
            async with _heartbeat() as ticks:
                png = await running.render(resolved_revision("line_chart"), component_id="cmp_1")
    finally:
        late_body.cancel()
        late_body.join()
        far_end.close()
        await running.stop()

    assert png is None
    assert len(ticks) >= 5
    (record,) = _chart_worker_records(caplog, level=logging.WARNING)
    assert "took over" in record.getMessage()
    assert not process.is_alive()


async def test_a_request_the_worker_does_not_read_ends_at_the_timeout_and_the_loop_runs_on(
    worker: ChartWorker, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A stopped worker reads nothing, so a request larger than the socket's
    # buffer cannot all be written. The timer ends a write that outlasts 20
    # timeouts, by killing the worker.
    stopped = await _worker_pid(worker)
    os.kill(stopped, signal.SIGSTOP)
    monkeypatch.setattr(worker, "_timeout_s", 0.5)
    unblock = threading.Timer(10.0, os.kill, (stopped, signal.SIGKILL))
    unblock.start()

    try:
        with caplog.at_level(logging.WARNING, logger=chart_worker.__name__):
            async with _heartbeat() as ticks:
                png = await worker.render(cast(Any, bytes(2**23)), component_id="cmp_1")
    finally:
        unblock.cancel()
        unblock.join()

    assert png is None
    assert len(ticks) >= 5
    (record,) = _chart_worker_records(caplog, level=logging.WARNING)
    assert "took over" in record.getMessage()
    assert _gone(stopped)
    warmed = await _warmed_pid(worker)
    assert warmed != stopped
    assert await _worker_pid(worker) == warmed


async def test_a_render_cancelled_while_it_waits_its_turn_keeps_the_worker(
    worker: ChartWorker,
) -> None:
    pid = await _worker_pid(worker)
    lock = worker._lock
    assert lock is not None
    warming = worker._warming

    async with lock:
        queued = asyncio.create_task(
            worker.render(resolved_revision("line_chart"), component_id="cmp_1")
        )
        await asyncio.sleep(0.05)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued

    assert worker._warming is warming
    assert await _worker_pid(worker) == pid


async def test_a_warm_up_whose_worker_dies_starts_no_second_warm_up(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # `sys.exit` raises `SystemExit` in the worker, which ends its process
    # before it replies.
    monkeypatch.setattr(chart_worker, "_warm", sys.exit)
    running = ChartWorker(timeout_s=5.0, warm=True)

    with caplog.at_level(logging.WARNING, logger=chart_worker.__name__):
        running.start()
        warming = running._warming
        assert warming is not None
        await warming
        await asyncio.sleep(0)
    try:
        assert running._warming is warming
        assert running._process is None
        (record,) = _chart_worker_records(caplog, level=logging.WARNING)
        assert "could not warm up" in record.getMessage()
        assert await running.render(resolved_revision("line_chart"), component_id="cmp_1")
    finally:
        await running.stop()


async def test_a_dead_worker_leaves_the_render_without_an_image_and_is_replaced(
    worker: ChartWorker, caplog: pytest.LogCaptureFixture
) -> None:
    dead = await _worker_pid(worker)
    os.kill(dead, signal.SIGKILL)
    envelope = resolved_revision("line_chart")

    with caplog.at_level(logging.WARNING, logger=chart_worker.__name__):
        broken = await worker.render(envelope, component_id="cmp_1")
    warmed = await _warmed_pid(worker)
    after = await worker.render(envelope, component_id="cmp_2")

    assert broken is None
    (record,) = _chart_worker_records(caplog, level=logging.WARNING)
    assert "cmp_1" in record.getMessage()
    assert after is not None
    assert after.startswith(_PNG_SIGNATURE)
    assert warmed != dead
    assert await _worker_pid(worker) == warmed


async def test_a_render_before_start_raises() -> None:
    with pytest.raises(RuntimeError, match="not running"):
        await ChartWorker(timeout_s=1.0, warm=False).render(
            resolved_revision("line_chart"), component_id="c"
        )


async def test_stop_waits_for_the_worker_to_exit(worker: ChartWorker) -> None:
    pid = await _worker_pid(worker)

    await worker.stop()

    assert _gone(pid)
    assert worker._process is None


async def test_stop_during_a_hung_render_kills_the_worker_within_its_bound(
    worker: ChartWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid = await _worker_pid(worker)
    lock = worker._lock
    assert lock is not None
    monkeypatch.setattr(chart_worker, "_STOP_WAIT_S", 0.5)
    monkeypatch.setattr(chart_image, "chart_png", time.sleep)
    hung = asyncio.create_task(worker.render(cast(Any, 30.0), component_id="cmp_1"))
    while not lock.locked():
        await asyncio.sleep(0)

    started = time.monotonic()
    await worker.stop()

    assert time.monotonic() - started < 2.0
    assert await hung is None
    assert _gone(pid)
    assert worker._process is None


@pytest.mark.parametrize("warm", [False, True], ids=["a render", "the warm-up"])
async def test_stop_while_a_worker_is_starting_starts_none_and_keeps_its_bound(
    warm: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # `_spawn` awaits the connection of its streams before it starts the
    # process, and `stop` runs inside that await.
    connecting = asyncio.Event()
    connect = asyncio.open_connection
    start = SpawnProcess.start
    started: list[str] = []

    async def _connect(**kwargs: Any) -> Any:
        connecting.set()
        return await connect(**kwargs)

    def _start(process: SpawnProcess) -> None:
        started.append(process.name)
        start(process)

    monkeypatch.setattr(asyncio, "open_connection", _connect)
    monkeypatch.setattr(SpawnProcess, "start", _start)
    monkeypatch.setattr(chart_worker, "_STOP_WAIT_S", 0.5)
    monkeypatch.setattr(chart_image, "chart_png", time.sleep)
    running = ChartWorker(timeout_s=5.0, warm=warm)
    running.start()
    render = asyncio.create_task(running.render(cast(Any, 30.0), component_id="cmp_1"))
    await connecting.wait()
    assert running._process is None

    began = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=chart_worker.__name__):
        await running.stop()
    took = time.monotonic() - began

    assert took < 2.0
    assert await render is None
    assert started == []
    assert running._process is None
    assert running._streams is None
    assert not [one for one in multiprocessing.active_children() if one.name == "chart-worker"]
    assert _chart_worker_records(caplog, level=logging.WARNING) == []
