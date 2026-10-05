"""The frame rows this process's tool calls hold in memory at once.

A call that decodes, computes over or encodes a frame allocates in proportion to the
frame's rows, so rows are what the budget counts. data-mcp, analysis-mcp and
presentation-mcp each claim rows at the point a call knows its row count and has not
yet made its large allocation, and release them when that work ends:

- data-mcp: `frame_materialisation`, after the store opens a held pair and before it
  reads it, for the read that confirms a cache hit; and after the artifact footer gives
  the row count, for the parse and the store write.
- analysis-mcp: `frame_ref_resolution.resolve_input_frames`, once the store has opened
  every input (`MaterialisationStore.open`) and before it reads any, for all the
  inputs' rows in one claim, held for the whole verb call (`holds_input_rows`).
- presentation-mcp: `frame_publish.publish_body_for`, after the store opens the frame
  and before it reads it, for the read, the publish body's encoding and the publish
  request that sends it.

A store's `open` reads no byte of the frame itself (`MaterialisationStore.open`), so
the download, its checks and the decode all run under the claim. A frame stored
without a recorded row count is claimed at the service's `max_frame_rows`, and its
read refuses a frame holding more rows than the claim before decoding it.

The harness mounts all three in one process, so one budget serves them all:
`row_budget()` builds it on first use from `BaseServiceSettings.frame_rows_in_flight_max`,
which every service's settings inherit from the same env var. A service run on its own
builds its own. `FrameServiceSettings` refuses a budget smaller than the largest claim,
a two-input verb's two `max_frame_rows` inputs, so every legal call fits an idle budget.

A claim that does not fit is refused at once and never queued: a queued call would spend
its own deadline waiting, as the combined server's in-flight cap
(`harness.combined_mcp._InFlightLimit`) argues. Each service turns
`RowBudgetExhausted` into its `service_unavailable` refusal with
`ROW_BUDGET_RETRY_AFTER_S`.

Work a claim covers can run in a worker thread that outlives the coroutine awaiting it:
a store read or write cut off by its deadline, or a verb call cancelled at the call
limit.
`RowClaim.hold_through` runs such work so the claim gives its rows back only once the
work and the claim's `with` block have both ended.

Every count changes under one lock, so claims from threads and from concurrent
coroutines on one event loop see one consistent count.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable
from typing import Any, Self

from common.settings import BaseServiceSettings

logger = logging.getLogger(__name__)

# The wait a caller is told before repeating a call the budget refused. At 250,000 rows
# the heaviest budgeted call's work runs in under a second, and a join in a few
# (`scripts/bench_frames.py`).
ROW_BUDGET_RETRY_AFTER_S = 2.0


class RowBudgetExhausted(Exception):  # noqa: N818 -- named for the state it reports
    """A claim for `requested` more rows did not fit: `free` rows were left of
    `max_rows`."""

    def __init__(self, *, requested: int, free: int, max_rows: int) -> None:
        self.requested = requested
        self.free = free
        self.max_rows = max_rows
        super().__init__(
            f"{requested} frame rows requested, {free} of {max_rows} free in this process"
        )


class RowBudget:
    """A count of frame rows in flight, bounded by `max_rows`."""

    def __init__(self, *, max_rows: int) -> None:
        if max_rows < 1:
            raise ValueError(f"max_rows must be at least 1, got {max_rows}")
        self.max_rows = max_rows
        self._in_flight = 0
        self._high_water = 0
        self._lock = threading.Lock()

    @property
    def in_flight(self) -> int:
        """Rows every open claim holds together."""
        return self._in_flight

    @property
    def high_water(self) -> int:
        """The most rows held at once since this budget was built.
        `scripts/bench_frames.py` divides a call's memory by it."""
        return self._high_water

    def claim(self, rows: int) -> RowClaim:
        """A claim on `rows`, released when its `with` block exits, or
        `RowBudgetExhausted` when they do not fit. A claim on zero rows always fits."""
        claim = RowClaim(self)
        claim.extend(rows)
        return claim

    def _take(self, rows: int) -> None:
        """Add `rows` to the count, or raise `RowBudgetExhausted` when they do not fit."""
        if rows < 0:
            raise ValueError(f"rows must not be negative, got {rows}")
        with self._lock:
            if self._in_flight + rows > self.max_rows:
                raise RowBudgetExhausted(
                    requested=rows, free=self.max_rows - self._in_flight, max_rows=self.max_rows
                )
            self._in_flight += rows
            self._high_water = max(self._high_water, self._in_flight)

    def _give_back(self, rows: int) -> None:
        with self._lock:
            self._in_flight -= rows


class RowClaim:
    """Rows one call holds against a `RowBudget`, released on exit from its `with`
    block, whether the block returns or raises, or once the last `hold_through` work
    ends when that is later."""

    def __init__(self, budget: RowBudget) -> None:
        self._budget = budget
        self._rows = 0
        self._released = False
        self._work: set[asyncio.Future[Any]] = set()
        # Work whose awaiting coroutine stopped before it ended, by the name it logs under.
        self._abandoned: dict[asyncio.Future[Any], str] = {}

    @property
    def rows(self) -> int:
        return self._rows

    def extend(self, rows: int) -> None:
        """Hold `rows` more, or raise `RowBudgetExhausted` and hold what was held.
        A call claims everything it will decode in one `extend`: a call that holds
        rows while it waits for more can hold them while every such call is refused."""
        self._budget._take(rows)
        self._rows += rows

    async def hold_through[T](self, work: Awaitable[T]) -> T:
        """Await `work`, shielded from cancellation, and keep this claim's rows held
        until it ends. A deadline or a cancellation that stops the awaiting coroutine
        leaves `work` running, as it leaves a worker thread running, and a `release`
        in the meantime gives the rows back only when `work` ends. Work that then
        fails has no caller left to raise to, so its exception is logged."""
        name = getattr(work, "__qualname__", repr(work))
        task = asyncio.ensure_future(work)
        self._work.add(task)
        task.add_done_callback(self._work_ended)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                self._abandoned[task] = name
            raise

    def _work_ended(self, task: asyncio.Future[Any]) -> None:
        self._work.discard(task)
        name = self._abandoned.pop(task, None)
        if name is not None and not task.cancelled() and (exc := task.exception()) is not None:
            logger.warning("%s failed after the call awaiting it had stopped", name, exc_info=exc)
        if self._released and not self._work:
            self._give_back()

    def release(self) -> None:
        """Give every held row back, now or once the last `hold_through` work ends.
        A second release gives back nothing."""
        self._released = True
        if not self._work:
            self._give_back()

    def _give_back(self) -> None:
        rows, self._rows = self._rows, 0
        self._budget._give_back(rows)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


_budget: RowBudget | None = None
_budget_lock = threading.Lock()


def row_budget() -> RowBudget:
    """The process-wide budget, built on first use from
    `BaseServiceSettings().frame_rows_in_flight_max`."""
    global _budget  # noqa: PLW0603 -- the one process-wide budget
    with _budget_lock:
        if _budget is None:
            _budget = RowBudget(max_rows=BaseServiceSettings().frame_rows_in_flight_max)
        return _budget


def set_row_budget(budget: RowBudget | None) -> None:
    """Install `budget` as the process-wide one, or clear it so the next
    `row_budget()` builds it again from settings."""
    global _budget  # noqa: PLW0603 -- the one process-wide budget
    with _budget_lock:
        _budget = budget
