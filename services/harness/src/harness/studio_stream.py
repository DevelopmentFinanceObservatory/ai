"""A Studio stream's owner: one per claimed segment.

The route that claims a segment hands its work to `open_stream` as its last
step and returns the response that call gives (`harness.studio_relay`).
From then on three tasks serve the stream. The work's drive (`Work.drive`)
runs the segment's calls and hands each event it makes to the owner over one
queue. The owner turns the events into frames and puts them on the channel,
which the response drains. Beside the queue and the channel they share the
client-gone flag, which the response sets and the owner reads, and the
segment's `renew_now` event (`harness.studio_state.renew_lease_now`).

The owner (`own_segment`) is a coroutine that never yields and that no
request cancels. For its segment it alone:

- translates each of the drive's events into frames (`Work.frames`), in the
  order the drive emitted them, which numbers the frames and writes the
  conversation state they change, and reads how the work ended once the
  drive has returned (`Work.end`). A Stop and a lost lease reach the owner
  only as that outcome (`harness.turn.loop.Outcome`);
- puts each frame on the channel, except the work's ending frame (a pause
  frame, `done`, or a stream-fatal `error`), which it holds;
- saves the pending interrupts once `plan.approved` has gone out, so a retry
  of the same answer does not send it again. The save reads them as the
  frames sent so far leave them, before its first await, and writes nothing
  else. A save that finds the lease lost ends the work `lost`;
- reads the client-gone flag in one place, while it waits for the drive's
  next event. Once the flag is set it cancels the drive with the turn's
  `StopScope` unmarked, so the cursor stays as last written for a later
  segment to carry on, waits for it to stop, and translates the events it
  emitted before it stopped;
- makes the segment's one closing write (`StudioStore.close_segment`), which
  saves the conversation, clears the lease and, when a Stop is live for the
  open turn, closes it `cancelled`, in one transaction. A segment whose
  work ended `lost` or `abandoned` makes none: the lease is another's;
- decides how the stream ends from the work's end and that write's result,
  once (`ending`);
- forgets this instance's copy of the conversation when the closing write
  did not commit as far as it knows, or was not made, so the next request
  reads the row (`StudioStore.evict`), and then frees the conversation's
  local lock;
- puts the closing frames on the channel: a `failed` line for every Activity
  line the stream left open when it ends on an `error` frame, then the
  ending frame, then the channel's end. The ending frame goes out after the
  closing write has returned, as the stream's last frame, so a client never
  reads a pause or `done` the store does not hold;
- opens, enters, ends and exports the segment's telemetry span, whose status
  is the ending's, with `dfo.saved` and `dfo.client_gone` beside it.

The closing frames are numbered from the conversation's `seq` as the owner
read it before the closing write, or from the work's ending frame's own
number when nothing went out after that frame, and that write stores `seq`
past every number they can take. A later segment numbers from the block its
claim reserves (`harness.studio_store.SEQ_BLOCK`). After the closing write
the owner changes no conversation state.

The drive's tool calls still meet the conversation in two ways. A tool
reads the live canvas and digest (`harness.studio_state.make_canvas_cards`
and its siblings). The `remove_cards` tool writes the canvas through
`StudioStore.change_canvas`, since the model reads its result in the same
model call: the store's write lock orders that write with the owner's, it
commits only the fields its change made, and those reach the live state
only once it commits. The recorders the drive's calls write
(`harness.studio_translate._RecordingExecutor` and `_RecordingSurface`)
are append-only queues: the drive marks each event with how many records
they held as it was emitted (`harness.studio_translate._EventRecords`), and
the owner alone takes them, a write token among them.

Past a graceful-shutdown timeout the server stops waiting for responses, and
the event loop's teardown would cancel an owner still in its closing write.
`LiveOwners` keeps the owners running on this instance, so the app's
lifespan shutdown waits for them (`LiveOwners.wait`) before it closes the
database.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Coroutine, Hashable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Protocol

from common.telemetry import (
    EventLevel,
    end_turn_span,
    flush_telemetry,
    start_turn_span,
    use_turn_span,
)
from pydantic import BaseModel

from harness.sentences import failure_sentence
from harness.studio_relay import Channel, StudioStream
from harness.studio_state import (
    ConversationState,
    DatabaseUnavailable,
    SegmentLost,
    _recorded_reply,
)
from harness.studio_store import SegmentClose, Snapshot
from harness.studio_translate import SEGMENT_LOST_DETAIL, STREAM_RETRY_AFTER_S
from harness.turn.loop import Outcome
from harness.wire import (
    ActivityStep,
    Done,
    ElicitationRequest,
    ErrorCode,
    ErrorFrame,
    PlanProposed,
    RetryMode,
    StepStatus,
    ToolError,
)

logger = logging.getLogger(__name__)

# How long the app's lifespan shutdown waits for the owners still running:
# inside Cloud Run's 10 s between SIGTERM and SIGKILL, with room for the rest
# of the shutdown. An owner's closing write is bounded at 5 s.
SHUTDOWN_WAIT_S = 7.0


class Finish(StrEnum):
    """How a segment's work ended: the turn loop's outcome
    (`harness.turn.loop.Outcome`), or `left` when the owner cancelled the
    work because the client had left."""

    PAUSED = Outcome.PAUSED.value
    DONE = Outcome.DONE.value
    FAILED = Outcome.FAILED.value
    CANCELLED = Outcome.CANCELLED.value
    UNCLOSED = Outcome.UNCLOSED.value
    LOST = Outcome.LOST.value
    ABANDONED = Outcome.ABANDONED.value
    LEFT = "left"


class Status(StrEnum):
    """How a stream ended, as its segment's span states it: the turn
    `paused`, was `done`, `failed` on a node's error, was `cancelled` by a
    Stop, or failed with its close unwritten (`unclosed`); the client
    `left`; the lease was `lost`; or the closing write did not commit, so a
    pause or `done` went `unsaved`."""

    DONE = "done"
    PAUSED = "paused"
    FAILED = "failed"
    UNCLOSED = "unclosed"
    CANCELLED = "cancelled"
    LEFT = "left"
    LOST = "lost"
    UNSAVED = "unsaved"


# The level a segment's span ends at, by its status.
_LEVELS: Final[dict[Status, EventLevel]] = {
    Status.DONE: "DEFAULT",
    Status.PAUSED: "DEFAULT",
    Status.FAILED: "ERROR",
    Status.UNCLOSED: "ERROR",
    Status.UNSAVED: "ERROR",
    Status.CANCELLED: "WARNING",
    Status.LEFT: "WARNING",
    Status.LOST: "WARNING",
}

# The frames that end a stream. Every `error` frame is stream-fatal.
type EndingFrame = PlanProposed | ElicitationRequest | Done | ErrorFrame
_ENDING_FRAMES: Final = (PlanProposed, ElicitationRequest, Done, ErrorFrame)

# The ending frame each finish needs from its work.
_FINISH_FRAMES: Final[dict[Finish, tuple[type[BaseModel], ...]]] = {
    Finish.PAUSED: (PlanProposed, ElicitationRequest),
    Finish.DONE: (Done,),
    Finish.FAILED: (ErrorFrame,),
    Finish.UNCLOSED: (ErrorFrame,),
}


@dataclass(frozen=True)
class WorkEnd:
    """How the work ended, and the ending frame it sent (`None` when it
    sent none)."""

    finish: Finish
    frame: EndingFrame | None = None


@dataclass(frozen=True)
class Ending:
    """How a stream ends: its ending frame (`None` for a client that left),
    its span's status, and whether the closing write committed (`saved`).
    A frame `ending` builds carries `seq` 0; the owner numbers it."""

    frame: EndingFrame | None
    status: Status
    saved: bool


def _cancelled_frame() -> ErrorFrame:
    """The frame that ends a stream on a Stop."""
    return ErrorFrame(
        event="error",
        seq=0,
        error=ToolError(
            error=ErrorCode.CANCELLED,
            retry=RetryMode.DO_NOT_RETRY,
            detail=failure_sentence(ErrorCode.CANCELLED, RetryMode.DO_NOT_RETRY),
        ),
    )


def _transient_frame() -> ErrorFrame:
    """The frame that ends a stream whose turn is left as stored: the lease
    is another instance's, or the closing write did not commit. The client
    retries after a delay."""
    return ErrorFrame(
        event="error",
        seq=0,
        error=ToolError(
            error=ErrorCode.TRANSIENT_UNAVAILABLE,
            retry=RetryMode.RETRY_AFTER_DELAY,
            detail=SEGMENT_LOST_DETAIL,
            retry_after_seconds=STREAM_RETRY_AFTER_S,
        ),
    )


class _Frame(StrEnum):
    """Which frame an ending sends: the work's own, a new `cancelled` or
    transient one, or none."""

    HELD = "held"
    CANCELLED = "cancelled"
    TRANSIENT = "transient"
    NONE = "none"


def _row(
    saved: tuple[_Frame, Status],
    stopped: tuple[_Frame, Status],
    lost: tuple[_Frame, Status],
    failed: tuple[_Frame, Status],
) -> dict[SegmentClose, tuple[_Frame, Status]]:
    return {
        SegmentClose.SAVED: saved,
        SegmentClose.STOPPED: stopped,
        SegmentClose.LOST: lost,
        SegmentClose.FAILED: failed,
    }


# `ending`'s table, by the work's finish and the closing write's result.
_ENDINGS: Final[dict[tuple[Finish, SegmentClose], tuple[_Frame, Status]]] = {
    (finish, close): cell
    for finish, row in {
        Finish.PAUSED: _row(
            (_Frame.HELD, Status.PAUSED),
            (_Frame.CANCELLED, Status.CANCELLED),
            (_Frame.TRANSIENT, Status.LOST),
            (_Frame.TRANSIENT, Status.UNSAVED),
        ),
        Finish.DONE: _row(
            (_Frame.HELD, Status.DONE),
            (_Frame.HELD, Status.DONE),
            (_Frame.TRANSIENT, Status.LOST),
            (_Frame.TRANSIENT, Status.UNSAVED),
        ),
        Finish.FAILED: _row(
            (_Frame.HELD, Status.FAILED),
            (_Frame.HELD, Status.FAILED),
            (_Frame.HELD, Status.FAILED),
            (_Frame.HELD, Status.FAILED),
        ),
        Finish.CANCELLED: _row(
            (_Frame.CANCELLED, Status.CANCELLED),
            (_Frame.CANCELLED, Status.CANCELLED),
            (_Frame.CANCELLED, Status.CANCELLED),
            (_Frame.CANCELLED, Status.CANCELLED),
        ),
        Finish.UNCLOSED: _row(
            (_Frame.HELD, Status.UNCLOSED),
            (_Frame.CANCELLED, Status.CANCELLED),
            (_Frame.TRANSIENT, Status.LOST),
            (_Frame.HELD, Status.UNCLOSED),
        ),
        Finish.LEFT: _row(
            (_Frame.NONE, Status.LEFT),
            (_Frame.NONE, Status.CANCELLED),
            (_Frame.NONE, Status.LOST),
            (_Frame.NONE, Status.LEFT),
        ),
    }.items()
    for close, cell in row.items()
}


def ending(work: WorkEnd, close: SegmentClose | None) -> Ending:
    """How a stream ends, from how its work ended and how its closing write
    ended (`None` for work that ended `lost` or `abandoned`, which makes no
    closing write).

    | Work        | `saved`         | `stopped`   | `lost`             | `failed`              |
    |-------------|-----------------|-------------|--------------------|-----------------------|
    | `paused`    | pause, paused   | cancelled   | transient, lost    | transient, unsaved    |
    | `done`      | done, done      | as `saved`  | transient, lost    | transient, unsaved    |
    | `failed`    | error, failed   | as `saved`  | error, failed      | error, failed         |
    | `cancelled` | cancelled       | cancelled   | cancelled          | cancelled             |
    | `unclosed`  | error, unclosed | cancelled   | transient, lost    | error, unclosed       |
    | `left`      | none, left      | none,       | none, lost         | none, left            |
    |             |                 | cancelled   |                    |                       |

    `lost` and `abandoned` work ends on the transient frame, `lost`. "pause",
    "done" and "error" are the work's own ending frame. A closing write that
    found a Stop live for the open turn (`stopped`) cannot meet a turn that
    ended `done` or `failed`, since a closed turn voids its Stop; it reads
    as `saved` there. A `cancelled` frame goes out whether or not the close
    committed: the Stop stays live, and the next claimant closes the turn.

    Raises:
        ValueError: `close` is `None` for work that makes a closing write.
    """
    if work.finish in (Finish.LOST, Finish.ABANDONED):
        return Ending(_transient_frame(), Status.LOST, saved=False)
    if close is None:
        raise ValueError(f"work that ended {work.finish} makes a closing write")
    frame, status = _ENDINGS[work.finish, close]
    built = {
        _Frame.HELD: work.frame,
        _Frame.CANCELLED: _cancelled_frame(),
        _Frame.TRANSIENT: _transient_frame(),
        _Frame.NONE: None,
    }[frame]
    return Ending(built, status, saved=close in (SegmentClose.SAVED, SegmentClose.STOPPED))


class Emit[E](Protocol):
    """How a drive hands the owner an event. An event emitted with a `key`
    replaces one with the same key that still waits, so a stream that falls
    behind its drive translates only the latest of them."""

    def __call__(self, event: E, /, *, key: Hashable | None = None) -> None: ...


class Work[E, R](Protocol):
    """A segment's work, as the route that claimed the segment builds it, in
    two halves.

    `drive` runs in a task of its own: the turn loop's segment, or the Edit
    Card's calls, which hands each event it makes to `emit` and writes no
    conversation state of its own beyond a card write whose result its
    model reads (module docstring). The owner's task runs the rest: it
    starts the work (`begin`) before the drive runs, translates each event
    into wire frames (`frames`), in order, and once the drive has returned,
    and every event it emitted is translated, reads how the work ended and
    the frames that close it (`end`)."""

    def begin(self) -> None:
        """Write the conversation state the work needs before its drive
        runs."""
        ...

    async def drive(self, emit: Emit[E]) -> R:
        """Run the segment's calls, `emit` each event, and return what `end`
        reads. Cancelling the task that runs it stops the segment with the
        turn's `StopScope` unmarked."""
        ...

    async def frames(self, event: E) -> list[BaseModel]:
        """The wire frames of `event`, numbered, with the conversation state
        it changes written."""
        ...

    async def end(self, result: R) -> tuple[Outcome, list[BaseModel]]:
        """How the segment ended, from `result`, what `drive` returned, and
        the frames that close it. A turn that pauses, ends, or fails sends its
        ending frame (a pause frame, `done`, or a stream-fatal `error`), in
        these frames or in earlier ones, and ends `paused`, `done`, `failed`
        or `unclosed`. A Stop that closed the turn ends `cancelled`, and a
        lost lease or a turn another instance abandoned `lost` or
        `abandoned`, with no ending frame."""
        ...

    def fatal(self, error: BaseException) -> ErrorFrame:
        """The stream-fatal frame for work that raised `error`, or that ended
        with no ending frame its outcome needs; it records the turn failed
        on it."""
        ...

    def close(self) -> None:
        """Let go of what the work kept for its segment, however it ended."""
        ...


class SegmentStore(Protocol):
    """The writes an owner makes (`harness.studio_store.StudioStore`)."""

    def snapshot(self, principal: str, conversation_id: str) -> Snapshot: ...

    async def checkpoint(
        self, principal: str, conversation_id: str, segment_id: str, snapshot: Snapshot
    ) -> None: ...

    async def close_segment(
        self, principal: str, conversation_id: str, segment_id: str
    ) -> SegmentClose: ...

    def evict(self, principal: str, conversation_id: str) -> None: ...


class _OpenLines:
    """The Activity lines a stream opened and has not closed.

    Studio shows a line as pending while its latest frame carries no
    `duration_ms`: a `running` or `pending` step report. A later frame on
    the same `step_id` with a `duration_ms` closes it. A stream that ends on
    an `error` frame sends no such frame, so the owner closes each open line
    itself.
    """

    def __init__(self) -> None:
        self._open: dict[str, tuple[ActivityStep, float]] = {}

    def __len__(self) -> int:
        return len(self._open)

    def observe(self, frame: BaseModel) -> None:
        if not isinstance(frame, ActivityStep):
            return
        if frame.duration_ms is None and frame.status in (StepStatus.RUNNING, StepStatus.PENDING):
            self._open.setdefault(frame.step_id, (frame, time.monotonic()))
        else:
            self._open.pop(frame.step_id, None)

    def close(self, *, detail: str, seq: Iterator[int]) -> list[ActivityStep]:
        """A `failed` frame for every open line, timed from its first frame."""
        now = time.monotonic()
        return [
            ActivityStep(
                event="activity.step",
                seq=next(seq),
                step_id=step_id,
                label=opened.label,
                status=StepStatus.FAILED,
                tool=opened.tool,
                detail=detail,
                duration_ms=int((now - started) * 1000),
            )
            for step_id, (opened, started) in self._open.items()
        ]


class _Events[E]:
    """The one queue from a drive to its owner: the events the drive
    emitted and the owner has not taken, in order, by key (`Emit`). An
    event emitted with no key takes a key of its own."""

    def __init__(self) -> None:
        self._waiting: OrderedDict[Hashable, E] = OrderedDict()
        self._arrived = asyncio.Event()

    def __bool__(self) -> bool:
        return bool(self._waiting)

    def put(self, event: E, /, *, key: Hashable | None = None) -> None:
        # A waiting event under `key` is replaced in its place.
        self._waiting[object() if key is None else key] = event
        self._arrived.set()

    def take(self) -> E:
        """The oldest event waiting; there must be one."""
        return self._waiting.popitem(last=False)[1]

    def arrived(self) -> Coroutine[Any, Any, bool]:
        """Wait until an event arrives after the last `take` that emptied
        the queue."""
        if not self._waiting:
            self._arrived.clear()
        return self._arrived.wait()


class _Gone:
    """The client left while the drive runs."""


class _Ended:
    """The drive ended and every event it emitted is taken."""


_GONE: Final = _Gone()
_ENDED: Final = _Ended()


class _Owner[E, R]:
    """`own_segment`'s state for one segment."""

    def __init__(
        self,
        work: Work[E, R],
        *,
        store: SegmentStore,
        conversation: ConversationState,
        principal: str,
        conversation_id: str,
        segment_id: str,
        channel: Channel,
        gone: asyncio.Event,
    ) -> None:
        self._work = work
        self._store = store
        self._conversation = conversation
        self._principal = principal
        self._conversation_id = conversation_id
        self._segment_id = segment_id
        self._channel = channel
        self._gone = gone
        self._events: _Events[E] = _Events()
        self._task: asyncio.Task[R] | None = None
        self._held: EndingFrame | None = None
        # Whether nothing went on the channel since the work sent the held
        # frame, so the number the work gave it is still unsent.
        self._held_last = False
        self._lines = _OpenLines()
        # The first number the closing frames take.
        self._first_seq = conversation.seq

    async def run(self) -> Ending:
        """Run the segment to its end and close it; return how it ended."""
        try:
            finish = await self._work_to_end()
            close = await self._close(finish)
            return ending(WorkEnd(finish, self._held), close)
        except Exception:
            # A fault in the owner itself: no frame of the work's can be
            # trusted to end the stream, and the closing write's outcome is
            # unknown. The transient frame takes the next unsent number.
            logger.exception(
                "studio stream: the owner of conversation %s segment %s failed",
                self._conversation_id,
                self._segment_id,
            )
            self._first_seq = self._conversation.seq
            if self._task is not None and not self._task.done():
                await self._stop(self._task)
            return Ending(_transient_frame(), Status.UNCLOSED, saved=False)
        finally:
            self._work.close()

    async def _work_to_end(self) -> Finish:
        """Translate the drive's events until it ends, the client leaves, or
        the save after `plan.approved` or a card write finds the lease lost;
        then how the work ended. When the owner cuts the drive off, it
        cancels it and waits for it. A client that left still has the events
        the drive emitted translated, unsaved, so the closing write stores
        the state they leave; a lost lease drops them, since no write of
        this segment's can land."""
        try:
            self._work.begin()
        except Exception as error:
            return self._failed(error)
        self._task = task = asyncio.create_task(self._work.drive(self._events.put))
        cut: Finish | None = None
        try:
            while cut is None:
                event = await self._next()
                if isinstance(event, _Ended):
                    return await self._ended(task)
                if isinstance(event, _Gone):
                    cut = Finish.LEFT
                elif await self._take(await self._work.frames(event)):
                    cut = Finish.LOST
            await self._stop(task)
            if cut is Finish.LOST:
                return cut
            await self._translate_left()
            return cut if task.cancelled() else await self._ended(task)
        except (SegmentLost, DatabaseUnavailable) as lost:
            # A card write fenced out by another instance's takeover, or one
            # that could not reach the database before the lease's deadline
            # (`harness.studio_store.StudioStore.change_canvas`).
            logger.warning(
                "studio stream: a card write ended conversation %s segment %s (trace_id=%s): %s",
                self._conversation_id,
                self._segment_id,
                self._conversation.trace_id,
                lost,
            )
            await self._stop(task)
            return Finish.LOST
        except Exception as error:
            await self._stop(task)
            return self._failed(error)

    @staticmethod
    async def _stop(task: asyncio.Task[Any]) -> None:
        task.cancel()
        await asyncio.wait({task})

    async def _next(self) -> E | _Gone | _Ended:
        """The drive's next event, `_GONE` once the client has left while
        the drive runs, or `_ENDED` once the drive has ended and every event
        it emitted is taken. The one place the owner reads the client-gone
        flag."""
        assert self._task is not None
        while True:
            if self._gone.is_set() and not self._task.done():
                return _GONE
            if self._events:
                return self._events.take()
            if self._task.done():
                return _ENDED
            arrived = asyncio.ensure_future(self._events.arrived())
            left = asyncio.ensure_future(self._gone.wait())
            try:
                await asyncio.wait({arrived, left, self._task}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                arrived.cancel()
                left.cancel()

    async def _translate_left(self) -> None:
        """Translate, unsaved, the events the drive emitted before the owner
        stopped it."""
        while self._events:
            await self._take(await self._work.frames(self._events.take()), saving=False)

    async def _take(self, frames: list[BaseModel], *, saving: bool = True) -> bool:
        """Hold the work's ending frame and put every other frame on the
        channel, in order. After `plan.approved`, when `saving`, save the
        conversation; returns whether that save found the lease lost. The
        frames after it are still taken, unsaved."""
        lost = False
        for frame in frames:
            if isinstance(frame, _ENDING_FRAMES):
                if self._held is None:
                    self._held = frame
                    self._held_last = True
                else:
                    logger.error(
                        "studio stream: conversation %s segment %s sent a second ending frame "
                        "(%s); it is dropped",
                        self._conversation_id,
                        self._segment_id,
                        frame.event,
                    )
                continue
            self._lines.observe(frame)
            self._channel.put_nowait(frame)
            self._held_last = False
            if saving and not lost and getattr(frame, "event", None) == "plan.approved":
                lost = await self._approved()
        return lost

    async def _approved(self) -> bool:
        """Record that `plan.approved` went out and save it, so a retry of
        the same answer after this segment dies does not send it again (a
        crash before the save sends it twice, which beats never). Returns
        whether the save found the lease lost; any other failure is logged
        and the stream goes on.

        The owner alone translates, so the pending interrupts the snapshot
        reads, with no await since the frame went on the channel, are those
        of the frames sent so far. The save writes those and nothing else
        (`harness.studio_store.StudioStore.checkpoint`). The drive runs on
        while the save waits, and its events wait for the owner."""
        self._conversation.mark_approved_sent(self._segment_id)
        try:
            snapshot = self._store.snapshot(self._principal, self._conversation_id)
            await self._store.checkpoint(
                self._principal, self._conversation_id, self._segment_id, snapshot
            )
        except (SegmentLost, DatabaseUnavailable):
            logger.warning(
                "studio stream: conversation %s segment %s lost its lease at the save after "
                "plan.approved",
                self._conversation_id,
                self._segment_id,
            )
            return True
        except Exception:
            logger.exception(
                "studio stream: conversation %s segment %s's save after plan.approved failed",
                self._conversation_id,
                self._segment_id,
            )
        return False

    async def _ended(self, task: asyncio.Task[R]) -> Finish:
        """How the work ended, once the drive has ended on its own and every
        event it emitted is taken: the work's outcome, with the frames that
        close it taken. A drive that raised, a cancellation of its own
        included, or work that ended without the ending frame its outcome
        needs, ends `unclosed` on the work's fatal frame. A lost lease the
        drive or a card write raised ends it `lost`."""
        if task.cancelled():
            return self._failed(asyncio.CancelledError())
        outcome, frames = await self._work.end(task.result())
        await self._take(frames, saving=False)
        finish = Finish(outcome.value)
        needed = _FINISH_FRAMES.get(finish)
        if needed is None or isinstance(self._held, needed):
            return finish
        return self._failed(RuntimeError(f"the segment ended {finish} with no ending frame for it"))

    def _failed(self, error: BaseException) -> Finish:
        """End the work `unclosed` on its fatal frame for `error`."""
        logger.error(
            "studio stream: conversation %s segment %s's work failed (trace_id=%s)",
            self._conversation_id,
            self._segment_id,
            self._conversation.trace_id,
            exc_info=error,
        )
        self._held = self._work.fatal(error)
        self._held_last = True
        return Finish.UNCLOSED

    async def _close(self, finish: Finish) -> SegmentClose | None:
        """The segment's closing write, or `None` for work that ended `lost`
        or `abandoned`. A Stop that ended the work closes the turn's record
        `cancelled` first, and every finish but `left` drops the pending
        entries resuming under the segment: a client that left keeps them,
        for a retry of the same answer to carry the turn on. The write stores
        `seq` past every number the closing frames can take. They start at
        the number the work gave its ending frame when nothing went out after
        it, so a stream that closes no lines keeps that number on the wire."""
        conversation = self._conversation
        held = self._held
        self._first_seq = held.seq if held is not None and self._held_last else conversation.seq
        if finish in (Finish.LOST, Finish.ABANDONED):
            return None
        if finish is Finish.CANCELLED:
            conversation.close_turn(
                final_text=conversation.turn_final_text, error=ErrorCode.CANCELLED.value
            )
        if finish is not Finish.LEFT:
            conversation.drop_resuming(self._segment_id)
        conversation.seq = max(conversation.seq, self._first_seq + len(self._lines) + 1)
        try:
            return await self._store.close_segment(
                self._principal, self._conversation_id, self._segment_id
            )
        except Exception:
            logger.exception(
                "studio stream: conversation %s segment %s's closing write raised",
                self._conversation_id,
                self._segment_id,
            )
            return SegmentClose.FAILED

    def closing_frames(self, end: Ending) -> list[BaseModel]:
        """The frames that end the stream: a `failed` line for every open
        Activity line before an `error` frame, then the ending frame, each
        numbered from the number read before the closing write. `done`
        carries the canvas the closing write read back."""
        frame = end.frame
        if frame is None:
            return []
        seq = itertools.count(self._first_seq)
        lines: list[BaseModel] = []
        if isinstance(frame, ErrorFrame):
            lines.extend(self._lines.close(detail=frame.error.detail, seq=seq))
        update: dict[str, Any] = {"seq": next(seq)}
        if isinstance(frame, Done):
            update["composition"] = self._conversation.composition
        return [*lines, frame.model_copy(update=update)]


async def own_segment[E, R](
    work: Work[E, R],
    *,
    store: SegmentStore,
    conversation: ConversationState,
    principal: str,
    conversation_id: str,
    segment_id: str,
    channel: Channel,
    gone: asyncio.Event,
    settle: Callable[[], Awaitable[bool]] | None = None,
) -> None:
    """Own the segment `segment_id` of `conversation`, whose lease and local
    lock the caller holds, to its end (module docstring), putting its frames
    on `channel` and reading `gone`, the client-gone flag.

    `settle`, when given, runs once the channel has ended and the span is
    exported: a wait that keeps the request open, and so its CPU, for work
    the segment started (`harness.early_mint.EarlyMints.settle`). When it
    returns `True` the spans are exported again.

    Raises nothing, and is never cancelled by a request.
    """
    owner = _Owner(
        work,
        store=store,
        conversation=conversation,
        principal=principal,
        conversation_id=conversation_id,
        segment_id=segment_id,
        channel=channel,
        gone=gone,
    )
    span = start_turn_span(conversation.trace_id, session_id=conversation_id)
    end = Ending(None, Status.UNCLOSED, saved=False)
    frames: list[BaseModel] = []
    reply: str | None = None
    try:
        with use_turn_span(span):
            end = await owner.run()
        frames = owner.closing_frames(end)
        if conversation.turn_recorded and conversation.turns:
            reply = _recorded_reply(conversation.turns[-1])
        if not end.saved:
            store.evict(principal, conversation_id)
    except Exception:
        logger.exception(
            "studio stream: conversation %s segment %s could not end its stream",
            conversation_id,
            segment_id,
        )
    finally:
        if conversation.lock.locked():
            conversation.lock.release()
        for frame in frames:
            channel.put_nowait(frame)
        channel.put_nowait(None)
        end_turn_span(
            span,
            request=conversation.turn_user_message,
            response=reply,
            error=end.status.value,
            level=_LEVELS[end.status],
            attributes={
                "dfo.saved": str(end.saved).lower(),
                "dfo.client_gone": str(gone.is_set()).lower(),
            },
        )
        await flush_telemetry()
    if settle is not None:
        try:
            if await settle():
                await flush_telemetry()
        except Exception:
            logger.exception("studio stream: conversation %s's settle failed", conversation_id)


class LiveOwners:
    """The owners running on this instance, each kept from its start until
    it finishes."""

    def __init__(self) -> None:
        self._running: set[asyncio.Task[None]] = set()

    def start(self, owner: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Run `owner` as a task of its own and keep it until it finishes."""
        task = asyncio.create_task(owner)
        self._running.add(task)
        task.add_done_callback(self._running.discard)
        return task

    async def wait(self, *, timeout: float = SHUTDOWN_WAIT_S) -> None:
        """Wait up to `timeout` seconds for every owner still running; log
        the ones still running then."""
        if not self._running:
            return
        _, running = await asyncio.wait(set(self._running), timeout=timeout)
        if running:
            logger.error(
                "studio stream: %d stream owner(s) still running at shutdown after %ss",
                len(running),
                timeout,
            )


def open_stream[E, R](
    work: Work[E, R],
    *,
    owners: LiveOwners,
    store: SegmentStore,
    conversation: ConversationState,
    principal: str,
    conversation_id: str,
    segment_id: str,
    settle: Callable[[], Awaitable[bool]] | None = None,
) -> StudioStream:
    """Start the owner of the claimed segment `segment_id` (`own_segment`)
    and return the response that relays it. The caller holds the segment's
    lease and the conversation's local lock, and hands both to the owner."""
    channel: Channel = asyncio.Queue()
    gone = asyncio.Event()
    owner = owners.start(
        own_segment(
            work,
            store=store,
            conversation=conversation,
            principal=principal,
            conversation_id=conversation_id,
            segment_id=segment_id,
            channel=channel,
            gone=gone,
            settle=settle,
        )
    )
    return StudioStream(channel=channel, owner=owner, gone=gone)
