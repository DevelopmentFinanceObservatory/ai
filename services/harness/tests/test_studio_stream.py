"""How a Studio stream's owner ends a segment (`harness.studio_stream`).

`test_every_ending_is_the_tables` pins `ending`, cell by cell.

`test_every_run_of_the_owner_keeps_the_invariants` runs the real owner and
the real response class over an in-memory model of the store and of a
segment's work, for every script, input, place the input lands, and fault of
the closing write, and checks ten invariants after each run:

- I1: a present client receives exactly one ending frame, and it is the last
  frame; a client that left is owed none, and gets nothing after one.
- I2: the wire implies the store: `done` means the turn is stored done, a
  pause frame that it is stored paused with the interrupt answerable, and
  `cancelled` that it is stored cancelled or holds a live Stop.
- I3: `seq` rises along the stream, every number sent lies inside the claim's
  reservation, and a closing write that committed stored `seq` past them.
- I4: a committed closing write cleared the lease, the local lock is free, no
  task of the segment runs, the owner finished uncancelled, and the span
  ended once with the table's status for how the work and the write ended.
- I5: every Activity line the stream opened is closed before an `error` frame.
- I6: no store write follows the closing write.
- I7: after a Stop the store accepted, the turn is cancelled, already closed,
  or holds a live Stop.
- I8: the instance's live state equals the stored row, or is evicted.
- I9: every event the drive emitted is translated, unless the lease was
  lost.

The response runs its real ASGI call over an in-memory transport that
delivers `http.disconnect` when the client leaves, as uvicorn's protocols do
at ASGI spec 2.3.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from types import SimpleNamespace
from typing import Any, cast

import pytest
from google.genai import types
from harness import studio as studio_module
from harness import studio_stream
from harness.conversation import current_conversation_id
from harness.fatal_codes import wire_error
from harness.studio import _SegmentWork, event_key
from harness.studio_relay import Channel, StudioStream
from harness.studio_state import (
    ConversationState,
    SegmentLost,
    _PendingInterrupt,
    land_component,
    remove_cards,
)
from harness.studio_store import SegmentClose, closed_turn_error
from harness.studio_stream import (
    Emit,
    Finish,
    LockHold,
    Status,
    WorkEnd,
    ending,
    own_segment,
)
from harness.studio_translate import (
    _RecordingExecutor,
    recorded_fatal,
)
from harness.tool_contracts import HarnessCode, harness_refusal
from harness.turn.loop import Outcome, Segment
from harness.turn.store import Cursor, TurnStatus
from harness.wire import (
    ActivityStep,
    Done,
    ErrorCode,
    ErrorFrame,
    MessageDelta,
    PlanApproved,
    PlanProposed,
    RetryMode,
    StepStatus,
    ToolError,
)
from orchestrator.run import ModelResponded, ToolsAnswered
from pydantic import BaseModel
from starlette.types import Message

from .failure_text import classed_sentence, is_user_sentence
from .turn_driver import unconnected_store

PRINCIPAL = "principal"
CONVERSATION = "conversation"
SEGMENT = "segment"
FIRST_SEQ = 100
RESERVED = FIRST_SEQ + 2**20

_PAUSE = PlanProposed(
    event="plan.proposed",
    seq=7,
    interrupt_id="i1",
    plan_version_id="pv",
    version=1,
    title="t",
    steps=[],
    actions=[],
)
_DONE = Done(event="done", seq=7, trace_id="tr")
_ERROR = ErrorFrame(
    event="error",
    seq=7,
    error=ToolError(
        error=ErrorCode.CAPABILITY_UNAVAILABLE, retry=RetryMode.DO_NOT_RETRY, detail="no"
    ),
)

# The ending table, as `ending` must give it: by the work's finish and the
# closing write's result, the ending frame (`held` for the work's own, an
# error code, or `None`) and the span's status. Written out here so the test
# does not read the table it checks.
_HELD = "held"
_CANCELLED = ErrorCode.CANCELLED.value
_TRANSIENT = ErrorCode.TRANSIENT_UNAVAILABLE.value
_S, _T, _L, _F = (
    SegmentClose.SAVED,
    SegmentClose.STOPPED,
    SegmentClose.LOST,
    SegmentClose.FAILED,
)
TABLE: dict[tuple[Finish, SegmentClose | None], tuple[str | None, Status]] = {
    (Finish.PAUSED, _S): (_HELD, Status.PAUSED),
    (Finish.PAUSED, _T): (_CANCELLED, Status.CANCELLED),
    (Finish.PAUSED, _L): (_TRANSIENT, Status.LOST),
    (Finish.PAUSED, _F): (_TRANSIENT, Status.UNSAVED),
    (Finish.DONE, _S): (_HELD, Status.DONE),
    (Finish.DONE, _T): (_HELD, Status.DONE),
    (Finish.DONE, _L): (_TRANSIENT, Status.LOST),
    (Finish.DONE, _F): (_TRANSIENT, Status.UNSAVED),
    (Finish.FAILED, _S): (_HELD, Status.FAILED),
    (Finish.FAILED, _T): (_HELD, Status.FAILED),
    (Finish.FAILED, _L): (_HELD, Status.FAILED),
    (Finish.FAILED, _F): (_HELD, Status.FAILED),
    (Finish.CANCELLED, _S): (_CANCELLED, Status.CANCELLED),
    (Finish.CANCELLED, _T): (_CANCELLED, Status.CANCELLED),
    (Finish.CANCELLED, _L): (_CANCELLED, Status.CANCELLED),
    (Finish.CANCELLED, _F): (_CANCELLED, Status.CANCELLED),
    (Finish.UNCLOSED, _S): (_HELD, Status.UNCLOSED),
    (Finish.UNCLOSED, _T): (_CANCELLED, Status.CANCELLED),
    (Finish.UNCLOSED, _L): (_TRANSIENT, Status.LOST),
    (Finish.UNCLOSED, _F): (_HELD, Status.UNCLOSED),
    (Finish.LEFT, _S): (None, Status.LEFT),
    (Finish.LEFT, _T): (None, Status.CANCELLED),
    (Finish.LEFT, _L): (None, Status.LOST),
    (Finish.LEFT, _F): (None, Status.LEFT),
    (Finish.LOST, None): (_TRANSIENT, Status.LOST),
    (Finish.ABANDONED, None): (_TRANSIENT, Status.LOST),
}

_HELD_FRAMES = {Finish.PAUSED: _PAUSE, Finish.DONE: _DONE, Finish.FAILED: _ERROR}


def _frame_name(frame: BaseModel | None) -> str | None:
    """`held`-free name of an ending frame: its error code, or its event."""
    if frame is None:
        return None
    if isinstance(frame, ErrorFrame):
        return frame.error.error.value
    return getattr(frame, "event", None)


@pytest.mark.parametrize(("finish", "close"), list(TABLE))
def test_every_ending_is_the_tables(finish: Finish, close: SegmentClose | None) -> None:
    held = _HELD_FRAMES.get(finish, _ERROR if finish is Finish.UNCLOSED else None)
    want_frame, want_status = TABLE[finish, close]
    end = ending(WorkEnd(finish, held), close)
    assert end.status is want_status
    assert end.saved is (close in (_S, _T))
    if want_frame == _HELD:
        assert end.frame is held
    else:
        assert _frame_name(end.frame) == want_frame


def test_work_that_makes_a_closing_write_needs_its_result() -> None:
    with pytest.raises(ValueError, match="closing write"):
        ending(WorkEnd(Finish.DONE, _DONE), None)


# The errors a failed turn's cursor can record (`harness.turn.loop.failure`).
_CURSOR_ERRORS = {
    "known-code": {"refusal": {"code": "capability_unavailable", "retry_mode": "do_not_retry"}},
    "unknown-code-retry": {
        "refusal": {"code": "surface_changed", "retry_mode": "retry_after_delay"}
    },
    "unknown-code-final": {"refusal": {"code": "surface_changed", "retry_mode": "do_not_retry"}},
    "transient": {"transient": True},
    "no-refusal": {},
    "timeout": {"timeout_tool": "get_data", "timeout_key": "get_data"},
}


def _cursor(status: TurnStatus, error: dict[str, Any] | None) -> Cursor:
    return Cursor(
        id=1,
        conversation=1,
        turn_index=1,
        node="orchestrate",
        node_input=None,
        status=status,
        round=0,
        pause=None,
        owed=None,
        inputs={},
        state={},
        error=error,
        version=1,
        record_count=0,
        spec_digests={},
    )


def _wire_code(recorded: str) -> ErrorCode:
    """The code a frame states for a record's error."""
    if recorded in ErrorCode:
        return ErrorCode(recorded)
    return wire_error(harness_refusal(code=HarnessCode(recorded), message=""))[0]


@pytest.mark.parametrize(
    ("finish", "status", "error"),
    [
        pytest.param(Finish.DONE, TurnStatus.DONE, None, id="done"),
        pytest.param(Finish.CANCELLED, TurnStatus.CANCELLED, None, id="cancelled"),
        *(
            pytest.param(Finish.FAILED, status, error, id=f"{status}-{name}")
            for status in (TurnStatus.FAILED, TurnStatus.ABANDONED)
            for name, error in _CURSOR_ERRORS.items()
        ),
    ],
)
def test_a_turn_whose_close_failed_is_recorded_with_the_code_its_stream_sent(
    finish: Finish, status: TurnStatus, error: dict[str, Any] | None
) -> None:
    # The work of every finish whose cursor closes, its closing write
    # failed: the load that finds the cursor closed records the turn with
    # the code of the frame `ending` sent. A failed turn's frame is the one
    # its cursor's error gives, and the live record it closed holds the
    # same code. A turn the segment closed `abandoned` on a node's failure
    # ends `failed` on that failure's frame.
    held: Any = _DONE if finish is Finish.DONE else None
    live = ConversationState(conversation_id=CONVERSATION)
    if finish is Finish.FAILED:
        held = recorded_fatal(live, error or {})
    sent = ending(WorkEnd(finish, held), SegmentClose.FAILED).frame
    assert isinstance(sent, ErrorFrame)
    recorded = closed_turn_error(_cursor(status, error))
    assert _wire_code(recorded) is sent.error.error
    if finish is Finish.FAILED:
        assert live.turns[-1]["error"] == recorded


# ---------------------------------------------------------------------------
# The enumerated runs of the owner
# ---------------------------------------------------------------------------


class Script(StrEnum):
    """What a segment's work does."""

    DONE = "done"
    PAUSES = "pauses"
    FAILS = "fails"
    HANGS = "hangs"
    EDIT = "edit"
    RAISES = "raises"
    CARD = "card"


class Point(StrEnum):
    """Where an input lands, at one of the owner's four await points: while
    it waits for the work (early, before the turn's boundary write, or late,
    after it), while it waits for
    cancelled work to stop, and during the closing write; each before and
    after the write or the work's cleanup."""

    WORK_EARLY = "work_early"
    WORK_LATE = "work_late"
    CANCEL_BEFORE = "cancel_before"
    CANCEL_AFTER = "cancel_after"
    CLOSE_BEFORE = "close_before"
    CLOSE_AFTER = "close_after"


class Fault(StrEnum):
    """How the closing write fails."""

    NONE = "none"
    UNCOMMITTED = "uncommitted"
    ACK_LOST = "ack_lost"
    FENCED = "fenced"
    TIMEOUT = "timeout"


STOP, LOST, GONE = "stop", "lost", "gone"
INPUTS: tuple[frozenset[str], ...] = (
    frozenset(),
    frozenset({STOP}),
    frozenset({LOST}),
    frozenset({GONE}),
    frozenset({STOP, GONE}),
    frozenset({LOST, GONE}),
)
_CANCEL_POINTS = (Point.CANCEL_BEFORE, Point.CANCEL_AFTER)


@dataclass(frozen=True)
class Case:
    script: Script
    inputs: frozenset[str]
    point: Point | None
    fault: Fault

    def __str__(self) -> str:
        return f"{self.script}/{'+'.join(sorted(self.inputs)) or 'none'}@{self.point}/{self.fault}"


def _cases() -> list[Case]:
    """Every case whose input can land where it says. An input lands at a
    cancel point only after the client left early, which is what cancels the
    work; the hanging work waits for an input that ends it."""
    cases: list[Case] = []
    for script, inputs, fault in itertools.product(Script, INPUTS, Fault):
        if not inputs:
            if script is not Script.HANGS:
                cases.append(Case(script, inputs, None, fault))
            continue
        for point in Point:
            if point is Point.WORK_LATE and script in (Script.HANGS, Script.RAISES):
                continue
            if point in _CANCEL_POINTS and (GONE not in inputs or len(inputs) == 1):
                continue
            if script is Script.HANGS and point in (Point.CLOSE_BEFORE, Point.CLOSE_AFTER):
                continue
            cases.append(Case(script, inputs, point, fault))
    return cases


type Pending = tuple[tuple[str, str | None], ...]
type Durable = tuple[bool, str | None, Pending, int]


def _durable(conversation: ConversationState) -> Durable:
    """The fields of the conversation the model's store keeps: whether the
    turn is recorded and its error, the pending interrupts, and `seq`."""
    return (
        conversation.turn_recorded,
        conversation.turns[-1].get("error") if conversation.turn_recorded else None,
        tuple(
            sorted((key, pending.resuming_segment) for key, pending in conversation.pending.items())
        ),
        conversation.seq,
    )


_OPEN = ("running", "paused")


@dataclass
class World:
    """The model of the store: the turn's cursor, the lease, a Stop, and the
    conversation row as last committed."""

    stored: Durable
    cursor: str = "running"
    lease: str | None = SEGMENT
    stop: bool = False
    stop_accepted: bool = False
    committed_close: bool = False
    evicted: bool = False
    writes: list[str] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    def stop_live(self) -> bool:
        return self.stop and self.cursor in _OPEN


class _Cut(Exception):  # noqa: N818 -- carries the outcome the cut leaves
    def __init__(self, outcome: Outcome) -> None:
        self.outcome = outcome


@dataclass
class Run:
    """One run of the owner over `case`."""

    case: Case
    conversation: ConversationState
    world: World
    disconnect: asyncio.Event = field(default_factory=asyncio.Event)
    gone: asyncio.Event = field(default_factory=asyncio.Event)
    landed: set[Point] = field(default_factory=set)
    # What the model saw: how the work ended and the closing write's result.
    finish: Finish | None = None
    close: SegmentClose | None = None
    work_task: asyncio.Task[Any] | None = None
    # How many events the drive has emitted.
    emitted: int = 0
    # Whether a card write in the owner's translation found the lease lost,
    # and whether the owner translated a card after the drive had stopped.
    card_lost: bool = False
    card_after_stop: bool = False
    translated: int = 0
    left: Callable[[], None] | None = None

    async def land(self, point: Point) -> None:
        """Apply the case's inputs that land at `point`."""
        case = self.case
        if case.point is None or point in self.landed:
            return
        if case.point in _CANCEL_POINTS and point is Point.WORK_EARLY:
            self.landed.add(point)
            await self._leave()
            await self._in_node()
        if point is not case.point:
            return
        self.landed.add(point)
        if STOP in case.inputs:
            self.world.stop = True
            self.world.stop_accepted = self.world.stop_accepted or self.world.cursor in _OPEN
        if LOST in case.inputs:
            self.world.lease = "other"
        self.world.changed.set()
        if GONE in case.inputs and case.point not in _CANCEL_POINTS:
            await self._leave()
            if point in (Point.WORK_EARLY, Point.WORK_LATE):
                await self._in_node()

    @staticmethod
    async def _in_node() -> None:
        """The work stays inside its node until the owner cancels it."""
        await asyncio.Future()

    async def _leave(self) -> None:
        """The client closes, and the response sets the gone flag. A drive
        that emits as the flag is set (`left`) emits then, with no await
        between, so its event waits for the owner past the flag."""
        self.disconnect.set()
        for _ in range(1000):
            if self.gone.is_set():
                if self.left is not None and not (self.work_task and self.work_task.done()):
                    self.left()
                return
            await asyncio.sleep(0)
        raise AssertionError("the response never set the client-gone flag")


# An event of a fake drive: what the owner's translation of it sends and
# changes, run in the owner's task.
type Make = Callable[[], list[BaseModel]]
type AsyncMake = Callable[[], Awaitable[list[BaseModel]]]


class FakeWork:
    """A segment's work over the model. Its drive moves the turn's cursor,
    as the turn loop does, sees a Stop or a lost lease at each of its ticks
    as the turn loop's heartbeat would, and emits events. The owner's
    translation of each event numbers its frames and writes the conversation
    state they change, as the translator does."""

    def __init__(self, run: Run) -> None:
        self._run = run
        self._c = run.conversation
        self._w = run.world
        self._emit: Emit | None = None

    def _send(self, make: Make) -> None:
        assert self._emit is not None
        self._run.emitted += 1
        self._emit(make)

    def _card(self) -> AsyncMake:
        """A card: its translation awaits the card's write, fenced on the
        lease, as `_Translator._land` does."""

        async def land() -> list[BaseModel]:
            run = self._run
            if run.work_task is not None and run.work_task.cancelled():
                run.card_after_stop = True
            await asyncio.sleep(0)
            if self._w.lease != SEGMENT:
                run.card_lost = True
                raise SegmentLost(CONVERSATION)
            return [
                ActivityStep(
                    event="activity.step",
                    seq=self._c.next_seq(),
                    step_id="card",
                    label="card",
                    status=StepStatus.COMPLETE,
                    duration_ms=5,
                )
            ]

        return land

    def _step(self, step_id: str, status: StepStatus, *, done: bool = False) -> Make:
        return lambda: [
            ActivityStep(
                event="activity.step",
                seq=self._c.next_seq(),
                step_id=step_id,
                label=step_id,
                status=status,
                duration_ms=5 if done else None,
            )
        ]

    async def _tick(self, point: Point | None) -> None:
        if point is not None:
            await self._run.land(point)
        await asyncio.sleep(0)
        world = self._w
        if world.lease != SEGMENT:
            if world.cursor == "paused":
                raise _Cut(Outcome.PAUSED)
            if world.cursor not in _OPEN:
                raise _Cut(Outcome(world.cursor))
            raise _Cut(Outcome.LOST)
        if world.stop_live():
            world.cursor = "cancelled"
            raise _Cut(Outcome.CANCELLED)

    def begin(self) -> None:
        return None

    async def drive(self, emit: Emit) -> Outcome:
        self._emit = emit
        self._run.work_task = asyncio.current_task()
        try:
            outcome = await self._script()
        except _Cut as cut:
            outcome = cut.outcome
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                await self._run.land(Point.CANCEL_BEFORE)
                await asyncio.sleep(0)
                await self._run.land(Point.CANCEL_AFTER)
                self._run.finish = Finish.LEFT
            else:
                self._run.finish = Finish.UNCLOSED
            raise
        self._run.finish = Finish(outcome.value)
        return outcome

    async def frames(self, event: Make | AsyncMake) -> list[BaseModel]:
        self._run.translated += 1
        made = event()
        return await made if inspect.isawaitable(made) else made

    async def end(self, result: object) -> tuple[Outcome, list[BaseModel]]:
        return cast(Outcome, result), []

    async def _script(self) -> Outcome:
        script = self._run.case.script
        c, w = self._c, self._w
        self._send(self._step("s1", StepStatus.RUNNING))
        if script is Script.CARD:
            self._send(self._card())
            # Another card, emitted as the client leaves.
            self._run.left = lambda: self._send(self._card())
        await self._tick(Point.WORK_EARLY)
        match script:
            case Script.HANGS:
                while True:
                    await w.changed.wait()
                    w.changed.clear()
                    await self._tick(None)
            case Script.RAISES:
                raise asyncio.CancelledError
            case Script.FAILS:
                w.cursor = "failed"
                self._send(lambda: [self._fatal(ErrorCode.CAPABILITY_UNAVAILABLE)])
                await self._tick(Point.WORK_LATE)
                return Outcome.FAILED
            case Script.PAUSES:
                self._send(self._step("s1", StepStatus.COMPLETE, done=True))
                w.cursor = "paused"

                def pause() -> list[BaseModel]:
                    c.pending["i1"] = _PendingInterrupt(adk_id="i1", kind="plan")
                    return [_PAUSE.model_copy(update={"seq": c.next_seq()})]

                self._send(pause)
                await self._tick(Point.WORK_LATE)
                return Outcome.PAUSED
            case Script.EDIT:
                # The edit's second line is still open when `done` comes.
                self._send(self._step("s2", StepStatus.RUNNING))
            case _:
                self._send(self._step("s1", StepStatus.COMPLETE, done=True))
        w.cursor = "done"

        def done() -> list[BaseModel]:
            c.close_turn(final_text="answer")
            return [_DONE.model_copy(update={"seq": c.next_seq()})]

        self._send(done)
        await self._tick(Point.WORK_LATE)
        return Outcome.DONE

    def _fatal(self, code: ErrorCode) -> ErrorFrame:
        self._c.close_turn(final_text="", error=code.value)
        return ErrorFrame(
            event="error",
            seq=self._c.next_seq(),
            error=ToolError(error=code, retry=RetryMode.DO_NOT_RETRY, detail="failed"),
        )

    def owed(self) -> list[BaseModel]:
        return []

    def fatal(self, error: BaseException) -> ErrorFrame:
        del error
        return self._fatal(ErrorCode.CAPABILITY_UNAVAILABLE)

    def close(self) -> None:
        return None


class FakeStore:
    """The owner's writes, over the model."""

    def __init__(self, run: Run) -> None:
        self._run = run
        self._c = run.conversation
        self._w = run.world

    async def close_segment(
        self, principal: str, conversation_id: str, segment_id: str
    ) -> SegmentClose:
        self._w.writes.append("close")
        await self._run.land(Point.CLOSE_BEFORE)
        result = self._close(segment_id)
        self._run.close = result
        await self._run.land(Point.CLOSE_AFTER)
        return result

    def _close(self, segment_id: str) -> SegmentClose:
        world, c = self._w, self._c
        fault = self._run.case.fault
        if fault is Fault.FENCED:
            world.lease = "other"
        if world.lease != segment_id:
            return SegmentClose.LOST
        if fault in (Fault.UNCOMMITTED, Fault.TIMEOUT):
            return SegmentClose.FAILED
        stopped = world.stop_live()
        if stopped:
            # The Stop closes the record `cancelled`, whatever error the
            # work's own close recorded first.
            world.cursor = "cancelled"
            world.stored = (True, ErrorCode.CANCELLED.value, (), c.seq)
        else:
            world.stored = _durable(c)
        world.lease = None
        world.committed_close = True
        if fault is Fault.ACK_LOST:
            return SegmentClose.FAILED
        if stopped:
            c.pending.clear()
            c.close_turn(final_text=c.turn_final_text, error=ErrorCode.CANCELLED.value)
            c.turns[-1] = {**c.turns[-1], "error": ErrorCode.CANCELLED.value}
        return SegmentClose.STOPPED if stopped else SegmentClose.SAVED

    def evict(self, principal: str, conversation_id: str) -> None:
        self._w.writes.append("evict")
        self._w.evicted = True


@dataclass
class Spans:
    started: int = 0
    ended: list[tuple[str | None, dict[str, str]]] = field(default_factory=list)


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Spans:
    recorded = Spans()

    def start(trace_id: str, **_: Any) -> object:
        recorded.started += 1
        return object()

    def end(handle: object, **kwargs: Any) -> None:
        recorded.ended.append((kwargs.get("status"), dict(kwargs.get("attributes") or {})))

    async def flush() -> None:
        return None

    monkeypatch.setattr(studio_stream, "start_turn_segment", start)
    monkeypatch.setattr(studio_stream, "end_turn_segment", end)
    monkeypatch.setattr(studio_stream, "flush_telemetry", flush)
    return recorded


_SCOPE = {
    "type": "http",
    "asgi": {"version": "3.0", "spec_version": "2.3"},
    "http_version": "1.1",
    "method": "POST",
    "path": "/api/studio/turn",
    "headers": [],
}
_ENDING_EVENTS = {"error", "done", "plan.proposed", "elicitation.request"}


async def _one(case: Case, spans: Spans) -> list[str]:
    """Run the owner and the response over `case`; the invariants it broke."""
    broken, _ = await _run_case(case, spans)
    return broken


async def _run_case(case: Case, spans: Spans) -> tuple[list[str], Run]:
    """Run the owner and the response over `case`; the invariants it broke,
    and the run."""
    conversation = ConversationState(conversation_id=CONVERSATION, trace_id="tr", seq=FIRST_SEQ)
    await conversation.lock.acquire()
    hold = LockHold(conversation.lock)
    world = World(stored=_durable(conversation))
    run = Run(case, conversation, world)
    channel: Channel = asyncio.Queue()
    owner = asyncio.create_task(
        own_segment(
            FakeWork(run),
            store=FakeStore(run),
            conversation=conversation,
            principal=PRINCIPAL,
            conversation_id=CONVERSATION,
            segment_id=SEGMENT,
            channel=channel,
            gone=run.gone,
            hold=hold,
        )
    )
    response = StudioStream(channel=channel, owner=owner, gone=run.gone, keepalive_s=60)
    body: list[bytes] = []

    async def receive() -> Message:
        await run.disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.body" and not run.disconnect.is_set():
            body.append(message.get("body", b""))

    spans.started, spans.ended = 0, []
    try:
        async with asyncio.timeout(5):
            await response(_SCOPE, receive, send)
            await asyncio.wait({owner})
    except TimeoutError:
        return ["the run did not finish"], run
    return _broken(run, owner, spans, b"".join(body).decode()), run


def _broken(run: Run, owner: asyncio.Task[None], spans: Spans, wire: str) -> list[str]:
    """The invariants `run` broke."""
    world, conversation, case = run.world, run.conversation, run.case
    broken: list[str] = []
    if case.point is not None and case.point not in run.landed:
        broken.append(f"the input never landed at {case.point}")
    frames = [
        json.loads(chunk.removeprefix("data: "))
        for chunk in wire.split("\n\n")
        if chunk.startswith("data: ")
    ]
    endings = [i for i, frame in enumerate(frames) if frame["event"] in _ENDING_EVENTS]
    left = GONE in case.inputs and run.disconnect.is_set()

    # I1
    if left:
        if len(endings) > 1 or (endings and endings[-1] != len(frames) - 1):
            broken.append(f"I1: a client that left got {len(endings)} ending frames, not last")
    elif len(endings) != 1 or endings[0] != len(frames) - 1:
        broken.append(f"I1: {len(endings)} ending frames, at {endings} of {len(frames)}")

    # I2
    last = frames[-1] if endings and endings[-1] == len(frames) - 1 else None
    if last is not None:
        recorded, error, pending, _ = world.stored
        match last["event"]:
            case "done":
                if not (recorded and error is None and world.cursor == "done"):
                    broken.append(f"I2: done sent, stored {world.stored} cursor {world.cursor}")
            case "plan.proposed":
                answerable = ("i1", None) in pending
                if recorded or not answerable or world.cursor != "paused":
                    broken.append(f"I2: pause sent, stored {world.stored} cursor {world.cursor}")
            case "error" if last["error"]["error"] == ErrorCode.CANCELLED.value:
                # A committed close stores the turn `cancelled`, record and
                # cursor both; an unsaved one leaves a cursor closed
                # `cancelled`, or a Stop live, for the next load or claim.
                closed = (
                    recorded and error == ErrorCode.CANCELLED.value and world.cursor == "cancelled"
                )
                unsaved = not world.committed_close and (
                    world.cursor == "cancelled" or world.stop_live()
                )
                if not (closed or unsaved):
                    broken.append(f"I2: cancelled sent, stored {world.stored} {world.cursor}")
            case _:
                pass

    # I3
    seqs = [frame["seq"] for frame in frames]
    if seqs != sorted(set(seqs)):
        broken.append(f"I3: seq does not rise: {seqs}")
    if seqs and max(seqs) >= RESERVED:
        broken.append("I3: a frame numbered past the reservation")
    if world.committed_close and seqs and max(seqs) >= world.stored[3]:
        broken.append(f"I3: sent {max(seqs)} with seq_next stored at {world.stored[3]}")

    # A card write that finds the lease lost ends the work `lost`, however its
    # drive ended.
    finish = Finish.LOST if run.card_lost else run.finish

    # I4
    if run.close in (SegmentClose.SAVED, SegmentClose.STOPPED) and world.lease == SEGMENT:
        broken.append("I4: a committed close left the lease held")
    if conversation.lock.locked():
        broken.append("I4: the local lock is held")
    if run.work_task is not None and not run.work_task.done():
        broken.append("I4: the work still runs")
    if not owner.done() or owner.cancelled():
        broken.append("I4: the owner did not finish uncancelled")
    else:
        assert finish is not None
        _, want = TABLE[finish, run.close]
        statuses = [status for status, _ in spans.ended]
        if statuses != [want.value] or spans.started != 1:
            broken.append(f"I4: spans {statuses}, want [{want}] for {finish}/{run.close}")
        elif (
            spans.ended[0][1]["dfo.saved"]
            != str(run.close in (SegmentClose.SAVED, SegmentClose.STOPPED)).lower()
        ):
            broken.append("I4: the span's dfo.saved")

    # I5
    open_lines: set[str] = set()
    for frame in frames:
        if frame["event"] == "activity.step":
            if "duration_ms" not in frame and frame["status"] in ("running", "pending"):
                open_lines.add(frame["step_id"])
            else:
                open_lines.discard(frame["step_id"])
        elif frame["event"] == "error" and open_lines:
            broken.append(f"I5: lines {sorted(open_lines)} open at the error frame")

    # I6
    if "close" in world.writes:
        after = world.writes[world.writes.index("close") + 1 :]
        if any(write != "evict" for write in after):
            broken.append(f"I6: writes after the close: {after}")

    # I7
    if world.stop_accepted and not (
        world.cursor in ("cancelled", "done", "failed") or world.stop_live()
    ):
        broken.append(f"I7: a Stop left the turn {world.cursor}")

    # I8
    if not world.evicted and _durable(conversation) != world.stored:
        broken.append(f"I8: live {_durable(conversation)} stored {world.stored}")

    # I9: every event the drive emitted is translated, unless the lease was
    # lost.
    if finish is not Finish.LOST and run.translated != run.emitted:
        broken.append(f"I9: translated {run.translated} of {run.emitted} events")

    return broken


async def test_every_run_of_the_owner_keeps_the_invariants(spans: Spans) -> None:
    cases = _cases()
    failures: dict[str, list[str]] = {}
    lost = after_stop = 0
    for case in cases:
        broken, run = await _run_case(case, spans)
        lost += run.card_lost
        after_stop += run.card_after_stop
        if broken:
            failures[str(case)] = broken
    print(
        f"{len(cases)} runs of the owner, "
        f"{lost} with a card write finding the lease lost, {after_stop} translating a card "
        "after the drive stopped"
    )
    assert cases
    assert lost
    assert after_stop
    assert not failures, (
        "\n".join(f"{case}: {'; '.join(broken)}" for case, broken in list(failures.items())[:40])
        + f"\n({len(failures)} of {len(cases)} runs broke an invariant)"
    )


async def test_the_owner_ends_a_present_clients_stream_on_the_pause_it_saved(
    spans: Spans,
) -> None:
    # A control: the run with no input and no fault ends on the pause frame,
    # stored, with the span `paused`.
    assert not await _one(Case(Script.PAUSES, frozenset(), None, Fault.NONE), spans)
    assert spans.ended[0][0] == Status.PAUSED.value


# --- the frames that end a stream, keepalives ---


class _ScriptedWork:
    """Work that sends `frames` and returns `outcome`, or raises `raises` when
    given."""

    def __init__(
        self,
        frames: list[BaseModel],
        outcome: Outcome,
        *,
        raises: Exception | None = None,
    ) -> None:
        self._frames = frames
        self._outcome = outcome
        self._raises = raises

    def begin(self) -> None:
        return None

    async def drive(self, emit: Emit) -> Outcome:
        for frame in self._frames:
            emit(frame)
        if self._raises is not None:
            raise self._raises
        return self._outcome

    async def frames(self, event: object) -> list[BaseModel]:
        return [cast(BaseModel, event)]

    async def end(self, result: object) -> tuple[Outcome, list[BaseModel]]:
        return cast(Outcome, result), []

    def owed(self) -> list[BaseModel]:
        return []

    def fatal(self, error: BaseException) -> ErrorFrame:
        raise AssertionError("the work does not fail") from error

    def close(self) -> None:
        return None


class _LandingStore:
    """The owner's writes, each landing, recorded in order."""

    def __init__(self) -> None:
        self.writes: list[str] = []

    async def close_segment(
        self, principal: str, conversation_id: str, segment_id: str
    ) -> SegmentClose:
        self.writes.append("close")
        return SegmentClose.SAVED

    def evict(self, principal: str, conversation_id: str) -> None:
        return None


async def _owned(
    work: _ScriptedWork, conversation: ConversationState, store: _LandingStore
) -> list[dict[str, Any]]:
    """Run the owner over `work` to its end; the frames it put on the
    channel, as the wire carries them."""
    channel: Channel = asyncio.Queue()
    await conversation.lock.acquire()
    await own_segment(
        work,
        store=store,
        conversation=conversation,
        principal=PRINCIPAL,
        conversation_id=CONVERSATION,
        segment_id=SEGMENT,
        channel=channel,
        gone=asyncio.Event(),
        hold=LockHold(conversation.lock),
    )
    frames: list[dict[str, Any]] = []
    while (frame := channel.get_nowait()) is not None:
        frames.append(json.loads(frame.model_dump_json(exclude_none=True)))
    return frames


class _RemovingSegment(_SegmentWork):
    """The Studio's own segment work, its translation, `owed`, `end` and
    `fatal` as served, over a conversation whose agent removes `cmp_01` of
    two cards. Its drive stands in for the turn loop: it emits the
    removal's call, removes the card once the call is translated, emits the
    call's answer when `answered`, a response drawing a chart when `draws`,
    and then raises `raises`, ends `ends`, or runs until it is cancelled.
    With `calls` false it makes no removal call and goes straight to the
    drawing. `hold` is the request's `hold_removals`, left out when `None`."""

    def __init__(
        self,
        conversation: ConversationState,
        *,
        remove: bool = True,
        answered: bool = True,
        raises: BaseException | None = None,
        ends: Outcome | None = None,
        draws: bool = False,
        calls: bool = True,
        hold: bool | None = None,
    ) -> None:
        runner = SimpleNamespace(
            recorder=_RecordingExecutor(_never_called, store=unconnected_store()),
            surface_recorder=None,
            place_options=None,
        )
        self.cursor = SimpleNamespace(state={}, pause=None, error=None)
        super().__init__(
            cast(Any, runner),
            principal=PRINCIPAL,
            conversation=conversation,
            cursor=cast(Any, self.cursor),
            segment_id=SEGMENT,
            **({} if hold is None else {"hold_removals": hold}),
        )
        self._draws = draws
        self._calls = calls
        self._remove = remove
        self._answered = answered
        self._raises = raises
        self._ends = ends
        self._responded = asyncio.Event()
        self.removed = asyncio.Event()

    async def drive(self, emit: Emit) -> Segment:
        def recorded(event: Any) -> None:
            emit(self._marks.recorded(event), key=event_key(event))

        if self._calls:
            call = types.FunctionCall(id="r", name="remove_cards", args={})
            recorded(ModelResponded(parts=(types.Part(function_call=call),)))
            # The call runs once its response is translated, as the agent's does.
            await self._responded.wait()
            if self._remove:
                remove_cards(self._conversation, ["cmp_01"], all_cards=False)
        self.removed.set()
        if self._calls and self._answered:
            answer = types.FunctionResponse(
                id="r", name="remove_cards", response={"removed": ["cmp_01"]}
            )
            recorded(ToolsAnswered(answers=(answer,), state={}))
        if self._draws:
            charts = {"charts": [{"frame": "f", "type": "stat_card", "title": "Aid"}]}
            drawing = types.FunctionCall(id="m", name="make_charts", args=charts)
            recorded(ModelResponded(parts=(types.Part(function_call=drawing),)))
        if self._raises is not None:
            raise self._raises
        if self._ends is not None:
            return Segment(cursor=cast(Any, self.cursor), outcome=self._ends)
        await asyncio.Event().wait()
        raise AssertionError("the drive runs until it is cancelled")

    async def frames(self, event: Any) -> list[BaseModel]:
        frames = await super().frames(event)
        self._responded.set()
        return frames


async def _never_called(*, tool: str, args: dict[str, object]) -> dict[str, object]:
    raise AssertionError(f"no tool call expected, got {tool}")


def _two_cards() -> ConversationState:
    conversation = ConversationState(conversation_id=CONVERSATION, trace_id="tr")
    conversation.begin_turn()
    for cid in ("cmp_01", "cmp_02"):
        land_component(
            conversation, {"component_id": cid, "revision": 1, "type": "line_chart", "title": "Aid"}
        )
    return conversation


_RAISED = RuntimeError("the drive failed")


@pytest.mark.parametrize(
    ("remove", "answered", "raises", "ends"),
    [
        (True, True, _RAISED, None),
        (False, True, _RAISED, None),
        (True, True, None, Outcome.ABANDONED),
        (True, True, None, Outcome.LOST),
        (True, False, _RAISED, None),
        (True, False, SegmentLost(CONVERSATION), None),
    ],
    ids=[
        "translated-then-raises",
        "nothing-removed",
        "abandoned",
        "lost",
        "raises-before-the-answer",
        "lease-lost-before-the-answer",
    ],
)
async def test_a_removal_goes_out_before_the_stream_ends_however_it_ends(
    spans: Spans,
    remove: bool,
    answered: bool,
    raises: BaseException | None,
    ends: Outcome | None,
) -> None:
    """With `hold_removals`, the drive raises, its turn is abandoned, or
    another instance takes its lease, after the agent removed a card, whether
    or not the removal's answer was translated. The stream sends the canvas
    without the card once, after the removal's line closed, then the `error`
    frame. With nothing removed, it sends the error alone."""
    current_conversation_id.set(CONVERSATION)
    conversation = _two_cards()
    work = _RemovingSegment(
        conversation, remove=remove, answered=answered, raises=raises, ends=ends, hold=True
    )

    frames = await _owned(cast(Any, work), conversation, _LandingStore())

    events = [f["event"] for f in frames if f["event"] != "activity.step"]
    assert events == (["composition.updated", "error"] if remove else ["error"])
    if remove and answered:
        # Held through the removal's answer, which the default sends ahead of.
        assert _order(frames).index("composition.updated") > _order(frames).index(
            "remove_cards:complete"
        )
    if remove:
        (update,) = [f for f in frames if f["event"] == "composition.updated"]
        assert [b["component_id"] for b in update["manifest"]["blocks"]] == ["cmp_02"]
    assert work.owed() == []
    assert not conversation.take_unreported_removal()


async def test_an_owner_that_fails_still_sends_the_removal_before_its_transient_frame(
    spans: Spans, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drive raises and the work's fatal frame raises too, a fault in the
    owner. The stream sends the removal before the transient frame."""
    current_conversation_id.set(CONVERSATION)
    conversation = _two_cards()
    work = _RemovingSegment(conversation, answered=False, raises=_RAISED, hold=True)

    def broken(conversation: ConversationState, error: Exception) -> ErrorFrame:
        raise RuntimeError("the fatal frame could not be built")

    monkeypatch.setattr(studio_module, "_stream_fatal", broken)

    frames = await _owned(cast(Any, work), conversation, _LandingStore())

    events = [f["event"] for f in frames if f["event"] != "activity.step"]
    assert events == ["composition.updated", "error"]
    assert frames[-1]["error"]["error"] == "transient_unavailable"


def _order(frames: list[dict[str, Any]]) -> list[str]:
    """The removal's and the drawing's lines, canvas updates and the error,
    in stream order."""
    return [
        f"{f.get('tool')}:{f['status']}" if f["event"] == "activity.step" else f["event"]
        for f in frames
        if f["event"] != "activity.step" or f.get("tool") in {"remove_cards", "make_charts"}
    ]


_BASE_ORDER = [
    "remove_cards:running",
    "composition.updated",
    "remove_cards:complete",
    "make_charts:running",
    "make_charts:failed",
    "error",
]
_HELD_ORDER = [
    "remove_cards:running",
    "remove_cards:complete",
    "make_charts:running",
    "composition.updated",
    "make_charts:failed",
    "error",
]


@pytest.mark.parametrize(
    ("hold", "answered", "order", "left_marked"),
    [
        (None, True, _BASE_ORDER, False),
        (False, True, _BASE_ORDER, False),
        (True, True, _HELD_ORDER, False),
        (None, False, ["remove_cards:running", "remove_cards:failed", "error"], True),
    ],
    ids=["absent", "false", "true", "absent-raises-before-the-answer"],
)
async def test_a_removal_then_a_drawing_streams_in_the_order_the_request_asks(
    spans: Spans, hold: bool | None, answered: bool, order: list[str], left_marked: bool
) -> None:
    """A turn removes a card, then draws one. Without `hold_removals` the
    canvas update goes out as the removal's answer is drained, ahead of the
    removal's closing line, as the base branch sends it. With it, the update
    waits for the drawing line and follows it. The abandoned turn fails the
    drawing line. Without the member, a drive that raises before the
    removal's answer sends no update and leaves the removal marked on the
    conversation for the turn's next stream, as the base branch does."""
    current_conversation_id.set(CONVERSATION)
    conversation = _two_cards()
    work = _RemovingSegment(
        conversation,
        answered=answered,
        draws=answered,
        ends=None if not answered else Outcome.ABANDONED,
        raises=None if answered else _RAISED,
        hold=hold,
    )

    frames = await _owned(cast(Any, work), conversation, _LandingStore())

    assert _order(frames) == order
    assert conversation.take_unreported_removal() is left_marked


@pytest.mark.parametrize(
    ("first", "answered", "then", "left_sent", "resumed_sends"),
    [
        (True, True, None, 1, None),
        (True, False, None, 1, None),
        (None, True, True, 1, None),
        (None, False, True, 0, "make_charts:running"),
        (None, False, None, 0, "first"),
    ],
    ids=[
        "held-then-not-answered",
        "held-then-not-unanswered",
        "not-then-held-answered",
        "not-then-held-unanswered",
        "not-then-not-unanswered",
    ],
)
async def test_a_client_that_leaves_has_its_removal_reported_once_whatever_the_next_stream_asks(
    spans: Spans,
    first: bool | None,
    answered: bool,
    then: bool | None,
    left_sent: int,
    resumed_sends: str | None,
) -> None:
    """The client leaves after the agent removed a card, before or after the
    removal's answer was translated, and a later stream picks the turn up
    and draws. A segment with `hold_removals` reports the removal as it
    ends, on the stream the client left. One without it reports it when its
    answer is drained, and otherwise leaves it marked on the conversation,
    as the base branch does. The later stream, whose first response draws,
    then reports it at its first drain, ahead of the drawing line, without
    the member, and holds it to right after that line with it. Across both
    streams it goes out once."""
    current_conversation_id.set(CONVERSATION)
    conversation = _two_cards()
    work = _RemovingSegment(conversation, answered=answered, hold=first)
    channel: Channel = asyncio.Queue()
    gone = asyncio.Event()
    await conversation.lock.acquire()
    owner = asyncio.ensure_future(
        own_segment(
            cast(Any, work),
            store=_LandingStore(),
            conversation=conversation,
            principal=PRINCIPAL,
            conversation_id=CONVERSATION,
            segment_id=SEGMENT,
            channel=channel,
            gone=gone,
            hold=LockHold(conversation.lock),
        )
    )
    await work.removed.wait()
    for _ in range(5):
        await asyncio.sleep(0)
    gone.set()
    await owner
    left: list[str] = []
    while not channel.empty() and (frame := channel.get_nowait()) is not None:
        left.append(frame.event)
    resumed = _RemovingSegment(
        conversation, calls=False, draws=True, ends=Outcome.ABANDONED, hold=then
    )

    frames = await _owned(cast(Any, resumed), conversation, _LandingStore())

    assert left.count("composition.updated") == left_sent
    order = _order(frames)
    assert order.count("composition.updated") == (0 if resumed_sends is None else 1)
    if resumed_sends == "first":
        assert order[0] == "composition.updated"
    elif resumed_sends is not None:
        assert order[order.index("composition.updated") - 1] == resumed_sends
    assert work.owed() == []
    assert not conversation.take_unreported_removal()


def _running(conversation: ConversationState, step_id: str, **fields: Any) -> ActivityStep:
    return ActivityStep(
        event="activity.step",
        seq=conversation.next_seq(),
        step_id=step_id,
        label="Fetch the data",
        status=StepStatus.RUNNING,
        **{"tool": "get_data", **fields},
    )


async def test_an_owner_whose_hold_let_go_leaves_the_next_request_s_lock_alone(
    spans: Spans,
) -> None:
    """The owner lets go of the lock through the hold the route handed it.
    A hold that already let go releases nothing, so the lock another
    request took since stays held."""
    conversation = ConversationState(conversation_id=CONVERSATION, trace_id="tr")
    conversation.begin_turn()
    await conversation.lock.acquire()
    hold = LockHold(conversation.lock)
    hold.release()
    await conversation.lock.acquire()

    await own_segment(
        _ScriptedWork([], Outcome.FAILED),
        store=_LandingStore(),
        conversation=conversation,
        principal=PRINCIPAL,
        conversation_id=CONVERSATION,
        segment_id=SEGMENT,
        channel=asyncio.Queue(),
        gone=asyncio.Event(),
        hold=hold,
    )

    assert conversation.lock.locked()


@pytest.mark.parametrize("outcome", [Outcome.CANCELLED, Outcome.FAILED, Outcome.LOST], ids=str)
async def test_a_stream_ending_on_an_error_closes_the_activity_lines_it_left_open(
    spans: Spans, outcome: Outcome
) -> None:
    """A Stop, a failed turn and a lost lease each end the stream mid-step on
    an `error` frame. Studio would show the step's line pending for good,
    since nothing reports on its `step_id` on this stream again, so the
    owner closes it as failed first, on the error's own sentence. A line
    already timed stays as sent, `seq` keeps rising, and a lost lease
    leaves the turn open for the instance that holds it now."""
    conversation = ConversationState(conversation_id=CONVERSATION, trace_id="tr")
    conversation.begin_turn()
    sent: list[BaseModel] = [
        _running(conversation, "s0"),
        _running(conversation, "s1", tool="propose_plan", duration_ms=5),
    ]
    if outcome is Outcome.FAILED:
        conversation.close_turn(final_text="", error=ErrorCode.TRANSIENT_UNAVAILABLE.value)
        sent.append(
            ErrorFrame(
                event="error",
                seq=conversation.next_seq(),
                error=ToolError(
                    error=ErrorCode.TRANSIENT_UNAVAILABLE,
                    retry=RetryMode.RETRY_AFTER_DELAY,
                    detail="The fetch did not finish.",
                    retry_after_seconds=30,
                ),
            )
        )

    frames = await _owned(_ScriptedWork(sent, outcome), conversation, _LandingStore())

    assert [(f["event"], f.get("step_id"), f.get("status")) for f in frames] == [
        ("activity.step", "s0", "running"),
        ("activity.step", "s1", "running"),
        ("activity.step", "s0", "failed"),
        ("error", None, None),
    ]
    closed, error = frames[2], frames[3]["error"]
    assert closed["detail"] == error["detail"]
    assert closed["duration_ms"] >= 0
    assert [f["seq"] for f in frames] == sorted({f["seq"] for f in frames})
    match outcome:
        case Outcome.CANCELLED:
            assert error["error"] == "cancelled"
            assert closed["detail"] == classed_sentence("cancelled", "do_not_retry")
        case Outcome.LOST:
            assert error["error"] == "transient_unavailable"
            assert closed["detail"] == classed_sentence(
                "transient_unavailable", "retry_after_delay"
            )
            assert not conversation.turn_recorded
        case _:
            assert closed["detail"] == "The fetch did not finish."
    if outcome is not Outcome.FAILED:
        assert is_user_sentence(closed["detail"])


async def test_plan_approved_goes_out_with_no_write_before_the_closing_write(
    spans: Spans,
) -> None:
    """The owner sends `plan.approved` as it sends any frame and makes no
    write for it: the segment's one store write is the closing write."""
    conversation = ConversationState(conversation_id=CONVERSATION, trace_id="tr")
    conversation.begin_turn()
    store = _LandingStore()
    approved = PlanApproved(
        event="plan.approved",
        seq=conversation.next_seq(),
        plan_version_id="pv",
        version=1,
        title="t",
        steps=[],
        choices=[],
    )
    delta = MessageDelta(event="message.delta", seq=conversation.next_seq(), text="next")
    conversation.close_turn(final_text="next")
    done = Done(event="done", seq=conversation.next_seq(), trace_id="tr")
    work = _ScriptedWork([approved, delta, done], Outcome.DONE)

    frames = await _owned(work, conversation, store)

    assert [frame["event"] for frame in frames] == ["plan.approved", "message.delta", "done"]
    assert store.writes == ["close"]
    assert [error for error, _ in spans.ended] == [Status.DONE.value]


async def test_an_owner_that_fails_numbers_its_transient_frame_past_every_frame_sent(
    spans: Spans,
) -> None:
    """The work raises, and so does its fatal frame, a fault in the owner
    itself. The owner ends the stream on the transient frame, numbered past
    the frames the work numbered as the owner translated them."""
    conversation = ConversationState(conversation_id=CONVERSATION, trace_id="tr")
    conversation.begin_turn()

    class Numbering(_ScriptedWork):
        async def frames(self, event: object) -> list[BaseModel]:
            return [cast(BaseModel, event).model_copy(update={"seq": conversation.next_seq()})]

    work = Numbering(
        [_running(conversation, "s0"), _running(conversation, "s1")],
        Outcome.DONE,
        raises=RuntimeError("the segment failed"),
    )
    conversation.seq = 0

    frames = await _owned(work, conversation, _LandingStore())

    assert [frame["event"] for frame in frames] == [
        "activity.step",
        "activity.step",
        "activity.step",
        "activity.step",
        "error",
    ]
    assert frames[-1]["error"]["error"] == "transient_unavailable"
    assert [frame["seq"] for frame in frames] == [0, 1, 2, 3, 4]
    assert [error for error, _ in spans.ended] == [Status.UNCLOSED.value]


async def test_the_response_sends_keepalive_comments_while_the_next_frame_is_slow() -> None:
    """The response sends `: ka` comment lines while the next frame is slow.
    They are SSE comments outside the frame sequence and carry no `seq`, and
    the first frame goes out undelayed."""
    channel: Channel = asyncio.Queue()

    async def owner() -> None:
        channel.put_nowait(MessageDelta(event="message.delta", seq=0, text="one"))
        await asyncio.sleep(0.12)
        channel.put_nowait(MessageDelta(event="message.delta", seq=1, text="two"))
        channel.put_nowait(None)

    response = StudioStream(
        channel=channel,
        owner=asyncio.create_task(owner()),
        gone=asyncio.Event(),
        keepalive_s=0.05,
    )
    lines = [str(line) async for line in response.body_iterator]

    assert [line for line in lines if line.startswith("data: ")] == [
        line for line in lines if not line.startswith(": ka")
    ]
    assert len([line for line in lines if line.startswith("data: ")]) == 2
    assert [line for line in lines if line.startswith(": ka")]
    assert lines[0].startswith("data: ")
