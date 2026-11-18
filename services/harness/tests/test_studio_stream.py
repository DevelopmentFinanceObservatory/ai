"""How a Studio stream's owner ends a segment (`harness.studio_stream`).

`test_every_ending_is_the_tables` pins `ending`, cell by cell.

`test_every_run_of_the_owner_keeps_the_invariants` runs the real owner and
the real response class over an in-memory model of the store and of a
segment's work, for every script, input, place the input lands, and fault of
the closing write, and checks eight invariants after each run:

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

The response runs its real ASGI call over an in-memory transport that
delivers `http.disconnect` when the client leaves, as uvicorn's protocols do
at ASGI spec 2.3.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pytest
from harness import studio_stream
from harness.fatal_codes import wire_error
from harness.studio_state import ConversationState, SegmentLost, _PendingInterrupt
from harness.studio_store import SegmentClose, closed_turn_error
from harness.studio_stream import (
    Finish,
    Status,
    StudioStream,
    WorkEnd,
    ending,
    own_segment,
)
from harness.studio_translate import recorded_fatal
from harness.tool_contracts import HarnessCode, harness_refusal
from harness.turn.loop import Outcome
from harness.turn.store import Cursor, TurnStatus
from harness.wire import (
    ActivityStep,
    Done,
    ErrorCode,
    ErrorFrame,
    PlanApproved,
    PlanProposed,
    RetryMode,
    StepStatus,
    ToolError,
)
from pydantic import BaseModel
from starlette.types import Message

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
            pytest.param(Finish.FAILED, TurnStatus.FAILED, error, id=f"failed-{name}")
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
    # same code.
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
    APPROVED = "approved"
    EDIT = "edit"
    RAISES = "raises"


class Point(StrEnum):
    """Where an input lands, at one of the owner's four await points: while
    it waits for the work (early, before the turn's boundary write, or late,
    after it), during the save after `plan.approved`, while it waits for
    cancelled work to stop, and during the closing write; each before and
    after the write or the work's cleanup."""

    WORK_EARLY = "work_early"
    WORK_LATE = "work_late"
    APPROVED_BEFORE = "approved_before"
    APPROVED_AFTER = "approved_after"
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
_APPROVED_POINTS = (Point.APPROVED_BEFORE, Point.APPROVED_AFTER)


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
            if point in _APPROVED_POINTS and script is not Script.APPROVED:
                continue
            if point is Point.WORK_LATE and script in (Script.HANGS, Script.RAISES):
                continue
            if point in _CANCEL_POINTS and (GONE not in inputs or len(inputs) == 1):
                continue
            if script is Script.HANGS and point in (Point.CLOSE_BEFORE, Point.CLOSE_AFTER):
                continue
            cases.append(Case(script, inputs, point, fault))
    return cases


type Durable = tuple[bool, str | None, tuple[tuple[str, str | None, bool], ...], int]


def _durable(conversation: ConversationState) -> Durable:
    """The fields of the conversation the model's store keeps: whether the
    turn is recorded and its error, the pending interrupts, and `seq`."""
    return (
        conversation.turn_recorded,
        conversation.turns[-1].get("error") if conversation.turn_recorded else None,
        tuple(
            sorted(
                (key, pending.resuming_segment, pending.approved_sent)
                for key, pending in conversation.pending.items()
            )
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
    approved_lost: bool = False

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
        """The client closes, and the response sets the gone flag."""
        self.disconnect.set()
        for _ in range(1000):
            if self.gone.is_set():
                return
            await asyncio.sleep(0)
        raise AssertionError("the response never set the client-gone flag")


class FakeWork:
    """A segment's work over the model: sends frames as the translator
    does, writes the turn's boundary, and sees a Stop or a lost lease at
    each of its ticks as the turn loop's heartbeat would."""

    def __init__(self, run: Run) -> None:
        self._run = run
        self._c = run.conversation
        self._w = run.world

    def _step(self, step_id: str, status: StepStatus, *, done: bool = False) -> ActivityStep:
        return ActivityStep(
            event="activity.step",
            seq=self._c.next_seq(),
            step_id=step_id,
            label=step_id,
            status=status,
            duration_ms=5 if done else None,
        )

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

    async def run(self, send: Callable[[BaseModel], None]) -> Outcome:
        self._run.work_task = asyncio.current_task()
        try:
            outcome = await self._script(send)
        except _Cut as cut:
            outcome = cut.outcome
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                await self._run.land(Point.CANCEL_BEFORE)
                await asyncio.sleep(0)
                await self._run.land(Point.CANCEL_AFTER)
                self._run.finish = Finish.LOST if self._run.approved_lost else Finish.LEFT
            else:
                self._run.finish = Finish.UNCLOSED
            raise
        self._run.finish = Finish(outcome.value)
        return outcome

    async def _script(self, send: Callable[[BaseModel], None]) -> Outcome:
        script = self._run.case.script
        c, w = self._c, self._w
        if script is Script.APPROVED:
            send(
                PlanApproved(
                    event="plan.approved",
                    seq=c.next_seq(),
                    plan_version_id="pv",
                    version=1,
                    title="t",
                    steps=[],
                    choices=[],
                )
            )
        send(self._step("s1", StepStatus.RUNNING))
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
                send(self._fatal(ErrorCode.CAPABILITY_UNAVAILABLE))
                await self._tick(Point.WORK_LATE)
                return Outcome.FAILED
            case Script.PAUSES:
                send(self._step("s1", StepStatus.COMPLETE, done=True))
                w.cursor = "paused"
                c.pending["i1"] = _PendingInterrupt(adk_id="i1", kind="plan")
                send(_PAUSE.model_copy(update={"seq": c.next_seq()}))
                await self._tick(Point.WORK_LATE)
                return Outcome.PAUSED
            case Script.EDIT:
                # The edit's second line is still open when `done` comes.
                send(self._step("s2", StepStatus.RUNNING))
            case _:
                send(self._step("s1", StepStatus.COMPLETE, done=True))
        w.cursor = "done"
        c.close_turn(final_text="answer")
        send(_DONE.model_copy(update={"seq": c.next_seq()}))
        await self._tick(Point.WORK_LATE)
        return Outcome.DONE

    def _fatal(self, code: ErrorCode) -> ErrorFrame:
        self._c.close_turn(final_text="", error=code.value)
        return ErrorFrame(
            event="error",
            seq=self._c.next_seq(),
            error=ToolError(error=code, retry=RetryMode.DO_NOT_RETRY, detail="failed"),
        )

    def fatal(self, error: BaseException) -> ErrorFrame:
        del error
        return self._fatal(ErrorCode.CAPABILITY_UNAVAILABLE)


class FakeStore:
    """The owner's writes, over the model."""

    def __init__(self, run: Run) -> None:
        self._run = run
        self._c = run.conversation
        self._w = run.world

    async def checkpoint(self, principal: str, conversation_id: str, segment_id: str) -> None:
        self._w.writes.append("checkpoint")
        await self._run.land(Point.APPROVED_BEFORE)
        if self._w.lease != segment_id:
            self._run.approved_lost = True
            raise SegmentLost(conversation_id)
        self._w.stored = _durable(self._c)
        await self._run.land(Point.APPROVED_AFTER)

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
            world.cursor = "cancelled"
            error = c.turns[-1].get("error") if c.turn_recorded else ErrorCode.CANCELLED.value
            world.stored = (True, error, (), c.seq)
        else:
            world.stored = _durable(c)
        world.lease = None
        world.committed_close = True
        if fault is Fault.ACK_LOST:
            return SegmentClose.FAILED
        if stopped:
            c.pending.clear()
            c.close_turn(final_text=c.turn_final_text, error=ErrorCode.CANCELLED.value)
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

    def start(external_id: str, *, session_id: str | None = None) -> object:
        recorded.started += 1
        return object()

    def end(handle: object, **kwargs: Any) -> None:
        recorded.ended.append((kwargs.get("error"), dict(kwargs.get("attributes") or {})))

    async def flush() -> None:
        return None

    monkeypatch.setattr(studio_stream, "start_turn_span", start)
    monkeypatch.setattr(studio_stream, "end_turn_span", end)
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
    conversation = ConversationState(conversation_id=CONVERSATION, trace_id="tr", seq=FIRST_SEQ)
    if case.script is Script.APPROVED:
        conversation.pending["i0"] = _PendingInterrupt(
            adk_id="i0", kind="plan", resuming_segment=SEGMENT
        )
    await conversation.lock.acquire()
    world = World(stored=_durable(conversation))
    run = Run(case, conversation, world)
    channel: studio_stream.Channel = asyncio.Queue()
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
        return ["the run did not finish"]
    return _broken(run, owner, spans, b"".join(body).decode())


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
                answerable = ("i1", None, False) in pending
                if recorded or not answerable or world.cursor != "paused":
                    broken.append(f"I2: pause sent, stored {world.stored} cursor {world.cursor}")
            case "error" if last["error"]["error"] == ErrorCode.CANCELLED.value:
                if not (
                    world.cursor == "cancelled"
                    or (recorded and error == ErrorCode.CANCELLED.value)
                    or world.stop_live()
                ):
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
        assert run.finish is not None
        _, want = TABLE[run.finish, run.close]
        statuses = [status for status, _ in spans.ended]
        if statuses != [want.value] or spans.started != 1:
            broken.append(f"I4: spans {statuses}, want [{want}] for {run.finish}/{run.close}")
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
    return broken


async def test_every_run_of_the_owner_keeps_the_invariants(spans: Spans) -> None:
    cases = _cases()
    failures: dict[str, list[str]] = {}
    for case in cases:
        if broken := await _one(case, spans):
            failures[str(case)] = broken
    print(f"{len(cases)} runs of the owner")
    assert cases
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
