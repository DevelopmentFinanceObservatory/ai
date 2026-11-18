"""The turn loop: run the node a turn's cursor stands at, and the nodes its
edges lead to, until the turn pauses or ends, writing the cursor at every
node boundary.

`drive` runs nodes. At each boundary it writes, in one `TurnStore` call, the
node the turn moves to and that node's input, the state with its spec
blocks, and the transcript records appended since the last write; a pause
writes the question and an end closes the turn. Only then does it run the
node's `after_commit` work and emit the event that reports the boundary, so
a client never sees a step the store does not hold. The next node takes its
input as the cursor stored it.

A node that raises, a boundary write that fails for any reason but a lost
lease, a moved cursor or an answer to no open pause, and an owed answer the
store lacks each close the turn `failed`, with `failure`'s record of the
exception as its error. A cursor naming a node the running `Graph` lacks, or
paused at a node that cannot resume, closes the turn `abandoned` with
a `TurnExpired` error, before any answer to it is accepted. No turn is
carried across a graph change. A close that cannot be written is logged and
leaves the turn running or paused as last written, for a Stop or the
conversation's next turn to close. The segment ends `unclosed` and reports
the error (`TurnFailed`).

A segment runs a node only on a cursor that is current (`TurnStore.load`).
A crash leaves the cursor as last written. The segment that takes the turn
over runs the node at the cursor from its start, with the state, specs and
transcript as of that write. It runs `run`, or `resume` with the answer the
turn accepted when the cursor owes it (`Cursor.owed`).

`run_segment` runs `drive` under the conversation's lease
(`StudioStore.renew`), renewing it every `HEARTBEAT_INTERVAL_S`. The renewal
reports the Stop live for the turn, posted from any instance
(`StudioStore.request_cancel`, `harness.studio_store.StopState`). While a
node holds the turn's `StopScope` armed, the first Stop ends only the
scope's work, and the turn's next pause write records it taken, which ends
it. Any other Stop cancels the node and closes the turn `cancelled`. A lost
lease cancels the node and leaves the cursor to the instance that holds the
lease. Either way the segment then reads the turn back, and reports a
close or a pause that committed before the cut as the store holds it. A
pause the segment wrote is reported (`Paused`) and kept, whichever cut came
after it.

The caller may hand `run_segment` an event that makes the heartbeat renew
at once (`renew_now`). A Stop posted on the instance that runs the segment
then reaches the node without waiting for the next beat.

A turn no segment runs is closed without running its node: `stop_turn`
closes it `cancelled` on a Stop that found no live segment (the turn
paused, or left running by a segment whose lease lapsed), `reject` closes a
paused turn `done` on the user's reject of its open pause, and `abandon`
closes an open turn `abandoned` when the conversation's next turn opens.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from compiler.surface import transient

from harness.studio_state import HEARTBEAT_INTERVAL_S, SegmentLost, _wire_id
from harness.turn import transcript as entries
from harness.turn.events import NodeDone, Paused, TurnEvent, TurnFailed
from harness.turn.node import Next, Node, NodeCtx, Pause, StopScope, TurnKey
from harness.turn.store import (
    CLOSED,
    Cursor,
    CursorStale,
    PauseNotOpen,
    TurnStatus,
    TurnStore,
    require_open_pause,
)
from harness.turn.transcript import Transcript

if TYPE_CHECKING:
    from harness.studio_store import StudioStore

logger = logging.getLogger(__name__)

# The edge target that ends the turn.
END = "end"


class TurnExpired(Exception):  # noqa: N818 -- names the turn's state, as the cursor does
    """The cursor names a node the running graph lacks, or stands paused at
    a node with no `resume`."""


class TurnAbandoned(Exception):  # noqa: N818 -- names the turn's state, as the cursor does
    """The conversation's next turn opened while this one was open."""


class NoEdge(Exception):  # noqa: N818 -- names the missing edge
    """A node returned an edge label its graph has no edge for."""


@dataclass(frozen=True)
class Graph[E]:
    """The nodes by name, the edges by `(node, label)`, and the node a turn
    opens at. Every edge leaves a node and reaches a node or `END`. Every
    edge a node names in `Node.stop_edges` reaches a node that pauses (one
    with a `resume`), so the Stop its `StopScope` took is recorded by that
    node's pause. `E` is the type of the events the nodes emit beside the
    loop's own."""

    start: str
    nodes: Mapping[str, Node[E]]
    edges: Mapping[tuple[str, str], str]

    def __post_init__(self) -> None:
        if self.start not in self.nodes:
            raise ValueError(f"start node {self.start!r} is not a node")
        for (source, label), target in self.edges.items():
            if source not in self.nodes or (target != END and target not in self.nodes):
                raise ValueError(f"edge {source!r} -{label}-> {target!r} joins an unknown node")
        for name, node in self.nodes.items():
            for label in node.stop_edges:
                target = self.edges.get((name, label))
                if target is None or target == END or self.nodes[target].resume is None:
                    raise ValueError(
                        f"stop edge {name!r} -{label}-> {target!r} reaches no node that pauses"
                    )


class Outcome(StrEnum):
    """How a segment ended: the turn's closing status, `paused`, `lost`
    when another instance took the conversation's lease, or `unclosed` when
    a node failed and the turn's close could not be written, which leaves
    the turn running or paused as last written."""

    PAUSED = "paused"
    UNCLOSED = "unclosed"
    DONE = TurnStatus.DONE.value
    FAILED = TurnStatus.FAILED.value
    ABANDONED = TurnStatus.ABANDONED.value
    CANCELLED = TurnStatus.CANCELLED.value
    LOST = "lost"


@dataclass(frozen=True)
class Answer:
    """The user's answer `value` to the turn's open pause `(interrupt_id,
    round)`."""

    interrupt_id: str
    round: int
    value: dict[str, Any]


@dataclass(frozen=True)
class _Moving:
    """A move or end the segment `segment_id` is writing, from the cursor at
    `version`, to `following` (`END` for the end), and the `NodeDone` that
    reports it."""

    done: NodeDone
    version: int
    following: str
    segment_id: str

    def committed(self, stored: Cursor) -> bool:
        """Whether `stored`, the turn as read back, holds this write: the
        one write after `version`, by this segment (`Cursor.written_by`), at
        `following` or closed `done`. A write another segment made at the
        same version, after a lost lease, is not this one."""
        if stored.version != self.version + 1 or stored.written_by != self.segment_id:
            return False
        if self.following == END:
            return stored.status is TurnStatus.DONE
        return stored.status is TurnStatus.RUNNING and stored.node == self.following


@dataclass
class Segment:
    """One run of the loop under one lease: the cursor as last written, how
    the run ended (`None` while it runs), the `Paused` event of a pause the
    segment is writing and has not yet reported, the move or end it is
    writing and has not yet reported (`moving`), and what undoes the
    pausing node's work after its commit when a cut closes the turn
    (`NodeCtx.on_cut_close`)."""

    cursor: Cursor
    outcome: Outcome | None = None
    pausing: Paused | None = None
    moving: _Moving | None = None
    cut_closes: list[Callable[[], None]] = field(default_factory=list)


# The attributes an exception sets to name the MCP call whose pinned timeout
# it hit (`harness.mcp_client`).
_TIMEOUT_ATTRIBUTES = ("timeout_tool", "timeout_key")


def failure(node: str, error: Exception) -> dict[str, Any]:
    """The error a turn closed on `error` at `node` records: `node`, the
    exception's `type` and `message`, whether a later attempt may not meet
    it (`transient`, `compiler.surface.transient`), the `refusal` it states
    when it states one, and `timeout_tool` and `timeout_key` when it hit an
    MCP call's pinned timeout. These are what the Studio error frame reads
    off the exception (`harness.studio_translate._stream_fatal`), so a
    reader of the closed turn with no exception at hand words the same
    frame."""
    recorded: dict[str, Any] = {
        "node": node,
        "type": type(error).__name__,
        "message": str(error),
        "transient": transient(error),
    }
    refusal = getattr(error, "refusal", None)
    if isinstance(refusal, Mapping) and refusal:
        recorded["refusal"] = dict(refusal)
    for name in _TIMEOUT_ATTRIBUTES:
        value = getattr(error, name, None)
        if isinstance(value, str):
            recorded[name] = value
    return recorded


async def _close_on_error(
    store: TurnStore,
    segment: Segment,
    *,
    key: TurnKey,
    segment_id: str,
    status: TurnStatus,
    node: str,
    error: Exception,
    emit: Callable[[TurnEvent], None],
) -> Segment:
    """Close the turn `status` on `error` at `node` and report `error`. A
    close the store refuses for any reason but those `_NOT_THE_TURNS` names
    is logged and leaves the turn as last written, running or paused, for a
    Stop or the conversation's next turn to close. The segment ends
    `unclosed` and `error` is reported. A close that fails while a cut is
    pending raises the cut (`_cut`), which the segment settles on."""
    try:
        segment.cursor = await store.finish(
            segment.cursor, segment_id=segment_id, status=status, error=failure(node, error)
        )
    except _NOT_THE_TURNS:
        raise
    except Exception as exc:
        _cut(exc)
        logger.exception("turn %s: closing the turn %s at node %s failed", key, status, node)
        segment.outcome = Outcome.UNCLOSED
        emit(TurnFailed(node=node, error=error))
        return segment
    segment.outcome = Outcome(status.value)
    emit(TurnFailed(node=node, error=error))
    return segment


async def _close_on_taken_stop(
    store: TurnStore, segment: Segment, *, key: TurnKey, segment_id: str
) -> Segment:
    """Close the turn `cancelled` on the Stop its `StopScope` took, whose
    pause never came, without writing the node's outcome. A close the store
    refuses for any reason but those `_NOT_THE_TURNS` names may still have
    committed (its acknowledgement lost), so the turn is read back as
    stored, as `_settle` does. A turn the store closed keeps that outcome.
    One that is open, or cannot be read, is logged and left running with
    the Stop live, for the segment's closing write to close
    (`StudioStore.close_segment`). The segment ends `unclosed` and emits no
    `TurnFailed`. A close or read that fails while a cut is pending raises
    the cut (`_cut`), which the segment settles on."""
    try:
        segment.cursor = await store.finish(
            segment.cursor, segment_id=segment_id, status=TurnStatus.CANCELLED
        )
    except _NOT_THE_TURNS:
        raise
    except Exception as exc:
        _cut(exc)
        logger.exception("turn %s: closing the turn on the Stop its scope took failed", key)
        try:
            stored = await store.turn(segment.cursor.id)
        except Exception as read_exc:
            _cut(read_exc)
            logger.exception("turn %s: reading the turn after its close failed", key)
            segment.outcome = Outcome.UNCLOSED
            return segment
        segment.cursor = stored
        segment.outcome = (
            Outcome(stored.status.value) if stored.status in CLOSED else Outcome.UNCLOSED
        )
        return segment
    segment.outcome = Outcome.CANCELLED
    return segment


async def _after_commit(ctx: NodeCtx[Any], node: str) -> None:
    """Run the node's `after_commit` work. A failure is logged. The write
    before it holds the node's outcome, and a crash between the two loses
    the work the same way."""
    for work in ctx.committed:
        try:
            await work()
        except Exception:
            logger.exception("turn node %s: work after its commit failed", node)


def _following(graph: Graph[Any], node: str, result: Next | Pause) -> str | None:
    """The node `result` leads to, or `None` for a pause."""
    if isinstance(result, Pause):
        return None
    following = graph.edges.get((node, result.label))
    if following is None:
        raise NoEdge(f"node {node!r} has no edge {result.label!r}")
    return following


# A boundary write fails with one of these when the turn belongs to another
# writer or the answer named no open pause. They pass up and close nothing.
_NOT_THE_TURNS = (SegmentLost, CursorStale, PauseNotOpen)


async def _write_boundary(
    store: TurnStore,
    cursor: Cursor,
    *,
    segment_id: str,
    node: str,
    node_input: Any,
    result: Next | Pause,
    following: str | None,
    interrupt_id: str,
    ctx: NodeCtx[Any],
) -> Cursor:
    """The write that records `node`'s `result`: a pause under
    `interrupt_id`, the turn's end, or the move to `following`. A pause
    records the Stop the turn's `StopScope` took (`StopScope.taken`), which
    ends that Stop."""
    if isinstance(result, Pause):
        ctx.transcript.append(
            entries.pause(
                kind=result.kind,
                interrupt_id=interrupt_id,
                round=cursor.round + 1,
                payload=result.payload,
                message=result.message,
            )
        )
        return await store.pause(
            cursor,
            segment_id=segment_id,
            node=node,
            node_input=node_input,
            kind=result.kind,
            interrupt_id=interrupt_id,
            payload=result.payload,
            state=ctx.state,
            records=ctx.transcript.take(),
            stop_taken_id=ctx.stop.taken,
        )
    if following == END:
        return await store.finish(
            cursor,
            segment_id=segment_id,
            status=TurnStatus.DONE,
            state=ctx.state,
            records=ctx.transcript.take(),
        )
    assert following is not None
    return await store.advance(
        cursor,
        segment_id=segment_id,
        node=following,
        node_input=result.output,
        state=ctx.state,
        records=ctx.transcript.take(),
    )


class _Closes(Exception):  # noqa: N818 -- carries the close, not an error kind
    """`_enter` found the turn cannot go on. The turn closes with `status`
    and `error`."""

    def __init__(self, status: TurnStatus, error: Exception) -> None:
        super().__init__(str(error))
        self.status = status
        self.error = error


async def _enter[E](
    store: TurnStore,
    segment: Segment,
    transcript: Transcript,
    *,
    key: TurnKey,
    graph: Graph[E],
    segment_id: str,
    answer: Answer | None,
) -> tuple[Node[E], dict[str, Any] | None]:
    """The node at the cursor and the answer it owes its `resume`, if any:
    `answer`, accepted here, or the answer the cursor owes from before a
    crash. Whether the running graph can run or resume the node is checked
    before `answer` is accepted, so a turn that expires stores no answer.

    Raises:
        PauseNotOpen: `answer` names a pause the turn does not have open.
        _Closes: the running graph cannot run or resume the node
            (`abandoned`), accepting the answer failed (`failed`), or the
            owed answer is not stored (`failed`).
    """
    cursor = segment.cursor
    node = cursor.node
    if answer is not None:
        require_open_pause(cursor, interrupt_id=answer.interrupt_id, round=answer.round)
    resuming = answer is not None or cursor.owed is not None
    target = graph.nodes.get(node)
    if target is None or (resuming and target.resume is None):
        expired = TurnExpired(
            f"the running graph cannot {'resume' if resuming else 'run'} {node!r}"
        )
        logger.warning("turn %s expired: %s", key, expired)
        raise _Closes(TurnStatus.ABANDONED, expired)

    owed: dict[str, Any] | None = None
    if answer is not None:
        transcript.append(
            entries.answer(interrupt_id=answer.interrupt_id, round=answer.round, value=answer.value)
        )
        try:
            segment.cursor = await store.record_answer(
                cursor,
                segment_id=segment_id,
                interrupt_id=answer.interrupt_id,
                round=answer.round,
                answer=answer.value,
                records=transcript.take(),
            )
        except _NOT_THE_TURNS:
            raise
        except Exception as exc:
            _cut(exc)
            logger.exception("turn %s: accepting the answer at node %s failed", key, node)
            raise _Closes(TurnStatus.FAILED, exc) from exc
        owed = answer.value
    elif cursor.owed is not None:
        owed = await store.answer(
            cursor, interrupt_id=cursor.owed.interrupt_id, round=cursor.owed.round
        )
        if owed is None:
            missing = RuntimeError(
                f"turn {cursor.turn_index} owes {node!r} the answer to "
                f"{cursor.owed.interrupt_id!r} round {cursor.owed.round}, which is not stored"
            )
            logger.error("turn %s: %s", key, missing)
            raise _Closes(TurnStatus.FAILED, missing)
    return target, owed


def _cut(error: Exception | None = None) -> None:
    """Raise the cancellation the running task was sent and has not taken
    back (`asyncio.Task.cancelling`), so the segment settles on its cut
    (`run_segment`) when the node or its work after the commit caught the
    cancellation and returned, or raised `error` in its place."""
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError from error


async def drive[E](
    store: TurnStore,
    segment: Segment,
    *,
    key: TurnKey,
    graph: Graph[E],
    segment_id: str,
    emit: Callable[[E | TurnEvent], None],
    stop: StopScope,
    answer: Answer | None = None,
) -> Segment:
    """Run the turn from where `segment.cursor` stands until it pauses or
    ends, writing each boundary under `segment_id`'s lease, and return
    `segment` with its last cursor and outcome.

    `emit` hears the loop's own `TurnEvent`s and every event of the graph's own
    type `E` its nodes emit, in the order they happen.

    The turn and its transcript are read in one snapshot that also checks
    the cursor is still current (`TurnStore.load`), so the caller reads the
    cursor after claiming the lease and no node runs on a cursor another
    write has moved or closed. A paused cursor takes `answer`, which the
    first write accepts (`TurnStore.record_answer`) before the paused node's
    `resume` runs. A running cursor runs its node from its start.

    A write that fails for any reason but those `_NOT_THE_TURNS` names (a
    value JSON cannot hold, a database error) closes the turn `failed` at
    the node whose outcome it was writing.

    A Stop the turn's `StopScope` took (`StopScope.taken`) must be recorded
    by a pause of the node that took it, or of the node one of its stop
    edges leads to (`Node.stop_edges`, which `Graph` checks reach a node
    that pauses). When the node that took it ends the turn or leaves on
    another edge, or the node after it returns anything but a pause, the
    turn closes `cancelled` in place of that node's write.

    A cut (a Stop that ends the turn, a lost lease, a client that left)
    reaches the node as a cancellation of the task running `drive`. The
    cancellation is raised again once the node returns or raises, once a
    boundary write or the answer's acceptance raises, and once the node's
    work after the commit has run, so work that catches it neither writes
    the node's result, closes the turn `failed`, nor leads to another node,
    and the segment settles on the cut as when the cancellation escapes
    (`run_segment`). A cut that lands once a move or end committed is
    settled on the turn as stored, which reports that boundary's `NodeDone`
    (`_settle`).

    Raises:
        ValueError: the cursor is paused and no `answer` was given.
        PauseNotOpen: `answer` names a pause the turn does not have open.
        SegmentLost: `segment_id` does not hold the conversation's lease.
        CursorStale: another writer moved the cursor, or the turn is closed.
    """
    cursor = segment.cursor
    if cursor.status in CLOSED:
        raise CursorStale(f"turn {cursor.turn_index} is {cursor.status.value}")
    if cursor.status is TurnStatus.PAUSED and answer is None:
        raise ValueError(f"turn {cursor.turn_index} is paused and needs an answer")
    cursor, stored = await store.load(cursor)
    segment.cursor = cursor
    transcript = Transcript(stored)
    node, node_input = cursor.node, cursor.node_input

    async def close(status: TurnStatus, node: str, error: Exception) -> Segment:
        return await _close_on_error(
            store,
            segment,
            key=key,
            segment_id=segment_id,
            status=status,
            node=node,
            error=error,
            emit=emit,
        )

    try:
        target, owed = await _enter(
            store, segment, transcript, key=key, graph=graph, segment_id=segment_id, answer=answer
        )
    except _Closes as closing:
        return await close(closing.status, node, closing.error)
    cursor = segment.cursor

    while True:
        # The node works on a copy, so whatever it changes in place, the
        # cursor keeps the state as stored until a write stores the node's.
        ctx = NodeCtx[E](
            key=key,
            inputs=cursor.inputs,
            state=copy.deepcopy(cursor.state),
            transcript=transcript,
            emit=emit,
            stop=stop,
        )
        taken_before = stop.taken
        try:
            if owed is not None and target.resume is not None:
                result = await target.resume(ctx, node_input, owed)
            else:
                result = await target.run(ctx, node_input)
            following = _following(graph, node, result)
        except Exception as exc:
            _cut(exc)
            logger.exception("turn %s: node %s failed", key, node)
            return await close(TurnStatus.FAILED, node, exc)
        _cut()
        owed = None
        if (
            stop.taken is not None
            and not isinstance(result, Pause)
            and (
                stop.taken == taken_before
                or following == END
                or result.label not in target.stop_edges
            )
        ):
            # The scope's Stop is recorded by a pause of the node that took it
            # or of the node one of its stop edges leads to. A turn that
            # leaves another way, ends, or goes past that node is stopped.
            return await _close_on_taken_stop(store, segment, key=key, segment_id=segment_id)

        interrupt_id = _wire_id("int")
        if isinstance(result, Pause):
            segment.pausing = Paused(
                kind=result.kind,
                interrupt_id=interrupt_id,
                round=cursor.round + 1,
                payload=result.payload,
                message=result.message,
            )
            segment.cut_closes = ctx.cut_closes
        else:
            assert following is not None
            segment.moving = _Moving(
                done=NodeDone(node=node, label=result.label, output=result.output),
                version=cursor.version,
                following=following,
                segment_id=segment_id,
            )
        try:
            cursor = segment.cursor = await _write_boundary(
                store,
                cursor,
                segment_id=segment_id,
                node=node,
                node_input=node_input,
                result=result,
                following=following,
                interrupt_id=interrupt_id,
                ctx=ctx,
            )
        except _NOT_THE_TURNS:
            raise
        except Exception as exc:
            # The write may have committed, which the cut's settlement
            # reads back and reports (`_settle`).
            _cut(exc)
            segment.pausing, segment.moving = None, None
            logger.exception("turn %s: the write after node %s failed", key, node)
            return await close(TurnStatus.FAILED, node, exc)
        await _after_commit(ctx, node)
        # A cut that lands from the write on is settled on the turn as
        # stored, which reports the boundary the write committed (`_settle`).
        _cut()

        if segment.pausing is not None:
            segment.outcome = Outcome.PAUSED
            emit(segment.pausing)
            segment.pausing, segment.cut_closes = None, []
            return segment
        assert segment.moving is not None
        emit(segment.moving.done)
        segment.moving = None
        if following == END:
            segment.outcome = Outcome.DONE
            return segment
        assert following is not None
        # The next node takes its input as stored, as a rerun reads it back.
        node, node_input, target = following, cursor.node_input, graph.nodes[following]


async def stop_turn(store: TurnStore, cursor: Cursor, *, segment_id: str) -> Cursor:
    """Close the open turn at `cursor` `cancelled` without running its node,
    writing under `segment_id`, the conversation lease the caller claimed.
    It serves the Stop of a turn no segment runs, paused on a question or
    left running by a segment whose lease lapsed. A paused turn's pause is
    dropped, so an answer to it is refused (`PauseNotOpen`).

    Raises:
        SegmentLost: `segment_id` does not hold the conversation's lease.
        CursorStale: another writer moved the cursor, or closed the turn.
    """
    return await store.finish(cursor, segment_id=segment_id, status=TurnStatus.CANCELLED)


async def reject(store: TurnStore, cursor: Cursor, *, segment_id: str, answer: Answer) -> Cursor:
    """Close the paused turn at `cursor` `done` on the user's reject of its
    open pause, writing under `segment_id`, the conversation lease the
    caller claimed. The paused node does not run. `answer` joins the
    turn's transcript in the same write.

    Raises:
        PauseNotOpen: `answer` names a pause the turn does not have open.
        SegmentLost: `segment_id` does not hold the conversation's lease.
        CursorStale: another writer moved the cursor, or closed the turn.
    """
    require_open_pause(cursor, interrupt_id=answer.interrupt_id, round=answer.round)
    return await store.finish(
        cursor,
        segment_id=segment_id,
        status=TurnStatus.DONE,
        records=[
            entries.answer(interrupt_id=answer.interrupt_id, round=answer.round, value=answer.value)
        ],
    )


async def abandon(store: TurnStore, cursor: Cursor, *, segment_id: str) -> Cursor:
    """Close the open turn at `cursor` `abandoned` without running its node,
    writing under `segment_id`, the conversation lease the caller claimed to
    open the conversation's next turn. Its error is a `TurnAbandoned` at the
    node it stands at. A paused turn's pause is dropped.

    Raises:
        SegmentLost: `segment_id` does not hold the conversation's lease.
        CursorStale: another writer moved the cursor, or closed the turn.
    """
    left = TurnAbandoned(
        f"turn {cursor.turn_index} was left {cursor.status.value} at {cursor.node!r} "
        "when the next turn opened"
    )
    return await store.finish(
        cursor,
        segment_id=segment_id,
        status=TurnStatus.ABANDONED,
        error=failure(cursor.node, left),
    )


async def _next_beat(seconds: float, renew_now: asyncio.Event | None) -> None:
    """Wait `seconds`, or until `renew_now` is set, which it clears."""
    if renew_now is None:
        await asyncio.sleep(seconds)
        return
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(renew_now.wait(), seconds)
    renew_now.clear()


async def _settle(
    store: TurnStore,
    segment: Segment,
    *,
    cut_by: Outcome,
    segment_id: str,
    emit: Callable[[TurnEvent], None],
) -> None:
    """Set `segment`'s cursor and outcome after the heartbeat cut its node
    off for a Stop or a lost lease. The node may have been inside a
    boundary's write, which may have committed, so the turn is read back
    as stored. A turn the store closed keeps that outcome. A Stop closes
    any other `cancelled`, a pause the segment wrote included, since the
    user stopped the turn and the work after the pause's write was cut off
    as a Stop. After a lost lease a pause the segment was writing that
    committed keeps the turn paused, and its `Paused` is emitted here when
    the cut came before `drive` emitted it, so the client gets the
    question the store holds. A turn still running is left to the instance
    that holds the lease. A move or end the segment was writing that
    committed has its `NodeDone` emitted here when the cut came before
    `drive` emitted it, whether the cut came while the write's
    acknowledgement was awaited, during the node's work after the commit,
    or after it. When a Stop closes a turn whose pause the segment wrote,
    what undoes the pausing node's work after its commit runs
    (`NodeCtx.on_cut_close`), so nothing it started goes on for the closed
    turn."""
    stored = await store.turn(segment.cursor.id)
    pausing, segment.pausing = segment.pausing, None
    moving, segment.moving = segment.moving, None
    cut_closes, segment.cut_closes = segment.cut_closes, []
    if moving is not None and moving.committed(stored):
        emit(moving.done)
    paused_here = (
        pausing is not None
        and stored.pause is not None
        and stored.pause.interrupt_id == pausing.interrupt_id
    )
    if stored.status in CLOSED:
        segment.cursor, segment.outcome = stored, Outcome(stored.status.value)
        return
    if cut_by is Outcome.LOST and stored.status is TurnStatus.PAUSED:
        segment.cursor, segment.outcome = stored, Outcome.PAUSED
        if paused_here:
            assert pausing is not None
            emit(pausing)
        return
    segment.cursor, segment.outcome = stored, cut_by
    if cut_by is Outcome.CANCELLED:
        segment.cursor = await store.finish(
            stored, segment_id=segment_id, status=TurnStatus.CANCELLED
        )
        if paused_here:
            for undo in cut_closes:
                try:
                    undo()
                except Exception:
                    logger.exception("turn %s: undoing a paused node's work failed", stored.id)


async def run_segment[E](
    store: TurnStore,
    lease: StudioStore,
    cursor: Cursor,
    *,
    principal: str,
    conversation_id: str,
    segment_id: str,
    graph: Graph[E],
    emit: Callable[[E | TurnEvent], None],
    answer: Answer | None = None,
    heartbeat_seconds: float = HEARTBEAT_INTERVAL_S,
    renew_now: asyncio.Event | None = None,
) -> Segment:
    """`drive` the turn at `cursor` under `segment_id`, the conversation
    lease the caller claimed, renewing it every `heartbeat_seconds`, or at
    once whenever `renew_now` is set, and return the segment once the turn
    pauses or ends, a Stop closes it, or the lease is lost.

    A renewal reports the Stop live for the turn (`Renewal.stop_id`). While
    a node holds the turn's `StopScope` armed, the first Stop it reports
    sets the scope's event and the scope keeps its id (`StopScope.take`),
    with no write. The node goes on, and the turn's next pause write records
    the id taken (`TurnStore.pause`), so renewals stop reporting it. That
    pause is the node's own or that of the node one of its stop edges leads
    to (a switch's offer, `Node.stop_edges`). A turn that leaves another
    way, ends, or goes further without it closes `cancelled` (`drive`).
    Until then the Stop stays live, and a client that leaves first leaves
    it to the segment's closing write. Any other Stop cancels the node,
    leaves no task of the segment running, and closes the turn `cancelled`.
    One a renewal reads after the drive finished is left to the segment's
    closing write, which acts on a Stop still live
    (`StudioStore.close_segment`). A renewal that finds the lease
    taken, or that keeps failing until the lease's deadline
    (`StudioStore.lease_deadline`, dated from the claim or the last renewal
    that landed), cancels the node and ends the segment `lost` without
    writing the cursor. After either cut the turn is read back (`_settle`),
    so a close or pause that committed before the cut is reported as
    stored.

    `cursor` must be read after the caller claimed the lease. `drive`
    refuses one another write has moved since (`CursorStale`).

    The node's work tells the two cuts apart through the scope. A Stop marks
    it (`StopScope.mark_turn_stopped`) before cancelling the node, so the
    work ends what it started (an executor run is recorded failed), and a
    lost lease leaves it unmarked, so that work is left for the instance
    that holds the lease now.

    Cancelling the caller (a client that left) cancels the node with the
    scope unmarked and leaves the cursor as last written, for a later
    segment to take over: the conversation's next `/turn`, which abandons
    the turn and carries its unfinished runs on, or a retried `/interrupt`,
    which runs the node at the cursor again.

    Raises:
        What `drive` raises, except `SegmentLost`, which ends the segment
        `lost`.
    """
    segment = Segment(cursor=cursor)
    stop = StopScope()
    key = TurnKey(principal, conversation_id, cursor.turn_index)
    task = asyncio.create_task(
        drive(
            store,
            segment,
            key=key,
            graph=graph,
            segment_id=segment_id,
            emit=emit,
            stop=stop,
            answer=answer,
        )
    )
    # Why the heartbeat cancelled the node, when it did.
    ended: list[Outcome] = []

    def end(outcome: Outcome) -> None:
        if outcome is Outcome.CANCELLED:
            stop.mark_turn_stopped()
        if task.cancel():
            ended.append(outcome)

    async def heartbeat() -> None:
        while True:
            await _next_beat(heartbeat_seconds, renew_now)
            try:
                renewal = await lease.renew(principal, conversation_id, segment_id)
            except Exception:
                logger.exception("turn %s: lease renewal failed", key)
                if lease.now() < lease.lease_deadline(principal, conversation_id):
                    continue
                end(Outcome.LOST)
                return
            if not renewal.held:
                end(Outcome.LOST)
                return
            if renewal.stop_id is None or renewal.stop_id == stop.taken:
                continue
            if stop.take(renewal.stop_id):
                continue
            if task.done():
                # The turn's last write landed before this renewal read the
                # Stop: the segment's closing write acts on it.
                return
            end(Outcome.CANCELLED)
            return

    beat = asyncio.create_task(heartbeat())
    try:
        return await task
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if not ended or (current is not None and current.cancelling()):
            raise
        segment.outcome = Outcome.LOST
        try:
            await _settle(store, segment, cut_by=ended[0], segment_id=segment_id, emit=emit)
        except SegmentLost:
            segment.outcome = Outcome.LOST
        except Exception:
            # A lease lost to failing renewals usually means the database
            # is unreachable, so the read can fail too. The segment is lost.
            if ended[0] is not Outcome.LOST:
                raise
            logger.exception("turn %s: reading the turn after losing the lease failed", key)
        return segment
    except SegmentLost:
        segment.outcome = Outcome.LOST
        return segment
    finally:
        beat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await beat
