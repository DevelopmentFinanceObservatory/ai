"""The turn loop: run the node a turn's cursor stands at, and the nodes its edges lead to, until the
turn pauses or ends, writing the cursor at every node boundary.

`drive` runs nodes. At each boundary it writes, in one `TurnStore` call, the node the turn moves to
and that node's input, the state with its spec blocks, and the transcript records appended since the
last write; a pause writes the question and an end closes the turn. Only then does it run the node's
`after_commit` work and emit the event that reports the boundary, so a client never sees a step the
store does not hold. The next node takes its input as the cursor stored it.

A node that raises, a boundary write that did not commit for any reason but a lost lease, a moved
cursor or an answer to no open pause, and an owed answer the store lacks each close the turn
`failed`, with `failure`'s record of the exception as its error. A boundary write, an answer's
acceptance or a close that raised any error but `WriteNotEncoded` is read back first (`_read_back`):
one that committed, its acknowledgement lost, carries on as written, and a turn another write closed
takes the stored outcome. A cursor naming a node the running `Graph` lacks, or paused at a node that
cannot resume, closes the turn `abandoned` with a `TurnExpired` error, before any answer to it is
accepted. No turn is carried across a graph change. A close that cannot be written is logged and
leaves the turn running or paused as last written, for a Stop or the conversation's next turn to
close. The segment ends `unclosed` and reports the error (`TurnFailed`).

A segment runs a node only on a cursor that is current (`TurnStore.load`). A crash leaves the cursor
as last written. The segment that takes the turn over runs the node at the cursor from its start,
with the state, specs and transcript as of that write. It runs `run`, or `resume` with the answer
the turn accepted when the cursor owes it (`Cursor.owed`).

`run_segment` runs `drive` under the conversation's lease (`StudioStore.renew`), renewing it every
`HEARTBEAT_INTERVAL_S`. The renewal reports the Stop live for the turn, posted from any instance
(`StudioStore.request_cancel`, `harness.studio_store.StopState`). While a node holds the turn's
`StopScope` armed, the first Stop ends only the scope's work, and the turn's next pause write
records it taken, which ends it. Any other Stop cancels the node and closes the turn `cancelled`. A
lost lease cancels the node and leaves the cursor to the instance that holds the lease. Either way
the segment then reads the turn back, and reports a close or a pause that committed before the cut
as the store holds it. A pause the segment wrote is reported (`Paused`) and kept, whichever cut came
after it.

The caller may hand `run_segment` an event that makes the heartbeat renew at once (`renew_now`). A
Stop posted on the instance that runs the segment then reaches the node without waiting for the next
beat.

A turn no segment runs is closed without running its node: `stop_turn` closes it `cancelled` on a
Stop that found no live segment (the turn paused, or left running by a segment whose lease lapsed),
`reject` closes a paused turn `done` on the user's reject of its open pause, and `abandon_on` closes
an open turn `abandoned` in the transaction that opens the conversation's next turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, override

from compiler.surface import transient

from harness.conversation import ledger_conversation_id
from harness.studio_state import HEARTBEAT_INTERVAL_S, SegmentLost, _wire_id
from harness.turn import transcript as entries
from harness.turn.events import NodeDone, Paused, TurnEvent, TurnFailed
from harness.turn.node import Next, Node, NodeCtx, Pause, StopScope, TurnKey
from harness.turn.store import (
    CLOSED,
    Cursor,
    CursorStale,
    OwedAnswer,
    PauseNotOpen,
    SegmentLineage,
    TurnStatus,
    TurnStore,
    WriteNotEncoded,
    finish_on,
    require_open_pause,
)
from harness.turn.transcript import Transcript

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

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
    round)`, and `request`, the answer as the client sent it, which the
    turn records beside it (`TurnStore.record_answer`)."""

    interrupt_id: str
    round: int
    value: dict[str, Any]
    request: dict[str, Any] | None = None


@dataclass
class _Owed:
    """The works after its commit that the node `node` asked for
    (`NodeCtx.after_commit`), each run at most once: `started` counts the
    works, in order, that have started. The write before them holds the
    node's outcome, and a crash between the two loses the works the same
    way."""

    ctx: NodeCtx[Any]
    node: str
    started: int = 0

    async def run(self) -> None:
        """Run, in order, each work that has not started yet. A failure is
        logged and the next work runs."""
        works = self.ctx.committed
        while self.started < len(works):
            work = works[self.started]
            self.started += 1
            try:
                await work()
            except Exception:
                logger.exception("turn node %s: work after its commit failed", self.node)


@dataclass(frozen=True)
class _Writing:
    """A boundary write the segment `segment_id` is making from the cursor
    at `version`, the event that reports it (a pause's `Paused`, or a move to
    `following` or the end, `END`, reported by a `NodeDone`), the node's
    work its commit is owed (`owed`), and the cursor it wrote once its commit
    landed (`landed`), which the commit's own task records, a COMMIT whose
    acknowledgement was lost and whose transaction the store read back as
    committed included (`TurnStore.pause`)."""

    reports: Paused | NodeDone
    version: int
    following: str | None
    segment_id: str
    owed: _Owed = field(compare=False, hash=False)
    landed: list[Cursor] = field(default_factory=list, compare=False, hash=False)

    def committed(self, stored: Cursor) -> bool:
        """Whether this write landed: its commit's task recorded it
        (`landed`), or, for a write whose COMMIT raised and whose
        transaction's status the store could not read in time, `stored`, the
        turn as read back, holds it: the one write after `version`, by this
        segment (`Cursor.written_by`), paused on this pause's
        `interrupt_id`, at `following`, or closed `done`. A write another
        segment made at the same version, after a lost lease, is not this
        one."""
        if self.landed:
            return True
        if stored.version != self.version + 1 or stored.written_by != self.segment_id:
            return False
        if isinstance(self.reports, Paused):
            return (
                stored.status is TurnStatus.PAUSED
                and stored.pause is not None
                and stored.pause.interrupt_id == self.reports.interrupt_id
            )
        if self.following == END:
            return stored.status is TurnStatus.DONE
        return stored.status is TurnStatus.RUNNING and stored.node == self.following


@dataclass
class Segment:
    """One run of the loop under one lease: the cursor as last written, how
    the run ended (`None` while it runs), the boundary write the segment
    is making and has not yet reported (`writing`), and what hears each
    pause the segment's writes commit (`paused`, `note_pause`)."""

    cursor: Cursor
    outcome: Outcome | None = None
    writing: _Writing | None = None
    paused: Callable[[Cursor], None] | None = None
    # The round of the last pause handed to `paused`.
    _noted: int | None = field(default=None, repr=False)

    def note_pause(self, stored: Cursor) -> None:
        """Hand `stored`, a pause this segment's write committed, to
        `paused`, once per pause: the pause write's commit hands it over
        (`TurnStore.pause`), and a read back that finds a pause write
        committed whose acknowledgement was lost does (`_after_failed_write`,
        `_settle`). A cursor that is not paused is ignored. A failure is
        logged."""
        if self.paused is None or stored.status is not TurnStatus.PAUSED or stored.pause is None:
            return
        if self._noted == stored.pause.round:
            return
        self._noted = stored.pause.round
        try:
            self.paused(stored)
        except Exception:
            logger.exception("turn %s: the pause's own work failed", stored.turn_index)


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
    """Close the turn `status` on `error` at `node` and report `error`. A close
    that cannot encode its values (`WriteNotEncoded`) never committed and is
    not read back. A close the store refuses for any other reason but those
    `_NOT_THE_TURNS` names may still have committed, its acknowledgement
    lost, so the turn is read back (`_read_back`). A close that committed
    ends as one that returned. A turn another write closed ends on its
    stored outcome, with `error` reported all the same: the node failed,
    whoever closed the turn, while a write that raised and finds the turn
    closed by another reports nothing, since no node failed
    (`_after_failed_write`, `_settle_answer`). A turn still open, or one
    that cannot be read, is left as last written, running or paused, for a
    Stop or the conversation's next turn to close: the segment ends
    `unclosed` and `error` is reported. The close and the read are steps of
    `drive` (`_step`), so a cut pending raises it, which the segment settles
    on."""
    try:
        segment.cursor = await _step(
            store.finish(
                segment.cursor, segment_id=segment_id, status=status, error=failure(node, error)
            )
        )
    except _NOT_THE_TURNS:
        raise
    except WriteNotEncoded as exc:
        logger.error(
            "turn %s: closing the turn %s at node %s failed", key, status, node, exc_info=exc
        )
        segment.outcome = Outcome.UNCLOSED
        emit(TurnFailed(node=node, error=error))
        return segment
    except Exception as exc:
        landed, stored = await _read_back(
            store,
            segment.cursor,
            key=key,
            what=f"closing the turn {status.value} at node {node}",
            error=exc,
            holds=_closes(segment.cursor, segment_id=segment_id, status=status),
        )
        if stored is not None:
            segment.cursor = stored
        if landed is _Landed.CLOSED:
            assert stored is not None
            segment.outcome = Outcome(stored.status.value)
            emit(TurnFailed(node=node, error=error))
            return segment
        if landed is not _Landed.COMMITTED:
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
    committed (its acknowledgement lost), so the turn is read back
    (`_read_back`). A turn the store closed keeps that outcome. One that is
    open, or cannot be read, is left running with the Stop live, for the
    segment's closing write to close (`StudioStore.close_segment`). The
    segment ends `unclosed` and emits no `TurnFailed`. The close and the
    read are steps of `drive` (`_step`), so a cut pending raises it, which
    the segment settles on."""
    try:
        segment.cursor = await _step(
            store.finish(segment.cursor, segment_id=segment_id, status=TurnStatus.CANCELLED)
        )
    except _NOT_THE_TURNS:
        raise
    except Exception as exc:
        _, stored = await _read_back(
            store,
            segment.cursor,
            key=key,
            what="closing the turn on the Stop its scope took",
            error=exc,
            holds=_closes(segment.cursor, segment_id=segment_id, status=TurnStatus.CANCELLED),
        )
        if stored is None:
            segment.outcome = Outcome.UNCLOSED
            return segment
        segment.cursor = stored
        segment.outcome = (
            Outcome(stored.status.value) if stored.status in CLOSED else Outcome.UNCLOSED
        )
        return segment
    segment.outcome = Outcome.CANCELLED
    return segment


class _Landed(StrEnum):
    """What a write that raised left in the store, as its read back found it
    (`_read_back`)."""

    COMMITTED = "committed"
    NOT_COMMITTED = "not committed"
    CLOSED = "closed by another write"
    MOVED = "moved by another write"
    UNREADABLE = "unreadable"


async def _read_back(
    store: TurnStore,
    before: Cursor,
    *,
    key: TurnKey,
    what: str,
    error: Exception,
    holds: Callable[[Cursor], bool],
    then: Callable[[Cursor], Awaitable[None]] | None = None,
    reraising: bool = False,
) -> tuple[_Landed, Cursor | None]:
    """Read the turn back after the write `what`, made from `before`, raised
    `error`, once any write to it still in flight has settled
    (`TurnStore.settled_turn`, bounded at `SETTLE_TIMEOUT_S`), and say what
    the write left with the turn as stored (`None` when it cannot be read).
    `holds` tells whether the stored turn holds the write. When it does,
    `then` runs in the same step as the read, so work a commit is owed runs
    even when a cut became pending during the read. The read is a step of
    `drive` (`_step`), so a cut pending raises it once the read and `then`
    return. The write's failure is logged as a warning before the read, and
    the outcome after it, with `error`'s traceback when the turn does not
    carry on as written, unless the caller logs it (`reraising`), whether it
    raises it again or not."""
    logger.warning("turn %s: %s raised %r; reading the turn back", key, what, error)

    async def read() -> Cursor:
        stored = await store.settled_turn(before.id)
        if then is not None and holds(stored):
            await then(stored)
        return stored

    try:
        stored = await _step(read())
    except Exception:
        logger.exception("turn %s: the turn cannot be read back after %s failed", key, what)
        if not reraising:
            logger.error("turn %s: %s failed: the turn is unreadable", key, what, exc_info=error)
        return _Landed.UNREADABLE, None
    if holds(stored):
        logger.warning("turn %s: %s committed; its acknowledgement was lost", key, what)
        return _Landed.COMMITTED, stored
    if stored.version == before.version:
        landed = _Landed.NOT_COMMITTED
    elif stored.status in CLOSED:
        landed = _Landed.CLOSED
    else:
        landed = _Landed.MOVED
    logger.log(
        logging.WARNING if reraising else logging.ERROR,
        "turn %s: %s failed: the turn is %s (version %d, %s, by %s)",
        key,
        what,
        landed.value,
        stored.version,
        stored.status.value,
        stored.written_by,
        exc_info=None if reraising else error,
    )
    return landed, stored


def _closes(before: Cursor, *, segment_id: str, status: TurnStatus) -> Callable[[Cursor], bool]:
    """Whether a stored turn holds the close `status` `segment_id` wrote
    from `before`: the one write after it, by that segment, closed
    `status`."""

    def holds(stored: Cursor) -> bool:
        return (
            stored.version == before.version + 1
            and stored.written_by == segment_id
            and stored.status is status
        )

    return holds


# The longest the commit's owed work `_settle` starts may run, and the
# tasks running it, each held until it ends (`pending_owed_work`).
OWED_WORK_TIMEOUT_S = 30.0
_OWED_WORK: set[asyncio.Task[None]] = set()


def pending_owed_work() -> frozenset[asyncio.Task[None]]:
    """The tasks running a committed write's owed work that a cut segment
    left behind (`_settle`), for the app's shutdown to wait on."""
    return frozenset(_OWED_WORK)


def _start_owed(owed: _Owed, key: TurnKey) -> None:
    """Run the works of `owed` not yet started in a task of their own,
    bounded at `OWED_WORK_TIMEOUT_S` and held in `_OWED_WORK` until it ends,
    when any exception it ended on, a cancellation included, is logged."""

    async def bounded() -> None:
        async with asyncio.timeout(OWED_WORK_TIMEOUT_S):
            await owed.run()

    task = asyncio.create_task(bounded(), name=f"owed work of turn {key} node {owed.node}")
    _OWED_WORK.add(task)

    def ended(done: asyncio.Task[None]) -> None:
        _OWED_WORK.discard(done)
        if done.cancelled():
            logger.error("turn %s: the owed work of node %s was cancelled", key, owed.node)
        elif (error := done.exception()) is not None:
            logger.error("turn %s: the owed work of node %s failed", key, owed.node, exc_info=error)

    task.add_done_callback(ended)


async def _commit_boundary(
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
    segment: Segment,
    lineage: SegmentLineage | None,
) -> Cursor:
    """Write the boundary after `node` (`_write_boundary`), hold the written
    cursor on `segment`, then run the node's work after the commit
    (`segment.writing.owed`). `drive` awaits the two as one step, so a write
    that caught a cut and returned still runs the work its commit is owed
    before the cut is raised again, and a cut during that work leaves
    `segment` holding the cursor the write committed. A cut that ends the
    write before every work started leaves them to `_settle`, which starts
    those not yet started once it finds the write committed. The write's
    commit records the cursor it wrote on `segment.writing` and hands a
    pause to `segment.note_pause`, in the commit's own task."""
    writing = segment.writing
    assert writing is not None

    def landed(written: Cursor) -> None:
        writing.landed.append(written)
        segment.note_pause(written)

    written = await _write_boundary(
        store,
        cursor,
        segment_id=segment_id,
        node=node,
        node_input=node_input,
        result=result,
        following=following,
        interrupt_id=interrupt_id,
        ctx=ctx,
        lineage=lineage,
        landed=landed,
    )
    segment.cursor = written
    assert segment.writing is not None
    await segment.writing.owed.run()
    return written


async def _after_failed_write(
    store: TurnStore,
    segment: Segment,
    *,
    key: TurnKey,
    segment_id: str,
    node: str,
    error: Exception,
    emit: Callable[[TurnEvent], None],
) -> bool:
    """Settle `segment` after its boundary write after `node`
    (`segment.writing`) raised `error`, for any reason but those
    `_NOT_THE_TURNS` names. A write that could not encode its values
    (`WriteNotEncoded`) never committed, and closes the turn `failed` at
    once (`_close_on_error`). A write whose COMMIT lost its acknowledgement
    and whose transaction committed returned already, since the store reads
    the transaction's status back (`TurnStore.pause`), so a write that
    raised here either did not commit or its status could not be read in
    time (`harness.db.commit.READ_BACK_LIMIT_S`). The turn is read back
    (`_read_back`). A turn
    that holds the write (`_Writing.committed`) takes the stored cursor and
    runs the node's work after the commit (`_Writing.owed`) in the same step
    as the read, as a write that returned does (`_commit_boundary`), and
    `True` is returned for `drive` to report the boundary and go on. The
    lineage generations a committed write stores are not recorded on the
    segment's `SegmentLineage`, so the segment's next write writes those
    rows again, at the same generations. A turn at the version the write
    started from holds no write, and is closed `failed` from the cursor as
    read. A turn another writer closed ends the segment on the stored
    outcome and reports nothing, since the node did not fail; a node's own
    failure is reported whoever closed the turn (`_close_on_error`). A read
    back that fails, or an open turn another write moved, is left as stored:
    the segment ends `unclosed` and reports `error`, as when a close cannot
    be written. That includes a write whose status could not be read, that
    landed, and that another segment then wrote over under a lease that
    lapsed: the stored turn no longer holds the write, so a pause it made
    writes no `paused` line. A cut pending raises with `segment.writing` still held,
    which the segment settles on."""
    writing = segment.writing
    assert writing is not None

    async def close_failed() -> bool:
        await _close_on_error(
            store,
            segment,
            key=key,
            segment_id=segment_id,
            status=TurnStatus.FAILED,
            node=node,
            error=error,
            emit=emit,
        )
        return False

    if isinstance(error, WriteNotEncoded):
        logger.error("turn %s: the write after node %s failed", key, node, exc_info=error)
        segment.writing = None
        return await close_failed()

    async def committed(stored: Cursor) -> None:
        segment.cursor = stored
        segment.note_pause(stored)
        await writing.owed.run()

    landed, stored = await _read_back(
        store,
        segment.cursor,
        key=key,
        what=f"the write after node {node}",
        error=error,
        holds=writing.committed,
        then=committed,
    )
    if landed is _Landed.COMMITTED:
        return True
    segment.writing = None
    if stored is not None:
        segment.cursor = stored
    if landed is _Landed.NOT_COMMITTED:
        return await close_failed()
    if landed is _Landed.CLOSED:
        assert stored is not None
        segment.outcome = Outcome(stored.status.value)
        return False
    segment.outcome = Outcome.UNCLOSED
    emit(TurnFailed(node=node, error=error))
    return False


def _following(graph: Graph[Any], node: str, result: Next | Pause) -> str | None:
    """The node `result` leads to, or `None` for a pause."""
    if isinstance(result, Pause):
        return None
    following = graph.edges.get((node, result.label))
    if following is None:
        raise NoEdge(f"node {node!r} has no edge {result.label!r}")
    return following


def _lineage_scope(key: TurnKey) -> str:
    """The conversation the process's frame lineage store holds `key`'s
    frames under (`harness.conversation.ledger_conversation_id`), which the
    turn store keeps and restores them by."""
    return ledger_conversation_id(key.principal, key.conversation_id)


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
    lineage: SegmentLineage | None,
    landed: Callable[[Cursor], None],
) -> Cursor:
    """The write that records `node`'s `result`: a pause under
    `interrupt_id`, the turn's end, or the move to `following`. A pause
    records the Stop the turn's `StopScope` took (`StopScope.taken`), which
    ends that Stop. Each hands the cursor it wrote to `landed` once it
    commits. A pause or move writes the frame lineage of the segment's
    `lineage`."""
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
            lineage=lineage,
            committed=landed,
        )
    if following == END:
        return await store.finish(
            cursor,
            segment_id=segment_id,
            status=TurnStatus.DONE,
            state=ctx.state,
            records=ctx.transcript.take(),
            committed=landed,
        )
    assert following is not None
    return await store.advance(
        cursor,
        segment_id=segment_id,
        node=following,
        node_input=result.output,
        state=ctx.state,
        records=ctx.transcript.take(),
        lineage=lineage,
        committed=landed,
    )


type _Close = Callable[[TurnStatus, str, Exception], Awaitable[Segment]]


class _Stops(Exception):  # noqa: N818 -- carries how the segment ends, not an error kind
    """`_enter` found the turn cannot go on (`_Closes`, `_Ends`)."""

    async def end(
        self, segment: Segment, *, node: str, emit: Callable[[TurnEvent], None], close: _Close
    ) -> Segment:
        """End `segment` at `node`, closing the turn through `close` when the
        turn closes."""
        raise NotImplementedError


class _Closes(_Stops):
    """The turn closes with `status` and `error`."""

    def __init__(self, status: TurnStatus, error: Exception) -> None:
        super().__init__(str(error))
        self.status = status
        self.error = error

    @override
    async def end(
        self, segment: Segment, *, node: str, emit: Callable[[TurnEvent], None], close: _Close
    ) -> Segment:
        return await close(self.status, node, self.error)


class _Ends(_Stops):
    """The turn as stored must not be closed here. The segment ends
    `outcome`, reporting `error` when one is given (`TurnFailed`)."""

    def __init__(self, outcome: Outcome, error: Exception | None) -> None:
        super().__init__(outcome.value)
        self.outcome = outcome
        self.error = error

    @override
    async def end(
        self, segment: Segment, *, node: str, emit: Callable[[TurnEvent], None], close: _Close
    ) -> Segment:
        segment.outcome = self.outcome
        if self.error is not None:
            emit(TurnFailed(node=node, error=self.error))
        return segment


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
            (`abandoned`), accepting the answer failed and did not commit
            (`failed`), or the owed answer is not stored (`failed`).
        _Ends: accepting the answer failed and the turn, read back, was
            closed by another write (its stored outcome), moved by one, or
            could not be read (`unclosed`, reporting the error).
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
            segment.cursor = await _step(
                store.record_answer(
                    cursor,
                    segment_id=segment_id,
                    interrupt_id=answer.interrupt_id,
                    round=answer.round,
                    answer=answer.value,
                    request=answer.request,
                    records=transcript.take(),
                )
            )
        except _NOT_THE_TURNS:
            raise
        except Exception as exc:
            _settle_answer(
                await _answer_read_back(
                    store, cursor, key=key, segment_id=segment_id, answer=answer, error=exc
                ),
                segment,
                error=exc,
            )
        owed = answer.value
    elif cursor.owed is not None:
        owed = await _step(
            store.answer(cursor, interrupt_id=cursor.owed.interrupt_id, round=cursor.owed.round)
        )
        if owed is None:
            missing = RuntimeError(
                f"turn {cursor.turn_index} owes {node!r} the answer to "
                f"{cursor.owed.interrupt_id!r} round {cursor.owed.round}, which is not stored"
            )
            logger.error("turn %s: %s", key, missing)
            raise _Closes(TurnStatus.FAILED, missing)
    return target, owed


async def _answer_read_back(
    store: TurnStore,
    before: Cursor,
    *,
    key: TurnKey,
    segment_id: str,
    answer: Answer,
    error: Exception,
) -> tuple[_Landed, Cursor | None]:
    """The read back after accepting `answer` from `before` raised `error`
    (`_read_back`): the accepting write is held by a stored turn one
    version on, written by `segment_id`, running and owing the answer to
    `answer`'s pause. An answer JSON cannot hold (`WriteNotEncoded`) never
    committed and is not read back."""
    if isinstance(error, WriteNotEncoded):
        logger.error("turn %s: accepting the answer failed", key, exc_info=error)
        return _Landed.NOT_COMMITTED, None
    owed = OwedAnswer(interrupt_id=answer.interrupt_id, round=answer.round)

    def holds(stored: Cursor) -> bool:
        return (
            stored.version == before.version + 1
            and stored.written_by == segment_id
            and stored.status is TurnStatus.RUNNING
            and stored.owed == owed
        )

    return await _read_back(
        store, before, key=key, what="accepting the answer", error=error, holds=holds
    )


def _settle_answer(
    read_back: tuple[_Landed, Cursor | None], segment: Segment, *, error: Exception
) -> None:
    """Take the turn `_answer_read_back` found onto `segment` and return for
    an accepted answer, or raise how the segment ends otherwise.

    Raises:
        _Closes: the answer did not commit (`failed`).
        _Ends: the turn was closed by another write, which reports nothing
            since no node failed, or was moved by one, or could not be read,
            which reports `error`.
    """
    landed, stored = read_back
    if stored is not None:
        segment.cursor = stored
    match landed:
        case _Landed.COMMITTED:
            return
        case _Landed.NOT_COMMITTED:
            raise _Closes(TurnStatus.FAILED, error) from error
        case _Landed.CLOSED:
            assert stored is not None
            raise _Ends(Outcome(stored.status.value), None)
    raise _Ends(Outcome.UNCLOSED, error)


async def _step[T](step: Awaitable[T]) -> T:
    """Await one step of `drive` (a node's run, a write, the work after a
    commit, a close, a read) and raise the cancellation the running task was
    sent and has not taken back (`asyncio.Task.cancelling`), once the step
    returns or in place of the exception it raised. A step that caught the
    cut and went on therefore neither writes, closes nor moves the turn on,
    and the segment settles on its cut (`run_segment`)."""
    try:
        result = await step
    except Exception as exc:
        _raise_pending_cut(exc)
        raise
    _raise_pending_cut(None)
    return result


def _raise_pending_cut(error: Exception | None) -> None:
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
    `resume` runs. A running cursor runs its node from its start. Once the
    turn goes on, and before its node runs, the lineage of each frame the
    turn names is brought into step with its rows, best effort
    (`TurnStore.restore_frame_lineage`), so an answer the turn refuses or a
    turn that closes does not wait for that read. The segment's
    `SegmentLineage` is made before the restore (`TurnStore.start_segment`),
    goes to the restore and each of the segment's pause and move writes, and
    is ended however the segment returns (`TurnStore.end_segment`).

    A write that cannot encode its values (`WriteNotEncoded`, a value JSON
    cannot hold) closes the turn `failed` at the node whose outcome it was writing.
    A write that fails for any other reason but those `_NOT_THE_TURNS`
    names (a database error, a lost connection) is read back
    (`_after_failed_write`). One that committed, its acknowledgement lost,
    is reported and goes on as a write that returned does. One that did not
    commit closes the turn `failed` at that node. One whose turn another
    writer closed ends on the stored outcome. One whose turn cannot be read,
    or is open and moved some other way, leaves the turn as stored and ends
    the segment `unclosed`.

    A Stop the turn's `StopScope` took (`StopScope.taken`) must be recorded
    by a pause of the node that took it, or of the node one of its stop
    edges leads to (`Node.stop_edges`, which `Graph` checks reach a node
    that pauses). When the node that took it ends the turn or leaves on
    another edge, or the node after it returns anything but a pause, the
    turn closes `cancelled` in place of that node's write.

    A cut (a Stop that ends the turn, a lost lease, a client that left)
    reaches the node as a cancellation of the task running `drive`. Every
    step `drive` awaits runs through `_step`, which raises the cancellation
    again once the step returns or raises, so work that catches it neither
    writes the node's result, closes the turn `failed`, nor leads to another
    node, and the segment settles on the cut as when the cancellation
    escapes (`run_segment`). A cut that lands once a move or end committed is
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
    cursor, stored = await _step(store.load(cursor))
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
    except _Stops as stopping:
        return await stopping.end(segment, node=node, emit=emit, close=close)
    cursor = segment.cursor
    # Made before any await, so the segment's kept frames stop being exempt
    # from the provenance store's bound however the segment ends, a cut
    # during or just after the restore included (`TurnStore.end_segment`).
    lineage = store.start_segment(lineage_scope=_lineage_scope(key))
    try:
        await _step(store.restore_frame_lineage(cursor, lineage))
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
                    result = await _step(target.resume(ctx, node_input, owed))
                else:
                    result = await _step(target.run(ctx, node_input))
                following = _following(graph, node, result)
            except Exception as exc:
                logger.exception("turn %s: node %s failed", key, node)
                return await close(TurnStatus.FAILED, node, exc)
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
            segment.writing = _Writing(
                reports=Paused(
                    kind=result.kind,
                    interrupt_id=interrupt_id,
                    round=cursor.round + 1,
                    payload=result.payload,
                    message=result.message,
                )
                if isinstance(result, Pause)
                else NodeDone(node=node, label=result.label, output=result.output),
                version=cursor.version,
                following=following,
                segment_id=segment_id,
                owed=_Owed(ctx, node),
            )
            # A cut that lands from the write on is settled on the turn as
            # stored, which reports the boundary a write committed (`_settle`).
            try:
                cursor = segment.cursor = await _step(
                    _commit_boundary(
                        store,
                        cursor,
                        segment_id=segment_id,
                        node=node,
                        node_input=node_input,
                        result=result,
                        following=following,
                        interrupt_id=interrupt_id,
                        ctx=ctx,
                        segment=segment,
                        lineage=lineage,
                    )
                )
            except _NOT_THE_TURNS:
                raise
            except Exception as exc:
                if not await _after_failed_write(
                    store,
                    segment,
                    key=key,
                    segment_id=segment_id,
                    node=node,
                    error=exc,
                    emit=emit,
                ):
                    return segment
                cursor = segment.cursor

            writing = segment.writing
            assert writing is not None
            if isinstance(writing.reports, Paused):
                segment.outcome = Outcome.PAUSED
                emit(writing.reports)
                segment.writing = None
                return segment
            emit(writing.reports)
            segment.writing = None
            if following == END:
                segment.outcome = Outcome.DONE
                return segment
            assert following is not None
            # The next node takes its input as stored, as a rerun reads it back.
            node, node_input, target = following, cursor.node_input, graph.nodes[following]

    finally:
        store.end_segment(lineage)


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


def _abandoned(cursor: Cursor) -> dict[str, Any]:
    """The error a turn closed `abandoned` at `cursor` records: a
    `TurnAbandoned` at the node it stands at."""
    left = TurnAbandoned(
        f"turn {cursor.turn_index} was left {cursor.status.value} at {cursor.node!r} "
        "when the next turn opened"
    )
    return failure(cursor.node, left)


async def abandon_on(conn: AsyncConnection, cursor: Cursor, *, segment_id: str) -> Cursor:
    """Close the open turn at `cursor` `abandoned` without running its node,
    on `conn`, in the caller's transaction (`finish_on`), writing under
    `segment_id`, the conversation lease the caller claimed to open the
    conversation's next turn. Its error is a `TurnAbandoned` at the node it
    stands at. A paused turn's pause is dropped. `/turn` (`harness.studio`)
    runs it in the transaction of the save that begins the next turn.

    Raises:
        SegmentLost: `segment_id` does not hold the conversation's lease.
        CursorStale: another writer moved the cursor, or closed the turn.
    """
    return await finish_on(
        conn,
        cursor,
        segment_id=segment_id,
        status=TurnStatus.ABANDONED,
        # Read only by a close `cancelled`.
        now=datetime.now(UTC),
        error=_abandoned(cursor),
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
    key: TurnKey,
    cut_by: Outcome,
    segment_id: str,
    emit: Callable[[TurnEvent], None],
) -> None:
    """Set `segment`'s cursor and outcome after the heartbeat cut its node off
    for a Stop or a lost lease. The node may have been inside a boundary's
    write, which may have committed, so the turn is read back as stored once
    any write to it still in flight has settled (`TurnStore.settled_turn`).
    When that read fails and the write's own acknowledgement holds the write
    (`segment.cursor`), the segment settles on that cursor; otherwise the
    read's error is raised. A turn the store closed keeps that outcome. A
    Stop closes any other `cancelled`, a pause the segment wrote included,
    since the user stopped the turn; work the pause's commit started (early
    mints) runs to completion and is discarded, and work it had not started
    is not started. After a lost lease a pause the segment was writing that
    committed keeps the turn paused, and its `Paused` is emitted here when
    the cut came before `drive` emitted it, so the client gets the question
    the store holds. A turn still running is left to the instance that holds
    the lease. A move or end the segment was writing that committed has its
    `NodeDone` emitted here when the cut came before `drive` emitted it,
    whether the cut came while the write's acknowledgement was awaited,
    during the node's work after the commit, or after it. A Stop's close
    that raises is read back (`_read_back`): a turn it, or another write,
    closed takes the stored outcome; any other raises the close's error. Any
    other write of the segment's that committed has the works its commit is
    owed (`_Writing.owed`) that had not started when the cut came started
    here in a task of their own (`_start_owed`), bounded at
    `OWED_WORK_TIMEOUT_S`, without waiting for them, so a Stop's close goes
    on at once. Whether the write committed is what its commit's task
    recorded (`_Writing.landed`), its transaction's status read back when
    the COMMIT raised (`TurnStore.pause`), so a write another segment's
    later write moved past still counts. Only a COMMIT whose status could
    not be read in time is judged on the turn as stored
    (`_Writing.committed`); such a write that another segment moved past is
    not found committed, so its owed work is dropped and the conversation's
    next turn lists the hand-back it would have acknowledged again."""
    try:
        stored = await store.settled_turn(segment.cursor.id)
    except Exception:
        # The write's own acknowledgement still tells the turn as written.
        if segment.writing is None or not segment.writing.committed(segment.cursor):
            raise
        logger.exception(
            "turn %s: reading the turn back after the cut failed; settling on the cursor "
            "the write acknowledged",
            key,
        )
        stored = segment.writing.landed[-1] if segment.writing.landed else segment.cursor
    writing, segment.writing = segment.writing, None
    reported = writing.reports if writing is not None and writing.committed(stored) else None
    if isinstance(reported, Paused):
        # The pause committed, whatever cut came after it. Its commit's task
        # handed it over already unless it was read back here.
        assert writing is not None
        segment.note_pause(writing.landed[-1] if writing.landed else stored)
    # A Stop closes a pause it cut, so the pause's owed work (early mints)
    # would be for a question no one can answer.
    if reported is not None and not (isinstance(reported, Paused) and cut_by is Outcome.CANCELLED):
        assert writing is not None
        _start_owed(writing.owed, key)
    if isinstance(reported, NodeDone):
        emit(reported)
    if stored.status in CLOSED:
        segment.cursor, segment.outcome = stored, Outcome(stored.status.value)
        return
    if cut_by is Outcome.LOST and stored.status is TurnStatus.PAUSED:
        segment.cursor, segment.outcome = stored, Outcome.PAUSED
        if isinstance(reported, Paused):
            emit(reported)
        return
    segment.cursor, segment.outcome = stored, cut_by
    if cut_by is not Outcome.CANCELLED:
        return
    try:
        segment.cursor = await store.finish(
            stored, segment_id=segment_id, status=TurnStatus.CANCELLED
        )
    except _NOT_THE_TURNS:
        raise
    except Exception as exc:
        landed, closed = await _read_back(
            store,
            stored,
            key=key,
            what="closing the stopped turn",
            error=exc,
            reraising=True,
            holds=_closes(stored, segment_id=segment_id, status=TurnStatus.CANCELLED),
        )
        if closed is None or closed.status not in CLOSED:
            raise
        # `_read_back` left the traceback to a caller that raises again.
        logger.log(
            logging.WARNING if landed is _Landed.COMMITTED else logging.ERROR,
            "turn %s: closing the stopped turn failed; the turn is %s",
            key,
            landed.value,
            exc_info=exc,
        )
        segment.cursor, segment.outcome = closed, Outcome(closed.status.value)


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
    paused: Callable[[Cursor], None] | None = None,
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
    leaves no task of the segment running but the work a committed write is
    owed, which belongs to the commit and runs on, bounded (`_settle`,
    `pending_owed_work`), and closes the turn `cancelled`. One a renewal
    reads after the drive finished is left to the segment's closing write,
    which acts on a Stop still live
    (`StudioStore.close_segment`). A renewal that finds the lease
    taken, or that keeps failing until the lease's deadline
    (`StudioStore.lease_deadline`, dated from the claim or the last renewal
    that landed), cancels the node and ends the segment `lost` without
    writing the cursor. After either cut the turn is read back (`_settle`),
    so a close or pause that committed before the cut is reported as
    stored.

    `cursor` must be read after the caller claimed the lease. `drive`
    refuses one another write has moved since (`CursorStale`).

    `paused`, when given, hears each pause the segment's writes commit, once
    per pause, with the paused cursor (`Segment.note_pause`): from the pause
    write's own commit (`TurnStore.pause`), or from the read back that finds
    a pause committed whose acknowledgement was lost, whatever cut followed
    it. It must not await.

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
    segment = Segment(cursor=cursor, paused=paused)
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
            await _settle(
                store, segment, key=key, cut_by=ended[0], segment_id=segment_id, emit=emit
            )
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
