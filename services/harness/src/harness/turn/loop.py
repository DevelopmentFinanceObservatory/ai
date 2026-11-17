"""The turn loop: run the node a turn's cursor stands at, and the nodes its
edges lead to, until the turn pauses or ends, writing the cursor at every
node boundary.

`drive` runs nodes. At each boundary it writes, in one `TurnStore` call, the
node the turn moves to and that node's input, the state, the whole spec
mapping and the transcript records appended since the last write; a pause
writes the question and an end closes the turn. Only then does it run the
node's `after_commit` work and emit the event that reports the boundary, so
a client never sees a step the store does not hold.

A node that raises closes the turn `failed`, with `{node, type, message}` as
its error. A cursor naming a node the running `Graph` lacks, or paused at a
node that can no longer resume, closes the turn `abandoned` with a
`TurnExpired` error: no turn is carried across a graph change.

A crash leaves the cursor as last written. The segment that takes the turn
over runs the node at the cursor again from its start, with the state, specs
and transcript as of that write: from `run`, or from `resume` with the
answer the turn accepted when the cursor still owes it (`Cursor.owed`).

`run_segment` runs `drive` under the conversation's lease
(`StudioStore.renew`), renewing it every `HEARTBEAT_INTERVAL_S`. The renewal
reports a Stop from any instance (the conversation's cancel flag). While a
node holds the turn's `StopScope` armed, the renewal consumes the flag and
the Stop ends only the scope's work; any other Stop cancels the node and
closes the turn `cancelled`. A lost lease cancels the node and leaves the
cursor to the instance that holds the lease now.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from harness.studio_state import HEARTBEAT_INTERVAL_S, SEGMENT_TTL_S, SegmentLost, _wire_id
from harness.studio_store import StudioStore
from harness.turn import transcript as entries
from harness.turn.events import NodeDone, Paused, TurnEvent, TurnFailed
from harness.turn.node import Next, Node, NodeCtx, Pause, StopScope, TurnKey
from harness.turn.store import Cursor, CursorStale, TurnStatus, TurnStore
from harness.turn.transcript import Transcript

logger = logging.getLogger(__name__)

# The edge target that ends the turn.
END = "end"

# A heartbeat whose renewals keep failing gives the lease up this long
# before it would lapse, counted on the instance's own clock from the last
# renewal that landed: an instance that deaf to its own renewal cannot be
# trusted to still hold the lease.
_RENEWAL_SLACK_S = 10


class TurnExpired(Exception):  # noqa: N818 -- names the turn's state, as the cursor does
    """The cursor names a node the running graph lacks, or stands paused at
    a node with no `resume`."""


class NoEdge(Exception):  # noqa: N818 -- names the missing edge
    """A node returned an edge label its graph has no edge for."""


@dataclass(frozen=True)
class Graph:
    """The nodes by name, the edges by `(node, label)`, and the node a turn
    opens at. Every edge leaves a node and reaches a node or `END`."""

    start: str
    nodes: Mapping[str, Node]
    edges: Mapping[tuple[str, str], str]

    def __post_init__(self) -> None:
        if self.start not in self.nodes:
            raise ValueError(f"start node {self.start!r} is not a node")
        for (source, label), target in self.edges.items():
            if source not in self.nodes or (target != END and target not in self.nodes):
                raise ValueError(f"edge {source!r} -{label}-> {target!r} joins an unknown node")


class Outcome(StrEnum):
    """How a segment ended: the turn's closing status, `paused`, or `lost`
    when another instance took the conversation's lease."""

    PAUSED = "paused"
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


@dataclass
class Segment:
    """One run of the loop under one lease: the cursor as last written, and
    how the run ended (`None` while it runs)."""

    cursor: Cursor
    outcome: Outcome | None = None


def _error(node: str, error: BaseException) -> dict[str, Any]:
    return {"node": node, "type": type(error).__name__, "message": str(error)}


async def _close_on_error(
    store: TurnStore,
    segment: Segment,
    *,
    segment_id: str,
    status: TurnStatus,
    node: str,
    error: BaseException,
    emit: Callable[[TurnEvent], None],
) -> Segment:
    segment.cursor = await store.finish(
        segment.cursor, segment_id=segment_id, status=status, error=_error(node, error)
    )
    segment.outcome = Outcome(status.value)
    emit(TurnFailed(node=node, error=error))
    return segment


async def _after_commit(ctx: NodeCtx, node: str) -> None:
    """Run the node's `after_commit` work. A failure is logged: the write
    before it holds the node's outcome, and a crash between the two loses
    the work the same way."""
    for work in ctx.committed:
        try:
            await work()
        except Exception:
            logger.exception("turn node %s: work after its commit failed", node)


def _following(graph: Graph, node: str, result: Next | Pause) -> str | None:
    """The node `result` leads to, or `None` for a pause."""
    if isinstance(result, Pause):
        return None
    following = graph.edges.get((node, result.label))
    if following is None:
        raise NoEdge(f"node {node!r} has no edge {result.label!r}")
    return following


async def drive(
    store: TurnStore,
    segment: Segment,
    *,
    key: TurnKey,
    graph: Graph,
    segment_id: str,
    emit: Callable[[TurnEvent], None],
    stop: StopScope,
    answer: Answer | None = None,
) -> Segment:
    """Run the turn from where `segment.cursor` stands until it pauses or
    ends, writing each boundary under `segment_id`'s lease, and return
    `segment` with its last cursor and outcome.

    A paused cursor takes `answer`, which the first write accepts
    (`TurnStore.record_answer`) before the paused node's `resume` runs. A
    running cursor runs its node again from its start.

    Raises:
        ValueError: the cursor is paused and no `answer` was given.
        PauseNotOpen: `answer` names a pause the turn does not have open.
        SegmentLost: `segment_id` no longer holds the conversation's lease.
        CursorStale: another writer moved the cursor.
    """
    cursor = segment.cursor
    if cursor.status is TurnStatus.PAUSED and answer is None:
        raise ValueError(f"turn {cursor.turn_index} is paused and needs an answer")
    specs = await store.specs(cursor)
    transcript = Transcript(await store.records(cursor))
    node, node_input = cursor.node, cursor.node_input
    if answer is not None:
        transcript.append(
            entries.answer(
                interrupt_id=answer.interrupt_id,
                round=answer.round,
                value=answer.value,
            )
        )
        cursor = segment.cursor = await store.record_answer(
            cursor,
            segment_id=segment_id,
            interrupt_id=answer.interrupt_id,
            round=answer.round,
            answer=answer.value,
            records=transcript.take(),
        )
        owed: dict[str, Any] | None = answer.value
    elif cursor.owed is not None:
        owed = await store.answer(
            cursor, interrupt_id=cursor.owed.interrupt_id, round=cursor.owed.round
        )
    else:
        owed = None

    target = graph.nodes.get(node)
    if target is None or (owed is not None and target.resume is None):
        expired = TurnExpired(f"the running graph cannot {'resume' if owed else 'run'} {node!r}")
        logger.warning("turn %s expired: %s", key, expired)
        return await _close_on_error(
            store,
            segment,
            segment_id=segment_id,
            status=TurnStatus.ABANDONED,
            node=node,
            error=expired,
            emit=emit,
        )

    while True:
        ctx = NodeCtx(
            key=key,
            inputs=cursor.inputs,
            state=cursor.state,
            specs=specs,
            transcript=transcript,
            emit=emit,
            stop=stop,
        )
        try:
            if owed is not None and target.resume is not None:
                result = await target.resume(ctx, node_input, owed)
            else:
                result = await target.run(ctx, node_input)
            following = _following(graph, node, result)
        except Exception as exc:
            logger.exception("turn %s: node %s failed", key, node)
            return await _close_on_error(
                store,
                segment,
                segment_id=segment_id,
                status=TurnStatus.FAILED,
                node=node,
                error=exc,
                emit=emit,
            )
        owed = None

        if isinstance(result, Pause):
            interrupt_id = _wire_id("int")
            transcript.append(
                entries.pause(
                    kind=result.kind,
                    interrupt_id=interrupt_id,
                    round=cursor.round + 1,
                    payload=result.payload,
                    message=result.message,
                )
            )
            cursor = segment.cursor = await store.pause(
                cursor,
                segment_id=segment_id,
                node=node,
                node_input=node_input,
                kind=result.kind,
                interrupt_id=interrupt_id,
                payload=result.payload,
                state=ctx.state,
                specs=ctx.specs,
                records=transcript.take(),
            )
            segment.outcome = Outcome.PAUSED
            await _after_commit(ctx, node)
            emit(
                Paused(
                    kind=result.kind,
                    interrupt_id=interrupt_id,
                    round=cursor.round,
                    payload=result.payload,
                    message=result.message,
                )
            )
            return segment

        assert isinstance(result, Next) and following is not None
        if following == END:
            cursor = segment.cursor = await store.finish(
                cursor,
                segment_id=segment_id,
                status=TurnStatus.DONE,
                state=ctx.state,
                specs=ctx.specs,
                records=transcript.take(),
            )
            segment.outcome = Outcome.DONE
        else:
            cursor = segment.cursor = await store.advance(
                cursor,
                segment_id=segment_id,
                node=following,
                node_input=result.output,
                state=ctx.state,
                specs=ctx.specs,
                records=transcript.take(),
            )
        await _after_commit(ctx, node)
        emit(NodeDone(node=node, label=result.label, output=result.output))
        if following == END:
            return segment
        node, node_input, target = following, result.output, graph.nodes[following]


async def _close_cancelled(
    store: TurnStore, segment: Segment, *, key: TurnKey, segment_id: str
) -> None:
    """Close the turn `cancelled` after a Stop cancelled its node. The
    cancelled node may have been inside a boundary's write, which may have
    committed, so a stale cursor is read again and closed when it is still
    this turn's open one."""
    try:
        segment.cursor = await store.finish(
            segment.cursor, segment_id=segment_id, status=TurnStatus.CANCELLED
        )
    except CursorStale:
        current = await store.open_cursor(key.principal, key.conversation_id)
        if current is None or current.id != segment.cursor.id:
            segment.outcome = Outcome.CANCELLED
            return
        segment.cursor = await store.finish(
            current, segment_id=segment_id, status=TurnStatus.CANCELLED
        )
    segment.outcome = Outcome.CANCELLED


async def run_segment(
    store: TurnStore,
    lease: StudioStore,
    cursor: Cursor,
    *,
    principal: str,
    conversation_id: str,
    segment_id: str,
    graph: Graph,
    emit: Callable[[TurnEvent], None],
    answer: Answer | None = None,
    heartbeat_seconds: float = HEARTBEAT_INTERVAL_S,
) -> Segment:
    """`drive` the turn at `cursor` under `segment_id`, the conversation
    lease the caller claimed, renewing it every `heartbeat_seconds`, and
    return the segment once the turn pauses or ends, a Stop closes it, or
    the lease is lost.

    A renewal that reports a Stop while a node holds the turn's `StopScope`
    armed consumes the conversation's cancel flag and sets the scope's
    event; the node goes on. Any other Stop cancels the node, leaves no task
    of the segment running, and closes the turn `cancelled`. A renewal that
    finds the lease taken, or that keeps failing until `_RENEWAL_SLACK_S`
    before the lease would lapse, cancels the node and ends the segment
    `lost` without writing the cursor.

    Cancelling the caller cancels the node and leaves the cursor as last
    written, for a later segment to take over.

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
        if task.cancel():
            ended.append(outcome)

    async def heartbeat() -> None:
        renewed_at = lease.now()
        while True:
            await asyncio.sleep(heartbeat_seconds)
            scoped = stop.armed
            attempted_at = lease.now()
            try:
                result = await lease.renew(
                    principal, conversation_id, segment_id, consume_cancel=scoped
                )
            except Exception:
                logger.exception("turn %s: lease renewal failed", key)
                if lease.now() < renewed_at + SEGMENT_TTL_S - _RENEWAL_SLACK_S:
                    continue
                end(Outcome.LOST)
                return
            if result == "lost":
                end(Outcome.LOST)
                return
            renewed_at = attempted_at
            # A consumed flag whose scope closed during the renewal still
            # asked for a Stop, so it ends the turn.
            if result == "cancelled" and not (scoped and stop.take()):
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
        if ended[0] is Outcome.CANCELLED:
            with contextlib.suppress(SegmentLost):
                await _close_cancelled(store, segment, key=key, segment_id=segment_id)
        return segment
    except SegmentLost:
        segment.outcome = Outcome.LOST
        return segment
    finally:
        beat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await beat
