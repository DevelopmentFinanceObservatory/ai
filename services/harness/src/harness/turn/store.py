"""`TurnStore`: each turn's cursor, spec blocks, answers and transcript in Postgres.

A turn is one `turn_cursors` row (defined in
`harness/db/migrations/0003_turns.sql`): the node it stands at and that
node's input, its status, its open pause or the answered pause its node
still owes `resume`, the inputs fixed when it opened,
the turn-scoped state its nodes share, and the error that closed it. Its
spec blocks are `turn_specs` rows, its accepted answers `turn_answers` rows
and its transcript `agent_records` rows. A conversation has at most one open (running or paused) turn, which
the database enforces.

Every write is one transaction that does two things:

- Fences on the conversation's lease: it reads the `conversations` row
  `FOR SHARE` where `lease_segment_id` is the writer's segment, and raises
  `SegmentLost` when it is not. A takeover's claim updates that row, so it
  waits for an in-flight write to commit, and the dead segment's next write
  finds the lease gone.
- Compare-and-swaps the cursor on the `version` its caller read, raising
  `CursorStale` when another writer moved it. A write that matches the
  version found the row exactly as its `Cursor` describes it.

Spec blocks are kept apart from `state` because they make up most of a
turn's stored bytes and a checkpoint runs at every node boundary. A write
takes the turn's whole spec mapping, compares each block's digest with the
digests its `Cursor` holds, and in the same transaction upserts the blocks
that are new or changed and deletes the spec ids that are gone, so a
checkpoint costs what changed rather than what the turn holds. The rows are
updated as well as inserted because the mapping is not append-only: a later
search reading of a held spec id rewrites its block
(`planner.plan.with_blocks`), and a switch that fails or is stopped puts the
mapping back as it was before the switch (`harness.graph.switch`).

The lease itself, the Stop flag and the frame numbering live on
`conversations` and are written by `harness.studio_store.StudioStore`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from harness.studio_state import SegmentLost


class TurnStatus(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"


# The statuses `finish` closes a turn with. `FAILED` and `ABANDONED` carry an
# error and the others none, which the table's check constraint enforces.
CLOSED = frozenset({TurnStatus.DONE, TurnStatus.FAILED, TurnStatus.CANCELLED, TurnStatus.ABANDONED})


class CursorStale(Exception):  # noqa: N818 -- named for the wire 409
    """The cursor moved since the caller read it, or is not in the status the
    write needs."""


class TurnAlreadyOpen(Exception):  # noqa: N818 -- names the state that refuses
    """`open_turn` found an open turn on the conversation; close it first."""


class PauseNotOpen(Exception):  # noqa: N818 -- names the state that refuses
    """An answer named a pause the turn does not have open."""


@dataclass(frozen=True)
class AgentRecord:
    """One transcript entry. `agent` is the writer: `loop` for the user's
    message, each pause and its answer; `planner` for the planner's run
    records. `kind` and `body` are the writer's own."""

    agent: Literal["loop", "planner"]
    kind: str
    body: Any


@dataclass(frozen=True)
class OpenPause:
    """A paused turn's question: its `kind`, the wire `interrupt_id` the
    client answers, its `round` (the turn's pauses so far, this one
    included), and the `payload` it was put with."""

    kind: str
    interrupt_id: str
    round: int
    payload: dict[str, Any]


@dataclass(frozen=True)
class OwedAnswer:
    """The pause `(interrupt_id, round)` whose accepted answer the node at
    a running cursor still owes its `resume`: the cursor has not moved since
    `TurnStore.record_answer` accepted it. `TurnStore.answer` reads the
    answer."""

    interrupt_id: str
    round: int


@dataclass(frozen=True)
class Cursor:
    """One turn as last read or written. `id` keys the turn's answers and
    records; `conversation` is the `conversations` row's id; `record_count`
    is the number of transcript entries stored; `spec_digests` is the digest
    of each stored spec block, by spec id (`TurnStore.specs` reads the
    blocks). `owed` is set on a running turn whose node still owes an
    accepted answer to its `resume`, and `None` otherwise."""

    id: int
    conversation: int
    turn_index: int
    node: str
    node_input: Any
    status: TurnStatus
    round: int
    pause: OpenPause | None
    owed: OwedAnswer | None
    inputs: dict[str, Any]
    state: dict[str, Any]
    error: dict[str, Any] | None
    version: int
    record_count: int
    spec_digests: dict[str, str]


def _jsonb(value: Any) -> str | None:
    """`value` as a jsonb parameter: JSON text, or SQL NULL for `None`."""
    return None if value is None else json.dumps(value)


_COLUMNS = (
    "id, conversation, turn_index, node, node_input, status, round, pause_kind, "
    "interrupt_id, pause_payload, owed_interrupt_id, inputs, state, error, version"
)


type Specs = Mapping[str, Mapping[str, Any]]


def _cursor(row: Any, record_count: int, spec_digests: dict[str, str]) -> Cursor:
    pause = (
        OpenPause(
            kind=row.pause_kind,
            interrupt_id=row.interrupt_id,
            round=row.round,
            payload=row.pause_payload,
        )
        if row.pause_kind is not None
        else None
    )
    return Cursor(
        id=row.id,
        conversation=row.conversation,
        turn_index=row.turn_index,
        node=row.node,
        node_input=row.node_input,
        status=TurnStatus(row.status),
        round=row.round,
        pause=pause,
        owed=(
            OwedAnswer(interrupt_id=row.owed_interrupt_id, round=row.round)
            if row.owed_interrupt_id is not None
            else None
        ),
        inputs=row.inputs,
        state=row.state,
        error=row.error,
        version=row.version,
        record_count=record_count,
        spec_digests=spec_digests,
    )


_FENCE = text(
    "SELECT id FROM conversations WHERE id = :conversation "
    "AND lease_segment_id = :segment_id FOR SHARE"
)
_FENCE_BY_KEY = text(
    "SELECT id FROM conversations WHERE principal = :principal "
    "AND conversation_id = :conversation_id AND lease_segment_id = :segment_id FOR SHARE"
)
_HAS_OPEN_TURN = text(
    "SELECT 1 FROM turn_cursors WHERE conversation = :conversation "
    "AND status IN ('running', 'paused')"
)
_APPEND = text(
    "INSERT INTO agent_records (turn, seq, agent, kind, body) "
    "VALUES (:turn, :seq, :agent, :kind, CAST(:body AS jsonb))"
)


async def _append(conn: AsyncConnection, cursor: Cursor, records: Sequence[AgentRecord]) -> None:
    if not records:
        return
    await conn.execute(
        _APPEND,
        [
            {
                "turn": cursor.id,
                "seq": cursor.record_count + offset,
                "agent": record.agent,
                "kind": record.kind,
                "body": _jsonb(record.body),
            }
            for offset, record in enumerate(records)
        ],
    )


_UPSERT_SPEC = text(
    "INSERT INTO turn_specs (turn, spec_id, block, digest) "
    "VALUES (:turn, :spec_id, CAST(:block AS jsonb), :digest) "
    "ON CONFLICT (turn, spec_id) DO UPDATE SET block = EXCLUDED.block, digest = EXCLUDED.digest"
)


def _spec_digest(block: Mapping[str, Any]) -> tuple[str, str]:
    """`block`'s canonical JSON and its SHA-256."""
    body = json.dumps(block, sort_keys=True, separators=(",", ":"))
    return body, hashlib.sha256(body.encode()).hexdigest()


async def _write_specs(conn: AsyncConnection, cursor: Cursor, specs: Specs) -> dict[str, str]:
    """Make the turn's stored spec blocks equal `specs`, writing only the
    blocks whose digest differs from `cursor.spec_digests` and deleting the
    spec ids `specs` lacks. Returns the digests now stored."""
    digests: dict[str, str] = {}
    changed: list[dict[str, Any]] = []
    for spec_id, block in specs.items():
        body, digest = _spec_digest(block)
        digests[spec_id] = digest
        if cursor.spec_digests.get(spec_id) != digest:
            changed.append({"turn": cursor.id, "spec_id": spec_id, "block": body, "digest": digest})
    if changed:
        await conn.execute(_UPSERT_SPEC, changed)
    if gone := [spec_id for spec_id in cursor.spec_digests if spec_id not in digests]:
        await conn.execute(
            text("DELETE FROM turn_specs WHERE turn = :turn AND spec_id = ANY(:gone)"),
            {"turn": cursor.id, "gone": gone},
        )
    return digests


class TurnStore:
    """The turn cursor API over `engine` (module docstring). Every write
    takes the writer's `segment_id`, the conversation lease it holds, and the
    `Cursor` it last read or wrote, and returns the cursor as written."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def _fence(self, conn: AsyncConnection, conversation: int, segment_id: str) -> None:
        held = await conn.execute(_FENCE, {"conversation": conversation, "segment_id": segment_id})
        if held.first() is None:
            raise SegmentLost(f"segment {segment_id!r} does not hold the conversation's lease")

    async def _swap(
        self,
        conn: AsyncConnection,
        cursor: Cursor,
        *,
        statuses: tuple[TurnStatus, ...],
        sets: str,
        params: dict[str, Any],
        specs: Specs | None,
        records: Sequence[AgentRecord],
    ) -> Cursor:
        """Apply `sets` and bump the version where the row still holds the
        version `cursor` read and one of `statuses`; write `specs` when
        given (`_write_specs`) and append `records`."""
        row = (
            await conn.execute(
                text(
                    f"UPDATE turn_cursors SET version = version + 1, updated_at = now(), {sets} "
                    "WHERE id = :id AND version = :version AND status = ANY(:statuses) "
                    f"RETURNING {_COLUMNS}"
                ),
                {
                    **params,
                    "id": cursor.id,
                    "version": cursor.version,
                    "statuses": [status.value for status in statuses],
                },
            )
        ).first()
        if row is None:
            raise CursorStale(f"turn {cursor.turn_index} moved past version {cursor.version}")
        digests = (
            await _write_specs(conn, cursor, specs) if specs is not None else cursor.spec_digests
        )
        await _append(conn, cursor, records)
        return _cursor(row, cursor.record_count + len(records), digests)

    async def open_turn(
        self,
        principal: str,
        conversation_id: str,
        *,
        segment_id: str,
        turn_index: int,
        node: str,
        inputs: dict[str, Any],
        state: dict[str, Any],
        records: Sequence[AgentRecord] = (),
    ) -> Cursor:
        """Open turn `turn_index` running at `node` with no node input, and
        append `records`.

        Raises:
            SegmentLost: `segment_id` does not hold the conversation's lease.
            TurnAlreadyOpen: the conversation has an open turn. `finish` it
                (`ABANDONED`) first.
        """
        async with self._engine.begin() as conn:
            conversation = (
                await conn.execute(
                    _FENCE_BY_KEY,
                    {
                        "principal": principal,
                        "conversation_id": conversation_id,
                        "segment_id": segment_id,
                    },
                )
            ).scalar_one_or_none()
            if conversation is None:
                raise SegmentLost(
                    f"segment {segment_id!r} does not hold {conversation_id!r}'s lease"
                )
            if (await conn.execute(_HAS_OPEN_TURN, {"conversation": conversation})).first():
                raise TurnAlreadyOpen(f"conversation {conversation_id!r} has an open turn")
            row = (
                await conn.execute(
                    text(
                        "INSERT INTO turn_cursors "
                        "(conversation, turn_index, node, status, inputs, state) VALUES "
                        "(:conversation, :turn_index, :node, 'running', "
                        "CAST(:inputs AS jsonb), CAST(:state AS jsonb)) "
                        f"RETURNING {_COLUMNS}"
                    ),
                    {
                        "conversation": conversation,
                        "turn_index": turn_index,
                        "node": node,
                        "inputs": _jsonb(inputs),
                        "state": _jsonb(state),
                    },
                )
            ).one()
            cursor = _cursor(row, 0, {})
            await _append(conn, cursor, records)
            return _cursor(row, len(records), {})

    async def open_cursor(self, principal: str, conversation_id: str) -> Cursor | None:
        """The conversation's open (running or paused) turn, or `None`."""
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        f"SELECT {_COLUMNS}, "
                        "(SELECT count(*) FROM agent_records r WHERE r.turn = t.id) "
                        "AS record_count FROM turn_cursors t WHERE t.conversation = "
                        "(SELECT id FROM conversations WHERE principal = :principal "
                        "AND conversation_id = :conversation_id) "
                        "AND t.status IN ('running', 'paused')"
                    ),
                    {"principal": principal, "conversation_id": conversation_id},
                )
            ).first()
            if row is None:
                return None
            digests = await conn.execute(
                text("SELECT spec_id, digest FROM turn_specs WHERE turn = :turn"),
                {"turn": row.id},
            )
            return _cursor(row, row.record_count, dict(digests.tuples().all()))

    async def specs(self, cursor: Cursor) -> dict[str, dict[str, Any]]:
        """The turn's spec blocks by spec id, in the order the turn first
        stored each."""
        async with self._engine.connect() as conn:
            rows = await conn.execute(
                text("SELECT spec_id, block FROM turn_specs WHERE turn = :turn ORDER BY seq"),
                {"turn": cursor.id},
            )
            return dict(rows.tuples().all())

    async def records(self, cursor: Cursor) -> list[AgentRecord]:
        """The turn's transcript, in the order it was appended."""
        async with self._engine.connect() as conn:
            rows = await conn.execute(
                text("SELECT agent, kind, body FROM agent_records WHERE turn = :turn ORDER BY seq"),
                {"turn": cursor.id},
            )
            return [AgentRecord(agent=row.agent, kind=row.kind, body=row.body) for row in rows]

    async def answer(
        self, cursor: Cursor, *, interrupt_id: str, round: int
    ) -> dict[str, Any] | None:
        """The answer accepted for the turn's pause `(interrupt_id, round)`,
        or `None` when none was."""
        async with self._engine.connect() as conn:
            return (
                await conn.execute(
                    text(
                        "SELECT answer FROM turn_answers WHERE turn = :turn "
                        "AND interrupt_id = :interrupt_id AND round = :round"
                    ),
                    {"turn": cursor.id, "interrupt_id": interrupt_id, "round": round},
                )
            ).scalar_one_or_none()

    async def advance(
        self,
        cursor: Cursor,
        *,
        segment_id: str,
        node: str,
        node_input: Any,
        state: dict[str, Any],
        specs: Specs,
        records: Sequence[AgentRecord] = (),
    ) -> Cursor:
        """Move the running turn to `node` with `node_input`, writing `state`
        and the turn's spec blocks `specs`, and appending `records`, with it.

        Raises:
            SegmentLost: `segment_id` does not hold the conversation's lease.
            CursorStale: the cursor moved, or the turn is not running.
        """
        async with self._engine.begin() as conn:
            await self._fence(conn, cursor.conversation, segment_id)
            return await self._swap(
                conn,
                cursor,
                statuses=(TurnStatus.RUNNING,),
                sets="node = :node, node_input = CAST(:node_input AS jsonb), "
                "state = CAST(:state AS jsonb), owed_interrupt_id = NULL",
                params={"node": node, "node_input": _jsonb(node_input), "state": _jsonb(state)},
                specs=specs,
                records=records,
            )

    async def pause(
        self,
        cursor: Cursor,
        *,
        segment_id: str,
        node: str,
        node_input: Any,
        kind: str,
        interrupt_id: str,
        payload: dict[str, Any],
        state: dict[str, Any],
        specs: Specs,
        records: Sequence[AgentRecord] = (),
    ) -> Cursor:
        """Pause the running turn at `node` on a question of `kind` under the
        wire id `interrupt_id`, as the turn's next round, writing `state` and
        the turn's spec blocks `specs`, and appending `records`, with it. The returned cursor's `pause.round` is
        the round an answer names.

        Raises:
            SegmentLost: `segment_id` does not hold the conversation's lease.
            CursorStale: the cursor moved, or the turn is not running.
        """
        async with self._engine.begin() as conn:
            await self._fence(conn, cursor.conversation, segment_id)
            return await self._swap(
                conn,
                cursor,
                statuses=(TurnStatus.RUNNING,),
                sets="status = 'paused', round = round + 1, node = :node, "
                "node_input = CAST(:node_input AS jsonb), pause_kind = :kind, "
                "interrupt_id = :interrupt_id, pause_payload = CAST(:payload AS jsonb), "
                "state = CAST(:state AS jsonb), owed_interrupt_id = NULL",
                params={
                    "node": node,
                    "node_input": _jsonb(node_input),
                    "kind": kind,
                    "interrupt_id": interrupt_id,
                    "payload": _jsonb(payload),
                    "state": _jsonb(state),
                },
                specs=specs,
                records=records,
            )

    async def record_answer(
        self,
        cursor: Cursor,
        *,
        segment_id: str,
        interrupt_id: str,
        round: int,
        answer: dict[str, Any],
        records: Sequence[AgentRecord] = (),
    ) -> Cursor:
        """Accept `answer` to the open pause `(interrupt_id, round)` and set
        the turn running again at the node it paused at, appending `records`.
        The returned cursor's `owed` names the pause until the next write
        moves the cursor.

        Raises:
            PauseNotOpen: the turn has no open pause `(interrupt_id, round)`.
            SegmentLost: `segment_id` does not hold the conversation's lease.
            CursorStale: the cursor moved.
        """
        pause = cursor.pause
        if pause is None or (pause.interrupt_id, pause.round) != (interrupt_id, round):
            raise PauseNotOpen(f"turn {cursor.turn_index} has no open pause {interrupt_id!r}")
        async with self._engine.begin() as conn:
            await self._fence(conn, cursor.conversation, segment_id)
            running = await self._swap(
                conn,
                cursor,
                statuses=(TurnStatus.PAUSED,),
                sets="status = 'running', pause_kind = NULL, interrupt_id = NULL, "
                "pause_payload = NULL, owed_interrupt_id = :interrupt_id",
                params={"interrupt_id": interrupt_id},
                specs=None,
                records=records,
            )
            await conn.execute(
                text(
                    "INSERT INTO turn_answers (turn, interrupt_id, round, answer) "
                    "VALUES (:turn, :interrupt_id, :round, CAST(:answer AS jsonb))"
                ),
                {
                    "turn": cursor.id,
                    "interrupt_id": interrupt_id,
                    "round": round,
                    "answer": json.dumps(answer),
                },
            )
            return running

    async def finish(
        self,
        cursor: Cursor,
        *,
        segment_id: str,
        status: TurnStatus,
        error: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        specs: Specs | None = None,
        records: Sequence[AgentRecord] = (),
    ) -> Cursor:
        """Close the open turn with `status`, one of `CLOSED`, appending
        `records`. `FAILED` and `ABANDONED` take an `error` and the others
        none. `state` and `specs` replace the stored state and spec blocks
        when given. A paused turn's
        pause is dropped, so no answer can reach it.

        Raises:
            ValueError: `status` is not a closing status.
            SegmentLost: `segment_id` does not hold the conversation's lease.
            CursorStale: the cursor moved, or the turn is already closed.
        """
        if status not in CLOSED:
            raise ValueError(f"{status!r} does not close a turn")
        state_set = ", state = CAST(:state AS jsonb)" if state is not None else ""
        async with self._engine.begin() as conn:
            await self._fence(conn, cursor.conversation, segment_id)
            return await self._swap(
                conn,
                cursor,
                statuses=(TurnStatus.RUNNING, TurnStatus.PAUSED),
                sets="status = :status, error = CAST(:error AS jsonb), pause_kind = NULL, "
                f"interrupt_id = NULL, pause_payload = NULL, owed_interrupt_id = NULL{state_set}",
                params={
                    "status": status.value,
                    "error": _jsonb(error),
                    **({"state": _jsonb(state)} if state is not None else {}),
                },
                specs=specs,
                records=records,
            )
