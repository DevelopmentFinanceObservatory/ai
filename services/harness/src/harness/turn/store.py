"""`TurnStore`: each turn's cursor, spec blocks, answers and transcript in Postgres.

A turn is one `turn_cursors` row (defined in
`harness/db/migrations/0003_turns.sql`): the node it stands at and that
node's input, its status, its open pause or the answered pause its node
owes `resume`, the inputs fixed when it opened, the turn-scoped state its
nodes share, the error that closed it, and the lease segment whose write
it last took (`written_by`). Its spec blocks are
`turn_specs` rows, its accepted answers `turn_answers` rows and its
transcript `agent_records` rows. A conversation has at most one open
(running or paused) turn, which the database enforces.

Every write is one transaction that does two things:

- Fences on the conversation's lease: it reads the `conversations` row
  `FOR SHARE` where `lease_segment_id` is the writer's segment, and raises
  `SegmentLost` when it is not. A takeover's claim updates that row, so it
  waits for an in-flight write to commit, and the dead segment's next write
  finds the lease gone.
- Compare-and-swaps the cursor on the `version` its caller read, raising
  `CursorStale` when another writer moved it. A write that matches the
  version found the row exactly as its `Cursor` describes it.

The turn's spec blocks, the mapping its state holds under `SPECS_KEY`,
are stored apart from the rest of the state because they make up most of a
turn's stored bytes and a checkpoint runs at every node boundary. A write
splits the mapping out of the state it is given, compares each block's
digest with the digests its `Cursor` holds, and in the same transaction
upserts the blocks that are new or changed, deletes the spec ids that are
gone and moves the blocks whose position changed, so a checkpoint writes
only what changed. Every block is serialised at every write to take its
digest, so a block changed in place is written like any other, and only a
changed block is decoded. The rows are updated as well as inserted because
a block can change after it is first written. A later search reading of a
held spec id rewrites its block (`planner.plan.with_blocks`), and a switch
that fails or is stopped puts the mapping back as it was before the switch
(`harness.graph.switch`). A read
merges the blocks back into the state under `SPECS_KEY`, in their stored
order and with their keys in the order the turn built them, and a write
returns each block decoded from the JSON it stored, or the cursor's own
decoded block where the digest did not change, so every `Cursor.state`
holds the state as stored, whether the next node runs live or reruns.

A close `cancelled` ends the conversation's unfinished runs in its own
transaction (`harness.run_ledger.end_unfinished`), so a cancelled turn
leaves no run computing. A close reads no Stop. Closing the turn a Stop
names ends that Stop (`harness.studio_store.StopState`). A pause write
records the Stop the turn's `StopScope` answered (`stop_taken_id`).

The lineage of each frame the turn's state names under `FRAMES_KEY` is a
`turn_frame_lineage` row, written by the `advance` or `pause` that first
stores the frame's entry, from the lineage the process's `ProvenanceStore`
holds then, and put back once when a segment that lacks one of them takes
the turn over (`TurnStore.restore_frame_lineage`). A frame whose lineage the
process no longer holds, or that JSON cannot hold, has no row, and the write
goes on without it. The rows are best effort: a card drawn on a frame with
none is withheld, as on the instance that fetched it once the lineage is
evicted.

The lease itself, the Stop request and the frame numbering live on
`conversations` and are written by `harness.studio_store.StudioStore`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from compiler.surface import transient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from harness.conversation import NO_CONVERSATION, current_conversation_id
from harness.db.jsonb import to_json_param
from harness.provenance import (
    CardLineageDecodeError,
    ProvenanceStore,
    lineage_from_json,
    lineage_json,
)
from harness.run_ledger import end_unfinished
from harness.studio_state import SegmentLost

logger = logging.getLogger(__name__)

# The state key whose mapping, spec blocks by spec id, the store keeps as
# `turn_specs` rows, outside the stored state.
SPECS_KEY = "orchestrator:specs"
# The state key whose mapping, frame entries by frame id, names the frames
# whose lineage the store keeps as `turn_frame_lineage` rows. Each entry
# names its frame by `ref` and, when the materialisation store holds it
# under another pair, `store_ref`.
FRAMES_KEY = "orchestrator:frames"

# The longest the one read of a turn's stored frame lineage may take
# (`TurnStore.restore_frame_lineage`), the bound a card's stored lineage read
# has (`harness.mcp_client._LINEAGE_READ_BUDGET_S`). The read is best effort
# and the segment waits for it before its node runs.
_FRAME_LINEAGE_READ_BUDGET_S = 2.0


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
_ERRORED = frozenset({TurnStatus.FAILED, TurnStatus.ABANDONED})


class CursorStale(Exception):  # noqa: N818 -- named for the wire 409
    """The cursor moved since the caller read it, or is not in the status the
    write needs."""


class TurnAlreadyOpen(Exception):  # noqa: N818 -- names the state that refuses
    """`open_turn` found an open turn on the conversation. Close it first."""


class PauseNotOpen(Exception):  # noqa: N818 -- names the state that refuses
    """An answer named a pause the turn does not have open."""


def require_open_pause(cursor: Cursor, *, interrupt_id: str, round: int) -> None:
    """Refuse an answer to `(interrupt_id, round)` unless it is the pause
    `cursor` has open.

    Raises:
        PauseNotOpen: the turn has no open pause `(interrupt_id, round)`.
    """
    pause = cursor.pause
    if pause is None or (pause.interrupt_id, pause.round) != (interrupt_id, round):
        raise PauseNotOpen(f"turn {cursor.turn_index} has no open pause {interrupt_id!r}")


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
    a running cursor owes its `resume`. The cursor has not moved since
    `TurnStore.record_answer` accepted it. `TurnStore.answer` reads the
    answer."""

    interrupt_id: str
    round: int


@dataclass(frozen=True)
class Cursor:
    """One turn as last read or written. `id` keys the turn's answers and
    records; `conversation` is the `conversations` row's id; `record_count`
    is the number of transcript entries stored. `state` holds the turn's
    spec blocks under `SPECS_KEY`, as stored; `spec_digests` is the digest
    of each stored block, by spec id in stored order, and `spec_blocks` the
    stored blocks as decoded, in a mapping of their own, so a change to the
    state's mapping leaves them as stored. `owed` is set on a running turn
    whose node owes an accepted answer to its `resume`, and `None`
    otherwise. `written_by` is the lease segment whose write the cursor
    last took, `None` on a row no write set it on."""

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
    spec_blocks: dict[str, dict[str, Any]] = field(default_factory=dict, compare=False, repr=False)
    written_by: str | None = None


def _json(value: Any) -> str | None:
    """`value` as a json parameter: JSON text, or SQL NULL for `None`.
    Raises `ValueError` for a NaN or an infinity, which JSON cannot hold,
    and `TypeError` for a value JSON cannot encode, before any statement
    runs."""
    return None if value is None else to_json_param(value)


_COLUMNS = (
    "id, conversation, turn_index, node, node_input, status, round, pause_kind, "
    "interrupt_id, pause_payload, owed_interrupt_id, inputs, state, error, version, "
    "written_by"
)


type Specs = Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class _StoredSpecs:
    """The spec blocks a write stored or a read found: each block, by spec
    id in stored order, and its digest."""

    blocks: dict[str, dict[str, Any]]
    digests: dict[str, str]


_NO_SPECS = _StoredSpecs(blocks={}, digests={})


def _split(state: Mapping[str, Any]) -> tuple[dict[str, Any], Specs]:
    """`state` without `SPECS_KEY`, and the spec blocks it holds there."""
    rest = {key: value for key, value in state.items() if key != SPECS_KEY}
    return rest, state.get(SPECS_KEY) or {}


def _cursor(row: Any, record_count: int, specs: _StoredSpecs) -> Cursor:
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
        state={**row.state, SPECS_KEY: dict(specs.blocks)},
        error=row.error,
        version=row.version,
        record_count=record_count,
        spec_digests=specs.digests,
        written_by=row.written_by,
        spec_blocks=specs.blocks,
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
    "VALUES (:turn, :seq, :agent, :kind, CAST(:body AS json))"
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
                "body": _json(record.body),
            }
            for offset, record in enumerate(records)
        ],
    )


_UPSERT_SPEC = text(
    "INSERT INTO turn_specs (turn, spec_id, seq, block, digest) "
    "VALUES (:turn, :spec_id, :seq, CAST(:block AS json), :digest) "
    "ON CONFLICT (turn, spec_id) DO UPDATE "
    "SET seq = EXCLUDED.seq, block = EXCLUDED.block, digest = EXCLUDED.digest"
)
_MOVE_SPEC = text("UPDATE turn_specs SET seq = :seq WHERE turn = :turn AND spec_id = :spec_id")


def _spec_digest(body: str) -> str:
    """The SHA-256 of a spec block's JSON text."""
    return hashlib.sha256(body.encode()).hexdigest()


async def _write_specs(conn: AsyncConnection, cursor: Cursor, specs: Specs) -> _StoredSpecs:
    """Make the turn's stored spec blocks equal `specs`, in its order.

    Each row's `seq` is its block's position in `specs`, so stored order is
    the order the turn's state held the blocks in. A block is written when
    its JSON's digest differs from the stored one. A block whose position
    changed but whose content did not has only its `seq` rewritten, and the
    spec ids `specs` lacks are deleted. Returns the blocks in the form a
    read of the rows gives: a written block decoded from its JSON, and an
    unchanged one as `cursor` holds it decoded.
    """
    stored_at = {spec_id: position for position, spec_id in enumerate(cursor.spec_digests)}
    blocks: dict[str, dict[str, Any]] = {}
    digests: dict[str, str] = {}
    changed: list[dict[str, Any]] = []
    moved: list[dict[str, Any]] = []
    for position, (spec_id, block) in enumerate(specs.items()):
        body = to_json_param(block)
        digests[spec_id] = digest = _spec_digest(body)
        if cursor.spec_digests.get(spec_id) == digest:
            # `spec_blocks` holds a block for every digest the cursor holds.
            blocks[spec_id] = cursor.spec_blocks[spec_id]
            if stored_at[spec_id] != position:
                moved.append({"turn": cursor.id, "spec_id": spec_id, "seq": position})
        else:
            blocks[spec_id] = json.loads(body)
            changed.append(
                {
                    "turn": cursor.id,
                    "spec_id": spec_id,
                    "seq": position,
                    "block": body,
                    "digest": digest,
                }
            )
    if gone := [spec_id for spec_id in cursor.spec_digests if spec_id not in digests]:
        await conn.execute(
            text("DELETE FROM turn_specs WHERE turn = :turn AND spec_id = ANY(:gone)"),
            {"turn": cursor.id, "gone": gone},
        )
    if changed:
        await conn.execute(_UPSERT_SPEC, changed)
    if moved:
        await conn.execute(_MOVE_SPEC, moved)
    return _StoredSpecs(blocks=blocks, digests=digests)


def _frame_keys(state: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The store keys the frame entries of `state` (`FRAMES_KEY`) name, each
    entry's `ref` and `store_ref`, in order and each once. A pair with a
    member missing names no key."""
    keys: dict[tuple[str, str], None] = {}
    for entry in (state.get(FRAMES_KEY) or {}).values():
        for ref in (entry.get("ref"), entry.get("store_ref")):
            if isinstance(ref, Mapping):
                spec_id, content_version = ref.get("spec_id"), ref.get("content_version")
                if isinstance(spec_id, str) and isinstance(content_version, str):
                    keys[(spec_id, content_version)] = None
    return list(keys)


def _frame_lineage_rows(
    provenance: ProvenanceStore | None, cursor: Cursor, state: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """The `turn_frame_lineage` rows of the store keys `state` names and the
    state `cursor` holds does not: the lineage `provenance` holds for each, its
    members as JSON text, and the count stored with it. A key `provenance` no
    longer holds has no row, and so has one whose lineage JSON cannot hold (a
    NaN in a manifest), logged at error. None outside a conversation."""
    conversation = current_conversation_id.get()
    if provenance is None or conversation == NO_CONVERSATION:
        return []
    stored = set(_frame_keys(cursor.state))
    rows = []
    for spec_id, content_version in _frame_keys(state):
        if (spec_id, content_version) in stored:
            continue
        lineage = provenance.get(
            conversation=conversation, spec_id=spec_id, content_version=content_version
        )
        if lineage is None:
            continue
        try:
            members = to_json_param(lineage_json(lineage))
        except (TypeError, ValueError):
            logger.exception(
                "turn store: the lineage of frame spec_id=%r content_version=%r cannot be "
                "stored as JSON; turn %s goes on without it",
                spec_id,
                content_version,
                cursor.turn_index,
            )
            continue
        rows.append(
            {
                "turn": cursor.id,
                "spec_id": spec_id,
                "content_version": content_version,
                "members": members,
                "stat_var_count": provenance.stat_var_count(
                    conversation=conversation, spec_id=spec_id, content_version=content_version
                ),
            }
        )
    return rows


_INSERT_FRAME_LINEAGE = text(
    "INSERT INTO turn_frame_lineage (turn, spec_id, content_version, members, stat_var_count) "
    "VALUES (:turn, :spec_id, :content_version, CAST(:members AS json), :stat_var_count) "
    "ON CONFLICT (turn, spec_id, content_version) DO NOTHING"
)


async def _insert_frame_lineage(
    conn: AsyncConnection, cursor: Cursor, rows: Sequence[Mapping[str, Any]]
) -> None:
    """Store `rows` (`_frame_lineage_rows`) in a savepoint, so a failed insert
    leaves the cursor write to commit. A transient fault raises, for the
    whole write to be retried; any other is logged at error and the turn
    goes on without the rows."""
    if not rows:
        return
    try:
        async with conn.begin_nested():
            await conn.execute(_INSERT_FRAME_LINEAGE, list(rows))
    except Exception as exc:
        if transient(exc):
            raise
        logger.exception(
            "turn store: the lineage of turn %s's frames was not stored; the turn goes on "
            "without it",
            cursor.turn_index,
        )


async def _read_cursor(conn: AsyncConnection, where: str, params: dict[str, Any]) -> Cursor | None:
    """The one turn matching `where` (over `turn_cursors t`), or `None`."""
    row = (
        await conn.execute(
            text(
                f"SELECT {_COLUMNS}, "
                "(SELECT count(*) FROM agent_records r WHERE r.turn = t.id) "
                f"AS record_count FROM turn_cursors t WHERE {where}"
            ),
            params,
        )
    ).first()
    if row is None:
        return None
    return _cursor(row, row.record_count, await _stored_specs(conn, row.id))


async def _stored_specs(conn: AsyncConnection, turn: int) -> _StoredSpecs:
    rows = (
        await conn.execute(
            text("SELECT spec_id, block, digest FROM turn_specs WHERE turn = :turn ORDER BY seq"),
            {"turn": turn},
        )
    ).all()
    return _StoredSpecs(
        blocks={row.spec_id: row.block for row in rows},
        digests={row.spec_id: row.digest for row in rows},
    )


async def _records(conn: AsyncConnection, turn: int) -> list[AgentRecord]:
    rows = await conn.execute(
        text("SELECT agent, kind, body FROM agent_records WHERE turn = :turn ORDER BY seq"),
        {"turn": turn},
    )
    return [AgentRecord(agent=row.agent, kind=row.kind, body=row.body) for row in rows]


async def _fence(conn: AsyncConnection, conversation: int, segment_id: str) -> None:
    held = await conn.execute(_FENCE, {"conversation": conversation, "segment_id": segment_id})
    if held.first() is None:
        raise SegmentLost(f"segment {segment_id!r} does not hold the conversation's lease")


async def _swap(
    conn: AsyncConnection,
    cursor: Cursor,
    *,
    segment_id: str,
    statuses: tuple[TurnStatus, ...],
    sets: str,
    params: dict[str, Any],
    state: Mapping[str, Any] | None,
    records: Sequence[AgentRecord],
    provenance: ProvenanceStore | None = None,
) -> Cursor:
    """Apply `sets`, bump the version and record `segment_id` as the
    writer where the row holds the version `cursor` read and one of
    `statuses`, and append `records`. When `state`
    is given, store it with its spec blocks as rows (`_write_specs`), and,
    when the turn stays open, the lineage `provenance` holds for each frame
    key it names first (`_frame_lineage_rows`).
    Otherwise keep the stored blocks `cursor` holds."""
    lineage_rows = _frame_lineage_rows(provenance, cursor, state) if state is not None else []
    if state is not None:
        rest, specs = _split(state)
        sets += ", state = CAST(:state AS json)"
        params = {**params, "state": _json(rest)}
    row = (
        await conn.execute(
            text(
                "UPDATE turn_cursors SET version = version + 1, updated_at = now(), "
                f"written_by = :written_by, {sets} "
                "WHERE id = :id AND version = :version AND status = ANY(:statuses) "
                f"RETURNING {_COLUMNS}"
            ),
            {
                **params,
                "written_by": segment_id,
                "id": cursor.id,
                "version": cursor.version,
                "statuses": [status.value for status in statuses],
            },
        )
    ).first()
    if row is None:
        raise CursorStale(f"turn {cursor.turn_index} moved past version {cursor.version}")
    stored = (
        await _write_specs(conn, cursor, specs)
        if state is not None
        else _StoredSpecs(blocks=cursor.spec_blocks, digests=cursor.spec_digests)
    )
    if TurnStatus(row.status) not in CLOSED:
        await _insert_frame_lineage(conn, cursor, lineage_rows)
    await _append(conn, cursor, records)
    return _cursor(row, cursor.record_count + len(records), stored)


class TurnStore:
    """The turn cursor API over `engine` (module docstring). Every write
    takes the writer's `segment_id`, the conversation lease it holds, and the
    `Cursor` it last read or wrote, and returns the cursor as written.
    `clock` is the `time.time`-shaped clock the runs a close `cancelled`
    ends are stamped with. `provenance` is the process's frame lineage store,
    whose lineage of each frame an open turn names the store keeps with the
    turn and puts back (`restore_frame_lineage`), read over `lineage_engine`
    (`harness.db.engine.DatabaseHandle.lineage_engine`, `engine` when none
    is given). With no `provenance` no lineage is kept."""

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        clock: Callable[[], float] = time.time,
        provenance: ProvenanceStore | None = None,
        lineage_engine: AsyncEngine | None = None,
    ) -> None:
        self._engine = engine
        self._clock = clock
        self._provenance = provenance
        self._lineage_engine = lineage_engine or engine
        # Reads past their budget, cancelled and held until they finish
        # their cleanup, so they are not collected mid-flight.
        self._abandoned_reads: set[asyncio.Task[Any]] = set()

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
        """Open turn `turn_index` running at `node` with no node input and
        with `state`, which holds no spec blocks yet, and append `records`.

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
                        "(conversation, turn_index, node, status, inputs, state, written_by) "
                        "VALUES (:conversation, :turn_index, :node, 'running', "
                        "CAST(:inputs AS json), CAST(:state AS json), :written_by) "
                        f"RETURNING {_COLUMNS}"
                    ),
                    {
                        "conversation": conversation,
                        "turn_index": turn_index,
                        "node": node,
                        "inputs": _json(inputs),
                        "state": _json(_split(state)[0]),
                        "written_by": segment_id,
                    },
                )
            ).one()
            cursor = _cursor(row, 0, _NO_SPECS)
            await _append(conn, cursor, records)
            return _cursor(row, len(records), _NO_SPECS)

    async def open_cursor(self, principal: str, conversation_id: str) -> Cursor | None:
        """The conversation's open (running or paused) turn, or `None`."""
        async with self._engine.connect() as conn:
            return await _read_cursor(
                conn,
                "t.conversation = (SELECT id FROM conversations WHERE principal = :principal "
                "AND conversation_id = :conversation_id) AND t.status IN ('running', 'paused')",
                {"principal": principal, "conversation_id": conversation_id},
            )

    async def turn_at(self, conversation: int, turn_index: int) -> Cursor | None:
        """The turn `turn_index` of the `conversations` row `conversation`,
        open or closed, or `None` when it has no cursor (`cursor_at`)."""
        async with self._engine.connect() as conn:
            return await cursor_at(conn, conversation, turn_index)

    async def turn(self, turn_id: int) -> Cursor:
        """The turn `turn_id` as stored, open or closed.

        Raises:
            LookupError: no turn has that id.
        """
        async with self._engine.connect() as conn:
            cursor = await _read_cursor(conn, "t.id = :id", {"id": turn_id})
        if cursor is None:
            raise LookupError(f"no turn {turn_id}")
        return cursor

    async def load(self, cursor: Cursor) -> tuple[Cursor, list[AgentRecord]]:
        """The turn's cursor and transcript, read in one snapshot that holds
        the cursor at `cursor.version`, so a segment runs a node only on the
        turn as its cursor describes it.

        Raises:
            CursorStale: a write moved the cursor past `cursor.version`.
        """
        async with self._engine.connect() as plain:
            conn = await plain.execution_options(isolation_level="REPEATABLE READ")
            async with conn.begin():
                current = await _read_cursor(conn, "t.id = :id", {"id": cursor.id})
                if current is None or current.version != cursor.version:
                    raise CursorStale(
                        f"turn {cursor.turn_index} moved past version {cursor.version}"
                    )
                return current, await _records(conn, cursor.id)

    async def restore_frame_lineage(self, cursor: Cursor) -> None:
        """Put the stored lineage of each frame `cursor`'s state names back in
        `provenance` (`ProvenanceStore.restore_frame`), when it lacks one of
        them: a segment on another instance, or after a restart, taking over
        an open turn.

        Best effort: the one read of the turn's rows waits at most
        `_FRAME_LINEAGE_READ_BUDGET_S`, and a read that fails or runs past it
        is logged and restores nothing, so the turn goes on with what the
        process holds. A read past its budget runs as a task of its own and
        is left to finish its cleanup. A row that does not decode is logged
        and skipped."""
        provenance = self._provenance
        conversation = current_conversation_id.get()
        if provenance is None or conversation == NO_CONVERSATION:
            return
        if all(
            provenance.get(conversation=conversation, spec_id=spec_id, content_version=version)
            is not None
            for spec_id, version in _frame_keys(cursor.state)
        ):
            return
        read = asyncio.create_task(self._frame_lineage(cursor.id))
        try:
            done, _ = await asyncio.wait({read}, timeout=_FRAME_LINEAGE_READ_BUDGET_S)
        except asyncio.CancelledError:
            self._abandon(read)
            raise
        if not done:
            self._abandon(read)
            logger.warning(
                "turn %s goes on without its stored frame lineage: the read ran past %gs",
                cursor.turn_index,
                _FRAME_LINEAGE_READ_BUDGET_S,
            )
            return
        try:
            rows = read.result()
        except Exception:
            logger.warning(
                "turn %s goes on without its stored frame lineage: the read failed",
                cursor.turn_index,
                exc_info=True,
            )
            return
        for row in rows:
            try:
                lineage = lineage_from_json(row.members)
            except CardLineageDecodeError:
                logger.exception(
                    "turn store: skipping turn %s's lineage of frame spec_id=%r "
                    "content_version=%r, which does not decode",
                    cursor.turn_index,
                    row.spec_id,
                    row.content_version,
                )
                continue
            provenance.restore_frame(
                conversation=conversation,
                frame=(row.spec_id, row.content_version),
                lineage=lineage,
                stat_var_count=row.stat_var_count,
            )

    async def _frame_lineage(self, turn: int) -> list[Any]:
        """The turn's `turn_frame_lineage` rows, read over the lineage engine."""
        async with self._lineage_engine.connect() as conn:
            rows = await conn.execute(
                text(
                    "SELECT spec_id, content_version, members, stat_var_count "
                    "FROM turn_frame_lineage WHERE turn = :turn"
                ),
                {"turn": turn},
            )
            return list(rows.all())

    def _abandon(self, read: asyncio.Task[Any]) -> None:
        """Cancel `read` and hold it until it finishes, without waiting."""
        read.cancel()
        self._abandoned_reads.add(read)
        read.add_done_callback(self._abandoned_read_done)

    def _abandoned_read_done(self, read: asyncio.Task[Any]) -> None:
        self._abandoned_reads.discard(read)
        if not read.cancelled() and (exc := read.exception()) is not None:
            logger.warning("an abandoned frame lineage read failed while it finished", exc_info=exc)

    async def specs(self, cursor: Cursor) -> dict[str, dict[str, Any]]:
        """The turn's spec blocks by spec id, in the order the turn's state
        held them at the last write."""
        async with self._engine.connect() as conn:
            return (await _stored_specs(conn, cursor.id)).blocks

    async def records(self, cursor: Cursor) -> list[AgentRecord]:
        """The turn's transcript, in the order it was appended."""
        async with self._engine.connect() as conn:
            return await _records(conn, cursor.id)

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
        records: Sequence[AgentRecord] = (),
    ) -> Cursor:
        """Move the running turn to `node` with `node_input`, writing `state`,
        its spec blocks included, and appending `records`, with it.

        Raises:
            SegmentLost: `segment_id` does not hold the conversation's lease.
            CursorStale: the cursor moved, or the turn is not running.
        """
        async with self._engine.begin() as conn:
            await _fence(conn, cursor.conversation, segment_id)
            return await _swap(
                conn,
                cursor,
                segment_id=segment_id,
                statuses=(TurnStatus.RUNNING,),
                sets="node = :node, node_input = CAST(:node_input AS json), "
                "owed_interrupt_id = NULL",
                params={"node": node, "node_input": _json(node_input)},
                state=state,
                records=records,
                provenance=self._provenance,
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
        records: Sequence[AgentRecord] = (),
        stop_taken_id: str | None = None,
    ) -> Cursor:
        """Pause the running turn at `node` on a question of `kind` under the
        wire id `interrupt_id`, as the turn's next round, writing `state`,
        its spec blocks included, and appending `records`, with it. The
        returned cursor's `pause.round` is the round an answer names.
        `stop_taken_id`, when given, is the Stop the turn's `StopScope`
        answered, recorded in the same update; a pause without one keeps
        the Stop recorded before.

        Raises:
            SegmentLost: `segment_id` does not hold the conversation's lease.
            CursorStale: the cursor moved, or the turn is not running.
        """
        async with self._engine.begin() as conn:
            await _fence(conn, cursor.conversation, segment_id)
            return await _swap(
                conn,
                cursor,
                segment_id=segment_id,
                statuses=(TurnStatus.RUNNING,),
                sets="status = 'paused', round = round + 1, node = :node, "
                "node_input = CAST(:node_input AS json), pause_kind = :kind, "
                "interrupt_id = :interrupt_id, pause_payload = CAST(:payload AS json), "
                "owed_interrupt_id = NULL, "
                "stop_taken_id = COALESCE(CAST(:stop_taken_id AS text), stop_taken_id)",
                params={
                    "stop_taken_id": stop_taken_id,
                    "node": node,
                    "node_input": _json(node_input),
                    "kind": kind,
                    "interrupt_id": interrupt_id,
                    "payload": _json(payload),
                },
                state=state,
                records=records,
                provenance=self._provenance,
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
        the turn running at the node it paused at, appending `records`.
        The returned cursor's `owed` names the pause until the next write
        moves the cursor.

        Raises:
            PauseNotOpen: the turn has no open pause `(interrupt_id, round)`.
            SegmentLost: `segment_id` does not hold the conversation's lease.
            CursorStale: the cursor moved.
        """
        require_open_pause(cursor, interrupt_id=interrupt_id, round=round)
        async with self._engine.begin() as conn:
            await _fence(conn, cursor.conversation, segment_id)
            running = await _swap(
                conn,
                cursor,
                segment_id=segment_id,
                statuses=(TurnStatus.PAUSED,),
                sets="status = 'running', pause_kind = NULL, interrupt_id = NULL, "
                "pause_payload = NULL, owed_interrupt_id = :interrupt_id",
                params={"interrupt_id": interrupt_id},
                state=None,
                records=records,
            )
            await conn.execute(
                text(
                    "INSERT INTO turn_answers (turn, interrupt_id, round, answer) "
                    "VALUES (:turn, :interrupt_id, :round, CAST(:answer AS json))"
                ),
                {
                    "turn": cursor.id,
                    "interrupt_id": interrupt_id,
                    "round": round,
                    "answer": _json(answer),
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
        records: Sequence[AgentRecord] = (),
    ) -> Cursor:
        """Close the open turn with `status` in a transaction of its own
        (`finish_on`), stamping the runs a close `CANCELLED` ends with this
        store's clock.

        Raises:
            ValueError: `status` is not a closing status, or `error` is
                missing for `FAILED` or `ABANDONED` or given for another.
            SegmentLost: `segment_id` does not hold the conversation's lease.
            CursorStale: the cursor moved, or the turn is closed.
        """
        async with self._engine.begin() as conn:
            return await finish_on(
                conn,
                cursor,
                segment_id=segment_id,
                status=status,
                error=error,
                state=state,
                records=records,
                now=datetime.fromtimestamp(self._clock(), UTC),
            )


async def finish_on(
    conn: AsyncConnection,
    cursor: Cursor,
    *,
    segment_id: str,
    status: TurnStatus,
    now: datetime,
    error: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    records: Sequence[AgentRecord] = (),
) -> Cursor:
    """Close the open turn with `status`, one of `CLOSED`, on `conn`, in the
    caller's transaction, appending `records`. `FAILED` and `ABANDONED`
    take an `error` and the others none. `state`, its spec blocks included,
    replaces the stored state when given. A paused turn's pause is dropped,
    so no answer can reach it. A close `CANCELLED` records failed every
    unfinished run of the conversation in the same transaction
    (`end_unfinished`), stamped `now`, so when that fails the turn stays
    open as stored. A close reads no Stop. `TurnStore.finish` runs it in a
    transaction of its own; `harness.studio_store.StudioStore.close_segment`
    runs it in the transaction that saves the conversation's record.

    Raises:
        ValueError: `status` is not a closing status, or `error` is
            missing for `FAILED` or `ABANDONED` or given for another.
        SegmentLost: `segment_id` does not hold the conversation's lease.
        CursorStale: the cursor moved, or the turn is closed.
    """
    if status not in CLOSED:
        raise ValueError(f"{status!r} does not close a turn")
    if (status in _ERRORED) != (error is not None):
        raise ValueError(f"a turn closed {status.value} takes an error only when it failed")
    await _fence(conn, cursor.conversation, segment_id)
    closed = await _swap(
        conn,
        cursor,
        segment_id=segment_id,
        statuses=(TurnStatus.RUNNING, TurnStatus.PAUSED),
        sets="status = :status, error = CAST(:error AS json), pause_kind = NULL, "
        "interrupt_id = NULL, pause_payload = NULL, owed_interrupt_id = NULL",
        params={"status": status.value, "error": _json(error)},
        state=state,
        records=records,
    )
    if status is TurnStatus.CANCELLED:
        await end_unfinished(conn, cursor.conversation, now=now)
    return closed


async def cursor_at(conn: AsyncConnection, conversation: int, turn_index: int) -> Cursor | None:
    """The turn `turn_index` of the `conversations` row `conversation`, open
    or closed, read on `conn` in the caller's transaction, or `None` when it
    has no cursor."""
    return await _read_cursor(
        conn,
        "t.conversation = :conversation AND t.turn_index = :turn_index",
        {"conversation": conversation, "turn_index": turn_index},
    )
