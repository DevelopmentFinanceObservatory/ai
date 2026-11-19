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

The lineage of each frame the turn's state names is a `turn_frame_lineage`
row, kept while the turn is open, when the store is given a `FrameLineage`:
the graph's reading of which frames a state names, and the process's
`ProvenanceStore`. An `advance` or `pause` writes a row for each named frame
whose lineage the process holds at a generation (`ProvenanceStore.generation`)
it has not written for the turn since, so a lineage replaced in process, or
held again after an eviction or a skipped write, is written at the next
write. A row the database refuses is logged and not tried again until its
generation changes, and the other rows of the write land. A segment that
lacks one of the named frames reads the rows of those it has not already
found absent (`TurnStore.restore_frame_lineage`) and puts them back. The
closing write deletes the turn's rows. A frame whose lineage no process
held, or that JSON cannot hold, has no row, and the write goes on without
it. The rows are best effort: a card drawn on a frame with none is
withheld, as on the instance that fetched it once the lineage is evicted.

The lease itself, the Stop request and the frame numbering live on
`conversations` and are written by `harness.studio_store.StudioStore`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from sqlalchemy import ARRAY, Text, bindparam, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from harness.db.best_effort import bounded_read, execute_in_savepoint
from harness.db.jsonb import json_or_none, to_json_param
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
# The longest the one read of a turn's stored frame lineage may take
# (`TurnStore.restore_frame_lineage`), the bound a card's stored lineage read
# has (`harness.mcp_client._LINEAGE_READ_BUDGET_S`). The read is best effort
# and the segment waits for it before its node runs.
_FRAME_LINEAGE_READ_BUDGET_S = 2.0

# The turns whose frame lineage a store remembers writing and reading
# (`_TurnLineage`), least recently used dropped past it. A turn closed on
# another instance is never closed here, so this bound is what drops it.
_MAX_TURNS_HELD = 256

# A frame store key: `(spec_id, content_version)`.
type FrameKey = tuple[str, str]


@dataclass(frozen=True)
class FrameLineage:
    """What the store keeps a turn's frame lineage from: `frame_keys`, the
    store keys of the frames a turn's state names, read the way the graph
    that wrote the state names them, and `source`, the process's frame
    lineage store."""

    source: ProvenanceStore
    frame_keys: Callable[[Mapping[str, Any]], Iterable[FrameKey]]


@dataclass
class _TurnLineage:
    """What this process knows of one turn's stored frame lineage as of the
    cursor at `version`. `written` holds, for each key, the source's
    generation (`ProvenanceStore.generation`) of the lineage this process
    stored or read back; `failed` the generation of a lineage whose row could
    not be stored (JSON cannot hold it, or the database refused the row),
    not tried again until the generation changes; and `absent` the keys a
    read found no usable row for."""

    version: int
    written: dict[FrameKey, int] = field(default_factory=dict)
    failed: dict[FrameKey, int] = field(default_factory=dict)
    absent: set[FrameKey] = field(default_factory=set)


@dataclass(frozen=True)
class _PendingRow:
    """One `turn_frame_lineage` row a write stores: its key, the source's
    generation of its lineage, and the statement's parameters."""

    key: FrameKey
    generation: int
    params: dict[str, Any]


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


_UPSERT_FRAME_LINEAGE = text(
    "INSERT INTO turn_frame_lineage (turn, spec_id, content_version, members, stat_var_count) "
    "VALUES (:turn, :spec_id, :content_version, CAST(:members AS json), :stat_var_count) "
    "ON CONFLICT (turn, spec_id, content_version) DO UPDATE "
    "SET members = EXCLUDED.members, stat_var_count = EXCLUDED.stat_var_count"
)
_DELETE_FRAME_LINEAGE = text("DELETE FROM turn_frame_lineage WHERE turn = :turn")
_SELECT_FRAME_LINEAGE = text(
    "SELECT l.spec_id, l.content_version, l.members, l.stat_var_count "
    "FROM turn_frame_lineage l "
    "JOIN unnest(:spec_ids, :content_versions) AS k (spec_id, content_version) "
    "ON l.spec_id = k.spec_id AND l.content_version = k.content_version "
    "WHERE l.turn = :turn"
).bindparams(
    bindparam("spec_ids", type_=ARRAY(Text)), bindparam("content_versions", type_=ARRAY(Text))
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
) -> Cursor:
    """Apply `sets`, bump the version and record `segment_id` as the
    writer where the row holds the version `cursor` read and one of
    `statuses`, and append `records`. When `state`
    is given, store it with its spec blocks as rows (`_write_specs`).
    Otherwise keep the stored blocks `cursor` holds."""
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
    await _append(conn, cursor, records)
    return _cursor(row, cursor.record_count + len(records), stored)


class TurnStore:
    """The turn cursor API over `engine` (module docstring). Every write
    takes the writer's `segment_id`, the conversation lease it holds, and the
    `Cursor` it last read or wrote, and returns the cursor as written.
    `clock` is the `time.time`-shaped clock the runs a close `cancelled`
    ends are stamped with. With `lineage`, the store keeps the lineage of
    each frame an open turn names with the turn, and puts it back for a
    segment that lacks it (module docstring), reading it over
    `lineage_engine` (`harness.db.engine.DatabaseHandle.lineage_engine`,
    `engine` when none is given). Without it no lineage is kept."""

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        clock: Callable[[], float] = time.time,
        lineage: FrameLineage | None = None,
        lineage_engine: AsyncEngine | None = None,
    ) -> None:
        self._engine = engine
        self._clock = clock
        self._lineage = lineage
        self._lineage_engine = lineage_engine or engine
        self._turns: OrderedDict[int, _TurnLineage] = OrderedDict()
        # Reads past their budget, cancelled and held until they finish
        # their cleanup (`harness.db.best_effort.bounded_read`).
        self._abandoned_reads: set[asyncio.Task[Any]] = set()

    def _turn_lineage(self, cursor: Cursor) -> _TurnLineage:
        """What this process knows of `cursor`'s turn's stored lineage. When
        another writer moved the cursor since this process last wrote or read
        it, the rows may have changed, and the turn starts afresh."""
        known = self._turns.get(cursor.id)
        if known is None or known.version != cursor.version:
            known = _TurnLineage(version=cursor.version)
            self._turns[cursor.id] = known
        self._turns.move_to_end(cursor.id)
        while len(self._turns) > _MAX_TURNS_HELD:
            self._turns.popitem(last=False)
        return known

    def _moved(self, before: Cursor, after: Cursor) -> None:
        """Carry what this process knows of the turn from `before` to the
        cursor its own write left, `after`."""
        known = self._turns.get(before.id)
        if known is not None and known.version == before.version:
            known.version = after.version

    def _lineage_rows(
        self, cursor: Cursor, state: Mapping[str, Any], scope: str | None
    ) -> list[_PendingRow]:
        """The `turn_frame_lineage` rows a write of `state` from `cursor`
        stores: one per store key `state` names whose lineage the process
        holds in conversation `scope` at a generation it has neither stored
        for the turn since nor failed to store (`_TurnLineage`), its members
        as JSON text. A key whose lineage JSON cannot hold (a NaN in a
        manifest) has no row, is logged and is recorded as failed at its
        generation. None without `lineage` or `scope`."""
        if self._lineage is None or scope is None:
            return []
        source = self._lineage.source
        known = self._turn_lineage(cursor)
        rows: list[_PendingRow] = []
        for key in dict.fromkeys(self._lineage.frame_keys(state)):
            spec_id, content_version = key
            generation = source.generation(
                conversation=scope, spec_id=spec_id, content_version=content_version
            )
            if generation is None or generation in (known.written.get(key), known.failed.get(key)):
                continue
            lineage = source.get(
                conversation=scope, spec_id=spec_id, content_version=content_version
            )
            assert lineage is not None  # the source holds a generation for the key
            members = json_or_none(
                lineage_json(lineage),
                logger=logger,
                failure=lambda key=key: (
                    f"turn store: the lineage of frame spec_id={key[0]!r} "
                    f"content_version={key[1]!r} cannot be stored as JSON; "
                    f"turn {cursor.turn_index} goes on without it"
                ),
            )
            if members is None:
                known.failed[key] = generation
                continue
            rows.append(
                _PendingRow(
                    key=key,
                    generation=generation,
                    params={
                        "turn": cursor.id,
                        "spec_id": spec_id,
                        "content_version": content_version,
                        "members": members,
                        "stat_var_count": source.stat_var_count(
                            conversation=scope, spec_id=spec_id, content_version=content_version
                        ),
                    },
                )
            )
        return rows

    async def _write_lineage(
        self, conn: AsyncConnection, cursor: Cursor, rows: Sequence[_PendingRow]
    ) -> tuple[dict[FrameKey, int], dict[FrameKey, int]]:
        """Upsert `rows` (`_lineage_rows`) in the write's transaction, in one
        savepoint (`execute_in_savepoint`), and return the generations stored
        and those that failed, by key. A batch the database refuses is tried
        again row by row, each in a savepoint of its own, so one bad row
        keeps no other from landing; each row that fails is logged."""
        if not rows:
            return {}, {}
        if len(rows) > 1 and await execute_in_savepoint(
            conn,
            _UPSERT_FRAME_LINEAGE,
            [row.params for row in rows],
            logger=None,
            failure=lambda: "",
        ):
            return {row.key: row.generation for row in rows}, {}
        stored: dict[FrameKey, int] = {}
        failed: dict[FrameKey, int] = {}
        for row in rows:
            landed = await execute_in_savepoint(
                conn,
                _UPSERT_FRAME_LINEAGE,
                row.params,
                logger=logger,
                failure=lambda row=row: (
                    f"turn store: the lineage of frame spec_id={row.key[0]!r} "
                    f"content_version={row.key[1]!r} was not stored; turn "
                    f"{cursor.turn_index} goes on without it"
                ),
            )
            (stored if landed else failed)[row.key] = row.generation
        return stored, failed

    def _stored(
        self,
        before: Cursor,
        after: Cursor,
        written: tuple[Mapping[FrameKey, int], Mapping[FrameKey, int]],
    ) -> None:
        """Record a committed write from `before` to `after`, which stored the
        first of `written` and failed to store the second."""
        self._moved(before, after)
        stored, failed = written
        if stored or failed:
            known = self._turn_lineage(after)
            known.written.update(stored)
            known.failed.update(failed)
            known.absent.difference_update(stored)

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

    async def restore_frame_lineage(self, cursor: Cursor, *, lineage_scope: str | None) -> None:
        """Put the stored lineage of each frame `cursor`'s state names and the
        `FrameLineage` source lacks in conversation `lineage_scope` back in it
        (`ProvenanceStore.restore_frame`): a segment on another instance, or
        after a restart, taking over an open turn. Rows of frames the state no
        longer names, or the source holds, are left unread.

        A key a read found no usable row for is not read again by this
        process until another writer moves the cursor, so a frame with no row
        anywhere costs one read, and a frame evicted after an earlier read is
        read at the next segment that lacks it. Best effort: the read waits at most `_FRAME_LINEAGE_READ_BUDGET_S`,
        and one that fails or runs past it is logged, restores nothing and is
        tried again at the next segment (`bounded_read`). A row that does not
        decode is logged and skipped. Nothing happens without `lineage` or
        `lineage_scope`."""
        if self._lineage is None or lineage_scope is None:
            return
        source = self._lineage.source
        known = self._turn_lineage(cursor)
        missing = [
            (spec_id, content_version)
            for spec_id, content_version in dict.fromkeys(self._lineage.frame_keys(cursor.state))
            if source.get(
                conversation=lineage_scope, spec_id=spec_id, content_version=content_version
            )
            is None
        ]
        unread = [key for key in missing if key not in known.absent]
        if not unread:
            return
        rows = await bounded_read(
            self._frame_lineage(cursor.id, unread),
            budget_s=_FRAME_LINEAGE_READ_BUDGET_S,
            held=self._abandoned_reads,
            logger=logger,
            what=lambda: (
                f"turn {cursor.turn_index} goes on without its stored frame lineage: the read"
            ),
        )
        if rows is None:
            return
        known.absent.update(unread)
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
            key = (row.spec_id, row.content_version)
            source.restore_frame(
                conversation=lineage_scope,
                frame=key,
                lineage=lineage,
                stat_var_count=row.stat_var_count,
            )
            known.absent.discard(key)
            generation = source.generation(
                conversation=lineage_scope, spec_id=row.spec_id, content_version=row.content_version
            )
            if generation is not None:
                known.written[key] = generation

    async def _frame_lineage(self, turn: int, keys: Sequence[FrameKey]) -> list[Any]:
        """The turn's `turn_frame_lineage` rows of `keys`, read over the
        lineage engine."""
        async with self._lineage_engine.connect() as conn:
            rows = await conn.execute(
                _SELECT_FRAME_LINEAGE,
                {
                    "turn": turn,
                    "spec_ids": [spec_id for spec_id, _ in keys],
                    "content_versions": [content_version for _, content_version in keys],
                },
            )
            return list(rows.all())

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
        lineage_scope: str | None,
    ) -> Cursor:
        """Move the running turn to `node` with `node_input`, writing `state`,
        its spec blocks included, and appending `records`, with it.
        `lineage_scope` is the conversation the store's `FrameLineage` source
        holds the turn's frames under; with `None` no lineage is written.

        Raises:
            SegmentLost: `segment_id` does not hold the conversation's lease.
            CursorStale: the cursor moved, or the turn is not running.
        """
        rows = self._lineage_rows(cursor, state, lineage_scope)
        async with self._engine.begin() as conn:
            await _fence(conn, cursor.conversation, segment_id)
            advanced = await _swap(
                conn,
                cursor,
                segment_id=segment_id,
                statuses=(TurnStatus.RUNNING,),
                sets="node = :node, node_input = CAST(:node_input AS json), "
                "owed_interrupt_id = NULL",
                params={"node": node, "node_input": _json(node_input)},
                state=state,
                records=records,
            )
            written = await self._write_lineage(conn, cursor, rows)
        self._stored(cursor, advanced, written)
        return advanced

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
        lineage_scope: str | None,
    ) -> Cursor:
        """Pause the running turn at `node` on a question of `kind` under the
        wire id `interrupt_id`, as the turn's next round, writing `state`,
        its spec blocks included, and appending `records`, with it. The
        returned cursor's `pause.round` is the round an answer names.
        `stop_taken_id`, when given, is the Stop the turn's `StopScope`
        answered, recorded in the same update; a pause without one keeps
        the Stop recorded before. `lineage_scope` is as `advance` takes it.

        Raises:
            SegmentLost: `segment_id` does not hold the conversation's lease.
            CursorStale: the cursor moved, or the turn is not running.
        """
        rows = self._lineage_rows(cursor, state, lineage_scope)
        async with self._engine.begin() as conn:
            await _fence(conn, cursor.conversation, segment_id)
            paused = await _swap(
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
            )
            written = await self._write_lineage(conn, cursor, rows)
        self._stored(cursor, paused, written)
        return paused

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
        self._moved(cursor, running)
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
            closed = await finish_on(
                conn,
                cursor,
                segment_id=segment_id,
                status=status,
                error=error,
                state=state,
                records=records,
                now=datetime.fromtimestamp(self._clock(), UTC),
            )
        self._turns.pop(cursor.id, None)
        return closed


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
    caller's transaction, appending `records`, and delete the turn's frame
    lineage rows, best effort: a delete that fails for good is logged and
    the close commits (`execute_in_savepoint`). `FAILED` and `ABANDONED`
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
    await execute_in_savepoint(
        conn,
        _DELETE_FRAME_LINEAGE,
        {"turn": cursor.id},
        logger=logger,
        failure=lambda: (
            f"turn store: the frame lineage of closed turn {cursor.turn_index} was not "
            "deleted; the close goes on"
        ),
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
