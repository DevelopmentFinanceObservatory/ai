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
`ProvenanceStore`. The rows commit with the cursor write or not at all, so
the rows a turn's cursor write commits are the lineage the process held for
every frame the state names, at that write. Each segment starts by making
its `SegmentLineage` (`TurnStore.start_segment`): the process's last
generation then (its restore point) and, as the segment goes on, the
generations it reads back and writes. The restore fills it by reading the
rows of every frame the state names, putting each over what the process
holds, in the state's frame order, and dropping a named frame with no row
(`TurnStore.restore_frame_lineage`), and the segment passes it to each of
its writes. An `advance` or `pause` writes each named frame
whose lineage the process holds at a generation (`ProvenanceStore.held`)
produced or restored after that point and not written in the segment
since, so an entry from before it that no read confirmed never reaches the
rows, and a failed read leaves them as they are. Each row is checked before
the transaction opens (JSON that encodes, a count Postgres `integer`
holds); a lineage that fails the check deletes its frame's row instead,
logged once per generation, so nothing the database receives can be
refused for its values, and any refusal fails the cursor write like a
refusal of the cursor statement. A key holding a NUL or text that is not
UTF-8 has no row. The closing write deletes the turn's rows. A frame whose
lineage no process held has no row: a card drawn on it is withheld, as on
the instance that fetched it once the lineage is evicted.

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
from typing import Any, Literal, NamedTuple

from sqlalchemy import ARRAY, Text, bindparam, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from harness.db.best_effort import bounded_read
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

# A frame store key: `(spec_id, content_version)`.
type FrameKey = tuple[str, str]


class FrameKeys(NamedTuple):
    """The store keys a turn's state names, in order and each once (`keys`),
    and what named none because it is not a mapping: the type of the frames
    value itself (`container`), or each frames entry as `frame_id:type`
    (`skipped`)."""

    keys: list[FrameKey]
    skipped: list[str]
    container: str | None = None


@dataclass(frozen=True)
class FrameLineage:
    """What the store keeps a turn's frame lineage from: `frame_keys`, the
    store keys of the frames a turn's state names, read the way the graph
    that wrote the state names them, with the entries it skipped as not
    mappings (`FrameKeys`), and `source`, the process's frame lineage
    store."""

    source: ProvenanceStore
    frame_keys: Callable[[Mapping[str, Any]], FrameKeys]


# The range of a Postgres `integer`, the `stat_var_count` column.
_INTEGER_RANGE = range(-(2**31), 2**31)


@dataclass
class SegmentLineage:
    """One segment's frame lineage bookkeeping, which
    `TurnStore.start_segment` makes and the segment's restore and each of its
    `advance` and `pause` writes takes: the conversation the source holds
    the turn's frames under (`scope`), the source's last generation
    (`ProvenanceStore.last_generation`) at the segment's restore point
    (`since`), before which no entry is written unless a read confirmed it,
    and for each key the source's generation (`ProvenanceStore.held`) of the
    lineage the segment committed (a row, or the delete of one whose lineage
    failed its check) or read back (`written`). `token` is the segment's
    identity in the source (`ProvenanceStore.segment_token`), which its kept
    frames are recorded under (`ProvenanceStore.keep`); tokens carry no
    order. `reported` is set once the segment has logged its state's
    malformed frames (`TurnStore._named`), so it logs them once."""

    scope: str
    since: int
    token: int
    written: dict[FrameKey, int] = field(default_factory=dict)
    reported: bool = False


@dataclass
class _LineageWrite:
    """What a cursor write commits of the turn's frame lineage: the rows to
    upsert, the keys whose rows to delete, and the generation each key
    records as written once the write commits."""

    upserts: list[dict[str, Any]] = field(default_factory=list)
    deletes: list[dict[str, Any]] = field(default_factory=list)
    generations: dict[FrameKey, int] = field(default_factory=dict)


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


class WriteNotEncoded(ValueError):  # noqa: N818 -- names the write's state, as the others do
    """A write's value JSON cannot hold: a NaN or an infinity
    (`executor.ledger.NonFiniteJsonError`), or a value JSON cannot encode,
    which is the cause. Values are encoded only while a write builds its
    parameters, before its transaction commits, so the transaction ends
    without committing and the write stored nothing."""


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


def _encode(value: Any) -> str:
    """`value` as JSON text for a json parameter.

    Raises:
        WriteNotEncoded: JSON cannot hold `value`.
    """
    try:
        return to_json_param(value)
    except (ValueError, TypeError) as exc:
        raise WriteNotEncoded(f"a turn write's value cannot be stored as JSON: {exc}") from exc


def _json(value: Any) -> str | None:
    """`value` as a json parameter: JSON text, or SQL NULL for `None`.

    Raises:
        WriteNotEncoded: JSON cannot hold `value`.
    """
    return None if value is None else _encode(value)


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
        body = _encode(block)
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
_DELETE_FRAME_LINEAGE_KEY = text(
    "DELETE FROM turn_frame_lineage WHERE turn = :turn AND spec_id = :spec_id "
    "AND content_version = :content_version"
)


def _text_key(key: FrameKey) -> bool:
    """Whether both parts of `key` are text Postgres can hold: UTF-8, with no
    NUL character."""
    try:
        for part in key:
            part.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return not any("\x00" in part for part in key)


async def _write_lineage(conn: AsyncConnection, write: _LineageWrite) -> None:
    """Commit `write` (`TurnStore._lineage_rows`) in the cursor write's
    transaction: the deletes, then the upserts in one statement. An error
    from either fails the write as an error of the cursor statement does."""
    if write.deletes:
        await conn.execute(_DELETE_FRAME_LINEAGE_KEY, write.deletes)
    if write.upserts:
        await conn.execute(_UPSERT_FRAME_LINEAGE, write.upserts)


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


# Waits for a write in flight to the turn's cursor row (`settled_turn`).
_SETTLE = text("SELECT 1 FROM turn_cursors WHERE id = :id FOR SHARE")

# The longest `settled_turn` waits for the turn's row lock, and the longest
# each of its statements runs. A write whose connection dropped before its
# COMMIT reached the server can hold the lock until the server notices.
SETTLE_TIMEOUT_S = 2.0


class TurnStore:
    """The turn cursor API over `engine` (module docstring). Every write
    takes the writer's `segment_id`, the conversation lease it holds, and the
    `Cursor` it last read or wrote, and returns the cursor as written.
    `clock` is the `time.time`-shaped clock the runs a close `cancelled`
    ends are stamped with. With `lineage`, the store keeps the lineage of
    each frame an open turn names with the turn, and puts it back for a
    segment that takes the turn over (module docstring), reading it over
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
        self._lineage_engine = lineage_engine or engine
        self._clock = clock
        self._lineage = lineage
        # Reads past their budget, cancelled and held until they finish
        # their cleanup (`harness.db.best_effort.bounded_read`).
        self._abandoned_reads: set[asyncio.Task[Any]] = set()

    def _lineage_rows(
        self, cursor: Cursor, state: Mapping[str, Any], segment: SegmentLineage | None
    ) -> _LineageWrite:
        """What a write of `state` from `cursor` commits of the turn's frame
        lineage: for each store key `state` names whose lineage the process
        holds in the segment's conversation at a generation produced or
        restored after its restore point (`SegmentLineage.since`) and not
        written in the segment since (`SegmentLineage.written`), its row with its members as
        JSON text, or, when the lineage fails the check (JSON cannot hold it,
        or its count is past Postgres `integer`), a delete of its row,
        logged. A key whose text is not UTF-8 (a lone surrogate) can have no
        row and is only logged. The source keeps every named key past its
        per-conversation bound (`ProvenanceStore.keep`). Nothing without
        `lineage` or `segment`."""
        write = _LineageWrite()
        if self._lineage is None or segment is None:
            return write
        source, scope = self._lineage.source, segment.scope
        turn_index = cursor.turn_index
        named = self._named(cursor, state, segment)
        source.keep(conversation=scope, keys=named, token=segment.token)
        for key in named:
            spec_id, content_version = key
            held = source.held(conversation=scope, spec_id=spec_id, content_version=content_version)
            if held is None:
                continue
            lineage, count, generation = held
            # An entry from before the restore point that no read confirmed
            # can be older than the turn's row, so it is never written.
            if generation <= segment.since or segment.written.get(key) == generation:
                continue
            write.generations[key] = generation
            if not _text_key(key):
                logger.error(
                    "turn store: frame %r of turn %s has a key Postgres text cannot hold; "
                    "it gets no lineage row",
                    key,
                    turn_index,
                )
                continue
            ref = {"turn": cursor.id, "spec_id": spec_id, "content_version": content_version}
            members = json_or_none(
                lineage_json(lineage),
                logger=logger,
                failure=lambda key=key: (
                    f"turn store: the lineage of frame spec_id={key[0]!r} "
                    f"content_version={key[1]!r} of turn {turn_index} cannot be stored as "
                    "JSON; its row is deleted"
                ),
            )
            if members is None:
                write.deletes.append(ref)
                continue
            if count is not None and count not in _INTEGER_RANGE:
                logger.error(
                    "turn store: the count %d of frame spec_id=%r content_version=%r of turn "
                    "%s is past the integer column; its row is deleted",
                    count,
                    spec_id,
                    content_version,
                    turn_index,
                )
                write.deletes.append(ref)
                continue
            write.upserts.append({**ref, "members": members, "stat_var_count": count})
        return write

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

    async def settled_turn(self, turn_id: int) -> Cursor:
        """The turn `turn_id` as stored once every write to its cursor that
        is in flight has committed or rolled back: the read first takes the
        `turn_cursors` row `FOR SHARE`, which waits for such a write's row
        lock, then reads the turn in a statement that sees what it committed.
        The wait for the lock and each statement are bounded at
        `SETTLE_TIMEOUT_S` (`lock_timeout`, `statement_timeout`). A lock held
        past it belongs to a stalled or orphaned write, so the read fails and
        the caller leaves the turn as stored, for the next segment, a Stop or
        the conversation's next turn to settle. The cost: a write that
        commits after the bound is not reported by the segment that made
        it.

        Raises:
            LookupError: no turn has that id.
            sqlalchemy.exc.DBAPIError: the lock or a statement ran past
                `SETTLE_TIMEOUT_S`, or the database failed the read.
        """
        bound = round(SETTLE_TIMEOUT_S * 1000)
        async with self._engine.begin() as conn:
            await conn.exec_driver_sql(f"SET LOCAL lock_timeout = {bound}")
            await conn.exec_driver_sql(f"SET LOCAL statement_timeout = {bound}")
            await conn.execute(_SETTLE, {"id": turn_id})
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

    def start_segment(self, *, lineage_scope: str | None) -> SegmentLineage | None:
        """The `SegmentLineage` of a segment starting now in conversation
        `lineage_scope`: its restore point is the source's last generation
        now (`SegmentLineage.since`), and its token is new. Synchronous, so
        the caller holds the value before any await and ends it however the
        segment ends (`end_segment`). `None` without `lineage` or
        `lineage_scope`: the segment writes no lineage."""
        if self._lineage is None or lineage_scope is None:
            return None
        source = self._lineage.source
        return SegmentLineage(
            scope=lineage_scope, since=source.last_generation(), token=source.segment_token()
        )

    async def restore_frame_lineage(self, cursor: Cursor, segment: SegmentLineage | None) -> None:
        """Bring the `FrameLineage` source's lineage of each frame `cursor`'s
        state names in `segment`'s conversation into step with the turn's
        rows, which commit with the cursor, at the start of the segment
        (`start_segment`).

        From the segment's restore point on, its writes store only entries
        produced or restored after it. The source keeps every named frame
        past its per-conversation bound (`ProvenanceStore.keep`) until the
        segment ends (`end_segment`). One read takes the rows of every named
        frame. Each row is put over what the source holds
        (`ProvenanceStore.restore_frame` with `overwrite`), in the state's
        frame order, and a named frame with no row is dropped from the
        source, since the turn vouches for no lineage of it. Rows of frames
        the state no longer names are left unread, and so are frames whose
        key Postgres text cannot hold. A frames value or entry that is not a
        mapping names no frame and is logged here, once per segment; the
        segment's writes skip it without a word.

        Best effort: the read waits at most `_FRAME_LINEAGE_READ_BUDGET_S`,
        and one that fails or runs past it is logged and restores nothing
        (`bounded_read`); the source keeps what it held, which the segment's
        writes then leave unwritten, so the rows stay as they are. Any other
        fault is logged and the segment goes on. A row that does not decode
        is logged, and its frame dropped. Nothing without `lineage` or
        `segment`."""
        if self._lineage is None or segment is None:
            return
        try:
            await self._restore(cursor, segment)
        except Exception:
            logger.exception(
                "turn %s goes on without its stored frame lineage: the restore failed",
                cursor.turn_index,
            )

    def end_segment(self, segment: SegmentLineage | None) -> None:
        """End `segment`: the frames it kept outside the source's
        per-conversation bound (`ProvenanceStore.release`) count against it
        again, unless another running segment of the conversation keeps
        them."""
        if self._lineage is not None and segment is not None:
            self._lineage.source.release(conversation=segment.scope, token=segment.token)

    def _named(
        self, cursor: Cursor, state: Mapping[str, Any], segment: SegmentLineage
    ) -> list[FrameKey]:
        """The store keys `state` names (`FrameLineage.frame_keys`). What
        names none because it is not a mapping is logged the first time the
        segment meets it, at its restore or a write (`SegmentLineage.
        reported`)."""
        assert self._lineage is not None
        frames = self._lineage.frame_keys(state)
        if not segment.reported and (frames.container is not None or frames.skipped):
            segment.reported = True
            if frames.container is not None:
                logger.warning(
                    "turn %s of conversation %s: its state's frames value is a %s, not a "
                    "mapping; it names no frame",
                    cursor.turn_index,
                    segment.scope,
                    frames.container,
                )
            else:
                logger.warning(
                    "turn %s of conversation %s: its state's frames hold %d entries that "
                    "are not mappings (%s); they name no frame",
                    cursor.turn_index,
                    segment.scope,
                    len(frames.skipped),
                    ", ".join(frames.skipped),
                )
        return frames.keys

    async def _restore(self, cursor: Cursor, segment: SegmentLineage) -> None:
        """The read and restore of `restore_frame_lineage`, into `segment`."""
        assert self._lineage is not None
        source, scope = self._lineage.source, segment.scope
        named = self._named(cursor, cursor.state, segment)
        source.keep(conversation=scope, keys=named, token=segment.token)
        readable = [key for key in named if _text_key(key)]
        if not readable:
            return
        turn_index = cursor.turn_index
        rows = await bounded_read(
            self._frame_lineage(cursor.id, readable),
            budget_s=_FRAME_LINEAGE_READ_BUDGET_S,
            held=self._abandoned_reads,
            logger=logger,
            what=lambda: f"turn {turn_index} goes on without its stored frame lineage: the read",
        )
        if rows is None:
            return
        by_key = {(row.spec_id, row.content_version): row for row in rows}
        for key in readable:
            spec_id, content_version = key
            row = by_key.get(key)
            lineage = None
            if row is not None:
                try:
                    lineage = lineage_from_json(row.members)
                except CardLineageDecodeError:
                    logger.exception(
                        "turn store: dropping turn %s's lineage of frame spec_id=%r "
                        "content_version=%r, whose row does not decode",
                        turn_index,
                        spec_id,
                        content_version,
                    )
            if row is None or lineage is None:
                source.drop(conversation=scope, spec_id=spec_id, content_version=content_version)
                continue
            generation = source.restore_frame(
                conversation=scope,
                frame=key,
                lineage=lineage,
                stat_var_count=row.stat_var_count,
                overwrite=True,
            )
            assert generation is not None  # an overwrite always stores
            segment.written[key] = generation

    async def _frame_lineage(self, turn: int, keys: Sequence[FrameKey]) -> list[Any]:
        """The turn's `turn_frame_lineage` rows of `keys`, read over the
        lineage engine, off the pool the cursor writes and lease renewals
        use."""
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
        lineage: SegmentLineage | None,
    ) -> Cursor:
        """Move the running turn to `node` with `node_input`, writing `state`,
        its spec blocks included, and appending `records`, with it.
        `lineage` is the segment's, which `start_segment` made and
        `restore_frame_lineage` filled; with `None` no lineage is written.

        Raises:
            SegmentLost: `segment_id` does not hold the conversation's lease.
            CursorStale: the cursor moved, or the turn is not running.
            WriteNotEncoded: JSON cannot hold a value of the write.
        """
        write = self._lineage_rows(cursor, state, lineage)
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
            await _write_lineage(conn, write)
        if lineage is not None:
            lineage.written.update(write.generations)
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
        lineage: SegmentLineage | None,
    ) -> Cursor:
        """Pause the running turn at `node` on a question of `kind` under the
        wire id `interrupt_id`, as the turn's next round, writing `state`,
        its spec blocks included, and appending `records`, with it. The
        returned cursor's `pause.round` is the round an answer names.
        `stop_taken_id`, when given, is the Stop the turn's `StopScope`
        answered, recorded in the same update; a pause without one keeps
        the Stop recorded before. `lineage` is as `advance` takes it.

        Raises:
            SegmentLost: `segment_id` does not hold the conversation's lease.
            CursorStale: the cursor moved, or the turn is not running.
            WriteNotEncoded: JSON cannot hold a value of the write.
        """
        write = self._lineage_rows(cursor, state, lineage)
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
            await _write_lineage(conn, write)
        if lineage is not None:
            lineage.written.update(write.generations)
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
            WriteNotEncoded: JSON cannot hold a value of the write.
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
    lineage rows in the same transaction. `FAILED` and `ABANDONED`
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
    await conn.execute(_DELETE_FRAME_LINEAGE, {"turn": cursor.id})
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
