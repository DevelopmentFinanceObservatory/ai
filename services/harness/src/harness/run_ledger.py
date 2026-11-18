"""The run ledger kept in the harness's own tables.

`StudioRunLedger` is the `executor.ledger.RunLedger` the studio harness runs
under. Each run is one `runs` row: its record, the canonical submission it
executes (written with the admission and never again) and the `get_data`
results its inputs were sealed from (written with each seal that changes the
inputs). Each idempotency key is one `run_keys` row naming its run, and the
plan that compiled a run's work one `run_plans` row per revision, written
once.

Every write is one transaction fenced on the conversation's live segment
(`StudioStore.fenced`): a writer whose segment another instance took over
gets `SegmentLost` and records nothing, publication included, and ends the
conversation's stream, and a takeover waits for an in-flight write to
commit. Two admissions of one key settle on the `run_keys` primary key: the
second waits for the first to commit and then decides against its run. A record change locks its run row. Nothing
retries.

A stored run is read with `executor.ledger.stored_run`, so one another
release wrote loads. A stored run this release still cannot read is recorded
failed (`run_record_unreadable`) where it is found unfinished or its key
admits work again, its status and failure written over the stored record.

The ledger finds its conversation the way the recorder does:
`current_conversation_key`, set by `harness.studio` for each streamed
segment. Outside a conversation it reads nothing and refuses every write
with `SegmentLost`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from executor.ledger import (
    TERMINAL_STATUSES,
    Admission,
    RunChange,
    RunNotFoundError,
    RunRecord,
    RunStatus,
    Seal,
    admit_against,
    advance,
    stored_run,
)
from pydantic import JsonValue, ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from harness.conversation import current_conversation_key
from harness.studio_state import SegmentLost, _end_stream, stream_loss
from harness.studio_store import StudioStore
from harness.tool_contracts import HarnessCode

logger = logging.getLogger(__name__)

_TERMINAL = frozenset(status.value for status in TERMINAL_STATUSES)

_SCOPED_RUNS = (
    "FROM runs r JOIN conversations c ON c.id = r.conversation "
    "WHERE c.principal = :principal AND c.conversation_id = :conversation_id"
)


def _unreadable(raw: Any, reason: str) -> dict[str, Any]:
    """Stored run record `raw`, which this release cannot read, recorded failed."""
    return {
        **raw,
        "status": RunStatus.FAILED.value,
        "failure": {"code": HarnessCode.RUN_RECORD_UNREADABLE.value, "reason": reason},
    }


def _status_of(raw: Any) -> str | None:
    """The status a stored run record names, read without validating the rest."""
    status = raw.get("status") if isinstance(raw, dict) else None
    return status if isinstance(status, str) else None


async def _put_record(conn: AsyncConnection, pk: int, run_id: str, record: Any) -> None:
    await conn.execute(
        text(
            "UPDATE runs SET record = CAST(:record AS jsonb) "
            "WHERE conversation = :pk AND run_id = :run_id"
        ),
        {"pk": pk, "run_id": run_id, "record": json.dumps(record)},
    )


class StudioRunLedger:
    """A `RunLedger` over `store`'s conversations."""

    def __init__(self, store: StudioStore) -> None:
        self._store = store

    def _now(self) -> datetime:
        return datetime.fromtimestamp(self._store.now(), UTC)

    def _segment(self) -> tuple[str, str, str]:
        """The current conversation's `(principal, conversation_id,
        segment_id)`, the segment every write is fenced on."""
        key = current_conversation_key.get()
        state = self._store.current(*key) if key is not None else None
        if key is None or state is None or state.segment is None:
            raise SegmentLost("no conversation segment is streaming on this instance")
        return key[0], key[1], state.segment.id

    @asynccontextmanager
    async def _fenced(self) -> AsyncIterator[tuple[AsyncConnection, int]]:
        """`StudioStore.fenced` on the current segment. A `SegmentLost` ends
        the conversation's stream (`_end_stream`) before it propagates:
        `executor.ledger.store_faults` reports it to the graph as an
        unavailable ledger, which a tool hands back as data, so the turn
        would otherwise run on without the lease."""
        key = current_conversation_key.get()
        try:
            async with self._store.fenced(*self._segment()) as held:
                yield held
        except SegmentLost as exc:
            conversation = self._store.current(*key) if key is not None else None
            if conversation is not None:
                _end_stream(conversation, stream_loss(exc))
            raise

    async def _column(self, column: str, run_id: str) -> Any:
        key = current_conversation_key.get()
        if key is None:
            return None
        async with self._store.engine.connect() as conn:
            value = (
                await conn.execute(
                    text(f"SELECT r.{column} {_SCOPED_RUNS} AND r.run_id = :run_id"),
                    {"principal": key[0], "conversation_id": key[1], "run_id": run_id},
                )
            ).scalar_one_or_none()
        return value

    async def admit(self, candidate: RunRecord, *, submission: JsonValue) -> Admission:
        async with self._fenced() as (conn, pk):
            # Insert the key first. A concurrent admission of the same key
            # waits here for the first to commit, then finds its row taken.
            fresh = (
                await conn.execute(
                    text(
                        "INSERT INTO run_keys (conversation, idempotency_key, run_id) "
                        "VALUES (:pk, :key, :run_id) ON CONFLICT DO NOTHING RETURNING run_id"
                    ),
                    {"pk": pk, "key": candidate.idempotency_key, "run_id": candidate.run_id},
                )
            ).first()
            existing = None
            if fresh is None:
                # The key row alone is locked, then its run read: a lock
                # taken through a join re-checks the join against the run
                # row it first read, so an admission that waited while
                # another moved the key to a new run would find no row.
                held_id = (
                    await conn.execute(
                        text(
                            "SELECT run_id FROM run_keys "
                            "WHERE conversation = :pk AND idempotency_key = :key FOR UPDATE"
                        ),
                        {"pk": pk, "key": candidate.idempotency_key},
                    )
                ).scalar_one()
                raw = (
                    await conn.execute(
                        text(
                            "SELECT record FROM runs "
                            "WHERE conversation = :pk AND run_id = :run_id FOR UPDATE"
                        ),
                        {"pk": pk, "run_id": held_id},
                    )
                ).scalar_one()
                # A failed run binds nothing (`admit_against`), readable or not,
                # and one this release cannot read is recorded failed here.
                if isinstance(raw, dict) and _status_of(raw) != RunStatus.FAILED:
                    try:
                        existing = stored_run(raw)
                    except ValidationError as exc:
                        logger.error("run ledger: stored run %s is unreadable: %s", held_id, exc)
                        reason = f"the stored run is unreadable: {exc}"
                        await _put_record(conn, pk, held_id, _unreadable(raw, reason))
            admission = admit_against(existing, candidate, now=self._now())
            if admission.resumed:
                return admission
            await conn.execute(
                text(
                    "INSERT INTO runs (conversation, run_id, record, submission) "
                    "VALUES (:pk, :run_id, CAST(:record AS jsonb), CAST(:submission AS jsonb))"
                ),
                {
                    "pk": pk,
                    "run_id": candidate.run_id,
                    "record": json.dumps(candidate.model_dump(mode="json")),
                    "submission": json.dumps(submission),
                },
            )
            if fresh is None:
                await conn.execute(
                    text(
                        "UPDATE run_keys SET run_id = :run_id "
                        "WHERE conversation = :pk AND idempotency_key = :key"
                    ),
                    {"pk": pk, "key": candidate.idempotency_key, "run_id": candidate.run_id},
                )
            return admission

    async def get(self, run_id: str) -> RunRecord | None:
        raw = await self._column("record", run_id)
        return stored_run(raw) if raw is not None else None

    async def submission(self, run_id: str) -> JsonValue:
        return await self._column("submission", run_id)

    async def retrievals(self, run_id: str) -> JsonValue:
        return await self._column("retrievals", run_id)

    async def store_plan(self, revision: str, plan: JsonValue) -> None:
        async with self._fenced() as (conn, pk):
            await conn.execute(
                text(
                    "INSERT INTO run_plans (conversation, revision, plan) "
                    "VALUES (:pk, :revision, CAST(:plan AS jsonb)) ON CONFLICT DO NOTHING"
                ),
                {"pk": pk, "revision": revision, "plan": json.dumps(plan)},
            )

    async def plan(self, revision: str) -> JsonValue:
        key = current_conversation_key.get()
        if key is None:
            return None
        async with self._store.engine.connect() as conn:
            value = (
                await conn.execute(
                    text(
                        "SELECT p.plan FROM run_plans p "
                        "JOIN conversations c ON c.id = p.conversation "
                        "WHERE c.principal = :principal AND c.conversation_id = :conversation_id "
                        "AND p.revision = :revision"
                    ),
                    {"principal": key[0], "conversation_id": key[1], "revision": revision},
                )
            ).scalar_one_or_none()
        return value

    async def unfinished(self) -> tuple[RunRecord, ...]:
        key = current_conversation_key.get()
        if key is None:
            return ()
        async with self._store.engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(f"SELECT r.run_id, r.record {_SCOPED_RUNS} ORDER BY r.run_id"),
                    {"principal": key[0], "conversation_id": key[1]},
                )
            ).all()
        runs: list[RunRecord] = []
        unreadable: dict[str, str] = {}
        for run_id, record in rows:
            raw = record
            # `carried_on`, read off the stored record before validating it.
            status = _status_of(raw)
            if status == RunStatus.FAILED:
                continue
            if status == RunStatus.COMPLETED and raw.get("delivered") is True:
                continue
            try:
                runs.append(stored_run(raw))
            except ValidationError as exc:
                logger.error("run ledger: stored run %s is unreadable: %s", run_id, exc)
                unreadable[run_id] = f"the stored run is unreadable: {exc}"
        if unreadable:
            await self._fail_unreadable(unreadable)
        return tuple(runs)

    async def _fail_unreadable(self, reasons: dict[str, str]) -> None:
        """Record each stored run in `reasons` failed, with its reason."""
        async with self._fenced() as (conn, pk):
            for run_id, reason in reasons.items():
                raw = (
                    await conn.execute(
                        text(
                            "SELECT record FROM runs WHERE conversation = :pk "
                            "AND run_id = :run_id FOR UPDATE"
                        ),
                        {"pk": pk, "run_id": run_id},
                    )
                ).scalar_one()
                if isinstance(raw, dict) and _status_of(raw) not in _TERMINAL:
                    await _put_record(conn, pk, run_id, _unreadable(raw, reason))

    async def record(self, run_id: str, change: RunChange) -> RunRecord:
        async with self._fenced() as (conn, pk):
            raw = (
                await conn.execute(
                    text(
                        "SELECT record FROM runs WHERE conversation = :pk "
                        "AND run_id = :run_id FOR UPDATE"
                    ),
                    {"pk": pk, "run_id": run_id},
                )
            ).scalar_one_or_none()
            if raw is None:
                raise RunNotFoundError(run_id)
            run = stored_run(raw)
            advanced = advance(run, change, now=self._now())
            if advanced == run:
                return advanced
            await _put_record(conn, pk, run_id, advanced.model_dump(mode="json"))
            if isinstance(change, Seal) and advanced.manifest != run.manifest:
                await conn.execute(
                    text(
                        "UPDATE runs SET retrievals = CAST(:retrievals AS jsonb) "
                        "WHERE conversation = :pk AND run_id = :run_id"
                    ),
                    {"pk": pk, "run_id": run_id, "retrievals": json.dumps(list(change.retrievals))},
                )
            return advanced
