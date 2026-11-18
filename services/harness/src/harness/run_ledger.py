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
second waits for the first to commit and then decides against its run. A
record change locks its run row. Nothing retries.

A plan's approval admits its run in the save that records the approval
(`admit_with_approval`, which `harness.studio`'s interrupt route runs in
the save): a savepoint of the transaction that holds the conversation's row
`FOR UPDATE` under the lease the route claimed, so an approval on record has
its run whenever the client leaves. An admission refused or faulted there
leaves the approval without a run, and `execute` admits it.

A Stop ends the runs of the turn it closes: `end_unfinished` records them
failed on the connection of the transaction that closes the turn
(`harness.turn.store.TurnStore.finish`).

A stored run is read with `executor.ledger.stored_run`, so one another
release wrote loads. A stored run this release cannot read is recorded
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
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from executor.classify import ExecutorCode
from executor.ledger import (
    TERMINAL_STATUSES,
    UNFINISHED_STATUSES,
    Admission,
    Fail,
    LedgerUnavailableError,
    RunChange,
    RunLedgerError,
    RunNotFoundError,
    RunRecord,
    RunStatus,
    Seal,
    admit_against,
    advance,
    store_faults,
    stored_run,
)
from pydantic import JsonValue, ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from harness.conversation import current_conversation_key
from harness.studio_state import SegmentLost, renew_lease_now
from harness.tool_contracts import HarnessCode

if TYPE_CHECKING:
    from harness.studio_store import StudioStore

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


async def end_unfinished(conn: AsyncConnection, conversation: int, *, now: datetime) -> None:
    """Record failed `cancelled`, on `conn` and in its transaction, every run
    of the conversation row `conversation` whose computation is unfinished
    (`admitted` or `running`), each run row locked. A run whose computation
    finished and whose publication is pending is left for the next turn. A
    stored run this release cannot read is recorded failed as unreadable.
    Raises what the database raises, which leaves the caller's transaction
    to roll back."""
    rows = (
        await conn.execute(
            text(
                "SELECT run_id, record FROM runs WHERE conversation = :pk "
                "AND record ->> 'status' = ANY(:statuses) ORDER BY run_id FOR UPDATE"
            ),
            {
                "pk": conversation,
                "statuses": sorted(status.value for status in UNFINISHED_STATUSES),
            },
        )
    ).all()
    for run_id, raw in rows:
        try:
            run = stored_run(raw)
        except ValidationError as exc:
            logger.error("run ledger: stored run %s is unreadable: %s", run_id, exc)
            await _put_record(
                conn, conversation, run_id, _unreadable(raw, f"the stored run is unreadable: {exc}")
            )
            continue
        ended = advance(
            run, Fail(ExecutorCode.CANCELLED, "the turn running it was stopped"), now=now
        )
        await _put_record(conn, conversation, run_id, ended.model_dump(mode="json"))


async def admit_on(
    conn: AsyncConnection,
    pk: int,
    candidate: RunRecord,
    *,
    submission: JsonValue,
    now: datetime,
) -> Admission:
    """Admit `candidate` to the conversation row `pk`, on `conn` and in its
    transaction, with `submission` stored beside a new run and `now` the
    time its authorisation is checked against: the run its key already
    names, or `candidate` (`admit_against`). `StudioRunLedger.admit` runs it
    fenced on the live segment, and `StudioRunLedger.admit_with_approval` in
    the save that records a plan's approval.

    Raises what `admit_against` raises, and what the database raises; either
    leaves the caller's transaction to roll back."""
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
    admission = admit_against(existing, candidate, now=now)
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


async def store_plan_on(conn: AsyncConnection, pk: int, revision: str, plan: JsonValue) -> None:
    """Store `plan` under `revision` for the conversation row `pk`, on `conn`
    and in its transaction, once: a revision already stored keeps its plan."""
    await conn.execute(
        text(
            "INSERT INTO run_plans (conversation, revision, plan) "
            "VALUES (:pk, :revision, CAST(:plan AS jsonb)) ON CONFLICT DO NOTHING"
        ),
        {"pk": pk, "revision": revision, "plan": json.dumps(plan)},
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
        """`StudioStore.fenced` on the current segment. A `SegmentLost` has
        the conversation's stream renew its lease at once (`renew_lease_now`),
        which ends the stream, before it propagates. `executor.ledger.store_faults`
        reports it to the graph as an unavailable ledger, which a tool hands
        back as data, so the turn would otherwise run on without the lease."""
        key = current_conversation_key.get()
        try:
            async with self._store.fenced(*self._segment()) as held:
                yield held
        except SegmentLost:
            conversation = self._store.current(*key) if key is not None else None
            if conversation is not None:
                renew_lease_now(conversation)
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
            return await admit_on(conn, pk, candidate, submission=submission, now=self._now())

    async def admit_with_approval(
        self,
        conn: AsyncConnection,
        pk: int,
        *,
        candidate: Callable[[datetime], RunRecord],
        submission: JsonValue,
        revision: str,
        plan: JsonValue,
    ) -> Admission | None:
        """Store `plan` under `revision` and admit `candidate`, minted at this
        ledger's clock, with `submission` beside it (`store_plan_on`,
        `admit_on`), for the conversation row `pk`, on `conn`, the connection
        of the `StudioStore.save` that records the plan's approval, in a
        savepoint of its transaction. Returns the `Admission`. A refusal
        (`RunLedgerError`: a key naming other work, an approval already
        expired) or a store fault in it (`store_faults`, a `run_keys` row
        with no `runs` row among them) rolls the savepoint back, is logged,
        and returns None, and the save commits the approval without a run
        for `execute` to admit. A fault the savepoint cannot roll back raises
        and fails the save."""
        now = self._now()
        try:
            async with conn.begin_nested(), store_faults(None):
                await store_plan_on(conn, pk, revision, plan)
                return await admit_on(conn, pk, candidate(now), submission=submission, now=now)
        except (RunLedgerError, LedgerUnavailableError) as exc:
            logger.warning(
                "run ledger: the approval of revision %s admits no run: %s", revision, exc
            )
            return None

    async def get(self, run_id: str) -> RunRecord | None:
        raw = await self._column("record", run_id)
        return stored_run(raw) if raw is not None else None

    async def submission(self, run_id: str) -> JsonValue:
        return await self._column("submission", run_id)

    async def retrievals(self, run_id: str) -> JsonValue:
        return await self._column("retrievals", run_id)

    async def store_plan(self, revision: str, plan: JsonValue) -> None:
        async with self._fenced() as (conn, pk):
            await store_plan_on(conn, pk, revision, plan)

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
