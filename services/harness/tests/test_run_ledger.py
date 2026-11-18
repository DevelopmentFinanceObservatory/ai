"""`StudioRunLedger`: runs kept in the harness's own tables, on Postgres
(the `pg_engine` fixture), each test in a schema of its own."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from executor.ledger import (
    Attempt,
    Complete,
    Delivered,
    ManifestEntry,
    PlanApproval,
    Published,
    Publishing,
    ResultFrame,
    RunRecord,
    RunStatus,
    Seal,
    SourcePolicy,
    Start,
    idempotency_key,
    new_run,
)
from harness import studio_state, studio_store
from harness.conversation import current_conversation_key
from harness.run_ledger import StudioRunLedger
from harness.studio_state import SegmentLost
from harness.studio_store import DATABASE_NOW, StudioStore
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .store_clock import StoreClock

PRINCIPAL = "principal"
CONVERSATION = "conv_01run00000"
GDP = "spec_" + "a" * 32
DERIVED = ResultFrame(spec_id="ops_" + "d" * 32, content_version="ocv_" + "2" * 32)
ROOT = ResultFrame(spec_id=GDP, content_version="cv_" + "1" * 32)
INPUTS = (ManifestEntry(spec_id=GDP, content_version="cv_" + "1" * 32),)
# The `get_data` result the inputs were sealed from.
RETRIEVALS = ({"frame_ref": {"spec_id": GDP, "content_version": "cv_" + "1" * 32}},)
START = 1_000_000.0
# What the admitted runs execute, stored beside each.
SUBMISSION = {"workflow": {"steps": ["s1"]}}


@pytest.fixture(autouse=True)
def conversation() -> Iterator[None]:
    token = current_conversation_key.set((PRINCIPAL, CONVERSATION))
    yield
    current_conversation_key.reset(token)


def _store(
    engine: AsyncEngine,
    *,
    clock: Callable[[], float] = time.time,
    time_sql: str = DATABASE_NOW,
) -> StudioStore:
    return StudioStore(
        engine,
        clock=clock,
        time_sql=time_sql,
    )


async def _another_engine(engine: AsyncEngine) -> AsyncEngine:
    """A second engine on `engine`'s database and schema, standing for
    another instance's connection pool."""
    async with engine.connect() as conn:
        schema = (await conn.execute(text("SELECT current_schema()"))).scalar_one()
    return create_async_engine(
        engine.url,
        pool_size=2,
        max_overflow=0,
        connect_args={"server_settings": {"search_path": schema}},
    )


def _candidate() -> RunRecord:
    policy = SourcePolicy(sources=(GDP,))
    at = datetime.fromtimestamp(START, UTC)
    return new_run(
        idempotency_key=idempotency_key(caller=PRINCIPAL, conversation_id=CONVERSATION, task="t"),
        submission_digest="d" * 64,
        authorisation=PlanApproval(
            source_policy=policy,
            expires_at=at + timedelta(hours=1),
            plan_version=1,
            approved_revision="r" * 64,
            approver=PRINCIPAL,
            at=at,
        ),
        now=at,
    )


async def _streaming_store(service: Any, clock: list[float] | StoreClock) -> StudioStore:
    """A store holding `CONVERSATION` with a claimed, saved segment, as a
    streaming turn does. A `StoreClock` moves its leases' time too. A list
    holds its instance clock's reading alone."""
    if isinstance(clock, StoreClock):
        store = _store(service, **clock.store_kwargs)
    else:
        store = _store(service, clock=lambda: clock[0])
    if await store.load(PRINCIPAL, CONVERSATION) is None:
        await store.create(PRINCIPAL, CONVERSATION)
    store.claim_segment(PRINCIPAL, CONVERSATION)
    await store.save(PRINCIPAL, CONVERSATION)
    return store


async def test_a_run_and_its_key_survive_into_a_fresh_store_over_the_same_service(
    pg_engine,
) -> None:
    store = await _streaming_store(pg_engine, [START])
    ledger = StudioRunLedger(store)
    run = (await ledger.admit(_candidate(), submission=SUBMISSION)).run
    await ledger.record(run.run_id, Start())
    await ledger.record(run.run_id, Attempt(step_id="s1", tool="get_data"))
    await ledger.record(run.run_id, Seal(INPUTS, retrievals=RETRIEVALS))
    await ledger.store_plan("rev-1", "the plan")
    await ledger.store_plan("rev-1", "another plan")
    await store.release(PRINCIPAL, CONVERSATION, store.current(PRINCIPAL, CONVERSATION).segment.id)

    fresh = StudioRunLedger(await _streaming_store(pg_engine, [START]))

    reloaded = await fresh.get(run.run_id)
    assert reloaded is not None
    assert reloaded.status is RunStatus.RUNNING
    assert [a.step_id for a in reloaded.attempts] == ["s1"]
    assert await fresh.retrievals(run.run_id) == list(RETRIEVALS)
    # Written once per revision.
    assert await fresh.plan("rev-1") == "the plan"
    assert await fresh.unfinished() == (reloaded,)
    again = await fresh.admit(_candidate(), submission=SUBMISSION)
    assert again.resumed
    assert again.run == reloaded


async def test_a_completed_run_stays_unfinished_until_its_delivery_is_recorded(
    pg_engine,
) -> None:
    ledger = StudioRunLedger(await _streaming_store(pg_engine, [START]))
    run = (await ledger.admit(_candidate(), submission=SUBMISSION)).run
    for change in (Start(), Seal(INPUTS), Publishing((ROOT,)), Complete()):
        run = await ledger.record(run.run_id, change)
    assert await ledger.unfinished() == (run,)

    await ledger.record(run.run_id, Delivered())

    assert await ledger.unfinished() == ()


async def test_run_items_stay_out_of_conversation_state_and_survive_its_saves(
    pg_engine,
) -> None:
    store = await _streaming_store(pg_engine, [START])
    ledger = StudioRunLedger(store)
    run = (await ledger.admit(_candidate(), submission=SUBMISSION)).run
    state = store.current(PRINCIPAL, CONVERSATION)
    segment_id = state.segment.id

    state.turn_final_text = "answered"
    await store.save(PRINCIPAL, CONVERSATION)
    await store.checkpoint(
        PRINCIPAL, CONVERSATION, segment_id, store.snapshot(PRINCIPAL, CONVERSATION)
    )

    async with pg_engine.connect() as conn:
        stored_keys = (
            await conn.execute(text("SELECT turn_state FROM conversations"))
        ).scalar_one()
    assert set(stored_keys) <= studio_store._SCALAR_KEYS
    assert await ledger.get(run.run_id) == run


async def test_a_writer_fenced_out_by_a_takeover_cannot_record_publication(
    pg_engine,
) -> None:
    service = pg_engine
    clock = await StoreClock.start(service, START)
    store_a = await _streaming_store(service, clock)
    ledger_a = StudioRunLedger(store_a)
    run = (await ledger_a.admit(_candidate(), submission=SUBMISSION)).run
    for change in (Start(), Seal(INPUTS), Publishing((DERIVED,))):
        await ledger_a.record(run.run_id, change)

    await clock.advance(studio_state.SEGMENT_TTL_S + 1)  # A's segment has expired
    await _streaming_store(service, clock)  # B takes the conversation over
    streaming = store_a.current(PRINCIPAL, CONVERSATION)
    streaming.renew_now = asyncio.Event()

    with pytest.raises(SegmentLost):
        await ledger_a.record(run.run_id, Published(DERIVED, "rpk_1"))

    # A's heartbeat renews at once, finds the lease lost and ends the stream,
    # since the graph reports the ledger's failure to the model as data and
    # would otherwise run on.
    assert streaming.renew_now.is_set()
    recorded = await StudioRunLedger(_store(service)).get(run.run_id)
    assert recorded.status is RunStatus.PUBLISHING
    assert recorded.results[0].replay_key is None


async def test_concurrent_admissions_of_one_key_settle_on_one_run(pg_engine) -> None:
    ledger = StudioRunLedger(await _streaming_store(pg_engine, [START]))

    first, second = await asyncio.gather(
        ledger.admit(_candidate(), submission=SUBMISSION),
        ledger.admit(_candidate(), submission=SUBMISSION),
    )

    assert first.run.run_id == second.run.run_id
    assert sorted([first.resumed, second.resumed]) == [False, True]


async def test_an_admission_racing_another_decides_against_the_other_writers_run(
    pg_engine: AsyncEngine,
) -> None:
    """Another instance admits the same key and has not committed when this
    admission inserts its key. This one waits on the key row, then finds that
    run and resumes it rather than writing a second run under the key."""
    second = await _another_engine(pg_engine)
    try:
        store = await _streaming_store(pg_engine, [START])
        ledger = StudioRunLedger(store)
        other = _candidate()
        segment_id = store.current(PRINCIPAL, CONVERSATION).segment.id
        other_store = _store(second, clock=lambda: START)
        async with other_store.fenced(PRINCIPAL, CONVERSATION, segment_id) as (conn, pk):
            await conn.execute(
                text(
                    "INSERT INTO run_keys (conversation, idempotency_key, run_id) "
                    "VALUES (:pk, :key, :run_id)"
                ),
                {"pk": pk, "key": other.idempotency_key, "run_id": other.run_id},
            )
            await conn.execute(
                text(
                    "INSERT INTO runs (conversation, run_id, record) "
                    "VALUES (:pk, :run_id, CAST(:record AS jsonb))"
                ),
                {"pk": pk, "run_id": other.run_id, "record": other.model_dump_json()},
            )
            admitting = asyncio.ensure_future(ledger.admit(_candidate(), submission=SUBMISSION))
            await asyncio.sleep(0.2)
            assert not admitting.done()  # waiting on the uncommitted key row

        admission = await asyncio.wait_for(admitting, timeout=5)

        assert admission.resumed
        assert admission.run.run_id == other.run_id
        assert [run.run_id for run in await ledger.unfinished()] == [other.run_id]
    finally:
        await second.dispose()


async def test_an_admission_waiting_while_another_replaces_a_failed_run_resumes_the_new_one(
    pg_engine: AsyncEngine,
) -> None:
    """The key's run failed. Another instance admits the key again: it locks
    the key, this admission waits on it, and it moves the key to a new run
    and commits. This one then finds the new run and resumes it."""
    second = await _another_engine(pg_engine)
    try:
        store = await _streaming_store(pg_engine, [START])
        ledger = StudioRunLedger(store)
        failed = (await ledger.admit(_candidate(), submission=SUBMISSION)).run
        await _store_run_item(
            store, failed.run_id, {**failed.model_dump(mode="json"), "status": "failed"}
        )
        other = _candidate()
        segment_id = store.current(PRINCIPAL, CONVERSATION).segment.id
        other_store = _store(second, clock=lambda: START)
        async with other_store.fenced(PRINCIPAL, CONVERSATION, segment_id) as (conn, pk):
            key = {"pk": pk, "key": other.idempotency_key}
            await conn.execute(
                text(
                    "SELECT run_id FROM run_keys "
                    "WHERE conversation = :pk AND idempotency_key = :key FOR UPDATE"
                ),
                key,
            )
            admitting = asyncio.ensure_future(ledger.admit(_candidate(), submission=SUBMISSION))
            await asyncio.sleep(0.2)
            assert not admitting.done()  # waiting on the locked key row
            await conn.execute(
                text(
                    "INSERT INTO runs (conversation, run_id, record) "
                    "VALUES (:pk, :run_id, CAST(:record AS jsonb))"
                ),
                {"pk": pk, "run_id": other.run_id, "record": other.model_dump_json()},
            )
            await conn.execute(
                text(
                    "UPDATE run_keys SET run_id = :run_id "
                    "WHERE conversation = :pk AND idempotency_key = :key"
                ),
                {**key, "run_id": other.run_id},
            )

        admission = await asyncio.wait_for(admitting, timeout=5)

        assert (admission.resumed, admission.run.run_id) == (True, other.run_id)
    finally:
        await second.dispose()


async def test_outside_a_conversation_reads_find_nothing_and_writes_are_refused(
    pg_engine,
) -> None:
    ledger = StudioRunLedger(_store(pg_engine))
    token = current_conversation_key.set(None)
    try:
        assert await ledger.unfinished() == ()
        with pytest.raises(SegmentLost):
            await ledger.admit(_candidate(), submission=SUBMISSION)
    finally:
        current_conversation_key.reset(token)


async def test_an_unreadable_stored_run_is_recorded_failed_and_left_out(pg_engine) -> None:
    store = await _streaming_store(pg_engine, [START])
    ledger = StudioRunLedger(store)
    readable = (await ledger.admit(_candidate(), submission=SUBMISSION)).run
    stored = {"run_id": "run_old", "status": "running", "shape": "of an older release"}
    await _store_run_item(store, "run_old", stored)

    assert await ledger.unfinished() == (readable,)

    failed = await _stored_record(store, "run_old")
    assert failed["status"] == "failed"
    assert failed["failure"]["code"] == "run_record_unreadable"
    assert await ledger.unfinished() == (readable,)


async def _store_run_item(store: StudioStore, run_id: str, raw: dict[str, Any]) -> None:
    """Write `raw` as run `run_id`'s stored record, as another release would."""
    segment_id = store.current(PRINCIPAL, CONVERSATION).segment.id
    async with store.fenced(PRINCIPAL, CONVERSATION, segment_id) as (conn, pk):
        await conn.execute(
            text(
                "INSERT INTO runs (conversation, run_id, record) "
                "VALUES (:pk, :run_id, CAST(:record AS jsonb)) "
                "ON CONFLICT (conversation, run_id) DO UPDATE SET record = EXCLUDED.record"
            ),
            {"pk": pk, "run_id": run_id, "record": json.dumps(raw)},
        )


async def _stored_record(store: StudioStore, run_id: str) -> dict[str, Any]:
    async with store.engine.connect() as conn:
        record = (
            await conn.execute(
                text("SELECT record FROM runs WHERE run_id = :run_id"), {"run_id": run_id}
            )
        ).scalar_one()
    return json.loads(record) if isinstance(record, str) else record


async def test_a_run_another_release_stored_with_more_keys_is_read_and_carried_on(
    pg_engine,
) -> None:
    store = await _streaming_store(pg_engine, [START])
    ledger = StudioRunLedger(store)
    run = (await ledger.admit(_candidate(), submission=SUBMISSION)).run
    raw = run.model_dump(mode="json")
    raw["added_later"] = 1
    raw["authorisation"]["added_later"] = 2
    await _store_run_item(store, run.run_id, raw)

    assert await ledger.get(run.run_id) == run
    assert await ledger.unfinished() == (run,)
    assert (await ledger.admit(_candidate(), submission=SUBMISSION)).run == run
    assert (await ledger.record(run.run_id, Start())).status is RunStatus.RUNNING


async def test_a_key_whose_stored_run_is_unreadable_admits_a_new_run(pg_engine) -> None:
    store = await _streaming_store(pg_engine, [START])
    ledger = StudioRunLedger(store)
    old = (await ledger.admit(_candidate(), submission=SUBMISSION)).run
    await _store_run_item(store, old.run_id, {"run_id": old.run_id, "status": "running"})

    admitted = await ledger.admit(_candidate(), submission=SUBMISSION)

    assert not admitted.resumed
    assert admitted.run.run_id != old.run_id
    failed = await _stored_record(store, old.run_id)
    assert (failed["status"], failed["failure"]["code"]) == ("failed", "run_record_unreadable")
    assert (await ledger.admit(_candidate(), submission=SUBMISSION)).run == admitted.run
