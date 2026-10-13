"""`StudioRunLedger`: runs kept in the conversation's ledger session.

Durability and the run items' isolation from `ConversationState` run over
both session backends. Fencing and the stale-retry admission run over the
database backend, where two `StudioStore`s sharing one session service can
race (`InMemorySessionService.append_event` has no compare-and-swap check).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
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
from google.adk.events import Event, EventActions
from google.adk.sessions import InMemorySessionService
from harness import studio_state
from harness.conversation import current_conversation_key
from harness.run_ledger import StudioRunLedger
from harness.studio_state import SegmentLost, StudioStore

from .sqlite_session import sqlite_session_service

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


@pytest.fixture(params=["memory", "database"])
def session_service(request, tmp_path) -> Any:
    if request.param == "memory":
        return InMemorySessionService()
    return sqlite_session_service(tmp_path / "studio.db")


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


async def _streaming_store(service: Any, clock: list[float]) -> StudioStore:
    """A store holding `CONVERSATION` with a claimed, saved segment, as a
    streaming turn does."""
    store = StudioStore(service, app_name="app", clock=lambda: clock[0])
    if await store.load(PRINCIPAL, CONVERSATION) is None:
        await store.create(PRINCIPAL, CONVERSATION)
    store.claim_segment(PRINCIPAL, CONVERSATION)
    await store.save(PRINCIPAL, CONVERSATION)
    return store


async def test_a_run_and_its_key_survive_into_a_fresh_store_over_the_same_service(
    session_service,
) -> None:
    store = await _streaming_store(session_service, [START])
    ledger = StudioRunLedger(store)
    run = (await ledger.admit(_candidate(), submission=SUBMISSION)).run
    await ledger.record(run.run_id, Start())
    await ledger.record(run.run_id, Attempt(step_id="s1", tool="get_data"))
    await ledger.record(run.run_id, Seal(INPUTS, retrievals=RETRIEVALS))
    await ledger.store_plan("rev-1", "the plan")
    await ledger.store_plan("rev-1", "another plan")
    await store.release(PRINCIPAL, CONVERSATION, store.current(PRINCIPAL, CONVERSATION).segment.id)

    fresh = StudioRunLedger(await _streaming_store(session_service, [START]))

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
    session_service,
) -> None:
    ledger = StudioRunLedger(await _streaming_store(session_service, [START]))
    run = (await ledger.admit(_candidate(), submission=SUBMISSION)).run
    for change in (Start(), Seal(INPUTS), Publishing((ROOT,)), Complete()):
        run = await ledger.record(run.run_id, change)
    assert await ledger.unfinished() == (run,)

    await ledger.record(run.run_id, Delivered())

    assert await ledger.unfinished() == ()


async def test_run_items_stay_out_of_conversation_state_and_survive_its_saves(
    session_service, caplog
) -> None:
    store = await _streaming_store(session_service, [START])
    ledger = StudioRunLedger(store)
    run = (await ledger.admit(_candidate(), submission=SUBMISSION)).run
    state = store.current(PRINCIPAL, CONVERSATION)
    segment_id = state.segment.id

    with caplog.at_level(logging.WARNING, logger=studio_state.__name__):
        state.turn_final_text = "answered"
        await store.save(PRINCIPAL, CONVERSATION)
        await store.checkpoint(PRINCIPAL, CONVERSATION, segment_id)
        await StudioStore(session_service, app_name="app").load(PRINCIPAL, CONVERSATION)

    assert not [r for r in caplog.records if "unknown state key" in r.getMessage()]
    assert await ledger.get(run.run_id) == run


async def test_a_writer_fenced_out_by_a_takeover_cannot_record_publication(tmp_path) -> None:
    service = sqlite_session_service(tmp_path / "studio.db")
    clock = [START]
    store_a = await _streaming_store(service, clock)
    ledger_a = StudioRunLedger(store_a)
    run = (await ledger_a.admit(_candidate(), submission=SUBMISSION)).run
    for change in (Start(), Seal(INPUTS), Publishing((DERIVED,))):
        await ledger_a.record(run.run_id, change)

    clock[0] += studio_state.SEGMENT_TTL_S + 1  # A's segment has expired
    await _streaming_store(service, clock)  # B takes the conversation over

    with pytest.raises(SegmentLost):
        await ledger_a.record(run.run_id, Published(DERIVED, "rpk_1"))

    recorded = await StudioRunLedger(StudioStore(service, app_name="app")).get(run.run_id)
    assert recorded.status is RunStatus.PUBLISHING
    assert recorded.results[0].replay_key is None


async def test_concurrent_admissions_of_one_key_settle_on_one_run(session_service) -> None:
    ledger = StudioRunLedger(await _streaming_store(session_service, [START]))

    first, second = await asyncio.gather(
        ledger.admit(_candidate(), submission=SUBMISSION),
        ledger.admit(_candidate(), submission=SUBMISSION),
    )

    assert first.run.run_id == second.run.run_id
    assert sorted([first.resumed, second.resumed]) == [False, True]


async def test_an_admission_gone_stale_decides_again_against_the_other_writers_run(
    tmp_path,
) -> None:
    """Another instance admits the same key between this admission's fetch
    and its append; the retry finds that run and resumes it rather than
    writing a second run under the key."""
    service = sqlite_session_service(tmp_path / "studio.db")
    store = await _streaming_store(service, [START])
    ledger = StudioRunLedger(store)
    other = _candidate()
    real_append_event = service.append_event
    appends = 0

    async def append_event_after_another_admission(session: Any, event: Any) -> Any:
        nonlocal appends
        appends += 1
        if appends == 1:
            fresh = await service.get_session(
                app_name="app.ledger", user_id=PRINCIPAL, session_id=CONVERSATION
            )
            await real_append_event(fresh, _admission_event(other))
        return await real_append_event(session, event)

    service.append_event = append_event_after_another_admission

    admission = await ledger.admit(_candidate(), submission=SUBMISSION)

    assert admission.resumed
    assert admission.run.run_id == other.run_id
    service.append_event = real_append_event
    assert [run.run_id for run in await ledger.unfinished()] == [other.run_id]


def _admission_event(run: RunRecord) -> Event:
    return Event(
        author="studio",
        actions=EventActions(
            state_delta={
                f"{studio_state.RUN_ITEM_PREFIX}{run.run_id}": run.model_dump(mode="json"),
                f"{studio_state.RUN_KEY_ITEM_PREFIX}{run.idempotency_key}": run.run_id,
            }
        ),
    )


async def test_outside_a_conversation_reads_find_nothing_and_writes_are_refused() -> None:
    ledger = StudioRunLedger(StudioStore(InMemorySessionService(), app_name="app"))
    token = current_conversation_key.set(None)
    try:
        assert await ledger.unfinished() == ()
        with pytest.raises(SegmentLost):
            await ledger.admit(_candidate(), submission=SUBMISSION)
    finally:
        current_conversation_key.reset(token)


async def test_an_unreadable_stored_run_is_recorded_failed_and_left_out(session_service) -> None:
    store = await _streaming_store(session_service, [START])
    ledger = StudioRunLedger(store)
    readable = (await ledger.admit(_candidate(), submission=SUBMISSION)).run
    item = f"{studio_state.RUN_ITEM_PREFIX}run_old"
    stored = {"run_id": "run_old", "status": "running", "shape": "of an older release"}
    await store.append_run_items(
        PRINCIPAL,
        CONVERSATION,
        store.current(PRINCIPAL, CONVERSATION).segment.id,
        lambda items: {item: stored},
    )

    assert await ledger.unfinished() == (readable,)

    failed = (await store.run_items(PRINCIPAL, CONVERSATION))[item]
    assert failed["status"] == "failed"
    assert failed["failure"]["code"] == "run_record_unreadable"
    assert await ledger.unfinished() == (readable,)


async def _store_run_item(store: StudioStore, run_id: str, raw: dict[str, Any]) -> None:
    await store.append_run_items(
        PRINCIPAL,
        CONVERSATION,
        store.current(PRINCIPAL, CONVERSATION).segment.id,
        lambda items: {f"{studio_state.RUN_ITEM_PREFIX}{run_id}": raw},
    )


async def test_a_run_another_release_stored_with_more_keys_is_read_and_carried_on(
    session_service,
) -> None:
    store = await _streaming_store(session_service, [START])
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


async def test_a_key_whose_stored_run_is_unreadable_admits_a_new_run(session_service) -> None:
    store = await _streaming_store(session_service, [START])
    ledger = StudioRunLedger(store)
    old = (await ledger.admit(_candidate(), submission=SUBMISSION)).run
    await _store_run_item(store, old.run_id, {"run_id": old.run_id, "status": "running"})

    admitted = await ledger.admit(_candidate(), submission=SUBMISSION)

    assert not admitted.resumed
    assert admitted.run.run_id != old.run_id
    failed = (await store.run_items(PRINCIPAL, CONVERSATION))[
        f"{studio_state.RUN_ITEM_PREFIX}{old.run_id}"
    ]
    assert (failed["status"], failed["failure"]["code"]) == ("failed", "run_record_unreadable")
    assert (await ledger.admit(_candidate(), submission=SUBMISSION)).run == admitted.run
