"""Writers that touch different parts of a conversation commit side by side.

Each instance is a `StudioStore` over an engine of its own (its own pool of
two connections, as a deployed instance has), all on one Postgres schema
(the `pg_engine` fixture's). A heartbeat renewal, a checkpoint, a canvas
edit, a cancel and run-ledger writes on one conversation all commit, and
none retries or is refused. Two claims on one turn state still conflict.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from executor.ledger import PlanApproval, SourcePolicy, Start, idempotency_key, new_run
from harness.conversation import current_conversation_key
from harness.run_ledger import StudioRunLedger
from harness.studio_state import _next_manifest, _PendingInterrupt
from harness.studio_store import ConversationStale, Renewal, StudioStore
from harness.wire import ComponentType, CompositionActor, CompositionBlock, CompositionManifest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .store_clock import StoreClock
from .test_run_ledger import _another_engine

PRINCIPAL = "principal"
CONVERSATION = "conv_01conflict0"
ROUNDS = 40


@pytest.fixture
async def engines(pg_engine: AsyncEngine) -> AsyncIterator[list[AsyncEngine]]:
    """Three instances' engines on one database schema."""
    others = [await _another_engine(pg_engine) for _ in range(2)]
    try:
        yield [pg_engine, *others]
    finally:
        for engine in others:
            await engine.dispose()


def _store(engine: AsyncEngine, *, clock: StoreClock | None = None) -> StudioStore:
    return StudioStore(
        engine,
        **(clock.store_kwargs if clock is not None else {}),
    )


def _block(component_id: str) -> CompositionBlock:
    return CompositionBlock(
        component_id=component_id, revision=1, type=ComponentType.LINE_CHART, title=component_id
    )


def _adding(component_id: str):
    def mutate(manifest: CompositionManifest | None, _components: object) -> CompositionManifest:
        blocks = [*(manifest.blocks if manifest else []), _block(component_id)]
        return _next_manifest(manifest, blocks, actor=CompositionActor.USER)

    return mutate


def _candidate(task: str):
    at = datetime.now(UTC)
    return new_run(
        idempotency_key=idempotency_key(caller=PRINCIPAL, conversation_id=CONVERSATION, task=task),
        submission_digest="d" * 64,
        authorisation=PlanApproval(
            source_policy=SourcePolicy(sources=("spec_" + "a" * 32,)),
            expires_at=at + timedelta(hours=1),
            plan_version=1,
            approved_revision="r" * 64,
            approver=PRINCIPAL,
            at=at,
        ),
        now=at,
    )


async def test_heartbeat_checkpoint_canvas_cancel_and_runs_commit_side_by_side(
    engines: list[AsyncEngine],
) -> None:
    """Instance A streams: it renews its lease, checkpoints and writes runs.
    Instance B edits the canvas and instance C presses Stop, over and over,
    at the same time. Every write lands; nothing raises."""
    owner, editor, stopper = (_store(engine) for engine in engines)
    state = await owner.create(PRINCIPAL, CONVERSATION)
    segment_id = owner.claim_segment(PRINCIPAL, CONVERSATION)
    state.begin_turn()
    await owner.save(PRINCIPAL, CONVERSATION)
    await editor.load(PRINCIPAL, CONVERSATION)
    ledger = StudioRunLedger(owner)

    async def heartbeat() -> list[Renewal]:
        return [await owner.renew(PRINCIPAL, CONVERSATION, segment_id) for _ in range(ROUNDS)]

    async def stream() -> None:
        token = current_conversation_key.set((PRINCIPAL, CONVERSATION))
        try:
            for index in range(ROUNDS):
                state.pending = {"i1": _PendingInterrupt(adk_id=f"answer {index}", kind="plan")}
                await owner.checkpoint(
                    PRINCIPAL, CONVERSATION, segment_id, owner.snapshot(PRINCIPAL, CONVERSATION)
                )
                run = await ledger.admit(_candidate(f"t{index}"), submission={})
                await ledger.record(run.run.run_id, Start())
        finally:
            current_conversation_key.reset(token)

    async def edits() -> None:
        for index in range(ROUNDS):
            await editor.mutate_composition(PRINCIPAL, CONVERSATION, _adding(f"cmp_{index:03d}"))

    async def stops() -> list[bool]:
        return [await stopper.request_cancel(PRINCIPAL, CONVERSATION) for _ in range(ROUNDS)]

    renewals, _, _, cancels = await asyncio.wait_for(
        asyncio.gather(heartbeat(), stream(), edits(), stops()), timeout=60
    )

    assert all(renewal.held for renewal in renewals)
    assert any(renewal.stop_id is not None for renewal in renewals)
    assert all(cancels)
    reloaded = await _store(engines[0]).load(PRINCIPAL, CONVERSATION)
    assert reloaded is not None
    assert [block.component_id for block in reloaded.composition.blocks] == [
        f"cmp_{index:03d}" for index in range(ROUNDS)
    ]
    assert reloaded.pending["i1"].adk_id == f"answer {ROUNDS - 1}"
    assert reloaded.segment is not None and reloaded.segment.id == segment_id
    async with engines[0].connect() as conn:
        stop_id = (await conn.execute(text("SELECT stop_id FROM conversations"))).scalar_one()
    assert stop_id is not None
    token = current_conversation_key.set((PRINCIPAL, CONVERSATION))
    try:
        assert len(await ledger.unfinished()) == ROUNDS
    finally:
        current_conversation_key.reset(token)


async def test_a_turn_claim_after_another_instances_canvas_edit_is_not_refused(
    engines: list[AsyncEngine],
) -> None:
    """The case behind the `/turn` 409s. Instance A loads a conversation, B
    edits its canvas, then A claims a turn and saves. The claim lands, and
    B's edit is kept beside A's own card."""
    first, other, _ = (_store(engine) for engine in engines)
    state = await first.create(PRINCIPAL, CONVERSATION)
    await first.save(PRINCIPAL, CONVERSATION)
    await other.load(PRINCIPAL, CONVERSATION)
    await other.mutate_composition(PRINCIPAL, CONVERSATION, _adding("cmp_user"))

    first.claim_segment(PRINCIPAL, CONVERSATION)
    state.begin_turn()
    await first.save(PRINCIPAL, CONVERSATION)

    reloaded = await _store(engines[2]).load(PRINCIPAL, CONVERSATION)
    assert reloaded is not None and reloaded.turn_index == 1
    assert [block.component_id for block in reloaded.composition.blocks] == ["cmp_user"]


async def test_two_claims_on_one_turn_state_still_conflict(engines: list[AsyncEngine]) -> None:
    """Two instances load the same conversation and both claim a turn. The
    second save is refused, since its turn state is stale."""
    first, second, _ = (_store(engine) for engine in engines)
    await first.create(PRINCIPAL, CONVERSATION)
    await first.save(PRINCIPAL, CONVERSATION)
    await second.load(PRINCIPAL, CONVERSATION)

    first.claim_segment(PRINCIPAL, CONVERSATION)
    first.current(PRINCIPAL, CONVERSATION).begin_turn()
    second.claim_segment(PRINCIPAL, CONVERSATION)
    second.current(PRINCIPAL, CONVERSATION).begin_turn()
    await first.save(PRINCIPAL, CONVERSATION)

    with pytest.raises(ConversationStale):
        await second.save(PRINCIPAL, CONVERSATION)


async def test_a_claim_racing_a_live_owners_renewal_is_refused(
    engines: list[AsyncEngine],
) -> None:
    """B loaded the conversation after A's lease had expired and claimed it in
    memory, but A's heartbeat renewed the lease before B's claim committed.
    B's claim is refused, so two instances never both hold the lease."""
    clock = await StoreClock.start(engines[0])
    owner = _store(engines[0], clock=clock)
    taker = _store(engines[1], clock=clock)
    await owner.create(PRINCIPAL, CONVERSATION)
    segment_id = owner.claim_segment(PRINCIPAL, CONVERSATION)
    await owner.save(PRINCIPAL, CONVERSATION)

    await clock.advance(50)  # past the 45 s TTL
    await taker.load(PRINCIPAL, CONVERSATION)
    taker.claim_segment(PRINCIPAL, CONVERSATION)
    assert (await owner.renew(PRINCIPAL, CONVERSATION, segment_id)).held

    with pytest.raises(ConversationStale):
        await taker.save(PRINCIPAL, CONVERSATION)
