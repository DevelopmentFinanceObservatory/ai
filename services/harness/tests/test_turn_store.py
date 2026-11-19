from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import harness.provenance as provenance_module
import pytest
from executor.ledger import PlanApproval, SourcePolicy, new_run
from harness.conversation import current_conversation_id
from harness.db.migrate import MIGRATIONS_DIR, apply_pending
from harness.provenance import ProvenanceStore
from harness.studio_state import SegmentLost
from harness.turn import store as turn_store
from harness.turn.store import (
    FRAMES_KEY,
    SPECS_KEY,
    AgentRecord,
    Cursor,
    CursorStale,
    OpenPause,
    OwedAnswer,
    PauseNotOpen,
    TurnAlreadyOpen,
    TurnStatus,
    TurnStore,
)
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

PRINCIPAL = "user-1"
CONVERSATION = "conv-1"
SEGMENT = "segment-a"


async def _conversation(engine: AsyncEngine, *, segment_id: str | None = SEGMENT) -> int:
    """Insert the conversation row, its lease held by `segment_id`."""
    async with engine.begin() as conn:
        return (
            await conn.execute(
                text(
                    "INSERT INTO conversations (principal, conversation_id, lease_segment_id, "
                    "lease_owner, lease_expires_at) VALUES (:principal, :conversation_id, "
                    ":segment_id, 'owner', CASE WHEN CAST(:segment_id AS text) IS NULL "
                    "THEN NULL ELSE extract(epoch FROM now()) + 60 END) RETURNING id"
                ),
                {"principal": PRINCIPAL, "conversation_id": CONVERSATION, "segment_id": segment_id},
            )
        ).scalar_one()


async def _move_lease(engine: AsyncEngine, segment_id: str) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE conversations SET lease_segment_id = :segment_id"),
            {"segment_id": segment_id},
        )


async def _open(store: TurnStore, *, turn_index: int = 0) -> Cursor:
    return await store.open_turn(
        PRINCIPAL,
        CONVERSATION,
        segment_id=SEGMENT,
        turn_index=turn_index,
        node="seed",
        inputs={"message": "aid to Kenya"},
        state={"plan": None},
        records=[AgentRecord(agent="loop", kind="user", body={"text": "aid to Kenya"})],
    )


async def _paused(store: TurnStore, cursor: Cursor, *, interrupt_id: str = "int_1") -> Cursor:
    return await store.pause(
        cursor,
        segment_id=SEGMENT,
        node="ask",
        node_input={"question": "which donor?"},
        kind="ask",
        interrupt_id=interrupt_id,
        payload={"question": "which donor?"},
        state=cursor.state,
    )


async def test_the_turn_tables_apply_on_top_of_the_conversation_tables(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    for name in ("0001_conversations.sql", "0002_adk_ledger_imports.sql"):
        shutil.copy(MIGRATIONS_DIR / name, tmp_path / name)
    await apply_pending(pg_empty_engine, directory=tmp_path)

    for name in ("0003_turns.sql", "0004_card_lineage.sql"):
        shutil.copy(MIGRATIONS_DIR / name, tmp_path / name)

    assert await apply_pending(pg_empty_engine, directory=tmp_path) == [
        "0003_turns.sql",
        "0004_card_lineage.sql",
    ]
    await _conversation(pg_empty_engine)
    cursor = await _open(TurnStore(pg_empty_engine))
    assert cursor.status is TurnStatus.RUNNING
    async with pg_empty_engine.connect() as conn:
        tables = await conn.execute(
            text(
                "SELECT to_regclass('conversation_card_lineage') IS NOT NULL, "
                "to_regclass('turn_frame_lineage') IS NOT NULL"
            )
        )
        assert tables.one() == (True, True)


async def test_a_turn_runs_pauses_resumes_and_closes_with_its_transcript_in_order(
    pg_engine: AsyncEngine,
) -> None:
    await _conversation(pg_engine)
    store = TurnStore(pg_engine)
    opened = await _open(store)

    advanced = await store.advance(
        opened,
        segment_id=SEGMENT,
        node="planner",
        node_input={"brief": "b"},
        state={"plan": {"steps": 1}, SPECS_KEY: {"spec_a": {"spec_id": "spec_a"}}},
        records=[AgentRecord(agent="planner", kind="response", body={"parts": [1]})],
    )
    paused = await store.pause(
        advanced,
        segment_id=SEGMENT,
        node="planner",
        node_input={"brief": "b"},
        kind="planner_ask",
        interrupt_id="int_1",
        payload={"question": "which donor?"},
        state=advanced.state,
        records=[AgentRecord(agent="loop", kind="pause", body={"question": "which donor?"})],
    )
    resumed = await store.record_answer(
        paused,
        segment_id=SEGMENT,
        interrupt_id="int_1",
        round=1,
        answer={"choice": "DAC"},
        records=[AgentRecord(agent="loop", kind="answer", body={"choice": "DAC"})],
    )
    closed = await store.finish(resumed, segment_id=SEGMENT, status=TurnStatus.DONE)

    assert (paused.status, paused.round, paused.pause) == (
        TurnStatus.PAUSED,
        1,
        OpenPause(
            kind="planner_ask", interrupt_id="int_1", round=1, payload={"question": "which donor?"}
        ),
    )
    assert (resumed.status, resumed.node, resumed.pause) == (TurnStatus.RUNNING, "planner", None)
    assert (closed.status, closed.error, closed.version) == (TurnStatus.DONE, None, 4)
    assert closed.state == {"plan": {"steps": 1}, SPECS_KEY: {"spec_a": {"spec_id": "spec_a"}}}
    assert await store.specs(closed) == {"spec_a": {"spec_id": "spec_a"}}
    assert [(r.agent, r.kind) for r in await store.records(closed)] == [
        ("loop", "user"),
        ("planner", "response"),
        ("loop", "pause"),
        ("loop", "answer"),
    ]
    assert await store.open_cursor(PRINCIPAL, CONVERSATION) is None


async def test_a_write_from_a_stale_cursor_is_refused_and_leaves_the_row_as_written(
    pg_engine: AsyncEngine,
) -> None:
    await _conversation(pg_engine)
    store = TurnStore(pg_engine)
    opened = await _open(store)
    await store.advance(
        opened,
        segment_id=SEGMENT,
        node="planner",
        node_input=None,
        state={"winner": True, SPECS_KEY: {"spec_a": {"by": "winner"}}},
    )

    with pytest.raises(CursorStale):
        await store.advance(
            opened,
            segment_id=SEGMENT,
            node="orchestrate",
            node_input=None,
            state={"winner": False, SPECS_KEY: {"spec_a": {"by": "loser"}, "spec_b": {}}},
            records=[AgentRecord(agent="planner", kind="response", body={})],
        )

    stored = await store.open_cursor(PRINCIPAL, CONVERSATION)
    assert stored is not None
    assert (stored.node, stored.state, stored.version, stored.record_count) == (
        "planner",
        {"winner": True, SPECS_KEY: {"spec_a": {"by": "winner"}}},
        1,
        1,
    )
    assert await store.specs(stored) == {"spec_a": {"by": "winner"}}


async def test_writes_are_refused_once_another_segment_holds_the_lease(
    pg_engine: AsyncEngine,
) -> None:
    await _conversation(pg_engine)
    store = TurnStore(pg_engine)
    opened = await _open(store)
    await _move_lease(pg_engine, "segment-b")

    with pytest.raises(SegmentLost):
        await store.advance(opened, segment_id=SEGMENT, node="planner", node_input=None, state={})
    with pytest.raises(SegmentLost):
        await store.finish(opened, segment_id=SEGMENT, status=TurnStatus.CANCELLED)

    stored = await store.open_cursor(PRINCIPAL, CONVERSATION)
    assert stored == opened


async def test_a_turn_opens_only_under_the_conversation_lease(pg_engine: AsyncEngine) -> None:
    await _conversation(pg_engine, segment_id="segment-b")

    with pytest.raises(SegmentLost):
        await _open(TurnStore(pg_engine))

    assert await TurnStore(pg_engine).open_cursor(PRINCIPAL, CONVERSATION) is None


async def test_one_open_turn_per_conversation(pg_engine: AsyncEngine) -> None:
    conversation = await _conversation(pg_engine)
    store = TurnStore(pg_engine)
    first = await _paused(store, await _open(store))

    with pytest.raises(TurnAlreadyOpen):
        await _open(store, turn_index=1)
    with pytest.raises(IntegrityError, match="turn_cursors_one_open"):
        async with pg_engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO turn_cursors (conversation, turn_index, node, status, inputs, "
                    "state) VALUES (:conversation, 1, 'seed', 'running', '{}', '{}')"
                ),
                {"conversation": conversation},
            )

    abandoned = await store.finish(
        first,
        segment_id=SEGMENT,
        status=TurnStatus.ABANDONED,
        error={"type": "abandoned", "message": "a new turn began"},
    )
    second = await _open(store, turn_index=1)

    assert (abandoned.status, abandoned.pause) == (TurnStatus.ABANDONED, None)
    assert await store.open_cursor(PRINCIPAL, CONVERSATION) == second
    with pytest.raises(PauseNotOpen):
        await store.record_answer(
            abandoned, segment_id=SEGMENT, interrupt_id="int_1", round=1, answer={}
        )


@pytest.mark.parametrize("next_write", ["advance", "pause", "finish"])
async def test_an_accepted_answer_is_owed_until_the_next_write_moves_the_cursor(
    pg_engine: AsyncEngine, next_write: str
) -> None:
    await _conversation(pg_engine)
    store = TurnStore(pg_engine)
    paused = await _paused(store, await _open(store))
    resumed = await store.record_answer(
        paused, segment_id=SEGMENT, interrupt_id="int_1", round=1, answer={"pick": "a"}
    )
    owed = OwedAnswer(interrupt_id="int_1", round=1)
    taken_over = await store.open_cursor(PRINCIPAL, CONVERSATION)

    if next_write == "advance":
        moved = await store.advance(
            resumed, segment_id=SEGMENT, node="planner", node_input=None, state={}
        )
    elif next_write == "pause":
        moved = await _paused(store, resumed, interrupt_id="int_2")
    else:
        moved = await store.finish(resumed, segment_id=SEGMENT, status=TurnStatus.DONE)

    assert (paused.owed, resumed.owed) == (None, owed)
    assert taken_over is not None and taken_over.owed == owed
    assert moved.owed is None


async def test_only_a_running_turn_can_owe_an_answer(pg_engine: AsyncEngine) -> None:
    await _conversation(pg_engine)
    store = TurnStore(pg_engine)
    await _paused(store, await _open(store))

    with pytest.raises(IntegrityError):
        async with pg_engine.begin() as conn:
            await conn.execute(text("UPDATE turn_cursors SET owed_interrupt_id = 'int_1'"))


async def test_answers_are_kept_per_turn_interrupt_and_round(pg_engine: AsyncEngine) -> None:
    await _conversation(pg_engine)
    store = TurnStore(pg_engine)
    first_pause = await _paused(store, await _open(store))
    resumed = await store.record_answer(
        first_pause, segment_id=SEGMENT, interrupt_id="int_1", round=1, answer={"pick": "a"}
    )
    second_pause = await _paused(store, resumed)
    await store.record_answer(
        second_pause, segment_id=SEGMENT, interrupt_id="int_1", round=2, answer={"pick": "b"}
    )

    assert second_pause.pause is not None and second_pause.pause.round == 2
    assert await store.answer(resumed, interrupt_id="int_1", round=1) == {"pick": "a"}
    assert await store.answer(resumed, interrupt_id="int_1", round=2) == {"pick": "b"}
    assert await store.answer(resumed, interrupt_id="int_2", round=1) is None
    # The same answer again from the cursor it was accepted on: the pause has
    # moved on, so a retried /interrupt reads the stored answer instead.
    with pytest.raises(CursorStale):
        await store.record_answer(
            first_pause, segment_id=SEGMENT, interrupt_id="int_1", round=1, answer={"pick": "a"}
        )
    with pytest.raises(PauseNotOpen):
        await store.record_answer(
            second_pause, segment_id=SEGMENT, interrupt_id="int_1", round=1, answer={"pick": "a"}
        )


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (TurnStatus.DONE, None),
        (TurnStatus.CANCELLED, None),
        (TurnStatus.FAILED, {"node": "planner", "type": "ValueError", "message": "boom"}),
        (TurnStatus.ABANDONED, {"type": "abandoned", "message": "lease lapsed"}),
    ],
)
async def test_a_turn_closes_with_an_error_exactly_when_it_failed_or_was_abandoned(
    pg_engine: AsyncEngine, status: TurnStatus, error: dict[str, str] | None
) -> None:
    await _conversation(pg_engine)
    store = TurnStore(pg_engine)
    opened = await _open(store)
    wrong = None if error is not None else {"type": "x", "message": "y"}

    with pytest.raises(ValueError):
        await store.finish(opened, segment_id=SEGMENT, status=status, error=wrong)
    # The table holds the same rule for a writer that bypasses `finish`.
    with pytest.raises(IntegrityError):
        async with pg_engine.begin() as conn:
            await conn.execute(
                text("UPDATE turn_cursors SET status = :status, error = CAST(:error AS jsonb)"),
                {"status": status.value, "error": json.dumps(wrong) if wrong else None},
            )
    closed = await store.finish(opened, segment_id=SEGMENT, status=status, error=error)

    assert (closed.status, closed.error) == (status, error)
    with pytest.raises(CursorStale):
        await store.finish(closed, segment_id=SEGMENT, status=status, error=error)


# A run record as admitted.
ADMITTED_RUN = new_run(
    idempotency_key="key",
    submission_digest="d" * 64,
    authorisation=PlanApproval(
        source_policy=SourcePolicy(sources=("spec_" + "a" * 32,)),
        expires_at=datetime(2030, 1, 1, tzinfo=UTC),
        plan_version=1,
        approved_revision="r" * 64,
        approver=PRINCIPAL,
        at=datetime(2026, 1, 1, tzinfo=UTC),
    ),
    now=datetime(2026, 1, 1, tzinfo=UTC),
).model_dump(mode="json")
# One run of each kind a close meets, by run id: computing (`admitted`,
# `running`), finished with its publication pending (`completed`), and one
# this release cannot read.
_RUNS = {
    "run_admitted": ADMITTED_RUN,
    "run_running": {**ADMITTED_RUN, "status": "running"},
    "run_completed": {**ADMITTED_RUN, "status": "completed"},
    "run_unreadable": {"status": "running", "shape": "from another release"},
}


async def _store_runs(engine: AsyncEngine, conversation: int) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO runs (conversation, run_id, record) "
                "VALUES (:pk, :run_id, CAST(:record AS jsonb))"
            ),
            [
                {"pk": conversation, "run_id": run_id, "record": json.dumps(record)}
                for run_id, record in _RUNS.items()
            ],
        )


async def _run_outcomes(engine: AsyncEngine) -> dict[str, tuple[str, str | None]]:
    """Each stored run's status and failure code, by run id."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT run_id, record ->> 'status' AS status, "
                "record -> 'failure' ->> 'code' AS code FROM runs"
            )
        )
        return {row.run_id: (row.status, row.code) for row in rows}


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (TurnStatus.CANCELLED, None),
        (TurnStatus.DONE, None),
        (TurnStatus.FAILED, {"type": "x", "message": "y"}),
        (TurnStatus.ABANDONED, {"type": "abandoned", "message": "a new turn began"}),
    ],
)
async def test_only_a_cancelled_close_ends_the_runs_left_computing_and_no_close_reads_a_stop(
    pg_engine: AsyncEngine, status: TurnStatus, error: dict[str, str] | None
) -> None:
    """A close `cancelled` records failed `cancelled` every run of the
    conversation still computing, in its own transaction, and leaves a run
    whose publication is pending for the next turn. Every other close leaves
    the runs as stored. No close reads or writes the Stop columns, so a turn
    abandoned with a Stop requested still closes `abandoned`."""
    conversation = await _conversation(pg_engine)
    await _store_runs(pg_engine, conversation)
    store = TurnStore(pg_engine)
    opened = await _open(store)
    stops = "lease_cancel = true, stop_id = 'stop_1', stop_turn = 0"
    async with pg_engine.begin() as conn:
        await conn.execute(text(f"UPDATE conversations SET {stops}"))

    closed = await store.finish(opened, segment_id=SEGMENT, status=status, error=error)

    assert closed.status is status
    ends = status is TurnStatus.CANCELLED
    assert await _run_outcomes(pg_engine) == {
        "run_admitted": ("failed", "cancelled") if ends else ("admitted", None),
        "run_running": ("failed", "cancelled") if ends else ("running", None),
        "run_completed": ("completed", None),
        "run_unreadable": ("failed", "run_record_unreadable") if ends else ("running", None),
    }
    async with pg_engine.connect() as conn:
        row = (
            await conn.execute(text("SELECT lease_cancel, stop_id, stop_turn FROM conversations"))
        ).one()
    assert tuple(row) == (True, "stop_1", 0)


async def test_a_cancelled_close_whose_runs_cannot_be_ended_leaves_the_turn_open(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A close `cancelled` and the end of its runs commit together: when the
    runs cannot be recorded, the turn stays open as stored."""
    conversation = await _conversation(pg_engine)
    await _store_runs(pg_engine, conversation)
    store = TurnStore(pg_engine)
    opened = await _open(store)

    async def unreachable(*_args: Any, **_kwargs: Any) -> None:
        raise ConnectionError("the run ledger is unreachable")

    monkeypatch.setattr(turn_store, "end_unfinished", unreachable)
    with pytest.raises(ConnectionError):
        await store.finish(opened, segment_id=SEGMENT, status=TurnStatus.CANCELLED)

    assert await store.open_cursor(PRINCIPAL, CONVERSATION) == opened
    assert (await _run_outcomes(pg_engine))["run_admitted"] == ("admitted", None)


async def _spec_rows(engine: AsyncEngine) -> dict[str, tuple[int, int]]:
    """Each stored spec block's (position, writing transaction id)."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT spec_id, seq, xmin::text::bigint AS written_by FROM turn_specs")
        )
        return {row.spec_id: (row.seq, row.written_by) for row in rows}


async def test_a_checkpoint_writes_only_the_spec_blocks_that_changed(
    pg_engine: AsyncEngine,
) -> None:
    await _conversation(pg_engine)
    store = TurnStore(pg_engine)
    first = await store.advance(
        await _open(store),
        segment_id=SEGMENT,
        node="orchestrate",
        node_input=None,
        state={SPECS_KEY: {"spec_a": {"v": 1}, "spec_b": {"v": 1}, "spec_c": {"v": 1}}},
    )
    before = await _spec_rows(pg_engine)

    # A search re-reading rewrites spec_b, a switch's restore drops spec_c,
    # and a new search adds spec_d; spec_a is carried unchanged.
    second = await store.advance(
        first,
        segment_id=SEGMENT,
        node="planner",
        node_input=None,
        state={SPECS_KEY: {"spec_a": {"v": 1}, "spec_b": {"v": 2}, "spec_d": {"v": 1}}},
    )
    after = await _spec_rows(pg_engine)

    assert after["spec_a"] == before["spec_a"]
    assert after["spec_b"][0] == before["spec_b"][0]
    assert after["spec_b"][1] != before["spec_b"][1]
    assert "spec_c" not in after
    assert await store.specs(second) == {
        "spec_a": {"v": 1},
        "spec_b": {"v": 2},
        "spec_d": {"v": 1},
    }
    # A segment that takes the turn over reads the digests the writer held,
    # so its first checkpoint diffs against what is stored.
    assert await store.open_cursor(PRINCIPAL, CONVERSATION) == second


async def test_a_spec_block_reads_back_with_its_keys_in_the_order_the_turn_built_them(
    pg_engine: AsyncEngine,
) -> None:
    await _conversation(pg_engine)
    store = TurnStore(pg_engine)
    block = {"spec_id": "spec_a", "title": "ODA", "dims": {"recipient": "KEN", "donor": "DAC"}}

    written = await store.advance(
        await _open(store),
        segment_id=SEGMENT,
        node="planner",
        node_input=None,
        state={SPECS_KEY: {"spec_a": block}},
    )
    taken_over = await store.open_cursor(PRINCIPAL, CONVERSATION)
    assert taken_over is not None
    loaded, _ = await store.load(taken_over)

    for read in (loaded.state[SPECS_KEY]["spec_a"], (await store.specs(written))["spec_a"]):
        assert list(read) == ["spec_id", "title", "dims"]
        assert list(read["dims"]) == ["recipient", "donor"]


async def test_spec_blocks_read_back_in_the_order_the_turn_last_held_them(
    pg_engine: AsyncEngine,
) -> None:
    """A switch drops spec_a and then, failing, puts the mapping back as it
    was: spec_a first again. The next segment reads that order, as it does
    after blocks move without changing and after an earlier one is dropped."""
    await _conversation(pg_engine)
    store = TurnStore(pg_engine)
    a, b, c = {"v": "a"}, {"v": "b"}, {"v": "c"}

    async def held(cursor: Cursor, specs: dict[str, Any]) -> tuple[Cursor, list[str]]:
        written = await store.advance(
            cursor, segment_id=SEGMENT, node="planner", node_input=None, state={SPECS_KEY: specs}
        )
        stored = await store.open_cursor(PRINCIPAL, CONVERSATION)
        assert stored is not None
        return written, list(stored.state[SPECS_KEY])

    cursor, _ = await held(await _open(store), {"spec_a": a, "spec_b": b})
    cursor, _ = await held(cursor, {"spec_b": b})
    cursor, restored = await held(cursor, {"spec_a": a, "spec_b": b})
    cursor, moved = await held(cursor, {"spec_b": {"v": "b"}, "spec_a": {"v": "a"}})
    cursor, appended = await held(cursor, {"spec_a": a, "spec_c": c})

    assert restored == ["spec_a", "spec_b"]
    assert moved == ["spec_b", "spec_a"]
    assert appended == ["spec_a", "spec_c"]


async def test_the_cursor_holds_spec_blocks_as_stored_and_a_block_changed_in_place_is_written(
    pg_engine: AsyncEngine,
) -> None:
    """A written cursor holds each block as its JSON decodes, the form a
    rerun reads, and a block a node then changes in place is written at the
    next checkpoint."""
    await _conversation(pg_engine)
    store = TurnStore(pg_engine)
    first = await store.advance(
        await _open(store),
        segment_id=SEGMENT,
        node="planner",
        node_input=None,
        state={SPECS_KEY: {"spec_a": {"v": 1, "pair": (1, 2), "by": {7: "x"}}}},
    )
    stored = {"spec_a": {"v": 1, "pair": [1, 2], "by": {"7": "x"}}}
    assert first.state[SPECS_KEY] == stored

    first.state[SPECS_KEY]["spec_a"]["v"] = 2
    second = await store.advance(
        first, segment_id=SEGMENT, node="approve", node_input=None, state=first.state
    )

    assert await store.specs(second) == {"spec_a": {**stored["spec_a"], "v": 2}}


async def test_a_write_decodes_only_the_spec_blocks_it_changed(pg_engine: AsyncEngine) -> None:
    """A block whose digest did not change is carried as the cursor holds it
    decoded, and a write without state keeps the cursor's blocks."""
    await _conversation(pg_engine)
    store = TurnStore(pg_engine)
    first = await store.advance(
        await _open(store),
        segment_id=SEGMENT,
        node="planner",
        node_input=None,
        state={SPECS_KEY: {"spec_a": {"v": 1}, "spec_b": {"v": 1}}},
    )
    held = first.state[SPECS_KEY]

    second = await store.advance(
        first,
        segment_id=SEGMENT,
        node="approve",
        node_input=None,
        state={SPECS_KEY: {"spec_a": {"v": 1}, "spec_b": {"v": 2}}},
    )
    closed = await store.finish(second, segment_id=SEGMENT, status=TurnStatus.DONE)

    assert second.state[SPECS_KEY]["spec_a"] is held["spec_a"]
    assert second.state[SPECS_KEY]["spec_b"] == {"v": 2}
    assert closed.state[SPECS_KEY]["spec_a"] is held["spec_a"]
    assert closed.state[SPECS_KEY]["spec_b"] is second.state[SPECS_KEY]["spec_b"]


async def test_a_pause_round_takes_one_answer(pg_engine: AsyncEngine) -> None:
    await _conversation(pg_engine)
    store = TurnStore(pg_engine)
    paused = await _paused(store, await _open(store))
    await store.record_answer(
        paused, segment_id=SEGMENT, interrupt_id="int_1", round=1, answer={"pick": "a"}
    )

    with pytest.raises(IntegrityError):
        async with pg_engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO turn_answers (turn, interrupt_id, round, answer) "
                    "VALUES (:turn, 'int_2', 1, '{}')"
                ),
                {"turn": paused.id},
            )


@pytest.mark.parametrize("where", ["state", "spec block"])
async def test_a_value_json_cannot_hold_is_refused_before_the_write(
    pg_engine: AsyncEngine, where: str
) -> None:
    await _conversation(pg_engine)
    store = TurnStore(pg_engine)
    opened = await _open(store)
    nan = float("nan")

    with pytest.raises(ValueError):
        await store.advance(
            opened,
            segment_id=SEGMENT,
            node="planner",
            node_input=None,
            state={"share": nan} if where == "state" else {SPECS_KEY: {"spec_a": {"share": nan}}},
        )
    assert await store.open_cursor(PRINCIPAL, CONVERSATION) == opened


# The conversation `current_conversation_id` names while a turn's frames are
# written and restored.
_LEDGER_CONVERSATION = f"{PRINCIPAL}/{CONVERSATION}"


@pytest.fixture
def in_conversation() -> Iterator[None]:
    token = current_conversation_id.set(_LEDGER_CONVERSATION)
    yield
    current_conversation_id.reset(token)


def _key(n: int) -> dict[str, str]:
    return {"spec_id": f"spec_{n}", "content_version": f"cv_{n}"}


def _held(provenance: ProvenanceStore, n: int, *, share: float = 0.5) -> None:
    """Put a one-retrieval lineage for `_key(n)`, counted as two StatVars."""
    provenance.put(
        conversation=_LEDGER_CONVERSATION, manifest={"citation": {"share": share}}, **_key(n)
    )
    assert provenance.alias(
        conversation=_LEDGER_CONVERSATION,
        **_key(n),
        sources=[(f"spec_{n}", f"cv_{n}")],
        stat_var_count=2,
    )


def _frames(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"plan": None, FRAMES_KEY: {f"frame_{i}": entry for i, entry in enumerate(entries)}}


async def _lineage_rows(engine: AsyncEngine) -> list[tuple[str, str, int | None]]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT spec_id, content_version, stat_var_count FROM turn_frame_lineage "
                "ORDER BY spec_id"
            )
        )
        return [tuple(row) for row in rows]


async def test_a_turns_frames_keep_their_lineage_once_and_another_store_restores_it(
    pg_engine: AsyncEngine, in_conversation: None
) -> None:
    """Each store key a frame entry names (`ref` and `store_ref`) is written
    once, by the write that first stores the entry, and a store whose
    provenance is empty puts each lineage back with its count."""
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    for n in (1, 2, 3):
        _held(provenance, n)
    store = TurnStore(pg_engine, provenance=provenance)
    first = _frames({"ref": _key(1), "store_ref": _key(2)})
    advanced = await store.advance(
        await _open(store), segment_id=SEGMENT, node="orchestrate", node_input=None, state=first
    )
    both = _frames({"ref": _key(1), "store_ref": _key(2)}, {"ref": _key(3)})
    paused = await store.pause(
        advanced,
        segment_id=SEGMENT,
        node="orchestrate",
        node_input=None,
        kind="ask",
        interrupt_id="int_1",
        payload={},
        state=both,
    )
    resumed = await store.record_answer(
        paused, segment_id=SEGMENT, interrupt_id="int_1", round=1, answer={}
    )
    await store.pause(
        resumed,
        segment_id=SEGMENT,
        node="orchestrate",
        node_input=None,
        kind="ask",
        interrupt_id="int_2",
        payload={},
        state=both,
    )

    assert await _lineage_rows(pg_engine) == [
        ("spec_1", "cv_1", 2),
        ("spec_2", "cv_2", 2),
        ("spec_3", "cv_3", 2),
    ]
    fresh = ProvenanceStore()
    other = TurnStore(pg_engine, provenance=fresh)
    open_cursor = await other.open_cursor(PRINCIPAL, CONVERSATION)
    assert open_cursor is not None
    await other.restore_frame_lineage(open_cursor)
    for n in (1, 2, 3):
        key = {"conversation": _LEDGER_CONVERSATION, **_key(n)}
        assert fresh.get(**key) == provenance.get(**key)
        assert fresh.stat_var_count(**key) == 2


async def test_a_frame_the_process_no_longer_holds_or_json_cannot_hold_writes_no_row(
    pg_engine: AsyncEngine,
    in_conversation: None,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The write goes on with the rows it can store: a key the provenance
    store evicted has none, and a lineage holding a NaN has none and is
    logged."""
    monkeypatch.setattr(provenance_module, "_MAX_MANIFESTS_PER_CONVERSATION", 2)
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    _held(provenance, 4)
    _held(provenance, 1)
    _held(provenance, 2, share=float("nan"))
    assert provenance.was_evicted(conversation=_LEDGER_CONVERSATION, **_key(4))
    store = TurnStore(pg_engine, provenance=provenance)

    advanced = await store.advance(
        await _open(store),
        segment_id=SEGMENT,
        node="orchestrate",
        node_input=None,
        state=_frames({"ref": _key(1)}, {"ref": _key(2)}, {"ref": _key(4)}),
    )

    assert advanced.status is TurnStatus.RUNNING
    assert set(advanced.state[FRAMES_KEY]) == {"frame_0", "frame_1", "frame_2"}
    assert await _lineage_rows(pg_engine) == [("spec_1", "cv_1", 2)]
    assert any("spec_2" in record.getMessage() for record in caplog.records)


async def test_a_closing_write_stores_no_frame_lineage(
    pg_engine: AsyncEngine, in_conversation: None
) -> None:
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    _held(provenance, 1)
    store = TurnStore(pg_engine, provenance=provenance)

    await store.finish(
        await _open(store),
        segment_id=SEGMENT,
        status=TurnStatus.DONE,
        state=_frames({"ref": _key(1)}),
    )

    assert await _lineage_rows(pg_engine) == []


@pytest.mark.parametrize("read", ["fails", "stalls"])
async def test_a_frame_lineage_read_that_fails_or_stalls_restores_nothing_and_returns(
    pg_engine: AsyncEngine,
    in_conversation: None,
    monkeypatch: pytest.MonkeyPatch,
    read: str,
) -> None:
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    _held(provenance, 1)
    store = TurnStore(pg_engine, provenance=provenance)
    advanced = await store.advance(
        await _open(store),
        segment_id=SEGMENT,
        node="orchestrate",
        node_input=None,
        state=_frames({"ref": _key(1)}),
    )
    fresh = ProvenanceStore()
    other = TurnStore(pg_engine, provenance=fresh)
    released = asyncio.Event()

    async def _read(turn: int) -> list[Any]:
        if read == "fails":
            raise OSError("connection reset")
        await released.wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(other, "_frame_lineage", _read)
    monkeypatch.setattr(turn_store, "_FRAME_LINEAGE_READ_BUDGET_S", 0.05)

    await asyncio.wait_for(other.restore_frame_lineage(advanced), 5)

    assert fresh.get(conversation=_LEDGER_CONVERSATION, **_key(1)) is None
