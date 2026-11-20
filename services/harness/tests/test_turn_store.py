from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import harness.provenance as provenance_module
import pytest
from executor.ledger import PlanApproval, SourcePolicy, new_run
from harness.db.migrate import MIGRATIONS_DIR, apply_pending
from harness.graph.state import ORCHESTRATOR_FRAMES_KEY, frame_store_keys
from harness.provenance import ProvenanceStore
from harness.studio_state import SegmentLost
from harness.turn import store as turn_store
from harness.turn.store import (
    SPECS_KEY,
    AgentRecord,
    Cursor,
    CursorStale,
    FrameLineage,
    OpenPause,
    OwedAnswer,
    PauseNotOpen,
    SegmentLineage,
    TurnAlreadyOpen,
    TurnStatus,
    TurnStore,
)
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

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
        lineage=None,
    )


async def test_the_turn_tables_apply_on_top_of_the_conversation_tables(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    for name in ("0001_conversations.sql", "0002_adk_ledger_imports.sql"):
        shutil.copy(MIGRATIONS_DIR / name, tmp_path / name)
    await apply_pending(pg_empty_engine, directory=tmp_path)

    assert await apply_pending(pg_empty_engine) == ["0003_turns.sql", "0004_card_lineage.sql"]
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
        lineage=None,
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
        lineage=None,
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
        lineage=None,
    )

    with pytest.raises(CursorStale):
        await store.advance(
            opened,
            segment_id=SEGMENT,
            node="orchestrate",
            node_input=None,
            state={"winner": False, SPECS_KEY: {"spec_a": {"by": "loser"}, "spec_b": {}}},
            records=[AgentRecord(agent="planner", kind="response", body={})],
            lineage=None,
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
        await store.advance(
            opened,
            segment_id=SEGMENT,
            node="planner",
            node_input=None,
            state={},
            lineage=None,
        )
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
            resumed,
            segment_id=SEGMENT,
            node="planner",
            node_input=None,
            state={},
            lineage=None,
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
        lineage=None,
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
        lineage=None,
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
        lineage=None,
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
            cursor,
            segment_id=SEGMENT,
            node="planner",
            node_input=None,
            state={SPECS_KEY: specs},
            lineage=None,
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
        lineage=None,
    )
    stored = {"spec_a": {"v": 1, "pair": [1, 2], "by": {"7": "x"}}}
    assert first.state[SPECS_KEY] == stored

    first.state[SPECS_KEY]["spec_a"]["v"] = 2
    second = await store.advance(
        first,
        segment_id=SEGMENT,
        node="approve",
        node_input=None,
        state=first.state,
        lineage=None,
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
        lineage=None,
    )
    held = first.state[SPECS_KEY]

    second = await store.advance(
        first,
        segment_id=SEGMENT,
        node="approve",
        node_input=None,
        state={SPECS_KEY: {"spec_a": {"v": 1}, "spec_b": {"v": 2}}},
        lineage=None,
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
            lineage=None,
        )
    assert await store.open_cursor(PRINCIPAL, CONVERSATION) == opened


# The conversation the provenance store holds a turn's frames under.
_SCOPE = f"{PRINCIPAL}/{CONVERSATION}"


def _key(n: int) -> dict[str, str]:
    return {"spec_id": f"spec_{n}", "content_version": f"cv_{n}"}


def _held(provenance: ProvenanceStore, n: int, *, share: float = 0.5, count: int = 2) -> None:
    """Put a one-retrieval lineage for `_key(n)`, counted as `count` StatVars."""
    provenance.put(conversation=_SCOPE, manifest={"citation": {"share": share}}, **_key(n))
    assert provenance.alias(
        conversation=_SCOPE, **_key(n), sources=[(f"spec_{n}", f"cv_{n}")], stat_var_count=count
    )


def _frames(*entries: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan": None,
        ORCHESTRATOR_FRAMES_KEY: {f"frame_{i}": entry for i, entry in enumerate(entries)},
    }


def _keeping(engine: AsyncEngine, provenance: ProvenanceStore) -> TurnStore:
    """A store keeping the lineage `provenance` holds of each frame a turn names."""
    return TurnStore(engine, lineage=FrameLineage(source=provenance, frame_keys=frame_store_keys))


async def _segment(store: TurnStore, cursor: Cursor) -> SegmentLineage:
    """Start a segment of `cursor`'s turn on `store` and restore its frame
    lineage, as the turn loop does (`TurnStore.start_segment`,
    `TurnStore.restore_frame_lineage`)."""
    segment = store.start_segment(lineage_scope=_SCOPE)
    assert segment is not None
    await store.restore_frame_lineage(cursor, segment)
    return segment


async def _opened(store: TurnStore, provenance: ProvenanceStore) -> tuple[Cursor, SegmentLineage]:
    """Open the turn and start its first segment (`_segment`), then store
    again every lineage the test put in `provenance`, as the segment's own
    fetches would, so each is produced after the segment's restore point."""
    cursor = await _open(store)
    segment = await _segment(store, cursor)
    for key in list(provenance._by_conversation.get(_SCOPE, {})):
        held = provenance.held(conversation=_SCOPE, spec_id=key[0], content_version=key[1])
        assert held is not None
        provenance.restore_frame(
            conversation=_SCOPE, frame=key, lineage=held[0], stat_var_count=held[1], overwrite=True
        )
    return cursor, segment


async def _advance(
    segment: SegmentLineage, store: TurnStore, cursor: Cursor, state: dict[str, Any]
) -> Cursor:
    return await store.advance(
        cursor,
        segment_id=SEGMENT,
        node="orchestrate",
        node_input=None,
        state=state,
        lineage=segment,
    )


async def _pause(
    segment: SegmentLineage,
    store: TurnStore,
    cursor: Cursor,
    state: dict[str, Any],
    *,
    interrupt_id: str,
) -> Cursor:
    return await store.pause(
        cursor,
        segment_id=SEGMENT,
        node="orchestrate",
        node_input=None,
        kind="ask",
        interrupt_id=interrupt_id,
        payload={},
        state=state,
        lineage=segment,
    )


async def _lineage_rows(engine: AsyncEngine) -> list[tuple[str, str, int | None, Any]]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT spec_id, content_version, stat_var_count, members "
                "FROM turn_frame_lineage ORDER BY spec_id"
            )
        )
        return [tuple(row) for row in rows]


async def _restored(engine: AsyncEngine) -> tuple[ProvenanceStore, TurnStore]:
    """A store with an empty provenance store, its turn's lineage restored."""
    fresh = ProvenanceStore()
    other = _keeping(engine, fresh)
    cursor = await other.open_cursor(PRINCIPAL, CONVERSATION)
    assert cursor is not None
    await _segment(other, cursor)
    return fresh, other


async def test_a_turns_frames_keep_their_lineage_and_another_store_restores_it(
    pg_engine: AsyncEngine,
) -> None:
    """Each store key a frame entry names (`ref` and `store_ref`) has one row
    across the turn's writes, and a store whose provenance is empty puts
    each lineage back with its count."""
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    for n in (1, 2, 3):
        _held(provenance, n)
    store = _keeping(pg_engine, provenance)
    first = _frames({"ref": _key(1), "store_ref": _key(2)})
    opened, seg = await _opened(store, provenance)
    advanced = await _advance(seg, store, opened, first)
    both = _frames({"ref": _key(1), "store_ref": _key(2)}, {"ref": _key(3)})
    paused = await _pause(seg, store, advanced, both, interrupt_id="int_1")
    resumed = await store.record_answer(
        paused, segment_id=SEGMENT, interrupt_id="int_1", round=1, answer={}
    )
    await _pause(seg, store, resumed, both, interrupt_id="int_2")

    assert [row[:3] for row in await _lineage_rows(pg_engine)] == [
        ("spec_1", "cv_1", 2),
        ("spec_2", "cv_2", 2),
        ("spec_3", "cv_3", 2),
    ]
    fresh, _ = await _restored(pg_engine)
    for n in (1, 2, 3):
        key = {"conversation": _SCOPE, **_key(n)}
        assert fresh.get(**key) == provenance.get(**key)
        assert fresh.stat_var_count(**key) == 2


async def test_a_lineage_replaced_in_process_replaces_its_row(pg_engine: AsyncEngine) -> None:
    """A key whose lineage the process replaces after its row was written
    (a later retrieval served under the same pair) is written again, so a
    restore states the lineage the process last held."""
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    _held(provenance, 1, share=0.25, count=1)
    store = _keeping(pg_engine, provenance)
    state = _frames({"ref": _key(1)})
    opened, seg = await _opened(store, provenance)
    advanced = await _advance(seg, store, opened, state)
    _held(provenance, 1, share=0.75, count=3)
    await _pause(seg, store, advanced, state, interrupt_id="int_1")

    (row,) = await _lineage_rows(pg_engine)
    assert row[2] == 3
    assert row[3][0]["manifest"] == {"citation": {"share": 0.75}}
    fresh, _ = await _restored(pg_engine)
    key = {"conversation": _SCOPE, **_key(1)}
    assert fresh.get(**key) == provenance.get(**key)
    assert fresh.stat_var_count(**key) == 3


async def test_a_frame_held_again_after_a_write_without_it_is_written_next(
    pg_engine: AsyncEngine,
) -> None:
    """A frame the process did not hold when the write that first named it
    ran is written by the next write after the process holds it again."""
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    store = _keeping(pg_engine, provenance)
    state = _frames({"ref": _key(1)})
    opened, seg = await _opened(store, provenance)
    advanced = await _advance(seg, store, opened, state)
    assert await _lineage_rows(pg_engine) == []

    _held(provenance, 1)
    await _pause(seg, store, advanced, state, interrupt_id="int_1")

    assert [row[:3] for row in await _lineage_rows(pg_engine)] == [("spec_1", "cv_1", 2)]


def _evict(provenance: ProvenanceStore, n: int) -> None:
    """Drop `_key(n)` from `provenance`, as its per-conversation bound does."""
    del provenance._by_conversation[_SCOPE][(f"spec_{n}", f"cv_{n}")]


async def test_a_segment_on_the_process_that_wrote_last_draws_without_a_new_fetch(
    pg_engine: AsyncEngine,
) -> None:
    """The next segment on the process that paused the turn reads its rows
    back over what it holds, so every named frame stays drawable with the
    lineage it was stored with, and a frame the turn dropped is not read."""
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    for n in (1, 2, 3):
        _held(provenance, n)
    store = _keeping(pg_engine, provenance)
    opened, seg = await _opened(store, provenance)
    advanced = await _advance(
        seg,
        store,
        opened,
        _frames({"ref": _key(1)}, {"ref": _key(2)}, {"ref": _key(3)}),
    )
    stored = {n: provenance.get(conversation=_SCOPE, **_key(n)) for n in (1, 2)}
    paused = await _pause(
        seg, store, advanced, _frames({"ref": _key(1)}, {"ref": _key(2)}), interrupt_id="i"
    )
    _evict(provenance, 1)
    _evict(provenance, 3)

    await _segment(store, paused)

    for n in (1, 2):
        assert provenance.get(conversation=_SCOPE, **_key(n)) == stored[n]
        assert provenance.stat_var_count(conversation=_SCOPE, **_key(n)) == 2
    assert provenance.get(conversation=_SCOPE, **_key(3)) is None


async def test_a_failed_restore_read_leaves_the_rows_as_another_writer_committed_them(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Instance A holds L1, B commits L2, and the turn comes back to A, whose
    read of the rows fails: A's next write stores nothing over L2, since A's
    entry predates the segment's restore point and no read confirmed it."""
    await _conversation(pg_engine)
    held_a = ProvenanceStore()
    _held(held_a, 1, share=0.25)
    a = _keeping(pg_engine, held_a)
    state = _frames({"ref": _key(1)})
    opened, seg_a = await _opened(a, held_a)
    paused = await _pause(seg_a, a, opened, state, interrupt_id="int_1")
    held_b = ProvenanceStore()
    b = _keeping(pg_engine, held_b)
    seg_b = await _segment(b, paused)
    resumed = await b.record_answer(
        paused, segment_id=SEGMENT, interrupt_id="int_1", round=1, answer={}
    )
    _held(held_b, 1, share=0.75, count=4)
    by_b = await _pause(seg_b, b, resumed, state, interrupt_id="int_2")

    async def _fails(self: TurnStore, turn: int, keys: Any) -> list[Any]:
        raise OSError("connection reset")

    with monkeypatch.context() as patch:
        patch.setattr(TurnStore, "_frame_lineage", _fails)
        seg_a = await _segment(a, by_b)
    resumed_a = await a.record_answer(
        by_b, segment_id=SEGMENT, interrupt_id="int_2", round=2, answer={}
    )
    await _advance(seg_a, a, resumed_a, state)

    (row,) = await _lineage_rows(pg_engine)
    assert (row[2], row[3][0]["manifest"]) == (4, {"citation": {"share": 0.75}})


async def test_a_named_frame_with_no_row_is_dropped_and_not_stored_again(
    pg_engine: AsyncEngine,
) -> None:
    """Instance A holds L1; B's lineage for the frame fails its check, so B's
    write deletes the row. Back on A, the restore finds no row and drops A's
    entry, and A's next write does not store L1 as the turn's record."""
    await _conversation(pg_engine)
    held_a = ProvenanceStore()
    _held(held_a, 1, share=0.25)
    a = _keeping(pg_engine, held_a)
    state = _frames({"ref": _key(1)})
    opened, seg_a = await _opened(a, held_a)
    paused = await _pause(seg_a, a, opened, state, interrupt_id="int_1")
    held_b = ProvenanceStore()
    b = _keeping(pg_engine, held_b)
    seg_b = await _segment(b, paused)
    resumed = await b.record_answer(
        paused, segment_id=SEGMENT, interrupt_id="int_1", round=1, answer={}
    )
    _held(held_b, 1, share=float("nan"))
    by_b = await _pause(seg_b, b, resumed, state, interrupt_id="int_2")
    assert await _lineage_rows(pg_engine) == []

    seg_a = await _segment(a, by_b)
    resumed_a = await a.record_answer(
        by_b, segment_id=SEGMENT, interrupt_id="int_2", round=2, answer={}
    )
    await _advance(seg_a, a, resumed_a, state)

    assert held_a.get(conversation=_SCOPE, **_key(1)) is None
    assert await _lineage_rows(pg_engine) == []


@pytest.mark.parametrize(
    "frames",
    [{"broken": ["ref"]}, ["ref"]],
    ids=["entry-not-a-mapping", "frames-not-a-mapping"],
)
async def test_a_malformed_frames_entry_names_no_frame_and_the_turn_goes_on(
    pg_engine: AsyncEngine,
    caplog: pytest.LogCaptureFixture,
    frames: Any,
) -> None:
    """A frames value, or a frames entry, that is not a mapping names no
    key: the writes and the next segment's restore go on, the writes store
    the well-formed frames' rows, and each segment logs the case once,
    worded for which of the two it is."""
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    _held(provenance, 1)
    store = _keeping(pg_engine, provenance)
    opened, seg = await _opened(store, provenance)
    if isinstance(frames, dict):
        state = {ORCHESTRATOR_FRAMES_KEY: {**frames, "frame_0": {"ref": _key(1)}}}
    else:
        state = {ORCHESTRATOR_FRAMES_KEY: frames}

    advanced = await _advance(seg, store, opened, state)
    advanced = await _advance(seg, store, advanced, state)
    await _segment(store, advanced)

    assert advanced.status is TurnStatus.RUNNING
    rows = [row[:2] for row in await _lineage_rows(pg_engine)]
    assert rows == ([("spec_1", "cv_1")] if isinstance(frames, dict) else [])
    wording = "are not mappings" if isinstance(frames, dict) else "is a list, not a mapping"
    assert len([r for r in caplog.records if wording in r.getMessage()]) == 2
    assert not any("restore failed" in record.getMessage() for record in caplog.records)


async def test_rows_written_by_another_writer_replace_what_the_process_held(
    pg_engine: AsyncEngine,
) -> None:
    """Instance A holds a frame as L1, instance B takes the turn over and
    commits L2 for it, and the turn comes back to A: A's restore puts the
    row over its own entry, so A draws from L2 and its next write keeps L2."""
    await _conversation(pg_engine)
    held_a = ProvenanceStore()
    _held(held_a, 1, share=0.25)
    a = _keeping(pg_engine, held_a)
    state = _frames({"ref": _key(1)})
    opened, seg_a = await _opened(a, held_a)
    paused = await _pause(seg_a, a, opened, state, interrupt_id="int_1")

    held_b = ProvenanceStore()
    b = _keeping(pg_engine, held_b)
    seg_b = await _segment(b, paused)
    resumed = await b.record_answer(
        paused, segment_id=SEGMENT, interrupt_id="int_1", round=1, answer={}
    )
    _held(held_b, 1, share=0.75, count=4)
    by_b = await _pause(seg_b, b, resumed, state, interrupt_id="int_2")

    seg_a = await _segment(a, by_b)
    resumed_a = await a.record_answer(
        by_b, segment_id=SEGMENT, interrupt_id="int_2", round=2, answer={}
    )
    await _advance(seg_a, a, resumed_a, state)

    key = {"conversation": _SCOPE, **_key(1)}
    assert held_a.get(**key) == held_b.get(**key)
    assert held_a.stat_var_count(**key) == 4
    (row,) = await _lineage_rows(pg_engine)
    assert (row[2], row[3][0]["manifest"]) == (4, {"citation": {"share": 0.75}})


@pytest.mark.parametrize("part", ["spec_id", "content_version"])
async def test_a_frame_key_holding_a_nul_gets_no_row_and_is_never_read(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, part: str
) -> None:
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    _held(provenance, 1)
    nul = {**_key(7), part: _key(7)[part] + "\x00"}
    provenance.put(conversation=_SCOPE, manifest={}, **nul)
    store = _keeping(pg_engine, provenance)

    opened, seg = await _opened(store, provenance)
    advanced = await _advance(seg, store, opened, _frames({"ref": _key(1)}, {"ref": nul}))

    assert [row[:2] for row in await _lineage_rows(pg_engine)] == [("spec_1", "cv_1")]
    reads: list[Any] = []
    read = TurnStore._frame_lineage

    async def recorded(self: TurnStore, turn: int, keys: Any) -> list[Any]:
        reads.append(list(keys))
        return await read(self, turn, keys)

    monkeypatch.setattr(TurnStore, "_frame_lineage", recorded)
    fresh = ProvenanceStore()
    await _segment(_keeping(pg_engine, fresh), advanced)
    assert reads == [[("spec_1", "cv_1")]]
    assert fresh.get(conversation=_SCOPE, **_key(1)) is not None


async def test_a_frame_the_process_no_longer_holds_or_json_cannot_hold_writes_no_row(
    pg_engine: AsyncEngine, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
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
    assert provenance.was_evicted(conversation=_SCOPE, **_key(4))
    store = _keeping(pg_engine, provenance)

    opened, seg = await _opened(store, provenance)
    advanced = await _advance(
        seg,
        store,
        opened,
        _frames({"ref": _key(1)}, {"ref": _key(2)}, {"ref": _key(4)}),
    )

    assert advanced.status is TurnStatus.RUNNING
    assert set(advanced.state[ORCHESTRATOR_FRAMES_KEY]) == {"frame_0", "frame_1", "frame_2"}
    assert [row[:3] for row in await _lineage_rows(pg_engine)] == [("spec_1", "cv_1", 2)]
    assert any("spec_2" in record.getMessage() for record in caplog.records)


async def test_a_closing_write_deletes_the_turns_frame_lineage(pg_engine: AsyncEngine) -> None:
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    _held(provenance, 1)
    store = _keeping(pg_engine, provenance)
    opened, seg = await _opened(store, provenance)
    advanced = await _advance(seg, store, opened, _frames({"ref": _key(1)}))
    assert len(await _lineage_rows(pg_engine)) == 1

    await store.finish(advanced, segment_id=SEGMENT, status=TurnStatus.DONE)

    assert await _lineage_rows(pg_engine) == []


@pytest.mark.parametrize("read", ["fails", "stalls"])
async def test_a_frame_lineage_read_that_fails_or_stalls_restores_nothing_and_returns(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, read: str
) -> None:
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    _held(provenance, 1)
    store = _keeping(pg_engine, provenance)
    opened, seg = await _opened(store, provenance)
    advanced = await _advance(seg, store, opened, _frames({"ref": _key(1)}))
    fresh = ProvenanceStore()
    other = _keeping(pg_engine, fresh)
    released = asyncio.Event()

    async def _read(turn: int, keys: Any) -> list[Any]:
        if read == "fails":
            raise OSError("connection reset")
        await released.wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(other, "_frame_lineage", _read)
    monkeypatch.setattr(turn_store, "_FRAME_LINEAGE_READ_BUDGET_S", 0.05)

    await asyncio.wait_for(_segment(other, advanced), 5)

    assert fresh.get(conversation=_SCOPE, **_key(1)) is None


async def test_a_burst_of_other_turns_mid_segment_keeps_the_segments_frames(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The segment holds its own restore point and kept set, so however many
    other segments the process starts and ends while it runs, its frames
    outlive the per-conversation bound and its next write stores the frames
    it produced."""
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    _held(provenance, 1)
    store = _keeping(pg_engine, provenance)
    opened, seg = await _opened(store, provenance)
    advanced = await _advance(seg, store, opened, _frames({"ref": _key(1)}))
    for turn in range(opened.id + 1, opened.id + 300):
        store.end_segment(await _segment(store, replace(opened, id=turn, state={})))
    monkeypatch.setattr(provenance_module, "_MAX_MANIFESTS_PER_CONVERSATION", 1)
    for n in (4, 5):
        _held(provenance, n)
    _held(provenance, 2)

    await _advance(seg, store, advanced, _frames({"ref": _key(1)}, {"ref": _key(2)}))

    assert provenance.get(conversation=_SCOPE, **_key(1)) is not None
    assert [row[:2] for row in await _lineage_rows(pg_engine)] == [
        ("spec_1", "cv_1"),
        ("spec_2", "cv_2"),
    ]


@pytest.mark.parametrize("second_names", ["nothing", "the same frame"])
async def test_a_second_segment_ending_leaves_the_first_segments_frames_kept(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, second_names: str
) -> None:
    """Each running segment keeps its own frames: a second segment of the
    conversation that names none, or the same frame, and then ends, takes
    nothing of the first segment's kept set."""
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    _held(provenance, 1)
    store = _keeping(pg_engine, provenance)
    opened, first = await _opened(store, provenance)
    advanced = await _advance(first, store, opened, _frames({"ref": _key(1)}))

    second_state = {} if second_names == "nothing" else _frames({"ref": _key(1)})
    second = await _segment(store, replace(advanced, state=second_state))
    store.end_segment(second)
    monkeypatch.setattr(provenance_module, "_MAX_MANIFESTS_PER_CONVERSATION", 1)
    for n in (4, 5):
        _held(provenance, n)

    assert provenance.get(conversation=_SCOPE, **_key(1)) is not None
    store.end_segment(first)
    _held(provenance, 6)
    assert provenance.get(conversation=_SCOPE, **_key(1)) is None


@pytest.mark.parametrize("frames", ["none", "named-without-rows"])
async def test_a_segment_that_keeps_no_frame_records_no_kept_entry(
    pg_engine: AsyncEngine, frames: str
) -> None:
    """A chat-only segment, or one whose named frames have no rows, gives the
    provenance store no bucket, so it records no kept entry either."""
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    store = _keeping(pg_engine, provenance)
    opened = await _open(store)
    if frames == "named-without-rows":
        opened = replace(opened, state=_frames({"ref": _key(1)}))

    segment = await _segment(store, opened)
    if frames == "named-without-rows":
        store.end_segment(segment)

    assert _SCOPE not in provenance._kept


async def test_a_superseded_segments_write_leaves_the_live_segments_kept_frames(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Segment A's lease lapses and segment B of the same process takes the
    turn and keeps its frame. A's write, which the cursor refuses, changes
    nothing of B's kept set, so B's frame outlives the bound."""
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    _held(provenance, 1)
    store = _keeping(pg_engine, provenance)
    opened, seg_a = await _opened(store, provenance)
    named = _frames({"ref": _key(1)})
    paused = await _pause(seg_a, store, opened, named, interrupt_id="i")
    seg_b = await _segment(store, paused)
    resumed = await store.record_answer(
        paused, segment_id=SEGMENT, interrupt_id="i", round=1, answer={}
    )

    with pytest.raises(CursorStale):
        await _advance(seg_a, store, paused, _frames({"ref": _key(9)}))
    monkeypatch.setattr(provenance_module, "_MAX_MANIFESTS_PER_CONVERSATION", 2)
    for n in (4, 5, 6):
        _held(provenance, n)

    assert provenance.get(conversation=_SCOPE, **_key(1)) is not None
    await _advance(seg_b, store, resumed, named)
    assert [row[:2] for row in await _lineage_rows(pg_engine)] == [("spec_1", "cv_1")]


async def test_a_restore_holds_every_frame_the_turn_names_past_the_per_conversation_bound(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn naming more frames than the provenance store's per-conversation
    bound gets every one back on a restore, and later stores in the
    conversation evict other entries before them."""
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    for n in (1, 2, 3):
        _held(provenance, n)
    store = _keeping(pg_engine, provenance)
    opened, seg = await _opened(store, provenance)
    named = _frames({"ref": _key(1)}, {"ref": _key(2)}, {"ref": _key(3)})
    paused = await _pause(seg, store, opened, named, interrupt_id="i")
    monkeypatch.setattr(provenance_module, "_MAX_MANIFESTS_PER_CONVERSATION", 2)
    fresh = ProvenanceStore()
    other = _keeping(pg_engine, fresh)

    await _segment(other, paused)
    for n in (4, 5, 6):
        _held(fresh, n)

    for n in (1, 2, 3):
        key = {"conversation": _SCOPE, **_key(n)}
        assert fresh.get(**key) == provenance.get(**key)
    assert fresh.was_evicted(conversation=_SCOPE, **_key(4))


async def _refuse_lineage_writes(engine: AsyncEngine, errcode: str) -> None:
    """Make the database refuse every insert or update of a lineage row with
    SQLSTATE `errcode`, as a trigger raising it."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE FUNCTION refuse_lineage() RETURNS trigger LANGUAGE plpgsql AS "
                f"$$ BEGIN RAISE EXCEPTION 'refused' USING ERRCODE = '{errcode}'; END $$"
            )
        )
        await conn.execute(
            text(
                "CREATE TRIGGER refuse_lineage BEFORE INSERT OR UPDATE ON turn_frame_lineage "
                "FOR EACH ROW EXECUTE FUNCTION refuse_lineage()"
            )
        )


async def _allow_lineage_writes(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("DROP TRIGGER refuse_lineage ON turn_frame_lineage"))


def _refusals(caplog: pytest.LogCaptureFixture, needle: str) -> int:
    return len([r for r in caplog.records if r.levelname == "ERROR" and needle in r.getMessage()])


async def test_a_transient_fault_of_the_lineage_write_fails_the_write_and_counts_nothing(
    pg_engine: AsyncEngine, caplog: pytest.LogCaptureFixture
) -> None:
    """A fault the store retries whole (here cannot_connect_now, 57P03)
    raises out of the cursor write, which does not commit, and the next
    write stores the row as if the fault never happened."""
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    _held(provenance, 1)
    store = _keeping(pg_engine, provenance)
    state = _frames({"ref": _key(1)})
    opened, seg = await _opened(store, provenance)
    await _refuse_lineage_writes(pg_engine, "57P03")

    with pytest.raises(DBAPIError):
        await _advance(seg, store, opened, state)
    assert await store.open_cursor(PRINCIPAL, CONVERSATION) == opened
    await _allow_lineage_writes(pg_engine)

    await _advance(seg, store, opened, state)

    assert [row[:2] for row in await _lineage_rows(pg_engine)] == [("spec_1", "cv_1")]
    assert _refusals(caplog, "turn store") == 0


async def test_an_unchanged_lineage_is_not_written_again_whatever_object_holds_it(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store compares the source's generation, so a source that hands out
    a copy of an unchanged lineage on each read causes no second upsert."""
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    _held(provenance, 1)
    get = ProvenanceStore.get

    def copying(self: ProvenanceStore, **kwargs: Any) -> Any:
        lineage = get(self, **kwargs)
        return None if lineage is None else tuple(member for member in lineage)

    monkeypatch.setattr(ProvenanceStore, "get", copying)
    store = _keeping(pg_engine, provenance)
    state = _frames({"ref": _key(1)})
    opened, seg = await _opened(store, provenance)
    advanced = await _advance(seg, store, opened, state)
    first = await _row_versions(pg_engine)
    await _pause(seg, store, advanced, state, interrupt_id="int_1")

    assert await _row_versions(pg_engine) == first


async def _row_versions(engine: AsyncEngine) -> list[tuple[str, str]]:
    """Each lineage row's key and the transaction that last wrote it."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT spec_id, xmin::text FROM turn_frame_lineage ORDER BY spec_id")
        )
        return [tuple(row) for row in rows]


async def test_a_close_fails_when_the_frame_lineage_delete_fails(pg_engine: AsyncEngine) -> None:
    """The close deletes the turn's rows in its own transaction, so a close
    whose delete fails does not commit and the turn stays open as written."""
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    _held(provenance, 1)
    store = _keeping(pg_engine, provenance)
    opened, seg = await _opened(store, provenance)
    advanced = await _advance(seg, store, opened, _frames({"ref": _key(1)}))
    async with pg_engine.begin() as conn:
        await conn.execute(text("ALTER TABLE turn_frame_lineage RENAME TO frame_lineage_gone"))

    with pytest.raises(DBAPIError):
        await store.finish(advanced, segment_id=SEGMENT, status=TurnStatus.DONE)

    assert await store.open_cursor(PRINCIPAL, CONVERSATION) == advanced


@pytest.mark.parametrize("invalid", ["nan", "count-past-integer"])
async def test_a_lineage_that_fails_its_check_deletes_the_older_row_and_the_write_commits(
    pg_engine: AsyncEngine, caplog: pytest.LogCaptureFixture, invalid: str
) -> None:
    """A lineage JSON cannot hold, or whose count the integer column cannot,
    never reaches the database: the key's older row is deleted, the other
    frames' rows land, the write commits, and the check is logged once while
    the lineage is unchanged."""
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    for n in (1, 9):
        _held(provenance, n)
    store = _keeping(pg_engine, provenance)
    state = _frames({"ref": _key(1)}, {"ref": _key(9)})
    opened, seg = await _opened(store, provenance)
    cursor = await _advance(seg, store, opened, state)
    if invalid == "nan":
        _held(provenance, 9, share=float("nan"))
    else:
        _held(provenance, 9, count=2**40)
    _held(provenance, 1, share=0.6)

    cursor = await _advance(seg, store, cursor, state)
    await _advance(seg, store, cursor, state)

    (row,) = await _lineage_rows(pg_engine)
    assert row[0] == "spec_1"
    assert row[3][0]["manifest"] == {"citation": {"share": 0.6}}
    assert len([r for r in caplog.records if "spec_9" in r.getMessage()]) == 1


async def test_a_frame_key_postgres_text_cannot_hold_gets_no_row_and_the_write_commits(
    pg_engine: AsyncEngine,
) -> None:
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    _held(provenance, 1)
    surrogate = {"spec_id": "spec_\ud800", "content_version": "cv_x"}
    provenance.put(conversation=_SCOPE, manifest={}, **surrogate)
    store = _keeping(pg_engine, provenance)

    opened, seg = await _opened(store, provenance)
    advanced = await _advance(seg, store, opened, _frames({"ref": _key(1)}, {"ref": surrogate}))

    assert advanced.status is TurnStatus.RUNNING
    assert [row[:2] for row in await _lineage_rows(pg_engine)] == [("spec_1", "cv_1")]
    fresh, _ = await _restored(pg_engine)
    assert fresh.get(conversation=_SCOPE, **_key(1)) is not None


async def test_a_database_refusal_of_the_lineage_fails_the_cursor_write_and_commits_nothing(
    pg_engine: AsyncEngine,
) -> None:
    """A refusal by the database (here disk_full, 53100) of the lineage
    statement fails the cursor write as a refusal of the cursor statement
    would: neither the cursor nor any lineage row moves."""
    await _conversation(pg_engine)
    provenance = ProvenanceStore()
    _held(provenance, 1)
    store = _keeping(pg_engine, provenance)
    state = _frames({"ref": _key(1)})
    opened, seg = await _opened(store, provenance)
    advanced = await _advance(seg, store, opened, state)
    stored = await _row_versions(pg_engine)
    _held(provenance, 1, share=0.6)
    await _refuse_lineage_writes(pg_engine, "53100")

    with pytest.raises(DBAPIError):
        await _advance(seg, store, advanced, state)

    assert await store.open_cursor(PRINCIPAL, CONVERSATION) == advanced
    assert await _row_versions(pg_engine) == stored


class _HeldCommit:
    """An engine whose next `begin` holds its COMMIT open until cancelled:
    before the COMMIT reaches the server (`landed=False`), or once the
    COMMIT landed, with its acknowledgement never arriving (`landed=True`).
    `holding` is set once it holds. Everything else is `engine`'s."""

    def __init__(self, engine: AsyncEngine, *, landed: bool) -> None:
        self._engine = engine
        self._landed = landed
        self._first = True
        self.holding = asyncio.Event()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def begin(self) -> Any:
        if not self._first:
            return self._engine.begin()
        self._first = False
        return self._held()

    @contextlib.asynccontextmanager
    async def _held(self) -> Any:
        async with self._engine.begin() as conn:
            yield conn
            if not self._landed:
                self.holding.set()
                await asyncio.sleep(3600)
        self.holding.set()
        await asyncio.sleep(3600)


class _LostAckThenOverwritten:
    """An engine whose next `begin` commits, then has segment B take the
    lease and write the turn's cursor over, then raises as a lost
    acknowledgement does. Everything else is `engine`'s."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._first = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def begin(self) -> Any:
        if not self._first:
            return self._engine.begin()
        self._first = False
        return self._lost()

    @contextlib.asynccontextmanager
    async def _lost(self) -> Any:
        async with self._engine.begin() as conn:
            yield conn
        await _move_lease(self._engine, "segment-b")
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE turn_cursors SET version = version + 1, written_by = 'segment-b', "
                    "status = 'running', interrupt_id = NULL"
                )
            )
        raise ConnectionResetError("the pause's acknowledgement was lost")


@pytest.mark.parametrize(
    "how",
    [
        "returns",
        "caller-cancelled-landed-held-open",
        "caller-cancelled-held-open-before-landing",
        "fenced-out",
        "lost-ack-overwritten",
    ],
)
async def test_a_pause_hands_its_cursor_to_committed_once_its_write_commits(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """`committed` runs once, with the paused cursor, once the pause's write
    commits. A cut of the caller while the COMMIT is held open is passed
    into the COMMIT, so the caller ends in the cancellation within the read
    back's limit, here 0.5 s (`PAUSE_READ_BACK_LIMIT_S`): a COMMIT that
    landed runs `committed` once, and one that had not stores nothing and
    runs nothing. A COMMIT whose acknowledgement is lost and whose
    transaction committed runs `committed` and raises its error, even after
    another segment wrote the turn over. A pause fenced out by a moved lease
    commits nothing and runs nothing."""
    monkeypatch.setattr(turn_store, "PAUSE_READ_BACK_LIMIT_S", 0.5)
    await _conversation(pg_engine)
    store = TurnStore(pg_engine)
    cursor = await _open(store)
    noted: list[Cursor] = []

    def pause() -> Any:
        return store.pause(
            cursor,
            segment_id=SEGMENT,
            node="ask",
            node_input={"question": "which donor?"},
            kind="ask",
            interrupt_id="int_1",
            payload={"question": "which donor?"},
            state=cursor.state,
            lineage=None,
            committed=noted.append,
        )

    if how == "returns":
        paused = await pause()
        assert noted == [paused]
    elif how == "lost-ack-overwritten":
        store._engine = _LostAckThenOverwritten(pg_engine)  # type: ignore[assignment]
        with pytest.raises(ConnectionResetError):
            await pause()
        store._engine = pg_engine  # type: ignore[assignment]
        [stored] = noted
        assert stored.status is TurnStatus.PAUSED
        assert (await store.turn(cursor.id)).written_by == "segment-b"
        return
    elif how == "fenced-out":
        await _move_lease(pg_engine, "segment-b")
        with pytest.raises(SegmentLost):
            await pause()
        assert noted == []
        return
    else:
        landed = how == "caller-cancelled-landed-held-open"
        held = _HeldCommit(pg_engine, landed=landed)
        store._engine = held  # type: ignore[assignment]
        task = asyncio.ensure_future(pause())
        await asyncio.wait_for(held.holding.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        store._engine = pg_engine  # type: ignore[assignment]
        if not landed:
            assert noted == []
            assert (await store.turn(cursor.id)).status is TurnStatus.RUNNING
            return
    [stored] = noted
    assert stored.status is TurnStatus.PAUSED
    assert stored.pause is not None and stored.pause.interrupt_id == "int_1"
    assert (await store.turn(cursor.id)).status is TurnStatus.PAUSED


async def test_a_cut_of_a_pause_whose_commit_stalls_frees_its_connection(
    pg_engine: AsyncEngine,
) -> None:
    """A real COMMIT held up by a deferred constraint trigger that sleeps is
    cancelled when the caller is cut: the cut is passed into the COMMIT,
    whose transaction aborts, so the caller ends in the cancellation within
    the read back's limit, `committed` never runs, and the pool has every
    connection back."""
    await _conversation(pg_engine)
    store = TurnStore(pg_engine)
    cursor = await _open(store)
    async with pg_engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE FUNCTION stall_commit() RETURNS trigger LANGUAGE plpgsql AS "
                "$$ BEGIN PERFORM pg_sleep(30); RETURN NULL; END $$"
            )
        )
        await conn.execute(
            text(
                "CREATE CONSTRAINT TRIGGER stall_commit AFTER UPDATE ON turn_cursors "
                "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION stall_commit()"
            )
        )
    noted: list[Cursor] = []
    task = asyncio.ensure_future(
        store.pause(
            cursor,
            segment_id=SEGMENT,
            node="ask",
            node_input={},
            kind="ask",
            interrupt_id="int_1",
            payload={},
            state=cursor.state,
            lineage=None,
            committed=noted.append,
        )
    )
    watch = create_async_engine(pg_engine.url)
    try:
        for _ in range(500):
            async with watch.connect() as conn:
                stalled = (
                    await conn.execute(
                        text(
                            "SELECT count(*) FROM pg_stat_activity WHERE datname = "
                            "current_database() AND query ILIKE 'COMMIT%' AND state = 'active'"
                        )
                    )
                ).scalar()
            if stalled:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("the COMMIT never stalled")
    finally:
        await watch.dispose()
    loop = asyncio.get_running_loop()
    started = loop.time()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=turn_store.PAUSE_READ_BACK_LIMIT_S + 1)

    assert loop.time() - started < turn_store.PAUSE_READ_BACK_LIMIT_S + 1
    assert noted == []
    assert pg_engine.pool.checkedout() == 0  # type: ignore[attr-defined]
    async with pg_engine.begin() as conn:
        await conn.execute(text("DROP TRIGGER stall_commit ON turn_cursors"))
    assert (await store.turn(cursor.id)).status is TurnStatus.RUNNING
