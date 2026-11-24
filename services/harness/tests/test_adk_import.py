"""`harness.adk_import`: Studio conversations from ADK ledger sessions into the
harness's own tables.

`fixtures/adk_import/ledger_sessions.json` holds ADK `sessions` rows that the
studio routes of the last release storing conversations in ADK sessions
wrote through `DatabaseSessionService` on Postgres, run with the scripted
model: `conv_01adkfull000` (a card, a pin through `/composition`, an approved
plan that ran, a plain answer, and a fourth turn whose approved plan ran and
then paused on a question, its turn session holding the block of the source
the plan used), `conv_01adkplan000` (paused on a plan card) and
`conv_01adktext000` (two plain turns). Turn sessions keep only their
`orchestrator:specs`, the one key the job reads. `expected` holds what that
release's `StudioStore.load` returned for each, its run-ledger items, and,
for an open turn, the conversation after `close_turn(error="abandoned")` and
`pending.clear()`.

Each test loads the rows into ADK's own tables on a schema of its own.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import common.logging as common_logging
import pytest
from compiler.sources import source_from_spec_block
from google.adk.events import Event, EventActions
from google.adk.sessions import DatabaseSessionService
from harness import adk_import
from harness.adk_import import (
    IMPORTED_SEQ_MARGIN,
    LEDGER_APP,
    AdkSchemaError,
    AdkTablesMissing,
    NamedTwice,
    NextReleaseMissing,
    TableBusy,
    TablesMissing,
    UnknownConversation,
    _stored_digest,
    _stored_runs,
    close_open_turns,
    cutover_gate,
    import_conversations,
    main,
    open_turn_specs,
    reimport_on_access,
    rollback_turn_sessions,
    verify,
)
from harness.conversation import current_conversation_key
from harness.db.engine import DatabaseHandle
from harness.db.migrate import apply_pending
from harness.graph.state import AGENT_NAME, ORCHESTRATOR_SPECS_KEY, store_search
from harness.run_ledger import StudioRunLedger
from harness.studio import _keep_turn_records
from harness.studio_state import ConversationState, SegmentLost
from harness.studio_store import (
    ConversationHeld,
    ConversationStale,
    StudioStore,
    _serialize_pending,
)
from harness.tool_contracts import resolver_digest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .orchestrator_fakes import (
    AUTH,
    orchestrator_app,
    request_text,
    stream,
    studio_client,
    turn_body,
)

RECORDED = json.loads(
    (Path(__file__).parent / "fixtures" / "adk_import" / "ledger_sessions.json").read_text()
)
EXPECTED: dict[str, Any] = RECORDED["expected"]
FULL = "conv_01adkfull000"
PLAN_PAUSED = "conv_01adkplan000"
TEXT_ONLY = "conv_01adktext000"
PRINCIPAL = next(row["user_id"] for row in RECORDED["sessions"])
# Readings `close_turn` derives carry the wall-clock time they were read at.
_VOLATILE = "observed_at"


def _name(conversation_id: str) -> str:
    return f"{PRINCIPAL}/{conversation_id}"


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k != _VOLATILE}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _view(state: ConversationState) -> dict[str, Any]:
    """The fields the fixture's `expected` views hold, read the same way."""
    segment = state.segment
    data = {
        "turn_index": state.turn_index,
        "seq": state.seq,
        "trace_id": state.trace_id,
        "pending": _serialize_pending(state.pending),
        "plan_version_ids": {str(k): v for k, v in state.plan_version_ids.items()},
        "latest_plan_version": state.latest_plan_version,
        "composition": state.composition.model_dump(mode="json") if state.composition else None,
        "segment": None
        if segment is None
        else [segment.id, segment.owner, segment.expires_at, segment.cancel],
        "auto_add": state.auto_add,
        "timed_out_calls": sorted(state.timed_out_calls),
        "turn_user_message": state.turn_user_message,
        "turn_user_words": state.turn_user_words,
        "turn_final_text": state.turn_final_text,
        "turn_components": state.turn_components,
        "turn_recorded": state.turn_recorded,
        "turn_plans": state.turn_plans,
        "turn_readings": state.turn_readings,
        "turn_applied_choice": state.turn_applied_choice,
        "turn_choice_baseline": state.turn_choice_baseline,
        "turns": state.turns,
        "components": state.components,
        "component_order": list(state.components),
        "write_tokens": state.write_tokens,
        "approvals": [a.model_dump(mode="json") for a in state.approvals],
        "turn_specs": state.turn_specs,
    }
    return json.loads(json.dumps(data))


def _store(engine: AsyncEngine) -> StudioStore:
    """A store over the tables as the job builds one: the reader the server
    serves, and a bridge that converts what it reads."""
    return adk_import._store(engine)


async def _load(engine: AsyncEngine, conversation_id: str) -> ConversationState | None:
    return await _store(engine).load(PRINCIPAL, conversation_id)


async def _write_adk(engine: AsyncEngine, sessions: list[dict[str, Any]]) -> None:
    """ADK's tables, created by ADK, holding `sessions`."""
    await DatabaseSessionService(db_engine=engine).prepare_tables()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO sessions (app_name, user_id, id, state, create_time, update_time) "
                "VALUES (:app_name, :user_id, :id, CAST(:state AS jsonb), now(), now())"
            ),
            [{**row, "state": json.dumps(row["state"])} for row in sessions],
        )


def _sessions(**ledger_changes: Any) -> list[dict[str, Any]]:
    """The fixture's rows, with each named conversation's ledger state
    replaced by what its callable returns for a copy of it."""
    rows = copy.deepcopy(RECORDED["sessions"])
    for row in rows:
        if row["app_name"] == LEDGER_APP and row["id"] in ledger_changes:
            row["state"] = ledger_changes[row["id"]](row["state"])
    return rows


async def _set_adk_state(
    engine: AsyncEngine, conversation_id: str, changes: dict[str, Any]
) -> None:
    """A write the previous release makes to a ledger session."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE sessions SET state = state || CAST(:changes AS jsonb), update_time = now() "
                "WHERE app_name = :app AND id = :id"
            ),
            {"changes": json.dumps(changes), "app": LEDGER_APP, "id": conversation_id},
        )


async def _row_id(engine: AsyncEngine, conversation_id: str) -> int:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text("SELECT id FROM conversations WHERE conversation_id = :id"),
                {"id": conversation_id},
            )
        ).scalar_one()


async def _count(engine: AsyncEngine) -> int:
    async with engine.connect() as conn:
        return (await conn.execute(text("SELECT count(*) FROM conversations"))).scalar_one()


# The columns of `turn_cursors` that `close-open-turns` reads. The next
# release's first migration creates the table, and `--apply` refuses until it
# exists.
_TURN_CURSORS = (
    "CREATE TABLE IF NOT EXISTS turn_cursors (id bigint GENERATED ALWAYS AS IDENTITY "
    "PRIMARY KEY, conversation bigint NOT NULL REFERENCES conversations (id) "
    "ON DELETE CASCADE, turn_index integer NOT NULL, status text NOT NULL, "
    "UNIQUE (conversation, turn_index))"
)


async def _next_release_started(engine: AsyncEngine) -> None:
    """The `turn_cursors` table the next release's first migration creates."""
    async with engine.begin() as conn:
        await conn.execute(text(_TURN_CURSORS))


@pytest.fixture
async def adk(pg_empty_engine: AsyncEngine) -> AsyncEngine:
    """A schema holding the recorded ADK rows and no harness tables."""
    await _write_adk(pg_empty_engine, RECORDED["sessions"])
    return pg_empty_engine


async def _assert_runs_equal(engine: AsyncEngine, conversation_id: str) -> None:
    items = EXPECTED[conversation_id]["run_items"]
    token = current_conversation_key.set((PRINCIPAL, conversation_id))
    try:
        ledger = StudioRunLedger(_store(engine))
        for key, raw in items.items():
            kind, _, ident = key.removeprefix("studio:").partition(":")
            if kind == "run":
                run = await ledger.get(ident)
                assert run is not None and run.model_dump(mode="json") == raw
            elif kind == "run_submission":
                assert await ledger.submission(ident) == raw
            elif kind == "run_retrievals":
                assert await ledger.retrievals(ident) == raw
            elif kind == "run_plan":
                assert await ledger.plan(ident) == raw
            else:
                async with engine.connect() as conn:
                    run_id = (
                        await conn.execute(
                            text("SELECT run_id FROM run_keys WHERE idempotency_key = :key"),
                            {"key": ident},
                        )
                    ).scalar_one()
                assert run_id == raw
    finally:
        current_conversation_key.reset(token)


async def test_a_dry_run_writes_nothing_and_apply_copies_what_adk_loaded(
    adk: AsyncEngine,
) -> None:
    every = sorted(_name(c) for c in EXPECTED)

    dry = await import_conversations(adk, apply=False)

    assert (sorted(dry.imported), dry.clean, dry.findings) == (every, True, {})
    assert await _count(adk) == 0

    applied = await import_conversations(adk, apply=True)

    assert (sorted(applied.imported), applied.clean, applied.findings) == (every, True, {})
    for conversation_id, expected in EXPECTED.items():
        state = await _load(adk, conversation_id)
        assert state is not None
        assert _view(state) == expected["view"], conversation_id
        await _assert_runs_equal(adk, conversation_id)
    # A resumed pause numbers its frames past any the previous release's
    # dead stream could have sent after its last save.
    async with adk.connect() as conn:
        rows = (await conn.execute(text("SELECT seq_next, seq_reserved FROM conversations"))).all()
    assert all(reserved == seq_next + IMPORTED_SEQ_MARGIN for seq_next, reserved in rows)
    # The pauses and the open turns' spec blocks came across.
    assert EXPECTED[FULL]["view"]["pending"] and EXPECTED[PLAN_PAUSED]["view"]["pending"]
    assert EXPECTED[FULL]["view"]["turn_specs"]
    assert EXPECTED[FULL]["run_items"]


async def _add_columns(engine: AsyncEngine) -> None:
    """Add a nullable column to the conversations table and to a child
    table, as a later release's migration does."""
    async with engine.begin() as conn:
        await conn.execute(text("ALTER TABLE conversations ADD COLUMN later_col text"))
        await conn.execute(text("ALTER TABLE conversation_turns ADD COLUMN later_col text"))


_LATER_COLUMNS = pytest.mark.parametrize(
    "later_columns", [False, True], ids=["schema-as-imported", "columns-added-since"]
)


@_LATER_COLUMNS
async def test_verify_counts_equal_conversations_and_names_each_difference(
    adk: AsyncEngine, later_columns: bool
) -> None:
    await import_conversations(adk, apply=True)
    if later_columns:
        await _add_columns(adk)

    assert (await verify(adk)).render() == "3 of 3 equal"

    async with adk.begin() as conn:
        await conn.execute(
            text(
                'UPDATE conversation_turns SET record = record || \'{"assistant": "edited"}\' '
                "WHERE position = 0 AND conversation = "
                "(SELECT id FROM conversations WHERE conversation_id = :full)"
            ),
            {"full": FULL},
        )
        await conn.execute(
            text("UPDATE conversations SET composition = NULL WHERE conversation_id = :full"),
            {"full": FULL},
        )
        await conn.execute(
            text("DELETE FROM conversations WHERE conversation_id = :text_only"),
            {"text_only": TEXT_ONLY},
        )
        # The edited rows are what the import wrote, so verify compares them.
        pk = (
            await conn.execute(
                text("SELECT id FROM conversations WHERE conversation_id = :full"), {"full": FULL}
            )
        ).scalar_one()
        await conn.execute(
            text(
                "UPDATE adk_ledger_imports SET imported_digest = :digest WHERE conversation_id = :full"
            ),
            {"digest": await _stored_digest(conn, pk), "full": FULL},
        )
    report = await verify(adk)

    assert EXPECTED[FULL]["view"]["composition"] is not None
    assert report.differences == {
        _name(FULL): ["composition", "turn:0"],
        _name(TEXT_ONLY): ["not in the tables"],
    }
    assert report.render().splitlines()[-1] == "1 of 3 equal"


async def test_verify_reads_adks_side_as_the_previous_release_did_so_a_mapping_error_differs(
    adk: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`read_ledger` maps one key to a wrong value. The import writes it, and
    verify, whose ADK side loads the ledger session as the previous release
    did, names the key as a difference."""
    read = adk_import.read_ledger

    def mismapped(state: dict[str, Any]) -> Any:
        ledger = read(state)
        ledger.flat["auto_add"] = not ledger.flat["auto_add"]
        return ledger

    monkeypatch.setattr(adk_import, "read_ledger", mismapped)
    await import_conversations(adk, apply=True)

    report = await verify(adk)

    assert report.differences == {_name(c): ["auto_add"] for c in sorted(EXPECTED)}
    assert not report.clean


async def _add_turn_sessions(engine: AsyncEngine, principal: str, ids: list[str]) -> None:
    """ADK turn sessions, as a release that runs turns on ADK creates them."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO sessions (app_name, user_id, id, state, create_time, update_time) "
                "VALUES (:app, :principal, :id, '{}'::jsonb, now(), now())"
            ),
            [{"app": AGENT_NAME, "principal": principal, "id": i} for i in ids],
        )


async def _turn_session_ids(engine: AsyncEngine) -> list[tuple[str, str]]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT user_id, id FROM sessions WHERE app_name = :app ORDER BY user_id, id"),
            {"app": AGENT_NAME},
        )
        return [tuple(row) for row in rows]


async def test_rollback_turn_sessions_deletes_each_turn_session_above_the_ledgers_turn_index(
    adk: AsyncEngine,
) -> None:
    """This release served turns 5 and 6 of a conversation whose ledger
    session ADK left at turn 4. Before the previous release serves again,
    the job deletes those turn sessions, and only those: not a turn session
    at or below the index, one whose number is not a turn number, or one the
    previous release claimed since (its index is written first). It deletes
    every turn session of a conversation with no ledger session, as one
    started on this release, whatever its id begins with or whose it is."""
    await import_conversations(adk, apply=True)
    ahead = [f"{FULL}#t5", f"{FULL}#t6"]
    await _add_turn_sessions(
        adk,
        PRINCIPAL,
        [*ahead, f"{FULL}#tx", f"{FULL}#t12345678901", f"{FULL}0#t10", f"{FULL}0#t9", "c#t#t1"],
    )
    await _add_turn_sessions(adk, "u_other", [f"{FULL}#t7"])
    before = await _turn_session_ids(adk)
    started_here = {
        f"{PRINCIPAL}/c#t": ["c#t#t1"],
        f"{PRINCIPAL}/{FULL}0": [f"{FULL}0#t9", f"{FULL}0#t10"],
        f"u_other/{FULL}": [f"{FULL}#t7"],
    }

    dry = await rollback_turn_sessions(adk, apply=False)

    assert (dry.ahead, dry.clean) == ({_name(FULL): ahead, **started_here}, False)
    assert await _turn_session_ids(adk) == before

    # The previous release claimed turn 5, closing the pause it held.
    await _set_adk_state(adk, FULL, {"studio:turn_index": 5, "studio:pending": {}})
    applied = await rollback_turn_sessions(adk, apply=True)

    assert (applied.ahead, applied.clean) == ({_name(FULL): [f"{FULL}#t6"], **started_here}, False)
    assert applied.render().splitlines()[-1].endswith("run again until none is found")
    deleted = {f"{FULL}#t6", *(i for ids in started_here.values() for i in ids)}
    assert await _turn_session_ids(adk) == [row for row in before if row[1] not in deleted]
    assert (await rollback_turn_sessions(adk, apply=True)).clean

    await _set_adk_state(adk, PLAN_PAUSED, {"studio:turn_index": "1"})
    unreadable = await rollback_turn_sessions(adk, apply=True)
    assert (unreadable.unreadable, unreadable.clean) == ([_name(PLAN_PAUSED)], True)

    # A stream of this release holds a conversation, with a ledger session or
    # without: `--apply` leaves both alone until the lease lapses.
    await _add_turn_sessions(adk, PRINCIPAL, [f"{TEXT_ONLY}#t9", "conv_late#t1"])
    async with adk.begin() as conn:
        await conn.execute(
            text("INSERT INTO conversations (principal, conversation_id) VALUES (:p, 'conv_late')"),
            {"p": PRINCIPAL},
        )
    await _lease(adk, TEXT_ONLY, 600)
    await _lease(adk, "conv_late", 600)
    streaming = await rollback_turn_sessions(adk, apply=True)
    assert (streaming.ahead, streaming.exit_status) == ({}, 5)
    assert sorted(streaming.live) == sorted([_name(TEXT_ONLY), f"{PRINCIPAL}/conv_late"])
    assert {(PRINCIPAL, f"{TEXT_ONLY}#t9"), (PRINCIPAL, "conv_late#t1")} <= set(
        await _turn_session_ids(adk)
    )
    await _lease(adk, TEXT_ONLY, -1)
    await _lease(adk, "conv_late", -1)
    lapsed = await rollback_turn_sessions(adk, apply=True)
    assert (lapsed.ahead, lapsed.live, lapsed.exit_status) == (
        {_name(TEXT_ONLY): [f"{TEXT_ONLY}#t9"], f"{PRINCIPAL}/conv_late": ["conv_late#t1"]},
        [],
        1,
    )
    assert (await rollback_turn_sessions(adk, apply=True)).exit_status == 0


async def _ledger_pending(engine: AsyncEngine, conversation_id: str) -> Any:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT state -> 'studio:pending' FROM sessions WHERE app_name = :app "
                    "AND id = :id"
                ),
                {"app": LEDGER_APP, "id": conversation_id},
            )
        ).scalar_one()


async def test_rollback_clears_a_pause_this_release_resumed_and_keeps_one_it_did_not(
    adk: AsyncEngine,
) -> None:
    """ADK holds FULL and PLAN_PAUSED paused on a card. This release resumed
    FULL's pause, so its row no longer holds that interrupt: the rollback
    clears the interrupt on ADK, so the previous release shows no card and
    cannot resume the turn a second time. PLAN_PAUSED's pause stands."""
    await import_conversations(adk, apply=True)
    paused = await _ledger_pending(adk, PLAN_PAUSED)
    async with adk.begin() as conn:
        await conn.execute(
            text(
                "UPDATE conversations SET turn_state = turn_state || '{\"pending\": {}}'::jsonb "
                "WHERE conversation_id = :id"
            ),
            {"id": FULL},
        )

    dry = await rollback_turn_sessions(adk, apply=False)
    assert (dry.cleared, dry.exit_status) == ({_name(FULL): 4}, 1)
    assert await _ledger_pending(adk, FULL)

    applied = await rollback_turn_sessions(adk, apply=True)

    assert applied.cleared == {_name(FULL): 4}
    assert await _ledger_pending(adk, FULL) == {}
    assert await _ledger_pending(adk, PLAN_PAUSED) == paused
    assert (await rollback_turn_sessions(adk, apply=True)).exit_status == 0


async def test_rollback_leaves_a_card_the_previous_release_pauses_on_after_the_move(
    adk: AsyncEngine,
) -> None:
    """After traffic moves back, the previous release pauses TEXT_ONLY on a
    card at turn 3. Its ledger session moved since the import, so the
    rollback reads the pause as the previous release's own and leaves it."""
    await import_conversations(adk, apply=True)
    assert (await rollback_turn_sessions(adk, apply=False)).cleared == {}
    card = await _ledger_pending(adk, PLAN_PAUSED)
    await _set_adk_state(adk, TEXT_ONLY, {"studio:turn_index": 3, "studio:pending": card})

    applied = await rollback_turn_sessions(adk, apply=True)

    assert (applied.cleared, applied.exit_status) == ({}, 0)
    assert await _ledger_pending(adk, TEXT_ONLY) == card


async def test_rollback_reruns_a_conversation_whose_ledger_session_appears_mid_run(
    adk: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The previous release creates a conversation's ledger session between
    the run's read of the turn sessions with no ledger session and their
    delete: the delete takes none of them, and the run lists the
    conversation and fails, so the next run reads it through its ledger."""
    # Read TEXT_ONLY's turn 2 as unledgered, as a read before its ledger
    # session appeared would.
    monkeypatch.setattr(
        adk_import,
        "_TURN_SESSIONS_UNLEDGERED",
        text(
            f"SELECT t.user_id, {adk_import._TURN_CONVERSATION} AS conversation_id, t.id "
            f"FROM sessions t WHERE t.app_name = :app AND t.id = '{TEXT_ONLY}#t2'"
        ),
    )

    applied = await rollback_turn_sessions(adk, apply=True)

    assert (applied.moved, applied.exit_status) == ([_name(TEXT_ONLY)], 1)
    assert (PRINCIPAL, f"{TEXT_ONLY}#t2") in await _turn_session_ids(adk)


@_LATER_COLUMNS
async def test_a_rerun_leaves_unmoved_conversations_and_reimports_what_only_adk_changed(
    adk: AsyncEngine, later_columns: bool
) -> None:
    await import_conversations(adk, apply=True)
    if later_columns:
        await _add_columns(adk)
    again = await import_conversations(adk, apply=True)
    assert (sorted(again.unchanged), again.imported, again.reimported) == (
        sorted(_name(c) for c in EXPECTED),
        [],
        [],
    )

    # The previous release writes one conversation during the drain. The new
    # release writes another one, which ADK did not change.
    await _set_adk_state(adk, TEXT_ONLY, {"studio:seq": 99})
    row_id = await _row_id(adk, TEXT_ONLY)
    store = _store(adk)
    plan_paused = await store.load(PRINCIPAL, PLAN_PAUSED)
    assert plan_paused is not None
    plan_paused.auto_add = not plan_paused.auto_add
    await store.save(PRINCIPAL, PLAN_PAUSED)

    rerun = await import_conversations(adk, apply=True)

    assert (rerun.reimported, rerun.clean) == ([_name(TEXT_ONLY)], True)
    assert sorted(rerun.unchanged) == [_name(FULL), _name(PLAN_PAUSED)]
    text_only = await _load(adk, TEXT_ONLY)
    assert text_only is not None and text_only.seq == 99
    # Rewritten in place: an instance holding the row's id still finds it.
    assert await _row_id(adk, TEXT_ONLY) == row_id
    kept = await _load(adk, PLAN_PAUSED)
    assert kept is not None and kept.auto_add == plan_paused.auto_add
    # Verify compares what the import wrote and sets the new release's
    # write apart.
    checked = await verify(adk)
    assert (checked.differences, checked.written_since_import) == ({}, [_name(PLAN_PAUSED)])
    assert checked.render().splitlines()[-1] == "2 of 2 equal"


async def test_a_conversation_changed_on_both_sides_is_a_conflict_until_adk_is_preferred(
    adk: AsyncEngine,
) -> None:
    await import_conversations(adk, apply=True)
    await _set_adk_state(adk, FULL, {"studio:seq": 99})
    store = _store(adk)
    full = await store.load(PRINCIPAL, FULL)
    assert full is not None
    full.auto_add = not full.auto_add
    await store.save(PRINCIPAL, FULL)

    conflict = await import_conversations(adk, apply=True)

    assert (conflict.conflicts, conflict.clean) == ([_name(FULL)], False)
    untouched = await _load(adk, FULL)
    assert untouched is not None and (untouched.seq, untouched.auto_add) == (
        EXPECTED[FULL]["view"]["seq"],
        full.auto_add,
    )

    # The name is the report's, so the same id under another principal names
    # no conversation, and stops the job before it writes.
    with pytest.raises(UnknownConversation, match=f"other/{FULL}"):
        await import_conversations(adk, apply=True, prefer_adk=frozenset({f"other/{FULL}"}))

    preferred = await import_conversations(adk, apply=True, prefer_adk=frozenset({_name(FULL)}))

    assert (preferred.reimported, preferred.clean) == ([_name(FULL)], True)
    reimported = await _load(adk, FULL)
    assert reimported is not None
    assert _view(reimported) == {**EXPECTED[FULL]["view"], "seq": 99}


def _drop(key: str) -> Any:
    return lambda state: {k: v for k, v in state.items() if k != key}


def _move(old: str, new: str) -> Any:
    return lambda state: {(new if k == old else k): v for k, v in state.items()}


def _add(**items: Any) -> Any:
    return lambda state: {**state, **items}


def _run_id() -> str:
    state = next(row["state"] for row in RECORDED["sessions"] if row["id"] == FULL)
    return next(k.removeprefix("studio:run:") for k in state if k.startswith("studio:run:"))


@pytest.mark.parametrize(
    ("change", "reasons"),
    [
        (_add(**{"studio:something_new": 1}), ["studio:something_new: a key this job"]),
        (
            _move("studio:turn:0001", "studio:turn:0005"),
            ["turn positions are not 0 to 2: [0, 2, 5]"],
        ),
        (_add(**{"studio:approval:first": {}}), ["studio:approval:first: the position is not"]),
        (_add(**{"studio:write_token:cmp_x": 7}), ["studio:write_token:cmp_x: the write token"]),
        (
            _drop(f"studio:run:{_run_id()}"),
            [
                f"the idempotency key of run {_run_id()} names a run",
                f"the submission of run {_run_id()} names a run",
                f"the retrievals of run {_run_id()} names a run",
            ],
        ),
        # A failure past the structural checks refuses the one conversation.
        (_add(**{"studio:segment": {"owner": "x"}}), ["the import failed: KeyError"]),
        # Values of a type the driver refuses when the write runs.
        (_add(**{"studio:seq": "7"}), ["the import failed: TypeError: seq is of type str"]),
        (
            _add(**{"studio:segment": {"id": "seg_x", "owner": 5, "expires_at": 1.0}}),
            ["the import failed: TypeError: segment owner is of type int"],
        ),
        (
            _add(**{"studio:segment": {"id": "seg_x", "owner": "x", "expires_at": "soon"}}),
            ["the import failed: TypeError: segment expiry is of type str"],
        ),
    ],
    ids=[
        "unknown-key",
        "position-gap",
        "position-not-a-number",
        "write-token",
        "dangling-run",
        "write-fails",
        "seq-not-an-int",
        "segment-owner-not-a-string",
        "segment-expiry-not-a-number",
    ],
)
async def test_a_structural_fault_refuses_the_conversation_by_every_reason_and_writes_none_of_it(
    pg_empty_engine: AsyncEngine, change: Any, reasons: list[str]
) -> None:
    await _write_adk(pg_empty_engine, _sessions(**{FULL: change}))

    # The dry run reports what the write refuses.
    dry = await import_conversations(pg_empty_engine, apply=False)
    report = await import_conversations(pg_empty_engine, apply=True)

    assert dry.refused == report.refused
    assert not report.clean and list(report.refused) == [_name(FULL)]
    refused = report.refused[_name(FULL)]
    assert len(refused) == len(reasons)
    assert all(any(r.startswith(reason) for r in refused) for reason in reasons), refused
    assert await _load(pg_empty_engine, FULL) is None
    assert sorted(report.imported) == [_name(PLAN_PAUSED), _name(TEXT_ONLY)]


async def test_records_the_turn_loader_skips_are_reported_and_copied_as_stored(
    pg_empty_engine: AsyncEngine,
) -> None:
    def unreadable_plan(state: dict[str, Any]) -> dict[str, Any]:
        state["studio:turn:0001"]["plans"].append({"not": "a plan record"})
        return state

    await _write_adk(pg_empty_engine, _sessions(**{FULL: unreadable_plan}))

    report = await import_conversations(pg_empty_engine, apply=True)

    assert report.clean and _name(FULL) in report.imported
    assert list(report.findings) == [_name(FULL)]
    assert report.findings[_name(FULL)][0].startswith("turn 1 plan 1: load skips it: not a")
    state = await _load(pg_empty_engine, FULL)
    assert state is not None and state.turns[1]["plans"][-1] == {"not": "a plan record"}
    assert (await verify(pg_empty_engine)).clean


async def test_the_close_open_turns_job_logs_each_close_as_a_json_studio_turn_line(
    adk: AsyncEngine, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`main`, run as a Cloud Run job execution runs it, brings logging and
    telemetry up as the service does, so each turn `close-open-turns` closes
    writes its `studio.turn` line as a JSON entry the analytics log model
    reads (`jsonPayload.event`), and the job flushes its outcomes, blocking,
    once its event loop has finished."""
    await import_conversations(adk, apply=True)
    await _next_release_started(adk)
    async with adk.connect() as conn:
        schema = (await conn.execute(text("SELECT current_schema()"))).scalar_one()
    url = adk.url

    def build_database(*, settings: Any) -> DatabaseHandle:
        # A fresh engine, built on the job's own event loop.
        engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
        return DatabaseHandle(engine=engine, aclose=engine.dispose)

    telemetry: list[str] = []

    def flush() -> None:
        # Once the job's event loop has finished.
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        telemetry.append("flush")

    monkeypatch.setattr(adk_import, "build_database", build_database)
    monkeypatch.setattr(adk_import, "init_telemetry", lambda **_: telemetry.append("init"))
    monkeypatch.setattr(adk_import, "flush_telemetry_before_exit", flush)
    monkeypatch.setenv("CLOUD_RUN_JOB", "ai-adk-import")
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    # `configure_logging` replaces the root handlers: put pytest's back after.
    monkeypatch.setattr(common_logging, "_configured", False)
    monkeypatch.setattr(logging.root, "handlers", list(logging.root.handlers))
    monkeypatch.setattr(logging.root, "level", logging.root.level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        monkeypatch.setattr(uvicorn_logger, "handlers", list(uvicorn_logger.handlers))
        monkeypatch.setattr(uvicorn_logger, "propagate", uvicorn_logger.propagate)
    otel_context = logging.getLogger("opentelemetry.context")
    monkeypatch.setattr(otel_context, "level", otel_context.level)

    status = await asyncio.to_thread(adk_import.main, ["close-open-turns", "--apply"])

    assert status == 0
    entries = [
        json.loads(line) for line in capsys.readouterr().err.splitlines() if line.startswith("{")
    ]
    lines = [entry for entry in entries if entry.get("event") == "studio.turn"]
    assert sorted((entry["status"], entry["error"]) for entry in lines) == [
        ("closed", "abandoned"),
        ("closed", "abandoned"),
    ]
    assert telemetry == ["init", "flush"]


async def test_close_open_turns_closes_each_paused_adk_turn_as_abandoned_with_its_blocks(
    adk: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each open turn's session row is read once, by the store's load.
    `--apply` refuses until the next release has created `turn_cursors`,
    since this release can still answer a turn paused on ADK until then."""
    await import_conversations(adk, apply=True)
    with pytest.raises(NextReleaseMissing, match="turn_cursors"):
        await close_open_turns(adk, apply=True)
    before = await _load(adk, PLAN_PAUSED)
    assert before is not None and not before.turn_recorded and before.pending
    await _next_release_started(adk)
    open_turns = [f"{_name(FULL)} turn 4", f"{_name(PLAN_PAUSED)} turn 1"]
    reads: list[str] = []
    read_row = adk_import._stored_turn_specs

    async def counted(conn: Any, principal: str, session_id: str) -> Any:
        reads.append(session_id)
        return await read_row(conn, principal, session_id)

    monkeypatch.setattr(adk_import, "_stored_turn_specs", counted)

    dry = await close_open_turns(adk, apply=False)

    assert sorted(reads) == sorted(set(reads)) and len(reads) == len(open_turns)
    monkeypatch.setattr(adk_import, "_stored_turn_specs", read_row)

    assert (dry.closed, dry.live_on_adk, dry.not_on_adk, dry.changed) == (open_turns, [], [], [])
    still_open = await _load(adk, FULL)
    assert still_open is not None and not still_open.turn_recorded

    applied = await close_open_turns(adk, apply=True)

    assert (applied.closed, applied.clean) == (open_turns, True)
    for conversation_id in (FULL, PLAN_PAUSED):
        state = await _load(adk, conversation_id)
        assert state is not None
        # The close names its turn's trace and usage on its record
        # (`ConversationState.close_turn`), which the recorded view predates.
        view = _view(state)
        view["turns"] = [*view["turns"][:-1], dict(view["turns"][-1])]
        abandoned = view["turns"][-1]
        assert abandoned.pop("trace_id") == state.trace_id, conversation_id
        assert abandoned.pop("usage")["trace_id"] == state.trace_id, conversation_id
        want = {**EXPECTED[conversation_id]["closed_view"], "turn_specs": {}}
        assert _strip(view) == _strip(want), conversation_id
    closed = await _load(adk, FULL)
    assert closed is not None
    assert closed.turns[-1]["error"] == "abandoned"
    assert closed.turns[-1]["specs"] == EXPECTED[FULL]["view"]["turn_specs"]
    assert (await close_open_turns(adk, apply=True)).closed == []


async def _lease(engine: AsyncEngine, conversation_id: str, seconds: int) -> None:
    """Set a lease on the conversation's row that lapses `seconds` from now."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE conversations SET lease_segment_id = 'seg_live', lease_owner = 'other', "
                "lease_expires_at = extract(epoch FROM now()) + :seconds "
                "WHERE conversation_id = :id"
            ),
            {"id": conversation_id, "seconds": seconds},
        )


async def test_close_open_turns_leaves_a_turn_a_stream_holds_or_that_never_ran_on_adk(
    adk: AsyncEngine,
) -> None:
    """A stream on an ADK turn fails the run until it ends; a stream on a
    turn with no ADK turn session, or one `turn_cursors` holds, does not."""
    await import_conversations(adk, apply=True)
    await _next_release_started(adk)
    await _lease(adk, PLAN_PAUSED, 600)
    async with adk.begin() as conn:
        await conn.execute(
            text("DELETE FROM sessions WHERE app_name = :app AND id = :id"),
            {"app": AGENT_NAME, "id": f"{FULL}#t4"},
        )
    plan_turn, full_turn = f"{_name(PLAN_PAUSED)} turn 1", f"{_name(FULL)} turn 4"

    report = await close_open_turns(adk, apply=True)

    assert (report.clean, report.closed, report.live_on_adk, report.not_on_adk) == (
        False,
        [],
        [plan_turn],
        [full_turn],
    )
    assert f"left open, a stream on ADK holds its lease, run again once it ends: {plan_turn}" in (
        report.render()
    )
    await _lease(adk, FULL, 600)
    async with adk.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO turn_cursors (conversation, turn_index, status) "
                "VALUES (:pk, 1, 'running')"
            ),
            {"pk": await _row_id(adk, PLAN_PAUSED)},
        )

    elsewhere = await close_open_turns(adk, apply=True)

    assert (elsewhere.clean, elsewhere.live_on_adk, sorted(elsewhere.live_not_adk)) == (
        True,
        [],
        sorted([plan_turn, full_turn]),
    )
    for conversation_id in (FULL, PLAN_PAUSED):
        state = await _load(adk, conversation_id)
        assert state is not None and not state.turn_recorded


@pytest.mark.parametrize(
    ("adk_write", "bucket"),
    [
        (
            {
                "studio:segment": {
                    "id": "seg_old",
                    "owner": "old",
                    "expires_at": time.time() + 600,
                    "cancel": False,
                }
            },
            "live_on_adk",
        ),
        ({"studio:seq": 99}, "stale_import"),
    ],
    ids=["adk-lease-live", "adk-wrote-since-import"],
)
async def test_close_open_turns_leaves_a_conversation_adk_streams_or_wrote_since_the_import(
    adk: AsyncEngine, adk_write: dict[str, Any], bucket: str
) -> None:
    """A stream of the previous release that started after the import writes
    its lease to ADK's ledger session, and the tables hold none. A
    conversation whose ledger session holds a live lease, or moved since its
    import, is left open and fails the run, dry or applied. A live lease is
    listed as live, whether or not the import is current."""
    await import_conversations(adk, apply=True)
    await _next_release_started(adk)
    await _set_adk_state(adk, PLAN_PAUSED, adk_write)

    for apply in (False, True):
        report = await close_open_turns(adk, apply=apply)

        assert (report.closed, report.clean) == ([f"{_name(FULL)} turn 4"], False)
        assert report.live_on_adk + report.stale_import == [_name(PLAN_PAUSED)]
        assert getattr(report, bucket) == [_name(PLAN_PAUSED)]
    state = await _load(adk, PLAN_PAUSED)
    assert state is not None and not state.turn_recorded


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE adk_internal_metadata SET value = '0'",
        "DROP TABLE adk_internal_metadata",
    ],
    ids=["other-version", "no-metadata"],
)
async def test_an_adk_schema_the_job_does_not_read_stops_it_before_any_write(
    adk: AsyncEngine, statement: str
) -> None:
    async with adk.begin() as conn:
        await conn.execute(text(statement))

    with pytest.raises(AdkSchemaError, match="schema version 1"):
        await import_conversations(adk, apply=True)

    async with adk.connect() as conn:
        assert (await conn.execute(text("SELECT to_regclass('conversations')"))).scalar() is None


async def test_the_command_exits_non_zero_after_importing_the_rest_when_one_is_refused(
    pg_database_url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = create_async_engine(pg_database_url)
    try:
        await _write_adk(engine, _sessions(**{TEXT_ONLY: _add(**{"studio:unknown": 1})}))
        monkeypatch.setenv("DATABASE_URL", pg_database_url)

        status = await asyncio.to_thread(main, ["import", "--apply"])

        assert status == 1
        out = capsys.readouterr().out
        assert f"refused: {_name(TEXT_ONLY)}\n  studio:unknown: a key this job does not know" in out
        assert (
            out.splitlines()[-1]
            == "2 imported, 0 re-imported, 0 unchanged, 0 conflicts, 0 kept, 0 discarded, "
            "1 refused"
        )
        assert await _count(engine) == 2
        assert await asyncio.to_thread(main, ["import", "--verify"]) == 1
        assert capsys.readouterr().out.splitlines()[-1] == "2 of 3 equal"
        assert await asyncio.to_thread(main, ["close-open-turns", "--apply"]) == 3
        assert "no turn_cursors table" in capsys.readouterr().err
        # The release that archives ADK's tables has started.
        async with engine.begin() as conn:
            await conn.execute(
                text("ALTER TABLE adk_ledger_imports RENAME TO adk_archive_adk_ledger_imports")
            )
        for verb in (["close-open-turns", "--apply"], ["import", "--verify"]):
            assert await asyncio.to_thread(main, verb) == 4
            assert "adk_archive_adk_ledger_imports resolves" in capsys.readouterr().err
    finally:
        await engine.dispose()


async def test_a_write_the_serving_release_makes_during_a_reimport_is_a_conflict_not_lost(
    adk: AsyncEngine,
) -> None:
    """The serving release's write holds the row lock when the re-import
    reaches the conversation. The import waits for it and then sees the
    tables changed; read before the lock, it would delete the write."""
    await import_conversations(adk, apply=True)
    await _set_adk_state(adk, TEXT_ONLY, {"studio:seq": 99})
    # The writer and the poll run beside the job's own pool of two.
    async with adk.connect() as conn:
        schema = (await conn.execute(text("SELECT current_schema()"))).scalar_one()
    side = create_async_engine(adk.url, connect_args={"server_settings": {"search_path": schema}})
    try:
        async with side.connect() as writer:
            transaction = await writer.begin()
            await writer.execute(
                text("UPDATE conversations SET version = version + 1 WHERE conversation_id = :id"),
                {"id": TEXT_ONLY},
            )
            rerun = asyncio.create_task(import_conversations(adk, apply=True))
            async with asyncio.timeout(10):
                while not await _waiting_on_a_lock(side):
                    await asyncio.sleep(0.02)
            await transaction.commit()
    finally:
        await side.dispose()
    report = await rerun

    assert (report.conflicts, report.reimported) == ([_name(TEXT_ONLY)], [])
    kept = await _load(adk, TEXT_ONLY)
    assert kept is not None and kept.seq == EXPECTED[TEXT_ONLY]["view"]["seq"]


async def _waiting_on_a_lock(engine: AsyncEngine) -> bool:
    async with engine.connect() as conn:
        return bool(
            (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE datname = current_database() AND wait_event_type = 'Lock'"
                    )
                )
            ).scalar_one()
        )


async def test_verify_and_close_open_turns_neither_migrate_nor_use_adks_session_service(
    adk: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    for job in (verify(adk), close_open_turns(adk, apply=False)):
        with pytest.raises(TablesMissing):
            await job
    async with adk.connect() as conn:
        assert (await conn.execute(text("SELECT to_regclass('conversations')"))).scalar() is None

    await import_conversations(adk, apply=True)

    async def no_service(*_: Any, **__: Any) -> Any:
        raise AssertionError("the job read a session through ADK's session service")

    monkeypatch.setattr(DatabaseSessionService, "get_session", no_service)

    assert (await verify(adk)).render() == "3 of 3 equal"
    assert (await close_open_turns(adk, apply=False)).closed == [
        f"{_name(FULL)} turn 4",
        f"{_name(PLAN_PAUSED)} turn 1",
    ]


async def test_the_report_gives_the_size_distribution_of_ledger_states_and_open_turn_specs(
    adk: AsyncEngine,
) -> None:
    async with adk.connect() as conn:
        largest = (
            await conn.execute(
                text("SELECT max(octet_length(state::text)) FROM sessions WHERE app_name = :app"),
                {"app": LEDGER_APP},
            )
        ).scalar_one()
        specs = (
            await conn.execute(
                text("SELECT octet_length((state -> :key)::text) FROM sessions WHERE id = :id"),
                {"key": ORCHESTRATOR_SPECS_KEY, "id": f"{FULL}#t4"},
            )
        ).scalar_one()

    lines = (await import_conversations(adk, apply=False)).render().splitlines()

    assert lines[-3].startswith("ledger state: p50 ")
    assert lines[-3].endswith(f", max {largest} bytes over 3")
    # The plan-paused turn's session holds no specs.
    assert lines[-2] == (
        f"open turn orchestrator:specs: p50 0, p95 {specs}, max {specs} bytes over 2"
    )


_G7_PART = json.loads(
    (Path(__file__).parent / "fixtures" / "search_part_g7_africa.json").read_text()
)["envelope"]


def _list_block() -> dict[str, Any]:
    """The G7 part's spec block as `store_search` writes it, `_bound_refs` as
    `BoundEntity` rows."""
    state: dict[str, Any] = {}
    store_search(state, resolver_digest(_G7_PART))
    (block,) = state[ORCHESTRATOR_SPECS_KEY].values()
    return block


def _mapping_block() -> dict[str, Any]:
    """The same block with `_bound_refs` as the mapping kept before `BoundEntity`."""
    return {**_list_block(), "_bound_refs": {"g7": ["GroupOfSeven"], "africa": ["DACAfrica"]}}


def _with_turn_specs(block: dict[str, Any]) -> Any:
    def change(state: dict[str, Any]) -> dict[str, Any]:
        state["studio:turn:0000"]["specs"] = {block["spec_id"]: block}
        return state

    return change


def _rows(block: dict[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    return [(row.text, row.refs) for row in source_from_spec_block(block).bound_refs]


# FULL's run item that holds what a run retrieved.
_RUN_RETRIEVALS = "studio:run_retrievals:run_46328735f3a24a1ba44570e6c79dd119"


def _with_run_retrieval(block: dict[str, Any]) -> Any:
    def change(state: dict[str, Any]) -> dict[str, Any]:
        state[_RUN_RETRIEVALS] = [*state[_RUN_RETRIEVALS], block]
        return state

    return change


@pytest.mark.parametrize("held_in", ["turn", "run item"])
async def test_a_mapping_form_block_is_imported_as_rows_and_verifies_equal(
    pg_empty_engine: AsyncEngine, held_in: str
) -> None:
    mapping = _mapping_block()
    change = _with_turn_specs(mapping) if held_in == "turn" else _with_run_retrieval(mapping)
    await _write_adk(pg_empty_engine, _sessions(**{FULL: change}))

    report = await import_conversations(pg_empty_engine, apply=True)

    assert (report.clean, report.converted_blocks, report.unconvertible_blocks) == (True, 1, 0)
    assert "1 converted, 0 unconvertible" in report.render()
    if held_in == "turn":
        state = await _load(pg_empty_engine, FULL)
        assert state is not None
        stored = state.turns[0]["specs"][mapping["spec_id"]]
    else:
        stored = (await _stored_runs(pg_empty_engine, PRINCIPAL, FULL)).retrievals[
            _RUN_RETRIEVALS.removeprefix("studio:run_retrievals:")
        ][-1]
    assert _rows(stored) == [("g7", ("GroupOfSeven",)), ("africa", ("DACAfrica",))]
    verified = await verify(pg_empty_engine)
    assert (verified.differences, verified.clean) == ({}, True)


async def test_a_block_that_does_not_convert_is_reported_and_copied_as_stored(
    pg_empty_engine: AsyncEngine,
) -> None:
    """The load path drops such a block and reads the rest of the
    conversation, so the import keeps the conversation too."""
    block = {**_list_block(), "_bound_refs": "g7"}
    await _write_adk(pg_empty_engine, _sessions(**{FULL: _with_turn_specs(block)}))

    report = await import_conversations(pg_empty_engine, apply=True)

    assert (report.clean, report.unconvertible_blocks) == (True, 1)
    assert _name(FULL) in report.imported
    (finding,) = report.findings[_name(FULL)]
    assert finding.startswith("studio:turn:0000: the _bound_refs of spec block")
    assert finding.endswith("does not convert: TypeError: _bound_refs is of type str")
    state = await _load(pg_empty_engine, FULL)
    assert state is not None and state.turns[0]["specs"][block["spec_id"]] == block


async def _put_open_turn_block(engine: AsyncEngine, block: dict[str, Any]) -> None:
    """`block` in FULL's open turn session, as a release before
    `BoundEntity` stored it."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE sessions SET state = jsonb_set(state, ARRAY[:key, :spec_id], "
                "CAST(:block AS jsonb)) WHERE app_name = :app AND id = :id"
            ),
            {
                "key": ORCHESTRATOR_SPECS_KEY,
                "spec_id": block["spec_id"],
                "block": json.dumps(block),
                "app": AGENT_NAME,
                "id": f"{FULL}#t4",
            },
        )


async def test_a_paused_turn_s_mapping_form_block_is_served_converted_through_its_close(
    adk: AsyncEngine,
) -> None:
    """The store the server builds reads the open turn's block converted,
    and a resumed stream, which carries the turn session's blocks as stored,
    keeps them converted, so the turn closes with a block the loader reads."""
    spec_id = next(iter(EXPECTED[FULL]["view"]["turn_specs"]))
    await _put_open_turn_block(adk, {**_mapping_block(), "spec_id": spec_id})
    assert (await import_conversations(adk, apply=True)).clean
    assert (await verify(adk)).clean
    store = _reimporting_store(adk)

    state = await store.load(PRINCIPAL, FULL)
    assert state is not None
    source_from_spec_block(state.turn_specs[spec_id])
    async with adk.connect() as conn:
        stored = (
            await conn.execute(
                text("SELECT state -> :key FROM sessions WHERE app_name = :app AND id = :id"),
                {"key": ORCHESTRATOR_SPECS_KEY, "app": AGENT_NAME, "id": f"{FULL}#t4"},
            )
        ).scalar_one()
    resumed = Event(
        author="orchestrator", actions=EventActions(state_delta={ORCHESTRATOR_SPECS_KEY: stored})
    )
    _keep_turn_records(state, resumed, store)
    state.close_turn(final_text="Reading B.")
    await store.save(PRINCIPAL, FULL)

    closed = await _load(adk, FULL)
    assert closed is not None and closed.turn_recorded
    assert _rows(closed.turns[-1]["specs"][spec_id]) == [
        ("g7", ("GroupOfSeven",)),
        ("africa", ("DACAfrica",)),
    ]


async def test_a_served_block_that_does_not_convert_is_logged_once_with_its_turn_and_reason(
    adk: AsyncEngine, caplog: pytest.LogCaptureFixture
) -> None:
    """The block the serving release keeps as stored is named once by the
    store's bridge, with its turn session, spec id and why, however many
    loads and resumed streams read it."""
    spec_id = next(iter(EXPECTED[FULL]["view"]["turn_specs"]))
    block = {**_mapping_block(), "spec_id": spec_id, "_bound_refs": "g7"}
    await _put_open_turn_block(adk, block)
    await import_conversations(adk, apply=True)
    store = _reimporting_store(adk)

    with caplog.at_level("WARNING", logger="harness.adk_import"):
        for _ in range(2):
            state = await store.load(PRINCIPAL, FULL)
            assert state is not None
            resumed = Event(
                author="orchestrator",
                actions=EventActions(state_delta={ORCHESTRATOR_SPECS_KEY: {spec_id: block}}),
            )
            _keep_turn_records(state, resumed, store)

    (logged,) = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert f"{FULL}#t4: the _bound_refs of spec block {spec_id} does not convert" in logged
    assert logged.endswith("TypeError: _bound_refs is of type str")


async def test_a_load_converts_an_open_turn_s_blocks_once(
    adk: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader returns the blocks as stored and the bridge converts them,
    so each block is converted once per load."""
    spec_id = next(iter(EXPECTED[FULL]["view"]["turn_specs"]))
    await _put_open_turn_block(adk, {**_mapping_block(), "spec_id": spec_id})
    await import_conversations(adk, apply=True)
    store = _reimporting_store(adk)
    converted: list[str] = []
    convert = adk_import._convert_block

    def counting(block: Any) -> Any:
        converted.append(block.get("spec_id"))
        return convert(block)

    monkeypatch.setattr(adk_import, "_convert_block", counting)

    state = await store.load(PRINCIPAL, FULL)

    assert state is not None
    assert converted.count(spec_id) == 1
    source_from_spec_block(state.turn_specs[spec_id])


def test_a_bridge_remembers_the_blocks_it_logged_within_its_bound(
    adk: AsyncEngine, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each bridge keeps its own memo of the blocks it logged, holding the
    latest `_LOGGED_BLOCKS_KEPT`, so a block pushed out by newer ones is
    logged again."""
    monkeypatch.setattr(adk_import, "_LOGGED_BLOCKS_KEPT", 1)
    first, second = reimport_on_access(adk), reimport_on_access(adk)

    def block(spec_id: str) -> dict[str, Any]:
        return {spec_id: {**_mapping_block(), "spec_id": spec_id, "_bound_refs": "g7"}}

    with caplog.at_level("WARNING", logger="harness.adk_import"):
        for bridge, spec_id in [
            (first, "spec_a"),
            (first, "spec_a"),
            (second, "spec_a"),
            (first, "spec_b"),
            (first, "spec_a"),
        ]:
            bridge.convert_specs(f"{FULL}#t4", block(spec_id))

    logged = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert [m.split("_bound_refs of spec block ")[1].split(" ")[0] for m in logged] == [
        "spec_a",
        "spec_a",
        "spec_b",
        "spec_a",
    ]


async def test_close_open_turns_keeps_a_mapping_form_turn_block_as_rows(
    adk: AsyncEngine,
) -> None:
    await import_conversations(adk, apply=True)
    await _next_release_started(adk)
    spec_id = next(iter(EXPECTED[FULL]["view"]["turn_specs"]))
    await _put_open_turn_block(adk, {**_mapping_block(), "spec_id": spec_id})

    await close_open_turns(adk, apply=True)

    closed = await _load(adk, FULL)
    assert closed is not None and closed.turns[-1]["error"] == "abandoned"
    assert _rows(closed.turns[-1]["specs"][spec_id]) == [
        ("g7", ("GroupOfSeven",)),
        ("africa", ("DACAfrica",)),
    ]


@pytest.mark.parametrize("changed", [False, True], ids=["closed", "changed"])
async def test_close_open_turns_closes_a_turn_with_a_block_that_does_not_convert_as_stored(
    adk: AsyncEngine, monkeypatch: pytest.MonkeyPatch, changed: bool
) -> None:
    """The dry run and the apply list the same block, and so does an apply
    whose write another instance moved first, under the outcome its turn
    had."""
    await import_conversations(adk, apply=True)
    await _next_release_started(adk)
    spec_id = next(iter(EXPECTED[FULL]["view"]["turn_specs"]))
    block = {**_mapping_block(), "spec_id": spec_id, "_bound_refs": "g7"}
    async with adk.begin() as conn:
        await conn.execute(
            text(
                "UPDATE sessions SET state = jsonb_set(state, ARRAY[:key, :spec_id], "
                "CAST(:block AS jsonb)) WHERE app_name = :app AND id = :id"
            ),
            {
                "key": ORCHESTRATOR_SPECS_KEY,
                "spec_id": spec_id,
                "block": json.dumps(block),
                "app": AGENT_NAME,
                "id": f"{FULL}#t4",
            },
        )

    dry = await close_open_turns(adk, apply=False)
    if changed:

        class Moved(StudioStore):
            async def save(self, principal: str, conversation_id: str, *_: Any, **__: Any) -> None:
                raise ConversationStale(conversation_id)

        monkeypatch.setattr(adk_import, "StudioStore", Moved)

    report = await close_open_turns(adk, apply=True)

    turn = f"{_name(FULL)} turn 4"
    assert report.findings == dry.findings
    (finding,) = report.findings[turn]
    assert finding.endswith("does not convert: TypeError: _bound_refs is of type str")
    outcome = "left open" if changed else "closed"
    assert (
        f"{outcome} with blocks as stored, which the turn loader drops: {turn}"
        in report.render().splitlines()
    )
    if changed:
        assert turn in report.changed
        return
    assert (report.clean, report.refused) == (True, {})
    assert turn in report.closed
    closed = await _load(adk, FULL)
    assert closed is not None and closed.turns[-1]["error"] == "abandoned"
    assert closed.turns[-1]["specs"][spec_id] == block


@pytest.mark.parametrize(
    ("bound_refs", "counts"),
    [({"g7": ["GroupOfSeven"]}, (1, 0)), ("g7", (0, 1))],
    ids=["mapping", "unconvertible"],
)
async def test_the_report_counts_the_blocks_of_open_turn_sessions_on_their_own(
    adk: AsyncEngine, bound_refs: Any, counts: tuple[int, int]
) -> None:
    """An open turn's blocks are read from its ADK session, so a dry run shows
    apart whether any of them is in mapping form, and names each one that
    does not convert."""
    spec_id = next(iter(EXPECTED[FULL]["view"]["turn_specs"]))
    async with adk.begin() as conn:
        await conn.execute(
            text(
                "UPDATE sessions SET state = jsonb_set(state, ARRAY[:key, :spec_id], "
                "CAST(:block AS jsonb)) WHERE app_name = :app AND id = :id"
            ),
            {
                "key": ORCHESTRATOR_SPECS_KEY,
                "spec_id": spec_id,
                "block": json.dumps(
                    {**_mapping_block(), "spec_id": spec_id, "_bound_refs": bound_refs}
                ),
                "app": AGENT_NAME,
                "id": f"{FULL}#t4",
            },
        )

    report = await import_conversations(adk, apply=False)

    assert (report.converted_blocks, report.unconvertible_blocks) == (0, 0)
    assert (report.open_turn_converted_blocks, report.open_turn_unconvertible_blocks) == counts
    assert (
        "open turn spec blocks with mapping-form _bound_refs: {} converted, {} unconvertible".format(
            *counts
        )
        in report.render().splitlines()
    )
    findings = report.open_turn_findings.get(_name(FULL), [])
    assert [f.startswith(f"open turn {FULL}#t4: the _bound_refs of") for f in findings] == (
        [True] * counts[1]
    )
    assert _name(FULL) not in report.findings
    heading = f"open turn blocks the release reads as stored: {_name(FULL)}"
    assert (heading in report.render().splitlines()) == bool(counts[1])


async def test_an_open_turn_that_cannot_be_read_refuses_the_conversation_with_nothing_written(
    adk: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    read = adk_import._read_turn_session

    async def failing(conn: Any, principal: str, session_id: str) -> Any:
        if session_id.startswith(FULL):
            raise ValueError("orchestrator:specs is not a mapping")
        return await read(conn, principal, session_id)

    monkeypatch.setattr(adk_import, "_read_turn_session", failing)

    report = await import_conversations(adk, apply=True)

    assert report.refused == {
        _name(FULL): ["the import failed: ValueError: orchestrator:specs is not a mapping"]
    }
    assert await _load(adk, FULL) is None
    assert sorted(report.imported) == sorted(_name(c) for c in (PLAN_PAUSED, TEXT_ONLY))
    # The refusal is one `--discard` clears.
    dropped = await import_conversations(adk, apply=True, discard=frozenset({_name(FULL)}))
    assert (dropped.discarded, dropped.refused) == ([_name(FULL)], {})
    assert (await cutover_gate(adk)).clean


async def test_a_dry_run_takes_no_row_lock(adk: AsyncEngine) -> None:
    """A dry run after traffic moved decides a conversation the serving
    release holds a row lock on without waiting for it."""
    await import_conversations(adk, apply=True)
    await _set_adk_state(adk, TEXT_ONLY, {"studio:seq": 99})
    async with adk.connect() as conn:
        schema = (await conn.execute(text("SELECT current_schema()"))).scalar_one()
    side = create_async_engine(adk.url, connect_args={"server_settings": {"search_path": schema}})
    try:
        async with side.connect() as writer:
            await writer.begin()
            await writer.execute(
                text("SELECT id FROM conversations WHERE conversation_id = :id FOR UPDATE"),
                {"id": TEXT_ONLY},
            )
            async with asyncio.timeout(5):
                dry = await import_conversations(adk, apply=False)
            await writer.rollback()
    finally:
        await side.dispose()

    assert dry.reimported == [_name(TEXT_ONLY)]


async def test_verify_fails_on_a_conversation_changed_on_both_sides(adk: AsyncEngine) -> None:
    await import_conversations(adk, apply=True)
    await _set_adk_state(adk, FULL, {"studio:seq": 99})
    store = _store(adk)
    full = await store.load(PRINCIPAL, FULL)
    assert full is not None
    full.auto_add = not full.auto_add
    await store.save(PRINCIPAL, FULL)

    report = await verify(adk)

    assert (report.clean, report.conflicts, report.differences) == (False, [_name(FULL)], {})
    assert "conflict, changed on both sides since the import" in report.render()
    assert report.render().splitlines()[-1] == "2 of 2 equal"


async def test_verify_lists_a_conversation_written_while_it_compares_as_written(
    adk: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The serving release writes the conversation after `--verify` found it
    as imported and before it reads the rows to compare."""
    await import_conversations(adk, apply=True)
    table_view = adk_import._table_view

    async def written_first(engine: AsyncEngine, session: Any) -> Any:
        if session.conversation_id == TEXT_ONLY:
            store = _store(engine)
            state = await store.load(PRINCIPAL, TEXT_ONLY)
            assert state is not None
            state.auto_add = not state.auto_add
            await store.save(PRINCIPAL, TEXT_ONLY)
        return await table_view(engine, session)

    monkeypatch.setattr(adk_import, "_table_view", written_first)

    checked = await verify(adk)

    assert (checked.clean, checked.written_since_import) == (True, [_name(TEXT_ONLY)])


async def test_verify_counts_a_conversation_written_since_the_import_as_checked(
    adk: AsyncEngine,
) -> None:
    """The serving release has written every conversation since the import:
    verify compares none and passes."""
    await import_conversations(adk, apply=True)
    store = _store(adk)
    for conversation_id in EXPECTED:
        state = await store.load(PRINCIPAL, conversation_id)
        assert state is not None
        state.auto_add = not state.auto_add
        await store.save(PRINCIPAL, conversation_id)

    report = await verify(adk)

    assert (report.clean, sorted(report.written_since_import), report.compared) == (
        True,
        sorted(_name(c) for c in EXPECTED),
        0,
    )


async def test_verify_fails_when_it_checks_no_conversation(adk: AsyncEngine) -> None:
    """Every conversation moved on ADK since the import: none was compared,
    written or kept."""
    await import_conversations(adk, apply=True)
    for conversation_id in EXPECTED:
        await _set_adk_state(adk, conversation_id, {"studio:seq": 99})

    report = await verify(adk)

    assert (report.clean, sorted(report.moved_on_adk)) == (
        False,
        sorted(_name(c) for c in EXPECTED),
    )
    assert report.render().splitlines()[-2:] == [
        "none of the 3 ledger sessions compared, written, kept or discarded",
        "0 of 0 equal",
    ]


# Every column of release 1's migrations the import digest leaves out: the
# row's id and timestamps, the uncompared lease field, the ids of the canvas
# changes applied under the lease (written by the serving release, not
# imported), each child's parent key, and the whole import ledger.
_UNDIGESTED_COLUMNS = {
    "conversations": {"id", "created_at", "updated_at", "lease_cancel", "canvas_writes"},
    **{table: {"conversation"} for table, _ in adk_import._CHILD_TABLES},
    "adk_ledger_imports": {
        "principal",
        "conversation_id",
        "source_digest",
        "source_updated_at",
        "imported_digest",
        "diverged",
        "kept_tables",
        "imported_at",
    },
}


async def test_every_column_of_the_migrations_is_digested_or_left_out_by_name(
    pg_empty_engine: AsyncEngine,
) -> None:
    """A column a migration adds fails this until it is sorted into one."""
    await apply_pending(pg_empty_engine)
    async with pg_empty_engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name <> 'harness_schema_migrations'"
            )
        )
        columns = set(result.tuples())
    digested = {
        (table, column) for table, names in adk_import._DIGESTED_COLUMNS.items() for column in names
    }
    left_out = {(table, column) for table, names in _UNDIGESTED_COLUMNS.items() for column in names}

    assert digested & left_out == set()
    assert columns == digested | left_out


async def test_verify_lists_a_conversation_only_adk_moved_apart_from_the_compared(
    adk: AsyncEngine,
) -> None:
    """Run straight after the warm-up import while the previous release
    still serves. Its new turns are re-imported on access and are not
    differences."""
    await import_conversations(adk, apply=True)
    await _set_adk_state(adk, TEXT_ONLY, {"studio:seq": 99})

    report = await verify(adk)

    assert (report.clean, report.moved_on_adk, report.differences) == (
        True,
        [_name(TEXT_ONLY)],
        {},
    )
    assert report.render().splitlines() == [
        f"moved on ADK since the import; re-imported on access: {_name(TEXT_ONLY)}",
        "2 of 2 equal",
    ]


async def test_a_stop_on_a_copied_lease_does_not_diverge_the_conversation(
    adk: AsyncEngine,
) -> None:
    """A stream the previous release still runs. A read on this release
    copies its live lease, and a Stop sets the flag on that copy, which
    `--verify` leaves out of the comparison. The stream's last write to ADK
    is then re-imported, not a conflict."""
    await import_conversations(adk, apply=True)
    lease = {"id": "seg_old", "owner": "old", "expires_at": time.time() + 60, "cancel": False}
    await _set_adk_state(adk, TEXT_ONLY, {"studio:segment": lease})
    store = _reimporting_store(adk)
    copied = await store.load(PRINCIPAL, TEXT_ONLY)
    assert copied is not None and copied.segment is not None

    assert await store.request_cancel(PRINCIPAL, TEXT_ONLY)
    streaming = await verify(adk)
    assert (streaming.differences, streaming.conflicts, streaming.streaming_on_adk) == (
        {},
        [],
        [_name(TEXT_ONLY)],
    )
    await _set_adk_state(adk, TEXT_ONLY, {"studio:segment": None, "studio:seq": 99})

    state = await store.load(PRINCIPAL, TEXT_ONLY)
    assert state is not None and (state.seq, state.segment) == (99, None)
    assert (await import_conversations(adk, apply=False)).unchanged == sorted(
        _name(c) for c in EXPECTED
    )


async def test_close_open_turns_leaves_a_diverged_conversation_alone_and_fails(
    adk: AsyncEngine,
) -> None:
    await _diverge(adk, FULL)
    await import_conversations(adk, apply=True)
    await _next_release_started(adk)

    report = await close_open_turns(adk, apply=True)

    assert (report.clean, report.diverged) == (False, [_name(FULL)])
    assert report.closed == [f"{_name(PLAN_PAUSED)} turn 1"]
    assert (
        f"left alone, diverged (--prefer-adk or --keep-tables first): {_name(FULL)}"
        in report.render()
    )
    kept = await _load(adk, FULL)
    assert kept is not None and not kept.turn_recorded


async def test_prefer_adk_refuses_a_conversation_a_stream_holds_unless_forced(
    adk: AsyncEngine,
) -> None:
    await import_conversations(adk, apply=True)
    await _set_adk_state(adk, FULL, {"studio:seq": 99})
    store = _store(adk)
    full = await store.load(PRINCIPAL, FULL)
    assert full is not None
    store.claim_segment(PRINCIPAL, FULL)
    await store.save(PRINCIPAL, FULL)
    preferred = frozenset({_name(FULL)})

    held = await import_conversations(adk, apply=True, prefer_adk=preferred)

    assert list(held.refused) == [_name(FULL)]
    assert held.refused[_name(FULL)][0].startswith("--prefer-adk would end the stream")
    kept = await _load(adk, FULL)
    assert kept is not None and kept.segment is not None

    forced = await import_conversations(adk, apply=True, prefer_adk=preferred, force=True)

    assert forced.reimported == [_name(FULL)]
    taken = await _load(adk, FULL)
    assert taken is not None and (taken.seq, taken.segment) == (99, None)


async def test_close_open_turns_refuses_a_conversation_it_cannot_close_and_closes_the_rest(
    adk: AsyncEngine,
) -> None:
    await import_conversations(adk, apply=True)
    await _next_release_started(adk)
    async with adk.begin() as conn:
        await conn.execute(
            text(
                "UPDATE conversations SET turn_state = turn_state || "
                '\'{"pending": {"int_broken": {}}}\'::jsonb WHERE conversation_id = :id'
            ),
            {"id": FULL},
        )

    report = await close_open_turns(adk, apply=True)

    assert list(report.refused) == [_name(FULL)]
    assert report.refused[_name(FULL)][0].startswith("the close failed: KeyError")
    assert report.closed == [f"{_name(PLAN_PAUSED)} turn 1"]


async def test_close_open_turns_fences_out_a_stream_whose_lease_lapsed(
    adk: AsyncEngine,
) -> None:
    """A stream still running under a lapsed lease cannot write over the turn
    the job closed."""
    await import_conversations(adk, apply=True)
    await _next_release_started(adk)
    async with adk.begin() as conn:
        await conn.execute(
            text(
                "UPDATE conversations SET lease_segment_id = 'seg_stalled', lease_owner = 'other', "
                "lease_expires_at = extract(epoch FROM now()) - 1 "
                "WHERE conversation_id = :plan_paused"
            ),
            {"plan_paused": PLAN_PAUSED},
        )
    stalled = _store(adk)
    state = await stalled.load(PRINCIPAL, PLAN_PAUSED)
    assert state is not None and state.segment is not None

    assert (await close_open_turns(adk, apply=True)).closed[-1] == f"{_name(PLAN_PAUSED)} turn 1"

    with pytest.raises(SegmentLost):
        await stalled.checkpoint(PRINCIPAL, PLAN_PAUSED, "seg_stalled")
    closed = await _load(adk, PLAN_PAUSED)
    assert closed is not None and closed.turn_recorded and closed.segment is None


async def test_close_open_turns_leaves_a_turn_cursors_turn_opened_while_its_write_waits(
    adk: AsyncEngine,
) -> None:
    """A writer opens turn 1 in `turn_cursors` under `FOR SHARE` on the
    conversation's row, and commits only once the job's closing write waits
    on that row. The write reads the cursor under its own row lock and
    leaves the turn open, its pending interrupts and blocks kept."""
    await import_conversations(adk, apply=True)
    await _next_release_started(adk)
    plan_paused = await _row_id(adk, PLAN_PAUSED)
    # The writer and the poll run beside the job's own pool of two.
    async with adk.connect() as conn:
        schema = (await conn.execute(text("SELECT current_schema()"))).scalar_one()
    side = create_async_engine(adk.url, connect_args={"server_settings": {"search_path": schema}})
    try:
        async with side.connect() as writer:
            transaction = await writer.begin()
            await writer.execute(
                text("SELECT id FROM conversations WHERE id = :pk FOR SHARE"), {"pk": plan_paused}
            )
            await writer.execute(
                text(
                    "INSERT INTO turn_cursors (conversation, turn_index, status) "
                    "VALUES (:pk, 1, 'paused')"
                ),
                {"pk": plan_paused},
            )
            run = asyncio.create_task(close_open_turns(adk, apply=True))
            async with asyncio.timeout(10):
                while not await _waiting_on_a_lock(side):
                    await asyncio.sleep(0.02)
            await transaction.commit()
    finally:
        await side.dispose()
    report = await run

    assert (report.closed, report.in_turn_cursors, report.clean) == (
        [f"{_name(FULL)} turn 4"],
        [f"{_name(PLAN_PAUSED)} turn 1"],
        True,
    )
    kept = await _load(adk, PLAN_PAUSED)
    assert kept is not None and not kept.turn_recorded and kept.pending
    dry = await close_open_turns(adk, apply=False)
    assert (dry.closed, dry.in_turn_cursors) == ([], [f"{_name(PLAN_PAUSED)} turn 1"])


async def test_close_open_turns_refuses_a_moved_turn_index_and_always_clears_the_lease(
    adk: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Between the job's load and its write, one conversation's row moves to
    another turn index, and the other's gains a lapsed lease, both without
    a version change. The first is listed as changed and left open, and the
    second closes with no lease on its row."""
    await import_conversations(adk, apply=True)
    await _next_release_started(adk)

    class Moving(StudioStore):
        async def load(self, principal: str, conversation_id: str) -> Any:
            state = await super().load(principal, conversation_id)
            async with adk.begin() as conn:
                if conversation_id == PLAN_PAUSED:
                    await conn.execute(
                        text(
                            "UPDATE conversations SET turn_state = "
                            "jsonb_set(turn_state, '{turn_index}', '9') WHERE conversation_id = :id"
                        ),
                        {"id": conversation_id},
                    )
                else:
                    await conn.execute(
                        text(
                            "UPDATE conversations SET lease_segment_id = 'seg_lapsed', "
                            "lease_owner = 'other', "
                            "lease_expires_at = extract(epoch FROM now()) - 1 "
                            "WHERE conversation_id = :id"
                        ),
                        {"id": conversation_id},
                    )
            return state

    monkeypatch.setattr(adk_import, "StudioStore", Moving)

    report = await close_open_turns(adk, apply=True)

    assert (report.closed, report.changed) == (
        [f"{_name(FULL)} turn 4"],
        [f"{_name(PLAN_PAUSED)} turn 1"],
    )
    async with adk.connect() as conn:
        leases = dict(
            (
                await conn.execute(
                    text("SELECT conversation_id, lease_segment_id FROM conversations")
                )
            ).all()
        )
    assert leases[FULL] is None
    monkeypatch.undo()
    left = await _load(adk, PLAN_PAUSED)
    assert left is not None and not left.turn_recorded and left.pending


async def test_verify_and_close_open_turns_keep_one_conversation_at_a_time(
    adk: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store keeps every conversation it loads, so the job's verbs build
    one per conversation and let it go."""
    await import_conversations(adk, apply=True)
    stores: list[StudioStore] = []

    class Recorded(StudioStore):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            stores.append(self)

    monkeypatch.setattr(adk_import, "StudioStore", Recorded)

    assert (await verify(adk)).clean
    assert (await close_open_turns(adk, apply=False)).clean

    assert stores and max(len(store._runtime) for store in stores) == 1


def _reimporting_store(engine: AsyncEngine) -> StudioStore:
    """A store as the server builds it."""
    return StudioStore(engine, adk=reimport_on_access(engine), turn_specs=open_turn_specs(engine))


async def test_a_turn_the_previous_release_served_after_the_import_reaches_the_next_turn(
    adk: AsyncEngine,
) -> None:
    """The previous release serves a third turn after the import, writing
    it and its turn session to ADK. The next `/turn` on this release reads
    it, and runs as turn 4 rather than claiming turn 3 again, whose ADK
    turn session exists."""
    await import_conversations(adk, apply=True)
    third = {"user": "a third question", "assistant": "Three."}
    await _set_adk_state(
        adk,
        TEXT_ONLY,
        {
            "studio:turn:0002": third,
            "studio:turn_index": 3,
            "studio:turn_user_message": third["user"],
            "studio:turn_final_text": third["assistant"],
        },
    )
    async with adk.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO sessions (app_name, user_id, id, state, create_time, update_time) "
                "VALUES (:app, :principal, :id, '{}'::jsonb, now(), now())"
            ),
            {"app": AGENT_NAME, "principal": PRINCIPAL, "id": f"{TEXT_ONLY}#t3"},
        )
    service = DatabaseSessionService(db_engine=adk)
    store = _reimporting_store(adk)
    app, model = orchestrator_app(engine=adk, session_service=service, store=store)

    async with studio_client(app) as client:
        frames = await stream(
            client, "/api/studio/turn", turn_body("and a fourth?", conversation_id=TEXT_ONLY)
        )

    assert frames[-1]["event"] == "done"
    assert "a third question" in request_text(model.requests[0])
    state = await _load(adk, TEXT_ONLY)
    assert state is not None
    assert (state.turn_index, state.turns[2]) == (4, third)


async def test_a_conversation_never_imported_is_imported_when_first_read(
    adk: AsyncEngine,
) -> None:
    """Started on the previous release after the last import run."""
    await apply_pending(adk)

    state = await _reimporting_store(adk).load(PRINCIPAL, PLAN_PAUSED)

    assert state is not None and _view(state) == EXPECTED[PLAN_PAUSED]["view"]
    assert (await import_conversations(adk, apply=False)).unchanged == [_name(PLAN_PAUSED)]


async def test_a_turn_racing_the_previous_release_claims_nothing_and_meets_its_lease(
    adk: AsyncEngine,
) -> None:
    """The previous release claims turn 3 and creates its turn session between
    this release's load and its session create. This release meets that
    session before its claim save, so it claims nothing and answers the
    retryable 409, leaving the session alone. The retry re-imports the
    previous release's turn and meets its live lease as a busy stream, with
    no conflict recorded. Once that lease lapses, a `/turn` streams to
    `done`."""
    await import_conversations(adk, apply=True)
    service = DatabaseSessionService(db_engine=adk)
    app, _ = orchestrator_app(engine=adk, session_service=service, store=_reimporting_store(adk))
    studio = app.state.studio
    open_session = studio._open_session
    lease = {"id": "seg_old", "owner": "old", "expires_at": time.time() + 600, "cancel": False}

    async def previous_release_claims_first(**kwargs: Any) -> bool:
        await _set_adk_state(
            adk, TEXT_ONLY, {"studio:segment": lease, "studio:turn_index": 3, "studio:seq": 99}
        )
        async with adk.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO sessions (app_name, user_id, id, state, create_time, "
                    "update_time) VALUES (:app, :principal, :id, '{}'::jsonb, now(), now())"
                ),
                {"app": AGENT_NAME, "principal": PRINCIPAL, "id": f"{TEXT_ONLY}#t3"},
            )
        studio._open_session = open_session
        return await open_session(**kwargs)

    studio._open_session = previous_release_claims_first
    body = turn_body("and a third?", conversation_id=TEXT_ONLY)

    async with studio_client(app) as client:
        first = await client.post("/api/studio/turn", json=body, headers=AUTH)
        async with adk.connect() as conn:
            leased = (
                await conn.execute(
                    text("SELECT lease_segment_id FROM conversations WHERE conversation_id = :id"),
                    {"id": TEXT_ONLY},
                )
            ).scalar_one()
        retry = await client.post("/api/studio/turn", json=body, headers=AUTH)

        assert (first.status_code, first.json()["error"]) == (409, "transient_unavailable")
        assert leased is None
        assert (retry.status_code, retry.json()["error"]) == (409, "transient_unavailable")
        assert retry.json()["retry_after_seconds"] > 0
        assert await _diverged_flags(adk) == []
        kept = await service.get_session(
            app_name=AGENT_NAME, user_id=PRINCIPAL, session_id=f"{TEXT_ONLY}#t3"
        )
        assert kept is not None
        reimported = await _load(adk, TEXT_ONLY)
        assert reimported is not None and reimported.segment is not None
        assert (reimported.turn_index, reimported.segment.id) == (3, "seg_old")

        await _set_adk_state(
            adk, TEXT_ONLY, {"studio:segment": {**lease, "expires_at": time.time() - 1}}
        )
        frames = await stream(client, "/api/studio/turn", body)

    assert frames[-1]["event"] == "done"
    assert await _diverged_flags(adk) == []


async def _diverged_flags(engine: AsyncEngine) -> list[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT conversation_id FROM adk_ledger_imports WHERE diverged")
        )
        return [row.conversation_id for row in rows]


async def _diverge(engine: AsyncEngine, conversation_id: str) -> None:
    """Import every conversation, then change `conversation_id` on ADK and
    in the tables."""
    await import_conversations(engine, apply=True)
    await _set_adk_state(engine, conversation_id, {"studio:seq": 99})
    store = _store(engine)
    state = await store.load(PRINCIPAL, conversation_id)
    assert state is not None
    state.auto_add = not state.auto_add
    await store.save(PRINCIPAL, conversation_id)


def _recording(engine: AsyncEngine) -> tuple[list[str], Any]:
    """The statements `engine` runs, and the listener that records them."""
    statements: list[str] = []

    def record(conn: Any, cursor: Any, statement: str, *args: Any) -> None:
        statements.append(statement)

    return statements, record


async def test_a_conversation_changed_on_both_sides_is_held_until_adk_is_preferred(
    adk: AsyncEngine, caplog: pytest.LogCaptureFixture
) -> None:
    """Its tables lack what ADK holds, so serving them would lose the
    previous release's turns. After the first refusal each load is the one
    lookup, and each logs the command that repairs it."""
    await _diverge(adk, FULL)
    store = _reimporting_store(adk)

    with caplog.at_level("ERROR", logger="harness.adk_import"):
        with pytest.raises(ConversationHeld):
            await store.load(PRINCIPAL, FULL)
        statements, record = _recording(adk)
        event.listen(adk.sync_engine, "before_cursor_execute", record)
        try:
            with pytest.raises(ConversationHeld):
                await store.load(PRINCIPAL, FULL)
        finally:
            event.remove(adk.sync_engine, "before_cursor_execute", record)

    assert len(statements) == 1 and "FROM sessions s LEFT JOIN adk_ledger_imports" in statements[0]
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 2
    assert all(f"--prefer-adk {_name(FULL)}" in message for message in errors)
    # The job still decides by digest.
    assert (await import_conversations(adk, apply=False)).conflicts == [_name(FULL)]
    assert (await verify(adk)).conflicts == [_name(FULL)]

    await import_conversations(adk, apply=True, prefer_adk=frozenset({_name(FULL)}))

    served = await store.load(PRINCIPAL, FULL)
    assert served is not None and served.seq == 99


_HELD_ROUTES = {
    "turn": turn_body("and a third?", conversation_id=TEXT_ONLY),
    "interrupt": {
        "kind": "interrupt.response",
        "schema_version": 1,
        "conversation_id": TEXT_ONLY,
        "interrupt_id": "int_any",
        "decision": "approve",
    },
    "composition": {
        "kind": "composition.mutate",
        "schema_version": 1,
        "conversation_id": TEXT_ONLY,
        "operations": [{"op": "pin", "component_id": "cmp_x"}],
    },
    # With no live lease to signal. With one, `/cancel` signals it and answers 204.
    "cancel": {"kind": "turn.cancel", "schema_version": 1, "conversation_id": TEXT_ONLY},
}


@pytest.mark.parametrize("route", list(_HELD_ROUTES))
async def test_a_held_conversation_is_refused_on_the_wire_without_taking_a_lease(
    adk: AsyncEngine, route: str
) -> None:
    await _diverge(adk, TEXT_ONLY)
    service = DatabaseSessionService(db_engine=adk)
    app, _ = orchestrator_app(engine=adk, session_service=service, store=_reimporting_store(adk))
    path, body = f"/api/studio/{route}", _HELD_ROUTES[route]

    async with studio_client(app) as client:
        turn = await client.post(path, json=body, headers=AUTH)
        again = await client.post(path, json=body, headers=AUTH)

    assert (turn.status_code, again.status_code) == (409, 409)
    body = turn.json()
    assert (body["error"], body["retry"], body["constraint_id"]) == (
        "constraint_violation",
        "do_not_retry",
        "conversation_diverged",
    )
    async with adk.connect() as conn:
        lease = (
            await conn.execute(
                text("SELECT lease_segment_id FROM conversations WHERE conversation_id = :id"),
                {"id": TEXT_ONLY},
            )
        ).scalar_one()
    assert lease is None


class _LostConnectionError(OSError):
    """A database fault the copy meets once."""


def _write_fails_once(monkeypatch: pytest.MonkeyPatch) -> None:
    write = adk_import._write
    calls = 0

    async def failing(*args: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _LostConnectionError("connection lost")
        await write(*args)

    monkeypatch.setattr(adk_import, "_write", failing)


async def test_a_copy_that_fails_on_a_fault_fails_the_load_and_the_retry_copies(
    adk: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Served stale, the request's own claim would make the conversation a
    conflict for good."""
    await import_conversations(adk, apply=True)
    await _set_adk_state(adk, TEXT_ONLY, {"studio:seq": 99})
    _write_fails_once(monkeypatch)
    store = _reimporting_store(adk)

    with pytest.raises(_LostConnectionError):
        await store.load(PRINCIPAL, TEXT_ONLY)

    assert (await import_conversations(adk, apply=False)).reimported == [_name(TEXT_ONLY)]
    state = await store.load(PRINCIPAL, TEXT_ONLY)
    assert state is not None and state.seq == 99
    assert (await import_conversations(adk, apply=False)).conflicts == []


async def test_a_turn_on_a_never_imported_conversation_whose_copy_fails_creates_no_row(
    adk: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Created afresh, it would hide its ADK history behind a conflict."""
    await apply_pending(adk)
    _write_fails_once(monkeypatch)
    service = DatabaseSessionService(db_engine=adk)
    app, _ = orchestrator_app(engine=adk, session_service=service, store=_reimporting_store(adk))
    body = turn_body("and a third?", conversation_id=TEXT_ONLY)

    async with studio_client(app) as client:
        with pytest.raises(_LostConnectionError):
            await client.post("/api/studio/turn", json=body, headers=AUTH)
        assert await _count(adk) == 0
        frames = await stream(client, "/api/studio/turn", body)

    assert frames[-1]["event"] == "done"
    state = await _load(adk, TEXT_ONLY)
    assert state is not None
    assert state.turns[:2] == EXPECTED[TEXT_ONLY]["view"]["turns"]


async def test_copies_waiting_on_the_row_lock_read_adk_in_the_order_they_commit(
    adk: AsyncEngine,
) -> None:
    """A copy that read ADK before it waited on the lock would write the
    older state over the newer one another copy committed."""
    await import_conversations(adk, apply=True)
    await _set_adk_state(adk, TEXT_ONLY, {"studio:seq": 98})
    async with adk.connect() as conn:
        schema = (await conn.execute(text("SELECT current_schema()"))).scalar_one()
    side = create_async_engine(adk.url, connect_args={"server_settings": {"search_path": schema}})
    try:
        async with side.connect() as holder:
            transaction = await holder.begin()
            await holder.execute(
                text("SELECT id FROM conversations WHERE conversation_id = :id FOR UPDATE"),
                {"id": TEXT_ONLY},
            )
            copy = asyncio.create_task(_reimporting_store(adk).load(PRINCIPAL, TEXT_ONLY))
            async with asyncio.timeout(10):
                while not await _waiting_on_a_lock(side):
                    await asyncio.sleep(0.02)
            async with side.begin() as writer:
                await writer.execute(
                    text(
                        "UPDATE sessions SET state = state || '{\"studio:seq\": 99}'::jsonb, "
                        "update_time = clock_timestamp() WHERE app_name = :app AND id = :id"
                    ),
                    {"app": LEDGER_APP, "id": TEXT_ONLY},
                )
            await transaction.commit()
    finally:
        await side.dispose()
    state = await copy

    assert state is not None and state.seq == 99
    assert (await import_conversations(adk, apply=False)).unchanged == sorted(
        _name(c) for c in EXPECTED
    )


@pytest.mark.parametrize(
    ("refusal", "reason"),
    [
        ({"studio:unknown": 1}, "studio:unknown: a key this job does not know"),
        ({"studio:seq": "7"}, "the import failed: TypeError"),
    ],
    ids=["ledger-fault", "rows-do-not-build"],
)
async def test_a_refused_conversation_the_tables_lack_starts_afresh_and_its_copy_is_tried_once(
    refusal: dict[str, Any],
    reason: str,
    pg_empty_engine: AsyncEngine,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Served as absent, so its next turn starts it afresh, and its later
    reads serve the row that turn made. Its old ADK turn sessions do not
    block the fresh turn 1."""
    rows = _sessions(**{TEXT_ONLY: _add(**refusal)})
    await _write_adk(pg_empty_engine, rows)
    await apply_pending(pg_empty_engine)
    service = DatabaseSessionService(db_engine=pg_empty_engine)
    store = _reimporting_store(pg_empty_engine)
    app, _ = orchestrator_app(engine=pg_empty_engine, session_service=service, store=store)

    with caplog.at_level("ERROR", logger="harness.adk_import"):
        assert await store.load(PRINCIPAL, TEXT_ONLY) is None
        statements, record = _recording(pg_empty_engine)
        event.listen(pg_empty_engine.sync_engine, "before_cursor_execute", record)
        try:
            assert await store.load(PRINCIPAL, TEXT_ONLY) is None
        finally:
            event.remove(pg_empty_engine.sync_engine, "before_cursor_execute", record)
        async with studio_client(app) as client:
            first = await stream(
                client, "/api/studio/turn", turn_body("hello", conversation_id=TEXT_ONLY)
            )
            second = await stream(
                client, "/api/studio/turn", turn_body("again", conversation_id=TEXT_ONLY)
            )

    # The copy is not tried again. The one read of ADK is the lookup.
    assert [s for s in statements if "sessions" in s] == [statements[0]]
    assert "FROM sessions s LEFT JOIN adk_ledger_imports" in statements[0]
    (error,) = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert reason in error
    assert "its next turn starts it afresh" in error
    assert (first[-1]["event"], second[-1]["event"]) == ("done", "done")
    state = await _load(pg_empty_engine, TEXT_ONLY)
    assert state is not None and state.turn_index == 2
    assert (await import_conversations(pg_empty_engine, apply=False)).conflicts == []


async def test_a_refused_conversation_the_tables_hold_is_served_as_they_hold_it(
    adk: AsyncEngine,
) -> None:
    await import_conversations(adk, apply=True)
    await _set_adk_state(adk, TEXT_ONLY, {"studio:unknown": 1})

    state = await _reimporting_store(adk).load(PRINCIPAL, TEXT_ONLY)

    assert state is not None and _view(state) == EXPECTED[TEXT_ONLY]["view"]


async def test_a_conversation_another_instance_found_diverged_counts_as_moved(
    adk: AsyncEngine,
) -> None:
    """The conflict is recorded at ADK's latest `update_time`, so the
    session's time no longer tells that ADK moved; the diverged flag does,
    and `_open_session` leaves the previous release's turn session alone."""
    await _diverge(adk, TEXT_ONLY)
    with pytest.raises(ConversationHeld):
        await reimport_on_access(adk).reimport(PRINCIPAL, TEXT_ONLY)

    assert await reimport_on_access(adk).moved(PRINCIPAL, TEXT_ONLY)


async def test_a_read_of_a_conversation_adk_has_not_moved_is_one_lookup(
    adk: AsyncEngine,
) -> None:
    await import_conversations(adk, apply=True)
    statements: list[str] = []

    def record(conn: Any, cursor: Any, statement: str, *args: Any) -> None:
        statements.append(statement)

    event.listen(adk.sync_engine, "before_cursor_execute", record)
    try:
        await reimport_on_access(adk).reimport(PRINCIPAL, FULL)
        await reimport_on_access(adk).reimport(PRINCIPAL, "conv_born_on_the_release")
    finally:
        event.remove(adk.sync_engine, "before_cursor_execute", record)

    assert len(statements) == 2
    assert all("FROM sessions s LEFT JOIN adk_ledger_imports" in s for s in statements)


async def _release_2_schema(engine: AsyncEngine) -> None:
    """The harness's tables as release 2's migrations leave them: release
    1's, plus columns on `conversations`, a table of its own, and its file
    in the migrations ledger, which release 1's `import` logs and passes."""
    await apply_pending(engine)
    async with engine.begin() as conn:
        await conn.execute(
            text("ALTER TABLE conversations ADD COLUMN stop_id text, ADD COLUMN stop_turn integer")
        )
        await conn.execute(
            text(
                "CREATE TABLE turn_cursors "
                "(conversation bigint NOT NULL, turn_index integer NOT NULL)"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO harness_schema_migrations (version, content_sha256) "
                "VALUES ('0003_turns.sql', 'release 2')"
            )
        )


_RELEASE_2_SCHEMA = pytest.mark.parametrize(
    "release_2_schema", [False, True], ids=["release-1-schema", "release-2-schema"]
)
LATE = "conv_01adklate000"


async def _gate_complete(engine: AsyncEngine) -> None:
    await import_conversations(engine, apply=True)


async def _gate_one_never_imported(engine: AsyncEngine) -> None:
    """A ledger session the previous revision started after the import."""
    await import_conversations(engine, apply=True)
    (late,) = [
        {**row, "id": LATE}
        for row in RECORDED["sessions"]
        if row["app_name"] == LEDGER_APP and row["id"] == TEXT_ONLY
    ]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO sessions (app_name, user_id, id, state, create_time, update_time) "
                "VALUES (:app_name, :user_id, :id, CAST(:state AS jsonb), now(), now())"
            ),
            {**late, "state": json.dumps(late["state"])},
        )


async def _gate_placeholder_only(engine: AsyncEngine) -> None:
    """The tables held the conversation before its first copy, so the copy
    records a conflict with no digest."""
    await apply_pending(engine)
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO conversations (principal, conversation_id) VALUES (:p, :c)"),
            {"p": PRINCIPAL, "c": FULL},
        )
    await import_conversations(engine, apply=True)
    async with engine.connect() as conn:
        record = (
            await conn.execute(
                text(
                    "SELECT source_digest, imported_digest, diverged FROM adk_ledger_imports "
                    "WHERE conversation_id = :c"
                ),
                {"c": FULL},
            )
        ).one()
    assert tuple(record) == (None, None, True)


async def _gate_adk_moved(engine: AsyncEngine) -> None:
    await import_conversations(engine, apply=True)
    await _set_adk_state(engine, TEXT_ONLY, {"studio:seq": 99})


async def _gate_kept(engine: AsyncEngine) -> None:
    """A refused conversation the tables hold, as imported before ADK's state
    gained a key the job does not know, kept with `--keep-tables`."""
    await import_conversations(engine, apply=True)
    await _set_adk_state(engine, TEXT_ONLY, {"studio:unknown": 1})
    report = await import_conversations(
        engine, apply=True, keep_tables=frozenset({_name(TEXT_ONLY)})
    )
    assert report.kept == [_name(TEXT_ONLY)]


async def _gate_discarded(engine: AsyncEngine) -> None:
    """A refused conversation the tables do not hold, given up with
    `--discard`."""
    await _set_adk_state(engine, TEXT_ONLY, {"studio:unknown": 1})
    report = await import_conversations(engine, apply=True, discard=frozenset({_name(TEXT_ONLY)}))
    assert report.discarded == [_name(TEXT_ONLY)]


async def _gate_kept_then_adk_moved(engine: AsyncEngine) -> None:
    await _gate_kept(engine)
    await _set_adk_state(engine, TEXT_ONLY, {"studio:seq": 5})


async def _gate_no_sessions(engine: AsyncEngine) -> None:
    del engine


async def _gate_foreign_sessions(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE TABLE sessions (app_name text, id text, note text)"))
        await conn.execute(
            text("INSERT INTO sessions VALUES (:app, 'conv_x', 'another service')"),
            {"app": LEDGER_APP},
        )


async def _gate_other_apps_only(engine: AsyncEngine) -> None:
    await _write_adk(engine, [row for row in RECORDED["sessions"] if row["app_name"] != LEDGER_APP])


async def _gate_never_imported(engine: AsyncEngine) -> None:
    await _write_adk(engine, RECORDED["sessions"])


@_RELEASE_2_SCHEMA
@pytest.mark.parametrize(
    ("setup", "adk_rows", "uncleared", "discarded"),
    [
        (_gate_complete, True, [], []),
        (_gate_one_never_imported, True, [LATE], []),
        (_gate_placeholder_only, True, [FULL], []),
        (_gate_adk_moved, True, [TEXT_ONLY], []),
        (_gate_kept, True, [], []),
        (_gate_discarded, True, [], [TEXT_ONLY]),
        (_gate_kept_then_adk_moved, True, [TEXT_ONLY], []),
        (_gate_no_sessions, False, None, []),
        (_gate_foreign_sessions, False, None, []),
        (_gate_other_apps_only, False, [], []),
        (_gate_never_imported, False, sorted(EXPECTED), []),
    ],
    ids=[
        "complete",
        "one-never-imported",
        "placeholder-only",
        "adk-moved",
        "kept",
        "discarded",
        "kept-then-adk-moved",
        "no-sessions",
        "foreign-sessions",
        "other-apps-only",
        "never-imported",
    ],
)
async def test_the_cutover_gate_clears_only_current_imported_or_kept_ledger_sessions(
    pg_empty_engine: AsyncEngine,
    setup: Any,
    adk_rows: bool,
    uncleared: list[str] | None,
    discarded: list[str],
    release_2_schema: bool,
) -> None:
    """`uncleared` None: no ADK `sessions` table resolves, so the gate cannot
    check anything and refuses, unless told the tables were archived and an
    archive resolves. A discarded conversation clears and is listed apart."""
    if adk_rows:
        await _write_adk(pg_empty_engine, RECORDED["sessions"])
    if release_2_schema:
        await _release_2_schema(pg_empty_engine)
    await setup(pg_empty_engine)

    if uncleared is None:
        with pytest.raises(AdkTablesMissing):
            await cutover_gate(pg_empty_engine)
        with pytest.raises(AdkTablesMissing, match="--adk-archived needs adk_archive_sessions"):
            await cutover_gate(pg_empty_engine, adk_archived=True)
        async with pg_empty_engine.begin() as conn:
            await conn.execute(text("CREATE TABLE adk_archive_sessions (id text)"))
        archived = await cutover_gate(pg_empty_engine, adk_archived=True)
        assert (archived.clean, archived.archived) == (True, True)
        return
    gate = await cutover_gate(pg_empty_engine)

    assert (gate.clean, gate.uncleared, gate.first, gate.discarded) == (
        not uncleared,
        len(uncleared),
        [_name(c) for c in uncleared],
        [_name(c) for c in discarded],
    )
    assert (f"  {_name(TEXT_ONLY)}" in gate.render().splitlines()) == (
        TEXT_ONLY in uncleared or TEXT_ONLY in discarded
    )


async def test_the_gate_command_names_what_it_has_not_cleared_and_passes_once_clear(
    pg_database_url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = create_async_engine(pg_database_url)
    try:
        monkeypatch.setenv("DATABASE_URL", pg_database_url)
        async with engine.begin() as conn:
            await conn.execute(text("ALTER TABLE sessions RENAME TO adk_archive_sessions"))
        assert await asyncio.to_thread(main, ["gate"]) == 2
        assert "no table named sessions carries ADK's columns" in capsys.readouterr().err
        assert await asyncio.to_thread(main, ["gate", "--adk-archived"]) == 0
        assert capsys.readouterr().out.startswith("cutover gate: no ADK sessions table")
        async with engine.begin() as conn:
            await conn.execute(text("ALTER TABLE adk_archive_sessions RENAME TO sessions"))
        await _write_adk(engine, RECORDED["sessions"])

        assert await asyncio.to_thread(main, ["gate"]) == 1
        out = capsys.readouterr().out.splitlines()
        assert out[:4] == [
            "cutover gate: 3 ledger sessions not cleared",
            *[f"  {_name(c)}" for c in sorted(EXPECTED)],
        ]
        assert out[-1].endswith("import --apply --discard PRINCIPAL/CONVERSATION_ID ...")

        assert await asyncio.to_thread(main, ["import", "--apply"]) == 0
        capsys.readouterr()
        assert await asyncio.to_thread(main, ["gate"]) == 0
        assert capsys.readouterr().out == "cutover gate: every ledger session is cleared\n"
    finally:
        await engine.dispose()


@_RELEASE_2_SCHEMA
async def test_a_refused_conversation_the_tables_lack_is_discarded_only_by_name_and_counted(
    pg_empty_engine: AsyncEngine, release_2_schema: bool
) -> None:
    """The tables lack the conversation. `--keep-tables` would keep nothing,
    so it refuses; `--discard` gives it up, adds no row, and the report, the
    gate and verify count it, once its user starts it afresh too, until ADK's
    state moves again."""
    name = _name(TEXT_ONLY)
    keep, drop = frozenset({name}), frozenset({name})
    await _write_adk(pg_empty_engine, _sessions(**{TEXT_ONLY: _add(**{"studio:unknown": 1})}))
    if release_2_schema:
        await _release_2_schema(pg_empty_engine)
    assert list((await import_conversations(pg_empty_engine, apply=True)).refused) == [name]
    assert (await cutover_gate(pg_empty_engine)).first == [name]

    for apply in (False, True):
        kept = await import_conversations(pg_empty_engine, apply=apply, keep_tables=keep)

        assert (kept.kept, list(kept.refused), kept.clean) == ([], [name], False)
        # The key's refusal and `--keep-tables`'s own.
        assert len(kept.refused[name]) == 2
        assert (await cutover_gate(pg_empty_engine)).first == [name]

    dry = await import_conversations(pg_empty_engine, apply=False, discard=drop)

    assert (dry.discarded, dry.refused, dry.clean) == ([name], {}, True)
    assert name in dry.render()
    assert not (await cutover_gate(pg_empty_engine)).clean

    applied = await import_conversations(pg_empty_engine, apply=True, discard=drop)

    assert (applied.discarded, applied.refused, applied.clean) == ([name], {}, True)
    # With the column in place, a repair takes no lock on the table beyond a
    # row write's, so a reader holding it does not stall the run.
    async with pg_empty_engine.connect() as reader:
        await reader.execute(text("LOCK TABLE adk_ledger_imports IN ACCESS SHARE MODE"))
        again = await asyncio.wait_for(
            import_conversations(pg_empty_engine, apply=True, discard=drop), timeout=5
        )
        await reader.rollback()
    assert again.discarded == [name]
    # The record keeps nothing of a conversation the tables lack.
    kept_absent = await import_conversations(pg_empty_engine, apply=True, keep_tables=keep)
    assert (list(kept_absent.refused), len(kept_absent.refused[name])) == ([name], 1)
    gate = await cutover_gate(pg_empty_engine)
    assert (gate.clean, gate.discarded) == (True, [name])
    assert await _count(pg_empty_engine) == 2
    assert (await import_conversations(pg_empty_engine, apply=True)).discarded == [name]
    assert await _reimporting_store(pg_empty_engine).load(PRINCIPAL, TEXT_ONLY) is None

    # Its user starts it afresh, and another conversation moves on ADK.
    async with pg_empty_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO conversations (principal, conversation_id) VALUES (:p, :c)"),
            {"p": PRINCIPAL, "c": TEXT_ONLY},
        )
    await _set_adk_state(pg_empty_engine, FULL, {"studio:seq": 7})

    gate = await cutover_gate(pg_empty_engine)
    assert (gate.first, gate.discarded) == ([_name(FULL)], [name])
    assert "  " + name in gate.render().splitlines()
    rerun = await import_conversations(pg_empty_engine, apply=False)
    assert (rerun.discarded, rerun.kept) == ([name], [])
    assert (await verify(pg_empty_engine)).discarded == [name]

    await _set_adk_state(pg_empty_engine, TEXT_ONLY, {"studio:seq": 5})

    gate = await cutover_gate(pg_empty_engine)
    assert (gate.uncleared, gate.first) == (2, sorted([_name(FULL), name]))
    assert name in (await import_conversations(pg_empty_engine, apply=True)).refused
    # The tables hold it now, so `--discard` refuses, and `--keep-tables`
    # keeps their copy with the mark the first gate listed.
    held = await import_conversations(pg_empty_engine, apply=True, discard=drop)
    assert list(held.refused) == [name]
    restarted = await import_conversations(pg_empty_engine, apply=True, keep_tables=keep)
    assert (restarted.kept, restarted.refused) == ([name], {})
    gate = await cutover_gate(pg_empty_engine)
    assert (gate.clean, gate.discarded) == (True, [name])


@_RELEASE_2_SCHEMA
async def test_keep_tables_serves_a_diverged_conversation_until_adk_moves_again(
    pg_empty_engine: AsyncEngine, release_2_schema: bool
) -> None:
    name = _name(FULL)
    await _write_adk(pg_empty_engine, RECORDED["sessions"])
    if release_2_schema:
        await _release_2_schema(pg_empty_engine)
    await _diverge(pg_empty_engine, FULL)
    # `--discard` refuses a conflict before the conflict is recorded.
    dropped = await import_conversations(pg_empty_engine, apply=True, discard=frozenset({name}))
    assert list(dropped.refused) == [name]
    async with pg_empty_engine.connect() as conn:
        marked = (
            await conn.execute(
                text("SELECT diverged FROM adk_ledger_imports WHERE conversation_id = :c"),
                {"c": FULL},
            )
        ).scalar_one()
    assert marked is False
    assert (await import_conversations(pg_empty_engine, apply=True)).conflicts == [name]
    with pytest.raises(ConversationHeld):
        await _reimporting_store(pg_empty_engine).load(PRINCIPAL, FULL)
    assert (await cutover_gate(pg_empty_engine)).first == [name]

    kept = await import_conversations(pg_empty_engine, apply=True, keep_tables=frozenset({name}))

    assert (kept.kept, kept.clean) == ([name], True)
    served = await _reimporting_store(pg_empty_engine).load(PRINCIPAL, FULL)
    assert served is not None
    assert served.auto_add != EXPECTED[FULL]["view"]["auto_add"] and served.seq != 99
    assert (await cutover_gate(pg_empty_engine)).clean
    checked = await verify(pg_empty_engine)
    assert (checked.clean, checked.kept, checked.conflicts) == (True, [name], [])

    await _set_adk_state(pg_empty_engine, FULL, {"studio:seq": 100})

    assert (await cutover_gate(pg_empty_engine)).first == [name]
    with pytest.raises(ConversationHeld):
        await _reimporting_store(pg_empty_engine).load(PRINCIPAL, FULL)
    assert (await cutover_gate(pg_empty_engine)).first == [name]


async def test_prefer_adk_takes_adks_copy_of_a_kept_conversation(
    pg_empty_engine: AsyncEngine,
) -> None:
    """The operator kept the tables' copy, then names the conversation to
    `--prefer-adk`: the job re-imports ADK's copy, the release serves it,
    and verify compares it as imported."""
    name = _name(FULL)
    await _write_adk(pg_empty_engine, RECORDED["sessions"])
    await _diverge(pg_empty_engine, FULL)
    await import_conversations(pg_empty_engine, apply=True, keep_tables=frozenset({name}))

    preferred = await import_conversations(
        pg_empty_engine, apply=True, prefer_adk=frozenset({name})
    )

    assert (preferred.reimported, preferred.kept, preferred.clean) == ([name], [], True)
    served = await _reimporting_store(pg_empty_engine).load(PRINCIPAL, FULL)
    assert served is not None and served.seq == 99
    checked = await verify(pg_empty_engine)
    assert (checked.clean, checked.kept, checked.differences) == (True, [], {})


async def test_keep_tables_refuses_a_conversation_that_imports_and_a_name_given_twice(
    adk: AsyncEngine,
) -> None:
    name = _name(TEXT_ONLY)
    await import_conversations(adk, apply=True)

    report = await import_conversations(adk, apply=True, keep_tables=frozenset({name}))

    assert report.refused == {
        name: [
            "--keep-tables keeps the tables' copy of a refused or diverged conversation, "
            "and this one is unchanged"
        ]
    }
    assert (await cutover_gate(adk)).clean
    discarded = await import_conversations(adk, apply=True, discard=frozenset({name}))
    assert discarded.refused == {
        name: [
            "--discard gives up a refused conversation the tables do not hold, "
            "and this one is unchanged"
        ]
    }
    assert await _count(adk) == 3
    for one, other in (("prefer_adk", "keep_tables"), ("keep_tables", "discard")):
        with pytest.raises(NamedTwice):
            await import_conversations(
                adk, apply=True, **{one: frozenset({name}), other: frozenset({name})}
            )
    for flag in ("keep_tables", "discard"):
        with pytest.raises(UnknownConversation, match=f"--{flag.replace('_', '-')}"):
            await import_conversations(adk, apply=True, **{flag: frozenset({"u_x/conv_x"})})


@pytest.mark.parametrize(
    ("expires_at", "streaming"),
    [(60, True), (-60, False), ("soon", False)],
    ids=["live", "lapsed", "not-a-number"],
)
async def test_verify_reports_and_the_gate_fails_on_a_ledger_session_that_holds_a_live_lease(
    adk: AsyncEngine, expires_at: int | str, streaming: bool
) -> None:
    """The previous revision's stream can still write the session. Verify
    names it and its count, and does not fail on it. The gate, run before
    that revision is deleted, fails on it."""
    await import_conversations(adk, apply=True)
    expiry = time.time() + expires_at if isinstance(expires_at, int) else expires_at
    lease = {"id": "seg_old", "owner": "old", "expires_at": expiry, "cancel": False}
    await _set_adk_state(adk, TEXT_ONLY, {"studio:segment": lease})
    await import_conversations(adk, apply=True)

    report = await verify(adk)
    gate = await cutover_gate(adk)

    assert (report.clean, report.streaming_on_adk) == (
        True,
        [_name(TEXT_ONLY)] if streaming else [],
    )
    assert gate.streaming_on_adk == ([_name(TEXT_ONLY)] if streaming else [])
    if isinstance(expires_at, int):
        # The import refuses a lease whose expiry is not a number, which the
        # gate then lists as not cleared.
        assert (gate.clean, gate.uncleared) == (not streaming, 0)
    assert ("1 stream holds its lease on ADK" in report.render().splitlines()) == streaming


async def test_adding_the_discarded_column_gives_up_on_a_table_another_reader_holds(
    pg_empty_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The column's `ALTER` waits behind a reader of the table only as long
    as its lock bound, then stops the run with nothing written."""
    name = _name(TEXT_ONLY)
    await _write_adk(pg_empty_engine, _sessions(**{TEXT_ONLY: _add(**{"studio:unknown": 1})}))
    await import_conversations(pg_empty_engine, apply=True)
    monkeypatch.setattr(adk_import, "_COLUMN_LOCK_TIMEOUT", "200ms")

    async with pg_empty_engine.connect() as reader:
        await reader.execute(text("LOCK TABLE adk_ledger_imports IN ACCESS SHARE MODE"))
        with pytest.raises(TableBusy):
            await import_conversations(pg_empty_engine, apply=True, discard=frozenset({name}))
        await reader.rollback()

    assert (await cutover_gate(pg_empty_engine)).first == [name]
