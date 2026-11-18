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
import time
from pathlib import Path
from typing import Any

import pytest
from compiler.sources import source_from_spec_block
from google.adk.sessions import DatabaseSessionService
from harness import adk_import
from harness.adk_import import (
    IMPORTED_SEQ_MARGIN,
    LEDGER_APP,
    AdkSchemaError,
    TablesMissing,
    UnknownConversation,
    _stored_digest,
    close_open_turns,
    import_conversations,
    main,
    open_turn_specs,
    reimport_on_access,
    verify,
)
from harness.conversation import current_conversation_key
from harness.db.migrate import apply_pending
from harness.graph.state import AGENT_NAME, ORCHESTRATOR_SPECS_KEY, store_search
from harness.run_ledger import StudioRunLedger
from harness.studio_state import ConversationState, SegmentLost
from harness.studio_store import ConversationHeld, StudioStore, _serialize_pending
from harness.tool_contracts import resolver_digest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .graph_turns import agent_read
from .studio_routes import AUTH, orchestrator_app, stream, studio_client, turn_body
from .turn_driver import NO_ADK

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
    return StudioStore(engine, adk=NO_ADK, turn_specs=open_turn_specs(engine))


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


async def test_verify_counts_equal_conversations_and_names_each_difference(
    adk: AsyncEngine,
) -> None:
    await import_conversations(adk, apply=True)

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


async def test_a_rerun_leaves_unmoved_conversations_and_reimports_what_only_adk_changed(
    adk: AsyncEngine,
) -> None:
    await import_conversations(adk, apply=True)
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


async def test_close_open_turns_closes_each_paused_adk_turn_as_abandoned_with_its_blocks(
    adk: AsyncEngine,
) -> None:
    await import_conversations(adk, apply=True)
    open_turns = [f"{_name(FULL)} turn 4", f"{_name(PLAN_PAUSED)} turn 1"]

    dry = await close_open_turns(adk, apply=False)

    assert (dry.closed, dry.live, dry.not_on_adk, dry.changed) == (open_turns, [], [], [])
    still_open = await _load(adk, FULL)
    assert still_open is not None and not still_open.turn_recorded

    applied = await close_open_turns(adk, apply=True)

    assert (applied.closed, applied.clean) == (open_turns, True)
    for conversation_id in (FULL, PLAN_PAUSED):
        state = await _load(adk, conversation_id)
        assert state is not None
        want = {**EXPECTED[conversation_id]["closed_view"], "turn_specs": {}}
        assert _strip(_view(state)) == _strip(want), conversation_id
    closed = await _load(adk, FULL)
    assert closed is not None
    assert closed.turns[-1]["error"] == "abandoned"
    assert closed.turns[-1]["specs"] == EXPECTED[FULL]["view"]["turn_specs"]
    assert (await close_open_turns(adk, apply=True)).closed == []


async def test_close_open_turns_leaves_a_turn_a_stream_holds_or_that_never_ran_on_adk(
    adk: AsyncEngine,
) -> None:
    await import_conversations(adk, apply=True)
    async with adk.begin() as conn:
        await conn.execute(
            text(
                "UPDATE conversations SET lease_segment_id = 'seg_live', lease_owner = 'other', "
                "lease_expires_at = extract(epoch FROM now()) + 600 "
                "WHERE conversation_id = :plan_paused"
            ),
            {"plan_paused": PLAN_PAUSED},
        )
        await conn.execute(
            text("DELETE FROM sessions WHERE app_name = :app AND id = :id"),
            {"app": AGENT_NAME, "id": f"{FULL}#t4"},
        )

    report = await close_open_turns(adk, apply=True)

    assert (report.closed, report.live, report.not_on_adk) == (
        [],
        [f"{_name(PLAN_PAUSED)} turn 1"],
        [f"{_name(FULL)} turn 4"],
    )
    for conversation_id in (FULL, PLAN_PAUSED):
        state = await _load(adk, conversation_id)
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
            out.splitlines()[-1] == "2 imported, 0 re-imported, 0 unchanged, 0 conflicts, 1 refused"
        )
        assert await _count(engine) == 2
        assert await asyncio.to_thread(main, ["import", "--verify"]) == 1
        assert capsys.readouterr().out.splitlines()[-1] == "2 of 3 equal"
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


async def test_a_mapping_form_block_is_imported_as_rows_and_verifies_equal(
    pg_empty_engine: AsyncEngine,
) -> None:
    mapping = _mapping_block()
    await _write_adk(pg_empty_engine, _sessions(**{FULL: _with_turn_specs(mapping)}))

    report = await import_conversations(pg_empty_engine, apply=True)

    assert (report.clean, report.converted_blocks, report.unconvertible_blocks) == (True, 1, 0)
    assert "1 converted, 0 unconvertible" in report.render()
    state = await _load(pg_empty_engine, FULL)
    assert state is not None
    stored = state.turns[0]["specs"][mapping["spec_id"]]
    assert _rows(stored) == [("g7", ("GroupOfSeven",)), ("africa", ("DACAfrica",))]
    assert (await verify(pg_empty_engine)).clean


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


async def test_close_open_turns_keeps_a_mapping_form_turn_block_as_rows(
    adk: AsyncEngine,
) -> None:
    await import_conversations(adk, apply=True)
    spec_id = next(iter(EXPECTED[FULL]["view"]["turn_specs"]))
    await _put_open_turn_block(adk, {**_mapping_block(), "spec_id": spec_id})

    await close_open_turns(adk, apply=True)

    closed = await _load(adk, FULL)
    assert closed is not None and closed.turns[-1]["error"] == "abandoned"
    assert _rows(closed.turns[-1]["specs"][spec_id]) == [
        ("g7", ("GroupOfSeven",)),
        ("africa", ("DACAfrica",)),
    ]


async def test_close_open_turns_closes_a_turn_with_a_block_that_does_not_convert_as_stored(
    adk: AsyncEngine,
) -> None:
    await import_conversations(adk, apply=True)
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

    report = await close_open_turns(adk, apply=True)

    turn = f"{_name(FULL)} turn 4"
    assert (report.clean, report.refused) == (True, {})
    assert turn in report.closed
    (finding,) = report.findings[turn]
    assert finding.endswith("does not convert: TypeError: _bound_refs is of type str")
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
    assert (await verify(adk)).clean
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

    report = await close_open_turns(adk, apply=True)

    assert (report.clean, report.diverged) == (False, [_name(FULL)])
    assert report.closed == [f"{_name(PLAN_PAUSED)} turn 1"]
    assert f"left alone, diverged (--prefer-adk first): {_name(FULL)}" in report.render()
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
    store = _reimporting_store(adk)
    app, model = orchestrator_app(engine=adk, store=store)

    async with studio_client(app) as client:
        frames = await stream(
            client, "/api/studio/turn", turn_body("and a fourth?", conversation_id=TEXT_ONLY)
        )

    assert frames[-1]["event"] == "done"
    assert "a third question" in agent_read(model.requests[0])
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
    app, _ = orchestrator_app(engine=adk, store=_reimporting_store(adk))
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
    app, _ = orchestrator_app(engine=adk, store=_reimporting_store(adk))
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
    store = _reimporting_store(pg_empty_engine)
    app, _ = orchestrator_app(engine=pg_empty_engine, store=store)

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
    session's time no longer tells that ADK moved; the diverged flag does."""
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
