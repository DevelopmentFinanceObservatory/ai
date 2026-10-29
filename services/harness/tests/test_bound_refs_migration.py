"""`harness.bound_refs_migration` over a sqlite-backed ADK session store.

The stored block is the G7 part's spec block as the release before
`BoundEntity` kept it: `_bound_refs` maps each entity text, case-folded, to
the refs its `entity_refs` row bound.
"""

from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
from collections.abc import AsyncIterator
from contextlib import closing
from pathlib import Path
from typing import Any

import harness.bound_refs_migration as migration
import pytest
from compiler.evidence import bound_refs_of
from compiler.sources import source_from_spec_block
from google.adk.events import Event, EventActions
from google.adk.sessions import DatabaseSessionService
from harness.bound_refs_migration import (
    LEDGER_APP,
    TURN_APP,
    Outcome,
    main,
    migrate_block,
    run_migration,
)
from harness.graph.state import ORCHESTRATOR_SPECS_KEY, store_search
from harness.tool_contracts import resolver_digest
from planner.plan import Entity

from .sqlite_session import sqlite_session_service

_G7_PART = json.loads(
    (Path(__file__).parent / "fixtures" / "search_part_g7_africa.json").read_text()
)["envelope"]
_ENTITIES = (Entity(name="G7", role="donor"), Entity(name="Africa", role="recipient"))


def _release_block() -> dict[str, Any]:
    """The block this release keeps for the G7 part, `_bound_refs` as `BoundEntity` rows."""
    state: dict[str, Any] = {}
    store_search(state, resolver_digest(_G7_PART))
    (block,) = state[ORCHESTRATOR_SPECS_KEY].values()
    return block


def _mapping_block() -> dict[str, Any]:
    """The same block as the release before `BoundEntity` stored it."""
    return {**_release_block(), "_bound_refs": {"g7": ["GroupOfSeven"], "africa": ["DACAfrica"]}}


def _refs(block: dict[str, Any]) -> list[tuple[str, ...]]:
    source = source_from_spec_block(block)
    return [bound_refs_of(entity, source) for entity in _ENTITIES]


@pytest.fixture
async def service(tmp_path: Path) -> AsyncIterator[DatabaseSessionService]:
    service = sqlite_session_service(tmp_path / "sessions.db")
    yield service
    await service.close()


async def _seed(service: DatabaseSessionService) -> None:
    """A ledger whose turn record keeps the block, an open turn holding it
    under the orchestrator's specs and a planner pin, and a ledger already
    in list form."""
    block = _mapping_block()
    spec_id = block["spec_id"]
    await service.create_session(
        app_name=LEDGER_APP,
        user_id="principal",
        session_id="conv",
        state={
            "studio:turn_index": 1,
            "studio:turn:0": {"user": "G7 aid", "specs": {spec_id: block}},
        },
    )
    await service.create_session(
        app_name=TURN_APP,
        user_id="principal",
        session_id="conv#t1",
        state={
            ORCHESTRATOR_SPECS_KEY: {spec_id: block},
            "planner:pin_sources": {spec_id: {"block": block, "settled": True}},
        },
    )
    await service.create_session(
        app_name=LEDGER_APP,
        user_id="principal",
        session_id="listed",
        state={"studio:turn:0": {"specs": {spec_id: _release_block()}}},
    )


async def _state(service: DatabaseSessionService, app_name: str, session_id: str) -> dict[str, Any]:
    session = await service.get_session(
        app_name=app_name, user_id="principal", session_id=session_id
    )
    assert session is not None
    return session.state


async def test_apply_rewrites_every_held_block_so_the_release_reads_the_refs_it_bound(
    service: DatabaseSessionService,
) -> None:
    await _seed(service)
    spec_id = _release_block()["spec_id"]
    with pytest.raises(ValueError):
        source_from_spec_block(_mapping_block())

    held = await service.get_session(app_name=LEDGER_APP, user_id="principal", session_id="conv")
    assert held is not None

    report = await run_migration(engine=service.db_engine, apply=True)

    # The write keeps the revision marker, so an instance holding the session appends on.
    await service.append_event(
        held, Event(author="studio", actions=EventActions(state_delta={"studio:seq": 1}))
    )
    ledger = await _state(service, LEDGER_APP, "conv")
    turn = await _state(service, TURN_APP, "conv#t1")
    stored = [
        ledger["studio:turn:0"]["specs"][spec_id],
        turn[ORCHESTRATOR_SPECS_KEY][spec_id],
        turn["planner:pin_sources"][spec_id]["block"],
    ]
    fresh = _refs(_release_block())
    assert fresh == [("GroupOfSeven",), ("DACAfrica",)]
    assert [_refs(block) for block in stored] == [fresh] * 3
    assert ledger["studio:turn_index"] == 1
    assert report.written == 2
    assert report.blocks == {
        (f"{LEDGER_APP} studio:turn:*", Outcome.CONVERTED): 1,
        (f"{LEDGER_APP} studio:turn:*", Outcome.ALREADY_LIST): 1,
        (f"{TURN_APP} {ORCHESTRATOR_SPECS_KEY}", Outcome.CONVERTED): 1,
        (f"{TURN_APP} planner:pin_sources", Outcome.CONVERTED): 1,
    }

    again = await run_migration(engine=service.db_engine, apply=True)

    assert again.written == 0
    assert set(again.blocks) == {(where, Outcome.ALREADY_LIST) for where, _ in report.blocks}
    assert await _state(service, LEDGER_APP, "conv") == ledger
    assert ledger["studio:seq"] == 1


async def test_a_stale_holders_write_back_is_converted_by_the_next_run(
    service: DatabaseSessionService,
) -> None:
    await _seed(service)
    held = await service.get_session(app_name=TURN_APP, user_id="principal", session_id="conv#t1")
    assert held is not None
    await run_migration(engine=service.db_engine, apply=True)

    # A turn in flight across the run stores a search as `store_search` does,
    # rewriting the whole of the specs it loaded before the run.
    specs = {**held.state[ORCHESTRATOR_SPECS_KEY], "spec_new": _release_block()}
    delta = {ORCHESTRATOR_SPECS_KEY: specs}
    await service.append_event(
        held, Event(author="orchestrator", actions=EventActions(state_delta=delta))
    )
    rerun = await run_migration(engine=service.db_engine, apply=True)

    assert rerun.blocks[f"{TURN_APP} {ORCHESTRATOR_SPECS_KEY}", Outcome.CONVERTED] == 1
    assert rerun.written == 1
    turn = await _state(service, TURN_APP, "conv#t1")
    assert all(
        _refs(block) == _refs(_release_block()) for block in turn[ORCHESTRATOR_SPECS_KEY].values()
    )


@pytest.mark.parametrize(("unconvertible", "status"), [(False, 0), (True, 1)])
def test_the_default_run_writes_nothing_and_fails_on_an_unconvertible_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unconvertible: bool, status: int
) -> None:
    async def ledger(*, seed: bool) -> tuple[dict[str, Any], float]:
        service = sqlite_session_service(tmp_path / "sessions.db")
        try:
            if seed:
                await _seed(service)
                if unconvertible:
                    await service.create_session(
                        app_name=TURN_APP,
                        user_id="principal",
                        session_id="bad#t1",
                        state={ORCHESTRATOR_SPECS_KEY: {"spec_bad": _unreadable_block()}},
                    )
            session = await service.get_session(
                app_name=LEDGER_APP, user_id="principal", session_id="conv"
            )
            assert session is not None
            return session.state, session.last_update_time
        finally:
            await service.close()

    before = asyncio.run(ledger(seed=True))
    # Settings read no developer `.env` and no other store's variables.
    monkeypatch.chdir(tmp_path)
    for name in (
        "SESSION_STORE_INSTANCE_CONNECTION_NAME",
        "SESSION_STORE_DATABASE",
        "SESSION_STORE_RUNTIME_DB_USER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}")

    assert main([]) == status

    assert asyncio.run(ledger(seed=False)) == before


async def test_a_session_written_during_the_run_is_left_for_a_rerun(
    service: DatabaseSessionService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed(service)
    migrate_state = migration._migrate_state

    def racing(app_name: str, session_id: str, state: Any, report: Any) -> dict[str, Any]:
        # Another instance appends to the ledger between the job's read and its write.
        if session_id == "conv":
            with closing(sqlite3.connect(tmp_path / "sessions.db")) as db, db:
                db.execute(
                    "UPDATE sessions SET state = json_set(state, '$.\"studio:seq\"', 7),"
                    " update_time = '2099-01-01 00:00:00.000000' WHERE id = 'conv'"
                )
        return migrate_state(app_name, session_id, state, report)

    monkeypatch.setattr(migration, "_migrate_state", racing)

    report = await run_migration(engine=service.db_engine, apply=True)

    assert report.changed_during_run == [f"{LEDGER_APP} conv"]
    assert report.written == 1
    ledger = await _state(service, LEDGER_APP, "conv")
    assert ledger["studio:seq"] == 7
    assert ledger["studio:turn:0"]["specs"] == {_release_block()["spec_id"]: _mapping_block()}


@pytest.mark.parametrize(
    ("bound", "rows", "dropped"),
    [
        (
            {"g7": ["GroupOfSeven", "GroupOfSeven"], "": ["x"], "congo": []},
            [{"identity": "g7", "text": "g7", "role": None, "refs": ["GroupOfSeven"]}],
            2,
        ),
        ({"congo": [], "chad": [3], "mali": "country/MLI"}, None, 3),
    ],
)
def test_entries_the_mapping_read_as_absent_are_dropped(
    bound: dict[str, Any], rows: list[dict[str, Any]] | None, dropped: int
) -> None:
    result = migrate_block({**_release_block(), "_bound_refs": bound})

    assert result.outcome is Outcome.CONVERTED
    assert result.block.get("_bound_refs") == rows
    assert result.dropped == dropped


def _unreadable_block() -> dict[str, Any]:
    """The mapping block with an entity the reader refuses, lacking its value and label."""
    block = copy.deepcopy(_mapping_block())
    block["resolved"]["entities"] = [{"role": "origin"}]
    return block


def _failing_reader(block: Any) -> Any:
    raise RuntimeError


@pytest.mark.parametrize(
    ("block", "reader", "error"),
    [
        (_unreadable_block(), source_from_spec_block, "ValueError"),
        ({**_release_block(), "_bound_refs": "g7"}, source_from_spec_block, "TypeError"),
        (_mapping_block(), _failing_reader, "RuntimeError"),
    ],
)
def test_a_block_that_does_not_convert_is_left_as_stored_naming_the_error(
    block: dict[str, Any], reader: Any, error: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(migration, "source_from_spec_block", reader)

    result = migrate_block(block)

    assert (result.outcome, result.error) == (Outcome.UNCONVERTIBLE, error)
    assert result.block is block


async def test_a_run_reports_each_unconvertible_block_and_converts_the_rest(
    service: DatabaseSessionService,
) -> None:
    await _seed(service)
    await service.create_session(
        app_name=TURN_APP,
        user_id="principal",
        session_id="bad#t1",
        state={ORCHESTRATOR_SPECS_KEY: {"spec_bad": _unreadable_block()}},
    )

    report = await run_migration(engine=service.db_engine, apply=True)

    where = f"{TURN_APP} {ORCHESTRATOR_SPECS_KEY}"
    spec_id = _release_block()["spec_id"]
    assert report.unconvertible == [("bad#t1", where, spec_id, "ValueError")]
    assert report.blocks[where, Outcome.CONVERTED] == 1
    assert report.written == 2
