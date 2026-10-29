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
    with pytest.raises(ValueError, match="valid tuple"):
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


def test_the_default_run_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def ledger(*, seed: bool) -> tuple[dict[str, Any], float]:
        service = sqlite_session_service(tmp_path / "sessions.db")
        try:
            if seed:
                await _seed(service)
            session = await service.get_session(
                app_name=LEDGER_APP, user_id="principal", session_id="conv"
            )
            assert session is not None
            return session.state, session.last_update_time
        finally:
            await service.close()

    before = asyncio.run(ledger(seed=True))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}")

    assert main([]) == 0

    assert asyncio.run(ledger(seed=False)) == before
    assert "would be converted: 3" in capsys.readouterr().out


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
        ({"congo": [], "chad": [3]}, None, 2),
    ],
)
def test_entries_the_mapping_read_as_absent_are_dropped(
    bound: dict[str, Any], rows: list[dict[str, Any]] | None, dropped: int
) -> None:
    result = migrate_block({**_release_block(), "_bound_refs": bound})

    assert result.outcome is Outcome.CONVERTED
    assert result.block.get("_bound_refs") == rows
    assert result.dropped == dropped


def test_a_block_the_reader_refuses_after_conversion_is_left_as_stored() -> None:
    block = copy.deepcopy(_mapping_block())
    block["resolved"]["entities"] = [{"role": "origin"}]

    result = migrate_block(block)

    assert result.outcome is Outcome.UNCONVERTIBLE
    assert result.block is block
