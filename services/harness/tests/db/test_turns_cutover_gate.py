"""The cutover gate: the release that stops reading ADK refuses to start,
with SQLSTATE DFA02 and nothing applied, while an ADK ledger session is not
cleared (imported or kept as of its last ADK write, 0003) or holds a live
`studio:segment` lease (0002_live_ledger_lease_gate, which sorts before 0003
and runs in its transaction). The case table is the one release 1's `gate`
verb and release 3's 0005 are pinned by."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import asyncpg
import pytest
from harness.db.migrate import MIGRATIONS_DIR, apply_pending
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

_MIGRATION = "0003_turns.sql"
# The migrations release 1 shipped.
_RELEASE_1 = ("0001_conversations.sql", "0002_adk_ledger_imports.sql")

# ADK's v1 `sessions` as google-adk 2.13 creates it on PostgreSQL
# (`google/adk/sessions/schemas/v1.py`, StorageSession).
_ADK_SESSIONS = (
    "CREATE TABLE sessions (app_name varchar(128) NOT NULL, user_id varchar(128) NOT NULL, "
    "id varchar(128) NOT NULL, state jsonb NOT NULL, create_time timestamp NOT NULL, "
    "update_time timestamp NOT NULL, PRIMARY KEY (app_name, user_id, id))"
)
_LEDGER_APP = "dfo_analyst_graph.ledger"
_UPDATED = "2026-11-01 10:05:00"
_MOVED = "2026-11-02 09:00:00"


def _session(
    cid: str, *, updated: str = _UPDATED, app: str = _LEDGER_APP, state: str = "{}"
) -> str:
    return (
        f"INSERT INTO sessions VALUES ('{app}', 'p', '{cid}', '{state}', '2026-11-01 10:00:00', "
        f"'{updated}')"
    )


def _leased(expires_at: object) -> str:
    """A ledger state whose `studio:segment` lease expires at `expires_at`,
    as the revision before the import wrote one."""
    return json.dumps(
        {"studio:segment": {"id": "seg", "owner": "old", "expires_at": expires_at, "cancel": False}}
    )


def _imported(cid: str, *, updated: str = _UPDATED) -> str:
    """The ledger row of a session the import copied."""
    return (
        "INSERT INTO adk_ledger_imports (principal, conversation_id, source_digest, "
        f"source_updated_at, imported_digest) VALUES ('p', '{cid}', 'src', '{updated}', 'rows')"
    )


def _kept(cid: str, *, updated: str = _UPDATED) -> str:
    """The ledger row of a session whose tables' copy the operator kept."""
    return (
        "INSERT INTO adk_ledger_imports (principal, conversation_id, source_updated_at, "
        f"diverged, kept_tables) VALUES ('p', '{cid}', '{updated}', true, true)"
    )


# A first-sight conflict: the tables held the conversation, so nothing was
# copied and both digests are NULL.
_PLACEHOLDER = (
    "INSERT INTO adk_ledger_imports (principal, conversation_id, source_updated_at, diverged) "
    f"VALUES ('p', 'c1', '{_UPDATED}', true)"
)


def _conversation(cid: str) -> str:
    return f"INSERT INTO conversations (principal, conversation_id) VALUES ('p', '{cid}')"


async def _migrate_before_0003(engine: AsyncEngine, tmp_path: Path) -> None:
    """Apply the migrations release 1 shipped, 0001 and 0002."""
    for name in _RELEASE_1:
        shutil.copy(MIGRATIONS_DIR / name, tmp_path / name)
    await apply_pending(engine, directory=tmp_path)


async def _execute(engine: AsyncEngine, *statements: str) -> None:
    async with engine.begin() as conn:
        for statement in statements:
            await conn.execute(text(statement))


async def _applied(engine: AsyncEngine) -> list[str]:
    """The migrations recorded as applied, none before the bookkeeping table
    exists."""
    async with engine.connect() as conn:
        if not (
            await conn.execute(text("SELECT to_regclass('harness_schema_migrations')"))
        ).scalar():
            return []
        rows = await conn.execute(
            text("SELECT version FROM harness_schema_migrations ORDER BY version")
        )
        return [row[0] for row in rows]


async def _refused(engine: AsyncEngine) -> str:
    """0003 raises DFA02 and applies nothing. Returns the error's message."""
    before = await _applied(engine)
    with pytest.raises(asyncpg.PostgresError) as refusal:
        await apply_pending(engine)
    assert refusal.value.sqlstate == "DFA02"
    assert await _applied(engine) == before
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT to_regclass('turn_cursors')"))).scalar() is None
    return str(refusal.value)


@pytest.mark.parametrize(
    ("tables", "rows", "uncleared"),
    [
        pytest.param(
            (_ADK_SESSIONS,),
            (
                *(_session("c1"), _session("c2"), _imported("c1"), _imported("c2")),
                *(_conversation("c1"), _conversation("c2")),
            ),
            [],
            id="complete import",
        ),
        pytest.param(
            (_ADK_SESSIONS,),
            (_session("c1"), _session("c2"), _imported("c1"), _conversation("c1")),
            ["p/c2"],
            id="one of two imported",
        ),
        pytest.param(
            (_ADK_SESSIONS,),
            (_session("c1"), _PLACEHOLDER, _conversation("c1")),
            ["p/c1"],
            id="conflict placeholder only",
        ),
        pytest.param(
            (_ADK_SESSIONS,),
            (_session("c1", updated=_MOVED), _imported("c1"), _conversation("c1")),
            ["p/c1"],
            id="ADK moved after the import",
        ),
        pytest.param((_ADK_SESSIONS,), (_session("c1"), _kept("c1")), [], id="kept, current"),
        pytest.param(
            (_ADK_SESSIONS,),
            (_session("c1", updated=_MOVED), _kept("c1")),
            ["p/c1"],
            id="kept, ADK moved after it",
        ),
        pytest.param(
            (_ADK_SESSIONS,),
            (
                _session("c1", state=_leased(time.time() + 600)),
                *(_imported("c1"), _conversation("c1")),
            ),
            ["p/c1"],
            id="imported, a live lease on ADK",
        ),
        pytest.param(
            (_ADK_SESSIONS,),
            (
                _session("c1", state=_leased(time.time() - 60)),
                *(_imported("c1"), _conversation("c1")),
            ),
            [],
            id="imported, a lapsed lease on ADK",
        ),
        pytest.param(
            (_ADK_SESSIONS,),
            (_session("c1", state=_leased("soon")), _kept("c1")),
            [],
            id="kept, a lease expiry that is not a number",
        ),
        pytest.param((), (), [], id="no sessions table"),
        pytest.param(
            ("CREATE TABLE sessions (id text PRIMARY KEY, app_name text)",),
            (f"INSERT INTO sessions VALUES ('c1', '{_LEDGER_APP}')",),
            [],
            id="foreign sessions table",
        ),
        pytest.param(
            (_ADK_SESSIONS,),
            (_session("c1#t1", app="dfo_analyst_graph"),),
            [],
            id="sessions of other apps only",
        ),
    ],
)
async def test_0003_waits_for_every_ledger_session_to_be_imported_or_kept(
    pg_empty_engine: AsyncEngine,
    tmp_path: Path,
    tables: tuple[str, ...],
    rows: tuple[str, ...],
    uncleared: list[str],
) -> None:
    await _migrate_before_0003(pg_empty_engine, tmp_path)
    await _execute(pg_empty_engine, *tables, *rows)

    if not uncleared:
        assert _MIGRATION in await apply_pending(pg_empty_engine)
        return
    message = await _refused(pg_empty_engine)
    assert message.startswith(f"{len(uncleared)} ledger session")
    assert f": {', '.join(uncleared)}." in message


async def test_the_dfa02_message_names_the_first_ten_and_the_commands_that_clear_them(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    await _migrate_before_0003(pg_empty_engine, tmp_path)
    await _execute(pg_empty_engine, _ADK_SESSIONS, *(_session(f"c{n:02}") for n in range(12)))

    message = await _refused(pg_empty_engine)

    assert message.startswith("12 ledger session")
    assert f": {', '.join(f'p/c{n:02}' for n in range(10))}, ...." in message
    for command in ("import --apply", "--prefer-adk", "--keep-tables", "`gate`"):
        assert command in message


async def test_one_run_from_no_harness_migration_refuses_a_never_imported_ledger_session(
    pg_empty_engine: AsyncEngine,
) -> None:
    """0001 to 0004 apply in one run, so 0002's empty ledger clears nothing,
    and the run applies none of them."""
    await _execute(pg_empty_engine, _ADK_SESSIONS, _session("c1"), _session("c2"))

    message = await _refused(pg_empty_engine)

    assert message.startswith("2 ledger session")
    assert await _applied(pg_empty_engine) == []
    async with pg_empty_engine.connect() as conn:
        assert (await conn.execute(text("SELECT to_regclass('conversations')"))).scalar() is None


_LEASE_GATE = "0002_live_ledger_lease_gate.sql"


async def test_the_lease_gate_runs_in_the_stage_2_transaction_before_0003(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    """Production at the stage-2 deploy: release 1 applied 0001 and 0002. The
    lease gate, 0003 and 0004 are pending and apply in one run, the gate
    first. A live lease refuses the run, which records and creates nothing;
    once it lapses the same run applies all three in that order."""
    await _migrate_before_0003(pg_empty_engine, tmp_path)
    leased = _session("c1", state=_leased(time.time() + 600))
    await _execute(pg_empty_engine, _ADK_SESSIONS, leased, _imported("c1"), _conversation("c1"))

    message = await _refused(pg_empty_engine)

    assert "hold a live `studio:segment` lease: p/c1." in message
    assert await _applied(pg_empty_engine) == sorted(_RELEASE_1)
    await _execute(
        pg_empty_engine,
        f"UPDATE sessions SET state = '{_leased(time.time() - 60)}'::jsonb",
    )
    ran = await apply_pending(pg_empty_engine)
    assert ran[:3] == [_LEASE_GATE, _MIGRATION, "0004_card_lineage.sql"]


async def test_the_lease_gate_applies_alone_and_checks_nothing_after_0005(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    """Dev: 0001 to 0005 were applied before the lease gate shipped, and
    0005 archived ADK's `sessions`. The gate sorts below applied files, which
    `apply_pending` allows for a file not yet applied: it runs alone, finds
    no `sessions` table, records itself, and a second run applies nothing.
    A live lease left in the archive is not read."""
    await _migrate_before_0003(pg_empty_engine, tmp_path)
    await _execute(pg_empty_engine, _ADK_SESSIONS, _session("c1"), _imported("c1"))
    await _execute(pg_empty_engine, _conversation("c1"))
    for path in MIGRATIONS_DIR.glob("*.sql"):
        if path.name != _LEASE_GATE:
            shutil.copy(path, tmp_path / path.name)
    assert await apply_pending(pg_empty_engine, directory=tmp_path) == [
        _MIGRATION,
        "0004_card_lineage.sql",
        "0005_rename_adk_tables.sql",
    ]
    await _execute(
        pg_empty_engine,
        f"UPDATE adk_archive_sessions SET state = '{_leased(time.time() + 600)}'::jsonb",
    )

    assert await apply_pending(pg_empty_engine) == [_LEASE_GATE]
    assert await apply_pending(pg_empty_engine) == []
    assert _LEASE_GATE in await _applied(pg_empty_engine)
