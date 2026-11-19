from __future__ import annotations

import contextlib
import json
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from harness.db.migrate import MIGRATIONS_DIR, apply_pending
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

_MIGRATION = "0005_rename_adk_tables.sql"
_ADK_TABLES = ("sessions", "events", "app_states", "user_states", "adk_internal_metadata")
_ARCHIVED = {*_ADK_TABLES, "adk_ledger_imports"}

# ADK's v1 tables as google-adk 2.13 creates them on PostgreSQL, from
# `google/adk/sessions/schemas/v1.py`: StorageSession (`sessions`),
# StorageEvent (`events`, with its foreign key to `sessions` and its index),
# StorageAppState (`app_states`), StorageUserState (`user_states`) and
# StorageMetadata (`adk_internal_metadata`). Key columns are varchar(128),
# `invocation_id` and metadata values varchar(256), JSON columns jsonb and
# timestamps `timestamp without time zone`.
_ADK_V1_DDL = (
    "CREATE TABLE adk_internal_metadata "
    '("key" varchar(128) NOT NULL, value varchar(256) NOT NULL, PRIMARY KEY ("key"))',
    "CREATE TABLE sessions (app_name varchar(128) NOT NULL, user_id varchar(128) NOT NULL, "
    "id varchar(128) NOT NULL, state jsonb NOT NULL, create_time timestamp NOT NULL, "
    "update_time timestamp NOT NULL, PRIMARY KEY (app_name, user_id, id))",
    "CREATE TABLE events (id varchar(128) NOT NULL, app_name varchar(128) NOT NULL, "
    "user_id varchar(128) NOT NULL, session_id varchar(128) NOT NULL, "
    "invocation_id varchar(256) NOT NULL, timestamp timestamp NOT NULL, event_data jsonb, "
    "PRIMARY KEY (id, app_name, user_id, session_id), "
    "FOREIGN KEY (app_name, user_id, session_id) "
    "REFERENCES sessions (app_name, user_id, id) ON DELETE CASCADE)",
    "CREATE INDEX idx_events_app_user_session_ts_id "
    "ON events (app_name, user_id, session_id, timestamp DESC, id DESC)",
    "CREATE TABLE app_states (app_name varchar(128) NOT NULL, state jsonb NOT NULL, "
    "update_time timestamp NOT NULL, PRIMARY KEY (app_name))",
    "CREATE TABLE user_states (app_name varchar(128) NOT NULL, user_id varchar(128) NOT NULL, "
    "state jsonb NOT NULL, update_time timestamp NOT NULL, PRIMARY KEY (app_name, user_id))",
)
_LEDGER_APP = "dfo_analyst_graph.ledger"
_UPDATED = "2026-11-01 10:05:00"


def _session(cid: str, *, updated: str = _UPDATED, app: str = _LEDGER_APP) -> str:
    return (
        f"INSERT INTO sessions VALUES ('{app}', 'p', '{cid}', '{{}}', '2026-11-01 10:00:00', "
        f"'{updated}')"
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
    turn_state = json.dumps({"turn_index": 3, "turn_recorded": False})
    return (
        "INSERT INTO conversations (principal, conversation_id, turn_state, lease_cancel, "
        f"version) VALUES ('p', '{cid}', '{turn_state}', true, 5)"
    )


_ADK_ROWS = (
    "INSERT INTO adk_internal_metadata VALUES ('schema_version', '1')",
    "INSERT INTO sessions VALUES ('dfo_analyst_graph.ledger', 'p', 'c1', "
    "'{\"studio:turn:0003\": 1}', '2026-11-01 10:00:00', '2026-11-01 10:05:00')",
    "INSERT INTO events VALUES ('e1', 'dfo_analyst_graph.ledger', 'p', 'c1', 'inv', "
    "'2026-11-01 10:01:00', '{\"a\": [1, 2]}')",
    "INSERT INTO app_states VALUES ('dfo_analyst_graph', '{}', '2026-11-01 10:00:00')",
    "INSERT INTO user_states VALUES ('dfo_analyst_graph', 'p', '{\"k\": 1}', '2026-11-01 10:00:00')",
    _imported("c1"),
    _conversation("c1"),
)


async def _migrate_through_0004(engine: AsyncEngine, tmp_path: Path) -> None:
    for path in MIGRATIONS_DIR.glob("*.sql"):
        if path.name < _MIGRATION:
            shutil.copy(path, tmp_path / path.name)
    await apply_pending(engine, directory=tmp_path)


async def _execute(engine: AsyncEngine, *statements: str) -> None:
    async with engine.begin() as conn:
        for statement in statements:
            await conn.execute(text(statement))


async def _tables(engine: AsyncEngine, schema: str | None = None) -> set[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = coalesce(:schema, current_schema())"
            ),
            {"schema": schema},
        )
        return {row[0] for row in rows}


async def _rows(engine: AsyncEngine, table: str) -> list[dict[str, Any]]:
    async with engine.connect() as conn:
        result = await conn.execute(text(f"SELECT * FROM {table} ORDER BY 1, 2"))
        return [dict(row._mapping) for row in result]


async def _stays_and_renames(engine: AsyncEngine, *, stays: set[str]) -> None:
    """Every archived table not in `stays` is renamed, and no table is lost."""
    tables = await _tables(engine)
    renamed = _ARCHIVED - stays
    assert {f"adk_archive_{name}" for name in renamed} <= tables
    assert not renamed & tables
    assert stays <= tables


async def _foreign_key_target(engine: AsyncEngine) -> str | None:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT confrelid::regclass::text FROM pg_constraint "
                    "WHERE conrelid = 'adk_archive_events'::regclass AND contype = 'f'"
                )
            )
        ).scalar()


async def test_an_adk_database_renames_all_six_tables_with_rows_and_foreign_key_intact(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    await _migrate_through_0004(pg_empty_engine, tmp_path)
    await _execute(pg_empty_engine, *_ADK_V1_DDL, *_ADK_ROWS)
    before = {table: await _rows(pg_empty_engine, table) for table in _ARCHIVED}
    conversations = await _rows(pg_empty_engine, "conversations")

    assert await apply_pending(pg_empty_engine) == [_MIGRATION]

    await _stays_and_renames(pg_empty_engine, stays={"conversation_card_lineage"})
    for table in _ARCHIVED:
        assert before[table]
        assert await _rows(pg_empty_engine, f"adk_archive_{table}") == before[table]
    assert await _rows(pg_empty_engine, "conversations") == conversations
    assert await _foreign_key_target(pg_empty_engine) == "adk_archive_sessions"


async def test_a_fresh_database_renames_only_the_empty_import_ledger(
    pg_empty_engine: AsyncEngine,
) -> None:
    ran = await apply_pending(pg_empty_engine)

    assert ran[-1] == _MIGRATION
    tables = await _tables(pg_empty_engine)
    assert "adk_archive_adk_ledger_imports" in tables
    assert not _ARCHIVED & tables


async def test_adks_v0_layout_without_internal_metadata_is_renamed(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    """v0's `events` has `author` and the other event columns, not `event_data`."""
    await _migrate_through_0004(pg_empty_engine, tmp_path)
    await _execute(
        pg_empty_engine,
        _ADK_V1_DDL[1],
        "CREATE TABLE events (id varchar(128) NOT NULL, app_name varchar(128) NOT NULL, "
        "user_id varchar(128) NOT NULL, session_id varchar(128) NOT NULL, "
        "invocation_id varchar(256), author varchar(256), timestamp timestamp NOT NULL, "
        "PRIMARY KEY (id, app_name, user_id, session_id))",
    )

    assert await apply_pending(pg_empty_engine) == [_MIGRATION]

    tables = await _tables(pg_empty_engine)
    assert {"adk_archive_sessions", "adk_archive_events"} <= tables
    assert not {"sessions", "events"} & tables


async def test_a_foreign_events_table_beside_no_adk_tables_stays_and_the_ledger_renames(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    await _migrate_through_0004(pg_empty_engine, tmp_path)
    await _execute(pg_empty_engine, "CREATE TABLE events (id text PRIMARY KEY, payload text)")

    assert await apply_pending(pg_empty_engine) == [_MIGRATION]

    tables = await _tables(pg_empty_engine)
    assert "events" in tables
    assert "adk_archive_events" not in tables
    assert "adk_archive_adk_ledger_imports" in tables


@contextlib.asynccontextmanager
async def _runtime_user(
    engine: AsyncEngine, *, owns: tuple[str, ...]
) -> AsyncIterator[AsyncEngine]:
    """An engine connected as a role holding only `USAGE, CREATE` on the test
    schema and owning `owns`, as the deployed runtime user does."""
    async with engine.connect() as conn:
        schema = (await conn.execute(text("SELECT current_schema()"))).scalar()
    role = f"{schema}_runtime"
    await _execute(
        engine,
        f"CREATE ROLE {role} NOSUPERUSER NOCREATEDB NOCREATEROLE",
        f"GRANT USAGE, CREATE ON SCHEMA {schema} TO {role}",
        *(f"ALTER TABLE {name} OWNER TO {role}" for name in owns),
    )
    runtime = create_async_engine(
        engine.url, connect_args={"server_settings": {"search_path": schema, "role": role}}
    )
    try:
        yield runtime
    finally:
        await runtime.dispose()
        await _execute(engine, f"DROP OWNED BY {role}", f"DROP ROLE {role}")


async def test_a_role_holding_only_usage_and_create_on_its_schema_runs_the_migration(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    """A rename needs ownership and `CREATE` on the schema, where `CREATE SCHEMA`
    would need more."""
    await _migrate_through_0004(pg_empty_engine, tmp_path)
    await _execute(pg_empty_engine, *_ADK_V1_DDL, *_ADK_ROWS)

    async with _runtime_user(
        pg_empty_engine, owns=(*_ARCHIVED, "harness_schema_migrations")
    ) as runtime:
        # DFA02 reads `conversations`, which the deployed runtime user owns.
        await _execute(pg_empty_engine, "GRANT SELECT ON conversations TO PUBLIC")
        assert await apply_pending(runtime) == [_MIGRATION]


async def test_a_table_the_current_user_does_not_own_refuses_at_once(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    await _migrate_through_0004(pg_empty_engine, tmp_path)
    await _execute(pg_empty_engine, *_ADK_V1_DDL, *_ADK_ROWS)
    owns = (*(_ARCHIVED - {"sessions"}), "harness_schema_migrations")

    async with _runtime_user(pg_empty_engine, owns=owns) as runtime:
        await _refused(runtime, "DFA03")


async def test_a_mixed_layout_refuses_and_leaves_every_table(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    await _migrate_through_0004(pg_empty_engine, tmp_path)
    await _execute(
        pg_empty_engine,
        *(statement for statement in _ADK_V1_DDL if "user_states" not in statement),
        "CREATE TABLE user_states (account text PRIMARY KEY)",
    )

    await _refused(pg_empty_engine, "DFA01")


async def test_a_foreign_sessions_table_beside_adks_events_is_a_mixed_layout(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    await _migrate_through_0004(pg_empty_engine, tmp_path)
    await _execute(
        pg_empty_engine,
        "CREATE TABLE sessions (id text PRIMARY KEY)",
        "CREATE TABLE events (id varchar(128) NOT NULL, app_name varchar(128) NOT NULL, "
        "user_id varchar(128) NOT NULL, session_id varchar(128) NOT NULL, "
        "invocation_id varchar(256), timestamp timestamp NOT NULL)",
    )

    await _refused(pg_empty_engine, "DFA01")


_MOVED = "2026-11-02 09:00:00"


@pytest.mark.parametrize(
    ("tables", "rows", "uncleared"),
    [
        pytest.param(
            _ADK_V1_DDL,
            (
                *(_session("c1"), _session("c2"), _imported("c1"), _imported("c2")),
                *(_conversation("c1"), _conversation("c2")),
            ),
            [],
            id="complete import",
        ),
        pytest.param(
            _ADK_V1_DDL,
            (_session("c1"), _session("c2"), _imported("c1"), _conversation("c1")),
            ["p/c2"],
            id="one of two imported",
        ),
        pytest.param(
            _ADK_V1_DDL,
            (_session("c1"), _PLACEHOLDER, _conversation("c1")),
            ["p/c1"],
            id="conflict placeholder only",
        ),
        pytest.param(
            _ADK_V1_DDL,
            (_session("c1", updated=_MOVED), _imported("c1"), _conversation("c1")),
            ["p/c1"],
            id="ADK moved after the import",
        ),
        pytest.param(_ADK_V1_DDL, (_session("c1"), _kept("c1")), [], id="kept, current"),
        pytest.param(
            _ADK_V1_DDL,
            (_session("c1", updated=_MOVED), _kept("c1")),
            ["p/c1"],
            id="kept, ADK moved after it",
        ),
        pytest.param(
            (*_ADK_V1_DDL, "DROP TABLE conversations CASCADE"),
            (_session("c1"), _imported("c1")),
            ["p/c1"],
            id="no conversations table",
        ),
        pytest.param((), (), [], id="no sessions table"),
        pytest.param(
            ("CREATE TABLE sessions (id text PRIMARY KEY, app_name text)",),
            (f"INSERT INTO sessions VALUES ('c1', '{_LEDGER_APP}')",),
            [],
            id="foreign sessions table",
        ),
        pytest.param(
            _ADK_V1_DDL,
            (_session("c1#t1", app="dfo_analyst_graph"),),
            [],
            id="sessions of other apps only",
        ),
    ],
)
async def test_the_archive_waits_for_every_ledger_session_to_be_imported_or_kept(
    pg_empty_engine: AsyncEngine,
    tmp_path: Path,
    tables: tuple[str, ...],
    rows: tuple[str, ...],
    uncleared: list[str],
) -> None:
    await _migrate_through_0004(pg_empty_engine, tmp_path)
    await _execute(pg_empty_engine, *tables, *rows)

    if not uncleared:
        assert await apply_pending(pg_empty_engine) == [_MIGRATION]
        return
    message = await _refused(pg_empty_engine, "DFA02")
    assert message.startswith(f"{len(uncleared)} ledger session")
    assert f": {', '.join(uncleared)}." in message


async def test_the_dfa02_message_names_the_first_ten_and_the_commands_that_clear_them(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    await _migrate_through_0004(pg_empty_engine, tmp_path)
    await _execute(pg_empty_engine, *_ADK_V1_DDL, *(_session(f"c{n:02}") for n in range(12)))

    message = await _refused(pg_empty_engine, "DFA02")

    assert message.startswith("12 ledger session")
    assert f": {', '.join(f'p/c{n:02}' for n in range(10))}, ...." in message
    for command in (
        "`import --apply`",
        "`import --apply --prefer-adk PRINCIPAL/CONVERSATION_ID`",
        "`import --apply --keep-tables PRINCIPAL/CONVERSATION_ID`",
        "`gate`",
    ):
        assert command in message


async def test_one_run_from_no_harness_migration_refuses_unimported_ledger_sessions(
    pg_empty_engine: AsyncEngine,
) -> None:
    """0001 to 0005 apply in one run, so 0002's empty ledger clears nothing and
    0003's gate refuses before 0005 runs."""
    await _execute(pg_empty_engine, *_ADK_V1_DDL, _session("c1"))

    with pytest.raises(asyncpg.PostgresError) as refusal:
        await apply_pending(pg_empty_engine)

    assert refusal.value.sqlstate == "DFA02"
    assert "adk_archive_sessions" not in await _tables(pg_empty_engine)


async def test_indexes_and_owned_sequences_are_renamed_so_adk_can_create_fresh_tables(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    await _migrate_through_0004(pg_empty_engine, tmp_path)
    await _execute(
        pg_empty_engine,
        *_ADK_V1_DDL,
        "ALTER TABLE adk_internal_metadata ADD COLUMN n serial",
    )

    await apply_pending(pg_empty_engine)

    async with pg_empty_engine.connect() as conn:
        relations = {
            row[0]
            for row in await conn.execute(
                text(
                    "SELECT relname FROM pg_class "
                    "WHERE relnamespace = current_schema()::regnamespace"
                )
            )
        }
    assert {
        "adk_archive_idx_events_app_user_session_ts_id",
        "adk_archive_sessions_pkey",
        "adk_archive_adk_internal_metadata_n_seq",
    } <= relations
    assert not {"idx_events_app_user_session_ts_id", "sessions_pkey"} & relations
    # `prepare_tables` creates the tables and the index again.
    await _execute(pg_empty_engine, *_ADK_V1_DDL[1:4])


@pytest.mark.parametrize(
    ("clash", "shadow"),
    [
        ("CREATE TABLE adk_archive_user_states (x int)", False),
        ("CREATE TYPE adk_archive_sessions AS ENUM ('a')", False),
        (
            "CREATE TABLE other (x int); CREATE INDEX adk_archive_idx_events_app_user_session_ts_id "
            "ON other (x)",
            False,
        ),
        # 52 bytes, which `adk_archive_` would take past 63.
        (f"CREATE INDEX {'i' * 52} ON events (invocation_id)", False),
        # A schema ahead of ADK's on `search_path`, where `adk_archive_sessions`
        # would hide the archive from `adk_archive`'s `drop-archive` and `restore`.
        ("CREATE TABLE {shadow}.adk_archive_sessions (x int)", True),
    ],
)
async def test_a_name_already_taken_or_too_long_refuses_and_renames_nothing(
    pg_empty_engine: AsyncEngine, tmp_path: Path, clash: str, shadow: bool
) -> None:
    await _migrate_through_0004(pg_empty_engine, tmp_path)
    async with pg_empty_engine.connect() as conn:
        schema = (await conn.execute(text("SELECT current_schema()"))).scalar()
    shadow_schema = f"{schema}_shadow"
    await _execute(
        pg_empty_engine,
        f"CREATE SCHEMA {shadow_schema}",
        *_ADK_V1_DDL,
        *clash.format(shadow=shadow_schema).split("; "),
    )
    engine = (
        create_async_engine(
            pg_empty_engine.url,
            connect_args={"server_settings": {"search_path": f"{shadow_schema},{schema}"}},
        )
        if shadow
        else pg_empty_engine
    )
    try:
        # A schema first on `search_path` takes the bookkeeping table too, so
        # the shadowed run applies 0001 to 0005 there.
        await _refused(engine, "DFA04", schema=schema)
    finally:
        if shadow:
            await engine.dispose()
        await _execute(pg_empty_engine, f"DROP SCHEMA {shadow_schema} CASCADE")


async def _refused(
    engine: AsyncEngine,
    sqlstate: str,
    *,
    schema: str | None = None,
) -> str:
    """0005 raises `sqlstate` and renames nothing in `schema`, the engine's
    current schema by default. Returns the error's message."""
    before = await _tables(engine, schema)
    with pytest.raises(asyncpg.PostgresError) as refusal:
        await apply_pending(engine)
    assert refusal.value.sqlstate == sqlstate
    assert await _tables(engine, schema) == before
    return str(refusal.value)


async def test_a_schema_the_current_user_cannot_create_in_refuses_at_once(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    """ADK's tables sit in a schema of their own where the runtime user owns
    them and holds `USAGE` only, so `RENAME` would fail with 42501."""
    await _migrate_through_0004(pg_empty_engine, tmp_path)
    async with pg_empty_engine.connect() as conn:
        schema = (await conn.execute(text("SELECT current_schema()"))).scalar()
    adk_schema = f"{schema}_adk"
    await _execute(
        pg_empty_engine,
        f"CREATE SCHEMA {adk_schema}",
        f"SET LOCAL search_path TO {adk_schema}",
        *_ADK_V1_DDL,
    )
    role = f"{schema}_runtime"
    owned = ("adk_ledger_imports", "harness_schema_migrations")
    await _execute(
        pg_empty_engine,
        f"CREATE ROLE {role} NOSUPERUSER NOCREATEDB NOCREATEROLE",
        f"GRANT USAGE, CREATE ON SCHEMA {schema} TO {role}",
        f"GRANT USAGE ON SCHEMA {adk_schema} TO {role}",
        *(f"ALTER TABLE {schema}.{name} OWNER TO {role}" for name in owned),
        *(f"ALTER TABLE {adk_schema}.{name} OWNER TO {role}" for name in _ADK_TABLES),
    )
    runtime = create_async_engine(
        pg_empty_engine.url,
        connect_args={"server_settings": {"search_path": f"{schema},{adk_schema}", "role": role}},
    )
    try:
        await _refused(runtime, "DFA03")
    finally:
        await runtime.dispose()
        await _execute(
            pg_empty_engine,
            f"DROP OWNED BY {role}",
            f"DROP ROLE {role}",
            f"DROP SCHEMA {adk_schema} CASCADE",
        )
