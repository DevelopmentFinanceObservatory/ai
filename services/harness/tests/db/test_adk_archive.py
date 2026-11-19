from __future__ import annotations

import contextlib
import re
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from harness.db import adk_archive
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .test_rename_adk_tables import (
    _ADK_ROWS,
    _ADK_TABLES,
    _ADK_V1_DDL,
    _ARCHIVED,
    _execute,
    _foreign_key_target,
    _migrate_through_0004,
    _rows,
    _session,
    _tables,
)


async def _resolved(engine: AsyncEngine, name: str) -> str | None:
    """The schema-qualified relation `name` resolves to through `search_path`."""
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT n.nspname || '.' || c.relname FROM pg_class c "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE c.oid = to_regclass(:name)"
                ),
                {"name": name},
            )
        ).scalar()


@contextlib.asynccontextmanager
async def _split_layout(
    engine: AsyncEngine, tmp_path: Path
) -> AsyncIterator[tuple[AsyncEngine, str, str]]:
    """The harness tables in the test schema and ADK's, with rows and an owned
    sequence, in a second schema after it on `search_path`, migrated through
    0004. Yields an engine on that `search_path` and both schemas."""
    await _migrate_through_0004(engine, tmp_path)
    async with engine.connect() as conn:
        schema = (await conn.execute(text("SELECT current_schema()"))).scalar()
    assert schema
    adk_schema = f"{schema}_adk"
    await _execute(
        engine,
        f"CREATE SCHEMA {adk_schema}",
        f"SET LOCAL search_path TO {adk_schema}",
        *_ADK_V1_DDL,
        "ALTER TABLE adk_internal_metadata ADD COLUMN n serial",
    )
    split = create_async_engine(
        engine.url, connect_args={"server_settings": {"search_path": f"{schema},{adk_schema}"}}
    )
    try:
        await _execute(split, *_ADK_ROWS)
        yield split, schema, adk_schema
    finally:
        await split.dispose()
        await _execute(engine, f"DROP SCHEMA {adk_schema} CASCADE")


def _home(name: str, schema: str, adk_schema: str) -> str:
    return adk_schema if name in _ADK_TABLES else schema


async def _archive_relations(engine: AsyncEngine, *schemas: str) -> set[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT n.nspname || '.' || c.relname FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = ANY (:schemas) AND c.relname LIKE 'adk\\_archive\\_%'"
            ),
            {"schemas": list(schemas)},
        )
        return {row[0] for row in rows}


def _archived_split(schema: str, adk_schema: str) -> set[str]:
    """Every relation `archive` renames in the split layout."""
    return (
        {f"{_home(table, schema, adk_schema)}.adk_archive_{table}" for table in _ARCHIVED}
        | {f"{_home(table, schema, adk_schema)}.adk_archive_{table}_pkey" for table in _ARCHIVED}
        | {
            f"{adk_schema}.adk_archive_idx_events_app_user_session_ts_id",
            f"{adk_schema}.adk_archive_adk_internal_metadata_n_seq",
        }
    )


async def _verb(engine: AsyncEngine, verb: str, *, apply: bool = True) -> int:
    return await adk_archive.run(engine, verb=verb, apply=apply)


async def test_archive_dry_runs_then_renames_a_split_layout_and_a_rerun_finds_nothing(
    pg_empty_engine: AsyncEngine, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    async with _split_layout(pg_empty_engine, tmp_path) as (split, schema, adk_schema):
        before = {table: await _rows(split, table) for table in _ARCHIVED}

        assert await _verb(split, "archive", apply=False) == 0
        printed = capsys.readouterr().out
        assert await _archive_relations(split, schema, adk_schema) == set()

        assert await _verb(split, "archive") == 0
        assert capsys.readouterr().out.splitlines()[1:] == printed.splitlines()[1:]
        assert len(printed.splitlines()) == 1 + len(_archived_split(schema, adk_schema))
        assert await _archive_relations(split, schema, adk_schema) == _archived_split(
            schema, adk_schema
        )
        for table in _ARCHIVED:
            assert await _rows(split, f"adk_archive_{table}") == before[table]
        assert await _foreign_key_target(split) == "adk_archive_sessions"

        assert await _verb(split, "archive") == 0
        assert "nothing to do" in capsys.readouterr().out


async def test_restore_brings_back_every_table_of_a_split_layout_and_drop_adk_discards_adks(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    """ADK's tables in one schema and the import ledger in another both rename
    back, with their indexes and sequences; `drop-adk` then drops ADK's five
    tables and leaves the ledger."""
    async with _split_layout(pg_empty_engine, tmp_path) as (split, schema, adk_schema):
        before = {table: await _rows(split, table) for table in _ARCHIVED}
        assert await _verb(split, "archive") == 0

        assert await _verb(split, "restore", apply=False) == 0
        assert await _archive_relations(split, schema, adk_schema) == _archived_split(
            schema, adk_schema
        )
        assert await _verb(split, "restore") == 0

        for table in _ARCHIVED:
            assert await _resolved(split, table) == f"{_home(table, schema, adk_schema)}.{table}"
            assert await _rows(split, table) == before[table]
        assert await _archive_relations(split, schema, adk_schema) == set()
        assert await _resolved(split, "idx_events_app_user_session_ts_id") == (
            f"{adk_schema}.idx_events_app_user_session_ts_id"
        )
        assert await _resolved(split, "adk_internal_metadata_n_seq") == (
            f"{adk_schema}.adk_internal_metadata_n_seq"
        )

        assert await _verb(split, "drop-adk") == 0

        assert await _tables(split, adk_schema) == set()
        assert await _resolved(split, "adk_ledger_imports") == f"{schema}.adk_ledger_imports"


async def test_restore_refuses_while_fresh_adk_tables_resolve_ahead_of_the_archive(
    pg_empty_engine: AsyncEngine, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ADK-reading image started after the archive creates its tables in the
    first schema on `search_path`, where they would hide the restored ones."""
    async with _split_layout(pg_empty_engine, tmp_path) as (split, schema, adk_schema):
        assert await _verb(split, "archive") == 0
        await _execute(split, *_ADK_V1_DDL)
        capsys.readouterr()

        for apply in (False, True):
            assert await _verb(split, "restore", apply=apply) == 1

        assert f"`sessions` resolves to {schema}.sessions" in capsys.readouterr().err
        assert await _archive_relations(split, schema, adk_schema) == _archived_split(
            schema, adk_schema
        )
        assert await _resolved(split, "sessions") == f"{schema}.sessions"


async def test_restore_rolls_back_when_a_restored_name_resolves_to_another_table(
    pg_empty_engine: AsyncEngine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The check after renaming holds without the check before it."""

    async def no_conflicts(conn: object, archived: object) -> list[str]:
        return []

    monkeypatch.setattr(adk_archive, "_restore_conflicts", no_conflicts)
    async with _split_layout(pg_empty_engine, tmp_path) as (split, schema, adk_schema):
        assert await _verb(split, "archive") == 0
        await _execute(split, *_ADK_V1_DDL)

        capsys.readouterr()
        assert await _verb(split, "restore") == 1

        assert "restore rolled back" in capsys.readouterr().err
        assert await _archive_relations(split, schema, adk_schema) == _archived_split(
            schema, adk_schema
        )


async def test_drop_archive_drops_every_archived_table_of_a_split_layout(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    async with _split_layout(pg_empty_engine, tmp_path) as (split, schema, adk_schema):
        conversations = await _rows(split, "conversations")
        assert await _verb(split, "archive") == 0

        assert await _verb(split, "drop-archive", apply=False) == 0
        assert await _archive_relations(split, schema, adk_schema) == _archived_split(
            schema, adk_schema
        )
        assert await _verb(split, "drop-archive") == 0

        assert await _archive_relations(split, schema, adk_schema) == set()
        assert await _tables(split, adk_schema) == set()
        assert await _rows(split, "conversations") == conversations


async def test_archive_refuses_an_unimported_session_until_drop_adk_discards_it(
    pg_empty_engine: AsyncEngine, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    await _migrate_through_0004(pg_empty_engine, tmp_path)
    await _execute(pg_empty_engine, *_ADK_V1_DDL, _session("c1"))
    tables = await _tables(pg_empty_engine)

    assert await _verb(pg_empty_engine, "archive") == 1
    assert "SQLSTATE DFA02" in capsys.readouterr().err
    assert await _tables(pg_empty_engine) == tables

    assert await _verb(pg_empty_engine, "drop-adk") == 0
    assert await _verb(pg_empty_engine, "archive") == 0
    assert not set(_ARCHIVED) & await _tables(pg_empty_engine)


async def test_archive_refuses_a_name_the_prefix_would_take_past_63_bytes(
    pg_empty_engine: AsyncEngine, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    await _migrate_through_0004(pg_empty_engine, tmp_path)
    await _execute(
        pg_empty_engine, *_ADK_V1_DDL, f"CREATE INDEX {'i' * 52} ON events (invocation_id)"
    )

    assert await _verb(pg_empty_engine, "archive") == 1

    assert "SQLSTATE DFA04" in capsys.readouterr().err
    assert await _resolved(pg_empty_engine, "i" * 52) is not None


async def test_drop_adk_leaves_a_table_of_an_adk_name_without_adks_columns(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    await _migrate_through_0004(pg_empty_engine, tmp_path)
    await _execute(
        pg_empty_engine,
        *(statement for statement in _ADK_V1_DDL if "user_states" not in statement),
        "CREATE TABLE user_states (account text PRIMARY KEY)",
    )

    assert await _verb(pg_empty_engine, "drop-adk") == 0

    tables = await _tables(pg_empty_engine)
    assert "user_states" in tables
    assert not (set(_ADK_TABLES) - {"user_states"}) & tables


def test_a_usage_error_exits_2() -> None:
    with pytest.raises(SystemExit) as exited:
        adk_archive.main(["rename"])

    assert exited.value.code == 2


def test_no_database_configured_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adk_archive, "build_database", lambda *, settings: None)

    assert adk_archive.main(["archive"]) == 2


def test_drop_adk_checks_the_column_signature_the_migration_checks() -> None:
    specs = adk_archive.ARCHIVE_MIGRATION.read_text().split("specs constant jsonb", 1)[1]
    specs = specs.split(";", 1)[0]
    pairs = re.findall(r"'(\w+)', jsonb_build_array\(([^)]*)\)", specs)

    assert {name: tuple(re.findall(r"'(\w+)'", columns)) for name, columns in pairs} == (
        adk_archive._ADK_COLUMNS
    )
