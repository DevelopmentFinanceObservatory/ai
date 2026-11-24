from __future__ import annotations

import contextlib
import re
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import asyncpg
import pytest
from harness import server as server_module
from harness.db import adk_archive
from harness.db.migrate import MIGRATIONS_DIR, apply_pending
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .test_rename_adk_tables import (
    _ADK_ROWS,
    _ADK_TABLES,
    _ADK_V1_DDL,
    _ARCHIVED,
    _conversation,
    _execute,
    _foreign_key_target,
    _imported,
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
    """Run `verb`, naming a backup id, which an applied drop needs."""
    return await adk_archive.run(engine, verb=verb, apply=apply, backup="1732441200000")


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
        # The heading, the renames and the line recording 0005 as applied.
        assert len(printed.splitlines()) == 2 + len(_archived_split(schema, adk_schema))
        assert "INSERT INTO harness_schema_migrations" in printed.splitlines()[-1]
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


async def _archived_by_startup(engine: AsyncEngine, tmp_path: Path) -> None:
    """ADK's tables with one imported ledger session, archived by a start
    that applied 0005 and recorded it."""
    await _migrate_through_0004(engine, tmp_path)
    await _execute(engine, *_ADK_V1_DDL, *_ADK_ROWS)
    assert await apply_pending(engine) == [adk_archive.ARCHIVE_MIGRATION.name]


async def _recorded(engine: AsyncEngine) -> bool:
    async with engine.connect() as conn:
        return bool(
            (
                await conn.execute(
                    text("SELECT count(*) FROM harness_schema_migrations WHERE version = :v"),
                    {"v": adk_archive.ARCHIVE_MIGRATION.name},
                )
            ).scalar()
        )


async def _start(engine: AsyncEngine) -> None:
    """Run startup's migrations and checks, failing on any retry."""

    async def no_retry(seconds: float) -> None:
        raise AssertionError(f"startup retried after {seconds}s")

    await server_module._apply_migrations_with_retries(engine, budget_s=60.0, sleep=no_retry)


async def _start_refused(engine: AsyncEngine) -> server_module._MigrationRefusedError:
    """Startup refuses with DFA05 at its first attempt. Returns the refusal."""
    with pytest.raises(server_module._MigrationRefusedError) as refusal:
        await _start(engine)
    assert refusal.value.sqlstate == "DFA05"
    return refusal.value


def _drop_adk_rows(printed: str) -> int:
    """The conversation count `drop-adk` printed."""
    (rows,) = re.findall(r"^adk_archive drop-adk: \S+ holds (\d+) conversations$", printed, re.M)
    return int(rows)


def _restore_lacks(printed: str) -> list[list[str]]:
    """The tables each of `restore`'s summary lines printed as missing from
    the archive."""
    found = re.findall(r"^adk_archive restore: .*; missing from the archive: (.+)$", printed, re.M)
    return [[] if names == "none" else names.split(", ") for names in found]


def _names(tables: tuple[str, ...]) -> set[str]:
    """The unqualified names of the schema-qualified `tables`."""
    return {table.rsplit(".", 1)[1] for table in tables}


async def test_after_restore_startup_refuses_until_archive_reruns_0005s_checks(
    pg_empty_engine: AsyncEngine, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A restore leaves 0005 recorded, so a start of this image beside the
    restored tables renames nothing and refuses. `archive` meets 0005's DFA02
    check for a session the ADK-reading image wrote, then renames once it is
    imported, and startup proceeds."""
    await _archived_by_startup(pg_empty_engine, tmp_path)
    assert await _verb(pg_empty_engine, "restore") == 0
    assert await _recorded(pg_empty_engine)
    capsys.readouterr()
    assert await _verb(pg_empty_engine, "restore") == 0
    assert "nothing to do" in capsys.readouterr().out
    await _execute(pg_empty_engine, _session("c2"))
    tables = await _tables(pg_empty_engine)

    refusal = await _start_refused(pg_empty_engine)

    assert _names(refusal.tables) == set(_ADK_TABLES)
    assert refusal.archive_present is False
    assert refusal.ledger_present is True
    assert await _tables(pg_empty_engine) == tables
    assert await _verb(pg_empty_engine, "archive") == 1
    assert "SQLSTATE DFA02" in capsys.readouterr().err
    assert await _tables(pg_empty_engine) == tables

    await _execute(pg_empty_engine, _imported("c2"), _conversation("c2"))
    assert await _verb(pg_empty_engine, "archive") == 0
    await _start(pg_empty_engine)
    assert {f"adk_archive_{table}" for table in _ARCHIVED} <= await _tables(pg_empty_engine)


async def test_startup_reports_no_archive_beside_a_foreign_table_of_an_archive_name(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    """A table of an `adk_archive_<name>` name without the archive's columns is
    another service's, so beside restored tables startup reports no archive."""
    await _archived_by_startup(pg_empty_engine, tmp_path)
    assert await _verb(pg_empty_engine, "restore") == 0
    await _execute(pg_empty_engine, "CREATE TABLE adk_archive_adk_ledger_imports (x int)")

    refusal = await _start_refused(pg_empty_engine)

    assert _names(refusal.tables) == set(_ADK_TABLES)
    assert refusal.archive_present is False
    assert _names(refusal.foreign_archive_names) == {"adk_archive_adk_ledger_imports"}
    assert refusal.ledger_present is True


async def test_startup_refuses_a_fresh_adk_table_created_after_the_archive_until_drop_adk(
    pg_empty_engine: AsyncEngine, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ADK-reading image started after the archive creates its tables and
    writes a ledger session. Startup refuses, naming the live table beside
    its own archive twin. `archive` refuses with 0005's DFA02, since no
    `adk_ledger_imports` resolves to clear it, so the remedy is `drop-adk`,
    which prints the conversations it discards in a dry run and with
    `--apply`, and refuses `--apply` without a numeric `--backup`. Startup
    proceeds once it has run. After `drop-archive` a fresh table has no twin,
    and startup refuses the same way."""
    await _archived_by_startup(pg_empty_engine, tmp_path)
    await _start(pg_empty_engine)
    sessions = next(statement for statement in _ADK_V1_DDL if "TABLE sessions" in statement)
    await _execute(
        pg_empty_engine, sessions, _session("c3"), _session("c3:turn1", app="dfo_analyst_graph")
    )
    archived = {f"adk_archive_{table}" for table in _ARCHIVED}

    refusal = await _start_refused(pg_empty_engine)

    assert _names(refusal.tables) == {"sessions"}
    assert refusal.archive_present is True
    assert refusal.ledger_present is False
    assert _names(refusal.beside) == {"sessions"}
    capsys.readouterr()
    assert await _verb(pg_empty_engine, "archive") == 1
    assert "SQLSTATE DFA02" in capsys.readouterr().err
    assert await _verb(pg_empty_engine, "drop-adk", apply=False) == 0
    assert _drop_adk_rows(capsys.readouterr().out) == 1
    for backup in (None, "", "\u00b2", "\u0661\u0662\u0663", "12a"):
        assert (
            await adk_archive.run(pg_empty_engine, verb="drop-adk", apply=True, backup=backup) == 2
        )
    assert "sessions" in await _tables(pg_empty_engine)
    assert await _verb(pg_empty_engine, "drop-adk") == 0
    printed = capsys.readouterr().out
    assert _drop_adk_rows(printed) == 1
    assert "1732441200000" in printed
    await _start(pg_empty_engine)
    tables = await _tables(pg_empty_engine)
    assert archived <= tables
    assert "sessions" not in tables

    assert await _verb(pg_empty_engine, "drop-archive") == 0
    await _execute(pg_empty_engine, sessions, _session("c4"))
    refusal = await _start_refused(pg_empty_engine)
    assert (_names(refusal.tables), refusal.beside) == ({"sessions"}, ())
    assert (refusal.archive_present, refusal.ledger_present) == (False, False)


async def test_startup_leaves_foreign_tables_of_adks_names_alone(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    await _migrate_through_0004(pg_empty_engine, tmp_path)
    await _execute(pg_empty_engine, *(f"CREATE TABLE {table} (x int)" for table in _ADK_TABLES))

    await _start(pg_empty_engine)

    assert set(_ADK_TABLES) <= await _tables(pg_empty_engine)


@pytest.mark.parametrize(
    ("tables", "missing"),
    [
        pytest.param(_ADK_V1_DDL[1:], ("adk_internal_metadata",), id="one of five missing"),
        pytest.param((), _ADK_TABLES, id="ledger only"),
        pytest.param(_ADK_V1_DDL, ("adk_ledger_imports",), id="no ledger"),
    ],
)
async def test_restore_refuses_an_archive_without_all_of_adks_tables_unless_allowed(
    pg_empty_engine: AsyncEngine,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    tables: tuple[str, ...],
    missing: tuple[str, ...],
) -> None:
    await _migrate_through_0004(pg_empty_engine, tmp_path)
    await _execute(pg_empty_engine, *tables)
    await apply_pending(pg_empty_engine)
    if "adk_ledger_imports" in missing:
        await _execute(pg_empty_engine, "DROP TABLE adk_archive_adk_ledger_imports")
    archived = await _tables(pg_empty_engine)
    capsys.readouterr()

    for apply in (False, True):
        assert await _verb(pg_empty_engine, "restore", apply=apply) == 1

    printed = capsys.readouterr()
    lacking = [name for name in (*_ADK_TABLES, "adk_ledger_imports") if name in missing]
    assert _restore_lacks(printed.out) == [lacking, lacking]
    assert await _tables(pg_empty_engine) == archived
    assert adk_archive._arguments(["restore", "--allow-missing"]).allow_missing
    assert (
        await adk_archive.run(pg_empty_engine, verb="restore", apply=True, allow_missing=True) == 0
    )
    restored = await _tables(pg_empty_engine)
    assert {*_ADK_TABLES, "adk_ledger_imports"} - set(missing) <= restored
    assert "adk_archive_adk_ledger_imports" not in restored


@pytest.mark.parametrize("verb", ["restore", "drop-archive"])
async def test_restore_and_drop_archive_refuse_a_foreign_table_of_an_archive_name_ahead(
    pg_empty_engine: AsyncEngine,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    """A table of an archive name earlier on `search_path` without the
    archived table's columns is another service's."""
    async with _split_layout(pg_empty_engine, tmp_path) as (split, schema, adk_schema):
        assert await _verb(split, "archive") == 0
        await _execute(split, f"CREATE TABLE {schema}.adk_archive_sessions (x int)")
        capsys.readouterr()

        assert await _verb(split, verb) == 1

        assert f"{schema}.adk_archive_sessions" in capsys.readouterr().err
        assert await _archive_relations(split, schema, adk_schema) == _archived_split(
            schema, adk_schema
        ) | {f"{schema}.adk_archive_sessions"}


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
        for backup in (None, " ", "BACKUP_ID", "12 3"):
            assert await adk_archive.run(split, verb="drop-archive", apply=True, backup=backup) == 2
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


@pytest.mark.parametrize(
    ("verb", "view"),
    [
        pytest.param("drop-adk", "events", id="drop-adk"),
        pytest.param("drop-archive", "adk_archive_events", id="drop-archive"),
    ],
)
async def test_a_drop_refuses_while_another_object_depends_on_a_table(
    pg_empty_engine: AsyncEngine,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    verb: str,
    view: str,
) -> None:
    """The drop names every table without `CASCADE`, so the foreign key
    between `events` and `sessions` resolves and a view on one refuses."""
    await _archived_by_startup(pg_empty_engine, tmp_path)
    if verb == "drop-adk":
        assert await _verb(pg_empty_engine, "restore") == 0
    await _execute(pg_empty_engine, f"CREATE VIEW event_ids AS SELECT id FROM {view}")
    tables = await _tables(pg_empty_engine)
    capsys.readouterr()

    assert await _verb(pg_empty_engine, verb) == 1

    assert "SQLSTATE 2BP01" in capsys.readouterr().err
    assert await _tables(pg_empty_engine) == tables

    await _execute(pg_empty_engine, "DROP VIEW event_ids")
    assert await _verb(pg_empty_engine, verb) == 0
    assert not {view, "sessions", "adk_archive_sessions"} & await _tables(pg_empty_engine)


def test_a_usage_error_exits_2() -> None:
    with pytest.raises(SystemExit) as exited:
        adk_archive.main(["rename"])

    assert exited.value.code == 2


def test_no_database_configured_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adk_archive, "build_database", lambda *, settings: None)

    assert adk_archive.main(["archive"]) == 2


def test_an_invalid_database_configuration_exits_2_without_printing_the_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both backends set fails `Settings` validation. Pydantic's own text
    quotes the start of the input, which holds `DATABASE_URL`."""
    monkeypatch.setenv("DATABASE_URL", "S3CRET")
    monkeypatch.setenv("SESSION_STORE_INSTANCE_CONNECTION_NAME", "proj:region:instance")

    assert adk_archive.main(["archive"]) == 2

    assert "S3CRET" not in capsys.readouterr().err


def test_drop_adk_checks_the_column_signature_the_migration_checks() -> None:
    specs = adk_archive.ARCHIVE_MIGRATION.read_text().split("specs constant jsonb", 1)[1]
    specs = specs.split(";", 1)[0]
    pairs = re.findall(r"'(\w+)', jsonb_build_array\(([^)]*)\)", specs)

    assert {name: tuple(re.findall(r"'(\w+)'", columns)) for name, columns in pairs} == (
        adk_archive._ADK_COLUMNS
    )


def _migration_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Directories of the files through 0002 and through 0004."""
    dirs = (tmp_path / "through_0002", tmp_path / "through_0004")
    for directory, last in zip(dirs, ("0002", "0004"), strict=True):
        directory.mkdir()
        for path in MIGRATIONS_DIR.glob("*.sql"):
            if path.name[:4] <= last:
                shutil.copy(path, directory / path.name)
    return dirs


async def _open_on_adk(engine: AsyncEngine) -> None:
    """Conversation c1, imported, with turn 1 open on its ADK turn session
    `c1#t1`, as the release that runs turns on ADK leaves a paused turn."""
    await _execute(
        engine,
        *_ADK_V1_DDL,
        _session("c1"),
        _session("c1#t1", app="dfo_analyst_graph"),
        _imported("c1"),
        "INSERT INTO conversations (principal, conversation_id, turn_state) "
        """VALUES ('p', 'c1', '{"turn_index": 1}')""",
    )


async def test_a_start_refuses_to_archive_adk_while_a_turn_is_open_on_it(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    """At 0002, with turn 1 open on ADK, a start of this image refuses with
    DFA07 and applies nothing, and so does `archive`. Once the turn is
    closed, as `close-open-turns` closes it, the same start applies 0003 to
    0005."""
    through_0002, _ = _migration_dirs(tmp_path)
    await apply_pending(pg_empty_engine, directory=through_0002)
    await _open_on_adk(pg_empty_engine)

    with pytest.raises(asyncpg.PostgresError) as refusal:
        await apply_pending(pg_empty_engine)

    assert refusal.value.sqlstate == "DFA07"
    assert "turn_cursors" not in await _tables(pg_empty_engine)
    await _execute(
        pg_empty_engine,
        """UPDATE conversations SET turn_state = '{"turn_index": 1, "turn_recorded": true}'""",
    )
    assert await apply_pending(pg_empty_engine) == [
        "0003_turns.sql",
        "0004_card_lineage.sql",
        "0005_rename_adk_tables.sql",
    ]


@pytest.mark.parametrize("held", [False, True], ids=["not in turn_cursors", "in turn_cursors"])
async def test_archive_refuses_a_turn_open_on_adk_that_turn_cursors_does_not_hold(
    pg_empty_engine: AsyncEngine, tmp_path: Path, capsys: pytest.CaptureFixture[str], held: bool
) -> None:
    """At 0004, a turn open on ADK refuses the start and `archive` with
    DFA07 unless `turn_cursors` holds the turn, opened there under the
    number of a leftover ADK turn session."""
    through_0002, through_0004 = _migration_dirs(tmp_path)
    await apply_pending(pg_empty_engine, directory=through_0002)
    await _open_on_adk(pg_empty_engine)
    await apply_pending(pg_empty_engine, directory=through_0004)
    if held:
        await _execute(
            pg_empty_engine,
            "INSERT INTO turn_cursors (conversation, turn_index, node, status, inputs, state) "
            "SELECT id, 1, 'plan', 'running', '{}', '{}' FROM conversations",
        )
    if held:
        assert await apply_pending(pg_empty_engine) == ["0005_rename_adk_tables.sql"]
    else:
        with pytest.raises(asyncpg.PostgresError) as refusal:
            await apply_pending(pg_empty_engine)
        assert refusal.value.sqlstate == "DFA07"
    capsys.readouterr()

    assert await _verb(pg_empty_engine, "archive") == (0 if held else 1)
    assert ("SQLSTATE DFA07" in capsys.readouterr().err) is not held


async def test_a_start_archives_beside_a_foreign_sessions_table(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    """At 0004, a `sessions` table of another service, with integer
    `app_name`, `user_id` and `id` and without ADK's other columns, is not
    ADK's: the DFA07 check passes over it and the start applies 0005."""
    _, through_0004 = _migration_dirs(tmp_path)
    await apply_pending(pg_empty_engine, directory=through_0004)
    await _execute(
        pg_empty_engine,
        "CREATE TABLE sessions (app_name integer, user_id integer, id integer)",
        "INSERT INTO conversations (principal, conversation_id, turn_state) "
        """VALUES ('p', 'c1', '{"turn_index": 1}')""",
    )

    assert await apply_pending(pg_empty_engine) == ["0005_rename_adk_tables.sql"]
