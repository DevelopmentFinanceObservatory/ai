from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from harness.db.migrate import MIGRATIONS_DIR, MigrationChecksumError, apply_pending
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


def _write(directory: Path, name: str, sql: str) -> None:
    (directory / name).write_text(sql)


async def _recorded(engine: AsyncEngine) -> list[str]:
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT version FROM harness_schema_migrations ORDER BY version")
        )
        return [row[0] for row in result]


async def test_shipped_migrations_apply_and_a_second_run_applies_nothing(
    pg_empty_engine: AsyncEngine,
) -> None:
    shipped = sorted(path.name for path in MIGRATIONS_DIR.glob("*.sql"))

    assert await apply_pending(pg_empty_engine) == shipped
    assert await apply_pending(pg_empty_engine) == []
    async with pg_empty_engine.connect() as conn:
        await conn.execute(text("SELECT id FROM conversations"))


async def test_files_apply_in_filename_order(pg_empty_engine: AsyncEngine, tmp_path: Path) -> None:
    _write(tmp_path, "0002_second.sql", "INSERT INTO ordered VALUES ('second');")
    _write(tmp_path, "0001_first.sql", "CREATE TABLE ordered (name text);")
    _write(tmp_path, "0003_third.sql", "INSERT INTO ordered VALUES ('third');")

    ran = await apply_pending(pg_empty_engine, directory=tmp_path)

    assert ran == ["0001_first.sql", "0002_second.sql", "0003_third.sql"]
    async with pg_empty_engine.connect() as conn:
        rows = (await conn.execute(text("SELECT name FROM ordered"))).scalars().all()
    assert rows == ["second", "third"]


async def test_only_files_added_since_the_last_run_apply(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    _write(tmp_path, "0001_first.sql", "CREATE TABLE a (x int);")
    await apply_pending(pg_empty_engine, directory=tmp_path)
    _write(tmp_path, "0002_second.sql", "CREATE TABLE b (x int);")

    assert await apply_pending(pg_empty_engine, directory=tmp_path) == ["0002_second.sql"]
    assert await _recorded(pg_empty_engine) == ["0001_first.sql", "0002_second.sql"]


async def test_a_failing_file_leaves_nothing_recorded_or_created(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    _write(tmp_path, "0001_bad.sql", "CREATE TABLE half (x int); SELECT 1 / 0;")

    with pytest.raises(Exception, match="division by zero"):
        await apply_pending(pg_empty_engine, directory=tmp_path)

    assert await _recorded(pg_empty_engine) == []
    async with pg_empty_engine.connect() as conn:
        assert (await conn.execute(text("SELECT to_regclass('half')"))).scalar() is None


async def test_a_changed_applied_file_is_refused(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    _write(tmp_path, "0001_first.sql", "CREATE TABLE a (x int);")
    await apply_pending(pg_empty_engine, directory=tmp_path)
    _write(tmp_path, "0001_first.sql", "CREATE TABLE a (x bigint);")

    with pytest.raises(MigrationChecksumError, match=r"0001_first\.sql"):
        await apply_pending(pg_empty_engine, directory=tmp_path)


async def test_a_missing_applied_file_is_refused_unless_a_later_release_applied_it(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    """An image older than the database runs beside the later release. An
    image that lost a file of its own does not start."""
    _write(tmp_path, "0001_first.sql", "CREATE TABLE a (x int);")
    _write(tmp_path, "0002_second.sql", "CREATE TABLE b (x int);")
    await apply_pending(pg_empty_engine, directory=tmp_path)
    (tmp_path / "0002_second.sql").unlink()

    assert await apply_pending(pg_empty_engine, directory=tmp_path) == []

    (tmp_path / "0001_first.sql").unlink()
    _write(tmp_path, "0001_renamed.sql", "CREATE TABLE a (x int);")
    with pytest.raises(MigrationChecksumError, match=r"\['0001_first\.sql'\]"):
        await apply_pending(pg_empty_engine, directory=tmp_path)


async def test_a_run_that_fails_part_way_applies_none_of_its_files(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    _write(tmp_path, "0001_first.sql", "CREATE TABLE a (x int);")
    _write(tmp_path, "0002_bad.sql", "SELECT 1 / 0;")

    with pytest.raises(Exception, match="division by zero"):
        await apply_pending(pg_empty_engine, directory=tmp_path)

    assert await _recorded(pg_empty_engine) == []
    async with pg_empty_engine.connect() as conn:
        assert (await conn.execute(text("SELECT to_regclass('a')"))).scalar() is None


async def test_a_statement_past_the_timeout_fails_and_leaves_the_lock_free(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    _write(tmp_path, "0001_slow.sql", "SELECT pg_sleep(2);")

    with pytest.raises(Exception, match="statement timeout"):
        await apply_pending(pg_empty_engine, directory=tmp_path, timeout_s=0.2)

    (tmp_path / "0001_slow.sql").write_text("CREATE TABLE a (x int);")
    async with asyncio.timeout(5):
        assert await apply_pending(pg_empty_engine, directory=tmp_path) == ["0001_slow.sql"]


async def test_a_lock_wait_past_the_timeout_fails_as_a_lock_timeout(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    """A migration statement waiting on a lock that live traffic holds gives
    up after the lock timeout, well inside the statement's own bound."""
    async with pg_empty_engine.begin() as conn:
        await conn.execute(text("CREATE TABLE held (x int)"))
    _write(tmp_path, "0001_alter.sql", "ALTER TABLE held ADD COLUMN y int;")
    async with pg_empty_engine.connect() as holder:
        await holder.begin()
        await holder.execute(text("LOCK TABLE held IN ACCESS EXCLUSIVE MODE"))

        with pytest.raises(Exception, match="lock timeout"):
            await apply_pending(
                pg_empty_engine, directory=tmp_path, timeout_s=5, lock_timeout_s=0.2
            )

        await holder.rollback()


async def test_two_concurrent_appliers_apply_each_file_once(
    pg_empty_engine: AsyncEngine, tmp_path: Path
) -> None:
    """Without the advisory lock the second applier's `CREATE TABLE` fails."""
    _write(tmp_path, "0001_first.sql", "CREATE TABLE a (x int); SELECT pg_sleep(0.3);")
    _write(tmp_path, "0002_second.sql", "CREATE TABLE b (x int);")

    results = await asyncio.gather(
        apply_pending(pg_empty_engine, directory=tmp_path),
        apply_pending(pg_empty_engine, directory=tmp_path),
    )

    assert sorted(results, key=len) == [[], ["0001_first.sql", "0002_second.sql"]]
