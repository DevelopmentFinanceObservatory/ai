"""Forward-only schema migrations for the harness's own tables.

The SQL files in `harness/db/migrations` run in filename order.
`harness_schema_migrations` records each applied file with the SHA-256 of
its bytes. A run refuses when an applied file changed, or when one is
missing and its name sorts at or below the newest file this image holds. A
fix is a new file. An applied file newer than every file here was written by
a later release. That release may have renamed or dropped what this one reads,
so the run logs the file and carries on, and an image started against a newer
schema runs only where the deploy notes say it may.

The bookkeeping table is created in a transaction of its own. The pending
files then apply in one transaction, every one of them or none. Each
transaction holds a transaction-level advisory lock, so two instances
starting together apply each file once, and the lock ends with the
transaction however the run ends. Each statement is bounded by
`statement_timeout` (`MIGRATION_TIMEOUT_S`), and each wait for a lock, the
advisory lock and the locks a migration's own statements take, by
`lock_timeout` (`MIGRATION_LOCK_TIMEOUT_S`). PostgreSQL applies
`statement_timeout` to each statement of a multi-statement file
separately. The client waits on each call it makes up to
`MIGRATION_CALL_TIMEOUT_S`, which replaces the engine's per-call
`command_timeout` (`harness.db.engine.COMMAND_TIMEOUT_S`), so a file whose
statements each pass `statement_timeout` is never cut short by the
engine's bound. Nothing bounds a run as a whole. At startup Cloud Run's
240 s startup probe bounds it. Past the probe the instance is
killed and the transaction rolls back, so a run that needs longer goes out
of band.

A pending file named in `harness.db.cutover_gates.BEFORE` has its check run
in the same transaction just before it, so a check added after the file was
recorded somewhere runs where the file applies and nowhere else, and adds
no file an older image would lack.

Each file runs through asyncpg's simple-query protocol (`Connection.execute`
with no arguments), which accepts several statements in one string.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine

from harness.db.cutover_gates import BEFORE

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
# The longest one statement of a run may take.
MIGRATION_TIMEOUT_S = 60.0
# The longest a statement of a run may wait for a lock. A statement waiting
# for a table lock holds up every query on that table queued behind it, so
# it gives up well before the statement bound and the run is retried.
MIGRATION_LOCK_TIMEOUT_S = 10.0
# The longest the client waits on one call of a run, a whole migration file
# sent as one call included, in place of the engine's `command_timeout`.
# It is the startup database budget (`harness.server._STARTUP_DB_RETRY_BUDGET_S`,
# 120 s): a call the startup could not wait for anyway.
MIGRATION_CALL_TIMEOUT_S = 120.0
# A fixed 64-bit number that names the lock in `pg_locks`.
_ADVISORY_LOCK_KEY = 0x6861726E657373  # "harness"
_BOOKKEEPING = """
CREATE TABLE IF NOT EXISTS harness_schema_migrations (
    version text PRIMARY KEY,
    content_sha256 text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


class MigrationChecksumError(RuntimeError):
    """An applied migration changed, or one this image should hold is
    missing. Fix forward with a new file."""


def sha256(path: Path) -> str:
    """The SHA-256 of the bytes of the file at `path`, as hex."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def bound_and_lock(
    conn: Any,
    *,
    timeout_s: float = MIGRATION_TIMEOUT_S,
    lock_timeout_s: float = MIGRATION_LOCK_TIMEOUT_S,
) -> None:
    """Bound each statement and lock wait of the asyncpg connection `conn`'s
    open transaction, then take the migrations' advisory lock until it ends."""
    await conn.execute(
        f"SET LOCAL statement_timeout = {round(timeout_s * 1000)}; "
        f"SET LOCAL lock_timeout = {round(lock_timeout_s * 1000)}",
        timeout=MIGRATION_CALL_TIMEOUT_S,
    )
    await conn.execute(
        "SELECT pg_advisory_xact_lock($1)", _ADVISORY_LOCK_KEY, timeout=MIGRATION_CALL_TIMEOUT_S
    )


async def record_applied(conn: Any, path: Path) -> None:
    """Record the file at `path` in `harness_schema_migrations` as applied,
    with the SHA-256 of its bytes."""
    await conn.execute(
        "INSERT INTO harness_schema_migrations (version, content_sha256) VALUES ($1, $2)",
        path.name,
        sha256(path),
        timeout=MIGRATION_CALL_TIMEOUT_S,
    )


async def apply_pending(
    engine: AsyncEngine,
    *,
    directory: Path = MIGRATIONS_DIR,
    timeout_s: float = MIGRATION_TIMEOUT_S,
    lock_timeout_s: float = MIGRATION_LOCK_TIMEOUT_S,
) -> list[str]:
    """Apply every file in `directory` not yet recorded, in filename order, in
    one transaction, and return their names. `timeout_s` bounds each
    statement (`statement_timeout`) and `lock_timeout_s` each wait for a
    lock (`lock_timeout`). The client waits on each call up to
    `MIGRATION_CALL_TIMEOUT_S`, whatever the engine's `command_timeout`.
    Nothing here bounds the run as a whole. At
    startup Cloud Run's startup probe does (module docstring).

    Raises:
        MigrationChecksumError: an applied file changed, or is missing and
            sorts at or below the newest file in `directory`.
    """
    files = sorted(directory.glob("*.sql"))
    newest = files[-1].name if files else ""
    async with engine.connect() as sa_conn:
        raw = await sa_conn.get_raw_connection()
        conn = raw.driver_connection
        assert conn is not None
        wait = MIGRATION_CALL_TIMEOUT_S
        async with conn.transaction():
            await bound_and_lock(conn, timeout_s=timeout_s, lock_timeout_s=lock_timeout_s)
            await conn.execute(_BOOKKEEPING, timeout=wait)
        async with conn.transaction():
            await bound_and_lock(conn, timeout_s=timeout_s, lock_timeout_s=lock_timeout_s)
            rows = await conn.fetch(
                "SELECT version, content_sha256 FROM harness_schema_migrations", timeout=wait
            )
            applied = {row["version"]: row["content_sha256"] for row in rows}
            missing = sorted(set(applied) - {path.name for path in files})
            if refused := [name for name in missing if name <= newest]:
                raise MigrationChecksumError(f"applied but not in {directory}: {refused}")
            if missing:
                logger.info("schema: applied by a later release, not in this image: %s", missing)
            ran: list[str] = []
            for path in files:
                digest = sha256(path)
                if path.name in applied:
                    if applied[path.name] != digest:
                        raise MigrationChecksumError(f"{path.name} changed since it was applied")
                    continue
                if (check := BEFORE.get(path.name)) is not None:
                    await conn.execute(check, timeout=wait)
                await conn.execute(path.read_text(), timeout=wait)
                await record_applied(conn, path)
                ran.append(path.name)
        for name in ran:
            logger.info("schema: applied %s", name)
        return ran
