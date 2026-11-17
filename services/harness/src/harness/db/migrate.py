"""Forward-only schema migrations for the harness's own tables.

The SQL files in `harness/db/migrations` run in filename order.
`harness_schema_migrations` records each applied file with the SHA-256 of
its bytes. A run refuses when an applied file changed, or when one is
missing and its name sorts at or below the newest file this image holds: a
fix is a new file. An applied file newer than every file here was written by
a later release, whose migrations add to this schema, so the run logs it and
carries on, and a revision of this release can still start beside the later
one or after a rollback.

The bookkeeping table is created in a transaction of its own; the pending
files then apply in one transaction, every one of them or none. Each
transaction holds a transaction-level advisory lock, so two instances
starting together apply each file once, and the lock ends with the
transaction however the run ends. Each statement is bounded by
`statement_timeout` (`MIGRATION_TIMEOUT_S`), and each wait for a lock, the
advisory lock and the locks a migration's own statements take, by
`lock_timeout` (`MIGRATION_LOCK_TIMEOUT_S`). PostgreSQL applies
`statement_timeout` to each statement of a multi-statement file
separately, so this module puts no bound on a run as a whole. At startup
Cloud Run's 240 s startup probe bounds it: past the probe the instance is
killed and the transaction rolls back, so a run that needs longer goes out
of band.

Each file runs through asyncpg's simple-query protocol (`Connection.execute`
with no arguments), which accepts several statements in one string.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
# The longest one statement of a run may take.
MIGRATION_TIMEOUT_S = 60.0
# The longest a statement of a run may wait for a lock. A statement waiting
# for a table lock holds up every query on that table queued behind it, so
# it gives up well before the statement bound and the run is retried.
MIGRATION_LOCK_TIMEOUT_S = 10.0
# Any fixed 64-bit number; names the lock in `pg_locks`.
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    lock (`lock_timeout`). Nothing here bounds the run as a whole; at
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
        timeout = (
            f"SET LOCAL statement_timeout = {round(timeout_s * 1000)}; "
            f"SET LOCAL lock_timeout = {round(lock_timeout_s * 1000)}"
        )
        async with conn.transaction():
            await conn.execute(timeout)
            await conn.execute("SELECT pg_advisory_xact_lock($1)", _ADVISORY_LOCK_KEY)
            await conn.execute(_BOOKKEEPING)
        async with conn.transaction():
            await conn.execute(timeout)
            await conn.execute("SELECT pg_advisory_xact_lock($1)", _ADVISORY_LOCK_KEY)
            rows = await conn.fetch("SELECT version, content_sha256 FROM harness_schema_migrations")
            applied = {row["version"]: row["content_sha256"] for row in rows}
            missing = sorted(set(applied) - {path.name for path in files})
            if refused := [name for name in missing if name <= newest]:
                raise MigrationChecksumError(f"applied but not in {directory}: {refused}")
            if missing:
                logger.info("schema: applied by a later release, not in this image: %s", missing)
            ran: list[str] = []
            for path in files:
                digest = _sha256(path)
                if path.name in applied:
                    if applied[path.name] != digest:
                        raise MigrationChecksumError(f"{path.name} changed since it was applied")
                    continue
                await conn.execute(path.read_text())
                await conn.execute(
                    "INSERT INTO harness_schema_migrations (version, content_sha256) "
                    "VALUES ($1, $2)",
                    path.name,
                    digest,
                )
                ran.append(path.name)
        for name in ran:
            logger.info("schema: applied %s", name)
        return ran
