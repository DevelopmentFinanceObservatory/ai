"""A sqlite-backed `DatabaseSessionService` for tests, in WAL journal mode.

In SQLite's default rollback-journal mode an open reader blocks a writer's
COMMIT. A cancelled turn leaves such a reader behind: the cancellation lands
while aiosqlite is stepping a `SELECT`, SQLAlchemy closes the connection, and
SQLite defers that close, lock included, until the half-read statement is
finalized. The turn's own end-of-segment save (`StudioStore.release`) then
waits out sqlite's 5-second busy timeout behind it. In WAL mode a writer
commits past open readers, as it does on Postgres.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from google.adk.sessions import DatabaseSessionService


def sqlite_session_service(db_path: Path) -> DatabaseSessionService:
    """A `DatabaseSessionService` over the sqlite file `db_path`, switched to
    WAL first. The journal mode is stored in the file, so every connection
    the service opens, and every other service over the same file, uses it."""
    with closing(sqlite3.connect(db_path)) as db:
        db.execute("PRAGMA journal_mode=WAL")
    return DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{db_path}")
