"""SQLite store for the turn loop: cursor with compare-and-swap, conversations, frames."""

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from pydantic_ai import Conversation, ConversationTypeAdapter

SCHEMA = """
create table if not exists turns(
    conversation_id text primary key,
    cursor text not null,
    version integer not null
);
create table if not exists conversations(
    conversation_id text not null,
    agent text not null,
    body text not null,
    primary key (conversation_id, agent)
);
create table if not exists frames(
    conversation_id text not null,
    seq integer not null,
    frame text not null,
    primary key (conversation_id, seq)
);
"""


class StaleCursorError(Exception):
    """The compare-and-swap on the turn's version found a different version."""


class Store:
    """One SQLite file per deployment; every write to `turns` is a compare-and-swap.

    Args:
        path: The database file.
        drop_agents: Agents whose conversation writes are silently skipped. A
            deliberately broken store for tests.
        unconditional: Ignore the expected version on write. A deliberately broken
            store for tests.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        drop_agents: frozenset[str] = frozenset(),
        unconditional: bool = False,
    ) -> None:
        self._db = sqlite3.connect(path, isolation_level=None, timeout=30)
        self._db.execute("pragma journal_mode=wal")
        self._db.executescript(SCHEMA)
        self._drop_agents = drop_agents
        self._unconditional = unconditional

    @contextmanager
    def _transaction(self) -> Generator[None]:
        self._db.execute("begin immediate")
        try:
            yield
        except BaseException:
            self._db.execute("rollback")
            raise
        self._db.execute("commit")

    def read_turn(self, conversation_id: str) -> tuple[str, int] | None:
        row = self._db.execute(
            "select cursor, version from turns where conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return None if row is None else (row[0], row[1])

    def create_turn(self, conversation_id: str, cursor: str) -> None:
        with self._transaction():
            try:
                self._db.execute("insert into turns values (?, ?, 0)", (conversation_id, cursor))
            except sqlite3.IntegrityError as error:
                raise StaleCursorError(f"turn {conversation_id} already exists") from error

    def checkpoint(
        self,
        conversation_id: str,
        *,
        expected_version: int,
        cursor: str,
        conversations: dict[str, Conversation],
    ) -> int:
        """Write the cursor and the changed conversations in one transaction.

        Returns:
            The new version.

        Raises:
            StaleCursorError: The stored version is not `expected_version`.
        """
        with self._transaction():
            guard = "" if self._unconditional else " and version = ?"
            args = (cursor, conversation_id) + (() if self._unconditional else (expected_version,))
            updated = self._db.execute(
                "update turns set cursor = ?, version = version + 1 "
                f"where conversation_id = ?{guard}",
                args,
            )
            if updated.rowcount != 1:
                raise StaleCursorError(
                    f"turn {conversation_id} is no longer at version {expected_version}"
                )
            for agent, conversation in conversations.items():
                if agent in self._drop_agents:
                    continue
                body = ConversationTypeAdapter.dump_json(conversation).decode()
                self._db.execute(
                    "insert or replace into conversations values (?, ?, ?)",
                    (conversation_id, agent, body),
                )
        row = self._db.execute(
            "select version from turns where conversation_id = ?", (conversation_id,)
        ).fetchone()
        return row[0]

    def load_conversation(self, conversation_id: str, agent: str) -> Conversation | None:
        row = self._db.execute(
            "select body from conversations where conversation_id = ? and agent = ?",
            (conversation_id, agent),
        ).fetchone()
        return None if row is None else ConversationTypeAdapter.validate_json(row[0])

    def require_conversation(self, conversation_id: str, agent: str) -> Conversation:
        stored = self.load_conversation(conversation_id, agent)
        if stored is None:
            raise LookupError(f"no stored {agent} conversation for {conversation_id}")
        return stored

    def append_frame(self, conversation_id: str, frame: dict) -> None:
        with self._transaction():
            self._db.execute(
                "insert into frames select ?, coalesce(max(seq) + 1, 0), ? "
                "from frames where conversation_id = ?",
                (conversation_id, json.dumps(frame, sort_keys=True), conversation_id),
            )

    def frames(self, conversation_id: str) -> list[dict]:
        rows = self._db.execute(
            "select frame from frames where conversation_id = ? order by seq",
            (conversation_id,),
        )
        return [json.loads(row[0]) for row in rows]

    def close(self) -> None:
        self._db.close()
