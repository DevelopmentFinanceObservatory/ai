"""Plan sessions kept in SQLite, so a restart of plan-mcp keeps every open session.

A session is one row: its id, its caller, and one JSON document holding the
session state, the conversation's spec blocks, the planner's first message, the
ADK events the planner's readers walk, each version declared ready, the answers
and approvals with who gave them, and each run's version, status and result. A
run in flight when the process stopped loads as `interrupted`, and the same
version may run again.

`plan_tools.MemorySessions` keeps nothing past the process, as plan-mcp did
before.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from google.adk.events import Event
from harness.studio_state import ConversationState
from plan_tools import INTERRUPTED, PlanSession, Run
from planner.plan import Plan


def dump(session: PlanSession) -> str:
    """`session` as the JSON document a row holds."""
    return json.dumps(
        {
            "state": session.state,
            "turn_specs": session.conversation.turn_specs,
            "first_message": session.first_message,
            "events": [
                event.model_dump(mode="json", exclude_none=True) for event in session.events
            ],
            "ready": {str(v): plan.model_dump(mode="json") for v, plan in session.ready.items()},
            "answers": session.answers,
            "approvals": {str(v): record for v, record in session.approvals.items()},
            "runs": {
                run_id: {"version": run.version, "status": run.status, "result": run.result}
                for run_id, run in session.runs.items()
            },
        },
    )


def load(session_id: str, caller: str | None, document: str) -> PlanSession:
    """The session a row's `document` holds."""
    held: dict[str, Any] = json.loads(document)
    conversation = ConversationState(conversation_id=session_id)
    conversation.turn_specs = dict(held["turn_specs"])
    runs = {}
    for run_id, run in held["runs"].items():
        finished = run["result"] is not None
        runs[run_id] = Run(
            run_id,
            run["version"],
            task=None,
            status=run["status"] if finished else INTERRUPTED,
            result=run["result"],
        )
    return PlanSession(
        session_id=session_id,
        caller=caller,
        state=held["state"],
        conversation=conversation,
        first_message=held["first_message"],
        events=[Event.model_validate(event) for event in held["events"]],
        ready={int(v): Plan.model_validate(plan) for v, plan in held["ready"].items()},
        answers=list(held["answers"]),
        approvals={int(v): record for v, record in held["approvals"].items()},
        runs=runs,
    )


class SqliteSessions:
    """Sessions in one SQLite file at `path`, written whole on every change."""

    def __init__(self, path: Path | str) -> None:
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS plan_sessions ("
                " session_id TEXT PRIMARY KEY, caller TEXT, document TEXT NOT NULL,"
                " updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )

    def save(self, session: PlanSession) -> None:
        document = dump(session)
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO plan_sessions (session_id, caller, document) VALUES (?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET document = excluded.document, "
                "updated_at = CURRENT_TIMESTAMP",
                (session.session_id, session.caller, document),
            )

    def load(self, session_id: str) -> PlanSession | None:
        with self._lock:
            row = self._db.execute(
                "SELECT caller, document FROM plan_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return None if row is None else load(session_id, row[0], row[1])

    def close(self) -> None:
        self._db.close()
