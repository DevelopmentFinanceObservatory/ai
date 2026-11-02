"""One process of X2, run by `test_kill.py` as its own pytest session so it gets
the harness suite's autouse isolation. Not collected on its own.

`BG_CHILD` holds the arguments as JSON:

- `arm`: `"durable"` or `"dev"`; `executor`: this process's DBOS executor id.
- `sessions`, `ledger`, `dbos`: the SQLite session database, the run ledger's
  JSON file and the DBOS system database, shared by every process.
- `steps`: `"turn"` posts the brief and reads its plan card; `"approve"`
  approves the card, or the interrupt `interrupt_id` names.
- `hold`: tools whose call never returns, so the parent can kill the process
  inside them. `clock_offset`: seconds added to the Studio store's clock, so a
  segment lease the killed process left behind has expired. `plans`: how many
  `plan` calls the scripted orchestrator makes before it answers "done".

Each line it prints that starts with `{` is one JSON record: `tool` lines from
the surface, `card`, `frames` and `end`.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from dbos import DBOS
from durable_execute import OPENED, launch
from fakes import FileRunLedger, SlowSurface
from google.adk.sessions import DatabaseSessionService
from harness.conversation import current_conversation_id
from harness.graph.state import AGENT_NAME
from harness.studio_state import StudioStore
from studio_app import BRIEF, build_app, interrupt_body
from tests.orchestrator_fakes import handed_back, stream, studio_client, turn_body


def emit(record: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(record) + "\n")
    sys.stdout.flush()


async def test_child(monkeypatch: pytest.MonkeyPatch) -> None:
    args = json.loads(os.environ["BG_CHILD"])
    offset = float(args.get("clock_offset", 0))
    session_service = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{args['sessions']}")
    store = StudioStore(session_service, app_name=AGENT_NAME, clock=lambda: time.time() + offset)
    ledger = FileRunLedger(Path(args["ledger"]), scope=current_conversation_id.get)
    surface = SlowSurface(
        log_file=Path(args["tool_log"]), echo=True, hold_forever=frozenset(args.get("hold", ()))
    )
    if args["arm"] == "durable":
        launch(Path(args["dbos"]), args["executor"])
    try:
        built = build_app(
            monkeypatch,
            arm=args["arm"],
            surface=surface,
            session_service=session_service,
            ledger=ledger,
            store=store,
            plans=int(args.get("plans", 1)),
        )
        interrupt_id = args.get("interrupt_id")
        async with studio_client(built.app) as client:
            if "turn" in args["steps"]:
                first = await stream(client, "/api/studio/turn", turn_body(BRIEF))
                interrupt_id = next(f for f in first if f["event"] == "plan.proposed")[
                    "interrupt_id"
                ]
                emit({"type": "card", "interrupt_id": interrupt_id})
            if "approve" in args["steps"]:
                assert interrupt_id is not None
                ran = await stream(
                    client, "/api/studio/interrupt", interrupt_body(interrupt_id, "approve")
                )
                emit(
                    {
                        "type": "frames",
                        "frames": [(f["event"], f.get("status"), f.get("tool")) for f in ran],
                    }
                )
        hand_back: dict[str, Any] | None = None
        if built.model.requests:
            try:
                _, hand_back = handed_back(built.model)
            except ValueError:
                hand_back = None
        books = json.loads(Path(args["ledger"]).read_text())
        emit(
            {
                "type": "end",
                "opened": OPENED,
                "hand_back": hand_back,
                "runs": [run for book in books.values() for run in book["runs"].values()],
            }
        )
    finally:
        if args["arm"] == "durable":
            DBOS.destroy()
