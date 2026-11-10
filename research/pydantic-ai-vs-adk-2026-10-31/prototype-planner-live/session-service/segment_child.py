"""One process's share of a conversation, run as a pytest test of its own.

The parent (`test_session_service.py`) writes a spec file and starts this file under pytest, so
the child runs under the same fixtures as the harness suite (no `.env`, no shared secret, no
graph calls). The spec names the arm, the script and which of its planner answers this
process's model gets, the actions to post, and where to write what they answered. The database
URL comes from `SESSION_DB_URL`. A process that blocks in the translation (`block`) never
finishes; its parent kills it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from entry import quiet_logs
from google.adk.sessions import DatabaseSessionService
from pai_node import PaiPlannerSeam
from planner.call import Step
from rig import SCRIPTS, build, run_actions

quiet_logs()


def _only_the_message(self: PaiPlannerSeam, ctx: Any) -> list[Step]:
    return [Step("message")]


PATCHES = {"no-earlier-steps": ("_earlier", _only_the_message)}


async def test_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = json.loads(Path(os.environ["SEGMENT_SPEC"]).read_text())
    task, steps = SCRIPTS[spec["script"]]
    service = DatabaseSessionService(db_url=os.environ["SESSION_DB_URL"])
    if spec.get("patch"):
        name, replacement = PATCHES[spec["patch"]]
        monkeypatch.setattr(PaiPlannerSeam, name, replacement)
    skew = spec.get("clock_skew", 0)
    block = Path(spec["block"]) if spec.get("block") else None
    try:
        rig = await build(
            spec["arm"],
            task,
            steps[spec.get("first", 0) : spec.get("last")],
            service=service,
            clock=lambda: time.time() + skew,
            pai=spec.get("pai"),
            block=block,
        )
        answered = await run_actions(rig, task, spec["actions"])
        Path(spec["out"]).write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "actions": answered,
                    "planner_requests": len(rig.client.calls),
                    "translations": len(rig.translations),
                }
            )
        )
    finally:
        await service.close()
