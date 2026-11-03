"""I7: sessions outlive a restart, and the tools carry a version.

Sessions, answers, approvals and runs persist in SQLite (`store.SqliteSessions`):
a second server over the same file, as after a restart, carries an open session
on, with its plan, approvals and finished runs. Without the store the second
server knows no session. A run the restart cut short loads `interrupted`, and its
version runs again.

Each tool is listed at version 2. A later schema registers its tools at version 3
on the same server: a listing shows the newest, and a client that pins version 2
(`_meta.fastmcp.version`) keeps calling this one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from checks import SCRIPTED_OPENING, call, client_of, opening, recorded_server, scripted_server
from entry import Task, load_tasks
from fastmcp import Client
from fastmcp.tools import Tool
from fastmcp.tools.base import ToolResult
from mcp.types import TextContent
from plan_tools import INTERRUPTED, SESSION_UNKNOWN
from scripted import PROPOSAL, amended
from store import SqliteSessions, dump, load
from validate import task_payloads

ROUND_TRIP = load_tasks()[::10]


async def _first_life(path: Path) -> str:
    """Open a session, accept and run version 1, all on one server over `path`."""
    first = scripted_server(store=SqliteSessions(path))
    async with client_of(first.server) as client:
        session = (await call(client, "start_plan", SCRIPTED_OPENING))["plan_session"]
        await call(client, "propose_plan", {"plan_session": session, **PROPOSAL})
        ran = await call(
            client, "run_plan", {"plan_session": session, "version": 1, "approved": True}
        )
    assert ran["status"] == "finished", ran
    return session


async def _second_life(store: Any, session: str) -> dict[str, Any]:
    second = scripted_server(store=store)
    async with client_of(second.server) as client:
        seen = {
            "ran_again": await call(client, "run_plan", {"plan_session": session, "version": 1}),
            "amended": await call(client, "propose_plan", {"plan_session": session, **amended(1)}),
            # Version 1 ran before the amendment: asking again finds that run.
            "old": await call(client, "run_plan", {"plan_session": session, "version": 1}),
            "new": await call(
                client, "run_plan", {"plan_session": session, "version": 2, "approved": True}
            ),
        }
    return seen


async def test_i7_a_session_outlives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite3"
    session = await _first_life(path)
    seen = await _second_life(SqliteSessions(path), session)
    print("I7:", {k: (v.get("status"), v.get("version"), v.get("code")) for k, v in seen.items()})
    assert (
        seen["ran_again"]["status"] == "finished" and seen["ran_again"]["approved_by"] == "client"
    )
    assert seen["amended"]["version"] == 2 and "summary" in seen["amended"]
    assert seen["old"]["run_id"] == seen["ran_again"]["run_id"]
    assert seen["new"]["status"] == "finished"
    assert "period 1990-2023" in seen["new"]["hand_back"]["plan_record"]
    stored = SqliteSessions(path).load(session)
    assert stored is not None and sorted(stored.approvals) == [1, 2]
    assert {run.status for run in stored.runs.values()} == {"finished"}


async def test_i7_control_without_the_store_a_restart_loses_the_session(tmp_path: Path) -> None:
    session = await _first_life(tmp_path / "sessions.sqlite3")
    seen = await _second_life(None, session)
    print("I7 control:", {k: v.get("code") for k, v in seen.items()})
    assert all(answer["code"] == SESSION_UNKNOWN for answer in seen.values())


async def test_i7_a_run_cut_short_runs_again(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite3"
    session_id = await _first_life(path)
    held = SqliteSessions(path).load(session_id)
    assert held is not None
    document = json.loads(dump(held))
    for run in document["runs"].values():
        run["status"], run["result"] = "running", None
    cut = load(session_id, held.caller, json.dumps(document))
    assert [run.status for run in cut.runs.values()] == [INTERRUPTED]
    store = SqliteSessions(path)
    store.save(cut)
    second = scripted_server(store=store)
    async with client_of(second.server) as client:
        ran = await call(client, "run_plan", {"plan_session": session_id, "version": 1})
    assert ran["status"] == "finished"
    assert len(second.server.service.sessions[session_id].runs) == 2


@pytest.mark.parametrize("task", ROUND_TRIP, ids=[task.id for task in ROUND_TRIP])
async def test_i7_a_proposed_session_round_trips_exactly(task: Task) -> None:
    recorded = await recorded_server(task)
    async with Client(recorded.server.mcp) as client:
        opened = await call(client, "start_plan", opening(task))
        body = {"plan_session": opened["plan_session"], **task_payloads(task)[0].body}
        await call(client, "propose_plan", body)
    held = recorded.server.service.sessions[opened["plan_session"]]
    again = load(held.session_id, held.caller, dump(held))
    # State is held as JSON, as dev's database session service holds it: a tuple
    # comes back a list.
    assert again.state == json.loads(json.dumps(held.state))
    assert [e.model_dump(mode="json") for e in again.events] == [
        e.model_dump(mode="json") for e in held.events
    ]
    assert again.ready == held.ready
    assert again.conversation.turn_specs == held.conversation.turn_specs


class _NextProposePlan(Tool):
    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text="version 3")])


async def test_i7_a_later_version_lives_beside_this_one() -> None:
    scripted = scripted_server()
    scripted.server.mcp.add_tool(
        _NextProposePlan(name="propose_plan", parameters={"type": "object"}, version="3")
    )
    async with Client(scripted.server.mcp) as client:
        listed = {tool.name: tool.meta for tool in await client.list_tools()}
        opened = await call(client, "start_plan", SCRIPTED_OPENING)
        pinned = await client.call_tool(
            "propose_plan",
            {"plan_session": opened["plan_session"], **PROPOSAL},
            version="2",
            raise_on_error=False,
        )
        newest = await client.call_tool("propose_plan", {}, raise_on_error=False)
    print("I7 versions:", listed["propose_plan"]["fastmcp"])
    assert listed["propose_plan"]["fastmcp"]["versions"] == ["3", "2"]
    assert (pinned.structured_content or {}).get("version") == 1
    assert newest.content[0].text == "version 3"  # type: ignore[union-attr]
