"""What the planner's model reads in each arm, what the change moves in the harness's
code, and what `run_plan` costs beside dev's `execute` node. Needs no model.

    python measure.py            # writes runs/measurements.json and prints it

- `declarations`: each tool declaration and the instruction, as compact JSON
  characters, arm A (the served planner) beside arm B (`McpPlannerSeam`).
- `results`: what the model reads back for the same call in each arm, and what it
  would read through ADK's `McpToolset` instead, which hands the model the whole
  `CallToolResult` (the text block and `structuredContent` both).
- `code_lines`: the line spans, at dev 190c7a9c, of the harness code a planner on
  plan-mcp would stop running in the harness process, and the lines it adds.
- `run_plan`: dev's `execute` and `plan_done` nodes run in process on the canned
  plan, against `run_plan` on plan-mcp over each transport, interleaved.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import json
import statistics
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ab import Current, build_server, http_source, run_b
from checks import COMBINATION_ASK, SCRIPTED_OPENING, scripted_server
from compiler.surface import McpSurface
from entry import (
    Probe,
    RecordedSurface,
    build_seam,
    load_recording,
    load_tasks,
    quiet_logs,
    run_entry,
    tool_calls,
)
from mcp_planner import McpPlannerSeam, PlanClient
from scripted import PROPOSAL
from validate import ScriptedLlm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEV = "190c7a9c"
TASKS = {task.id: task for task in load_tasks()}


def chars(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str))


# --- Declarations ----------------------------------------------------------------


def _declared(tools: Sequence[Any]) -> dict[str, int]:
    out = {}
    for tool in tools:
        declaration = tool._get_declaration()
        out[tool.name] = chars(declaration.model_dump(mode="json", exclude_none=True))
    return out


async def declarations(client: PlanClient) -> dict[str, Any]:
    surface = RecordedSurface({})
    a = build_seam(probe=Probe(), surface=surface, prefix_cache=None)
    b = McpPlannerSeam(client=client, hand_off=_no_hand_off, surface=surface, prefix_cache=None)
    sides = {}
    for name, seam in (("A", a), ("B", b)):
        tools = _declared(seam._agent.tools)
        instruction = len(str(seam._agent.instruction))
        sides[name] = {
            "tools": tools,
            "instruction": instruction,
            "total": sum(tools.values()) + instruction,
        }
    return sides


async def _no_hand_off() -> tuple[list[Any], list[dict[str, Any]]]:
    raise AssertionError("declarations open no session")


# --- Results ---------------------------------------------------------------------


def _toolset_form(answer: Mapping[str, Any]) -> dict[str, Any]:
    """`answer` as ADK's `McpTool` hands it to the model: the `CallToolResult` plan-mcp
    sends, its text block holding the answer as compact JSON beside `structuredContent`."""
    text = json.dumps(answer, separators=(",", ":"), default=str)
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": dict(answer),
        "isError": False,
    }


async def _answers(
    task_id: str, body: Mapping[str, Any], client: PlanClient, current: Current
) -> tuple[dict[str, Any], dict[str, Any]]:
    task = TASKS[task_id]
    recording = load_recording(task_id)
    a = await run_entry(
        task,
        recording,
        prefix_cache=None,
        model=ScriptedLlm(script=[[("propose_plan", dict(body))]]),
        timeout_s=60,
    )
    b = await run_b(
        task,
        recording,
        client=client,
        current=current,
        prefix_cache=None,
        model=ScriptedLlm(script=[[("propose_plan", dict(body))]]),
        timeout_s=60,
    )

    def first(run: Any) -> dict[str, Any]:
        return next(c["response"] for c in tool_calls(run.events) if c["name"] == "propose_plan")

    return first(a), first(b)


async def results(client: PlanClient, current: Current) -> dict[str, Any]:
    growth = copy.deepcopy(TASKS["growth-total-oda-ghana"].reference_payload)
    growth["evidence_requirements"][0]["period"] = {"start": "twenty twelve", "end": 2021}
    cases = {
        "combination_unasked (sum-oda-oof-ethiopia reference)": (
            "sum-oda-oof-ethiopia",
            TASKS["sum-oda-oof-ethiopia"].reference_payload,
        ),
        "invalid_plan (growth-total-oda-ghana, period.start 'twenty twelve')": (
            "growth-total-oda-ghana",
            growth,
        ),
    }
    out = {}
    for label, (task_id, body) in cases.items():
        a, b = await _answers(task_id, body, client, current)
        out[label] = {
            "A": chars(a),
            "B": chars(b),
            "B through McpToolset": chars(_toolset_form(b)),
            "A code": a.get("code"),
            "B code": b.get("code"),
            "B adds": sorted(set(b) - set(a)),
        }
    out["record_user_answer ask, as the model sends it"] = {"chars": chars(COMBINATION_ASK)}
    return out


# --- Code --------------------------------------------------------------------------

# The harness code a planner on plan-mcp no longer runs in the harness process, by
# file and top-level name (`Class.method` for one method).
GOES = {
    "services/harness/src/planner/agent.py": [
        "ToolContextSession",
        "build_planner_tools",
        "_AskTool",
        "declare_ready",
        "_execution_blocked",
        "_approval_summary",
    ],
    "services/harness/src/harness/planner_seam.py": [
        "PlannerSeam.__call__",
        "PlannerSeam._persist_durable",
        "PlannerSeam.revise",
        "PlannerSeam._revised",
        "store_searched_pin_readings",
        "_reading_pin_sources",
        "_readings_ledger",
    ],
    "services/harness/src/harness/graph/execute.py": ["*"],
}
# What arm B adds, in this directory, and plan-mcp itself.
COMES = {
    "harness-ab/mcp_planner.py": "the seam and tool wrappers",
    "plan_tools.py": "plan-mcp: sessions over the planner's tools",
    "server.py": "plan-mcp: the MCP server",
    "outward.py": "plan-mcp: answers as an outside model reads them",
    "execution.py": "plan-mcp: dev's execute node under a Runner",
    "store.py": "plan-mcp: sessions in SQLite",
}


def _spans(source: str, names: Sequence[str]) -> dict[str, int]:
    tree = ast.parse(source)
    found: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        assert node.end_lineno is not None
        if node.name in names:
            found[node.name] = node.end_lineno - node.lineno + 1
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                key = f"{node.name}.{getattr(item, 'name', '')}"
                if key in names and item.end_lineno is not None:
                    found[key] = item.end_lineno - item.lineno + 1
    return found


def code_lines() -> dict[str, Any]:
    goes = {}
    for path, names in GOES.items():
        source = subprocess.run(
            ["git", "show", f"{DEV}:{path}"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
        goes[path] = (
            {"whole file": len(source.splitlines())} if names == ["*"] else _spans(source, names)
        )
    comes = {name: len((HERE.parent / name).read_text().splitlines()) for name in COMES if name}
    return {"goes_at_dev": goes, "comes": comes}


# --- run_plan against dev's execute node -----------------------------------------


async def run_plan_latency(reps: int) -> dict[str, Any]:
    scripted = scripted_server()
    server = scripted.server
    times: dict[str, list[float]] = {"dev execute": [], "run_plan inproc": [], "run_plan http": []}
    async with http_source(server) as url:  # type: ignore[arg-type]
        clients = {
            "inproc": await PlanClient.connect(McpSurface(sources={"plan-mcp": server.mcp})),
            "http": await PlanClient.connect(McpSurface(sources={"plan-mcp": url})),
        }
        try:
            for rep in range(reps + 1):
                for transport in ("inproc", "http"):
                    client = clients[transport]
                    opened = await client.call("start_plan", SCRIPTED_OPENING)
                    session = opened["plan_session"]
                    accepted = await client.call(
                        "propose_plan", {"plan_session": session, **PROPOSAL}
                    )
                    assert "summary" in accepted, accepted
                    started = time.perf_counter()
                    ran = await client.call(
                        "run_plan", {"plan_session": session, "version": 1, "approved": True}
                    )
                    took = time.perf_counter() - started
                    assert ran.get("status") == "finished", ran
                    if rep:
                        times[f"run_plan {transport}"].append(took)
                    held = server.service.sessions[session]
                    plan = held.ready[1].approve(at=datetime.now(UTC))
                    started = time.perf_counter()
                    done = await server.service._run_graph(held, plan)
                    took = time.perf_counter() - started
                    assert done["status"] == "finished", done
                    if rep and transport == "inproc":
                        times["dev execute"].append(took)
        finally:
            for client in clients.values():
                await client.surface.aclose()
    return {
        name: {
            "n": len(values),
            "median_ms": round(statistics.median(values) * 1000, 2),
            "p90_ms": round(sorted(values)[int(0.9 * (len(values) - 1))] * 1000, 2),
            "mean_ms": round(statistics.fmean(values) * 1000, 2),
        }
        for name, values in times.items()
    }


async def main() -> None:
    quiet_logs()
    current = Current()
    server = build_server(current)
    surface = McpSurface(sources={"plan-mcp": server.mcp})
    client = await PlanClient.connect(surface)
    try:
        measured = {
            "declarations": await declarations(client),
            "results": await results(client, current),
            "code_lines": code_lines(),
            "run_plan": await run_plan_latency(30),
        }
    finally:
        await surface.aclose()
    (HERE / "runs" / "measurements.json").write_text(json.dumps(measured, indent=2) + "\n")
    print(json.dumps(measured, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
