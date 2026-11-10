"""Pass line 5: the MCP overhead of `propose_plan` through the Pydantic AI client.

Measured as `harness-ab` measures it: the client's time for the call less
`HarnessPlanMcp.server_times`, the server's own work on it. The Pydantic AI side times
`MCPToolset.direct_call_tool`, which is the whole client path of a model's call (the
`process_tool_call` hook, the request, the result mapping); arm B's side is
`PlanClient.call`, the harness's `McpSurface` path. Each repetition proposes the same
accepted plan to a fresh plan session through each client, in alternating order, over a
connection held open and warmed before timing starts.

The control puts 20 ms of delay in the server's transport layer, outside the time
`server_times` records; the measure has to show it.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastmcp.server.middleware import Middleware
from pai_plan_mcp import PlanDeps, open_plan, plan_agent, plan_toolsets
from rig import TASKS, Rig, Script, b_client, call, hand_off_of, say

TASK = "series-gdp-ghana"
REPS = 33
WARMUP = 3
DELAY_S = 0.020


def summary(seconds: list[float]) -> dict[str, float]:
    ordered = sorted(seconds)
    return {
        "n": len(ordered),
        "median_ms": round(statistics.median(ordered) * 1000, 2),
        "p90_ms": round(ordered[int(0.9 * (len(ordered) - 1))] * 1000, 2),
    }


async def measure(rig: Rig) -> dict[str, dict[str, float]]:
    """Per-call overhead of `propose_plan` for each client, over `REPS` calls."""
    task = TASKS[TASK]
    body = dict(task.reference_payload)
    asks, searches = hand_off_of(task)
    mcp, view = plan_toolsets(rig.source)
    pai_client: list[float] = []
    original = mcp.direct_call_tool

    async def timed(name: str, args: dict[str, Any], **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return await original(name, args, **kwargs)
        finally:
            if name == "propose_plan":
                pai_client.append(time.perf_counter() - started)

    mcp.direct_call_tool = timed  # type: ignore[method-assign] -- the timing hook

    async def pai_propose(planner_message: str, session: str) -> None:
        script = Script([call("propose_plan", body, id="c1"), say("done")])
        await plan_agent(script.model, view, instruction="").run(
            planner_message, deps=PlanDeps(session)
        )

    out: dict[str, list[float]] = {
        "pai_client": [],
        "pai_overhead": [],
        "b_client": [],
        "b_overhead": [],
    }
    async with mcp, b_client(rig) as b:

        async def one(rep: int, arm: str) -> None:
            rig.use(TASK)
            if arm == "pai":
                opened = await open_plan(mcp, brief=task.question, asks=asks, searches=searches)
                before = len(rig.server.server_times)
                await pai_propose(opened["planner_message"], opened["plan_session"])
                client_s = pai_client[-1]
            else:
                opened = await b.call(
                    "start_plan", {"brief": task.question, "asks": asks, "searches": searches}
                )
                before = len(rig.server.server_times)
                answer = await b.call(
                    "propose_plan", {"plan_session": opened["plan_session"], **body}
                )
                assert answer["outcome"] == "ok", answer
                client_s = b.calls[-1].client_s
            (served,) = rig.server.server_times[before:]
            assert served.tool == "propose_plan"
            if rep >= WARMUP:
                out[f"{arm}_client"].append(client_s)
                out[f"{arm}_overhead"].append(client_s - served.server_s)

        for rep in range(WARMUP + REPS):
            for arm in ("pai", "b") if rep % 2 else ("b", "pai"):
                await one(rep, arm)
    return {name: summary(values) for name, values in out.items()}


class Delay(Middleware):
    """A pause in the server's transport layer, before the tool body starts."""

    async def on_call_tool(self, context: Any, call_next: Callable[[Any], Awaitable[Any]]) -> Any:
        await asyncio.sleep(DELAY_S)
        return await call_next(context)


async def test_overhead_of_propose_plan(rig: Rig) -> None:
    measured = await measure(rig)
    print("overhead", rig.transport, json.dumps(measured))
    for arm in ("pai", "b"):
        assert measured[f"{arm}_overhead"]["n"] == REPS >= 30
        assert measured[f"{arm}_overhead"]["median_ms"] > 0


async def test_control_a_delay_outside_the_server_time_shows_in_the_overhead(rig: Rig) -> None:
    clean = await measure(rig)
    rig.server.mcp.add_middleware(Delay())
    delayed = await measure(rig)
    print(
        "control",
        rig.transport,
        "median overhead ms:",
        clean["pai_overhead"]["median_ms"],
        "->",
        delayed["pai_overhead"]["median_ms"],
    )
    for arm in ("pai", "b"):
        gain = delayed[f"{arm}_overhead"]["median_ms"] - clean[f"{arm}_overhead"]["median_ms"]
        assert gain >= DELAY_S * 1000 * 0.9
