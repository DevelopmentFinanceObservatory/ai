"""Planner entries in two arms over the planner-entry eval's recordings.

Arm A is `entry.run_entry` as `evals/planner_entry/run.py` runs it: the served
planner with its planning tools in process. Arm B is the same entry with the
planner seam swapped for `mcp_planner.McpPlannerSeam`, whose `start_plan`,
`propose_plan` and `record_user_answer` go to plan-mcp through `McpSurface`,
either in process (FastMCP's memory transport) or over loopback Streamable HTTP
(uvicorn on 127.0.0.1, in the same event loop, as the harness reaches its own
mounted data-mcp). Everything else in `run_entry` is shared: the recorded
searches and skill reads, the graph's `route`, `plan_start` and `planner` nodes,
the turn budget, the prefix cache, and the probe that counts each model call.

plan-mcp serves every B entry from one `HarnessPlanMcp`, its searches, skill
reads and first-year reads answered from the entry's recording (`Current`).
Its translation is the eval's stop: like `entry.Probe.translate` in arm A, it
records the plan and translates nothing, and the harness side ends the entry
on seeing it, so both arms stop at the first plan `propose_plan` records.
Entries run one at a time, which `Current` relies on.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import entry
import uvicorn
from compiler.surface import McpSurface, ToolCallResult, ToolSchema
from entry import (
    EntryRun,
    PlanRecorded,
    Probe,
    RecordedResolver,
    RecordedSurface,
    RequestEdit,
    Task,
    entry_record,
    run_entry,
)
from google.adk.models.base_llm import BaseLlm
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.genai import types
from harness.prefix_cache import PrefixCache
from mcp_planner import HarnessPlanMcp, McpPlannerSeam, PlanClient
from outward import ASK_TOOL
from plan_tools import PlanningService
from planner.plan import Plan, load_plan

STOP_KEY = "eval_plan_recorded"
ARMS = ("A", "B-inproc", "B-http", "B-served")


@dataclass
class Current:
    """The entry plan-mcp answers for: its recording's searches and reads, and the
    plan its first recorded proposal left."""

    resolver: RecordedResolver | None = None
    surface: RecordedSurface | None = None
    plan: Plan | None = None


class _Resolver:
    def __init__(self, current: Current) -> None:
        self._current = current

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        assert self._current.resolver is not None, "no entry is running"
        return await self._current.resolver(**kwargs)


class _Surface:
    def __init__(self, current: Current) -> None:
        self._current = current

    async def list_tools(self) -> list[ToolSchema]:
        return []

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        assert self._current.surface is not None, "no entry is running"
        return await self._current.surface.call_tool(**kwargs)


def _stop_translate(current: Current) -> Any:
    async def translate(tool_context: Any) -> dict[str, object]:
        current.plan = load_plan(tool_context.state)
        return {"outcome": "ok", STOP_KEY: True}

    return translate


async def _no_run(*args: Any) -> dict[str, Any]:
    raise AssertionError("a planner entry runs no plan")


def build_server(current: Current) -> HarnessPlanMcp:
    """plan-mcp answering from whichever entry `current` holds."""
    service = PlanningService(
        resolver=_Resolver(current),
        surface=_Surface(current),
        translate=_stop_translate(current),
        run_graph=_no_run,
    )
    return HarnessPlanMcp(service)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@contextlib.asynccontextmanager
async def http_source(server: HarnessPlanMcp) -> AsyncIterator[str]:
    """plan-mcp served over Streamable HTTP on a loopback port in this event loop."""
    port = _free_port()
    config = uvicorn.Config(
        server.mcp.http_app(path="/mcp"), host="127.0.0.1", port=port, log_level="warning"
    )
    served = uvicorn.Server(config)
    task = asyncio.create_task(served.serve())
    try:
        while not served.started:
            if task.done():
                task.result()
            await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        served.should_exit = True
        await task


@contextlib.asynccontextmanager
async def plan_client(server: HarnessPlanMcp, transport: str) -> AsyncIterator[PlanClient]:
    """The harness's client of `server` over `transport` (`inproc` or `http`), warmed
    with one call so its held connection is open before an entry is timed."""
    async with contextlib.AsyncExitStack() as stack:
        source: Any = server.mcp
        if transport == "http":
            source = await stack.enter_async_context(http_source(server))
        surface = McpSurface(sources={"plan-mcp": source})
        stack.push_async_callback(surface.aclose)
        client = await PlanClient.connect(surface)
        await client.call("propose_plan", {"plan_session": "ps_warm"})
        client.calls.clear()
        yield client


def _hand_off(task: Task) -> Any:
    async def hand_off() -> tuple[list[Any], list[dict[str, Any]]]:
        asks = [ask if isinstance(ask, str) else ask.model_dump() for ask in task.asks]
        return asks, [search.model_dump() for search in task.searches]

    return hand_off


async def run_b(
    task: Task,
    recording: Mapping[str, Any],
    *,
    client: PlanClient,
    current: Current,
    prefix_cache: PrefixCache | None,
    model: BaseLlm | None = None,
    answers: Sequence[Mapping[str, Any]] = (),
    edits: Sequence[RequestEdit] = (),
    timeout_s: float = 300.0,
    served_declaration: bool = False,
) -> EntryRun:
    """`entry.run_entry` for `task` with the planner's tools on plan-mcp, `propose_plan`
    declared as plan-mcp lists it or, with `served_declaration`, as the served planner
    declares it."""
    current.resolver = RecordedResolver(recording["searches"])
    current.plan = None

    def build_seam(
        *,
        probe: Probe,
        surface: RecordedSurface,
        prefix_cache: PrefixCache | None,
        edits: Sequence[RequestEdit] = (),
        model: BaseLlm | None = None,
    ) -> McpPlannerSeam:
        current.surface = surface

        def on_propose(answer: Mapping[str, Any], tool_context: Any) -> None:
            translation = answer.get("translation")
            if isinstance(translation, Mapping) and translation.get(STOP_KEY):
                probe.plan = current.plan
                probe.plan_call = tool_context.function_call_id
                raise PlanRecorded

        def edit_request(callback_context: Any, llm_request: Any) -> None:
            del callback_context
            for edit in edits:
                edit(llm_request)

        return McpPlannerSeam(
            client=client,
            hand_off=_hand_off(task),
            on_propose=on_propose,
            served_declaration=served_declaration,
            surface=surface,
            prefix_cache=prefix_cache,
            model=model,
            before_model_callbacks=[edit_request, probe.before_model],
            after_model_callbacks=[probe.after_model],
        )

    # `run_entry` resumes a paused entry by the ask's name and answers it with a
    # response under that name; in arm B the ask is plan-mcp's `record_user_answer`.
    served = (entry.build_seam, entry.REQUEST_INPUT_FUNCTION_CALL_NAME)
    entry.build_seam = build_seam  # type: ignore[assignment] -- the one seam B swaps
    entry.REQUEST_INPUT_FUNCTION_CALL_NAME = ASK_TOOL
    entry.create_request_input_response = _ask_response
    try:
        return await run_entry(
            task,
            recording,
            prefix_cache=prefix_cache,
            edits=edits,
            model=model,
            timeout_s=timeout_s,
            answers=answers,
        )
    finally:
        entry.build_seam, entry.REQUEST_INPUT_FUNCTION_CALL_NAME = served
        entry.create_request_input_response = create_request_input_response
        current.resolver = current.surface = None


def _ask_response(call_id: str, response: Mapping[str, Any]) -> types.Part:
    """The user's answer to a paused `record_user_answer` call, as the Studio's
    interrupt route sends one to the planner's own ask."""
    return types.Part(
        function_response=types.FunctionResponse(id=call_id, name=ASK_TOOL, response=dict(response))
    )


def tool_times(run: EntryRun) -> list[dict[str, Any]]:
    """Each tool call's time, from its call event to its response event, and the
    characters of the response the model reads, as compact JSON."""
    called = {
        call.id: (call.name, event.timestamp)
        for event in run.events
        for call in event.get_function_calls()
    }
    times = []
    for event in run.events:
        for response in event.get_function_responses():
            if response.id not in called:
                continue
            name, at = called[response.id]
            body = json.dumps(response.response, separators=(",", ":"), default=str)
            times.append(
                {"tool": name, "s": round(event.timestamp - at, 4), "response_chars": len(body)}
            )
    return times


@dataclass
class Arm:
    name: str
    client: PlanClient | None = None
    server: HarnessPlanMcp | None = None
    served_declaration: bool = False


def record(
    task: Task, run: EntryRun, *, arm: Arm, rep: int, model: str, revision: str, started: int
) -> dict[str, Any]:
    """`entry.entry_record` for the run, with its tool times, and in arm B each MCP
    call's client and server time. An ask awaiting the user in arm B is named
    `record_user_answer`, so its ending is read as `entry.stop_reason` reads arm A's."""
    out = entry_record(
        task=task,
        run=run,
        arm=arm.name,
        rep=rep,
        model=model,
        revision=revision,
        experiment=arm.name,
    )
    if out["stop_reason"] not in ("accepted", "timeout", "error") and any(
        call["name"] == ASK_TOOL and call["response"] is None
        for call in entry.tool_calls(run.events)
    ):
        out["stop_reason"] = "asked_user"
    out["tool_times"] = tool_times(run)
    if arm.client is not None and arm.server is not None:
        out["mcp_calls"] = [
            {"tool": c.tool, "client_s": round(c.client_s, 4), "chars": c.chars}
            for c in arm.client.calls
        ]
        out["server_times"] = [
            {"tool": t.tool, "server_s": round(t.server_s, 4)}
            for t in arm.server.server_times[started:]
        ]
        arm.client.calls.clear()
    out["at"] = time.time()
    return out
