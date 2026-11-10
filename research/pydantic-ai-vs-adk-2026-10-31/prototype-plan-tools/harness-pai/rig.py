"""What the tests here drive: plan-mcp over a transport, `harness-ab`'s arm B beside it,
and the Pydantic AI planner with a scripted model.

`Rig` is one `HarnessPlanMcp` (as `harness-ab/ab.py` builds it, answering from whichever
recorded entry `Rig.use` names) reached in process or over loopback Streamable HTTP.
`b_client` and `b_run` are arm B's side of it, `harness-ab`'s own code.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ab import Current, build_server, http_source, run_b
from compiler.surface import McpSurface
from entry import (
    EntryRun,
    RecordedResolver,
    RecordedSurface,
    Task,
    load_recording,
    load_tasks,
    tool_calls,
)
from mcp_planner import HarnessPlanMcp, McpPlannerSeam, PlanClient
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition
from validate import ScriptedLlm

TASKS = {task.id: task for task in load_tasks()}
TRANSPORTS = ["inproc", "http"]


@dataclass
class Rig:
    """plan-mcp reached at `source`: its FastMCP server in process, or its URL."""

    server: HarnessPlanMcp
    current: Current
    source: Any
    transport: str

    def use(self, task_id: str) -> Task:
        """Make plan-mcp answer for `task_id`'s recording, and return the task."""
        recording = load_recording(task_id)
        self.current.resolver = RecordedResolver(recording["searches"])
        self.current.surface = RecordedSurface(recording.get("tools") or {})
        self.current.plan = None
        return TASKS[task_id]


@contextlib.asynccontextmanager
async def serve(transport: str) -> AsyncIterator[Rig]:
    current = Current()
    server = build_server(current)
    async with contextlib.AsyncExitStack() as stack:
        source: Any = server.mcp
        if transport == "http":
            source = await stack.enter_async_context(http_source(server))
        yield Rig(server=server, current=current, source=source, transport=transport)


def hand_off_of(task: Task) -> tuple[list[Any], list[dict[str, Any]]]:
    """The orchestrator's hand-off a plan session opens from: its asks and searches."""
    asks = [ask if isinstance(ask, str) else ask.model_dump() for ask in task.asks]
    return asks, [search.model_dump() for search in task.searches]


# --- Arm B (harness-ab) ----------------------------------------------------------


@contextlib.asynccontextmanager
async def b_client(rig: Rig) -> AsyncIterator[PlanClient]:
    surface = McpSurface(sources={"plan-mcp": rig.source})
    try:
        yield await PlanClient.connect(surface)
    finally:
        await surface.aclose()


async def b_run(
    rig: Rig,
    client: PlanClient,
    task_id: str,
    script: Sequence[Sequence[tuple[str, Mapping[str, Any]]]],
    answers: Sequence[Mapping[str, Any]] = (),
) -> EntryRun:
    """The planner entry of `task_id` in arm B, its model scripted."""
    return await run_b(
        TASKS[task_id],
        load_recording(task_id),
        client=client,
        current=rig.current,
        prefix_cache=None,
        model=ScriptedLlm(script=[[(name, dict(args)) for name, args in turn] for turn in script]),
        answers=answers,
        timeout_s=60.0,
    )


def b_response(run: EntryRun, tool: str, *, nth: int = 0) -> dict[str, Any]:
    """What arm B's model read back from its `nth` call of `tool`."""
    return [c["response"] for c in tool_calls(run.events) if c["name"] == tool][nth]


async def b_seam(client: PlanClient) -> McpPlannerSeam:
    """Arm B's seam, whose agent holds the declarations B's model is offered."""

    async def no_hand_off() -> tuple[list[Any], list[dict[str, Any]]]:
        raise AssertionError("declarations open no session")

    return McpPlannerSeam(
        client=client, hand_off=no_hand_off, surface=RecordedSurface({}), prefix_cache=None
    )


# --- The Pydantic AI side ----------------------------------------------------------


def call(name: str, args: Mapping[str, Any], *, id: str) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=name, args=dict(args), tool_call_id=id)])


def say(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


@dataclass
class Script:
    """A scripted model: its responses in order, and what each request carried."""

    responses: list[ModelResponse]
    requests: list[list[ModelMessage]] = field(default_factory=list)
    offered: list[list[ToolDefinition]] = field(default_factory=list)
    instructions: list[str | None] = field(default_factory=list)

    @property
    def model(self) -> FunctionModel:
        return FunctionModel(self._respond)

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.requests.append(list(messages))
        self.offered.append(list(info.function_tools))
        self.instructions.append(info.instructions)
        return self.responses[len(self.requests) - 1]
