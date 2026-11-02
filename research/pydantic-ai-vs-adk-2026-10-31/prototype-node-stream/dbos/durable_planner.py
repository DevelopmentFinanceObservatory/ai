"""The planner made durable with DBOS: one workflow per planner segment.

A segment runs the planner from the brief, or from a paused `Conversation` and the
user's answer, to the next `DeferredToolRequests` or the final text. Its input and
output are plain strings and dicts. The agent and its `Deps` are not inputs: the
workflow reads them from the process's `Runtime`, so a recovered workflow uses the
recovering process's own.

Progress reaches the node through `DBOSDurability(event_stream_handler=...)`. Each
`PlanProgress` a function tool emits runs that handler in its own DBOS step, and the
handler hands the event to the node over a `Sink` keyed by workflow id. A sink is
thread-safe, because a workflow DBOS recovers runs on DBOS's own event loop thread
while the node waits on the request's loop.
"""

import asyncio
import json
import os
import threading
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from dbos import DBOS
from planner_agent import Deps, PlanProgress, build_planner
from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    Conversation,
    ConversationTypeAdapter,
    DeferredToolRequests,
    RunContext,
)
from pydantic_ai.durable_exec.dbos import DBOSDurability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCalls, FunctionModel
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.toolsets import FunctionToolset

PLANNER_CONVERSATION_ID = "planner"
SEGMENT_WORKFLOW = "planner_segment"
AGENT_NAME = "planner"
BUFFERED_AGENT_NAME = "planner_buffered"


@dataclass(kw_only=True)
class DurableDeps(Deps):
    """`Deps` whose log file lines are JSON carrying the process id and the DBOS
    workflow id they ran under, and whose model can sleep before each call.

    `model_delay` makes every model call wait that long, logging `slow model start`
    and `slow model end` around the wait, so a cancel can land inside a call.
    """

    model_delay: float = 0.0
    lines: list[dict[str, Any]] = field(default_factory=list)

    def record(self, line: str) -> None:
        entry = {"pid": os.getpid(), "wf": DBOS.workflow_id, "line": line}
        self.log.append(line)
        self.lines.append(entry)
        if self.log_file is not None:
            with self.log_file.open("a") as handle:
                handle.write(json.dumps(entry) + "\n")


class SegmentStart(TypedDict):
    """A segment's workflow input. `answer` answers the stored conversation's one
    deferred call; `buffered` runs the agent with no live handler."""

    prompt: str | None
    conversation: str | None
    answer: str | None
    buffered: bool


class SegmentEnd(TypedDict):
    """A segment's workflow output. `progress` is filled only by a buffered run."""

    conversation: str
    asked: bool
    text: str | None
    progress: list[dict[str, int]]


def progress_payload(event: PlanProgress) -> dict[str, int]:
    return {"call": event.call, "done": event.done, "total": event.total}


# Sinks ------------------------------------------------------------------------------


class Sink:
    """Progress for one workflow id. Items delivered before a node attaches wait in
    `buffer`; afterwards they go straight to the node's queue on the node's loop."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.buffer: list[dict[str, int]] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queue: asyncio.Queue[dict[str, int]] | None = None


_SINKS: dict[str, Sink] = {}
_SINKS_LOCK = threading.Lock()


def _sink(workflow_id: str) -> Sink:
    with _SINKS_LOCK:
        return _SINKS.setdefault(workflow_id, Sink())


def deliver(workflow_id: str, item: dict[str, int]) -> None:
    """Hand one progress item to whoever reads `workflow_id`, from any thread."""
    sink = _sink(workflow_id)
    with sink.lock:
        if sink.queue is None or sink.loop is None:
            sink.buffer.append(item)
        else:
            sink.loop.call_soon_threadsafe(sink.queue.put_nowait, item)


def attach(workflow_id: str) -> asyncio.Queue[dict[str, int]]:
    """The queue the calling loop reads `workflow_id`'s progress from, holding what
    arrived before it attached."""
    sink = _sink(workflow_id)
    queue: asyncio.Queue[dict[str, int]] = asyncio.Queue()
    with sink.lock:
        for item in sink.buffer:
            queue.put_nowait(item)
        sink.buffer.clear()
        sink.loop, sink.queue = asyncio.get_running_loop(), queue
    return queue


def detach(workflow_id: str) -> None:
    with _SINKS_LOCK:
        _SINKS.pop(workflow_id, None)


# The agents and the workflow ----------------------------------------------------------


async def relay_progress(
    ctx: RunContext[DurableDeps], stream: AsyncIterable[AgentStreamEvent]
) -> None:
    """The durability handler: each `PlanProgress` goes to its workflow's sink."""
    async for event in stream:
        if isinstance(event, PlanProgress):
            ctx.deps.record(f"progress handler call={event.call} done={event.done}")
            workflow_id = DBOS.workflow_id
            assert workflow_id is not None
            deliver(workflow_id, progress_payload(event))


def _slowed(deps: DurableDeps, base: FunctionModel) -> FunctionModel:
    stream_function = base.stream_function
    assert stream_function is not None

    async def slow(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[Any]:
        if deps.model_delay:
            deps.record("slow model start")
            await asyncio.sleep(deps.model_delay)
            deps.record("slow model end")
        chunk: str | DeltaToolCalls
        async for chunk in stream_function(messages, info):
            yield chunk

    return FunctionModel(stream_function=slow, model_name=base.model_name)


def build_durable_planner(
    deps: DurableDeps, *, name: str, live: bool
) -> Agent[DurableDeps, str | DeferredToolRequests]:
    """Stage A's scripted planner, its model and tools reused, with `DBOSDurability`
    attached; `live` gives it the progress handler."""
    base = build_planner(deps)
    assert isinstance(base.model, FunctionModel)
    (toolset,) = base.toolsets
    assert isinstance(toolset, FunctionToolset)
    tools = list(toolset.tools.values())
    durability: DBOSDurability[DurableDeps] = (
        DBOSDurability(event_stream_handler=relay_progress) if live else DBOSDurability()
    )
    return Agent(
        _slowed(deps, base.model),
        deps_type=DurableDeps,
        output_type=str | DeferredToolRequests,
        name=name,
        tools=tools,
        capabilities=[durability],
    )


@dataclass
class Runtime:
    """What a segment workflow runs with in this process."""

    deps: DurableDeps
    live: Agent[DurableDeps, str | DeferredToolRequests]
    buffered: Agent[DurableDeps, str | DeferredToolRequests]


# This process's runtime, under the key "current".
_RUNTIME: dict[str, Runtime] = {}


def install(deps: DurableDeps) -> Runtime:
    """Build this process's planner agents around `deps` for segment workflows."""
    runtime = _RUNTIME["current"] = Runtime(
        deps=deps,
        live=build_durable_planner(deps, name=AGENT_NAME, live=True),
        buffered=build_durable_planner(deps, name=BUFFERED_AGENT_NAME, live=False),
    )
    return runtime


def _deferred_results(conversation: Conversation, answer: str) -> Any:
    requests = conversation.deferred_tool_requests
    assert requests is not None
    (call,) = requests.calls
    return requests.build_results(calls={call.tool_call_id: answer})


@DBOS.workflow(name=SEGMENT_WORKFLOW)
async def planner_segment(start: SegmentStart) -> SegmentEnd:
    """Run the planner to its next pause or its end."""
    runtime = _RUNTIME.get("current")
    assert runtime is not None, "install() was not called in this process"
    deps = runtime.deps
    if start["conversation"] is None:
        conversation = Conversation(conversation_id=PLANNER_CONVERSATION_ID)
        results = None
    else:
        conversation = ConversationTypeAdapter.validate_json(start["conversation"])
        assert start["answer"] is not None
        results = _deferred_results(conversation, start["answer"])
    progress: list[dict[str, int]] = []
    if start["buffered"]:
        result = None
        async with runtime.buffered.run_stream_events(
            start["prompt"], deps=deps, conversation=conversation, deferred_tool_results=results
        ) as events:
            async for event in events:
                if isinstance(event, PlanProgress):
                    progress.append(progress_payload(event))
                elif isinstance(event, AgentRunResultEvent):
                    result = event.result
        assert result is not None
    else:
        result = await runtime.live.run(
            start["prompt"], deps=deps, conversation=conversation, deferred_tool_results=results
        )
    output = result.output
    return {
        "conversation": ConversationTypeAdapter.dump_json(result.conversation).decode(),
        "asked": isinstance(output, DeferredToolRequests),
        "text": output if isinstance(output, str) else None,
        "progress": progress,
    }


def dbos_config(db: Path, executor_id: str) -> dict[str, Any]:
    return {
        "name": "pai_node_stream",
        "system_database_url": f"sqlite:///{db}",
        "application_version": "planner-1",
        "executor_id": executor_id,
        "log_level": "ERROR",
    }


def launch(db: Path, executor_id: str) -> None:
    """Start DBOS over the SQLite system database `db` as `executor_id`. Launching
    recovers this executor's PENDING workflows of this application version."""
    DBOS(config=dbos_config(db, executor_id))  # type: ignore[arg-type]
    DBOS.launch()
