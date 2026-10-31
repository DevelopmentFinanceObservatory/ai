"""A three-node ADK workflow, `orchestrate` -> `planner` -> `approve`, whose
`planner` node runs the Pydantic AI planner and streams its progress live.

`Mode.LIVE` is the design under test. The other modes are the controls, each built
to fail one pass line:

- `AFTER_AWAIT`: dev's `planner_node`. Reports queue while the planner runs and are
  relayed once it returns; a pause drops them.
- `RESTART`: ignores the stored `Conversation` and runs the planner again from the
  brief on every entry, answering the ask from `ctx.resume_inputs` inline.
- `ORPHAN`: runs the planner in a task of its own that nothing cancels when the
  node is cancelled.
- `QUEUE_ONLY`: waits on the progress queue alone, never on the planner task, so a
  planner that raises leaves the node waiting.
- `PROCESS_MEMORY`: keeps the paused `Conversation` in this process's memory
  instead of the session state, so a resume in another process does not find it.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from google.adk import Context, Event
from google.adk.apps import App
from google.adk.events import EventActions, RequestInput
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService
from google.adk.workflow import START, Edge, Workflow, node
from planner_agent import Deps, PlanProgress, plan_of
from pydantic_ai import Agent, Conversation, ConversationTypeAdapter, DeferredToolRequests
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.tools import DeferredToolResults

APP_NAME = "pai_node_stream"
BRIEF = "ODA to Ghana"
BRIEF_KEY = "brief"
# The paused planner: `{"invocation_id": ..., "conversation": <Conversation JSON>}`.
# Written in the same event as the ask's interrupt, cleared when the planner ends.
CONVERSATION_KEY = "planner:conversation"
APPROVE_INTERRUPT = "approve"
PLANNER_CONVERSATION_ID = "planner"


class Mode(StrEnum):
    LIVE = "live"
    AFTER_AWAIT = "after_await"
    RESTART = "restart"
    ORPHAN = "orphan"
    QUEUE_ONLY = "queue_only"
    PROCESS_MEMORY = "process_memory"


# Where `Mode.PROCESS_MEMORY` keeps the paused conversation, by invocation id.
_PROCESS_MEMORY: dict[str, dict[str, str]] = {}


def _stored(ctx: Context, mode: Mode) -> dict[str, str] | None:
    if mode is Mode.PROCESS_MEMORY:
        return _PROCESS_MEMORY.get(ctx.invocation_id)
    return ctx.state.get(CONVERSATION_KEY)


def _store(ctx: Context, mode: Mode, value: dict[str, str] | None) -> None:
    if mode is Mode.PROCESS_MEMORY:
        _PROCESS_MEMORY[ctx.invocation_id] = value or {}
    else:
        ctx.state[CONVERSATION_KEY] = value


def ask_interrupt_id(tool_call_id: str) -> str:
    """The ADK interrupt id of the planner's ask with this tool call id."""
    return f"planner_ask:{tool_call_id}"


def progress_payload(event: PlanProgress) -> dict[str, int]:
    return {"call": event.call, "done": event.done, "total": event.total}


def progress_event(event: PlanProgress) -> Event:
    return Event(custom_metadata={"progress": progress_payload(event)})


def orchestrate(ctx: Context, node_input: Any) -> dict[str, Any]:
    """The stub orchestrator: writes the brief for the planner."""
    ctx.state[BRIEF_KEY] = BRIEF
    return {}


async def approve(ctx: Context, node_input: dict[str, Any]) -> AsyncIterator[Event | RequestInput]:
    """Offer the plan for approval, and end the turn with it once answered."""
    answer = ctx.resume_inputs.get(APPROVE_INTERRUPT)
    if answer is None:
        yield (
            RequestInput(
                interrupt_id=APPROVE_INTERRUPT,
                message="Approve this plan?",
                payload={"plan": node_input["plan"]},
            )
        )
        return
    yield Event(output={"approved": node_input["plan"]})


@dataclass
class _Entry:
    """Where a planner node entry starts: a prompt, or a stored conversation."""

    prompt: str | None
    conversation: Conversation
    results: DeferredToolResults | None


def _answer_for(ctx: Context, requests: DeferredToolRequests) -> DeferredToolResults | None:
    """The deferred results the resume inputs give, or `None` when the ask is
    not answered."""
    (call,) = requests.calls
    answer = ctx.resume_inputs.get(ask_interrupt_id(call.tool_call_id))
    if answer is None:
        return None
    return requests.build_results(calls={call.tool_call_id: answer["answer"]})


def _entry(ctx: Context, *, mode: Mode) -> _Entry:
    """Continue the stored, paused planner when this invocation stored it and the
    user answered its ask; otherwise start from the brief."""
    stored = _stored(ctx, mode)
    if mode is not Mode.RESTART and stored and stored["invocation_id"] == ctx.invocation_id:
        conversation = ConversationTypeAdapter.validate_json(stored["conversation"])
        requests = conversation.deferred_tool_requests
        assert requests is not None
        results = _answer_for(ctx, requests)
        if results is not None:
            return _Entry(None, conversation, results)
    return _Entry(
        ctx.state[BRIEF_KEY],
        Conversation(conversation_id=PLANNER_CONVERSATION_ID),
        None,
    )


type Relay = Callable[[Agent[Deps, Any], Deps, _Entry], AsyncIterator[Event | AgentRunResult]]


async def _relay_live(
    agent: Agent[Deps, Any], deps: Deps, entry: _Entry
) -> AsyncIterator[Event | AgentRunResult]:
    """Each progress report as its own event the moment it arrives, then the result.

    `run_stream_events` runs the agent in a task of its own and hands events over a
    zero-buffer stream, and leaving the `async with` cancels and drains that task.
    So a cancellation thrown into this generator stops the planner too.
    """
    async with agent.run_stream_events(
        entry.prompt,
        deps=deps,
        conversation=entry.conversation,
        deferred_tool_results=entry.results,
    ) as events:
        async for event in events:
            if isinstance(event, PlanProgress):
                yield progress_event(event)
            elif isinstance(event, AgentRunResultEvent):
                yield event.result


async def _relay_after_await(
    agent: Agent[Deps, Any], deps: Deps, entry: _Entry
) -> AsyncIterator[Event | AgentRunResult]:
    """Dev's behaviour: queue the reports while the planner runs, relay them once
    it returns, drop them when it pauses."""
    queued: list[PlanProgress] = []
    result: AgentRunResult | None = None
    async with agent.run_stream_events(
        entry.prompt,
        deps=deps,
        conversation=entry.conversation,
        deferred_tool_results=entry.results,
    ) as events:
        async for event in events:
            if isinstance(event, PlanProgress):
                queued.append(event)
            elif isinstance(event, AgentRunResultEvent):
                result = event.result
    assert result is not None
    if not isinstance(result.output, DeferredToolRequests):
        for event in queued:
            yield progress_event(event)
    yield result


async def _relay_orphan(
    agent: Agent[Deps, Any], deps: Deps, entry: _Entry
) -> AsyncIterator[Event | AgentRunResult]:
    """Live relay from a task this node starts and never cancels."""
    queue: asyncio.Queue[PlanProgress] = asyncio.Queue()

    async def run() -> AgentRunResult:
        async with agent.run_stream_events(
            entry.prompt,
            deps=deps,
            conversation=entry.conversation,
            deferred_tool_results=entry.results,
        ) as events:
            async for event in events:
                if isinstance(event, PlanProgress):
                    queue.put_nowait(event)
                elif isinstance(event, AgentRunResultEvent):
                    return event.result
        raise RuntimeError("planner ended without a result")

    task = asyncio.create_task(run())
    while not task.done():
        getter = asyncio.ensure_future(queue.get())
        await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
        if not getter.done():
            getter.cancel()
            break
        yield progress_event(getter.result())
    while not queue.empty():
        yield progress_event(queue.get_nowait())
    yield task.result()


async def _relay_queue_only(
    agent: Agent[Deps, Any], deps: Deps, entry: _Entry
) -> AsyncIterator[Event | AgentRunResult]:
    """Live relay that waits on the queue alone: the task's result arrives on the
    queue, so a task that raises puts nothing there."""
    queue: asyncio.Queue[PlanProgress | AgentRunResult] = asyncio.Queue()

    async def run() -> None:
        async with agent.run_stream_events(
            entry.prompt,
            deps=deps,
            conversation=entry.conversation,
            deferred_tool_results=entry.results,
        ) as events:
            async for event in events:
                if isinstance(event, PlanProgress):
                    queue.put_nowait(event)
                elif isinstance(event, AgentRunResultEvent):
                    queue.put_nowait(event.result)

    task = asyncio.create_task(run())
    try:
        while True:
            item = await queue.get()
            if isinstance(item, PlanProgress):
                yield progress_event(item)
            else:
                yield item
                return
    finally:
        task.cancel()


_RELAYS: dict[Mode, Relay] = {
    Mode.LIVE: _relay_live,
    Mode.AFTER_AWAIT: _relay_after_await,
    Mode.RESTART: _relay_live,
    Mode.ORPHAN: _relay_orphan,
    Mode.QUEUE_ONLY: _relay_queue_only,
    Mode.PROCESS_MEMORY: _relay_live,
}


def planner_node(
    *, agent: Agent[Deps, Any], deps: Deps, mode: Mode = Mode.LIVE
) -> Callable[[Context, Any], AsyncIterator[Event | RequestInput]]:
    """The `planner` node, running `agent` with `deps` under `mode`."""
    relay = _RELAYS[mode]

    async def planner(ctx: Context, node_input: Any) -> AsyncIterator[Event | RequestInput]:
        entry = _entry(ctx, mode=mode)
        while True:
            result: AgentRunResult | None = None
            async for item in relay(agent, deps, entry):
                if isinstance(item, AgentRunResult):
                    result = item
                else:
                    yield item
            assert result is not None
            output = result.output
            if not isinstance(output, DeferredToolRequests):
                break
            # The restart control reaches an ask the user already answered.
            answered = _answer_for(ctx, output)
            if answered is None:
                _store(
                    ctx,
                    mode,
                    {
                        "invocation_id": ctx.invocation_id,
                        "conversation": ConversationTypeAdapter.dump_json(
                            result.conversation
                        ).decode(),
                    },
                )
                (call,) = output.calls
                yield (
                    RequestInput(
                        interrupt_id=ask_interrupt_id(call.tool_call_id),
                        message=call.args_as_dict()["message"],
                        payload=call.args_as_dict(),
                    )
                )
                return
            entry = _Entry(None, result.conversation, answered)
        _store(ctx, mode, None)
        yield Event(
            output={
                "plan": plan_of(result.conversation.messages),
                "text": output,
                "history": ConversationTypeAdapter.dump_json(result.conversation).decode(),
            },
            actions=EventActions(route="ok"),
        )

    return planner


def build_runner(
    session_service: BaseSessionService,
    *,
    agent: Agent[Deps, Any],
    deps: Deps,
    mode: Mode = Mode.LIVE,
) -> Runner:
    orchestrate_n = node(orchestrate, name="orchestrate")
    planner_n = node(
        planner_node(agent=agent, deps=deps, mode=mode), name="planner", rerun_on_resume=True
    )
    approve_n = node(approve, name="approve", rerun_on_resume=True)
    workflow = Workflow(
        name=APP_NAME,
        edges=[
            (START, orchestrate_n),
            (orchestrate_n, planner_n),
            Edge(from_node=planner_n, to_node=approve_n, route="ok"),
        ],
    )
    return Runner(app=App(name=APP_NAME, root_agent=workflow), session_service=session_service)
