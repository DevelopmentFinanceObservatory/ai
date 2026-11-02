"""Stage A's three-node workflow with a `planner` node that runs each planner
segment as a DBOS workflow.

The pause still lives in ADK: a segment ending in an ask stores the conversation in
session state and the node yields a `RequestInput`. No DBOS workflow stays open
across a pause.

`DMode.DURABLE` is the design under test. The other modes are controls:

- `RANDOM_ID`: a fresh workflow id on every entry, so a re-entered node starts a
  new workflow instead of finding the half-finished one.
- `NO_HANDLER`: the agent has no live handler; the workflow buffers progress into
  its output and the node relays it once the segment returns.
- `LOCAL_CANCEL`: on stop, the node cancels its local task and leaves the DBOS
  workflow PENDING.
- `QUEUE_ONLY`: waits on the progress queue alone, never on the segment's task, so
  a segment that fails leaves the node waiting.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from enum import StrEnum
from typing import Any

from dbos import DBOS, SetWorkflowID
from durable_planner import DurableDeps, SegmentEnd, SegmentStart, attach, detach, install
from durable_planner import planner_segment as segment_workflow
from google.adk import Context, Event
from google.adk.apps import App
from google.adk.events import EventActions, RequestInput
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService
from google.adk.workflow import START, Edge, Workflow, node
from graph import (
    APP_NAME,
    BRIEF_KEY,
    CONVERSATION_KEY,
    approve,
    ask_interrupt_id,
    orchestrate,
)
from planner_agent import plan_of
from pydantic_ai import ConversationTypeAdapter


class DMode(StrEnum):
    DURABLE = "durable"
    RANDOM_ID = "random_id"
    NO_HANDLER = "no_handler"
    LOCAL_CANCEL = "local_cancel"
    QUEUE_ONLY = "queue_only"


def workflow_id(invocation_id: str, segment: int) -> str:
    """The segment's DBOS workflow id: the ADK invocation that runs it and the
    segment's position in it, so a node re-entered after a crash asks for the same
    workflow."""
    return f"{invocation_id}:{segment}"


def progress_event(item: dict[str, int]) -> Event:
    return Event(custom_metadata={"progress": item})


def _entry(ctx: Context, *, buffered: bool) -> tuple[int, SegmentStart]:
    """The segment this entry runs: the one after the stored pause when this
    invocation stored it and the user answered, else segment 0 from the brief."""
    stored = ctx.state.get(CONVERSATION_KEY)
    if stored and stored["invocation_id"] == ctx.invocation_id:
        conversation = ConversationTypeAdapter.validate_json(stored["conversation"])
        requests = conversation.deferred_tool_requests
        assert requests is not None
        (call,) = requests.calls
        answer = ctx.resume_inputs.get(ask_interrupt_id(call.tool_call_id))
        if answer is not None:
            start: SegmentStart = {
                "prompt": None,
                "conversation": stored["conversation"],
                "answer": answer["answer"],
                "buffered": buffered,
            }
            return stored["segment"] + 1, start
    brief = ctx.state[BRIEF_KEY]
    return 0, {"prompt": brief, "conversation": None, "answer": None, "buffered": buffered}


async def open_segment(
    workflow: str, start: SegmentStart, deps: DurableDeps
) -> asyncio.Task[SegmentEnd]:
    """A task that ends with the segment's output.

    A new id runs the workflow inline in the task, on this loop. A finished one
    returns its recorded output. A PENDING one that another executor owns is taken
    over with `resume_workflow`, which re-enqueues it so that one runner claims it
    and continues from its last recorded step; whichever process's queue runner
    claims it executes it on DBOS's own event loop. A PENDING or ENQUEUED one this
    executor already holds, as after recovery at launch, is only awaited.
    """
    status = await DBOS.get_workflow_status_async(workflow)
    if status is None:
        deps.record(f"segment {workflow} started")

        async def run() -> SegmentEnd:
            with SetWorkflowID(workflow):
                return await segment_workflow(start)

        return asyncio.create_task(run())
    if status.status == "PENDING" and status.executor_id != DBOS.executor_id:
        deps.record(f"segment {workflow} resumed from {status.executor_id}")
        await DBOS.resume_workflow_async(workflow)
    else:
        deps.record(f"segment {workflow} awaited in status {status.status}")
    handle = await DBOS.retrieve_workflow_async(workflow)
    return asyncio.create_task(handle.get_result(polling_interval_sec=0.1))


async def _relay(
    workflow: str, start: SegmentStart, deps: DurableDeps, mode: DMode
) -> AsyncIterator[dict[str, int] | SegmentEnd]:
    """Each progress item as it arrives, then the segment's output. Leaving early
    cancels the segment's task and, except under `LOCAL_CANCEL`, the workflow."""
    queue = attach(workflow)
    task = await open_segment(workflow, start, deps)
    getter: asyncio.Future[dict[str, int]] | None = None
    try:
        if mode is DMode.QUEUE_ONLY:
            task.add_done_callback(lambda t: t.cancelled() or t.exception() or None)
            while True:
                yield await queue.get()
        while True:
            getter = asyncio.ensure_future(queue.get())
            done, _ = await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
            if getter not in done:
                getter.cancel()
                break
            yield getter.result()
        while not queue.empty():
            yield queue.get_nowait()
        yield task.result()
    finally:
        detach(workflow)
        if getter is not None and not getter.done():
            getter.cancel()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if mode is not DMode.LOCAL_CANCEL:
                await DBOS.cancel_workflow_async(workflow)
                deps.record(f"segment {workflow} cancelled")


def planner_node(
    *, deps: DurableDeps, mode: DMode = DMode.DURABLE
) -> Callable[[Context, Any], AsyncIterator[Event | RequestInput]]:
    """The `planner` node running segment workflows with `deps` under `mode`."""
    buffered = mode is DMode.NO_HANDLER

    async def planner(ctx: Context, node_input: Any) -> AsyncIterator[Event | RequestInput]:
        segment, start = _entry(ctx, buffered=buffered)
        if mode is DMode.RANDOM_ID:
            workflow = workflow_id(uuid.uuid4().hex, segment)
        else:
            workflow = workflow_id(ctx.invocation_id, segment)
        end: SegmentEnd | None = None
        async for item in _relay(workflow, start, deps, mode):
            if "conversation" in item:
                end = item  # type: ignore[assignment]
            else:
                yield progress_event(item)  # type: ignore[arg-type]
        assert end is not None
        for item in end["progress"]:
            yield progress_event(item)
        conversation = ConversationTypeAdapter.validate_json(end["conversation"])
        if end["asked"]:
            ctx.state[CONVERSATION_KEY] = {
                "invocation_id": ctx.invocation_id,
                "segment": segment,
                "conversation": end["conversation"],
            }
            requests = conversation.deferred_tool_requests
            assert requests is not None
            (call,) = requests.calls
            yield RequestInput(
                interrupt_id=ask_interrupt_id(call.tool_call_id),
                message=call.args_as_dict()["message"],
                payload=call.args_as_dict(),
            )
            return
        ctx.state[CONVERSATION_KEY] = None
        yield Event(
            output={
                "plan": plan_of(conversation.messages),
                "text": end["text"],
                "history": end["conversation"],
            },
            actions=EventActions(route="ok"),
        )

    return planner


def build_runner(
    session_service: BaseSessionService, *, deps: DurableDeps, mode: DMode = DMode.DURABLE
) -> Runner:
    """Stage A's workflow with the durable `planner` node. Builds this process's
    planner agents around `deps`; DBOS must be launched before the runner runs."""
    install(deps)
    orchestrate_n = node(orchestrate, name="orchestrate")
    planner_n = node(planner_node(deps=deps, mode=mode), name="planner", rerun_on_resume=True)
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
