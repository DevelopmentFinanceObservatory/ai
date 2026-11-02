"""Dev's `execute` node with the approved plan's run moved into a DBOS workflow.

Dev's node (`harness.graph.execute.execute_node`) runs
`executor.run.execute_approved_plan` as a task beside itself, relays its
progress, and cancels the task when the node exits (`_stop`). Two names in that
module are swapped, and the node body is otherwise dev's:

- `execute_approved_plan` becomes `run_in_background`. It starts, finds or
  resumes the DBOS workflow `execute_plan` under the id
  `execute:{turn session id}:{plan revision}`, relays the workflow's progress to
  the node's `on_progress` while the node listens, and returns what dev's call
  returns. The workflow calls dev's own `execute_approved_plan` against the
  same ledger, publisher and tool surface.
- `_stop` becomes `stop`. On Studio's Stop (the conversation's
  `cancel_reason` is `"cancelled"`) it cancels the DBOS workflow and the task
  running it. On any other exit (a disconnect, a lost segment) it stops only
  the node's wait, and the workflow runs on.

Each tool call of the run is one DBOS step: `DurableSurface` wraps the bare
`ToolSurface`, under Studio's `_RecordingSurface`, and inside a workflow runs
`call_tool` as the step `execute_tool_call`. A retried call is one step per
attempt. Everything else in the run (ledger writes, statistics fetches, label
lookups, publication) is workflow code and runs again on replay.

`background_execute_node` replaces `harness.graph.workflow.execute_node`: it
registers what the workflow runs with in this process (`Runtime`), so a
workflow resumed here after another process died finds it, and sets the turn's
session id for the workflow id.

Progress crosses from the workflow to the node through `listen` / `publish`,
keyed by workflow id and safe across threads, because a workflow DBOS resumes
runs on DBOS's own event loop thread. A report nobody listens for is dropped.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from compiler.surface import ProgressCallback, ProgressReport, ToolCallResult, ToolSurface
from dbos import DBOS, SetWorkflowID
from executor import run as executor_run
from executor.ledger import LedgerUnavailableError, RunLedger
from executor.outcomes import outcome_from_json
from google.adk import Context
from harness.conversation import current_conversation_id, current_conversation_key
from harness.graph import execute as execute_module
from harness.graph import workflow as workflow_module
from harness.studio_state import StudioStore
from planner.plan import Plan, plan_revision
from planner.workflow import ToolId

EXECUTE_WORKFLOW = "execute_plan"
TOOL_STEP = "execute_tool_call"
APPLICATION_VERSION = "execute-1"

# Dev's own functions, kept before anything is swapped.
DEV_EXECUTE_APPROVED_PLAN = executor_run.execute_approved_plan
DEV_EXECUTE_NODE = execute_module.execute_node
DEV_STOP = execute_module._stop


class BMode(StrEnum):
    """`DURABLE` is the design. The others are controls."""

    DURABLE = "durable"
    # Progress is collected into the workflow's result and relayed only once
    # the run has ended.
    BUFFERED = "buffered"
    # Stop cancels the local task running the workflow, and leaves the DBOS
    # workflow as it stands.
    LOCAL_CANCEL = "local_cancel"


# ---------------------------------------------------------------------------
# Progress between the workflow and the node
# ---------------------------------------------------------------------------

_LISTENERS: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]]]] = {}
_LISTENERS_LOCK = threading.Lock()


def listen(workflow_id: str) -> asyncio.Queue[dict[str, Any]]:
    """The queue this loop reads `workflow_id`'s progress from, from now on."""
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    with _LISTENERS_LOCK:
        _LISTENERS[workflow_id] = (asyncio.get_running_loop(), queue)
    return queue


def unlisten(workflow_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
    with _LISTENERS_LOCK:
        if _LISTENERS.get(workflow_id, (None, None))[1] is queue:
            del _LISTENERS[workflow_id]


def publish(workflow_id: str, item: dict[str, Any]) -> None:
    """Hand one progress report to whoever listens for `workflow_id`, from any thread."""
    with _LISTENERS_LOCK:
        listener = _LISTENERS.get(workflow_id)
    if listener is not None:
        loop, queue = listener
        loop.call_soon_threadsafe(queue.put_nowait, item)


# ---------------------------------------------------------------------------
# One DBOS step per tool call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Call:
    surface: ToolSurface
    on_progress: ProgressCallback | None


_CALL: ContextVar[_Call | None] = ContextVar("durable_execute_call", default=None)


@DBOS.step(name=TOOL_STEP)
async def execute_tool_call(tool: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """One tool call of a run. Its result is recorded, so a replay returns it."""
    call = _CALL.get()
    assert call is not None, "execute_tool_call runs only through DurableSurface"
    result = await call.surface.call_tool(
        tool=ToolId.model_validate(tool), arguments=arguments, on_progress=call.on_progress
    )
    return result.model_dump(mode="json")


class DurableSurface:
    """A `ToolSurface` whose calls inside a DBOS workflow are DBOS steps."""

    def __init__(self, surface: ToolSurface) -> None:
        self._surface = surface

    async def list_tools(self) -> Any:
        return await self._surface.list_tools()

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        if DBOS.workflow_id is None:
            return await self._surface.call_tool(
                tool=tool, arguments=arguments, on_progress=on_progress
            )
        token = _CALL.set(_Call(self._surface, on_progress))
        try:
            raw = await execute_tool_call(tool.model_dump(mode="json"), dict(arguments))
        finally:
            _CALL.reset(token)
        return ToolCallResult.model_validate(raw)


# ---------------------------------------------------------------------------
# The workflow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Runtime:
    """What a run's workflow runs with in this process: the arguments dev's
    `execute_node` was built with, and how the node stops."""

    tool_surface: ToolSurface
    run_ledger: RunLedger
    publisher: Any
    max_concurrent_steps: int
    fetch_series_stats: Any
    label_refs: Any
    store: StudioStore | None
    mode: BMode


_RUNTIME: dict[str, Runtime] = {}


@DBOS.workflow(name=EXECUTE_WORKFLOW)
async def execute_plan(start: dict[str, Any]) -> dict[str, Any]:
    """Run an approved plan as dev's `execute_approved_plan` does.

    The conversation the ledger and the recorder key on is set from `start`,
    because a workflow DBOS resumes runs on its own thread with none set.
    """
    runtime = _RUNTIME.get("current")
    assert runtime is not None, "background_execute_node was not built in this process"
    current_conversation_id.set(start["scope"])
    current_conversation_key.set(tuple(start["key"]) if start["key"] else None)
    workflow_id = DBOS.workflow_id
    assert workflow_id is not None
    progress: list[dict[str, Any]] = []

    def on_progress(step_id: str, report: ProgressReport) -> None:
        item = {
            "step_id": step_id,
            "progress": report.progress,
            "total": report.total,
            "message": report.message,
        }
        if start["buffered"]:
            progress.append(item)
        else:
            publish(workflow_id, item)

    try:
        finished, outcome = await DEV_EXECUTE_APPROVED_PLAN(
            Plan.model_validate_json(start["plan"]),
            surface=runtime.tool_surface,
            ledger=runtime.run_ledger,
            publisher=runtime.publisher,
            approver=start["approver"],
            conversation_id=start["conversation_id"],
            max_concurrent_steps=runtime.max_concurrent_steps,
            fetch_series_stats=runtime.fetch_series_stats,
            label_refs=runtime.label_refs,
            on_progress=on_progress,
        )
    except LedgerUnavailableError as exc:
        return {"ledger_unavailable": True, "run_id": exc.run_id, "progress": progress}
    return {
        "plan": finished.model_dump_json(),
        "outcome": outcome.model_dump(mode="json"),
        "progress": progress,
    }


def workflow_id(turn: str, revision: str) -> str:
    """The run's DBOS workflow id: the turn's ADK session and the plan's
    revision, so the node re-entered in the same turn finds the same run."""
    return f"execute:{turn}:{revision}"


# Workflows this process runs inline, by id, so a re-entered node and Stop
# reach the task.
_LOCAL: dict[str, asyncio.Task[dict[str, Any]]] = {}
# The node's waiting task, to the workflow it waits on.
_WAITERS: dict[asyncio.Task[Any], str] = {}
# The turn's ADK session id, set by the node for the workflow id.
_TURN: ContextVar[str | None] = ContextVar("durable_execute_turn", default=None)
# What happened to each workflow id the node opened in this process, in order.
OPENED: list[tuple[str, str]] = []


async def open_run(workflow: str, start: dict[str, Any]) -> asyncio.Task[dict[str, Any]]:
    """A task that ends with the workflow's result.

    A new id runs the workflow inline in a task of its own, which outlives the
    node. One this process already runs is that task. One `PENDING` under
    another executor is taken over with `resume_workflow`; DBOS runs it on its
    own loop, and the returned task polls for its result. Any other is only
    polled: a finished one returns its recorded result.
    """
    if (local := _LOCAL.get(workflow)) is not None:
        OPENED.append((workflow, "local"))
        return local
    status = await DBOS.get_workflow_status_async(workflow)
    if status is None:
        OPENED.append((workflow, "started"))

        async def run() -> dict[str, Any]:
            with SetWorkflowID(workflow):
                return await execute_plan(start)

        task = asyncio.create_task(run())
        _LOCAL[workflow] = task
        task.add_done_callback(lambda _: _LOCAL.pop(workflow, None))
        return task
    if status.status == "PENDING" and status.executor_id != DBOS.executor_id:
        OPENED.append((workflow, f"resumed from {status.executor_id}"))
        await DBOS.resume_workflow_async(workflow)
    else:
        OPENED.append((workflow, f"polled in status {status.status}"))
    handle = await DBOS.retrieve_workflow_async(workflow)
    return asyncio.create_task(handle.get_result(polling_interval_sec=0.05))


def _report(item: dict[str, Any]) -> ProgressReport:
    return ProgressReport(progress=item["progress"], total=item["total"], message=item["message"])


async def run_in_background(
    plan: Plan,
    *,
    surface: ToolSurface,
    ledger: RunLedger | None = None,
    publisher: Any = None,
    approver: str = "unattributed",
    conversation_id: str = "",
    on_progress: Callable[[str, ProgressReport], None] | None = None,
    **_: Any,
) -> tuple[Plan, Any]:
    """`execute_approved_plan` as dev's node calls it, run as the DBOS workflow.

    `surface`, `ledger` and `publisher` are the node's own, which `Runtime`
    holds; the workflow reads them from there.
    """
    del surface, ledger, publisher
    runtime = _RUNTIME["current"]
    turn = _TURN.get()
    assert turn is not None, "run_in_background runs only under background_execute_node"
    workflow = workflow_id(turn, plan_revision(plan))
    key = current_conversation_key.get()
    start = {
        "plan": plan.model_dump_json(),
        "approver": approver,
        "conversation_id": conversation_id,
        "scope": current_conversation_id.get(),
        "key": list(key) if key is not None else None,
        "buffered": runtime.mode is BMode.BUFFERED,
    }
    me = asyncio.current_task()
    assert me is not None
    _WAITERS[me] = workflow
    queue = listen(workflow)
    task = await open_run(workflow, start)
    getter: asyncio.Future[dict[str, Any]] | None = None
    try:
        while True:
            getter = asyncio.ensure_future(queue.get())
            done, _ = await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
            if getter not in done:
                getter.cancel()
                break
            if on_progress is not None:
                item = getter.result()
                on_progress(item["step_id"], _report(item))
        while not queue.empty():
            item = queue.get_nowait()
            if on_progress is not None:
                on_progress(item["step_id"], _report(item))
        result = task.result()
    finally:
        unlisten(workflow, queue)
        del _WAITERS[me]
        if getter is not None and not getter.done():
            getter.cancel()
        if task is not _LOCAL.get(workflow) and not task.done():
            # A poll of a workflow another loop runs: only the poll stops.
            task.cancel()
    if on_progress is not None:
        for item in result["progress"]:
            on_progress(item["step_id"], _report(item))
    if result.get("ledger_unavailable"):
        raise LedgerUnavailableError(result["run_id"])
    return Plan.model_validate_json(result["plan"]), outcome_from_json(result["outcome"])


def _stop_requested(store: StudioStore | None) -> bool:
    """Whether the conversation's stream ended on Studio's Stop."""
    key = current_conversation_key.get()
    state = store.current(*key) if store is not None and key is not None else None
    return state is not None and state.cancel_reason == "cancelled"


async def stop(task: asyncio.Task[Any]) -> None:
    """Dev's `_stop` for the node's waiting task. On Stop, the workflow it
    waits on is cancelled first, in DBOS and then its local task."""
    workflow = _WAITERS.get(task)
    runtime = _RUNTIME.get("current")
    if (
        workflow is not None
        and runtime is not None
        and not task.done()
        and _stop_requested(runtime.store)
    ):
        if runtime.mode is not BMode.LOCAL_CANCEL:
            await DBOS.cancel_workflow_async(workflow)
        local = _LOCAL.get(workflow)
        if local is not None and not local.done():
            local.cancel()
            await asyncio.wait({local})
    await DEV_STOP(task)


def background_execute_node(**kwargs: Any) -> Callable[[Context, Any], Any]:
    """`harness.graph.workflow.execute_node`'s replacement: dev's node, with
    this process's `Runtime` registered and the turn's session id set."""
    settings = _SETTINGS["current"]
    _RUNTIME["current"] = Runtime(
        tool_surface=kwargs["tool_surface"],
        run_ledger=kwargs["run_ledger"],
        publisher=kwargs["publisher"],
        max_concurrent_steps=kwargs["max_concurrent_steps"],
        fetch_series_stats=kwargs["fetch_series_stats"],
        label_refs=kwargs["label_refs"],
        store=settings.store,
        mode=settings.mode,
    )
    inner = DEV_EXECUTE_NODE(**kwargs)

    async def execute(ctx: Context, node_input: Any) -> Any:
        _TURN.set(ctx.session.id)
        async for item in inner(ctx, node_input):
            yield item

    return execute


@dataclass(frozen=True)
class _Settings:
    store: StudioStore | None
    mode: BMode


_SETTINGS: dict[str, _Settings] = {}


def install(patch: Any, *, store: StudioStore | None, mode: BMode = BMode.DURABLE) -> None:
    """Build every root workflow with the background `execute` node. `patch`
    is a `pytest.MonkeyPatch`, or anything with its `setattr`."""
    _SETTINGS["current"] = _Settings(store=store, mode=mode)
    patch.setattr(workflow_module, "execute_node", background_execute_node)
    patch.setattr(execute_module, "execute_approved_plan", run_in_background)
    patch.setattr(execute_module, "_stop", stop)


def dbos_config(db: Path, executor_id: str) -> dict[str, Any]:
    return {
        "name": "background_execute",
        "system_database_url": f"sqlite:///{db}",
        "application_version": APPLICATION_VERSION,
        "executor_id": executor_id,
        "log_level": "ERROR",
    }


def launch(db: Path, executor_id: str) -> None:
    """Start DBOS over the SQLite system database `db` as `executor_id`."""
    DBOS(config=dbos_config(db, executor_id))  # type: ignore[arg-type]
    DBOS.launch()
