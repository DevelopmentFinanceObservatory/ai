"""The pass-line checks for the DBOS-backed planner node. Each raises when its
property fails and takes the `mode` that builds its control.

In-process checks expect DBOS launched over a scratch system database; the kill
check launches its own processes.
"""

import asyncio
import gc
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from checks import _repeats, history, observe_tool_failure, reference_run
from dbos import DBOS
from durable_graph import DMode, build_runner
from durable_planner import DurableDeps, launch
from durable_procs import cancel_from_another_process, run_driver
from google.adk import Event
from google.adk.runners import Runner
from graph import Mode
from pydantic_ai import ConversationTypeAdapter
from turn import (
    USER,
    events,
    interrupt_of,
    last_interrupt,
    new_session,
    progress_list,
    progress_of,
    segment,
    session_service,
    user_text,
)

SESSION = "s1"
STAGE_A = "stage_a"


def _runner(db: Path, deps: DurableDeps, mode: DMode) -> Runner:
    return build_runner(session_service(db), deps=deps, mode=mode)


def _workflows(deps: DurableDeps) -> list[str]:
    """The workflow ids the node started, in order."""
    return [line.split()[1] for line in deps.log if line.endswith(" started")]


async def _status(workflow: str) -> str | None:
    status = await DBOS.get_workflow_status_async(workflow)
    return None if status is None else status.status


async def relaunch(dbos_db: Path, executor_id: str = "test") -> None:
    """Stop DBOS and launch it again over the same system database, as a restarted
    process would."""
    DBOS.destroy()
    launch(dbos_db, executor_id)


# D2: live progress and a lossless pause ----------------------------------------------


async def check_live_progress(db: Path, *, mode: DMode = DMode.DURABLE) -> float:
    """The first progress event reaches the `run_async` consumer while
    `propose_plan` waits for the client's signal. Returns the seconds it took."""
    deps = DurableDeps(gate=asyncio.Event(), gate_call=1, gate_timeout=10)
    runner = _runner(db, deps, mode)
    await new_session(runner, SESSION)
    started = time.monotonic()

    async def until_first_progress() -> None:
        async for event in events(runner, SESSION, user_text("go")):
            if progress_of(event) is not None:
                assert "propose_plan call=1 end" not in deps.log
                return

    try:
        await asyncio.wait_for(until_first_progress(), timeout=2)
        return time.monotonic() - started
    finally:
        assert deps.gate is not None
        deps.gate.set()


async def check_pause_loses_nothing(db: Path, *, mode: DMode = DMode.DURABLE) -> int:
    """All progress events arrive in order before the ask's interrupt, read by a
    client slower than the planner. Returns the count."""
    progress = 40
    deps = DurableDeps(progress=progress)
    runner = _runner(db, deps, mode)
    await new_session(runner, SESSION)
    seen = await segment(runner, SESSION, user_text("go"), on_event=lambda _: asyncio.sleep(0.005))
    got = progress_list(seen)
    assert got == [(1, done) for done in range(1, progress + 1)], f"got progress {got}"
    assert interrupt_of(seen[-1]) is not None, "the interrupt is not the last event"
    return len(got)


# D1 and D3: a kill repeats no model call; re-entry finds the workflow -------------


@dataclass
class KillResult:
    killed_returncode: int
    repeated_model_calls: list[str]
    repeated_tool_lines: list[str]
    repeated_handler_runs: list[str]
    repeated_progress: list[tuple[int, int]]
    segments: list[list[str]]
    model_calls: list[tuple[int, str, str]]
    progress_by_process: list[list[tuple[int, int]]]
    stored_progress_events: list[tuple[int, int]]
    stored_ask_answers: int
    plan: dict[str, Any]


def _lines(log: Path) -> list[dict[str, Any]]:
    return [json.loads(raw) for raw in log.read_text().splitlines()]


async def check_kill(
    tmp_path: Path,
    *,
    mode: str = DMode.DURABLE,
    executors: tuple[str, str, str] = ("p1", "p2", "p3"),
) -> KillResult:
    """Pause in process 1, answer in process 2 and SIGKILL it inside `propose_plan`
    after its first progress event, answer again in process 3 under another DBOS
    executor id: no model call runs twice, process 3 continues process 2's
    workflow, and the plan and history equal a run with no pause."""
    db, dbos_db, log = tmp_path / "s.db", tmp_path / "dbos.sqlite", tmp_path / "log"
    common = ("--log-file", str(log), "--mode", mode)
    first = run_driver(db, dbos_db, executors[0], *common, "start")
    assert first.returncode == 0, first.stderr
    assert first.lines[-2] == {"type": "interrupt", "id": "planner_ask:ask-1"}
    after_first = len(_lines(log))

    killed = run_driver(
        db,
        dbos_db,
        executors[1],
        *common,
        "--hold-call",
        "2",
        "answer",
        kill_when=lambda line: line["type"] == "progress",
    )
    assert killed.returncode == -9, "the process was not killed while alive"
    after_killed = len(_lines(log))

    third = run_driver(db, dbos_db, executors[2], *common, "answer", "approve")
    assert third.returncode == 0, third.stderr
    outputs = [line["output"] for line in third.lines if line["type"] == "output"]
    drafted, approved = outputs[0], outputs[-1]

    session = await session_service(db).get_session(
        app_name="pai_node_stream", user_id=USER, session_id=SESSION
    )
    assert session is not None
    entries = _lines(log)
    lines = [entry["line"] for entry in entries]
    sent = [
        (line["call"], line["done"])
        for run in (first, killed, third)
        for line in run.lines
        if line["type"] == "progress"
    ]
    spans = [entries[:after_first], entries[after_first:after_killed], entries[after_killed:]]
    result = KillResult(
        killed_returncode=killed.returncode,
        repeated_model_calls=_repeats([line for line in lines if line.startswith("model call")]),
        repeated_tool_lines=_repeats(
            [line for line in lines if line.startswith(("propose_plan", "ask_user"))]
        ),
        repeated_handler_runs=_repeats([line for line in lines if line.startswith("progress")]),
        repeated_progress=_repeats(sent),
        segments=[[e["line"] for e in span if e["line"].startswith("segment")] for span in spans],
        model_calls=[
            (process, e["wf"], e["line"])
            for process, span in enumerate(spans, start=1)
            for e in span
            if e["line"].startswith("model call")
        ],
        progress_by_process=[
            [(line["call"], line["done"]) for line in run.lines if line["type"] == "progress"]
            for run in (first, killed, third)
        ],
        stored_progress_events=progress_list(session.events),
        stored_ask_answers=sum(
            1
            for event in session.events
            for response in event.get_function_responses()
            if response.id == "planner_ask:ask-1"
        ),
        plan=drafted["plan"],
    )

    assert not result.repeated_model_calls, f"model called again: {result.repeated_model_calls}"
    ref_plan, ref_conversation, _ = await reference_run()
    assert drafted["plan"] == ref_plan == approved["approved"], "plan differs"
    stored = ConversationTypeAdapter.validate_json(drafted["history"])
    assert history(stored) == history(ref_conversation), "planner history differs"
    (started,) = result.segments[1]
    workflow = started.split()[1]
    assert result.segments[2][0].startswith(f"segment {workflow} "), (
        f"process 3 did not re-enter {workflow}: {result.segments[2]}"
    )
    assert "started" not in result.segments[2][0], f"process 3 restarted: {result.segments[2]}"
    per_workflow = Counter((wf, line) for _, wf, line in result.model_calls)
    assert set(per_workflow.values()) == {1}, f"a workflow ran a model call twice: {per_workflow}"
    return result


# D4: stop -----------------------------------------------------------------------


@dataclass
class StopResult:
    workflow: str
    status_after_stop: str | None
    left_running: list[str] = field(default_factory=list)
    unretrieved: list[str] = field(default_factory=list)


async def _stop_turn(runner: Runner, deps: DurableDeps) -> tuple[list[str], list[str]]:
    """Start a turn, cancel its consumer at the first progress event, and return
    the tasks left running and the exceptions nobody retrieved."""
    loop = asyncio.get_running_loop()
    unretrieved: list[str] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: unretrieved.append(context["message"]))
    baseline = asyncio.all_tasks()
    first_progress = asyncio.Event()

    async def consume() -> None:
        async for event in events(runner, SESSION, user_text("go")):
            if progress_of(event) is not None:
                first_progress.set()

    consumer = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(first_progress.wait(), timeout=5)
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        await asyncio.sleep(0.2)
        left = [
            task.get_name() for task in asyncio.all_tasks() - baseline - {asyncio.current_task()}
        ]
        gc.collect()
        await asyncio.sleep(0)
        return left, unretrieved
    finally:
        for task in asyncio.all_tasks() - baseline - {asyncio.current_task()}:
            task.cancel()
        loop.set_exception_handler(previous_handler)


async def check_stop(db: Path, *, mode: DMode = DMode.DURABLE) -> StopResult:
    """Cancelling the consumer mid-planning cancels the planner and its workflow:
    no task is left, no exception goes unretrieved, the workflow is CANCELLED, and a
    new turn on the session runs."""
    deps = DurableDeps(gate=asyncio.Event(), gate_call=1, gate_timeout=30)
    runner = _runner(db, deps, mode)
    await new_session(runner, SESSION)
    left, unretrieved = await _stop_turn(runner, deps)
    (workflow,) = _workflows(deps)
    result = StopResult(workflow, await _status(workflow), left, unretrieved)
    assert not left, f"tasks left running: {left}"
    assert not unretrieved, f"unretrieved exceptions: {unretrieved}"
    assert "propose_plan call=1 cancelled" in deps.log, "the planner was not cancelled"
    assert result.status_after_stop == "CANCELLED", f"workflow left {result.status_after_stop}"

    deps.gate = None
    seen = await asyncio.wait_for(segment(runner, SESSION, user_text("again")), timeout=5)
    assert not [e.error_code for e in seen if e.error_code]
    assert progress_list(seen) == [(1, 1), (1, 2), (1, 3)]
    assert last_interrupt(seen)["interruptId"] == "planner_ask:ask-1"
    return result


@dataclass
class AbandonedResult:
    status_after_stop: str | None
    status_after_relaunch: str | None
    lines_after_relaunch: list[str]


async def observe_abandoned_workflow(db: Path, dbos_db: Path) -> AbandonedResult:
    """Stop a turn with the `LOCAL_CANCEL` node, then relaunch DBOS as the same
    executor and record what its recovery does with the abandoned workflow."""
    deps = DurableDeps(gate=asyncio.Event(), gate_call=1, gate_timeout=30)
    runner = _runner(db, deps, DMode.LOCAL_CANCEL)
    await new_session(runner, SESSION)
    await _stop_turn(runner, deps)
    (workflow,) = _workflows(deps)
    after_stop = await _status(workflow)
    deps.gate = None
    mark = len(deps.log)
    await relaunch(dbos_db)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and await _status(workflow) in ("PENDING", "ENQUEUED"):
        await asyncio.sleep(0.1)
    return AbandonedResult(after_stop, await _status(workflow), deps.log[mark:])


@dataclass
class InFlightResult:
    model_call_completed: bool
    log: list[str]


async def measure_cancel_during_model_call(db: Path) -> InFlightResult:
    """Cancel the consumer while the first model call sleeps for 1 s, then wait
    past the call's end: did the call complete?"""
    deps = DurableDeps(model_delay=1.0)
    runner = _runner(db, deps, DMode.DURABLE)
    await new_session(runner, SESSION)

    async def consume() -> None:
        async for _ in events(runner, SESSION, user_text("go")):
            pass

    consumer = asyncio.create_task(consume())
    while "slow model start" not in deps.log:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.3)
    consumer.cancel()
    await asyncio.gather(consumer, return_exceptions=True)
    await asyncio.sleep(1.2)
    return InFlightResult("slow model end" in deps.log, list(deps.log))


@dataclass
class ExternalCancelResult:
    where: str
    seconds_to_stop: float
    model_call_completed: bool
    consumer_saw: str
    status: str | None


async def measure_external_cancel(db: Path, dbos_db: Path, *, where: str) -> ExternalCancelResult:
    """Cancel a running segment with `DBOSClient.cancel_workflow` from another
    process and time how long the node takes to stop after the cancel commits.

    `where="model"`: the segment is inside a 6 s model call. `where="tool"`:
    `propose_plan` is emitting 400 progress events 25 ms apart, each a handler step.
    """
    if where == "model":
        deps = DurableDeps(model_delay=6.0)
        ready = "slow model start"
    else:
        deps = DurableDeps(progress=400, delay=0.025)
        ready = "propose_plan call=1 start"
    runner = _runner(db, deps, DMode.DURABLE)
    await new_session(runner, SESSION)
    seen: list[Event] = []
    outcome: list[str] = []

    async def consume() -> None:
        try:
            async for event in events(runner, SESSION, user_text("go")):
                seen.append(event)  # noqa: PERF401
            outcome.append("stream ended")
        except Exception as error:
            outcome.append(f"{type(error).__name__}: {error}")

    consumer = asyncio.create_task(consume())
    while ready not in deps.log:
        await asyncio.sleep(0.01)
    (workflow,) = _workflows(deps)
    committed = await asyncio.to_thread(cancel_from_another_process, dbos_db, workflow)
    await asyncio.wait_for(consumer, timeout=30)
    stopped = time.time()
    codes = [e.error_code for e in seen if e.error_code]
    return ExternalCancelResult(
        where,
        stopped - committed,
        "slow model end" in deps.log,
        f"{outcome[0]}; error events {codes}",
        await _status(workflow),
    )


# D5: a planner-side failure ---------------------------------------------------------


@dataclass
class DurableFailure:
    error_codes: list[str | None]
    raised: str | None
    interrupts: int
    status: str | None
    status_after_relaunch: str | None
    lines_after_relaunch: list[str]


async def check_tool_failure(
    db: Path, dbos_db: Path, *, mode: DMode = DMode.DURABLE
) -> DurableFailure:
    """`propose_plan` raising ends the stream as the node's failure within 5 s, as
    stage A's live node does, and the workflow ends in ERROR and stays there across
    a relaunch."""
    deps = DurableDeps(fail_in_call=1)
    runner = _runner(db, deps, mode)
    await new_session(runner, SESSION)
    seen: list[Event] = []
    raised = None

    async def consume() -> None:
        nonlocal raised
        try:
            async for event in events(runner, SESSION, user_text("go")):
                seen.append(event)  # noqa: PERF401
        except Exception as error:
            raised = f"{type(error).__name__}: {error}"

    await asyncio.wait_for(consume(), timeout=5)
    (workflow,) = _workflows(deps)
    status = await _status(workflow)
    mark = len(deps.log)
    await relaunch(dbos_db)
    await asyncio.sleep(1.5)
    got = DurableFailure(
        error_codes=[e.error_code for e in seen if e.error_code],
        raised=raised,
        interrupts=sum(1 for e in seen if interrupt_of(e) is not None),
        status=status,
        status_after_relaunch=await _status(workflow),
        lines_after_relaunch=deps.log[mark:],
    )
    stage_a = await observe_tool_failure(db.with_name("stage_a.db"), mode=Mode.LIVE)
    assert (got.error_codes, got.raised, got.interrupts) == (
        stage_a.error_codes,
        stage_a.raised,
        stage_a.interrupts,
    ), f"{got} differs from stage A's {stage_a}"
    assert got.status == got.status_after_relaunch == "ERROR", got
    assert not got.lines_after_relaunch, f"retried after relaunch: {got.lines_after_relaunch}"
    return got
