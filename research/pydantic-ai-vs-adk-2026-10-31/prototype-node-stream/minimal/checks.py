"""The pass-line checks. Each raises when its property fails and takes the `mode` that
builds its control, so the tests can show the control fails."""

import asyncio
import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google.adk import Event
from google.adk.runners import Runner
from graph import (
    APPROVE_INTERRUPT,
    BRIEF,
    CONVERSATION_KEY,
    PLANNER_CONVERSATION_ID,
    Mode,
    build_runner,
)
from planner_agent import Deps, build_planner, plan_of
from procs import run_driver
from pydantic_ai import Conversation, ConversationTypeAdapter, ModelMessagesTypeAdapter
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from turn import (
    USER,
    answer,
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

SCOPE = "Ghana only"
SESSION = "s1"


def _runner(db: Path, deps: Deps, mode: Mode) -> Runner:
    return build_runner(session_service(db), agent=build_planner(deps), deps=deps, mode=mode)


def _model_calls(log: list[str]) -> list[str]:
    return [line for line in log if line.startswith("model call")]


# History comparison -----------------------------------------------------------------


def _strip(value: Any, drop: set[str]) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v, drop) for k, v in value.items() if k not in drop}
    if isinstance(value, list):
        return [_strip(v, drop) for v in value]
    return value


def history(conversation: Conversation) -> str:
    """The messages as canonical JSON without timestamps, each `run_id` replaced by
    the order in which it first appears."""
    messages = json.loads(ModelMessagesTypeAdapter.dump_json(conversation.messages))
    messages = _strip(messages, {"timestamp"})
    seen: dict[str, int] = {}
    for message in messages:
        message["run_id"] = seen.setdefault(message["run_id"], len(seen))
    return json.dumps(messages, sort_keys=True)


async def reference_run() -> tuple[dict[str, Any], Conversation, list[str]]:
    """The planner as a bare agent, the ask answered inline: no ADK, no pause, no
    store. Returns the plan, the conversation and the log."""
    deps = Deps()
    agent = build_planner(deps)

    async def run(
        prompt: str | None, conversation: Conversation, results: Any = None
    ) -> AgentRunResult[Any]:
        async with agent.run_stream_events(
            prompt, deps=deps, conversation=conversation, deferred_tool_results=results
        ) as stream:
            async for event in stream:
                if isinstance(event, AgentRunResultEvent):
                    return event.result
        raise RuntimeError("run ended without a result")

    asked = await run(BRIEF, Conversation(conversation_id=PLANNER_CONVERSATION_ID))
    (call,) = asked.output.calls
    results = asked.output.build_results(calls={call.tool_call_id: SCOPE})
    planned = await run(None, asked.conversation, results)
    return plan_of(planned.conversation.messages), planned.conversation, deps.log


# N1 ---------------------------------------------------------------------------------


async def check_live_progress(db: Path, *, mode: Mode = Mode.LIVE) -> float:
    """The first progress event reaches the `run_async` consumer while
    `propose_plan` waits for the client's signal. Returns the seconds it took."""
    deps = Deps(gate=asyncio.Event(), gate_call=1, gate_timeout=10)
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


# N2 ---------------------------------------------------------------------------------


async def check_pause_loses_nothing(db: Path, *, mode: Mode = Mode.LIVE) -> int:
    """All progress events arrive in order before the ask's interrupt, read by a
    client slower than the planner. Returns the count."""
    progress = 40
    deps = Deps(progress=progress)
    runner = _runner(db, deps, mode)
    await new_session(runner, SESSION)
    seen = await segment(runner, SESSION, user_text("go"), on_event=lambda _: asyncio.sleep(0.005))
    got = progress_list(seen)
    assert got == [(1, done) for done in range(1, progress + 1)], f"got progress {got}"
    assert interrupt_of(seen[-1]) is not None, "the interrupt is not the last event"
    return len(got)


# N3 ---------------------------------------------------------------------------------


@dataclass
class ResumeResult:
    model_calls_before: list[str]
    model_calls_after: list[str]
    progress: list[tuple[int, int]] = field(default_factory=list)


async def check_resume_continues(db: Path, *, mode: Mode = Mode.LIVE) -> ResumeResult:
    """Answering the ask continues the stored planner: no model call, tool body or
    progress event repeats, and the plan and history equal a run with no pause."""
    deps = Deps()
    runner = _runner(db, deps, mode)
    await new_session(runner, SESSION)
    paused = await segment(runner, SESSION, user_text("go"))
    ask = last_interrupt(paused)
    before = _model_calls(deps.log)

    # The pause wrote the conversation in the interrupt's own event, so a second
    # service over the same file reads it.
    reread = await session_service(db).get_session(
        app_name=runner.app_name, user_id=USER, session_id=SESSION
    )
    assert reread is not None and reread.state[CONVERSATION_KEY]
    assert (paused[-1].actions.state_delta or {}).get(CONVERSATION_KEY)

    resumed = await segment(runner, SESSION, answer(ask["interruptId"], {"answer": SCOPE}))
    after = _model_calls(deps.log)[len(before) :]
    drafted = next(e.output for e in resumed if e.output and "plan" in e.output)
    assert last_interrupt(resumed)["interruptId"] == APPROVE_INTERRUPT
    done = await segment(runner, SESSION, answer(APPROVE_INTERRUPT, {"approved": True}))

    ref_plan, ref_conversation, ref_log = await reference_run()
    assert after == _model_calls(ref_log)[len(before) :], f"model called again: {after}"
    for body in ("propose_plan call=1 start", "propose_plan call=2 start", "ask_user body"):
        assert deps.log.count(body) == 1, f"{body!r} ran {deps.log.count(body)} times"
    progress = progress_list(paused) + progress_list(resumed)
    assert len(progress) == len(set(progress)), f"progress sent twice: {progress}"
    approved = done[-1].output
    assert isinstance(approved, dict)
    assert drafted["plan"] == ref_plan == approved["approved"]
    stored = ConversationTypeAdapter.validate_json(drafted["history"])
    assert history(stored) == history(ref_conversation), "planner history differs"
    return ResumeResult(before, after, progress)


# N4 ---------------------------------------------------------------------------------


@dataclass
class KillResult:
    killed_returncode: int
    repeated_model_calls: list[str]
    repeated_tool_lines: list[str]
    repeated_progress: list[tuple[int, int]]
    stored_progress_events: list[tuple[int, int]]
    stored_ask_answers: int
    plan: dict[str, Any]


def _repeats(items: list[Any]) -> list[Any]:
    return sorted({item for item in items if items.count(item) > 1}, key=items.index)


async def check_survives_sigkill(tmp_path: Path, *, mode: Mode = Mode.LIVE) -> KillResult:
    """Pause in process 1, resume in process 2 and kill it inside `propose_plan`
    after its progress is sent, resume again in process 3: the turn completes
    with the plan and history of a run with no pause, and nothing from before the
    pause runs or is sent again."""
    db, log = tmp_path / "s.db", tmp_path / "log"
    common = ("--log-file", str(log), "--mode", mode)
    first = run_driver(db, *common, "start")
    assert first.returncode == 0, first.stderr
    assert first.lines[-2] == {"type": "interrupt", "id": "planner_ask:ask-1"}

    killed = run_driver(
        db, *common, "--hold-call", "2", "answer", kill_when=lambda line: line["type"] == "progress"
    )
    assert killed.returncode == -9, "the process was not killed while alive"
    assert killed.lines[-1]["type"] == "progress"

    third = run_driver(db, *common, "answer", "approve")
    assert third.returncode == 0, third.stderr
    outputs = [line["output"] for line in third.lines if line["type"] == "output"]
    drafted, approved = outputs[0], outputs[-1]

    ref_plan, ref_conversation, _ = await reference_run()
    assert drafted["plan"] == ref_plan == approved["approved"], "plan differs"
    stored = ConversationTypeAdapter.validate_json(drafted["history"])
    assert history(stored) == history(ref_conversation), "planner history differs"

    session = await session_service(db).get_session(
        app_name="pai_node_stream", user_id=USER, session_id=SESSION
    )
    assert session is not None
    lines = log.read_text().splitlines()
    sent = [
        (line["call"], line["done"])
        for run in (first, killed, third)
        for line in run.lines
        if line["type"] == "progress"
    ]
    result = KillResult(
        killed_returncode=killed.returncode,
        repeated_model_calls=_repeats([line for line in lines if line.startswith("model")]),
        repeated_tool_lines=_repeats([line for line in lines if not line.startswith("model")]),
        repeated_progress=_repeats(sent),
        stored_progress_events=progress_list(session.events),
        stored_ask_answers=sum(
            1
            for event in session.events
            for response in event.get_function_responses()
            if response.id == "planner_ask:ask-1"
        ),
        plan=drafted["plan"],
    )
    before_pause = {"model call after 1 messages", "model call after 3 messages"}
    repeated = set(result.repeated_model_calls) | set(result.repeated_tool_lines)
    assert not repeated & (before_pause | {"propose_plan call=1 start", "ask_user body"}), (
        f"repeated from before the pause: {repeated}"
    )
    assert not [p for p in result.repeated_progress if p[0] == 1], (
        f"repeated from before the pause: {result.repeated_progress}"
    )
    return result


# N5 ---------------------------------------------------------------------------------


@dataclass
class StopResult:
    left_running: list[str]
    unretrieved: list[str]


async def check_stop(db: Path, *, mode: Mode = Mode.LIVE) -> StopResult:
    """Cancelling the consumer mid-planning cancels the planner: no task is left
    running, no exception goes unretrieved, and a new turn on the session runs."""
    loop = asyncio.get_running_loop()
    unretrieved: list[str] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: unretrieved.append(context["message"]))
    baseline = asyncio.all_tasks()
    deps = Deps(gate=asyncio.Event(), gate_call=1, gate_timeout=30)
    runner = _runner(db, deps, mode)
    await new_session(runner, SESSION)
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
        assert not left, f"tasks left running: {left}"
        assert not unretrieved, f"unretrieved exceptions: {unretrieved}"
        assert "propose_plan call=1 cancelled" in deps.log, "the planner was not cancelled"

        deps.gate = None
        seen = await asyncio.wait_for(segment(runner, SESSION, user_text("again")), timeout=5)
        assert not [e.error_code for e in seen if e.error_code]
        assert progress_list(seen) == [(1, 1), (1, 2), (1, 3)]
        assert last_interrupt(seen)["interruptId"] == "planner_ask:ask-1"
        return StopResult(left, unretrieved)
    finally:
        for task in asyncio.all_tasks() - baseline - {asyncio.current_task()}:
            task.cancel()
        loop.set_exception_handler(previous_handler)


# N6 ---------------------------------------------------------------------------------


@dataclass
class FailureResult:
    error_codes: list[str | None]
    raised: str | None
    node_outputs: list[Any]
    interrupts: int


async def observe_tool_failure(db: Path, *, mode: Mode) -> FailureResult:
    """What a `run_async` consumer sees when `propose_plan` raises."""
    deps = Deps(fail_in_call=1)
    runner = _runner(db, deps, mode)
    await new_session(runner, SESSION)
    seen: list[Event] = []
    raised = None
    try:
        # A loop, not a comprehension: the events before the raise must be kept.
        async for event in events(runner, SESSION, user_text("go")):
            seen.append(event)  # noqa: PERF401
    except Exception as error:
        raised = f"{type(error).__name__}: {error}"
    return FailureResult(
        error_codes=[e.error_code for e in seen if e.error_code],
        raised=raised,
        node_outputs=[e.output for e in seen if e.output is not None],
        interrupts=sum(1 for e in seen if interrupt_of(e) is not None),
    )


async def check_tool_failure(tmp_path: Path, *, mode: Mode = Mode.LIVE) -> FailureResult:
    """A tool's exception ends the stream as the node's failure, as it does for the
    awaited planner, within 5 s."""
    got = await asyncio.wait_for(observe_tool_failure(tmp_path / "a.db", mode=mode), timeout=5)
    awaited = await observe_tool_failure(tmp_path / "b.db", mode=Mode.AFTER_AWAIT)
    assert got == awaited, f"{got} differs from the awaited planner's {awaited}"
    assert got.error_codes and got.interrupts == 0
    return got
