"""O1 to O5: the Pydantic AI `orchestrate` node inside real Studio turns.

Each turn runs the real root workflow behind the real Studio routes, with dev's
`ScriptedPlannerModel` (or, in O5, dev's ADK `PlannerSeam`), `ScriptedExecutor`
and `ScriptedSurface`. The `pai` arm swaps in `pai_orchestrate_node`; the `dev`
arm keeps dev's node and `LlmAgent`, both playing the same `script.Step`s.

Each pass line is a function returning the ways a run fails it, so the test and
its control assert on the same check: the test that it finds nothing, the control
that it finds the failure the control was built to cause.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from compiler.surface import FixtureSurface
from google.adk.sessions import InMemorySessionService
from harness.graph.state import ORCHESTRATOR_PLAN_RECORDS_KEY
from harness.planner_seam import PlannerSeam
from harness.scripted import ScriptedSurface
from harness.settings import Settings
from harness.tool_contracts import HarnessCode
from orchestrator.agent import INSTRUCTION
from orchestrator_node import CONVERSATION_KEY, LEAK_SENTINEL, Variant
from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart
from script import Played, Step, ask_step, plan_step, search_step
from tests.context_probe import RecordingLlm
from tests.orchestrator_fakes import (
    CONVERSATION,
    request_text,
    stream,
    studio_client,
    turn_body,
)
from wiring import (
    AUTH,
    CANCEL,
    Arm,
    build_app,
    differences,
    events,
    frame,
    interrupt_body,
    stable,
    turn_sessions,
)

BRIEF = "Kenya's debt service since 2015"
REPLY = ["Kenya's debt service ", "rose after 2015, ", "as the plan shows."]
THOUGHT = ["Reading the plan's ", "results."]


def _replies(frames: list[dict[str, Any]]) -> tuple[list[str], str | None]:
    """The `message.delta` texts of a segment and its `message.complete` text."""
    deltas = [f["text"] for f in frames if f["event"] == "message.delta"]
    complete = [f["text"] for f in frames if f["event"] == "message.complete"]
    return deltas, (complete[0] if complete else None)


# ---------------------------------------------------------------------------
# A full plan turn: question, plan, planner, approval, execute, plan_done, reply
# ---------------------------------------------------------------------------


@dataclass
class PlanTurn:
    first: list[dict[str, Any]]
    ran: list[dict[str, Any]]
    played: Played
    stored: dict[str, Any] = field(default_factory=dict)


async def _plan_turn(
    monkeypatch: pytest.MonkeyPatch, arm: Arm, variant: Variant = Variant.LIVE
) -> PlanTurn:
    steps = [
        plan_step(BRIEF, thought=["The question needs ", "a calculation."]),
        Step(THOUGHT, REPLY),
    ]
    with monkeypatch.context() as patch:
        session_service = InMemorySessionService()
        app, played = build_app(
            patch,
            steps,
            arm=arm,
            variant=variant,
            tool_surface=ScriptedSurface(),
            session_service=session_service,
        )
        async with studio_client(app) as client:
            first = await stream(client, "/api/studio/turn", turn_body(f"{BRIEF}?"))
            card = frame(first, "plan.proposed")
            ran = await stream(
                client, "/api/studio/interrupt", interrupt_body(card["interrupt_id"], "approve")
            )
        store = app.state.studio.store
        (principal,) = session_service.sessions["dfo_analyst_graph"]
        conversation = await store.load(principal, CONVERSATION)
    (session,) = turn_sessions(session_service)
    state = dict(session.state)
    return PlanTurn(
        first=first,
        ran=ran,
        played=played,
        stored={
            "state": state,
            "approvals": [a.model_dump(mode="json") for a in conversation.approvals],
            "turns": list(conversation.turns),
        },
    )


def o1_failures(pai: PlanTurn, dev: PlanTurn) -> list[str]:
    """How the `pai` turn fails O1 against `dev`'s."""
    failures: list[str] = []
    deltas, complete = _replies(pai.ran)
    if events(pai.ran)[-3:] != ["message.complete", "suggestions", "done"]:
        failures.append(f"the turn did not end on a reply: {events(pai.ran)[-3:]}")
    if deltas != REPLY or complete != "".join(REPLY):
        failures.append(f"reply not streamed chunk by chunk: {deltas!r} then {complete!r}")
    for name, a, b in (("first", pai.first, dev.first), ("ran", pai.ran, dev.ran)):
        if found := differences(stable(a), stable(b)):
            failures.append(f"{name} segment frames differ from dev's: {found[:5]}")
    pai_state, dev_state = pai.stored["state"], dev.stored["state"]
    extra = set(pai_state) - set(dev_state)
    if extra != {CONVERSATION_KEY} or pai_state[CONVERSATION_KEY] is not None:
        failures.append(f"state keys only pai holds: {extra}")
    if missing := set(dev_state) - set(pai_state):
        failures.append(f"state keys only dev holds: {missing}")
    orchestrator_keys = sorted(
        k for k in dev_state if k.startswith("orchestrator") and k != CONVERSATION_KEY
    )
    failures += [
        f"state[{key}] differs: {found[:3]}"
        for key in [*orchestrator_keys, "plan", "awaiting_approve"]
        if (found := differences(stable(pai_state.get(key)), stable(dev_state.get(key))))
    ]
    failures += [
        f"{key} differ: {found[:3]}"
        for key in ("approvals", "turns")
        if (found := differences(stable(pai.stored[key]), stable(dev.stored[key])))
    ]
    return failures


async def test_o1_a_full_plan_turn_streams_its_reply_and_stores_what_dev_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pai = await _plan_turn(monkeypatch, "pai")
    dev = await _plan_turn(monkeypatch, "dev")

    print(f"pai reply frames: {_replies(pai.ran)}")
    print(f"pai visited: {events(pai.first)} / {events(pai.ran)}")
    assert pai.stored["state"][ORCHESTRATOR_PLAN_RECORDS_KEY]
    assert o1_failures(pai, dev) == []
    # Dev's own agent streams the same chunks, so both arms send the same frames.
    assert _replies(dev.ran) == _replies(pai.ran)


async def test_o1_control_a_buffering_node_sends_the_reply_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pai = await _plan_turn(monkeypatch, "pai", Variant.BUFFERED)
    dev = await _plan_turn(monkeypatch, "dev")

    failures = o1_failures(pai, dev)
    print(f"buffered control: {failures}")
    assert _replies(pai.ran) == (["".join(REPLY)], "".join(REPLY))
    assert any(f.startswith("reply not streamed") for f in failures)


# ---------------------------------------------------------------------------
# O3: the hand-back is the result of the agent's `plan` call
# ---------------------------------------------------------------------------


def _dev_plan_result(dev: PlanTurn) -> dict[str, Any]:
    request = dev.played.requests[-1]
    (result,) = [
        part.function_response.response or {}
        for content in request.contents
        for part in content.parts or []
        if part.function_response is not None and part.function_response.name == "plan"
    ]
    return result


def o3_failures(pai: PlanTurn, dev: PlanTurn) -> list[str]:
    """How the `pai` turn fails O3: the request after `plan_done` must answer the
    `plan` call with the hand-back's `text` and `body`, as `_call_result` builds
    it on dev, and carry nothing else new."""
    failures: list[str] = []
    messages = pai.played.requests[-1]
    last = messages[-1]
    assert isinstance(last, ModelRequest)
    returns = [p for p in last.parts if isinstance(p, ToolReturnPart) and p.tool_name == "plan"]
    prompts = [p for p in last.parts if isinstance(p, UserPromptPart)]
    if len(returns) != 1:
        failures.append(f"{len(returns)} plan results in the last request")
        return failures
    (returned,) = returns
    expected = _dev_plan_result(dev)
    if found := differences(stable(returned.content), stable(expected)):
        failures.append(f"plan result differs from dev's hand-back: {found[:3]}")
    if prompts:
        failures.append(f"the hand-back came with {len(prompts)} user messages")
    if not str(expected.get("result")).startswith("The approved plan ran."):
        failures.append("dev's hand-back is not the plan's run")
    return failures


async def test_o3_the_plan_hand_back_is_the_result_of_the_plan_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pai = await _plan_turn(monkeypatch, "pai")
    dev = await _plan_turn(monkeypatch, "dev")

    returned = pai.played.requests[-1][-1].parts[0]
    print(f"pai plan result keys: {sorted(returned.content)}")
    assert o3_failures(pai, dev) == []
    assert pai.played.log == ["model call after 1 messages", "model call after 3 messages"]


async def test_o3_control_a_hand_back_sent_as_a_user_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pai = await _plan_turn(monkeypatch, "pai", Variant.HAND_BACK_AS_MESSAGE)
    dev = await _plan_turn(monkeypatch, "dev")

    failures = o3_failures(pai, dev)
    print(f"hand-back-as-message control: {failures}")
    assert any(f.startswith("plan result differs") for f in failures)
    assert any("user messages" in f for f in failures)


# ---------------------------------------------------------------------------
# O2: the clarify pause
# ---------------------------------------------------------------------------

QUESTION = "Which years do you mean?"
ANSWERED = ["Since 2015 ", "it is."]


@dataclass
class AskTurn:
    first: list[dict[str, Any]]
    resumed: list[dict[str, Any]]
    played: Played


async def _ask_turn(
    monkeypatch: pytest.MonkeyPatch, arm: Arm, variant: Variant = Variant.LIVE
) -> AskTurn:
    steps = [ask_step(QUESTION, ["Since 2015", "Since 2020"]), Step(text=ANSWERED)]
    with monkeypatch.context() as patch:
        app, played = build_app(patch, steps, arm=arm, variant=variant)
        async with studio_client(app) as client:
            first = await stream(client, "/api/studio/turn", turn_body("Kenya's debt lately?"))
            asked = frame(first, "elicitation.request")
            answer = {"q1": {"kind": "text", "text": "Since 2015"}}
            resumed = await stream(
                client,
                "/api/studio/interrupt",
                interrupt_body(asked["interrupt_id"], "answer", answers=answer),
            )
    return AskTurn(first=first, resumed=resumed, played=played)


def _dev_ask_result(dev: AskTurn) -> dict[str, Any]:
    (result,) = [
        part.function_response.response or {}
        for content in dev.played.requests[-1].contents
        for part in content.parts or []
        if part.function_response is not None and part.function_response.name == "ask_user"
    ]
    return result


def o2_failures(pai: AskTurn, dev: AskTurn) -> list[str]:
    failures: list[str] = []
    if found := differences(stable(pai.first), stable(dev.first)):
        failures.append(f"the question's frames differ from dev's: {found[:3]}")
    if events(pai.first)[-1] != "elicitation.request":
        failures.append(f"the first segment ends on {events(pai.first)[-1]}")
    counts = pai.played.log
    if len(set(counts)) != len(counts):
        failures.append(f"a model call repeated: {counts}")
    if pai.played.requests and len(pai.played.requests) >= 2:
        last = pai.played.requests[-1][-1]
        returns = [
            p for p in last.parts if isinstance(p, ToolReturnPart) and p.tool_name == "ask_user"
        ]
        expected = _dev_ask_result(dev)
        if [r.content for r in returns] != [expected]:
            failures.append(f"ask_user result {[r.content for r in returns]} != {expected}")
    if _replies(pai.resumed) != (ANSWERED, "".join(ANSWERED)):
        failures.append(f"resumed reply: {_replies(pai.resumed)}")
    if found := differences(stable(pai.resumed), stable(dev.resumed)):
        failures.append(f"the resumed frames differ from dev's: {found[:3]}")
    return failures


async def test_o2_the_answer_resumes_the_stored_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pai = await _ask_turn(monkeypatch, "pai")
    dev = await _ask_turn(monkeypatch, "dev")

    print(f"question as Studio shows it: {frame(pai.first, 'elicitation.request')['questions']}")
    print(f"pai model log: {pai.played.log}; dev: {dev.played.log}")
    assert o2_failures(pai, dev) == []
    assert pai.played.log == ["model call after 1 messages", "model call after 3 messages"]


async def test_o2_control_restarting_from_the_users_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pai = await _ask_turn(monkeypatch, "pai", Variant.RESTART)
    dev = await _ask_turn(monkeypatch, "dev")

    failures = o2_failures(pai, dev)
    print(f"restart control: {pai.played.log}; {failures}")
    assert pai.played.log == ["model call after 1 messages", "model call after 1 messages"]
    assert any(f.startswith("a model call repeated") for f in failures)
    assert any(f.startswith("ask_user result") for f in failures)


# ---------------------------------------------------------------------------
# O4: a tool that raises
# ---------------------------------------------------------------------------

FAILED_REPLY = ["Search failed, ", "so I cannot give Kenya's GDP."]


async def _breaking_resolver(**kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("upstream at https://internal.example/search refused")


async def _failing_turn(
    monkeypatch: pytest.MonkeyPatch, arm: Arm, *, answer_failures: bool = True
) -> tuple[list[dict[str, Any]], Played]:
    steps = [search_step("Kenya's GDP?", "GDP"), Step(text=FAILED_REPLY)]
    with monkeypatch.context() as patch:
        app, played = build_app(
            patch, steps, arm=arm, answer_failures=answer_failures, resolver=_breaking_resolver
        )
        async with studio_client(app) as client:
            frames = await stream(client, "/api/studio/turn", turn_body("Kenya's GDP?"))
    return frames, played


def o4_failures(
    frames: list[dict[str, Any]], played: Played, dev_frames: list[dict[str, Any]]
) -> list[str]:
    failures: list[str] = []
    if events(frames)[-1] != "done":
        failures.append(f"the turn ended on {events(frames)[-1]}")
    if len(played.requests) < 2:
        failures.append(f"the model was called {len(played.requests)} times")
    else:
        (answer,) = [
            p.content
            for p in played.requests[-1][-1].parts
            if isinstance(p, ToolReturnPart) and p.tool_name == "search"
        ]
        if not str(answer.get("error", "")).startswith("search failed inside DFO (RuntimeError)."):
            failures.append(f"search answered {answer}")
        if (answer.get("code"), answer.get("retry_mode")) != (HarnessCode.RAISED, "do_not_retry"):
            failures.append(f"search answered code {answer.get('code')}")
        if "internal.example" in str(answer):
            failures.append("the exception's message reached the model")
    if found := differences(stable(frames), stable(dev_frames)):
        failures.append(f"frames differ from dev's: {found[:3]}")
    return failures


async def test_o4_a_tool_that_raises_answers_the_model_and_the_turn_goes_on(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR, logger="orchestrator.tool_failures"):
        frames, played = await _failing_turn(monkeypatch, "pai")
    (record,) = [r for r in caplog.records if r.name == "orchestrator.tool_failures"]
    dev_frames, _ = await _failing_turn(monkeypatch, "dev")

    print(f"pai frames: {events(frames)}")
    assert o4_failures(frames, played, dev_frames) == []
    assert record.exc_info is not None and record.exc_info[0] is RuntimeError


async def test_o4_control_no_failure_handling_ends_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames, played = await _failing_turn(monkeypatch, "pai", answer_failures=False)
    dev_frames, _ = await _failing_turn(monkeypatch, "dev")

    failures = o4_failures(frames, played, dev_frames)
    print(f"no-handling control: {events(frames)}; {frame(frames, 'error')['error']}")
    assert events(frames)[-1] == "error"
    assert any(f.startswith("the turn ended on error") for f in failures)


# ---------------------------------------------------------------------------
# O5: isolation, against dev's ADK planner
# ---------------------------------------------------------------------------

ORCHESTRATOR_SENTINEL = "ORCHESTRATOR-ONLY-SENTINEL"
PLANNER_SENTINEL = "PLANNER-ONLY-SENTINEL"


@dataclass
class Isolation:
    orchestrator_texts: list[str]
    planner_texts: list[str]
    planner_tools: set[str]
    frames: list[dict[str, Any]]
    session_text: str


async def _isolation_turn(monkeypatch: pytest.MonkeyPatch, variant: Variant) -> Isolation:
    seam = PlannerSeam(settings=Settings(), surface=FixtureSurface(schemas=[]))
    planner_llm = RecordingLlm(
        script=[f"I cannot plan this. {PLANNER_SENTINEL}"], agent_name="planner", scenario="o5"
    )
    seam._agent.model = planner_llm
    steps = [
        Step(
            thought=[f"Thinking it over. {ORCHESTRATOR_SENTINEL}"],
            text=[f"Let me plan this. {ORCHESTRATOR_SENTINEL}"],
            calls=[("plan", {"brief": BRIEF, "asks": [BRIEF]})],
        ),
        Step(text=["No plan could be made."]),
    ]
    session_service = InMemorySessionService()
    with monkeypatch.context() as patch:
        app, played = build_app(
            patch, steps, variant=variant, planner=seam, session_service=session_service
        )
        async with studio_client(app) as client:
            frames = await stream(client, "/api/studio/turn", turn_body(f"{BRIEF}?"))
    (session,) = turn_sessions(session_service)
    planner_tools = {tool.name for tool in seam._agent.tools if hasattr(tool, "name")}
    return Isolation(
        orchestrator_texts=[repr(messages) for messages in played.requests],
        planner_texts=[
            request_text(r)
            + "\n".join(str(p) for c in r.contents for p in c.parts or [])
            + str(r.config.system_instruction)
            for r in planner_llm.captured
        ],
        planner_tools=planner_tools,
        frames=frames,
        session_text="\n".join(str(event.content) for event in session.events),
    )


def o5_failures(run: Isolation) -> list[str]:
    failures: list[str] = []
    if not run.planner_texts:
        failures.append("the planner made no model call")
    orchestrator_only = [
        ORCHESTRATOR_SENTINEL,
        LEAK_SENTINEL,
        INSTRUCTION.splitlines()[0],
        "make_charts",
        "call-1",
    ]
    for text in run.planner_texts:
        failures += [f"planner request holds {s!r}" for s in orchestrator_only if s in text]
    planner_only = [PLANNER_SENTINEL, *run.planner_tools]
    for text in run.orchestrator_texts:
        failures += [f"orchestrator request holds {s!r}" for s in planner_only if s in text]
    return failures


async def test_o5_neither_agent_sees_the_others_internals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = await _isolation_turn(monkeypatch, Variant.LIVE)

    print(f"planner tools: {sorted(run.planner_tools)}; frames: {events(run.frames)}")
    assert o5_failures(run) == []
    # The checks read the right requests: each holds its own agent's material.
    assert run.planner_tools
    assert any(BRIEF in t for t in run.planner_texts)
    assert all(any(tool in t for t in run.planner_texts) for tool in run.planner_tools)
    assert any(ORCHESTRATOR_SENTINEL in t for t in run.orchestrator_texts)
    assert len(run.orchestrator_texts) == 2


async def test_o5_control_a_sentinel_leaked_into_state_reaches_the_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one channel from this node to the planner is session state: a node
    that writes orchestrator text into the pending hand-off's brief is caught."""
    run = await _isolation_turn(monkeypatch, Variant.LEAK_STATE)

    failures = o5_failures(run)
    print(f"state leak control: {failures}")
    assert f"planner request holds {LEAK_SENTINEL!r}" in failures


async def test_o5_probe_an_unbranched_orchestrator_event_misses_the_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe: the orchestrator's own text written into the session on no branch.
    Dev's planner runs as a `single_turn` node agent, which reads only the events
    of its own node path and its ancestors, so the event never reaches it."""
    run = await _isolation_turn(monkeypatch, Variant.LEAK_EVENT)

    assert LEAK_SENTINEL in run.session_text
    assert o5_failures(run) == []


# ---------------------------------------------------------------------------
# Stop mid-reply (measured; not a pass line)
# ---------------------------------------------------------------------------


def _tasks() -> set[asyncio.Task[Any]]:
    return {t for t in asyncio.all_tasks() if not t.done()}


@pytest.fixture
async def baseline_tasks() -> AsyncIterator[set[asyncio.Task[Any]]]:
    yield _tasks()


async def test_stop_mid_reply_cancels_the_orchestrator(
    monkeypatch: pytest.MonkeyPatch, baseline_tasks: set[asyncio.Task[Any]]
) -> None:
    """Studio's Stop while the reply streams: the model stream is cancelled, the
    turn's stream ends with the `cancelled` error after the chunk already sent,
    no task is left, no conversation is stored, and the next turn runs."""
    steps = [Step(text=["First half, ", "second half."], stall=30.0), Step(text=["Ghana."])]
    session_service = InMemorySessionService()
    app, played = build_app(monkeypatch, steps, session_service=session_service)

    async def wait_for(predicate: Any) -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    async with studio_client(app) as client:
        turn = asyncio.ensure_future(stream(client, "/api/studio/turn", turn_body("Kenya?")))
        await asyncio.wait_for(wait_for(lambda: played.log), 5)
        await asyncio.sleep(0.2)
        cancel = await client.post("/api/studio/cancel", json=CANCEL, headers=AUTH)
        stopped = await asyncio.wait_for(turn, 5)
        await asyncio.sleep(0.1)
        left = _tasks() - baseline_tasks - {asyncio.current_task()}
        nxt = await stream(client, "/api/studio/turn", turn_body("And Ghana?"))

    print(f"stopped frames: {events(stopped)}")
    assert cancel.status_code == 204
    assert events(stopped)[-1] == "error"
    assert frame(stopped, "error")["error"]["error"] == "cancelled"
    assert [f["text"] for f in stopped if f["event"] == "message.delta"] == ["First half, "]
    assert "message.complete" not in events(stopped)
    assert not left, [t.get_coro() for t in left]
    first, *_ = sorted(turn_sessions(session_service), key=lambda s: s.id)
    assert first.state.get(CONVERSATION_KEY) is None
    assert _replies(nxt) == (["Ghana."], "Ghana.")
