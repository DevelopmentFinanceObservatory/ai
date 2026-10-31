"""S1 to S5: stage A's planner node inside real Studio turns.

Each turn runs the real root workflow (`create_root_workflow`) behind the real
Studio routes (`orchestrator_app`), with the scripted orchestrator model, the
`ScriptedExecutor` and, where a plan runs, the `ScriptedSurface`. Only the
`planner` node differs between the two arms: `pai` is `node.pai_planner_node`,
`dev` is dev's `planner_node` drafting with `node.DevStubSeam`.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

import pytest
from google.adk.sessions import InMemorySessionService
from harness.graph.state import AGENT_NAME, ORCHESTRATOR_PLAN_RECORDS_KEY
from harness.scripted import ScriptedSurface
from harness.studio_translate import _CHECKING
from node import ASK_PREFIX, DevStubSeam, Leak
from studio_planner import SENTINEL, StudioDeps
from tests.orchestrator_fakes import (
    CONVERSATION,
    carries_an_interrupt,
    orchestrator_app,
    plan_call,
    quotes_another_agent,
    request_text,
    stream,
    studio_client,
    turn_body,
)
from wiring import (
    AUTH,
    Timed,
    events,
    frame,
    is_progress,
    read_timed,
    show_planner_progress,
    streaming_client,
    use_pai_planner,
)

BRIEF = "ODA to Ghana since 2015"
type Arm = Literal["pai", "dev"]


def build_app(
    monkeypatch: pytest.MonkeyPatch,
    deps: StudioDeps,
    *,
    arm: Arm = "pai",
    leak: Leak | None = None,
    tool_surface: Any = None,
    session_service: InMemorySessionService | None = None,
    plans: int = 1,
) -> tuple[Any, Any]:
    show_planner_progress(monkeypatch)
    planner = None
    if arm == "pai":
        use_pai_planner(monkeypatch, deps, leak=leak)
    else:
        planner = DevStubSeam(deps)
    return orchestrator_app(
        *[plan_call(BRIEF)] * plans,
        planner=planner,
        tool_surface=tool_surface,
        session_service=session_service,
    )


def interrupt_body(interrupt_id: str, decision: str, **extra: Any) -> dict[str, Any]:
    return {
        "kind": "interrupt.response",
        "schema_version": 1,
        "conversation_id": CONVERSATION,
        "interrupt_id": interrupt_id,
        "decision": decision,
        **extra,
    }


CANCEL = {"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION}


def turn_sessions(session_service: InMemorySessionService) -> list[Any]:
    """Every per-turn ADK session Studio created, whoever the principal is."""
    return [
        session
        for by_user in session_service.sessions[AGENT_NAME].values()
        for session in by_user.values()
        if session.id.startswith(CONVERSATION + "#")
    ]


def details(frames: list[dict[str, Any]]) -> list[str]:
    return [f["detail"] for f in frames if is_progress(f)]


# ---------------------------------------------------------------------------
# S1: progress reaches the stream while `propose_plan` waits
# ---------------------------------------------------------------------------


async def _first_progress(monkeypatch: pytest.MonkeyPatch, arm: Arm) -> dict[str, Any]:
    """Run a turn to its plan card over the streaming client. `propose_plan`
    waits after its first report until the client has read a progress frame, or
    3 s. Returns when the first progress frame arrived, whether `propose_plan` had
    ended by then, and the frames."""
    deps = StudioDeps(ask=False, gate=asyncio.Event(), gate_timeout=3.0)
    app, _ = build_app(monkeypatch, deps, arm=arm)
    timed = Timed()
    seen: dict[str, Any] = {}

    def on_frame(parsed: dict[str, Any]) -> None:
        if is_progress(parsed) and "frame" not in seen:
            seen["frame"] = parsed
            seen["ended_before"] = "propose_plan call=1 end" in deps.log
            assert deps.gate is not None
            deps.gate.set()

    started = time.monotonic()
    async with (
        streaming_client(app) as client,
        client.stream("POST", "/api/studio/turn", json=turn_body(BRIEF), headers=AUTH) as response,
    ):
        assert response.status_code == 200
        await asyncio.wait_for(read_timed(response, timed, started, on_frame=on_frame), 15)
    first = timed.at[timed.frames.index(seen["frame"])]
    return {**seen, "seconds": first, "frames": timed.frames, "log": deps.log}


async def test_s1_progress_streams_while_propose_plan_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = await _first_progress(monkeypatch, "pai")
    print(f"pai: first progress frame at {seen['seconds']:.3f} s")

    assert seen["seconds"] < 2.0
    assert not seen["ended_before"]
    assert "propose_plan call=1 gate timed out" not in seen["log"]
    first = seen["frame"]
    assert (first["label"], first["status"], first["detail"]) == (
        _CHECKING,
        "running",
        "check 1 of 3",
    )
    assert details(seen["frames"]) == ["check 1 of 3", "check 2 of 3", "check 3 of 3"]
    assert events(seen["frames"])[-1] == "plan.proposed"


async def test_s1_control_dev_node_relays_progress_only_after_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = await _first_progress(monkeypatch, "dev")
    print(f"dev: first progress frame at {seen['seconds']:.3f} s")

    assert seen["seconds"] >= 3.0
    assert seen["ended_before"]
    assert "propose_plan call=1 gate timed out" in seen["log"]
    assert details(seen["frames"]) == ["check 1 of 3", "check 2 of 3", "check 3 of 3"]
    assert events(seen["frames"])[-1] == "plan.proposed"


# ---------------------------------------------------------------------------
# S2: the planner's ask is a Studio question, and its answer resumes the planner
# ---------------------------------------------------------------------------


async def test_s2_ask_shows_as_a_question_and_its_answer_resumes_the_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps = StudioDeps(ask=True, progress=10)
    session_service = InMemorySessionService()
    app, _ = build_app(monkeypatch, deps, session_service=session_service)

    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", turn_body(BRIEF))
        asked = frame(first, "elicitation.request")
        answer = {"q1": {"kind": "text", "text": "Ghana only"}}
        body = interrupt_body(asked["interrupt_id"], "answer", answers=answer)
        resumed = await stream(client, "/api/studio/interrupt", body)
        again = await client.post("/api/studio/interrupt", json=body, headers=AUTH)

    # Every report of the first proposal, in order, then the question.
    ask_at = events(first).index("elicitation.request")
    assert details(first) == [f"check {n} of 10" for n in range(1, 11)]
    assert all(i < ask_at for i, f in enumerate(first) if is_progress(f))
    assert events(first)[-1] == "elicitation.request"
    (question,) = asked["questions"]
    assert question["question"] == f"Which scope should the plan cover? {SENTINEL}"
    print(f"question as Studio shows it: {question}")

    # The answer continues the stored run: no model call or tool body repeats.
    assert details(resumed) == [f"check {n} of 10" for n in range(1, 11)]
    assert events(resumed)[-1] == "plan.proposed"
    assert deps.log == [
        "model call after 1 messages",
        "propose_plan call=1 start",
        "propose_plan call=1 end",
        "model call after 3 messages",
        "ask_user body",
        "model call after 5 messages",
        "propose_plan call=2 start",
        "propose_plan call=2 end",
        "propose_plan call=2 scope=Ghana only",
        "model call after 7 messages",
    ]

    # The same answer posted again: the route no longer holds the question.
    print(f"same answer again: {again.status_code} {again.json()}")
    assert again.status_code == 404
    (session,) = turn_sessions(session_service)
    responses = [
        response.id
        for event in session.events
        for response in event.get_function_responses()
        if (response.id or "").startswith(ASK_PREFIX)
    ]
    assert responses == [ASK_PREFIX + "ask-1"]


# ---------------------------------------------------------------------------
# S3: the orchestrator's model sees none of the planner's messages
# ---------------------------------------------------------------------------


async def _orchestrator_requests(
    monkeypatch: pytest.MonkeyPatch,
    *,
    leak: Leak | None = None,
    session_service: InMemorySessionService | None = None,
) -> list[Any]:
    """Run ask, answer, approve and the plan's run; every orchestrator request."""
    deps = StudioDeps(ask=True)
    app, model = build_app(
        monkeypatch,
        deps,
        leak=leak,
        tool_surface=ScriptedSurface(),
        session_service=session_service,
    )
    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", turn_body(BRIEF))
        asked = frame(first, "elicitation.request")
        answer = {"q1": {"kind": "text", "text": "Ghana only"}}
        resumed = await stream(
            client,
            "/api/studio/interrupt",
            interrupt_body(asked["interrupt_id"], "answer", answers=answer),
        )
        card = frame(resumed, "plan.proposed")
        ran = await stream(
            client, "/api/studio/interrupt", interrupt_body(card["interrupt_id"], "approve")
        )
    assert events(ran)[-1] == "done", events(ran)
    return model.requests


async def test_s3_orchestrator_requests_hold_no_planner_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = await _orchestrator_requests(monkeypatch)

    assert len(requests) >= 2  # the plan call, and the hand-back of the plan's run
    for request in requests:
        text = request_text(request) + "\n".join(
            str(p) for c in request.contents for p in c.parts or []
        )
        assert SENTINEL not in text
        for planner_only in ("propose_plan", "ask_user", "propose-1", "ask-1", ASK_PREFIX):
            assert planner_only not in text
        assert not carries_an_interrupt(request)
        assert not quotes_another_agent(request)


async def test_s3_control_the_requests_show_what_reaches_the_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check above reads every content the orchestrator is sent: its last
    request holds the user's brief and the plan's hand-back."""
    requests = await _orchestrator_requests(monkeypatch)

    text = request_text(requests[-1])
    assert BRIEF in text
    assert "The approved plan ran." in text


@pytest.mark.parametrize("leak", list(Leak))
async def test_s3_planner_content_in_the_session_still_misses_the_orchestrator(
    monkeypatch: pytest.MonkeyPatch, leak: Leak
) -> None:
    """Probe: the planner's text written into the turn's session as content, in
    three shapes. None reaches a request; see NOTES.md for why."""
    session_service = InMemorySessionService()
    requests = await _orchestrator_requests(monkeypatch, leak=leak, session_service=session_service)

    (session,) = turn_sessions(session_service)
    assert any(SENTINEL in str(event.content) for event in session.events)
    assert all(SENTINEL not in str(request.contents) for request in requests)


# ---------------------------------------------------------------------------
# S4: Stop or disconnect mid-planning
# ---------------------------------------------------------------------------


def _tasks() -> set[asyncio.Task[Any]]:
    return {t for t in asyncio.all_tasks() if not t.done()}


@pytest.fixture
async def baseline_tasks() -> AsyncIterator[set[asyncio.Task[Any]]]:
    yield _tasks()


async def _wait_for(predicate: Any, seconds: float) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), seconds)


async def _next_turn_runs(client: Any, deps: StudioDeps) -> list[dict[str, Any]]:
    deps.gate = None
    return await stream(client, "/api/studio/turn", turn_body("And to Kenya?"))


async def test_s4_stop_mid_planning_cancels_the_planner(
    monkeypatch: pytest.MonkeyPatch, baseline_tasks: set[asyncio.Task[Any]]
) -> None:
    deps = StudioDeps(ask=False, gate=asyncio.Event(), gate_timeout=30.0)
    app, _ = build_app(monkeypatch, deps, plans=2)

    async with studio_client(app) as client:
        turn = asyncio.ensure_future(stream(client, "/api/studio/turn", turn_body(BRIEF)))
        await _wait_for(lambda: "propose_plan call=1 start" in deps.log, 5)
        cancel = await client.post("/api/studio/cancel", json=CANCEL, headers=AUTH)
        stopped = await asyncio.wait_for(turn, 5)
        await asyncio.sleep(0.1)
        left = _tasks() - baseline_tasks - {asyncio.current_task()}
        nxt = await _next_turn_runs(client, deps)

    assert cancel.status_code == 204
    assert events(stopped)[-1] == "error"
    assert frame(stopped, "error")["error"]["error"] == "cancelled"
    assert "propose_plan call=1 cancelled" in deps.log
    assert (
        "propose_plan call=1 end" not in deps.log[: deps.log.index("propose_plan call=1 cancelled")]
    )
    assert not left, [t.get_coro() for t in left]
    assert events(nxt)[-1] == "plan.proposed"
    assert "error" not in events(nxt)


async def test_s4_disconnect_mid_planning_cancels_the_planner(
    monkeypatch: pytest.MonkeyPatch, baseline_tasks: set[asyncio.Task[Any]]
) -> None:
    deps = StudioDeps(ask=False, gate=asyncio.Event(), gate_timeout=30.0)
    app, _ = build_app(monkeypatch, deps, plans=2)
    timed = Timed()

    async with streaming_client(app) as client:
        async with client.stream(
            "POST", "/api/studio/turn", json=turn_body(BRIEF), headers=AUTH
        ) as response:

            async def read_to_first_progress() -> None:
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and '"check 1 of 3"' in line:
                        timed.frames.append({"line": line})
                        return

            await asyncio.wait_for(read_to_first_progress(), 5)
        # Leaving the block closed the response: the app has seen the disconnect
        # and finished (`_Call.close` waits for it).
        await asyncio.sleep(0.1)
        left = _tasks() - baseline_tasks - {asyncio.current_task()}
        first_turn_log = list(deps.log)
        nxt = await stream_after_disconnect(app, deps)

    assert timed.frames
    assert first_turn_log == [
        "model call after 1 messages",
        "propose_plan call=1 start",
        "propose_plan call=1 cancelled",
    ]
    assert not left, [t.get_coro() for t in left]
    assert events(nxt)[-1] == "plan.proposed"


async def stream_after_disconnect(app: Any, deps: StudioDeps) -> list[dict[str, Any]]:
    async with studio_client(app) as client:
        return await _next_turn_runs(client, deps)


# ---------------------------------------------------------------------------
# S5: the stored records match dev's
# ---------------------------------------------------------------------------

# Fields that differ between any two runs: times, and ids minted per run.
# JSON held as a string is compared parsed.
_VOLATILE = {
    "observed_at",
    "run_id",
    "plan_version_id",
    "approved_at",
    "at",
    "seq",
    "interrupt_id",
    "step_id",
    "trace_id",
    "duration_ms",
}


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if k not in _VOLATILE}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    if isinstance(value, str) and value.startswith("{"):
        return _stable(json.loads(value))
    return value


def _differences(a: Any, b: Any, path: str = "") -> list[str]:
    """The paths at which `a` and `b` differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            d
            for key in sorted(set(a) | set(b), key=str)
            for d in _differences(a.get(key), b.get(key), f"{path}.{key}")
        ]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in _differences(x, y, f"{path}[{i}]")
        ]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


async def _stored(monkeypatch: pytest.MonkeyPatch, arm: Arm) -> dict[str, Any]:
    """Run turn, approval and the plan's run; what the session and the store hold."""
    with monkeypatch.context() as patch:
        deps = StudioDeps(ask=False)
        session_service = InMemorySessionService()
        app, _ = build_app(
            patch, deps, arm=arm, tool_surface=ScriptedSurface(), session_service=session_service
        )
        async with studio_client(app) as client:
            first = await stream(client, "/api/studio/turn", turn_body(BRIEF))
            card = frame(first, "plan.proposed")
            ran = await stream(
                client, "/api/studio/interrupt", interrupt_body(card["interrupt_id"], "approve")
            )
        store = app.state.studio.store
        (principal,) = session_service.sessions[AGENT_NAME]
        conversation = await store.load(principal, CONVERSATION)
    assert events(ran)[-1] == "done"
    (session,) = turn_sessions(session_service)
    state = dict(session.state)
    return {
        "card": card,
        "approved": frame(ran, "plan.approved"),
        "state_keys": set(state),
        "plan": state["plan"],
        "awaiting_approve": state.get("awaiting_approve"),
        "plan_records": state.get(ORCHESTRATOR_PLAN_RECORDS_KEY),
        "planner_conversation": state.get("planner:conversation", "absent"),
        "approvals": [a.model_dump(mode="json") for a in conversation.approvals],
        "turns": list(conversation.turns),
    }


async def test_s5_stored_records_match_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    pai = await _stored(monkeypatch, "pai")
    dev = await _stored(monkeypatch, "dev")

    print(f"state keys only pai holds: {pai['state_keys'] - dev['state_keys']}")
    print(f"state keys only dev holds: {dev['state_keys'] - pai['state_keys']}")
    assert pai["state_keys"] - dev["state_keys"] == {"planner:conversation"}
    assert pai["planner_conversation"] is None
    assert dev["state_keys"] - pai["state_keys"] == set()
    assert pai["card"]["revision"] == dev["card"]["revision"]
    assert _stable(pai["card"]) == _stable(dev["card"])
    assert _stable(pai["approved"]) == _stable(dev["approved"])
    assert pai["awaiting_approve"] is dev["awaiting_approve"] is False
    assert _stable(pai["plan"]) == _stable(dev["plan"])
    assert _stable(pai["plan_records"]) == _stable(dev["plan_records"])
    assert _stable(pai["approvals"]) == _stable(dev["approvals"])
    assert _differences(_stable(pai["turns"]), _stable(dev["turns"])) == []
