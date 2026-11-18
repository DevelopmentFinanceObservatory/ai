"""What a Studio turn puts on the wire to Langfuse.

The real Langfuse SDK, tracer provider, span filter and Studio routes run; only
the Langfuse server is fake (`fake_langfuse`). These tests cover the parts an
in-memory span exporter cannot see: which spans the export filter keeps, how a
client that leaves mid-stream ends the turn's telemetry, and the parent chain
of a turn that pauses for approval and resumes.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
from collections.abc import Callable
from contextlib import suppress
from typing import Any

import httpx2
import pytest
from common.telemetry import init_telemetry
from google.genai import types
from harness.scripted import ScriptedExecutor, ScriptedSurface
from harness.studio import StudioRunner
from harness.studio_relay import Channel
from harness.studio_state import ConversationState
from harness.studio_store import SegmentClose, Snapshot
from harness.studio_stream import Emit, own_segment
from harness.turn.loop import Outcome
from harness.wire import ErrorFrame, MessageDelta
from langfuse import Langfuse
from opentelemetry import trace
from opentelemetry.proto.trace.v1.trace_pb2 import Span
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncEngine

from ..scripted_tools import plan_call
from ..studio_routes import orchestrator_app

CONVERSATION = "conv_trace_export_01"
AUTH = {"Authorization": "Bearer test"}
QUESTION = "comprehensive overview of life expectancy in Kenya"
ANSWER = "Life expectancy in Kenya rose."
# The name of the stand-in for a model call span in the test that drives a
# stream's owner with scripted work.
MODEL_CALL = "generate_content gemini-x"


def _attribute(span: Span, key: str) -> str | int | float | bool | None:
    for attribute in span.attributes:
        if attribute.key == key:
            return getattr(attribute.value, attribute.value.WhichOneof("value"))
    return None


def _attribute_keys(span: Span) -> set[str]:
    return {attribute.key for attribute in span.attributes}


def _message_parts(span: Span, key: str) -> list[dict[str, Any]]:
    """The parts of every message in the GenAI messages attribute `key`."""
    messages = json.loads(str(_attribute(span, key)))
    return [part for message in messages for part in message["parts"]]


def _spans_with_an_unexported_parent(spans: list[Span]) -> list[Span]:
    exported = {span.span_id for span in spans}
    return [span for span in spans if span.parent_span_id not in exported]


async def _until(condition: Callable[[], bool], *, seconds: float = 15.0) -> None:
    """Wait for `condition`, failing the test rather than hanging it."""
    deadline = asyncio.get_running_loop().time() + seconds
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail("timed out waiting for the stream to finish unwinding")
        gc.collect()
        await asyncio.sleep(0.05)


# --- a turn that pauses for approval and resumes ---------------------------------


def _planned_app(engine: AsyncEngine) -> Any:
    """The studio with an agent that hands the question to the scripted
    planner, then answers once the approved plan is handed back."""
    app, _model = orchestrator_app(
        plan_call(QUESTION),
        types.Part.from_text(text=ANSWER),
        executor=ScriptedExecutor(),
        tool_surface=ScriptedSurface(),
        engine=engine,
    )
    return app


async def _stream(client: httpx2.AsyncClient, path: str, body: dict[str, Any]) -> list[dict]:
    async with client.stream("POST", path, json=body, headers=AUTH) as response:
        assert response.status_code == 200
        return [
            json.loads(line[len("data: ") :])
            async for line in response.aiter_lines()
            if line.startswith("data: ")
        ]


async def _approve_a_planned_turn(engine: AsyncEngine) -> list[dict]:
    """Ask the planned studio `QUESTION`, approve the plan it proposes, and
    return the frames of the approval's stream."""
    app = _planned_app(engine)
    base = {"schema_version": 1, "conversation_id": CONVERSATION}
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://studio.test"
    ) as client:
        first = await _stream(
            client,
            "/api/studio/turn",
            {
                **base,
                "kind": "turn.request",
                "message": QUESTION,
            },
        )
        plan = first[-1]
        assert plan["event"] == "plan.proposed"
        second = await _stream(
            client,
            "/api/studio/interrupt",
            {
                **base,
                "kind": "interrupt.response",
                "interrupt_id": plan["interrupt_id"],
                "decision": "approve",
            },
        )
    assert second[-1]["event"] == "done"
    return second


async def test_a_paused_and_resumed_turn_exports_a_complete_parent_chain(
    fake_langfuse: Any, pg_engine: AsyncEngine
) -> None:
    """Both segments of a turn run the orchestrator's agent, each under a
    `turn` span of its own on the turn's one trace. Each segment's
    `invoke_agent` span is a child of its segment's `turn`, and every span on
    the wire has its parent on the wire too, except the two `turn` spans,
    whose parent is the synthetic one the SDK creates for a seeded trace id.
    Every exported span is the SDK's own or carries `gen_ai.*` attributes.
    Each `turn` carries the question as its input, and the resumed one the
    answer as its output."""
    init_telemetry(service_name="harness")
    second = await _approve_a_planned_turn(pg_engine)

    scoped = fake_langfuse.scoped_spans()
    spans = [span for _, span in scoped]
    paused, resumed = sorted(
        (span for span in spans if span.name == "turn"), key=lambda span: span.start_time_unix_nano
    )
    trace_id = Langfuse.create_trace_id(seed=second[-1]["trace_id"])
    assert {span.trace_id.hex() for span in spans} == {trace_id}
    runs = sorted(
        (span for span in spans if span.name.startswith("invoke_agent")),
        key=lambda span: span.start_time_unix_nano,
    )
    assert [span.parent_span_id for span in runs] == [paused.span_id, resumed.span_id]
    assert sorted(
        _spans_with_an_unexported_parent(spans), key=lambda span: span.start_time_unix_nano
    ) == [paused, resumed]
    assert _attribute(paused, "langfuse.observation.input") == QUESTION
    assert _attribute(resumed, "langfuse.observation.input") == QUESTION
    assert _attribute(resumed, "langfuse.observation.output")
    for scope, span in scoped:
        assert scope == "langfuse-sdk" or any(
            key.startswith("gen_ai") for key in _attribute_keys(span)
        ), (scope, span.name)


async def test_a_model_call_exports_its_prompt_and_response(
    fake_langfuse: Any, pg_engine: AsyncEngine
) -> None:
    """With no content setting in the environment, the `chat` span
    of the orchestrator's first call carries the user's question in its input
    messages and the `plan` call it made in its output messages, and the span
    of its last call carries the answer, so Langfuse shows each generation's
    prompt and response."""
    init_telemetry(service_name="harness")
    await _approve_a_planned_turn(pg_engine)

    calls = sorted(
        (span for span in fake_langfuse.spans() if span.name.startswith("chat")),
        key=lambda span: span.start_time_unix_nano,
    )
    assert len(calls) >= 2
    assert {"type": "text", "content": QUESTION} in _message_parts(
        calls[0], "gen_ai.input.messages"
    )
    assert [part["name"] for part in _message_parts(calls[0], "gen_ai.output.messages")] == ["plan"]
    assert {"type": "text", "content": ANSWER} in _message_parts(
        calls[-1], "gen_ai.output.messages"
    )


# --- a client that leaves once the turn paused -------------------------------------


async def _post_and_leave_at_the_pause(app: Any, body: dict[str, Any], *, how: str) -> str:
    """POST a turn straight to the ASGI app and drop the response once
    `plan.proposed` has been sent; returns that frame's interrupt id.
    `how="cancel"` cancels the response task (ASGI spec 2.4 and later, where
    Starlette does nothing else), `how="disconnect"` delivers
    `http.disconnect` (earlier spec versions)."""
    raw = json.dumps(body).encode()
    sent = False
    disconnect = asyncio.Event()
    at_pause = asyncio.Event()
    chunks: list[str] = []

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": raw, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] != "http.response.body":
            return
        chunk = message.get("body", b"").decode()
        chunks.append(chunk)
        if '"plan.proposed"' in chunk:
            at_pause.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4" if how == "cancel" else "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/studio/turn",
        "raw_path": b"/api/studio/turn",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"authorization", b"Bearer test"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(raw)).encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("studio.test", 80),
    }
    task = asyncio.create_task(app(scope, receive, send))
    await asyncio.wait_for(at_pause.wait(), timeout=30)
    if how == "cancel":
        task.cancel()
    else:
        disconnect.set()
    done, _ = await asyncio.wait({task}, timeout=15)
    assert done, "the response task did not end"
    with suppress(asyncio.CancelledError):
        await task
    (plan,) = [
        json.loads(chunk[len("data: ") :])
        for chunk in chunks
        if chunk.startswith("data: ") and '"plan.proposed"' in chunk
    ]
    return plan["interrupt_id"]


@pytest.mark.parametrize("how", ["cancel", "disconnect"])
async def test_a_client_that_leaves_at_the_approval_pause_keeps_the_turn_on_one_trace(
    fake_langfuse: Any, how: str, pg_engine: AsyncEngine
) -> None:
    """The client goes away right after `plan.proposed`, the stream's last
    frame. The turn stays open and unrecorded, so the `/interrupt` that
    approves the plan resumes it on the same trace: the paused segment's
    `turn` span and the resumed one's, each ended, the execution spans under
    the second, and no cancellation on the turn's record."""
    init_telemetry(service_name="harness")
    app = _planned_app(pg_engine)
    studio: StudioRunner = app.state.studio
    body = {
        "schema_version": 1,
        "conversation_id": CONVERSATION,
        "kind": "turn.request",
        "message": "comprehensive overview of life expectancy in Kenya",
    }

    interrupt_id = await _post_and_leave_at_the_pause(app, body, how=how)
    principal = "u_" + hashlib.sha256(b"test").hexdigest()[:16]
    conversation = studio.store.current(principal, CONVERSATION)
    assert conversation is not None
    assert not conversation.lock.locked()
    assert conversation.awaiting_answer

    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://studio.test"
    ) as client:
        resumed = await _stream(
            client,
            "/api/studio/interrupt",
            {
                "schema_version": 1,
                "conversation_id": CONVERSATION,
                "kind": "interrupt.response",
                "interrupt_id": interrupt_id,
                "decision": "approve",
            },
        )
    assert resumed[-1]["event"] == "done"

    spans = fake_langfuse.spans()
    turns = [span for span in spans if span.name == "turn"]
    assert len(turns) == 2
    assert all(turn.end_time_unix_nano > 0 for turn in turns)
    trace_id = Langfuse.create_trace_id(seed=resumed[-1]["trace_id"])
    assert {span.trace_id.hex() for span in spans} == {trace_id}
    # The orchestrator's run of each segment, the resumed one after the plan
    # executed among them.
    assert len([span for span in spans if span.name.startswith("invoke_agent")]) == 2
    (record,) = conversation.turns
    assert "error" not in record


class _SlowWork:
    """Work that makes a model call, sends one frame, then waits on a slow
    tool call."""

    def __init__(self, conversation: ConversationState) -> None:
        self._conversation = conversation
        self.running = asyncio.Event()

    def begin(self) -> None:
        return None

    async def drive(self, emit: Emit) -> Outcome:
        with trace.get_tracer("test.segment").start_as_current_span(MODEL_CALL) as span:
            span.set_attribute("gen_ai.operation.name", "generate_content")
        emit("one")
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def frames(self, event: object) -> list[BaseModel]:
        self.running.set()
        return [
            MessageDelta(event="message.delta", seq=self._conversation.next_seq(), text=str(event))
        ]

    async def end(self, result: object) -> tuple[Outcome, list[BaseModel]]:
        raise AssertionError("the drive never returns")

    def fatal(self, error: BaseException) -> ErrorFrame:
        raise AssertionError("the work never fails") from error

    def close(self) -> None:
        return None


class _SavingStore:
    """The owner's writes, each landing."""

    def snapshot(self, principal: str, conversation_id: str) -> Snapshot:
        return {}

    async def checkpoint(
        self, principal: str, conversation_id: str, segment_id: str, snapshot: Snapshot
    ) -> None:
        return None

    async def close_segment(
        self, principal: str, conversation_id: str, segment_id: str
    ) -> SegmentClose:
        return SegmentClose.SAVED

    def evict(self, principal: str, conversation_id: str) -> None:
        return None


async def test_a_client_that_leaves_while_the_turn_runs_ends_the_turn_span_as_left(
    fake_langfuse: Any,
) -> None:
    """A turn that has not paused stays open past the client's leaving, for
    the next request to carry on, and the segment's `turn` span ends once,
    at level `WARNING` as `left`, with the model call under it and
    `dfo.client_gone` set; nothing is recorded."""
    init_telemetry(service_name="harness")
    conversation = ConversationState(conversation_id=CONVERSATION)
    conversation.begin_turn()
    conversation.turn_user_message = "q"
    work = _SlowWork(conversation)
    channel: Channel = asyncio.Queue()
    gone = asyncio.Event()
    owner = asyncio.create_task(
        own_segment(
            work,
            store=_SavingStore(),
            conversation=conversation,
            principal="u_a",
            conversation_id=CONVERSATION,
            segment_id="seg_1",
            channel=channel,
            gone=gone,
        )
    )
    await asyncio.wait_for(work.running.wait(), timeout=15)
    gone.set()
    await asyncio.wait_for(owner, timeout=15)
    await _until(lambda: any(span.name == "turn" for span in fake_langfuse.spans()))

    spans = fake_langfuse.spans()
    (turn,) = [span for span in spans if span.name == "turn"]
    (call,) = [span for span in spans if span.name == MODEL_CALL]
    assert turn.end_time_unix_nano > 0
    assert call.parent_span_id == turn.span_id
    assert (
        _attribute(turn, "langfuse.observation.level"),
        _attribute(turn, "langfuse.observation.status_message"),
        _attribute(turn, "dfo.client_gone"),
    ) == ("WARNING", "left", "true")
    assert not conversation.turn_recorded
    assert conversation.turns == []
