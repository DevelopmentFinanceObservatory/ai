"""What a Studio turn puts on the wire to Langfuse.

The real Langfuse SDK, tracer provider, span filter and Studio routes run; only
the Langfuse server is fake (`fake_langfuse`). These tests cover the parts an
in-memory span exporter cannot see: which spans the export filter keeps, how a
response that is torn down mid-stream ends the turn's telemetry, the parent
chain of a turn that pauses for approval and resumes, and that no span outlives
the request that opened it (`Quiescence.assert_quiescent`), across instances
too.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import inspect
import json
import logging
import sys
import time
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import aclosing, contextmanager, suppress
from functools import partial
from types import AsyncGeneratorType, SimpleNamespace
from typing import Any

import common.telemetry as telemetry_module
import harness.studio as studio_module
import httpx2
import pytest
from common.logging import JSON_FIELDS
from common.telemetry import init_telemetry
from compiler.surface import record_tool_result
from data_mcp.mint_admission import BACKGROUND_OUTCOME_KEY
from google.adk.sessions import DatabaseSessionService
from google.genai import types
from harness.conversation import current_conversation_caller
from harness.early_mint import EarlyMints
from harness.mcp_client import McpExecutor
from harness.mcp_deadlines import McpTimeoutPolicy
from harness.scripted import ScriptedExecutor, ScriptedSurface
from harness.settings import Settings
from harness.studio import SSE_MEDIA_TYPE, StudioRunner, _sse_stream
from harness.studio_state import ConversationState, TurnUsage
from harness.studio_store import ConversationStale, StudioStore
from harness.wire import MessageDelta
from langfuse import Langfuse, get_client
from opentelemetry import trace
from opentelemetry.proto.trace.v1.trace_pb2 import Span
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.responses import StreamingResponse

from ..orchestrator_fakes import (
    AGENT_NAME,
    NO_ADK,
    no_turn_specs,
    orchestrator_app,
    plan_call,
    session_turn_specs,
)
from .conftest import Quiescence

CONVERSATION = "conv_trace_export_01"
AUTH = {"Authorization": "Bearer test"}
QUESTION = "comprehensive overview of life expectancy in Kenya"
ANSWER = "Life expectancy in Kenya rose."
# The name of the stand-in for an ADK model call span in the tests that drive
# `_sse_stream` with a scripted segment.
MODEL_CALL = "generate_content gemini-x"


def _attribute(span: Span, key: str) -> object:
    for attribute in span.attributes:
        if attribute.key == key:
            value = getattr(attribute.value, attribute.value.WhichOneof("value"))
            # The SDK JSON-encodes `langfuse.*.metadata.*` values.
            return json.loads(value) if ".metadata." in key else value
    return None


def _attribute_keys(span: Span) -> set[str]:
    return {attribute.key for attribute in span.attributes}


def _message_parts(span: Span, key: str) -> list[dict[str, Any]]:
    """The parts of every message in the GenAI messages attribute `key`."""
    messages = json.loads(str(_attribute(span, key)))
    return [part for message in messages for part in message["parts"]]


def _closes(spans: list[Span]) -> list[Span]:
    """The `turn` observations of `spans` that close a turn, not pause it."""
    return [
        span
        for span in spans
        if span.name == "turn"
        and _attribute(span, "langfuse.observation.metadata.dfo.turn_status") is None
    ]


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


# --- a response torn down while the stream sits at a frame ------------------------


@contextmanager
def _abandoned_streams() -> Iterator[Callable[[], list[AsyncGeneratorType]]]:
    """Yield a function returning the response generators the event loop has
    taken over for closing: a generator dropped while suspended is handed to
    the loop, which closes it later on a task of its own. The loop takes over
    only the response's outermost generator: `_sse_stream` when a test serves
    it bare through a plain `StreamingResponse`. A Studio route's response
    (`harness.studio._ClaimedStream`) closes its body itself as it ends, so
    the loop takes over none of it. Only generators first iterated inside the
    block are seen."""
    hooks = sys.get_asyncgen_hooks()
    handed_over: list[AsyncGeneratorType] = []

    def finalizer(generator: AsyncGeneratorType) -> None:
        handed_over.append(generator)
        assert hooks.finalizer is not None
        hooks.finalizer(generator)

    sys.set_asyncgen_hooks(firstiter=hooks.firstiter, finalizer=finalizer)
    try:
        responses = (_sse_stream.__code__, StudioRunner._settling.__code__)
        yield lambda: [g for g in handed_over if g.ag_code in responses]
    finally:
        sys.set_asyncgen_hooks(firstiter=hooks.firstiter, finalizer=hooks.finalizer)


@contextmanager
def _route_bodies() -> Iterator[Callable[[], list[AsyncGeneratorType]]]:
    """Yield a function returning the Studio route bodies
    (`StudioRunner._settling`) first iterated inside the block."""
    hooks = sys.get_asyncgen_hooks()
    started: list[AsyncGeneratorType] = []

    def firstiter(generator: AsyncGeneratorType) -> None:
        if generator.ag_code is StudioRunner._settling.__code__:
            started.append(generator)
        if hooks.firstiter is not None:
            hooks.firstiter(generator)

    sys.set_asyncgen_hooks(firstiter=firstiter, finalizer=hooks.finalizer)
    try:
        yield lambda: list(started)
    finally:
        sys.set_asyncgen_hooks(firstiter=hooks.firstiter, finalizer=hooks.finalizer)


async def _abandon_a_response_at_its_second_frame(
    conversation: ConversationState, *, how: str
) -> None:
    """Serve a stream whose `send` never completes for the second frame, then
    tear the response down: `how="cancel"` cancels the response task (ASGI
    spec 2.4 and later, where Starlette does nothing else), `how="disconnect"`
    delivers `http.disconnect` (earlier spec versions, where Starlette cancels
    the streaming task itself). The stream generator is left suspended at its
    second `yield`; nothing here closes it, so the event loop does."""

    async def segment() -> AsyncGenerator[MessageDelta]:
        try:
            with trace.get_tracer("test.segment").start_as_current_span(MODEL_CALL) as span:
                span.set_attribute("gen_ai.operation.name", "generate_content")
            yield MessageDelta(event="message.delta", seq=0, text="one")
            conversation.close_turn(final_text="one")
            yield MessageDelta(event="message.delta", seq=1, text="two")
        finally:
            # The segment's end, as `StudioRunner._locked` ends it.
            conversation.end_segment_span("saved")

    response = StreamingResponse(
        _sse_stream(segment(), conversation=conversation), media_type=SSE_MEDIA_TYPE
    )
    disconnect = asyncio.Event()
    second_frame_sent = asyncio.Event()
    never = asyncio.Event()
    chunks = 0

    async def receive() -> dict[str, Any]:
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        nonlocal chunks
        if message["type"] == "http.response.body" and message.get("more_body"):
            chunks += 1
            if chunks == 2:
                second_frame_sent.set()
                await never.wait()  # the client's socket stops draining

    spec_version = "2.4" if how == "cancel" else "2.3"
    task = asyncio.create_task(
        response({"type": "http", "asgi": {"spec_version": spec_version}}, receive, send)
    )
    await asyncio.wait_for(second_frame_sent.wait(), timeout=15)
    if how == "cancel":
        task.cancel()
    else:
        disconnect.set()
    done, _ = await asyncio.wait({task}, timeout=15)
    assert done, "the response task did not end"


@pytest.mark.parametrize("how", ["cancel", "disconnect"])
async def test_a_response_torn_down_at_a_frame_still_exports_the_turn_once_and_cleanly(
    quiescence: Quiescence, caplog: pytest.LogCaptureFixture, how: str
) -> None:
    """The response task ends while the stream generator sits at a `yield`, so
    the event loop closes it later in a context of its own. The segment span,
    ended by the segment, is exported by the flush in the generator's
    `finally`, and leaving the telemetry scope there does not make
    OpenTelemetry log a failed detach. The close, which no store saves, writes
    no `turn` outcome."""
    fake_langfuse = quiescence.fake
    conversation = ConversationState(conversation_id=CONVERSATION)
    conversation.begin_turn()
    conversation.turn_user_message = "q"

    # The capture handler runs at INFO (`quiescence`), below the ERROR a
    # failed detach logs at.
    with _abandoned_streams() as abandoned:
        await _abandon_a_response_at_its_second_frame(conversation, how=how)
        # The segment reaches the wire from the flush inside the generator's
        # `finally`, before the scope exit that would log a failed detach, so
        # the log is read once the generator has finished closing.
        await _until(
            lambda: any(
                inspect.getasyncgenstate(generator) == inspect.AGEN_CLOSED
                for generator in abandoned()
            )
        )
        await _until(lambda: any(span.name == "turn.segment" for span in fake_langfuse.spans()))

    spans = fake_langfuse.spans()
    (segment,) = [span for span in spans if span.name == "turn.segment"]
    (call,) = [span for span in spans if span.name == MODEL_CALL]
    assert segment.end_time_unix_nano > 0
    assert call.parent_span_id == segment.span_id
    assert conversation.cancel_event is None
    assert [r for r in caplog.records if r.name == "opentelemetry.context"] == []
    assert quiescence.outcomes() == []
    quiescence.assert_quiescent()


# --- a turn that pauses for approval and resumes ---------------------------------


def _planned_app(
    engine: AsyncEngine, surface: ScriptedSurface | None = None, *, open_auth: bool = False
) -> Any:
    """The studio with an agent that hands the question to the scripted
    planner, then answers once the approved plan is handed back."""
    app, _model = orchestrator_app(
        plan_call(QUESTION),
        types.Part.from_text(text=ANSWER),
        executor=ScriptedExecutor(),
        tool_surface=surface or ScriptedSurface(),
        engine=engine,
        settings=Settings(studio_api_token=None if open_auth else "test"),
    )
    return app


async def _stream(
    client: httpx2.AsyncClient,
    path: str,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> list[dict]:
    async with client.stream(
        "POST", path, json=body, headers={**AUTH, **(headers or {})}
    ) as response:
        assert response.status_code == 200
        return [
            json.loads(line[len("data: ") :])
            async for line in response.aiter_lines()
            if line.startswith("data: ")
        ]


async def _approve_a_planned_turn(
    engine: AsyncEngine,
    *,
    headers: dict[str, str] | None = None,
    interrupt_headers: dict[str, str] | None = None,
    surface: ScriptedSurface | None = None,
    open_auth: bool = False,
    after_each: Callable[[], None] = lambda: None,
) -> list[dict]:
    """Ask the planned studio `QUESTION` with `headers` beside the bearer
    token, approve the plan it proposes with `interrupt_headers`, and return
    the frames of the approval's stream. `after_each` runs once each
    response has ended."""
    app = _planned_app(engine, surface, open_auth=open_auth)
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
            headers,
        )
        after_each()
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
            interrupt_headers,
        )
        after_each()
    assert second[-1]["event"] == "done"
    return second


ACCOUNT = "a" * 64
VISITOR = "ip:" + "b" * 32
# The principal the studio derives from the bearer token `test`.
PRINCIPAL = "u_" + hashlib.sha256(b"test").hexdigest()[:16]


class _ProbedSurface(ScriptedSurface):
    """`ScriptedSurface` that notes the caller the harness's MCP clients
    would name on each call (`current_conversation_caller`) and the spec of
    each `get_data` call, and records each result as `McpSurface` does."""

    def __init__(self) -> None:
        super().__init__()
        self.callers: list[str | None] = []
        self.fetched: list[str] = []

    async def call_tool(self, *, tool: Any, arguments: Any, on_progress: Any = None) -> Any:
        self.callers.append(current_conversation_caller())
        result = await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)
        if tool.name == "get_data":
            self.fetched.append(str(arguments["spec_id"]))
        record_tool_result(
            tool.name, is_error=result.is_error, structured_content=result.structured_content
        )
        return result


def _turn_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    """The JSON fields of every `studio.turn` line a close logged."""
    return [
        fields
        for record in caplog.records
        if isinstance(fields := getattr(record, JSON_FIELDS, None), dict)
        and fields.get("event") == "studio.turn"
        and fields.get("status") == "closed"
    ]


def _tags(span: Span) -> list[str] | None:
    for attribute in span.attributes:
        if attribute.key == "langfuse.trace.tags":
            return [value.string_value for value in attribute.value.array_value.values]
    return None


@pytest.mark.parametrize(
    ("identity", "user_id", "account"),
    [
        ({"x-dfo-account": ACCOUNT, "x-dfo-caller": VISITOR}, ACCOUNT, ACCOUNT),
        ({"x-dfo-caller": VISITOR}, VISITOR, None),
        ({"x-dfo-account": "not-a-digest", "x-dfo-caller": VISITOR}, VISITOR, None),
        ({"x-dfo-account": ACCOUNT.upper(), "x-dfo-caller": VISITOR}, ACCOUNT, ACCOUNT),
        ({"x-dfo-caller": VISITOR.upper()}, VISITOR, None),
    ],
    ids=["signed-in", "signed-out", "malformed-account", "upper-case-account", "upper-case-caller"],
)
async def test_a_studio_turn_names_who_asked_it_on_its_trace_and_its_log_line(
    fake_langfuse: Any,
    caplog: pytest.LogCaptureFixture,
    pg_engine: AsyncEngine,
    identity: dict[str, str],
    user_id: str,
    account: str | None,
) -> None:
    """The account the website names, else the visitor, is the Langfuse user
    of the turn, kept through the approval's `/interrupt`, which sends no
    identity headers. The turn's one `studio.turn` line carries the same
    account and visitor, the caller the turn's MCP calls name, the wire trace
    id, and the spec each `get_data` of the plan fetched."""
    init_telemetry(service_name="harness")
    surface = _ProbedSurface()
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        second = await _approve_a_planned_turn(pg_engine, headers=identity, surface=surface)

    (turn,) = _closes(fake_langfuse.spans())
    assert _attribute(turn, "user.id") == user_id
    assert _tags(turn) == ["studio"]
    (line,) = _turn_lines(caplog)
    assert line["account"] == account
    assert line["visitor"] == VISITOR
    assert line["trace_id"] == second[-1]["trace_id"]
    assert line["error"] is None
    assert surface.callers
    assert set(surface.callers) == {line["caller"]}
    assert line["caller"].startswith("conversation:")
    assert surface.fetched
    assert line["fetched_specs"] == list(dict.fromkeys(surface.fetched))


async def test_an_open_studio_names_nobody_whatever_the_headers_say(
    fake_langfuse: Any, caplog: pytest.LogCaptureFixture, pg_engine: AsyncEngine
) -> None:
    """With `STUDIO_API_TOKEN` unset no proxy vouches for the identity
    headers, so the turn has no account, no visitor and no Langfuse user."""
    init_telemetry(service_name="harness")
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        await _approve_a_planned_turn(
            pg_engine, headers={"x-dfo-account": ACCOUNT, "x-dfo-caller": VISITOR}, open_auth=True
        )

    (turn,) = _closes(fake_langfuse.spans())
    assert _attribute(turn, "user.id") is None
    (line,) = _turn_lines(caplog)
    assert (line["account"], line["visitor"]) == (None, None)


async def test_a_resumed_turn_that_recorded_no_usage_takes_it_from_the_interrupt(
    fake_langfuse: Any,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    pg_engine: AsyncEngine,
) -> None:
    """A paused turn whose stored usage names no one, as one paused before
    `TurnUsage` was stored, takes the `/interrupt` request's account and visitor,
    the caller its MCP calls name, and the surface, and still hears the spec
    each `get_data` of the resumed plan fetched."""
    monkeypatch.setattr(studio_module, "TurnUsage", lambda **_: TurnUsage())
    init_telemetry(service_name="harness")
    surface = _ProbedSurface()
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        await _approve_a_planned_turn(
            pg_engine,
            interrupt_headers={"x-dfo-account": ACCOUNT, "x-dfo-caller": VISITOR},
            surface=surface,
        )

    (line,) = _turn_lines(caplog)
    assert line["account"] == ACCOUNT
    assert line["visitor"] == VISITOR
    assert line["surface"] == "studio"
    assert surface.callers
    assert {line["caller"]} == set(surface.callers)
    assert line["caller"].startswith("conversation:")
    assert surface.fetched
    assert line["fetched_specs"] == list(dict.fromkeys(surface.fetched))


async def test_a_paused_and_resumed_turn_exports_a_complete_parent_chain(
    quiescence: Quiescence, pg_engine: AsyncEngine
) -> None:
    """Both segments of a turn run ADK's real workflow, each inside a
    `turn.segment` span of its own request, and no span is open once either
    response has ended, the pause included. Each segment's `invoke_workflow`
    span is a child of its own `turn.segment`, and every span on the wire has
    its parent on the wire too, except the two segments, the pause's
    `turn.pause` and the close's `turn`, each a root under the synthetic
    parent the SDK creates for a seeded trace id. The pause exports no `turn`,
    so only a close is a `turn` to a reader that sees names alone. Every
    exported span is the SDK's own or carries `gen_ai.*` attributes. The
    close's `turn` carries the question and the answer as its input and
    output, and the segments carry neither."""
    outcomes_after_each: list[list[str]] = []

    def after_each() -> None:
        quiescence.assert_quiescent()
        outcomes_after_each.append(
            sorted(
                span.name for span in quiescence.fake.spans() if span.name in {"turn", "turn.pause"}
            )
        )

    second = await _approve_a_planned_turn(pg_engine, after_each=after_each)

    assert outcomes_after_each == [["turn.pause"], ["turn", "turn.pause"]]

    scoped = quiescence.fake.scoped_spans()
    spans = [span for _, span in scoped]
    (turn,) = _closes(spans)
    segments = [span for span in spans if span.name == "turn.segment"]
    assert len(segments) == 2
    trace_id = Langfuse.create_trace_id(seed=second[-1]["trace_id"])
    assert {span.trace_id.hex() for span in spans} == {trace_id}
    workflows = [span for span in spans if span.name.startswith("invoke_workflow")]
    assert len(workflows) == 2
    assert sorted(span.parent_span_id for span in workflows) == sorted(
        span.span_id for span in segments
    )
    roots = _spans_with_an_unexported_parent(spans)
    assert sorted(span.name for span in roots) == [
        "turn",
        "turn.pause",
        "turn.segment",
        "turn.segment",
    ]
    assert all(_attribute(span, "langfuse.internal.is_app_root") is True for span in roots)
    assert _attribute(turn, "langfuse.observation.input") == QUESTION
    assert _attribute(turn, "langfuse.observation.output")
    for segment in segments:
        assert _attribute(segment, "langfuse.observation.input") is None
        assert _attribute(segment, "langfuse.observation.output") is None
    for scope, span in scoped:
        assert scope == "langfuse-sdk" or any(
            key.startswith("gen_ai") for key in _attribute_keys(span)
        ), (scope, span.name)


async def test_a_model_call_exports_its_prompt_and_response(
    fake_langfuse: Any, pg_engine: AsyncEngine
) -> None:
    """With no content setting in the environment, the `generate_content` span
    of the orchestrator's first call carries the user's question in its input
    messages and the `plan` call it made in its output messages, and the span
    of its last call carries the answer, so Langfuse shows each generation's
    prompt and response."""
    init_telemetry(service_name="harness")
    await _approve_a_planned_turn(pg_engine)

    calls = sorted(
        (span for span in fake_langfuse.spans() if span.name.startswith("generate_content")),
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


# --- a client that leaves while the turn waits for the next frame -----------------


async def _post_and_leave_at_the_pause(
    app: Any,
    body: dict[str, Any],
    *,
    how: str,
    at_pause: asyncio.Event,
) -> str:
    """POST a turn straight to the ASGI app and drop the response once the
    stream is parked after `plan.proposed`; returns that frame's interrupt id.
    `how="cancel"` cancels the response task while the stream awaits its next
    frame (`asyncio.CancelledError` at the wait); `how="close"` lets the stream
    emit a heartbeat that never finishes sending, then cancels the response
    task while the generator sits at the heartbeat `yield`, so the response
    (`harness.studio._ClaimedStream`) closes it there as it ends
    (`GeneratorExit`)."""
    raw = json.dumps(body).encode()
    sent = False
    disconnect = asyncio.Event()
    heartbeat_blocked = asyncio.Event()
    never = asyncio.Event()
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
        if how == "close" and at_pause.is_set() and chunk.startswith(": ka"):
            heartbeat_blocked.set()
            await never.wait()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
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
    if how == "close":
        await asyncio.wait_for(heartbeat_blocked.wait(), timeout=15)
    task.cancel()
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


@pytest.mark.parametrize("how", ["cancel", "close"])
async def test_a_client_that_leaves_at_the_approval_pause_keeps_the_turn_on_one_trace(
    fake_langfuse: Any, monkeypatch: pytest.MonkeyPatch, how: str, pg_engine: AsyncEngine
) -> None:
    """The client goes away right after `plan.proposed`, while the stream
    awaits the next frame. The turn stays open and unrecorded, so the
    `/interrupt` that approves the plan resumes it on the same trace: the
    paused segment's `turn.segment` span and the resumed one's, each ended,
    one closing `turn` outcome, the execution spans on that trace, and no
    cancellation on the turn's record."""
    init_telemetry(service_name="harness")
    app = _planned_app(pg_engine)
    studio: StudioRunner = app.state.studio
    at_pause = asyncio.Event()
    paused_once = False
    frames_of_segment = studio._frames

    async def parked_after_the_plan(**kwargs: Any) -> AsyncGenerator[Any]:
        nonlocal paused_once
        async with aclosing(frames_of_segment(**kwargs)) as frames:
            async for frame in frames:
                yield frame
                if frame.event == "plan.proposed" and not paused_once:
                    paused_once = True
                    at_pause.set()
                    await asyncio.Event().wait()  # the run has not drained yet

    monkeypatch.setattr(studio, "_frames", parked_after_the_plan)
    monkeypatch.setattr(studio_module, "_sse_stream", partial(_sse_stream, heartbeat_seconds=0.02))
    body = {
        "schema_version": 1,
        "conversation_id": CONVERSATION,
        "kind": "turn.request",
        "message": "comprehensive overview of life expectancy in Kenya",
    }

    with _abandoned_streams() as abandoned, _route_bodies() as bodies:
        interrupt_id = await _post_and_leave_at_the_pause(app, body, how=how, at_pause=at_pause)
        conversation = studio.store.current(PRINCIPAL, CONVERSATION)
        assert conversation is not None
        await _until(lambda: not conversation.lock.locked() and conversation.cancel_event is None)
        if how == "close":
            # The response closed the body as it ended, in the request task,
            # so the event loop was handed none of it to close.
            (route_body,) = bodies()
            assert inspect.getasyncgenstate(route_body) == inspect.AGEN_CLOSED
            assert abandoned() == []
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
    segments = [span for span in spans if span.name == "turn.segment"]
    assert len(segments) == 2
    assert all(segment.end_time_unix_nano > 0 for segment in segments)
    (turn,) = _closes(spans)
    assert turn.end_time_unix_nano > 0
    trace_id = Langfuse.create_trace_id(seed=resumed[-1]["trace_id"])
    assert {span.trace_id.hex() for span in spans} == {trace_id}
    assert any(span.name.startswith("invoke_node execute") for span in spans)
    (record,) = conversation.turns
    assert "error" not in record


async def test_a_client_that_leaves_while_the_turn_runs_writes_its_outcome_as_cancelled(
    quiescence: Quiescence, caplog: pytest.LogCaptureFixture, pg_engine: AsyncEngine
) -> None:
    """A turn that has not paused is closed by the disconnect: the segment
    ends once, parenting the segment's own spans, the release save that
    commits the close writes one `turn` outcome at `WARNING` naming the
    cancellation and one `studio.turn` line, and the record carries it. The
    stream records that the client left on its `_StreamEnding`, and the
    segment closes the turn, releases it and ends its span as
    `StudioRunner._locked` does."""
    fake_langfuse = quiescence.fake
    store = StudioStore(pg_engine, adk=NO_ADK, turn_specs=no_turn_specs)
    conversation = await store.create("principal", CONVERSATION)
    conversation.begin_turn()
    conversation.turn_user_message = "q"
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    running = asyncio.Event()
    ending = studio_module._StreamEnding()

    async def segment() -> AsyncGenerator[MessageDelta]:
        try:
            with trace.get_tracer("test.segment").start_as_current_span(MODEL_CALL) as span:
                span.set_attribute("gen_ai.operation.name", "generate_content")
            yield MessageDelta(event="message.delta", seq=0, text="one")
            running.set()
            await asyncio.Event().wait()  # a slow tool call
            yield MessageDelta(event="message.delta", seq=1, text="two")
        finally:
            # The segment's end, as `StudioRunner._locked` ends it.
            if ending.client_gone and not conversation.awaiting_answer:
                conversation.close_turn(final_text="", error="cancelled")
            try:
                await store.release("principal", CONVERSATION, segment_id)
            finally:
                conversation.end_segment_span("disconnected")

    async def consume() -> None:
        async for _ in _sse_stream(
            segment(), conversation=conversation, heartbeat_seconds=15, ending=ending
        ):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(running.wait(), timeout=15)
    await asyncio.sleep(0.05)  # the stream is parked at its wait for the next frame
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _until(lambda: any(span.name == "turn.segment" for span in fake_langfuse.spans()))

    spans = fake_langfuse.spans()
    (segment_span,) = [span for span in spans if span.name == "turn.segment"]
    (turn,) = _closes(spans)
    (call,) = [span for span in spans if span.name == MODEL_CALL]
    assert segment_span.end_time_unix_nano > 0
    assert call.parent_span_id == segment_span.span_id
    assert _attribute(turn, "langfuse.observation.level") == "WARNING"
    assert _attribute(turn, "langfuse.observation.status_message") == "cancelled"
    assert conversation.turn_recorded
    (record,) = conversation.turns
    assert (record["user"], record["assistant"], record["error"]) == ("q", "", "cancelled")
    (line,) = _turn_lines(caplog)
    assert (line["trace_id"], line["error"]) == (conversation.trace_id, "cancelled")
    await quiescence.assert_store_agrees(
        pg_engine, principal="principal", conversation_id=CONVERSATION
    )


# --- no span outlives its request -------------------------------------------------


def _turn_body(message: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "conversation_id": CONVERSATION,
        "kind": "turn.request",
        "message": message,
    }


def _approve_body(interrupt_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "conversation_id": CONVERSATION,
        "kind": "interrupt.response",
        "interrupt_id": interrupt_id,
        "decision": "approve",
    }


def _client(app: Any) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://studio.test"
    )


def _instance(engine: AsyncEngine, *script: types.Part) -> Any:
    """One harness instance on the shared database `engine`, with a store and
    an ADK session service of its own on it, its agent answering from
    `script`."""
    session_service = DatabaseSessionService(db_engine=engine)
    app, _model = orchestrator_app(
        *script,
        executor=ScriptedExecutor(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
        store=StudioStore(
            engine,
            adk=NO_ADK,
            turn_specs=session_turn_specs(session_service, app_name=AGENT_NAME),
        ),
        settings=Settings(studio_api_token="test"),
    )
    return app


async def test_a_turn_whose_claim_save_is_stale_writes_no_outcome_and_holds_no_span(
    quiescence: Quiescence, pg_engine: AsyncEngine
) -> None:
    """A `/turn` on a paused conversation whose claim save meets another
    instance's write answers 409 and leaves nothing behind: no open span, no
    `turn` outcome for the paused turn it would have abandoned, and no line."""
    app = _planned_app(pg_engine)
    studio: StudioRunner = app.state.studio
    async with _client(app) as client:
        paused = await _stream(client, "/api/studio/turn", _turn_body(QUESTION))
        assert paused[-1]["event"] == "plan.proposed"
        quiescence.assert_quiescent()
        real_save = studio.store.save

        async def stale_save(*args: Any, **kwargs: Any) -> bool:
            studio.store.save = real_save  # type: ignore[method-assign]
            raise ConversationStale(CONVERSATION)

        studio.store.save = stale_save  # type: ignore[method-assign]
        response = await client.post(
            "/api/studio/turn", json=_turn_body("aid to kenya"), headers=AUTH
        )
    assert response.status_code == 409
    get_client().flush()  # nothing the refused request queued may reach the wire
    assert quiescence.outcomes() == []
    quiescence.assert_quiescent()


async def test_a_turn_resumed_on_another_instance_is_one_trace_with_one_outcome(
    quiescence: Quiescence, pg_engine: AsyncEngine
) -> None:
    """Two instances share one database. A takes a turn to the approval pause,
    B answers the `/interrupt` through `done`, and A takes the next turn. Each
    response leaves no span open on the process, and each instance's spans
    end in its own request. The first turn's trace holds one `turn` outcome
    carrying the question and the answer and one `turn.segment` per instance,
    and every exported span has its parent on the wire or is a root."""
    app_a = _instance(pg_engine, plan_call(QUESTION), types.Part.from_text(text=ANSWER))
    app_b = _instance(pg_engine)

    async with _client(app_a) as client_a:
        paused = await _stream(client_a, "/api/studio/turn", _turn_body(QUESTION))
    plan = paused[-1]
    assert plan["event"] == "plan.proposed"
    quiescence.assert_quiescent()

    async with _client(app_b) as client_b:
        resumed = await _stream(
            client_b, "/api/studio/interrupt", _approve_body(plan["interrupt_id"])
        )
    assert resumed[-1]["event"] == "done"
    quiescence.assert_quiescent()

    async with _client(app_a) as client_a:
        later = await _stream(client_a, "/api/studio/turn", _turn_body("aid to kenya"))
    assert later[-1]["event"] == "done"
    quiescence.assert_quiescent()

    trace_id = resumed[-1]["trace_id"]
    first_trace = Langfuse.create_trace_id(seed=trace_id)
    spans = quiescence.fake.spans()
    first = [span for span in spans if span.trace_id.hex() == first_trace]
    (turn,) = _closes(first)
    assert _attribute(turn, "langfuse.observation.input") == QUESTION
    assert _attribute(turn, "langfuse.observation.output")
    assert _attribute(turn, "langfuse.observation.level") is None
    assert [span.name for span in first].count("turn.segment") == 2
    assert quiescence.outcomes() == [(trace_id, None), (later[-1]["trace_id"], None)]
    for span in _spans_with_an_unexported_parent(spans):
        assert _attribute(span, "langfuse.internal.is_app_root") is True, span.name


async def test_a_close_the_store_never_saw_writes_one_outcome_as_abandoned(
    quiescence: Quiescence, pg_engine: AsyncEngine
) -> None:
    """A turn closes, but the close never reaches the store, so it writes no
    line and no outcome. Reloaded from the store, the next turn's
    `begin_turn` closes it as `abandoned`, and its save writes the one `turn`
    outcome, at `WARNING`, and the one line the stored record agrees with."""
    store = StudioStore(pg_engine, adk=NO_ADK, turn_specs=no_turn_specs)
    conversation = await store.create("principal", CONVERSATION)
    conversation.begin_turn()
    conversation.turn_user_message = "q"
    await store.save("principal", CONVERSATION)
    trace_id = conversation.trace_id
    conversation.close_turn(final_text="a")  # its save never lands
    conversation.end_segment_span("done")

    reloaded = await StudioStore(pg_engine, adk=NO_ADK, turn_specs=no_turn_specs).load(
        "principal", CONVERSATION
    )
    assert reloaded is not None
    assert not reloaded.turn_recorded
    get_client().flush()
    assert quiescence.outcomes() == []
    other = StudioStore(pg_engine, adk=NO_ADK, turn_specs=no_turn_specs)
    fresh = await other.load("principal", CONVERSATION)
    assert fresh is not None
    fresh.begin_turn()
    await other.save("principal", CONVERSATION)
    fresh.end_segment_span("not_started")
    get_client().flush()

    assert quiescence.outcomes() == [(trace_id, "abandoned")]
    await quiescence.assert_store_agrees(
        pg_engine, principal="principal", conversation_id=CONVERSATION
    )
    (turn,) = _closes(quiescence.fake.spans())
    assert _attribute(turn, "langfuse.observation.level") == "WARNING"


async def test_a_streams_last_flush_already_holds_its_segment(
    quiescence: Quiescence, monkeypatch: pytest.MonkeyPatch, pg_engine: AsyncEngine
) -> None:
    """The segment span ends in `StudioRunner._locked`, before the stream's
    final `flush_telemetry`, so that flush exports it with its children. Left
    for a later flush, a throttled instance would hold it and orphan every
    span under it."""
    real_flush = studio_module.flush_telemetry
    seen_at_flush: list[list[str]] = []

    async def recording_flush(*args: Any, **kwargs: Any) -> None:
        await real_flush(*args, **kwargs)
        seen_at_flush.append([span.name for span in quiescence.fake.spans()])

    monkeypatch.setattr(studio_module, "flush_telemetry", recording_flush)
    app = _planned_app(pg_engine)
    async with _client(app) as client:
        paused = await _stream(client, "/api/studio/turn", _turn_body(QUESTION))
    assert paused[-1]["event"] == "plan.proposed"

    last = seen_at_flush[-1]
    assert last.count("turn.segment") == 1
    assert any(name.startswith("invoke_workflow") for name in last)
    quiescence.assert_quiescent()


async def test_a_cancel_on_an_instance_that_never_ran_the_paused_turn_writes_its_outcome(
    quiescence: Quiescence, pg_engine: AsyncEngine
) -> None:
    """`/cancel` reaches a paused turn on an instance that loads it fresh from
    the store. That instance never held anything for the turn, and still
    writes its one `turn` outcome at `WARNING` naming the cancellation."""
    app_a = _instance(pg_engine, plan_call(QUESTION))
    app_b = _instance(pg_engine)

    async with _client(app_a) as client_a:
        paused = await _stream(client_a, "/api/studio/turn", _turn_body(QUESTION))
    assert paused[-1]["event"] == "plan.proposed"
    stored = await StudioStore(pg_engine, adk=NO_ADK, turn_specs=no_turn_specs).load(
        PRINCIPAL, CONVERSATION
    )
    assert stored is not None
    async with _client(app_b) as client_b:
        cancel = await client_b.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
    assert cancel.status_code == 204

    assert quiescence.outcomes() == [(stored.trace_id, "cancelled")]
    await quiescence.assert_store_agrees(
        pg_engine, principal=PRINCIPAL, conversation_id=CONVERSATION
    )
    (turn,) = _closes(quiescence.fake.spans())
    assert _attribute(turn, "langfuse.observation.level") == "WARNING"
    assert _attribute(turn, "langfuse.observation.input") == QUESTION


async def test_a_rejected_plan_streams_inside_a_segment_of_its_own(
    quiescence: Quiescence, pg_engine: AsyncEngine
) -> None:
    """The `/interrupt` that rejects a plan runs no workflow, and still streams
    inside a `turn.segment`: one segment per stream, beside the one `turn`
    outcome the rejection writes."""
    app = _planned_app(pg_engine)
    async with _client(app) as client:
        paused = await _stream(client, "/api/studio/turn", _turn_body(QUESTION))
        assert paused[-1]["event"] == "plan.proposed"
        quiescence.assert_quiescent()
        rejected = await _stream(
            client,
            "/api/studio/interrupt",
            {**_approve_body(paused[-1]["interrupt_id"]), "decision": "reject"},
        )
    assert rejected[-1]["event"] == "done"
    quiescence.assert_quiescent()

    trace_id = rejected[-1]["trace_id"]
    seeded = Langfuse.create_trace_id(seed=trace_id)
    names = [span.name for span in quiescence.fake.spans() if span.trace_id.hex() == seeded]
    assert names.count("turn.segment") == 2  # the `/turn` stream and the rejection's
    assert quiescence.outcomes() == [(trace_id, None)]
    (turn,) = _closes(quiescence.fake.spans())
    assert _attribute(turn, "langfuse.observation.output") == "Plan rejected."


@pytest.mark.parametrize(
    "failure", ["leftover-session-replace-fails", "claim-committed-then-fails"]
)
async def test_a_turn_that_ends_with_no_stream_flushes_the_abandoned_close_it_wrote(
    quiescence: Quiescence, pg_engine: AsyncEngine, failure: str
) -> None:
    """A `/turn` on a paused conversation writes the paused turn's `abandoned`
    close as its claim save commits, then ends with no stream: replacing a
    leftover session of the new turn fails, or the claim save raised after it
    committed. The close's outcome reaches the wire with no stream's flush to
    carry it."""
    app = _planned_app(pg_engine)
    studio: StudioRunner = app.state.studio
    async with _client(app) as client:
        paused = await _stream(client, "/api/studio/turn", _turn_body(QUESTION))
        assert paused[-1]["event"] == "plan.proposed"
        live = studio.store.current(PRINCIPAL, CONVERSATION)
        assert live is not None
        paused_trace_id = live.trace_id
        if failure == "claim-committed-then-fails":
            real_save = studio.store.save

            async def save_then_fail(*args: Any, **kwargs: Any) -> bool:
                studio.store.save = real_save  # type: ignore[method-assign]
                await real_save(*args, **kwargs)
                raise RuntimeError("the store answered late")

            studio.store.save = save_then_fail  # type: ignore[method-assign]
        else:
            # The new turn's session exists, as one a crash left, so the
            # claim save is followed by its replacement, which fails.
            await studio.runner.session_service.create_session(
                app_name=AGENT_NAME, user_id=PRINCIPAL, session_id=f"{CONVERSATION}#t2"
            )

            async def replace_fails(**_: Any) -> None:
                raise RuntimeError("session store unavailable")

            studio._replace_leftover_session = replace_fails  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            await client.post("/api/studio/turn", json=_turn_body("aid to kenya"), headers=AUTH)

    # No flush here, since the request's own flush must have exported the outcome.
    assert quiescence.outcomes() == [(paused_trace_id, "abandoned")]
    quiescence.assert_quiescent()


class _SlowDataClient:
    """A `Client` stand-in whose `call_tool` answers a background mint as
    `posted` once `release` is set."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __aenter__(self) -> _SlowDataClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def call_tool(self, tool: str, arguments: dict[str, Any], **_: Any) -> Any:
        self.entered.set()
        await self.release.wait()
        return SimpleNamespace(
            is_error=False,
            structured_content={BACKGROUND_OUTCOME_KEY: "posted", "spec_id": arguments["spec_id"]},
        )


async def test_an_early_mint_that_ends_during_the_settle_wait_is_not_logged_unsent(
    quiescence: Quiescence,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    pg_engine: AsyncEngine,
) -> None:
    """An early mint that answers while the response waits for it ends inside
    that wait though exporting its root is slow. It starts the flush and
    does not wait for it, so it logs its own outcome and not
    `unsent_at_stream_end`."""
    real_flush_backend = telemetry_module._flush_backend

    def slow_flush_backend() -> None:
        time.sleep(0.3)  # an export slower than the settle wait
        real_flush_backend()

    monkeypatch.setattr(telemetry_module, "_flush_backend", slow_flush_backend)
    data = _SlowDataClient()
    monkeypatch.setattr(
        McpTimeoutPolicy, "client", lambda self, url, *, auth, caller, tool=None: data
    )
    executor = McpExecutor(settings=Settings())
    early = EarlyMints(send=executor.get_data_in_background, max_concurrent=4, settle_s=0.2)
    real_settle = early.settle

    async def settle() -> bool:
        data.release.set()  # the mint answers once the wait has begun
        return await real_settle()

    early.settle = settle  # type: ignore[method-assign]
    app, _model = orchestrator_app(
        plan_call(QUESTION),
        executor=ScriptedExecutor(),
        tool_surface=ScriptedSurface(),
        settings=Settings(studio_api_token="test"),
        early_mints=early,
        engine=pg_engine,
    )
    try:
        with caplog.at_level(logging.INFO, logger="harness.early_mint"):
            async with _client(app) as client:
                paused = await _stream(client, "/api/studio/turn", _turn_body(QUESTION))
    finally:
        await early.aclose()
        await executor.aclose()

    assert paused[-1]["event"] == "plan.proposed"
    lines = [r.getMessage() for r in caplog.records if r.name == "harness.early_mint"]
    assert len(lines) == 1
    assert "outcome=posted" in lines[0]


async def test_an_early_mint_that_ends_after_its_pause_exports_its_result_on_the_turns_trace(
    quiescence: Quiescence, monkeypatch: pytest.MonkeyPatch, pg_engine: AsyncEngine
) -> None:
    """The card's early `get_data` is still running when the stream that
    offered it pauses and ends its segment. When it ends, its `dfo.tool_result`
    goes into a `turn.early_mint` root on the turn's trace, flushed at once, so
    it reaches the wire with no request left to flush it."""
    data = _SlowDataClient()
    monkeypatch.setattr(
        McpTimeoutPolicy, "client", lambda self, url, *, auth, caller, tool=None: data
    )
    executor = McpExecutor(settings=Settings())
    early = EarlyMints(send=executor.get_data_in_background, max_concurrent=4, settle_s=0.05)
    app, _model = orchestrator_app(
        plan_call(QUESTION),
        executor=ScriptedExecutor(),
        tool_surface=ScriptedSurface(),
        settings=Settings(studio_api_token="test"),
        early_mints=early,
        engine=pg_engine,
    )
    try:
        async with _client(app) as client:
            paused = await _stream(client, "/api/studio/turn", _turn_body(QUESTION))
        assert paused[-1]["event"] == "plan.proposed"
        await asyncio.wait_for(data.entered.wait(), timeout=5)
        quiescence.assert_quiescent()
        live = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
        assert live is not None
        trace_id = live.trace_id

        data.release.set()
        await _until(
            lambda: any(span.name == "dfo.tool_result" for span in quiescence.fake.spans())
        )
    finally:
        await early.aclose()
        await executor.aclose()

    spans = quiescence.fake.spans()
    (result,) = [span for span in spans if span.name == "dfo.tool_result"]
    (root,) = [span for span in spans if span.name == "turn.early_mint"]
    assert result.parent_span_id == root.span_id
    assert _attribute(result, "langfuse.observation.metadata.mint") == "background"
    assert root.trace_id.hex() == Langfuse.create_trace_id(seed=trace_id)
    # A root by the markers the turn's own roots carry, Langfuse's root flags,
    # under a synthetic parent that is never exported.
    exported = {span.span_id for span in spans}
    (segment,) = [span for span in spans if span.name == "turn.segment"]
    for span in (root, segment):
        assert _attribute(span, "langfuse.internal.as_root") is True, span.name
        assert _attribute(span, "langfuse.internal.is_app_root") is True, span.name
        assert span.parent_span_id, span.name
        assert span.parent_span_id not in exported, span.name
    quiescence.assert_quiescent()
