"""What a Studio turn puts on the wire to Langfuse.

The real Langfuse SDK, tracer provider, span filter and Studio routes run; only
the Langfuse server is fake (`fake_langfuse`). These tests cover the parts an
in-memory span exporter cannot see: which spans the export filter keeps, how a
response that is torn down mid-stream ends the turn's telemetry, and the parent
chain of a turn that pauses for approval and resumes.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import inspect
import json
import logging
import sys
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import aclosing, contextmanager, suppress
from functools import partial
from types import AsyncGeneratorType
from typing import Any

import harness.studio as studio_module
import httpx2
import pytest
from common.telemetry import init_telemetry
from harness.server import create_app
from harness.settings import Settings
from harness.studio import SSE_MEDIA_TYPE, StudioRunner, _sse_stream
from harness.studio_state import ConversationState
from harness.wire import MessageDelta
from langfuse import Langfuse
from opentelemetry import trace
from opentelemetry.proto.trace.v1.trace_pb2 import Span
from starlette.responses import StreamingResponse

CONVERSATION = "conv_trace_export_01"
AUTH = {"Authorization": "Bearer test"}


def _attribute_keys(span: Span) -> set[str]:
    return {attribute.key for attribute in span.attributes}


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
    """Yield a function returning the `_sse_stream` generators the event loop
    has taken over for closing: a generator dropped while suspended is handed
    to the loop, which closes it later on a task of its own. Only generators
    first iterated inside the block are seen."""
    hooks = sys.get_asyncgen_hooks()
    handed_over: list[AsyncGeneratorType] = []

    def finalizer(generator: AsyncGeneratorType) -> None:
        handed_over.append(generator)
        assert hooks.finalizer is not None
        hooks.finalizer(generator)

    sys.set_asyncgen_hooks(firstiter=hooks.firstiter, finalizer=finalizer)
    try:
        yield lambda: [g for g in handed_over if g.ag_code is _sse_stream.__code__]
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
        with trace.get_tracer("test.segment").start_as_current_span("call_llm") as span:
            span.set_attribute("gen_ai.system", "test")
        yield MessageDelta(event="message.delta", seq=0, text="one")
        conversation.close_turn(final_text="one")
        yield MessageDelta(event="message.delta", seq=1, text="two")

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
    fake_langfuse: Any, caplog: pytest.LogCaptureFixture, how: str
) -> None:
    """The response task ends while the stream generator sits at a `yield`, so
    the event loop closes it later in a context of its own. The turn span,
    ended by the segment, is exported by the flush in the generator's
    `finally`, and leaving the telemetry scope there does not make
    OpenTelemetry log a failed detach."""
    init_telemetry(service_name="harness")
    conversation = ConversationState(conversation_id=CONVERSATION)
    conversation.begin_turn()
    conversation.turn_user_message = "q"

    with (
        caplog.at_level(logging.ERROR, logger="opentelemetry.context"),
        _abandoned_streams() as (abandoned),
    ):
        await _abandon_a_response_at_its_second_frame(conversation, how=how)
        # The turn span reaches the wire from the flush inside the generator's
        # `finally`, before the scope exit that would log a failed detach, so
        # the log is read once the generator has finished closing.
        await _until(
            lambda: any(
                inspect.getasyncgenstate(generator) == inspect.AGEN_CLOSED
                for generator in abandoned()
            )
        )
        await _until(lambda: any(span.name == "turn" for span in fake_langfuse.spans()))

    spans = fake_langfuse.spans()
    (turn,) = [span for span in spans if span.name == "turn"]
    (call,) = [span for span in spans if span.name == "call_llm"]
    assert turn.end_time_unix_nano > 0
    assert call.parent_span_id == turn.span_id
    assert conversation.cancel_event is None
    assert [r for r in caplog.records if r.name == "opentelemetry.context"] == []


# --- a scripted turn that pauses for approval and resumes -------------------------


async def _stream(client: httpx2.AsyncClient, path: str, body: dict[str, Any]) -> list[dict]:
    async with client.stream("POST", path, json=body, headers=AUTH) as response:
        assert response.status_code == 200
        return [
            json.loads(line[len("data: ") :])
            async for line in response.aiter_lines()
            if line.startswith("data: ")
        ]


async def test_a_paused_and_resumed_turn_exports_a_complete_parent_chain(
    fake_langfuse: Any,
) -> None:
    """Both segments of a scripted turn run ADK's real workflow. Every span on
    the wire has its parent on the wire too, except `turn`, whose parent is the
    synthetic one the SDK creates for a seeded trace id; ADK's two `invocation`
    spans are exported, and nothing else without attributes is."""
    init_telemetry(service_name="harness")
    app = create_app(settings=Settings(studio_scripted=True))
    base = {"schema_version": 1, "conversation_id": CONVERSATION}
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://scripted.test"
    ) as client:
        first = await _stream(
            client,
            "/api/studio/turn",
            {
                **base,
                "kind": "turn.request",
                "message": "comprehensive overview of life expectancy in Kenya",
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

    scoped = fake_langfuse.scoped_spans()
    spans = [span for _, span in scoped]
    (turn,) = [span for span in spans if span.name == "turn"]
    assert turn.trace_id.hex() == Langfuse.create_trace_id(seed=second[-1]["trace_id"])
    assert {span.trace_id for span in spans} == {turn.trace_id}
    invocations = [span for span in spans if span.name == "invocation"]
    workflows = [span for span in spans if span.name.startswith("invoke_workflow")]
    assert len(invocations) == len(workflows) == 2
    assert {span.parent_span_id for span in invocations} == {turn.span_id}
    assert {span.parent_span_id for span in workflows} == {span.span_id for span in invocations}
    assert _spans_with_an_unexported_parent(spans) == [turn]
    # The filter drops attribute-less spans other than ADK's `invocation`.
    for scope, span in scoped:
        assert (
            scope == "langfuse-sdk"
            or span.name == "invocation"
            or any(key.startswith("gen_ai") for key in _attribute_keys(span))
        ), (scope, span.name)


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
    task, so the event loop closes the abandoned generator at the heartbeat
    `yield` (`GeneratorExit`)."""
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
    fake_langfuse: Any, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """The client goes away right after `plan.proposed`, while the stream
    awaits the next frame. The turn stays open and unrecorded, so the
    `/interrupt` that approves the plan resumes it inside the same `turn` span:
    one trace, one `turn` span ended once, the execution spans under it, and
    no cancellation on the turn's record."""
    init_telemetry(service_name="harness")
    app = create_app(settings=Settings(studio_scripted=True))
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

    with _abandoned_streams() as abandoned:
        interrupt_id = await _post_and_leave_at_the_pause(app, body, how=how, at_pause=at_pause)
        principal = "u_" + hashlib.sha256(b"test").hexdigest()[:16]
        conversation = studio.store.current(principal, CONVERSATION)
        assert conversation is not None
        await _until(lambda: not conversation.lock.locked() and conversation.cancel_event is None)
        if how == "close":
            await _until(
                lambda: any(
                    inspect.getasyncgenstate(generator) == inspect.AGEN_CLOSED
                    for generator in abandoned()
                )
            )
    assert conversation.awaiting_answer

    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://scripted.test"
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
    (turn,) = [span for span in spans if span.name == "turn"]
    assert turn.end_time_unix_nano > 0
    assert {span.trace_id for span in spans} == {turn.trace_id}
    assert turn.trace_id.hex() == Langfuse.create_trace_id(seed=resumed[-1]["trace_id"])
    assert any(span.name.startswith("invoke_node execute") for span in spans)
    (record,) = conversation.turns
    assert "error" not in record


async def test_a_client_that_leaves_while_the_turn_runs_ends_the_turn_span_as_cancelled(
    fake_langfuse: Any,
) -> None:
    """A turn that has not paused is closed by the disconnect: the `turn` span
    ends once and the record carries the cancellation."""
    init_telemetry(service_name="harness")
    conversation = ConversationState(conversation_id=CONVERSATION)
    conversation.begin_turn()
    conversation.turn_user_message = "q"
    running = asyncio.Event()

    async def segment() -> AsyncGenerator[MessageDelta]:
        with trace.get_tracer("test.segment").start_as_current_span("call_llm") as span:
            span.set_attribute("gen_ai.system", "test")
        yield MessageDelta(event="message.delta", seq=0, text="one")
        running.set()
        await asyncio.Event().wait()  # a slow tool call
        yield MessageDelta(event="message.delta", seq=1, text="two")

    async def consume() -> None:
        async for _ in _sse_stream(segment(), conversation=conversation, heartbeat_seconds=15):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(running.wait(), timeout=15)
    await asyncio.sleep(0.05)  # the stream is parked at its wait for the next frame
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _until(lambda: any(span.name == "turn" for span in fake_langfuse.spans()))

    spans = fake_langfuse.spans()
    (turn,) = [span for span in spans if span.name == "turn"]
    (call,) = [span for span in spans if span.name == "call_llm"]
    assert turn.end_time_unix_nano > 0
    assert call.parent_span_id == turn.span_id
    assert conversation.turn_recorded
    assert conversation.turns == [{"user": "q", "assistant": "", "error": "cancelled"}]
