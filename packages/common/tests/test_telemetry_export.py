"""End-to-end telemetry tests against a local stand-in for Langfuse.

The real Langfuse SDK, tracer provider and span filter run; only the server is
fake. A local HTTP server answers the credentials check and records every OTLP
trace export, and the tests assert on the spans that arrived on the wire. The
unit tests in `test_telemetry.py` mock the client, so they cannot see whether a
span is exported at all.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mcp.shared._otel as mcp_otel
import pytest
from common import telemetry
from common.telemetry import (
    SESSION_META_KEY,
    TAGS_META_KEY,
    USER_META_KEY,
    backend,
    end_turn_segment,
    flush_telemetry,
    init_telemetry,
    record_event,
    record_turn_outcome,
    root_if_span_ended,
    start_turn_segment,
    turn_call_meta,
    turn_segment,
    use_call_session,
    use_request_span,
)
from common.uri_redaction import UriRedactingExporter
from fastmcp import Client, FastMCP
from langfuse import Langfuse, get_client
from langfuse import __version__ as langfuse_version
from langfuse._client.resource_manager import LangfuseResourceManager
from langfuse._client.span_exporter import LangfuseTransformingSpanExporter
from langfuse._client.span_processor import LangfuseSpanProcessor
from opentelemetry import baggage, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

_PROJECTS_BODY = {
    "data": [
        {"id": "p", "name": "proj", "organization": {"id": "o", "name": "org"}, "metadata": {}}
    ]
}


@dataclass
class FakeLangfuse:
    """A local Langfuse: `projects_status` is what the credentials check gets,
    `exports` holds the decoded body of every OTLP trace export and
    `export_headers` its request headers, and an export blocks until
    `release_exports` is set while `hang_exports` is true."""

    base_url: str
    projects_status: int = 200
    hang_exports: bool = False
    release_exports: threading.Event = field(default_factory=threading.Event)
    exports: list[ExportTraceServiceRequest] = field(default_factory=list)
    export_headers: list[dict[str, str]] = field(default_factory=list)

    def spans(self) -> list[Span]:
        return [
            span
            for request in self.exports
            for resource_spans in request.resource_spans
            for scope_spans in resource_spans.scope_spans
            for span in scope_spans.spans
        ]


@pytest.fixture
def fake_langfuse(isolated_telemetry, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeLangfuse]:
    state = FakeLangfuse(base_url="")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def _reply(self, status: int, body: bytes = b"{}") -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.startswith("/api/public/projects"):
                if state.projects_status != 200:
                    self._reply(state.projects_status)
                else:
                    self._reply(200, json.dumps(_PROJECTS_BODY).encode())
            else:
                self._reply(404)

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            if self.path == "/api/public/otel/v1/traces":
                if state.hang_exports:
                    state.release_exports.wait(timeout=30)
                request = ExportTraceServiceRequest()
                request.ParseFromString(body)
                state.exports.append(request)
                state.export_headers.append(
                    {name.lower(): value for name, value in self.headers.items()}
                )
                self._reply(200)
            else:
                self._reply(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", state.base_url)
    try:
        yield state
    finally:
        # Shutting the client down flushes it, which needs the server up.
        state.release_exports.set()
        LangfuseResourceManager.reset()
        server.shutdown()
        server.server_close()


def _join_auth_check_thread() -> None:
    for thread in threading.enumerate():
        if thread.name == telemetry._AUTH_CHECK_THREAD_NAME:
            thread.join(timeout=30)
            assert not thread.is_alive()


def _attribute(span: Span, key: str) -> object:
    """The attribute's value. The SDK JSON-encodes `langfuse.*.metadata.*` values, so
    they are decoded here."""
    for attribute in span.attributes:
        if attribute.key == key:
            value = getattr(attribute.value, attribute.value.WhichOneof("value"))
            if ".metadata." in key:
                return json.loads(value)
            return value
    return None


def _array_attribute(span: Span, key: str) -> list[str] | None:
    for attribute in span.attributes:
        if attribute.key == key:
            return [value.string_value for value in attribute.value.array_value.values]
    return None


def _child_span(name: str) -> None:
    """Open and close a span with a `gen_ai` attribute, the kind ADK and
    fastmcp emit, which Langfuse's default filter exports."""
    with trace.get_tracer("test.child").start_as_current_span(name) as child:
        child.set_attribute("gen_ai.system", "test")


def _one_turn(external_id: str) -> None:
    """Run a child span inside one `turn.segment`, then write the turn's
    `turn` outcome."""
    with turn_segment(external_id):
        _child_span("generate_content gemini-x")
    record_turn_outcome(external_id, request="q", response="a")
    get_client().flush()


def _assert_turn_root_exported(fake: FakeLangfuse, external_id: str) -> None:
    spans = fake.spans()
    (segment,) = [span for span in spans if span.name == "turn.segment"]
    (turn,) = [span for span in spans if span.name == "turn"]
    (child,) = [span for span in spans if span.name == "generate_content gemini-x"]

    expected_trace_id = Langfuse.create_trace_id(seed=external_id)
    assert {span.trace_id.hex() for span in spans} == {expected_trace_id}
    for root in (segment, turn):
        assert _attribute(root, "langfuse.observation.metadata.dfo.external_id") == external_id
        # The seeded trace id comes from a synthetic remote parent that is
        # never exported, so each root is identified by Langfuse's own marker.
        assert _attribute(root, "langfuse.internal.is_app_root") is True
    assert child.parent_span_id == segment.span_id


def test_turn_root_span_is_exported_with_the_trace_id_derived_from_the_wire_id(
    fake_langfuse: FakeLangfuse,
) -> None:
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    assert backend() == "langfuse"
    _one_turn("tr_export0001")

    _assert_turn_root_exported(fake_langfuse, "tr_export0001")


def test_turn_root_span_is_exported_when_the_credentials_check_returns_503(
    fake_langfuse: FakeLangfuse, caplog: pytest.LogCaptureFixture
) -> None:
    """The exporter is attached as soon as the client exists and the backend
    stays `langfuse` after a failing credentials check, so the turn root and
    its children are exported."""
    fake_langfuse.projects_status = 503

    with caplog.at_level("WARNING"):
        init_telemetry(service_name="harness")
        _join_auth_check_thread()

    assert backend() == "langfuse"
    assert any("auth check" in record.message for record in caplog.records)
    _one_turn("tr_export0002")

    _assert_turn_root_exported(fake_langfuse, "tr_export0002")


def test_record_event_exports_a_langfuse_event_under_the_turn_segment(
    fake_langfuse: FakeLangfuse,
) -> None:
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with turn_segment("tr_export0003"):
        record_event("compose.refused", metadata={"reason": "no_rows", "count": 3, "ok": False})
    get_client().flush()

    spans = fake_langfuse.spans()
    (turn,) = [span for span in spans if span.name == "turn.segment"]
    (event,) = [span for span in spans if span.name == "compose.refused"]
    assert event.trace_id == turn.trace_id
    assert event.parent_span_id == turn.span_id
    assert _attribute(event, "langfuse.observation.level") == "WARNING"
    assert _attribute(event, "langfuse.observation.metadata.reason") == "no_rows"
    assert _attribute(event, "langfuse.observation.metadata.count") == 3
    assert _attribute(event, "langfuse.observation.metadata.ok") is False


def test_record_event_adds_a_span_event_on_the_otlp_backend(isolated_telemetry) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    telemetry._backend = "otel"

    with turn_segment("tr_export0004"):
        record_event("compose.refused", metadata={"reason": "no_rows", "count": 3})

    (span,) = exporter.get_finished_spans()
    (event,) = span.events
    assert span.name == "turn.segment"
    assert event.name == "compose.refused"
    assert dict(event.attributes or {}) == {"reason": "no_rows", "count": 3, "level": "WARNING"}


def test_a_turn_segment_ended_with_attributes_exports_them_on_the_otlp_backend(
    isolated_telemetry,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    telemetry._backend = "otel"

    end_turn_segment(
        start_turn_segment("tr_export0005"), status="paused", attributes={"dfo.stored": "paused"}
    )

    (span,) = exporter.get_finished_spans()
    assert span.name == "turn.segment"
    assert (span.attributes or {}).get("dfo.stored") == "paused"


def test_a_turn_segment_ended_with_attributes_exports_them_on_langfuse(
    fake_langfuse: FakeLangfuse,
) -> None:
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    end_turn_segment(
        start_turn_segment("tr_export0006"),
        status="done",
        level="DEFAULT",
        attributes={"dfo.stored": "done"},
    )
    get_client().flush()

    (segment,) = [span for span in fake_langfuse.spans() if span.name == "turn.segment"]
    assert _attribute(segment, "dfo.stored") == "done"
    assert _attribute(segment, "langfuse.observation.level") == "DEFAULT"
    assert _attribute(segment, "langfuse.observation.status_message") == "done"
    assert not [span for span in fake_langfuse.spans() if span.name == "turn"]


async def test_a_turns_otlp_spans_share_one_synthetic_parent(isolated_telemetry) -> None:
    """On the OTLP backend two segments of one turn, its outcome, and an
    early-mint root opened after its segment ended are children of one
    remote parent span, so a trace viewer shows them as siblings under one
    missing root and not as four roots."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    telemetry._backend = "otel"
    segment_ended = asyncio.Event()

    async def late_mint() -> None:
        await segment_ended.wait()
        with root_if_span_ended("turn.early_mint") as rooted:
            assert rooted

    with turn_segment("tr_export0023"):
        mint = asyncio.ensure_future(late_mint())
    segment_ended.set()
    await mint
    with turn_segment("tr_export0023"):
        pass
    record_turn_outcome("tr_export0023", request="q", response="a")

    spans = exporter.get_finished_spans()
    assert sorted(span.name for span in spans) == [
        "turn",
        "turn.early_mint",
        "turn.segment",
        "turn.segment",
    ]
    parents = {(span.parent.trace_id, span.parent.span_id) for span in spans if span.parent}
    assert len(parents) == 1
    assert all(span.parent is not None and span.parent.is_remote for span in spans)
    assert {span.context.trace_id for span in spans} == {next(iter(parents))[0]}


async def test_an_early_mint_under_an_ended_inner_span_roots_only_once_its_segment_ends(
    isolated_telemetry,
) -> None:
    """A mint offered under an inner span of the segment records under the
    live segment while the segment runs, though the inner span has ended,
    and opens its own root only once the segment has ended."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    telemetry._backend = "otel"
    inner_ended, segment_ended = asyncio.Event(), asyncio.Event()
    rooted: list[bool] = []

    async def mint() -> None:
        for ended in (inner_ended, segment_ended):
            await ended.wait()
            with root_if_span_ended("turn.early_mint") as opened:
                rooted.append(opened)

    with turn_segment("tr_export0024"):
        with trace.get_tracer("test").start_as_current_span("tool"):
            task = asyncio.ensure_future(mint())
        inner_ended.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    segment_ended.set()
    await task

    assert rooted == [False, True]
    assert sorted(span.name for span in exporter.get_finished_spans()) == [
        "tool",
        "turn.early_mint",
        "turn.segment",
    ]


async def test_an_early_mint_after_a_segment_the_sampler_dropped_opens_no_root(
    isolated_telemetry,
) -> None:
    """A segment the sampler dropped records nothing and counts as live, so a
    mint that ends after it opens no root, which would be dropped too."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=ALWAYS_OFF)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    telemetry._backend = "otel"
    segment_ended = asyncio.Event()
    rooted: list[bool] = []

    async def late_mint() -> None:
        await segment_ended.wait()
        with root_if_span_ended("turn.early_mint") as opened:
            rooted.append(opened)

    with turn_segment("tr_export0025"):
        mint = asyncio.ensure_future(late_mint())
    segment_ended.set()
    await mint

    assert rooted == [False]
    assert exporter.get_finished_spans() == ()


async def test_a_turn_continued_in_a_second_task_exports_as_one_trace(
    fake_langfuse: FakeLangfuse, caplog: pytest.LogCaptureFixture
) -> None:
    """A turn whose segments run in two tasks and whose outcome is written in
    a third (a turn paused for approval and resumed by a separate request) is
    one trace: each segment is a root, the spans and events of each task nest
    under that task's segment, and the `turn` outcome is a root of its own.
    The turn's Langfuse session id reaches the spans of both segments and
    stops with each segment's block."""
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    async def first_segment() -> None:
        with turn_segment("tr_export0005", session_id="conv-1"):
            _child_span("plan_call")
        _child_span("plan_call_after_block")

    async def resumed_segment() -> None:
        with turn_segment("tr_export0005", session_id="conv-1"):
            _child_span("execute_call")
            record_event("compose.write_refused", metadata={"code": "missing_field"})
        _child_span("execute_call_after_block")

    async def close_turn() -> None:
        record_turn_outcome("tr_export0005", session_id="conv-1", request="q", response="a")

    with caplog.at_level(logging.WARNING):
        await asyncio.create_task(first_segment())
        await asyncio.create_task(resumed_segment())
        await asyncio.create_task(close_turn())
    get_client().flush()

    exported = fake_langfuse.spans()
    spans = [span for span in exported if not span.name.endswith("_after_block")]
    after_block = [span for span in exported if span.name.endswith("_after_block")]
    assert {span.trace_id.hex() for span in spans} == {
        Langfuse.create_trace_id(seed="tr_export0005")
    }
    by_id = {span.span_id: span for span in spans}
    for name, parent in (
        ("plan_call", "turn.segment"),
        ("execute_call", "turn.segment"),
        ("compose.write_refused", "turn.segment"),
    ):
        (span,) = [span for span in spans if span.name == name]
        assert by_id[span.parent_span_id].name == parent, name
    (plan_call,) = [span for span in spans if span.name == "plan_call"]
    (execute_call,) = [span for span in spans if span.name == "execute_call"]
    assert plan_call.parent_span_id != execute_call.parent_span_id
    # Every span but the roots has a parent inside the trace.
    assert sorted(
        span.name for span in spans if _attribute(span, "langfuse.internal.is_app_root") is True
    ) == ["turn", "turn.segment", "turn.segment"]
    assert _spans_with_an_unexported_parent(spans) == [
        span for span in spans if span.name in {"turn", "turn.segment"}
    ]
    assert not [r for r in caplog.records if r.name == "opentelemetry.context"]
    for span in spans:
        assert _attribute(span, "session.id") == "conv-1", span.name
    assert len(after_block) == 2
    for span in after_block:
        assert _attribute(span, "session.id") is None, span.name


def test_a_segment_whose_block_raises_still_ends_and_exports_at_level_error(
    fake_langfuse: FakeLangfuse,
) -> None:
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with pytest.raises(RuntimeError), turn_segment("tr_export0020"):
        _child_span("generate_content gemini-x")
        raise RuntimeError("boom")
    get_client().flush()

    assert sorted(span.name for span in fake_langfuse.spans()) == [
        "generate_content gemini-x",
        "turn.segment",
    ]
    (segment,) = [span for span in fake_langfuse.spans() if span.name == "turn.segment"]
    # Only the class name is exported, since exception text may carry what the
    # trace must not.
    assert _attribute(segment, "langfuse.observation.level") == "ERROR"
    assert _attribute(segment, "langfuse.observation.status_message") == "RuntimeError"


async def test_flush_telemetry_exports_queued_spans_without_waiting_for_the_interval(
    fake_langfuse: FakeLangfuse, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SDK's batch worker wakes every `LANGFUSE_FLUSH_INTERVAL` seconds;
    with a five-minute interval only `flush_telemetry` can deliver the spans."""
    monkeypatch.setenv("LANGFUSE_FLUSH_INTERVAL", "300")
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with turn_segment("tr_export0006"):
        _child_span("generate_content gemini-x")
    assert fake_langfuse.spans() == []

    await flush_telemetry()

    assert {span.name for span in fake_langfuse.spans()} == {
        "turn.segment",
        "generate_content gemini-x",
    }


async def test_flush_telemetry_returns_at_its_bound_when_the_server_hangs(
    fake_langfuse: FakeLangfuse, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGFUSE_FLUSH_INTERVAL", "300")
    init_telemetry(service_name="harness")
    _join_auth_check_thread()
    with turn_segment("tr_export0007"):
        _child_span("generate_content gemini-x")
    fake_langfuse.hang_exports = True

    started = time.monotonic()
    await flush_telemetry(timeout_seconds=0.3)
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    assert fake_langfuse.spans() == []  # still blocked in the export

    # Released, the abandoned flush completes in the background and frees the gate.
    fake_langfuse.release_exports.set()
    for thread in threading.enumerate():
        if thread.name == telemetry._FLUSH_THREAD_NAME:
            thread.join(timeout=30)
            assert not thread.is_alive()
    assert {span.name for span in fake_langfuse.spans()} == {
        "turn.segment",
        "generate_content gemini-x",
    }


async def test_a_flush_requested_during_a_running_one_is_served_before_it_finishes(
    fake_langfuse: FakeLangfuse, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn that ends after the running flush has drained the queue, but
    before that flush releases single-flight, still has its spans exported
    without waiting for the batch interval. The test parks the running flush
    in exactly that window."""
    monkeypatch.setenv("LANGFUSE_FLUSH_INTERVAL", "300")
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    real_flush = telemetry._flush_backend
    in_window = threading.Event()
    leave_window = threading.Event()
    passes: list[int] = []

    def flush_then_wait_once() -> None:
        real_flush()
        passes.append(1)
        if len(passes) == 1:
            in_window.set()
            leave_window.wait(timeout=30)

    monkeypatch.setattr(telemetry, "_flush_backend", flush_then_wait_once)

    with turn_segment("tr_export0008"):
        _child_span("first_call")
    running = asyncio.create_task(flush_telemetry())
    assert await asyncio.to_thread(in_window.wait, 30)
    assert {span.name for span in fake_langfuse.spans()} == {"turn.segment", "first_call"}

    with turn_segment("tr_export0009"):
        _child_span("second_call")
    # Waits for the running flush, parked here, until its own bound.
    await asyncio.wait_for(flush_telemetry(timeout_seconds=0.1), timeout=2)
    assert "second_call" not in {span.name for span in fake_langfuse.spans()}

    leave_window.set()
    # The first caller wakes when its pass ends; the second pass, which the
    # withdrawn second caller asked for, still runs before the thread ends.
    await asyncio.wait_for(running, timeout=30)
    for thread in threading.enumerate():
        if thread.name == telemetry._FLUSH_THREAD_NAME:
            await asyncio.to_thread(thread.join, 30)

    assert "second_call" in {span.name for span in fake_langfuse.spans()}
    assert len(passes) == 2
    assert not telemetry._flush_running


def _spans_with_an_unexported_parent(spans: list[Span]) -> list[Span]:
    """The spans whose parent was not exported."""
    exported = {span.span_id for span in spans}
    return [span for span in spans if span.parent_span_id not in exported]


async def test_adks_spans_export_under_the_turn_and_bare_spans_do_not(
    fake_langfuse: FakeLangfuse,
) -> None:
    """ADK's schema v2 spans carry `gen_ai.*` attributes, so the Langfuse
    default filter exports them, `invoke_workflow` as a child of the segment.
    A span with no such attribute, from ADK's scope or any other, is
    dropped."""
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with turn_segment("tr_export0010"):
        adk = trace.get_tracer("gcp.vertex.agent")
        with adk.start_as_current_span("invoke_workflow wf") as workflow:
            workflow.set_attribute("gen_ai.operation.name", "invoke_workflow")
        with adk.start_as_current_span("bare_adk_span"):
            pass
        with trace.get_tracer("some.other.library").start_as_current_span("bare"):
            pass
    get_client().flush()

    spans = fake_langfuse.spans()
    assert sorted(span.name for span in spans) == ["invoke_workflow wf", "turn.segment"]
    (turn,) = [span for span in spans if span.name == "turn.segment"]
    (workflow,) = [span for span in spans if span.name == "invoke_workflow wf"]
    assert workflow.parent_span_id == turn.span_id
    # The only parent that was never exported is the SDK's synthetic one for
    # the segment.
    assert _spans_with_an_unexported_parent(spans) == [turn]


def _request_span(fake: FakeLangfuse, name: str) -> Span:
    get_client().flush()
    (span,) = [span for span in fake.spans() if span.name == name]
    return span


def test_a_request_span_that_raises_exports_as_an_error(fake_langfuse: FakeLangfuse) -> None:
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with pytest.raises(RuntimeError), use_request_span("combined mcp tools/call search"):
        raise RuntimeError("gs://bucket/secret-path")

    span = _request_span(fake_langfuse, "combined mcp tools/call search")
    assert _attribute(span, "langfuse.observation.level") == "ERROR"
    assert _attribute(span, "langfuse.observation.status_message") == "RuntimeError"


def test_a_request_span_exports_the_outcome_it_records(fake_langfuse: FakeLangfuse) -> None:
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with use_request_span("combined mcp tools/call get_data") as request:
        request.record_outcome(level="WARNING", status_message="request_invalid")

    span = _request_span(fake_langfuse, "combined mcp tools/call get_data")
    assert _attribute(span, "langfuse.observation.level") == "WARNING"
    assert _attribute(span, "langfuse.observation.status_message") == "request_invalid"
    assert span.status.code == span.status.STATUS_CODE_UNSET


def test_a_request_span_that_records_nothing_exports_at_the_default_level(
    fake_langfuse: FakeLangfuse,
) -> None:
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with use_request_span("combined mcp resources/read"):
        pass

    span = _request_span(fake_langfuse, "combined mcp resources/read")
    assert _attribute(span, "langfuse.observation.level") is None
    assert _attribute(span, "langfuse.observation.status_message") is None
    assert span.status.code == span.status.STATUS_CODE_UNSET


def test_a_request_span_exports_no_whole_credential_uri_from_the_error_it_raised(
    fake_langfuse: FakeLangfuse,
) -> None:
    """fastmcp raises `NotFoundError("Unknown resource: '<uri>'")` for a read
    it cannot route, and the SDK records that text in the span's status
    description and exception event. A `dfo://artifact` URI is a bearer
    credential, so the export keeps its scheme and host alone."""
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with pytest.raises(LookupError), use_request_span("combined mcp resources/read"):
        raise LookupError("Unknown resource: 'dfo://artifact/SECRETTOKEN'")

    span = _request_span(fake_langfuse, "combined mcp resources/read")
    (event,) = span.events
    assert event.name == "exception"
    assert "Unknown resource: 'dfo://artifact'" in span.status.message
    attributes = {attribute.key: attribute.value.string_value for attribute in event.attributes}
    assert attributes["exception.message"] == "Unknown resource: 'dfo://artifact'"
    assert b"SECRETTOKEN" not in b"".join(
        request.SerializeToString() for request in fake_langfuse.exports
    )


def test_a_resource_read_span_exports_a_presigned_urls_path_without_its_query(
    fake_langfuse: FakeLangfuse,
) -> None:
    """fastmcp sets a read's URI on its span as `mcp.resource.uri`. An AWS
    presigned URL there reads the object until it expires, so the export keeps
    its path and drops its query."""
    path = "https://dfo-exports.s3.eu-west-1.amazonaws.com/working/abc.csv"
    presigned = (
        f"{path}?X-Amz-Algorithm=AWS4-HMAC-SHA256"
        "&X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20260930%2Feu-west-1%2Fs3%2Faws4_request"
        "&X-Amz-Date=20260930T000000Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host"
        "&X-Amz-Signature=0123456789abcdef"
    )
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with use_request_span("combined mcp resources/read"):
        trace.get_current_span().set_attribute("mcp.resource.uri", presigned)

    span = _request_span(fake_langfuse, "combined mcp resources/read")
    assert _attribute(span, "mcp.resource.uri") == path
    wire = b"".join(request.SerializeToString() for request in fake_langfuse.exports)
    assert b"0123456789abcdef" not in wire
    assert b"AKIAIOSFODNN7EXAMPLE" not in wire


def test_an_observation_output_exports_a_presigned_urls_path_without_its_query(
    fake_langfuse: FakeLangfuse,
) -> None:
    """The SDK serialises an observation's output with `json.dumps`, which writes
    a non-ASCII character as a `\\uXXXX` escape. A presigned URL with one in its
    path still loses its query on export."""
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with use_request_span("combined mcp resources/read"):
        get_client().update_current_span(
            output={
                "url": "https://example.org/donn\u00e9es.csv?X-Amz-Signature=SECRETSIG"  # codespell:ignore
            }
        )

    span = _request_span(fake_langfuse, "combined mcp resources/read")
    assert (
        _attribute(span, "langfuse.observation.output")
        == '{"url": "https://example.org/donn\\u00e9es.csv"}'  # codespell:ignore
    )
    wire = b"".join(request.SerializeToString() for request in fake_langfuse.exports)
    assert b"SECRETSIG" not in wire


def test_the_langfuse_export_carries_the_sdks_credentials_and_headers(
    fake_langfuse: FakeLangfuse,
) -> None:
    """`init_telemetry` hands the SDK its own exporter, so it sets the
    credentials and SDK headers the SDK's default exporter would, and the
    ingestion-version header the SDK asks of a custom exporter."""
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    _one_turn("tr_export0011")

    headers = fake_langfuse.export_headers[0]
    assert headers["authorization"] == "Basic " + base64.b64encode(b"pk-lf-test:sk-lf-test").decode(
        "ascii"
    )
    assert headers["x-langfuse-public-key"] == "pk-lf-test"
    assert headers["x-langfuse-sdk-name"] == "python"
    assert headers["x-langfuse-sdk-version"] == langfuse_version
    assert headers["x-langfuse-ingestion-version"] == "4"


def _sdk_default_exporter() -> OTLPSpanExporter:
    """The OTLP exporter the Langfuse SDK builds for a client given no
    `span_exporter`, read off a client on its own tracer provider. The SDK and
    OpenTelemetry expose neither the processor list nor the exporter's
    settings, so this reads their private attributes; a rename fails the test
    that uses it."""
    provider = TracerProvider()
    Langfuse(tracer_provider=provider)
    (processor,) = [
        processor
        for processor in provider._active_span_processor._span_processors
        if isinstance(processor, LangfuseSpanProcessor)
    ]
    exporter = processor.span_exporter
    if isinstance(exporter, LangfuseTransformingSpanExporter):
        exporter = exporter._exporter
    assert isinstance(exporter, OTLPSpanExporter)
    return exporter


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"LANGFUSE_TIMEOUT": "7", "LANGFUSE_OTEL_TRACES_EXPORT_PATH": "custom/v1/traces"},
        {"LANGFUSE_OTEL_MAX_BATCH_SIZE_BYTES": "1000000"},
    ],
)
def test_the_langfuse_exporter_matches_the_one_the_sdk_builds_by_default(
    fake_langfuse: FakeLangfuse, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    """`_langfuse_exporter` copies the SDK's defaults and adds the v4
    ingestion header, so an SDK upgrade that changes the endpoint, headers,
    timeout, compression or request-size cap it would use fails here."""
    for name in (
        "LANGFUSE_OTEL_COMPRESSION",
        "OTEL_EXPORTER_OTLP_TRACES_COMPRESSION",
        "OTEL_EXPORTER_OTLP_COMPRESSION",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    ours = telemetry._langfuse_exporter()
    sdk = _sdk_default_exporter()

    assert isinstance(ours, UriRedactingExporter)
    inner = ours._exporter
    assert isinstance(inner, OTLPSpanExporter)
    assert inner._endpoint == sdk._endpoint
    assert inner._client._headers == {
        **sdk._client._headers,
        "x-langfuse-ingestion-version": "4",
    }
    assert inner._client._timeout == sdk._client._timeout
    assert inner._compression.value == sdk._compression.value
    assert inner._max_request_size == sdk._max_request_size


def test_the_langfuse_exporter_sends_no_compression_when_the_setting_says_none(
    fake_langfuse: FakeLangfuse, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGFUSE_OTEL_COMPRESSION", "none")

    exporter = telemetry._langfuse_exporter()

    assert exporter._exporter._compression.value == "none"


# --- one Langfuse session per conversation ------------------------------------


def _adk_child_span(name: str) -> None:
    """A span like ADK's `generate_content`: a separate tracer, with ADK's own
    per-turn conversation id."""
    with trace.get_tracer("gcp.vertex.agent").start_as_current_span(name) as span:
        span.set_attribute("gen_ai.conversation.id", "conv-1#t1")


def test_a_turn_with_a_session_id_exports_it_on_the_turn_and_on_spans_a_library_opens(
    fake_langfuse: FakeLangfuse,
) -> None:
    """The session, user and tags a turn's segment and outcome are given reach
    both and every span a library opens under the segment."""
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with turn_segment("tr_export0013", session_id="conv-1", user_id="a" * 64, tags=["studio"]):
        _adk_child_span("generate_content gemini-x")
    record_turn_outcome("tr_export0013", session_id="conv-1", user_id="a" * 64, tags=["studio"])
    get_client().flush()

    spans = fake_langfuse.spans()
    assert sorted(span.name for span in spans) == [
        "generate_content gemini-x",
        "turn",
        "turn.segment",
    ]
    for span in spans:
        assert _attribute(span, "session.id") == "conv-1", span.name
        assert _attribute(span, "langfuse.trace.name") == "turn", span.name
        assert _attribute(span, "user.id") == "a" * 64, span.name
        assert _array_attribute(span, "langfuse.trace.tags") == ["studio"], span.name


def test_the_exported_spans_name_the_cloud_run_revision_as_their_release(
    fake_langfuse: FakeLangfuse, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("K_REVISION", "dfo-dev-ai-00042-abc")
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    _one_turn("tr_export0016")

    releases = {
        attribute.value.string_value
        for request in fake_langfuse.exports
        for resource_spans in request.resource_spans
        for attribute in resource_spans.resource.attributes
        if attribute.key == "langfuse.release"
    }
    assert releases == {"dfo-dev-ai-00042-abc"}


def test_a_turn_without_a_session_id_exports_no_langfuse_session(
    fake_langfuse: FakeLangfuse,
) -> None:
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with turn_segment("tr_export0014"):
        _child_span("generate_content gemini-x")
    record_turn_outcome("tr_export0014")
    get_client().flush()

    spans = fake_langfuse.spans()
    assert {span.name for span in spans} == {"turn", "turn.segment", "generate_content gemini-x"}
    for span in spans:
        assert _attribute(span, "session.id") is None, span.name
        assert _attribute(span, "langfuse.trace.name") == "turn", span.name
        assert _attribute(span, "user.id") is None, span.name
        assert _array_attribute(span, "langfuse.trace.tags") is None, span.name


def test_a_turn_exports_its_request_and_response_on_the_outcome_observation_alone(
    fake_langfuse: FakeLangfuse,
) -> None:
    """The turn's request and response go on `turn`, the outcome observation,
    and on neither the segment nor its children. The outcome sets the trace's
    input and output to them too, for trace-level views and evaluators."""
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with turn_segment("tr_export0018", session_id="conv-1"):
        _adk_child_span("generate_content gemini-x")
    record_turn_outcome(
        "tr_export0018",
        session_id="conv-1",
        request="life expectancy in Kenya",
        response="It rose to 66.",
    )
    get_client().flush()

    io = {
        span.name: (
            _attribute(span, "langfuse.observation.input"),
            _attribute(span, "langfuse.observation.output"),
        )
        for span in fake_langfuse.spans()
    }
    assert io == {
        "turn": ("life expectancy in Kenya", "It rose to 66."),
        "turn.segment": (None, None),
        "generate_content gemini-x": (None, None),
    }
    (turn,) = [span for span in fake_langfuse.spans() if span.name == "turn"]
    assert turn.end_time_unix_nano - turn.start_time_unix_nano < 50_000_000


@pytest.mark.parametrize(
    ("error", "paused", "level", "status_message", "status"),
    [
        (None, False, None, None, None),
        ("state_not_saved", False, "ERROR", "state_not_saved", None),
        ("cancelled", False, "WARNING", "cancelled", None),
        ("abandoned", False, "WARNING", "abandoned", None),
        (None, True, None, None, "paused"),
    ],
    ids=["clean", "error", "cancelled", "abandoned", "paused"],
)
def test_a_turn_closed_with_an_error_exports_its_outcome_at_level_error(
    fake_langfuse: FakeLangfuse,
    error: str | None,
    paused: bool,
    level: str | None,
    status_message: str | None,
    status: str | None,
) -> None:
    """A turn ended with an error code marks the outcome observation `ERROR`,
    naming the code in its status message and its `dfo.turn_error` metadata,
    and `WARNING` for a Stop's `cancelled` and a newer turn's `abandoned`,
    which no failure ended. A clean turn leaves it at the default level with no `dfo.turn_error`. A
    pause is a `turn.pause` observation, never a `turn`, and says so in
    `dfo.turn_status` too, which a close leaves out."""
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with turn_segment("tr_export0019"):
        _child_span("generate_content gemini-x")
    record_turn_outcome("tr_export0019", request="q", response="a", error=error, paused=paused)
    get_client().flush()

    marks = {
        span.name: (
            _attribute(span, "langfuse.observation.level"),
            _attribute(span, "langfuse.observation.status_message"),
            _attribute(span, "langfuse.observation.metadata.dfo.turn_error"),
            _attribute(span, "langfuse.observation.metadata.dfo.turn_status"),
        )
        for span in fake_langfuse.spans()
    }
    assert marks == {
        "turn.pause" if paused else "turn": (level, status_message, error, status),
        "turn.segment": (None, None, None, None),
        "generate_content gemini-x": (None, None, None, None),
    }


def test_a_turn_closed_twice_exports_two_outcomes(fake_langfuse: FakeLangfuse) -> None:
    """Each close writes its own `turn` observation on the one trace, and
    nothing merges them."""
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    record_turn_outcome("tr_export0021", request="q", response="a")
    record_turn_outcome("tr_export0021", request="q", response="", error="abandoned")
    get_client().flush()

    turns = [span for span in fake_langfuse.spans() if span.name == "turn"]
    assert [_attribute(span, "langfuse.observation.metadata.dfo.turn_error") for span in turns] == [
        None,
        "abandoned",
    ]
    assert {span.trace_id.hex() for span in turns} == {
        Langfuse.create_trace_id(seed="tr_export0021")
    }
    assert turns[0].span_id != turns[1].span_id


# --- the session across an MCP call ------------------------------------------


def test_turn_call_meta_carries_the_session_the_turn_exports(fake_langfuse: FakeLangfuse) -> None:
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with turn_segment("tr_export0015", session_id="conv-1"):
        inside = turn_call_meta()
    get_client().flush()

    (turn,) = [span for span in fake_langfuse.spans() if span.name == "turn.segment"]
    assert inside == {SESSION_META_KEY: _attribute(turn, "session.id")}
    assert inside[SESSION_META_KEY] == "conv-1"


def test_turn_call_meta_carries_the_user_and_tags_the_turn_exports(
    fake_langfuse: FakeLangfuse,
) -> None:
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with turn_segment("tr_export0018", session_id="conv-1", user_id="u-1", tags=["studio"]):
        inside = turn_call_meta()

    assert inside == {
        SESSION_META_KEY: "conv-1",
        USER_META_KEY: "u-1",
        TAGS_META_KEY: ["studio"],
    }


def test_turn_call_meta_is_empty_outside_a_turn_session(fake_langfuse: FakeLangfuse) -> None:
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with turn_segment("tr_export0016"):
        in_turn_without_session = turn_call_meta()
    with turn_segment("tr_export0017", session_id="conv-1"):
        pass
    after_block = turn_call_meta()

    assert turn_call_meta() == {}
    assert in_turn_without_session == {}
    assert after_block == {}


def test_use_call_session_exports_the_session_on_the_current_span_and_its_children(
    fake_langfuse: FakeLangfuse,
) -> None:
    init_telemetry(service_name="mcp")
    _join_auth_check_thread()

    with get_client().start_as_current_observation(name="server", as_type="span"):
        with use_call_session("conv-1", user_id="u-1", tags=["studio"]):
            _child_span("tool_body")
            assert "langfuse_session_id" not in baggage.get_all()
        _child_span("after_call")
    get_client().flush()

    marks = {
        span.name: (
            _attribute(span, "session.id"),
            _attribute(span, "langfuse.trace.name"),
            _attribute(span, "user.id"),
            _array_attribute(span, "langfuse.trace.tags"),
        )
        for span in fake_langfuse.spans()
    }
    assert marks == {
        "server": ("conv-1", "turn", "u-1", ["studio"]),
        "tool_body": ("conv-1", "turn", "u-1", ["studio"]),
        "after_call": (None, None, None, None),
    }


@pytest.mark.parametrize("session_id", [None, ""])
def test_use_call_session_without_a_session_leaves_spans_unmarked(
    fake_langfuse: FakeLangfuse, session_id: str | None
) -> None:
    init_telemetry(service_name="mcp")
    _join_auth_check_thread()

    with (
        get_client().start_as_current_observation(name="server", as_type="span"),
        use_call_session(session_id),
    ):
        _child_span("tool_body")
    get_client().flush()

    spans = fake_langfuse.spans()
    assert {span.name for span in spans} == {"server", "tool_body"}
    assert all(_attribute(span, "session.id") is None for span in spans)
    assert all(_attribute(span, "langfuse.trace.name") is None for span in spans)


# --- MCP calls through a real fastmcp client ---------------------------------


@pytest.fixture
def fresh_mcp_tracer() -> Iterator[None]:
    """Clear the `mcp` SDK's import-time tracer, a `ProxyTracer` that keeps the
    first provider it resolved, so its spans reach this test's provider."""
    proxy = mcp_otel._tracer
    assert isinstance(proxy, trace.ProxyTracer)
    proxy._real_tracer = None
    try:
        yield
    finally:
        proxy._real_tracer = None


async def test_mcp_calls_through_fastmcp_export_no_span_whose_parent_is_missing(
    fake_langfuse: FakeLangfuse, fresh_mcp_tracer: None
) -> None:
    """A real fastmcp `Client` lists and calls a tool on an in-process server,
    once outside a turn and once inside a segment. Every exported span's
    parent is exported, except for two roots. One is the segment, whose parent
    is the SDK's synthetic one. The other is the client span of the call
    outside the turn, which has none. The listings export nothing."""
    init_telemetry(service_name="harness")
    _join_auth_check_thread()
    server = FastMCP("probe")

    @server.tool
    def echo(text: str) -> str:
        return text

    async with Client(server) as client:
        await client.list_tools()
        await client.call_tool("echo", {"text": "outside"})
        with turn_segment("tr_export0022"):
            await client.list_tools()
            await client.call_tool("echo", {"text": "inside"})
    get_client().flush()

    spans = fake_langfuse.spans()
    by_id = {span.span_id: span for span in spans}
    (segment,) = [span for span in spans if span.name == "turn.segment"]
    # Control: each call's client, `mcp` SDK and server spans all arrived.
    for trace_id in {span.trace_id for span in spans}:
        names = sorted(span.name for span in spans if span.trace_id == trace_id)
        expected = ["MCP send tools/call echo", "tools/call echo", "tools/call echo"]
        if trace_id == segment.trace_id:
            expected = sorted([*expected, "turn.segment"])
        assert names == expected
    assert len({span.trace_id for span in spans}) == 2
    orphans = [
        span.name
        for span in spans
        if span.parent_span_id
        and span.parent_span_id not in by_id
        and span.span_id != segment.span_id
    ]
    assert orphans == []
