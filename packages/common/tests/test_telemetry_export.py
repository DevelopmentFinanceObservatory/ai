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

import pytest
from common import telemetry
from common.telemetry import (
    SESSION_META_KEY,
    backend,
    end_turn_span,
    flush_telemetry,
    init_telemetry,
    record_event,
    start_turn_span,
    turn_call_meta,
    use_call_session,
    use_request_span,
    use_turn_span,
)
from common.uri_redaction import UriRedactingExporter
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
                state.export_headers.append(dict(self.headers))
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


def _attribute(span: Span, key: str) -> str | int | float | bool | None:
    for attribute in span.attributes:
        if attribute.key == key:
            return getattr(attribute.value, attribute.value.WhichOneof("value"))
    return None


def _child_span(name: str) -> None:
    """Open and close a span with a `gen_ai` attribute, the kind ADK and
    fastmcp emit, which Langfuse's default filter exports."""
    with trace.get_tracer("test.child").start_as_current_span(name) as child:
        child.set_attribute("gen_ai.system", "test")


def _one_turn(external_id: str) -> None:
    """Open a `turn` span, a child span inside it, and close the turn."""
    handle = start_turn_span(external_id)
    with use_turn_span(handle):
        _child_span("call_llm")
    end_turn_span(handle)
    get_client().flush()


def _assert_turn_root_exported(fake: FakeLangfuse, external_id: str) -> None:
    spans = fake.spans()
    (turn,) = [span for span in spans if span.name == "turn"]
    (child,) = [span for span in spans if span.name == "call_llm"]

    expected_trace_id = Langfuse.create_trace_id(seed=external_id)
    assert turn.trace_id.hex() == expected_trace_id
    assert _attribute(turn, "langfuse.observation.metadata.dfo.external_id") == external_id
    assert child.trace_id == turn.trace_id
    assert child.parent_span_id == turn.span_id
    # The seeded trace id comes from a synthetic remote parent that is never
    # exported, so the root is identified by Langfuse's own marker.
    assert _attribute(turn, "langfuse.internal.is_app_root") is True


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


def test_record_event_exports_a_langfuse_event_under_the_turn_span(
    fake_langfuse: FakeLangfuse,
) -> None:
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    handle = start_turn_span("tr_export0003")
    with use_turn_span(handle):
        record_event("compose.refused", metadata={"reason": "no_rows", "count": 3, "ok": False})
    end_turn_span(handle)
    get_client().flush()

    spans = fake_langfuse.spans()
    (turn,) = [span for span in spans if span.name == "turn"]
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

    handle = start_turn_span("tr_export0004")
    with use_turn_span(handle):
        record_event("compose.refused", metadata={"reason": "no_rows", "count": 3})
    end_turn_span(handle)

    (span,) = exporter.get_finished_spans()
    (event,) = span.events
    assert span.name == "turn"
    assert event.name == "compose.refused"
    assert dict(event.attributes or {}) == {"reason": "no_rows", "count": 3, "level": "WARNING"}


async def test_a_turn_continued_in_a_second_task_exports_as_one_trace(
    fake_langfuse: FakeLangfuse, caplog: pytest.LogCaptureFixture
) -> None:
    """A turn opened in one task, continued by the requests that follow it in
    two more tasks, and closed in a fourth (a turn paused for approval and
    resumed by a separate request) is one trace with one `turn` span, and
    spans and events from every task nest under it. The turn's Langfuse session
    id reaches the spans of both continuing tasks and stops with each task's
    block."""
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    async def open_turn() -> object:
        return start_turn_span("tr_export0005", session_id="conv-1")

    async def first_segment(handle: object) -> None:
        with use_turn_span(handle):
            _child_span("plan_call")
        _child_span("plan_call_after_block")

    async def resumed_segment(handle: object) -> None:
        with use_turn_span(handle):
            _child_span("execute_call")
            record_event("compose.write_refused", metadata={"code": "missing_field"})
        _child_span("execute_call_after_block")

    async def close_turn(handle: object) -> None:
        end_turn_span(handle)

    with caplog.at_level(logging.WARNING):
        handle = await asyncio.create_task(open_turn())
        await asyncio.create_task(first_segment(handle))
        await asyncio.create_task(resumed_segment(handle))
        await asyncio.create_task(close_turn(handle))
    get_client().flush()

    exported = fake_langfuse.spans()
    spans = [span for span in exported if not span.name.endswith("_after_block")]
    after_block = [span for span in exported if span.name.endswith("_after_block")]
    (turn,) = [span for span in spans if span.name == "turn"]
    assert turn.trace_id.hex() == Langfuse.create_trace_id(seed="tr_export0005")
    assert {span.trace_id for span in spans} == {turn.trace_id}
    for name in ("plan_call", "execute_call", "compose.write_refused"):
        (span,) = [span for span in spans if span.name == name]
        assert span.parent_span_id == turn.span_id, name
    # Every span but the root has a parent inside the trace.
    assert _attribute(turn, "langfuse.internal.is_app_root") is True
    assert [
        span.name for span in spans if _attribute(span, "langfuse.internal.is_app_root") is True
    ] == ["turn"]
    assert not [r for r in caplog.records if r.name == "opentelemetry.context"]
    for name in ("turn", "plan_call", "execute_call", "compose.write_refused"):
        (span,) = [span for span in spans if span.name == name]
        assert _attribute(span, "session.id") == "conv-1", name
    assert len(after_block) == 2
    for span in after_block:
        assert _attribute(span, "session.id") is None, span.name


async def test_flush_telemetry_exports_queued_spans_without_waiting_for_the_interval(
    fake_langfuse: FakeLangfuse, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SDK's batch worker wakes every `LANGFUSE_FLUSH_INTERVAL` seconds;
    with a five-minute interval only `flush_telemetry` can deliver the spans."""
    monkeypatch.setenv("LANGFUSE_FLUSH_INTERVAL", "300")
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    handle = start_turn_span("tr_export0006")
    with use_turn_span(handle):
        _child_span("call_llm")
    end_turn_span(handle)
    assert fake_langfuse.spans() == []

    await flush_telemetry()

    assert {span.name for span in fake_langfuse.spans()} == {"turn", "call_llm"}


async def test_flush_telemetry_returns_at_its_bound_when_the_server_hangs(
    fake_langfuse: FakeLangfuse, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGFUSE_FLUSH_INTERVAL", "300")
    init_telemetry(service_name="harness")
    _join_auth_check_thread()
    handle = start_turn_span("tr_export0007")
    with use_turn_span(handle):
        _child_span("call_llm")
    end_turn_span(handle)
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
    assert {span.name for span in fake_langfuse.spans()} == {"turn", "call_llm"}


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

    first = start_turn_span("tr_export0008")
    with use_turn_span(first):
        _child_span("first_call")
    end_turn_span(first)
    running = asyncio.create_task(flush_telemetry())
    assert await asyncio.to_thread(in_window.wait, 30)
    assert {span.name for span in fake_langfuse.spans()} == {"turn", "first_call"}

    second = start_turn_span("tr_export0009")
    with use_turn_span(second):
        _child_span("second_call")
    end_turn_span(second)
    await asyncio.wait_for(flush_telemetry(), timeout=2)  # returns at once: one is running
    assert "second_call" not in {span.name for span in fake_langfuse.spans()}

    leave_window.set()
    await asyncio.wait_for(running, timeout=30)

    assert "second_call" in {span.name for span in fake_langfuse.spans()}
    assert len(passes) == 2
    assert not telemetry._flush_running


def _spans_with_an_unexported_parent(spans: list[Span]) -> list[Span]:
    """The spans whose parent was not exported."""
    exported = {span.span_id for span in spans}
    return [span for span in spans if span.parent_span_id not in exported]


async def test_adks_invocation_span_is_exported_and_other_bare_spans_are_not(
    fake_langfuse: FakeLangfuse,
) -> None:
    """ADK wraps each run in an attribute-less `invocation` span that is the
    parent of its `invoke_workflow` span. The Langfuse default filter drops it;
    the client built by `init_telemetry` keeps it (matched on scope and name),
    while an attribute-less span from any other scope, or of another name in
    ADK's scope, is dropped."""
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    handle = start_turn_span("tr_export0010")
    with use_turn_span(handle):
        adk = trace.get_tracer("gcp.vertex.agent")
        with adk.start_as_current_span("invocation"):
            with adk.start_as_current_span("invoke_workflow wf") as workflow:
                workflow.set_attribute("gen_ai.operation.name", "invoke_workflow")
            with adk.start_as_current_span("bare_adk_span"):
                pass
        other = trace.get_tracer("some.other.library")
        with other.start_as_current_span("invocation"), other.start_as_current_span("bare"):
            pass
    end_turn_span(handle)
    get_client().flush()

    spans = fake_langfuse.spans()
    assert sorted(span.name for span in spans) == ["invocation", "invoke_workflow wf", "turn"]
    (turn,) = [span for span in spans if span.name == "turn"]
    (invocation,) = [span for span in spans if span.name == "invocation"]
    (workflow,) = [span for span in spans if span.name == "invoke_workflow wf"]
    assert invocation.parent_span_id == turn.span_id
    assert workflow.parent_span_id == invocation.span_id
    # The only parent that was never exported is the SDK's synthetic one for `turn`.
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
    credentials and SDK headers the SDK's default exporter would."""
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    _one_turn("tr_export0011")

    headers = fake_langfuse.export_headers[0]
    assert headers["Authorization"] == "Basic " + base64.b64encode(b"pk-lf-test:sk-lf-test").decode(
        "ascii"
    )
    assert headers["x-langfuse-public-key"] == "pk-lf-test"
    assert headers["x-langfuse-sdk-name"] == "python"
    assert headers["x-langfuse-sdk-version"] == langfuse_version


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
    [{}, {"LANGFUSE_TIMEOUT": "7", "LANGFUSE_OTEL_TRACES_EXPORT_PATH": "custom/v1/traces"}],
)
def test_the_langfuse_exporter_matches_the_one_the_sdk_builds_by_default(
    fake_langfuse: FakeLangfuse, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    """`_langfuse_exporter` copies the SDK's defaults, so an SDK upgrade that
    changes the endpoint, headers or timeout it would use fails here."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    ours = telemetry._langfuse_exporter()
    sdk = _sdk_default_exporter()

    assert isinstance(ours, UriRedactingExporter)
    inner = ours._exporter
    assert isinstance(inner, OTLPSpanExporter)
    assert inner._endpoint == sdk._endpoint
    assert inner._headers == sdk._headers
    assert inner._timeout == sdk._timeout


_USAGE = {
    "gen_ai.usage.input_tokens": 120,
    "gen_ai.usage.output_tokens": 30,
    "gen_ai.usage.reasoning.output_tokens": 12,
    "gen_ai.usage.experimental.system_instruction_tokens": 40,
}


_LLM_RESPONSE_KEY = "gcp.vertex.agent.llm_response"
_LLM_RESPONSE = json.dumps(
    {"content": {"parts": [{"text": "hi"}]}, "usage_metadata": {"cached_content_token_count": 7}}
)


def _model_call(call_llm_scope: str, *, call_llm_usage: dict[str, int] = _USAGE) -> None:
    """Open ADK's two nested spans for one model call: `call_llm` from
    `call_llm_scope`, carrying `call_llm_usage`, and the `generate_content
    gemini-x` inference span inside it, carrying the token usage."""
    handle = start_turn_span("tr_export0012")
    call_llm_tracer = trace.get_tracer(call_llm_scope)
    inference_tracer = trace.get_tracer(telemetry._ADK_SCOPE_NAME)
    with (
        use_turn_span(handle),
        call_llm_tracer.start_as_current_span("call_llm") as call_llm,
    ):
        call_llm.set_attributes(
            {
                "gen_ai.request.model": "gemini-x",
                _LLM_RESPONSE_KEY: _LLM_RESPONSE,
                **call_llm_usage,
            }
        )
        with inference_tracer.start_as_current_span("generate_content gemini-x") as inference:
            inference.set_attributes({"gen_ai.request.model": "gemini-x", **_USAGE})
    end_turn_span(handle)
    get_client().flush()


def _usage_on(span: Span) -> dict[str, str | int | float | bool | None]:
    return {
        attribute.key: _attribute(span, attribute.key)
        for attribute in span.attributes
        if attribute.key.startswith("gen_ai.usage.")
    }


@pytest.mark.parametrize("call_llm_usage", [_USAGE, {}], ids=["with_usage", "without_usage"])
def test_adks_call_llm_span_exports_as_a_plain_span_with_empty_usage_details(
    fake_langfuse: FakeLangfuse, call_llm_usage: dict[str, int]
) -> None:
    """Langfuse prices every generation, and types a span that carries a model
    as one. ADK writes the same usage on `call_llm` and on the `generate_content`
    span inside it, so the export types every `call_llm` as a span, usage or
    not, and gives it an explicit empty `usage_details`, which Langfuse reads
    before any other usage source. The `gen_ai.usage.*` attributes are kept."""
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    _model_call(telemetry._ADK_SCOPE_NAME, call_llm_usage=call_llm_usage)

    spans = fake_langfuse.spans()
    (call_llm,) = [span for span in spans if span.name == "call_llm"]
    (inference,) = [span for span in spans if span.name == "generate_content gemini-x"]
    assert _attribute(call_llm, "langfuse.observation.type") == "span"
    assert _usage_on(call_llm) == call_llm_usage
    assert _attribute(call_llm, "langfuse.observation.usage_details") == "{}"
    assert _attribute(call_llm, _LLM_RESPONSE_KEY) == _LLM_RESPONSE
    assert _attribute(call_llm, "gen_ai.request.model") == "gemini-x"
    assert _attribute(inference, "langfuse.observation.type") is None
    assert _attribute(inference, "langfuse.observation.usage_details") is None
    assert _usage_on(inference) == _USAGE
    assert inference.parent_span_id == call_llm.span_id


def test_a_call_llm_span_from_another_scope_exports_untouched(
    fake_langfuse: FakeLangfuse,
) -> None:
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    _model_call("some.other.library")

    (call_llm,) = [span for span in fake_langfuse.spans() if span.name == "call_llm"]
    assert _usage_on(call_llm) == _USAGE
    assert _attribute(call_llm, "langfuse.observation.type") is None
    assert _attribute(call_llm, "langfuse.observation.usage_details") is None


def test_a_failing_span_mask_leaves_the_export_batch_intact_and_warns_once(
    fake_langfuse: FakeLangfuse, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The SDK drops a whole export batch when its mask hook raises, so the hook
    catches its own failures and the spans arrive as they were. A failure brings
    double pricing back, so the first one in a process logs at WARNING and later
    ones at DEBUG."""

    def fail(**_: object) -> None:
        raise RuntimeError("patch construction failed")

    monkeypatch.setattr("langfuse.types.OtelSpanPatch", fail)
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    with caplog.at_level(logging.DEBUG, logger=telemetry.logger.name):
        _model_call(telemetry._ADK_SCOPE_NAME)
        _model_call(telemetry._ADK_SCOPE_NAME)

    spans = fake_langfuse.spans()
    assert {span.name for span in spans} == {"turn", "call_llm", "generate_content gemini-x"}
    for call_llm in [span for span in spans if span.name == "call_llm"]:
        assert _usage_on(call_llm) == _USAGE
    failures = [record for record in caplog.records if "call_llm" in record.message]
    assert [record.levelno for record in failures] == [logging.WARNING, logging.DEBUG]


# --- one Langfuse session per conversation ------------------------------------


def _adk_child_span(name: str) -> None:
    """A span like ADK's `call_llm`: a separate tracer, with ADK's own
    per-turn conversation id."""
    with trace.get_tracer("gcp.vertex.agent").start_as_current_span(name) as span:
        span.set_attribute("gen_ai.conversation.id", "conv-1#t1")


def test_a_turn_with_a_session_id_exports_it_on_the_turn_and_on_spans_a_library_opens(
    fake_langfuse: FakeLangfuse,
) -> None:
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    handle = start_turn_span("tr_export0013", session_id="conv-1")
    with use_turn_span(handle):
        _adk_child_span("call_llm")
    end_turn_span(handle)
    get_client().flush()

    spans = fake_langfuse.spans()
    assert {span.name for span in spans} == {"turn", "call_llm"}
    for span in spans:
        assert _attribute(span, "session.id") == "conv-1", span.name


def test_a_turn_without_a_session_id_exports_no_langfuse_session(
    fake_langfuse: FakeLangfuse,
) -> None:
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    handle = start_turn_span("tr_export0014")
    with use_turn_span(handle):
        _child_span("call_llm")
    end_turn_span(handle)
    get_client().flush()

    spans = fake_langfuse.spans()
    assert {span.name for span in spans} == {"turn", "call_llm"}
    for span in spans:
        assert _attribute(span, "session.id") is None, span.name


# --- the session across an MCP call ------------------------------------------


def test_turn_call_meta_carries_the_session_the_turn_exports(fake_langfuse: FakeLangfuse) -> None:
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    handle = start_turn_span("tr_export0015", session_id="conv-1")
    with use_turn_span(handle):
        inside = turn_call_meta()
    end_turn_span(handle)
    get_client().flush()

    (turn,) = [span for span in fake_langfuse.spans() if span.name == "turn"]
    assert inside == {SESSION_META_KEY: _attribute(turn, "session.id")}
    assert inside[SESSION_META_KEY] == "conv-1"


def test_turn_call_meta_is_empty_outside_a_turn_session(fake_langfuse: FakeLangfuse) -> None:
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    without_session = start_turn_span("tr_export0016")
    with use_turn_span(without_session):
        in_turn_without_session = turn_call_meta()
    end_turn_span(without_session)
    with_session = start_turn_span("tr_export0017", session_id="conv-1")
    with use_turn_span(with_session):
        pass
    after_block = turn_call_meta()
    end_turn_span(with_session)

    assert turn_call_meta() == {}
    assert in_turn_without_session == {}
    assert after_block == {}


def test_use_call_session_exports_the_session_on_the_current_span_and_its_children(
    fake_langfuse: FakeLangfuse,
) -> None:
    init_telemetry(service_name="mcp")
    _join_auth_check_thread()

    with get_client().start_as_current_observation(name="server", as_type="span"):
        with use_call_session("conv-1"):
            _child_span("tool_body")
            assert "langfuse_session_id" not in baggage.get_all()
        _child_span("after_call")
    get_client().flush()

    sessions = {span.name: _attribute(span, "session.id") for span in fake_langfuse.spans()}
    assert sessions == {"server": "conv-1", "tool_body": "conv-1", "after_call": None}


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
