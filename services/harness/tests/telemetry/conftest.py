"""Fixtures for the harness tests that run the real Langfuse SDK against a
local stand-in for the Langfuse server, so a test can assert on the spans that
reach the wire.

`isolated_telemetry` starts a test from an unconfigured `common.telemetry`
with no telemetry env vars, no global tracer provider and no tracer left bound
by ADK's module-level tracers, and puts back what it found: OpenTelemetry
allows one global provider per process and other suites in the session install
their own. `fake_langfuse` points the SDK's env vars at a local server that
answers the credentials check and records every OTLP trace export. Both mirror
`packages/common/tests/conftest.py` and `test_telemetry_export.py`; a fixture
cannot be imported across suites.
"""

from __future__ import annotations

import gzip
import json
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from common import telemetry
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry import trace
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span
from opentelemetry.sdk import trace as sdk_trace
from opentelemetry.util._once import Once

_PROJECTS_BODY = {
    "data": [
        {"id": "p", "name": "proj", "organization": {"id": "o", "name": "org"}, "metadata": {}}
    ]
}


@dataclass
class FakeLangfuse:
    """`exports` holds the decoded body of every OTLP trace export."""

    exports: list[ExportTraceServiceRequest] = field(default_factory=list)

    def scoped_spans(self) -> list[tuple[str, Span]]:
        """Every exported span with the name of its instrumentation scope."""
        return [
            (scope_spans.scope.name, span)
            for request in self.exports
            for resource_spans in request.resource_spans
            for scope_spans in resource_spans.scope_spans
            for span in scope_spans.spans
        ]

    def spans(self) -> list[Span]:
        return [span for _, span in self.scoped_spans()]


def _adk_tracer_slots() -> Iterator[tuple[Any, str, Any]]:
    """Every module-level `(module, attribute, tracer)` in the `google.adk` modules."""
    for name, module in list(sys.modules.items()):
        if not name.startswith("google.adk"):
            continue
        for attribute, value in list(getattr(module, "__dict__", {}).items()):
            if isinstance(value, trace.ProxyTracer | sdk_trace.Tracer):
                yield module, attribute, value


def _unbind_adk_tracers() -> list[tuple[Any, str, Any]]:
    """Make ADK's module-level tracers resolve the current global provider, and
    return the SDK tracers that were replaced so `_rebind_adk_tracers` can put
    them back.

    ADK builds each tracer once at import. When no provider is global then, it
    is an OpenTelemetry `ProxyTracer`, which keeps the first real tracer it
    resolves for the life of the process; when the shared span provider from
    the root `conftest.py` is already global, it is an SDK tracer bound to that
    provider for good. Left alone, either would keep feeding whichever provider
    it first met, and a test that installs its own provider would see none of
    ADK's spans. A proxy is cleared, and an SDK tracer is swapped for a proxy
    that resolves the provider current when it is first used.
    """
    replaced: list[tuple[Any, str, Any]] = []
    for module, attribute, value in _adk_tracer_slots():
        if isinstance(value, trace.ProxyTracer):
            value._real_tracer = None
            continue
        info = value.instrumentation_info
        proxy = trace.ProxyTracer(info.name, info.version, info.schema_url)
        setattr(module, attribute, proxy)
        replaced.append((module, attribute, value))
    return replaced


def _rebind_adk_tracers(replaced: list[tuple[Any, str, Any]]) -> None:
    """Put back the SDK tracers `_unbind_adk_tracers` replaced, and clear any proxy."""
    for module, attribute, original in replaced:
        setattr(module, attribute, original)
    for _, _, value in _adk_tracer_slots():
        if isinstance(value, trace.ProxyTracer):
            value._real_tracer = None


@pytest.fixture
def isolated_telemetry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for var in (*telemetry._LANGFUSE_ENV_VARS, "OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_SDK_DISABLED"):
        monkeypatch.delenv(var, raising=False)
    saved_provider = trace._TRACER_PROVIDER
    saved_once = trace._TRACER_PROVIDER_SET_ONCE
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = Once()
    replaced_tracers = _unbind_adk_tracers()
    telemetry._configured = False
    telemetry._backend = None
    telemetry._event_failure_warned = False
    try:
        yield
    finally:
        LangfuseResourceManager.reset()
        trace._TRACER_PROVIDER = saved_provider
        trace._TRACER_PROVIDER_SET_ONCE = saved_once
        _rebind_adk_tracers(replaced_tracers)
        telemetry._configured = False
        telemetry._backend = None
        telemetry._event_failure_warned = False


@pytest.fixture
def fake_langfuse(
    isolated_telemetry: None, monkeypatch: pytest.MonkeyPatch
) -> Iterator[FakeLangfuse]:
    state = FakeLangfuse()

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
                self._reply(200, json.dumps(_PROJECTS_BODY).encode())
            else:
                self._reply(404)

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            if self.path == "/api/public/otel/v1/traces":
                request = ExportTraceServiceRequest()
                request.ParseFromString(body)
                state.exports.append(request)
                self._reply(200)
            else:
                self._reply(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    try:
        yield state
    finally:
        # Shutting the client down flushes it, which needs the server up. It
        # leaves the tracer provider's batch thread running, and libraries that
        # cache the provider's tracer (ADK does) would keep feeding it spans
        # for later tests to export at a server that is gone, so the provider
        # is shut down as well.
        provider = trace.get_tracer_provider()
        LangfuseResourceManager.reset()
        shutdown = getattr(provider, "shutdown", None)
        if shutdown is not None:
            shutdown()
        server.shutdown()
        server.server_close()
