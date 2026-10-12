"""Workspace-wide pytest configuration.

Lives at the workspace root so it applies to every member's suite, whether
run together (`just test`) or one at a time (`just test-harness`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from common import auth as mcp_auth
from common.row_budget import set_row_budget
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# presentation-mcp's `Settings` has no default for `PRESENTATION_PUBLIC_URL` and
# `presentation_mcp.server` builds its settings at import, so any suite that
# imports the server (the harness mounts it) collects only with a value set.
# Set at import time because collection runs before any fixture.
os.environ.setdefault("PRESENTATION_PUBLIC_URL", "http://localhost:8080/mcp/presentation")

# presentation-mcp's `PRESENTATION_RELEASE_REQUIRED` defaults to true, which
# retries an unreachable platform for 90 s and then fails startup. No suite
# runs a real platform, so a server started without a stub comes up tool-less
# at once instead. A test that wants the deployed behaviour passes
# `presentation_release_required=True` to `Settings`.
os.environ.setdefault("PRESENTATION_RELEASE_REQUIRED", "false")

# presentation-mcp's lifespan starts a chart worker process and, with
# `CHART_WORKER_WARM` on, has it import the renderer at once, about 0.3 s of
# CPU per server start. Most servers the suites start never draw a chart, so
# the worker starts on its first render instead.
os.environ.setdefault("CHART_WORKER_WARM", "false")

# OpenTelemetry permits exactly one global TracerProvider per process and logs
# a warning while ignoring any later one. `testpaths` collects every member's
# suite into this one process, so the provider is installed once, here, and
# every span test across every package shares it and the `exported_spans`
# fixture, which empties the exporter between them.
_SPAN_EXPORTER = InMemorySpanExporter()
_SPAN_PROVIDER = TracerProvider()
_SPAN_PROVIDER.add_span_processor(SimpleSpanProcessor(_SPAN_EXPORTER))
trace.set_tracer_provider(_SPAN_PROVIDER)


@pytest.fixture
def exported_spans() -> InMemorySpanExporter:
    """The shared in-memory exporter, emptied so a test sees only its own spans."""
    _SPAN_EXPORTER.clear()
    return _SPAN_EXPORTER


@pytest.fixture(autouse=True)
def _isolate_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run every test from an empty directory so a real `.env` can't leak in.

    `BaseServiceSettings` sets `env_file=".env"`, a relative path that
    pydantic-settings resolves against the working directory. `just env` writes
    exactly such a file at the workspace root, so without this fixture that
    file becomes an input to the suite, and assertions about default values
    pass or fail depending on what it happens to contain.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _reset_mcp_auth_gate() -> Any:
    """`common.auth` resolves its bearer-token policy once per process
    (first caller wins), and every member's suite runs in this one process,
    so a test that resolves it (directly, or by building any FastMCP app
    that calls `apply_gate`) must not leak into a later, unrelated test.
    """
    yield
    mcp_auth.reset()


@pytest.fixture(autouse=True)
def _reset_row_budget() -> Any:
    """`common.row_budget` holds one budget per process, and every member's
    suite runs in this one process, so a budget a test installs, and any
    claim it leaves open, must not reach a later test. The next
    `row_budget()` builds a fresh one from settings.
    """
    yield
    set_row_budget(None)
