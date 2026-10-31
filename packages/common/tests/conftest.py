"""Shared fixtures for the common test suite.

`_reset_materialisation_store` is autouse: this package's own tests run in
the same one-process pytest session as every service's, so a test that
calls `set_store(...)` here must not leak into a later test anywhere else
in the suite.

`isolated_telemetry` is opt-in: the telemetry tests use it to start from an
unconfigured `common.telemetry` and to leave the process-wide OpenTelemetry
tracer provider as they found it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest
from common import telemetry
from common.materialisation import set_store
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry import trace
from opentelemetry.util._once import Once


@pytest.fixture(autouse=True)
async def _reset_materialisation_store() -> Any:
    yield
    set_store(None)


@pytest.fixture
def isolated_telemetry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give a test an unconfigured `common.telemetry`, no telemetry env vars, and
    an unset global tracer provider, then restore all of it.

    OpenTelemetry allows one global tracer provider per process, and the
    repository's root `conftest.py` installs the shared in-memory one at
    import, so the provider and its set-once latch are saved and put back.
    Langfuse clients created by a test are shut down
    (which flushes them) so none outlives it.
    """
    for var in (
        *telemetry._LANGFUSE_ENV_VARS,
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        telemetry._CAPTURE_MESSAGE_CONTENT_VAR,
    ):
        monkeypatch.delenv(var, raising=False)
    saved_provider = trace._TRACER_PROVIDER
    saved_once = trace._TRACER_PROVIDER_SET_ONCE
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = Once()
    telemetry._configured = False
    telemetry._backend = None
    telemetry._event_failure_warned = False
    try:
        yield
    finally:
        LangfuseResourceManager.reset()
        # `init_telemetry` sets this variable when it was unset, which
        # `monkeypatch` does not record. `monkeypatch` restores a value it
        # removed above.
        os.environ.pop(telemetry._CAPTURE_MESSAGE_CONTENT_VAR, None)
        trace._TRACER_PROVIDER = saved_provider
        trace._TRACER_PROVIDER_SET_ONCE = saved_once
        telemetry._configured = False
        telemetry._backend = None
        telemetry._event_failure_warned = False
