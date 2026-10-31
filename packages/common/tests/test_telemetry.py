import asyncio
import contextvars
import logging
import os
import threading
from unittest.mock import MagicMock, call

import langfuse
import pytest
from common import telemetry
from common.telemetry import (
    backend,
    current_trace_id,
    end_turn_span,
    flush_telemetry,
    init_telemetry,
    record_event,
    record_span,
    start_turn_span,
    turn_call_meta,
    use_call_session,
    use_request_span,
    use_turn_span,
)
from common.uri_redaction import UriRedactingExporter
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture(autouse=True)
def _isolated(isolated_telemetry):
    """`init_telemetry` is idempotent via module-level state; every test here
    starts unconfigured, with no telemetry env vars and the global tracer
    provider unset."""


@pytest.fixture
def langfuse_env(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://example.invalid")


@pytest.fixture
def langfuse_client(monkeypatch):
    """A stand-in for the SDK client `init_telemetry` builds. `Langfuse` is
    replaced by a factory that returns it and records the keyword arguments
    the client was built with in `client.built_with`."""
    client = MagicMock()
    client.auth_check.return_value = True
    client.built_with = []

    def build(**kwargs):
        client.built_with.append(kwargs)
        return client

    monkeypatch.setattr(langfuse, "Langfuse", build)
    return client


@pytest.fixture
def live_span():
    """A real, recording span made current, as the one `record_event` runs
    under in a turn."""
    with TracerProvider().get_tracer("test.live").start_as_current_span("live"):
        yield


def _join_auth_check_thread():
    for thread in threading.enumerate():
        if thread.name == telemetry._AUTH_CHECK_THREAD_NAME:
            thread.join(timeout=10)
            assert not thread.is_alive()


def test_noop_without_any_env_vars(caplog):
    with caplog.at_level("INFO"):
        init_telemetry(service_name="harness")

    assert any("Telemetry disabled" in record.message for record in caplog.records)
    assert backend() == "off"


def test_backend_is_off_before_init():
    assert backend() == "off"


def test_off_is_logged_at_warning(caplog):
    with caplog.at_level("WARNING"):
        init_telemetry(service_name="harness")

    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert "Telemetry disabled for harness" in record.message


def test_is_idempotent(caplog):
    with caplog.at_level("INFO"):
        init_telemetry(service_name="harness")
        caplog.clear()
        init_telemetry(service_name="harness")

    assert caplog.records == []


def test_partial_langfuse_env_vars_does_not_enable_langfuse(monkeypatch, caplog):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-secret-value")

    with caplog.at_level("INFO"):
        init_telemetry(service_name="harness")

    assert backend() == "off"
    (record,) = [r for r in caplog.records if "Telemetry disabled" in r.message]
    assert record.levelno == logging.WARNING
    # The warning names the variables to set and leaves out the key's value.
    assert "LANGFUSE_SECRET_KEY" in record.message
    assert "LANGFUSE_BASE_URL" in record.message
    assert "pk-secret-value" not in record.message


def test_otel_endpoint_resolves_to_the_otel_backend(monkeypatch, caplog):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setattr(telemetry, "_init_otel", lambda *, service_name: None)

    with caplog.at_level("INFO"):
        init_telemetry(service_name="harness")

    assert backend() == "otel"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_langfuse_env_vars_resolve_to_the_langfuse_backend(langfuse_env, langfuse_client, caplog):
    with caplog.at_level("INFO"):
        init_telemetry(service_name="harness")
    _join_auth_check_thread()

    assert backend() == "langfuse"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("configured for harness via Langfuse" in r.message for r in caplog.records)


@pytest.mark.parametrize("failure", ["returns_false", "raises"])
def test_a_failing_auth_check_warns_and_leaves_langfuse_active(
    langfuse_env, langfuse_client, caplog, failure
):
    """`start_turn_span` and `record_event` read `_backend` alone, and the
    exporter is attached as soon as `get_client()` returns, so a rejected or
    unreachable credentials check only logs a WARNING and the backend stays
    `langfuse`."""
    if failure == "returns_false":
        langfuse_client.auth_check.return_value = False
    else:
        langfuse_client.auth_check.side_effect = RuntimeError("503 from langfuse")

    with caplog.at_level("INFO"):
        init_telemetry(service_name="harness")
        _join_auth_check_thread()

    assert backend() == "langfuse"
    (record,) = [r for r in caplog.records if "auth check" in r.message]
    assert record.levelno == logging.WARNING
    assert (record.exc_info is not None) == (failure == "raises")


def test_the_auth_check_does_not_block_startup(langfuse_env, langfuse_client):
    """`auth_check` blocks for the SDK's request timeout; a slow or hung
    Langfuse must not delay `init_telemetry`, so the check runs off-thread."""
    release = threading.Event()
    langfuse_client.auth_check.side_effect = lambda: release.wait(timeout=10)

    try:
        init_telemetry(service_name="harness")

        assert backend() == "langfuse"
        assert any(t.name == telemetry._AUTH_CHECK_THREAD_NAME for t in threading.enumerate())
    finally:
        release.set()
        _join_auth_check_thread()


def test_a_client_that_cannot_be_created_leaves_telemetry_off(langfuse_env, monkeypatch, caplog):
    def _boom(**kwargs):
        raise RuntimeError("no client")

    monkeypatch.setattr(langfuse, "Langfuse", _boom)

    with caplog.at_level("WARNING"):
        init_telemetry(service_name="harness")  # must not raise

    assert backend() == "off"
    assert any(r.exc_info is not None for r in caplog.records)


@pytest.mark.parametrize(
    ("var", "value", "reason"),
    [
        ("LANGFUSE_TRACING_ENABLED", "false", "LANGFUSE_TRACING_ENABLED=false"),
        ("LANGFUSE_TRACING_ENABLED", "False", "LANGFUSE_TRACING_ENABLED=false"),
        ("LANGFUSE_SAMPLE_RATE", "0", "LANGFUSE_SAMPLE_RATE=0"),
        ("LANGFUSE_SAMPLE_RATE", "0.0", "LANGFUSE_SAMPLE_RATE=0"),
    ],
)
def test_langfuse_env_that_makes_the_sdk_export_nothing_resolves_to_off(
    langfuse_env, monkeypatch, caplog, var, value, reason
):
    """The SDK uses a no-op tracer for `LANGFUSE_TRACING_ENABLED=false` and
    samples nothing at rate 0, so the backend is `off`, no client is created,
    and the WARNING names the variable."""
    monkeypatch.setenv(var, value)

    def _unexpected(**kwargs):
        raise AssertionError("a disabled Langfuse must not create a client")

    monkeypatch.setattr(langfuse, "Langfuse", _unexpected)

    with caplog.at_level("INFO"):
        init_telemetry(service_name="harness")

    assert backend() == "off"
    (record,) = [r for r in caplog.records if "Telemetry disabled" in r.message]
    assert record.levelno == logging.WARNING
    assert reason in record.message


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("LANGFUSE_TRACING_ENABLED", "true"),
        ("LANGFUSE_SAMPLE_RATE", "0.25"),
        ("LANGFUSE_SAMPLE_RATE", "1"),
    ],
)
def test_langfuse_env_that_leaves_export_on_resolves_to_langfuse(
    langfuse_env, langfuse_client, monkeypatch, var, value
):
    monkeypatch.setenv(var, value)

    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    assert backend() == "langfuse"


@pytest.mark.parametrize("value", ["true", "TRUE", " True "])
@pytest.mark.parametrize("configured", ["langfuse", "otel"])
def test_otel_sdk_disabled_resolves_to_off_for_either_backend(
    monkeypatch, caplog, langfuse_client, configured, value
):
    """The OpenTelemetry SDK's tracer provider hands out no-op tracers when
    `OTEL_SDK_DISABLED` is true (compared lowercased and stripped), so neither
    backend exports anything and the state is `off`, with the variable named."""
    if configured == "langfuse":
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://example.invalid")
    else:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")

    def _unexpected(**kwargs):
        raise AssertionError("a disabled SDK must not bring up an exporter")

    monkeypatch.setattr(langfuse, "Langfuse", _unexpected)
    monkeypatch.setattr(telemetry, "_init_otel", _unexpected)
    monkeypatch.setenv("OTEL_SDK_DISABLED", value)

    with caplog.at_level("INFO"):
        init_telemetry(service_name="harness")

    assert backend() == "off"
    (record,) = [r for r in caplog.records if "Telemetry disabled" in r.message]
    assert record.levelno == logging.WARNING
    assert "OTEL_SDK_DISABLED=true" in record.message


@pytest.mark.parametrize("value", ["false", "", "0"])
def test_otel_sdk_disabled_leaves_the_backend_on_unless_it_is_true(
    langfuse_env, langfuse_client, monkeypatch, value
):
    monkeypatch.setenv("OTEL_SDK_DISABLED", value)

    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    assert backend() == "langfuse"


def test_the_langfuse_client_is_built_with_a_redacting_exporter_and_the_default_filter(
    langfuse_env, langfuse_client
):
    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    (kwargs,) = langfuse_client.built_with
    assert kwargs.keys() == {"span_exporter"}
    assert isinstance(kwargs["span_exporter"], UriRedactingExporter)


@pytest.mark.parametrize("backend_env", ["langfuse", "otel"])
def test_a_backend_that_comes_up_turns_on_message_content_on_spans(
    langfuse_client, monkeypatch, backend_env
):
    if backend_env == "langfuse":
        for var in telemetry._LANGFUSE_ENV_VARS:
            monkeypatch.setenv(var, "https://example.invalid")
    else:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
        monkeypatch.setattr(telemetry, "_init_otel", lambda *, service_name: None)

    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    assert backend() == backend_env
    assert os.environ[telemetry._CAPTURE_MESSAGE_CONTENT_VAR] == "SPAN_ONLY"


@pytest.mark.parametrize("value", ["NO_CONTENT", ""])
def test_message_content_set_in_the_environment_is_left_as_it_is(
    langfuse_env, langfuse_client, monkeypatch, value
):
    monkeypatch.setenv(telemetry._CAPTURE_MESSAGE_CONTENT_VAR, value)

    init_telemetry(service_name="harness")
    _join_auth_check_thread()

    assert os.environ[telemetry._CAPTURE_MESSAGE_CONTENT_VAR] == value


def test_no_backend_leaves_message_content_unset():
    init_telemetry(service_name="harness")

    assert backend() == "off"
    assert telemetry._CAPTURE_MESSAGE_CONTENT_VAR not in os.environ


def test_an_existing_langfuse_client_is_reported_and_startup_continues(
    langfuse_env, langfuse_client, monkeypatch, caplog
):
    monkeypatch.setattr(telemetry, "_langfuse_client_exists", lambda: True)

    with caplog.at_level("INFO"):
        init_telemetry(service_name="harness")
    _join_auth_check_thread()

    assert backend() == "langfuse"
    (record,) = [r for r in caplog.records if "already exists" in r.message]
    assert record.levelno == logging.WARNING


def test_no_existing_langfuse_client_means_no_warning(langfuse_env, langfuse_client, caplog):
    with caplog.at_level("INFO"):
        init_telemetry(service_name="harness")
    _join_auth_check_thread()

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_the_client_registry_decides_whether_a_langfuse_client_exists(monkeypatch):
    assert telemetry._langfuse_client_exists() is False
    monkeypatch.setattr(LangfuseResourceManager, "_instances", {"pk-lf-x": MagicMock()})
    assert telemetry._langfuse_client_exists() is True


def _span(scope: str, name: str):
    """A live SDK span, which is a `ReadableSpan`, from a tracer of `scope`."""
    return TracerProvider().get_tracer(scope).start_span(name)


def test_a_failed_auth_check_thread_start_warns_and_leaves_langfuse_active(
    langfuse_env, monkeypatch, caplog
):
    """The exporter is attached when `get_client()` returns, so the backend is
    `langfuse` whether or not the diagnostic thread starts."""
    monkeypatch.setattr(langfuse, "Langfuse", lambda **kwargs: MagicMock())

    def _cannot_start(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", _cannot_start)

    with caplog.at_level("INFO"):
        init_telemetry(service_name="harness")

    assert backend() == "langfuse"
    (record,) = [r for r in caplog.records if "could not be started" in r.message]
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None


# --- start_turn_span / end_turn_span ---------------------------------------


def test_start_turn_span_is_a_noop_with_no_backend():
    assert start_turn_span("tr_abc123") is None


def test_start_turn_span_langfuse_uses_a_seeded_trace_id(monkeypatch):
    """The Langfuse path must derive the real trace id from the wire id via
    `Langfuse.create_trace_id` -- the SDK's documented external-id
    correlation pattern -- and pass it through `trace_context`, so a person
    holding only the wire id can recompute the same trace id later."""
    telemetry._backend = "langfuse"
    seeds_seen = []

    def fake_create_trace_id(*, seed):
        seeds_seen.append(seed)
        return "d" * 32

    fake_span_cm = MagicMock()
    fake_client = MagicMock()
    fake_client.start_as_current_observation.return_value = fake_span_cm
    monkeypatch.setattr(langfuse.Langfuse, "create_trace_id", staticmethod(fake_create_trace_id))
    monkeypatch.setattr(langfuse, "get_client", lambda: fake_client)

    handle = start_turn_span("tr_abc123")

    assert seeds_seen == ["tr_abc123"]
    _, kwargs = fake_client.start_as_current_observation.call_args
    assert kwargs["trace_context"] == {"trace_id": "d" * 32}
    assert kwargs["metadata"] == {"dfo.external_id": "tr_abc123"}
    fake_span_cm.__enter__.assert_called_once()
    handle.context.run(handle.scope.__exit__, None, None, None)
    fake_span_cm.__exit__.assert_called_once()


def test_start_turn_span_otel_sets_the_id_as_an_attribute(monkeypatch):
    """Without Langfuse there is no `create_trace_id`, so the wire id goes on
    the span as a plain attribute."""
    telemetry._backend = "otel"
    fake_span_cm = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_as_current_span.return_value = fake_span_cm
    monkeypatch.setattr(otel_trace, "get_tracer", lambda name: fake_tracer)

    handle = start_turn_span("tr_abc123")

    _, kwargs = fake_tracer.start_as_current_span.call_args
    assert kwargs["attributes"] == {"dfo.external_id": "tr_abc123"}
    fake_span_cm.__enter__.assert_called_once()
    assert handle.scope is fake_span_cm


def test_start_turn_span_swallows_errors(monkeypatch):
    """Telemetry opening a span must never be the reason a turn fails."""

    def _boom(*, seed):
        raise RuntimeError("boom")

    telemetry._backend = "langfuse"
    monkeypatch.setattr(langfuse.Langfuse, "create_trace_id", staticmethod(_boom))

    assert start_turn_span("tr_abc123") is None


def test_end_turn_span_is_a_noop_for_none():
    end_turn_span(None)  # must not raise


def test_end_turn_span_swallows_errors(monkeypatch):
    """Telemetry closing a span must never break the caller."""
    telemetry._backend = "otel"
    fake_span_cm = MagicMock()
    fake_span_cm.__exit__.side_effect = ValueError("close failed")
    fake_tracer = MagicMock()
    fake_tracer.start_as_current_span.return_value = fake_span_cm
    monkeypatch.setattr(otel_trace, "get_tracer", lambda name: fake_tracer)
    handle = start_turn_span("tr_abc123")

    end_turn_span(handle)  # must not raise

    fake_span_cm.__exit__.assert_called_once_with(None, None, None)


def test_end_turn_span_closes_the_span_when_recording_its_input_and_output_fails(monkeypatch):
    """A turn whose input and output cannot be recorded still ends its span."""
    telemetry._backend = "langfuse"
    fake_span_cm = MagicMock()
    fake_span_cm.__enter__.return_value.update.side_effect = ValueError("update failed")
    fake_client = MagicMock()
    fake_client.start_as_current_observation.return_value = fake_span_cm
    monkeypatch.setattr(langfuse, "get_client", lambda: fake_client)
    handle = start_turn_span("tr_abc123")

    end_turn_span(handle, request="q", response="a")  # must not raise

    fake_span_cm.__enter__.return_value.update.assert_called_once_with(
        input="q", output="a", level=None, status_message=None
    )
    fake_span_cm.__exit__.assert_called_once()


async def test_end_turn_span_runs_in_the_context_the_span_was_opened_in(monkeypatch):
    """The scope is exited in the copy of the context it was entered in, from
    whichever task closes the turn, so its context tokens are always reset in
    the context that made them."""
    telemetry._backend = "otel"
    seen: dict[str, object] = {}
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker", default="unset")

    class Scope:
        def __enter__(self):
            marker.set("entered")

        def __exit__(self, *exc_info):
            seen["marker_on_exit"] = marker.get()

    fake_tracer = MagicMock()
    fake_tracer.start_as_current_span.return_value = Scope()
    monkeypatch.setattr(otel_trace, "get_tracer", lambda name: fake_tracer)

    handle = await asyncio.create_task(_start("tr_abc123"))
    assert marker.get() == "unset"  # opening the span leaves the caller's context alone
    await asyncio.create_task(_end(handle))

    assert seen["marker_on_exit"] == "entered"


async def _start(external_id):
    return start_turn_span(external_id)


async def _end(handle):
    end_turn_span(handle)


# --- use_turn_span ------------------------------------------------------------


def test_use_turn_span_is_a_noop_for_none():
    with use_turn_span(None):
        pass  # must not raise


def test_use_turn_span_never_raises_on_a_handle_that_is_not_a_turn_span():
    with use_turn_span(object()):
        pass  # must not raise


def _turn_handle() -> telemetry._TurnSpan:
    """A handle whose OpenTelemetry context holds a live span, as
    `start_turn_span` builds one, without any backend."""
    span = _span("test.turn", "turn")
    return telemetry._TurnSpan(
        scope=None,
        context=contextvars.copy_context(),
        otel_context=otel_trace.set_span_in_context(span),
        observation=None,
    )


def test_use_turn_span_makes_the_span_current_and_restores_the_context_after():
    handle = _turn_handle()
    before = otel_context.get_current()

    with use_turn_span(handle):
        assert otel_trace.get_current_span(
            otel_context.get_current()
        ) is otel_trace.get_current_span(handle.otel_context)
        assert otel_trace.get_current_span().get_span_context().is_valid

    assert otel_context.get_current() == before
    assert not otel_trace.get_current_span().get_span_context().is_valid


def test_use_turn_span_never_raises_when_attaching_fails(monkeypatch):
    detached = []

    def _boom(context):
        raise RuntimeError("attach failed")

    monkeypatch.setattr(otel_context, "attach", _boom)
    monkeypatch.setattr(otel_context, "detach", detached.append)
    ran = []

    with use_turn_span(_turn_handle()):
        ran.append(True)

    assert ran == [True]
    assert detached == []  # nothing was attached, so nothing is detached


def test_use_turn_span_never_raises_when_detaching_fails(monkeypatch):
    real_detach = otel_context.detach

    def _boom(token):
        real_detach(token)  # keeps the test's own context clean
        raise RuntimeError("detach failed")

    monkeypatch.setattr(otel_context, "detach", _boom)
    ran = []

    with use_turn_span(_turn_handle()):
        ran.append(True)

    assert ran == [True]


def test_use_turn_span_left_in_another_context_detaches_nothing_and_logs_nothing(caplog):
    """An async generator abandoned at a `yield` is closed by the event loop in
    a context of its own. Detaching there makes OpenTelemetry log a ValueError
    at ERROR; `use_turn_span` recognises the foreign context and leaves it."""
    scope = use_turn_span(_turn_handle())
    contextvars.copy_context().run(scope.__enter__)

    with caplog.at_level(logging.ERROR, logger="opentelemetry.context"):
        contextvars.copy_context().run(scope.__exit__, None, None, None)

    assert caplog.records == []


# --- flush_telemetry ------------------------------------------------------------


async def test_flush_telemetry_is_a_noop_with_no_backend(monkeypatch):
    """With telemetry off, no flush thread starts and no flush runs."""
    flushed = []
    monkeypatch.setattr(telemetry, "_flush_backend", lambda: flushed.append(True))
    started = []
    real_start = threading.Thread.start
    monkeypatch.setattr(
        threading.Thread, "start", lambda self: (started.append(self.name), real_start(self))
    )

    await flush_telemetry()

    assert flushed == []
    assert telemetry._FLUSH_THREAD_NAME not in started


async def test_flush_telemetry_flushes_the_langfuse_client(monkeypatch):
    telemetry._backend = "langfuse"
    fake_client = MagicMock()
    monkeypatch.setattr(langfuse, "get_client", lambda: fake_client)

    await flush_telemetry()

    fake_client.flush.assert_called_once_with()


async def test_flush_telemetry_never_raises_when_the_client_fails(monkeypatch, caplog):
    telemetry._backend = "langfuse"
    fake_client = MagicMock()
    fake_client.flush.side_effect = RuntimeError("flush failed")
    monkeypatch.setattr(langfuse, "get_client", lambda: fake_client)

    with caplog.at_level("WARNING"):
        await flush_telemetry()  # must not raise

    assert any("telemetry flush failed" in r.message for r in caplog.records)
    assert not telemetry._flush_running


async def test_flush_telemetry_returns_at_the_bound_and_skips_while_one_is_running(
    monkeypatch, caplog
):
    telemetry._backend = "langfuse"
    release = threading.Event()
    started = threading.Event()
    fake_client = MagicMock()

    def _hang():
        started.set()
        release.wait(timeout=10)

    fake_client.flush.side_effect = _hang
    monkeypatch.setattr(langfuse, "get_client", lambda: fake_client)

    try:
        with caplog.at_level("WARNING"):
            await asyncio.wait_for(flush_telemetry(timeout_seconds=0.1), timeout=2)
        assert started.is_set()
        assert any("still running" in r.message for r in caplog.records)

        await asyncio.wait_for(flush_telemetry(timeout_seconds=0.1), timeout=2)
        await asyncio.wait_for(flush_telemetry(timeout_seconds=0.1), timeout=2)
        assert fake_client.flush.call_count == 1  # neither call started another
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name == telemetry._FLUSH_THREAD_NAME:
                thread.join(timeout=10)
    assert not telemetry._flush_running
    # The calls that found one running are served by a single second pass.
    assert fake_client.flush.call_count == 2
    assert not telemetry._flush_pending


# --- record_event -----------------------------------------------------------


def test_record_event_is_a_noop_with_no_backend(monkeypatch):
    def _unexpected():
        raise AssertionError("no backend must not reach for a client")

    monkeypatch.setattr(langfuse, "get_client", _unexpected)
    span = MagicMock()
    monkeypatch.setattr(otel_trace, "get_current_span", lambda: span)

    record_event("compose.refused", metadata={"reason": "x"})

    span.add_event.assert_not_called()


def test_record_event_langfuse_creates_an_event_with_level_and_metadata(monkeypatch, live_span):
    telemetry._backend = "langfuse"
    fake_client = MagicMock()
    monkeypatch.setattr(langfuse, "get_client", lambda: fake_client)

    record_event("compose.refused", metadata={"reason": "no_rows", "count": 3})

    fake_client.create_event.assert_called_once_with(
        name="compose.refused", level="WARNING", metadata={"reason": "no_rows", "count": 3}
    )


def test_record_event_langfuse_outside_any_span_creates_nothing(monkeypatch):
    """Langfuse would make an event created outside a span a one-event root
    trace of its own."""
    telemetry._backend = "langfuse"
    fake_client = MagicMock()
    monkeypatch.setattr(langfuse, "get_client", lambda: fake_client)

    record_event("compose.refused", metadata={"reason": "no_rows"})

    fake_client.create_event.assert_not_called()


def test_record_event_otel_adds_a_span_event_to_the_current_span(monkeypatch):
    telemetry._backend = "otel"
    span = MagicMock()
    monkeypatch.setattr(otel_trace, "get_current_span", lambda: span)

    record_event("compose.refused", level="ERROR", metadata={"reason": "no_rows"})

    span.add_event.assert_called_once_with(
        "compose.refused", attributes={"reason": "no_rows", "level": "ERROR"}
    )


def test_record_event_keeps_only_scalar_values_and_truncates_strings(monkeypatch, live_span):
    telemetry._backend = "langfuse"
    fake_client = MagicMock()
    monkeypatch.setattr(langfuse, "get_client", lambda: fake_client)

    record_event(
        "n" * 1000,
        metadata={
            "text": "x" * 1000,
            "count": 2,
            "ratio": 0.5,
            "flag": True,
            "nested": {"a": 1},
            "items": [1, 2],
            "nothing": None,
        },
    )

    _, kwargs = fake_client.create_event.call_args
    assert kwargs["name"] == "n" * 500
    assert kwargs["metadata"] == {"text": "x" * 500, "count": 2, "ratio": 0.5, "flag": True}


def test_record_event_caps_the_number_of_metadata_entries(monkeypatch, live_span):
    telemetry._backend = "langfuse"
    fake_client = MagicMock()
    monkeypatch.setattr(langfuse, "get_client", lambda: fake_client)

    record_event("e", metadata={f"k{i}": i for i in range(100)})

    _, kwargs = fake_client.create_event.call_args
    assert len(kwargs["metadata"]) == 32
    assert next(iter(kwargs["metadata"])) == "k0"


@pytest.mark.parametrize("backend_name", ["langfuse", "otel"])
def test_record_event_never_propagates_a_backend_failure(monkeypatch, backend_name):
    """Telemetry must not be the reason a turn fails."""
    telemetry._backend = backend_name
    fake_client = MagicMock()
    fake_client.create_event.side_effect = RuntimeError("langfuse down")
    monkeypatch.setattr(langfuse, "get_client", lambda: fake_client)
    span = MagicMock()
    span.add_event.side_effect = RuntimeError("otel down")
    monkeypatch.setattr(otel_trace, "get_current_span", lambda: span)

    record_event("compose.refused", metadata={"reason": "x"})  # must not raise


def test_record_event_logs_the_first_failure_at_warning_and_later_ones_at_debug(
    monkeypatch, caplog, live_span
):
    telemetry._backend = "langfuse"
    fake_client = MagicMock()
    fake_client.create_event.side_effect = RuntimeError("langfuse down")
    monkeypatch.setattr(langfuse, "get_client", lambda: fake_client)

    with caplog.at_level("DEBUG", logger=telemetry.logger.name):
        record_event("compose.refused")
        record_event("compose.refused")

    failures = [r for r in caplog.records if "failed to record telemetry event" in r.message]
    assert [r.levelno for r in failures] == [logging.WARNING, logging.DEBUG]


# --- record_span -------------------------------------------------------


def test_record_span_langfuse_opens_and_closes_one_observation(monkeypatch):
    """Langfuse's span processor exports only spans from its own tracer scope
    (plus `gen_ai.*`-attributed or listed-instrumentor spans), so a marker
    span recorded while Langfuse is the backend must go through
    `start_as_current_observation`, not the plain OTel tracer, or it is
    silently dropped rather than exported."""
    telemetry._backend = "langfuse"
    fake_observation_cm = MagicMock()
    fake_client = MagicMock()
    fake_client.start_as_current_observation.return_value = fake_observation_cm
    monkeypatch.setattr(langfuse, "get_client", lambda: fake_client)

    record_span("plan.approved", {"dfo.depth": "standard", "dfo.parts": 2})

    fake_client.start_as_current_observation.assert_called_once_with(
        name="plan.approved",
        as_type="span",
        metadata={"dfo.depth": "standard", "dfo.parts": 2},
    )
    fake_observation_cm.__enter__.assert_called_once()
    fake_observation_cm.__exit__.assert_called_once()


def test_record_span_otel_lands_on_the_exporter():
    """No Langfuse backend means the plain OTel tracer, unfiltered."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    telemetry._backend = None

    record_span("plan.approved", {"dfo.depth": "standard", "dfo.parts": 2})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "plan.approved"
    assert dict(spans[0].attributes) == {"dfo.depth": "standard", "dfo.parts": 2}


def test_record_span_nests_under_the_turn_span_made_current_by_use_turn_span():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    telemetry._backend = "otel"

    handle = start_turn_span("tr_marker0001")
    with use_turn_span(handle):
        record_span("plan.approved", {"dfo.parts": 2})
    end_turn_span(handle)

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["plan.approved"].parent is not None
    assert spans["plan.approved"].parent.span_id == spans["turn"].context.span_id
    assert spans["plan.approved"].context.trace_id == spans["turn"].context.trace_id


# --- use_request_span and current_trace_id -------------------------------------


def test_use_request_span_under_langfuse_opens_one_observation(monkeypatch):
    """Langfuse drops the request's own OTel spans, so the block gets an
    observation that it records."""
    telemetry._backend = "langfuse"
    fake_observation_cm = MagicMock()
    fake_client = MagicMock()
    fake_client.start_as_current_observation.return_value = fake_observation_cm
    monkeypatch.setattr(langfuse, "get_client", lambda: fake_client)

    with use_request_span("combined mcp tools/call search"):
        pass

    fake_client.start_as_current_observation.assert_called_once_with(
        name="combined mcp tools/call search", as_type="span"
    )
    fake_observation_cm.__enter__.assert_called_once()
    fake_observation_cm.__exit__.assert_called_once()


@pytest.fixture
def otel_exporter():
    """A plain OTel backend whose finished spans land in the returned exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    telemetry._backend = "otel"
    return exporter


def test_use_request_span_keeps_a_valid_current_span_and_opens_none(otel_exporter):
    """Under plain OTel the request's own span is current already; a second one
    would split it."""
    with (
        otel_trace.get_tracer(__name__).start_as_current_span("request") as request,
        use_request_span("combined mcp tools/call search"),
    ):
        inside = current_trace_id()

    assert inside == otel_trace.format_trace_id(request.get_span_context().trace_id)
    assert [span.name for span in otel_exporter.get_finished_spans()] == ["request"]


def _fastmcp_request_span():
    """A current `request` span in place of fastmcp's SERVER span, with exception
    recording off so the test sees only what `use_request_span` records."""
    return otel_trace.get_tracer(__name__).start_as_current_span(
        "request", record_exception=False, set_status_on_exception=False
    )


def test_use_request_span_under_otel_marks_an_error_outcome_on_the_request_span(otel_exporter):
    with _fastmcp_request_span(), use_request_span("combined mcp tools/call search") as request:
        request.record_outcome(level="ERROR", status_message="upstream_error: bad_shape")

    (span,) = otel_exporter.get_finished_spans()
    assert span.status.status_code is otel_trace.StatusCode.ERROR
    assert span.status.description == "upstream_error: bad_shape"


def test_use_request_span_under_otel_leaves_a_warning_outcome_off_the_status(otel_exporter):
    with _fastmcp_request_span(), use_request_span("combined mcp tools/call search") as request:
        request.record_outcome(level="WARNING", status_message="request_invalid")

    (span,) = otel_exporter.get_finished_spans()
    assert span.status.status_code is otel_trace.StatusCode.UNSET


def test_use_request_span_under_otel_marks_a_raised_exception_by_its_class_name(otel_exporter):
    with (
        pytest.raises(RuntimeError),
        _fastmcp_request_span(),
        use_request_span("combined mcp tools/call search"),
    ):
        raise RuntimeError("gs://bucket/secret-path")

    (span,) = otel_exporter.get_finished_spans()
    assert span.status.status_code is otel_trace.StatusCode.ERROR
    assert span.status.description == "RuntimeError"


def test_use_request_span_under_langfuse_records_the_outcome_on_its_observation(monkeypatch):
    telemetry._backend = "langfuse"
    observation = MagicMock()
    fake_client = MagicMock()
    fake_client.start_as_current_observation.return_value.__enter__.return_value = observation
    monkeypatch.setattr(langfuse, "get_client", lambda: fake_client)

    with pytest.raises(ValueError), use_request_span("combined mcp resources/read") as handle:
        handle.record_outcome(level="DEFAULT", status_message="operation_still_processing")
        raise ValueError("dfo://artifact/secret")

    assert observation.update.call_args_list == [
        call(level="DEFAULT", status_message="operation_still_processing"),
        call(level="ERROR", status_message="ValueError"),
    ]


def test_record_outcome_swallows_a_failing_observation():
    observation = MagicMock()
    observation.update.side_effect = RuntimeError("exporter gone")

    telemetry.RequestSpan(observation=observation, span=None).record_outcome(
        level="ERROR", status_message="internal_error"
    )


def test_use_request_span_without_a_valid_span_names_one_random_id_per_block():
    telemetry._backend = None

    ids = []
    for _ in range(2):
        with use_request_span("combined mcp resources/read"):
            first, second = current_trace_id(), current_trace_id()
        assert first == second
        ids.append(first)

    assert all(len(trace_id) == 32 and int(trace_id, 16) for trace_id in ids)
    assert ids[0] != ids[1]
    assert not otel_trace.get_current_span().get_span_context().is_valid


def test_current_trace_id_without_a_span_is_fresh_each_call():
    assert current_trace_id() != current_trace_id()


def test_turn_call_meta_is_empty_with_no_backend():
    handle = start_turn_span("tr_meta0001", session_id="conv-1")

    with use_turn_span(handle):
        assert turn_call_meta() == {}


def test_turn_call_meta_is_empty_on_the_otlp_backend(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    init_telemetry(service_name="svc")
    handle = start_turn_span("tr_meta0002", session_id="conv-1")

    try:
        with use_turn_span(handle):
            assert turn_call_meta() == {}
    finally:
        end_turn_span(handle)


def test_turn_call_meta_never_raises(monkeypatch):
    monkeypatch.setattr(otel_context, "get_value", MagicMock(side_effect=RuntimeError("boom")))

    assert turn_call_meta() == {}


@pytest.mark.parametrize("backend_name", [None, "otel"])
def test_use_call_session_outside_langfuse_runs_the_block_untouched(monkeypatch, backend_name):
    propagate = MagicMock()
    monkeypatch.setattr(langfuse, "propagate_attributes", propagate)
    monkeypatch.setattr(telemetry, "_backend", backend_name)
    ran = []

    with use_call_session("conv-1"):
        ran.append(True)

    assert ran == [True]
    propagate.assert_not_called()


def test_use_call_session_never_raises_on_a_propagation_failure(monkeypatch):
    monkeypatch.setattr(telemetry, "_backend", "langfuse")
    monkeypatch.setattr(
        langfuse, "propagate_attributes", MagicMock(side_effect=RuntimeError("boom"))
    )
    ran = []

    with use_call_session("conv-1"):
        ran.append(True)

    assert ran == [True]
