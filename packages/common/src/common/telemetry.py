"""Telemetry bootstrap shared by all dfo-ai services.

`init_telemetry` selects a backend from the environment and `backend()` reports
it: `"langfuse"`, `"otel"` or `"off"`. Service-specific instrumentation (e.g.
ADK or FastMCP span attributes) is layered on top by each service; this module
owns env-var detection so services don't each hand-roll it. Third-party imports
are lazy so the module stays cheap to import when telemetry is off, as it is in
tests and in local dev without creds.

Recognised env vars:
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` -- all
  three enable Langfuse export. A deployed service needs all three;
  `LANGFUSE_TRACING_ENVIRONMENT` (read by the Langfuse SDK) tags the traces,
  which separates deployed from local ones in a shared project.
  `LANGFUSE_TRACING_ENABLED=false` and `LANGFUSE_SAMPLE_RATE=0` make the SDK
  export nothing, so with either set the state is off.
- `OTEL_EXPORTER_OTLP_ENDPOINT` -- enables a generic OTLP/HTTP span exporter,
  checked only if the Langfuse vars above are not all set.
- `OTEL_SDK_DISABLED=true` (case-insensitive, surrounding whitespace ignored) --
  the OpenTelemetry SDK hands out no-op tracers, so whichever backend the two
  groups above select would export nothing; the state is off.
- None of them set -- the state is off.

When a backend comes up, `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`
is set to `SPAN_ONLY` unless the environment already sets it, so ADK's
`generate_content` spans carry each model call's prompt and response and its
`execute_tool` spans each tool's arguments and result. ADK writes every
streamed response as its own output message and thought summaries as plain
text parts.

The spans follow ADK's telemetry schema v2, its default.
`ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN=otel_semconv_1_36` selects schema v1,
whose spans reach Langfuse as ADK writes them: each model call is priced
twice, on `call_llm` and on `generate_content`, and Langfuse's default span
filter drops ADK's attribute-less `invocation` spans, so the `invoke_workflow`
spans under them lose their parent.

`start_turn_span`/`end_turn_span` open and close one root span per unit of
work (a harness turn, say) on whichever backend is active. `use_turn_span`
makes that span current in the task that runs a `with` block, so a unit of
work that spans several requests (a turn paused for approval and resumed by a
second request) stays one trace. `turn_call_meta` gives the `_meta` entries
that carry the turn's session to an MCP server, and `use_call_session` makes
the server's spans under a call carry it. `record_event` attaches a named
event, and `record_span` a zero-duration marker span, to the current trace, and
`flush_telemetry` exports what is queued before a response ends.
`use_request_span` gives a request that has no span of the service's own
one the backend records and marks it with the request's outcome, and
`current_trace_id` names the current trace for a log line. A caller elsewhere
in the service doesn't need to know which backend is active or whether
telemetry is on.
"""

# ruff: noqa: PLC0415 -- every third-party import below is deliberately lazy
# (see module docstring); that's this file's whole reason for existing.

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import secrets
import threading
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

logger = logging.getLogger(__name__)

_LANGFUSE_ENV_VARS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")

_AUTH_CHECK_THREAD_NAME = "langfuse-auth-check"
_FLUSH_THREAD_NAME = "telemetry-flush"

# The OpenTelemetry GenAI variable that decides whether ADK writes message
# content on its spans (`_capture_message_content`).
_CAPTURE_MESSAGE_CONTENT_VAR = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"

# The longest a response waits on `flush_telemetry`. An export to a healthy
# Langfuse takes well under a second; the exporter's own request timeout is
# 5 seconds (`LANGFUSE_TIMEOUT`), and a longer wait would hold a response open
# on an unreachable backend.
FLUSH_TIMEOUT_SECONDS = 3.0

# Longest string a `record_event` name or metadata value keeps, and the most
# metadata entries it keeps; exporters and the Langfuse UI handle large
# attributes poorly, and an event is a marker.
_MAX_EVENT_STR_CHARS = 500
_MAX_EVENT_METADATA_ITEMS = 32

# The name of the root span `start_turn_span` opens, and the Langfuse trace
# name every span of a turn carries, on the server side of its MCP calls too.
TURN_SPAN_NAME = "turn"

# The MCP request `_meta` key that carries a turn's Langfuse session to a server
# (`turn_call_meta`). The MCP spec namespaces a `_meta` key with a DNS-style
# prefix and reserves the prefixes whose second label is `modelcontextprotocol`
# or `mcp`.
SESSION_META_KEY = "org.one.dfo/langfuse-session"

type Backend = Literal["langfuse", "otel", "off"]
type EventLevel = Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]

_configured = False
# Which backend `init_telemetry` brought up, if any -- "langfuse", "otel", or
# None (nothing configured, or the exporter could not be constructed). Set
# once from the environment and never revisited, and read by `backend()`,
# `start_turn_span` and `record_event` so none of them needs env lookups of
# its own or can drift from what `init_telemetry` decided.
_backend: Literal["langfuse", "otel"] | None = None
# Whether a `record_event` failure has been logged at WARNING yet; later ones
# log at DEBUG so a broken exporter is visible once without a line per event.
_event_failure_warned = False
# Single-flight state for `flush_telemetry`, guarded by `_flush_lock`:
# `_flush_running` is true while a flush thread owns the backend, and
# `_flush_pending` records that a call found it running. The thread checks
# `_flush_pending` and clears `_flush_running` in one locked step, so a call
# is either seen by the running thread or starts its own flush.
_flush_lock = threading.Lock()
_flush_running = False
_flush_pending = False

# Set in the context that entered a `use_turn_span` block, so leaving the block
# in any other context is recognisable.
_turn_scope_marker: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "dfo_turn_scope_marker", default=False
)


def backend() -> Backend:
    """The backend `init_telemetry` resolved, `"off"` when there is none.

    `"off"` before `init_telemetry` runs as well.
    """
    return _backend or "off"


def init_telemetry(*, service_name: str) -> None:
    """Configure tracing for `service_name` from the environment variables the
    module docstring lists, and log the outcome once: at INFO when a backend
    came up, at WARNING when none did (naming the reason), and at ERROR, with
    the traceback, when the exporter could not be constructed. The environment
    and the exporter's construction decide the outcome; no network call does.

    Idempotent -- safe to call multiple times.
    """
    global _configured  # noqa: PLW0603 -- deliberate idempotent-init state
    if _configured:
        return
    _configured = True

    try:
        disabled_reason = _start_backend(service_name=service_name)
    except Exception:
        logger.exception(
            "Telemetry disabled for %s: the exporter could not be constructed", service_name
        )
        return
    if disabled_reason is not None:
        logger.warning(
            "Telemetry disabled for %s (%s); no traces will be exported",
            service_name,
            disabled_reason,
        )


def _start_backend(*, service_name: str) -> str | None:
    """Bring up the backend the environment selects, setting `_backend`.

    Returns why no backend was selected, or `None` when one came up. Raises
    when the selected exporter cannot be constructed.
    """
    global _backend  # noqa: PLW0603 -- the backend is fixed once the exporter exists
    disabled_reason = _otel_sdk_disabled_reason()
    if disabled_reason is not None:
        return disabled_reason

    missing = [var for var in _LANGFUSE_ENV_VARS if not os.getenv(var)]
    if not missing:
        disabled_reason = _langfuse_disabled_reason()
        if disabled_reason is None:
            _capture_message_content()
            _init_langfuse(service_name=service_name)
        return disabled_reason
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        _capture_message_content()
        _init_otel(service_name=service_name)
        _backend = "otel"
        return None
    if len(missing) == len(_LANGFUSE_ENV_VARS):
        return "no LANGFUSE_* or OTEL_EXPORTER_OTLP_ENDPOINT set"
    return f"LANGFUSE_* incomplete, missing {', '.join(missing)}"


def _capture_message_content() -> None:
    """Have ADK write prompts, responses, tool arguments and tool results on
    its spans, unless the environment already says otherwise.

    ADK's telemetry schema v2 puts a model call's messages on its
    `generate_content` span and a tool's arguments and result on its
    `execute_tool` span only when `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`
    is `SPAN_ONLY` or `SPAN_AND_EVENT`. Unset, the spans carry none. ADK reads
    the variable each time it builds a telemetry config, which it does per
    model and tool call, so setting it here, before the first run, reaches
    every call. A value already set, even an empty one, is left as it is.
    """
    os.environ.setdefault(_CAPTURE_MESSAGE_CONTENT_VAR, "SPAN_ONLY")


def _otel_sdk_disabled_reason() -> str | None:
    """`"OTEL_SDK_DISABLED=true"` when the OpenTelemetry SDK is told to disable
    itself, else `None`. The SDK's tracer provider then hands out no-op tracers,
    so neither backend exports anything. It compares the value lowercased and
    stripped, and so does this."""
    if os.getenv("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return "OTEL_SDK_DISABLED=true"
    return None


def _langfuse_disabled_reason() -> str | None:
    """Why the Langfuse SDK would export nothing although its credentials are
    set, read from the same public env vars the SDK reads, or `None`.

    `LANGFUSE_TRACING_ENABLED=false` makes the SDK use a no-op tracer, and
    `LANGFUSE_SAMPLE_RATE=0` makes its sampler drop every trace. A sample rate
    the SDK rejects (not a number, outside 0 to 1) is left to it: creating the
    client raises, which `init_telemetry` reports as a construction failure.
    """
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() == "false":
        return "LANGFUSE_TRACING_ENABLED=false"
    with suppress(ValueError):
        if float(os.getenv("LANGFUSE_SAMPLE_RATE", "1.0")) == 0.0:
            return "LANGFUSE_SAMPLE_RATE=0"
    return None


def _init_langfuse(*, service_name: str) -> None:
    """Create the Langfuse client, which attaches its exporter to the global
    tracer provider, and start a background credentials check.

    The client is built with `_langfuse_exporter` and the SDK's default span
    filter, which exports its own spans and every span with a `gen_ai.*`
    attribute, so every span ADK opens. The SDK keeps the first client built
    in a process and ignores the arguments of a later one, and `get_client()`
    returns that first client, so this has to run before anything else builds
    one. A client that already exists is reported.

    Once the client exists, spans are exported whatever the check finds, so the
    backend is recorded then and the check only reports.
    """
    global _backend  # noqa: PLW0603 -- the backend is fixed the moment the exporter exists
    from langfuse import Langfuse

    if _langfuse_client_exists():
        logger.warning(
            "A Langfuse client already exists in %s, so its exporter stays in force and "
            "may export credentials in span URIs unredacted",
            service_name,
        )
    client = Langfuse(span_exporter=_langfuse_exporter())
    _backend = "langfuse"
    logger.info("Telemetry configured for %s via Langfuse", service_name)
    try:
        threading.Thread(
            target=_check_langfuse_auth,
            args=(client, service_name),
            name=_AUTH_CHECK_THREAD_NAME,
            daemon=True,
        ).start()
    except Exception:
        logger.warning(
            "Langfuse auth check for %s could not be started; credentials are unchecked",
            service_name,
            exc_info=True,
        )


def _langfuse_exporter() -> Any:
    """The OTLP exporter the Langfuse SDK builds by default, behind
    `common.uri_redaction.UriRedactingExporter` so no span carries a whole
    credential URI.

    The SDK wires no endpoint, credentials or timeout into an exporter it is
    handed, so this sets the ones its own would get: the endpoint under
    `LANGFUSE_BASE_URL` (the path `LANGFUSE_OTEL_TRACES_EXPORT_PATH` names, if
    set), Basic auth from the key pair, the SDK's name, version and public-key
    headers, and `LANGFUSE_TIMEOUT` seconds, 5 by default. It also sends
    `x-langfuse-ingestion-version: 4`, which the SDK asks of a custom exporter
    so that a Langfuse v4 server processes the spans in real time.
    """
    import base64

    from langfuse import __version__ as langfuse_version
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    from common.uri_redaction import UriRedactingExporter

    public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    credentials = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
    path = os.getenv("LANGFUSE_OTEL_TRACES_EXPORT_PATH") or "api/public/otel/v1/traces"
    return UriRedactingExporter(
        OTLPSpanExporter(
            endpoint=f"{os.environ['LANGFUSE_BASE_URL']}/{path}",
            headers={
                "Authorization": f"Basic {credentials}",
                "x-langfuse-sdk-name": "python",
                "x-langfuse-sdk-version": langfuse_version,
                "x-langfuse-public-key": public_key,
                "x-langfuse-ingestion-version": "4",
            },
            timeout=int(os.getenv("LANGFUSE_TIMEOUT", "5")),
        )
    )


def _langfuse_client_exists() -> bool:
    """Whether the SDK already holds a client in this process. The SDK has no
    public query for it, so this reads its client registry, and answers `False`
    if that registry moves: the answer only decides whether to log a WARNING."""
    try:
        from langfuse._client.resource_manager import LangfuseResourceManager

        return bool(LangfuseResourceManager._instances)
    except Exception:
        logger.debug("could not tell whether a Langfuse client exists", exc_info=True)
        return False


def _check_langfuse_auth(client: Any, service_name: str) -> None:
    """Log a WARNING when Langfuse rejects the credentials or cannot be reached.

    `auth_check` blocks for up to the SDK's request timeout, which is why this
    runs on its own thread. It changes no state: an unreachable or misconfigured
    Langfuse surfaces here, and the exporter keeps retrying on its own.
    """
    try:
        if client.auth_check():
            return
    except Exception:
        logger.warning(
            "Langfuse auth check errored for %s; traces may not be reaching Langfuse "
            "(check LANGFUSE_BASE_URL and the key pair)",
            service_name,
            exc_info=True,
        )
        return
    logger.warning(
        "Langfuse auth check failed for %s; traces may not be reaching Langfuse "
        "(check LANGFUSE_BASE_URL and the key pair)",
        service_name,
    )


def _init_otel(*, service_name: str) -> None:
    """Wire up a generic OTLP/HTTP span exporter, behind
    `common.uri_redaction.UriRedactingExporter` so no span carries a whole
    credential URI."""
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from common.uri_redaction import UriRedactingExporter

    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
    provider.add_span_processor(BatchSpanProcessor(UriRedactingExporter(OTLPSpanExporter())))
    trace.set_tracer_provider(provider)
    logger.info("Telemetry configured for %s via OTLP", service_name)


@dataclass(frozen=True, slots=True)
class _TurnSpan:
    """The handle `start_turn_span` returns.

    `scope` is the entered span context manager (on Langfuse an `ExitStack`
    that also holds the session propagation). `context` is the copy of the
    caller's `contextvars` context it was entered in, and `end_turn_span` exits
    it there: a context token can only be reset in the context that created
    it, and the span is ended by whichever request closes the turn.
    `otel_context` is the OpenTelemetry context that holds the span (and, on
    Langfuse, the trace-root claim the SDK puts beside it), which
    `use_turn_span` attaches. `observation` is the Langfuse observation the
    span is, `None` off Langfuse; `end_turn_span` sets its input and output.
    """

    scope: Any
    context: contextvars.Context
    otel_context: Any
    observation: Any | None


def start_turn_span(external_id: str, *, session_id: str | None = None) -> Any | None:
    """Open one root span for a unit of work identified by `external_id`
    (a harness turn's wire `trace_id`, say). Returns an opaque handle for
    `use_turn_span` and `end_turn_span`, or `None` when telemetry never came
    up or opening the span itself errors; either way the caller proceeds
    untraced rather than failing on it.

    The span is opened inside a private copy of the caller's context, so it is
    not current on the caller's task and nothing is left attached to it.
    Spans nest under it only inside `use_turn_span(handle)`. That is what lets
    one turn span cover several requests: each request's stream attaches the
    same span for its own duration.

    Only the Langfuse backend gets an exact match: `Langfuse.create_trace_id`
    seeded on `external_id` is the SDK's own documented pattern for
    correlating an external id with a Langfuse trace (a sha256 of the seed,
    truncated to a valid 32-hex-character trace id), so a person holding
    only `external_id` can recompute the same trace id and open it directly
    -- no time-window search needed. It is also set as the span's
    `dfo.external_id` attribute, for a text search over the same value. A
    generic OTLP endpoint (no Langfuse) has no equivalent deterministic-id
    call in the plain OTel SDK, so there `external_id` goes on the span as
    that same attribute alone, filterable in whatever reads the export.

    `session_id` is the Langfuse session the turn belongs to, so turns that
    share it group in Langfuse's Sessions view. It and the trace name
    `TURN_SPAN_NAME` reach every span started under `use_turn_span(handle)` in
    this process, ADK's included, and only on the Langfuse backend. Spans an
    MCP server opens for a call it receives over HTTP start in the server's own
    context, and `turn_call_meta` carries the session to it. The Langfuse SDK
    drops a `session_id` over 200 characters with a warning, so the trace
    carries no session and ADK's spans fall back to their
    per-turn `gen_ai.conversation.id`.
    """
    if _backend is None:
        return None
    try:
        from opentelemetry import context as otel_context_api

        context = contextvars.copy_context()
        scope, observation = context.run(_enter_turn_scope, external_id, session_id)
        otel_context = context.run(otel_context_api.get_current)
        return _TurnSpan(
            scope=scope, context=context, otel_context=otel_context, observation=observation
        )
    except Exception:
        logger.exception("failed to open a turn span for %s -- continuing untraced", external_id)
        return None


def _enter_turn_scope(external_id: str, session_id: str | None) -> tuple[Any, Any | None]:
    """Open the turn span as the current span of the calling context and
    return the entered context manager with the Langfuse observation, `None`
    off Langfuse. On Langfuse, the trace name and a `session_id` are
    propagated to every span started under the turn, the session is held for
    `turn_call_meta`, and the returned scope exits both before the span."""
    if _backend == "langfuse":
        from langfuse import Langfuse, get_client, propagate_attributes

        derived = Langfuse.create_trace_id(seed=external_id)
        with ExitStack() as stack:
            observation = stack.enter_context(
                get_client().start_as_current_observation(
                    name=TURN_SPAN_NAME,
                    as_type="span",
                    trace_context={"trace_id": derived},
                    metadata={"dfo.external_id": external_id},
                )
            )
            stack.enter_context(
                propagate_attributes(trace_name=TURN_SPAN_NAME, session_id=session_id or None)
            )
            if session_id:
                stack.enter_context(_hold_turn_session(session_id))
            scope = stack.pop_all()
        logger.debug("turn span opened: %s -> langfuse trace %s", external_id, derived)
        return scope, observation

    from opentelemetry import trace

    scope = trace.get_tracer(__name__).start_as_current_span(
        TURN_SPAN_NAME, attributes={"dfo.external_id": external_id}
    )
    scope.__enter__()
    return scope, None


@cache
def _turn_session_key() -> str:
    """The OpenTelemetry context key under which a turn holds its Langfuse
    session, so it rides with every `use_turn_span` attach. A private key, so no
    other code can set or read it by name."""
    from opentelemetry import context as otel_context_api

    return otel_context_api.create_key("dfo.turn_session_id")


@contextmanager
def _hold_turn_session(session_id: str) -> Iterator[None]:
    """Hold `session_id` in the current OpenTelemetry context for the block."""
    from opentelemetry import context as otel_context_api

    token = otel_context_api.attach(otel_context_api.set_value(_turn_session_key(), session_id))
    try:
        yield
    finally:
        otel_context_api.detach(token)


def turn_call_meta() -> dict[str, str]:
    """The MCP request `_meta` entries for a call made now: `{SESSION_META_KEY:
    session}` inside a turn that has a Langfuse session, `{}` anywhere else.

    The value is read from the OpenTelemetry context the turn span lives in, so
    it is the `session_id` that `start_turn_span` was given. Pass it as the
    `meta` of a client's `call_tool`. The server's `use_call_session` reads it
    back. Never raises.
    """
    try:
        from opentelemetry import context as otel_context_api

        session = otel_context_api.get_value(_turn_session_key())
    except Exception:
        logger.debug("failed to read the turn session -- continuing", exc_info=True)
        return {}
    return {SESSION_META_KEY: session} if isinstance(session, str) and session else {}


@contextmanager
def use_call_session(session_id: str | None) -> Iterator[None]:
    """Make every span opened in the block, and the current span, carry the
    Langfuse session `session_id` and the trace name `TURN_SPAN_NAME`, for an
    MCP server handling a call whose client sent the session in `_meta` (see
    `turn_call_meta`, which sends it only from inside a turn).

    A no-op for an empty or `None` id and off the Langfuse backend. The session
    stays in this process: it is not put in baggage, so no outbound HTTP
    request carries it. Never raises.
    """
    with ExitStack() as stack:
        if _backend == "langfuse" and session_id:
            try:
                from langfuse import propagate_attributes

                stack.enter_context(
                    propagate_attributes(
                        session_id=session_id, trace_name=TURN_SPAN_NAME, as_baggage=False
                    )
                )
            except Exception:
                logger.debug("failed to propagate a call session -- continuing", exc_info=True)
        yield


@contextmanager
def use_turn_span(handle: Any | None) -> Iterator[None]:
    """Make the span behind `handle` current for the enclosed block, so every
    span opened and every `record_event` made while it runs, on this task or
    on any task started inside the block, is a descendant of that span in the
    same trace. A no-op for `None`. Never raises and never ends the span; only
    `end_turn_span` does.

    It attaches a context token, and a token can only be detached in the
    context that made it. A block that an async generator holds open across a
    `yield` is left in another context when the event loop closes the
    generator after its task was cancelled; there is nothing to restore in that
    context, so leaving the block there detaches nothing. Any other block is
    entered and left on one task.
    """
    attached: tuple[Any, contextvars.Token[bool]] | None = None
    if isinstance(handle, _TurnSpan):
        try:
            from opentelemetry import context as otel_context_api

            token = otel_context_api.attach(handle.otel_context)
            attached = (token, _turn_scope_marker.set(True))
        except Exception:
            logger.debug("failed to attach a turn span -- continuing", exc_info=True)
    try:
        yield
    finally:
        if attached is not None:
            _leave_turn_scope(*attached)


def _leave_turn_scope(token: Any, marker: contextvars.Token[bool]) -> None:
    """Undo `use_turn_span`'s attach, when this is the context that made it."""
    try:
        _turn_scope_marker.reset(marker)
    except ValueError:
        logger.debug("turn span scope left in another context; nothing to detach")
        return
    except Exception:
        logger.debug("failed to leave a turn span scope -- continuing", exc_info=True)
        return
    try:
        from opentelemetry import context as otel_context_api

        otel_context_api.detach(token)
    except Exception:
        logger.debug("failed to detach a turn span -- continuing", exc_info=True)


def end_turn_span(
    handle: Any | None,
    *,
    request: Any = None,
    response: Any = None,
    error: str | None = None,
) -> None:
    """Close a span opened by `start_turn_span`. A no-op for `None`.

    On Langfuse, `request` and `response` become the input and output of the
    turn's observation, which is the trace's root, so they are the trace's
    overall input and output and what an evaluator on that observation reads.
    An `error` (a machine-readable code, never exception text) makes the
    observation's level `ERROR` with `error` as its status message. Off
    Langfuse all three are ignored.

    Callable from any asyncio task: the span's scope is exited in the context
    it was entered in. Swallows any error from closing it, since telemetry
    closing a span must not break whatever the caller was already doing.
    """
    if not isinstance(handle, _TurnSpan):
        return
    if handle.observation is not None:
        try:
            handle.observation.update(
                input=request,
                output=response,
                level="ERROR" if error else None,
                status_message=error or None,
            )
        except Exception:
            logger.debug("failed to record a turn's input and output -- continuing", exc_info=True)
    try:
        handle.context.run(handle.scope.__exit__, None, None, None)
    except Exception:
        logger.debug("failed to close a turn span -- continuing", exc_info=True)


@dataclass(frozen=True, slots=True)
class RequestSpan:
    """The handle `use_request_span` yields: the Langfuse observation it opened,
    if any, and the OpenTelemetry span current in the block."""

    observation: Any | None
    span: Any

    def record_outcome(self, *, level: EventLevel, status_message: str) -> None:
        """Mark the request's span with how the request ended. Never raises.

        Under Langfuse the observation takes `level` and `status_message`, and
        an `ERROR` also sets the span's OpenTelemetry status. Under plain
        OpenTelemetry only an `ERROR` changes the span, to `Status(ERROR,
        status_message)`. The other levels have no OpenTelemetry status, and a
        server span leaves its status unset for a failure the caller caused.
        """
        try:
            if self.observation is not None:
                self.observation.update(level=level, status_message=status_message)
            elif level == "ERROR":
                from opentelemetry.trace import Status, StatusCode

                self.span.set_status(Status(StatusCode.ERROR, status_message))
        except Exception:
            logger.debug("failed to record a request outcome -- continuing", exc_info=True)


@contextmanager
def use_request_span(name: str) -> Iterator[RequestSpan]:
    """Make sure the block runs in a span whose trace the backend records, so
    `current_trace_id` in it names a trace an operator can open, and yield a
    `RequestSpan` that marks that span with the request's outcome.

    For one request that has no span of this service's own, such as an outside
    MCP call. fastmcp opens a SERVER span per request and honours an incoming
    `_meta` traceparent, so under a plain OTel backend the current span is
    already the request's and nothing opens: a second span would split
    fastmcp's one SERVER span per request in two. Langfuse's default span
    filter exports only its own spans and those with `gen_ai.*` attributes,
    which fastmcp's lack, so under Langfuse the block gets an observation
    named `name` from the Langfuse client, in the same trace. With no valid span current (no tracer provider, or the SDK
    disabled), a non-recording span with fresh random ids is made current, so
    every `current_trace_id` in the block still names one id. A span that
    fails to open leaves the block as it was; it never fails the request.

    A block that raises is recorded as an `ERROR`. Under Langfuse its status
    message is the exception's class name. The SDK also records the exception
    itself on the span, text included, as an exception event and the span's
    OpenTelemetry status description, and the export cuts the credentials out
    of any URI in that text by the rule in `common.uri_redaction`.
    """
    from opentelemetry import trace

    observation = None
    with ExitStack() as stack:
        try:
            if _backend == "langfuse":
                from langfuse import get_client

                observation = stack.enter_context(
                    get_client().start_as_current_observation(name=name, as_type="span")
                )
            elif not trace.get_current_span().get_span_context().is_valid:
                stack.enter_context(trace.use_span(_untraced_span(), end_on_exit=False))
        except Exception:
            logger.debug("failed to open request span %s -- continuing as is", name, exc_info=True)
        handle = RequestSpan(observation=observation, span=trace.get_current_span())
        try:
            yield handle
        except Exception as exc:
            handle.record_outcome(level="ERROR", status_message=type(exc).__name__)
            raise


def _untraced_span() -> Any:
    """A non-recording span with fresh random trace and span ids."""
    from opentelemetry import trace

    context = trace.SpanContext(
        trace_id=secrets.randbits(128) or 1,
        span_id=secrets.randbits(64) or 1,
        is_remote=False,
    )
    return trace.NonRecordingSpan(context)


def current_trace_id() -> str:
    """The id of the OpenTelemetry trace current on this task, as 32 hex digits.

    With no valid span current (telemetry off, or nothing traced), a fresh
    random id of the same form, so a log line and the reply it explains still
    share one id an operator can search for.
    """
    from opentelemetry import trace

    span_context = trace.get_current_span().get_span_context()
    if span_context.is_valid:
        return trace.format_trace_id(span_context.trace_id)
    return secrets.token_hex(16)


def record_span(name: str, attributes: dict[str, Any]) -> None:
    """Open and immediately close a child span named `name` carrying `attributes`,
    nesting under whatever span is current on this asyncio task (e.g. the turn
    span `use_turn_span` made current). For a point-in-time fact with no work of
    its own to wrap -- a plan getting approved, say -- a zero-duration span is
    the way OpenTelemetry attaches attributes to a moment on the current trace.

    Branches on `_backend` the same way `start_turn_span` does, and for the
    same reason: Langfuse's span processor exports only spans from its own
    tracer scope, spans carrying a `gen_ai.*` attribute, or spans from a
    listed LLM instrumentor scope, so a plain-OTel span opened while Langfuse
    is the backend is dropped rather than exported. The Langfuse backend
    therefore opens the span through `get_client().start_as_current_observation`
    (`attributes` passed as `metadata`), which nests it under the current
    OTel context -- no `trace_context` needed, since the caller here always
    runs inside a span another instrumentor already made current (e.g. ADK's
    own span). Any other backend keeps using the plain OTel tracer, which
    Langfuse's own auto-configuration is not present to filter.

    Swallows any error opening or closing the span, the same contract as
    `start_turn_span`/`end_turn_span`: a marker span must never break the
    caller recording it.
    """
    try:
        if _backend == "langfuse":
            from langfuse import get_client

            with get_client().start_as_current_observation(
                name=name, as_type="span", metadata=attributes
            ):
                pass
            return

        from opentelemetry import trace

        with trace.get_tracer(__name__).start_as_current_span(name, attributes=attributes):
            pass
    except Exception:
        logger.debug("failed to record span %s -- continuing", name, exc_info=True)


async def flush_telemetry(*, timeout_seconds: float = FLUSH_TIMEOUT_SECONDS) -> None:
    """Export the spans queued so far, waiting at most `timeout_seconds`.
    Never raises (a cancelled caller still sees its `CancelledError`), and
    does nothing when telemetry is off.

    Both exporters batch on a background thread that wakes every five seconds
    by default, and a serverless platform that throttles CPU between requests
    can leave that thread idle while a turn's last spans sit in memory. Calling
    this before a response ends exports them while the request still has CPU.

    The SDK's `flush()` blocks its caller for a whole export and ignores any
    timeout it is given (`BatchSpanProcessor.force_flush` runs the export on
    the calling thread), so it runs on its own daemon thread and this awaits
    that thread for at most `timeout_seconds`. A flush that outlives the wait
    finishes in the background. One flush runs at a time, so a backend that
    hangs cannot pile up threads. A call that finds one running returns at once
    after marking that it wants one; the running thread flushes again before it
    finishes, so spans queued after its export started still go out.
    """
    global _flush_running, _flush_pending  # noqa: PLW0603 -- single-flight state
    if _backend is None:
        return
    with _flush_lock:
        if _flush_running:
            _flush_pending = True
            logger.debug("telemetry flush deferred to the one already running")
            return
        _flush_running = True

    loop = asyncio.get_running_loop()
    finished = asyncio.Event()

    def flush_off_the_loop() -> None:
        global _flush_running, _flush_pending  # noqa: PLW0603 -- single-flight state
        try:
            while True:
                try:
                    _flush_backend()
                except Exception:
                    logger.warning("telemetry flush failed", exc_info=True)
                with _flush_lock:
                    if not _flush_pending:
                        _flush_running = False
                        return
                    _flush_pending = False
        finally:
            with suppress(RuntimeError):  # the loop closed while the flush ran
                loop.call_soon_threadsafe(finished.set)

    try:
        threading.Thread(target=flush_off_the_loop, name=_FLUSH_THREAD_NAME, daemon=True).start()
    except Exception:
        with _flush_lock:
            _flush_running = False
            _flush_pending = False
        logger.warning("telemetry flush thread could not be started", exc_info=True)
        return
    try:
        await asyncio.wait_for(finished.wait(), timeout=timeout_seconds)
    except TimeoutError:
        logger.warning(
            "telemetry flush still running after %.1fs; the response proceeds and the "
            "export finishes in the background",
            timeout_seconds,
        )


def _flush_backend() -> None:
    """Block until the active backend has exported what it has queued."""
    if _backend == "langfuse":
        from langfuse import get_client

        get_client().flush()
        return

    from opentelemetry import trace

    force_flush = getattr(trace.get_tracer_provider(), "force_flush", None)
    if force_flush is not None:
        force_flush()


def record_event(
    name: str,
    *,
    level: EventLevel = "WARNING",
    metadata: Mapping[str, str | int | float | bool] | None = None,
) -> None:
    """Attach a named event to the current trace. Never raises.

    On Langfuse it is an `EVENT` observation under the current span, which
    the exporter's default span filter lets through (an arbitrary attribute
    on a span from another library is not exported). On an OTLP backend it is
    a span event on the current span, with `level` as an attribute. With no
    backend, or no live current span, it does nothing: Langfuse would turn an
    event made outside any span into a one-event trace of its own. Only `str`,
    `int`, `float` and `bool` metadata values are kept, strings are cut to 500
    characters, and at most 32 entries are kept. A failure logs at WARNING the
    first time in a process and at DEBUG afterwards.
    """
    global _event_failure_warned  # noqa: PLW0603 -- one process-wide first-failure latch
    if _backend is None:
        return
    try:
        from opentelemetry import trace

        current = trace.get_current_span()
        if not current.get_span_context().is_valid:
            return
        attributes = _clean_event_metadata(metadata or {})
        name = name[:_MAX_EVENT_STR_CHARS]
        if _backend == "langfuse":
            from langfuse import get_client

            get_client().create_event(name=name, level=level, metadata=attributes)
            return

        current.add_event(name, attributes={**attributes, "level": level})
    except Exception:
        if _event_failure_warned:
            logger.debug("failed to record telemetry event %r -- continuing", name, exc_info=True)
        else:
            _event_failure_warned = True
            logger.warning(
                "failed to record telemetry event %r; later failures log at DEBUG",
                name,
                exc_info=True,
            )


def _clean_event_metadata(metadata: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    """Keep the leading scalar entries of `metadata`, with strings truncated."""
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if len(cleaned) == _MAX_EVENT_METADATA_ITEMS:
            break
        if not isinstance(key, str) or not isinstance(value, str | int | float | bool):
            continue
        cleaned[key[:_MAX_EVENT_STR_CHARS]] = (
            value[:_MAX_EVENT_STR_CHARS] if isinstance(value, str) else value
        )
    return cleaned
