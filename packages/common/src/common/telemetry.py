"""Telemetry bootstrap shared by all dfo-ai services.

`init_telemetry` selects a backend from the environment and `backend()` reports
it: `"langfuse"`, `"otel"` or `"off"`. Service-specific instrumentation (e.g.
ADK or FastMCP span attributes) is layered on top by each service; this module
owns env-var detection so services don't each hand-roll it. Third-party imports
are lazy so the module stays cheap to import when telemetry is off, as it is in
tests and in local dev without creds.

Recognised env vars:
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`: all
  three enable Langfuse export. A deployed service needs all three;
  `LANGFUSE_TRACING_ENVIRONMENT` (read by the Langfuse SDK) tags the traces,
  which separates deployed from local ones in a shared project.
  `LANGFUSE_TRACING_ENABLED=false` and `LANGFUSE_SAMPLE_RATE=0` make the SDK
  export nothing, so with either set the state is off.
- `OTEL_EXPORTER_OTLP_ENDPOINT`: enables a generic OTLP/HTTP span exporter,
  checked only if the Langfuse vars above are not all set.
- `OTEL_SDK_DISABLED=true` (case-insensitive, surrounding whitespace ignored):
  the OpenTelemetry SDK hands out no-op tracers, so whichever backend the two
  groups above select would export nothing; the state is off.
- With none of them set, the state is off.
- `LANGFUSE_RELEASE`, else `K_REVISION` (set by Cloud Run): the Langfuse
  client's `release`, so every exported span names the revision that
  produced it (`deploy_release`). With neither set the Langfuse SDK may take
  a release from a CI variable of its own (`GITHUB_SHA` and others), which
  `deploy_release` does not read.

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

`turn_segment` runs a `with` block inside one segment span of a unit of work
(a harness turn, say) on whichever backend is active, opened and ended by the
block, and `record_turn_outcome` writes how the unit of work ended as a
zero-duration observation. `start_turn_segment`, `use_turn_segment` and
`end_turn_segment` open, enter and end the same segment span from different
places, and end it with a status, level and attributes of the caller's. Both land on one trace seeded on the unit's id, so
a unit of work that spans several requests (a turn paused for approval and
resumed by a second request, on any instance) stays one trace and no span
outlives the request that opened it. `turn_call_meta` gives the `_meta` entries
that carry the turn's session, user and tags to an MCP server, and
`use_call_session` makes the server's spans under a call carry all three.
`record_event` attaches a named event, and `record_span` a zero-duration marker
span, to the current trace. `flush_telemetry` exports what is queued before a
response ends, and `start_telemetry_flush` starts that export without waiting.
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
import hashlib
import logging
import os
import secrets
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager, suppress
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

# The name of the observation `record_turn_outcome` writes for a close, and the
# Langfuse trace name every span of a turn carries, on the server side of its
# MCP calls too.
TURN_SPAN_NAME = "turn"
# The name of the observation `record_turn_outcome` writes for a pause. A reader
# that sees names and no metadata, as the Langfuse BigQuery export does, tells a
# pause from a close by it.
TURN_PAUSE_NAME = "turn.pause"
# The name of the span `turn_segment` opens for each request's share of a turn.
TURN_SEGMENT_NAME = "turn.segment"

# The MCP request `_meta` keys that carry a turn's Langfuse session, user and
# tags to a server (`turn_call_meta`). The MCP spec namespaces a `_meta` key
# with a DNS-style prefix and reserves the prefixes whose second label is
# `modelcontextprotocol` or `mcp`.
SESSION_META_KEY = "org.one.dfo/langfuse-session"
USER_META_KEY = "org.one.dfo/langfuse-user"
TAGS_META_KEY = "org.one.dfo/langfuse-tags"

# The environment variables naming the deployed revision, in the order
# `deploy_release` reads them: the Langfuse SDK's own, which an operator sets
# by hand, then the one Cloud Run sets on every instance.
_RELEASE_VARS = ("LANGFUSE_RELEASE", "K_REVISION")

# The instrumentation scope of the spans the `mcp` SDK opens for each request
# a client sends (`_export_span`).
_MCP_SDK_SCOPE = "mcp-python-sdk"

type Backend = Literal["langfuse", "otel", "off"]
type EventLevel = Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]

_configured = False
# Which backend `init_telemetry` brought up, if any -- "langfuse", "otel", or
# None (nothing configured, or the exporter could not be constructed). Set
# once from the environment and never revisited, and read by `backend()`,
# `_start_turn_span` and `record_event` so none of them needs env lookups of
# its own or can drift from what `init_telemetry` decided.
_backend: Literal["langfuse", "otel"] | None = None
# Whether a `record_event` failure has been logged at WARNING yet; later ones
# log at DEBUG so a broken exporter is visible once without a line per event.
_event_failure_warned = False
# Single-flight state for `flush_telemetry`, guarded by `_flush_lock`:
# `_flush_running` is true while a flush thread owns the backend,
# `_flush_pending` records that a call found it running, and `_flush_waiters`
# holds the wake-up of each call no export pass has yet started for. Each pass
# takes the waiters registered before it starts and wakes them when it ends.
# The thread checks `_flush_pending` and clears `_flush_running` in one locked
# step, so a call is either seen by the running thread or starts its own flush,
# and takes that step before it wakes the pass's waiters.
_flush_lock = threading.Lock()
_flush_running = False
_flush_pending = False
_flush_waiters: list[Callable[[], None]] = []

# Set in the context that entered a `_use_turn_span` block, so leaving the block
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

    The client is built with `_langfuse_exporter` and `_export_span`, the
    SDK's default span filter, which exports its own spans and every span with
    a `gen_ai.*` attribute, so every span ADK opens, widened to the `mcp`
    SDK's spans. The SDK keeps the first client built
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
    client = Langfuse(
        span_exporter=_langfuse_exporter(),
        release=deploy_release(),
        should_export_span=_export_span,
    )
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
    """The handle `_start_turn_span` returns.

    `scope` is the entered span context manager (on Langfuse an `ExitStack`
    that also holds the attribute propagation). `context` is the copy of the
    caller's `contextvars` context it was entered in, and `_end_turn_span`
    exits it there: a context token can only be reset in the context that
    created it, and an async generator can be closed in another context.
    `otel_context` is the OpenTelemetry context that holds the span (and, on
    Langfuse, the trace-root claim the SDK puts beside it), which
    `_use_turn_span` attaches. `observation` is the Langfuse observation the
    span is, `None` off Langfuse; `_end_turn_span` sets its input and output.
    """

    scope: Any
    context: contextvars.Context
    otel_context: Any
    observation: Any | None


def _export_span(span: Any) -> bool:
    """Whether the Langfuse exporter sends `span`: the SDK's default filter,
    or the `mcp` SDK's (`_MCP_SDK_SCOPE`) `MCP send tools/call` span.

    The `mcp` SDK opens a CLIENT span, `MCP send <method>`, for every request
    a client sends, inside fastmcp's own CLIENT span for it, and puts its own
    id in the request's `_meta` traceparent, so the server's span for the
    request takes it as its parent. Of those server spans only `tools/call`
    carries `gen_ai.tool.name` and passes the default filter, so only its
    `MCP send` span is kept: dropped, every tool call's server span would
    reach Langfuse with a parent that is not on the trace. The other methods'
    server spans are dropped with their `MCP send` spans.
    """
    from langfuse import is_default_export_span

    if is_default_export_span(span):
        return True
    scope = span.instrumentation_scope
    return (
        scope is not None
        and scope.name == _MCP_SDK_SCOPE
        and (span.attributes or {}).get("mcp.method.name") == "tools/call"
    )


def deploy_release() -> str | None:
    """The revision this process runs as: `LANGFUSE_RELEASE` when set, else
    `K_REVISION`, or `None` where neither is set, as in local dev and tests.

    The Langfuse client takes it as its `release`, so every span the process
    exports carries it, and the harness's `studio.turn` log line names it.
    Where it is `None` the line names no release, and the SDK may still give
    the spans one from a CI variable (`GITHUB_SHA` and others).
    """
    return next((value for name in _RELEASE_VARS if (value := os.environ.get(name))), None)


@contextmanager
def turn_segment(
    external_id: str,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: Sequence[str] = (),
) -> Iterator[None]:
    """Run the block inside one `TURN_SEGMENT_NAME` span of the unit of work
    identified by `external_id` (a harness turn's wire `trace_id`, say), opened
    on entry and ended on exit. Every span opened and every `record_event` made
    in the block, on this task or on a task started inside it, is a
    descendant of the segment span. The segment carries no input or output.
    `record_turn_outcome` writes those.

    Every segment and outcome of one `external_id` lands in one trace. On
    Langfuse the trace id is `Langfuse.create_trace_id(seed=external_id)`, the
    SDK's documented pattern for correlating an external id with a trace (a
    sha256 of the seed, cut to 32 hex characters), so a person holding only
    `external_id` can recompute it and open the trace. A generic OTLP endpoint
    gets the same trace id. Both backends set `external_id` as the span's
    `dfo.external_id` attribute too. On Langfuse each segment is a root of
    its trace. On OTLP it is a child of the turn's synthetic parent, which
    every segment and outcome of `external_id` shares
    (`_seeded_trace_context`). On both, a unit of work that pauses and
    resumes in another request, or on another instance, holds one segment per
    request.

    `session_id` is the Langfuse session the work belongs to, so turns that
    share it group in Langfuse's Sessions view. It, `user_id`, `tags` and the
    trace name `TURN_SPAN_NAME` reach every span started in the block in this
    process, ADK's included, and only on the Langfuse backend.
    `turn_call_meta` carries the three to an MCP server, whose spans start in
    the server's own context. The Langfuse SDK drops a `session_id` over 200
    characters with a warning, so the trace carries no session.

    A block that raises ends the segment at level `ERROR` on Langfuse, with
    the exception's class name as its status message and no exception text.
    A cancel or a close (`CancelledError`, `GeneratorExit`) ends it at the
    default level, since a stopped or dropped stream is no fault, and the
    turn's outcome (`record_turn_outcome`) carries the `cancelled` code.
    A no-op when telemetry never came up. A span that fails to open or close
    leaves the block running untraced and never raises into the caller.
    """
    handle = start_turn_segment(external_id, session_id=session_id, user_id=user_id, tags=tags)
    error: str | None = None
    try:
        with use_turn_segment(handle):
            yield
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        end_turn_segment(handle, status=error)


def start_turn_segment(
    external_id: str,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: Sequence[str] = (),
) -> Any | None:
    """Open the `TURN_SEGMENT_NAME` span `turn_segment` opens, for a caller
    whose segment starts and ends in different places, and return an opaque
    handle for `use_turn_segment` and `end_turn_segment`. `None` when
    telemetry never came up or the span fails to open, and the caller then
    runs untraced. The span is not current anywhere until `use_turn_segment`
    makes it current for a block."""
    return _start_turn_span(
        external_id,
        name=TURN_SEGMENT_NAME,
        session_id=session_id,
        user_id=user_id,
        tags=tags,
    )


def use_turn_segment(handle: Any | None) -> AbstractContextManager[None]:
    """Make the segment behind `handle` (`start_turn_segment`) current for a
    `with` block, on the block's task and every task started inside it. A
    no-op for `None`. Never ends the span; `end_turn_segment` does."""
    return _use_turn_span(handle if isinstance(handle, _TurnSpan) else None)


def end_turn_segment(
    handle: Any | None,
    *,
    status: str | None = None,
    level: EventLevel = "ERROR",
    attributes: Mapping[str, str] | None = None,
) -> None:
    """End the segment behind `handle` (`start_turn_segment`), from any task.
    A `status` (a machine-readable code) sets the Langfuse observation's
    level to `level` with `status` as its status message. `attributes` go on
    the span, and on Langfuse into the observation's metadata. The segment
    carries no input or output; `record_turn_outcome` writes those. A no-op
    for `None`. Never raises."""
    _end_turn_span(
        handle if isinstance(handle, _TurnSpan) else None,
        error=status,
        level=level,
        attributes=attributes,
    )


# The close codes `record_turn_outcome` records at `WARNING`, defined here
# alone: `cancelled`, a unit of work a Stop or the client's leaving ended,
# and `abandoned`, one a newer unit of work closed. Neither is a failure.
WARNING_TURN_ERRORS = frozenset({"cancelled", "abandoned"})


def record_turn_outcome(
    external_id: str,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: Sequence[str] = (),
    request: Any = None,
    response: Any = None,
    error: str | None = None,
    paused: bool = False,
) -> None:
    """Write how the unit of work identified by `external_id` ended as one
    zero-duration `TURN_SPAN_NAME` observation on its trace (the trace
    `turn_segment` writes to), or that it paused for the user (`paused`) as
    one `TURN_PAUSE_NAME` observation, opened and ended in this call.

    On Langfuse `request` and `response` are the observation's input and
    output, what an evaluator filtered on the name `turn` reads. An `error`
    (a machine-readable code such as `state_not_saved`) makes its level
    `ERROR` with `error` as its status message, and `WARNING` for a code in
    `WARNING_TURN_ERRORS`, so a Stop is never reported as a failure. Its
    metadata holds `dfo.external_id`, and `dfo.turn_error` holds `error`
    when there is one. A clean close writes
    no `dfo.turn_error`. A pause holds `dfo.turn_status` `paused`, and a close
    holds no `dfo.turn_status`. `session_id`, `user_id` and `tags` mean what
    they do for `turn_segment`. Off Langfuse the observation is a plain span
    with the `dfo.external_id` attribute, and the arguments other than
    `paused` are ignored.

    Each call writes its own observation, so a unit of work that paused before
    it closed holds a `TURN_PAUSE_NAME` for each pause and a `TURN_SPAN_NAME`
    for each close, and a reader picks one close by rule. Callable from any
    task, inside a segment or outside one. A no-op when telemetry never came
    up. Never raises.
    """
    metadata = {"dfo.turn_error": error} if error else {}
    if paused:
        metadata["dfo.turn_status"] = "paused"
    handle = _start_turn_span(
        external_id,
        name=TURN_PAUSE_NAME if paused else TURN_SPAN_NAME,
        session_id=session_id,
        user_id=user_id,
        tags=tags,
        metadata=metadata,
    )
    _end_turn_span(
        handle,
        request=request,
        response=response,
        error=error,
        level="WARNING" if error in WARNING_TURN_ERRORS else "ERROR",
    )


@contextmanager
def root_if_span_ended(name: str) -> Iterator[bool]:
    """Run the block inside a short-lived root observation named `name` on the
    current turn's trace when the enclosing `turn_segment`'s span has ended,
    and yield `True`. While that span is live, run the block as it is and
    yield `False`.

    A task started inside a `turn_segment` keeps the segment's context after
    the segment's request ended, as an early mint does (`harness.early_mint`),
    and the root gives what it records a live parent that a flush exports.
    The test reads the segment span the context holds, and a segment the
    sampler dropped counts as live, so a turn it does not export gets no root
    and no flush. A mint offered under an
    inner span that ended while the segment runs records under the live
    segment (`record_event`), which its request flushes. The root takes the
    trace id of the segment and, from its context, the session, user and tags
    `turn_call_meta` reads, and it sits where a `turn` outcome does: a root
    of that trace on Langfuse, a child of the turn's synthetic parent on
    OTLP. The root ends with the block, so a caller that sees `True`
    flushes (`flush_telemetry`) to export it. Never raises.
    """
    handle: _TurnSpan | None = None
    try:
        from opentelemetry import context as otel_context_api
        from opentelemetry import trace

        segment = otel_context_api.get_value(_turn_span_key())
        if not isinstance(segment, trace.Span):
            segment = trace.INVALID_SPAN
        context = segment.get_span_context()
        # A segment the sampler dropped is live and records nothing, so only a
        # sampled one that stopped recording has ended.
        if (
            _backend is not None
            and context.is_valid
            and context.trace_flags.sampled
            and not segment.is_recording()
        ):
            call_meta = turn_call_meta()
            session = call_meta.get(SESSION_META_KEY)
            user = call_meta.get(USER_META_KEY)
            tags = call_meta.get(TAGS_META_KEY)
            handle = _start_turn_span(
                None,
                name=name,
                session_id=session if isinstance(session, str) else None,
                user_id=user if isinstance(user, str) else None,
                tags=[tag for tag in tags if isinstance(tag, str)]
                if isinstance(tags, list)
                else (),
                trace_id=context.trace_id,
            )
    except Exception:
        logger.debug("failed to open a root for an ended span -- continuing", exc_info=True)
    try:
        with _use_turn_span(handle):
            yield handle is not None
    finally:
        _end_turn_span(handle)


def _start_turn_span(
    external_id: str | None,
    *,
    name: str,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: Sequence[str] = (),
    metadata: Mapping[str, str] | None = None,
    trace_id: int | None = None,
) -> _TurnSpan | None:
    """Open one root span named `name` on the trace of `external_id`, or on
    `trace_id` when the caller holds that and not an `external_id` (then the
    span carries no `dfo.external_id`), and return the handle `_use_turn_span` and
    `_end_turn_span` take, or `None` when telemetry never came up or opening
    the span errors.

    The span opens inside a private copy of the caller's context, so it is not
    current on the caller's task and nothing is left attached to it. The copy
    starts from an empty OpenTelemetry context, so whatever span or Langfuse
    trace-root claim is current where it is called (a segment, when a turn
    closes mid-stream), the span is a root of its trace that Langfuse marks as
    one, or on OTLP a child of the turn's synthetic parent, the same shape a
    close on another instance writes. `metadata` joins `dfo.external_id` in
    the Langfuse observation's metadata.
    """
    if _backend is None:
        return None
    try:
        from opentelemetry import context as otel_context_api
        from opentelemetry import trace

        context = contextvars.copy_context()
        # The attach lives in the private copy and dies with it.
        context.run(otel_context_api.attach, otel_context_api.Context())
        if trace_id is None:
            assert external_id is not None, "a turn span names its trace"
            trace_id = _seeded_trace_id(external_id)
        scope, observation = context.run(
            _enter_turn_scope,
            external_id,
            trace_id,
            name,
            session_id,
            user_id,
            list(tags),
            dict(metadata or {}),
        )
        otel_context = context.run(otel_context_api.get_current)
        # Held beside the call meta, so `root_if_span_ended` reads the turn
        # span itself however deep the current span sits under it.
        otel_context = otel_context_api.set_value(
            _turn_span_key(), trace.get_current_span(otel_context), otel_context
        )
        return _TurnSpan(
            scope=scope, context=context, otel_context=otel_context, observation=observation
        )
    except Exception:
        logger.exception("failed to open a turn span for %s -- continuing untraced", external_id)
        return None


def _enter_turn_scope(
    external_id: str | None,
    trace_id: int,
    name: str,
    session_id: str | None,
    user_id: str | None,
    tags: list[str],
    extra_metadata: dict[str, str],
) -> tuple[Any, Any | None]:
    """Open the turn span named `name` as the current span of the calling
    context and return the entered context manager with the Langfuse
    observation, `None` off Langfuse. On Langfuse, the trace name, a
    `session_id`, a `user_id` and the `tags` are propagated to every span
    started under the turn, the session, user and tags are held for
    `turn_call_meta`, and the returned scope exits the attribute propagation
    and the `_hold_turn_call_meta` hold before the span."""
    attributes = {"dfo.external_id": external_id} if external_id is not None else {}
    if _backend == "langfuse":
        from langfuse import get_client, propagate_attributes

        derived = format(trace_id, "032x")
        metadata: dict[str, Any] = {**attributes, **extra_metadata}
        with ExitStack() as stack:
            observation = stack.enter_context(
                get_client().start_as_current_observation(
                    name=name,
                    as_type="span",
                    trace_context={"trace_id": derived},
                    metadata=metadata,
                )
            )
            stack.enter_context(
                propagate_attributes(
                    trace_name=TURN_SPAN_NAME,
                    session_id=session_id or None,
                    user_id=user_id or None,
                    tags=tags or None,
                )
            )
            call_meta: dict[str, Any] = {}
            if session_id:
                call_meta[SESSION_META_KEY] = session_id
            if user_id:
                call_meta[USER_META_KEY] = user_id
            if tags:
                call_meta[TAGS_META_KEY] = list(tags)
            if call_meta:
                stack.enter_context(_hold_turn_call_meta(call_meta))
            scope = stack.pop_all()
        logger.debug("%s span opened: %s -> langfuse trace %s", name, external_id, derived)
        return scope, observation

    from opentelemetry import trace

    scope = trace.get_tracer(__name__).start_as_current_span(
        name, context=_seeded_trace_context(trace_id), attributes=attributes
    )
    scope.__enter__()
    return scope, None


def _seeded_trace_id(external_id: str) -> int:
    """The trace id `Langfuse.create_trace_id(seed=external_id)` computes, the
    first 16 bytes of the seed's sha256, so every span of one `external_id`
    shares a trace on a plain OTLP backend as it does on Langfuse."""
    return int.from_bytes(hashlib.sha256(external_id.encode("utf-8")).digest()[:16])


def _seeded_trace_context(trace_id: int) -> Any:
    """An OpenTelemetry context whose current span is the turn's synthetic
    parent on `trace_id`: remote, sampled, and with a span id read off
    `trace_id` (the first 8 bytes of its sha256, nonzero), so every
    segment, outcome and early-mint root of one turn is a child of the one
    parent, an early-mint root that knows only the trace id included."""
    from opentelemetry import trace

    digest = hashlib.sha256(trace_id.to_bytes(16)).digest()
    parent = trace.SpanContext(
        trace_id=trace_id,
        span_id=int.from_bytes(digest[:8]) or 1,
        is_remote=True,
        trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
    )
    return trace.set_span_in_context(trace.NonRecordingSpan(parent))


@cache
def _turn_call_meta_key() -> str:
    """The OpenTelemetry context key under which a turn holds the `_meta`
    entries for its MCP calls, so they ride with every `_use_turn_span` attach.
    A private key, so no other code can set or read it by name."""
    from opentelemetry import context as otel_context_api

    return otel_context_api.create_key("dfo.turn_call_meta")


@cache
def _turn_span_key() -> str:
    """The OpenTelemetry context key under which a turn span's context holds
    the span itself, so every `_use_turn_span` attach carries it. A private
    key, as `_turn_call_meta_key` is."""
    from opentelemetry import context as otel_context_api

    return otel_context_api.create_key("dfo.turn_span")


@contextmanager
def _hold_turn_call_meta(call_meta: dict[str, Any]) -> Iterator[None]:
    """Hold `call_meta` in the current OpenTelemetry context for the block."""
    from opentelemetry import context as otel_context_api

    token = otel_context_api.attach(otel_context_api.set_value(_turn_call_meta_key(), call_meta))
    try:
        yield
    finally:
        otel_context_api.detach(token)


def turn_call_meta() -> dict[str, Any]:
    """The MCP request `_meta` entries for a call made now: `SESSION_META_KEY`
    (the session), `USER_META_KEY` (the user) and `TAGS_META_KEY` (the tags
    list) for each the enclosing `turn_segment` was opened with, `{}` outside
    a segment or in a segment with none of them.

    The values are read from the OpenTelemetry context the segment span lives
    in. Pass the result as the `meta` of a client's `call_tool`. The server's
    `use_call_session` reads them back. Never raises.
    """
    try:
        from opentelemetry import context as otel_context_api

        call_meta = otel_context_api.get_value(_turn_call_meta_key())
    except Exception:
        logger.debug("failed to read the turn call meta -- continuing", exc_info=True)
        return {}
    if not isinstance(call_meta, dict):
        return {}
    return {str(key): value for key, value in call_meta.items()}


@contextmanager
def use_call_session(
    session_id: str | None, *, user_id: str | None = None, tags: Sequence[str] = ()
) -> Iterator[None]:
    """Make every span opened in the block, and the current span, carry the
    Langfuse session `session_id`, `user_id`, `tags` and the trace name
    `TURN_SPAN_NAME`, for an MCP server handling a call whose client sent them
    in `_meta` (see `turn_call_meta`, which sends them only from inside a turn).

    A no-op for an empty or `None` session id and off the Langfuse backend. The
    attributes stay in this process: they are not put in baggage, so no
    outbound HTTP request carries them. Never raises.
    """
    with ExitStack() as stack:
        if _backend == "langfuse" and session_id:
            try:
                from langfuse import propagate_attributes

                stack.enter_context(
                    propagate_attributes(
                        session_id=session_id,
                        user_id=user_id or None,
                        tags=list(tags) or None,
                        trace_name=TURN_SPAN_NAME,
                        as_baggage=False,
                    )
                )
            except Exception:
                logger.debug("failed to propagate a call session -- continuing", exc_info=True)
        yield


@contextmanager
def _use_turn_span(handle: _TurnSpan | None) -> Iterator[None]:
    """Make the span behind `handle` current for the enclosed block, so every
    span opened and every `record_event` made while it runs, on this task or
    on any task started inside the block, is a descendant of that span in the
    same trace. A no-op for `None`. Never raises and never ends the span; only
    `_end_turn_span` does.

    It attaches a context token, and a token can only be detached in the
    context that made it. A block that an async generator holds open across a
    `yield` is left in another context when the generator is closed outside
    the task that iterated it: by a response that closes its body in the
    request task after the send task was cancelled (the harness's
    `_ClaimedStream`), or by the event loop after its task was cancelled.
    There is nothing to restore in that context, so leaving the block there
    detaches nothing. Any other block is entered and left on one task.
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
    """Undo `_use_turn_span`'s attach, when this is the context that made it."""
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


def _end_turn_span(
    handle: _TurnSpan | None,
    *,
    request: Any = None,
    response: Any = None,
    error: str | None = None,
    level: EventLevel = "ERROR",
    attributes: Mapping[str, str] | None = None,
) -> None:
    """End a span opened by `_start_turn_span`. A no-op for `None`.

    On Langfuse, `request` and `response` become the observation's input and
    output, and an `error` (a machine-readable code, with no exception text)
    sets its level to `level` with `error` as its status message. Off
    Langfuse `request`, `response`, `error` and `level` are ignored.
    `attributes` go on the span itself, on any backend, and on Langfuse into
    the observation's metadata as well.

    Callable from any asyncio task: the span's scope is exited in the context
    it was entered in. Swallows any error from closing it, since telemetry
    closing a span must not break whatever the caller was already doing.
    """
    if not isinstance(handle, _TurnSpan):
        return
    if attributes:
        try:
            from opentelemetry import trace

            trace.get_current_span(handle.otel_context).set_attributes(dict(attributes))
        except Exception:
            logger.debug("failed to set a turn span's attributes -- continuing", exc_info=True)
    if handle.observation is not None:
        try:
            handle.observation.update(
                input=request,
                output=response,
                level=level if error else None,
                status_message=error or None,
                **({"metadata": dict(attributes)} if attributes else {}),
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
    span `turn_segment` made current). For a point-in-time fact with no work of
    its own to wrap -- a plan getting approved, say -- a zero-duration span is
    the way OpenTelemetry attaches attributes to a moment on the current trace.

    Branches on `_backend` the same way `_start_turn_span` does, and for the
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
    `turn_segment`: a marker span must never break the
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
    hangs cannot pile up threads. A call that finds one running marks that it
    wants one and waits, within the same bound, for that thread, which flushes
    again before it finishes, so spans queued after its export started still
    go out. Each call wakes when the first export pass that started after it
    ends, so under a steady stream of calls each waits at most about two
    passes, and a call that stops waiting takes its wake-up back.
    """
    if _backend is None:
        return
    loop = asyncio.get_running_loop()
    finished = asyncio.Event()

    def wake() -> None:
        with suppress(RuntimeError):  # the loop closed while the flush ran
            loop.call_soon_threadsafe(finished.set)

    if not _start_flush(wake):
        return
    try:
        await asyncio.wait_for(finished.wait(), timeout=timeout_seconds)
    except TimeoutError:
        logger.warning(
            "telemetry flush still running after %.1fs; the response proceeds and the "
            "export finishes in the background",
            timeout_seconds,
        )
    finally:
        with _flush_lock, suppress(ValueError):  # a pass already took it
            _flush_waiters.remove(wake)


def flush_telemetry_before_exit() -> None:
    """Export everything queued so far, blocking until the export ends, for a
    process about to exit, such as a job whose `flush_telemetry` may have
    returned while its export ran on a daemon thread that exit would kill.
    Never raises, and does nothing when telemetry is off."""
    if _backend is None:
        return
    try:
        _flush_backend()
    except Exception:
        logger.warning("telemetry flush before exit failed", exc_info=True)


def start_telemetry_flush() -> None:
    """Start the export `flush_telemetry` runs and return without waiting for
    it, for a caller that must not await, such as a cancelled task's unwind.
    Never raises, and does nothing when telemetry is off."""
    if _backend is not None:
        _start_flush(None)


def _start_flush(on_done: Callable[[], None] | None) -> bool:
    """Have `flush_telemetry`'s single-flight flush thread run, start it when
    none runs, and say whether a thread will call `on_done`. A call that finds
    one running marks that it wants another pass. Each pass takes the
    `on_done` callbacks registered before it starts and calls them when it
    ends, so a callback runs after the first full pass that follows its
    registration. `False` only when no thread could be started."""
    global _flush_running, _flush_pending  # noqa: PLW0603 -- single-flight state
    with _flush_lock:
        if on_done is not None:
            _flush_waiters.append(on_done)
        if _flush_running:
            _flush_pending = True
            logger.debug("telemetry flush deferred to the one already running")
            return True
        _flush_running = True

    def flush_off_the_loop() -> None:
        global _flush_running, _flush_pending  # noqa: PLW0603 -- single-flight state
        while True:
            with _flush_lock:
                waiters = _flush_waiters[:]
                _flush_waiters.clear()
                _flush_pending = False
            try:
                _flush_backend()
            except Exception:
                logger.warning("telemetry flush failed", exc_info=True)
            # Settle the state before waking anyone, so a woken caller finds
            # this thread finished unless another pass was asked for.
            with _flush_lock:
                done = not _flush_pending
                if done:
                    _flush_running = False
            for waiter in waiters:
                waiter()
            if done:
                return

    try:
        threading.Thread(target=flush_off_the_loop, name=_FLUSH_THREAD_NAME, daemon=True).start()
    except Exception:
        with _flush_lock:
            _flush_running = False
            _flush_pending = False
            _flush_waiters.clear()
        logger.warning("telemetry flush thread could not be started", exc_info=True)
        return False
    return True


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
    a span event on the current span, with `level` as an attribute; when that
    span has ended, as an inner span an early mint was offered under, it goes
    on the live `turn_segment` span the context holds. With no
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

        if not current.is_recording():
            from opentelemetry import context as otel_context_api

            segment = otel_context_api.get_value(_turn_span_key())
            if isinstance(segment, trace.Span) and segment.is_recording():
                current = segment
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
