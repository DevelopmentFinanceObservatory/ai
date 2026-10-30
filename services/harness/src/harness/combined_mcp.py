"""The combined MCP server: the one endpoint an outside MCP client connects to.

`build_combined_mcp` returns a FastMCP parent that mounts data-mcp's,
analysis-mcp's and presentation-mcp's server objects unnamespaced, so an
outside client sees one server carrying every data, analysis and
presentation tool, and presentation-mcp's `ui://` renderer resource. The
tools only the harness calls (`HARNESS_ONLY_TOOLS`) are left out: an outside
client neither lists nor calls them.
`mount()` runs a child's tool in process through the child's own
middleware, with no second MCP session. `harness.server.create_app` serves
the parent at `COMBINED_MCP_MOUNT_PATH` beside the three per-server mounts,
which the harness's own clients keep calling and which answer loopback
clients alone. presentation-mcp's per-server mount also serves the theme
stylesheet a view's iframe loads, which the parent does not.

The parent mounts the very objects the per-server mounts serve. FastMCP
reference-counts a server's lifespan per object, so a child that is served
on its own mount and mounted here starts once. A second `FastMCP` built with
a child's lifespan would keep its own count and run that lifespan again,
replacing the stores the first run installed. `children_start_once` refuses
startup when that happens.

Middleware added to the parent runs only for calls that enter through the
combined server. A call to a per-server mount runs its child's own
middleware alone, so the parent's chain is where behaviour for outside
callers goes. `build_combined_mcp` adds it outermost first:

- `_ExternalCallBudget` runs each call under data-mcp's
  `EXTERNAL_CALL_BUDGET`, whose `get_data` deadline ends 3 s inside the
  call limit.
- `_RequestLog` runs each tool call, resource read and resource listing in a span the
  telemetry backend records, marks a refusal or a raised exception on that
  span, and logs its tool or resource URI, outcome code, protocol revision,
  caller, trace id, and the client's name, version and MCP Apps declaration.
  It logs each initialize request with the same client fields.
- `_InFlightLimit` refuses a tool call, resource read or resource listing
  while `Settings.combined_mcp_max_in_flight` others run on this instance.
- `_ToolCallRateLimit` draws every tool call from one token bucket.
- `_CallLimit` cancels a tool call or resource read still running at
  `EXTERNAL_CALL_LIMIT_S`, 45 s, and a resource listing, which reads the
  skill catalogue upstream (`data_mcp.skills_resources`), at
  `_SKILLS_READ_LIMIT_S`, 10 s.
- `_ShapeResults` drops `structuredContent` and `search`'s envelope in
  `_meta` from successful results of tools that declare no MCP App view,
  and `outputSchema` from `tools/list`.
- `_PublicArguments` lists and calls each tool in the public form its child
  declares (`PUBLIC_SHAPES`, `common.public_shape`): `get_data` without
  `place_types` and with it true on every call, every verb without
  `rolling_follows_tree` and `rebuild_lost_frames` and with both true,
  `describe_frame` and `write_component` without `rebuild_lost_frames` and
  with it true, and `growth`, `rank` and `filter` without the harness's
  `rolling` and `anchor_key`. A call that sends a hidden argument the public form does not
  force is refused as an unknown argument; a forced one is accepted and
  overwritten.
- `_RefuseRaisedErrors` answers a call that raised with a refusal: an
  unknown tool name, arguments that fail the tool's input schema, or an
  exception from the tool itself. A resource read that raised an
  unexpected exception fails with fixed text.

The parent registers the Skills over MCP extension
(`data_mcp.skills_extension.SkillsExtension`), since a mounted child's
extensions do not reach it. Middleware sees no extension method, so
`_guard_extension_method` wraps each of its handlers in the request span, a
slot of the same `_InFlightLimit`, and `_SKILLS_READ_LIMIT_S`. The
token bucket is left out: each method costs one Skills API read, and a host
calls them once per session.

The two admission checks sit outside the call limit, so a refusal they
write is immediate and spends no time from it. The in-flight cap sits
outside the token bucket, so a call it refuses spends no token.

The refusals the parent writes itself (`rate_limited`, `service_unavailable`,
`tool_not_found`, `request_invalid`, `internal_error`) carry their envelope
in the text through `common.refusal.refusal_text`, and their code in `_meta`
through `common.refusal.refusal_meta`, as data-mcp's and analysis-mcp's
`to_tool_result` do. A resource read has no envelope, so its
failures put the code at the head of the `ResourceError` text, as
`"<code>: <prose>"`, as data-mcp's artifact resource does.

data-mcp limits the apis mint slots outside callers hold, since only it
knows when an upstream mint has ended (`data_mcp.mint_ledger`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from importlib import metadata
from typing import Any, override

from analysis_mcp.server import PUBLIC_SHAPES as ANALYSIS_PUBLIC_SHAPES
from analysis_mcp.server import healthz as analysis_healthz
from analysis_mcp.server import mcp as analysis_mcp
from common.auth import BearerTokenVerifier
from common.caller import current_caller
from common.echoed_text import truncate_value
from common.materialisation import get_store as get_frame_store
from common.public_shape import PublicShape
from common.refusal import (
    REFUSAL_META_KEY,
    internal_error_message,
    log_unexpected_error,
    refusal_meta,
    refusal_text,
)
from common.telemetry import EventLevel, RequestSpan, current_trace_id, use_request_span
from common.uri_redaction import loggable_uri
from data_mcp.deadline import (
    DEADLINE_RETRY_AFTER_S,
    EXTERNAL_CALL_BUDGET,
    EXTERNAL_CALL_LIMIT_S,
    use_call_budget,
)
from data_mcp.definition_store import get_store as get_definition_store
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.server import PUBLIC_SHAPES as DATA_PUBLIC_SHAPES
from data_mcp.server import healthz as data_healthz
from data_mcp.server import mcp as data_mcp
from data_mcp.skills_extension import SkillsExtension
from fastmcp import FastMCP
from fastmcp.apps import UI_EXTENSION_ID
from fastmcp.exceptions import FastMCPError, NotFoundError, ResourceError, ValidationError
from fastmcp.resources.base import Resource, ResourceResult
from fastmcp.server.extensions import ExtensionRequestHandler
from fastmcp.server.http import StarletteWithLifespan
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.server.middleware.rate_limiting import TokenBucketRateLimiter
from fastmcp.tools.base import Tool, ToolResult
from mcp import types as mt
from mcp.server.context import ServerRequestContext
from mcp.shared.exceptions import MCPError
from mcp_types import INTERNAL_ERROR
from presentation_mcp.server import PUBLIC_SHAPES as PRESENTATION_PUBLIC_SHAPES
from presentation_mcp.server import healthz as presentation_healthz
from presentation_mcp.server import mcp as presentation_mcp
from presentation_mcp.server import view_support
from pydantic import ValidationError as PydanticValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import BaseRoute
from starlette.types import Lifespan, StatelessLifespan

from harness.settings import Settings

logger = logging.getLogger(__name__)

COMBINED_MCP_NAME = "dfo"

# What a client reads when it connects, before it lists a tool. It names every
# tool the combined server lists.
COMBINED_MCP_INSTRUCTIONS = (
    "The Development Finance Observatory serves development-finance data, such as aid, "
    "debt and economic indicators. `search` turns a data question into dataset "
    "definitions, each identified by a spec_id, and `get_data` fetches one as a frame, "
    "with its source. `sources` lists what DFO holds and how a dataset is made; "
    "`describe` explains graph ids read from a result. The analysis tools "
    "(`aggregate`, `compare`, `growth`, `rank`, `share_of`, `filter`, `join`) take frames "
    "and return a new frame. `describe_frame` summarises a frame, and "
    "`write_component` turns a frame into a chart, table or card, which "
    "`describe_component` reads back. `export_data` turns a frame into a CSV, Excel, "
    "Parquet or zip file and returns a lasting link to it. A `search` result can list "
    "skills: `get_skill` loads one, or a file it lists."
)

# data-mcp's `resolve_definition`, which resolves each Search result a Studio
# turn starts from, and a seed's identity over a plan's years, again through
# Search and stores its definition, and `catalogue_counts`, the counts
# `explain_dfo` quotes. `sources` gives an outside client the same counts. The
# harness calls both at `DATA_MCP_URL`
# (`translator.discovery.resolve_definition`,
# `harness.dfo_faq.read_catalogue_counts`). The harness also calls
# `sum_of`, `quotient_of` and `get_skill_file`, whose work an outside client
# reaches through `compare`, `share_of` and `get_skill` (`_MERGED_NAMES`), and
# `resolve_anchor`, which writes the `rolling` and `anchor_key` arguments the
# public list leaves out (`analysis_mcp.server.PUBLIC_SHAPES`).
HARNESS_ONLY_TOOLS = frozenset(
    {
        "resolve_definition",
        "catalogue_counts",
        "sum_of",
        "quotient_of",
        "get_skill_file",
        "resolve_anchor",
    }
)

# Each harness-only name whose work a listed tool does, with that tool and the
# `operation` that picks the form. A call to one refuses `tool_not_found` naming
# the listed call.
_MERGED_NAMES: dict[str, tuple[str, str | None]] = {
    "sum_of": ("compare", "sum"),
    "quotient_of": ("share_of", "quotient"),
    "get_skill_file": ("get_skill", None),
}

# How the public list shows and calls each tool whose form differs from its
# mount's, as each child declares it beside the tool.
_PUBLIC_SHAPES: dict[str, PublicShape] = {
    **DATA_PUBLIC_SHAPES,
    **ANALYSIS_PUBLIC_SHAPES,
    **PRESENTATION_PUBLIC_SHAPES,
}

# The wait a caller is told before repeating a call the in-flight cap refused. A call
# runs for up to `EXTERNAL_CALL_LIMIT_S`, and most finish within a few seconds.
_IN_FLIGHT_RETRY_AFTER_S = 5.0

# The longest an extension method or a `resources/list` runs. Each makes one
# Skills API GET, which data-mcp does not cache, so this sits well above a slow
# read and well below the call limit.
_SKILLS_READ_LIMIT_S = 10.0

# Each mounted child with its own `/healthz` handler, which the parent's
# `/healthz` calls in process. An outside client's `write_component` reaches
# presentation-mcp here directly, so it does not pass the rules for a card whose
# period follows the latest data, which `harness.mcp_client.McpExecutor` applies
# to the Studio's own writes (`harness.rolling_cards`).
_CHILDREN: tuple[tuple[FastMCP, Callable[[Request], Awaitable[Response]]], ...] = (
    (data_mcp, data_healthz),
    (analysis_mcp, analysis_healthz),
    (presentation_mcp, presentation_healthz),
)


class _CombinedMCP(FastMCP):
    """A FastMCP parent whose HTTP app carries only the routes it registers.

    FastMCP copies every mounted child's `@custom_route` onto the parent's
    app, after the gated MCP route and outside the auth gate
    (`TransportMixin._get_additional_http_routes`). Every child registers
    `/healthz`, so each would sit, unreachable, under the parent's own.
    presentation-mcp's `/theme-{version}.css` stays on its per-server mount,
    at the URL `PRESENTATION_PUBLIC_URL` names.
    """

    @override
    def _get_additional_http_routes(self) -> list[BaseRoute]:
        return list(self._additional_http_routes)


def build_combined_mcp(
    *, settings: Settings, call_limit_s: float = EXTERNAL_CALL_LIMIT_S
) -> FastMCP:
    """Build the combined server over data-mcp's, analysis-mcp's and presentation-mcp's
    server objects.

    The parent carries its own `BearerTokenVerifier`; a child's `auth` is
    never consulted under a parent. Call `common.auth.apply_gate` on the
    result before building its app, as for every other mount. `settings`
    sizes the rate limit and the in-flight cap, which the Skills over MCP
    extension's methods share. `call_limit_s` is the longest a tool call or
    resource read runs; tests shorten it.

    `/healthz` reports each child's own `/healthz` payload under the child's
    name. It answers 200 with `status: "ok"` when every child answered 200
    with `status: "ok"` in its body, and 503 with `status: "degraded"`
    otherwise. presentation-mcp answers 200 with `status: "degraded"` when
    it started without the platform's tools, which it does only with
    `PRESENTATION_RELEASE_REQUIRED` false; otherwise it fails startup.
    """
    combined = _CombinedMCP(
        name=COMBINED_MCP_NAME,
        instructions=COMBINED_MCP_INSTRUCTIONS,
        version=metadata.version("harness"),
        auth=BearerTokenVerifier(),
    )
    for child, _ in _CHILDREN:
        combined.mount(child)
    combined.disable(names=set(HARNESS_ONLY_TOOLS), components={"tool"})
    in_flight = _InFlightLimit(max_in_flight=settings.combined_mcp_max_in_flight)
    for middleware in (
        _ExternalCallBudget(),
        _RequestLog(),
        in_flight,
        _ToolCallRateLimit(
            calls_per_s=settings.combined_mcp_requests_per_s, burst=settings.combined_mcp_burst
        ),
        _CallLimit(limit_s=call_limit_s),
        _ShapeResults(server=combined),
        _RefuseRaisedErrors(server=combined),
        _PublicArguments(),
    ):
        combined.add_middleware(middleware)
    combined.add_extension(
        SkillsExtension(
            guard=lambda method, handler: _guard_extension_method(
                handler, method=method, in_flight=in_flight
            )
        )
    )

    @combined.custom_route("/healthz", methods=["GET"])
    async def healthz(request: Request) -> JSONResponse:
        servers: dict[str, object] = {}
        healthy = True
        for child, child_healthz in _CHILDREN:
            response = await child_healthz(request)
            body = json.loads(bytes(response.body))
            servers[child.name] = body
            healthy = healthy and response.status_code == 200 and body.get("status") == "ok"
        return JSONResponse(
            {"status": "ok" if healthy else "degraded", "servers": servers},
            status_code=200 if healthy else 503,
        )

    return combined


def _refusal(
    *,
    code: str,
    retry_mode: str,
    message: str,
    fix: dict[str, Any],
    retry_after_s: float | None = None,
) -> ToolResult:
    """A refusal the parent writes itself, with the text the children's
    `to_tool_result` writes for the same fields."""
    envelope: dict[str, Any] = {"code": code, "retry_mode": retry_mode, "fix": fix}
    if retry_after_s is not None:
        envelope["retry_after_s"] = retry_after_s
    return ToolResult(
        content=refusal_text(
            message=message,
            code=code,
            retry_mode=retry_mode,
            fix=fix,
            retry_after_s=retry_after_s,
        ),
        structured_content={**envelope, "message": message},
        meta=refusal_meta(code=code),
        is_error=True,
    )


class _ExternalCallBudget(Middleware):
    """Runs each tool call under `EXTERNAL_CALL_BUDGET`, which `get_data` reads to
    size its deadline. A call to data-mcp's own mount keeps the direct budget.

    That deadline ends `EXTERNAL_CALL_MARGIN_S` before `EXTERNAL_CALL_LIMIT_S`, so
    `get_data`'s own refusal at its deadline reaches the caller before `_CallLimit`
    cancels the call."""

    @override
    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        with use_call_budget(EXTERNAL_CALL_BUDGET):
            return await call_next(context)


class _RequestLog(Middleware):
    """Runs each tool call, resource read and resource listing in a recorded span,
    and logs one line for it: the tool, the outcome, the protocol revision, the caller, the trace
    id, and the host fields. Logs one line for each initialize request too, with
    the same fields.

    The host fields name the client (`clientInfo`'s name and version) and say
    what it declared about drawing MCP Apps views. `views` is
    `presentation_mcp.server.view_support` of its capabilities, the reading
    presentation-mcp's tools answer by, and `view_mime_types` the mime types
    it listed under the `io.modelcontextprotocol/ui` extension. A 2025-11-25
    client sends both on initialize, and its later requests reach this
    stateless server with neither, so they log no client and `views=unknown`.
    A 2026-07-28 client sends its capabilities on every request, and its
    client info when it chooses to. Every client-supplied string is cut to
    `_CLIENT_TEXT_LIMIT` characters and written as a JSON string, so it cannot
    forge another field. A resource read logs its URI as `uri`: a `ui://` URI
    whole, and any other URI as its scheme and host alone.

    fastmcp opens a SERVER span per request around this whole chain, and an
    `internal_error`'s logged trace id (`common.refusal.log_unexpected_error`) is
    that span's trace. `common.telemetry.use_request_span` covers the two cases where
    that trace would not open in the backend: under Langfuse, which exports
    only its own spans, it adds a Langfuse observation in the same trace; with
    no tracer provider it makes one random id current, so the log lines of one
    call still agree. A tool call's observation is named for the tool as
    the log line writes it; a resource read's has a fixed name, since a
    `dfo://artifact` URI is a bearer credential.

    The outcome is `ok`, the refusal's code, or the name of the exception the
    call raised. A refusal's code is the one every mounted service writes in
    the result's `_meta` under `REFUSAL_META_KEY` (`common.refusal.refusal_meta`),
    since presentation-mcp's `structuredContent` names it `error`, or holds a
    view in its place for an MCP Apps host. No line carries a header, a tool's
    arguments or its result.
    The span records a refusal at the level `_refusal_level` gives its code,
    and `use_request_span` records a raised exception as an `ERROR`. A call
    that succeeds keeps the default level.
    """

    @override
    async def on_initialize(
        self,
        context: MiddlewareContext[mt.InitializeRequest],
        call_next: CallNext[mt.InitializeRequest, mt.InitializeResult | None],
    ) -> mt.InitializeResult | None:
        params = context.message.params if context.message is not None else None
        try:
            result = await call_next(context)
        except Exception as exc:
            _log_initialize(params, outcome=type(exc).__name__, revision=None)
            raise
        revision = result.protocol_version if result is not None else None
        _log_initialize(params, outcome="ok", revision=revision)
        return result

    @override
    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        name = _loggable_tool_name(context.message.name)
        subject = f"tool={name}"
        with use_request_span(f"combined mcp tools/call {name}") as span:
            try:
                result = await call_next(context)
            except Exception as exc:
                _log_request(context, subject=subject, outcome=type(exc).__name__)
                raise
            outcome = _record_refusal(span, result) if result.is_error else "ok"
            _log_request(context, subject=subject, outcome=outcome)
            return result

    @override
    async def on_read_resource(
        self,
        context: MiddlewareContext[mt.ReadResourceRequestParams],
        call_next: CallNext[mt.ReadResourceRequestParams, ResourceResult],
    ) -> ResourceResult:
        uri = loggable_uri(str(context.message.uri))
        with use_request_span("combined mcp resources/read"):
            try:
                result = await call_next(context)
            except Exception as exc:
                _log_request(context, subject="resource", outcome=type(exc).__name__, uri=uri)
                raise
            _log_request(context, subject="resource", outcome="ok", uri=uri)
            return result

    @override
    async def on_list_resources(
        self,
        context: MiddlewareContext[mt.ListResourcesRequest],
        call_next: CallNext[mt.ListResourcesRequest, Sequence[Resource]],
    ) -> Sequence[Resource]:
        with use_request_span("combined mcp resources/list"):
            try:
                result = await call_next(context)
            except Exception as exc:
                _log_request(context, subject="resource_list", outcome=type(exc).__name__)
                raise
            _log_request(context, subject="resource_list", outcome="ok")
            return result


# Refusal codes a call's span records as an `ERROR`: the server or a service behind
# it failed. presentation-mcp answers `transient_unavailable` when the platform
# could not be reached or failed, and passes on the platform's `timeout`. It
# refuses `publish_not_configured`, or passes on the platform's
# `publish_not_permitted`, when this deployment cannot publish a frame, and passes
# on `publish_conflict` when the platform holds an analysis-mcp frame's
# content-addressed pair with other bytes or derivation. The platform's
# `release_mismatch` is a catalogue release this deployment does not serve, or a
# stored payload awaiting migration (503), and `capability_unavailable` a surface
# or format this deployment does not mount (501).
# `service_unavailable` includes a call the frame row budget (`common.row_budget`)
# had no room for, which marks an instance sized below its load, as data-mcp's
# `mint_slots_full` does for apis mint slots.
# `operation_still_processing` is a call still running, which the caller collects
# later, so it keeps the default level. Every other refusal answers something the
# caller can change or wait out, and is a `WARNING`.
_ERROR_CODES = frozenset(
    {
        "upstream_error",
        "service_unavailable",
        "internal_error",
        "frame_integrity_failed",
        "transient_unavailable",
        "timeout",
        "publish_not_configured",
        "publish_not_permitted",
        "publish_conflict",
        "release_mismatch",
        "capability_unavailable",
    }
)


def _refusal_level(code: str | None) -> EventLevel:
    """The level a refusal with `code` gives its call's span. A refusal with no
    code breaks the envelope every tool writes, so it is an `ERROR`."""
    if code is None or code in _ERROR_CODES:
        return "ERROR"
    if code == "operation_still_processing":
        return "DEFAULT"
    return "WARNING"


def _record_refusal(span: RequestSpan, result: ToolResult) -> str:
    """Mark `span` with the refusal `result` carries and return its code, `error`
    for a refusal with none. The code and the upstream service's own are read
    from `result`'s `_meta` under `REFUSAL_META_KEY`. The status message is the
    code, followed by `: <upstream_code>` when the refusal names one."""
    refusal = (result.meta or {}).get(REFUSAL_META_KEY) or {}
    raw_code = refusal.get("code")
    code = str(raw_code) if raw_code else None
    upstream_code = refusal.get("upstream_code")
    outcome = code or "error"
    span.record_outcome(
        level=_refusal_level(code),
        status_message=f"{outcome}: {upstream_code}" if upstream_code else outcome,
    )
    return outcome


# The longest client-supplied string a log line carries, and the most mime types.
_CLIENT_TEXT_LIMIT = 128
_MIME_TYPES_LIMIT = 8
# A tool name the log line writes bare.
_TOOL_NAME = re.compile(rf"[A-Za-z0-9_.\-]{{1,{_CLIENT_TEXT_LIMIT}}}")


def _log_request(
    context: MiddlewareContext[Any], *, subject: str, outcome: str, uri: str | None = None
) -> None:
    fastmcp_context = context.fastmcp_context
    request_context = fastmcp_context.request_context if fastmcp_context else None
    try:
        session = fastmcp_context.session if fastmcp_context else None
    except RuntimeError:
        session = None
    if session is None:
        # A call with no session is answered as `UNDECLARED`, which empty capabilities give.
        client_info = None
        capabilities = mt.ClientCapabilities()
    else:
        client_info = session.client_params.client_info if session.client_params else None
        capabilities = session.client_capabilities
    logger.info(
        "combined mcp call %s outcome=%s revision=%s caller=%s trace_id=%s %s%s",
        subject,
        outcome,
        request_context.protocol_version if request_context else None,
        current_caller(),
        current_trace_id(),
        _host_fields(client_info=client_info, capabilities=capabilities),
        f" uri={_client_text(uri)}" if uri is not None else "",
    )


def _log_initialize(
    params: mt.InitializeRequestParams | None, *, outcome: str, revision: str | None
) -> None:
    """Log an initialize request, whose `params` carry a 2025-11-25 client's
    `clientInfo` and capabilities. `revision` is the one the server answered with."""
    logger.info(
        "combined mcp initialize outcome=%s revision=%s caller=%s %s",
        outcome,
        revision,
        current_caller(),
        _host_fields(
            client_info=params.client_info if params else None,
            capabilities=params.capabilities if params else None,
        ),
    )


def _host_fields(
    *,
    client_info: mt.Implementation | None,
    capabilities: mt.ClientCapabilities | None,
) -> str:
    """The `client`, `client_version`, `views` and `view_mime_types` fields.

    `views` is `view_support(capabilities)`.
    `view_mime_types` lists what the client put under the MCP Apps
    extension's `mimeTypes`, and is `None` when it listed none.
    """
    settings = (capabilities.extensions or {}).get(UI_EXTENSION_ID) if capabilities else None
    mime_types = settings.get("mimeTypes") if isinstance(settings, dict) else None
    listed = (
        json.dumps([str(mime)[:_CLIENT_TEXT_LIMIT] for mime in mime_types[:_MIME_TYPES_LIMIT]])
        if isinstance(mime_types, list)
        else None
    )
    name = _client_text(client_info.name) if client_info else None
    version = _client_text(client_info.version) if client_info else None
    return (
        f"client={name} client_version={version} views={view_support(capabilities)} "
        f"view_mime_types={listed}"
    )


def _client_text(text: str) -> str:
    """`text`, cut to `_CLIENT_TEXT_LIMIT` characters, as a JSON string."""
    return json.dumps(text[:_CLIENT_TEXT_LIMIT])


def _loggable_tool_name(name: str) -> str:
    """`name` bare when it is a plain tool name, and as a JSON string otherwise,
    since a client can send any string, including one that is no tool's."""
    return name if _TOOL_NAME.fullmatch(name) else _client_text(name)


class _InFlightLimit(Middleware):
    """Refuses a tool call, resource read or resource listing while `max_in_flight`
    others run.

    The token bucket limits how often calls start; this limits how many run at
    once on this instance. A tool call over the cap refuses `rate_limited`, a
    resource read fails with a `ResourceError` marked `rate_limited`, and a
    resource listing, or an extension method, which takes its slot through
    `_guard_extension_method`, fails with `-32603` marked the same way. FastMCP
    answers any other exception a listing raises with fixed text. None waits for a slot, since a
    queued call would spend its own time limit waiting. The check and the count's update run on the event loop with no
    await between them, so concurrent calls need no lock.
    """

    def __init__(self, *, max_in_flight: int) -> None:
        self._max_in_flight = max_in_flight
        self._in_flight = 0

    @override
    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        if self.full:
            return _refusal(
                code="rate_limited",
                retry_mode="retry_after_delay",
                message=f"{self.full_prose()} Repeat the call unchanged after "
                f"{_IN_FLIGHT_RETRY_AFTER_S:g}s.",
                fix={"max_in_flight": self._max_in_flight},
                retry_after_s=_IN_FLIGHT_RETRY_AFTER_S,
            )
        with self.slot():
            return await call_next(context)

    @override
    async def on_read_resource(
        self,
        context: MiddlewareContext[mt.ReadResourceRequestParams],
        call_next: CallNext[mt.ReadResourceRequestParams, ResourceResult],
    ) -> ResourceResult:
        if self.full:
            raise ResourceError(
                f"rate_limited: {self.full_prose()} Read the resource again after "
                f"{_IN_FLIGHT_RETRY_AFTER_S:g}s."
            )
        with self.slot():
            return await call_next(context)

    @override
    async def on_list_resources(
        self,
        context: MiddlewareContext[mt.ListResourcesRequest],
        call_next: CallNext[mt.ListResourcesRequest, Sequence[Resource]],
    ) -> Sequence[Resource]:
        if self.full:
            logger.warning("combined mcp: resources/list refused over the in-flight cap")
            raise self.refused(retry="List the resources again")
        with self.slot():
            return await call_next(context)

    @property
    def full(self) -> bool:
        """Whether `max_in_flight` calls are running."""
        return self._in_flight >= self._max_in_flight

    def full_prose(self) -> str:
        """The sentence a refusal over the cap opens with."""
        return (
            f"This server is already running as many calls as it runs at once "
            f"({self._max_in_flight})."
        )

    def refused(self, *, retry: str) -> MCPError:
        """The `-32603` a JSON-RPC request over the cap fails with; `retry` opens
        the sentence naming the wait."""
        return MCPError(
            code=INTERNAL_ERROR,
            message=f"rate_limited: {self.full_prose()} {retry} after "
            f"{_IN_FLIGHT_RETRY_AFTER_S:g}s.",
        )

    @contextmanager
    def slot(self) -> Iterator[None]:
        """Count the block as one call in flight."""
        self._in_flight += 1
        try:
            yield
        finally:
            self._in_flight -= 1


def _guard_extension_method(
    handler: ExtensionRequestHandler, *, method: str, in_flight: _InFlightLimit
) -> ExtensionRequestHandler:
    """`handler` run as middleware runs a resource read: in a request span, in a
    slot of `in_flight`, and within `_SKILLS_READ_LIMIT_S`.

    Over the in-flight cap it fails at once with `-32603`, as does a call the
    limit stops. An exception that is not an `MCPError`, which the handler
    raises with text written for the caller, fails with fixed `internal_error`
    text, and the exception is logged under a trace id.
    """

    async def guarded(ctx: ServerRequestContext[Any, Any], params: Any) -> Any:
        with use_request_span(f"combined mcp {method}"):
            if in_flight.full:
                raise in_flight.refused(retry="Repeat the request")
            with in_flight.slot():
                limit = asyncio.timeout(_SKILLS_READ_LIMIT_S)
                try:
                    async with limit:
                        return await handler(ctx, params)
                except MCPError:
                    raise
                except Exception as exc:
                    if isinstance(exc, TimeoutError) and limit.expired():
                        raise _extension_method_stopped(method) from None
                    log_unexpected_error(logger, f"combined mcp: {method}")
                    raise MCPError(
                        code=INTERNAL_ERROR,
                        message="internal_error: " + internal_error_message(subject=method),
                    ) from None

    return guarded


def _extension_method_stopped(method: str) -> MCPError:
    logger.warning("combined mcp: %s stopped at the %gs limit", method, _SKILLS_READ_LIMIT_S)
    return MCPError(
        code=INTERNAL_ERROR,
        message=(
            f"service_unavailable: {method} did not finish within the "
            f"{_SKILLS_READ_LIMIT_S:g}s this server allows it, so it was stopped. "
            f"Repeat it after {DEADLINE_RETRY_AFTER_S:g}s."
        ),
    )


class _ToolCallRateLimit(Middleware):
    """Draws each tool call from one token bucket, refilled at `calls_per_s` and
    holding at most `burst`, and refuses a call over it `rate_limited`.

    The bucket is shared by every caller on this instance. Only `tools/call`
    draws on it. Initialize, `tools/list`, resource reads and pings pass
    untouched, since a host sends several of them around each tool call, and
    a session DELETE never reaches middleware.
    """

    def __init__(self, *, calls_per_s: float, burst: int) -> None:
        self._bucket = TokenBucketRateLimiter(capacity=burst, refill_rate=calls_per_s)

    @override
    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        if not await self._bucket.consume():
            retry_after_s = max(1.0, 1 / self._bucket.refill_rate)
            return _refusal(
                code="rate_limited",
                retry_mode="retry_after_delay",
                message=(
                    "This server is receiving more calls than it accepts across all of "
                    f"its callers. Repeat the call unchanged after {retry_after_s:.1f}s."
                ),
                fix={},
                retry_after_s=retry_after_s,
            )
        return await call_next(context)


class _CallLimit(Middleware):
    """Cancels a tool call or resource read still running after `limit_s` seconds,
    and a resource listing after `_SKILLS_READ_LIMIT_S`.

    A tool call over the limit refuses `service_unavailable` with
    `retry_after_delay`, naming the tool and the limit, as data-mcp refuses
    work its own deadline left no time for. A resource read over it fails with
    a `ResourceError` marked `service_unavailable`, and a listing with `-32603`
    marked the same way.

    A listing's one slow part is data-mcp's read of the skill catalogue, the
    same Skills API read an extension method makes, so it shares that
    method's limit and stays well inside the upstream client's 120 s retry budget.

    Cancelling unwinds the call. The upstream client's connection pool closes
    the connection a cancelled request held, and a `get_data` mint POST
    cancelled while it waits keeps its mint-ledger slot until apis has
    stopped running the mint (`data_mcp.mint_ledger`). Work a tool handed to
    a thread runs to its end, and its result is dropped.
    """

    def __init__(self, *, limit_s: float) -> None:
        self._limit_s = limit_s

    @override
    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        result = await self._within_limit(call_next(context))
        if not isinstance(result, _Stopped):
            return result
        name = context.message.name
        logger.warning("combined mcp: tool %s stopped at the %gs call limit", name, self._limit_s)
        return _refusal(
            code="service_unavailable",
            retry_mode="retry_after_delay",
            message=(
                f"{name} did not finish within the {self._limit_s:g}s this server allows "
                f"a call, so the call was stopped. Repeat it unchanged after "
                f"{DEADLINE_RETRY_AFTER_S:g}s."
            ),
            fix={"tool": name, "limit_s": self._limit_s},
            retry_after_s=DEADLINE_RETRY_AFTER_S,
        )

    @override
    async def on_read_resource(
        self,
        context: MiddlewareContext[mt.ReadResourceRequestParams],
        call_next: CallNext[mt.ReadResourceRequestParams, ResourceResult],
    ) -> ResourceResult:
        result = await self._within_limit(call_next(context))
        if not isinstance(result, _Stopped):
            return result
        logger.warning("combined mcp: resource read stopped at the %gs call limit", self._limit_s)
        raise ResourceError(
            f"service_unavailable: reading this resource did not finish within the "
            f"{self._limit_s:g}s this server allows a request, so it was stopped. Read it "
            f"again after {DEADLINE_RETRY_AFTER_S:g}s."
        )

    @override
    async def on_list_resources(
        self,
        context: MiddlewareContext[mt.ListResourcesRequest],
        call_next: CallNext[mt.ListResourcesRequest, Sequence[Resource]],
    ) -> Sequence[Resource]:
        result = await self._within_limit(call_next(context), limit_s=_SKILLS_READ_LIMIT_S)
        if not isinstance(result, _Stopped):
            return result
        logger.warning(
            "combined mcp: resources/list stopped at the %gs limit", _SKILLS_READ_LIMIT_S
        )
        raise MCPError(
            code=INTERNAL_ERROR,
            message=f"service_unavailable: listing resources did not finish within the "
            f"{_SKILLS_READ_LIMIT_S:g}s this server allows it, so it was stopped. List them "
            f"again after {DEADLINE_RETRY_AFTER_S:g}s.",
        )

    async def _within_limit[T](
        self, call: Awaitable[T], *, limit_s: float | None = None
    ) -> T | _Stopped:
        """`call`'s result, or `_Stopped` when the limit cancelled it. `limit_s`
        defaults to this middleware's call limit.

        A `TimeoutError` the call raised itself, rather than the limit's, is
        re-raised.
        """
        timeout = asyncio.timeout(self._limit_s if limit_s is None else limit_s)
        try:
            async with timeout:
                return await call
        except TimeoutError:
            if not timeout.expired():
                raise
        return _Stopped()


class _Stopped:
    """What `_CallLimit._within_limit` returns for a call the limit cancelled."""


class _ShapeResults(Middleware):
    """Drops `structuredContent` and `search`'s envelope from a successful
    result, and `outputSchema` from `tools/list`.

    The digest in `content` is the surface built for a model. A TypeScript-SDK
    client rejects a result without structured content from a tool that
    declares an output schema, so `tools/list` drops every tool's schema. The
    parent registers no tool of its own, so every tool it serves is a child's.

    A tool whose `tools/list` entry declares `_meta.ui.resourceUri` is an MCP
    App tool: its result carries the view's layout in `structuredContent` and
    the values the view binds in `_meta`, which the host's renderer reads. Its
    results pass whole, found from the tool's declaration at call time, so a
    new app tool needs no change here. Refusals pass whole from every tool.
    `search` puts the whole upstream envelope in its result's `_meta` under
    `SEARCH_ENVELOPE_META_KEY` for the harness's own readers, which call the
    data-mcp mount directly. An outside client has no reader for it, and it
    runs to tens of kilobytes, so the rest of `_meta` passes and that key
    does not.
    """

    def __init__(self, *, server: FastMCP) -> None:
        self._server = server

    @override
    async def on_list_tools(
        self,
        context: MiddlewareContext[mt.ListToolsRequest],
        call_next: CallNext[mt.ListToolsRequest, Sequence[Tool]],
    ) -> Sequence[Tool]:
        return [
            tool.model_copy(update={"output_schema": None}) for tool in await call_next(context)
        ]

    @override
    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        result = await call_next(context)
        if result.is_error or _declares_app_view(await self._server.get_tool(context.message.name)):
            return result
        meta = {
            key: value
            for key, value in (result.meta or {}).items()
            if key != SEARCH_ENVELOPE_META_KEY
        }
        return ToolResult(content=result.content, meta=meta or None)


class _PublicArguments(Middleware):
    """Lists each tool as `public_tool` declares it, and writes a tool's
    `PublicShape.forced` arguments over every call's.

    A call that sends an argument the public form hides and does not force raises
    the `ValidationError` an unlisted name raises, so `_RefuseRaisedErrors`
    refuses it as `request_invalid` before the child validates or runs it. A
    forced argument the caller sends is accepted and overwritten, since clients
    send `get_data`'s `place_types`."""

    @override
    async def on_list_tools(
        self,
        context: MiddlewareContext[mt.ListToolsRequest],
        call_next: CallNext[mt.ListToolsRequest, Sequence[Tool]],
    ) -> Sequence[Tool]:
        return [public_tool(tool) for tool in await call_next(context)]

    @override
    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        name = context.message.name
        shape = _PUBLIC_SHAPES.get(name)
        if shape is None:
            return await call_next(context)
        sent = context.message.arguments or {}
        if hidden := [
            argument
            for argument in sent
            if argument in shape.hidden and argument not in shape.forced
        ]:
            raise ValidationError(f"{name} takes no {', '.join(hidden)}") from (
                PydanticValidationError.from_exception_data(
                    name,
                    [
                        {
                            "type": "unexpected_keyword_argument",
                            "loc": (argument,),
                            "input": sent[argument],
                        }
                        for argument in hidden
                    ],
                )
            )
        if not shape.forced:
            return await call_next(context)
        arguments = {**sent, **shape.forced}
        message = context.message.model_copy(update={"arguments": arguments})
        return await call_next(context.copy(message=message))


def public_tool(tool: Tool) -> Tool:
    """`tool` as the combined server lists it, under its child's `PublicShape`:
    the shape's paragraphs after the mount's description, and every property the
    shape hides left out wherever the schema holds one. A tool with no shape is
    listed as its server declares it."""
    shape = _PUBLIC_SHAPES.get(tool.name)
    if shape is None:
        return tool
    return tool.model_copy(
        update={
            "description": "\n\n".join((tool.description or "", *shape.paragraphs)),
            "parameters": _without_properties(tool.parameters, shape.hidden),
        }
    )


def _without_properties(node: Any, hidden: frozenset[str]) -> Any:
    if isinstance(node, list):
        return [_without_properties(item, hidden) for item in node]
    if not isinstance(node, dict):
        return node
    kept: dict[str, Any] = {}
    for key, value in node.items():
        if key == "properties" and isinstance(value, dict):
            kept[key] = {
                name: _without_properties(prop, hidden)
                for name, prop in value.items()
                if name not in hidden
            }
        elif key == "required" and isinstance(value, list):
            kept[key] = [name for name in value if name not in hidden]
        else:
            kept[key] = _without_properties(value, hidden)
    return kept


def _declares_app_view(tool: Tool | None) -> bool:
    """Whether `tool` declares an MCP App view, `_meta.ui.resourceUri`, in `tools/list`."""
    ui = (tool.meta or {}).get("ui") if tool is not None else None
    return isinstance(ui, dict) and bool(ui.get("resourceUri"))


class _RefuseRaisedErrors(Middleware):
    """Answers a tool call that raised with a refusal in the envelope a model reads.

    fastmcp reports a raised call to the client as bare text with no code or
    retry mode. This middleware sits inside every other but `_PublicArguments`,
    so it sees what the call raised and what `_PublicArguments` raised for it:

    - `NotFoundError`, a name no tool carries: `tool_not_found`, naming the
      tools `server` serves. The parent resolves the name before any child
      sees the call. A name in `_MERGED_NAMES` is answered with the listed
      call that does its work instead.
    - `ValidationError`, arguments that fail the tool's input schema:
      `request_invalid`, whose `fix` holds `problems`, one `field: problem`
      line per error pydantic found, and `accepted`, the arguments the tool's
      listing (`public_tool`) declares.
    - Any other exception, which fastmcp has already wrapped in a `ToolError`
      for an exception the tool raised: `internal_error`, which repeating the
      call will not clear. Its text can hold a bucket path or a service
      account, so the exception is logged under a trace id and the refusal
      carries fixed text.

    A resource read that raised gets the same split. fastmcp raises an error
    written for the caller (an unknown URI, and every failure of data-mcp's
    artifact resource, which logs its own) as is, and wraps any other
    exception in a `ResourceError` raised from it. An error with a `__cause__`
    is therefore one fastmcp wrapped, and fails with fixed `internal_error`
    text.
    Log lines leave out the URI, since a `dfo://artifact` URI is a bearer
    credential.

    A call to a per-server mount never passes through here and keeps
    fastmcp's own reporting.
    """

    def __init__(self, *, server: FastMCP) -> None:
        self._server = server

    @override
    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        name = context.message.name
        try:
            return await call_next(context)
        except NotFoundError:
            if name in _MERGED_NAMES:
                return _merged_name_refusal(name)
            tools = sorted(
                tool.name for tool in await self._server.list_tools(run_middleware=False)
            )
            return _refusal(
                code="tool_not_found",
                retry_mode="correct_and_retry",
                message=(
                    f"No tool is named {truncate_value(name)!r}. Call one of: {', '.join(tools)}."
                ),
                fix={"tools": tools},
            )
        except ValidationError as exc:
            tool = await self._server.get_tool(name)
            parameters = public_tool(tool).parameters if tool is not None else {}
            problems = _problems(exc, parameters)
            return _refusal(
                code="request_invalid",
                retry_mode="correct_and_retry",
                message=(
                    f"The arguments to {name} do not match its input schema: "
                    f"{'; '.join(problems)}. Correct them and call again."
                ),
                fix={
                    "tool": name,
                    "problems": problems,
                    "accepted": list(parameters.get("properties", {})),
                },
            )
        except Exception:
            log_unexpected_error(logger, f"combined mcp: tool {name}")
            return _refusal(
                code="internal_error",
                retry_mode="do_not_retry",
                message=internal_error_message(subject=name),
                fix={"tool": name},
            )

    @override
    async def on_read_resource(
        self,
        context: MiddlewareContext[mt.ReadResourceRequestParams],
        call_next: CallNext[mt.ReadResourceRequestParams, ResourceResult],
    ) -> ResourceResult:
        try:
            return await call_next(context)
        except Exception as exc:
            if isinstance(exc, FastMCPError) and exc.__cause__ is None:
                raise
            log_unexpected_error(logger, "combined mcp: resource read")
            raise ResourceError(
                "internal_error: " + internal_error_message(subject="Reading this resource")
            ) from None


def _merged_name_refusal(name: str) -> ToolResult:
    """`tool_not_found` for `name`, naming the listed call that does its work."""
    tool, operation = _MERGED_NAMES[name]
    if operation is None:
        message = (
            f"No tool is named {name!r}. Call {tool} with the same skill_id, path, offset "
            "and digest."
        )
        # `get_skill` reads one file of a skill when `path` is set.
        fix: dict[str, Any] = {"tool": tool, "argument": "path"}
    else:
        message = (
            f'No tool is named {name!r}. Call {tool} with operation="{operation}" and the '
            "same arguments."
        )
        fix = {"tool": tool, "operation": operation}
    return _refusal(code="tool_not_found", retry_mode="correct_and_retry", message=message, fix=fix)


def _problems(exc: ValidationError, parameters: dict[str, Any]) -> list[str]:
    """One `field: problem` line per error in the pydantic `ValidationError` fastmcp
    raised `exc` from. A value outside a top-level enum in `parameters` names the
    values the enum allows. What the caller sent, an argument's name included, is
    echoed capped (`echoed_text.truncate_value`)."""
    cause = exc.__cause__
    if not isinstance(cause, PydanticValidationError):
        return [str(exc)]
    properties = parameters.get("properties", {})
    problems = []
    for error in cause.errors(include_url=False):
        field = truncate_value(".".join(str(part) for part in error["loc"]))
        allowed = properties.get(field, {}).get("enum")
        if error["type"] == "literal_error" and allowed:
            values = ", ".join(str(value) for value in allowed)
            sent = truncate_value(repr(error["input"]))
            problems.append(f"{field}: must be one of {values}, not {sent}")
        else:
            problems.append(f"{field}: {error['msg']}")
    return problems


def combined_http_app(combined: FastMCP) -> StarletteWithLifespan:
    """The ASGI app serving `combined` at the root of whatever prefix mounts it.

    `path="/"` puts the MCP endpoint at the mount prefix itself, `/mcp/`;
    FastMCP's default path would stack a second `/mcp` under it. A bare
    `/mcp` gets Starlette's 307 to `/mcp/`, so clients use the trailing
    slash. `stateless_http=True` serves a 2025-revision client without an
    `Mcp-Session-Id`, as the 2026-07-28 revision always is, so any instance
    serves any request. `json_response=False` keeps SSE responses.
    """
    return combined.http_app(
        transport="streamable-http", path="/", stateless_http=True, json_response=False
    )


def _installed_stores() -> dict[str, object]:
    """The process-wide stores the children's lifespans install at startup.

    data-mcp's lifespan sets the definition store on every entry, and
    analysis-mcp's (or data-mcp's, with a frame-store bucket) installs the
    frame store. data-mcp's upstream client holder is written only at
    shutdown, so a second entry could not change it at startup.
    """
    return {"definition store": get_definition_store(), "frame store": get_frame_store()}


def children_start_once[AppT](lifespan: Lifespan[AppT]) -> StatelessLifespan[AppT]:
    """Wrap the combined app's `lifespan` to refuse startup if it re-ran a child's lifespan.

    Enter the wrapped lifespan after every per-server mount's own, so the
    stores are installed before it starts. Each store must be the same
    object after the combined server starts as before.
    """

    @asynccontextmanager
    async def checked(app: AppT) -> AsyncIterator[None]:
        before = _installed_stores()
        async with lifespan(app):
            replaced = [
                name for name, store in _installed_stores().items() if store is not before[name]
            ]
            if replaced:
                raise RuntimeError(
                    f"the combined MCP server replaced the {' and '.join(replaced)} when it "
                    "started: it ran a child's lifespan a second time. Mount the same FastMCP "
                    "objects the per-server mounts serve."
                )
            yield

    return checked
