"""The tool surface: three pass-through tools over the assembler.

The server publishes the tools, forwards each call with the
caller's type attached, and returns the reply unchanged, with one exception.
A write_component or describe_component result also carries the platform's
resolved revision in its `_meta` under `RESOLVED_PAYLOAD_META_KEY`. Its
`content` leads with the text projection, or a one-line delta for a revise
that changed only presentation fields. A client that declared the MCP Apps
extension, or whose capabilities the server cannot see (`_view_support`),
gets the projection without its data table, and a Prefab shell as
`structuredContent` in place of the model vocabulary: the component's layout
and labels without its data. The data rides in `_meta` under
`VIEW_STATE_META_KEY`, which the renderer page merges into the shell's state,
so the view needs no further call. A client that declared nothing about
the extension (`ViewSupport.UNDECLARED`) cannot draw the view, and gets a
line or bar chart as a PNG `ImageContent` block (`_chart_image`) after the
payload and before the write token, unless `include_projection` is `none`
or the chart's value axis is logarithmic.
The component id the platform mints lets any call describe the component.
Revising it also needs the write token the create returned. A create's
result carries the token on the last line of its `content`, and at the top
level of `structuredContent` when that is the model vocabulary, and never in
`_meta` or a Prefab shell. The server never asks who is calling.
The assembler validates intents and answers with typed refusals.
The server owns two behaviours of its own: the safety net, which turns an
unexpected exception into an `internal_error` refusal naming the trace id
it was logged under instead of a raw error, and the startup wiring, which
fetches the platform's release, admits it, and mounts the tools before the
server answers a request. A platform that cannot be reached is retried
until a deadline, and startup fails past it, or at once when the release
answers and cannot be admitted. Local development can opt out and come up
tool-less instead.

Refusals leave as MCP error results (`isError: true`). `content` carries the
`error`/`retry`/`detail` teaching payload, matching data-mcp and
analysis-mcp. `describe_frame` carries the same payload in
`structuredContent`. write_component and describe_component send it there
too, except to a client `_view_support` answers as an MCP Apps host, which
gets a one-line Prefab notice view in `structuredContent` instead. Every
refusal also carries its `error` code in `_meta`
(`common.refusal.refusal_meta`), where the harness's combined server reads it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict
from enum import StrEnum
from typing import Any, Literal

from common.auth import BearerTokenVerifier, apply_gate
from common.logging import configure_logging
from common.materialisation import install_store_if_unset
from common.mcp_telemetry import (
    WRITE_TOKEN_KEY,
    LangfuseSessionMiddleware,
    LangfuseToolIoMiddleware,
)
from common.refusal import internal_error_message, log_unexpected_error, refusal_meta
from common.telemetry import init_telemetry
from fastmcp import Context, FastMCP
from fastmcp.apps import (
    UI_EXTENSION_ID,
    UI_MIME_TYPE,
    AppConfig,
    ResourceCSP,
    app_config_to_meta_dict,
)
from fastmcp.resources import ResourceContent, ResourceResult
from fastmcp.tools import Tool, ToolResult
from fastmcp.utilities.types import Image
from mcp.types import ClientCapabilities, ImageContent, TextContent
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from presentation_mcp.app_view import app_shell, notice_view, plots_as_chart, view_state
from presentation_mcp.assembler_holder import get_assembler, set_assembler
from presentation_mcp.chart_worker import ChartWorker
from presentation_mcp.client import PlatformAssembler
from presentation_mcp.platform_contract import (
    RESOLVED_PAYLOAD_META_KEY,
    VIEW_STATE_META_KEY,
    Caller,
    Refusal,
    RetryMode,
    WriteReply,
)
from presentation_mcp.projection import (
    ROWLESS_NOTICE,
    UNKNOWN_HOST_ROWLESS_NOTICE,
    delta_projection,
    text_projection,
)
from presentation_mcp.settings import Settings
from presentation_mcp.theme import THEME_CSS, THEME_HASH, renderer_csp, renderer_html, renderer_uri
from presentation_mcp.tool_schema import (
    describe_component_schema,
    describe_frame_schema,
    write_component_schema,
)

logger = logging.getLogger(__name__)

settings = Settings()

# HTTP is the only transport this server has, and HTTP callers are always
# external. A first-party path would need a second transport, not a flag here.
_CALLER = Caller.EXTERNAL

# The `ui://` resource serving Prefab's CDN renderer page, a stub that loads
# renderer.js and renderer.css from jsDelivr at the installed prefab-ui version,
# with the script that merges a result's view state into the view. Its URI
# carries a hash of the Prefab version, the CSP, and the page, so a host that
# caches resources by URI fetches a changed page or CSP. The CSP admits
# jsDelivr, the font hosts, and the origin `settings.presentation_public_url`
# names, where the view's iframe loads the theme stylesheet from.
# write_component and describe_component open it.
_RENDERER_URI = renderer_uri(settings.presentation_public_url)
_APP_META = {"ui": app_config_to_meta_dict(AppConfig(resource_uri=_RENDERER_URI))}

# Per member type, the intent fields in the catalogue's intent schemas that
# change how a component looks and none of the facts it shows: `title` on
# every member that has one, and `scale` and `range` on the value axis of
# line_chart and bar_chart. Under `include_projection="auto"`, a revise that
# changed only these returns a one-line delta. A revise that changed any
# other field returns the full text projection, whether the field is
# `select`, `window`, a stat_card's `change`, a note_narration's `text`, or
# one a later catalogue adds, and so does any revise of a member type this
# map does not name.
_PRESENTATION_ONLY_FIELDS: dict[str, frozenset[str]] = {
    "line_chart": frozenset({"title", "scale", "range"}),
    "bar_chart": frozenset({"title", "scale", "range"}),
    "table": frozenset({"title"}),
    "stat_card": frozenset({"title"}),
    "note_narration": frozenset(),
    "source_card": frozenset(),
}

type IncludeProjection = Literal["auto", "full", "none"]

# Draws the chart images, in a worker process the lifespan starts and stops.
_chart_worker = ChartWorker(
    timeout_s=settings.chart_render_timeout_s, warm=settings.chart_worker_warm
)

# What `/healthz` reports, mutated in place by the lifespan so the route
# closure always reads the current state.
_health: dict[str, Any] = {"status": "starting"}


def _reply(
    reply: Mapping[str, Any] | WriteReply | Refusal,
    *,
    view: dict[str, Any] | None = None,
) -> ToolResult:
    """A reply as a `ToolResult` whose `content` is the payload as JSON.

    `structuredContent` is `view` when given, and otherwise the identical
    payload. `None` fields are dropped from dataclass replies, so absence
    stays absent on the wire. A `Refusal` sets `is_error`, and its `_meta`
    is `refusal_meta` of its `error`. `WriteReply` and a mapping reply
    (`describe_frame`, `describe_component`) both leave `is_error` false
    and carry no `_meta`.
    """
    payload = _payload(reply)
    return ToolResult(
        content=payload,
        structured_content=payload if view is None else view,
        meta=refusal_meta(code=reply.error) if isinstance(reply, Refusal) else None,
        is_error=isinstance(reply, Refusal),
    )


class ViewSupport(StrEnum):
    """What the calling client declared about drawing MCP Apps views."""

    DECLARED = "declared"
    UNDECLARED = "undeclared"
    UNKNOWN = "unknown"


# The notice standing in for the data table, per client that gets the
# projection without it.
_ROWLESS_NOTICES = {
    ViewSupport.DECLARED: ROWLESS_NOTICE,
    ViewSupport.UNKNOWN: UNKNOWN_HOST_ROWLESS_NOTICE,
}


def _view_support(ctx: Context) -> ViewSupport:
    """What the calling client declared about the MCP Apps extension.

    A 2025-11-25 client declares its capabilities on `initialize`, and a
    2026-07-28 client declares them in every request's `_meta`, with or
    without its client info. The combined server at `/mcp/` is stateless, so
    a 2025-11-25 request there carries no initialize params, and its
    capabilities are `UNKNOWN`. A call with no session at all is answered as
    `UNDECLARED`.
    """
    try:
        session = ctx.session
    except RuntimeError:
        return ViewSupport.UNDECLARED
    return view_support(session.client_capabilities)


def view_support(capabilities: ClientCapabilities | None) -> ViewSupport:
    """What `capabilities` declare about the MCP Apps extension, or `UNKNOWN`
    when the server holds none for the client."""
    if capabilities is None:
        return ViewSupport.UNKNOWN
    if UI_EXTENSION_ID in (capabilities.extensions or {}):
        return ViewSupport.DECLARED
    return ViewSupport.UNDECLARED


def _write_token_text(write_token: str) -> str:
    """The content block carrying a create's write token on its last line."""
    return (
        "Keep this write_token with the component_id. Revising the component needs "
        "both, and the component_id alone lets anyone read it. Without the token, "
        "create a new component.\n"
        f"{WRITE_TOKEN_KEY}: {write_token}"
    )


def _write_result(
    payload: dict[str, Any],
    *,
    write_token: str | None,
    lead: str | None = None,
    image: ImageContent | None = None,
    view: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> ToolResult:
    """A successful write_component or describe_component result.

    `content` is `lead` when given, then `payload` as JSON, then, on a
    create, `_write_token_text`, so the token is the last line of the text
    every client reads. `image`, when given, sits between the payload and the
    token block. `structuredContent` is `view` when given, the Prefab
    shell or a notice view for an Apps client, and otherwise `payload` with
    the write token at its top level. Neither the view nor `meta` ever
    carries the token.
    """
    vocabulary = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    content = [TextContent(type="text", text=vocabulary)]
    if lead is not None:
        content.insert(0, TextContent(type="text", text=lead))
    structured = payload
    if image is not None:
        content.append(image)
    if write_token is not None:
        content.append(TextContent(type="text", text=_write_token_text(write_token)))
        structured = {**payload, WRITE_TOKEN_KEY: write_token}
    return ToolResult(
        content=content,
        structured_content=structured if view is None else view,
        meta=meta,
    )


async def _component_reply(
    reply: Mapping[str, Any] | WriteReply | Refusal,
    *,
    views: ViewSupport,
    include_projection: IncludeProjection,
    revise: bool = False,
) -> ToolResult:
    """A write_component or describe_component reply as a `ToolResult`.

    A successful reply's `resolved_payload` leaves the payload and rides in
    `_meta` under `RESOLVED_PAYLOAD_META_KEY`. `content` is the projection
    `_projection` picks, when it picks one, then the payload, the model
    vocabulary, as JSON, then a create's write token (`_write_result`).
    `structuredContent` is the envelope's Prefab shell for an Apps client,
    one whose `views` is `DECLARED` or `UNKNOWN`, and the payload otherwise.
    The shell holds no data value, so for an Apps client `_meta` also
    carries `view_state` of the envelope under `VIEW_STATE_META_KEY`, the
    values the shell binds. Only a `WriteReply` carries a write token. A
    client whose `views` is `UNDECLARED` gets `_chart_image` in `content`.

    The write has already committed when this runs, and refusing it would
    invite a duplicate, so a reply without an envelope, or one whose
    envelope fails to render, still succeeds with the model vocabulary
    alone and logs why. For those replies and for a refusal, an Apps
    client's renderer draws a `notice_view` saying why there is nothing to
    draw, and `content` is unchanged.
    """
    app = views is not ViewSupport.UNDECLARED
    if isinstance(reply, Refusal):
        return _reply(reply, view=_notice(app, f"The request was refused: {reply.detail}"))
    payload = _payload(reply)
    write_token = payload.pop(WRITE_TOKEN_KEY, None) if isinstance(reply, WriteReply) else None
    envelope = payload.pop("resolved_payload", None)
    component_id = payload.get("component_id")
    if envelope is None:
        logger.warning("platform reply for %s carried no resolved_payload", component_id)
        return _write_result(
            payload,
            write_token=write_token,
            view=_notice(app, "The platform returned no revision of this component to draw."),
        )
    meta = {RESOLVED_PAYLOAD_META_KEY: envelope}
    try:
        lead = _projection(
            envelope,
            payload.get("changed", ()),
            include_projection=include_projection,
            revise=revise,
            views=views,
        )
        if not app:
            image = await _chart_image(
                envelope, include_projection=include_projection, component_id=component_id
            )
            return _write_result(
                payload, write_token=write_token, lead=lead, image=image, meta=meta
            )
        return _write_result(
            payload,
            write_token=write_token,
            lead=lead,
            view=app_shell(envelope, base_url=settings.presentation_public_url),
            meta={**meta, VIEW_STATE_META_KEY: view_state(envelope)},
        )
    except Exception:
        trace_id = log_unexpected_error(
            logger, f"presentation-mcp rendering the resolved revision of {component_id}"
        )
        return _write_result(
            payload,
            write_token=write_token,
            view=_notice(
                app, f"This revision of the component could not be drawn (trace id {trace_id})."
            ),
            meta=meta,
        )


async def _chart_image(
    envelope: Mapping[str, Any],
    *,
    include_projection: IncludeProjection,
    component_id: object,
) -> ImageContent | None:
    """The envelope's chart as a PNG content block, or `None` for none.

    The caller has already limited this to a client that cannot draw the
    view. `include_projection="none"` asks for no lead, and the image counts
    as one. Only an envelope that `plots_as_chart` goes to `_chart_worker`,
    so a table, a stat card, or a log-axis chart skips the worker's queue
    and its envelope stays in the server process. The render runs in the
    worker's process, off the event loop, and a chart past `chart_image`'s
    size limits, a render past `settings.chart_render_timeout_s`, which
    kills the worker, or a worker that died leaves the result without the
    image. The write has committed, so a render that raises is logged and
    the result goes out without the image too.
    """
    if include_projection == "none":
        return None
    try:
        if not plots_as_chart(envelope):
            return None
        png = await _chart_worker.render(envelope, component_id=component_id)
    except Exception:
        log_unexpected_error(logger, f"presentation-mcp drawing the chart of {component_id}")
        return None
    if png is None:
        return None
    return Image(data=png, format="png").to_image_content()


def _projection(
    envelope: Mapping[str, Any],
    changed: Sequence[Mapping[str, Any]],
    *,
    include_projection: IncludeProjection,
    revise: bool,
    views: ViewSupport,
) -> str | None:
    """The text leading `content`, or `None` for none.

    `auto` returns the one-line delta on a revise whose `changed` fields
    all fall in the component type's `_PRESENTATION_ONLY_FIELDS` entry, and
    the projection on a create, on a describe, on every other revise, and on
    any revise of a type the map does not name. That projection leaves out
    the data table for a client that declared it draws views, and for one
    whose capabilities are unknown, each with its own notice in the table's
    place (`_ROWLESS_NOTICES`). `full` forces the projection with its table
    for any client, and `none` suppresses it.
    """
    if include_projection == "none":
        return None
    if include_projection == "full":
        return text_projection(envelope)
    touched = {entry.get("field") for entry in changed}
    presentation_only = _PRESENTATION_ONLY_FIELDS.get(envelope["component"]["type"])
    if revise and presentation_only is not None and touched <= presentation_only:
        return delta_projection(envelope, changed)
    return text_projection(envelope, rowless_notice=_ROWLESS_NOTICES.get(views))


def _notice(app: bool, text: str) -> dict[str, Any] | None:
    """`notice_view(text)` for an Apps client, else `None`."""
    return notice_view(text, base_url=settings.presentation_public_url) if app else None


def _payload(reply: Mapping[str, Any] | WriteReply | Refusal) -> dict[str, Any]:
    if isinstance(reply, WriteReply | Refusal):
        return {k: v for k, v in asdict(reply).items() if v is not None}
    return dict(reply)


async def _safely(call: Awaitable[Any]) -> Any:
    """The safety net: an unexpected exception leaves as an `internal_error`
    refusal that names the trace id it was logged under and none of its text."""
    try:
        return await call
    except Exception:
        trace_id = log_unexpected_error(logger, "presentation-mcp assembler call")
        return Refusal(
            error="internal_error",
            retry=RetryMode.DO_NOT_RETRY,
            detail=internal_error_message(subject="This call", trace_id=trace_id),
        )


def _missing_write_token(component_id: str) -> Refusal:
    """The local refusal for a revise that names no write token."""
    return Refusal(
        error="request_invalid",
        retry=RetryMode.CORRECT_AND_RETRY,
        detail=f"Revising {component_id} needs the write_token its create returned.",
        fix=(
            "Pass that write_token with component_id. Without it, create a new "
            "component: call write_component without component_id. "
            "describe_component on this id returns its intent to start from."
        ),
        pointer="/write_token",
    )


async def write_component(
    intent: dict[str, Any],
    ctx: Context,
    component_id: str | None = None,
    write_token: str | None = None,
    include_projection: IncludeProjection = "auto",
) -> ToolResult:
    """Write a component: one of the platform's six typed members, line_chart, stat_card, bar_chart, table, note_narration, or source_card.

    Omitting component_id creates a new component from a complete intent.
    Passing component_id and write_token, which the component's create
    returned, edits that component from intent as a patch holding only the
    changed fields.

    The reply's fact_summary.change populates only for a stat_card whose
    intent authors a change object with a basis. line_chart, bar_chart,
    table, note_narration, and source_card carry no change property,
    whatever the intent sends.

    The component_id a reply returns is the only way back to that component.
    A create's reply also returns a write_token. Keep both: a revise needs
    the token, and the component_id alone lets anyone read the component.
    Without the token, create a new component; describe_component returns
    the old one's intent to start from.

    intent's frame_ref is the spec_id and content_version pair a data tool's
    reply returned. A frame_ref whose spec_id starts with ops_ and whose
    content_version starts with ocv_ names an Analysis Toolkit result, and
    is presentable like any other frame_ref.

    A create's reply leads with a Markdown table of the component's data,
    and so does an edit. An edit whose patch changes only title, scale, or
    range leads with one line naming what changed instead. A client that
    draws the component as an interactive view gets that lead without the
    table, and so does a client whose capabilities the server cannot see.
    include_projection overrides this: full always sends the table, none
    leaves it out.
    """
    assembler = get_assembler()
    if component_id is None:
        reply = await _safely(assembler.create_component(intent=intent, caller=_CALLER))
    elif write_token is None:
        reply = _missing_write_token(component_id)
    else:
        reply = await _safely(
            assembler.revise_component(
                component_id=component_id, patch=intent, write_token=write_token, caller=_CALLER
            )
        )
    return await _component_reply(
        reply,
        views=_view_support(ctx),
        include_projection=include_projection,
        revise=component_id is not None,
    )


async def describe_frame(
    frame_ref: dict[str, Any],
    dimension: str | None = None,
    query: str | None = None,
) -> ToolResult:
    """Describe a frame: its dimensions, their values, and the periods it covers.

    frame_ref is the spec_id and content_version pair a data tool's reply
    returned. Any caller holding a frame_ref can use it, whichever caller
    minted it.
    """
    reply = await _safely(
        get_assembler().get_frame_view(frame_ref=frame_ref, dimension=dimension, query=query)
    )
    return _reply(reply)


async def describe_component(
    component_id: str,
    ctx: Context,
    revision: int | None = None,
    include_projection: IncludeProjection = "auto",
) -> ToolResult:
    """Describe a stored component: its intent and fact_summary, at the current revision or an older one.

    revision is the count the platform increments on every write, starting
    at 1 on create. Omit it to read the current revision.

    component_id is the id a previous write_component reply returned.

    The reply leads with a Markdown table of the component's data. A client
    that draws the component as an interactive view gets that lead without
    the table, and so does a client whose capabilities the server cannot
    see. include_projection overrides this: full always sends the table,
    none leaves it out.
    """
    reply = await _safely(
        get_assembler().get_component(component_id=component_id, revision=revision)
    )
    return await _component_reply(
        reply,
        views=_view_support(ctx),
        include_projection=include_projection,
    )


def _register_tools(
    *,
    write_schema: dict[str, Any],
    frame_schema: dict[str, Any],
    describe_schema: dict[str, Any],
) -> tuple[str, ...]:
    """Publish the three tools from already-assembled schemas.

    FastMCP keeps parsing incoming arguments against the function signatures,
    which stay permissive on purpose; only what is advertised changes.

    write_component and describe_component are Prefab app tools. Their
    listing carries `_meta.ui.resourceUri`, `_RENDERER_URI`.

    Schema assembly happens in the caller, before any tool is added here, so
    a `ValueError` from a divergent release document never reaches this
    function. What this function can still fail partway through is
    registration itself; on any such failure it unregisters whatever of the
    three it had already added before re-raising, so a partial failure never
    leaves `mcp` with, say, `write_component` mounted but not the others.

    Returns:
        The registered tools' names, so the lifespan can unregister exactly
        these on shutdown.
    """
    registered: list[str] = []
    try:
        write_tool = Tool.from_function(write_component, meta=_APP_META)
        write_tool.parameters = write_schema
        mcp.add_tool(write_tool)
        registered.append(write_tool.name)

        frame_tool = Tool.from_function(describe_frame)
        frame_tool.parameters = frame_schema
        mcp.add_tool(frame_tool)
        registered.append(frame_tool.name)

        describe_tool = Tool.from_function(describe_component, meta=_APP_META)
        describe_tool.parameters = describe_schema
        mcp.add_tool(describe_tool)
        registered.append(describe_tool.name)
    except Exception:
        for name in registered:
            mcp.local_provider.remove_tool(name)
        raise
    return tuple(registered)


class _ReleaseNotAdmittedError(Exception):
    """The platform answered, and its release cannot be mounted. Retrying
    does not help, unlike a failure to reach the platform."""


def _set_health(status: str, reason: str | None = None) -> None:
    _health.clear()
    _health.update(status=status)
    if reason is not None:
        _health.update(reason=reason)


async def _mount(assembler: PlatformAssembler) -> tuple[str, ...]:
    """Fetch the platform's release, admit it, register the tools, and wire
    `assembler` for them.

    Raises whatever `release()` raises when the platform cannot be reached,
    and `_ReleaseNotAdmittedError` when it answers with a release this
    deployment cannot mount: a `schema_version` that is malformed or outruns
    `settings.max_schema_version`, member schemas whose `$defs` diverge
    (`tool_schema.py`'s `ValueError`), or a registration failure. On
    `_ReleaseNotAdmittedError` no tool stays registered and the holder is
    clear.

    Returns:
        The registered tools' names, for the lifespan to unregister on
        shutdown.
    """
    release = await assembler.release()
    try:
        schema_version = release["schema_version"]
        version_too_new = schema_version > settings.max_schema_version
    except (KeyError, TypeError) as exc:
        msg = f"platform release document carries a malformed schema_version: {exc}"
        raise _ReleaseNotAdmittedError(msg) from exc
    if version_too_new:
        msg = (
            f"platform schema_version {schema_version} exceeds this deployment's "
            f"max_schema_version {settings.max_schema_version}"
        )
        raise _ReleaseNotAdmittedError(msg)
    try:
        write_schema = write_component_schema(release=release)
        frame_schema = describe_frame_schema(release=release)
        describe_schema = describe_component_schema()
        tool_names = _register_tools(
            write_schema=write_schema, frame_schema=frame_schema, describe_schema=describe_schema
        )
    except Exception as exc:
        msg = f"could not mount the platform's tools: {exc}"
        raise _ReleaseNotAdmittedError(msg) from exc
    set_assembler(assembler)
    return tool_names


async def _mount_within_deadline(
    assembler: PlatformAssembler,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[str, ...]:
    """Retry `_mount` until the platform answers or
    `settings.release_startup_deadline_s` passes.

    Waits `settings.release_retry_base_s` after the first failed fetch and
    doubles the wait after each later one, up to `settings.release_retry_max_s`.
    No fetch starts after the deadline, and one still in flight at the
    deadline is cancelled. A release that answers and fails admission raises
    `_ReleaseNotAdmittedError` at once, since asking again returns the same
    document. `sleep` is replaced in tests.

    Returns:
        The registered tools' names, as `_mount` returns them.

    Raises:
        RuntimeError: No fetch reached the platform before the deadline,
            chained from the last fetch's error.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + settings.release_startup_deadline_s
    delay = settings.release_retry_base_s
    attempt = 0
    while True:
        attempt += 1
        try:
            async with asyncio.timeout_at(deadline):
                return await _mount(assembler)
        except _ReleaseNotAdmittedError:
            raise
        except Exception as exc:
            if loop.time() + delay >= deadline:
                msg = (
                    f"could not reach the platform within {settings.release_startup_deadline_s:g}s "
                    f"(fetches: {attempt}; last error: {exc!r})"
                )
                raise RuntimeError(msg) from exc
            logger.warning(
                "could not fetch the platform's release (attempt %d); retrying in %.1fs: %r",
                attempt,
                delay,
                exc,
            )
        await sleep(delay)
        delay = min(delay * 2, settings.release_retry_max_s)


@asynccontextmanager
async def _lifespan(_server: FastMCP) -> AsyncIterator[None]:
    """Starts the chart worker, then wires the platform (`_platform_lifespan`).
    Shutdown stops the worker last.

    This lifespan, not `main()`, is what actually runs in production: the
    harness mounts this app's ASGI sub-app via `combine_lifespans` and never
    calls `main()`. The harness's combined server at `/mcp/` mounts the same
    `mcp` object, and FastMCP counts a server's lifespan entries per object,
    so the worker starts once for both. The worker starts first so it boots
    while the release is fetched.
    """
    _chart_worker.start()
    try:
        async with _platform_lifespan():
            yield
    finally:
        await _chart_worker.stop()


@asynccontextmanager
async def _platform_lifespan() -> AsyncIterator[None]:
    """Wires the platform assembler, admits its release, and mounts the tools.

    Startup finishes, and the server answers its first request, only once
    the three tools are mounted or the attempt to mount them has failed for
    good, so no `tools/list` ever answers from a half-started server.

    With `settings.presentation_release_required` set, the default, the
    release fetch is retried until `settings.release_startup_deadline_s`
    (`_mount_within_deadline`), and any failure to reach the platform, admit
    its release, or mount its tools fails startup and exits the process.
    Unset, which only local development and tests do, the release is
    fetched once, and any failure brings the server up tool-less with
    `/healthz` reporting `degraded` until its next restart. Shutdown
    unregisters whatever mounted, clears the holder, and closes the client.

    `settings.frame_store_bucket` set installs the shared frame
    materialisation store here too, mirroring `data_mcp.server`'s own
    guarded `install_store_if_unset` call: in this service's own deployment
    (mounted in-process alongside data-mcp and analysis-mcp,
    `harness.server.create_app`), one of those two services' lifespans has
    already installed it before this one runs (`combine_lifespans` enters
    every mounted lifespan in mount order), so this call is normally a
    no-op; it only does real work for a hypothetical standalone deployment
    of this service against a durable bucket. This service never installs
    the in-memory fallback itself and never closes what it did not
    install -- reading through the holder's own lazy default (`get_store()`)
    is enough for the publish hook below, and whichever of the other two
    services' lifespans installed a real store also owns closing it.

    That install can raise `ContentVersionDeriverUnregistered` -- a
    standalone deployment with `frame_store_verify_content_version` set has
    no `analysis_mcp` in the process to register a deriver (`settings.py`'s
    own field docstring). It runs before the platform is reached, fails
    startup under `settings.presentation_release_required`, and otherwise
    degrades `/healthz` with no tools.
    """
    if settings.frame_store_bucket is not None:
        try:
            install_store_if_unset(
                bucket_name=settings.frame_store_bucket,
                verify_content_version_on_read=settings.frame_store_verify_content_version,
            )
        except Exception as exc:
            logger.exception("could not install the shared frame materialisation store at startup")
            if settings.presentation_release_required:
                raise
            _set_health("degraded", f"could not install the frame store: {exc}")
            yield
            return

    running = PlatformAssembler(settings=settings)
    mounted: tuple[str, ...] = ()
    try:
        try:
            if settings.presentation_release_required:
                mounted = await _mount_within_deadline(running)
            else:
                mounted = await _mount(running)
        except _ReleaseNotAdmittedError as exc:
            logger.exception("the platform's release could not be mounted at startup")
            if settings.presentation_release_required:
                raise RuntimeError(str(exc)) from exc
            _set_health("degraded", str(exc))
        except Exception as exc:
            logger.exception("could not fetch the platform's release at startup")
            if settings.presentation_release_required:
                raise
            _set_health("degraded", f"could not reach the platform: {exc}")
        else:
            _set_health("ok")
        yield
    finally:
        for name in mounted:
            mcp.local_provider.remove_tool(name)
        set_assembler(None)
        await running.aclose()


mcp = FastMCP(name="presentation-mcp", lifespan=_lifespan, auth=BearerTokenVerifier())
mcp.add_middleware(LangfuseSessionMiddleware())
mcp.add_middleware(LangfuseToolIoMiddleware())


def _renderer_result() -> ResourceResult:
    """The current renderer page, as `text/html;profile=mcp-app` with the CSP
    `renderer_csp` declares in the content's `_meta.ui.csp`. The page's script
    and stylesheet, and the payload's stylesheets, load from those origins."""
    csp = ResourceCSP.model_validate(renderer_csp(settings.presentation_public_url))
    return ResourceResult(
        [
            ResourceContent(
                renderer_html(),
                mime_type=UI_MIME_TYPE,
                meta={"ui": {"csp": csp.model_dump(by_alias=True, exclude_none=True)}},
            )
        ]
    )


@mcp.resource(
    _RENDERER_URI,
    name="Prefab renderer",
    mime_type=UI_MIME_TYPE,
    app=AppConfig(csp=ResourceCSP.model_validate(renderer_csp(settings.presentation_public_url))),
)
def renderer() -> ResourceResult:
    """The renderer page under its current URI, `_RENDERER_URI`."""
    return _renderer_result()


@mcp.resource(
    "ui://presentation-mcp/renderer-{version}.html",
    name="Prefab renderer at any version",
    mime_type=UI_MIME_TYPE,
    app=AppConfig(csp=ResourceCSP.model_validate(renderer_csp(settings.presentation_public_url))),
)
def renderer_at_any_version(version: str) -> ResourceResult:
    """The current renderer page under any `renderer-{version}.html` URI.

    A host may keep the URI from when its user added the app and read it after
    the server has moved to a new page, so an old URI answers with the
    current page and CSP, and the view still draws."""
    del version
    return _renderer_result()


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(_request: Request) -> JSONResponse:
    return JSONResponse(_health)


@mcp.custom_route("/theme-{version}.css", methods=["GET"])
async def theme_css(request: Request) -> Response:
    """The theme stylesheet the views' payloads name (`theme.theme_url`).

    A custom route sits outside the MCP bearer gate, as `/healthz` does, which
    the view's iframe needs since it cannot send the token. The current hash
    is immutable for a year. Any other hash is a view saved under an older
    theme, and still gets the current CSS, but is revalidated every time.
    """
    current = request.path_params["version"] == THEME_HASH
    cache = "public, max-age=31536000, immutable" if current else "no-cache"
    return Response(
        THEME_CSS,
        media_type="text/css",
        headers={"Cache-Control": cache, "Access-Control-Allow-Origin": "*"},
    )


def main() -> None:
    """Console-script entry point: run presentation-mcp over streamable HTTP."""
    configure_logging()
    init_telemetry(service_name="presentation-mcp")
    apply_gate(mcp, token=settings.mcp_api_token, disabled=settings.mcp_auth_disabled)
    logger.info("Starting presentation-mcp on port %d", settings.port)
    mcp.run(transport="streamable-http", host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
