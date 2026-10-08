"""FastMCP server scaffold for analysis-mcp.

Holds infrastructure: the FastMCP instance, the `/healthz` liveness route,
the fixture-store lifespan, and the console-script entry point (settings,
logging, telemetry, transport). Registers the closed nine-verb set, one
tool per verb, each under an explicit wire name, and `resolve_anchor`
(`rolling_tool.py`), which finds the latest period rolling calls follow and
computes no frame. `filter` needs the explicit name because the Python
function backing it cannot be called `filter` without shadowing the builtin
(see `filter_tool.py`'s own docstring).

Refusal policy: every anticipated failure gets its own typed refusal, built
and returned via `to_tool_result` inside that verb's own tool module (see
each `*_tool.py`'s own module docstring). Underneath all nine sits the
backstop: `_with_internal_error_backstop` wraps each verb at its
registration site, so an exception the verb's own typed handling missed
becomes an `internal_error` refusal instead of a raw traceback crossing the
MCP wire. Wrapping the verb puts that backstop inside the tool boundary,
after `fastmcp` has resolved the tool and validated its arguments, which
leaves a bad argument to report itself as the protocol error it is. An
unknown tool name is also left to report itself as a protocol error, but
`_NameUnknownVerbsMiddleware` names the nine registered verbs and points
the caller at `tools/list` in that error's text.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

import mcp.types as mt
from common.auth import BearerTokenVerifier, apply_gate
from common.logging import configure_logging
from common.materialisation import (
    InMemoryMaterialisationStore,
    MaterialisationStore,
    aclose_store,
    build_store,
    set_store,
    store_is_set,
)
from common.mcp_telemetry import LangfuseSessionMiddleware, LangfuseToolIoMiddleware
from common.refusal import log_unexpected_error
from common.telemetry import init_telemetry
from common.tool_annotations import READ_ONLY_CATALOGUE
from fastmcp import FastMCP
from fastmcp.exceptions import NotFoundError, ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult
from starlette.requests import Request
from starlette.responses import JSONResponse

from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.compare_tool import compare
from analysis_mcp.filter_tool import filter_verb
from analysis_mcp.fixtures import seed_fixture_store
from analysis_mcp.growth_tool import growth
from analysis_mcp.join_tool import join
from analysis_mcp.quotient_of_tool import quotient_of
from analysis_mcp.rank_tool import rank
from analysis_mcp.result import IMPLEMENTATION_VERSIONS, VERB_OUTPUT_SCHEMA
from analysis_mcp.rolling_tool import (
    RESOLVE_ANCHOR_OUTPUT_SCHEMA,
    RESOLVE_ANCHOR_VERSION,
    resolve_anchor,
)
from analysis_mcp.settings import Settings
from analysis_mcp.share_of_tool import share_of
from analysis_mcp.sum_of_tool import sum_of
from analysis_mcp.tool_errors import ToolRefusal, to_tool_result

logger = logging.getLogger(__name__)


def _with_internal_error_backstop(
    fn: Callable[..., Awaitable[ToolResult]], wire_name: str
) -> Callable[..., Awaitable[ToolResult]]:
    """Wrap a verb function so an exception its own typed handling missed
    becomes an `internal_error` refusal rather than escaping raw. The broad
    `except` is the point: it catches what the typed handling could not
    enumerate in advance. The exception is logged under a trace id, and the
    refusal names that id and none of the exception's text.

    `functools.wraps` sets the `__wrapped__` attribute that
    `inspect.signature` follows by default, and that signature is what
    `fastmcp.tools.function_parsing` reads to build a tool's schema, so a
    verb registered through this wrapper produces the schema the bare
    function would.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> ToolResult:
        try:
            return await fn(*args, **kwargs)
        except Exception:
            trace_id = log_unexpected_error(logger, f"analysis-mcp tool {wire_name}")
            return to_tool_result(ToolRefusal.internal_error(tool=wire_name, trace_id=trace_id))

    return wrapper


async def _build_store(settings: Settings) -> MaterialisationStore:
    """Build the materialisation store this process serves: a
    `GcsMaterialisationStore` when `settings.frame_store_bucket` names a
    bucket, an in-process `InMemoryMaterialisationStore` otherwise (see
    `common.materialisation.build_store`).

    Fixture seeding (`settings.seed_fixture_frames`, default off) only ever
    targets the in-memory store: development-finance fixtures are for local
    development against the store's own default, restart-losing backend,
    never for a durable bucket a real deployment points this process at.
    """
    store = build_store(
        bucket_name=settings.frame_store_bucket,
        verify_content_version_on_read=settings.frame_store_verify_content_version,
    )
    if settings.seed_fixture_frames and isinstance(store, InMemoryMaterialisationStore):
        seeded = await seed_fixture_store(store)
        logger.info("Seeded fixture pairs: %s", seeded)
    return store


@asynccontextmanager
async def _lifespan(_server: FastMCP) -> AsyncIterator[None]:
    """Builds the materialisation store on startup, unless one is already installed;
    closes it on shutdown regardless of which side installed it.

    This lifespan, not `main()`, is what actually runs in production: the
    harness mounts this app's ASGI sub-app via `combine_lifespans` and never
    calls `main()`. `combine_lifespans` runs every mounted service's own lifespan
    before the composed app starts serving, and data-mcp's lifespan may run first and
    already have installed a durable store over the same bucket (see
    `common.materialisation.factory.install_store_if_unset`'s own docstring for the
    ownership rule) -- `store_is_set()` is this lifespan's side of that same rule: skip
    building (and, with it, fixture seeding) entirely rather than replace what's
    already there.
    """
    if not store_is_set():
        set_store(await _build_store(Settings()))
    try:
        yield
    finally:
        await aclose_store()


mcp = FastMCP(name="analysis-mcp", lifespan=_lifespan, auth=BearerTokenVerifier())


_TITLES: Mapping[str, str] = {
    "aggregate": "Aggregate by group",
    "compare": "Compare two series",
    "growth": "Calculate growth",
    "rank": "Rank entities",
    "share_of": "Calculate share of total",
    "join": "Join two results",
    "filter": "Filter rows",
    "sum_of": "Add two series",
    "quotient_of": "Divide two series",
}


def _register(fn: Callable[..., Awaitable[ToolResult]], wire_name: str) -> None:
    """Register one verb under its wire name, behind the internal-error
    backstop, declaring its `_TITLES` and `IMPLEMENTATION_VERSIONS` entries and
    `VERB_OUTPUT_SCHEMA` -- the schema `result.VerbResult` derives and
    `tool_pipeline.store_result_and_render` builds every successful payload
    from. `-> ToolResult` on the wrapped function suppresses fastmcp's own
    schema derivation, so `output_schema` is passed explicitly here rather
    than left to be derived.
    """
    mcp.tool(
        _with_internal_error_backstop(fn, wire_name),
        name=wire_name,
        version=IMPLEMENTATION_VERSIONS[wire_name],
        title=_TITLES[wire_name],
        annotations=READ_ONLY_CATALOGUE,
        output_schema=VERB_OUTPUT_SCHEMA,
    )


_register(aggregate, "aggregate")
_register(compare, "compare")
_register(growth, "growth")
_register(rank, "rank")
_register(share_of, "share_of")
_register(join, "join")
_register(filter_verb, "filter")
_register(sum_of, "sum_of")
_register(quotient_of, "quotient_of")

# `resolve_anchor` computes no frame, so it declares its own result schema and
# version in place of the verbs'.
mcp.tool(
    _with_internal_error_backstop(resolve_anchor, "resolve_anchor"),
    name="resolve_anchor",
    version=RESOLVE_ANCHOR_VERSION,
    title="Find the latest period",
    annotations=READ_ONLY_CATALOGUE,
    output_schema=RESOLVE_ANCHOR_OUTPUT_SCHEMA,
)


class _NameUnknownVerbsMiddleware(Middleware):
    """On an unknown tool name, replaces fastmcp's bare `NotFoundError` with
    a `ToolError` that also names the server's registered verbs and points
    the caller at `tools/list`.

    `FastMCP.call_tool` runs middleware before it resolves the tool name, so
    `call_next` raises `NotFoundError` for a name no tool is registered
    under, and this handler is the one place that sees that failure before
    it crosses the MCP wire as the caller's error text. The replacement is
    a `ToolError`: the MCP SDK's own `call_tool` handler catches
    `NotFoundError` past this point and discards its message in favour of a
    fixed `Unknown tool: {name!r}`, which would erase everything this
    handler adds.
    """

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        try:
            return await call_next(context)
        except NotFoundError as exc:
            verbs = sorted(tool.name for tool in await mcp.list_tools(run_middleware=False))
            raise ToolError(
                f"{exc} -- analysis-mcp registers these verbs: {', '.join(verbs)}. "
                "Call tools/list to see the current set."
            ) from exc


mcp.add_middleware(LangfuseSessionMiddleware())
mcp.add_middleware(LangfuseToolIoMiddleware())
mcp.add_middleware(_NameUnknownVerbsMiddleware())


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def main() -> None:
    """Console-script entry point: run analysis-mcp over streamable HTTP."""
    settings = Settings()
    configure_logging()
    init_telemetry(service_name="analysis-mcp")
    apply_gate(mcp, token=settings.mcp_api_token, disabled=settings.mcp_auth_disabled)
    logger.info("Starting analysis-mcp on port %d", settings.port)
    mcp.run(transport="streamable-http", host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
