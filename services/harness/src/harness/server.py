"""FastAPI app factory and entrypoint for the harness service."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx2
import uvicorn
from analysis_mcp.server import mcp as analysis_mcp
from common.auth import apply_gate, bearer_token
from common.logging import configure_logging
from common.mcp_telemetry import LOOPBACK_SCOPE_KEY
from common.telemetry import backend as telemetry_backend
from common.telemetry import init_telemetry
from data_mcp.server import mcp as data_mcp
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.utilities.lifespan import combine_lifespans
from google.adk.sessions import DatabaseSessionService
from mcp import MCPError
from mcp.shared._httpx_utils import McpHttpClientFactory
from orchestrator.agent import create_orchestrator_agent
from presentation_mcp.server import mcp as presentation_mcp
from starlette.routing import get_route_path
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose

from harness import __version__
from harness.agents import build_tool_surface
from harness.combined_mcp import build_combined_mcp, children_start_once, combined_http_app
from harness.early_mint import EarlyMints
from harness.generative import GeminiEdit
from harness.graph.state import AGENT_NAME
from harness.graph.workflow import create_root_workflow
from harness.live_refresh import build_live_refresh, create_live_refresh_router
from harness.mcp_client import McpExecutor, McpResolver
from harness.orchestrator_tools import build_orchestrator_tools
from harness.prefix_cache import PrefixCache
from harness.provenance import ProvenanceStore
from harness.run_ledger import StudioRunLedger
from harness.session_store import build_session_service
from harness.settings import (
    ANALYSIS_MCP_MOUNT_PATH,
    COMBINED_MCP_MOUNT_PATH,
    DATA_MCP_MOUNT_PATH,
    PRESENTATION_MCP_MOUNT_PATH,
    Settings,
)
from harness.studio import StudioRunner, create_studio_router
from harness.studio_state import (
    StudioStore,
    make_canvas_cards,
    make_canvas_series,
    make_recall_part,
    make_remove_cards,
)
from harness.wiring import build_planner, handing_back_planner_failures

logger = logging.getLogger(__name__)

SERVICE_NAME = "harness"

_RUN_SSE_GONE = (
    "run_sse was a placeholder and is retired. The studio v1 interface "
    "replaces it: POST /api/studio/turn (SSE stream), /api/studio/interrupt, "
    "/api/studio/composition -- see docs/studio-contract-v1.md."
)


def _loopback_mcp_client(app: FastAPI, mounted_mcp_path: str, *, statuses: list[int]) -> Client:
    """An MCP client that talks to one mount on `app` over ASGI in-process.

    `mounted_mcp_path` is a mount path plus fastmcp's own protocol suffix,
    e.g. `f"{DATA_MCP_MOUNT_PATH}/mcp"`. `statuses` collects the HTTP status
    of every response the mount sends: mcp's client reports a non-2xx answer
    as `MCPError(INTERNAL_ERROR)` without its status, so this list is the
    only place a caller can see a 401.

    Used only by the startup self-check below. The composed app isn't
    listening on a real socket yet during its own lifespan's startup phase
    (uvicorn opens the port only after startup completes), so this drives
    the same route and middleware chain (`RequireAuthMiddleware` included)
    that a live request over loopback HTTP would hit, without one.
    """

    async def record(response: httpx2.Response) -> None:
        statuses.append(response.status_code)

    def factory(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url="http://selfcheck",
            event_hooks={"response": [record]},
            **kwargs,
        )

    transport = StreamableHttpTransport(
        url=f"http://selfcheck{mounted_mcp_path}",
        auth=bearer_token(),
        httpx_client_factory=cast(McpHttpClientFactory, factory),
    )
    return Client(transport)


# The paths a direct mount serves to a client off loopback: each MCP
# package's `@custom_route("/healthz")`, and presentation-mcp's
# `/theme-{version}.css`, which a view's iframe loads from the host's
# browser without the bearer. The theme route answers every version, since
# a saved view names the hash it was drawn under.
_DIRECT_MOUNT_PUBLIC_PATH = re.compile(r"/healthz|/theme-[^/]+\.css")


def _is_loopback(client: tuple[str, int] | None) -> bool:
    if client is None:
        return False
    try:
        return ipaddress.ip_address(client[0]).is_loopback
    except ValueError:
        return False


def _loopback_only(app: ASGIApp) -> ASGIApp:
    """`app`, refusing every request from off loopback except its public paths.

    The direct mounts serve the harness's own clients, which reach them at
    `http://127.0.0.1:{port}` (`Settings`' URL defaults). An outside client
    uses the combined server at `/mcp/`, where the rate limit, the
    `get_data` budget and result shaping apply. A request from anywhere else
    gets 403 before the bearer check, whatever token it carries.

    A request from loopback reaches `app` with `LOOPBACK_SCOPE_KEY` set in its
    scope, which is what lets `common.mcp_telemetry.LangfuseSessionMiddleware`
    trust the Langfuse session in its `_meta`. A request through the combined
    server never passes this guard, so it never carries the mark.

    The client address is the socket peer's. On Cloud Run that is the
    front end's address, never loopback. uvicorn replaces it from
    `X-Forwarded-For` only when the peer is in `FORWARDED_ALLOW_IPS`, which
    defaults to 127.0.0.1, and the harness's own clients send no such
    header.
    """

    async def guarded(scope: Scope, receive: Receive, send: Send) -> None:
        if _is_loopback(scope.get("client")):
            await app({**scope, LOOPBACK_SCOPE_KEY: True}, receive, send)
        elif scope["type"] == "lifespan" or _DIRECT_MOUNT_PUBLIC_PATH.fullmatch(
            get_route_path(scope)
        ):
            await app(scope, receive, send)
        elif scope["type"] == "http":
            refusal = JSONResponse(
                {
                    "detail": f"this endpoint serves loopback only; MCP clients use {COMBINED_MCP_MOUNT_PATH}/"
                },
                status_code=403,
            )
            await refusal(scope, receive, send)
        else:
            await WebSocketClose()(scope, receive, send)

    return guarded


# Total wall-clock budget `_prepare_tables_with_retries` spends retrying
# `DatabaseSessionService.prepare_tables` at startup. A runtime identity's
# IAM grant can still be propagating when a fresh revision starts, taking
# minutes rather than seconds to land; Cloud Run's default startup probe
# window is 240 s, so this leaves headroom for the rest of startup.
_PREPARE_TABLES_RETRY_BUDGET_S = 120.0
_PREPARE_TABLES_INITIAL_BACKOFF_S = 1.0
_PREPARE_TABLES_MAX_BACKOFF_S = 30.0


async def _prepare_tables_with_retries(
    session_service: DatabaseSessionService,
    *,
    connect_timeout_s: float,
    budget_s: float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    now: Callable[[], float] = time.monotonic,
) -> None:
    """Retry `prepare_tables` with bounded exponential backoff.

    Each attempt is bounded by `connect_timeout_s`, the same deadline that
    bounds one connector dial elsewhere; a failed attempt is logged with its
    exception and followed by a `sleep` pause that doubles up to
    `_PREPARE_TABLES_MAX_BACKOFF_S`, capped so it never pushes the call past
    `budget_s` total. The last exception is re-raised once no time remains
    in the budget. `sleep` and `now` are seams for tests to replace real
    waiting and a real clock with instant, deterministic fakes.
    """
    deadline = now() + budget_s
    backoff_s = _PREPARE_TABLES_INITIAL_BACKOFF_S
    attempt = 0
    while True:
        attempt += 1
        try:
            async with asyncio.timeout(connect_timeout_s):
                await session_service.prepare_tables()
            return
        except Exception:
            remaining_s = deadline - now()
            if remaining_s <= 0:
                logger.exception(
                    "prepare_tables attempt %d failed and the %.0fs retry budget is exhausted",
                    attempt,
                    budget_s,
                )
                raise
            logger.exception(
                "prepare_tables attempt %d failed, retrying (%.0fs left in the retry budget)",
                attempt,
                remaining_s,
            )
            await sleep(min(backoff_s, remaining_s))
            backoff_s = min(backoff_s * 2, _PREPARE_TABLES_MAX_BACKOFF_S)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    init_telemetry(service_name=SERVICE_NAME)
    # Runs at startup: an instance that can't reach its database fails its
    # startup probe (`DatabaseSessionService.prepare_tables`'s own docstring
    # recommends this for latency-sensitive services), since every studio
    # route needs the session service. It's a no-op after the first call, so
    # this costs nothing beyond startup. `_prepare_tables_with_retries`
    # bounds each attempt by `session_store_connect_timeout_s` and retries
    # transient failures within its own budget.
    if isinstance(app.state.session_service, DatabaseSessionService):
        await _prepare_tables_with_retries(
            app.state.session_service,
            connect_timeout_s=app.state.settings.session_store_connect_timeout_s,
            budget_s=_PREPARE_TABLES_RETRY_BUDGET_S,
        )
    # A 401 from a mounted app reaches an MCP client as a raised `MCPError`,
    # not a tool result: `raise_on_error=False` on `call_tool` doesn't catch
    # it, and nothing upstream of `McpExecutor`'s call sites has a broad
    # handler for it either. Without this check a misconfigured token surfaces as an
    # unhandled exception on a user's first turn instead of failing startup.
    # This runs after every mounted app's own lifespan has started (this
    # lifespan is last in `combine_lifespans`, below), so their session
    # managers are already up.
    statuses: list[int] = []
    try:
        async with _loopback_mcp_client(
            app, f"{DATA_MCP_MOUNT_PATH}/mcp", statuses=statuses
        ) as client:
            await client.list_tools()
    except MCPError as exc:
        if 401 not in statuses:
            raise
        raise RuntimeError(
            "harness could not authenticate to its own data-mcp mount at startup -- "
            "check MCP_API_TOKEN (and MCP_AUTH_DISABLED) agree across this process"
        ) from exc
    yield
    # Early mints still running are cancelled before the clients they call
    # through close. apis finishes any mint already posted.
    try:
        if app.state.early_mints is not None:
            await app.state.early_mints.aclose()
    except Exception:
        logger.exception("failed to cancel the early mints during shutdown")
    # `StudioRunner.aclose` releases the studio executor's resources, which
    # are `McpExecutor`'s long-lived per-conversation presentation-mcp
    # clients, the one long-lived client this process opens (every other MCP
    # call site is `async with` and closes per call). Guarded so a broken
    # close during shutdown is logged rather than left to crash the ASGI
    # shutdown sequence.
    try:
        await app.state.studio.aclose()
    except Exception:
        logger.exception("failed to close the studio executor during shutdown")
    try:
        await app.state.live_refresh.aclose()
    except Exception:
        logger.exception("failed to close the live refresh clients during shutdown")
    # Disposes whatever `build_session_service` opened for the session
    # service above: a no-op for the in-memory backend, the engine dispose
    # `DatabaseSessionService.close` already does for the plain-`DATABASE_URL`
    # backend, and the engine-dispose plus connector-close for the Cloud SQL
    # Connector path (`harness.session_store.SessionServiceHandle`). Guarded
    # the same way as the studio close above.
    try:
        await app.state.session_service_aclose()
    except Exception:
        logger.exception("failed to close the session service during shutdown")


def create_app(
    *,
    settings: Settings | None = None,
) -> FastAPI:
    """Build the harness FastAPI app.

    Builds the session service eagerly so wiring failures surface at
    startup rather than on first request. `settings` is injectable for
    tests.

    Logging is configured here as well as in the lifespan, so eager
    construction below, which logs its session-service choice, has its
    output captured ahead of the logging module's pre-configuration
    defaults. `configure_logging` is idempotent, so the lifespan's call is
    a no-op.

    The data-mcp, analysis-mcp, and presentation-mcp FastMCP servers are
    mounted as ASGI sub-apps rather than run as separate services: one warm
    Cloud Run instance instead of four, no cold starts on agent tool calls,
    and loopback instead of network hops (see docs/architecture.md). The
    packages stay separate, so splitting any back out later is a new
    Dockerfile and a URL change (`DATA_MCP_URL` / `ANALYSIS_MCP_URL` /
    `PRESENTATION_MCP_URL`), not a rewrite.

    A fourth mount at `COMBINED_MCP_MOUNT_PATH` serves the combined server
    (`harness.combined_mcp`) to outside MCP clients: one FastMCP parent over
    the same three objects. The harness's own clients keep calling the
    per-server mounts, which answer only loopback clients beyond their
    public paths (`_loopback_only`).

    Each mounted FastMCP app owns a `StreamableHTTPSessionManager` that only
    starts inside that app's own lifespan. `combine_lifespans` runs the
    harness's lifespan and every mounted app's lifespan together, so each
    mounted sub-app's MCP session manager actually starts, per fastmcp's
    ASGI-integration contract for sub-app mounting.
    """
    configure_logging()
    resolved_settings = settings or Settings()

    # Every mounted `FastMCP(...)` already carries a `BearerTokenVerifier`:
    # the three singletons from import time (see `common.auth`'s own module
    # docstring), the combined parent from `build_combined_mcp`. `apply_gate`
    # resolves the process-wide token/disabled policy (first caller wins, and
    # every call here reads the same `resolved_settings`, so they agree) and
    # only has to act when the override says the gate should come off. This
    # must run before `.http_app()`, which bakes `auth`'s presence into the
    # route table it builds.
    combined_mcp = build_combined_mcp(settings=resolved_settings)
    for mounted in (data_mcp, analysis_mcp, presentation_mcp, combined_mcp):
        apply_gate(
            mounted,
            token=resolved_settings.mcp_api_token,
            disabled=resolved_settings.mcp_auth_disabled,
        )

    data_mcp_app = data_mcp.http_app(transport="streamable-http")
    analysis_mcp_app = analysis_mcp.http_app(transport="streamable-http")
    presentation_mcp_app = presentation_mcp.http_app(transport="streamable-http")
    combined_mcp_app = combined_http_app(combined_mcp)

    app = FastAPI(
        title=SERVICE_NAME,
        # `combine_lifespans` enters these in order. The combined server
        # starts after the per-server mounts, so its children's lifespans
        # have already run and `children_start_once` can check it does not
        # run them again. `_lifespan` runs last. Its own pre-yield code,
        # including the startup auth self-check, executes only once every
        # mount has started.
        lifespan=combine_lifespans(
            data_mcp_app.lifespan,
            analysis_mcp_app.lifespan,
            presentation_mcp_app.lifespan,
            children_start_once(combined_mcp_app.lifespan),
            _lifespan,
        ),
    )
    app.state.settings = resolved_settings
    # `create_app` needs the `aclose` half too, for `_lifespan`'s shutdown
    # phase.
    session_service_handle = build_session_service(settings=resolved_settings)
    app.state.session_service = session_service_handle.service
    app.state.session_service_aclose = session_service_handle.aclose

    # Built before the studio stack, which replays a card's frame through it.
    live_refresh = build_live_refresh(settings=resolved_settings)
    app.state.live_refresh = live_refresh

    # The studio stack: the orchestrator agent and its workflow behind the
    # v1 wire contract (docs/studio-contract-v1.md). Resolver/executor speak
    # MCP to the mounts below over loopback; the agents build their genai
    # clients lazily, so construction here stays keyless. A missing
    # GOOGLE_API_KEY fails the first turn, visibly, as a stream-fatal error
    # frame. The planner node is `planner.agent`'s planner with
    # `translator.tool.build_translate`'s translation attached to
    # `propose_plan`, behind `harness.planner_seam.PlannerSeam`.
    # The server allows exactly the origins listed in STUDIO_CORS_ORIGINS (a
    # browser page on another port during local testing); with the list
    # empty it sends no CORS headers, which is the deployed posture.
    if resolved_settings.studio_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved_settings.studio_cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    if resolved_settings.studio_api_token is None:
        # PLACEHOLDER ACCESS CONTROL. Without the shared secret every
        # non-empty bearer token is accepted, so anyone who holds the URL
        # can drive the agent. Set STUDIO_API_TOKEN on any server that is
        # reachable beyond the developer's machine.
        logger.warning(
            "STUDIO AUTH IS OPEN: STUDIO_API_TOKEN is unset, so any non-empty "
            "bearer token is accepted. Set it before exposing this server."
        )
    # Shared with `StudioRunner` below (`provenance=provenance`), so the
    # approved-plan path's `get_data` writer (`_RecordingSurface`) and
    # `McpExecutor`'s own direct-fetch writer land in the same store
    # component assembly reads from -- see `harness.provenance`.
    provenance = ProvenanceStore()
    # The executor replays a rolling frame through the refresh's clients when
    # a card draws other series than the frame's latest period was taken over.
    executor = McpExecutor(
        settings=resolved_settings,
        provenance=provenance,
        card_replay=live_refresh.replay,
        dataset_versions=live_refresh.dataset_versions,
    )
    # Each card's retrievals, sent before approval through the executor's
    # own `get_data` clients, as many at once as a run's steps.
    early_mints = (
        EarlyMints(
            send=executor.get_data_in_background,
            max_concurrent=resolved_settings.executor_max_concurrent_steps,
            settle_s=resolved_settings.early_mint_wait_s,
        )
        if resolved_settings.early_mint
        else None
    )
    app.state.early_mints = early_mints
    if early_mints is not None:
        logger.info(
            "early mints: on; a card's stream waits up to %gs for their mint POSTs",
            resolved_settings.early_mint_wait_s,
        )
    else:
        logger.info("early mints: off (EARLY_MINT=false)")
    # The store exists before the workflow because the recall seam is
    # bound to it (conversation resolved per request via contextvar).
    store = StudioStore(app.state.session_service, app_name=AGENT_NAME)
    # Built once and threaded into both the `execute` node below and
    # `build_planner`'s planner (which reads skills through it and
    # passes it on to the translator), so the two compile against one set
    # of schemas, read once, rather than each fetching its own copy
    # through a separate `McpSurface`. Opens no connection at construction.
    tool_surface = build_tool_surface(settings=resolved_settings)
    # One for the process, shared by the planner and the orchestrator
    # across every conversation (`harness.prefix_cache`).
    prefix_cache = PrefixCache() if resolved_settings.studio_prefix_cache else None
    logger.info("orchestrator thinking level: %s", resolved_settings.orchestrator_thinking_level)

    def create_workflow(recorder: Any, surface_recorder: Any) -> Any:
        # Runs are kept in the conversation's ledger session, and a run
        # completes once its derived results are published with their
        # replay recipes.
        run_ledger = StudioRunLedger(store)
        # The orchestrator's tools run through the same recorder and
        # recording surface as the `execute` node, so their steps and cards
        # reach the wire.
        orchestrator = create_orchestrator_agent(
            model_name=resolved_settings.model_name,
            prefix_cache=prefix_cache,
            thinking_level=resolved_settings.orchestrator_thinking_level,
            tools=build_orchestrator_tools(
                resolver=McpResolver(settings=resolved_settings),
                executor=recorder,
                surface=surface_recorder,
                dfo_faq_surface=tool_surface,
                definition_surface=tool_surface,
                recall_part=make_recall_part(store),
                canvas_series=make_canvas_series(store),
                canvas_cards=make_canvas_cards(store),
                remove_cards=make_remove_cards(store),
            ),
        )
        return create_root_workflow(
            # A planner that cannot plan hands its reason back to the agent
            # and the turn goes on.
            planner=handing_back_planner_failures(
                build_planner(
                    settings=resolved_settings, surface=tool_surface, prefix_cache=prefix_cache
                )
            ),
            orchestrator=orchestrator,
            executor=recorder,
            # The recorded wrapper, not the bare `tool_surface` above: the
            # `execute` node must run against the same recorder that turns
            # every plan-step call into an `activity.step` frame
            # (`_RecordingSurface`), or the served turns go dark. The bare
            # surface stays the one `build_planner` compiles the
            # planner/translator schemas against and the planner reads
            # skills through; neither is a plan step. The `seed` node
            # stores definitions through the bare surface, off the
            # Activity feed.
            tool_surface=surface_recorder,
            definition_surface=tool_surface,
            run_ledger=run_ledger,
            publisher=executor.publish_result,
            max_concurrent_steps=resolved_settings.executor_max_concurrent_steps,
            early_mints=early_mints,
        )

    studio = StudioRunner(
        create_workflow=create_workflow,
        executor=executor,
        session_service=app.state.session_service,
        store=store,
        edit_model=GeminiEdit(
            model_name=resolved_settings.model_name,
            timeout_s=resolved_settings.turn_model_time_s,
        ),
        tool_surface=tool_surface,
        provenance=provenance,
        early_mints=early_mints,
    )
    app.state.studio = studio
    app.include_router(create_studio_router(studio, settings=resolved_settings))

    # `POST /api/studio/refresh` sits on a router of its own: it takes the
    # live-refresh principal as its bearer and runs no model.
    if resolved_settings.live_refresh_principal is None:
        logger.warning(
            "LIVE REFRESH IS CLOSED: LIVE_REFRESH_PRINCIPAL is unset, so "
            "POST /api/studio/refresh refuses every caller with 401."
        )
    app.include_router(create_live_refresh_router(settings=resolved_settings))

    # Each mounted app keeps its own `/healthz` (registered by each MCP
    # package's server module) under its prefix, e.g. `/mcp/data/healthz`,
    # at a distinct path from the harness's own top-level `/healthz` below.
    # The combined server's `/mcp/healthz` reports its children's. Starlette
    # matches mounts in registration order, so the combined mount at `/mcp`
    # goes last.
    app.mount(DATA_MCP_MOUNT_PATH, _loopback_only(data_mcp_app))
    app.mount(ANALYSIS_MCP_MOUNT_PATH, _loopback_only(analysis_mcp_app))
    app.mount(PRESENTATION_MCP_MOUNT_PATH, _loopback_only(presentation_mcp_app))
    app.mount(COMBINED_MCP_MOUNT_PATH, combined_mcp_app)

    # `telemetry` is the backend `init_telemetry` resolved at startup:
    # "langfuse", "otel" or "off". Cloud Run's Google Frontend answers the
    # exact path `/healthz` itself, so a request there never reaches this
    # app; `/status/telemetry` serves the same one value from a path that does.
    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"service": SERVICE_NAME, "version": __version__, "telemetry": telemetry_backend()}

    @app.get("/status/telemetry")
    async def telemetry_status() -> dict[str, str]:
        return {"telemetry": telemetry_backend()}

    @app.post("/run_sse")
    async def run_sse() -> JSONResponse:
        return JSONResponse(status_code=410, content={"detail": _RUN_SSE_GONE})

    return app


def main() -> None:
    """Entry point for the `harness-server` console script."""
    settings = Settings()
    uvicorn.run(create_app(settings=settings), host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
