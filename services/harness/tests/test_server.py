from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import httpx2
import pytest
from analysis_mcp.fixtures import seed_fixture_store
from common import telemetry
from common.materialisation import get_store as get_analysis_mcp_store
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from harness import __version__
from harness import server as server_module
from harness.mcp_client import McpExecutor
from harness.server import create_app
from harness.session_store import build_session_service
from harness.settings import (
    ANALYSIS_MCP_MOUNT_PATH,
    DATA_MCP_MOUNT_PATH,
    PRESENTATION_MCP_MOUNT_PATH,
    Settings,
)
from harness.studio_translate import _RecordingSurface
from mcp import MCPError
from mcp.shared._httpx_utils import McpHttpClientFactory
from presentation_mcp import server as presentation_server
from presentation_mcp.theme import THEME_CSS, THEME_HASH


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate tests from any of these vars set in the ambient environment."""
    for var in (
        "DATA_MCP_URL",
        "ANALYSIS_MCP_URL",
        "PRESENTATION_MCP_URL",
        "DATABASE_URL",
        "SESSION_STORE_INSTANCE_CONNECTION_NAME",
        "SESSION_STORE_DATABASE",
        "SESSION_STORE_RUNTIME_DB_USER",
        "SESSION_STORE_POOL_SIZE",
        "SESSION_STORE_MAX_OVERFLOW",
        "SESSION_STORE_CONNECT_TIMEOUT_S",
        "PORT",
    ):
        monkeypatch.delenv(var, raising=False)


@asynccontextmanager
async def _running_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Drive `app`'s ASGI lifespan by hand for an in-process MCP handshake.

    `httpx2.ASGITransport` only forwards HTTP request/response events, not
    lifespan events -- but each mounted FastMCP app's
    `StreamableHTTPSessionManager` starts inside that app's own lifespan
    (composed into the harness app's lifespan by `combine_lifespans` in
    `harness.server.create_app`). Without running the lifespan, a request to
    a mounted MCP path fails with "session manager task group was not
    initialized". Real servers (uvicorn, Cloud Run) send these events
    automatically; this replays just enough of the ASGI lifespan protocol --
    including the `state` key Starlette's lifespan handling requires -- to
    exercise the same startup/shutdown path in a test.
    """
    receive_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    startup_complete = asyncio.Event()
    shutdown_complete = asyncio.Event()

    async def receive() -> MutableMapping[str, Any]:
        return await receive_queue.get()

    async def send(message: MutableMapping[str, Any]) -> None:
        if message["type"] == "lifespan.startup.complete":
            startup_complete.set()
        elif message["type"] == "lifespan.shutdown.complete":
            shutdown_complete.set()

    task = asyncio.create_task(app({"type": "lifespan", "state": {}}, receive, send))
    await receive_queue.put({"type": "lifespan.startup"})
    await asyncio.wait(
        {task, asyncio.ensure_future(startup_complete.wait())},
        return_when=asyncio.FIRST_COMPLETED,
    )
    if task.done():
        task.result()  # surface a startup failure instead of hanging forever
    try:
        yield
    finally:
        await receive_queue.put({"type": "lifespan.shutdown"})
        await asyncio.wait(
            {task, asyncio.ensure_future(shutdown_complete.wait())},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not task.done():
            await task


def _mounted_mcp_transport(
    app: FastAPI, mounted_path: str, *, statuses: list[int] | None = None
) -> StreamableHttpTransport:
    """Build a fastmcp `Client` transport that talks MCP to `mounted_path` on `app`.

    Routes through `httpx2.ASGITransport` so the handshake (initialize ->
    list_tools) exercises the real streamable-HTTP wire protocol against the
    composed app in-process, no real socket or network hop required.
    `statuses`, when given, collects the HTTP status of every response: mcp's
    client raises a non-2xx answer as `MCPError(INTERNAL_ERROR)` without its
    status, so a test reads a 401 off this list.
    """
    recorded = statuses if statuses is not None else []

    async def record(response: httpx2.Response) -> None:
        recorded.append(response.status_code)

    # fastmcp's `StreamableHttpTransport` also passes `follow_redirects=True` to
    # this factory at call time, which `McpHttpClientFactory` doesn't declare
    # (fastmcp's own factories hit the same mismatch -- see
    # `StreamableHttpTransport._make_verify_factory`) -- `**kwargs` absorbs it
    # and `cast` tells the type checker to trust the resulting shape.
    def factory(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url="http://testserver",
            event_hooks={"response": [record]},
            **kwargs,
        )

    return StreamableHttpTransport(
        url=f"http://testserver{mounted_path}",
        httpx_client_factory=cast(McpHttpClientFactory, factory),
    )


def test_create_app_builds_session_service_without_network() -> None:
    app = create_app(settings=Settings())

    assert isinstance(app.state.session_service, InMemorySessionService)


async def test_lifespan_shutdown_closes_the_studio_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server's studio executor is `McpExecutor`,
    which opens one long-lived presentation-mcp client per conversation and
    has nothing else on the served path to close it
    (`harness.mcp_client.McpExecutor.aclose`). `_lifespan`'s shutdown must
    route to it via `StudioRunner.aclose`."""
    closed: list[bool] = []
    original_aclose = McpExecutor.aclose

    async def tracking_aclose(self: McpExecutor) -> None:
        closed.append(True)
        await original_aclose(self)

    monkeypatch.setattr(McpExecutor, "aclose", tracking_aclose)
    app = create_app(settings=Settings())
    assert isinstance(app.state.studio.recorder._executor, McpExecutor)

    async with _running_lifespan(app):
        assert closed == []

    assert closed == [True]


async def test_the_app_starts_presentation_mcps_chart_worker_once_and_stops_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """presentation-mcp is served on its own mount and inside the combined
    server, and its lifespan, which owns the chart worker, runs once for both."""
    worker = presentation_server._chart_worker
    monkeypatch.setattr(worker, "_warm", True)
    app = create_app(settings=Settings())

    async with _running_lifespan(app):
        assert worker._warming is not None
        await worker._warming
        process = worker._process
        assert process is not None
        pid = process.pid
        assert pid is not None
        assert pid != os.getpid()

    assert worker._process is None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_the_console_script_module_leaves_the_harness_unimported() -> None:
    """A worker process spawned from `harness-server` runs the script's module
    again, so importing it must not import `harness.server`."""
    probe = "import sys, harness.cli; print('harness.server' in sys.modules)"

    ran = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert ran.stdout.strip() == "False"


def test_a_chart_worker_started_from_the_console_script_module_leaves_the_harness_unimported(
    tmp_path: Path,
) -> None:
    """`spawn` runs the parent's `__main__` file again in the worker, so the
    worker reports `harness.server` imported if `harness.cli` imports it. The
    script is a file, since a `-c` script has no `__main__` to run again."""
    script = tmp_path / "console_script.py"
    script.write_text(
        "import asyncio\n"
        "import logging\n"
        "import sys\n"
        "\n"
        "from harness.cli import main  # noqa: F401\n"
        "from presentation_mcp.chart_worker import ChartWorker\n"
        "\n"
        "\n"
        "async def run() -> None:\n"
        "    worker = ChartWorker(timeout_s=5.0, warm=True)\n"
        "    worker.start()\n"
        "    assert worker._warming is not None\n"
        "    await worker._warming\n"
        "    await worker.stop()\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    logging.basicConfig(level=logging.INFO, stream=sys.stdout)\n"
        "    asyncio.run(run())\n",
        encoding="utf-8",
    )

    ran = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=True, timeout=60
    )

    assert "harness.server not imported" in ran.stdout


def test_a_server_without_a_shared_secret_logs_that_auth_is_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `configure_logging` replaces the root handlers, so the warning is read
    # off the module logger rather than through caplog.
    warnings: list[str] = []
    monkeypatch.setattr(
        server_module.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    create_app(settings=Settings(studio_api_token=None))
    assert any("STUDIO AUTH IS OPEN" in w for w in warnings)


def test_a_server_with_a_shared_secret_does_not_log_open_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        server_module.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    create_app(settings=Settings(studio_api_token="s3cret"))
    assert not any("STUDIO AUTH IS OPEN" in w for w in warnings)


def test_the_execute_node_runs_against_the_recorded_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `execute` node must run the approved plan against the RECORDED
    surface (`_RecordingSurface`, wrapping `StudioRunner.surface_recorder`),
    not the bare `ToolSurface` `build_tool_surface` returns -- otherwise
    every plan step's `activity.step` frame goes missing on the served path
    and only compose's presentation calls show up. A test that only checked
    the constructor lambda's arity would pass while the feed stayed dark;
    this asserts on the actual object `create_root_workflow` receives.
    """
    captured: dict[str, Any] = {}
    real_create_root_workflow = server_module.create_root_workflow

    def _spy(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_create_root_workflow(**kwargs)

    monkeypatch.setattr(server_module, "create_root_workflow", _spy)

    app = create_app(settings=Settings())

    surface_recorder = app.state.studio.surface_recorder
    assert isinstance(surface_recorder, _RecordingSurface)
    assert captured["tool_surface"] is surface_recorder


def test_healthz_returns_service_version_and_telemetry_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telemetry, "_backend", None)
    client = TestClient(create_app(settings=Settings()))

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"service": "harness", "version": __version__, "telemetry": "off"}


@pytest.mark.parametrize("state", ["langfuse", "otel", "off"])
def test_telemetry_state_is_readable_from_a_route_that_reaches_the_app(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    """Cloud Run answers `/healthz` itself, so the resolved backend is also
    served from `/status/telemetry`, and that route carries only that value."""
    monkeypatch.setattr(telemetry, "_backend", None if state == "off" else state)
    client = TestClient(create_app(settings=Settings()))

    response = client.get("/status/telemetry")

    assert response.status_code == 200
    assert response.json() == {"telemetry": state}
    assert client.get("/healthz").json()["telemetry"] == state


async def test_healthz_nested_under_mounted_mcp_prefixes() -> None:
    """Each mounted FastMCP app keeps its own `/healthz` under its prefix,
    at a distinct path from the harness's top-level `/healthz` above. This
    route bypasses the MCP session manager entirely, so it works without
    driving the app's lifespan.
    """
    app = create_app(settings=Settings())
    transport = httpx2.ASGITransport(app=app)

    async with httpx2.AsyncClient(transport=transport, base_url="http://testserver") as client:
        data_response = await client.get(f"{DATA_MCP_MOUNT_PATH}/healthz")
        analysis_response = await client.get(f"{ANALYSIS_MCP_MOUNT_PATH}/healthz")
        presentation_response = await client.get(f"{PRESENTATION_MCP_MOUNT_PATH}/healthz")

    assert data_response.status_code == 200
    assert analysis_response.status_code == 200
    assert presentation_response.status_code == 200


_OUTSIDE_CLIENT = ("203.0.113.7", 443)  # TEST-NET-3: any address off loopback
_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "outside", "version": "1"},
    },
}


@pytest.mark.parametrize(
    "mount", [DATA_MCP_MOUNT_PATH, ANALYSIS_MCP_MOUNT_PATH, PRESENTATION_MCP_MOUNT_PATH]
)
async def test_a_direct_mount_refuses_an_outside_client_whatever_token_it_holds(
    mount: str,
) -> None:
    """Outside MCP clients use the combined server at `/mcp/`. A direct mount
    answers an outside request 403 before the bearer check, and keeps
    `/healthz` open."""
    app = create_app(settings=Settings(mcp_api_token="test-token"))
    headers = {
        "authorization": "Bearer test-token",
        "accept": "application/json, text/event-stream",
    }

    async with _running_lifespan(app):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app, client=_OUTSIDE_CLIENT),
            base_url="http://testserver",
        ) as outside:
            refused = await outside.post(f"{mount}/mcp", json=_INITIALIZE, headers=headers)
            health = await outside.get(f"{mount}/healthz")
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://testserver"
        ) as loopback:
            # The same route from loopback reaches the bearer check, and past it.
            unauthenticated = await loopback.post(f"{mount}/mcp", json=_INITIALIZE)
            served = await loopback.post(f"{mount}/mcp", json=_INITIALIZE, headers=headers)

    assert refused.status_code == 403
    assert "/mcp/" in refused.json()["detail"]
    assert health.status_code == 200
    assert unauthenticated.status_code == 401
    assert served.status_code == 200


async def test_the_combined_server_serves_an_outside_client() -> None:
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with (
        _running_lifespan(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app, client=_OUTSIDE_CLIENT),
            base_url="http://testserver",
        ) as outside,
    ):
        response = await outside.post(
            "/mcp/",
            json=_INITIALIZE,
            headers={
                "authorization": "Bearer test-token",
                "accept": "application/json, text/event-stream",
            },
        )

    assert response.status_code == 200


async def test_the_presentation_theme_stylesheet_is_served_to_an_outside_client_without_a_token() -> (
    None
):
    """A view's iframe loads its theme from `<mount>/theme-<hash>.css`, from
    the host's browser, and cannot send the bearer, so the path stays open to
    an outside client while the MCP route beside it refuses one."""
    app = create_app(settings=Settings(mcp_api_token="test-token"))
    theme_path = f"{PRESENTATION_MCP_MOUNT_PATH}/theme-{THEME_HASH}.css"

    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app, client=_OUTSIDE_CLIENT),
        base_url="http://testserver",
    ) as outside:
        theme = await outside.get(theme_path)
        refused = await outside.post(f"{PRESENTATION_MCP_MOUNT_PATH}/mcp", json={})

    assert refused.status_code == 403
    assert theme.status_code == 200
    assert theme.text == THEME_CSS
    assert theme.headers["content-type"] == "text/css; charset=utf-8"


async def test_mounted_data_mcp_reports_its_tools() -> None:
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with _running_lifespan(app):
        transport = _mounted_mcp_transport(app, f"{DATA_MCP_MOUNT_PATH}/mcp")
        async with Client(transport, auth="test-token") as client:
            tools = await client.list_tools()

    assert [tool.name for tool in tools] == [
        "search",
        "get_data",
        "describe",
        "get_skill",
        "get_skill_file",
        "sources",
        "resolve_definition",
        "catalogue_counts",
    ]


async def test_a_held_data_mcp_client_authenticates_every_call_not_just_the_first() -> None:
    """Regression for a suspected failure mode of `harness.mcp_client`'s
    held-per-conversation clients: a `Client` entered once and reused for
    several `call_tool`s must attach its bearer token to every one, not
    just whichever request happened while it was auth'd during `__aenter__`.
    Drives the real `RequireAuthMiddleware` gate three calls deep on one
    entered session -- a 401 on any call but the first would raise
    `MCPError` and fail this test, the same way the mount
    already refuses an unauthenticated request in the test above."""
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with _running_lifespan(app):
        transport = _mounted_mcp_transport(app, f"{DATA_MCP_MOUNT_PATH}/mcp")
        async with Client(transport, auth="test-token") as client:
            for i in range(3):
                # `search`'s own upstream call has nothing to reach in this
                # test environment and refuses -- that refusal is a normal,
                # successful MCP result (`is_error=True`); only a 401 would
                # raise, and it would raise before this line returns.
                result = await client.call_tool(
                    "search",
                    {"parts": [{"variable": f"x{i}"}], "raw_query": "x"},
                    raise_on_error=False,
                )
                assert result is not None


async def test_mounted_data_mcp_refuses_an_unauthenticated_request() -> None:
    """The mount itself, not just the harness's own client, rejects a caller
    with no token, since the gate is enforced at the FastMCP layer instead
    of only by threading `auth=` through `harness.mcp_client`."""
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with _running_lifespan(app):
        statuses: list[int] = []
        transport = _mounted_mcp_transport(app, f"{DATA_MCP_MOUNT_PATH}/mcp", statuses=statuses)
        with pytest.raises(MCPError):
            async with Client(transport) as client:
                await client.list_tools()

    assert set(statuses) == {401}


async def test_mounted_analysis_mcp_reports_its_tools() -> None:
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with _running_lifespan(app):
        transport = _mounted_mcp_transport(app, f"{ANALYSIS_MCP_MOUNT_PATH}/mcp")
        async with Client(transport, auth="test-token") as client:
            tools = await client.list_tools()

    assert {tool.name for tool in tools} == {
        "aggregate",
        "compare",
        "growth",
        "rank",
        "share_of",
        "quotient_of",
        "join",
        "filter",
        "sum_of",
        "resolve_anchor",
    }


async def test_mounted_analysis_mcp_refuses_an_unauthenticated_request() -> None:
    """The mount itself, not just the harness's own client, rejects a caller
    with no token, since the gate is enforced at the FastMCP layer instead
    of only by threading `auth=` through `harness.mcp_client`."""
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with _running_lifespan(app):
        statuses: list[int] = []
        transport = _mounted_mcp_transport(app, f"{ANALYSIS_MCP_MOUNT_PATH}/mcp", statuses=statuses)
        with pytest.raises(MCPError):
            async with Client(transport) as client:
                await client.list_tools()

    assert set(statuses) == {401}


async def test_mounted_analysis_mcp_aggregate_returns_a_real_digest_through_the_wire() -> None:
    """`test_mounted_analysis_mcp_reports_its_tools` above proves the mount
    lists seven tools, but never calls one -- every verb's own correctness
    still rests entirely on analysis-mcp's own test suite, and the composed
    application (this harness's own ASGI mount, routed through the real
    streamable-HTTP handshake) is never itself exercised. This drives the
    same in-process handshake, then actually calls `aggregate` through it
    and reads the real digest back -- the analysis-mcp process-wide store
    (`common.materialisation`) is seeded only after the mounted app's own
    lifespan has built it, the same store the mounted tool call resolves
    `frame_ref` against, since analysis-mcp's own `_lifespan` overwrites
    whatever store existed before `_running_lifespan` drives startup."""
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with _running_lifespan(app):
        handles = await seed_fixture_store(get_analysis_mcp_store())
        transport = _mounted_mcp_transport(app, f"{ANALYSIS_MCP_MOUNT_PATH}/mcp")
        async with Client(transport, auth="test-token") as client:
            result = await client.call_tool(
                "aggregate",
                {
                    "frame_ref": handles.plain_single_unit,
                    "aggregation": "sum",
                    "group_by": ["reporter"],
                },
            )

    assert result.is_error is False
    digest = result.content[0].text
    # Every number is engine-computed, from `plain_single_unit_frame`'s own
    # fixture data (see `analysis_mcp.fixtures`'s `_ODA_DISBURSED_BY_DONOR_YEAR`).
    assert "reporter=dcid:USA" in digest
    assert "166683000000" in digest  # USA: 47_805 + 55_303 + 63_575 (billions)
    assert "unit: dcid:USDollar" in digest


async def test_mounted_presentation_mcp_reports_zero_tools_when_no_platform_is_reachable() -> None:
    """presentation-mcp's own lifespan fetches apis' `/api/presentation/release`
    at startup and mounts its tools from that document; with no platform
    reachable in this in-process test, it comes up tool-less and degraded
    instead (see `presentation_mcp.server`'s tests, which mock that fetch and
    pin the three-tools-mounted case).
    """
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with _running_lifespan(app):
        transport = _mounted_mcp_transport(app, f"{PRESENTATION_MCP_MOUNT_PATH}/mcp")
        async with Client(transport, auth="test-token") as client:
            tools = await client.list_tools()

    assert tools == []


async def test_mounted_presentation_mcp_refuses_an_unauthenticated_request() -> None:
    """The mount itself, not just the harness's own client, rejects a caller
    with no token, since the gate is enforced at the FastMCP layer instead
    of only by threading `auth=` through `harness.mcp_client`."""
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with _running_lifespan(app):
        statuses: list[int] = []
        transport = _mounted_mcp_transport(
            app, f"{PRESENTATION_MCP_MOUNT_PATH}/mcp", statuses=statuses
        )
        with pytest.raises(MCPError):
            async with Client(transport) as client:
                await client.list_tools()

    assert set(statuses) == {401}


async def test_startup_fails_loudly_when_its_own_token_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The startup self-check sends a token the mounts refuse, so startup
    fails with the message naming `MCP_API_TOKEN` rather than a bare
    `MCPError` on a user's first turn."""
    monkeypatch.setattr(server_module, "bearer_token", lambda: "not-the-token")
    app = create_app(settings=Settings(mcp_api_token="test-token"))

    # Each mounted app's session manager wraps the failure in its own task
    # group on the way out.
    with pytest.RaisesGroup(
        pytest.RaisesExc(RuntimeError, match="could not authenticate"), flatten_subgroups=True
    ):
        async with _running_lifespan(app):
            pass


async def test_mcp_auth_disabled_lets_an_unauthenticated_request_through() -> None:
    app = create_app(settings=Settings(mcp_auth_disabled=True))

    async with _running_lifespan(app):
        transport = _mounted_mcp_transport(app, f"{DATA_MCP_MOUNT_PATH}/mcp")
        async with Client(transport) as client:
            tools = await client.list_tools()

    assert [tool.name for tool in tools] == [
        "search",
        "get_data",
        "describe",
        "get_skill",
        "get_skill_file",
        "sources",
        "resolve_definition",
        "catalogue_counts",
    ]


async def test_unset_mcp_api_token_mints_one_the_harness_can_call_itself_with() -> None:
    """Default settings (`MCP_API_TOKEN` and `MCP_AUTH_DISABLED` both unset).
    A per-process token is minted, and `harness.server._lifespan`'s own
    startup self-check (a real `list_tools` call against the data-mcp
    mount, over the same ASGI/auth path a live request would take) proves
    the harness can authenticate to itself with it. A misconfigured token
    would fail `_running_lifespan`'s own `task.result()` re-raise instead of
    reaching the end of this `async with` block.
    """
    app = create_app(settings=Settings())

    async with _running_lifespan(app):
        pass


def test_run_sse_is_retired_in_favour_of_the_studio_routes() -> None:
    client = TestClient(create_app(settings=Settings()))

    response = client.post("/run_sse")

    assert response.status_code == 410
    assert "/api/studio/turn" in response.json()["detail"]


def test_studio_routes_are_mounted() -> None:
    """The studio surface is part of the served app: an unauthenticated turn
    against `create_app` hits the auth seam and gets a 401."""
    client = TestClient(create_app(settings=Settings()))

    response = client.post(
        "/api/studio/turn",
        json={
            "kind": "turn.request",
            "schema_version": 1,
            "conversation_id": "conv_mount_check",
            "message": "hi",
        },
    )

    assert response.status_code == 401


def test_build_session_service_defaults_to_in_memory() -> None:
    service = build_session_service(settings=Settings()).service

    assert isinstance(service, InMemorySessionService)


async def test_build_session_service_database_url_yields_a_working_database_session_service(
    tmp_path,
) -> None:
    """`DATABASE_URL=sqlite+aiosqlite://...` yields a real
    `DatabaseSessionService`: a session created through it round-trips
    through `get_session`, and `close` disposes the engine it built for
    itself without raising."""
    settings = Settings(database_url=f"sqlite+aiosqlite:///{tmp_path}/sessions.db")

    service = build_session_service(settings=settings).service

    assert isinstance(service, DatabaseSessionService)
    await service.create_session(app_name="harness", user_id="user-1", session_id="conv-1")
    reloaded = await service.get_session(app_name="harness", user_id="user-1", session_id="conv-1")
    assert reloaded is not None
    assert reloaded.id == "conv-1"
    await service.close()


async def test_lifespan_prepares_database_tables_eagerly_and_disposes_on_shutdown(
    monkeypatch, tmp_path
) -> None:
    """DatabaseSessionService over sqlite+aiosqlite: the lifespan's startup
    phase calls `prepare_tables` before the app accepts traffic (so a
    session created right after startup needs no lazy table creation of its
    own), and its shutdown phase disposes the engine `build_session_service`
    built for it (`harness.session_store.SessionServiceHandle`)."""
    prepared: list[bool] = []
    original_prepare_tables = DatabaseSessionService.prepare_tables

    async def tracking_prepare_tables(self: DatabaseSessionService) -> None:
        prepared.append(True)
        await original_prepare_tables(self)

    monkeypatch.setattr(DatabaseSessionService, "prepare_tables", tracking_prepare_tables)

    app = create_app(settings=Settings(database_url=f"sqlite+aiosqlite:///{tmp_path}/sessions.db"))
    assert isinstance(app.state.session_service, DatabaseSessionService)

    closed: list[bool] = []
    original_aclose = app.state.session_service_aclose

    async def tracking_aclose() -> None:
        closed.append(True)
        await original_aclose()

    monkeypatch.setattr(app.state, "session_service_aclose", tracking_aclose)

    async with _running_lifespan(app):
        assert prepared == [True]
        assert closed == []
        session = await app.state.session_service.create_session(
            app_name="harness", user_id="user-1", session_id="conv-1"
        )
        assert session.id == "conv-1"

    assert closed == [True]


async def test_lifespan_startup_prepare_tables_honors_the_configured_connect_timeout(
    monkeypatch, tmp_path
) -> None:
    """`_lifespan`'s eager `prepare_tables` call is bounded by
    `session_store_connect_timeout_s`, not a hardcoded deadline: a setting
    far below how long the call actually takes fails startup. The retry
    budget is zeroed so the first timeout exhausts it immediately -- the
    retry behaviour itself is covered directly, against
    `_prepare_tables_with_retries`, below."""
    monkeypatch.setattr(server_module, "_PREPARE_TABLES_RETRY_BUDGET_S", 0.0)
    original_prepare_tables = DatabaseSessionService.prepare_tables

    async def slow_prepare_tables(self: DatabaseSessionService) -> None:
        await asyncio.sleep(0.2)
        await original_prepare_tables(self)

    monkeypatch.setattr(DatabaseSessionService, "prepare_tables", slow_prepare_tables)

    app = create_app(
        settings=Settings(
            database_url=f"sqlite+aiosqlite:///{tmp_path}/sessions.db",
            session_store_connect_timeout_s=0.01,
        )
    )
    assert isinstance(app.state.session_service, DatabaseSessionService)

    # The mounted MCP apps' own lifespans wrap this in nested exception
    # groups (`fastmcp`'s `combine_lifespans`); the `TimeoutError` itself is
    # in there somewhere.
    with pytest.raises(BaseException) as excinfo:
        async with _running_lifespan(app):
            pass
    assert excinfo.group_contains(TimeoutError)


def _fake_clock_and_sleep(
    start: float = 0.0,
) -> tuple[Callable[[], float], Callable[[float], Awaitable[None]]]:
    """A mutable fake clock plus a `sleep` that advances it by the requested
    duration instead of actually waiting -- `_prepare_tables_with_retries`'s
    deadline arithmetic runs against this pair, so a test exercises the
    real budget/backoff logic without any real delay."""
    clock = [start]

    def now() -> float:
        return clock[0]

    async def sleep(seconds: float) -> None:
        clock[0] += seconds

    return now, sleep


async def test_prepare_tables_with_retries_succeeds_after_transient_failures() -> None:
    """A `prepare_tables` that fails twice then succeeds returns normally,
    having paused between attempts through the injected `sleep`/`now` seam
    -- no real delay, so the test stays fast."""
    attempts: list[int] = []

    class _FlakyService:
        async def prepare_tables(self) -> None:
            attempts.append(len(attempts) + 1)
            if len(attempts) < 3:
                raise RuntimeError("transient connector failure")

    now, sleep = _fake_clock_and_sleep()

    await server_module._prepare_tables_with_retries(
        cast(DatabaseSessionService, _FlakyService()),
        connect_timeout_s=5.0,
        budget_s=60.0,
        sleep=sleep,
        now=now,
    )

    assert attempts == [1, 2, 3]


async def test_prepare_tables_with_retries_raises_after_the_budget_is_exhausted() -> None:
    """A `prepare_tables` that always fails re-raises the last error once the
    retry budget runs out, rather than retrying forever."""
    attempts: list[int] = []

    class _AlwaysFailingService:
        async def prepare_tables(self) -> None:
            attempts.append(len(attempts) + 1)
            raise RuntimeError(f"connector failure {len(attempts)}")

    now, sleep = _fake_clock_and_sleep()

    with pytest.raises(RuntimeError, match="connector failure"):
        await server_module._prepare_tables_with_retries(
            cast(DatabaseSessionService, _AlwaysFailingService()),
            connect_timeout_s=5.0,
            budget_s=3.0,
            sleep=sleep,
            now=now,
        )

    # backoff sequence: 1s, 2s -- a third 4s pause would exceed the 3s
    # budget, so the third attempt's failure is the one that re-raises.
    assert attempts == [1, 2, 3]


async def _preflight(app: FastAPI, origin: str) -> httpx2.Response:
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://cors.test"
    ) as client:
        return await client.options(
            "/api/studio/turn",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )


async def test_cors_origins_setting_admits_listed_origins_only() -> None:
    """The server answers browser preflights for the origins in
    STUDIO_CORS_ORIGINS and for no other; with the list empty (the deployed
    posture) it sends no CORS headers at all."""
    listed = "http://localhost:8091"
    app = create_app(settings=Settings(studio_cors_origins=[listed]))
    allowed = await _preflight(app, listed)
    assert allowed.status_code == 200
    assert allowed.headers.get("access-control-allow-origin") == listed
    refused = await _preflight(app, "http://evil.example")
    assert "access-control-allow-origin" not in refused.headers

    closed = await _preflight(create_app(settings=Settings()), listed)
    assert "access-control-allow-origin" not in closed.headers


def test_cors_origins_setting_parses_a_comma_separated_env_value() -> None:
    """The env var is a comma-separated list, the shape a shell can set."""
    settings = Settings(studio_cors_origins="http://localhost:8091, http://localhost:3000")  # type: ignore[arg-type]
    assert settings.studio_cors_origins == ["http://localhost:8091", "http://localhost:3000"]
    assert Settings(studio_cors_origins="").studio_cors_origins == []  # type: ignore[arg-type]
