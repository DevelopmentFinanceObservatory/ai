from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import asyncpg
import httpx2
import pytest
from analysis_mcp.fixtures import seed_fixture_store
from common import telemetry
from common.materialisation import get_store as get_analysis_mcp_store
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from harness import __version__
from harness import server as server_module
from harness.mcp_client import McpExecutor
from harness.server import create_app
from harness.settings import (
    ANALYSIS_MCP_MOUNT_PATH,
    DATA_MCP_MOUNT_PATH,
    PRESENTATION_MCP_MOUNT_PATH,
    Settings,
)
from harness.studio_stream import SHUTDOWN_WAIT_S
from harness.studio_translate import _RecordingSurface
from harness.wire import TURN_INDEX_HEADER
from mcp import MCPError
from mcp.shared._httpx_utils import McpHttpClientFactory
from presentation_mcp import server as presentation_server
from presentation_mcp.theme import THEME_CSS, THEME_HASH
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


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


@pytest.mark.parametrize("database_url", [None, "sqlite+aiosqlite:///:memory:"])
def test_create_app_refuses_a_missing_or_non_postgres_database(database_url: str | None) -> None:
    """The studio stores conversations in Postgres, so an app with no
    database, or one that is not Postgres, is refused at construction."""
    with pytest.raises(RuntimeError, match="Postgres"):
        create_app(settings=Settings(database_url=database_url))


async def test_lifespan_shutdown_closes_the_studio_executor(
    monkeypatch: pytest.MonkeyPatch,
    pg_database_url: str,
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
    app = create_app(settings=Settings(database_url=pg_database_url))
    assert isinstance(app.state.studio.recorder._executor, McpExecutor)

    async with _running_lifespan(app):
        assert closed == []

    assert closed == [True]


async def test_lifespan_shutdown_waits_for_the_stream_owners_before_it_closes_the_database(
    pg_database_url: str,
) -> None:
    """A Studio stream's owner still making its closing write at shutdown
    finishes before `_lifespan` closes the database it writes to
    (`harness.studio_stream.LiveOwners`). The early mints are cancelled
    first, which ends an owner's wait for its stream's mints, so the wait
    for the owners is spent on closing writes."""
    app = create_app(settings=Settings(database_url=pg_database_url))
    order: list[str] = []
    close_database = app.state.database_aclose
    cancelled = asyncio.Event()

    class Mints:
        async def aclose(self) -> None:
            order.append("mints cancelled")
            cancelled.set()

    async def recorded_close() -> None:
        order.append("database closed")
        await close_database()

    async def settling() -> None:
        # The closing write, then the wait for the stream's mints, which
        # gives up after a while.
        await asyncio.sleep(0.05)
        await asyncio.wait({asyncio.ensure_future(cancelled.wait())}, timeout=0.5)
        order.append("owner finished")

    app.state.database_aclose = recorded_close
    async with _running_lifespan(app):
        app.state.early_mints = Mints()
        app.state.studio.owners.start(settling())

    assert order == ["mints cancelled", "owner finished", "database closed"]


async def test_the_app_starts_presentation_mcps_chart_worker_once_and_stops_it(
    monkeypatch: pytest.MonkeyPatch,
    pg_database_url: str,
) -> None:
    """presentation-mcp is served on its own mount and inside the combined
    server, and its lifespan, which owns the chart worker, runs once for both."""
    worker = presentation_server._chart_worker
    monkeypatch.setattr(worker, "_warm", True)
    app = create_app(settings=Settings(database_url=pg_database_url))

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
    pg_database_url: str,
) -> None:
    # `configure_logging` replaces the root handlers, so the warning is read
    # off the module logger rather than through caplog.
    warnings: list[str] = []
    monkeypatch.setattr(
        server_module.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    create_app(settings=Settings(database_url=pg_database_url, studio_api_token=None))
    assert any("STUDIO AUTH IS OPEN" in w for w in warnings)


def test_a_server_with_a_shared_secret_does_not_log_open_auth(
    monkeypatch: pytest.MonkeyPatch,
    pg_database_url: str,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        server_module.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    create_app(settings=Settings(database_url=pg_database_url, studio_api_token="s3cret"))
    assert not any("STUDIO AUTH IS OPEN" in w for w in warnings)


@pytest.mark.parametrize("on", [True, False], ids=["on", "off"])
def test_the_server_says_at_startup_whether_early_mints_are_on(
    monkeypatch: pytest.MonkeyPatch,
    on: bool,
    pg_database_url: str,
) -> None:
    """`EARLY_MINT` decides whether the studio gets early mints at all, and the
    server logs which way it went."""
    infos: list[str] = []
    monkeypatch.setattr(server_module.logger, "info", lambda msg, *args: infos.append(msg % args))

    app = create_app(settings=Settings(database_url=pg_database_url, early_mint=on))

    assert (app.state.studio.early_mints is not None) is on
    assert any(line.startswith(f"early mints: {'on' if on else 'off'}") for line in infos)


def test_the_execute_node_runs_against_the_recorded_surface(
    monkeypatch: pytest.MonkeyPatch,
    pg_database_url: str,
) -> None:
    """The `execute` node must run the approved plan against the RECORDED
    surface (`_RecordingSurface`, wrapping `StudioRunner.surface_recorder`),
    not the bare `ToolSurface` `build_tool_surface` returns -- otherwise
    every plan step's `activity.step` frame goes missing on the served path
    and only compose's presentation calls show up. A test that only checked
    the constructor lambda's arity would pass while the feed stayed dark;
    this asserts on the actual object `studio_graph` receives.
    """
    captured: dict[str, Any] = {}
    real_studio_graph = server_module.studio_graph

    def _spy(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_studio_graph(**kwargs)

    monkeypatch.setattr(server_module, "studio_graph", _spy)

    app = create_app(settings=Settings(database_url=pg_database_url))

    surface_recorder = app.state.studio.surface_recorder
    assert isinstance(surface_recorder, _RecordingSurface)
    assert captured["tool_surface"] is surface_recorder


def test_healthz_returns_service_version_and_telemetry_state(
    monkeypatch: pytest.MonkeyPatch,
    pg_database_url: str,
) -> None:
    monkeypatch.setattr(telemetry, "_backend", None)
    client = TestClient(create_app(settings=Settings(database_url=pg_database_url)))

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"service": "harness", "version": __version__, "telemetry": "off"}


@pytest.mark.parametrize("state", ["langfuse", "otel", "off"])
def test_telemetry_state_is_readable_from_a_route_that_reaches_the_app(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    pg_database_url: str,
) -> None:
    """Cloud Run answers `/healthz` itself, so the resolved backend is also
    served from `/status/telemetry`, and that route carries only that value."""
    monkeypatch.setattr(telemetry, "_backend", None if state == "off" else state)
    client = TestClient(create_app(settings=Settings(database_url=pg_database_url)))

    response = client.get("/status/telemetry")

    assert response.status_code == 200
    assert response.json() == {"telemetry": state}
    assert client.get("/healthz").json()["telemetry"] == state


async def test_healthz_nested_under_mounted_mcp_prefixes(pg_database_url: str) -> None:
    """Each mounted FastMCP app keeps its own `/healthz` under its prefix,
    at a distinct path from the harness's top-level `/healthz` above. This
    route bypasses the MCP session manager entirely, so it works without
    driving the app's lifespan.
    """
    app = create_app(settings=Settings(database_url=pg_database_url))
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
    pg_database_url: str,
) -> None:
    """Outside MCP clients use the combined server at `/mcp/`. A direct mount
    answers an outside request 403 before the bearer check, and keeps
    `/healthz` open."""
    app = create_app(settings=Settings(database_url=pg_database_url, mcp_api_token="test-token"))
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


async def test_the_combined_server_serves_an_outside_client(pg_database_url: str) -> None:
    app = create_app(settings=Settings(database_url=pg_database_url, mcp_api_token="test-token"))

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


async def test_the_presentation_theme_stylesheet_is_served_to_an_outside_client_without_a_token(
    pg_database_url: str,
) -> None:
    """A view's iframe loads its theme from `<mount>/theme-<hash>.css`, from
    the host's browser, and cannot send the bearer, so the path stays open to
    an outside client while the MCP route beside it refuses one."""
    app = create_app(settings=Settings(database_url=pg_database_url, mcp_api_token="test-token"))
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


async def test_mounted_data_mcp_reports_its_tools(pg_database_url: str) -> None:
    app = create_app(settings=Settings(database_url=pg_database_url, mcp_api_token="test-token"))

    async with _running_lifespan(app):
        transport = _mounted_mcp_transport(app, f"{DATA_MCP_MOUNT_PATH}/mcp")
        async with Client(transport, auth="test-token") as client:
            tools = await client.list_tools()

    assert [tool.name for tool in tools] == [
        "search",
        "get_data",
        "describe",
        "get_skill",
        "sources",
        "export_data",
        "resolve_definition",
        "catalogue_counts",
    ]


async def test_a_held_data_mcp_client_authenticates_every_call_not_just_the_first(
    pg_database_url: str,
) -> None:
    """Regression for a suspected failure mode of `harness.mcp_client`'s
    held-per-conversation clients: a `Client` entered once and reused for
    several `call_tool`s must attach its bearer token to every one, not
    just whichever request happened while it was auth'd during `__aenter__`.
    Drives the real `RequireAuthMiddleware` gate three calls deep on one
    entered session -- a 401 on any call but the first would raise
    `MCPError` and fail this test, the same way the mount
    already refuses an unauthenticated request in the test above."""
    app = create_app(settings=Settings(database_url=pg_database_url, mcp_api_token="test-token"))

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


async def test_mounted_data_mcp_refuses_an_unauthenticated_request(pg_database_url: str) -> None:
    """The mount itself, not just the harness's own client, rejects a caller
    with no token, since the gate is enforced at the FastMCP layer instead
    of only by threading `auth=` through `harness.mcp_client`."""
    app = create_app(settings=Settings(database_url=pg_database_url, mcp_api_token="test-token"))

    async with _running_lifespan(app):
        statuses: list[int] = []
        transport = _mounted_mcp_transport(app, f"{DATA_MCP_MOUNT_PATH}/mcp", statuses=statuses)
        with pytest.raises(MCPError):
            async with Client(transport) as client:
                await client.list_tools()

    assert set(statuses) == {401}


async def test_mounted_analysis_mcp_reports_its_tools(pg_database_url: str) -> None:
    app = create_app(settings=Settings(database_url=pg_database_url, mcp_api_token="test-token"))

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


async def test_mounted_analysis_mcp_refuses_an_unauthenticated_request(
    pg_database_url: str,
) -> None:
    """The mount itself, not just the harness's own client, rejects a caller
    with no token, since the gate is enforced at the FastMCP layer instead
    of only by threading `auth=` through `harness.mcp_client`."""
    app = create_app(settings=Settings(database_url=pg_database_url, mcp_api_token="test-token"))

    async with _running_lifespan(app):
        statuses: list[int] = []
        transport = _mounted_mcp_transport(app, f"{ANALYSIS_MCP_MOUNT_PATH}/mcp", statuses=statuses)
        with pytest.raises(MCPError):
            async with Client(transport) as client:
                await client.list_tools()

    assert set(statuses) == {401}


async def test_mounted_analysis_mcp_aggregate_returns_a_real_digest_through_the_wire(
    pg_database_url: str,
) -> None:
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
    app = create_app(settings=Settings(database_url=pg_database_url, mcp_api_token="test-token"))

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


async def test_mounted_presentation_mcp_reports_zero_tools_when_no_platform_is_reachable(
    pg_database_url: str,
) -> None:
    """presentation-mcp's own lifespan fetches apis' `/api/presentation/release`
    at startup and mounts its tools from that document; with no platform
    reachable in this in-process test, it comes up tool-less and degraded
    instead (see `presentation_mcp.server`'s tests, which mock that fetch and
    pin the three-tools-mounted case).
    """
    app = create_app(settings=Settings(database_url=pg_database_url, mcp_api_token="test-token"))

    async with _running_lifespan(app):
        transport = _mounted_mcp_transport(app, f"{PRESENTATION_MCP_MOUNT_PATH}/mcp")
        async with Client(transport, auth="test-token") as client:
            tools = await client.list_tools()

    assert tools == []


async def test_mounted_presentation_mcp_refuses_an_unauthenticated_request(
    pg_database_url: str,
) -> None:
    """The mount itself, not just the harness's own client, rejects a caller
    with no token, since the gate is enforced at the FastMCP layer instead
    of only by threading `auth=` through `harness.mcp_client`."""
    app = create_app(settings=Settings(database_url=pg_database_url, mcp_api_token="test-token"))

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
    pg_database_url: str,
) -> None:
    """The startup self-check sends a token the mounts refuse, so startup
    fails with the message naming `MCP_API_TOKEN` rather than a bare
    `MCPError` on a user's first turn."""
    monkeypatch.setattr(server_module, "bearer_token", lambda: "not-the-token")
    app = create_app(settings=Settings(database_url=pg_database_url, mcp_api_token="test-token"))

    # Each mounted app's session manager wraps the failure in its own task
    # group on the way out.
    with pytest.RaisesGroup(
        pytest.RaisesExc(RuntimeError, match="could not authenticate"), flatten_subgroups=True
    ):
        async with _running_lifespan(app):
            pass


async def test_mcp_auth_disabled_lets_an_unauthenticated_request_through(
    pg_database_url: str,
) -> None:
    app = create_app(settings=Settings(database_url=pg_database_url, mcp_auth_disabled=True))

    async with _running_lifespan(app):
        transport = _mounted_mcp_transport(app, f"{DATA_MCP_MOUNT_PATH}/mcp")
        async with Client(transport) as client:
            tools = await client.list_tools()

    assert [tool.name for tool in tools] == [
        "search",
        "get_data",
        "describe",
        "get_skill",
        "sources",
        "export_data",
        "resolve_definition",
        "catalogue_counts",
    ]


async def test_unset_mcp_api_token_mints_one_the_harness_can_call_itself_with(
    pg_database_url: str,
) -> None:
    """Default settings (`MCP_API_TOKEN` and `MCP_AUTH_DISABLED` both unset).
    A per-process token is minted, and `harness.server._lifespan`'s own
    startup self-check (a real `list_tools` call against the data-mcp
    mount, over the same ASGI/auth path a live request would take) proves
    the harness can authenticate to itself with it. A misconfigured token
    would fail `_running_lifespan`'s own `task.result()` re-raise instead of
    reaching the end of this `async with` block.
    """
    app = create_app(settings=Settings(database_url=pg_database_url))

    async with _running_lifespan(app):
        pass


def test_run_sse_is_retired_in_favour_of_the_studio_routes(pg_database_url: str) -> None:
    client = TestClient(create_app(settings=Settings(database_url=pg_database_url)))

    response = client.post("/run_sse")

    assert response.status_code == 410
    assert "/api/studio/turn" in response.json()["detail"]


def test_studio_routes_are_mounted(pg_database_url: str) -> None:
    """The studio surface is part of the served app: an unauthenticated turn
    against `create_app` hits the auth check and gets a 401."""
    client = TestClient(create_app(settings=Settings(database_url=pg_database_url)))

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


async def test_lifespan_disposes_the_database_on_shutdown(
    monkeypatch, pg_database_url: str
) -> None:
    """The lifespan's shutdown phase disposes the engine `build_database`
    built (`harness.db.engine.DatabaseHandle`), and not before."""
    app = create_app(settings=Settings(database_url=pg_database_url))

    closed: list[bool] = []
    original_aclose = app.state.database_aclose

    async def tracking_aclose() -> None:
        closed.append(True)
        await original_aclose()

    monkeypatch.setattr(app.state, "database_aclose", tracking_aclose)

    async with _running_lifespan(app):
        assert closed == []

    assert closed == [True]


async def test_lifespan_applies_the_harness_migrations_on_postgres(
    pg_empty_engine: AsyncEngine,
) -> None:
    """Startup on a Postgres `DATABASE_URL` creates the harness's tables. The
    test database is a throwaway one, since the migrations
    write to whichever schema the connection's `search_path` names."""
    base_url = pg_empty_engine.url
    name = f"t_{uuid.uuid4().hex[:12]}"
    admin = create_async_engine(base_url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f"CREATE DATABASE {name}"))
    url = base_url.set(database=name)
    try:
        app = create_app(settings=Settings(database_url=url.render_as_string(hide_password=False)))
        async with _running_lifespan(app):
            engine = app.state.database_engine
            async with engine.connect() as conn:
                tables = set(
                    (
                        await conn.execute(
                            text(
                                "SELECT table_name FROM information_schema.tables "
                                "WHERE table_schema = 'public'"
                            )
                        )
                    ).scalars()
                )
        assert {"conversations", "runs", "harness_schema_migrations"} <= tables
    finally:
        async with admin.connect() as conn:
            await conn.execute(text(f"DROP DATABASE {name} WITH (FORCE)"))
        await admin.dispose()


def _fake_clock_and_sleep(
    start: float = 0.0,
) -> tuple[Callable[[], float], Callable[[float], Awaitable[None]]]:
    """A mutable fake clock plus a `sleep` that advances it by the requested
    duration instead of actually waiting -- `_retry_startup_step`'s deadline
    arithmetic runs against this pair, so a test exercises the
    real budget/backoff logic without any real delay."""
    clock = [start]

    def now() -> float:
        return clock[0]

    async def sleep(seconds: float) -> None:
        clock[0] += seconds

    return now, sleep


async def test_a_startup_step_succeeds_after_transient_failures() -> None:
    """A startup step that fails twice then succeeds returns normally,
    having paused between attempts through the injected `sleep` and `now`
    -- no real delay, so the test stays fast."""
    attempts: list[int] = []

    async def flaky_step() -> None:
        attempts.append(len(attempts) + 1)
        if len(attempts) < 3:
            raise RuntimeError("transient connector failure")

    now, sleep = _fake_clock_and_sleep()

    await server_module._retry_startup_step(
        "flaky_step",
        flaky_step,
        budget_s=60.0,
        sleep=sleep,
        now=now,
    )

    assert attempts == [1, 2, 3]


async def test_a_startup_step_raises_after_the_budget_is_exhausted() -> None:
    """A startup step that always fails re-raises the last error once the
    retry budget runs out, rather than retrying forever."""
    attempts: list[int] = []

    async def failing_step() -> None:
        attempts.append(len(attempts) + 1)
        raise RuntimeError(f"connector failure {len(attempts)}")

    now, sleep = _fake_clock_and_sleep()

    with pytest.raises(RuntimeError, match="connector failure"):
        await server_module._retry_startup_step(
            "failing_step",
            failing_step,
            budget_s=3.0,
            sleep=sleep,
            now=now,
        )

    # backoff sequence: 1s, 2s -- a third 4s pause would exceed the 3s
    # budget, so the third attempt's failure is the one that re-raises.
    assert attempts == [1, 2, 3]


async def test_a_slow_migration_is_not_cut_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An attempt has no bound of its own, since `apply_pending` bounds each
    statement, so a slow migration completes on its first attempt."""
    attempts: list[int] = []

    async def slow_apply_pending(engine: object) -> list[str]:
        attempts.append(len(attempts) + 1)
        await asyncio.sleep(0.05)
        return []

    async def no_adk_tables(engine: object) -> list[str]:
        return []

    monkeypatch.setattr(server_module, "apply_pending", slow_apply_pending)
    monkeypatch.setattr(server_module, "live_adk_tables", no_adk_tables)

    await server_module._apply_migrations_with_retries(cast(AsyncEngine, object()), budget_s=0.0)

    assert attempts == [1]


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


async def test_cors_origins_setting_admits_listed_origins_only(pg_database_url: str) -> None:
    """The server answers browser preflights for the origins in
    STUDIO_CORS_ORIGINS and for no other; with the list empty (the deployed
    posture) it sends no CORS headers at all. A listed origin may read the
    turn index header a stream names its turn in."""
    listed = "http://localhost:8091"
    app = create_app(settings=Settings(database_url=pg_database_url, studio_cors_origins=[listed]))
    allowed = await _preflight(app, listed)
    assert allowed.status_code == 200
    assert allowed.headers.get("access-control-allow-origin") == listed
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://cors.test"
    ) as client:
        read = await client.get("/healthz", headers={"Origin": listed})
    assert TURN_INDEX_HEADER in read.headers.get("access-control-expose-headers", "").split(",")
    refused = await _preflight(app, "http://evil.example")
    assert "access-control-allow-origin" not in refused.headers

    closed = await _preflight(create_app(settings=Settings(database_url=pg_database_url)), listed)
    assert "access-control-allow-origin" not in closed.headers


def test_cors_origins_setting_parses_a_comma_separated_env_value() -> None:
    """The env var is a comma-separated list, the shape a shell can set."""
    settings = Settings(studio_cors_origins="http://localhost:8091, http://localhost:3000")  # type: ignore[arg-type]
    assert settings.studio_cors_origins == ["http://localhost:8091", "http://localhost:3000"]
    assert Settings(studio_cors_origins="").studio_cors_origins == []  # type: ignore[arg-type]


# Cloud Run's gap between SIGTERM and SIGKILL.
CLOUD_RUN_SIGKILL_AFTER_S = 10.0


@pytest.mark.parametrize(
    ("name", "resolved"),
    [
        ("gemini-3.8-flash", "gemini-3.8-flash"),
        ("Gemini:gemini-3.8-flash", "gemini-3.8-flash"),
        ("gemini:gemini-3.8-flash", "gemini-3.8-flash"),
        ("models/gemini-3.8-flash", "gemini-3.8-flash"),
        ("Gemini:models/gemini-3.8-flash", "gemini-3.8-flash"),
    ],
)
def test_the_model_name_is_resolved_to_the_one_the_developer_api_is_sent(
    name: str, resolved: str
) -> None:
    assert server_module.resolved_model_name(name) == resolved


@pytest.mark.parametrize(
    "name",
    [
        "projects/p/locations/l/publishers/google/models/gemini-3.8-flash",
        "Gemini:projects/p/locations/l/publishers/google/models/gemini-3.8-flash",
        "claude-sonnet-5-5",
        "Other:gemini-3.8-flash",
        "Gemini:",
        "Gemini:models/",
    ],
)
def test_a_model_name_the_developer_api_cannot_serve_refuses_startup(
    name: str, pg_database_url: str
) -> None:
    with pytest.raises(server_module.ModelNameRefusedError):
        create_app(settings=Settings(database_url=pg_database_url, model_name=name))


def test_create_app_gives_the_orchestrator_and_edit_card_one_resolved_model_name(
    pg_database_url: str,
) -> None:
    app = create_app(
        settings=Settings(database_url=pg_database_url, model_name="Gemini:gemini-3.8-flash")
    )

    assert app.state.studio.edit_model.model_name == "gemini-3.8-flash"


def test_main_bounds_the_request_drain_so_the_owners_close_before_sigkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cloud Run sends SIGKILL 10 s after SIGTERM. Uvicorn's drain, the wait
    for the stream owners' closing writes and the engine close fit inside it,
    with a second to spare."""
    passed: dict[str, Any] = {}

    def fake_app(*, settings: Settings) -> object:
        return object()

    monkeypatch.setattr(server_module, "create_app", fake_app)
    monkeypatch.setattr(server_module.uvicorn, "run", lambda app, **kwargs: passed.update(kwargs))
    monkeypatch.setattr(server_module, "Settings", lambda: Settings(database_url=None))

    server_module.main()

    engine_close_s = 0.5
    assert (
        passed["timeout_graceful_shutdown"] + SHUTDOWN_WAIT_S + engine_close_s
        <= CLOUD_RUN_SIGKILL_AFTER_S - 1.0
    )


def _postgres_error(sqlstate: str | None) -> asyncpg.PostgresError:
    error = asyncpg.PostgresError("failed")
    error.sqlstate = sqlstate
    return error


async def test_a_migration_refusal_is_fatal_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A SQLSTATE beginning `DFA`, a migration's own refusal, is not retried."""
    attempts: list[int] = []

    async def refusing_apply_pending(engine: object) -> list[str]:
        attempts.append(len(attempts) + 1)
        raise _postgres_error("DFA04")

    monkeypatch.setattr(server_module, "apply_pending", refusing_apply_pending)
    now, sleep = _fake_clock_and_sleep()

    with pytest.raises(server_module._MigrationRefusedError) as refusal:
        await server_module._apply_migrations_with_retries(
            cast(AsyncEngine, object()), budget_s=60.0, sleep=sleep, now=now
        )

    assert attempts == [1]
    assert refusal.value.sqlstate == "DFA04"
    assert refusal.value.__cause__ is not None


async def test_an_uncleared_adk_ledger_session_refuses_startup_with_nothing_applied(
    pg_empty_engine: AsyncEngine,
) -> None:
    """0003's cutover gate (DFA02) stops startup at the first attempt while an
    ADK ledger session was never imported, and the run applies no migration."""
    async with pg_empty_engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE TABLE sessions (app_name varchar(128) NOT NULL, "
                "user_id varchar(128) NOT NULL, id varchar(128) NOT NULL, state jsonb NOT NULL, "
                "create_time timestamp NOT NULL, update_time timestamp NOT NULL)"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO sessions VALUES ('dfo_analyst_graph.ledger', 'p', 'c1', '{}', "
                "now(), now())"
            )
        )
    now, sleep = _fake_clock_and_sleep()

    with pytest.raises(server_module._MigrationRefusedError) as refusal:
        await server_module._apply_migrations_with_retries(
            pg_empty_engine, budget_s=60.0, sleep=sleep, now=now
        )

    assert refusal.value.sqlstate == "DFA02"
    async with pg_empty_engine.connect() as conn:
        applied = (
            await conn.execute(text("SELECT count(*) FROM harness_schema_migrations"))
        ).scalar()
        conversations = (await conn.execute(text("SELECT to_regclass('conversations')"))).scalar()
    assert (applied, conversations) == (0, None)


async def test_any_other_migration_database_error_retries_until_the_budget_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []

    async def failing_apply_pending(engine: object) -> list[str]:
        attempts.append(len(attempts) + 1)
        raise _postgres_error("55P03")

    monkeypatch.setattr(server_module, "apply_pending", failing_apply_pending)
    now, sleep = _fake_clock_and_sleep()

    with pytest.raises(asyncpg.PostgresError):
        await server_module._apply_migrations_with_retries(
            cast(AsyncEngine, object()), budget_s=10.0, sleep=sleep, now=now
        )

    assert len(attempts) >= 2
