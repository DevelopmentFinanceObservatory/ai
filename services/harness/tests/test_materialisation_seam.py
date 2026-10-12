"""The cross-service seam this test proves: a frame `data_mcp.get_data`
registers is resolvable by a mounted `analysis_mcp` verb, through the real
composed ASGI app -- not two services' own unit tests asserting compatible
shapes in isolation.

`_running_lifespan`/`_mounted_mcp_transport` mirror `test_server.py`'s own
helpers rather than importing them: `services/harness/tests` carries no
`__init__.py` and this package's `pyproject.toml` sets
`--import-mode=importlib`, so a sibling test module has no stable name to
import from (confirmed by attempting both a bare and a relative import here
first; both raise `ModuleNotFoundError`).
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, MutableMapping
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx2
import pytest
from common.materialisation import frame_to_parquet_bytes
from data_mcp import client_holder, frame_materialisation
from data_mcp.client import UpstreamClient
from data_mcp.settings import Settings as DataMcpSettings
from dfo_frame_grammar import Frame, frame_sha256, mint_content_version
from fastapi import FastAPI
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from harness.server import create_app
from harness.settings import ANALYSIS_MCP_MOUNT_PATH, DATA_MCP_MOUNT_PATH, Settings
from mcp.shared._httpx_utils import McpHttpClientFactory

_SPEC_ID = "spec_" + "5" * 32


def _frame() -> Frame:
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:Test",
                "reporter": "country/KEN",
                "time_period": "2020",
                "value": 1.5,
            }
        ],
    )


# The version's token, minted from its canonical frame as dfo-apis mints it.
_CONTENT_VERSION = mint_content_version(frame_sha256=frame_sha256(_frame()))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate this module from any of these vars set in the ambient environment."""
    for var in ("DATA_MCP_URL", "ANALYSIS_MCP_URL", "PRESENTATION_MCP_URL", "DATABASE_URL", "PORT"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
async def _reset_data_mcp_client() -> AsyncIterator[None]:
    """Nothing in `data_mcp.server`'s own lifespan touches `client_holder` (only the
    definition store and the frame store are lifespan-managed), so a client set here
    would otherwise leak into a later test -- including one in another package
    entirely, since this workspace's whole suite runs in one process.
    """
    yield
    client_holder.set_client(None)


@asynccontextmanager
async def _running_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Drive `app`'s ASGI lifespan by hand for an in-process MCP handshake.

    See `test_server.py`'s own copy of this helper for the full rationale
    (`httpx2.ASGITransport` never forwards lifespan events, but a mounted
    FastMCP app's session manager only starts inside its own lifespan).
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


def _mounted_mcp_transport(app: FastAPI, mounted_path: str) -> StreamableHttpTransport:
    """Build a fastmcp `Client` transport that talks MCP to `mounted_path` on `app`,
    routed through `httpx2.ASGITransport` -- see `test_server.py`'s own copy for the
    full rationale."""

    def factory(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://testserver", **kwargs
        )

    return StreamableHttpTransport(
        url=f"http://testserver{mounted_path}",
        httpx_client_factory=cast(McpHttpClientFactory, factory),
    )


def _mint_response() -> dict[str, Any]:
    """A mint response shaped like `data-mcp`'s own `test_get_data_tool.py` fixtures,
    with one difference: `content_version` here actually matches `FrameRef`'s pair
    pattern (`^cv_[0-9a-f]{32}$`) -- most of that module's own fixtures deliberately
    don't, since they stub `materialise_and_register` rather than exercising it, the
    same way this test needs to.
    """
    return {
        "schema_version": 1,
        "definition": {
            "schema_version": 1,
            "spec_id": _SPEC_ID,
            "identity": {
                "identity_version": "spec1",
                "shape": {"id": "SeamTestShape", "kind": "stat_var"},
                "entities": {"reporter": ["country/KEN"]},
                "facets": {},
                "requested_window": {"kind": "all"},
                "slots": [],
            },
        },
        "content_version": _CONTENT_VERSION,
        "coverage": {"rows": 1, "dimensions": []},
        "citation": {"text": "seam test fixture", "sources": []},
        "trace": {"graph_revision": "g1", "facets": []},
        "warnings": [],
        "artifacts": [
            {
                "artifact_digest": _artifact_digest(),
                "rendering": {"bundle": False, "format": "parquet", "layout": "frame"},
                "bytes": 100,
                "media_type": "application/vnd.apache.parquet",
                "created_at": "2026-08-11T00:00:00+00:00",
                "tier": "working",
                "url": "/links/should-not-be-fetched",
            }
        ],
    }


def _parquet_bytes() -> bytes:
    """One row in the canonical layout: the version's canonical frame, which the
    default mint registers, and its frame file."""
    return frame_to_parquet_bytes(_frame())


def _artifact_digest() -> str:
    """`_mint_response`'s own advertised `artifact_digest`, computed truthfully from
    `_parquet_bytes()` so the fetch-parse-register path's own digest gate passes."""
    return f"sha256:{hashlib.sha256(_parquet_bytes()).hexdigest()}"


async def test_a_frame_data_mcp_registers_resolves_through_a_mounted_analysis_mcp_verb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`get_data`, called through the mounted `data_mcp` app, fetches and registers a
    frame in the shared `common.materialisation` store; a `filter` call through the
    mounted `analysis_mcp` app then resolves that exact `{spec_id, content_version}`
    pair -- the actual seam this test proves, through the real streamable-HTTP wire
    and the real ASGI composition, not two services' own unit tests asserting
    compatible shapes in isolation.

    The bare signed-URL GET is the one hop nothing in `data_mcp.get_data_tool`
    threads a test transport into (see `frame_materialisation.materialise_and_register`'s
    own `transport` parameter, a seam only `data_mcp`'s own suite exercises directly);
    patched here at `_fetch_artifact_bytes`, the same seam `data-mcp`'s own
    `test_get_data_tool.py::test_successful_get_data_call_registers_a_frame_end_to_end`
    uses to prove the tool-to-store wiring without opening a real socket.
    """

    async def fake_fetch(**_kwargs: Any) -> bytes:
        return _parquet_bytes()

    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", fake_fetch)

    def mint_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(200, json=_mint_response())
        if request.url.path == f"/api/datasets/{_SPEC_ID}/{_CONTENT_VERSION}":
            # The snapshot lists the version's canonical frame beside the frame file.
            snapshot = _mint_response()
            snapshot["artifacts"].append(
                snapshot["artifacts"][0]
                | {"rendering": {"format": "frame"}, "url": "/links/canonical"}
            )
            return httpx2.Response(200, json=snapshot)
        raise AssertionError(f"unexpected call to {request.url.path}")

    client_holder.set_client(
        UpstreamClient(settings=DataMcpSettings(), transport=httpx2.MockTransport(mint_handler))
    )

    app = create_app(settings=Settings(mcp_api_token="test-token"))

    async with _running_lifespan(app):
        data_transport = _mounted_mcp_transport(app, f"{DATA_MCP_MOUNT_PATH}/mcp")
        async with Client(data_transport, auth="test-token") as data_client:
            get_data_result = await data_client.call_tool(
                "get_data", {"spec_id": _SPEC_ID}, raise_on_error=False
            )

        assert get_data_result.is_error is False
        digest = get_data_result.content[0].text
        assert _SPEC_ID in digest
        assert _CONTENT_VERSION in digest

        analysis_transport = _mounted_mcp_transport(app, f"{ANALYSIS_MCP_MOUNT_PATH}/mcp")
        async with Client(analysis_transport, auth="test-token") as analysis_client:
            filter_result = await analysis_client.call_tool(
                "filter",
                {"frame_ref": {"spec_id": _SPEC_ID, "content_version": _CONTENT_VERSION}},
                raise_on_error=False,
            )

    # The one assertion that is the seam: a genuine `frame_not_held` would mean the
    # store `data_mcp.get_data` wrote into and the store this mounted `filter` call
    # resolved against were never actually the same store.
    assert filter_result.is_error is False
