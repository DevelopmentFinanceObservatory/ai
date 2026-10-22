"""Tests for the process-wide upstream client holder: lazy build and rebuild-after-close."""

from __future__ import annotations

from typing import Any

import pytest
from common.apis_auth import IdTokenProvider
from data_mcp import client_holder
from data_mcp.client import UpstreamClient


@pytest.fixture(autouse=True)
async def _reset_client() -> Any:
    """Every test wires its own client state; nothing should leak to the next test."""
    yield
    client_holder.set_client(None)


# Client wiring — the lazy default the deployed topology actually relies on,
# since harness mounts data-mcp in-process and never calls `main()`.


async def test_client_holder_lazily_builds_a_real_upstream_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DFO_APIS_URL", "https://dfo-apis.internal.example")

    client = client_holder.get_client()

    try:
        assert isinstance(client, UpstreamClient)
        assert str(client._client.base_url) == "https://dfo-apis.internal.example"
        # Cached, not rebuilt, on a second call.
        assert client_holder.get_client() is client
    finally:
        await client.aclose()


async def test_client_holder_rebuilds_after_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second server lifespan must not inherit the first's closed client."""
    monkeypatch.setenv("DFO_APIS_URL", "https://dfo-apis.internal.example")

    first = client_holder.get_client()
    await client_holder.aclose_client()

    assert client_holder._holder._client is None
    assert first._client.is_closed is True

    second = client_holder.get_client()
    try:
        assert second is not first
        assert second._client.is_closed is False
    finally:
        await second.aclose()


async def test_on_cloud_run_the_lazy_client_carries_the_apis_id_token_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`main()` builds its client through the same `build_client`."""
    monkeypatch.setenv("K_SERVICE", "dfo-dev-ai")
    monkeypatch.setenv("DFO_APIS_URL", "https://dfo-apis.internal.example/")
    monkeypatch.setenv("DFO_SITE_ORIGIN", "https://site.example")

    client = client_holder.get_client()

    try:
        assert isinstance(client._auth, IdTokenProvider)
        assert client._auth.audience == "https://dfo-apis.internal.example"
    finally:
        await client.aclose()


async def test_off_cloud_run_the_lazy_client_sends_no_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("K_SERVICE", raising=False)

    client = client_holder.get_client()

    try:
        assert client._auth is not None
        assert await client._auth() == {}
    finally:
        await client.aclose()
