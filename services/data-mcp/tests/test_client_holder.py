"""Tests for the process-wide upstream client holder: lazy build and rebuild-after-close."""

from __future__ import annotations

from typing import Any

import pytest
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
