"""`common.auth`: policy resolution, `apply_gate`, and `BearerTokenVerifier`.

The root `conftest.py`'s `_reset_mcp_auth_gate` fixture clears the
process-wide policy after every test in the workspace. Nothing here needs
to reset it itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx2
import pytest
from common import auth as auth_module
from common.auth import BearerTokenVerifier, apply_gate, bearer_token, reset
from fastmcp import FastMCP


def test_bearer_token_is_none_before_anything_resolves() -> None:
    assert bearer_token() is None


async def test_verify_token_rejects_everything_before_apply_gate_runs() -> None:
    """The fail-closed default. A `FastMCP(auth=BearerTokenVerifier())` built
    at import time, with no `apply_gate` call yet, locks its mount rather
    than leaving it open."""
    verifier = BearerTokenVerifier()

    assert await verifier.verify_token("anything") is None


async def test_apply_gate_with_a_configured_token_accepts_exactly_that_token() -> None:
    mcp = SimpleNamespace(auth="unused")

    apply_gate(mcp, token="s3cret", disabled=False)

    assert bearer_token() == "s3cret"
    verifier = BearerTokenVerifier()
    assert await verifier.verify_token("s3cret") is not None
    assert await verifier.verify_token("wrong") is None
    # The enabled path always installs a fresh verifier, rather than trusting
    # whatever `.auth` already carries, so a prior disabled call on the same
    # object (tests do this via `reset()`) can't leave it permanently open.
    assert isinstance(mcp.auth, BearerTokenVerifier)


async def test_apply_gate_with_no_token_mints_one() -> None:
    mcp = SimpleNamespace(auth="unused")

    apply_gate(mcp, token=None, disabled=False)

    minted = bearer_token()
    assert minted is not None
    verifier = BearerTokenVerifier()
    assert await verifier.verify_token(minted) is not None


def test_apply_gate_disabled_clears_auth_and_bearer_token() -> None:
    mcp = SimpleNamespace(auth="unused")

    apply_gate(mcp, token=None, disabled=True)

    assert mcp.auth is None
    assert bearer_token() is None


def test_apply_gate_first_call_wins() -> None:
    """Every server's own `apply_gate` call resolves the same process-wide
    policy. Only the first one actually decides it (mirrors
    `common.materialisation.factory.install_store_if_unset`)."""
    first = SimpleNamespace(auth="unused")
    second = SimpleNamespace(auth="unused")

    apply_gate(first, token="first-token", disabled=False)
    apply_gate(second, token="second-token", disabled=False)

    assert bearer_token() == "first-token"


@pytest.mark.parametrize("bad_token", ["", "s3cretx", "S3cret"])
async def test_verify_token_is_exact(bad_token: str) -> None:
    mcp = SimpleNamespace(auth="unused")
    apply_gate(mcp, token="s3cret", disabled=False)

    assert await BearerTokenVerifier().verify_token(bad_token) is None


def test_apply_gate_warns_when_no_token_is_given_on_a_cloud_run_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("K_SERVICE", "data-mcp")
    warnings: list[str] = []
    monkeypatch.setattr(
        auth_module.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    mcp = SimpleNamespace(auth="unused")

    apply_gate(mcp, token=None, disabled=False)

    minted = bearer_token()
    assert minted is not None
    assert any("MCP_API_TOKEN" in w and "data-mcp" in w for w in warnings)
    assert not any(minted in w for w in warnings)


def test_apply_gate_does_not_warn_when_a_token_is_given_on_a_cloud_run_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("K_SERVICE", "data-mcp")
    warnings: list[str] = []
    monkeypatch.setattr(
        auth_module.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    mcp = SimpleNamespace(auth="unused")

    apply_gate(mcp, token="s3cret", disabled=False)

    assert warnings == []


def test_apply_gate_does_not_warn_when_no_token_is_given_off_cloud_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("K_SERVICE", raising=False)
    warnings: list[str] = []
    monkeypatch.setattr(
        auth_module.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    mcp = SimpleNamespace(auth="unused")

    apply_gate(mcp, token=None, disabled=False)

    assert warnings == []


async def test_apply_gate_restores_the_gate_after_a_disabled_reset_re_enabled_cycle() -> None:
    """Drives the real disable -> reset -> re-enable sequence against a real
    `FastMCP` object and its ASGI app, not the `SimpleNamespace` stand-ins the
    rest of this file uses. A stand-in can't observe `RequireAuthMiddleware`,
    which is exactly why a disabled `apply_gate` call leaving a later,
    enabled call's mount open went uncaught.
    """
    mcp = FastMCP("gate-restore-test", auth=BearerTokenVerifier())
    apply_gate(mcp, token=None, disabled=True)
    reset()

    apply_gate(mcp, token="s3cret", disabled=False)

    app = mcp.http_app(transport="streamable-http")
    initialize_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "unauthenticated-probe", "version": "0.0"},
        },
    }
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://testserver"
        ) as client,
    ):
        # No `Authorization` header: this is the request a caller with no
        # token sends. `Accept` has to name both, or fastmcp's streamable-http
        # transport 406s before `RequireAuthMiddleware` gets a say -- a
        # content-negotiation failure this test must not mistake for the
        # auth rejection it's actually checking.
        response = await client.post(
            "/mcp",
            json=initialize_request,
            headers={"Accept": "application/json, text/event-stream"},
        )

    assert response.status_code == 401
