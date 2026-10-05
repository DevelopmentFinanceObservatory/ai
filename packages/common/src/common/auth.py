"""Bearer-token auth for the MCP mounts, shared by every service that builds one.

Three states, resolved once per process (first caller wins, mirroring
`common.materialisation.factory.install_store_if_unset`'s ownership rule for
the shared store holder):

- `MCP_API_TOKEN` set: that token is required.
- Unset: a random token is minted once and required. This locks an
  unconfigured deployment down instead of leaving it open, rotates every
  deploy, and never lands in a repo or a log.
- `MCP_AUTH_DISABLED=true`: no token is required at all.

`apply_gate` is the entry point every FastMCP-building code path calls
(`harness.server.create_app` for the co-mounted deployment,
`{analysis,data,presentation}_mcp.server.main` for a standalone one) before
its server's `.http_app()` / `.run()` builds the ASGI app. fastmcp bakes the
presence of `auth` into the route table at that point, and its
`RequireAuthMiddleware` rejects a request with no `Authorization` header
before any verifier gets a say. A lenient verifier alone cannot turn the
gate off. The toggle has to happen before the app is built, not from inside
a lifespan.

The three `FastMCP(...)` servers this guards are module-level singletons
built at import time, before any `Settings` exists, so each is constructed
directly with `auth=BearerTokenVerifier()` (`analysis_mcp.server`,
`data_mcp.server`, `presentation_mcp.server`). The verifier carries no
token of its own: it reads the resolved policy through this module's shared
state on every `verify_token` call. `bearer_token()` returns `None` until a
caller resolves the policy, and `BearerTokenVerifier` rejects every token
while it does, so a new code path that builds one of these apps without
calling `apply_gate` first locks the mount rather than leaving it open.
`apply_gate` sets `.auth` to `None` when the resolved policy is disabled,
and installs a fresh `BearerTokenVerifier()` when it is not. A singleton
can see both calls across a `reset()` (tests do this; nothing in
production rebuilds a mount's app twice), so the enabled path always
reinstates a verifier rather than assuming the constructor's is still
sitting on `.auth`.

A plain module-level singleton, not `common.holder.Holder`: that holder's
generic bound requires an `aclose()`-able resource, and a bearer token has
nothing to close.
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import TYPE_CHECKING

from fastmcp.server.auth import AccessToken, TokenVerifier

if TYPE_CHECKING:
    from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# The `client_id` every verified `AccessToken` carries. One shared bearer
# authenticates "the harness calling its own mounts", not an individual
# caller, so there is exactly one identity to name (see this module's own
# docstring on `apply_gate` for the ownership rule that keeps it singular).
_CLIENT_ID = "mcp-mount"


class _Policy:
    """Process-wide holder for the resolved auth policy, mirroring the
    `_ClientHolder` / `_Holder` shape every other module-level singleton in
    this workspace already uses (`data_mcp.client_holder`,
    `common.materialisation.holder`). A plain instance rather than module
    globals, so `_resolve`/`reset` mutate state through `self` instead of a
    `global` statement.
    """

    def __init__(self) -> None:
        self.resolved = False
        self.disabled = False
        self.token: str | None = None

    def resolve(self, *, token: str | None, disabled: bool) -> None:
        """First caller wins. Every service's own `Settings` reads the same
        env vars, so in the co-mounted deployment they agree on
        `token`/`disabled` regardless of which one resolves first. A later
        call with different values is a no-op: the process-wide policy
        stays whatever the first caller set.
        """
        if self.resolved:
            return
        self.disabled = disabled
        if disabled:
            logger.error("MCP_AUTH_DISABLED is set: the MCP mounts require no bearer token.")
        else:
            if token is None and os.environ.get("K_SERVICE"):
                logger.warning(
                    "MCP_API_TOKEN is unset on a Cloud Run instance (K_SERVICE=%s); "
                    "minting an ephemeral token reachable only from inside this "
                    "process. Set MCP_API_TOKEN to a known value for an external "
                    "caller to reach the MCP mounts.",
                    os.environ["K_SERVICE"],
                )
            self.token = token or secrets.token_urlsafe(32)
        self.resolved = True

    def reset(self) -> None:
        self.resolved = False
        self.disabled = False
        self.token = None


_policy = _Policy()


def bearer_token() -> str | None:
    """The token every mount requires, or `None` when the gate is disabled
    or not yet resolved (`apply_gate` hasn't run). Both read the same to a
    caller, since neither state has a token worth sending."""
    return _policy.token if _policy.resolved and not _policy.disabled else None


def apply_gate(mcp: FastMCP, *, token: str | None, disabled: bool) -> None:
    """Resolve the process-wide auth policy (first call wins) and set
    `mcp`'s gate to match: `None` if the resolved policy is disabled, a
    fresh `BearerTokenVerifier()` otherwise. Call this before `.http_app()`
    / `.run()` builds `mcp`'s ASGI app. A call after has no effect on the
    routes already built.

    Always installs the verifier on the enabled path, rather than trusting
    the constructor's, so a `reset()` between a disabled call and an
    enabled one (as in tests) restores the gate instead of leaving `mcp`
    permanently open.
    """
    _policy.resolve(token=token, disabled=disabled)
    if _policy.disabled:
        mcp.auth = None
    else:
        mcp.auth = BearerTokenVerifier()


def reset() -> None:
    """Clear the resolved policy so the next `apply_gate` call resolves fresh.

    Test-only. This workspace's whole suite runs in one process, so a test
    that resolves the policy (directly, or by building an app that calls
    `apply_gate`) must not leak into a later, unrelated test.
    """
    _policy.reset()


class BearerTokenVerifier(TokenVerifier):
    """Constant-time bearer check against the token `apply_gate` resolves.

    No `base_url` is configured, so this advertises no RFC 9728 discovery
    metadata on a 401, which is moot for a static shared secret since no
    OAuth discovery flow ever runs against it.
    """

    async def verify_token(self, token: str) -> AccessToken | None:
        expected = bearer_token()
        if expected is None or not secrets.compare_digest(token, expected):
            return None
        return AccessToken(token=token, client_id=_CLIENT_ID, scopes=[])
