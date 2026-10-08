"""The studio auth seam: one function between the routes and the token.

v1 treats the bearer token as opaque (contract delta 5): presence is
required on every studio request, and when the `STUDIO_API_TOKEN` setting
is set the token must equal it (the dev-time shared secret). Full JWT
validation (the website's Better Auth) plugs in HERE later -- `principal_for`
is the only place a token is ever read, so swapping the check never touches
a route.

`POST /api/studio/refresh` is the one studio route that does not go through
`principal_for`. It takes the live-refresh principal alone
(`live_refresh_admits`), and that value opens no other route.

The returned principal is a stable derivation of the token: it becomes the
ADK session `user_id`, and session ids must not embed credentials. Two
requests with the same token always land on the
same principal, which is what keys conversation isolation.
"""

from __future__ import annotations

import hashlib
import secrets

from fastapi import HTTPException

from harness.settings import Settings

_BEARER_PREFIX = "bearer "


def principal_for(authorization: str | None, *, settings: Settings) -> str:
    """Resolve an `Authorization` header to a principal id, or raise 401.

    The scheme comparison is case-insensitive per RFC 9110; the token is
    compared to `settings.studio_api_token` (when set) in constant time.
    """
    if authorization is None or not authorization.lower().startswith(_BEARER_PREFIX):
        raise HTTPException(
            status_code=401,
            detail="studio requests require an 'Authorization: Bearer <token>' header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization[len(_BEARER_PREFIX) :].strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail="the bearer token is empty",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if live_refresh_admits(authorization, settings=settings) or (
        settings.studio_api_token is not None
        and not secrets.compare_digest(token, settings.studio_api_token)
    ):
        raise HTTPException(
            status_code=401,
            detail="the bearer token was not accepted",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return "u_" + hashlib.sha256(token.encode()).hexdigest()[:16]


def live_refresh_admits(authorization: str | None, *, settings: Settings) -> bool:
    """Whether `authorization` is `Bearer <LIVE_REFRESH_PRINCIPAL>`.

    False for every header while `settings.live_refresh_principal` is unset, so
    the refresh route is closed until the secret is mounted. The value is
    compared in constant time, as bytes, so a non-ASCII header is refused
    without raising.
    """
    principal = settings.live_refresh_principal
    if (
        principal is None
        or authorization is None
        or not authorization.lower().startswith(_BEARER_PREFIX)
    ):
        return False
    token = authorization[len(_BEARER_PREFIX) :].strip()
    return secrets.compare_digest(token.encode(), principal.encode())
