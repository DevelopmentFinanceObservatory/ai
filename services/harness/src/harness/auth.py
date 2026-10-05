"""The studio auth seam: one function between the routes and the token.

v1 treats the bearer token as opaque (contract delta 5): presence is
required on every studio request, and when the `STUDIO_API_TOKEN` setting
is set the token must equal it (the dev-time shared secret). Full JWT
validation (the website's Better Auth) plugs in HERE later -- `principal_for`
is the only place a token is ever read, so swapping the check never touches
a route.

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
    if settings.studio_api_token is not None and not secrets.compare_digest(
        token, settings.studio_api_token
    ):
        raise HTTPException(
            status_code=401,
            detail="the bearer token was not accepted",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return "u_" + hashlib.sha256(token.encode()).hexdigest()[:16]
