"""Cloud Run ID tokens for calls from this service to dfo-apis.

dfo-apis runs on Cloud Run, and its only identity boundary is Cloud Run IAM:
a caller must present a Google ID token whose audience is the apis service
URL, minted for an identity that holds `roles/run.invoker` on the service.

`apis_auth_provider` is the entry point. On Cloud Run, in a service
(`K_SERVICE` set) or a job (`CLOUD_RUN_JOB` set), it returns an
`IdTokenProvider`, which asks the metadata server for a token for the runtime
service account and caches it until `REFRESH_MARGIN_S` before it expires.
Anywhere else it returns a provider that sends no token. Local development
reaches dfo-apis through `gcloud run services proxy`, which authenticates on
the developer's behalf.

The token travels in `X-Serverless-Authorization`. Cloud Run checks that
header in preference to `Authorization` and strips its signature before the
request reaches the container, so it never collides with an application
credential in `Authorization`.

On Cloud Run, a refresh that fails while the held token is still before its
`exp` logs a warning and sends the held token, and the provider waits
`REFRESH_BACKOFF_S` (cut to that `exp`) before asking the metadata server
again, so callers queued behind the failed refresh share its outcome. A
failure with no unexpired token held raises `ApisAuthError`. It never degrades
to an anonymous request, which would fail at the IAM boundary with a 403 that
hides the cause.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from urllib.parse import urlsplit

import httpx2

logger = logging.getLogger(__name__)

# The async credential hook every dfo-apis client takes as `auth=`: called once
# per call, returning the headers to add to it.
type AuthProvider = Callable[[], Awaitable[Mapping[str, str]]]

TOKEN_HEADER = "X-Serverless-Authorization"

METADATA_IDENTITY_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity"
)

# Metadata-server tokens live an hour. Refreshing five minutes early keeps a
# token that is about to expire off a request that may queue or retry.
REFRESH_MARGIN_S = 300.0

# After a failed refresh with an unexpired token held, how long the held token
# is sent before the metadata server is asked again.
REFRESH_BACKOFF_S = 15.0

_METADATA_TIMEOUT_S = 5.0
_ERROR_BODY_CHARS = 200


class ApisAuthError(RuntimeError):
    """The metadata server did not return a usable ID token. No request was made."""


class IdTokenProvider:
    """Mints and caches the runtime service account's ID token for one audience.

    Callable as an `AuthProvider`. Concurrent calls that find the cache stale
    share one metadata request.
    """

    def __init__(
        self,
        *,
        audience: str,
        transport: httpx2.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Build a provider for `audience`.

        Args:
            audience: The dfo-apis origin, e.g. `https://dfo-dev-apis-xyz.a.run.app`.
            transport: `httpx2` transport for the metadata request,
                `httpx2.MockTransport` in tests.
            clock: Wall-clock seconds, compared with the token's `exp`.
        """
        self.audience = audience
        self._transport = transport
        self._clock = clock
        self._token: str | None = None
        self._expires_at = 0.0
        self._retry_at = 0.0
        self._lock = asyncio.Lock()

    async def __call__(self) -> Mapping[str, str]:
        """The token header for one call, minting a fresh token when the cached one is stale.

        A failed refresh falls back to the held token while it is before its `exp`,
        and the next `REFRESH_BACKOFF_S` of calls send it without asking again.

        Raises:
            ApisAuthError: the metadata server failed or answered something
                other than a JWT with an `exp`, and no unexpired token is held.
        """
        if not self._usable():
            async with self._lock:
                if not self._usable():
                    await self._refresh()
        return {TOKEN_HEADER: f"Bearer {self._token}"}

    def _usable(self) -> bool:
        """Whether the held token can be sent without asking the metadata server.

        True before the refresh margin, and inside the back-off after a failed
        refresh, which never runs past the token's `exp`.
        """
        if self._token is None:
            return False
        now = self._clock()
        return now < self._expires_at - REFRESH_MARGIN_S or now < self._retry_at

    async def _refresh(self) -> None:
        try:
            self._token, self._expires_at = await self._mint()
        except ApisAuthError:
            now = self._clock()
            remaining_s = self._expires_at - now
            if self._token is None or remaining_s <= 0:
                raise
            self._retry_at = min(now + REFRESH_BACKOFF_S, self._expires_at)
            logger.warning(
                "Refreshing the dfo-apis ID token failed; sending the held token, "
                "which expires in %.0fs",
                remaining_s,
            )

    async def _mint(self) -> tuple[str, float]:
        try:
            async with httpx2.AsyncClient(
                transport=self._transport, timeout=_METADATA_TIMEOUT_S
            ) as client:
                response = await client.get(
                    METADATA_IDENTITY_URL,
                    # `full` adds the `email` claim, which dfo-apis logs as the caller.
                    params={"audience": self.audience, "format": "full"},
                    headers={"Metadata-Flavor": "Google"},
                )
        except httpx2.TransportError as exc:
            logger.error("Metadata server unreachable minting a dfo-apis ID token: %r", exc)
            raise ApisAuthError(f"metadata server unreachable: {type(exc).__name__}") from exc
        if not response.is_success:
            detail = response.text[:_ERROR_BODY_CHARS]
            logger.error(
                "Metadata server answered %d minting a dfo-apis ID token: %s",
                response.status_code,
                detail,
            )
            raise ApisAuthError(f"metadata server answered {response.status_code}: {detail}")
        token = response.text.strip()
        expires_at = _expiry(token)
        if expires_at is None:
            logger.error("Metadata server answered a dfo-apis ID token with no readable exp")
            raise ApisAuthError("metadata server answered a token with no readable exp")
        return token, expires_at


def _expiry(token: str) -> float | None:
    """The `exp` claim of a JWT, read without verifying the signature, or `None`."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    # binascii.Error, UnicodeDecodeError and JSONDecodeError are all ValueErrors.
    except ValueError:
        return None
    exp = claims.get("exp") if isinstance(claims, dict) else None
    if isinstance(exp, bool) or not isinstance(exp, int | float):
        return None
    return float(exp)


async def _no_token() -> Mapping[str, str]:
    return {}


def apis_auth_provider(*, base_url: str) -> AuthProvider:
    """The credential hook for a client calling dfo-apis at `base_url`.

    An `IdTokenProvider` whose audience is `base_url`'s origin on Cloud Run,
    in a service (`K_SERVICE`) or a job (`CLOUD_RUN_JOB`), and a provider that
    adds no header anywhere else.
    """
    if not (os.environ.get("K_SERVICE") or os.environ.get("CLOUD_RUN_JOB")):
        return _no_token
    parts = urlsplit(base_url)
    return IdTokenProvider(audience=f"{parts.scheme}://{parts.netloc}")
