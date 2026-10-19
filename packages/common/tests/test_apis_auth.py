"""`common.apis_auth`: minting, caching and refreshing the Cloud Run ID token."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable, Mapping

import httpx2
import pytest
from common.apis_auth import (
    METADATA_IDENTITY_URL,
    REFRESH_MARGIN_S,
    TOKEN_HEADER,
    ApisAuthError,
    IdTokenProvider,
    apis_auth_provider,
)

AUDIENCE = "https://dfo-dev-apis-abc123.a.run.app"
NOW = 1_800_000_000.0


def jwt(claims: object) -> str:
    """A JWT-shaped string carrying `claims`, with a dummy header and signature."""
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"eyJhbGciOiJSUzI1NiJ9.{payload}.c2ln"


class Metadata:
    """A metadata server that answers each request with the next token, expiring an hour on."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self.requests: list[httpx2.Request] = []
        self._clock = clock

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        token = jwt({"n": len(self.requests), "exp": int(self._clock()) + 3600})
        return httpx2.Response(200, text=token)


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> float:
        return self.now


def provider(
    handler: Callable[[httpx2.Request], httpx2.Response], clock: Callable[[], float] = lambda: NOW
) -> IdTokenProvider:
    return IdTokenProvider(audience=AUDIENCE, transport=httpx2.MockTransport(handler), clock=clock)


def token_number(headers: Mapping[str, str]) -> int:
    scheme, token = headers[TOKEN_HEADER].split(" ")
    assert scheme == "Bearer"
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["n"]


async def test_mints_from_the_metadata_server_for_the_audience() -> None:
    clock = Clock()
    metadata = Metadata(clock)

    headers = await provider(metadata, clock)()

    (request,) = metadata.requests
    assert str(request.url.copy_with(query=None)) == METADATA_IDENTITY_URL
    assert dict(request.url.params) == {"audience": AUDIENCE, "format": "full"}
    assert request.headers["Metadata-Flavor"] == "Google"
    assert list(headers) == [TOKEN_HEADER]
    assert token_number(headers) == 1


async def test_caches_the_token_until_the_refresh_margin_then_mints_again() -> None:
    clock = Clock()
    metadata = Metadata(clock)
    auth = provider(metadata, clock)

    first = await auth()
    clock.now = NOW + 3600 - REFRESH_MARGIN_S - 1
    cached = await auth()
    clock.now = NOW + 3600 - REFRESH_MARGIN_S
    refreshed = await auth()

    assert [token_number(h) for h in (first, cached, refreshed)] == [1, 1, 2]
    assert len(metadata.requests) == 2


async def test_concurrent_calls_on_a_stale_cache_share_one_request() -> None:
    clock = Clock()
    metadata = Metadata(clock)
    release = asyncio.Event()

    async def slow(request: httpx2.Request) -> httpx2.Response:
        await release.wait()
        return metadata(request)

    auth = IdTokenProvider(audience=AUDIENCE, transport=httpx2.MockTransport(slow), clock=clock)
    calls = [asyncio.create_task(auth()) for _ in range(5)]
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(*calls)

    assert len(metadata.requests) == 1
    assert {token_number(h) for h in results} == {1}


def _raise_connect_error(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("no route to metadata.google.internal", request=request)


@pytest.mark.parametrize(
    "handler",
    [
        _raise_connect_error,
        lambda _: httpx2.Response(500, text="internal error"),
        lambda _: httpx2.Response(404, text="not found"),
        lambda _: httpx2.Response(200, text="not-a-jwt"),
        lambda _: httpx2.Response(200, text=jwt({"sub": "no exp"})),
        lambda _: httpx2.Response(200, text=jwt({"exp": "soon"})),
        lambda _: httpx2.Response(200, text="a.!!!.c"),
    ],
    ids=["unreachable", "5xx", "4xx", "not-a-jwt", "no-exp", "exp-not-a-number", "bad-base64"],
)
async def test_a_metadata_failure_raises_and_logs(
    handler: Callable[[httpx2.Request], httpx2.Response], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR, logger="common.apis_auth"), pytest.raises(ApisAuthError):
        await provider(handler)()

    assert caplog.records


async def test_a_failed_refresh_sends_the_held_token_until_its_exp(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = Clock()
    metadata = Metadata(clock)
    down = False

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(503, text="unavailable") if down else metadata(request)

    auth = provider(handler, clock)
    await auth()
    down = True
    clock.now = NOW + 3600 - 1

    with caplog.at_level(logging.WARNING, logger="common.apis_auth"):
        held = await auth()

    assert token_number(held) == 1
    assert any(record.levelno == logging.WARNING for record in caplog.records)

    clock.now = NOW + 3600
    with pytest.raises(ApisAuthError):
        await auth()


async def test_one_failed_refresh_covers_the_callers_queued_behind_it() -> None:
    clock = Clock()
    metadata = Metadata(clock)
    down = False
    release = asyncio.Event()

    async def handler(request: httpx2.Request) -> httpx2.Response:
        if not down:
            return metadata(request)
        metadata.requests.append(request)
        await release.wait()
        return httpx2.Response(503, text="unavailable")

    auth = provider(handler, clock)
    await auth()
    down = True
    clock.now = NOW + 3600 - REFRESH_MARGIN_S

    calls = [asyncio.create_task(auth()) for _ in range(5)]
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(*calls)

    assert len(metadata.requests) == 2
    assert {token_number(h) for h in results} == {1}


async def test_off_cloud_run_the_provider_adds_no_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("CLOUD_RUN_JOB", raising=False)

    auth = apis_auth_provider(base_url=AUDIENCE)

    assert not isinstance(auth, IdTokenProvider)
    assert await auth() == {}


@pytest.mark.parametrize(
    ("env_var", "base_url"),
    [
        ("K_SERVICE", AUDIENCE),
        ("K_SERVICE", f"{AUDIENCE}/"),
        ("K_SERVICE", f"{AUDIENCE}/api/search?q=x"),
        ("CLOUD_RUN_JOB", AUDIENCE),
    ],
    ids=str,
)
async def test_on_cloud_run_the_audience_is_the_base_url_origin(
    env_var: str, base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("CLOUD_RUN_JOB", raising=False)
    monkeypatch.setenv(env_var, "dfo-dev-ai")

    auth = apis_auth_provider(base_url=base_url)

    assert isinstance(auth, IdTokenProvider)
    assert auth.audience == AUDIENCE
