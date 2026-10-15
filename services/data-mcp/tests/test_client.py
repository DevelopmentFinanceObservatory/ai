import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Coroutine, Mapping

import httpx2
import pytest
from common.apis_auth import TOKEN_HEADER, ApisAuthError, AuthProvider
from data_mcp.client import (
    _PROBE_TIMEOUT_S,
    DATASETS_PREFIX,
    DEFAULT_ACCEPT_HEADER,
    GRAPH_PREFIX,
    SEARCH_PREFIX,
    SKILLS_PREFIX,
    UpstreamClient,
)
from data_mcp.deadline import CallDeadline
from data_mcp.errors import UpstreamError, UpstreamSurfaceUnavailable, UpstreamTransportError
from data_mcp.settings import Settings
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

Handler = (
    Callable[[httpx2.Request], httpx2.Response]
    | Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]]
)


def _settings(**overrides: object) -> Settings:
    """Settings tuned for fast tests, with short timeouts, near-zero backoff, and a generous budget."""
    defaults: dict[str, object] = {
        "upstream_connect_timeout_s": 1.0,
        "upstream_read_timeout_s": 1.0,
        "upstream_retry_budget_s": 10.0,
        "upstream_max_attempts": 3,
        "upstream_backoff_base_s": 0.01,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _recording_sleep() -> tuple[list[float], Callable[[float], Awaitable[None]]]:
    delays: list[float] = []

    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    return delays, sleep


def _client(
    handler: Handler, *, settings: Settings | None = None, **kwargs: object
) -> UpstreamClient:
    return UpstreamClient(
        settings=settings or _settings(),
        transport=httpx2.MockTransport(handler),
        **kwargs,
    )


ID_TOKEN = "Bearer test-id-token"


def id_token_auth(*, mints: int | None = None) -> AuthProvider:
    """An auth hook sending `ID_TOKEN` in `TOKEN_HEADER`, which fails with
    `ApisAuthError` once it has answered `mints` calls, when `mints` is set."""
    answered = 0

    async def auth() -> Mapping[str, str]:
        nonlocal answered
        if mints is not None and answered >= mints:
            raise ApisAuthError("metadata server unreachable: ConnectError")
        answered += 1
        return {TOKEN_HEADER: ID_TOKEN}

    return auth


def _problem_body(*, code: str, retryable: bool, **extra: object) -> bytes:
    payload: dict[str, object] = {
        "code": code,
        "title": "refused",
        "detail": "because",
        "retryable": retryable,
        "trace_id": "trace-1",
        "capability_version": "v1",
        **extra,
    }
    return json.dumps(payload).encode()


def text_of(result: CallToolResult) -> str:
    """The text of a tool result's first content block.

    `CallToolResult.content` is a union of five block types and only `TextContent`
    carries `.text`; every test here expects a text response, so narrow once here
    instead of asserting `isinstance` at each call site.
    """
    block = result.content[0]
    assert isinstance(block, TextContent), f"expected text content, got {type(block).__name__}"
    return block.text


def _counting_handler(response: httpx2.Response) -> tuple[list[httpx2.Request], Handler]:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return response

    return requests, handler


async def test_base_url_and_prefixes_resolve_from_settings() -> None:
    assert SEARCH_PREFIX == "/api/search"
    assert DATASETS_PREFIX == "/api/datasets"

    requests, handler = _counting_handler(httpx2.Response(200, json={"ok": True}))
    settings = _settings(dfo_apis_url="https://dfo-apis.internal")

    async with _client(handler, settings=settings) as client:
        await client.request("GET", f"{SEARCH_PREFIX}/query", route=f"{SEARCH_PREFIX}/query")
        await client.request("GET", f"{DATASETS_PREFIX}/abc", route=f"{DATASETS_PREFIX}/{{id}}")

    assert str(requests[0].url) == "https://dfo-apis.internal/api/search/query"
    assert str(requests[1].url) == "https://dfo-apis.internal/api/datasets/abc"


async def test_per_call_timeout_override_reaches_the_request() -> None:
    requests, handler = _counting_handler(httpx2.Response(200, json={"ok": True}))

    async with _client(handler) as client:
        await client.request("GET", "/api/datasets", route="/api/datasets", timeout=42.5)

    # Assert all four deadlines, not just `read`. `httpx2` expands a bare
    # float into every deadline at once, so an override that only pinned
    # `read` would silently raise `connect` to match and leave an
    # unreachable dfo-apis hanging for a full read timeout per attempt.
    # `read` is `min(attempt_timeout, remaining)` against the call's
    # wall-clock deadline, and 42.5 exceeds the default retry budget, so
    # `remaining` is chosen: 42.5 less the real time elapsed since the
    # deadline was set, which a loaded machine can stretch past any relative
    # tolerance, and which stays far under a second.
    timeout = requests[0].extensions["timeout"]
    assert timeout["connect"] == 1.0
    assert 41.5 < timeout["read"] <= 42.5
    assert timeout["write"] == 1.0
    assert timeout["pool"] == 1.0


async def test_default_timeout_uses_the_configured_read_deadline() -> None:
    """Omitting `timeout` leaves `upstream_read_timeout_s` in force."""
    requests, handler = _counting_handler(httpx2.Response(200, json={"ok": True}))
    settings = _settings(upstream_connect_timeout_s=2.0, upstream_read_timeout_s=7.0)

    async with _client(handler, settings=settings) as client:
        await client.request("GET", "/api/datasets", route="/api/datasets")

    assert requests[0].extensions["timeout"] == {
        "connect": 2.0,
        "read": 7.0,
        "write": 2.0,
        "pool": 2.0,
    }


async def test_4xx_is_not_retried_even_when_it_claims_to_be_retryable() -> None:
    """A 4xx describes the request, so repeating it unchanged reproduces it."""
    calls, handler = _counting_handler(
        httpx2.Response(400, content=_problem_body(code="bad_request", retryable=True))
    )
    _delays, sleep = _recording_sleep()

    async with _client(handler, sleep=sleep) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", "/api/search/query", route="/api/search/query")

    assert len(calls) == 1
    assert exc_info.value.status == 400


async def test_oversized_problem_body_keeps_its_code() -> None:
    """The body cap bounds what is retained, leaving the parsed code intact."""
    padded = _problem_body(code="over_ceiling", retryable=False, padding="x" * (80 * 1024))
    calls, handler = _counting_handler(httpx2.Response(422, content=padded))

    async with _client(handler) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("POST", "/api/datasets", route="/api/datasets")

    assert len(calls) == 1
    assert exc_info.value.code == "over_ceiling"
    assert len(exc_info.value.body) == 64 * 1024


async def test_extension_members_survive_onto_the_error() -> None:
    """A problem field beyond the six the constructor names lands in `extensions`."""
    calls, handler = _counting_handler(
        httpx2.Response(
            422,
            content=_problem_body(code="default_facet_undefined", retryable=False, axis="country"),
        )
    )

    async with _client(handler) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", "/api/search/query", route="/api/search/query")

    assert len(calls) == 1
    assert exc_info.value.extensions == {"axis": "country"}


async def test_oversized_problem_body_keeps_extensions_empty() -> None:
    """The 64KB cap that bounds `body` also bounds `extensions`, not just what's retained."""
    padded = _problem_body(
        code="over_ceiling", retryable=False, axis="country", padding="x" * (80 * 1024)
    )
    calls, handler = _counting_handler(httpx2.Response(422, content=padded))

    async with _client(handler) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("POST", "/api/datasets", route="/api/datasets")

    assert len(calls) == 1
    assert exc_info.value.code == "over_ceiling"
    assert exc_info.value.extensions == {}


async def test_transport_error_every_attempt_raises_after_max_attempts() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("boom", request=request)

    delays, sleep = _recording_sleep()
    settings = _settings(upstream_max_attempts=3)

    async with _client(handler, settings=settings, sleep=sleep) as client:
        with pytest.raises(UpstreamTransportError) as exc_info:
            await client.request("GET", "/api/search/query", route="/api/search/query")

    assert exc_info.value.status is None
    assert exc_info.value.code is None
    assert len(delays) == settings.upstream_max_attempts - 1


async def test_retry_exhaustion_on_retryable_503_raises() -> None:
    calls, handler = _counting_handler(
        httpx2.Response(503, content=_problem_body(code="temporarily_unavailable", retryable=True))
    )
    _delays, sleep = _recording_sleep()
    settings = _settings(upstream_max_attempts=3)

    async with _client(handler, settings=settings, sleep=sleep) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", "/api/search/query", route="/api/search/query")

    assert len(calls) == settings.upstream_max_attempts
    assert exc_info.value.status == 503
    assert exc_info.value.code == "temporarily_unavailable"


async def test_422_over_ceiling_raises_on_first_attempt_no_retry() -> None:
    calls, handler = _counting_handler(
        httpx2.Response(422, content=_problem_body(code="over_ceiling", retryable=False))
    )

    async with _client(handler) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", "/api/search/query", route="/api/search/query")

    assert len(calls) == 1
    assert exc_info.value.status == 422
    assert exc_info.value.code == "over_ceiling"


async def test_5xx_with_retryable_false_does_not_retry() -> None:
    # 501 capability_unavailable is 5xx, so a 4xx-only guard would wrongly
    # retry it; the `retryable: false` check is what has to catch it.
    calls, handler = _counting_handler(
        httpx2.Response(501, content=_problem_body(code="capability_unavailable", retryable=False))
    )

    async with _client(handler) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", "/api/datasets", route="/api/datasets")

    assert len(calls) == 1
    assert exc_info.value.status == 501
    assert exc_info.value.code == "capability_unavailable"


async def test_upstream_version_changed_does_not_retry_despite_retryable_true() -> None:
    calls, handler = _counting_handler(
        httpx2.Response(503, content=_problem_body(code="upstream_version_changed", retryable=True))
    )

    async with _client(handler) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", "/api/datasets", route="/api/datasets")

    assert len(calls) == 1
    assert exc_info.value.code == "upstream_version_changed"


async def test_unparseable_body_falls_back_to_status_classification() -> None:
    calls, handler = _counting_handler(httpx2.Response(500, content=b"<html>server error</html>"))
    settings = _settings(upstream_max_attempts=1)

    async with _client(handler, settings=settings) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", "/api/search/query", route="/api/search/query")

    assert len(calls) == 1
    assert exc_info.value.status == 500
    assert exc_info.value.code is None
    # 500 falls outside the fixed 5xx fallback set (502/503/504), so it's
    # treated as retryable only when a parsed problem body says so.
    assert exc_info.value.retryable is False


async def test_json_body_without_retryable_falls_back_to_status_classification() -> None:
    """A valid-JSON body that isn't a dfo-apis problem document is treated the same as an unparsable one.

    An edge or proxy can return well-formed JSON, e.g. `{"error":
    "unavailable"}`, on a 503 without it being dfo-apis' own problem+json.
    Trusting it as one would default `retryable` to `False` and silently
    swallow a genuinely transient failure.
    """
    calls, handler = _counting_handler(
        httpx2.Response(503, content=json.dumps({"error": "unavailable"}).encode())
    )
    _delays, sleep = _recording_sleep()
    settings = _settings(upstream_max_attempts=3)

    async with _client(handler, settings=settings, sleep=sleep) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", "/api/search/query", route="/api/search/query")

    assert len(calls) == settings.upstream_max_attempts
    assert exc_info.value.retryable is True
    assert exc_info.value.code is None


async def test_an_attempt_that_outlives_the_budget_stops_the_loop() -> None:
    """A single attempt run past the call's wall-clock budget is cut off rather than left to retry to exhaustion.

    The handler awaits for real, since a synchronous one returns instantly
    regardless of how long `upstream_read_timeout_s` or
    `upstream_retry_budget_s` are configured and could never reproduce an
    attempt actually overrunning the budget.
    """
    calls = 0

    async def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.2)
        return httpx2.Response(200, json={"ok": True})

    _delays, sleep = _recording_sleep()
    settings = _settings(
        upstream_max_attempts=5,
        upstream_backoff_base_s=0.0,
        upstream_read_timeout_s=0.05,
        upstream_retry_budget_s=0.05,
    )

    async with _client(handler, settings=settings, sleep=sleep) as client:
        with pytest.raises(UpstreamTransportError) as exc_info:
            await client.request("GET", "/api/search/query", route="/api/search/query")

    assert calls == 1
    assert exc_info.value.attempts == 1


async def test_mint_route_passes_retry_false_and_is_attempted_once() -> None:
    calls, handler = _counting_handler(
        httpx2.Response(503, content=_problem_body(code="temporarily_unavailable", retryable=True))
    )

    async with _client(handler) as client:
        with pytest.raises(UpstreamError):
            await client.request(
                "POST",
                "/api/datasets",
                route="/api/datasets",
                json={"kind": "csv"},
                retry=False,
                timeout=125.0,
            )

    assert len(calls) == 1
    # `pytest.approx`, not exact, for the same reason as
    # `test_per_call_timeout_override_reaches_the_request`. 125.0 exceeds
    # the default retry budget, so the read timeout actually sent is
    # `remaining`, trimmed by whatever real time already elapsed.
    assert calls[0].extensions["timeout"]["read"] == pytest.approx(125.0)


async def test_backoff_delays_use_full_jitter_up_to_the_exponential_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pin jitter to its upper bound so the recorded delays are deterministic.
    monkeypatch.setattr("data_mcp.client.random.uniform", lambda _low, high: high)
    calls, handler = _counting_handler(
        httpx2.Response(503, content=_problem_body(code="temporarily_unavailable", retryable=True))
    )
    delays, sleep = _recording_sleep()
    settings = _settings(
        upstream_max_attempts=3, upstream_backoff_base_s=1.0, upstream_retry_budget_s=100.0
    )

    async with _client(handler, settings=settings, sleep=sleep) as client:
        with pytest.raises(UpstreamError):
            await client.request("GET", "/api/search/query", route="/api/search/query")

    assert len(calls) == 3
    assert delays == [1.0, 2.0]


async def test_auth_seam_adds_header_default_adds_none() -> None:
    requests, handler = _counting_handler(httpx2.Response(200, json={"ok": True}))

    async with _client(handler) as client:
        await client.request("GET", "/api/search/query", route="/api/search/query")

    assert "authorization" not in requests[0].headers

    requests, handler = _counting_handler(httpx2.Response(200, json={"ok": True}))

    async def auth() -> Mapping[str, str]:
        return {"Authorization": "Bearer test-token"}

    async with _client(handler, auth=auth) as client:
        await client.request("GET", "/api/search/query", route="/api/search/query")

    assert requests[0].headers["authorization"] == "Bearer test-token"


@pytest.mark.parametrize(
    ("link", "sent"),
    [("/api/artifacts/a1", True), ("https://other/x", False), ("//other/x", False)],
    ids=["root-relative", "absolute", "scheme-relative"],
)
async def test_stream_link_sends_the_token_to_the_dfo_apis_origin_alone(
    link: str, sent: bool
) -> None:
    """A link off the dfo-apis origin is refused before a token is minted. On it,
    the token wins over a caller's header of the same name."""
    seen: list[httpx2.Request] = []
    mints = 0

    async def auth() -> Mapping[str, str]:
        nonlocal mints
        mints += 1
        return {TOKEN_HEADER: ID_TOKEN}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200)

    async def fetch() -> None:
        async with client.stream_link(
            link,
            timeout_s=1.0,
            headers={TOKEN_HEADER: "Bearer caller"},
            transport=httpx2.MockTransport(handler),
        ):
            pass

    async with _client(handler, auth=auth) as client:
        if sent:
            await fetch()
        else:
            with pytest.raises(ValueError, match="not on the dfo-apis origin"):
                await fetch()

    assert [r.headers[TOKEN_HEADER] for r in seen] == ([ID_TOKEN] if sent else [])
    assert mints == (1 if sent else 0)


async def test_default_accept_header_requests_the_buffered_body() -> None:
    """The parsed door serves SSE or a buffered body from one route, chosen by `Accept`."""
    requests, handler = _counting_handler(httpx2.Response(200, json={"ok": True}))

    async with _client(handler) as client:
        await client.request("GET", f"{SEARCH_PREFIX}/query", route=f"{SEARCH_PREFIX}/query")

    assert requests[0].headers["accept"] == DEFAULT_ACCEPT_HEADER == "application/json"


async def test_one_call_span_with_child_attempt_spans(
    exported_spans: InMemorySpanExporter,
) -> None:
    responses = iter(
        [
            httpx2.Response(
                503, content=_problem_body(code="temporarily_unavailable", retryable=True)
            ),
            httpx2.Response(200, json={"ok": True}),
        ]
    )

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return next(responses)

    settings = _settings(upstream_max_attempts=3, upstream_backoff_base_s=0.0)

    async with _client(handler, settings=settings) as client:
        await client.request("GET", "/api/datasets/abc", route=f"{DATASETS_PREFIX}/{{id}}")

    spans = exported_spans.get_finished_spans()
    call_spans = [s for s in spans if s.name == "upstream.request"]
    attempt_spans = [s for s in spans if s.name == "upstream.attempt"]

    assert len(call_spans) == 1
    assert len(attempt_spans) == 2

    call_span = call_spans[0]
    assert call_span.attributes is not None
    assert call_span.attributes["url.template"] == f"{DATASETS_PREFIX}/{{id}}"
    assert call_span.attributes["http.response.status_code"] == 200
    assert call_span.attributes["http.request.resend_count"] == 1

    for attempt_span in attempt_spans:
        assert attempt_span.parent is not None
        assert attempt_span.parent.span_id == call_span.context.span_id


async def test_datasets_404_probe_fires_once_and_is_cached() -> None:
    # Case 1: the probe also 404s, so the surface is not deployed.
    probe_calls = {"count": 0}

    def unavailable_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == f"{DATASETS_PREFIX}/capabilities":
            probe_calls["count"] += 1
            return httpx2.Response(404, content=b"{}")
        return httpx2.Response(404, content=_problem_body(code="not_found", retryable=False))

    async with _client(unavailable_handler) as client:
        with pytest.raises(UpstreamSurfaceUnavailable):
            await client.request("GET", "/api/datasets/abc", route=f"{DATASETS_PREFIX}/{{id}}")
        with pytest.raises(UpstreamSurfaceUnavailable):
            await client.request("GET", "/api/datasets/def", route=f"{DATASETS_PREFIX}/{{id}}")

    assert probe_calls["count"] == 1

    # Case 2: the probe succeeds, so the original not_found is a genuine
    # refusal that passes through unchanged.
    probe_calls = {"count": 0}

    def available_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == f"{DATASETS_PREFIX}/capabilities":
            probe_calls["count"] += 1
            return httpx2.Response(200, json={"ok": True})
        return httpx2.Response(404, content=_problem_body(code="not_found", retryable=False))

    async with _client(available_handler) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", "/api/datasets/abc", route=f"{DATASETS_PREFIX}/{{id}}")
        assert not isinstance(exc_info.value, UpstreamSurfaceUnavailable)
        assert exc_info.value.code == "not_found"

        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", "/api/datasets/def", route=f"{DATASETS_PREFIX}/{{id}}")
        assert exc_info.value.code == "not_found"

    assert probe_calls["count"] == 1


@pytest.mark.parametrize("probe_status", [404, 200])
async def test_a_skills_404_probes_the_healthz_door_the_skills_api_serves(
    probe_status: int,
) -> None:
    # The Skills API serves `healthz`, not `capabilities`, so a mounted one
    # answers the probe 200 and an unknown id's refusal passes through.
    probed: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == f"{SKILLS_PREFIX}/healthz":
            probed.append(request.url.path)
            return httpx2.Response(probe_status, content=b"{}")
        return httpx2.Response(404, content=_problem_body(code="not_found", retryable=False))

    async with _client(handler) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", "/api/skills/x", route=f"{SKILLS_PREFIX}/{{skill_id}}")

    assert probed == [f"{SKILLS_PREFIX}/healthz"]
    assert isinstance(exc_info.value, UpstreamSurfaceUnavailable) is (probe_status == 404)


async def test_concurrent_first_refusals_fire_one_probe() -> None:
    """Concurrent calls share a single probe.

    The handler yields to the event loop, so several calls reach the
    unprobed check before any of them records a verdict. A synchronous
    handler never suspends, so it cannot reproduce this.
    """
    probe_calls = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal probe_calls
        await asyncio.sleep(0)
        if request.url.path == f"{DATASETS_PREFIX}/capabilities":
            probe_calls += 1
            await asyncio.sleep(0)
            return httpx2.Response(404, content=b"{}")
        return httpx2.Response(404, content=_problem_body(code="not_found", retryable=False))

    async with _client(handler) as client:
        results = await asyncio.gather(
            *(
                client.request("GET", f"/api/datasets/{n}", route=f"{DATASETS_PREFIX}/{{id}}")
                for n in range(5)
            ),
            return_exceptions=True,
        )

    assert all(isinstance(r, UpstreamSurfaceUnavailable) for r in results)
    assert probe_calls == 1


async def test_concurrent_inconclusive_probes_share_one_round_trip() -> None:
    """Callers piled up on an inconclusive probe (401) share it instead of each firing their own.

    Without sharing, N concurrent 404s would cost N sequential 5s probes,
    all outside the retry budget. A later call, arriving once the wave has
    settled, gets to probe again.
    """
    probe_calls = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal probe_calls
        await asyncio.sleep(0)
        if request.url.path == f"{DATASETS_PREFIX}/capabilities":
            probe_calls += 1
            await asyncio.sleep(0)
            return httpx2.Response(401, content=b"{}")
        return httpx2.Response(404, content=_problem_body(code="not_found", retryable=False))

    async with _client(handler) as client:
        results = await asyncio.gather(
            *(
                client.request("GET", f"/api/datasets/{n}", route=f"{DATASETS_PREFIX}/{{id}}")
                for n in range(5)
            ),
            return_exceptions=True,
        )
        assert probe_calls == 1
        assert all(
            isinstance(r, UpstreamError) and not isinstance(r, UpstreamSurfaceUnavailable)
            for r in results
        )

        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", "/api/datasets/later", route=f"{DATASETS_PREFIX}/{{id}}")
        assert not isinstance(exc_info.value, UpstreamSurfaceUnavailable)

    assert probe_calls == 2


async def test_inconclusive_probe_leaves_the_verdict_open() -> None:
    """A probe answering something other than 404 or 2xx settles nothing.

    Caching "surface present" off a 401 would disable the diagnosis for the
    client's whole lifetime, so the verdict stays open for the next call.
    """
    probe_status = 401
    probe_calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal probe_calls
        if request.url.path == f"{DATASETS_PREFIX}/capabilities":
            probe_calls += 1
            return httpx2.Response(probe_status, content=b"{}")
        return httpx2.Response(404, content=_problem_body(code="not_found", retryable=False))

    async with _client(handler) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", "/api/datasets/abc", route=f"{DATASETS_PREFIX}/{{id}}")
        assert not isinstance(exc_info.value, UpstreamSurfaceUnavailable)
        assert exc_info.value.code == "not_found"

        probe_status = 404
        with pytest.raises(UpstreamSurfaceUnavailable):
            await client.request("GET", "/api/datasets/def", route=f"{DATASETS_PREFIX}/{{id}}")

    assert probe_calls == 2


async def test_probe_carries_its_own_short_deadline() -> None:
    """The probe runs outside the retry budget, so it gets its own deadline."""
    probe_timeouts: list[float] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == f"{DATASETS_PREFIX}/capabilities":
            probe_timeouts.append(request.extensions["timeout"]["read"])
            return httpx2.Response(404, content=b"{}")
        return httpx2.Response(404, content=_problem_body(code="not_found", retryable=False))

    settings = _settings(upstream_read_timeout_s=50.0)

    async with _client(handler, settings=settings) as client:
        with pytest.raises(UpstreamSurfaceUnavailable):
            await client.request("GET", "/api/datasets/abc", route=f"{DATASETS_PREFIX}/{{id}}")

    assert probe_timeouts == [_PROBE_TIMEOUT_S]
    assert settings.upstream_read_timeout_s > _PROBE_TIMEOUT_S


async def test_failing_probe_leaves_the_original_refusal_intact() -> None:
    """A probe that cannot complete must not replace the error it was diagnosing."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == f"{DATASETS_PREFIX}/capabilities":
            raise httpx2.ConnectError("probe unreachable")
        return httpx2.Response(404, content=_problem_body(code="not_found", retryable=False))

    async with _client(handler) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", "/api/datasets/abc", route=f"{DATASETS_PREFIX}/{{id}}")

    assert not isinstance(exc_info.value, UpstreamSurfaceUnavailable)
    assert exc_info.value.code == "not_found"
    assert exc_info.value.status == 404
    assert exc_info.value.trace_id == "trace-1"


async def test_probe_carries_the_calls_auth_headers() -> None:
    """The probe authenticates as the call it diagnoses.

    An unauthenticated probe would answer 401 rather than 404 once a
    credential is in place, and read as "the surface is fine".
    """
    probe_headers: list[str | None] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == f"{DATASETS_PREFIX}/capabilities":
            probe_headers.append(request.headers.get("authorization"))
            return httpx2.Response(404, content=b"{}")
        return httpx2.Response(404, content=_problem_body(code="not_found", retryable=False))

    async def auth() -> Mapping[str, str]:
        return {"authorization": "Bearer test-token"}

    async with _client(handler, auth=auth) as client:
        with pytest.raises(UpstreamSurfaceUnavailable):
            await client.request("GET", "/api/datasets/abc", route=f"{DATASETS_PREFIX}/{{id}}")

    assert probe_headers == ["Bearer test-token"]


async def test_call_span_carries_the_status_when_the_call_fails(
    exported_spans: InMemorySpanExporter,
) -> None:
    """The failure path is the one worth tracing, so it carries the status too."""
    _, handler = _counting_handler(
        httpx2.Response(422, content=_problem_body(code="over_ceiling", retryable=False))
    )

    async with _client(handler) as client:
        with pytest.raises(UpstreamError):
            await client.request("POST", "/api/datasets", route=DATASETS_PREFIX)

    call_span = next(s for s in exported_spans.get_finished_spans() if s.name == "upstream.request")
    assert call_span.attributes is not None
    assert call_span.attributes["http.response.status_code"] == 422
    assert call_span.attributes["url.template"] == DATASETS_PREFIX


async def test_graph_prefix_gets_the_same_surface_probe() -> None:
    """Graph ships behind a flag of its own, so it needs the Datasets treatment."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == f"{GRAPH_PREFIX}/capabilities":
            return httpx2.Response(404, content=b"{}")
        return httpx2.Response(404, content=_problem_body(code="not_found", retryable=False))

    async with _client(handler) as client:
        with pytest.raises(UpstreamSurfaceUnavailable) as exc_info:
            await client.request("POST", f"{GRAPH_PREFIX}/resolve", route=f"{GRAPH_PREFIX}/resolve")

    assert exc_info.value.prefix == GRAPH_PREFIX


async def test_a_verdict_on_one_surface_does_not_settle_another() -> None:
    """Each prefix carries its own verdict, so one missing surface says nothing about the next."""
    probed: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/capabilities"):
            probed.append(request.url.path)
            missing = request.url.path.startswith(GRAPH_PREFIX)
            return httpx2.Response(404 if missing else 200, content=b"{}")
        return httpx2.Response(404, content=_problem_body(code="not_found", retryable=False))

    async with _client(handler) as client:
        with pytest.raises(UpstreamSurfaceUnavailable):
            await client.request("POST", f"{GRAPH_PREFIX}/resolve", route=f"{GRAPH_PREFIX}/resolve")

        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", "/api/datasets/abc", route=f"{DATASETS_PREFIX}/{{id}}")
        assert not isinstance(exc_info.value, UpstreamSurfaceUnavailable)

    assert probed == [f"{GRAPH_PREFIX}/capabilities", f"{DATASETS_PREFIX}/capabilities"]


async def test_search_not_found_is_never_probed() -> None:
    """Search mounts unconditionally, so a `not_found` there is always genuine."""
    calls, handler = _counting_handler(
        httpx2.Response(404, content=_problem_body(code="not_found", retryable=False))
    )

    async with _client(handler) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request(
                "POST", f"{SEARCH_PREFIX}/resolve", route=f"{SEARCH_PREFIX}/resolve"
            )

    assert not isinstance(exc_info.value, UpstreamSurfaceUnavailable)
    assert len(calls) == 1


async def test_error_carries_what_the_call_already_spent() -> None:
    """`retry_after_delay`'s interval derives from the attempts and the time spent."""
    calls, handler = _counting_handler(
        httpx2.Response(503, content=_problem_body(code="temporarily_unavailable", retryable=True))
    )
    _delays, sleep = _recording_sleep()
    settings = _settings(upstream_max_attempts=3)

    async with _client(handler, settings=settings, sleep=sleep) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("GET", f"{SEARCH_PREFIX}/query", route=f"{SEARCH_PREFIX}/query")

    assert exc_info.value.attempts == len(calls) == 3
    # A lower bound of zero would hold for any float, including the 0.0 the
    # field is initialised to, so it would pass even if the retry loop never
    # filled it in. The clock is monotonic and three round trips separate the
    # two readings, so a strict bound is what actually proves it was set.
    assert exc_info.value.elapsed_s > 0.0


async def test_a_near_zero_budget_still_raises_a_typed_error() -> None:
    """Every failure leaves this client as an `UpstreamError`, misconfiguration included.

    `Settings` refuses a literal `0.0` for `upstream_retry_budget_s` /
    `upstream_read_timeout_s` (both are wait bounds, per `Settings`' own docstring),
    so this exercises the smallest value it still accepts: real wall-clock time
    spent constructing `budget` already exceeds it, the same "expired before any
    attempt" path a literal zero used to reach.
    """
    calls, handler = _counting_handler(httpx2.Response(200, json={"ok": True}))
    settings = _settings(upstream_retry_budget_s=1e-9, upstream_read_timeout_s=1e-9)

    async with _client(handler, settings=settings) as client:
        with pytest.raises(UpstreamTransportError):
            await client.request("GET", f"{SEARCH_PREFIX}/query", route=f"{SEARCH_PREFIX}/query")

    assert calls == []


async def test_a_first_attempt_failure_reports_one_attempt() -> None:
    """A refusal raised immediately is distinguishable from one that exhausted retries."""
    calls, handler = _counting_handler(
        httpx2.Response(422, content=_problem_body(code="over_ceiling", retryable=False))
    )

    async with _client(handler) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request("POST", "/api/datasets", route=DATASETS_PREFIX)

    assert len(calls) == 1
    assert exc_info.value.attempts == 1


async def test_surface_unavailable_carries_the_calls_attempts_and_elapsed_time() -> None:
    """`UpstreamSurfaceUnavailable` reports what the retry loop actually spent, not a fresh instance's defaults."""
    responses = iter(
        [
            httpx2.Response(
                503, content=_problem_body(code="temporarily_unavailable", retryable=True)
            ),
            httpx2.Response(404, content=_problem_body(code="not_found", retryable=False)),
        ]
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == f"{DATASETS_PREFIX}/capabilities":
            return httpx2.Response(404, content=b"{}")
        return next(responses)

    _delays, sleep = _recording_sleep()
    settings = _settings(upstream_max_attempts=3)

    async with _client(handler, settings=settings, sleep=sleep) as client:
        with pytest.raises(UpstreamSurfaceUnavailable) as exc_info:
            await client.request("GET", "/api/datasets/abc", route=f"{DATASETS_PREFIX}/{{id}}")

    assert exc_info.value.attempts == 2
    assert exc_info.value.elapsed_s > 0.0


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_deadline_shorter_than_the_attempt_timeout_cuts_the_read_timeout() -> None:
    requests, handler = _counting_handler(httpx2.Response(200, json={"ok": True}))
    deadline = CallDeadline.after(2.0, clock=_FakeClock())

    async with _client(handler) as client:
        await client.request(
            "GET", "/api/search/query", route="/api/search/query", timeout=50.0, deadline=deadline
        )

    # Exact: the fake clock stands still, so the deadline leaves exactly 2s,
    # well inside the call's own ~50s budget.
    assert requests[0].extensions["timeout"]["read"] == 2.0


async def test_deadline_shortens_a_non_retryable_mint_below_its_own_timeout() -> None:
    requests, handler = _counting_handler(httpx2.Response(200, json={"ok": True}))
    deadline = CallDeadline.after(30.0, clock=_FakeClock())

    async with _client(handler) as client:
        await client.request(
            "POST",
            "/api/datasets",
            route="/api/datasets",
            retry=False,
            timeout=125.0,
            deadline=deadline,
        )

    assert requests[0].extensions["timeout"]["read"] == 30.0


async def test_deadline_stops_retries_the_retry_budget_would_allow() -> None:
    """Each attempt spends a second on the fake clock, so a 2.5s deadline admits three."""
    clock = _FakeClock()
    calls = 0

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        clock.now += 1.0
        return httpx2.Response(
            503, content=_problem_body(code="temporarily_unavailable", retryable=True)
        )

    _delays, sleep = _recording_sleep()
    settings = _settings(upstream_max_attempts=10, upstream_backoff_base_s=0.0)
    deadline = CallDeadline.after(2.5, clock=clock)

    async with _client(handler, settings=settings, sleep=sleep) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request(
                "GET", "/api/search/query", route="/api/search/query", deadline=deadline
            )

    assert calls == 3
    assert exc_info.value.attempts == 3
    # The last real refusal survives; the deadline only stopped a fourth try.
    assert exc_info.value.status == 503


async def test_a_passed_deadline_raises_a_transport_error_without_calling_upstream() -> None:
    requests, handler = _counting_handler(httpx2.Response(200, json={"ok": True}))
    clock = _FakeClock()
    deadline = CallDeadline.after(1.0, clock=clock)
    clock.now += 2.0

    async with _client(handler) as client:
        with pytest.raises(UpstreamTransportError) as exc_info:
            await client.request(
                "GET", "/api/search/query", route="/api/search/query", deadline=deadline
            )

    assert requests == []
    assert exc_info.value.attempts == 0
    assert exc_info.value.detail == "the call deadline expired before any attempt"


async def test_an_attempt_that_outlives_the_deadline_raises_a_transport_error() -> None:
    """The handler awaits for real, since `MockTransport` ignores httpx2's own timeouts."""
    calls = 0

    async def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(1.0)
        return httpx2.Response(200, json={"ok": True})

    async with _client(handler) as client:
        with pytest.raises(UpstreamTransportError) as exc_info:
            await client.request(
                "GET",
                "/api/search/query",
                route="/api/search/query",
                deadline=CallDeadline.after(0.05),
            )

    assert calls == 1
    assert exc_info.value.detail == "attempt exceeded the call deadline"
    assert exc_info.value.elapsed_s is not None
    assert exc_info.value.elapsed_s < 0.5


async def test_probe_is_skipped_once_the_deadline_has_passed() -> None:
    clock = _FakeClock()
    probed: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == f"{DATASETS_PREFIX}/capabilities":
            probed.append(request.url.path)
            return httpx2.Response(404, content=b"{}")
        clock.now += 10.0
        return httpx2.Response(404, content=_problem_body(code="not_found", retryable=False))

    deadline = CallDeadline.after(5.0, clock=clock)

    async with _client(handler) as client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.request(
                "GET", "/api/datasets/abc", route=f"{DATASETS_PREFIX}/{{id}}", deadline=deadline
            )

    assert probed == []
    assert not isinstance(exc_info.value, UpstreamSurfaceUnavailable)
    assert exc_info.value.code == "not_found"


async def test_probe_timeout_is_clipped_to_the_deadline() -> None:
    probe_timeouts: list[float] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == f"{DATASETS_PREFIX}/capabilities":
            probe_timeouts.append(request.extensions["timeout"]["read"])
            return httpx2.Response(404, content=b"{}")
        return httpx2.Response(404, content=_problem_body(code="not_found", retryable=False))

    deadline = CallDeadline.after(2.0, clock=_FakeClock())

    async with _client(handler) as client:
        with pytest.raises(UpstreamSurfaceUnavailable):
            await client.request(
                "GET", "/api/datasets/abc", route=f"{DATASETS_PREFIX}/{{id}}", deadline=deadline
            )

    assert probe_timeouts == [2.0]
    assert _PROBE_TIMEOUT_S > 2.0


async def test_a_probe_that_outlives_the_deadline_leaves_the_refusal_intact() -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == f"{DATASETS_PREFIX}/capabilities":
            await asyncio.sleep(1.0)
            return httpx2.Response(404, content=b"{}")
        return httpx2.Response(404, content=_problem_body(code="not_found", retryable=False))

    async with _client(handler) as client:
        started = time.monotonic()
        with pytest.raises(UpstreamError) as exc_info:
            await client.request(
                "GET",
                "/api/datasets/abc",
                route=f"{DATASETS_PREFIX}/{{id}}",
                deadline=CallDeadline.after(0.1),
            )
        elapsed = time.monotonic() - started

    assert not isinstance(exc_info.value, UpstreamSurfaceUnavailable)
    assert exc_info.value.code == "not_found"
    assert elapsed < 0.5


async def test_waiting_on_another_callers_probe_ends_with_the_deadline() -> None:
    """A call queued behind a slow probe gives up at its own deadline, not the probe's end."""
    probe_calls = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal probe_calls
        if request.url.path == f"{DATASETS_PREFIX}/capabilities":
            probe_calls += 1
            await asyncio.sleep(0.5)
            return httpx2.Response(404, content=b"{}")
        return httpx2.Response(404, content=_problem_body(code="not_found", retryable=False))

    async def bounded_call(client: UpstreamClient) -> tuple[UpstreamError, float]:
        # Let the unbounded call take the probe lock first.
        await asyncio.sleep(0.05)
        started = time.monotonic()
        try:
            await client.request(
                "GET",
                "/api/datasets/def",
                route=f"{DATASETS_PREFIX}/{{id}}",
                deadline=CallDeadline.after(0.1),
            )
        except UpstreamError as exc:
            return exc, time.monotonic() - started
        raise AssertionError("expected an UpstreamError")

    async with _client(handler) as client:
        unbounded, bounded = await asyncio.gather(
            client.request("GET", "/api/datasets/abc", route=f"{DATASETS_PREFIX}/{{id}}"),
            bounded_call(client),
            return_exceptions=True,
        )

    assert isinstance(unbounded, UpstreamSurfaceUnavailable)
    assert isinstance(bounded, tuple)
    bounded_error, bounded_elapsed = bounded
    assert not isinstance(bounded_error, UpstreamSurfaceUnavailable)
    assert bounded_error.code == "not_found"
    assert bounded_elapsed < 0.3
    assert probe_calls == 1
