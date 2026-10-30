"""Async HTTP client for the apis surfaces.

One `httpx2.AsyncClient` behind a retry loop bounded by wall-clock, problem-body
parsing that keeps a refusal's `code` intact for #26, a span per call, and a
lazy probe that separates an unmounted surface from a missing id.

The retry budget is wall-clock and covers all attempts of one call together.
A caller's `CallDeadline` shortens it and bounds the probe as well.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from typing import Any, Self

import httpx2
from common.apis_auth import ApisAuthError, AuthProvider
from common.caller import ADDRESS_CALLER, CALLER_HEADER, CALLER_PRINCIPAL_HEADER, current_caller
from opentelemetry import trace

from data_mcp.deadline import CallDeadline
from data_mcp.errors import (
    UpstreamAuthUnavailable,
    UpstreamError,
    UpstreamSurfaceUnavailable,
    UpstreamTransportError,
)
from data_mcp.settings import Settings

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# The suffix of the `httpcore` trace event that marks a request body written
# out, over HTTP/1.1 (`http11.`) or HTTP/2 (`http2.`).
_REQUEST_SENT_EVENT = "send_request_body.complete"


def _sent_trace(on_sent: Callable[[], None]) -> Callable[[str, dict[str, Any]], Awaitable[None]]:
    """An `httpcore` trace extension that calls `on_sent` once the request body is written."""

    async def trace_event(event: str, _info: dict[str, Any]) -> None:
        if event.endswith(_REQUEST_SENT_EVENT):
            on_sent()

    return trace_event


SEARCH_PREFIX = "/api/search"
DATASETS_PREFIX = "/api/datasets"
# apis mounts Graph once that surface lands.
GRAPH_PREFIX = "/api/graph"
SKILLS_PREFIX = "/api/skills"

# apis mounts these behind env flags, so an unmounted one answers
# `not_found` on every path beneath it. Each maps to the path its probe reads,
# one the surface serves whenever it is mounted: Datasets and Graph serve
# `capabilities`, the Skills API serves `healthz`. Search mounts
# unconditionally and needs no probe.
GATED_PREFIXES: Mapping[str, str] = {
    DATASETS_PREFIX: f"{DATASETS_PREFIX}/capabilities",
    GRAPH_PREFIX: f"{GRAPH_PREFIX}/capabilities",
    SKILLS_PREFIX: f"{SKILLS_PREFIX}/healthz",
}

# Codes that reproduce identically on retry, whatever `retryable` the body
# carries. `upstream_version_changed` is a 503 with `retryable: true`
# (doors/mint.py:423, materialise/errors.py:148, version.py:149), which #26
# handles as `refresh_then_retry`. `operation_failed` is the poll route's
# answer for an operation that has ended: its `retryable` says whether
# submitting the mint again could succeed, and every further poll of the same
# operation returns the same problem.
NON_RETRYABLE_CODES = frozenset({"upstream_version_changed", "operation_failed"})

# Used only when no problem body parsed. 501 stays out deliberately. A parsed
# `capability_unavailable` carries its own `retryable: false`, and an
# unparsable 501 is better left non-retryable than guessed at.
_RETRYABLE_STATUS_FALLBACK = frozenset({502, 503, 504})

# Bounds what is kept on the error, not what is parsed.
_ERROR_BODY_CAP_BYTES = 64 * 1024

# The six fields `UpstreamError.__init__` already reads off a problem, plus
# RFC 9457's other core members (`type`, `status`, `instance`), so nothing
# from the problem document lands twice under two different names.
_KNOWN_PROBLEM_FIELDS = frozenset(
    {
        "type",
        "status",
        "code",
        "title",
        "detail",
        "instance",
        "retryable",
        "trace_id",
        "capability_version",
    }
)

# The probe runs after the retry loop, outside `upstream_retry_budget_s`, so
# it carries a timeout of its own, clipped to a caller's deadline.
_PROBE_TIMEOUT_S = 5.0

# apis' parsed door serves SSE or a buffered body from one route, chosen
# by `Accept`. Every caller here wants the buffered body, so this rides as a
# client default rather than a `request()` parameter — httpx2 merges a
# per-request header over a client default, so a future caller that genuinely
# needs SSE can still override it.
DEFAULT_ACCEPT_HEADER = "application/json"

# The redirects `stream_link` follows. dfo-apis answers a stable artifact link
# with 302 to a short-lived signed URL.
_LINK_REDIRECT_STATUSES = frozenset({302, 303, 307})

SleepFn = Callable[[float], Awaitable[None]]


async def _default_sleep(seconds: float) -> None:
    """Real sleep; tests inject a fake to keep backoff assertions fast."""
    await asyncio.sleep(seconds)


def _time_left(budget: CallDeadline, deadline: CallDeadline | None) -> float:
    """Seconds left before the retry budget or the caller's deadline, whichever ends first."""
    if deadline is None:
        return budget.remaining()
    return min(budget.remaining(), deadline.remaining())


def _spent_limit(budget: CallDeadline, deadline: CallDeadline | None, budget_s: float) -> str:
    """Name the bound that ran out, for an error's `detail`."""
    if deadline is not None and deadline.remaining() <= budget.remaining():
        return "the call deadline"
    return f"the call's {budget_s:.1f}s retry budget"


def _parse_problem_json(body: bytes) -> dict[str, Any] | None:
    """Parse `body` as an apis service problem document, ignoring `Content-Type`.

    Returns `None` unless `retryable` is present as a real bool, which is what
    separates an apis service problem (`dfo_platform/problem.py`) from a proxy's
    `{"error": "unavailable"}` on the same status. `isinstance` rather than
    truthiness, since `isinstance(True, int)` would admit `retryable: 1`.
    Callers fall back to `_RETRYABLE_STATUS_FALLBACK` on `None`.
    """
    if not body:
        return None
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    if not isinstance(parsed.get("retryable"), bool):
        return None
    return parsed


def _retry_after_s(response: httpx2.Response) -> float | None:
    """The response's `Retry-After` as seconds, or `None` when it names none.

    apis writes delay-seconds. An HTTP-date form reads as `None`.
    """
    value = response.headers.get("Retry-After", "").strip()
    return float(value) if value.isdigit() else None


def is_retryable(error: UpstreamError) -> bool:
    """Whether repeating the request that failed with `error` could succeed.

    A transport error, or a 5xx reporting `retryable: true` with a `code`
    outside `NON_RETRYABLE_CODES`. The client's own retry loop and any caller
    that repeats a request itself use this one test.
    """
    if isinstance(error, UpstreamTransportError):
        return True
    if error.code in NON_RETRYABLE_CODES:
        return False
    # A 4xx describes the request, so repeating it reproduces it. 501
    # `capability_unavailable` is 5xx and falls to the `retryable` check.
    if error.status is not None and error.status < 500:
        return False
    return error.retryable


class UpstreamClient:
    """Async client over apis' surfaces.

    Holds one `httpx2.AsyncClient` and, per gated surface, the cached
    verdict of its probe once that has fired. Use as an async context
    manager or call `aclose()` directly to release the connection pool.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        auth: AuthProvider | None = None,
        sleep: SleepFn = _default_sleep,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        """Build a client from `settings`.

        Args:
            settings: The service's `Settings`, supplying the base URL and
                the `upstream_*` timeout/retry knobs.
            auth: Async credential provider, called once per call rather than
                per attempt. The surface probe sends its headers too. The
                deployed client gets `common.apis_auth.apis_auth_provider`'s
                Cloud Run ID token. Injects nothing by default.
            sleep: Backoff sleep, which tests replace to avoid waiting.
            transport: `httpx2` transport override, `httpx2.MockTransport` in
                tests.
        """
        self._retry_budget_s = settings.upstream_retry_budget_s
        self._max_attempts = settings.upstream_max_attempts
        self._backoff_base_s = settings.upstream_backoff_base_s
        self._read_timeout_s = settings.upstream_read_timeout_s
        self._connect_timeout_s = settings.upstream_connect_timeout_s
        self._auth = auth
        self._sleep = sleep
        self._public_origin = settings.dfo_site_origin
        self._caller_principal = (
            None
            if settings.dfo_caller_principal is None
            else settings.dfo_caller_principal.get_secret_value()
        )
        self._client = httpx2.AsyncClient(
            base_url=settings.dfo_apis_url,
            timeout=httpx2.Timeout(
                settings.upstream_connect_timeout_s, read=settings.upstream_read_timeout_s
            ),
            transport=transport,
            headers={"Accept": DEFAULT_ACCEPT_HEADER},
        )
        # Prefix -> whether apis has that surface unmounted. Absent means
        # unprobed. See `_probe_surface`.
        self._surface_missing: dict[str, bool] = {}
        self._probe_lock = asyncio.Lock()
        self._probe_epoch = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release the underlying connection pool."""
        await self._client.aclose()

    def public_url(self, link: str) -> str:
        """`link`, a root-relative apis path, as a caller outside this service reaches it.

        On `Settings.dfo_site_origin` when set, the website that proxies apis'
        paths with its own ID token, else on the apis origin this client calls.
        """
        base = self._client.base_url if self._public_origin is None else self._public_origin
        return str(httpx2.URL(base).join(link))

    def caller_headers(self) -> dict[str, str]:
        """The headers that name the current request's caller to dfo-apis.

        The caller principal and the caller, when this process holds the principal
        (`Settings.dfo_caller_principal`) and the website named the caller by address
        (`x-dfo-caller: ip:<digest>`), so dfo-apis counts that client's allowance.
        Empty otherwise, and dfo-apis counts the call against this service.
        """
        caller = current_caller()
        if self._caller_principal is None or caller is None:
            return {}
        if not ADDRESS_CALLER.fullmatch(caller):
            return {}
        return {CALLER_PRINCIPAL_HEADER: self._caller_principal, CALLER_HEADER: caller}

    @asynccontextmanager
    async def stream_link(
        self,
        link: str,
        *,
        timeout_s: float,
        headers: Mapping[str, str] | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> AsyncIterator[httpx2.Response]:
        """GET `link`, a root-relative apis path, and yield the streamed response.

        The first hop goes to the apis origin this client calls, with the auth
        provider's headers, which win over a header of the same name in `headers`.
        A 302, 303 or 307 from it is followed once, with `headers` alone. httpx
        keeps custom headers across a redirect it follows itself, so automatic
        following would hand the ID token to the signed-URL host. Any other answer,
        a second redirect included, is yielded as it is.

        Args:
            link: The root-relative path, e.g. an artifact's stable link.
            timeout_s: Each hop's timeout.
            headers: Headers for both hops.
            transport: `httpx2` transport override, `httpx2.MockTransport` in
                tests. Separate from this client's own, which carries the
                snapshot and mint calls.

        Raises:
            ValueError: `link` resolves off the apis origin, as an absolute or
                scheme-relative URL for another host does. Nothing was sent and no
                token was minted.
            httpx2.TransportError: a hop failed, or the auth provider could not
                mint a token, in which case nothing was sent.
        """
        base = self._client.base_url
        url = base.join(link)
        if (url.scheme, url.host, url.port) != (base.scheme, base.host, base.port):
            raise ValueError(f"link {link!r} is not on the dfo-apis origin")
        try:
            auth = dict(await self._auth()) if self._auth else {}
        except ApisAuthError as exc:
            raise httpx2.TransportError(f"no dfo-apis ID token: {exc}") from exc
        hop_headers = dict(headers or {})
        async with httpx2.AsyncClient(
            transport=transport, timeout=httpx2.Timeout(timeout_s)
        ) as client:
            response = await client.send(
                client.build_request("GET", url, headers={**hop_headers, **auth}),
                stream=True,
            )
            if response.status_code in _LINK_REDIRECT_STATUSES and "Location" in response.headers:
                await response.aclose()
                target = response.url.join(response.headers["Location"])
                response = await client.send(
                    client.build_request("GET", target, headers=hop_headers), stream=True
                )
            try:
                yield response
            finally:
                await response.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        route: str,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        retry: bool = True,
        deadline: CallDeadline | None = None,
        headers: Mapping[str, str] | None = None,
        on_sent: Callable[[], None] | None = None,
    ) -> httpx2.Response:
        """Issue one apis call, retried within a wall-clock budget.

        Args:
            method: HTTP method, e.g. "GET" or "POST".
            path: The interpolated request path, e.g. `/api/datasets/abc123`.
            route: The path *template*, e.g. `/api/datasets/{id}`. Kept
                separate from `path` because it lands on the span as
                `url.template`, which bounds span cardinality.
            json: JSON request body.
            params: Query parameters.
            timeout: Read-timeout override in seconds, defaulting to
                `Settings.upstream_read_timeout_s`. Mint passes `125.0`.
            retry: Whether transient failures may be retried. Mint passes
                `False`, because a timed-out mint upstream refuses
                cancellation and holds one of four concurrency permits until
                it finishes.
            deadline: The caller's deadline for the whole call. Every
                attempt's timeout and the retry budget are cut to the time it
                leaves, and the surface probe is clipped to it, or skipped
                once it has passed. `None` bounds the call by the retry budget
                alone.
            headers: Headers sent on every attempt beside the auth provider's,
                which win over a header of the same name.
            on_sent: Called each time an attempt's request body has been
                written to the connection (`httpcore`'s `send_request_body`
                trace event), before any response is read.

        Returns:
            The 2xx `httpx2.Response`.

        Raises:
            UpstreamError: apis returned a non-2xx response.
            UpstreamAuthUnavailable: the auth provider could not mint a token,
                so nothing was sent. A subclass of `UpstreamTransportError`.
            UpstreamTransportError: no response was ever received, including
                when `deadline` passed before or during an attempt.
            UpstreamSurfaceUnavailable: a gated-prefix call 404d and the probe
                confirmed the surface is unmounted.
        """
        attempt_timeout = timeout if timeout is not None else self._read_timeout_s
        try:
            auth = dict(await self._auth()) if self._auth else {}
        except ApisAuthError as exc:
            raise UpstreamAuthUnavailable(detail=f"no dfo-apis ID token: {exc}") from exc
        sent_headers = {**(headers or {}), **auth}
        # The budget bounds retrying, not one attempt a caller asked for.
        # Mint's 125s exceeds the 120s budget and must survive it.
        call_budget_s = max(self._retry_budget_s, attempt_timeout)

        with tracer.start_as_current_span(
            "upstream.request",
            attributes={"http.request.method": method, "url.template": route},
        ) as call_span:
            start = time.monotonic()
            budget = CallDeadline(expires_at=start + call_budget_s)
            attempt = 0
            result: UpstreamError | None = None

            while True:
                remaining = _time_left(budget, deadline)
                if remaining <= 0:
                    break
                attempt += 1
                with tracer.start_as_current_span(
                    "upstream.attempt",
                    attributes={"http.request.resend_count": attempt - 1},
                ) as attempt_span:
                    try:
                        async with asyncio.timeout(remaining):
                            attempted = await self._attempt(
                                method,
                                path,
                                json=json,
                                params=params,
                                headers=sent_headers,
                                timeout=min(attempt_timeout, remaining),
                                on_sent=on_sent,
                            )
                    except TimeoutError:
                        # `asyncio.timeout` bounds connect, write and pool as
                        # well as read, so an attempt that keeps trickling
                        # bytes cannot outlive the budget. Builtin
                        # `TimeoutError`, distinct from
                        # `httpx2.TimeoutException`.
                        limit = _spent_limit(budget, deadline, call_budget_s)
                        attempted = UpstreamTransportError(detail=f"attempt exceeded {limit}")
                    if isinstance(attempted, httpx2.Response):
                        attempt_span.set_attribute(
                            "http.response.status_code", attempted.status_code
                        )
                        call_span.set_attribute("http.response.status_code", attempted.status_code)
                        call_span.set_attribute("http.request.resend_count", attempt - 1)
                        return attempted
                    result = attempted
                    if result.status is not None:
                        attempt_span.set_attribute("http.response.status_code", result.status)

                should_retry = (
                    self._should_retry(result, retry=retry) and attempt < self._max_attempts
                )
                backoff = 0.0
                if should_retry:
                    backoff = random.uniform(0, self._backoff_base_s * (2 ** (attempt - 1)))
                    if backoff >= _time_left(budget, deadline):
                        should_retry = False

                if not should_retry:
                    break
                await self._sleep(backoff)

            if result is None:
                # Only on a zero or negative budget, or a deadline already
                # passed. A typed error keeps every failure leaving here an
                # `UpstreamError`.
                limit = _spent_limit(budget, deadline, call_budget_s)
                result = UpstreamTransportError(detail=f"{limit} expired before any attempt")
            call_span.set_attribute("http.request.resend_count", attempt - 1)
            if result.status is not None:
                call_span.set_attribute("http.response.status_code", result.status)
            # #26 derives `retry_after_delay`'s interval from these.
            result.attempts = attempt
            result.elapsed_s = time.monotonic() - start
            raise await self._translate_gated_surface_404(
                path, result, headers=auth, deadline=deadline
            )

    async def _attempt(
        self,
        method: str,
        path: str,
        *,
        json: Any | None,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str],
        timeout: float,
        on_sent: Callable[[], None] | None = None,
    ) -> httpx2.Response | UpstreamError:
        """Issue one HTTP attempt, returning the response or a classified error.

        Never raises for ordinary failure modes (transport errors, non-2xx
        responses). Those become the return value, so `request()`'s retry
        loop decides what to do with them.
        """
        try:
            response = await self._client.request(
                method,
                path,
                json=json,
                params=params,
                headers=headers,
                timeout=self._attempt_timeout(timeout),
                extensions=None if on_sent is None else {"trace": _sent_trace(on_sent)},
            )
        except httpx2.TransportError as exc:
            return UpstreamTransportError(detail=str(exc))
        if response.is_success:
            return response
        return self._to_upstream_error(response)

    def _attempt_timeout(self, read: float) -> httpx2.Timeout:
        """Build the per-attempt timeout, varying only the read deadline.

        `httpx2` expands a bare float into all four deadlines, so passing
        mint's 125s would discard `upstream_connect_timeout_s` and leave an
        unreachable apis hanging a full read timeout per attempt.
        """
        return httpx2.Timeout(self._connect_timeout_s, read=read)

    def _to_upstream_error(self, response: httpx2.Response) -> UpstreamError:
        """Classify a non-2xx response into an `UpstreamError`.

        Falls back to `_RETRYABLE_STATUS_FALLBACK` when the body is not a
        problem document, leaving `code` as `None`. `code` is only ever the
        problem's own `code`, read off the wire. A `cause` member (a failed
        operation names the refusal that failed it) lands in `extensions`
        with every other extension member.
        """
        # Parse in full, cap only what is kept. Capping first truncates an
        # oversized problem mid-JSON and loses the `code` #26 reads.
        problem = _parse_problem_json(response.content)
        body = response.content[:_ERROR_BODY_CAP_BYTES]
        if problem is None:
            return UpstreamError(
                status=response.status_code,
                code=None,
                retryable=response.status_code in _RETRYABLE_STATUS_FALLBACK,
                body=body,
            )
        # Only within the body cap: an oversized problem's extensions would
        # otherwise carry the full, uncapped dict onto the exception even
        # though `body` itself got truncated above.
        extensions = (
            {k: v for k, v in problem.items() if k not in _KNOWN_PROBLEM_FIELDS}
            if len(response.content) <= _ERROR_BODY_CAP_BYTES
            else {}
        )
        return UpstreamError(
            status=response.status_code,
            code=problem.get("code"),
            title=problem.get("title"),
            detail=problem.get("detail"),
            retryable=bool(problem.get("retryable", False)),
            trace_id=problem.get("trace_id"),
            capability_version=problem.get("capability_version"),
            body=body,
            extensions=extensions,
            retry_after_s=_retry_after_s(response),
        )

    def _should_retry(self, error: UpstreamError, *, retry: bool) -> bool:
        """Whether `error` is eligible for another attempt on a call that did not opt out."""
        return retry and is_retryable(error)

    async def _translate_gated_surface_404(
        self,
        path: str,
        error: UpstreamError,
        *,
        headers: Mapping[str, str],
        deadline: CallDeadline | None,
    ) -> UpstreamError:
        """Separate an unmounted surface from a genuine "no such id" refusal.

        Both answer `not_found`, `retryable: false`, so a probe of the path
        `GATED_PREFIXES` names for the prefix is what tells them apart. It
        runs on the error path only, never at startup.

        `headers` carries the call's own auth headers, so the probe
        authenticates as the call it diagnoses. An unauthenticated probe would
        answer 401 rather than 404 once the auth seam is filled, and read as
        "the surface is fine".

        A failed probe returns the original error untouched. This exists to
        improve an error, so it must never replace one with its own failure.
        So does a probe that `deadline` leaves no time for: the wait for
        another caller's probe and the probe itself both end with it.
        """
        if error.code != "not_found":
            return error
        prefix = next((p for p in GATED_PREFIXES if path.startswith(p)), None)
        if prefix is None:
            return error
        if prefix not in self._surface_missing:
            if deadline is None:
                await self._probe_surface(prefix, headers=headers, timeout_s=_PROBE_TIMEOUT_S)
            elif (time_left := deadline.remaining()) > 0:
                with suppress(TimeoutError):
                    async with asyncio.timeout(time_left):
                        await self._probe_surface(
                            prefix, headers=headers, timeout_s=min(_PROBE_TIMEOUT_S, time_left)
                        )
        if not self._surface_missing.get(prefix):
            return error
        return UpstreamSurfaceUnavailable(
            prefix=prefix,
            probe_path=GATED_PREFIXES[prefix],
            trace_id=error.trace_id,
            capability_version=error.capability_version,
            attempts=error.attempts,
            elapsed_s=error.elapsed_s,
        )

    async def _probe_surface(
        self, prefix: str, *, headers: Mapping[str, str], timeout_s: float
    ) -> None:
        """Probe `prefix` once, caching a conclusive verdict and sharing an inconclusive one.

        The lock stops concurrent callers each firing a probe. The epoch stops
        them firing one *in turn* once the first comes back inconclusive, which
        would cost a 5s round trip per caller. A call arriving after the wave
        snapshots a fresh epoch and stays free to probe again.

        An inconclusive outcome records no verdict. Recording "present" off a
        401 would disable the diagnosis for the client's whole lifetime. A
        probe cancelled by its caller's deadline counts as inconclusive and
        still closes the wave.
        """
        epoch = self._probe_epoch
        async with self._probe_lock:
            if prefix in self._surface_missing:
                return
            if self._probe_epoch != epoch:
                return
            try:
                probe = await self._client.get(
                    GATED_PREFIXES[prefix],
                    headers=headers,
                    timeout=self._attempt_timeout(timeout_s),
                )
            except httpx2.TransportError:
                logger.debug("%s surface probe failed; leaving the refusal as it arrived", prefix)
                return
            finally:
                self._probe_epoch += 1
            if probe.status_code == 404:
                self._surface_missing[prefix] = True
            elif probe.is_success:
                self._surface_missing[prefix] = False
