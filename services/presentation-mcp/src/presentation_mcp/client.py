"""Async HTTP `Assembler` implementation over apis' presentation surface.

Copies `data_mcp.client.UpstreamClient`'s shape: one `httpx2.AsyncClient`
behind a retry loop bounded by wall-clock, problem-body parsing that keeps a
refusal's `code` intact, a span per call, injectable transport and sleep, and
`route` kept separate from the interpolated path for span cardinality. No
surface probe: the server's startup `release()` fetch already proves the
surface is mounted before any tool call reaches this client.

The `caller` kwarg on `create_component`/`revise_component` is
presentation-mcp's own reading of the transport a write arrived on
(`server._transport_caller`), and decides which caller headers the write
carries (`PlatformAssembler._card_write_headers`). The kind itself never goes
on the wire.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import json
import logging
import random
import re
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Self

import httpx2
from common.apis_auth import ApisAuthError, AuthProvider
from common.caller import ADDRESS_CALLER, CALLER_HEADER, CALLER_PRINCIPAL_HEADER, current_caller
from common.materialisation import FrameRef, get_store
from common.row_budget import ROW_BUDGET_RETRY_AFTER_S, RowBudgetExhausted
from opentelemetry import trace
from pydantic import ValidationError

from presentation_mcp.errors import PlatformError, PlatformTransportError
from presentation_mcp.frame_publish import PublishBody, PublishBodyNotBuildable, publish_body_for
from presentation_mcp.platform_contract import (
    Caller,
    NormalFormConflict,
    Refusal,
    RetryMode,
    WriteReply,
)
from presentation_mcp.settings import Settings

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

_COMPONENTS_ROUTE = "/api/presentation/components"
_COMPONENT_ROUTE = "/api/presentation/components/{component_id}"
_FRAME_ROUTE = "/api/presentation/frames/{spec_id}/{content_version}"
# The newest version of a `spec_` frame, with its `spec_id` and
# `content_version` beside the pair route's body.
_SPEC_FRAME_ROUTE = "/api/presentation/frames/{spec_id}"
_RELEASE_ROUTE = "/api/presentation/release"
# apis' own `datasets_api` surface, not `presentation_api`. See
# `_ensure_published_pair`'s own docstring for why this client calls it.
_PUBLISH_ROUTE = "/api/datasets/frames/publish"

# These match apis' presentation path-parameter checks. They deliberately
# admit well-formed unknown ids so the platform, rather than this client,
# decides whether a referenced component or frame exists.
_COMPONENT_ID_PATTERN = re.compile(r"^cmp_[A-Za-z0-9_-]{1,64}$")
_WRITE_TOKEN_PATTERN = re.compile(r"^cmw_[A-Za-z0-9_-]{1,64}$")
_SPEC_ID_PATTERN = re.compile(r"^(?:spec|ops)_[a-z0-9]{1,64}$")
_CONTENT_VERSION_PATTERN = re.compile(r"^(?:cv|ocv)_[a-z0-9]{1,64}$")

# The largest request Cloud Run admits, measured on the wire. apis serves
# HTTP/1, and Cloud Run rejects an HTTP/1 request over 32 MiB with a bare 413
# before apis sees it. The publish body is gzipped, so this bounds the
# compressed bytes; `_publish_pair` refuses a larger one locally, with the figures.
_PUBLISH_BODY_CEILING_BYTES = 32 * 1024 * 1024

# The apis service's `MAX_PUBLISH_BODY_BYTES`: the most bytes its publish door decodes
# from a gzipped body. `_publish_pair` refuses an uncompressed body over it, so
# the door never refuses for size a body this client sends.
_PUBLISH_DECODED_BODY_CEILING_BYTES = 129_000_000

# Level 1 compresses a 52 MiB body in about 0.23 s; higher levels cost more
# time for a small gain on JSON this repetitive.
_PUBLISH_GZIP_LEVEL = 1

# The `constraint_id` values of the 409 `publish_conflict` apis answers when a
# publish carries a recipe that disagrees with the recipes it holds: the pair has a
# recipe with another normal form or another replay key, or another frame holds
# this replay key. apis records the frame before it compares recipes, so the
# pair is published either way and keeps the recipe apis already holds for it,
# if any (`_publish_pair`).
_NORMAL_FORM_CONFLICT = "recipe_requires_matching_normal_form_hash"
_RECIPE_CONFLICT_CONSTRAINTS = frozenset(
    {
        _NORMAL_FORM_CONFLICT,
        "recipe_requires_matching_replay_key",
        "replay_key_requires_one_frame",
    }
)

# Sent on the three calls whose apis route reads it: the publish call
# (`_ensure_published_pair`'s POST to `_PUBLISH_ROUTE`, which
# `datasets_api.doors.publish` authorizes by it), and a first-party component
# create and revise. apis spends one unit of a per-caller daily card allowance
# on each create and revise, and exempts a principal holding the publish grant,
# so Studio's conversations, which share the harness's one address, spend
# none. No other route reads it, so no other call carries it.
PRINCIPAL_HEADER = "x-dfo-principal"

# Sent only on a revise: the component's write token, which apis checks
# before it merges a patch.
WRITE_TOKEN_HEADER = "x-dfo-write-token"

# The code apis answers a revise with when its write token is missing or
# wrong (403, not retryable). The apis service's own detail and fix name its REST
# routes and the write-token header, which a caller of this server cannot
# use, so `_refusal` answers this code with the tool-level text below.
WRITE_REFUSED = "write_refused"
_WRITE_REFUSED_DETAIL = (
    "The write_token sent does not grant revising this component: it is not the "
    "token this component's create returned, or the component predates write tokens. "
    "To go on, call write_component without component_id. describe_component on this "
    "id returns the intent to start from."
)
_WRITE_REFUSED_FIX = (
    "Create a new component: call write_component without component_id. "
    "describe_component on this id returns its intent, which the new "
    "component can start from."
)

# The parsed door serves SSE or a buffered body from one route, chosen by
# `Accept`. Every caller here wants the buffered body.
DEFAULT_ACCEPT_HEADER = "application/json"

SleepFn = Callable[[float], Awaitable[None]]


async def _default_sleep(seconds: float) -> None:
    """Real sleep; tests inject a fake to keep backoff assertions fast."""
    await asyncio.sleep(seconds)


def _new_write_token() -> str:
    """Mint a component write token: `cmw_` and 22 url-safe characters (128 bits)."""
    return f"cmw_{secrets.token_urlsafe(16)}"


def _component_id_refusal(component_id: str) -> Refusal | None:
    """A local refusal for an invalid `component_id` path segment.

    Guards `get_component` and `revise_component`, both of which format
    `_COMPONENT_ROUTE` from this id before issuing a request. This check
    catches empty ids and strings that would alter the route, so neither call
    ever reaches the platform with an invalid component id.
    """
    if _COMPONENT_ID_PATTERN.fullmatch(component_id):
        return None
    return Refusal(
        error="request_invalid",
        retry=RetryMode.DO_NOT_RETRY,
        detail="component_id must be a valid id a previous write_component reply returned.",
        pointer="/component_id",
    )


def _write_token_refusal(write_token: str) -> Refusal | None:
    """A local refusal for a malformed `write_token`, which would otherwise
    reach the wire as a header value."""
    if _WRITE_TOKEN_PATTERN.fullmatch(write_token):
        return None
    return Refusal(
        error="request_invalid",
        retry=RetryMode.DO_NOT_RETRY,
        detail="write_token must be the token this component's create returned.",
        fix=_WRITE_REFUSED_FIX,
        pointer="/write_token",
    )


def _publish_body_refusal(
    *, body_bytes: int, rows: int, ceiling: int, ceiling_kind: str, what: str
) -> Refusal:
    """The refusal for a publish body over one of its two byte ceilings.

    `publish_body_bytes` bounds the gzipped bytes on the wire
    (`_PUBLISH_BODY_CEILING_BYTES`) and `publish_body_decoded_bytes` bounds the
    uncompressed JSON (`_PUBLISH_DECODED_BODY_CEILING_BYTES`); `what` names the
    measured bytes in the message. The figures come from this frame's own body:
    its bytes a row, in the same unit as `body_bytes`, and the rows of that
    shape that fit under `ceiling`.
    """
    bytes_per_row = body_bytes / max(rows, 1)
    fitting_rows = int(ceiling / bytes_per_row)
    return Refusal(
        error="over_ceiling",
        retry=RetryMode.CORRECT_AND_RETRY,
        detail=(
            f"This frame is too large to publish to the platform: its {rows:,} rows encode to "
            f"{body_bytes:,} {what} bytes, over the {ceiling:,}-byte limit. "
            f"At {bytes_per_row:,.0f} bytes a row, about {fitting_rows:,} rows of this shape "
            "fit. Narrow the frame to fewer rows."
        ),
        fix={
            "ceiling_kind": ceiling_kind,
            "body_bytes": body_bytes,
            "ceiling": ceiling,
        },
    )


async def _gzipped_publish_body(published: PublishBody) -> bytes | Refusal:
    """The gzipped publish body, or the refusal for a body over either ceiling.

    Compressed once, so every retry resends the same bytes. Off the event loop:
    compressing a large body takes a fraction of a second and would stall every
    call this instance serves. Run through the body's row claim, so a cancelled
    publish leaves the rows held until the compression thread ends.
    """
    if len(published.data) > _PUBLISH_DECODED_BODY_CEILING_BYTES:
        return _publish_body_refusal(
            body_bytes=len(published.data),
            rows=published.rows,
            ceiling=_PUBLISH_DECODED_BODY_CEILING_BYTES,
            ceiling_kind="publish_body_decoded_bytes",
            what="uncompressed",
        )
    body = await published.hold_through(
        asyncio.to_thread(gzip.compress, published.data, _PUBLISH_GZIP_LEVEL, mtime=0)
    )
    if len(body) > _PUBLISH_BODY_CEILING_BYTES:
        return _publish_body_refusal(
            body_bytes=len(body),
            rows=published.rows,
            ceiling=_PUBLISH_BODY_CEILING_BYTES,
            ceiling_kind="publish_body_bytes",
            what="compressed",
        )
    return body


# What a publish answers when this process was given no principal to publish as.
_PUBLISH_NOT_CONFIGURED = Refusal(
    error="publish_not_configured",
    retry=RetryMode.DO_NOT_RETRY,
    detail=(
        "This deployment has no datasets_first_party_principal "
        "configured, so it cannot publish this frame to the platform."
    ),
)


def _version_created(response: httpx2.Response) -> bool:
    """Whether the publish door's answer says this call recorded the version:
    its `published.created`. An answer that states neither way reads as recorded."""
    try:
        body = response.json()
    except ValueError:
        return True
    published = body.get("published") if isinstance(body, Mapping) else None
    return not isinstance(published, Mapping) or published.get("created") is not False


def _row_budget_refusal(exc: RowBudgetExhausted) -> Refusal:
    """The refusal for a publish whose frame rows the process's frame row
    budget (`common.row_budget`) has no room for."""
    return Refusal(
        error="service_unavailable",
        retry=RetryMode.RETRY_AFTER_DELAY,
        detail=(
            "This server is already working on as many frame rows as its memory allows at "
            f"once ({exc.max_rows:,}). This frame has {exc.requested:,} rows and "
            f"{exc.free:,} are free. Retry after {ROW_BUDGET_RETRY_AFTER_S:.1f}s."
        ),
        fix={
            "service": "presentation-mcp",
            "rows_requested": exc.requested,
            "rows_free": exc.free,
            "rows_in_flight_max": exc.max_rows,
        },
        retry_after_s=ROW_BUDGET_RETRY_AFTER_S,
    )


def _frame_ref_incomplete(spec_id: str) -> Refusal:
    """The refusal for a `frame_ref` that names `spec_id` without its
    `content_version` where `spec_id` is not a `spec_` id. The platform
    resolves only a `spec_` id to its newest version."""
    return Refusal(
        error="frame_ref_incomplete",
        retry=RetryMode.CORRECT_AND_RETRY,
        detail=(
            f"frame_ref names {spec_id} without its content_version. Only a spec_ id "
            "can be passed alone. Pass the frame_ref the call that made this frame returned."
        ),
        fix=(
            "Pass frame_ref as the spec_id and content_version pair the call that made "
            "the frame returned."
        ),
        pointer="/frame_ref/content_version",
    )


def _resolves_alone(spec_id: str) -> bool:
    """Whether the platform resolves `spec_id` with no `content_version`: a
    `spec_` id names a fetch, and its newest version is the one drawn."""
    return spec_id.startswith("spec_")


def _frame_ref_or_refusal(frame_ref: Mapping[str, Any]) -> tuple[str, str | None] | Refusal:
    """Return valid frame path segments, or a local refusal before routing.

    `spec_id` and `content_version` are interpolated into `_FRAME_ROUTE`, so
    accepting route syntax in either field could select a different platform
    resource after URL normalization. `content_version` is `None` for a
    `spec_` id passed alone, which `_SPEC_FRAME_ROUTE` resolves.
    """
    spec_id = frame_ref.get("spec_id")
    content_version = frame_ref.get("content_version")
    if not isinstance(spec_id, str) or not _SPEC_ID_PATTERN.fullmatch(spec_id):
        return Refusal(
            error="request_invalid",
            retry=RetryMode.DO_NOT_RETRY,
            detail="frame_ref.spec_id must be a valid id a previous frame reference returned.",
            pointer="/frame_ref/spec_id",
        )
    if content_version is None:
        return (spec_id, None) if _resolves_alone(spec_id) else _frame_ref_incomplete(spec_id)
    if not isinstance(content_version, str) or not _CONTENT_VERSION_PATTERN.fullmatch(
        content_version
    ):
        return Refusal(
            error="request_invalid",
            retry=RetryMode.DO_NOT_RETRY,
            detail=(
                "frame_ref.content_version must be a valid id a previous frame reference returned."
            ),
            pointer="/frame_ref/content_version",
        )
    return spec_id, content_version


def _parse_problem_json(body: bytes) -> dict[str, Any] | None:
    """Parse `body` as an apis service problem document, ignoring `Content-Type`.

    Returns `None` unless `retryable` is present as a real bool, which is
    what separates an apis service problem (`dfo_platform.problem.ProblemDetails`)
    from a proxy's `{"error": "unavailable"}` on the same status.
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


def _render_alternatives(alternatives: Sequence[Mapping[str, Any]]) -> str:
    """Each legal value as `label (ref)`, or its ref alone where the two match.

    A retry's select names values by ref (`country/USA`), so the ref rides
    beside the label the model reads.
    """
    rendered = []
    for alt in alternatives:
        label, ref = str(alt["label"]), str(alt["ref"])
        rendered.append(ref if label == ref else f"{label} ({ref})")
    return ", ".join(rendered)


def _render_fix(exc: PlatformError) -> str | None:
    """A short, model-facing rendering of what would satisfy the request instead.

    `constraint_id` names the violated rule, `alternatives` is a flat list of
    legal values for the axis that triggered the refusal, and `issues` is one
    such list per axis/member the refusal concerns. Each list gets its own
    more-values-exist note from its own `truncated` flag: the top-level
    `truncated` for `alternatives`, and each issue's own `truncated` for that
    issue's list -- the two are computed independently on the wire and do not
    imply one another.
    """
    parts: list[str] = []
    if exc.constraint_id:
        parts.append(f"violates constraint {exc.constraint_id!r}")
    if exc.alternatives:
        labels = _render_alternatives(exc.alternatives)
        if exc.truncated:
            labels += " (more values exist than are listed)"
        parts.append(f"legal values: {labels}")
    if exc.issues:
        for issue in exc.issues:
            axis = issue.get("axis") or issue.get("member") or issue.get("pointer") or "field"
            alternatives = issue.get("alternatives") or []
            if alternatives:
                labels = _render_alternatives(alternatives)
                if issue.get("truncated"):
                    labels += " (more values exist than are listed)"
                parts.append(f"{axis}: {labels}")
    if not parts:
        return None
    return "; ".join(parts)


class PlatformAssembler:
    """`Assembler` implementation calling apis' `/api/presentation/*` surface.

    Use as an async context manager or call `aclose()` directly to release
    the connection pool.
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
                per attempt. The deployed assembler gets
                `common.apis_auth.apis_auth_provider`'s Cloud Run ID token.
                Injects nothing by default.
            sleep: Backoff sleep, which tests replace to avoid waiting.
            transport: `httpx2` transport override, `httpx2.MockTransport` in
                tests.
        """
        self._retry_budget_s = settings.upstream_retry_budget_s
        self._max_attempts = settings.upstream_max_attempts
        self._backoff_base_s = settings.upstream_backoff_base_s
        self._read_timeout_s = settings.upstream_read_timeout_s
        self._connect_timeout_s = settings.upstream_connect_timeout_s
        self._publish_principal = settings.datasets_first_party_principal
        self._caller_principal = (
            None
            if settings.dfo_caller_principal is None
            else settings.dfo_caller_principal.get_secret_value()
        )
        self._max_frame_rows = settings.max_frame_rows
        self._auth = auth
        self._sleep = sleep
        self._client = httpx2.AsyncClient(
            base_url=settings.dfo_apis_url,
            timeout=httpx2.Timeout(
                settings.upstream_connect_timeout_s, read=settings.upstream_read_timeout_s
            ),
            transport=transport,
            headers={"Accept": DEFAULT_ACCEPT_HEADER},
        )
        # code -> RetryMode, cached from the last successful `release()`
        # call. The server lifespan always calls `release()` before any tool
        # call, so this is populated in practice by the time a refusal needs
        # to compose a retry mode.
        self._retry_modes: dict[str, RetryMode] = {}
        # `(spec_id, content_version)` pairs this process has already
        # published successfully -- see `_ensure_published_pair`'s own
        # docstring for why a per-process memo, not a stronger cache, is the
        # right amount of idempotency here. One `PlatformAssembler` lives
        # for the whole process (the server lifespan builds exactly one),
        # so this is a plain instance set, not a holder of its own.
        self._published: set[tuple[str, str]] = set()
        # One `asyncio.Lock` per pair ever seen, single-flighting concurrent
        # first publishes of the same pair onto one network call instead of
        # each racing its own (`_ensure_published_pair`'s own docstring).
        # Never pruned, same as `_published` above: both grow with the
        # number of distinct pairs this process ever publishes, not with
        # call volume.
        self._publish_locks: dict[tuple[str, str], asyncio.Lock] = {}

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release the underlying connection pool."""
        await self._client.aclose()

    async def create_component(
        self,
        *,
        intent: Mapping[str, Any],
        caller: Caller,
    ) -> WriteReply | Refusal:
        """Create a component, retrying transport failures and retryable 5xx replies.

        The write token is minted here, once per call, and sent in the body of
        every attempt. apis treats it as the create's idempotency key: a
        repeat carrying the same token and intent returns the draft the first
        attempt stored, so a create is retried like any other call. One
        duplicate path remains: when every attempt fails, the model may call
        `write_component` again, which mints a fresh token and can store a
        second draft.

        `caller` picks the headers that name who the card counts against
        (`_card_write_headers`).
        """
        write_token = _new_write_token()
        # A create's intent is complete. The platform resolves a `spec_` id
        # with no content_version to its newest version and stores that pair,
        # so only another id alone is refused. A revise's patch merges into
        # the stored frame_ref, so it goes to the platform as is.
        frame_ref = intent.get("frame_ref")
        if (
            isinstance(frame_ref, Mapping)
            and isinstance(spec_id := frame_ref.get("spec_id"), str)
            and frame_ref.get("content_version") is None
            and not _resolves_alone(spec_id)
        ):
            return _frame_ref_incomplete(spec_id)
        # Publish before create: `create_component` never validates
        # `intent["frame_ref"]` locally beyond that (see `_presentable`'s own
        # docstring), so this is the only chance to publish an analysis-mcp
        # result before the platform's own `ops_unsupported` refusal would
        # otherwise answer for it.
        presentable = await self._presentable(intent)
        if isinstance(presentable, Refusal):
            return presentable
        intent = presentable
        try:
            response = await self.request(
                "POST",
                _COMPONENTS_ROUTE,
                route=_COMPONENTS_ROUTE,
                json={"intent": dict(intent), "write_token": write_token},
                headers=self._card_write_headers(caller),
            )
        except PlatformError as exc:
            return self._refusal(exc)
        return self._write_reply(response.json())

    async def revise_component(
        self,
        *,
        component_id: str,
        patch: Mapping[str, Any],
        write_token: str,
        caller: Caller,
    ) -> WriteReply | Refusal:
        # `caller` picks the headers that name who the card counts against
        # (`_card_write_headers`). A revise names the component
        # it targets and merges the same patch into the same stored intent
        # every time -- a repeated identical patch is the platform's
        # structural no-op, so it is safe to retry.
        refusal = _component_id_refusal(component_id) or _write_token_refusal(write_token)
        if refusal is not None:
            return refusal
        # Same reasoning as `create_component`'s own publish hook: a patch
        # naming a new `frame_ref` is never locally validated either, so
        # publish it before it reaches the platform.
        presentable = await self._presentable(patch)
        if isinstance(presentable, Refusal):
            return presentable
        patch = presentable
        path = _COMPONENT_ROUTE.format(component_id=component_id)
        try:
            response = await self.request(
                "PATCH",
                path,
                route=_COMPONENT_ROUTE,
                json={"intent": dict(patch)},
                headers={**self._card_write_headers(caller), WRITE_TOKEN_HEADER: write_token},
            )
        except PlatformError as exc:
            return self._refusal(exc)
        return self._write_reply(response.json())

    def _card_write_headers(self, caller: Caller) -> dict[str, str]:
        """The headers that tell dfo-apis whose card allowance a create or revise spends.

        A first-party write, Studio's, carries the publish principal, which
        dfo-apis exempts. An external write carries the caller principal and
        the caller the website named, when this process holds the principal
        and the request's `x-dfo-caller` is an `ip:` digest, so dfo-apis
        counts that client. Any other external write carries neither, and
        dfo-apis counts it against the address it sees for this service.
        """
        if caller is Caller.FIRST_PARTY:
            principal = self._publish_principal
            return {} if principal is None else {PRINCIPAL_HEADER: principal}
        named = current_caller()
        if self._caller_principal is None or named is None or not ADDRESS_CALLER.fullmatch(named):
            return {}
        return {CALLER_PRINCIPAL_HEADER: self._caller_principal, CALLER_HEADER: named}

    async def get_frame_view(
        self,
        *,
        frame_ref: Mapping[str, Any],
        dimension: str | None = None,
        query: str | None = None,
    ) -> Mapping[str, Any] | Refusal:
        frame_ref_or_refusal = _frame_ref_or_refusal(frame_ref)
        if isinstance(frame_ref_or_refusal, Refusal):
            return frame_ref_or_refusal
        spec_id, content_version = frame_ref_or_refusal
        if content_version is None:
            # apis resolves the newest version and names it in the body.
            route, path = _SPEC_FRAME_ROUTE, _SPEC_FRAME_ROUTE.format(spec_id=spec_id)
        else:
            spec_id, content_version = await self._platform_pair(spec_id, content_version)
            publish_refusal = await self._ensure_published_pair(spec_id, content_version)
            if publish_refusal is not None:
                return publish_refusal
            route = _FRAME_ROUTE
            path = _FRAME_ROUTE.format(spec_id=spec_id, content_version=content_version)
        params = {
            k: v for k, v in {"dimension": dimension, "query": query}.items() if v is not None
        }
        try:
            response = await self.request("GET", path, route=route, params=params or None)
        except PlatformError as exc:
            return self._refusal(exc)
        return response.json()

    async def get_component(
        self,
        *,
        component_id: str,
        revision: int | None = None,
    ) -> Mapping[str, Any] | Refusal:
        refusal = _component_id_refusal(component_id)
        if refusal is not None:
            return refusal
        path = _COMPONENT_ROUTE.format(component_id=component_id)
        params = {"revision": revision} if revision is not None else None
        try:
            response = await self.request(
                "GET",
                path,
                route=_COMPONENT_ROUTE,
                params=params,
            )
        except PlatformError as exc:
            return self._refusal(exc)
        return response.json()

    async def release(self) -> Mapping[str, Any]:
        """The platform's release document, fetched fresh every call.

        Raises `PlatformError`/`PlatformTransportError` on failure -- the one
        `Assembler` call with no `Refusal` path. On success, caches this
        document's `retry_modes` table for `_refusal` to compose a retry mode
        from.
        """
        response = await self.request("GET", _RELEASE_ROUTE, route=_RELEASE_ROUTE)
        document = response.json()
        retry_modes = document.get("retry_modes")
        if isinstance(retry_modes, Mapping):
            parsed = {code: self._parse_retry_mode(mode) for code, mode in retry_modes.items()}
            self._retry_modes = {code: mode for code, mode in parsed.items() if mode is not None}
        return document

    @staticmethod
    def _parse_retry_mode(value: Any) -> RetryMode | None:
        try:
            return RetryMode(value)
        except ValueError:
            return None

    def _write_reply(self, body: Mapping[str, Any]) -> WriteReply:
        """The wire's create/revise reply, mapped onto `WriteReply`.

        `catalog_version`, `composition_id`, and `composition_revision` are
        deliberately omitted from `WriteReply`; `catalog_version` is logged
        here so the dropped field still leaves a trace. `resolved_payload`
        is kept verbatim for the server to carry in the tool result's
        `_meta`. `write_token` is present on a create's reply alone.

        The wire's `affordance` is the REST door's line naming its `PATCH`
        and `GET` routes. This service omits it too. The equivalent
        guidance lives in `tool_schema.py`'s parameter descriptions, which
        reach a model through `tools/list`.
        """
        logger.debug("platform call minted catalog_version %s", body.get("catalog_version"))
        return WriteReply(
            component_id=body["component_id"],
            revision=body["revision"],
            changed=tuple(body.get("changed") or ()),
            fact_summary=body["fact_summary"],
            resolved_payload=body.get("resolved_payload"),
            write_token=body.get("write_token"),
        )

    def _refusal(self, exc: PlatformError) -> Refusal:
        """Translate a `PlatformError` into the model-facing `Refusal`."""
        if isinstance(exc, PlatformTransportError):
            return Refusal(
                error="transient_unavailable",
                retry=RetryMode.RETRY_AFTER_DELAY,
                detail=exc.detail or "The platform call failed before a response arrived.",
            )
        if exc.code is None:
            # No problem body parsed: classified by status alone. `detail`
            # holds prose describing the failure for an integrator to read.
            # The status code is logged here for tracing.
            logger.debug("platform call answered %s with no problem body", exc.status)
            if exc.status is not None and exc.status < 500:
                detail = exc.detail or "The platform rejected this call and returned no reason."
                return Refusal(error="request_invalid", retry=RetryMode.DO_NOT_RETRY, detail=detail)
            detail = exc.detail or "The platform failed to answer this call."
            return Refusal(
                error="transient_unavailable", retry=RetryMode.RETRY_AFTER_DELAY, detail=detail
            )
        retry = RetryMode.DO_NOT_RETRY if not exc.retryable else self._retry_mode_for(exc.code)
        if exc.code == WRITE_REFUSED:
            detail, fix = _WRITE_REFUSED_DETAIL, _WRITE_REFUSED_FIX
        else:
            detail, fix = exc.detail or "The platform refused this call.", _render_fix(exc)
        if isinstance(exc.trace_id, str) and exc.trace_id:
            logger.warning(
                "platform refused a call code=%s status=%s platform_trace_id=%s",
                exc.code,
                exc.status,
                exc.trace_id,
            )
        return Refusal(error=exc.code, retry=retry, detail=detail, fix=fix, pointer=exc.pointer)

    def _retry_mode_for(self, code: str) -> RetryMode:
        """The retry mode for a retryable `code`, from the cached `release()` table.

        Falls back to `RETRY_AFTER_DELAY` for a code the table doesn't name,
        or before the table has ever been populated.
        """
        return self._retry_modes.get(code, RetryMode.RETRY_AFTER_DELAY)

    async def _presentable(self, body: Mapping[str, Any]) -> Mapping[str, Any] | Refusal:
        """`body` (an intent or a patch) with its `frame_ref` turned into the
        pair the platform holds and published when it needs publishing --
        `create_component`/`revise_component`'s own hook. Unlike
        `get_frame_view`'s own `frame_ref` argument, the one under `body` is
        never locally validated at all (`platform_contract.py`'s own
        docstring: intents and patches stay opaque to this client, `frame_ref`
        being the one exception `get_frame_view` reads). Reads only the two
        fields it needs to decide whether to act -- `body` itself for anything
        that isn't a mapping carrying both as strings, deferring entirely to
        whatever the platform's own intent validation makes of it.
        """
        frame_ref = body.get("frame_ref")
        if not isinstance(frame_ref, Mapping):
            return body
        spec_id = frame_ref.get("spec_id")
        content_version = frame_ref.get("content_version")
        if not isinstance(spec_id, str) or not isinstance(content_version, str):
            return body
        platform_spec_id, platform_content_version = await self._platform_pair(
            spec_id, content_version
        )
        publish_refusal = await self._ensure_published_pair(
            platform_spec_id, platform_content_version
        )
        if publish_refusal is not None:
            return publish_refusal
        if (platform_spec_id, platform_content_version) == (spec_id, content_version):
            return body
        return {
            **body,
            "frame_ref": {
                **frame_ref,
                "spec_id": platform_spec_id,
                "content_version": platform_content_version,
            },
        }

    async def _platform_pair(self, spec_id: str, content_version: str) -> tuple[str, str]:
        """The pair apis holds for the frame `{spec_id, content_version}`.

        `get_data` reports the pair its frame is held under in the shared
        store. A fetch with `place_types`, other `columns`, a `stat_vars` sum or
        a `breakdown` is held under a pair of its own, which apis never
        stored, so the platform refuses it as `frame_unavailable`. The mint
        record the store keeps beside that frame names the pair apis does
        hold (`MintRecord.dataset`): the rendering's own dataset version, or
        the summed version apis mints for a collapse. That pair is
        returned when it differs from the one given, and the platform draws
        the dataset version's canonical frame.

        Any other pair is returned as given: an `ops_` pair, a `spec_` pair
        equal to its dataset pair, a `spec_` pair with no mint record, and a
        value that is not a pair the store could hold. The record is written
        best effort, so a derived pair whose write was given up still refuses
        `frame_unavailable` until `get_data` is called for it again. A lookup
        that fails for any reason is logged and the pair returned as given, so
        a derived pair then refuses `frame_unavailable` as it does with no
        record.
        """
        if not spec_id.startswith("spec_"):
            return spec_id, content_version
        try:
            ref = FrameRef(spec_id=spec_id, content_version=content_version)
        except ValidationError:
            return spec_id, content_version
        try:
            record = await get_store().get_mint_record(ref)
        except Exception:
            # Broad on purpose: the store's failure modes belong to its backend
            # (the storage client, its auth and its transport), which this module
            # does not name, and the lookup only ever improves a call.
            logger.warning(
                "mint record of %s/%s unreadable; sending the pair as given",
                spec_id,
                content_version,
                exc_info=True,
            )
            return spec_id, content_version
        if record is None or record.frame == record.dataset:
            return spec_id, content_version
        return record.dataset.spec_id, record.dataset.content_version

    async def _ensure_published_pair(self, spec_id: str, content_version: str) -> Refusal | None:
        """Ensure `{spec_id, content_version}` is published to apis'
        `datasets_api` surface before the caller's own platform request
        that needs it, when it needs publishing at all.

        A `spec_`-prefixed pair is left alone: it names a Datasets
        snapshot, already resolvable through the platform's ordinary
        mint/pin path, and posting it to the publish door (`ops_`-only on
        the wire) would just be refused. An `ops_`/`ocv_` pair analysis-mcp's
        verbs return is one dfo-apis' analysis runner computed and
        published itself, so apis holds it already and the shared store holds
        nothing to publish. A pair an earlier in-process analysis-mcp wrote to
        the shared store is published from there when this process has a
        principal. Without one, the caller's own platform call proceeds and
        apis answers for the pair.

        Memoised per process in `self._published`: a repeat call for a pair
        already published this process's lifetime is a same-request no-op,
        never a second network call -- the route is idempotent server-side
        too (a second publish of identical bytes is safe), but a session
        that both describes and then writes the same ref in one turn is the
        common case this memo is for, and skipping the call outright there
        is cheaper than relying on the server's own dedup. Scoped to this
        process: a second `PlatformAssembler` (this server restarted, or one
        day horizontally scaled) republishes once, which is exactly as safe.
        Single-flighted per pair via `self._publish_locks`, so two concurrent
        first calls for the same pair still publish it only once, rather than
        each racing to build a body and post it before the other records the
        memo above.

        Returns a `Refusal` only when publishing was needed and failed; the
        caller's own platform call proceeds either way otherwise. See
        `frame_publish.publish_body_for` for what "cannot be published"
        means before any network call, and this method's own `except
        PlatformError` branch for what it means after one.
        """
        if not spec_id.startswith("ops_"):
            return None
        key = (spec_id, content_version)
        if key in self._published:
            return None

        if self._publish_principal is None:
            # Nothing can be published without a principal, and a runner's
            # pair needs no publishing, since the platform answers for the pair.
            return None

        # Single-flighted per pair: concurrent first callers for the same
        # pair await the same lock rather than each building a body and
        # posting it independently. The route tolerates the redundant POSTs
        # this would otherwise cause (idempotent server-side), so this is a
        # load/latency fix, not a correctness one -- see this method's own
        # docstring.
        lock = self._publish_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._published:
                # Someone else already won this pair's publish while this
                # call waited on the lock above.
                return None
            published = await self._publish_pair(spec_id=spec_id, content_version=content_version)
            return published if isinstance(published, Refusal) else None

    async def publish_frame(
        self, *, spec_id: str, content_version: str, recipe: Mapping[str, Any]
    ) -> bool | Refusal | NormalFormConflict:
        """Publish the derived frame `{spec_id, content_version}` to apis
        with `recipe`, for a caller that writes no component.

        This is the publish `_ensure_published_pair` runs ahead of a component
        write, with the caller's recipe in the body in place of one built here.
        It posts whether or not this process has published the pair before,
        since an earlier publish may have carried no recipe, and apis
        answers a repeat with the version it holds. It takes the pair's publish
        lock, and a success is remembered as `_ensure_published_pair`'s is.

        Returns:
            `True` when apis recorded the version with this call, and `False`
            when it already held it, including when it kept another recipe of the
            same normal form for the pair. A `NormalFormConflict` when it kept a
            recipe of another normal form, so no card of `recipe` is drawn on the
            frame. A `Refusal` when nothing
            was published: no principal is configured, the store holds no frame
            under the pair, the body cannot be built or is over a ceiling, or
            apis refused it.
        """
        if self._publish_principal is None:
            return _PUBLISH_NOT_CONFIGURED
        lock = self._publish_locks.setdefault((spec_id, content_version), asyncio.Lock())
        async with lock:
            published = await self._publish_pair(
                spec_id=spec_id, content_version=content_version, recipe=recipe
            )
        if published is None:
            return Refusal(
                error="request_invalid",
                retry=RetryMode.DO_NOT_RETRY,
                detail=(
                    "Publishing this frame to the platform failed: no frame is held "
                    f"for {spec_id!r}/{content_version!r}."
                ),
            )
        return published

    async def _publish_pair(
        self, *, spec_id: str, content_version: str, recipe: Mapping[str, Any] | None = None
    ) -> bool | Refusal | NormalFormConflict | None:
        """Build and post the publish-door body for `(spec_id, content_version)`,
        called under the pair's publish lock once a principal is known to be
        configured. `_ensure_published_pair` and `publish_frame` are its callers;
        see their docstrings for what each makes of the answer.

        The body carries `recipe`, or without one the frame's recipe when one
        can be built (`frame_publish.publish_body_for`). A 409 that names a
        recipe conflict (`_RECIPE_CONFLICT_CONSTRAINTS`) means the frame is
        published and apis kept the recipe it held, so it is logged at
        warning with the pair and the publish stands.

        Returns:
            Whether apis recorded the version with this call (`published
            .created` of its answer, and `False` on a recipe conflict), a
            `NormalFormConflict` when the recipe it kept has another normal form,
            `None` when the store holds nothing to publish under the pair, or the
            `Refusal` of a publish that failed.
        """
        created: bool | NormalFormConflict = False
        async with contextlib.AsyncExitStack() as stack:
            try:
                published = await stack.enter_async_context(
                    publish_body_for(
                        spec_id=spec_id,
                        content_version=content_version,
                        max_frame_rows=self._max_frame_rows,
                        recipe=recipe,
                    )
                )
            except PublishBodyNotBuildable as exc:
                if not exc.refuse_locally:
                    # Nothing (yet) to publish from this process's own view of
                    # the store -- not a publish failure. `_ensure_published_pair`
                    # proceeds to its caller's own platform call unmodified and
                    # lets the platform, the one party that knows whether the ref
                    # has ever resolved anywhere, answer -- honestly
                    # `ops_unsupported` if it never has.
                    return None
                return Refusal(
                    error="request_invalid",
                    retry=RetryMode.DO_NOT_RETRY,
                    detail=f"Publishing this frame to the platform failed: {exc}",
                )
            except RowBudgetExhausted as exc:
                return _row_budget_refusal(exc)

            body = await _gzipped_publish_body(published)
            if isinstance(body, Refusal):
                return body

            # Inside `publish_body_for`'s block, so the frame's row claim stays held
            # while the encoded body is in memory for the request.
            try:
                response = await self.request(
                    "POST",
                    _PUBLISH_ROUTE,
                    route=_PUBLISH_ROUTE,
                    content=body,
                    principal=self._publish_principal,
                    headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
                )
                created = _version_created(response)
            except PlatformError as exc:
                if (
                    exc.code != "publish_conflict"
                    or exc.constraint_id not in _RECIPE_CONFLICT_CONSTRAINTS
                ):
                    # Honest about which call failed: `_refusal` alone would read
                    # as a refusal of the caller's own request (the create/revise/
                    # describe call), when what actually failed is the publish this
                    # method ran ahead of it. Same code and retry mode either way --
                    # the apis service's publish door and its ordinary presentation surface
                    # already share one code vocabulary (`request_invalid`,
                    # `service_unavailable`, ...), just the detail text is reworded.
                    base = self._refusal(exc)
                    return replace(
                        base, detail=f"Publishing this frame to the platform failed: {base.detail}"
                    )
                logger.warning(
                    "published %s/%s, and dfo-apis kept another recipe: %s",
                    spec_id,
                    content_version,
                    exc.constraint_id,
                )
                if exc.constraint_id == _NORMAL_FORM_CONFLICT:
                    created = NormalFormConflict(detail=exc.detail or exc.constraint_id)
        self._published.add((spec_id, content_version))
        return created

    async def request(
        self,
        method: str,
        path: str,
        *,
        route: str,
        json: Any | None = None,
        content: bytes | None = None,
        params: Mapping[str, Any] | None = None,
        principal: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx2.Response:
        """Issue one apis presentation call, retried within a wall-clock budget.

        Args:
            method: HTTP method, e.g. "GET" or "POST".
            path: The interpolated request path, e.g.
                `/api/presentation/components/cmp_1`.
            route: The path *template*, e.g.
                `/api/presentation/components/{component_id}`. Kept separate
                from `path` because it lands on the span as `url.template`,
                which bounds span cardinality.
            json: JSON request body.
            content: An already-encoded request body, sent as is on every
                attempt. The publish call sends its body this way.
            params: Query parameters.
            principal: The `x-dfo-principal` header value, or `None` to send
                none at all. Set on the publish call only; a component create
                and revise send theirs through `headers`
                (`_card_write_headers`). `PRINCIPAL_HEADER`'s comment says why.
            headers: Further request headers, sent on every attempt.
                `revise_component` sends `WRITE_TOKEN_HEADER` here.

        Returns:
            The 2xx `httpx2.Response`.

        Raises:
            PlatformError: apis returned a non-2xx response.
            PlatformTransportError: no response was ever received, or the auth
                provider could not mint a token and nothing was sent.
        """
        try:
            request_headers = dict(await self._auth()) if self._auth else {}
        except ApisAuthError as exc:
            raise PlatformTransportError(detail=f"no dfo-apis ID token: {exc}") from exc
        if principal is not None:
            request_headers[PRINCIPAL_HEADER] = principal
        request_headers.update(headers or {})
        call_budget_s = max(self._retry_budget_s, self._read_timeout_s)

        with tracer.start_as_current_span(
            "platform.request",
            attributes={"http.request.method": method, "url.template": route},
        ) as call_span:
            start = time.monotonic()
            deadline = start + call_budget_s
            attempt = 0
            result: PlatformError | None = None

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                attempt += 1
                with tracer.start_as_current_span(
                    "platform.attempt",
                    attributes={"http.request.resend_count": attempt - 1},
                ) as attempt_span:
                    try:
                        async with asyncio.timeout(remaining):
                            attempted = await self._attempt(
                                method,
                                path,
                                json=json,
                                content=content,
                                params=params,
                                headers=request_headers,
                                timeout=min(self._read_timeout_s, remaining),
                            )
                    except TimeoutError:
                        # Builtin `TimeoutError`, distinct from
                        # `httpx2.TimeoutException`. `asyncio.timeout` bounds
                        # connect, write and pool as well as read, so an
                        # attempt that keeps trickling bytes cannot outlive
                        # the budget.
                        attempted = PlatformTransportError(
                            detail=f"attempt exceeded the call's {call_budget_s:.1f}s retry budget"
                        )
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

                should_retry = self._should_retry(result) and attempt < self._max_attempts
                backoff = 0.0
                if should_retry:
                    backoff = random.uniform(0, self._backoff_base_s * (2 ** (attempt - 1)))
                    if time.monotonic() + backoff >= deadline:
                        should_retry = False

                if not should_retry:
                    break
                await self._sleep(backoff)

            if result is None:
                # Only on a zero or negative budget. A typed error keeps
                # every failure leaving here a `PlatformError`.
                result = PlatformTransportError(
                    detail=f"the call's {call_budget_s:.1f}s budget expired before any attempt"
                )
            call_span.set_attribute("http.request.resend_count", attempt - 1)
            if result.status is not None:
                call_span.set_attribute("http.response.status_code", result.status)
            raise result

    async def _attempt(
        self,
        method: str,
        path: str,
        *,
        json: Any | None,
        content: bytes | None,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str],
        timeout: float,
    ) -> httpx2.Response | PlatformError:
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
                content=content,
                params=params,
                headers=headers,
                timeout=self._attempt_timeout(timeout),
            )
        except httpx2.TransportError as exc:
            return PlatformTransportError(detail=str(exc))
        if response.is_success:
            return response
        return self._to_platform_error(response)

    def _attempt_timeout(self, read: float) -> httpx2.Timeout:
        """Build the per-attempt timeout, varying only the read deadline."""
        return httpx2.Timeout(self._connect_timeout_s, read=read)

    def _to_platform_error(self, response: httpx2.Response) -> PlatformError:
        """Classify a non-2xx response into a `PlatformError`.

        Falls back to treating any unparsed 5xx as retryable and any
        unparsed 4xx as not, leaving `code` as `None` either way -- `code` is
        only ever read off the wire.
        """
        problem = _parse_problem_json(response.content)
        if problem is None:
            return PlatformError(
                status=response.status_code,
                code=None,
                retryable=response.status_code >= 500,
            )
        return PlatformError(
            status=response.status_code,
            code=problem.get("code"),
            retryable=bool(problem.get("retryable", False)),
            detail=problem.get("detail"),
            pointer=problem.get("pointer"),
            alternatives=problem.get("alternatives"),
            issues=problem.get("issues"),
            constraint_id=problem.get("constraint_id"),
            truncated=bool(problem.get("truncated", False)),
            trace_id=problem.get("trace_id"),
        )

    def _should_retry(self, error: PlatformError) -> bool:
        """Whether `error` is eligible for another attempt.

        Every call this client makes is safe to repeat: a read changes nothing,
        a revise merges the same patch into the same stored intent, a publish
        is idempotent server-side, and a create carries its own write token, so
        the platform answers a repeat with the draft the first attempt stored.
        A transport error is therefore always eligible, and so is a received
        5xx that reports `retryable: true` (a platform `timeout` 504 can arrive
        while the worker still commits). A 4xx describes the request, so
        repeating it reproduces it.
        """
        if isinstance(error, PlatformTransportError):
            return True
        if error.status is not None and error.status < 500:
            return False
        return error.retryable
