"""One async client for the apis routes live Studio shares use.

Three consumers share it: the harness refresh endpoint and replayer, the recipe
backfill job, and the recovery routine for root frames. Each call is one apis
route and returns a small typed result or raises `LiveApisError`.

Two credentials, sent in apis' principal header (`PRINCIPAL_HEADER`):

- `datasets_principal` (`DATASETS_FIRST_PARTY_PRINCIPAL`, the publish door's own)
  on the recipe routes: `put_recipe`, `read_recipe`, `find_replay`, `list_versions`.
- `live_refresh_principal` (`LIVE_REFRESH_PRINCIPAL`) on `refresh_frame`,
  `render` and `stream_render_data`.

`read_component` needs neither. A call whose principal is unset or blank raises
`LiveApisNotConfigured` before any request is made. Neither value appears in an
exception message, an attribute or a `repr`.

Every call also carries the headers of the `auth` hook, once per call: the Cloud
Run ID token from `common.apis_auth.apis_auth_provider`, which Cloud Run IAM in
front of dfo-apis checks. A hook that raises `ApisAuthError` fails the call
with `LiveApisTransportError` before any request is made, as an unreachable
dfo-apis does.

Every wire path and body is built in this module, in the `_*_PATH` constants and
the method that uses each. A refusal is an RFC 9457 problem body; `LiveApisError`
carries its `code` and `constraint_id`.

Every call returns the same answer on a repeat (a PUT of the same recipe, a
refresh behind single-flight, a render that appends no revision), so transport
failures and `retryable` 5xx refusals are retried up to `max_attempts`.
`stream_render_data` sends one attempt, since a body already begun cannot be resent.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any, Literal, Self
from urllib.parse import quote

import httpx2

from common.apis_auth import ApisAuthError, AuthProvider

# apis' principal header (`principal_header`, default `x-dfo-principal`).
PRINCIPAL_HEADER = "x-dfo-principal"

_RECIPE_PATH = "/api/datasets/frames/{spec_id}/{content_version}/recipe"
_REPLAY_PATH = "/api/datasets/recipes/replays/{replay_key}"
_VERSIONS_PATH = "/api/datasets/{spec_id}/versions"
_REFRESH_PATH = "/api/datasets/frames/refresh"
_COMPONENT_PATH = "/api/presentation/components/{component_id}"
_RENDER_PATH = "/api/presentation/components/{component_id}/revisions/{revision}/render"
_RENDER_DATA_PATH = f"{_RENDER_PATH}/data"

# Codes apis registers that a caller branches on.
RECIPE_NOT_FOUND = "recipe_not_found"
PUBLISH_CONFLICT = "publish_conflict"
REFRESH_NOT_PERMITTED = "refresh_not_permitted"
FRAME_NOT_REFRESHABLE = "frame_not_refreshable"
RENDERING_NOT_RECORDED = "rendering_not_recorded"
RENDER_TARGET_MISMATCH = "render_target_mismatch"

# The one code this module mints itself, for a 2xx answer outside the contract.
RESPONSE_INVALID = "response_invalid"

type RecipeBody = Mapping[str, Any]


class LiveApisNotConfigured(RuntimeError):  # noqa: N818 -- a configuration state, not a failed call
    """A principal this call needs is unset or blank. No request was made."""


class LiveApisError(Exception):
    """An apis service call failed.

    `status` is the HTTP status and `code` the problem body's machine-readable
    code, `None` when the body is not a problem document. `constraint_id` names the
    cause of a 409. `detail` is apis' own text.
    """

    def __init__(
        self,
        *,
        status: int | None,
        code: str | None,
        retryable: bool,
        constraint_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        self.status = status
        self.code = code
        self.retryable = retryable
        self.constraint_id = constraint_id
        self.detail = detail
        super().__init__(f"dfo-apis call failed: status={status} code={code}")


class LiveApisTransportError(LiveApisError):
    """No response arrived, from a connection failure or a timeout."""

    def __init__(self, *, detail: str) -> None:
        super().__init__(status=None, code=None, retryable=True, detail=detail)


@dataclass(frozen=True)
class FramePair:
    """A stored frame, named by a dataset (`spec_`) or derived (`ops_`) id and its version."""

    spec_id: str
    content_version: str

    def as_json(self) -> dict[str, str]:
        return {"spec_id": self.spec_id, "content_version": self.content_version}


@dataclass(frozen=True)
class RecipeWritten:
    """Answer to `put_recipe`. `created` is false when the frame already held it."""

    frame: FramePair
    created: bool


# How a stored version was rendered: `plain` for a raw version, `groupby` for a
# collapse whose `groupby` apis recorded, `without_party_totals` for a frame cut of
# the group totals of its open party roles and not summed, `not_recorded` for a derived
# version whose options apis recorded none of.
VersionRendering = Literal["plain", "groupby", "without_party_totals", "not_recorded"]


@dataclass(frozen=True)
class DatasetVersion:
    """One stored version of a dataset. `base_content_version` names the raw
    version a derived version (a groupby collapse, a party-totals cut, or both) was
    built from, else `None`. `groupby` is the mint request's `options.groupby` that
    made the collapse, set exactly when `rendering` is `groupby`.
    `without_party_totals` is true when the version's frame leaves out the group totals
    of its open party roles: on every `without_party_totals` rendering, and on a
    `groupby` one that summed such a frame.

    `enrichment_stamp` is the stamp a file of this version holding party-kind cells
    carries: the code index's digest when the version's frame holds a party role, and
    `None` when it holds none, or when apis lists no stamp on its versions. `served_stamp` is the answer's top-level stamp, the answering code index's
    digest, the same on every version of one answer, and `None` when it lists none."""

    content_version: str
    base_content_version: str | None
    rendering: VersionRendering
    groupby: Mapping[str, Any] | None
    without_party_totals: bool = False
    enrichment_stamp: str | None = None
    served_stamp: str | None = None


@dataclass(frozen=True)
class Current:
    """The head stands and `frame` is the named frame's rendering of it."""

    frame: FramePair


@dataclass(frozen=True)
class Moved:
    """The named frame is stale, and `frame` is the current frame of the same rendering."""

    frame: FramePair


@dataclass(frozen=True)
class Pending:
    """A rebuild over the inline budget is queued as `operation_id`."""

    operation_id: str
    retry_after_s: float


type RefreshResult = Current | Moved | Pending


@dataclass(frozen=True)
class ComponentRead:
    """A component revision as apis stores it. `frame_ref` is the stored
    intent's frame, `None` for a revision with no frame. `body` is the answer
    untouched."""

    frame_ref: FramePair | None
    body: dict[str, Any]


@dataclass
class RenderDataStream:
    """An open render data response. `chunks` is single use and ends with the
    `stream_render_data` block."""

    content_type: str | None
    content_disposition: str | None
    content_length: int | None
    chunks: AsyncIterator[bytes]


async def _default_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _segment(value: str) -> str:
    """`value` as one URL path segment, so an id cannot alter the route."""
    return quote(value, safe="")


def _problem(body: bytes) -> dict[str, Any] | None:
    """The body as an apis service problem document, else `None`.

    A document counts only with a bool `retryable`, which separates a
    apis refusal from a proxy's own error body on the same status.
    """
    try:
        parsed = json.loads(body) if body else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("retryable"), bool):
        return None
    return parsed


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _error_from(response: httpx2.Response) -> LiveApisError:
    problem = _problem(response.content)
    if problem is None:
        return LiveApisError(
            status=response.status_code, code=None, retryable=response.status_code >= 500
        )
    return LiveApisError(
        status=response.status_code,
        code=_text(problem.get("code")),
        retryable=problem["retryable"],
        constraint_id=_text(problem.get("constraint_id")),
        detail=_text(problem.get("detail")),
    )


def _invalid(response: httpx2.Response, what: str) -> LiveApisError:
    return LiveApisError(
        status=response.status_code,
        code=RESPONSE_INVALID,
        retryable=False,
        detail=f"{what} is outside the contract",
    )


def _json_object(response: httpx2.Response, what: str) -> dict[str, Any]:
    try:
        parsed = response.json()
    except ValueError:
        parsed = None
    if not isinstance(parsed, dict):
        raise _invalid(response, what)
    return parsed


def _frame_pair(value: object) -> FramePair | None:
    if not isinstance(value, Mapping):
        return None
    spec_id, content_version = _text(value.get("spec_id")), _text(value.get("content_version"))
    if spec_id is None or content_version is None:
        return None
    return FramePair(spec_id=spec_id, content_version=content_version)


def _dataset_version(item: dict[str, Any]) -> DatasetVersion | None:
    """One entry of the versions answer, or `None` when it is outside the contract:
    no `content_version`, a `rendering` that is none of the four, a `groupby` present on
    anything but a `groupby` rendering, or a `without_party_totals` that is not a
    boolean. apis leaves a false `without_party_totals` out, and a
    `without_party_totals` rendering implies it. apis also leaves out the
    `enrichment_stamp` of a version that has none."""
    content_version = _text(item.get("content_version"))
    base_content_version = _text(item.get("base_content_version"))
    groupby = item.get("groupby")
    without_party_totals = item.get("without_party_totals", False)
    stamp = _text(item.get("enrichment_stamp"))
    if content_version is None or not isinstance(without_party_totals, bool):
        return None
    match item.get("rendering"):
        case "groupby" if isinstance(groupby, dict):
            return DatasetVersion(
                content_version,
                base_content_version,
                "groupby",
                groupby,
                without_party_totals,
                enrichment_stamp=stamp,
            )
        case "without_party_totals" if groupby is None:
            return DatasetVersion(
                content_version,
                base_content_version,
                "without_party_totals",
                None,
                True,
                enrichment_stamp=stamp,
            )
        case ("plain" | "not_recorded") as rendering if groupby is None:
            return DatasetVersion(
                content_version, base_content_version, rendering, None, enrichment_stamp=stamp
            )
    return None


def _required_frame(response: httpx2.Response, value: object, what: str) -> FramePair:
    frame = _frame_pair(value)
    if frame is None:
        raise _invalid(response, what)
    return frame


class LiveApisClient:
    """Typed calls to the apis routes live Studio shares use.

    Use as an async context manager, or call `aclose`.
    """

    def __init__(
        self,
        *,
        base_url: str,
        datasets_principal: str | None = None,
        live_refresh_principal: str | None = None,
        auth: AuthProvider | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 50.0,
        max_attempts: int = 3,
        backoff_base_s: float = 1.0,
        sleep: Callable[[float], Awaitable[None]] = _default_sleep,
    ) -> None:
        """Build a client for the apis service at `base_url` (`DFO_APIS_URL`).

        Args:
            base_url: Origin of apis.
            datasets_principal: `DATASETS_FIRST_PARTY_PRINCIPAL`, for the recipe routes.
            live_refresh_principal: `LIVE_REFRESH_PRINCIPAL`, for refresh and render.
            auth: Async credential hook, called once per call. Adds nothing by default.
            transport: An `httpx2` transport, `httpx2.MockTransport` in tests.
            connect_timeout_s: Connection deadline of each attempt.
            read_timeout_s: Read deadline of each attempt.
            max_attempts: Attempts of one call, retries included.
            backoff_base_s: The first retry waits up to this long, each later one twice as long.
            sleep: Backoff sleep, replaced in tests.
        """
        self._datasets_principal = datasets_principal
        self._live_refresh_principal = live_refresh_principal
        self._auth = auth
        self._max_attempts = max_attempts
        self._backoff_base_s = backoff_base_s
        self._sleep = sleep
        self._client = httpx2.AsyncClient(
            base_url=base_url,
            timeout=httpx2.Timeout(connect_timeout_s, read=read_timeout_s),
            transport=transport,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release the connection pool."""
        await self._client.aclose()

    # Recipe routes, called with the publish principal.

    async def put_recipe(
        self, *, spec_id: str, content_version: str, recipe: RecipeBody
    ) -> RecipeWritten:
        """Store `recipe` for a published `ops_` frame; idempotent.

        Raises:
            LiveApisError: `snapshot_not_found` for a pair apis never stored;
                `publish_conflict` (see `constraint_id`) for a different recipe or
                a replay key another frame holds.
        """
        response = await self._send(
            "PUT",
            _RECIPE_PATH.format(
                spec_id=_segment(spec_id), content_version=_segment(content_version)
            ),
            headers=self._datasets_headers(),
            json=recipe,
        )
        body = _json_object(response, "recipe write answer")
        created = body.get("created")
        if not isinstance(created, bool):
            raise _invalid(response, "recipe write answer")
        return RecipeWritten(
            frame=_required_frame(response, body, "recipe write answer"), created=created
        )

    async def read_recipe(self, *, spec_id: str, content_version: str) -> RecipeBody | None:
        """The frame's recipe as stored, or `None` on `recipe_not_found`."""
        try:
            response = await self._send(
                "GET",
                _RECIPE_PATH.format(
                    spec_id=_segment(spec_id), content_version=_segment(content_version)
                ),
                headers=self._datasets_headers(),
            )
        except LiveApisError as exc:
            if exc.code == RECIPE_NOT_FOUND:
                return None
            raise
        return _json_object(response, "recipe")

    async def find_replay(self, *, replay_key: str) -> FramePair | None:
        """The frame stored under `replay_key`, or `None` on `recipe_not_found`."""
        try:
            response = await self._send(
                "GET",
                _REPLAY_PATH.format(replay_key=_segment(replay_key)),
                headers=self._datasets_headers(),
            )
        except LiveApisError as exc:
            if exc.code == RECIPE_NOT_FOUND:
                return None
            raise
        return _required_frame(response, _json_object(response, "replay answer"), "replay answer")

    async def list_versions(self, *, spec_id: str) -> tuple[DatasetVersion, ...]:
        """The dataset's stored versions, newest first, each with its own
        `enrichment_stamp` and the answer's top-level one as `served_stamp`."""
        response = await self._send(
            "GET",
            _VERSIONS_PATH.format(spec_id=_segment(spec_id)),
            headers=self._datasets_headers(),
        )
        answer = _json_object(response, "versions answer")
        versions = answer.get("versions")
        if not isinstance(versions, list):
            raise _invalid(response, "versions answer")
        stamp = _text(answer.get("enrichment_stamp"))
        parsed: list[DatasetVersion] = []
        for item in versions:
            version = _dataset_version(item) if isinstance(item, dict) else None
            if version is None:
                raise _invalid(response, "versions answer")
            parsed.append(replace(version, served_stamp=stamp))
        return tuple(parsed)

    # Refresh and render routes, called with the live-refresh principal.

    async def refresh_frame(self, *, spec_id: str, content_version: str) -> RefreshResult:
        """Whether the dataset frame named is current, and if not, what is.

        Raises:
            LiveApisError: `frame_not_refreshable` for a derived frame,
                `rendering_not_recorded` for a collapsed frame with no recorded groupby,
                `refresh_not_permitted` for a rejected principal, and
                `snapshot_not_found` or `spec_not_found` for an unknown frame.
        """
        response = await self._send(
            "POST",
            _REFRESH_PATH,
            headers=self._live_headers(),
            json={"spec_id": spec_id, "content_version": content_version},
        )
        body = _json_object(response, "refresh answer")
        match body.get("status"):
            case "current":
                return Current(_required_frame(response, body.get("frame"), "refresh answer"))
            case "moved":
                return Moved(_required_frame(response, body.get("frame"), "refresh answer"))
            case "pending":
                operation_id = _text(body.get("operation_id"))
                retry_after = _number(body.get("retry_after_s"))
                if operation_id is None or retry_after is None:
                    raise _invalid(response, "refresh answer")
                return Pending(operation_id=operation_id, retry_after_s=retry_after)
        raise _invalid(response, "refresh answer")

    async def render(self, *, component_id: str, revision: int, frame: FramePair) -> dict[str, Any]:
        """The component revision drawn on `frame`, as apis answers it.

        Raises:
            LiveApisError: `render_target_mismatch` when `frame` is not an
                acceptable target for the revision, `refresh_not_permitted` for a
                rejected principal.
        """
        response = await self._send(
            "POST",
            _RENDER_PATH.format(component_id=_segment(component_id), revision=revision),
            headers=self._live_headers(),
            json={"frame": frame.as_json()},
        )
        return _json_object(response, "render answer")

    @asynccontextmanager
    async def stream_render_data(
        self,
        *,
        component_id: str,
        revision: int,
        frame: FramePair,
        format: str | None = None,
        bundle: bool | None = None,
    ) -> AsyncIterator[RenderDataStream]:
        """The revision's data export drawn on `frame`, as an open byte stream.

        `format` and `bundle` go to apis when set, which validates them.

        Raises:
            LiveApisError: on a refusal, raised on entering the block.
            LiveApisTransportError: on a transport failure, including one that cuts
                the body short.
        """
        headers = await self._with_auth(self._live_headers())
        params: dict[str, str] = frame.as_json()
        if format is not None:
            params["format"] = format
        if bundle is not None:
            params["bundle"] = "true" if bundle else "false"
        path = _RENDER_DATA_PATH.format(component_id=_segment(component_id), revision=revision)
        try:
            async with self._client.stream("GET", path, params=params, headers=headers) as response:
                if not response.is_success:
                    await response.aread()
                    raise _error_from(response)
                length = response.headers.get("content-length", "")
                yield RenderDataStream(
                    content_type=response.headers.get("content-type"),
                    content_disposition=response.headers.get("content-disposition"),
                    content_length=int(length) if length.isdigit() else None,
                    chunks=response.aiter_bytes(),
                )
        except httpx2.TransportError as exc:
            raise LiveApisTransportError(detail=type(exc).__name__) from exc

    # No credential.

    async def read_component(
        self, *, component_id: str, revision: int | None = None
    ) -> ComponentRead:
        """A component revision (the head when `revision` is `None`).

        Raises:
            LiveApisError: `not_found` for an unknown component or revision.
        """
        response = await self._send(
            "GET",
            _COMPONENT_PATH.format(component_id=_segment(component_id)),
            params={"revision": revision} if revision is not None else None,
        )
        body = _json_object(response, "component read")
        intent = body.get("stored_intent")
        frame_ref = _frame_pair(intent.get("frame_ref")) if isinstance(intent, dict) else None
        return ComponentRead(frame_ref=frame_ref, body=body)

    # Wire.

    def _datasets_headers(self) -> dict[str, str]:
        return _principal_headers(self._datasets_principal, "DATASETS_FIRST_PARTY_PRINCIPAL")

    def _live_headers(self) -> dict[str, str]:
        return _principal_headers(self._live_refresh_principal, "LIVE_REFRESH_PRINCIPAL")

    async def _send(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> httpx2.Response:
        """Issue one call, retrying a transport failure or a `retryable` 5xx.

        Returns the 2xx response.
        """
        headers = await self._with_auth(headers)
        attempt = 0
        while True:
            attempt += 1
            error = await self._attempt(method, path, headers=headers, json=json, params=params)
            if isinstance(error, httpx2.Response):
                return error
            transient = error.retryable and (error.status is None or error.status >= 500)
            if not transient or attempt >= self._max_attempts:
                raise error
            await self._sleep(random.uniform(0, self._backoff_base_s * 2 ** (attempt - 1)))

    async def _with_auth(self, headers: Mapping[str, str] | None) -> dict[str, str]:
        """`headers` plus the auth hook's, minted after the principal check has passed.

        The auth hook's headers win over a header of the same name in `headers`.

        Raises:
            LiveApisTransportError: the hook could not mint a token.
        """
        try:
            auth = await self._auth() if self._auth else {}
        except ApisAuthError as exc:
            raise LiveApisTransportError(detail=f"no dfo-apis ID token: {exc}") from exc
        return {**(headers or {}), **auth}

    async def _attempt(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
        json: Any | None,
        params: Mapping[str, Any] | None,
    ) -> httpx2.Response | LiveApisError:
        try:
            response = await self._client.request(
                method, path, headers=headers, json=json, params=params
            )
        except httpx2.TransportError as exc:
            return LiveApisTransportError(detail=type(exc).__name__)
        return response if response.is_success else _error_from(response)


def _principal_headers(principal: str | None, setting: str) -> dict[str, str]:
    if principal is None or not principal.strip():
        raise LiveApisNotConfigured(f"{setting} is not set, so this dfo-apis call cannot be made")
    return {PRINCIPAL_HEADER: principal}


def _number(value: object) -> float | None:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None
