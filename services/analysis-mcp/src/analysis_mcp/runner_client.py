"""The client of dfo-apis' analysis runner, where every verb and `resolve_anchor`
computes.

`run_tree` posts a request tree (`runner_tree`), with a rolling call's anchor
roots, to `POST /api/datasets/analysis`, and `find_anchor` posts roots and
populations to `POST /api/datasets/analysis/anchor`. Each returns the runner's
answer, or the `ToolRefusal` its reply maps to:

- 200 from the analysis route is the stored frame's ref and the `summary` of
  the call (`summary.VerbSummary`). From the anchor route it is `AnchorAnswer`.
- 202 is a root dataset still building, `ToolRefusal.analysis_pending` with the
  runner's `retry_after_s`.
- A problem document keeps the runner's `code`, `detail` and `fix` where this
  service defines the code, which holds for every engine code. A root the
  runner cannot read (`snapshot_not_found`, `recipe_not_found`,
  `artifact_expired`) is the caller's own frame ref not held or aged out. A
  root the call did not name is one behind a derived (`ops_`) input, which the
  runner computes again from its stored recipe, so that input is
  `derived_input_unrecoverable`. `over_ceiling` is `frame_too_large`, and a
  busy runner's retryable `service_unavailable` keeps its code with a delay.
  Any other code is `upstream_error` naming it.
- An answer outside that contract, an unreachable dfo-apis, a timeout and a
  failed credential are `upstream_error`, the last three retryable.

`read_derived_manifest` and `read_derived_file` read a frame the runner stored,
by its `ops_` pair, from `GET /api/datasets/derived/{spec_id}/{content_version}`
and its `/file`, for a caller that needs its recipe or its rows. Each raises
`DerivedUnread` naming the problem document's code.

Every call carries the Cloud Run ID token (`common.apis_auth`) and, for a caller
the website named (`x-dfo-caller: ip:<digest>`), the caller principal beside it,
so dfo-apis resolves the website's client as it does on the website's own
calls (`dfo_platform.caller.resolve_caller`). A `conversation:` caller from the
harness names no client address, so it is not sent.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

import httpx2
from common.apis_auth import ApisAuthError, AuthProvider, apis_auth_provider
from common.caller import CALLER_HEADER, current_caller
from common.holder import Holder
from common.materialisation import FrameRef, LeftOutSeries
from pydantic import BaseModel, ConfigDict, ValidationError

from analysis_mcp.settings import Settings
from analysis_mcp.summary import VerbSummary
from analysis_mcp.tool_errors import ToolErrorCode, ToolRefusal

logger = logging.getLogger(__name__)

ANALYSIS_PATH = "/api/datasets/analysis"
ANCHOR_PATH = "/api/datasets/analysis/anchor"
DERIVED_PATH = "/api/datasets/derived/{spec_id}/{content_version}"
DERIVED_FILE_PATH = f"{DERIVED_PATH}/file"
# The derived file query that answers the frame's rows as one Parquet file.
DERIVED_FILE_QUERY: Mapping[str, str] = {"format": "parquet", "bundle": "false"}
CALLER_PRINCIPAL_HEADER = "x-dfo-caller-principal"

# The caller the website names, which dfo-apis believes beside the principal.
_ADDRESS_CALLER = re.compile(r"ip:[0-9a-f]{32}")

# The wait a retryable refusal or a failed transport names when dfo-apis sent none.
DEFAULT_RETRY_AFTER_S = 5.0

# Codes this service defines for its own failures.
_LOCAL_CODES = frozenset(
    {
        ToolErrorCode.FRAME_NOT_HELD,
        ToolErrorCode.FRAME_EXPIRED,
        ToolErrorCode.FRAME_INTEGRITY_FAILED,
        ToolErrorCode.DERIVED_INPUT_UNRECOVERABLE,
        ToolErrorCode.SERVICE_UNAVAILABLE,
        ToolErrorCode.UPSTREAM_ERROR,
        ToolErrorCode.INTERNAL_ERROR,
    }
)
_RUNNER_CODES = frozenset(code.value for code in ToolErrorCode if code not in _LOCAL_CODES)
_UNHELD_ROOT_CODES = frozenset({"snapshot_not_found", "recipe_not_found"})
_EXPIRED_ROOT_CODE = "artifact_expired"
_ROOT_CODES = _UNHELD_ROOT_CODES | {_EXPIRED_ROOT_CODE}
_OVER_CEILING_CODE = "over_ceiling"
_BUSY_CODE = "service_unavailable"
_RESPONSE_INVALID = "response_invalid"

type Pair = tuple[str, str]


class AnchorAnswer(BaseModel):
    """The anchor route's answer, with the latest `period`, its `rule`, and each series
    that holds no value at it, its `root` the index of its population."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    period: str
    rule: str
    left_out: tuple[LeftOutSeries, ...] = ()


class RunnerClient:
    """One HTTP client for dfo-apis' analysis routes."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float,
        caller_principal: str | None,
        auth: AuthProvider,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx2.AsyncClient(base_url=base_url, timeout=timeout_s, transport=transport)
        self._caller_principal = caller_principal
        self._auth = auth

    def _caller_headers(self) -> dict[str, str]:
        caller = current_caller()
        if self._caller_principal is None or caller is None:
            return {}
        if not _ADDRESS_CALLER.fullmatch(caller):
            return {}
        return {CALLER_PRINCIPAL_HEADER: self._caller_principal, CALLER_HEADER: caller}

    async def post(self, path: str, body: Mapping[str, Any]) -> httpx2.Response | ToolRefusal:
        """POST `body` to `path`, or the retryable refusal for a call that got no answer."""
        return await self._send("POST", path, json=body)

    async def get(
        self, path: str, *, params: Mapping[str, str] | None = None
    ) -> httpx2.Response | ToolRefusal:
        """GET `path`, or the retryable refusal for a call that got no answer."""
        return await self._send("GET", path, params=params)

    async def _send(self, method: str, path: str, **request: Any) -> httpx2.Response | ToolRefusal:
        try:
            headers = {**(await self._auth()), **self._caller_headers()}
            return await self._client.request(method, path, headers=headers, **request)
        except ApisAuthError as exc:
            detail = f"no dfo-apis credential: {exc}"
        except httpx2.TimeoutException:
            detail = "dfo-apis did not answer in time"
        except httpx2.TransportError as exc:
            detail = f"dfo-apis is unreachable: {type(exc).__name__}"
        logger.warning("dfo-apis call to %s failed: %s", path, detail)
        return ToolRefusal.upstream_error(
            upstream_code=None, detail=detail, retry_after_s=DEFAULT_RETRY_AFTER_S
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _default_client() -> RunnerClient:
    settings = Settings()
    principal = settings.dfo_caller_principal
    return RunnerClient(
        base_url=settings.dfo_apis_url,
        timeout_s=settings.analysis_runner_timeout_s,
        caller_principal=None if principal is None else principal.get_secret_value(),
        auth=apis_auth_provider(base_url=settings.dfo_apis_url),
    )


def _build() -> RunnerClient:
    # `_default_client` is looked up on each build, so a test can replace it.
    return _default_client()


runner_holder: Holder[RunnerClient] = Holder(default_factory=_build)


def _json(response: httpx2.Response) -> Mapping[str, Any] | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, Mapping) else None


def _retry_after_s(response: httpx2.Response) -> float:
    try:
        return max(float(response.headers.get("retry-after", "")), 0.0)
    except ValueError:
        return DEFAULT_RETRY_AFTER_S


def _unusable(detail: str) -> ToolRefusal:
    return ToolRefusal.upstream_error(
        upstream_code=_RESPONSE_INVALID, detail=detail, retry_after_s=None
    )


def _caller_ref(fix: Mapping[str, Any], refs_by_root: Mapping[Pair, FrameRef]) -> FrameRef | None:
    """The caller's frame ref for the root a refusal's `fix` names."""
    return refs_by_root.get((fix.get("spec_id"), fix.get("content_version")))


def _derived_inputs(refs_by_root: Mapping[Pair, FrameRef]) -> list[dict[str, str]]:
    """The derived (`ops_`) frame refs the request sent, each under its own pair."""
    return [ref.model_dump() for ref in refs_by_root.values() if ref.spec_id.startswith("ops_")]


def refusal_for(response: httpx2.Response, *, refs_by_root: Mapping[Pair, FrameRef]) -> ToolRefusal:
    """The refusal a runner's non-2xx `response` maps to. See the module docstring.

    `refs_by_root` maps each root pair the request named to the caller's frame ref.
    """
    problem = _json(response)
    if problem is None or not isinstance(problem.get("code"), str):
        return ToolRefusal.upstream_error(
            upstream_code=None,
            detail=f"dfo-apis answered {response.status_code} with no problem document",
            retry_after_s=_retry_after_s(response) if response.status_code >= 500 else None,
        )
    code: str = problem["code"]
    detail = str(problem.get("detail") or problem.get("title") or code)
    given = problem.get("fix")
    fix: dict[str, Any] = dict(given) if isinstance(given, Mapping) else {}
    retryable = problem.get("retryable") is True
    root = _caller_ref(fix, refs_by_root)
    if code in _RUNNER_CODES:
        refusal = ToolRefusal(code=ToolErrorCode(code), message=detail, fix=fix)
    elif code in _UNHELD_ROOT_CODES and root is not None:
        refusal = ToolRefusal.frame_not_held(
            spec_id=root.spec_id, content_version=root.content_version
        )
    elif code == _EXPIRED_ROOT_CODE and root is not None:
        refusal = ToolRefusal.frame_expired(
            spec_id=root.spec_id, content_version=root.content_version
        )
    elif code in _ROOT_CODES and (derived := _derived_inputs(refs_by_root)):
        refusal = ToolRefusal.derived_input_root_lost(
            frame_refs=derived, root=fix, upstream_code=code
        )
    elif code == _OVER_CEILING_CODE:
        refusal = ToolRefusal(code=ToolErrorCode.FRAME_TOO_LARGE, message=detail, fix=fix)
    elif code == _BUSY_CODE and retryable:
        wait = _retry_after_s(response)
        refusal = ToolRefusal(
            code=ToolErrorCode.SERVICE_UNAVAILABLE,
            message=f"The analysis service is busy: {detail}. Retry after {wait:.1f}s.",
            fix={"service": "dfo-apis"},
            retry_after_s=wait,
        )
    else:
        refusal = ToolRefusal.upstream_error(
            upstream_code=code,
            detail=detail,
            retry_after_s=_retry_after_s(response) if retryable else None,
            fix=fix,
        )
    return refusal


def _pending(response: httpx2.Response) -> ToolRefusal:
    body = _json(response) or {}
    retry_after_s = body.get("retry_after_s")
    operations = body.get("operations")
    return ToolRefusal.analysis_pending(
        retry_after_s=float(retry_after_s)
        if isinstance(retry_after_s, int | float)
        else _retry_after_s(response),
        operations=[str(op) for op in operations] if isinstance(operations, list) else [],
    )


async def run_tree(
    node: Mapping[str, Any],
    *,
    anchor_roots: Sequence[Mapping[str, Any]] = (),
    refs_by_root: Mapping[Pair, FrameRef],
) -> VerbSummary | ToolRefusal:
    """The summary of the tree `node` computed by the runner, or its refusal.
    `anchor_roots`, when given, are the roots the runner takes a rolling call's
    latest period over."""
    request: dict[str, Any] = {"tree": node, "latest": False}
    if anchor_roots:
        request["anchor_roots"] = list(anchor_roots)
    response = await runner_holder.get().post(ANALYSIS_PATH, request)
    if isinstance(response, ToolRefusal):
        return response
    if response.status_code == httpx2.codes.ACCEPTED:
        return _pending(response)
    if not response.is_success:
        return refusal_for(response, refs_by_root=refs_by_root)
    body = _json(response)
    if (
        body is None
        or body.get("status") != "ready"
        or not isinstance(body.get("summary"), Mapping)
    ):
        return _unusable("the analysis answer holds no summary")
    try:
        return VerbSummary.parse(body["summary"])
    except (ValidationError, KeyError) as exc:
        logger.warning("analysis runner summary unreadable: %s", exc)
        return _unusable(f"the analysis summary does not fit this service: {type(exc).__name__}")


async def find_anchor(
    roots: Sequence[Mapping[str, Any]],
    populations: Sequence[Mapping[str, Any]],
    *,
    refs_by_root: Mapping[Pair, FrameRef],
) -> AnchorAnswer | ToolRefusal:
    """The latest period of `roots` over `populations`, or the refusal."""
    response = await runner_holder.get().post(
        ANCHOR_PATH, {"roots": list(roots), "populations": list(populations), "latest": False}
    )
    if isinstance(response, ToolRefusal):
        return response
    if response.status_code == httpx2.codes.ACCEPTED:
        return _pending(response)
    if not response.is_success:
        return refusal_for(response, refs_by_root=refs_by_root)
    body = _json(response)
    try:
        return AnchorAnswer.model_validate(body)
    except ValidationError as exc:
        logger.warning("analysis runner anchor answer unreadable: %s", exc)
        return _unusable("the anchor answer does not fit this service")


class DerivedUnread(Exception):  # noqa: N818 -- a read's outcome, as `ToolRefusal` is
    """A derived frame dfo-apis did not serve. `code` is its problem document's code,
    `upstream_error` for an answer outside the contract or no answer at all, and
    `retryable` says whether the same read can succeed later."""

    def __init__(self, *, code: str, retryable: bool, detail: str) -> None:
        self.code = code
        self.retryable = retryable
        self.detail = detail
        super().__init__(f"derived frame unread ({code}): {detail}")


def _derived_unread(answer: httpx2.Response | ToolRefusal) -> DerivedUnread:
    if isinstance(answer, ToolRefusal):
        return DerivedUnread(
            code=ToolErrorCode.UPSTREAM_ERROR.value, retryable=True, detail=answer.message
        )
    problem = _json(answer) or {}
    code = problem.get("code")
    return DerivedUnread(
        code=code if isinstance(code, str) else ToolErrorCode.UPSTREAM_ERROR.value,
        retryable=problem.get("retryable") is True
        if isinstance(code, str)
        else answer.status_code >= 500,
        detail=str(problem.get("detail") or f"dfo-apis answered {answer.status_code}"),
    )


def _derived_path(path: str, ref: FrameRef) -> str:
    return path.format(spec_id=ref.spec_id, content_version=ref.content_version)


async def read_derived_manifest(ref: FrameRef) -> Mapping[str, Any]:
    """The manifest dfo-apis serves for the derived frame `ref`
    (`GET /api/datasets/derived/{spec_id}/{content_version}`): its `recipe`, with
    `replay_key` and `tree`, its `anchor`, its `roots` and the rest.

    Raises:
        DerivedUnread: dfo-apis holds no such frame (`snapshot_not_found`), its
            stored frame has expired (`artifact_expired`), or the read failed.
    """
    answer = await runner_holder.get().get(_derived_path(DERIVED_PATH, ref))
    if isinstance(answer, ToolRefusal) or not answer.is_success:
        raise _derived_unread(answer)
    body = _json(answer)
    if body is None:
        raise DerivedUnread(
            code=ToolErrorCode.UPSTREAM_ERROR.value,
            retryable=False,
            detail="the derived manifest is no JSON object",
        )
    return body


async def read_derived_file(ref: FrameRef) -> bytes:
    """The rows of the derived frame `ref` as one Parquet file
    (`GET /api/datasets/derived/{spec_id}/{content_version}/file`).

    Raises:
        DerivedUnread: As `read_derived_manifest`.
    """
    answer = await runner_holder.get().get(
        _derived_path(DERIVED_FILE_PATH, ref), params=DERIVED_FILE_QUERY
    )
    if isinstance(answer, ToolRefusal) or not answer.is_success:
        raise _derived_unread(answer)
    return answer.content
