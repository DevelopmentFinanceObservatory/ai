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
  service defines the code, which holds for every engine code, except that
  `place_levels_mixed` is written again from its `fix`
  (`ToolRefusal.place_levels_mixed`), so its message names this service's calls. Under
  `rolling: "tree"`, a derived input with no stored recipe (`recipe_not_found`)
  is `rolling_reanchor_unavailable`, since it cannot be computed again under the
  tree's latest year. A root the
  runner cannot read (`snapshot_not_found`, `recipe_not_found`,
  `artifact_expired`) is, in this order: the caller's own frame refs read as
  that root, not held or aged out (`ToolRefusal.frames_lost`, which names
  every side read from it); a root of a rolling call's anchor record,
  `rolling_anchor_unresolved` with `anchor_root_lost`, since the record must
  be resolved again; or a root behind a derived (`ops_`) input, which the
  runner computes again from its stored recipe, so that input is
  `derived_input_unrecoverable`. The refusal names only the derived inputs
  whose manifest lists the root, and every one when a manifest cannot be read.
  `over_ceiling` is `frame_too_large`, and a busy runner's retryable
  `service_unavailable` keeps its code with a delay. Any other code is
  `upstream_error` naming it.
- An answer outside that contract, an unreachable dfo-apis, a timeout and a
  failed credential are `upstream_error`, the last three retryable.

A rolling call's answer is checked against its anchor record (`Anchor`): every
root the answer's ref names must be one the record was resolved over, else the
call refuses `rolling_anchor_unresolved` with `root_not_in_record`. The runner
takes the latest period over the anchor roots and checks the literals against
it, so a call over data the record never saw would otherwise compute a window
that ends at another dataset's latest period. The ref's roots are the roots
the runner read, those behind a derived input and the anchor roots included,
so the check covers a derived input as it covers a fetched frame.

`replay_derived` asks the runner to compute a derived frame again from its stored
recipe with `latest: true`, which reads its roots at their datasets' current
versions, and returns the pair it computed (`common.derived_replay`, which
presentation-mcp shares). A call whose derived input lost a
root sends the replayed pair in its place (`tool_pipeline.run_on_runner`). Every
other call sends `latest: false`, reading each root at the version it names.

A call with relative years states no anchor record: the runner fills them from
the latest year of the call's whole tree and answers each with its literal
(`resolved_periods`). A public call over a derived input sends `rolling: "tree"`,
so the input's own rolling years follow that one latest year too, and the
summary names each input that took another year alone (`run_tree`).

`read_derived_manifest` and `read_derived_file` read a frame the runner stored,
by its `ops_` pair, from `GET /api/datasets/derived/{spec_id}/{content_version}`
and its `/file`, for a caller that needs its recipe or its rows. Each raises
`DerivedUnread` naming the problem document's code. A read waits
`derived_read_timeout_s`, far less than a compute call's
`analysis_runner_timeout_s`, so a stalled dfo-apis fails a read in seconds.

Every call carries the Cloud Run ID token (`common.apis_auth`) and, for a caller
the website named (`x-dfo-caller: ip:<digest>`), the caller principal beside it,
so dfo-apis resolves the website's client as it does on the website's own
calls (`dfo_platform.caller.resolve_caller`). A `conversation:` caller from the
harness names no client address, so it is not sent.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

import httpx2
from common.apis_auth import ApisAuthError, AuthProvider, apis_auth_provider
from common.caller import ADDRESS_CALLER, CALLER_HEADER, CALLER_PRINCIPAL_HEADER, current_caller
from common.derived_replay import (
    ANALYSIS_PATH,
    DEFAULT_RETRY_AFTER_S,
    ReplayBusy,
    ReplayFailed,
    ReplayPending,
    busy,
    pending,
)
from common.derived_replay import problem_json as _json
from common.derived_replay import replay_derived as _replay
from common.derived_replay import retry_after_s as _retry_after_s
from common.holder import Holder
from common.materialisation import FrameRef, LeftOutSeries
from dfo_frame_grammar import shift_period
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from analysis_mcp.settings import Settings
from analysis_mcp.summary import RecomputedInput, ResolvedPeriod, Rolling, VerbSummary
from analysis_mcp.tool_errors import ToolErrorCode, ToolRefusal

logger = logging.getLogger(__name__)

ANCHOR_PATH = "/api/datasets/analysis/anchor"
DERIVED_PATH = "/api/datasets/derived/{spec_id}/{content_version}"
DERIVED_FILE_PATH = f"{DERIVED_PATH}/file"
# The derived file query that answers the frame's rows as one Parquet file.
DERIVED_FILE_QUERY: Mapping[str, str] = {"format": "parquet", "bundle": "false"}

# Codes this service defines for its own failures.
_LOCAL_CODES = frozenset(
    {
        ToolErrorCode.FRAME_NOT_HELD,
        ToolErrorCode.FRAME_EXPIRED,
        ToolErrorCode.FRAME_INTEGRITY_FAILED,
        ToolErrorCode.DERIVED_INPUT_UNRECOVERABLE,
        ToolErrorCode.ROLLING_REANCHOR_UNAVAILABLE,
        ToolErrorCode.SERVICE_UNAVAILABLE,
        ToolErrorCode.UPSTREAM_ERROR,
        ToolErrorCode.INTERNAL_ERROR,
    }
)
# Codes the runner's engine refuses a computation with.
RUNNER_CODES = frozenset(code.value for code in ToolErrorCode if code not in _LOCAL_CODES)
_UNHELD_ROOT_CODES = frozenset({"snapshot_not_found", "recipe_not_found"})
_EXPIRED_ROOT_CODE = "artifact_expired"
_ROOT_CODES = _UNHELD_ROOT_CODES | {_EXPIRED_ROOT_CODE}
_RECIPE_MISSING_CODE = "recipe_not_found"
# What dfo-apis answers when it cannot serve a Spec root. These refusals carry no `fix`.
_SPEC_ROOT_CODES = _ROOT_CODES | {"spec_not_found"}
_OVER_CEILING_CODE = "over_ceiling"
_RESPONSE_INVALID = "response_invalid"

# An answer's `resolved_periods`, one entry per relative period of the recipe.
_RESOLVED_PERIODS: TypeAdapter[tuple[ResolvedPeriod, ...]] = TypeAdapter(
    Annotated[tuple[ResolvedPeriod, ...], Field(min_length=1)]
)

type Pair = tuple[str, str]
# The caller's frame refs each root pair of a request stands for: two sides
# fetched from one dataset version with different options share one root.
type RefsByRoot = Mapping[Pair, tuple[FrameRef, ...]]


@dataclass(frozen=True, kw_only=True)
class Anchor:
    """The anchor record a rolling call is checked against: its `key`, the root
    node of each of its roots (`nodes`), sent as the request's `anchor_roots`, and
    the pair each node names (`pairs`)."""

    key: str
    nodes: tuple[dict[str, Any], ...]
    pairs: frozenset[Pair]


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
        read_timeout_s: float,
        caller_principal: str | None,
        auth: AuthProvider,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx2.AsyncClient(base_url=base_url, timeout=timeout_s, transport=transport)
        self._read_timeout_s = read_timeout_s
        self._caller_principal = caller_principal
        self._auth = auth

    def _caller_headers(self) -> dict[str, str]:
        caller = current_caller()
        if self._caller_principal is None or caller is None:
            return {}
        if not ADDRESS_CALLER.fullmatch(caller):
            return {}
        return {CALLER_PRINCIPAL_HEADER: self._caller_principal, CALLER_HEADER: caller}

    async def post(self, path: str, body: Mapping[str, Any]) -> httpx2.Response | ToolRefusal:
        """POST `body` to `path`, or the retryable refusal for a call that got no answer."""
        return await self._send("POST", path, json=body)

    async def get(
        self, path: str, *, params: Mapping[str, str] | None = None
    ) -> httpx2.Response | ToolRefusal:
        """GET `path` within the read timeout, or the retryable refusal for a call
        that got no answer."""
        return await self._send("GET", path, params=params, timeout=self._read_timeout_s)

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
        read_timeout_s=settings.derived_read_timeout_s,
        caller_principal=None if principal is None else principal.get_secret_value(),
        auth=apis_auth_provider(base_url=settings.dfo_apis_url),
    )


def _build() -> RunnerClient:
    # `_default_client` is looked up on each build, so a test can replace it.
    return _default_client()


runner_holder: Holder[RunnerClient] = Holder(default_factory=_build)


def _unusable(detail: str) -> ToolRefusal:
    return ToolRefusal.upstream_error(
        upstream_code=_RESPONSE_INVALID, detail=detail, retry_after_s=None
    )


def _pairs(roots: object) -> frozenset[Pair] | None:
    """The pair of each root in `roots`, a JSON list of `{spec_id, content_version}`
    objects, or `None` when it is not one."""
    if not isinstance(roots, list):
        return None
    pairs: set[Pair] = set()
    for root in roots:
        if not isinstance(root, Mapping):
            return None
        spec_id, content_version = root.get("spec_id"), root.get("content_version")
        if not isinstance(spec_id, str) or not isinstance(content_version, str):
            return None
        pairs.add((spec_id, content_version))
    return frozenset(pairs)


async def _manifest_roots(ref: FrameRef) -> frozenset[Pair] | None:
    """The root pairs the manifest of the derived frame `ref` lists, or `None` when
    it cannot be read."""
    try:
        manifest = await read_derived_manifest(ref)
    except DerivedUnread:
        return None
    return _pairs(manifest.get("roots"))


async def _derived_inputs_over(root: Pair, refs_by_root: RefsByRoot) -> list[dict[str, str]]:
    """The derived (`ops_`) inputs the request sent that were computed from `root`:
    each whose manifest lists it, or that cannot be told apart. A lone derived input
    is the one, and reads nothing."""
    derived = [
        ref for refs in refs_by_root.values() for ref in refs if ref.spec_id.startswith("ops_")
    ]
    if len(derived) > 1:
        async with asyncio.TaskGroup() as group:
            reads = [group.create_task(_manifest_roots(ref)) for ref in derived]
        over = [
            ref
            for ref, read in zip(derived, reads, strict=True)
            if (roots := read.result()) is None or root in roots
        ]
        derived = over or derived
    return [ref.model_dump() for ref in derived]


def _anchor_root_lost(anchor: Anchor, root: Pair, code: str) -> ToolRefusal:
    spec_id, content_version = root
    return ToolRefusal.rolling_anchor_unresolved(
        detail="anchor_root_lost",
        reason=(
            f"the record under {anchor.key} was resolved over the data {spec_id}, "
            f"{content_version}, which the analysis service no longer holds ({code})"
        ),
        context={
            "anchor_key": anchor.key,
            "root": {"spec_id": spec_id, "content_version": content_version},
        },
    )


async def refusal_for(
    response: httpx2.Response,
    *,
    refs_by_root: RefsByRoot,
    spec_roots: Collection[str] = (),
    anchor: Anchor | None = None,
    tree_rolling: bool = False,
) -> ToolRefusal:
    """The refusal a runner's non-2xx `response` maps to. See the module docstring.

    `refs_by_root` maps each root pair the request named as a frame to the
    caller's frame refs, `spec_roots` are the Specs it sent alone, `anchor` is a
    rolling call's anchor record, and `tree_rolling` says the request sent
    `rolling: "tree"`.
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
    root: Pair = (str(fix.get("spec_id")), str(fix.get("content_version")))
    if code == ToolErrorCode.PLACE_LEVELS_MIXED and isinstance(fix.get("column"), str):
        refusal = _place_levels_mixed(fix)
    elif code in RUNNER_CODES:
        refusal = ToolRefusal(code=ToolErrorCode(code), message=detail, fix=fix)
    elif (
        tree_rolling
        and code == _RECIPE_MISSING_CODE
        and root[0].startswith("ops_")
        and (refs := refs_by_root.get(root))
    ):
        refusal = ToolRefusal.rolling_reanchor_unavailable(
            frame_refs=[ref.model_dump() for ref in refs]
        )
    elif code in _ROOT_CODES and (refs := refs_by_root.get(root)):
        refusal = ToolRefusal.frames_lost(
            frame_refs=[ref.model_dump() for ref in refs], expired=code == _EXPIRED_ROOT_CODE
        )
    elif (
        code in _SPEC_ROOT_CODES and spec_roots and (root[0] in spec_roots or "spec_id" not in fix)
    ):
        refusal = ToolRefusal.bare_frames_lost(
            spec_ids=sorted(spec_roots), expired=code == _EXPIRED_ROOT_CODE
        )
    elif code in _ROOT_CODES and anchor is not None and root in anchor.pairs:
        refusal = _anchor_root_lost(anchor, root, code)
    elif code in _ROOT_CODES and (derived := await _derived_inputs_over(root, refs_by_root)):
        refusal = ToolRefusal.derived_input_root_lost(
            frame_refs=derived, root=fix, upstream_code=code
        )
    elif code == _OVER_CEILING_CODE:
        refusal = ToolRefusal(code=ToolErrorCode.FRAME_TOO_LARGE, message=detail, fix=fix)
    elif (refused := busy(response)) is not None:
        refusal = _busy(refused)
    else:
        refusal = ToolRefusal.upstream_error(
            upstream_code=code,
            detail=detail,
            retry_after_s=_retry_after_s(response) if retryable else None,
            fix=fix,
        )
    return refusal


def _place_levels_mixed(fix: Mapping[str, Any]) -> ToolRefusal:
    """The runner's `place_levels_mixed`, written from its `fix` as this service
    writes it, so its message names the call this service's tools take."""
    whole_class = fix.get("whole_class")
    whole_total = fix.get("whole_total")
    return ToolRefusal.place_levels_mixed(
        column=fix["column"],
        types=[str(cell) for cell in fix.get("types") or ()],
        whole_total=whole_total if isinstance(whole_total, str) else None,
        whole_class=whole_class if isinstance(whole_class, str) else None,
        member_count=int(fix.get("member_count") or 0),
        members=[str(member) for member in fix.get("example_members") or ()],
    )


def _busy(refused: ReplayBusy) -> ToolRefusal:
    wait = refused.retry_after_s
    return ToolRefusal(
        code=ToolErrorCode.SERVICE_UNAVAILABLE,
        message=f"The analysis service is busy: {refused.detail}. Retry after {wait:.1f}s.",
        fix={"service": "dfo-apis"},
        retry_after_s=wait,
    )


def _pending(response: httpx2.Response) -> ToolRefusal:
    return _pending_refusal(pending(response))


def _pending_refusal(waiting: ReplayPending) -> ToolRefusal:
    return ToolRefusal.analysis_pending(
        retry_after_s=waiting.retry_after_s, operations=list(waiting.operations)
    )


def _ref_roots(body: Mapping[str, Any]) -> frozenset[Pair] | None:
    """The root pairs every member of a ready answer's ref names, or `None` when the
    answer holds no such ref."""
    ref = body.get("ref")
    members = ref.get("members") if isinstance(ref, Mapping) else None
    if not isinstance(members, list) or not members:
        return None
    pairs: set[Pair] = set()
    for member in members:
        roots = _pairs(member.get("roots")) if isinstance(member, Mapping) else None
        if roots is None:
            return None
        pairs |= roots
    return frozenset(pairs)


def _uncovered(body: Mapping[str, Any], anchor: Anchor) -> ToolRefusal | None:
    """The refusal for a ready answer `body` that read a root `anchor`'s record was
    not resolved over. See the module docstring."""
    roots = _ref_roots(body)
    if roots is None:
        return _unusable("the analysis answer names no roots")
    uncovered = sorted(roots - anchor.pairs)
    if not uncovered:
        return None
    spec_id, content_version = uncovered[0]
    return ToolRefusal.rolling_anchor_unresolved(
        detail="root_not_in_record",
        reason=(
            f"the call read the data {spec_id}, {content_version}, which the record was "
            "not resolved over"
        ),
        context={
            "anchor_key": anchor.key,
            "root": {"spec_id": spec_id, "content_version": content_version},
        },
    )


def _unread_summary(exc: ValidationError | KeyError) -> str:
    """What of a summary did not fit: the place of its first field error, or the
    verb no facts model reads."""
    if isinstance(exc, KeyError):
        return f"no facts model reads the verb {exc.args[0]!r}"
    error = exc.errors()[0]
    place = ".".join(str(part) for part in error["loc"]) or "summary"
    return f"{place}: {error['msg']}"


async def run_tree(
    node: Mapping[str, Any],
    *,
    anchor: Anchor | None = None,
    refs_by_root: RefsByRoot,
    spec_roots: Collection[str] = (),
    follows_tree: bool = False,
) -> VerbSummary | ToolRefusal:
    """The summary of the tree `node` computed by the runner, or its refusal.
    `anchor`, for a rolling call, sends its record's roots, which the runner takes
    the latest period over, and checks the answer read no other root.

    The runner computes each derived (`ops_`) input again from its stored recipe.
    With `follows_tree`, a tree with a derived input sends `rolling: "tree"`, so the
    input's rolling years follow the one latest year of the whole tree. Any other
    request omits the field: a dfo-apis that does not know it still answers, and
    the harness's rolling steps keep the populations they state. Under
    `rolling: "tree"`, an input whose recipe is missing refuses
    `rolling_reanchor_unavailable`. The summary's `rolling` names the latest year
    the answer's `resolved_periods` were filled from and, under `rolling: "tree"`
    alone, each derived input the runner computed again under another year than
    its own (`_rolling`)."""
    derived = _derived_reads(refs_by_root)
    tree_rolling = follows_tree and bool(derived)
    request: dict[str, Any] = {"tree": node, "latest": False}
    if tree_rolling:
        request["rolling"] = "tree"
    if anchor is not None:
        request["anchor_roots"] = list(anchor.nodes)
    response = await runner_holder.get().post(ANALYSIS_PATH, request)
    if isinstance(response, ToolRefusal):
        return response
    if response.status_code == httpx2.codes.ACCEPTED:
        return _pending(response)
    if not response.is_success:
        return await refusal_for(
            response,
            refs_by_root=refs_by_root,
            spec_roots=spec_roots,
            anchor=anchor,
            tree_rolling=tree_rolling,
        )
    body = _json(response)
    summary = _ready_summary(body, anchor=anchor)
    if isinstance(summary, ToolRefusal):
        return summary
    assert body is not None
    rolling = await _rolling(body.get("resolved_periods"), derived=derived if tree_rolling else ())
    if isinstance(rolling, ToolRefusal):
        return rolling
    return summary if rolling is None else summary.model_copy(update={"rolling": rolling})


def _derived_reads(refs_by_root: RefsByRoot) -> tuple[FrameRef, ...]:
    """The derived (`ops_`) frames the request reads, each by the pair it names, which
    for a rebuilt frame is the replay's and no longer the caller's."""
    return tuple(
        FrameRef(spec_id=spec_id, content_version=content_version)
        for spec_id, content_version in refs_by_root
        if spec_id.startswith("ops_")
    )


async def _own_latest(ref: FrameRef) -> RecomputedInput | None:
    """`ref` with its top verb and the latest year its own recipe was filled from;
    `None` for a recipe with no rolling year, and for a manifest that cannot be
    read, which is logged, since whether the input was computed again is unknown."""
    try:
        manifest = await read_derived_manifest(ref)
    except DerivedUnread as exc:
        logger.warning(
            "derived input %s/%s unchecked for a recomputed year: %s",
            ref.spec_id,
            ref.content_version,
            exc,
        )
        return None
    anchor = manifest.get("anchor")
    period = anchor.get("period") if isinstance(anchor, Mapping) else None
    if not isinstance(period, str):
        return None
    recipe = manifest.get("recipe")
    tree = recipe.get("tree") if isinstance(recipe, Mapping) else None
    op_spec = tree.get("op_spec") if isinstance(tree, Mapping) else None
    verb = op_spec.get("verb") if isinstance(op_spec, Mapping) else None
    return RecomputedInput(
        **ref.model_dump(), verb=verb if isinstance(verb, str) else None, latest=period
    )


async def _rolling(
    resolved: object, *, derived: Sequence[FrameRef]
) -> Rolling | None | ToolRefusal:
    """The latest year the answer's `resolved` periods were filled from, `None`
    when it names none, with each of the `derived` inputs whose own recipe took
    another latest year. `derived` is empty unless the request sent
    `rolling: "tree"`, since without it each input keeps its stored periods. A
    derived input with no rolling year of its own, one that took the same year,
    and one whose manifest cannot be read are not named."""
    if not resolved:
        return None
    try:
        periods = _RESOLVED_PERIODS.validate_python(resolved)
        latest = shift_period(periods[0].period, -periods[0].offset)
    except ValueError:
        return _unusable("the analysis answer's resolved_periods do not fit this service")
    recomputed: list[RecomputedInput] = []
    if derived:
        async with asyncio.TaskGroup() as group:
            reads = [group.create_task(_own_latest(ref)) for ref in derived]
        owns = [read.result() for read in reads]
        recomputed = [own for own in owns if own is not None and own.latest != latest]
    return Rolling(latest=latest, periods=periods, recomputed=tuple(recomputed))


def _ready_summary(
    body: Mapping[str, Any] | None, *, anchor: Anchor | None
) -> VerbSummary | ToolRefusal:
    """The summary of the ready answer `body`, or the refusal for an answer outside the contract
    or, for a rolling call, one that read a root outside `anchor`."""
    if (
        body is None
        or body.get("status") != "ready"
        or not isinstance(body.get("summary"), Mapping)
    ):
        return _unusable("the analysis answer holds no summary")
    if anchor is not None and (refusal := _uncovered(body, anchor)) is not None:
        return refusal
    try:
        return VerbSummary.parse(body["summary"])
    except (ValidationError, KeyError) as exc:
        logger.warning("analysis runner summary unreadable: %s", exc)
        return _unusable(f"the analysis summary does not fit this service: {_unread_summary(exc)}")


async def replay_derived(ref: FrameRef) -> FrameRef | ToolRefusal:
    """The pair of the derived frame `ref` computed again from its stored recipe on
    its datasets' current versions (`common.derived_replay`), or the refusal for a
    runner that computes no frame: 202 is `analysis_pending` and a busy runner
    `service_unavailable`, each with its delay. Any other refusal is logged with
    `ref`."""
    replayed = await _replay(ref, runner_holder.get().post, logger=logger)
    match replayed:
        case ReplayPending():
            return _pending_refusal(replayed)
        case ReplayBusy():
            return _busy(replayed)
        case ReplayFailed() if replayed.response.status_code == httpx2.codes.OK:
            # A 200 that names no single frame.
            return _unusable(replayed.detail)
        case ReplayFailed():
            return await refusal_for(replayed.response, refs_by_root={})
    return replayed


async def find_anchor(
    roots: Sequence[Mapping[str, Any]],
    populations: Sequence[Mapping[str, Any]],
    *,
    refs_by_root: RefsByRoot,
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
        return await refusal_for(response, refs_by_root=refs_by_root)
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
