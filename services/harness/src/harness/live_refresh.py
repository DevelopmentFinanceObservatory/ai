"""`POST /api/studio/refresh`: a card drawn on a derived frame, on current data.

The website calls this server to server for one card of a live share. The
request names a stored component revision, `{"component_id", "revision"}`, and
carries `Authorization: Bearer <LIVE_REFRESH_PRINCIPAL>`
(`harness.auth.live_refresh_admits`). No conversation, model, or session is
involved, and nothing about a refresh is kept between requests beyond the frames
in the materialisation store and the map of runs in flight.

One refresh (`LiveRefresh._refresh`):

1. Reads the component revision from apis and takes its stored intent's
   frame. A dataset (`spec_`) frame is refused. The website refreshes those
   against apis itself.
2. Reads the frame's recipe from apis.
3. Asks apis' frame refresh about each root's dataset pair, at most
   `MAX_CONCURRENT_ROOT_CHECKS` at once. A root still building answers `pending`.
   Otherwise a root whose mint record says its replay mints party kinds is named
   under the stamp the versions route lists on the root's current version
   (`_listed_stamp`), which is the stamp of the replay's kind file and of the mint
   record the replay keeps. The route is read once per dataset in a refresh
   (`LiveRefresh._listed_stamps`). The card is `current` when every root's dataset
   pair stands and each root's identity (`common.recipe.root_identity`: the pair,
   and the stamp a kind root is named under) is the one its mint record holds. A
   root recorded with neither columns nor a stamp counts as minted without kinds
   here, so a flow root recorded without kinds stands until its dataset moves.
4. Computes the replay key of the recipe on the roots' current identities
   (`common.recipe.replay_key`) and asks apis for the frame stored under it. A
   root recorded with neither columns nor a stamp whose current version lists a
   stamp is named under it only when its replay mints kinds, so its stored
   definition is read first to learn that (`LiveRefresh._settle_kinds`). A hit is
   rendered without replaying, so a card replays once for each change of a root's
   pair or stamp: the replay publishes under the key the next refresh looks up.
5. Otherwise replays the recipe (`harness.recipe_replay.replay_recipe`), which
   mints the roots, runs the verbs, and publishes the frame with its recipe.
   The replayer reads the replay key once more before it publishes, under the
   key of the recipe it built, so a frame another instance published while this
   one replayed is not sent twice.
6. Has apis render the stored revision on the frame, and answers the render
   as apis gave it.

Time. A refresh starts a deadline of `Settings.live_refresh_deadline_s`. The
replayer starts no mint, verb, or publish after it. A step in flight is not cut
by the deadline, so the whole run is cancelled `HARD_STOP_GRACE_S` later and
answers `pending`. The steps that finished are in the store and the next call
skips them.

Load. Refreshes draw on a token bucket and an in-flight cap of their own, apart
from the combined MCP server's, and on a row budget of their own
(`common.row_budget.RowBudget`). A request over any of the three answers
`pending`, except a replay step whose rows exceed the whole row budget, which is
refused `rows_over_refresh_budget`. Requests for one `(component_id, revision)`
share one run in this process, and the cap counts runs: a run holds its slot
until it ends, even after every request waiting on it has gone, and a request
that joins a run takes none.

`REFRESH_CODES` lists every refusal this route answers, with its HTTP status and
whether repeating the request can succeed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from analysis_mcp.execution_identity import current_execution_identity
from common import live_apis
from common.apis_auth import apis_auth_provider
from common.live_apis import (
    DatasetVersion,
    FramePair,
    LiveApisClient,
    LiveApisError,
    LiveApisNotConfigured,
    RefreshResult,
)
from common.materialisation import MintRecord, get_store
from common.recipe import replay_key, root_identity
from common.row_budget import RowBudget
from common.telemetry import EventLevel, use_request_span
from compiler.surface import McpSurface, ToolSurfaceUnavailableError
from data_mcp.deadline import EXTERNAL_CALL_BUDGET, CallDeadline, use_call_budget
from data_mcp.frame_materialisation import replayed_mint_options, requests_kinds
from data_mcp.get_data_tool import recorded_mint_request
from data_mcp.tool_errors import ToolErrorCode
from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse
from fastmcp.server.middleware.rate_limiting import TokenBucketRateLimiter
from fastmcp.tools import ToolResult
from planner.workflow import ToolId
from presentation_mcp.client import PlatformAssembler
from presentation_mcp.settings import Settings as PresentationSettings
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from harness.agents import build_tool_surface
from harness.auth import live_refresh_admits
from harness.recipe_replay import (
    BUSY_RETRY_AFTER_S,
    RESUME_RETRY_AFTER_S,
    CallTool,
    Pending,
    PublishFrame,
    Refused,
    Replayed,
    ToolAnswer,
    refusal_outcome,
    replay_recipe,
)
from harness.settings import LIVE_REFRESH_HARD_STOP_GRACE_S, Settings

__all__ = [
    "HARD_STOP_GRACE_S",
    "MAX_CONCURRENT_ROOT_CHECKS",
    "REFRESH_CODES",
    "LiveRefresh",
    "RefreshCode",
    "build_live_refresh",
    "create_live_refresh_router",
]

logger = logging.getLogger(__name__)

# Frame refresh calls to apis in flight at once for one card's roots. Each can
# take one of the two materialise slots an apis instance has.
MAX_CONCURRENT_ROOT_CHECKS: Final = 2

# How long after its deadline a run is cancelled. The deadline stops new steps.
# This bounds the one already in flight. `harness.settings` holds the value,
# because the largest deadline it accepts is worked out from it.
HARD_STOP_GRACE_S: Final = LIVE_REFRESH_HARD_STOP_GRACE_S

# The wait a `pending` names when the in-flight cap is full. Most refreshes end in
# a second or two. One that replays runs for up to the deadline.
IN_FLIGHT_RETRY_AFTER_S: Final = 2.0

_DATASET_PREFIX, _DERIVED_PREFIX = "spec_", "ops_"


@dataclass(frozen=True, kw_only=True)
class RefreshCode:
    """One refusal code of the route: the HTTP status it is answered under, and
    whether the same request can succeed later with nothing else changed."""

    code: str
    status: int
    retry_can_succeed: bool
    when: str


UNAUTHORIZED: Final = "unauthorized"
REQUEST_INVALID: Final = "request_invalid"
COMPONENT_NOT_FOUND: Final = "component_not_found"
FRAME_UNSUPPORTED: Final = "frame_unsupported"
DATASET_FRAME: Final = "dataset_frame"
RECIPE_MISSING: Final = "recipe_missing"
RECIPE_INVALID: Final = "recipe_invalid"
DFO_APIS_REFUSED: Final = "dfo_apis_refused"
UPSTREAM_UNAVAILABLE: Final = "upstream_unavailable"
NOT_CONFIGURED: Final = "refresh_not_configured"

# Every refusal the route writes under a code of its own. Any other code in a 422
# answer is passed through unchanged from the service that refused, and is
# permanent: an apis service refusal that is not retryable (frame refresh, replay
# lookup, render), or a `harness.recipe_replay.Refused` (the replayer's own codes,
# `rows_over_refresh_budget` among them, and the mint's, verb's, and publish's).
# `upstream_unavailable` is the only refusal a retry can clear. A wait is a 200
# `pending`, not a refusal.
REFRESH_CODES: Final[tuple[RefreshCode, ...]] = (
    RefreshCode(
        code=UNAUTHORIZED,
        status=401,
        retry_can_succeed=False,
        when="the bearer is missing or is not LIVE_REFRESH_PRINCIPAL, or that setting is unset",
    ),
    RefreshCode(
        code=REQUEST_INVALID,
        status=422,
        retry_can_succeed=False,
        when="the body is not {component_id, revision} with a cmp_ id and a revision of 1 or more",
    ),
    RefreshCode(
        code=COMPONENT_NOT_FOUND,
        status=422,
        retry_can_succeed=False,
        when="dfo-apis holds no such component, or no such revision of it",
    ),
    RefreshCode(
        code=FRAME_UNSUPPORTED,
        status=422,
        retry_can_succeed=False,
        when="the stored intent names no frame, or one that is neither spec_ nor ops_",
    ),
    RefreshCode(
        code=DATASET_FRAME,
        status=422,
        retry_can_succeed=False,
        when="the card is drawn on a dataset (spec_) frame, which dfo-apis refreshes itself",
    ),
    RefreshCode(
        code=RECIPE_MISSING,
        status=422,
        retry_can_succeed=False,
        when="dfo-apis holds no recipe for the card's frame",
    ),
    RefreshCode(
        code=RECIPE_INVALID,
        status=422,
        retry_can_succeed=False,
        when="the stored recipe has no normal form hash, or a root that is no mint record",
    ),
    RefreshCode(
        code=DFO_APIS_REFUSED,
        status=422,
        retry_can_succeed=False,
        when="dfo-apis refused for good with a body that names no code",
    ),
    RefreshCode(
        code=UPSTREAM_UNAVAILABLE,
        status=503,
        retry_can_succeed=True,
        when=(
            "dfo-apis did not answer, answered 5xx, or marked its refusal retryable; "
            "or an analysis tool call did not complete"
        ),
    ),
    RefreshCode(
        code=NOT_CONFIGURED,
        status=503,
        retry_can_succeed=False,
        when="DATASETS_FIRST_PARTY_PRINCIPAL is unset, so recipes cannot be read",
    ),
)
_STATUS_OF: Final = {row.code: row.status for row in REFRESH_CODES}
_PASSED_THROUGH_STATUS: Final = 422


class _RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    component_id: str = Field(pattern=r"^cmp_[A-Za-z0-9_-]{1,64}$")
    revision: int = Field(ge=1)


@dataclass(frozen=True, kw_only=True)
class _Answer:
    """What one refresh answers: the HTTP status, the JSON body, and the word the
    log line and the span carry for it (a status or a refusal code)."""

    status: int
    body: dict[str, Any]
    outcome: str

    def response(self) -> JSONResponse:
        return JSONResponse(
            self.body, status_code=self.status, headers={"Cache-Control": "no-store"}
        )


def _refusal(code: str, detail: str) -> _Answer:
    return _Answer(
        status=_STATUS_OF.get(code, _PASSED_THROUGH_STATUS),
        body={"code": code, "detail": detail},
        outcome=code,
    )


def _pending(retry_after_s: float, *, reason: str) -> _Answer:
    return _Answer(
        status=200,
        body={"status": "pending", "retry_after_s": retry_after_s},
        outcome=f"pending:{reason}",
    )


_CURRENT: Final = _Answer(status=200, body={"status": "current"}, outcome="current")


def _apis_refusal(exc: LiveApisError) -> _Answer:
    """What a failed apis call makes of the refresh: `upstream_unavailable` for
    no answer, a 5xx, or a refusal apis marks retryable, else its own code."""
    if exc.retryable or exc.status is None or exc.status >= 500:
        return _refusal(UPSTREAM_UNAVAILABLE, f"dfo-apis did not serve the call: {exc}")
    return _refusal(exc.code or DFO_APIS_REFUSED, exc.detail or str(exc))


def _level(answer: _Answer) -> EventLevel:
    if answer.status >= 500:
        return "ERROR"
    return "WARNING" if answer.status >= 400 else "DEFAULT"


def _answer_of(outcome: Pending | Refused) -> _Answer:
    """What a replay's wait or refusal makes of the refresh."""
    if isinstance(outcome, Pending):
        return _pending(outcome.retry_after_s, reason=outcome.reason)
    return _refusal(outcome.code, outcome.detail)


class _Stop(Exception):  # noqa: N818 -- a control-flow signal carrying an answer
    """The refresh ended early. `answer` is what the route answers."""

    def __init__(self, answer: _Answer) -> None:
        self.answer = answer
        super().__init__(answer.outcome)


class _Refusal(_Stop):
    """The refresh ended in the refusal `code`."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(_refusal(code, detail))


def _normal_form_and_roots(recipe: Mapping[str, Any]) -> tuple[str, list[MintRecord]]:
    """The stored recipe's normal form hash and its roots' mint records.

    Raises:
        _Refusal: `recipe_invalid` for a recipe that states neither.
    """
    normal_form_hash, roots = recipe.get("normal_form_hash"), recipe.get("roots")
    if not isinstance(normal_form_hash, str) or not isinstance(roots, list):
        raise _Refusal(RECIPE_INVALID, "the stored recipe has no normal form hash or roots")
    try:
        return normal_form_hash, [MintRecord.model_validate(root) for root in roots]
    except ValidationError as exc:
        raise _Refusal(
            RECIPE_INVALID,
            f"a root of the stored recipe is no mint record: {exc.error_count()} error(s)",
        ) from exc


def _log_another_key(
    component_id: str,
    *,
    looked_up: str,
    identities: Sequence[Mapping[str, Any]],
    replayed: Replayed,
) -> None:
    """Log a replay published under another key than the refresh looked up. The
    replay then named a root under another stamp than the versions route listed
    (`_listed_stamp`), as when two apis instances on different code
    indexes answer the versions route and the mint during a deploy, and the next
    refresh replays again until the two agree."""
    published = replayed.recipe.get("replay_key")
    if published is None or published == looked_up:
        return
    logger.warning(
        "studio refresh: component=%s looked_up=%s published=%s roots=%s replayed_roots=%s",
        component_id,
        looked_up,
        published,
        list(identities),
        [
            root_identity(root["dataset"], enrichment_stamp=root.get("enrichment_stamp"))
            for root in replayed.recipe.get("roots", ())
        ],
    )


@dataclass(kw_only=True)
class _RootCheck:
    """One root's frame refresh answer, whether its replay mints party kinds (`kinds`,
    `None` when the record does not say, and it stays `None` when the root's version lists
    no stamp, since only a listed stamp makes `_settle_kinds` read the stored
    definition), and the stamp a replay names it under (`stamp`, `_listed_stamp`)."""

    result: RefreshResult
    kinds: bool | None
    stamp: str | None = None

    def identity(self) -> dict[str, str]:
        """The root's identity on the dataset pair the refresh answered
        (`common.recipe.root_identity`). Not for a `Pending` answer."""
        assert not isinstance(self.result, live_apis.Pending)
        return root_identity(self.result.frame.as_json(), enrichment_stamp=self.stamp)


def _kinds_from_record(record: MintRecord) -> bool | None:
    """Whether a replay mints `record`'s root with party kinds, as far as the record
    says, or `None` when it depends on the stored definition. A recorded stamp says
    yes: only a mint that asked for kinds keeps one (`recorded_stamp`), and a replay of
    the same `spec_id` reads the same identity. Recorded columns decide on their own
    (`replayed_mint_options` reads the identity only when none are recorded)."""
    if record.enrichment_stamp is not None:
        return True
    if not record.options.get("columns"):
        return None
    replayed = replayed_mint_options(record.options, identity=None)
    return requests_kinds(replayed["columns"])


def _listed_stamp(
    record: MintRecord, result: RefreshResult, versions: Sequence[DatasetVersion]
) -> str | None:
    """The stamp a replay names `record`'s kind root under: the `enrichment_stamp`
    `versions`, the versions route's answer, lists on the root's current version
    (`result`'s pair), which is the stamp of the kind file a mint of that version
    serves. apis lists one exactly when the version's frame holds a party role, and
    none on a frame with no party role, whose kind file carries none. A version the
    route does not list gives `None`; the replay then publishes under the stamp the
    next refresh finds listed.

    A root recorded with a stamp whose version lists none is logged, since the
    recorded stamp shows its frame held a party role when it was minted.

    Not for a `Pending` answer.
    """
    assert not isinstance(result, live_apis.Pending)
    version = next((v for v in versions if v.content_version == result.frame.content_version), None)
    stamp = None if version is None else version.enrichment_stamp
    if stamp is None and record.enrichment_stamp is not None:
        logger.warning(
            "studio refresh: dataset=%s version=%s lists no enrichment_stamp for a root "
            "recorded with one",
            record.dataset.spec_id,
            result.frame.content_version,
        )
    return stamp


async def _gathered[T](calls: Iterable[Coroutine[Any, Any, T]]) -> list[T]:
    """The results of `calls`, run concurrently, in order.

    Raises:
        Exception: The first failure among the calls, after which the rest are cancelled.
    """
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(call) for call in calls]
    except ExceptionGroup as failures:
        raise failures.exceptions[0] from None
    return [task.result() for task in tasks]


class LiveRefresh:
    """The refresh of one card, and the limits every refresh in this process shares."""

    def __init__(
        self,
        *,
        apis: LiveApisClient,
        call_tool: CallTool,
        publish: PublishFrame,
        row_budget: RowBudget,
        deadline_s: float,
        requests_per_s: float,
        burst: int,
        max_in_flight: int,
        grace_s: float = HARD_STOP_GRACE_S,
        replay: Callable[..., Awaitable[Replayed | Pending | Refused]] = replay_recipe,
        closers: Sequence[Callable[[], Awaitable[None]]] = (),
    ) -> None:
        """Build the refresh over its dependencies.

        Args:
            apis: The apis client, holding both principals.
            call_tool: Calls an analysis-mcp tool by name, for the replayer.
            publish: Publishes a replayed frame with its recipe.
            row_budget: The rows replays may hold at once, apart from the
                process-wide budget.
            deadline_s: Seconds a refresh has before the replayer starts no new step.
            requests_per_s: Refill rate of the route's token bucket.
            burst: Capacity of that bucket.
            max_in_flight: Refresh runs the route executes at once.
            grace_s: Seconds after the deadline at which a run is cancelled.
            replay: `harness.recipe_replay.replay_recipe`, replaced in tests.
            closers: Called by `aclose`, to release what the dependencies hold.
        """
        self._apis = apis
        self._call_tool = call_tool
        self.publish = publish
        self.row_budget = row_budget
        self._deadline_s = deadline_s
        self._grace_s = grace_s
        self._bucket = TokenBucketRateLimiter(capacity=burst, refill_rate=requests_per_s)
        self._max_in_flight = max_in_flight
        self._in_flight = 0
        self._replay_recipe = replay
        self._closers = closers
        # The run of each card being refreshed now, so concurrent requests for one
        # card replay it once.
        self._runs: dict[tuple[str, int], asyncio.Task[_Answer]] = {}

    async def aclose(self) -> None:
        """Release the dependencies' connections."""
        for close in self._closers:
            await close()

    async def dataset_versions(self, *, spec_id: str) -> tuple[DatasetVersion, ...]:
        """The stored versions of the dataset `spec_id` names, newest first,
        read through this refresh's apis client: the reader
        `data_mcp.mint_recovery.recover_mint_record` rebuilds a root's mint
        record from (`harness.mcp_client.McpExecutor.publish_result`)."""
        return await self._apis.list_versions(spec_id=spec_id)

    async def answer(self, *, component_id: str, revision: int) -> _Answer:
        """The answer to one admitted request, drawing on the route's rate bucket
        and joining a run already under way for the card. A new run takes a
        slot of the in-flight cap, which it holds until it ends, whether or not
        anybody still waits on it."""
        if not await self._bucket.consume():
            return _pending(max(1.0, 1 / self._bucket.refill_rate), reason="rate")
        key = (component_id, revision)
        # The map's read, the cap's check and the count's update have no await
        # between them, so concurrent requests on the event loop need no lock.
        run = self._runs.get(key)
        if run is None:
            if self._in_flight >= self._max_in_flight:
                return _pending(IN_FLIGHT_RETRY_AFTER_S, reason="in_flight")
            self._in_flight += 1
            run = asyncio.create_task(self._run(component_id=component_id, revision=revision))
            self._runs[key] = run
            run.add_done_callback(lambda _: self._release(key))
        # Shielded, so a caller that goes away leaves the run to the others
        # waiting on it. The run ends itself at its hard stop.
        return await asyncio.shield(run)

    def _release(self, key: tuple[str, int]) -> None:
        """Free the slot and the map entry of the run for `key`, once it has ended."""
        del self._runs[key]
        self._in_flight -= 1

    async def _run(self, *, component_id: str, revision: int) -> _Answer:
        """One refresh under its deadline and hard stop, with every expected
        failure turned into an answer."""
        deadline = CallDeadline.after(self._deadline_s)
        try:
            async with asyncio.timeout(self._deadline_s + self._grace_s) as hard_stop:
                return await self._refresh(
                    component_id=component_id, revision=revision, deadline=deadline
                )
        except TimeoutError:
            if not hard_stop.expired():
                raise
            return _pending(RESUME_RETRY_AFTER_S, reason="hard_stop")
        except _Stop as stop:
            return stop.answer
        except LiveApisNotConfigured as exc:
            return _refusal(NOT_CONFIGURED, str(exc))
        except LiveApisError as exc:
            return _apis_refusal(exc)
        except ToolSurfaceUnavailableError as exc:
            return _refusal(UPSTREAM_UNAVAILABLE, f"an analysis tool call did not complete: {exc}")

    async def _refresh(
        self, *, component_id: str, revision: int, deadline: CallDeadline
    ) -> _Answer:
        frame = await self._derived_frame(component_id=component_id, revision=revision)
        recipe = await self._apis.read_recipe(
            spec_id=frame.spec_id, content_version=frame.content_version
        )
        if recipe is None:
            raise _Refusal(
                RECIPE_MISSING,
                f"dfo-apis holds no recipe for {frame.spec_id} at {frame.content_version}",
            )
        normal_form_hash, records = _normal_form_and_roots(recipe)

        checks = await self._check_roots(records)
        waits = [
            check.result.retry_after_s
            for check in checks
            if isinstance(check.result, live_apis.Pending)
        ]
        if waits:
            return _pending(max(waits), reason="root_rebuilding")
        # A failed versions read refuses the refresh even when every pair stands: the
        # listed stamp is all that says whether the party kinds the card was drawn with
        # still stand, so `current` would vouch for kinds nothing checked. A retryable
        # failure ends as `upstream_unavailable`, which a retry can clear
        # (`REFRESH_CODES`); a non-retryable 4xx passes on apis' own code
        # (`_apis_refusal`).
        listed: dict[str, tuple[DatasetVersion, ...]] = {}
        kind_roots = [
            (record, check) for record, check in zip(records, checks, strict=True) if check.kinds
        ]
        stamps = await self._listed_stamps(kind_roots, listed)
        for (_, check), stamp in zip(kind_roots, stamps, strict=True):
            check.stamp = stamp
        recorded = [
            root_identity(record.dataset.model_dump(), enrichment_stamp=record.enrichment_stamp)
            for record in records
        ]
        if all(isinstance(check.result, live_apis.Current) for check in checks) and (
            [check.identity() for check in checks] == recorded
        ):
            return _CURRENT

        await self._settle_kinds(records, checks, listed, deadline=deadline)
        identities = [check.identity() for check in checks]
        key = replay_key(
            normal_form_hash=normal_form_hash,
            root_datasets=identities,
            execution_identities=[current_execution_identity().canonical_string()],
        )
        target = await self._apis.find_replay(replay_key=key)
        if target is None:
            replayed = await self.replay(recipe, deadline=deadline)
            if isinstance(replayed, Pending | Refused):
                return _answer_of(replayed)
            _log_another_key(component_id, looked_up=key, identities=identities, replayed=replayed)
            target = FramePair(replayed.frame.spec_id, replayed.frame.content_version)

        render = await self._apis.render(component_id=component_id, revision=revision, frame=target)
        return _Answer(status=200, body={"status": "redrawn", "render": render}, outcome="redrawn")

    async def _derived_frame(self, *, component_id: str, revision: int) -> FramePair:
        """The derived frame the stored revision is drawn on.

        Raises:
            _Refusal: `component_not_found`, `frame_unsupported`, or `dataset_frame`.
        """
        try:
            component = await self._apis.read_component(
                component_id=component_id, revision=revision
            )
        except LiveApisError as exc:
            if exc.status == 404:
                raise _Refusal(
                    COMPONENT_NOT_FOUND, f"dfo-apis holds no revision {revision} of {component_id}"
                ) from exc
            raise
        frame = component.frame_ref
        if frame is None or not frame.spec_id.startswith((_DATASET_PREFIX, _DERIVED_PREFIX)):
            raise _Refusal(FRAME_UNSUPPORTED, "the stored intent names no dataset or derived frame")
        if frame.spec_id.startswith(_DATASET_PREFIX):
            raise _Refusal(
                DATASET_FRAME,
                f"{frame.spec_id} is a dataset frame; dfo-apis' frame refresh serves it",
            )
        return frame

    async def _check_roots(self, records: Sequence[MintRecord]) -> list[_RootCheck]:
        """apis' frame refresh answer for each root's dataset pair, in order, at
        most `MAX_CONCURRENT_ROOT_CHECKS` at once, with whether its replay mints party
        kinds as far as its record says (`_kinds_from_record`). A root whose record does
        not say is left to `_settle_kinds` and counts as minted without kinds until then.

        Raises:
            LiveApisError: The first failure among the calls, after which the rest are cancelled.
        """
        slots = asyncio.Semaphore(MAX_CONCURRENT_ROOT_CHECKS)

        async def check(record: MintRecord) -> _RootCheck:
            async with slots:
                result = await self._apis.refresh_frame(
                    spec_id=record.dataset.spec_id,
                    content_version=record.dataset.content_version,
                )
            return _RootCheck(result=result, kinds=_kinds_from_record(record))

        return await _gathered(check(record) for record in records)

    async def _listed_stamps(
        self,
        roots: Sequence[tuple[MintRecord, _RootCheck]],
        listed: dict[str, tuple[DatasetVersion, ...]],
    ) -> list[str | None]:
        """The stamp the versions route lists on the current version of each of
        `roots` (`_listed_stamp`), in order. `listed` holds the route's answer for each
        dataset this refresh has read it for, and gains one for each dataset of
        `roots` it lacks, so a refresh reads the route once per `spec_id`.

        Raises:
            LiveApisError: The first failed read, after which the rest are cancelled.
        """
        unread = list(
            dict.fromkeys(
                record.dataset.spec_id
                for record, _ in roots
                if record.dataset.spec_id not in listed
            )
        )
        answers = await _gathered(self._apis.list_versions(spec_id=spec_id) for spec_id in unread)
        listed.update(zip(unread, answers, strict=True))
        return [
            _listed_stamp(record, check.result, listed[record.dataset.spec_id])
            for record, check in roots
        ]

    async def _settle_kinds(
        self,
        records: Sequence[MintRecord],
        checks: Sequence[_RootCheck],
        listed: dict[str, tuple[DatasetVersion, ...]],
        *,
        deadline: CallDeadline,
    ) -> None:
        """Set the stamp a replay names each root under whose record does not say
        whether its replay mints party kinds (`kinds` is `None`). A root whose current
        version lists no stamp is named under its pair either way, so its stored
        definition is not read. For any other, the options its replay sends decide
        (`data_mcp.get_data_tool.recorded_mint_request`), and one whose replay mints
        kinds is named under the listed stamp. Only a refresh that goes on to a replay
        key calls this.

        The definition read runs under `EXTERNAL_CALL_BUDGET`, the budget the replay's
        mints run under. The refresh deadline, `Settings.live_refresh_deadline_s`, is
        at most 52 s, so `DIRECT_CALL_BUDGET`'s 50 s pre-mint reserve would leave a
        definition read no time.

        Raises:
            LiveApisError: The first failure among the calls, after which the rest are cancelled.
            _Stop: A definition read refused, as a wait or a refusal.
        """
        undecided = [
            (record, check)
            for record, check in zip(records, checks, strict=True)
            if check.kinds is None
        ]
        stamps = await self._listed_stamps(undecided, listed)

        async def settle(record: MintRecord, check: _RootCheck, stamp: str) -> None:
            with use_call_budget(EXTERNAL_CALL_BUDGET):
                request = await recorded_mint_request(record, deadline=deadline)
            if isinstance(request, ToolResult):
                raise _Stop(_answer_of(refusal_outcome(request.structured_content, stage="mint")))
            _, options = request
            check.kinds = requests_kinds(options.get("columns") or ())
            if check.kinds:
                check.stamp = stamp

        await _gathered(
            settle(record, check, stamp)
            for (record, check), stamp in zip(undecided, stamps, strict=True)
            if stamp is not None
        )

    async def replay(
        self,
        recipe: Mapping[str, Any],
        *,
        deadline: CallDeadline,
        populations: Sequence[Mapping[str, Any]] | None = None,
    ) -> Replayed | Pending | Refused:
        """Replay `recipe` with this refresh's clients, publisher, and row budget.

        A dataset that moved between a root's mint and its fetch refuses
        `upstream_version_changed`. The replay runs once more, and a second such
        refusal is a wait. `populations` replaces the populations a rolling recipe
        recorded (`harness.recipe_replay.replay_recipe`). A caller outside the
        route draws on neither its token bucket nor its in-flight cap.

        Raises:
            LiveApisNotConfigured: The datasets principal is unset.
            LiveApisError: An apis service call the replayer does not turn into an outcome.
            ToolSurfaceUnavailableError: An analysis tool call did not complete.
        """
        for _ in range(2):
            outcome = await self._replay_recipe(
                recipe,
                deadline=deadline,
                call_tool=self._call_tool,
                store=get_store(),
                apis=self._apis,
                publish=self.publish,
                row_budget=self.row_budget,
                populations=populations,
            )
            if not (
                isinstance(outcome, Refused)
                and outcome.code == ToolErrorCode.UPSTREAM_VERSION_CHANGED
            ):
                return outcome
        return Pending(retry_after_s=BUSY_RETRY_AFTER_S, reason="busy")


def build_live_refresh(*, settings: Settings) -> LiveRefresh:
    """The process's `LiveRefresh`, built from settings. Opens no connection.

    apis' origin and the datasets principal are presentation-mcp's
    (`DFO_APIS_URL`, `DATASETS_FIRST_PARTY_PRINCIPAL`), since the publish goes
    through its `PlatformAssembler`, built here so it does not depend on the
    presentation tools having mounted. The verbs run on a tool surface of the
    refresh's own over the loopback mounts.

    Raises:
        ValueError: `live_refresh_rows_in_flight_max` is below twice
            `max_frame_rows`, so a two-input verb on two full frames could never
            be replayed.
    """
    presentation = PresentationSettings()
    if settings.live_refresh_rows_in_flight_max < 2 * presentation.max_frame_rows:
        raise ValueError(
            f"live_refresh_rows_in_flight_max ({settings.live_refresh_rows_in_flight_max}) is "
            f"below twice max_frame_rows ({presentation.max_frame_rows}), so a two-input verb "
            "on two max_frame_rows inputs could never be replayed; raise "
            "LIVE_REFRESH_ROWS_IN_FLIGHT_MAX"
        )
    auth = apis_auth_provider(base_url=presentation.dfo_apis_url)
    apis = LiveApisClient(
        base_url=presentation.dfo_apis_url,
        datasets_principal=presentation.datasets_first_party_principal,
        live_refresh_principal=settings.live_refresh_principal,
        auth=auth,
    )
    assembler = PlatformAssembler(settings=presentation, auth=auth)
    surface = build_tool_surface(settings=settings)
    return LiveRefresh(
        apis=apis,
        call_tool=_analysis_tools(surface),
        publish=assembler.publish_frame,
        row_budget=RowBudget(max_rows=settings.live_refresh_rows_in_flight_max),
        deadline_s=settings.live_refresh_deadline_s,
        requests_per_s=settings.live_refresh_requests_per_s,
        burst=settings.live_refresh_burst,
        max_in_flight=settings.live_refresh_max_in_flight,
        closers=(apis.aclose, assembler.aclose, surface.aclose),
    )


def _analysis_tools(surface: McpSurface) -> CallTool:
    async def call_tool(*, name: str, arguments: Mapping[str, object]) -> ToolAnswer:
        return await surface.call_tool(
            tool=ToolId(server="analysis-mcp", name=name), arguments=arguments
        )

    return call_tool


def create_live_refresh_router(*, settings: Settings) -> APIRouter:
    """The router of `POST /api/studio/refresh`.

    The route reads the `LiveRefresh` from `app.state.live_refresh` on each
    request. It takes only the live-refresh principal. `harness.auth.principal_for`,
    which the other `/api/studio` routes call, is not consulted, so
    `STUDIO_API_TOKEN` does not open it.
    """
    router = APIRouter(prefix="/api/studio")

    @router.post("/refresh")
    async def refresh(request: Request, authorization: str | None = Header(default=None)) -> Any:
        if not live_refresh_admits(authorization, settings=settings):
            answer = _refusal(UNAUTHORIZED, "this route takes the live-refresh principal")
            response = answer.response()
            response.headers["WWW-Authenticate"] = "Bearer"
            return response
        try:
            body = _RefreshRequest.model_validate_json(await request.body())
        except ValidationError as exc:
            return _refusal(
                REQUEST_INVALID,
                f"the body is not {{component_id, revision}}: {exc.error_count()} error(s)",
            ).response()

        service: LiveRefresh = request.app.state.live_refresh
        started = time.monotonic()
        with use_request_span("studio.refresh") as span:
            answer = await service.answer(component_id=body.component_id, revision=body.revision)
            span.record_outcome(level=_level(answer), status_message=answer.outcome)
        logger.info(
            "studio refresh: component=%s revision=%d status=%d outcome=%s elapsed_ms=%d",
            body.component_id,
            body.revision,
            answer.status,
            answer.outcome,
            (time.monotonic() - started) * 1000,
        )
        return answer.response()

    return router
