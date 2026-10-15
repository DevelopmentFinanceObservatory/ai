"""`POST /api/studio/refresh`: a card drawn on a derived frame, on current data.

The website calls this server to server for one card of a live share. The
request names a stored component revision, `{"component_id", "revision"}`, and
carries `Authorization: Bearer <LIVE_REFRESH_PRINCIPAL>`
(`harness.auth.live_refresh_admits`). No conversation, model, or session is
involved, and nothing about a refresh is kept between requests beyond the frames
in the materialisation store and the map of runs in flight.

One refresh (`LiveRefresh._refresh`):

1. Reads the component revision from dfo-apis and takes its stored intent's
   frame. A dataset (`spec_`) frame is refused. The website refreshes those
   against dfo-apis itself.
2. Reads the frame's recipe from dfo-apis.
3. Asks dfo-apis' frame refresh about each root's dataset pair, at most
   `MAX_CONCURRENT_ROOT_CHECKS` at once. A root still building answers
   `pending`, and every root current answers `current`.
4. Computes the replay key of the recipe on the roots' current pairs
   (`common.recipe.replay_key`) and asks dfo-apis for the frame stored under it.
   A hit is rendered without replaying.
5. Otherwise replays the recipe (`harness.recipe_replay.replay_recipe`), which
   mints the roots, runs the verbs, and publishes the frame with its recipe.
   The replayer reads the replay key once more before it publishes, under the
   key of the recipe it built, so a frame another instance published while this
   one replayed is not sent twice.
6. Has dfo-apis render the stored revision on the frame, and answers the render
   as dfo-apis gave it.

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
from collections.abc import Awaitable, Callable, Mapping, Sequence
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
from common.recipe import replay_key
from common.row_budget import RowBudget
from common.telemetry import EventLevel, use_request_span
from compiler.surface import McpSurface, ToolSurfaceUnavailableError
from data_mcp.deadline import CallDeadline
from data_mcp.tool_errors import ToolErrorCode
from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse
from fastmcp.server.middleware.rate_limiting import TokenBucketRateLimiter
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

# Frame refresh calls to dfo-apis in flight at once for one card's roots. Each can
# take one of the two materialise slots a dfo-apis instance has.
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
# permanent: a dfo-apis refusal that is not retryable (frame refresh, replay
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
    """What a failed dfo-apis call makes of the refresh: `upstream_unavailable` for
    no answer, a 5xx, or a refusal dfo-apis marks retryable, else its own code."""
    if exc.retryable or exc.status is None or exc.status >= 500:
        return _refusal(UPSTREAM_UNAVAILABLE, f"dfo-apis did not serve the call: {exc}")
    return _refusal(exc.code or DFO_APIS_REFUSED, exc.detail or str(exc))


def _level(answer: _Answer) -> EventLevel:
    if answer.status >= 500:
        return "ERROR"
    return "WARNING" if answer.status >= 400 else "DEFAULT"


class _Refusal(Exception):  # noqa: N818 -- a control-flow signal carrying an answer
    """The refresh ended in the refusal `code`. `answer` is what the route answers."""

    def __init__(self, code: str, detail: str) -> None:
        self.answer = _refusal(code, detail)
        super().__init__(code)


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
            apis: The dfo-apis client, holding both principals.
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
        read through this refresh's dfo-apis client: the reader
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
        except _Refusal as refusal:
            return refusal.answer
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
        waits = [check.retry_after_s for check in checks if isinstance(check, live_apis.Pending)]
        if waits:
            return _pending(max(waits), reason="root_rebuilding")
        if all(isinstance(check, live_apis.Current) for check in checks):
            return _CURRENT

        key = replay_key(
            normal_form_hash=normal_form_hash,
            root_datasets=[
                check.frame.as_json()
                for check in checks
                if not isinstance(check, live_apis.Pending)
            ],
            execution_identities=[current_execution_identity().canonical_string()],
        )
        target = await self._apis.find_replay(replay_key=key)
        if target is None:
            replayed = await self.replay(recipe, deadline=deadline)
            if isinstance(replayed, Pending):
                return _pending(replayed.retry_after_s, reason=replayed.reason)
            if isinstance(replayed, Refused):
                raise _Refusal(replayed.code, replayed.detail)
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

    async def _check_roots(self, records: Sequence[MintRecord]) -> list[RefreshResult]:
        """dfo-apis' frame refresh answer for each root's dataset pair, in order.

        Raises:
            LiveApisError: The first failure among the calls, after which the rest are cancelled.
        """
        slots = asyncio.Semaphore(MAX_CONCURRENT_ROOT_CHECKS)

        async def check(record: MintRecord) -> RefreshResult:
            async with slots:
                return await self._apis.refresh_frame(
                    spec_id=record.dataset.spec_id, content_version=record.dataset.content_version
                )

        try:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(check(record)) for record in records]
        except ExceptionGroup as failures:
            raise failures.exceptions[0] from None
        return [task.result() for task in tasks]

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
            LiveApisError: A dfo-apis call the replayer does not turn into an outcome.
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

    dfo-apis' origin and the datasets principal are presentation-mcp's
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
