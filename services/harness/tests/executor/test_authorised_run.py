"""A run's ledger record around execution: admission before the first dispatch,
an attempt before every call, the sealed manifest checked against the
authorisation before any derived step, and the completion gate that holds
`completed` back until every derived result is published.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from common.materialisation import FrameRef
from compiler.compile import compile_plan
from compiler.surface import ProgressCallback, ToolCallResult, ToolSchema
from executor.completion import (
    PUBLISH_ATTEMPTS,
    FramePublished,
    PublishFault,
    unrecorded_hand_back,
)
from executor.ledger import (
    MAX_PENDING_AGE,
    Attempt,
    Complete,
    Fail,
    InMemoryRunLedger,
    LedgerUnavailableError,
    PlanApproval,
    Published,
    Publishing,
    RequestAuthorisation,
    RunChange,
    RunRecord,
    RunStatus,
    Seal,
    SourcePolicy,
    idempotency_key,
    submission_content,
    submission_digest,
)
from executor.outcomes import Accepted, RefusalKind, Refused
from executor.run import (
    PLAN_APPROVAL_TTL,
    Submission,
    current_step_id,
    default_fetch_series_stats,
    execute_approved_plan,
    execute_authorised,
    load_submission,
    resume_run,
)
from google.adk.errors import StaleSessionError
from harness.graph.resume import run_resumer
from harness.graph.state import ORCHESTRATOR_FRAMES_KEY
from harness.studio_state import SegmentLost
from planner.operations import RetrieveOperation, Translation
from planner.plan import AnswerMode, DatasetRequirement, Decision, Plan, PlanStatus
from planner.workflow import SelectedSource, ToolId

from .conftest import (
    CONTENT_VERSION,
    RESULT_FRAME,
    SPEC_KENYA,
    SPEC_POPULATION,
    ScriptedSurface,
    ok,
)

PlanFactory = Callable[..., Plan]
ManifestFactory = Callable[..., dict[str, Any]]

DERIVED = FrameRef(spec_id=RESULT_FRAME["spec_id"], content_version=RESULT_FRAME["content_version"])
REPLAY_KEY = "rpk_" + "e" * 64


class _Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


class _Publisher:
    """Answers each publish from `answers` in turn, repeating the last, and
    records every frame it was asked to publish."""

    def __init__(self, *answers: FramePublished | PublishFault) -> None:
        self.answers = list(answers)
        self.frames: list[FrameRef] = []

    async def __call__(self, frame: FrameRef) -> FramePublished | PublishFault:
        self.frames.append(frame)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


class _ProcessDiedError(BaseException):
    """The process running a run stopped at this point: nothing in the run
    handles it, as nothing handles a process dying."""


class _Dies:
    """A `RunLedger` over `alive` that loses every write once `dies_at`
    accepts a change, as a ledger seen by a process that died there."""

    def __init__(self, alive: InMemoryRunLedger, dies_at: Callable[[RunChange], bool]) -> None:
        self.alive = alive
        self.dies_at = dies_at
        self.dead = False

    async def admit(self, candidate: RunRecord, *, submission: Any) -> Any:
        return await self.alive.admit(candidate, submission=submission)

    async def get(self, run_id: str) -> RunRecord | None:
        return await self.alive.get(run_id)

    async def submission(self, run_id: str) -> Any:
        return await self.alive.submission(run_id)

    async def retrievals(self, run_id: str) -> Any:
        return await self.alive.retrievals(run_id)

    async def store_plan(self, revision: str, plan: Any) -> None:
        await self.alive.store_plan(revision, plan)

    async def plan(self, revision: str) -> Any:
        return await self.alive.plan(revision)

    async def unfinished(self) -> tuple[RunRecord, ...]:
        return await self.alive.unfinished()

    async def record(self, run_id: str, change: RunChange) -> RunRecord:
        self.dead = self.dead or self.dies_at(change)
        if self.dead:
            raise _ProcessDiedError(type(change).__name__)
        return await self.alive.record(run_id, change)


def _surface(real_schemas: list[ToolSchema], manifest: dict[str, Any]) -> ScriptedSurface:
    return ScriptedSurface(
        schemas=real_schemas,
        responses={"get_data": [ok(manifest)], "aggregate": [ok(RESULT_FRAME)]},
    )


def _tools(surface: ScriptedSurface) -> list[str]:
    return [tool for tool, _ in surface.calls]


async def _only_run(ledger: InMemoryRunLedger) -> RunRecord:
    (run,) = await ledger.unfinished()
    return run


HOUR = timedelta(hours=1)


def _submission(plan: Plan) -> Submission:
    assert plan.workflow is not None and plan.approved_revision is not None
    return Submission(
        revision=plan.approved_revision,
        workflow=plan.workflow,
        requirements=tuple(
            r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)
        ),
    )


def _approval(plan: Plan, sources: SourcePolicy, *, expires_at: datetime) -> PlanApproval:
    assert plan.approved_revision is not None
    return PlanApproval(
        source_policy=sources,
        expires_at=expires_at,
        plan_version=plan.version,
        approved_revision=plan.approved_revision,
        approver="u_approver",
        at=datetime.now(UTC),
    )


async def _resume(run: RunRecord, *, ledger: InMemoryRunLedger, **kwargs: Any) -> Any:
    return await resume_run(run, await load_submission(ledger, run.run_id), ledger=ledger, **kwargs)


async def test_a_run_is_admitted_and_each_attempt_recorded_before_its_call_goes_out(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    ledger = InMemoryRunLedger()
    seen: list[tuple[str | None, RunStatus, tuple[str, ...]]] = []

    class _Watching(ScriptedSurface):
        async def call_tool(
            self,
            *,
            tool: ToolId,
            arguments: Mapping[str, Any],
            on_progress: ProgressCallback | None = None,
        ) -> ToolCallResult:
            (run,) = await ledger.unfinished()
            seen.append((current_step_id.get(), run.status, tuple(a.step_id for a in run.attempts)))
            return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)

    surface = _Watching(
        schemas=real_schemas,
        responses={"get_data": [ok(manifest())], "aggregate": [ok(RESULT_FRAME)]},
    )

    _, outcome = await execute_approved_plan(
        approved_plan(),
        surface=surface,
        ledger=ledger,
        publisher=_Publisher(FramePublished(replay_key=REPLAY_KEY)),
    )

    assert isinstance(outcome, Accepted), outcome
    assert seen == [
        ("debt", RunStatus.RUNNING, ("debt",)),
        ("totals", RunStatus.RUNNING, ("debt", "totals")),
    ]
    run = await ledger.get(outcome.run_id or "")
    assert run is not None and run.status is RunStatus.COMPLETED


async def test_approving_the_same_revision_again_carries_on_the_run_a_dead_process_left(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    plan = approved_plan()
    alive = InMemoryRunLedger()
    dies = _Dies(alive, lambda change: isinstance(change, Attempt) and change.step_id == "totals")
    publisher = _Publisher(FramePublished(replay_key=REPLAY_KEY))

    with pytest.raises(_ProcessDiedError):
        await execute_approved_plan(
            plan, surface=_surface(real_schemas, manifest()), ledger=dies, publisher=publisher
        )
    left = await _only_run(alive)
    assert left.status is RunStatus.RUNNING

    surface = _surface(real_schemas, manifest())
    _, outcome = await execute_approved_plan(
        plan, surface=surface, ledger=alive, publisher=publisher
    )

    assert isinstance(outcome, Accepted), outcome
    assert outcome.run_id == left.run_id
    run = await alive.get(left.run_id)
    assert run is not None and run.status is RunStatus.COMPLETED
    assert [(a.step_id, a.attempt) for a in run.attempts] == [
        ("debt", 1),
        ("debt", 2),
        ("totals", 1),
    ]


async def test_a_run_that_died_before_publishing_publishes_once_when_carried_on(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    alive = InMemoryRunLedger()

    async def dies(frame: FrameRef) -> FramePublished:
        raise _ProcessDiedError(frame.spec_id)

    surface = _surface(real_schemas, manifest())
    with pytest.raises(_ProcessDiedError):
        await execute_approved_plan(approved_plan(), surface=surface, ledger=alive, publisher=dies)
    left = await _only_run(alive)
    assert left.status is RunStatus.PUBLISHING
    computed = len(surface.calls)

    publisher = _Publisher(FramePublished(replay_key=REPLAY_KEY))
    outcome = await _resume(left, ledger=alive, surface=surface, publisher=publisher)

    assert isinstance(outcome, Accepted), outcome
    assert outcome.run_id == left.run_id
    assert len(surface.calls) == computed
    assert publisher.frames == [DERIVED]
    run = await alive.get(left.run_id)
    assert run is not None and run.status is RunStatus.COMPLETED
    # Carried on again until a hand-back delivers its result.
    assert await alive.unfinished() == (run,)


async def test_a_recipe_that_cannot_be_built_fails_the_run_without_a_retry(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    ledger = InMemoryRunLedger()
    publisher = _Publisher(
        PublishFault(code="recipe_unbuildable", reason="no trace", retryable=False)
    )
    sleeps = _Sleeps()

    finished, outcome = await execute_approved_plan(
        approved_plan(),
        surface=_surface(real_schemas, manifest()),
        ledger=ledger,
        publisher=publisher,
        sleep=sleeps,
    )

    run = await ledger.get(outcome.run_id or "")
    assert run is not None and run.status is RunStatus.FAILED
    assert run.failure is not None and run.failure.code == "recipe_unbuildable"
    assert publisher.frames == [DERIVED]
    assert sleeps.delays == []
    assert await ledger.unfinished() == ()
    assert finished.status is PlanStatus.FAILED


async def test_executing_a_plan_whose_run_failed_admits_a_new_run(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    plan = approved_plan()
    ledger = InMemoryRunLedger()
    final = PublishFault(code="recipe_unbuildable", reason="no trace", retryable=False)
    _, failed = await execute_approved_plan(
        plan, surface=_surface(real_schemas, manifest()), ledger=ledger, publisher=_Publisher(final)
    )

    finished, outcome = await execute_approved_plan(
        plan,
        surface=_surface(real_schemas, manifest()),
        ledger=ledger,
        publisher=_Publisher(FramePublished(replay_key=REPLAY_KEY)),
    )

    assert isinstance(outcome, Accepted), outcome
    assert outcome.run_id != failed.run_id
    run = await ledger.get(outcome.run_id or "")
    assert run is not None and run.status is RunStatus.COMPLETED
    assert finished.status is PlanStatus.COMPLETED


async def test_an_approval_in_force_runs_under_a_lifetime_measured_from_when_it_was_given(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    approved_at = datetime.now(UTC) - 23 * HOUR
    ledger = InMemoryRunLedger()

    _, outcome = await execute_approved_plan(
        approved_plan(approved_at=approved_at),
        surface=_surface(real_schemas, manifest()),
        ledger=ledger,
        publisher=_Publisher(FramePublished(replay_key=REPLAY_KEY)),
    )

    assert isinstance(outcome, Accepted), outcome
    run = await ledger.get(outcome.run_id or "")
    assert run is not None
    assert run.authorisation.at == approved_at
    assert run.authorisation.expires_at == approved_at + PLAN_APPROVAL_TTL


@pytest.mark.parametrize(
    ("age", "after_a_failed_run"),
    [(25 * HOUR, False), (25 * HOUR, True), (None, False)],
    ids=["expired", "expired_after_a_failed_run", "no_approval_time"],
)
async def test_a_new_run_of_a_plan_whose_approval_is_not_in_force_dispatches_nothing(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    age: timedelta | None,
    after_a_failed_run: bool,
) -> None:
    """Admitting a new run, a retry of a failed one included, needs the approval
    in force; a plan with no approval time has none to measure from."""
    now = [datetime.now(UTC)]
    ledger = InMemoryRunLedger(clock=lambda: now[0])
    if age is None:
        plan = approved_plan().model_copy(update={"approved_at": None})
    else:
        plan = approved_plan(approved_at=now[0] - age)
    if after_a_failed_run:
        assert plan.approved_at is not None
        now[0] = plan.approved_at + HOUR
        final = PublishFault(code="recipe_unbuildable", reason="no trace", retryable=False)
        _, failed = await execute_approved_plan(
            plan,
            surface=_surface(real_schemas, manifest()),
            ledger=ledger,
            publisher=_Publisher(final),
        )
        failed_run = await ledger.get(failed.run_id or "")
        assert failed_run is not None and failed_run.status is RunStatus.FAILED
        now[0] = plan.approved_at + age
    surface = _surface(real_schemas, manifest())

    finished, outcome = await execute_approved_plan(
        plan,
        surface=surface,
        ledger=ledger,
        publisher=_Publisher(FramePublished(replay_key=REPLAY_KEY)),
    )

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.AUTHORISATION_EXPIRED
    assert outcome.run_id is None
    assert surface.calls == []
    assert await ledger.unfinished() == ()
    assert finished.status is PlanStatus.FAILED


async def test_the_stored_submission_holds_no_unset_added_field_and_reads_back_as_admitted(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    """Admission stores the content its digest covers, so a field a later
    release adds stays out of what an earlier release has to read."""
    plan = approved_plan()
    assert plan.workflow is not None
    ledger = InMemoryRunLedger()

    _, outcome = await execute_approved_plan(
        plan,
        surface=_surface(real_schemas, manifest()),
        ledger=ledger,
        publisher=_Publisher(FramePublished(replay_key=REPLAY_KEY)),
    )

    run = await ledger.get(outcome.run_id or "")
    assert run is not None
    stored = await ledger.submission(run.run_id)
    assert isinstance(stored, dict)
    (dataset,) = stored["requirements"]
    assert isinstance(dataset, dict) and "period_end_open" not in dataset
    loaded = await load_submission(ledger, run.run_id)
    assert loaded == _submission(plan)
    assert loaded.requirements[0].period_end_open is False
    digest = submission_digest(submission=loaded, source_policy=run.authorisation.source_policy)
    assert digest == run.submission_digest


@pytest.mark.parametrize(
    ("raised", "status", "code", "plan_status"),
    [
        (ConnectionResetError("reset"), RunStatus.PENDING, None, PlanStatus.EXECUTING),
        (KeyError("replay_key"), RunStatus.FAILED, "publication_error", PlanStatus.FAILED),
    ],
)
async def test_an_exception_from_the_publisher_is_classified_not_raised(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    raised: Exception,
    status: RunStatus,
    code: str | None,
    plan_status: PlanStatus,
) -> None:
    async def raises(frame: FrameRef) -> FramePublished:
        raise raised

    ledger = InMemoryRunLedger()
    finished, outcome = await execute_approved_plan(
        approved_plan(),
        surface=_surface(real_schemas, manifest()),
        ledger=ledger,
        publisher=raises,
        sleep=_Sleeps(),
    )

    run = await ledger.get(outcome.run_id or "")
    assert run is not None and run.status is status
    assert (run.failure.code if run.failure else None) == code
    assert finished.status is plan_status


async def test_a_retryable_publish_fault_holds_the_run_pending_until_a_later_call_publishes_it(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    ledger = InMemoryRunLedger()
    busy = PublishFault(code="service_unavailable", reason="busy", retryable=True, retry_after_s=7)
    publisher = _Publisher(busy, busy, FramePublished(replay_key=REPLAY_KEY))
    sleeps = _Sleeps()
    surface = _surface(real_schemas, manifest())

    _, outcome = await execute_approved_plan(
        approved_plan(),
        surface=surface,
        ledger=ledger,
        publisher=publisher,
        sleep=sleeps,
        jitter=lambda low, high: low,
    )

    run = await ledger.get(outcome.run_id or "")
    assert run is not None and run.status is RunStatus.PENDING
    assert run.pending is not None and run.pending.retry_after_s == 7
    assert sleeps.delays == [7]
    assert await ledger.unfinished() == (run,)

    await _resume(run, ledger=ledger, surface=surface, publisher=publisher)

    finished = await ledger.get(run.run_id)
    assert finished is not None and finished.status is RunStatus.COMPLETED
    assert publisher.frames == [DERIVED, DERIVED, DERIVED]


async def test_a_pending_run_finishes_publication_after_its_authorisation_expires(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    plan = approved_plan()
    now = [datetime.now(UTC)]
    ledger = InMemoryRunLedger(clock=lambda: now[0])
    busy = PublishFault(code="service_unavailable", reason="busy", retryable=True)
    publisher = _Publisher(busy, busy, FramePublished(replay_key=REPLAY_KEY))
    surface = _surface(real_schemas, manifest())
    submission = _submission(plan)
    outcome = await execute_authorised(
        submission,
        authorisation=_approval(plan, submission.source_policy(), expires_at=now[0] + HOUR),
        idempotency_key=idempotency_key(caller="u_approver", conversation_id="c", task="t"),
        ledger=ledger,
        surface=surface,
        publisher=publisher,
        sleep=_Sleeps(),
    )
    run = await ledger.get(outcome.run_id or "")
    assert run is not None and run.status is RunStatus.PENDING

    now[0] += 2 * HOUR
    await _resume(run, ledger=ledger, surface=surface, publisher=publisher)

    finished = await ledger.get(run.run_id)
    assert finished is not None and finished.status is RunStatus.COMPLETED


@pytest.mark.parametrize(
    ("later", "refusal", "code"),
    [
        (2 * HOUR, RefusalKind.AUTHORISATION_EXPIRED, "authorisation_expired"),
        (MAX_PENDING_AGE + HOUR, RefusalKind.RUN_ABANDONED, "computation_abandoned"),
    ],
    ids=["authorisation_expired", "too_old_to_resume"],
)
async def test_an_unfinished_computation_carried_on_too_late_dispatches_nothing(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    later: timedelta,
    refusal: RefusalKind,
    code: str,
) -> None:
    """The run's authorisation is in force for an hour; the resume bound
    (`MAX_PENDING_AGE`) ends a computation left unfinished, as it ends a
    pending publication."""
    now = [datetime.now(UTC)]
    alive = InMemoryRunLedger(clock=lambda: now[0])
    left = await _left_running(alive, approved_plan(), _surface(real_schemas, manifest()))

    now[0] += later
    surface = _surface(real_schemas, manifest())
    outcome = await _resume(left, ledger=alive, surface=surface, publisher=None)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is refusal
    assert surface.calls == []
    run = await alive.get(left.run_id)
    assert run is not None and run.status is RunStatus.FAILED
    assert run.failure is not None and run.failure.code == code


async def test_a_retrieval_outside_the_authorised_sources_stops_before_any_derived_step(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    plan = approved_plan()
    submission = _submission(plan)
    elsewhere = _approval(
        plan, SourcePolicy(sources=(SPEC_POPULATION,)), expires_at=datetime.now(UTC) + HOUR
    )
    ledger = InMemoryRunLedger()
    surface = _surface(real_schemas, manifest())

    outcome = await execute_authorised(
        submission,
        authorisation=elsewhere,
        idempotency_key=idempotency_key(caller="u_approver", conversation_id="c", task="t"),
        ledger=ledger,
        surface=surface,
        publisher=None,
    )

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.MANIFEST_OUTSIDE_AUTHORISATION
    assert _tools(surface) == ["get_data"]
    run = await ledger.get(outcome.run_id or "")
    assert run is not None and run.status is RunStatus.FAILED
    assert run.manifest is None


def _retrieval_plan(
    decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    source: Callable[..., SelectedSource],
    real_schemas: list[ToolSchema],
) -> Plan:
    """An approved plan answered by its one retrieval."""
    dataset = requirement()
    draft = Plan(
        question="What has Kenya paid to service its debt each year?",
        answer_mode=AnswerMode.SERIES,
        decisions=[decision()],
        evidence_requirements=[dataset],
        status=PlanStatus.DRAFT,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id=dataset.id, requirement_id=dataset.id, spec_id=source().spec_id)
        ],
        terminal=[dataset.id],
    )
    compiled = compile_plan(
        plan=draft,
        translation=translation,
        schemas={schema.name: schema for schema in real_schemas},
        discovered={dataset.id: source()},
    )
    assert compiled.workflow is not None, compiled.issues
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve(
        at=datetime.now(UTC)
    )


async def test_a_run_answered_by_its_retrieval_completes_with_nothing_to_publish(
    decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    source: Callable[..., SelectedSource],
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
) -> None:
    plan = _retrieval_plan(decision, requirement, source, real_schemas)

    async def never(frame: FrameRef) -> FramePublished:
        raise AssertionError(f"a root frame is not published: {frame}")

    ledger = InMemoryRunLedger()
    _, outcome = await execute_approved_plan(
        plan,
        surface=ScriptedSurface(schemas=real_schemas, responses={"get_data": [ok(manifest())]}),
        ledger=ledger,
        publisher=never,
    )

    assert isinstance(outcome, Accepted), outcome
    run = await ledger.get(outcome.run_id or "")
    assert run is not None and run.status is RunStatus.COMPLETED
    assert run.manifest is not None
    assert [entry.spec_id for entry in run.manifest.entries] == [source().spec_id]


@pytest.mark.parametrize(
    ("answer", "listed"),
    [
        ({"outcome": "ok", "view": {}}, False),
        (
            {"outcome": "refusal", "refusal": {"code": "gone", "retry_mode": "do_not_retry"}},
            True,
        ),
    ],
    ids=["described", "refused"],
)
async def test_a_carried_on_run_describes_its_frames_by_the_pair_dfo_apis_serves(
    decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    source: Callable[..., SelectedSource],
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    answer: dict[str, Any],
    listed: bool,
) -> None:
    """The retrieval was minted with extra columns, under a pair of data-mcp's
    own; it is described, as `plan_done` describes it, by the manifest's
    pair and registered with data-mcp's as the store's, and a refusal is
    listed not chartable."""
    minted = {"spec_id": SPEC_KENYA, "content_version": "cv_" + "7" * 32}
    ledger = InMemoryRunLedger()
    surface = ScriptedSurface(
        schemas=real_schemas, responses={"get_data": [ok(manifest(frame_ref=minted))]}
    )
    _, outcome = await execute_approved_plan(
        _retrieval_plan(decision, requirement, source, real_schemas),
        surface=surface,
        ledger=ledger,
        publisher=None,
    )
    assert isinstance(outcome, Accepted), outcome
    described: list[dict[str, Any]] = []

    async def describes(*, tool: str, args: dict[str, Any]) -> Any:
        described.append(args["frame_ref"])
        return answer

    resume = run_resumer(
        run_ledger=ledger,
        publisher=None,
        executor=describes,
        tool_surface=surface,
        max_concurrent_steps=4,
        fetch_series_stats=default_fetch_series_stats,
    )

    ctx = _Context()
    handed_back = await resume(ctx)  # type: ignore[arg-type] -- reads only `state`

    served = {"spec_id": SPEC_KENYA, "content_version": CONTENT_VERSION}
    assert described == [served]
    registered = (ctx.state.get(ORCHESTRATOR_FRAMES_KEY) or {}).values()
    assert [(frame["ref"], frame["store_ref"]) for frame in registered] == (
        [] if listed else [(served, minted)]
    )
    assert handed_back is not None
    (standing,) = _runs(handed_back.hand_back["text"])
    assert len(standing["frames"]) == (0 if listed else 1)
    assert ("frames_not_chartable" in standing) is listed


async def test_a_pending_run_past_its_age_fails_without_another_publish(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    now = [datetime.now(UTC)]
    ledger = InMemoryRunLedger(clock=lambda: now[0])
    busy = PublishFault(code="service_unavailable", reason="busy", retryable=True)
    publisher = _Publisher(busy)
    surface = _surface(real_schemas, manifest())
    _, outcome = await execute_approved_plan(
        approved_plan(), surface=surface, ledger=ledger, publisher=publisher, sleep=_Sleeps()
    )
    run = await ledger.get(outcome.run_id or "")
    assert run is not None and run.status is RunStatus.PENDING

    now[0] += MAX_PENDING_AGE + HOUR
    await _resume(run, ledger=ledger, surface=surface, publisher=publisher)

    finished = await ledger.get(run.run_id)
    assert finished is not None and finished.status is RunStatus.FAILED
    assert finished.failure is not None and finished.failure.code == "publication_abandoned"
    assert len(publisher.frames) == PUBLISH_ATTEMPTS


class _Context:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}


def _runs(text: str) -> list[dict[str, Any]]:
    """The run standings a resume hand-back's text lists."""
    return json.loads(text.split("\n", 1)[1])["runs"]


async def _describes(*, tool: str, args: dict[str, Any]) -> Any:
    return {"outcome": "ok", "view": {}}


class _Broken(ScriptedSurface):
    async def call_tool(self, **_: Any) -> ToolCallResult:
        raise RuntimeError("a defect")


async def test_carrying_runs_on_records_each_that_cannot_be_and_never_raises(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    """One run's stored submission is unreadable and another's computation
    raises; each is recorded failed and the turn goes on."""
    alive = InMemoryRunLedger()
    for task in ("a", "b"):
        dies = _Dies(alive, lambda change: isinstance(change, Attempt))
        submission = _submission(approved_plan())
        with pytest.raises(_ProcessDiedError):
            await execute_authorised(
                submission,
                authorisation=_approval(
                    approved_plan(), submission.source_policy(), expires_at=datetime.now(UTC) + HOUR
                ),
                idempotency_key=idempotency_key(caller="u", conversation_id="c", task=task),
                ledger=dies,
                surface=_surface(real_schemas, manifest()),
                publisher=None,
            )
    unreadable, broken = await alive.unfinished()
    alive._book().submissions[unreadable.run_id] = {"shape": "of an older release"}
    resume = run_resumer(
        run_ledger=alive,
        publisher=None,
        executor=_describes,
        tool_surface=_Broken(schemas=real_schemas, responses={}),
        max_concurrent_steps=4,
        fetch_series_stats=default_fetch_series_stats,
    )

    handed_back = await resume(_Context())  # type: ignore[arg-type] -- reads only `state`

    assert handed_back is not None
    assert await alive.unfinished() == ()
    codes = {
        run_id: (await alive.get(run_id)).failure.code  # type: ignore[union-attr]
        for run_id in (unreadable.run_id, broken.run_id)
    }
    assert codes == {unreadable.run_id: "run_record_unreadable", broken.run_id: "internal_error"}


class _Flaky:
    """A `RunLedger` over `alive` whose `submission` raises `fault` on its
    first `faults` reads, or with `at` its `record` of each `at` change does,
    or with `on_admit` its `admit` does, as a store that is briefly
    unreachable does."""

    def __init__(
        self,
        alive: InMemoryRunLedger,
        fault: Exception,
        *,
        faults: int = 1,
        at: type[RunChange] | None = None,
        on_admit: bool = False,
    ) -> None:
        self.alive = alive
        self.fault = fault
        self.faults = faults
        self.at = at
        self.on_admit = on_admit

    async def admit(self, candidate: RunRecord, *, submission: Any) -> Any:
        if self.on_admit:
            self._fault()
        return await self.alive.admit(candidate, submission=submission)

    async def get(self, run_id: str) -> RunRecord | None:
        return await self.alive.get(run_id)

    def _fault(self) -> None:
        if self.faults:
            self.faults -= 1
            raise self.fault

    async def submission(self, run_id: str) -> Any:
        if self.at is None:
            self._fault()
        return await self.alive.submission(run_id)

    async def retrievals(self, run_id: str) -> Any:
        return await self.alive.retrievals(run_id)

    async def store_plan(self, revision: str, plan: Any) -> None:
        await self.alive.store_plan(revision, plan)

    async def plan(self, revision: str) -> Any:
        return await self.alive.plan(revision)

    async def unfinished(self) -> tuple[RunRecord, ...]:
        return await self.alive.unfinished()

    async def record(self, run_id: str, change: RunChange) -> RunRecord:
        if self.at is not None and isinstance(change, self.at):
            self._fault()
        return await self.alive.record(run_id, change)


async def _left_running(
    alive: InMemoryRunLedger, plan: Plan, surface: ScriptedSurface
) -> RunRecord:
    """The run of `plan` a process left `running` in `alive`."""
    submission = _submission(plan)
    dies = _Dies(alive, lambda change: isinstance(change, Attempt))
    with pytest.raises(_ProcessDiedError):
        await execute_authorised(
            submission,
            authorisation=_approval(
                plan, submission.source_policy(), expires_at=datetime.now(UTC) + HOUR
            ),
            idempotency_key=idempotency_key(caller="u", conversation_id="c", task="t"),
            ledger=dies,
            surface=surface,
            publisher=None,
        )
    return await _only_run(alive)


class _TooManyConnectionsError(Exception):
    """As `asyncpg.TooManyConnectionsError`: a store fault at connect time,
    of a type no list of transient faults names."""


@pytest.mark.parametrize(
    "fault",
    [StaleSessionError("another writer landed first"), _TooManyConnectionsError("too many")],
    ids=["stale", "connect_time"],
)
@pytest.mark.parametrize("at", [None, Attempt], ids=["reading_the_submission", "on_an_attempt"])
async def test_a_transient_ledger_fault_leaves_the_run_unfinished_for_the_next_resume(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    at: type[RunChange] | None,
    fault: Exception,
) -> None:
    alive = InMemoryRunLedger()
    left = await _left_running(alive, approved_plan(), _surface(real_schemas, manifest()))
    resume = run_resumer(
        run_ledger=_Flaky(alive, fault, at=at),  # type: ignore[arg-type] -- the ledger's protocol, structurally
        publisher=_Publisher(FramePublished(replay_key=REPLAY_KEY)),
        executor=_describes,
        tool_surface=_surface(real_schemas, manifest()),
        max_concurrent_steps=4,
        fetch_series_stats=default_fetch_series_stats,
    )

    assert await resume(_Context()) is None  # type: ignore[arg-type] -- reads only `state`
    unfinished = await alive.get(left.run_id)
    assert unfinished is not None and unfinished.status is RunStatus.RUNNING

    handed_back = await resume(_Context())  # type: ignore[arg-type] -- reads only `state`

    assert handed_back is not None
    finished = await alive.get(left.run_id)
    assert finished is not None and finished.status is RunStatus.COMPLETED


@pytest.mark.parametrize(
    "fault",
    [StaleSessionError("another writer landed first"), SegmentLost("the segment was taken over")],
    ids=["stale", "segment_lost"],
)
@pytest.mark.parametrize(
    ("at", "left_at"),
    [
        (Attempt, RunStatus.RUNNING),
        (Seal, RunStatus.RUNNING),
        (Published, RunStatus.PUBLISHING),
        (Complete, RunStatus.PUBLISHING),
    ],
)
async def test_a_ledger_fault_mid_run_leaves_the_run_unfinished_for_the_next_resume(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    fault: Exception,
    at: type[RunChange],
    left_at: RunStatus,
) -> None:
    """A store fault on an attempt, the seal or the publication says nothing
    about the computation: the run is left unfinished, never failed, and the
    next resume completes it."""
    alive = InMemoryRunLedger()
    plan = approved_plan()
    submission = _submission(plan)

    with pytest.raises(LedgerUnavailableError) as raised:
        await execute_authorised(
            submission,
            authorisation=_approval(
                plan, submission.source_policy(), expires_at=datetime.now(UTC) + HOUR
            ),
            idempotency_key=idempotency_key(caller="u", conversation_id="c", task="t"),
            ledger=_Flaky(alive, fault, at=at),  # type: ignore[arg-type] -- the ledger's protocol, structurally
            surface=_surface(real_schemas, manifest()),
            publisher=_Publisher(FramePublished(replay_key=REPLAY_KEY)),
        )

    left = await _only_run(alive)
    assert raised.value.run_id == left.run_id
    assert left.status is left_at and left.failure is None
    resumed = await _resume(
        left,
        ledger=alive,
        surface=_surface(real_schemas, manifest()),
        publisher=_Publisher(FramePublished(replay_key=REPLAY_KEY)),
    )
    assert isinstance(resumed, Accepted), resumed
    finished = await alive.get(left.run_id)
    assert finished is not None and finished.status is RunStatus.COMPLETED


@pytest.mark.parametrize("route", ["planned", "authorised"])
async def test_a_store_fault_at_admission_is_a_ledger_fault(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    route: str,
) -> None:
    alive = InMemoryRunLedger()
    ledger = _Flaky(alive, StaleSessionError("another writer landed first"), on_admit=True)
    plan = approved_plan()
    surface = _surface(real_schemas, manifest())
    publisher = _Publisher(FramePublished(replay_key=REPLAY_KEY))

    with pytest.raises(LedgerUnavailableError) as raised:
        if route == "planned":
            await execute_approved_plan(
                plan,
                surface=surface,
                ledger=ledger,  # type: ignore[arg-type] -- the ledger's protocol, structurally
                publisher=publisher,
            )
        else:
            submission = _submission(plan)
            await execute_authorised(
                submission,
                authorisation=_approval(
                    plan, submission.source_policy(), expires_at=datetime.now(UTC) + HOUR
                ),
                idempotency_key=idempotency_key(caller="u", conversation_id="c", task="t"),
                ledger=ledger,  # type: ignore[arg-type] -- the ledger's protocol, structurally
                surface=surface,
                publisher=publisher,
            )

    assert raised.value.run_id is None
    assert unrecorded_hand_back(raised.value.run_id) == {
        "status": "not_started",
        "code": "run_not_started",
    }
    assert await alive.unfinished() == ()
    assert surface.calls == []


async def test_a_run_carried_on_after_its_seal_completes_on_what_its_source_holds_now(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    """The process died after the seal; when the run is carried on, its
    authorised source holds a newer version, which the run reads and seals."""
    alive = InMemoryRunLedger()
    dies = _Dies(alive, lambda change: isinstance(change, Attempt) and change.step_id == "totals")
    with pytest.raises(_ProcessDiedError):
        await execute_approved_plan(
            approved_plan(),
            surface=_surface(real_schemas, manifest()),
            ledger=dies,
            publisher=None,
        )
    left = await _only_run(alive)
    assert left.manifest is not None
    newer = "cv_" + "8" * 32

    outcome = await _resume(
        left,
        ledger=alive,
        surface=_surface(real_schemas, manifest(content_version=newer)),
        publisher=_Publisher(FramePublished(replay_key=REPLAY_KEY)),
    )

    assert isinstance(outcome, Accepted), outcome
    run = await alive.get(left.run_id)
    assert run is not None and run.status is RunStatus.COMPLETED
    assert run.manifest is not None
    assert [entry.content_version for entry in run.manifest.entries] == [newer]


async def test_resuming_from_a_snapshot_the_run_has_moved_past_drives_from_the_ledger(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    """The snapshot says `running`; another driver has since completed the
    run. Driving from the snapshot would record an attempt on a completed run."""
    alive = InMemoryRunLedger()
    stale = await _left_running(alive, approved_plan(), _surface(real_schemas, manifest()))
    kwargs: dict[str, Any] = {
        "ledger": alive,
        "publisher": _Publisher(FramePublished(replay_key=REPLAY_KEY)),
    }
    first = await _resume(stale, surface=_surface(real_schemas, manifest()), **kwargs)
    assert isinstance(first, Accepted), first
    surface = _surface(real_schemas, manifest())

    second = await _resume(stale, surface=surface, **kwargs)

    assert second == first
    assert _tools(surface) == []


async def test_two_concurrent_submissions_of_one_key_run_the_work_once(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    """As two identical tool calls of one model response run: the second
    admission finds the first's run and waits for its outcome."""
    plan = approved_plan()
    submission = _submission(plan)
    authorisation = _approval(plan, submission.source_policy(), expires_at=datetime.now(UTC) + HOUR)
    key = idempotency_key(caller="u_approver", conversation_id="c", task="t")
    ledger = InMemoryRunLedger()
    surface = _surface(real_schemas, manifest())
    publisher = _Publisher(FramePublished(replay_key=REPLAY_KEY))

    first, second = await asyncio.gather(
        *(
            execute_authorised(
                submission,
                authorisation=authorisation,
                idempotency_key=key,
                ledger=ledger,
                surface=surface,
                publisher=publisher,
            )
            for _ in range(2)
        )
    )

    assert isinstance(first, Accepted), first
    assert second == first
    assert _tools(surface) == ["get_data", "aggregate"]
    assert publisher.frames == [DERIVED]
    run = await ledger.get(first.run_id or "")
    assert run is not None and run.status is RunStatus.COMPLETED


async def test_an_authorisation_of_other_work_admits_no_run(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    plan = approved_plan()
    submission = _submission(plan)
    authorisation = _approval(
        plan, submission.source_policy(), expires_at=datetime.now(UTC) + HOUR
    ).model_copy(update={"approved_revision": "f" * 64})
    ledger = InMemoryRunLedger()
    surface = _surface(real_schemas, manifest())

    outcome = await execute_authorised(
        submission,
        authorisation=authorisation,
        idempotency_key=idempotency_key(caller="u", conversation_id="c", task="t"),
        ledger=ledger,
        surface=surface,
        publisher=None,
    )

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.AUTHORISATION_MISMATCH
    assert outcome.run_id is None
    assert await ledger.unfinished() == ()
    assert _tools(surface) == []


def _stored_request_authorisation(
    sources: SourcePolicy, *, at: datetime, digest: str
) -> RequestAuthorisation:
    """A request authorisation as the ledger stores one, from its JSON."""
    return RequestAuthorisation.model_validate(
        {
            "kind": "request",
            "source_policy": sources.model_dump(mode="json"),
            "expires_at": (at + HOUR).isoformat(),
            "task": "relative growth of life expectancy at birth in Kenya from 2015 to 2022",
            "recipe_id": "growth",
            "recipe_revision": "0123456789abcdef",
            "graph_digest": digest,
            "limits": {"steps": 2, "concurrent_steps": 4},
            "parameter_origins": {
                "spec_id": "recorded",
                "start_year": "recorded",
                "end_year": "recorded",
                "change": "model_stated",
            },
            "at": at.isoformat(),
        }
    )


@pytest.mark.parametrize(
    ("later", "status", "code"),
    [
        (timedelta(0), RunStatus.COMPLETED, None),
        (2 * HOUR, RunStatus.FAILED, "authorisation_expired"),
    ],
    ids=["in_force", "expired"],
)
async def test_the_next_turn_carries_on_a_stored_run_a_request_authorised(
    approved_plan: PlanFactory,
    manifest: ManifestFactory,
    real_schemas: list[ToolSchema],
    later: timedelta,
    status: RunStatus,
    code: str | None,
) -> None:
    """A run stored under a `RequestAuthorisation` and left unfinished is
    carried on by the next turn's resume like a plan's: run to completion
    while its authorisation is in force, recorded failed once it is not."""
    now = [datetime.now(UTC)]
    ledger = InMemoryRunLedger(clock=lambda: now[0])
    plan = approved_plan()
    assert plan.workflow is not None
    digest = "a" * 64
    submission = _submission(plan).model_copy(update={"revision": digest})
    authorisation = _stored_request_authorisation(
        submission.source_policy(), at=now[0], digest=digest
    )
    stored = RunRecord(
        run_id="run_" + "b" * 32,
        idempotency_key=idempotency_key(caller="u", conversation_id="c", task="t"),
        submission_digest=submission_digest(
            submission=submission, source_policy=authorisation.source_policy
        ),
        authorisation=authorisation,
        status=RunStatus.ADMITTED,
        admitted_at=now[0],
    )
    admitted = await ledger.admit(stored, submission=submission_content(submission))
    now[0] += later
    resume = run_resumer(
        run_ledger=ledger,
        publisher=_Publisher(FramePublished(replay_key=REPLAY_KEY)),
        executor=_describes,
        tool_surface=_surface(real_schemas, manifest()),
        max_concurrent_steps=4,
        fetch_series_stats=default_fetch_series_stats,
    )

    handed_back = await resume(_Context())  # type: ignore[arg-type] -- reads only `state`

    assert handed_back is not None
    run = await ledger.get(admitted.run.run_id)
    assert run is not None and run.status is status
    assert (run.failure.code if run.failure is not None else None) == code
    await handed_back.acknowledge()
    assert await ledger.unfinished() == ()


async def test_carrying_on_a_run_failed_before_its_outcome_answers_that_it_was_abandoned(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    alive = InMemoryRunLedger()
    left = await _left_running(alive, approved_plan(), _surface(real_schemas, manifest()))
    await alive.record(left.run_id, Fail("cancelled", "the turn running it was cancelled"))
    surface = _surface(real_schemas, manifest())

    outcome = await _resume(left, ledger=alive, surface=surface, publisher=None)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.RUN_ABANDONED
    assert outcome.run_id == left.run_id
    assert _tools(surface) == []


async def test_a_cancelled_driver_hands_its_waiter_the_abandoned_run_not_its_cancellation(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    plan = approved_plan()
    submission = _submission(plan)
    authorisation = _approval(plan, submission.source_policy(), expires_at=datetime.now(UTC) + HOUR)
    ledger = InMemoryRunLedger()
    surface = ScriptedSurface(
        schemas=real_schemas,
        responses={"get_data": [ok(manifest())], "aggregate": [ok(RESULT_FRAME)]},
        delays={"get_data": 3600},
    )

    def submit() -> asyncio.Task[Any]:
        return asyncio.create_task(
            execute_authorised(
                submission,
                authorisation=authorisation,
                idempotency_key=idempotency_key(caller="u", conversation_id="c", task="t"),
                ledger=ledger,
                surface=surface,
                publisher=None,
            )
        )

    driver = submit()
    while not surface.calls:
        await asyncio.sleep(0)
    waiter = submit()
    for _ in range(20):
        await asyncio.sleep(0)
    driver.cancel()
    with pytest.raises(asyncio.CancelledError):
        await driver

    outcome = await asyncio.wait_for(waiter, timeout=5)

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.RUN_ABANDONED
    run = await ledger.get(outcome.run_id or "")
    assert run is not None and run.status is RunStatus.FAILED
    assert run.failure is not None and run.failure.code == "cancelled"
    assert _tools(surface) == ["get_data"]


def _jsonb(value: Any) -> Any:
    """`value` with every object's keys in the order Postgres `jsonb` keeps
    them: shorter keys first, then by bytes."""
    if isinstance(value, dict):
        return {key: _jsonb(value[key]) for key in sorted(value, key=lambda k: (len(k), k))}
    if isinstance(value, list):
        return [_jsonb(item) for item in value]
    return value


class _JsonbLedger(InMemoryRunLedger):
    """Stores each recorded outcome as `jsonb` hands it back."""

    async def record(self, run_id: str, change: RunChange) -> RunRecord:
        if isinstance(change, Publishing):
            change = Publishing(change.results, outcome=_jsonb(change.outcome))
        return await super().record(run_id, change)


async def test_a_carried_on_outcome_keeps_the_workflows_terminal_order(
    approved_plan: PlanFactory, manifest: ManifestFactory, real_schemas: list[ToolSchema]
) -> None:
    plan = approved_plan()
    assert plan.workflow is not None
    terminal = ("totals", "debt")
    plan = plan.model_copy(
        update={"workflow": plan.workflow.model_copy(update={"terminal_steps": terminal})}
    )
    submission = _submission(plan)
    ledger = _JsonbLedger()

    async def dies(frame: FrameRef) -> FramePublished:
        raise _ProcessDiedError(frame.spec_id)

    with pytest.raises(_ProcessDiedError):
        await execute_authorised(
            submission,
            authorisation=_approval(
                plan, submission.source_policy(), expires_at=datetime.now(UTC) + HOUR
            ),
            idempotency_key=idempotency_key(caller="u", conversation_id="c", task="t"),
            ledger=ledger,
            surface=_surface(real_schemas, manifest()),
            publisher=dies,
        )
    left = await _only_run(ledger)
    assert list(left.outcome["results"]) == ["debt", "totals"]  # type: ignore[index] -- an Accepted's JSON

    outcome = await _resume(
        left,
        ledger=ledger,
        surface=_surface(real_schemas, manifest()),
        publisher=_Publisher(FramePublished(replay_key=REPLAY_KEY)),
    )

    assert isinstance(outcome, Accepted), outcome
    assert tuple(outcome.results) == terminal
