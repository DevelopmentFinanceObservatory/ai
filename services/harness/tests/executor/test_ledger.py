"""The run ledger's rules: admission by idempotency key and submission digest,
the status machine, the sealed manifest's check against the authorisation, and
the completion gate (a run completes only once each derived result is
published). Run against `InMemoryRunLedger`; the studio-backed ledger shares
`admit_against` and `advance` and is covered in `tests/test_run_ledger.py`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Annotated

import pytest
from common.hashed_fields import HASHED_WHEN_SET
from common.materialisation import FrameRef
from executor.classify import MAX_RETRY_AFTER_SECONDS
from executor.completion import PUBLISH_ATTEMPTS, PublishFault, publish_results, run_hand_back
from executor.ledger import (
    MAX_DELIVERY_ATTEMPTS,
    MAX_PENDING_AGE,
    MAX_PUBLICATION_RESUMES,
    Attempt,
    AuthorisationExpiredError,
    Complete,
    Delivered,
    Fail,
    Hold,
    IdempotencyConflictError,
    IllegalTransitionError,
    InMemoryRunLedger,
    ManifestEntry,
    ManifestOutsideAuthorisationError,
    ParameterOrigin,
    PlanApproval,
    Publication,
    Published,
    Publishing,
    RequestAuthorisation,
    ResultFrame,
    Resume,
    RunChange,
    RunNotFoundError,
    RunRecord,
    RunStatus,
    Seal,
    SourcePolicy,
    Start,
    Undelivered,
    idempotency_key,
    new_run,
    stored_run,
    submission_digest,
)
from planner.records import StrictModel

GDP = "spec_" + "a" * 32
ODA = "spec_" + "b" * 32
UNASKED = "spec_" + "c" * 32
NOW = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


class _Work(StrictModel):
    steps: tuple[str, ...]


class _WorkLater(_Work):
    """`_Work` as a later release reads it, with a field added at a default."""

    limit: Annotated[int | None, HASHED_WHEN_SET] = None


WORKFLOW = _Work(steps=("s1",))

ROOT = ResultFrame(spec_id=GDP, content_version="cv_" + "1" * 32)
DERIVED = ResultFrame(spec_id="ops_" + "d" * 32, content_version="ocv_" + "2" * 32)
INPUTS = (
    ManifestEntry(spec_id=ODA, content_version="cv_" + "3" * 32),
    ManifestEntry(spec_id=GDP, content_version="cv_" + "1" * 32),
)

# A retryable publication fault that names a wait.
TIMEOUT = Hold("timeout", "the publisher timed out", retry_after_s=5.0)

# What the admitted runs execute, stored beside each.
SUBMISSION = {"workflow": WORKFLOW}


def _policy(*sources: str) -> SourcePolicy:
    return SourcePolicy(sources=sources or (GDP, ODA))


def _approval(*, expires_at: datetime = NOW + timedelta(hours=1)) -> PlanApproval:
    return PlanApproval(
        source_policy=_policy(),
        expires_at=expires_at,
        plan_version=1,
        approved_revision="r" * 64,
        approver="u_0123456789abcdef",
        at=NOW,
    )


def _candidate(
    *, workflow: _Work = WORKFLOW, key: str | None = None, expires_at: datetime | None = None
) -> RunRecord:
    authorisation = _approval(expires_at=expires_at or NOW + timedelta(hours=1))
    return new_run(
        idempotency_key=key or idempotency_key(caller="u_1", conversation_id="c_1", task="plan:1"),
        submission_digest=submission_digest(
            submission=workflow, source_policy=authorisation.source_policy
        ),
        authorisation=authorisation,
        now=NOW,
    )


class _Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def ledger(clock: _Clock) -> InMemoryRunLedger:
    return InMemoryRunLedger(clock=clock)


async def _run_to(ledger: InMemoryRunLedger, status: RunStatus) -> RunRecord:
    """A run admitted and advanced to `status` along the ordinary path."""
    steps: dict[RunStatus, tuple[RunChange, ...]] = {
        RunStatus.ADMITTED: (),
        RunStatus.RUNNING: (Start(),),
        RunStatus.PUBLISHING: (Start(), Seal(INPUTS), Publishing((ROOT, DERIVED))),
        RunStatus.PENDING: (Start(), Seal(INPUTS), Publishing((ROOT, DERIVED)), TIMEOUT),
        RunStatus.COMPLETED: (
            Start(),
            Seal(INPUTS),
            Publishing((ROOT, DERIVED)),
            Published(DERIVED, "rpk_1"),
            Complete(),
        ),
        RunStatus.FAILED: (Fail("step_refused", "the tool refused"),),
    }
    run = (await ledger.admit(_candidate(), submission=SUBMISSION)).run
    for change in steps[status]:
        run = await ledger.record(run.run_id, change)
    assert run.status is status
    return run


# --- admission -----------------------------------------------------------------


async def test_the_same_key_and_digest_returns_the_recorded_run_as_it_stands(ledger) -> None:
    first = await ledger.admit(_candidate(), submission=SUBMISSION)
    await ledger.record(first.run.run_id, Start())

    again = await ledger.admit(_candidate(), submission=SUBMISSION)

    assert not first.resumed
    assert again.resumed
    assert again.run.run_id == first.run.run_id
    assert again.run.status is RunStatus.RUNNING


async def test_the_same_key_for_a_different_submission_is_a_conflict(ledger) -> None:
    first = await ledger.admit(_candidate(), submission=SUBMISSION)

    with pytest.raises(IdempotencyConflictError) as refused:
        await ledger.admit(_candidate(workflow=_Work(steps=())), submission=SUBMISSION)

    assert refused.value.run_id == first.run.run_id
    assert await ledger.unfinished() == (first.run,)


async def test_a_refresh_mints_a_new_key_and_so_a_new_run(ledger) -> None:
    first = await ledger.admit(_candidate(), submission=SUBMISSION)
    refreshed_key = idempotency_key(
        caller="u_1", conversation_id="c_1", task="plan:1", generation=1
    )

    refreshed = await ledger.admit(_candidate(key=refreshed_key), submission=SUBMISSION)

    assert not refreshed.resumed
    assert refreshed.run.run_id != first.run.run_id


async def test_a_failed_runs_key_admits_a_new_run_of_the_same_work(ledger) -> None:
    failed = await _run_to(ledger, RunStatus.FAILED)

    again = await ledger.admit(_candidate(), submission=SUBMISSION)

    assert not again.resumed
    assert again.run.run_id != failed.run_id
    assert again.run.status is RunStatus.ADMITTED
    assert (await ledger.admit(_candidate(), submission=SUBMISSION)).run == again.run


async def test_a_completed_runs_key_returns_the_completed_run(ledger) -> None:
    completed = await _run_to(ledger, RunStatus.COMPLETED)

    again = await ledger.admit(_candidate(), submission=SUBMISSION)

    assert again.resumed
    assert again.run == completed


async def test_each_scope_keeps_its_own_runs(clock) -> None:
    scope = ["conversation a"]
    ledger = InMemoryRunLedger(clock=clock, scope=lambda: scope[0])
    first = (await ledger.admit(_candidate(), submission=SUBMISSION)).run

    scope[0] = "conversation b"

    assert await ledger.unfinished() == ()
    assert await ledger.get(first.run_id) is None
    assert not (await ledger.admit(_candidate(), submission=SUBMISSION)).resumed


async def test_a_new_run_under_an_expired_authorisation_is_refused(ledger) -> None:
    with pytest.raises(AuthorisationExpiredError):
        await ledger.admit(_candidate(expires_at=NOW), submission=SUBMISSION)

    assert await ledger.unfinished() == ()


@pytest.mark.parametrize(
    "other",
    [
        {"caller": "u_2"},
        {"conversation_id": "c_2"},
        {"task": "plan:2"},
        {"generation": 1},
    ],
)
def test_the_key_is_scoped_to_caller_conversation_task_and_generation(other) -> None:
    scope = {"caller": "u_1", "conversation_id": "c_1", "task": "plan:1"}

    assert idempotency_key(**scope) == idempotency_key(**scope)
    assert idempotency_key(**scope) != idempotency_key(**(scope | other))


def test_the_submission_digest_covers_the_source_policy() -> None:
    assert submission_digest(submission=WORKFLOW, source_policy=_policy(GDP, ODA)) == (
        submission_digest(submission=WORKFLOW, source_policy=_policy(ODA, GDP, ODA))
    )
    assert submission_digest(submission=WORKFLOW, source_policy=_policy(GDP, ODA)) != (
        submission_digest(submission=WORKFLOW, source_policy=_policy(GDP))
    )


def test_a_field_a_later_release_adds_at_its_default_leaves_the_submission_digest() -> None:
    policy = _policy(GDP)
    digest = submission_digest(submission=WORKFLOW, source_policy=policy)

    assert submission_digest(submission=_WorkLater(steps=("s1",)), source_policy=policy) == digest
    assert (
        submission_digest(submission=_WorkLater(steps=("s1",), limit=3), source_policy=policy)
        != digest
    )


def test_a_plan_approval_round_trips_through_a_run_record() -> None:
    run = new_run(idempotency_key="rk_1", submission_digest="d", authorisation=_approval(), now=NOW)
    assert RunRecord.model_validate(run.model_dump(mode="json")) == run


# A run record as the ledger stores one admitted under a request authorisation.
_STORED_REQUEST_RUN = {
    "run_id": "run_15e10ab66c594efb9b4decc70c05b4ec",
    "idempotency_key": "rk_1",
    "submission_digest": "d" * 64,
    "authorisation": {
        "kind": "request",
        "source_policy": {"sources": [GDP], "version_policy": "latest"},
        "expires_at": "2026-10-12T13:00:00Z",
        "task": "relative growth of GDP in Kenya from 2015 to 2022",
        "recipe_id": "growth",
        "recipe_revision": "0123456789abcdef",
        "graph_digest": "g" * 64,
        "limits": {"steps": 2, "concurrent_steps": 4},
        "parameter_origins": {"spec_id": "recorded", "change": "model_stated"},
        "at": "2026-10-12T12:00:00Z",
    },
    "status": "completed",
    "admitted_at": "2026-10-12T12:00:00Z",
    "resumes": 0,
    "delivered": True,
}


def test_a_stored_run_admitted_under_a_request_authorisation_loads() -> None:
    run = stored_run(_STORED_REQUEST_RUN)

    assert isinstance(run.authorisation, RequestAuthorisation)
    assert run.authorisation.parameter_origins == {
        "spec_id": ParameterOrigin.RECORDED,
        "change": ParameterOrigin.MODEL_STATED,
    }
    assert run.model_dump(mode="json")["authorisation"] == _STORED_REQUEST_RUN["authorisation"]


# --- the status machine -----------------------------------------------------------


async def test_a_run_completes_once_its_derived_result_is_published(ledger) -> None:
    run = await _run_to(ledger, RunStatus.PENDING)
    assert run.pending is not None

    resumed = await ledger.record(run.run_id, Resume())
    assert (resumed.status, resumed.pending, resumed.resumes) == (RunStatus.PUBLISHING, None, 1)
    await ledger.record(run.run_id, Published(DERIVED, "rpk_1"))
    run = await ledger.record(run.run_id, Complete())

    assert run.status is RunStatus.COMPLETED
    assert run.pending is None
    assert [(r.frame, r.publication, r.replay_key) for r in run.results] == [
        (ROOT, Publication.NOT_REQUIRED, None),
        (DERIVED, Publication.PUBLISHED, "rpk_1"),
    ]


async def test_attempts_are_numbered_per_step_in_dispatch_order(ledger) -> None:
    run = await _run_to(ledger, RunStatus.RUNNING)
    for step in ("s1", "s2", "s1"):
        run = await ledger.record(run.run_id, Attempt(step_id=step, tool="get_data"))

    assert [(a.step_id, a.attempt) for a in run.attempts] == [("s1", 1), ("s2", 1), ("s1", 2)]


@pytest.mark.parametrize(
    ("status", "change"),
    [
        (RunStatus.ADMITTED, Attempt(step_id="s1", tool="get_data")),
        (RunStatus.ADMITTED, Seal(INPUTS)),
        (RunStatus.RUNNING, Start()),
        (RunStatus.RUNNING, Complete()),
        (RunStatus.RUNNING, Published(DERIVED, "rpk_1")),
        (RunStatus.PUBLISHING, Complete()),
        (RunStatus.PUBLISHING, Attempt(step_id="s1", tool="get_data")),
        (RunStatus.PENDING, Published(DERIVED, "rpk_1")),
        (RunStatus.PENDING, Complete()),
        (RunStatus.COMPLETED, Fail("late", "too late")),
        (RunStatus.FAILED, Start()),
        (RunStatus.COMPLETED, Resume()),
        (RunStatus.FAILED, Resume()),
        (RunStatus.PUBLISHING, Delivered()),
        (RunStatus.PENDING, Delivered()),
        (RunStatus.FAILED, Delivered()),
        (RunStatus.PENDING, Undelivered()),
        (RunStatus.FAILED, Undelivered()),
    ],
)
async def test_a_change_from_the_wrong_status_is_refused_and_writes_nothing(
    ledger, status, change
) -> None:
    run = await _run_to(ledger, status)

    with pytest.raises(IllegalTransitionError):
        await ledger.record(run.run_id, change)

    assert await ledger.get(run.run_id) == run


async def test_a_completed_run_is_carried_on_until_its_result_is_delivered(ledger) -> None:
    run = await _run_to(ledger, RunStatus.COMPLETED)
    assert await ledger.unfinished() == (run,)

    delivered = await ledger.record(run.run_id, Delivered())

    assert (delivered.status, delivered.delivered) == (RunStatus.COMPLETED, True)
    assert await ledger.record(run.run_id, Delivered()) == delivered
    assert await ledger.record(run.run_id, Undelivered()) == delivered
    assert await ledger.unfinished() == ()


async def test_a_completed_run_is_carried_on_no_more_once_delivery_failed_too_often(
    ledger,
) -> None:
    run = await _run_to(ledger, RunStatus.COMPLETED)
    for _ in range(MAX_DELIVERY_ATTEMPTS - 1):
        run = await ledger.record(run.run_id, Undelivered())
        assert not run.delivered
    assert await ledger.unfinished() == (run,)

    given_up = await ledger.record(run.run_id, Undelivered())

    assert (given_up.undelivered, given_up.delivered) == (MAX_DELIVERY_ATTEMPTS, True)
    assert await ledger.unfinished() == ()


async def test_a_completed_run_is_carried_on_no_more_once_too_old(ledger, clock) -> None:
    run = await _run_to(ledger, RunStatus.COMPLETED)
    clock.now = run.admitted_at + MAX_PENDING_AGE

    given_up = await ledger.record(run.run_id, Undelivered())

    assert (given_up.undelivered, given_up.delivered) == (1, True)
    assert await ledger.unfinished() == ()


async def test_results_cannot_be_reported_over_unsealed_inputs(ledger) -> None:
    run = await _run_to(ledger, RunStatus.RUNNING)

    with pytest.raises(IllegalTransitionError):
        await ledger.record(run.run_id, Publishing((ROOT,)))


async def test_a_run_of_root_results_alone_completes_without_publication(ledger) -> None:
    run = await _run_to(ledger, RunStatus.RUNNING)
    await ledger.record(run.run_id, Seal(INPUTS))
    await ledger.record(run.run_id, Publishing((ROOT,)))

    with pytest.raises(IllegalTransitionError):
        await ledger.record(run.run_id, TIMEOUT)
    assert (await ledger.record(run.run_id, Complete())).status is RunStatus.COMPLETED


async def test_a_publication_cannot_be_rerecorded_under_another_replay_key(ledger) -> None:
    run = await _run_to(ledger, RunStatus.PUBLISHING)
    await ledger.record(run.run_id, Published(DERIVED, "rpk_1"))

    assert await ledger.record(run.run_id, Published(DERIVED, "rpk_1")) == await ledger.get(
        run.run_id
    )
    with pytest.raises(IllegalTransitionError):
        await ledger.record(run.run_id, Published(DERIVED, "rpk_2"))


async def test_a_change_to_an_unknown_run_is_refused(ledger) -> None:
    with pytest.raises(RunNotFoundError):
        await ledger.record("run_missing", Start())


async def test_starting_after_the_authorisation_expires_is_refused(ledger, clock) -> None:
    run = (await ledger.admit(_candidate(), submission=SUBMISSION)).run
    clock.now = run.authorisation.expires_at

    with pytest.raises(AuthorisationExpiredError):
        await ledger.record(run.run_id, Start())


# --- the sealed manifest ------------------------------------------------------------


async def test_a_manifest_is_sealed_sorted_and_resealing_it_is_a_no_op(ledger) -> None:
    run = await _run_to(ledger, RunStatus.RUNNING)

    sealed = await ledger.record(run.run_id, Seal(INPUTS))

    assert sealed.manifest is not None
    assert [e.spec_id for e in sealed.manifest.entries] == [GDP, ODA]
    assert await ledger.record(run.run_id, Seal(tuple(reversed(INPUTS)))) == sealed


async def test_the_retrievals_are_stored_with_the_first_seal_only(ledger) -> None:
    run = await _run_to(ledger, RunStatus.RUNNING)
    assert await ledger.retrievals(run.run_id) is None

    await ledger.record(run.run_id, Seal(INPUTS, retrievals=({"step": "s1"},)))
    await ledger.record(run.run_id, Seal(INPUTS, retrievals=({"step": "again"},)))

    assert await ledger.retrievals(run.run_id) == [{"step": "s1"}]


@pytest.mark.parametrize("sealed_before", [False, True], ids=["first seal", "reseal"])
@pytest.mark.parametrize(
    "entries",
    [
        pytest.param(
            (*INPUTS, ManifestEntry(spec_id=UNASKED, content_version="cv_" + "4" * 32)),
            id="a source the authorisation does not name",
        ),
        pytest.param((), id="no inputs"),
    ],
)
async def test_a_manifest_outside_the_authorisation_is_refused(
    ledger, entries, sealed_before
) -> None:
    run = await _run_to(ledger, RunStatus.RUNNING)
    if sealed_before:
        run = await ledger.record(run.run_id, Seal(INPUTS))

    with pytest.raises(ManifestOutsideAuthorisationError):
        await ledger.record(run.run_id, Seal(entries))

    assert (await ledger.get(run.run_id)).manifest == run.manifest


async def test_one_source_may_be_sealed_at_several_versions(ledger) -> None:
    run = await _run_to(ledger, RunStatus.RUNNING)
    second = ManifestEntry(spec_id=GDP, content_version="cv_" + "5" * 32)

    sealed = await ledger.record(run.run_id, Seal((*INPUTS, second)))

    assert sealed.manifest is not None
    assert [(e.spec_id, e.content_version) for e in sealed.manifest.entries] == [
        (GDP, "cv_" + "1" * 32),
        (GDP, "cv_" + "5" * 32),
        (ODA, "cv_" + "3" * 32),
    ]


async def test_sealing_after_the_authorisation_expires_is_refused(ledger, clock) -> None:
    run = await _run_to(ledger, RunStatus.RUNNING)
    clock.now = run.authorisation.expires_at

    with pytest.raises(AuthorisationExpiredError):
        await ledger.record(run.run_id, Seal(INPUTS))


async def test_a_run_carried_on_after_its_seal_reseals_the_versions_its_sources_hold_now(
    ledger,
) -> None:
    """Under the `latest` policy a computation run again may read newer
    versions of the authorised sources; nothing was published under the
    earlier seal, so the new one replaces it, with its retrievals."""
    run = await _run_to(ledger, RunStatus.RUNNING)
    await ledger.record(run.run_id, Seal(INPUTS, retrievals=({"read": "first"},)))
    moved = (INPUTS[0], ManifestEntry(spec_id=GDP, content_version="cv_" + "6" * 32))

    resealed = await ledger.record(run.run_id, Seal(moved, retrievals=({"read": "again"},)))

    assert resealed.manifest is not None
    assert [(e.spec_id, e.content_version) for e in resealed.manifest.entries] == [
        (GDP, "cv_" + "6" * 32),
        (ODA, "cv_" + "3" * 32),
    ]
    assert await ledger.retrievals(run.run_id) == [{"read": "again"}]


# --- resuming -------------------------------------------------------------------

UNFINISHED = [
    (RunStatus.ADMITTED, "computation_abandoned"),
    (RunStatus.RUNNING, "computation_abandoned"),
    (RunStatus.PUBLISHING, "publication_abandoned"),
    (RunStatus.PENDING, "publication_abandoned"),
]


@pytest.mark.parametrize(("status", "code"), UNFINISHED)
async def test_an_unfinished_run_is_abandoned_once_resumed_too_often(ledger, status, code) -> None:
    run = await _run_to(ledger, status)
    for _ in range(MAX_PUBLICATION_RESUMES):
        run = await ledger.record(run.run_id, Resume())
        assert run.status is (RunStatus.PUBLISHING if status is RunStatus.PENDING else status)
        if status is RunStatus.PENDING:
            run = await ledger.record(run.run_id, TIMEOUT)

    abandoned = await ledger.record(run.run_id, Resume())

    assert abandoned.status is RunStatus.FAILED
    assert run_hand_back(abandoned) == {"run_id": run.run_id, "status": "failed", "code": code}
    assert await ledger.unfinished() == ()


@pytest.mark.parametrize(("status", "code"), UNFINISHED)
async def test_an_unfinished_run_is_abandoned_once_too_old(ledger, clock, status, code) -> None:
    run = await _run_to(ledger, status)
    clock.now = run.admitted_at + MAX_PENDING_AGE

    abandoned = await ledger.record(run.run_id, Resume())

    assert abandoned.status is RunStatus.FAILED
    assert run_hand_back(abandoned) == {"run_id": run.run_id, "status": "failed", "code": code}


async def test_a_pending_run_failed_hands_back_no_hold(ledger) -> None:
    run = await _run_to(ledger, RunStatus.PENDING)
    assert run_hand_back(run)["retry_after_s"] == TIMEOUT.retry_after_s

    failed = await ledger.record(run.run_id, Fail("publication_error", "a defect"))

    assert failed.pending is None
    assert run_hand_back(failed) == {
        "run_id": run.run_id,
        "status": "failed",
        "code": "publication_error",
    }


def test_a_stored_record_from_another_release_loads_with_its_unknown_keys_ignored() -> None:
    run = _candidate()
    raw = run.model_dump(mode="json")
    raw["added_later"] = 1
    raw["authorisation"]["added_later"] = 2
    raw["authorisation"]["source_policy"]["added_later"] = 3

    assert stored_run(raw) == run


# --- the completion gate -------------------------------------------------------------


@pytest.mark.parametrize(
    ("retry_after_s", "held_for"),
    [
        (4.0, 4.0),
        (MAX_RETRY_AFTER_SECONDS * 10, MAX_RETRY_AFTER_SECONDS),
        (None, None),
        (-3.0, None),
        (float("inf"), None),
    ],
)
async def test_a_publish_is_retried_after_a_jittered_wait_and_held_with_a_usable_one(
    ledger, retry_after_s: float | None, held_for: float | None
) -> None:
    run = await _run_to(ledger, RunStatus.PUBLISHING)
    busy = PublishFault(
        code="service_unavailable", reason="busy", retryable=True, retry_after_s=retry_after_s
    )
    tried: list[FrameRef] = []
    slept: list[float] = []
    spreads: list[tuple[float, float]] = []

    async def publisher(frame: FrameRef) -> PublishFault:
        tried.append(frame)
        return busy

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    def jitter(low: float, high: float) -> float:
        spreads.append((low, high))
        return high

    held = await publish_results(
        run, ledger=ledger, publisher=publisher, sleep=sleep, jitter=jitter
    )

    assert len(tried) == PUBLISH_ATTEMPTS
    assert slept == [high for _, high in spreads]
    assert len(slept) == PUBLISH_ATTEMPTS - 1
    assert all(0 < low <= high <= MAX_RETRY_AFTER_SECONDS for low, high in spreads)
    assert held.status is RunStatus.PENDING
    assert held.pending is not None and held.pending.retry_after_s == held_for
    assert run_hand_back(held) == {
        "run_id": run.run_id,
        "status": "pending",
        "code": "service_unavailable",
        **({} if held_for is None else {"retry_after_s": held_for}),
    }


@pytest.mark.parametrize("retryable", [False, True])
async def test_a_runs_hand_back_carries_its_code_and_the_ledger_its_reason(
    ledger, retryable: bool
) -> None:
    """The reason names the frame and quotes the platform, so it stays in the
    ledger and the log."""
    run = await _run_to(ledger, RunStatus.PUBLISHING)
    reason = f"frame {DERIVED.spec_id}/{DERIVED.content_version} cannot be replayed"
    fault = PublishFault(code="replay_unavailable", reason=reason, retryable=retryable)

    async def publisher(frame: FrameRef) -> PublishFault:
        return fault

    async def sleep(seconds: float) -> None:
        return None

    stood = await publish_results(run, ledger=ledger, publisher=publisher, sleep=sleep)

    hand_back = run_hand_back(stood)
    assert hand_back == {
        "run_id": run.run_id,
        "status": stood.status.value,
        "code": "replay_unavailable",
    }
    assert DERIVED.spec_id not in json.dumps(hand_back)
    if retryable:
        assert stood.pending is not None and stood.pending.reason == reason
    else:
        assert stood.failure is not None and stood.failure.reason == reason
