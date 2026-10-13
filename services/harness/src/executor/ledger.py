"""The run ledger: one durable record per executed calculation.

A run is admitted before its first dispatch with an idempotency key, the
digest of the submission it executes and the typed authorisation that lets it
run. Every later fact about it (a tool attempt, the sealed input manifest, its
terminal results and their publication, a pending hold, a failure) is a
`RunChange` that `advance` applies to the stored `RunRecord`. `advance` is pure
and owns every rule: which status a change is legal from, what the manifest
must satisfy, and that a run completes only once each derived terminal result
is published.

The key names the work, the digest names its content. `idempotency_key` scopes
a key to the caller, the conversation and the task, and `admit_against` binds
it to the digest of the run it names: the same key with the same digest
returns that run, whether it is still unfinished or completed, and with a
different digest it is an `IdempotencyConflictError`. A failed run binds
nothing: its key admits a new run of the work. An intentional refresh passes
the next `generation`, which mints a new key and so a new run.

Admission also stores the canonical submission the run executes
(`submission_content`, the same content its digest covers), once and apart
from the record, so a run left unfinished by a restart can be carried
on from it. The authorisation must be in force to admit a new run and for
every dispatch and the seal; a resumed run may still finish publication once
it has expired. Each carry-on of an unfinished run records `Resume`, whatever
its status: a run is resumed at most `MAX_PUBLICATION_RESUMES` times and
within `MAX_PENDING_AGE` of its admission; past either it fails,
`publication_abandoned` once its computation came to an outcome and
`computation_abandoned` before.

A run carried on after its seal computes again and seals what its sources
hold then, which under the `latest` version policy may be newer versions.
That seal replaces the earlier one: a run records its results only after its
seal, so nothing was published under the earlier one. Every seal is checked
against the authorisation's sources.

A stored record is read with `stored_run`, which ignores keys this release
does not know, so a record another release wrote loads as long as every
field this release adds has a default.

A completed run's result is delivered once the conversation has recorded the
hand-back listing it, and `Delivered` records that. Until then the run stays
among those a later turn carries on (`RunLedger.unfinished`), so a hand-back
lost before it was recorded is made again. Each turn that carries it on
without delivering its result records `Undelivered`; once that has happened
`MAX_DELIVERY_ATTEMPTS` times, or `MAX_PENDING_AGE` after its admission, the
run is carried on no more.

`RunLedger` is the storage seam. `InMemoryRunLedger` serves tests and the
standalone harness, keeping each conversation's runs apart by the scope its
caller names; `harness.run_ledger.StudioRunLedger` keeps runs in the
conversation's ledger session.
"""

from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated, Any, Literal, Protocol

import rfc8785
from common.hashed_fields import revision_content
from planner.records import SpecId, StrictModel
from pydantic import AfterValidator, BaseModel, Field, JsonValue


class RunStatus(StrEnum):
    ADMITTED = "admitted"
    RUNNING = "running"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    PENDING = "pending"
    FAILED = "failed"


TERMINAL_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.FAILED})

# How many times an unfinished run is resumed, and how long after its
# admission, before the run fails (`Resume`).
MAX_PUBLICATION_RESUMES = 5
MAX_PENDING_AGE = timedelta(hours=24)

# How many turns carry a completed run on without delivering its result
# before no later turn carries it on. `MAX_PENDING_AGE` bounds it too.
MAX_DELIVERY_ATTEMPTS = MAX_PUBLICATION_RESUMES


def _sorted_unique(sources: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(sources)))


class SourcePolicy(StrictModel):
    """Which source definitions a run may read, and which of their versions.

    `sources` are root `spec_` ids, held sorted and unique so two policies
    naming the same sources digest the same. `latest` lets retrieval resolve
    whatever version each source holds when it runs; the sealed manifest
    records the versions it found."""

    sources: Annotated[tuple[SpecId, ...], AfterValidator(_sorted_unique), Field(min_length=1)]
    version_policy: Literal["latest"] = "latest"


class PlanApproval(StrictModel):
    """A user's approval of one Plan revision, the planned route's authorisation."""

    kind: Literal["plan_approval"] = "plan_approval"
    source_policy: SourcePolicy
    expires_at: datetime
    plan_version: int
    approved_revision: str
    approver: str
    at: datetime


class ParameterOrigin(StrEnum):
    """Where a direct request's parameter came from: `recorded` when the
    harness checked it against what the conversation recorded (a source a
    search settled on, years inside its reported coverage), `model_stated`
    when it is the model's reading of the user's words, which nothing
    checks."""

    RECORDED = "recorded"
    MODEL_STATED = "model_stated"


class RequestAuthorisation(StrictModel):
    """The harness's authorisation of a direct request to run one reviewed
    recipe. `parameter_origins` names where each recipe parameter came from."""

    kind: Literal["request"] = "request"
    source_policy: SourcePolicy
    expires_at: datetime
    task: str
    recipe_id: str
    recipe_revision: str
    graph_digest: str
    limits: dict[str, int]
    parameter_origins: dict[str, ParameterOrigin]
    at: datetime


Authorisation = Annotated[PlanApproval | RequestAuthorisation, Field(discriminator="kind")]


class StepAttempt(StrictModel):
    """One dispatch of one step's tool, recorded before the call goes out.
    `attempt` counts from 1 per step."""

    step_id: str
    tool: str
    attempt: int
    at: datetime


class ManifestEntry(StrictModel):
    """One retrieved root input at the exact version the run read. One source
    may be read at several versions, by retrievals of different arguments."""

    spec_id: SpecId
    content_version: str = Field(pattern=r"^cv_[0-9a-f]{32}$")


class InputManifest(StrictModel):
    """The run's retrieved inputs, sealed before any derived step runs."""

    entries: tuple[ManifestEntry, ...]
    sealed_at: datetime


class ResultFrame(StrictModel):
    """One terminal frame a run produced: a root `spec_` frame or a derived `ops_` one."""

    spec_id: str = Field(pattern=r"^(?:spec|ops)_[0-9a-f]{32}$")
    content_version: str = Field(pattern=r"^(?:cv|ocv)_[0-9a-f]{32}$")


class Publication(StrEnum):
    """Where a terminal result stands on durable publication. A root `spec_`
    frame needs none; a derived `ops_` frame is `required` until published with
    its replay recipe."""

    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    PUBLISHED = "published"


class TerminalResult(StrictModel):
    frame: ResultFrame
    publication: Publication
    replay_key: str | None = None


class PendingHold(StrictModel):
    """Why a run's publication is unfinished, and when a retry may succeed.
    `code` is the fault's code; `reason` its text, which stays in the ledger."""

    code: str = "publication_unfinished"
    reason: str
    retry_after_s: float | None = None


class RunFailure(StrictModel):
    code: str
    reason: str


class RunRecord(StrictModel):
    """Everything durable about one run. `submission_digest` is
    `submission_digest()` of the submission and source policy the run
    executes. `outcome` is what its computation came to, as the executor
    states it, recorded once with `Publishing` or `Fail`. `resumes` counts
    how often the unfinished run was carried on (`Resume`). `delivered` says
    no later turn carries a completed run on: the conversation recorded the
    hand-back of its result, or `undelivered` reached the delivery bound
    (`Undelivered`). `undelivered` counts the turns that carried a completed
    run on without delivering its result."""

    run_id: str
    idempotency_key: str
    submission_digest: str
    authorisation: Authorisation
    status: RunStatus
    admitted_at: datetime
    attempts: tuple[StepAttempt, ...] = ()
    manifest: InputManifest | None = None
    results: tuple[TerminalResult, ...] = ()
    pending: PendingHold | None = None
    failure: RunFailure | None = None
    outcome: JsonValue = None
    resumes: int = 0
    delivered: bool = False
    undelivered: int = 0


def stored_run(raw: JsonValue) -> RunRecord:
    """The run record a store holds as `raw`, keys this release does not
    know ignored, at every level. Raises `pydantic.ValidationError` when it
    is still not a record this release reads."""
    return RunRecord.model_validate(raw, extra="ignore")


# --- changes -----------------------------------------------------------------


@dataclass(frozen=True)
class Start:
    """`admitted` to `running`, while the authorisation is still in force."""


@dataclass(frozen=True)
class Attempt:
    """Record one dispatch of `step_id`'s tool, before it goes out, while the
    authorisation is still in force."""

    step_id: str
    tool: str


@dataclass(frozen=True)
class Seal:
    """Seal the retrieved inputs. Each must be a source the authorisation
    names; resealing the same inputs is a no-op, and sealing other inputs, as
    a run carried on after its seal does, replaces them. `retrievals` are the
    `get_data` results the inputs were read from, which the ledger stores
    with each seal that changes the inputs, apart from the record
    (`RunLedger.retrievals`)."""

    entries: tuple[ManifestEntry, ...]
    retrievals: tuple[JsonValue, ...] = ()


@dataclass(frozen=True)
class Publishing:
    """`running` to `publishing` with the run's terminal frames and the
    outcome that names them, once its inputs are sealed."""

    results: tuple[ResultFrame, ...]
    outcome: JsonValue = None


@dataclass(frozen=True)
class Published:
    """One derived terminal frame is published with its replay recipe."""

    frame: ResultFrame
    replay_key: str


@dataclass(frozen=True)
class Hold:
    """`publishing` to `pending`: a recoverable fault, `code`, left publication
    unfinished."""

    code: str
    reason: str
    retry_after_s: float | None = None


@dataclass(frozen=True)
class Resume:
    """A later turn carries an unfinished run on: `pending` back to
    `publishing`, any other unfinished status kept, counted in `resumes`. A
    run already resumed `MAX_PUBLICATION_RESUMES` times, or admitted
    `MAX_PENDING_AGE` ago, fails instead: `publication_abandoned` once its
    computation came to an outcome, `computation_abandoned` before."""


@dataclass(frozen=True)
class Complete:
    """`publishing` to `completed`, once no terminal result awaits publication."""


@dataclass(frozen=True)
class Fail:
    """Any unfinished run to `failed`, with the reason it cannot finish, and
    the outcome of a computation that stopped. A run that already holds an
    outcome keeps it."""

    code: str
    reason: str
    outcome: JsonValue = None


@dataclass(frozen=True)
class Delivered:
    """A completed run's result is handed back and the conversation recorded
    it. Recording it again is a no-op."""


@dataclass(frozen=True)
class Undelivered:
    """A turn carried a completed run on and could not deliver its result.
    The run is recorded `delivered` once this has happened
    `MAX_DELIVERY_ATTEMPTS` times or the run was admitted `MAX_PENDING_AGE`
    ago, and no later turn carries it on. A no-op on a delivered run."""


type RunChange = (
    Start
    | Attempt
    | Seal
    | Publishing
    | Published
    | Hold
    | Resume
    | Complete
    | Fail
    | Delivered
    | Undelivered
)


# --- errors ------------------------------------------------------------------


class RunLedgerError(Exception):
    """Base of every refusal the ledger raises. Nothing is written when one is raised."""


class IllegalTransitionError(RunLedgerError):
    def __init__(self, run: RunRecord, change: RunChange, why: str) -> None:
        super().__init__(
            f"run {run.run_id} is {run.status.value}; {type(change).__name__} refused: {why}"
        )
        self.run_id = run.run_id
        self.status = run.status


class IdempotencyConflictError(RunLedgerError):
    """The key already names a run of different work."""

    def __init__(self, existing: RunRecord, submitted_digest: str) -> None:
        super().__init__(
            f"idempotency key {existing.idempotency_key} already names run {existing.run_id} "
            f"of submission {existing.submission_digest}, not {submitted_digest}"
        )
        self.run_id = existing.run_id
        self.recorded_digest = existing.submission_digest
        self.submitted_digest = submitted_digest


class AuthorisationExpiredError(RunLedgerError):
    def __init__(self, expires_at: datetime) -> None:
        super().__init__(f"the authorisation expired at {expires_at.isoformat()}")
        self.expires_at = expires_at


class ManifestOutsideAuthorisationError(RunLedgerError):
    """The run read inputs its authorisation does not cover."""

    def __init__(self, run_id: str, reason: str) -> None:
        super().__init__(f"run {run_id}: {reason}")
        self.run_id = run_id


class RunNotFoundError(RunLedgerError):
    def __init__(self, run_id: str) -> None:
        super().__init__(f"no run {run_id} is recorded")
        self.run_id = run_id


class LedgerUnavailableError(Exception):
    """The ledger's store could not take a change to run `run_id`: it was
    unreachable, or another writer held the conversation. The run stands as
    the ledger last recorded it, unfinished, for a later carry-on to take on
    from there. `run_id` is `None` when the fault came at admission, so no run
    was stored and nothing was run. Not a `RunLedgerError`: the ledger refused
    nothing."""

    def __init__(self, run_id: str | None) -> None:
        if run_id is None:
            message = "the run could not be admitted; nothing was run"
        else:
            message = f"run {run_id}'s change could not be recorded; the run is unfinished"
        super().__init__(message)
        self.run_id = run_id


@asynccontextmanager
async def store_faults(run_id: str | None) -> AsyncIterator[None]:
    """Classify what a read or write of run `run_id`, `None` for an
    admission, raises: a
    `RunLedgerError` is the ledger refusing a change and propagates as it
    is; any other exception is the ledger's store failing, raised as
    `LedgerUnavailableError`, since it says nothing about the run."""
    try:
        yield
    except RunLedgerError:
        raise
    except Exception as exc:
        raise LedgerUnavailableError(run_id) from exc


async def record_change(ledger: RunLedger, run_id: str, change: RunChange) -> RunRecord:
    """Record `change` to run `run_id` of `ledger`, a store fault raised as
    `LedgerUnavailableError` (`store_faults`)."""
    async with store_faults(run_id):
        return await ledger.record(run_id, change)


# --- identity and admission --------------------------------------------------


def _digest(content: JsonValue) -> str:
    return hashlib.sha256(rfc8785.dumps(content)).hexdigest()


def idempotency_key(*, caller: str, conversation_id: str, task: str, generation: int = 0) -> str:
    """The key naming one piece of work for one caller in one conversation.
    `task` names the work (a plan version, a direct request); an intentional
    refresh passes the next `generation`."""
    content = {
        "caller": caller,
        "conversation_id": conversation_id,
        "task": task,
        "generation": generation,
    }
    return f"rk_{_digest(content)[:32]}"


def submission_content(submission: BaseModel) -> dict[str, Any]:
    """The canonical content of a run's submission: what admission stores with
    the run and what `submission_digest` covers.

    It is `common.hashed_fields.revision_content`, as a plan's revision is, so
    a release that adds a `HASHED_WHEN_SET` field neither writes the field
    while it holds its default, which keeps the stored form one an earlier
    release reads, nor moves the digest, so a key bound to it keeps naming the
    same run."""
    return revision_content(submission)


def submission_digest(*, submission: BaseModel, source_policy: SourcePolicy) -> str:
    """The digest of what a run executes: its submission's canonical content
    (`submission_content`) and the source policy it reads under."""
    content = {
        "submission": submission_content(submission),
        "source_policy": source_policy.model_dump(mode="json"),
    }
    return _digest(content)


def new_run(
    *, idempotency_key: str, submission_digest: str, authorisation: Authorisation, now: datetime
) -> RunRecord:
    """A freshly minted `admitted` run, the candidate `RunLedger.admit` stores
    unless its key already names a run."""
    return RunRecord(
        run_id=f"run_{uuid.uuid4().hex}",
        idempotency_key=idempotency_key,
        submission_digest=submission_digest,
        authorisation=authorisation,
        status=RunStatus.ADMITTED,
        admitted_at=now,
    )


@dataclass(frozen=True)
class Admission:
    """The run a submission executes under. `resumed` is true when its key
    already named this run, whatever its status."""

    run: RunRecord
    resumed: bool


def admit_against(existing: RunRecord | None, candidate: RunRecord, *, now: datetime) -> Admission:
    """Admit `candidate` where `existing` is the run its key already names, if any.

    A failed `existing` binds nothing, so `candidate` is admitted as a new run.

    Raises `IdempotencyConflictError` when an unfailed `existing` ran a different
    submission and `AuthorisationExpiredError` when a new run's authorisation is no
    longer in force.
    """
    if existing is not None and existing.status is not RunStatus.FAILED:
        if existing.submission_digest != candidate.submission_digest:
            raise IdempotencyConflictError(existing, candidate.submission_digest)
        return Admission(run=existing, resumed=True)
    _require_in_force(candidate.authorisation, now)
    return Admission(run=candidate, resumed=False)


def _require_in_force(authorisation: Authorisation, now: datetime) -> None:
    if now >= authorisation.expires_at:
        raise AuthorisationExpiredError(authorisation.expires_at)


# --- transitions -------------------------------------------------------------


def _require_status(run: RunRecord, change: RunChange, *allowed: RunStatus) -> None:
    if run.status not in allowed:
        wanted = " or ".join(status.value for status in allowed)
        raise IllegalTransitionError(run, change, f"it needs a run that is {wanted}")


def _sealed(run: RunRecord, change: Seal, now: datetime) -> RunRecord:
    # The manifest is the set of (source, version) pairs read.
    entries = tuple(sorted(set(change.entries), key=lambda e: (e.spec_id, e.content_version)))
    if run.manifest is not None and run.manifest.entries == entries:
        return run
    _require_in_force(run.authorisation, now)
    if not entries:
        raise ManifestOutsideAuthorisationError(run.run_id, "a manifest names at least one input")
    authorised = set(run.authorisation.source_policy.sources)
    outside = sorted({entry.spec_id for entry in entries} - authorised)
    if outside:
        raise ManifestOutsideAuthorisationError(
            run.run_id, f"inputs the authorisation does not name: {', '.join(outside)}"
        )
    return run.model_copy(update={"manifest": InputManifest(entries=entries, sealed_at=now)})


def _with_results(run: RunRecord, change: Publishing) -> RunRecord:
    if run.manifest is None:
        raise IllegalTransitionError(run, change, "its inputs are not sealed")
    frames = list(dict.fromkeys(change.results))
    if not frames:
        raise IllegalTransitionError(run, change, "a run publishes at least one terminal result")
    results = tuple(
        TerminalResult(
            frame=frame,
            publication=(
                Publication.REQUIRED
                if frame.spec_id.startswith("ops_")
                else Publication.NOT_REQUIRED
            ),
        )
        for frame in frames
    )
    return run.model_copy(
        update={"status": RunStatus.PUBLISHING, "results": results, "outcome": change.outcome}
    )


def _published(run: RunRecord, change: Published) -> RunRecord:
    for index, result in enumerate(run.results):
        if result.frame != change.frame:
            continue
        if result.publication is Publication.NOT_REQUIRED:
            raise IllegalTransitionError(run, change, "a root frame needs no publication")
        if result.publication is Publication.PUBLISHED:
            if result.replay_key != change.replay_key:
                raise IllegalTransitionError(
                    run, change, "it is already published under another key"
                )
            return run
        published = result.model_copy(
            update={"publication": Publication.PUBLISHED, "replay_key": change.replay_key}
        )
        results = (*run.results[:index], published, *run.results[index + 1 :])
        return run.model_copy(update={"results": results})
    raise IllegalTransitionError(run, change, f"{change.frame.spec_id} is not one of its results")


def _awaiting_publication(run: RunRecord) -> int:
    return sum(result.publication is Publication.REQUIRED for result in run.results)


_UNFINISHED_PUBLICATION = frozenset({RunStatus.PUBLISHING, RunStatus.PENDING})


def _resumed(run: RunRecord, now: datetime) -> RunRecord:
    """Unfinished `run` carried on once more (`Resume`), or failed once it
    has been resumed too often or is too old."""
    if run.resumes >= MAX_PUBLICATION_RESUMES:
        why = f"after {run.resumes} resumptions"
    elif now - run.admitted_at >= MAX_PENDING_AGE:
        why = f"{MAX_PENDING_AGE} after it was admitted"
    else:
        status = RunStatus.PUBLISHING if run.status is RunStatus.PENDING else run.status
        return run.model_copy(
            update={"status": status, "pending": None, "resumes": run.resumes + 1}
        )
    stage = "publication" if run.status in _UNFINISHED_PUBLICATION else "computation"
    failure = RunFailure(code=f"{stage}_abandoned", reason=f"the run was still unfinished {why}")
    return run.model_copy(update={"status": RunStatus.FAILED, "failure": failure, "pending": None})


def _undelivered(run: RunRecord, now: datetime) -> RunRecord:
    """Completed `run` with one more turn that could not deliver its result,
    recorded `delivered` once the delivery bound is reached."""
    if run.delivered:
        return run
    undelivered = run.undelivered + 1
    given_up = undelivered >= MAX_DELIVERY_ATTEMPTS or now - run.admitted_at >= MAX_PENDING_AGE
    return run.model_copy(update={"undelivered": undelivered, "delivered": given_up})


def advance(run: RunRecord, change: RunChange, *, now: datetime) -> RunRecord:  # noqa: PLR0911 -- one return per change kind
    """`run` with `change` applied, or `run` itself when the change is already
    recorded (a repeated seal, publication or delivery). Raises a
    `RunLedgerError` when the change is not legal for the run as it stands."""
    if run.status in TERMINAL_STATUSES and not isinstance(change, Delivered | Undelivered):
        raise IllegalTransitionError(run, change, "the run has finished")
    match change:
        case Start():
            _require_status(run, change, RunStatus.ADMITTED)
            _require_in_force(run.authorisation, now)
            return run.model_copy(update={"status": RunStatus.RUNNING})
        case Attempt(step_id=step_id, tool=tool):
            _require_status(run, change, RunStatus.RUNNING)
            _require_in_force(run.authorisation, now)
            number = 1 + sum(attempt.step_id == step_id for attempt in run.attempts)
            attempt = StepAttempt(step_id=step_id, tool=tool, attempt=number, at=now)
            return run.model_copy(update={"attempts": (*run.attempts, attempt)})
        case Seal():
            _require_status(run, change, RunStatus.RUNNING)
            return _sealed(run, change, now)
        case Publishing():
            _require_status(run, change, RunStatus.RUNNING)
            return _with_results(run, change)
        case Published():
            _require_status(run, change, RunStatus.PUBLISHING)
            return _published(run, change)
        case Hold(code=code, reason=reason, retry_after_s=retry_after_s):
            _require_status(run, change, RunStatus.PUBLISHING)
            if not _awaiting_publication(run):
                raise IllegalTransitionError(run, change, "every result is already published")
            hold = PendingHold(code=code, reason=reason, retry_after_s=retry_after_s)
            return run.model_copy(update={"status": RunStatus.PENDING, "pending": hold})
        case Resume():
            return _resumed(run, now)
        case Complete():
            _require_status(run, change, RunStatus.PUBLISHING)
            if waiting := _awaiting_publication(run):
                raise IllegalTransitionError(run, change, f"{waiting} result(s) await publication")
            return run.model_copy(update={"status": RunStatus.COMPLETED})
        case Fail(code=code, reason=reason, outcome=outcome):
            failure = RunFailure(code=code, reason=reason)
            kept = run.outcome if run.outcome is not None else outcome
            return run.model_copy(
                update={
                    "status": RunStatus.FAILED,
                    "failure": failure,
                    "outcome": kept,
                    "pending": None,
                }
            )
        case Delivered():
            _require_status(run, change, RunStatus.COMPLETED)
            return run if run.delivered else run.model_copy(update={"delivered": True})
        case Undelivered():
            _require_status(run, change, RunStatus.COMPLETED)
            return _undelivered(run, now)


# --- storage -----------------------------------------------------------------


class RunLedger(Protocol):
    """Where runs are kept. Every method that writes raises before writing
    anything when its change is refused."""

    async def admit(self, candidate: RunRecord, *, submission: JsonValue) -> Admission:
        """Store `candidate`, and `submission` beside it, unless its key
        already names a run (`admit_against`). `submission` is the
        `submission_content` `candidate.submission_digest` digests; it is written once,
        with the run, and never again."""
        ...

    async def get(self, run_id: str) -> RunRecord | None: ...

    async def submission(self, run_id: str) -> JsonValue:
        """The canonical submission stored with run `run_id`, or `None`."""
        ...

    async def retrievals(self, run_id: str) -> JsonValue:
        """The `get_data` results run `run_id` sealed its inputs from
        (`Seal.retrievals`), or `None` when its inputs are not sealed."""
        ...

    async def store_plan(self, revision: str, plan: JsonValue) -> None:
        """Store `plan`, the plan that compiled the work `revision` names
        (`Submission.revision`), unless one is stored for it already. The
        caller writes it before the work's run is admitted, outside the
        submission and its digest."""
        ...

    async def plan(self, revision: str) -> JsonValue:
        """The plan stored for `revision` (`store_plan`), or `None`."""
        ...

    async def unfinished(self) -> tuple[RunRecord, ...]:
        """Every run in this ledger a later turn carries on (`carried_on`). A
        stored run this release cannot read is recorded failed
        (`run_record_unreadable`) and left out."""
        ...

    async def record(self, run_id: str, change: RunChange) -> RunRecord:
        """Apply `change` to the stored run (`advance`) and return the result,
        storing `Seal.retrievals` with a seal that changes the inputs. Raises
        `RunNotFoundError` when no run `run_id` is recorded."""
        ...


def carried_on(run: RunRecord) -> bool:
    """Whether a later turn carries `run` on: it is neither completed nor
    failed, or it completed and its result is not yet delivered."""
    if run.status is RunStatus.COMPLETED:
        return not run.delivered
    return run.status is not RunStatus.FAILED


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class _Book:
    """One scope's runs, their keys, their stored submissions and retrievals,
    and the plans that compiled them, by revision."""

    runs: dict[str, RunRecord] = field(default_factory=dict)
    by_key: dict[str, str] = field(default_factory=dict)
    submissions: dict[str, JsonValue] = field(default_factory=dict)
    retrievals: dict[str, JsonValue] = field(default_factory=dict)
    plans: dict[str, JsonValue] = field(default_factory=dict)


class InMemoryRunLedger:
    """A `RunLedger` held in this process, for tests and the standalone harness.

    `scope` names the conversation each call belongs to, and each scope's runs
    are kept apart: one conversation never sees, resumes or carries on
    another's. With no `scope`, every call shares one."""

    def __init__(
        self, *, clock: Callable[[], datetime] = _utc_now, scope: Callable[[], str] = lambda: ""
    ) -> None:
        self._clock = clock
        self._scope = scope
        self._books: defaultdict[str, _Book] = defaultdict(_Book)

    def _book(self) -> _Book:
        return self._books[self._scope()]

    async def admit(self, candidate: RunRecord, *, submission: JsonValue) -> Admission:
        book = self._book()
        existing_id = book.by_key.get(candidate.idempotency_key)
        existing = book.runs[existing_id] if existing_id is not None else None
        admission = admit_against(existing, candidate, now=self._clock())
        if not admission.resumed:
            book.runs[candidate.run_id] = candidate
            book.by_key[candidate.idempotency_key] = candidate.run_id
            book.submissions[candidate.run_id] = submission
        return admission

    async def get(self, run_id: str) -> RunRecord | None:
        return self._book().runs.get(run_id)

    async def submission(self, run_id: str) -> JsonValue:
        return self._book().submissions.get(run_id)

    async def retrievals(self, run_id: str) -> JsonValue:
        return self._book().retrievals.get(run_id)

    async def store_plan(self, revision: str, plan: JsonValue) -> None:
        self._book().plans.setdefault(revision, plan)

    async def plan(self, revision: str) -> JsonValue:
        return self._book().plans.get(revision)

    async def unfinished(self) -> tuple[RunRecord, ...]:
        return tuple(run for run in self._book().runs.values() if carried_on(run))

    async def record(self, run_id: str, change: RunChange) -> RunRecord:
        book = self._book()
        run = book.runs.get(run_id)
        if run is None:
            raise RunNotFoundError(run_id)
        advanced = advance(run, change, now=self._clock())
        if isinstance(change, Seal) and advanced.manifest != run.manifest:
            book.retrievals[run_id] = list(change.retrievals)
        book.runs[run_id] = advanced
        return advanced
