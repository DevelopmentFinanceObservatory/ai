"""What a run produces: accepted evidence, a choice for the planner, or a refusal.

Every outcome carries the plan revision it ran, so a reader can tie it to
exactly what was approved, and the id of the run the ledger recorded it under
(`executor.ledger`), `None` only for a refusal made before admission. Every
refusal says which step or requirement it is about and what the tool or the
check said.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Literal

from common.materialisation import FrameRef
from compiler.evidence import EvidenceIssue, EvidenceIssueCode, errors
from planner.plan import SessionState
from planner.records import Identifier, Period, SpecId, StrictModel
from planner.workflow import ToolId
from pydantic import Field, JsonValue, TypeAdapter

from executor.classify import (
    Disposition,
    ExecutorCode,
    ToolFailure,
    classify_refusal,
    is_harness_fault,
)

# The session-state slot holding the latest `RunOutcome`. Separate from the
# plan's own slot, so the plan stays a control record.
RUN_STATE_KEY = "executor:run"

logger = logging.getLogger(__name__)


class RefusalKind(StrEnum):
    """Why a run stopped."""

    NOT_APPROVED = "not_approved"
    STALE_REVISION = "stale_revision"
    SURFACE_CHANGED = "surface_changed"
    TRANSPORT_FAILED = "transport_failed"
    BINDING_INVALID = "binding_invalid"
    STEP_REFUSED = "step_refused"
    STEP_RESULT_UNUSABLE = "step_result_unusable"
    REQUIREMENT_UNMET = "requirement_unmet"
    EMPTY_ANSWER = "empty_answer"
    SHAPE_MISMATCH = "shape_mismatch"
    COLUMN_UNBOUND = "column_unbound"
    ROLLING_WINDOW_OUTSIDE_DATASET = "rolling_window_outside_dataset"
    ROLLING_ANCHOR_BEHIND_PLAN = "rolling_anchor_behind_plan"
    AUTHORISATION_EXPIRED = "authorisation_expired"
    MANIFEST_OUTSIDE_AUTHORISATION = "manifest_outside_authorisation"
    AUTHORISATION_MISMATCH = "authorisation_mismatch"
    RUN_ABANDONED = "run_abandoned"


class LeftOutSeries(StrictModel):
    """One series counted for the newest period that holds no value at it.

    `root` indexes the populations `resolve_anchor` was sent, in their
    canonical order, and `coordinates` is the series' value in each of its
    columns.
    """

    root: int
    coordinates: dict[str, str]


class AnchorResult(StrictModel):
    """The newest period one group of a run's rolling steps stands on, as
    `resolve_anchor` answered it (`compiler.rolling.anchor_groups`).

    `anchor_key` names the stored resolution every rolling call of the group
    was sent, `period` is the newest period held by at least half of the
    series of each population, and `left_out` the series that hold no value
    at it. `steps` are the rolling steps of every group that stands on it
    (`AnchorGroup.steps`), `roots` the retrieval steps of those groups
    (`AnchorGroup.roots`), and `populations` the populations the period was
    taken over, spelled the way the retrieved frames spell their columns
    (`compiler.frame_columns.bound_populations`). The run writes all three;
    `resolve_anchor` does not send them.
    """

    anchor_key: str
    period: str
    left_out: tuple[LeftOutSeries, ...] = ()
    steps: tuple[str, ...] = ()
    roots: tuple[str, ...] = ()
    populations: tuple[JsonValue, ...] = ()


class StepRecord(StrictModel):
    """One step that ran: what it called, how many attempts it took, what it produced.

    `row_count`, `role_keys` and `stat_var_count` are the row count, the
    frame's role columns and the distinct StatVars a successful call
    declared in its own `structured_content`, or `None` when it declared
    none. `get_data`, whose structured result is a `Manifest`, declares no
    `rows`, and declares `role_keys` and `stat_var_count` whenever it read
    the frame. `dropped_key_count` and `right_dropped_key_count` are how many
    keys of its first and second input the call declared it left out of its
    result (`executor.artifacts.declared_dropped_key_counts`): the numerator
    keys a keyed `quotient_of` or `share_of` found nothing to divide by, and
    each side's entities a `join` matched in no period. Both are 0 for every
    other call, and `right_dropped_key_count` for every call but a `join`.
    `ranked_before_limit` is how many entries a `rank` call ranked before its
    `limit` cut them (`executor.artifacts.declared_ranked_before_limit`),
    `None` when the call set no limit.

    `frame_ref` is the pair later steps read the frame by. `manifest_ref` is
    the manifest's own pair for a `get_data` step whose `frame_ref` differs
    from it (a frame minted with extra columns,
    `executor.artifacts.manifest_frame_ref`), and `None` otherwise.
    """

    step_id: Identifier
    tool: ToolId
    attempts: int
    frame_ref: FrameRef
    manifest_ref: FrameRef | None = None
    digest: str
    row_count: int | None = None
    role_keys: tuple[str, ...] | None = None
    stat_var_count: int | None = None
    dropped_key_count: int = Field(default=0, ge=0)
    right_dropped_key_count: int = Field(default=0, ge=0)
    ranked_before_limit: int | None = Field(default=None, ge=0)

    @property
    def presented_ref(self) -> FrameRef:
        """The pair the frame is described and drawn by: the one apis serves."""
        return self.manifest_ref or self.frame_ref


class EvidenceRecord(StrictModel):
    """One requirement's evidence: the frame that carries it and what the checks found.

    `gaps` holds the findings the requirement accepts -- years missing under
    `report_gaps`, a named entity the retrieval left out, a unit the source
    did not state -- so an answer built on this evidence can say what it
    rests on.
    """

    requirement_id: Identifier
    step_id: Identifier
    frame_ref: FrameRef
    spec_id: SpecId
    stat_vars: tuple[str, ...]
    window: Period | None
    gaps: tuple[EvidenceIssue, ...]
    citation: str


class Accepted(StrictModel):
    """Every step ran and every requirement is met.

    `anchors` holds each newest period the run's rolling steps were written
    from, in the order resolved, one for each group of rolling steps
    (`compiler.rolling.anchor_groups`), with groups over the same series and
    frames sharing one. Empty when no step rolls.
    """

    kind: Literal["accepted"] = "accepted"
    revision: str
    run_id: str | None = None
    evidence: tuple[EvidenceRecord, ...]
    steps: tuple[StepRecord, ...]
    results: dict[str, FrameRef]
    anchors: tuple[AnchorResult, ...] = ()


class MethodChoiceNeeded(StrictModel):
    """A tool refused with a choice only the planner can make with the user.

    `candidates` is whatever the refusal offered to choose from, verbatim.
    `also_stopped` names the wave's other stops, wave order, when this one
    is the wave's first and so the one a run raises (`executor.run._Run.execute`).
    `completed` and `evidence` hold the steps that finished before the stop
    and the evidence their retrievals recorded, in the plan's own step order.
    `answers` holds, by step id, each finished terminal step's frame that
    passed the answer check an accepted run applies (`executor.run._Run.
    _answer_refusal`). `anchors` holds the newest period of each group of
    rolling steps the run resolved before it stopped.
    """

    kind: Literal["method_choice_needed"] = "method_choice_needed"
    revision: str
    run_id: str | None = None
    step_id: Identifier
    tool: ToolId
    code: str
    candidates: JsonValue
    refusal: ToolFailure
    completed: tuple[StepRecord, ...]
    evidence: tuple[EvidenceRecord, ...] = ()
    answers: dict[str, FrameRef] = Field(default_factory=dict)
    anchors: tuple[AnchorResult, ...] = ()
    also_stopped: tuple[MethodChoiceNeeded | Refused | RediscoveryNeeded, ...] = ()


class Refused(StrictModel):
    """The run stopped: what stopped it, where, and what was said.

    `also_stopped` names the wave's other stops, wave order, when this one
    is the wave's first and so the one a run raises (`executor.run._Run.execute`).
    `completed` and `evidence` hold the steps that finished before the stop
    and the evidence their retrievals recorded, in the plan's own step order.
    `answers` holds, by step id, each finished terminal step's frame that
    passed the answer check an accepted run applies (`executor.run._Run.
    _answer_refusal`). `anchors` holds the newest period of each group of
    rolling steps the run resolved before it stopped.

    `spec_id` names the source a `REQUIREMENT_UNMET` stop read, when the
    requirement has a binding, as a `RediscoveryNeeded` does. Other stops leave
    it unset. A replan avoids the source of a `Refused` stop only when an error
    finding says its unit is unverified and found no unit text
    (`harness.planner_adapter._SOURCE_FAULT_CODES`).

    `retryable` says whether a later attempt can get past a
    `TRANSPORT_FAILED` stop: `False` when the tool surface reported a failure
    no retry changes (`compiler.surface.ToolSurfaceUnavailableError`), such as
    a server it does not cover. It is `True` for every other stop.
    """

    kind: Literal["refused"] = "refused"
    revision: str
    run_id: str | None = None
    refusal: RefusalKind
    detail: str
    retryable: bool = True
    step_id: Identifier | None = None
    requirement_id: Identifier | None = None
    # Left out of a dump while unset, so a stored refusal reads as before.
    spec_id: SpecId | None = Field(default=None, exclude_if=lambda spec_id: spec_id is None)
    payload: ToolFailure | None = None
    issues: tuple[EvidenceIssue, ...] = ()
    completed: tuple[StepRecord, ...] = ()
    evidence: tuple[EvidenceRecord, ...] = ()
    answers: dict[str, FrameRef] = Field(default_factory=dict)
    anchors: tuple[AnchorResult, ...] = ()
    also_stopped: tuple[MethodChoiceNeeded | Refused | RediscoveryNeeded, ...] = ()


class RediscoveryNeeded(StrictModel):
    """A retrieval's source needs discovering again; a bare retry would only repeat the refusal.

    Discovering the source again is the translator's job under a new plan
    revision, not a retry this loop can make on its own. `requirement_id`
    and `spec_id` name the requirement and the source the step was approved
    to retrieve, read off its `EvidenceBinding`; a step without one has
    nothing to rediscover, so its refusal comes back as `Refused` instead.
    `also_stopped` names the wave's other stops, wave order, when this one
    is the wave's first and so the one a run raises (`executor.run._Run.execute`).
    `completed` and `evidence` hold the steps that finished before the stop
    and the evidence their retrievals recorded, in the plan's own step order.
    `answers` holds, by step id, each finished terminal step's frame that
    passed the answer check an accepted run applies (`executor.run._Run.
    _answer_refusal`). `anchors` holds the newest period of each group of
    rolling steps the run resolved before it stopped.
    """

    kind: Literal["rediscovery_needed"] = "rediscovery_needed"
    revision: str
    run_id: str | None = None
    step_id: Identifier
    tool: ToolId
    requirement_id: Identifier
    spec_id: SpecId
    refusal: ToolFailure
    completed: tuple[StepRecord, ...]
    evidence: tuple[EvidenceRecord, ...] = ()
    answers: dict[str, FrameRef] = Field(default_factory=dict)
    anchors: tuple[AnchorResult, ...] = ()
    also_stopped: tuple[MethodChoiceNeeded | Refused | RediscoveryNeeded, ...] = ()


MethodChoiceNeeded.model_rebuild()
Refused.model_rebuild()
RediscoveryNeeded.model_rebuild()

RunOutcome = Annotated[
    Accepted | MethodChoiceNeeded | Refused | RediscoveryNeeded, Field(discriminator="kind")
]

_OUTCOME_ADAPTER: TypeAdapter[Any] = TypeAdapter(RunOutcome)


def outcome_from_json(
    raw: JsonValue,
) -> Accepted | MethodChoiceNeeded | Refused | RediscoveryNeeded:
    """The outcome a run record holds (`executor.ledger.RunRecord.outcome`)."""
    return _OUTCOME_ADAPTER.validate_python(raw)


def load_outcome(
    state: SessionState,
) -> Accepted | MethodChoiceNeeded | Refused | RediscoveryNeeded | None:
    """Return the latest run outcome held in session `state`, or `None` when none is."""
    raw = state.get(RUN_STATE_KEY)
    return _OUTCOME_ADAPTER.validate_json(raw) if raw else None


def save_outcome(
    state: SessionState, outcome: Accepted | MethodChoiceNeeded | Refused | RediscoveryNeeded
) -> None:
    """Record `outcome` as the latest run in session `state`."""
    state[RUN_STATE_KEY] = outcome.model_dump_json()


# What the orchestrator is told of a stop no tool refused and no evidence check
# explains, by its kind. `Refused.detail` stays out of the hand-back, since it
# can quote an exception, the ledger's failure reason or literal spec ids, and
# is logged instead.
STOP_MESSAGES: dict[RefusalKind, str] = {
    RefusalKind.NOT_APPROVED: "the calculation was not approved",
    RefusalKind.STALE_REVISION: "the calculation changed after it was approved",
    RefusalKind.SURFACE_CHANGED: "a tool the calculation uses changed after it was approved",
    RefusalKind.TRANSPORT_FAILED: "a data service could not be reached in time",
    RefusalKind.BINDING_INVALID: "a step's input could not be passed on to the next step",
    RefusalKind.STEP_REFUSED: "a tool refused the step",
    RefusalKind.STEP_RESULT_UNUSABLE: "a step returned a result that could not be used",
    RefusalKind.REQUIREMENT_UNMET: "the data retrieved does not meet what the calculation asked for",
    RefusalKind.EMPTY_ANSWER: "the step's result holds no rows",
    RefusalKind.SHAPE_MISMATCH: "a step's result does not have the shape the calculation expects",
    RefusalKind.COLUMN_UNBOUND: "a step names a column its input does not have",
    RefusalKind.ROLLING_WINDOW_OUTSIDE_DATASET: (
        "the latest period asked for falls outside the dataset"
    ),
    RefusalKind.ROLLING_ANCHOR_BEHIND_PLAN: (
        "the latest year the data holds is before the one the plan counted from"
    ),
    RefusalKind.AUTHORISATION_EXPIRED: "the approval has expired, so the plan needs approving again",
    RefusalKind.MANIFEST_OUTSIDE_AUTHORISATION: "the run read data the approval does not cover",
    RefusalKind.AUTHORISATION_MISMATCH: "the approval names other work than the run",
    RefusalKind.RUN_ABANDONED: "the run failed before it came to a result",
}


# What the orchestrator is told of a transport stop no later attempt can get
# past (`Refused.retryable`), which it reads as the system fault it is.
PERMANENT_TRANSPORT_MESSAGE = "a data service this step needs is not available"


class StopCause(StrEnum):
    """The cause of a refused run's stop, which decides where a caller outside the run routes it."""

    # A later try can get past it.
    TRANSIENT = "transient"
    # The compiler or the run wrote something wrong, or a server failed of
    # itself, so nothing the user asked for is wrong.
    HARNESS = "harness"
    # A fact of the data or of the plan, which the user decides on.
    DATA = "data"


# The cause of a stop no tool refused, by its kind. The approval, revision and
# ledger kinds are the run's own state, so a stop on one is no fact of the data.
# A stop a tool refused is told by its refusal (`stop_cause`).
STOP_CAUSES: dict[RefusalKind, StopCause] = {
    RefusalKind.NOT_APPROVED: StopCause.HARNESS,
    RefusalKind.STALE_REVISION: StopCause.HARNESS,
    RefusalKind.SURFACE_CHANGED: StopCause.HARNESS,
    RefusalKind.TRANSPORT_FAILED: StopCause.TRANSIENT,
    RefusalKind.BINDING_INVALID: StopCause.HARNESS,
    RefusalKind.STEP_REFUSED: StopCause.HARNESS,
    RefusalKind.STEP_RESULT_UNUSABLE: StopCause.HARNESS,
    RefusalKind.REQUIREMENT_UNMET: StopCause.DATA,
    RefusalKind.EMPTY_ANSWER: StopCause.DATA,
    RefusalKind.SHAPE_MISMATCH: StopCause.HARNESS,
    RefusalKind.COLUMN_UNBOUND: StopCause.HARNESS,
    RefusalKind.ROLLING_WINDOW_OUTSIDE_DATASET: StopCause.DATA,
    RefusalKind.ROLLING_ANCHOR_BEHIND_PLAN: StopCause.DATA,
    RefusalKind.AUTHORISATION_EXPIRED: StopCause.HARNESS,
    RefusalKind.MANIFEST_OUTSIDE_AUTHORISATION: StopCause.HARNESS,
    RefusalKind.AUTHORISATION_MISMATCH: StopCause.HARNESS,
    RefusalKind.RUN_ABANDONED: StopCause.HARNESS,
}


def permanent_transport_stop(outcome: Refused) -> bool:
    """Whether `outcome` is a transport stop no later attempt can get past."""
    return outcome.refusal is RefusalKind.TRANSPORT_FAILED and not outcome.retryable


def stop_cause(outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused) -> StopCause:
    """Whose `outcome`'s stop is.

    A method choice and a source to discover again are the data's, which a
    plan settles with the user. A stop carrying a tool's refusal is transient
    when the tool says waiting helps (`classify_refusal`), the harness's when
    the code says the harness wrote the call wrong or a server failed of
    itself (`is_harness_fault`), and the data's otherwise. A transport stop
    is transient when a later attempt can get past it (`Refused.retryable`)
    and the harness's when none can. Any other stop is told by its kind
    (`STOP_CAUSES`).
    """
    if not isinstance(outcome, Refused):
        return StopCause.DATA
    failure = outcome.payload
    if failure is None:
        if permanent_transport_stop(outcome):
            return StopCause.HARNESS
        return STOP_CAUSES[outcome.refusal]
    if classify_refusal(failure) is Disposition.RETRY_LATER:
        return StopCause.TRANSIENT
    if is_harness_fault(failure):
        return StopCause.HARNESS
    return StopCause.DATA


# What the user can do about a run the evidence check stopped, one sentence
# written for the end user, by the code of its first error (`failed_step`). A
# run's `entity_missing` names a party the approved source does not resolve
# on a side it binds, so the name is the user's to correct. A name on a side
# it leaves open is refused before approval
# (`compiler.evidence.named_on_open_axes`), so asking without the names would
# answer another question. `entity_no_series` has no remedy, since the
# source resolves the name and the retrieval holds no series for it.
ISSUE_REMEDIES: dict[EvidenceIssueCode, str] = {
    EvidenceIssueCode.ENTITY_MISSING: "Ask again with each name as the data lists it",
}


def failed_step(outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused) -> dict[str, Any]:
    """The step a stopped run failed at, as its caller is told it: `step_id`, `tool`, `code` and `message`.

    `code` is the run's own: a tool's code when one refused, the specific
    `EvidenceIssueCode` when the evidence check did, `rediscovery_needed` for
    a source to discover again, and the stop's kind otherwise. A `remedy`
    says what the user can do: a tool's own (`_refused_step`), or the one
    `ISSUE_REMEDIES` gives the evidence check's first error. A stop on a
    tool's refusal also carries that refusal's `retry_mode` and `fix`, which
    word the user's sentence for it (`harness.sentences.failure_sentence`),
    and the `disposition` `classify_refusal` gave it. Every stop carries its
    `cause` (`stop_cause`), and the `wave_remedy` of the stops its wave made
    (`wave_remedy`), which `remedy` reads.
    """
    return {**_own_step(outcome), "wave_remedy": wave_remedy(outcome).value}


def _own_step(outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused) -> dict[str, Any]:
    """`failed_step` for `outcome` alone, without its wave's remedy."""
    return {**_stopped_at(outcome), "cause": stop_cause(outcome).value}


def _stopped_at(outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused) -> dict[str, Any]:
    """`failed_step` without its `cause` and its wave's remedy."""
    if isinstance(outcome, MethodChoiceNeeded):
        return {
            "step_id": outcome.step_id,
            "tool": outcome.tool.name,
            "code": outcome.code,
            "message": outcome.refusal.message,
            **_refusal_terms(outcome.refusal),
        }
    if isinstance(outcome, RediscoveryNeeded):
        return {
            "step_id": outcome.step_id,
            "tool": outcome.tool.name,
            "code": outcome.kind,
            "message": outcome.refusal.message
            or f"the source for requirement {outcome.requirement_id!r} needs discovering again",
            **_refusal_terms(outcome.refusal),
        }
    # `payload`, set on every `STEP_REFUSED` stop, carries the tool's own
    # `code` and `message` from its `ToolFailure`. `outcome.detail` wraps the
    # tool's text ("'aggregate' refused: ...") only when the call left one,
    # since `_refused` falls back to `failure.code` with no `result.text`
    # (`executor.run._Run._stopped_by_refusal`). A `Refused` raised for any
    # other reason (`EMPTY_ANSWER`, `SHAPE_MISMATCH`, ...) has no tool call
    # to report and is told by its kind (`STOP_MESSAGES`), unless it carries
    # `issues` (the `REQUIREMENT_UNMET` stops of `executor.run._Run`'s
    # `_check_evidence` and `_check_left_out_narrowings` are their only
    # source). `outcome.detail` there is generic, naming no sub-check of
    # `check_evidence` and no expected or actual value, so the specific
    # `EvidenceIssueCode` and its own expected/actual detail lead instead.
    if outcome.payload is not None:
        return _refused_step(step_id=outcome.step_id, failure=outcome.payload)
    if outcome.issues:
        failing = errors(list(outcome.issues)) or list(outcome.issues)
        failed: dict[str, Any] = {
            "step_id": outcome.step_id,
            "tool": None,
            "code": failing[0].code.value,
            "message": "; ".join(issue.detail for issue in failing),
        }
        if (remedy := ISSUE_REMEDIES.get(failing[0].code)) is not None:
            failed["remedy"] = remedy
        return failed
    if permanent_transport_stop(outcome):
        return {
            "step_id": outcome.step_id,
            "tool": None,
            "code": ExecutorCode.INTERNAL_ERROR.value,
            "message": PERMANENT_TRANSPORT_MESSAGE,
        }
    return {
        "step_id": outcome.step_id,
        "tool": None,
        "code": outcome.refusal.value,
        "message": STOP_MESSAGES[outcome.refusal],
    }


def _refusal_terms(failure: ToolFailure) -> dict[str, Any]:
    """`failure`'s `retry_mode`, its `fix` and its disposition (`classify_refusal`),
    as `failed_step` carries them."""
    return {
        "retry_mode": failure.retry_mode,
        "fix": dict(failure.fix),
        "disposition": classify_refusal(failure).value,
    }


def _refused_step(*, step_id: str | None, failure: ToolFailure) -> dict[str, Any]:
    """`failed_step` for a step the tool itself refused.

    A tool's `fix` can carry its own `remedy`, one sentence written for the
    end user, where the rest of `fix` holds code/detail prose for a retrying
    caller. `data_mcp.tool_errors` sets one for an `over_ceiling` on
    `member_cardinality` and for the `member_not_resolvable` a mint apis
    refuses. `plan_done` hands it to the orchestrator under `failed_step`.
    """
    failed: dict[str, Any] = {
        "step_id": step_id,
        "tool": None,
        "code": failure.code,
        "message": failure.message,
        **_refusal_terms(failure),
    }
    remedy = failure.fix.get("remedy")
    if isinstance(remedy, str) and remedy:
        failed["remedy"] = remedy
    return failed


class Remedy(StrEnum):
    """What gets past a stopped run's failure (`remedy`), least severe first."""

    # A plan drafted afresh, from another brief or approved again.
    REPLAN = "replan"
    # The user, who accepts a fact of the data or changes the question.
    USER = "user"
    # Waiting: the same plan can run once a service recovers.
    WAIT = "wait"
    # Nothing the user or a plan does: the fault is the system's.
    SYSTEM = "system"


# The stops of the harness's own that are approvals a changed plan or tool
# surface made stale, or that expired. A plan drafted and approved afresh gets
# past each, so they are the harness's stops that a new plan remedies.
LAPSED_APPROVAL_KINDS = frozenset(
    {
        RefusalKind.NOT_APPROVED,
        RefusalKind.STALE_REVISION,
        RefusalKind.SURFACE_CHANGED,
        RefusalKind.AUTHORISATION_EXPIRED,
    }
)

# What gets past a stop of the data's that no disposition settles, by its
# `failed_step` code: a tool's code, the executor's own, a stop's kind or an
# evidence finding. `remedy` reads it after the cause and the disposition, and
# `tests/executor/test_outcomes.py` keeps its keys equal to the codes that
# reach it, so a new code is placed on purpose.
#
# `REPLAN` is a failure a plan drafted from another brief avoids, reading
# another source, spec, scope, frame or method. `USER` is a fact of the data
# the user accepts or changes the question for, a tool no workflow step calls,
# or a stop no other brief changes this turn.
_DATA_STOP_REMEDY: dict[str, Remedy] = {
    # data-mcp: a facet, node, spec or family a different spec avoids.
    "facet_not_available": Remedy.REPLAN,
    "default_facet_undefined": Remedy.REPLAN,
    "node_not_held": Remedy.REPLAN,
    "spec_not_exportable": Remedy.REPLAN,
    "member_not_resolvable": Remedy.REPLAN,
    "rollup_not_summable": Remedy.REPLAN,
    # A member-cardinality or candidate-window ceiling is fixed by the spec's
    # own definition: the user narrows the topic or names a breakdown. A
    # ceiling a narrower request avoids is a method choice.
    "over_ceiling": Remedy.USER,
    "capability_unavailable": Remedy.USER,
    "entity_not_found": Remedy.USER,
    # Only a definition the platform refused to register, which another spec
    # avoids (`is_upstream_fault`).
    "upstream_error": Remedy.REPLAN,
    # Tools no workflow step calls (`compiler.capabilities.EXCLUDED_TOOLS`).
    "skill_not_found": Remedy.USER,
    "skill_resource_not_found": Remedy.USER,
    "allowance_exhausted": Remedy.USER,
    "provenance_unavailable": Remedy.USER,
    "licence_not_redistributable": Remedy.USER,
    # analysis-mcp: a frame, unit, join or method a different plan avoids.
    "stat_var_unresolved": Remedy.REPLAN,
    "frame_too_large": Remedy.REPLAN,
    # A stored frame that failed its integrity check: a narrower scope mints
    # another content version, which gets past it.
    "frame_integrity_failed": Remedy.REPLAN,
    "non_finite_result": Remedy.REPLAN,
    "frame_not_held": Remedy.REPLAN,
    "frame_expired": Remedy.REPLAN,
    "mixed_time_grain": Remedy.REPLAN,
    "filter_unit_conversion_unsupported": Remedy.REPLAN,
    "join_output_collision": Remedy.REPLAN,
    "derived_input_unusable": Remedy.REPLAN,
    "derived_input_unrecoverable": Remedy.REPLAN,
    "derived_input_ordinal": Remedy.REPLAN,
    "rolling_reanchor_unavailable": Remedy.REPLAN,
    "quotient_of_side_unit_unsupported": Remedy.REPLAN,
    "rendering_not_recorded": Remedy.REPLAN,
    "rolling_grain_unsupported": Remedy.REPLAN,
    # No year half of the series hold, or a pairing whose sides share no key.
    "rolling_anchor_unresolved": Remedy.USER,
    # The executor's own stops on the data a plan read. Each holds for any plan
    # over the same data, and its refusal tells the user what to ask instead:
    # the latest year the series reach, a later start, or the family read apart.
    "rolling_anchor_behind_plan": Remedy.USER,
    "rolling_window_outside_dataset": Remedy.USER,
    "family_not_summed": Remedy.USER,
    # The run's own stops on what it retrieved.
    "requirement_unmet": Remedy.REPLAN,
    "empty_answer": Remedy.REPLAN,
    # The evidence check's findings on what a retrieval holds, which another
    # source, scope or reading avoids.
    "no_data_at_source": Remedy.REPLAN,
    "data_unverified": Remedy.REPLAN,
    "identity_unverified": Remedy.REPLAN,
    "entity_unverified": Remedy.REPLAN,
    "entity_role_ambiguous": Remedy.REPLAN,
    "entity_role_mismatch": Remedy.REPLAN,
    "entity_role_unverified": Remedy.REPLAN,
    "group_read_as_total": Remedy.REPLAN,
    "open_axis_unresolved": Remedy.REPLAN,
    "scope_entities_unbound": Remedy.REPLAN,
    "scope_entities_wider": Remedy.REPLAN,
    "unit_mismatch": Remedy.REPLAN,
    "unit_unverified": Remedy.REPLAN,
    "unit_price_basis_mismatch": Remedy.REPLAN,
    "unit_price_basis_unverified": Remedy.REPLAN,
    "source_mismatch": Remedy.REPLAN,
    "source_unverified": Remedy.REPLAN,
    "source_drift": Remedy.REPLAN,
    "rows_empty": Remedy.REPLAN,
    "coverage_gaps": Remedy.REPLAN,
    "grain_ambiguous": Remedy.REPLAN,
    # The findings no other brief changes this turn: the user's name as the
    # data lists it, a named party with no series, a period the data does not
    # reach or confirm, a family the turn approves again as it was, a rebound
    # binding every search meets, and a manifest that read another pin or
    # states no statistics.
    "entity_missing": Remedy.USER,
    "entity_no_series": Remedy.USER,
    "window_short": Remedy.USER,
    "scope_window_short": Remedy.USER,
    "period_unverified": Remedy.USER,
    "stat_var_family_moved": Remedy.USER,
    "stat_var_membership_drift": Remedy.USER,
    "source_pin_mismatch": Remedy.USER,
    "statistics_unavailable": Remedy.USER,
    # No check reports these as a failure.
    "source_substituted": Remedy.USER,
    "unit_matches": Remedy.USER,
}


# The remedies from least to most severe. A wave's is its most severe stop's.
_SEVERITY = (Remedy.REPLAN, Remedy.WAIT, Remedy.USER, Remedy.SYSTEM)


def remedy(failed: Mapping[str, object]) -> Remedy:
    """What gets past the stopped run `failed`, its `failed_step`, names: the
    `wave_remedy` it carries. A failed step that carries none is the system's.

    `REPLAN`: a plan drafted afresh, so the turn's plan call is given back once.
    `WAIT`: a service did not answer, so no plan is given back and the user is
    told to try again shortly. `USER`: the user accepts a fact of the data or
    changes the question. `SYSTEM`: the fault is the system's, and the user is
    not asked to change the question.
    """
    stated = failed.get("wave_remedy")
    return Remedy(stated) if isinstance(stated, str) and stated in set(Remedy) else Remedy.SYSTEM


def wave_remedy(outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused) -> Remedy:
    """The most severe remedy (`_SEVERITY`) of `outcome`'s stop and of each
    stop in `outcome.also_stopped`, so a new plan gets past the wave only when
    it gets past every stop in it (`stop_remedy`)."""
    stops = (outcome, *outcome.also_stopped)
    return max((stop_remedy(_own_step(stop)) for stop in stops), key=_SEVERITY.index)


def stop_remedy(failed: Mapping[str, object]) -> Remedy:
    """What gets past one stop, `failed` as `failed_step` states it alone.

    Read in order: a transient stop waits, and a stop of the harness's own is
    the system's, except an approval a fresh plan gets past
    (`LAPSED_APPROVAL_KINDS`). A data stop a method choice or a rediscovery
    settles (`Disposition`) is re-planned. Any other is decided by its code
    (`_DATA_STOP_REMEDY`), and an unplaced code is the user's, so it never
    costs the user a second approval. A stop with no cause is the system's.
    """
    cause = failed.get("cause")
    code = failed.get("code")
    if cause == StopCause.TRANSIENT:
        return Remedy.WAIT
    if cause == StopCause.HARNESS:
        return Remedy.REPLAN if code in LAPSED_APPROVAL_KINDS else Remedy.SYSTEM
    if cause != StopCause.DATA:
        return Remedy.SYSTEM
    if failed.get("disposition") in (Disposition.METHOD_CHOICE, Disposition.REDISCOVER):
        return Remedy.REPLAN
    placed = _DATA_STOP_REMEDY.get(code) if isinstance(code, str) else None
    if placed is None:
        logger.warning("no remedy placed for the data stop %r", code)
        return Remedy.USER
    return placed
