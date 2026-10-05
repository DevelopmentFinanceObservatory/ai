"""What a run produces: accepted evidence, a choice for the planner, or a refusal.

Every outcome carries the plan revision it ran, so a reader can tie it to
exactly what was approved, and every refusal says which step or requirement
it is about and what the tool or the check said.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from common.materialisation import FrameRef
from compiler.evidence import EvidenceIssue
from planner.plan import SessionState
from planner.records import Identifier, Period, SpecId, StrictModel
from planner.workflow import ToolId
from pydantic import Field, JsonValue, TypeAdapter

from executor.classify import ToolFailure

# The session-state slot holding the latest `RunOutcome`. Separate from the
# plan's own slot, so the plan stays a control record.
RUN_STATE_KEY = "executor:run"


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


class StepRecord(StrictModel):
    """One step that ran: what it called, how many attempts it took, what it produced.

    `row_count`, `role_keys` and `stat_var_count` are the row count, the
    frame's role columns and the distinct StatVars a successful call
    declared in its own `structured_content`, or `None` when it declared
    none. `get_data`, whose structured result is a `Manifest`, declares no
    `rows`, and declares `role_keys` and `stat_var_count` whenever it read
    the frame.
    """

    step_id: Identifier
    tool: ToolId
    attempts: int
    frame_ref: FrameRef
    digest: str
    row_count: int | None = None
    role_keys: tuple[str, ...] | None = None
    stat_var_count: int | None = None


class EvidenceRecord(StrictModel):
    """One requirement's evidence: the frame that carries it and what the checks found.

    `gaps` holds the findings the requirement accepts -- years missing under
    `report_gaps`, a unit the source did not state -- so an answer built on
    this evidence can say what it rests on.
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
    """Every step ran and every requirement is met."""

    kind: Literal["accepted"] = "accepted"
    revision: str
    evidence: tuple[EvidenceRecord, ...]
    steps: tuple[StepRecord, ...]
    results: dict[str, FrameRef]


class MethodChoiceNeeded(StrictModel):
    """A tool refused with a choice only the planner can make with the user.

    `candidates` is whatever the refusal offered to choose from, verbatim.
    `also_stopped` names the wave's other stops, wave order, when this one
    is the wave's first and so the one a run raises (`executor.run._Run.execute`).
    `completed` and `evidence` hold the steps that finished before the stop
    and the evidence their retrievals recorded, in the plan's own step order.
    `answers` holds, by step id, each finished terminal step's frame that
    passed the answer check an accepted run applies (`executor.run._Run.
    _answer_refusal`).
    """

    kind: Literal["method_choice_needed"] = "method_choice_needed"
    revision: str
    step_id: Identifier
    tool: ToolId
    code: str
    candidates: JsonValue
    refusal: ToolFailure
    completed: tuple[StepRecord, ...]
    evidence: tuple[EvidenceRecord, ...] = ()
    answers: dict[str, FrameRef] = Field(default_factory=dict)
    also_stopped: tuple[MethodChoiceNeeded | Refused | RediscoveryNeeded, ...] = ()


class Refused(StrictModel):
    """The run stopped: what stopped it, where, and what was said.

    `also_stopped` names the wave's other stops, wave order, when this one
    is the wave's first and so the one a run raises (`executor.run._Run.execute`).
    `completed` and `evidence` hold the steps that finished before the stop
    and the evidence their retrievals recorded, in the plan's own step order.
    `answers` holds, by step id, each finished terminal step's frame that
    passed the answer check an accepted run applies (`executor.run._Run.
    _answer_refusal`).
    """

    kind: Literal["refused"] = "refused"
    revision: str
    refusal: RefusalKind
    detail: str
    step_id: Identifier | None = None
    requirement_id: Identifier | None = None
    payload: ToolFailure | None = None
    issues: tuple[EvidenceIssue, ...] = ()
    completed: tuple[StepRecord, ...] = ()
    evidence: tuple[EvidenceRecord, ...] = ()
    answers: dict[str, FrameRef] = Field(default_factory=dict)
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
    _answer_refusal`).
    """

    kind: Literal["rediscovery_needed"] = "rediscovery_needed"
    revision: str
    step_id: Identifier
    tool: ToolId
    requirement_id: Identifier
    spec_id: SpecId
    refusal: ToolFailure
    completed: tuple[StepRecord, ...]
    evidence: tuple[EvidenceRecord, ...] = ()
    answers: dict[str, FrameRef] = Field(default_factory=dict)
    also_stopped: tuple[MethodChoiceNeeded | Refused | RediscoveryNeeded, ...] = ()


MethodChoiceNeeded.model_rebuild()
Refused.model_rebuild()
RediscoveryNeeded.model_rebuild()

RunOutcome = Annotated[
    Accepted | MethodChoiceNeeded | Refused | RediscoveryNeeded, Field(discriminator="kind")
]

_OUTCOME_ADAPTER: TypeAdapter[Any] = TypeAdapter(RunOutcome)


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
