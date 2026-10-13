"""Compiling a direct recipe's requirements through the planner's own chain.

`compile_direct` runs a recipe's requirements through exactly what a
planner's proposal goes through: the shared submission service
(`submission.service.admit`), which writes the source decision a settled
pin needs and windows the pin to the requirement's years, then the
translation `propose_plan` runs (`translator.tool.build_translate`), which
confirms the pin and compiles the workflow. No model is called. Both run on a
scratch session (`_DirectSession`) seeded only with the pinned source as the
conversation recorded it and the spec_ids the user chose, so nothing of the
planner's session state is read or written, and the same requirements
compile to the same workflow on either route.

The `Plan` the chain builds carries no approval or amendment state. The
caller records it as the turn's plan record once its run completes
(`harness.graph.plan_done.record_plan`), as a planned run's is.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import rfc8785
from common.hashed_fields import revision_content
from compiler.surface import ToolSurface
from google.adk.tools.tool_context import ToolContext
from planner.plan import (
    AnswerMode,
    DatasetRequirement,
    EvidenceRequirement,
    PinSource,
    Plan,
    SessionState,
    UnsupportedOperation,
    load_plan,
    record_unsupported,
    save_pin_sources,
    save_plan,
    save_plan_rounds,
    save_windowed_pins,
)
from planner.tool_results import Outcome
from planner.workflow import Workflow
from translator.tool import build_translate

from submission.service import Admission, Combination, Proposal, Proposer, admit


def graph_digest(workflow: Workflow) -> str:
    """The digest of a compiled workflow's canonical revision content
    (`common.hashed_fields.revision_content`), as `executor.ledger.
    submission_digest` hashes a submission's, so a release that adds a
    `HASHED_WHEN_SET` field at its default leaves it unchanged."""
    return hashlib.sha256(rfc8785.dumps(revision_content(workflow))).hexdigest()


@dataclass
class _DirectSession:
    """A `submission.service.SessionView` over a scratch state.

    `elected` are the spec_ids the user chose, as the conversation recorded
    them, so a measure the user picked is the user's decision in the plan.
    A direct route asks nothing, so the other readers answer nothing, and it
    runs on the user's own new request over a state that recorded no earlier
    plan, so the user has spoken since any plan version. An admission is
    recorded as `ToolContextSession` records one.
    """

    state: SessionState = field(default_factory=dict)
    elected: frozenset[str] = frozenset()

    def elected_spec_ids(self) -> frozenset[str]:
        return self.elected

    def answers_since_referral(self) -> tuple[bool, frozenset[str]]:
        return False, frozenset()

    def combination_picks(self) -> tuple[dict[Combination, str | None], bool]:
        return {}, False

    def user_spoke_since(self, version: int) -> bool:
        return True

    def record_unsupported(self, operation: UnsupportedOperation) -> None:
        record_unsupported(self.state, operation)

    def record_admission(self, admission: Admission) -> None:
        if admission.windowed_pins is not None:
            save_windowed_pins(self.state, admission.windowed_pins)
        save_plan(self.state, admission.plan)
        save_plan_rounds(self.state, admission.rounds)


@dataclass(frozen=True)
class _ScratchContext:
    """What the translation reads of a tool context: its state."""

    state: SessionState


@dataclass(frozen=True)
class Compiled:
    """A direct request's admitted plan, its compiled workflow and the dataset
    requirements its retrievals are checked against."""

    plan: Plan
    workflow: Workflow
    requirements: tuple[DatasetRequirement, ...]

    @property
    def digest(self) -> str:
        return graph_digest(self.workflow)


async def compile_direct(
    *,
    question: str,
    requirements: Sequence[EvidenceRequirement],
    pins: Mapping[str, PinSource],
    surface: ToolSurface,
    proposer: Proposer,
    elected: frozenset[str] = frozenset(),
) -> Compiled | dict[str, Any]:
    """Compile `requirements` as the planner's chain would, over `pins`, the
    sources the conversation recorded, by spec_id, and `elected`, the
    spec_ids the user chose (`submission.service.SessionView.elected_spec_ids`).

    `surface` is the one the planner's translation runs against: it confirms
    a pin through data-mcp when the pin is windowed to other years, and lists
    the tool schemas the workflow is compiled against. `proposer` is the
    tool the request came through, which any `action` in the service's
    refusal names.

    Returns:
        The compiled workflow, or the refusal that stopped it: the service's
        typed problem, or the translation's referral.
    """
    session = _DirectSession(elected=elected)
    save_pin_sources(session.state, pins)
    admitted = admit(
        session,
        Proposal(
            question=question,
            answer_mode=AnswerMode.COMPUTED,
            evidence_requirements=list(requirements),
        ),
        question_fixed=False,
        proposer=proposer,
    )
    if not isinstance(admitted, Admission):
        return admitted
    session.record_admission(admitted)
    translate = build_translate(surface=surface)
    verdict = await translate(cast("ToolContext", _ScratchContext(session.state)))
    plan = load_plan(session.state)
    if verdict.get("outcome") != Outcome.OK.value or plan is None or plan.workflow is None:
        return dict(verdict)
    return Compiled(
        plan=plan,
        workflow=plan.workflow,
        requirements=tuple(
            r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)
        ),
    )
