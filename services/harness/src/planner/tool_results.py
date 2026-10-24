"""One typed result shape for the planner's tools and the translator's core functions.

A tool that answers `{"error": "..."}` hands its caller a sentence to parse
back into a decision: does this fail because a plan is missing, because a
budget ran out, because a proposal has a bad shape? A model has to guess from
prose, and every caller that wants to branch on the answer has to guess the
same way independently. Every tool below returns one of `ok`, `problem` or
`referred` instead: a stable `outcome` a caller switches on, a `code` from one
shared vocabulary (`ProblemCode`) that names what went wrong without parsing a
sentence, and, where the next step is known, an `action` -- which tool to call
next and with which arguments -- so a model that hits a problem is pointed at
its recovery rather than left to reason about the message.

`ProblemCode` is one vocabulary for the planner and the translator so that a
code never has to be renamed when a function moves from one to the other, and
so a caller reading both learns one set of names, not two.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import JsonValue, model_serializer

from planner.records import StrictModel


class Outcome(StrEnum):
    """How a tool call landed, at the coarsest grain a caller branches on.

    `UNSUPPORTED` means the call asked for something the system cannot do:
    no change to the input makes it work, so the caller records a decline.

    A translator referral's own `outcome` (`translator.referral.OUTCOME_OF`)
    reads from this vocabulary too, with three members only a translation
    lands on: `CHOICE_REQUIRED`, a choice between sources the planner or
    the user settles; `TRANSIENT`, a service that did not answer just now,
    so the same input is worth one more try; and `INTERNAL`, a fault in the
    system itself, carrying a diagnostic reference and never retried.
    `INVALID_INPUT` there is input the planner can repair.
    """

    OK = "ok"
    INVALID_INPUT = "invalid_input"
    STATE_MISSING = "state_missing"
    REFERRED = "referred"
    REFUSED = "refused"
    UNSUPPORTED = "unsupported"
    CHOICE_REQUIRED = "choice_required"
    TRANSIENT = "transient"
    INTERNAL = "internal"


class ProblemCode(StrEnum):
    """What, specifically, stopped a tool from doing what it was asked.

    One vocabulary for every tool of the planner and the translator, so a
    code names the same thing wherever it appears.
    """

    PLAN_MISSING = "plan_missing"
    PLAN_NOT_DRAFT = "plan_not_draft"
    WORKFLOW_MISSING = "workflow_missing"
    PLAN_REJECTED = "plan_rejected"
    REFERRAL_PENDING = "referral_pending"
    TRANSITION_REFUSED = "transition_refused"
    APPROVAL_DECLINED = "approval_declined"
    REQUIREMENT_UNKNOWN = "requirement_unknown"
    REQUIREMENT_NOT_A_DATASET = "requirement_not_a_dataset"
    DECISION_UNKNOWN = "decision_unknown"
    INVALID_PLAN = "invalid_plan"
    INVALID_REFERRAL = "invalid_referral"
    INVALID_ARGUMENTS = "invalid_arguments"
    COMPILE_FAILED = "compile_failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    QUESTION_FIXED = "question_fixed"
    SEARCH_REFUSED = "search_refused"
    SURFACE_UNAVAILABLE = "surface_unavailable"
    # Reading the newest year a plan's rolling years count from, before
    # approval, stopped where the plan's run would (`translator.tool`).
    RUN_WOULD_STOP = "run_would_stop"
    NO_CANDIDATES_RECORDED = "no_candidates_recorded"
    ASK_NOT_UNDERSTOOD = "ask_not_understood"
    SOURCE_NOT_OFFERED = "source_not_offered"
    COMBINATION_UNASKED = "combination_unasked"
    OPERATION_UNSUPPORTED = "operation_unsupported"
    DECLINE_MISSING = "decline_missing"
    DECLINE_UNEARNED = "decline_unearned"
    NOTHING_TO_PLAN = "nothing_to_plan"
    ASKS_UNACCOUNTED = "asks_unaccounted"
    ASK_ROLE_DROPPED = "ask_role_dropped"
    RESULT_SCOPE_UNSTATED = "result_scope_unstated"
    PERIOD_REDUCTION_UNSTATED = "period_reduction_unstated"
    USER_DECISION_CONTRADICTED = "user_decision_contradicted"
    USER_DECISION_DROPPED = "user_decision_dropped"
    SHARED_DECISION_CONTRADICTED = "shared_decision_contradicted"


class Action(StrictModel):
    """What to call next to move past a problem, and with which arguments.

    `arguments` is left unset when the next call needs judgement a tool
    cannot supply -- which requirement to describe, say -- and set when the
    tool already knows exactly what the retry looks like. A third case sits
    between those two: `arguments` holds what is already known and `needs`
    names the rest, the arguments the caller has to supply itself before the
    call is complete.
    """

    call: str
    arguments: dict[str, JsonValue] | None = None
    needs: tuple[str, ...] = ()

    @model_serializer(mode="wrap")
    def _omit_needs_at_its_default(self, handler: Any) -> dict[str, object]:
        """Drop `needs` from the dump when it is `()`, scoped to this model alone.

        `exclude_none`/`exclude_defaults` on an enclosing `model_dump` call
        would apply to every field of every model in the tree, including
        ones a future field could add to `ToolProblem` with a non-`None`
        default -- a field silently missing from the model-facing payload
        because it happened to hold its default. Serializing the omission
        here keeps it local to the one field it is for.
        """
        data = handler(self)
        if not self.needs:
            data.pop("needs", None)
        return data


class ToolProblem(StrictModel):
    """One reason a tool call did not succeed, and what to do about it."""

    outcome: Literal[
        Outcome.INVALID_INPUT, Outcome.STATE_MISSING, Outcome.REFUSED, Outcome.UNSUPPORTED
    ]
    code: ProblemCode
    message: str
    action: Action | None = None
    detail: dict[str, JsonValue] | None = None


def problem(
    code: ProblemCode,
    message: str,
    *,
    outcome: Literal[
        Outcome.INVALID_INPUT, Outcome.STATE_MISSING, Outcome.REFUSED, Outcome.UNSUPPORTED
    ],
    action: Action | None = None,
    detail: dict[str, JsonValue] | None = None,
) -> dict[str, object]:
    """Return the typed shape a tool answers with when it does not succeed."""
    return ToolProblem(
        outcome=outcome, code=code, message=message, action=action, detail=detail
    ).model_dump(mode="json", exclude_none=True)


def ok(**fields: object) -> dict[str, object]:
    """Return the typed shape a tool answers with when it succeeds.

    Raises:
        TypeError: If `fields` names `outcome`; the shape sets it, and every
            caller is code in this repository, so passing one is a programmer error.
    """
    if "outcome" in fields:
        raise TypeError("ok() sets 'outcome' itself; a caller may not pass one")
    return {"outcome": Outcome.OK.value, **fields}


def referred(referral: dict[str, object]) -> dict[str, object]:
    """Return the typed shape a tool answers with when it hands the plan back."""
    return {"outcome": Outcome.REFERRED.value, "referral": referral}
