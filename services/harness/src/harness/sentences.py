"""The sentence a user reads for every reason a plan is not made, declined or cut short.

One table per vocabulary, each total over its enum, so a new code without a
sentence fails `tests/test_sentences.py`:

- `REFERRAL_REASONS`, per `translator.referral.ReferralKind`;
- `COMPILE_REASONS`, per `compiler.compile.CompileIssueCode`, which a
  `requirement_needs_repair` referral names as its `constraint`;
- `DECLINE_REASONS`, per `planner.plan.UnsupportedOperation`, the reason the
  plan card and `harness.graph.plan_done` give for an ask the plan declines;
- `SEAM_REASONS`, per `SeamCode` and `planner.tool_results.ProblemCode`: why
  a revision in code was refused and nothing more, which
  `harness.graph.switch` shows under the choice a switch did not apply to;
- `SEAM_SENTENCES`, per `SeamCode` and `planner.tool_results.ProblemCode`:
  the whole sentence `harness.planner_seam` answers when an entry into the
  planner, or a revision in code (`PlannerSeam.revise`), ends without a
  plan, which `harness.graph.planning` hands the orchestrator to quote and
  `harness.graph.switch` shows on a switch that did not apply, and the
  final message `harness.turn_budget` writes when a request spends a bound;
- `FAILURE_CLASS`, per code a service, the executor, a direct calculation, the
  harness or a refused plan can fail a step with, naming one of the classes
  `FAILURE_SENTENCES` words. `CODE_SENTENCES` words a code the user can act
  on in place of its class, and `CEILING_SENTENCES` an `over_ceiling` by the
  ceiling it names. `failure_sentence` reads them with the refusal's retry
  mode, and `RETRY_MODE_SENTENCES` words a code no class names.

`referral_sentence` states a referral's reason as a whole sentence, the lead of
the question a referral the user decides puts to them.

The first three hold reason clauses, lower case and without a full stop, so a
seam sentence can carry one (`{why}`) and a declined ask reads "not computed
because <reason>". A `SEAM_SENTENCES` entry is a template filled by
`refusal_sentence` from the refusal's own `detail`. The tables are separate
because the vocabularies share values: `requirement_unmeetable` is both a
referral kind and a compile issue code, and `discovery_unavailable` both a
referral kind and a seam code. A seam code has two texts: its `SEAM_SENTENCES`
template, which the orchestrator reads and quotes, and its `FAILURE_CLASS`
sentence, which the Activity log shows.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from compiler.compile import CompileIssueCode
from data_mcp.tool_errors import ToolErrorCode
from executor.classify import ExecutorCode
from planner.plan import MAX_PLAN_ROUNDS, UnsupportedOperation
from planner.tool_results import ProblemCode
from translator.referral import ReferralKind

from harness.scope_choices import option_axis
from harness.tool_contracts import HarnessCode
from harness.wire import RetryMode

logger = logging.getLogger(__name__)


class SeamCode(StrEnum):
    """Why `harness.planner_seam` ended an entry into the planner without a plan,
    or `harness.turn_budget` ended a request before it finished."""

    # The request spent its model-time budget (`harness.turn_budget.TurnBudget`).
    TURN_MODEL_TIME_SPENT = "turn_model_time_spent"
    # The request made as many model calls as its cap allows.
    TURN_MODEL_CALLS_SPENT = "turn_model_calls_spent"
    # The request reached its wall-clock bound, tool time included.
    TURN_WALL_TIME_SPENT = "turn_wall_time_spent"
    # A translation answered with an `internal_failure` referral.
    TRANSLATION_INTERNAL_FAILURE = "translation_internal_failure"
    # A translation failed the way it failed before, on an unchanged plan.
    TRANSLATION_REPEATED = "translation_repeated"
    # A search spent the discovery budget and timed out, or the entry had no
    # discovery time left after a transient referral.
    DISCOVERY_UNAVAILABLE = "discovery_unavailable"
    # One of the planner's own tools raised.
    PLANNER_TOOL_FAILED = "planner_tool_failed"
    # The planner stopped, or repeated a referral with nothing new, before the
    # plan compiled.
    PLAN_NOT_READY = "plan_not_ready"


# The `propose_plan` refusals that record nothing and end the entry.
TERMINAL_PROPOSE_CODES = frozenset(
    {
        ProblemCode.BUDGET_EXHAUSTED,
        ProblemCode.QUESTION_FIXED,
        ProblemCode.NOTHING_TO_PLAN,
        ProblemCode.PRICES_UNSERVED,
    }
)

# Clauses more than one vocabulary gives.
_CANNOT_ANSWER = "the data found cannot answer this as asked"
_NO_BREAKDOWN = "the data found does not break this down by the party asked"
_NO_CALCULATION = "the system has no calculation that does this"
_OVER_COMBINED = "the system cannot calculate further on a comparison or a combined table"
_NOT_COVERED = (
    "the data found does not cover what was asked, such as the period or the places named"
)
_SOURCE_MISMATCH = "the source named holds none of this data, but other sources do"
_PLAN_FAULT = "the drafted plan was malformed, a fault in the system rather than in the request"
_TOOLS_FAULT = (
    "the calculation tools did not match what the system expects, a fault in the system "
    "rather than in the request"
)

REFERRAL_REASONS: Mapping[ReferralKind, str] = {
    ReferralKind.SOURCE_AMBIGUOUS: "more than one dataset fits, and which one to use is not settled",
    ReferralKind.SOURCE_NAME_AMBIGUOUS: (
        "the source named could be more than one publisher or table"
    ),
    ReferralKind.ENTITY_NAME_AMBIGUOUS: "a place or party named could be more than one in the data",
    ReferralKind.NO_DATA: "no data was found for what was asked",
    ReferralKind.UNSUPPORTED_ASK: _CANNOT_ANSWER,
    ReferralKind.REQUIREMENT_UNMEETABLE: _NOT_COVERED,
    ReferralKind.NO_OPERATION_COVERS: _NO_CALCULATION,
    ReferralKind.DISCOVERY_BUDGET_EXHAUSTED: (
        "the request asks for more data at once than one plan can look up"
    ),
    ReferralKind.PINNED_SOURCE_NOT_FOUND: "the dataset the plan named could not be found again",
    ReferralKind.ASK_NOT_UNDERSTOOD: "part of the request could not be matched to data",
    ReferralKind.DISCOVERY_UNAVAILABLE: "the data service did not answer in time",
    ReferralKind.REQUIREMENT_NEEDS_REPAIR: (
        "the plan asked for its data in a form the calculation cannot use"
    ),
    ReferralKind.SURFACE_UNAVAILABLE: "a service the plan needs could not be reached",
    ReferralKind.INTERNAL_FAILURE: "a fault inside the system stopped the plan",
}

COMPILE_REASONS: Mapping[CompileIssueCode, str] = {
    CompileIssueCode.DUPLICATE_ID: _PLAN_FAULT,
    CompileIssueCode.REQUIREMENT_UNKNOWN: _PLAN_FAULT,
    CompileIssueCode.REQUIREMENT_UNSATISFIED: "part of the plan has no calculation that carries it out",
    CompileIssueCode.DUPLICATE_SATISFACTION: _PLAN_FAULT,
    CompileIssueCode.KIND_MISMATCH: _PLAN_FAULT,
    CompileIssueCode.OPERATION_MISMATCH: _PLAN_FAULT,
    CompileIssueCode.DANGLING_INPUT: _PLAN_FAULT,
    CompileIssueCode.INPUT_IS_TERMINAL: _OVER_COMBINED,
    CompileIssueCode.INPUT_SHAPE_UNSUPPORTED: (
        "a calculation was asked of data in a shape it cannot read"
    ),
    CompileIssueCode.RANK_AXIS_REQUIRED: "a ranking needs to know what it ranks",
    CompileIssueCode.SHARE_AXIS_REQUIRED: "a share needs to know what it is a share of",
    CompileIssueCode.SHARE_KEY_UNKNOWN: "a share names a part the data does not hold",
    CompileIssueCode.QUOTIENT_AXIS_REQUIRED: "a division needs to know which values to pair",
    CompileIssueCode.QUOTIENT_KEY_UNKNOWN: "a division names a value the data does not hold",
    CompileIssueCode.KEYS_DISJOINT: "the two sides being paired share no place or party",
    CompileIssueCode.CYCLE: _PLAN_FAULT,
    CompileIssueCode.UNKNOWN_TERMINAL: _PLAN_FAULT,
    CompileIssueCode.OPERATION_UNREACHABLE: _PLAN_FAULT,
    CompileIssueCode.SOURCE_UNSELECTED: "no dataset was chosen for one part of the plan",
    CompileIssueCode.SOURCE_NOT_DISCOVERED_FOR_REQUIREMENT: (
        "the plan used a dataset the search did not find for that part of it"
    ),
    CompileIssueCode.SOURCE_PIN_VIOLATED: "the plan used a different dataset from the one it named",
    CompileIssueCode.SOURCE_UNUSABLE: "the dataset found cannot serve that part of the request",
    CompileIssueCode.SOURCE_WIDER_THAN_REQUIREMENT: (
        "the dataset found covers more than was asked, and the plan did not narrow it"
    ),
    CompileIssueCode.WINDOW_OUTSIDE_PERIOD: (
        "the years asked for fall outside the period the plan covers"
    ),
    CompileIssueCode.ROLLING_NEEDS_OPEN_PERIOD: (
        "a rolling window needs a period that runs to the newest data"
    ),
    CompileIssueCode.ROLLING_POPULATION_UNUSABLE: (
        "the period that follows the newest data cannot be worked out for these series"
    ),
    CompileIssueCode.ROLLING_ANCHOR_BEHIND: (
        "the latest year most of these series report leaves the period asked for no years"
    ),
    CompileIssueCode.ENTITY_REQUIRED: (
        "a calculation needs one country or entity on each side, and the plan named several"
    ),
    CompileIssueCode.UNKNOWN_ENTITY: "the plan names a country or entity the data does not hold",
    CompileIssueCode.ENTITY_AMBIGUOUS: (
        "a country or entity named could be more than one in the data"
    ),
    CompileIssueCode.ENTITY_UNRESOLVED: "a country or entity named could not be matched in the data",
    CompileIssueCode.ENTITY_NOT_ON_FRAME: (
        "a country or entity named is not broken out in the data the calculation reads"
    ),
    CompileIssueCode.DEGENERATE_SIDES: "a comparison set one series against itself",
    CompileIssueCode.SUM_SIDES_OVERLAP: (
        "the parts of a sum overlap, so adding them would count some values twice"
    ),
    CompileIssueCode.SUM_OVERLAP_UNDECIDED: (
        "it is not known whether the parts of a sum overlap, so adding them could count "
        "values twice"
    ),
    CompileIssueCode.FAMILY_UNSUMMABLE: (
        "these figures are different measures that cannot be added into one total"
    ),
    CompileIssueCode.SUM_UNITS_DIFFER: "the parts of a sum are in different units",
    CompileIssueCode.SHARE_UNITS_DIFFER: "a share sets figures in different units against each other",
    CompileIssueCode.FRAME_UNITS_DIFFER: "the figures being combined are in different units",
    CompileIssueCode.FRAME_UNIT_UNBOUND: "one of the figures being combined states no unit",
    CompileIssueCode.FRAME_UNIT_FORMS_DIFFER: (
        "the figures being combined state their units in different forms"
    ),
    CompileIssueCode.FRAME_PRICE_BASES_DIFFER: (
        "the figures being combined are in prices of different base years"
    ),
    CompileIssueCode.FRAME_UNITS_IRRECONCILABLE: (
        "the figures being combined are in units that cannot be converted to one another"
    ),
    CompileIssueCode.PERIODS_DISJOINT: "the figures being compared share no year",
    CompileIssueCode.NO_CAPABILITY: "the calculation service has no tool for one of the steps",
    CompileIssueCode.AMBIGUOUS_CAPABILITY: _TOOLS_FAULT,
    CompileIssueCode.UNKNOWN_TOOL: _TOOLS_FAULT,
    CompileIssueCode.BAD_TOOL_SCHEMA: _TOOLS_FAULT,
    CompileIssueCode.CAPABILITY_SCHEMA_DRIFT: _TOOLS_FAULT,
    CompileIssueCode.CAPABILITY_UNPINNED: _TOOLS_FAULT,
    CompileIssueCode.ARTIFACT_MISMATCH: _PLAN_FAULT,
    CompileIssueCode.BAD_ARGS: _PLAN_FAULT,
}

DECLINE_REASONS: Mapping[UnsupportedOperation, str] = {
    UnsupportedOperation.CALCULATION_OVER_COMBINED: _OVER_COMBINED,
    UnsupportedOperation.NO_OPERATION_COVERS: _NO_CALCULATION,
    UnsupportedOperation.UNSUPPORTED_ASK: _CANNOT_ANSWER,
    UnsupportedOperation.BREAKDOWN_NOT_KEPT: _NO_BREAKDOWN,
}

# A problem the planner's model repairs within its entry, which ends a
# revision in code (`harness.planner_seam.PlannerSeam.revise`), where no model
# is there to repair it.
_NOT_DRAFTED = (
    "The plan could not be drafted{at}, a fault in the system rather than in the request, "
    "so nothing was recorded."
)
_NOT_DRAFTED_CODES = frozenset(
    {
        ProblemCode.PLAN_MISSING,
        ProblemCode.PLAN_NOT_DRAFT,
        ProblemCode.WORKFLOW_MISSING,
        ProblemCode.REFERRAL_PENDING,
        ProblemCode.TRANSITION_REFUSED,
        ProblemCode.REQUIREMENT_UNKNOWN,
        ProblemCode.REQUIREMENT_NOT_A_DATASET,
        ProblemCode.DECISION_UNKNOWN,
        ProblemCode.INVALID_PLAN,
        ProblemCode.INVALID_REFERRAL,
        ProblemCode.INVALID_ARGUMENTS,
        ProblemCode.COMPILE_FAILED,
        ProblemCode.NO_CANDIDATES_RECORDED,
        ProblemCode.SOURCE_NOT_OFFERED,
        ProblemCode.COMBINATION_UNASKED,
        ProblemCode.DECLINE_MISSING,
        ProblemCode.DECLINE_UNEARNED,
        ProblemCode.ASKS_UNACCOUNTED,
        ProblemCode.ASK_ROLE_DROPPED,
        ProblemCode.RESULT_SCOPE_UNSTATED,
        ProblemCode.PERIOD_REDUCTION_UNSTATED,
        ProblemCode.BASIS_UNKNOWN,
        ProblemCode.MEASURE_FIXED_BY_PIN,
    }
)
_DECISION_KEPT = (
    "The plan would have changed a choice already made{at}, so nothing was recorded. Make "
    "that choice again on the plan to change it."
)

# Filled by `refusal_sentence` for the planner and the orchestrator, which
# repair a plan by its requirement ids: `about` names the requirement a refusal
# is about by its id ("the plan" when it names none), `at` is " for <about>"
# or nothing,
# `why` the reason clause of the referral it ended on, `asks` the declined asks,
# `unsupported` why a calculation cannot be made, `reading` the series a
# requirement pins, by its spec_id, `chosen` the prices the user chose, and `rounds`,
# `tool` and `reference` come straight from the refusal's detail.
SEAM_SENTENCES: Mapping[SeamCode | ProblemCode, str] = {
    SeamCode.TURN_MODEL_TIME_SPENT: (
        "This request used all the time it allows the model to work before it finished, so "
        "it stopped here. Your next message starts with a fresh budget."
    ),
    SeamCode.TURN_MODEL_CALLS_SPENT: (
        "This request took as many model steps as it allows before it finished, so it "
        "stopped here. Your next message starts with a fresh budget."
    ),
    SeamCode.TURN_WALL_TIME_SPENT: (
        "This request ran for as long as one request may before it finished, so it stopped "
        "here. Your next message starts with a fresh budget."
    ),
    SeamCode.TRANSLATION_INTERNAL_FAILURE: (
        "Translation failed internally (reference {reference}), so nothing was recorded. The "
        "fault is in the system, not the request, and planning the same request again fails "
        "the same way."
    ),
    SeamCode.TRANSLATION_REPEATED: (
        "The plan failed the same way twice{at}: {why}. Nothing was recorded, and "
        "planning the same request again fails the same way; what that part of the request "
        "asks for has to change."
    ),
    SeamCode.DISCOVERY_UNAVAILABLE: (
        "The data service did not answer while finding data for {about}, so nothing was "
        "recorded. The service is unavailable; try again shortly."
    ),
    SeamCode.PLANNER_TOOL_FAILED: (
        "The planner's {tool} tool failed inside DFO, so no plan was made. The fault is in "
        "the system, not the request."
    ),
    SeamCode.PLAN_NOT_READY: "No plan was made{at}: {why}.",
    ProblemCode.QUESTION_FIXED: (
        "The plan already on record answers this turn's question, so this was not recorded "
        "as a change to it. Ask the other question as its own message."
    ),
    ProblemCode.BUDGET_EXHAUSTED: (
        "This request has used its budget of {rounds} plan drafts, so nothing was recorded. "
        "Your next message starts with a fresh budget."
    ),
    ProblemCode.NOTHING_TO_PLAN: (
        "This cannot be calculated by this system, so nothing was planned: {asks}."
    ),
    ProblemCode.PRICES_UNSERVED: (
        "The plan reads {reading}, and no series this turn's searches found measures it in "
        "{chosen}, which the user chose, so nothing was recorded. Search once for {reading} in "
        "{chosen}."
    ),
    ProblemCode.OPERATION_UNSUPPORTED: (
        "This cannot be calculated as planned{at}, because {unsupported}, so nothing was recorded."
    ),
    ProblemCode.BASES_UNALIGNED: (
        "The figures the plan combines{at} could not be put on one measure basis and provider "
        "scope, so nothing was recorded."
    ),
    ProblemCode.WHOLE_IS_ITS_MEMBER: (
        "A share{at} would divide by one of its own parts in place of their total, so nothing "
        "was recorded."
    ),
    ProblemCode.SHARE_WHOLE_UNDERIVABLE: (
        "A share{at} could not be divided by a total from the same dataset as its part, so "
        "nothing was recorded."
    ),
    ProblemCode.PROVIDERS_UNALIGNED: (
        "The figures the plan combines{at} count different providers' spending, so nothing "
        "was recorded."
    ),
    ProblemCode.ASK_NOT_UNDERSTOOD: (
        "Part of the request could not be matched to data{at}, so nothing was recorded."
    ),
    ProblemCode.SEARCH_REFUSED: "Search refused to look up the data{at}, so nothing was recorded.",
    ProblemCode.SURFACE_UNAVAILABLE: (
        "A service the plan needs could not be reached{at}, so nothing was recorded. Try "
        "again shortly."
    ),
    ProblemCode.RUN_WOULD_STOP: (
        "The newest year the plan counts from could not be read{at}, so nothing was recorded."
    ),
    ProblemCode.PLAN_REJECTED: "The plan was rejected, so nothing was recorded.",
    ProblemCode.APPROVAL_DECLINED: "The plan was not approved, so nothing was run.",
    ProblemCode.USER_DECISION_CONTRADICTED: _DECISION_KEPT,
    ProblemCode.USER_DECISION_DROPPED: _DECISION_KEPT,
    ProblemCode.SHARED_DECISION_CONTRADICTED: _DECISION_KEPT,
    **dict.fromkeys(_NOT_DRAFTED_CODES, _NOT_DRAFTED),
}

# Why a refusal ended a planner entry or a revision in code, as a clause with
# no advice and no account of what was kept, filled as `SEAM_SENTENCES` is,
# with `lead` "for <name>, " or nothing and `mixed` why a calculation cannot
# be made. `refusal_reason` makes it a sentence.
SEAM_REASONS: Mapping[SeamCode | ProblemCode, str] = {
    SeamCode.TURN_MODEL_TIME_SPENT: "this request used all the time it allows the model to work",
    SeamCode.TURN_MODEL_CALLS_SPENT: "this request took as many model steps as it allows",
    SeamCode.TURN_WALL_TIME_SPENT: "this request ran for as long as one request may",
    SeamCode.TRANSLATION_INTERNAL_FAILURE: "a fault inside the system stopped the plan",
    SeamCode.TRANSLATION_REPEATED: "{lead}{why}, and it failed the same way twice",
    SeamCode.DISCOVERY_UNAVAILABLE: "the data service did not answer while finding data for {about}",
    SeamCode.PLANNER_TOOL_FAILED: "the planner's {tool} tool failed inside the system",
    SeamCode.PLAN_NOT_READY: "{lead}{why}",
    ProblemCode.QUESTION_FIXED: "the plan on record answers another question",
    ProblemCode.BUDGET_EXHAUSTED: "this request has used its budget of {rounds} plan drafts",
    ProblemCode.NOTHING_TO_PLAN: "this system cannot calculate what was asked",
    ProblemCode.PRICES_UNSERVED: "{lead}no series found is in the prices chosen",
    ProblemCode.OPERATION_UNSUPPORTED: "{mixed}",
    ProblemCode.BASES_UNALIGNED: (
        "{lead}the figures the plan combines could not be put on one measure basis and "
        "provider scope"
    ),
    ProblemCode.WHOLE_IS_ITS_MEMBER: (
        "{lead}a share would divide by one of its own parts in place of their total"
    ),
    ProblemCode.SHARE_WHOLE_UNDERIVABLE: (
        "{lead}a share could not be divided by a total from the same dataset as its part"
    ),
    ProblemCode.PROVIDERS_UNALIGNED: (
        "{lead}the figures the plan combines count different providers' spending"
    ),
    ProblemCode.ASK_NOT_UNDERSTOOD: "{lead}part of the request could not be matched to data",
    ProblemCode.SEARCH_REFUSED: "Search refused to look up the data{at}",
    ProblemCode.SURFACE_UNAVAILABLE: "a service the plan needs could not be reached",
    ProblemCode.RUN_WOULD_STOP: "the newest year the plan counts from could not be read{at}",
    ProblemCode.PLAN_REJECTED: "the plan was rejected",
    ProblemCode.APPROVAL_DECLINED: "the plan was not approved",
    ProblemCode.USER_DECISION_CONTRADICTED: "the plan would have changed a choice already made{at}",
    ProblemCode.USER_DECISION_DROPPED: "the plan would have changed a choice already made{at}",
    ProblemCode.SHARED_DECISION_CONTRADICTED: (
        "the plan would have changed a choice already made{at}"
    ),
    **dict.fromkeys(
        _NOT_DRAFTED_CODES,
        "the plan could not be drafted{at}, a fault in the system rather than in the request",
    ),
}

# What an input that cannot be counted on its calculation's options would
# then do beside the calculation's other inputs, by the axis of those options.
_REALIGNED: Mapping[str | None, str] = {
    "measure_basis": "be counted on a different measure basis from",
    "provider": "count different providers' spending from",
    None: "be counted on a different measure basis or provider scope from",
}

# Why a calculation cannot be made, for an `operation_unsupported` refusal
# whose input cannot be counted on its calculation's basis or provider scope
# (`detail.basis`), by the axis of the options it was wanted on.
_UNALIGNED: Mapping[str | None, str] = {
    "measure_basis": "its figures would be counted on different measure bases",
    "provider": "its figures would count different providers' spending",
    None: "its figures would be counted on different measure bases or provider scopes",
}

# The sentence for a code no table names: a refusal built outside this
# module's vocabularies.
_UNKNOWN_REFUSAL = "No plan was made for this request, and nothing was recorded."

# The reason clause for a referral that names no kind any table holds, and for
# an entry that ended with no referral at all.
_STOPPED = "the planner stopped before the plan was ready"


def referral_reason(
    kind: object, constraint: object = None, *, source_mismatch: bool = False
) -> str:
    """The reason clause for a referral of `kind`.

    A `requirement_needs_repair` referral reads the compile issue it names as
    `constraint`, which says more than the kind. A `requirement_unmeetable`
    referral marked `source_mismatch` says the source named holds none of the
    data. A kind or constraint outside the vocabularies gives the plain stop
    clause.
    """
    if source_mismatch and kind == ReferralKind.REQUIREMENT_UNMEETABLE:
        return _SOURCE_MISMATCH
    if kind == ReferralKind.REQUIREMENT_NEEDS_REPAIR and isinstance(constraint, str):
        with suppress(ValueError):
            return COMPILE_REASONS[CompileIssueCode(constraint)]
    if isinstance(kind, str):
        with suppress(ValueError):
            return REFERRAL_REASONS[ReferralKind(kind)]
    return _STOPPED


def referral_sentence(
    kind: object, constraint: object = None, *, source_mismatch: bool = False
) -> str:
    """`referral_reason` as a sentence of its own, which leads the question
    `harness.planner_seam` puts to the user for a referral they decide."""
    reason = referral_reason(kind, constraint, source_mismatch=source_mismatch)
    return f"{reason[0].upper()}{reason[1:]}."


@dataclass(frozen=True)
class PlanWords:
    """How the requirements of the plan a refusal is about read
    (`harness.plan_display.plan_words`): `names`, each one's name on the plan
    card, `inputs`, the ids each one reads, and `options`, the scope axes
    ("provider", "measure_basis") whose option a retrieval's name states, by
    requirement id. A requirement with no name is left unnamed, and its id
    reaches no reader."""

    names: Mapping[str, str] = field(default_factory=dict)
    inputs: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    options: Mapping[str, frozenset[str]] = field(default_factory=dict)


_NO_WORDS = PlanWords()


def refusal_sentence(refused: Mapping[str, object]) -> str:
    """The sentence a refusal that ends a planner entry becomes for the
    planner and the orchestrator, naming a requirement by its id, which a
    repair cites.

    `refused` is `{"code", "detail"}`, the shape `harness.planner_seam` stamps
    and `propose_plan` refuses with. A code in no table gives a plain
    sentence that claims nothing about why.
    """
    code = refused.get("code")
    template = SEAM_SENTENCES.get(code, _UNKNOWN_REFUSAL) if isinstance(code, str) else None
    return (template or _UNKNOWN_REFUSAL).format_map(_refusal_fields(refused, None))


def refusal_reason(refused: Mapping[str, object], *, words: PlanWords = _NO_WORDS) -> str:
    """Why `refused` refused, as one sentence for the user (`SEAM_REASONS`),
    naming the plan's requirements by `words`, since an id reads as a
    fault. A code in no table gives a plain clause that claims nothing
    about why."""
    code = refused.get("code")
    template = SEAM_REASONS.get(code, _STOPPED) if isinstance(code, str) else _STOPPED
    reason = template.format_map(_refusal_fields(refused, words))
    first = next(index for index, char in enumerate(reason) if char.isalpha())
    return f"{reason[:first]}{reason[first].upper()}{reason[first + 1 :]}."


def _refusal_fields(refused: Mapping[str, object], words: PlanWords | None) -> dict[str, object]:
    detail = refused.get("detail")
    return _fields(detail if isinstance(detail, Mapping) else {}, words)


def _fields(detail: Mapping[Any, object], words: PlanWords | None) -> dict[str, object]:
    """What a `SEAM_SENTENCES` or `SEAM_REASONS` template reads, from a
    refusal's `detail`: a requirement by its reader's name in `words`, or by
    its id when `words` is `None`."""
    basis = detail.get("basis")
    basis = basis if isinstance(basis, Mapping) else None
    requirement_id = detail.get("requirement_id") or (basis or {}).get("requirement_id")
    about = None
    if isinstance(requirement_id, str) and requirement_id:
        about = (
            f"requirement {requirement_id!r}" if words is None else words.names.get(requirement_id)
        )
    words = words or _NO_WORDS
    declined = detail.get("declined")
    unsupported = _unsupported_reason(detail.get("operation"), basis)
    return {
        "about": about or "the plan",
        "at": f" for {about}" if about else "",
        "lead": f"for {about}, " if about else "",
        "mixed": _realigned(basis, about, words)
        if basis is not None
        else f"{f'for {about}, ' if about else ''}{unsupported}",
        "why": referral_reason(detail.get("kind"), detail.get("constraint")),
        "asks": "; ".join(
            str(decline.get("ask"))
            for decline in (declined if isinstance(declined, list) else [])
            if isinstance(decline, Mapping)
        ),
        "unsupported": unsupported,
        "rounds": detail.get("rounds", MAX_PLAN_ROUNDS),
        "tool": detail.get("tool"),
        "reference": detail.get("reference"),
        "reading": f"the reading {reading}"
        if isinstance(reading := detail.get("reading") or detail.get("part"), str)
        else "one reading",
        "chosen": detail.get("chosen") or "other prices",
    }


def _realigned(basis: Mapping[Any, object], about: str | None, words: PlanWords) -> str:
    """What the input `basis` names would then count beside its
    calculation's other inputs, each by its name on the card: "official
    development assistance would then count different providers' spending
    from health aid"."""
    requirement_id, step_id = basis.get("requirement_id"), basis.get("step_id")
    others = [
        words.names[input_id]
        for input_id in (words.inputs.get(step_id, ()) if isinstance(step_id, str) else ())
        if input_id != requirement_id and input_id in words.names
    ]
    return (
        f"{about or 'one of its figures'} would then {_REALIGNED[_option_axis(basis)]} "
        f"{' and '.join(others) or 'the other figures'}"
    )


def _option_axis(basis: Mapping[Any, object]) -> str | None:
    """The one axis the options `basis` names are on, `None` for none or several."""
    option_ids = basis.get("option_ids")
    axes = {
        option_axis(str(option)) for option in (option_ids if isinstance(option_ids, list) else [])
    }
    axis = next(iter(axes)) if len(axes) == 1 else None
    return axis if axis in _REALIGNED else None


def _unsupported_reason(operation: object, basis: Mapping[Any, object] | None) -> str:
    """Why an `operation_unsupported` refusal's calculation cannot be made.

    One whose input cannot be counted on its calculation's options (`basis`)
    names the axis those options are on; any other reads the declined
    operation it names.
    """
    if basis is not None:
        return _UNALIGNED[_option_axis(basis)]
    if isinstance(operation, str):
        with suppress(ValueError):
            return DECLINE_REASONS[UnsupportedOperation(operation)]
    return _CANNOT_ANSWER


class FailureClass(StrEnum):
    """What a user can do about a failure, which is what its sentence says."""

    TRANSIENT = "transient"
    TOO_BIG = "too_big"
    EMPTY = "empty"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    NOT_COVERED = "not_covered"
    UNITS = "units"
    CANNOT_CALCULATE = "cannot_calculate"
    SOURCE_GONE = "source_gone"
    UNVERIFIED = "unverified"
    CARD_REFUSED = "card_refused"
    EDIT_CONFLICT = "edit_conflict"
    SYSTEM_FAULT = "system_fault"
    UNAVAILABLE = "unavailable"
    NEEDS_DETAIL = "needs_detail"
    BUDGET_SPENT = "budget_spent"
    CANCELLED = "cancelled"
    STEP_FAILED = "step_failed"


@dataclass(frozen=True)
class FailureText:
    """The parts `failure_sentence` joins into one sentence."""

    # What happened, with no full stop.
    cause: str
    # What the user can do, joined to the cause by a comma. Without one the
    # refusal's retry mode gives the advice (`_RETRY_ADVICE`).
    advice: str | None = None
    # `cause` with a `{fill}` slot for the code's `_FILL_KEYS` value.
    filled: str | None = None


# The longest sentence Studio's Activity log shows whole. It cuts a step's
# detail at the first full stop followed by a space, at a line break, and at
# this many characters.
MAX_SENTENCE_CHARS = 139

_TRY_AGAIN = "and trying again in a moment may work"

# One text for each class, whose cause is true of every code in the class.
FAILURE_SENTENCES: Mapping[FailureClass, FailureText] = {
    FailureClass.TRANSIENT: FailureText("A service this step needs did not complete the request"),
    FailureClass.TOO_BIG: FailureText(
        "That covers more than one request or chart can hold",
        "so ask about something narrower",
    ),
    FailureClass.EMPTY: FailureText("Nothing was found for what was asked"),
    FailureClass.NOT_FOUND: FailureText(
        "A place or organisation you named is not in the data", "so check its name or try another"
    ),
    FailureClass.AMBIGUOUS: FailureText(
        "Part of the question could mean more than one thing in the data",
        "so say which one you mean",
    ),
    FailureClass.NOT_COVERED: FailureText(
        "The data does not cover what was asked for, such as the period or the breakdown"
    ),
    FailureClass.UNITS: FailureText("The figures are in units this step cannot work with"),
    FailureClass.CANNOT_CALCULATE: FailureText(
        "The system cannot work that out from the data it found"
    ),
    FailureClass.SOURCE_GONE: FailureText(
        "The data this step needed has changed or is not held",
        "so ask again to fetch it afresh",
    ),
    FailureClass.UNVERIFIED: FailureText(
        "The data found could not be confirmed as what was asked for"
    ),
    FailureClass.CARD_REFUSED: FailureText("The card could not be drawn as asked"),
    FailureClass.EDIT_CONFLICT: FailureText(
        "The card changed while it was being edited",
        "so ask again to edit the latest version",
    ),
    FailureClass.SYSTEM_FAULT: FailureText(
        "A fault inside the system stopped this step", "and your question did not cause it"
    ),
    FailureClass.UNAVAILABLE: FailureText("The system is not set up to do this"),
    FailureClass.NEEDS_DETAIL: FailureText(
        "This needs one more detail, or an agreed plan, before it can run"
    ),
    FailureClass.BUDGET_SPENT: FailureText(
        "This request used all the work allowed for one question",
        "so ask again for a fresh start",
    ),
    FailureClass.CANCELLED: FailureText(
        "The request was stopped before it finished", "so ask again to run it"
    ),
    FailureClass.STEP_FAILED: FailureText("This step could not be completed"),
}

# Every code a user can be told about, under the class that words it. A code
# two vocabularies share (`service_unavailable`, `internal_error`) sits once.
# A code of `SEAM_SENTENCES` sits here too, since its class or its own text
# words the failed step a refused plan shows, with none of the refusal's detail.
_CLASS_CODES: Mapping[FailureClass, frozenset[str]] = {
    FailureClass.TRANSIENT: frozenset(
        {
            "service_unavailable",
            "upstream_error",
            "operation_still_processing",
            "transport_failed",
            "timeout",
            "rate_limited",
            HarnessCode.TRANSIENT_UNAVAILABLE,
            HarnessCode.DEADLINE_EXCEEDED,
            HarnessCode.ROLLING_REPLAY_PENDING,
            HarnessCode.STATE_NOT_SAVED,
            HarnessCode.WRITE_UNCONFIRMED,
            HarnessCode.MEMBER_KIND_UNREAD,
            HarnessCode.STORE_UNAVAILABLE,
            HarnessCode.DATASET_VERSIONS_UNAVAILABLE,
            ExecutorCode.PUBLICATION_UNFINISHED,
            ExecutorCode.RUN_NOT_STARTED,
            ExecutorCode.LEDGER_UNAVAILABLE,
            SeamCode.DISCOVERY_UNAVAILABLE,
            ProblemCode.SURFACE_UNAVAILABLE,
        }
    ),
    FailureClass.TOO_BIG: frozenset(
        {
            "over_ceiling",
            "frame_too_large",
            "resource_exhausted",
            HarnessCode.SERIES_LIMIT_EXCEEDED,
        }
    ),
    FailureClass.EMPTY: frozenset(
        {
            "empty_answer",
            "rows_empty",
            "no_data_at_source",
            HarnessCode.EMPTY_RESULT,
            ProblemCode.ASK_NOT_UNDERSTOOD,
        }
    ),
    FailureClass.NOT_FOUND: frozenset({"entity_not_found", "entity_missing"}),
    FailureClass.AMBIGUOUS: frozenset({"entity_ambiguous", "entity_role_ambiguous"}),
    FailureClass.NOT_COVERED: frozenset(
        {
            "facet_not_available",
            "default_facet_undefined",
            "coverage_insufficient",
            "coverage_gaps",
            "entity_no_series",
            "window_short",
            "scope_window_short",
            "scope_entities_unbound",
            "group_read_as_total",
            "requirement_unmet",
            "comparison_periods_disjoint",
            "growth_gap_year",
            "facet_heterogeneous",
            "cross_frame_facet_mismatch",
            "quotient_of_no_key_matched",
            "share_of_key_unmatched",
            "share_of_grain_mismatch",
            "rolling_anchor_unresolved",
            "open_role_unavailable",
            ExecutorCode.ROLLING_WINDOW_OUTSIDE_DATASET,
            ExecutorCode.ROLLING_ANCHOR_BEHIND_PLAN,
            HarnessCode.RELATIVE_PERIOD_ON_CLOSED_WINDOW,
            HarnessCode.NO_SHARED_PERIOD,
            ProblemCode.RUN_WOULD_STOP,
            ProblemCode.PRICES_UNSERVED,
        }
    ),
    FailureClass.UNITS: frozenset(
        {
            "unit_heterogeneous",
            "cross_frame_unit_mismatch",
            "cross_frame_unit_unbound",
            "unit_mismatch",
            "unit_price_basis_mismatch",
            "unit_price_basis_unverified",
            "unit_unverified",
            "filter_unit_conversion_unsupported",
            "composed_unit_sum_unsupported",
            "quotient_of_side_unit_unsupported",
        }
    ),
    FailureClass.CANNOT_CALCULATE: frozenset(
        {
            "non_finite_result",
            "share_of_zero_denominator",
            "quotient_of_zero_denominator",
            "rollup_not_summable",
            "mixed_time_grain",
            "rolling_grain_unsupported",
            "operation_uncovered",
            "derived_input_ordinal",
            "derived_input_unrecoverable",
            "derived_input_unusable",
            "join_output_collision",
            "grain_ambiguous",
            "spec_not_exportable",
            "step_result_unusable",
            "shape_mismatch",
            "column_unbound",
            ExecutorCode.FAMILY_NOT_SUMMED,
            ProblemCode.NOTHING_TO_PLAN,
            ProblemCode.OPERATION_UNSUPPORTED,
            ProblemCode.BASES_UNALIGNED,
            ProblemCode.WHOLE_IS_ITS_MEMBER,
            ProblemCode.SHARE_WHOLE_UNDERIVABLE,
            ProblemCode.PROVIDERS_UNALIGNED,
        }
    ),
    FailureClass.SOURCE_GONE: frozenset(
        {
            "frame_expired",
            "frame_not_held",
            "frame_unavailable",
            "definition_not_held",
            "node_not_held",
            "upstream_version_changed",
            "stale_revision",
            "surface_changed",
            "source_drift",
            "stat_var_membership_drift",
            "not_found",
            HarnessCode.REDISCOVERY_NEEDED,
            HarnessCode.RESOLVED_REVISION_UNAVAILABLE,
            HarnessCode.REPLAY_UNAVAILABLE,
        }
    ),
    FailureClass.UNVERIFIED: frozenset(
        {
            "data_unverified",
            "identity_unverified",
            "entity_unverified",
            "source_mismatch",
            "source_pin_mismatch",
            "source_substituted",
            "source_unverified",
            "entity_role_mismatch",
            "entity_role_unverified",
            "period_unverified",
            "scope_entities_wider",
            "stat_var_unresolved",
            "member_not_resolvable",
            "open_axis_unresolved",
            "statistics_unavailable",
            "authorisation_mismatch",
            "manifest_outside_authorisation",
            HarnessCode.PROVENANCE_UNAVAILABLE,
        }
    ),
    FailureClass.CARD_REFUSED: frozenset(
        {
            "binding_invalid",
            "constraint_violation",
            "schema_invalid",
            "write_refused",
            "render_target_mismatch",
            "unsupported_schema_version",
            "unsupported_version",
            "unsupported_component_type",
            "release_mismatch",
            HarnessCode.SELECT_INVALID,
            HarnessCode.CHART_NOT_DRAWN,
            HarnessCode.TITLE_NAMES_PERIOD,
            HarnessCode.FIELD_CREATE_ONLY,
            HarnessCode.TITLE_QUOTES_ROLLING_PERIOD,
            HarnessCode.ROLLING_REPLAY_REFUSED,
            HarnessCode.ROLLING_SELECT_NEEDS_NEW_FRAME,
            HarnessCode.FRAME_UNKNOWN,
        }
    ),
    FailureClass.EDIT_CONFLICT: frozenset({"edit_conflict"}),
    FailureClass.SYSTEM_FAULT: frozenset(
        {
            "internal_error",
            "request_invalid",
            "op_spec_invalid",
            "role_ambiguous",
            "join_no_common_key_columns",
            "filter_predicate_type_mismatch",
            "share_of_key_unknown",
            "place_levels_mixed",
            "frame_integrity_failed",
            ExecutorCode.MALFORMED_REFUSAL,
            HarnessCode.UNKNOWN_TOOL,
            HarnessCode.RUN_RECORD_UNREADABLE,
            HarnessCode.RESUME_FAILED,
            HarnessCode.FIGURES_UNREAD,
            HarnessCode.STORE_REFUSED,
            HarnessCode.RECIPE_UNBUILDABLE,
            HarnessCode.RECIPE_NORMAL_FORM_CONFLICT,
            ExecutorCode.PUBLICATION_ERROR,
            SeamCode.TRANSLATION_INTERNAL_FAILURE,
            SeamCode.PLANNER_TOOL_FAILED,
            *_NOT_DRAFTED_CODES,
        }
    ),
    FailureClass.UNAVAILABLE: frozenset(
        {
            "capability_unavailable",
            "skill_not_found",
            "skill_resource_not_found",
            "refresh_not_permitted",
            HarnessCode.ROLLING_REPLAY_NOT_CONFIGURED,
            HarnessCode.TOOL_NOT_CONFIGURED,
            HarnessCode.PUBLISH_UNAVAILABLE,
        }
    ),
    FailureClass.NEEDS_DETAIL: frozenset(
        {
            "not_approved",
            "authorisation_expired",
            HarnessCode.HAND_OFF_PENDING,
            ProblemCode.PLAN_REJECTED,
            ProblemCode.APPROVAL_DECLINED,
        }
    ),
    FailureClass.BUDGET_SPENT: frozenset(
        {
            HarnessCode.TOOL_BUDGET_SPENT,
            SeamCode.TURN_MODEL_TIME_SPENT,
            SeamCode.TURN_MODEL_CALLS_SPENT,
            SeamCode.TURN_WALL_TIME_SPENT,
            ProblemCode.BUDGET_EXHAUSTED,
        }
    ),
    FailureClass.CANCELLED: frozenset(
        {
            "run_abandoned",
            "cancelled",
            ExecutorCode.PUBLICATION_ABANDONED,
            ExecutorCode.COMPUTATION_ABANDONED,
        }
    ),
    FailureClass.STEP_FAILED: frozenset(
        {
            "step_refused",
            "unit_matches",
            HarnessCode.NO_SEARCH_PARTS,
            HarnessCode.TOOL_ARGUMENTS_INVALID,
            HarnessCode.UNSPECIFIED,
            HarnessCode.CARD_NOT_FOUND,
            HarnessCode.RAISED,
            SeamCode.TRANSLATION_REPEATED,
            SeamCode.PLAN_NOT_READY,
            ProblemCode.QUESTION_FIXED,
            ProblemCode.SEARCH_REFUSED,
            ProblemCode.USER_DECISION_CONTRADICTED,
            ProblemCode.USER_DECISION_DROPPED,
            ProblemCode.SHARED_DECISION_CONTRADICTED,
        }
    ),
}

FAILURE_CLASS: Mapping[str, FailureClass] = {
    str(code): failure_class for failure_class, codes in _CLASS_CODES.items() for code in codes
}

# A card refused for something the model wrote, such as its select. The user
# has nothing to change.
_MODEL_MAY_CORRECT = FailureText(
    FAILURE_SENTENCES[FailureClass.CARD_REFUSED].cause, "and asking for it again may work"
)

# The text of a code the user can act on, read in place of its class's.
CODE_SENTENCES: Mapping[str, FailureText] = {
    "operation_still_processing": FailureText(
        "The data is still being prepared", "and asking again in a moment may pick it up"
    ),
    "entity_ambiguous": FailureText(
        "A place or organisation you named could be more than one in the data",
        "so say which one you mean",
    ),
    "facet_not_available": FailureText(
        "The data is not broken down the way that was asked",
        "so ask for a breakdown it holds",
    ),
    "member_not_resolvable": FailureText(
        "The data holds indicators the system cannot confirm", "so ask for it from another source"
    ),
    "growth_gap_year": FailureText(
        "The data has no figure in the first or the last year of that period",
        "so name years it holds",
    ),
    "comparison_periods_disjoint": FailureText(
        "The figures being compared share no period", "so name a period both cover"
    ),
    "binding_invalid": FailureText(
        "The data behind this card does not hold a value it was asked to show",
        filled="The data behind this card holds no figures for {fill}",
    ),
    ExecutorCode.ROLLING_WINDOW_OUTSIDE_DATASET: FailureText(
        "The period asked for starts before the data does", "so ask for a later start"
    ),
    ExecutorCode.ROLLING_ANCHOR_BEHIND_PLAN: FailureText(
        "Fewer than half of these series now report the year the plan counted from",
        "so ask again to plan from the latest year they report",
    ),
    HarnessCode.EMPTY_RESULT: FailureText(
        "The data holds no figures for the places and period asked",
        "so try other places or another period",
    ),
    HarnessCode.DEADLINE_EXCEEDED: FailureText(
        "Fetching the data took longer than one request allows",
        "so ask about fewer places or a shorter period",
    ),
    HarnessCode.SERIES_LIMIT_EXCEEDED: FailureText(
        "That chart would draw more series than one chart can hold",
        "so ask for fewer of them",
    ),
    HarnessCode.NO_SHARED_PERIOD: FailureText(
        "No one period is held by at least half of these series",
        "so name the period you want",
    ),
    HarnessCode.FIELD_CREATE_ONLY: FailureText(
        "That part of a card is set when the card is made", "so ask for a new card instead"
    ),
    HarnessCode.SELECT_INVALID: _MODEL_MAY_CORRECT,
    HarnessCode.TITLE_NAMES_PERIOD: _MODEL_MAY_CORRECT,
    HarnessCode.TITLE_QUOTES_ROLLING_PERIOD: FailureText(
        "The title names a period this card will move past as newer data arrives",
        "so ask for a title without it",
        filled="The title names {fill}, which this card will move past as newer data arrives",
    ),
    HarnessCode.ROLLING_REPLAY_PENDING: FailureText(
        "The latest period of the series on this card could not be worked out in time",
        "and asking again shortly may work",
    ),
    HarnessCode.WRITE_UNCONFIRMED: FailureText(
        "The request to draw this card got no answer and the card may already be on the canvas",
        "so look there before asking again",
    ),
    HarnessCode.ROLLING_SELECT_NEEDS_NEW_FRAME: FailureText(
        "The series now chosen have another latest period than this card was made for",
        "so ask for a new card",
    ),
    # The model writes a tool call's arguments, and a later try may write
    # them well.
    HarnessCode.TOOL_ARGUMENTS_INVALID: FailureText(
        "This step was written in a form the system could not run", "and asking again may work"
    ),
    HarnessCode.NO_SEARCH_PARTS: FailureText(
        "The question could not be turned into a search", "so try wording it another way"
    ),
    # The model names the frame and the cards, as it writes the select.
    HarnessCode.FRAME_UNKNOWN: _MODEL_MAY_CORRECT,
    HarnessCode.CARD_NOT_FOUND: FailureText(
        "No card matched the one that was to be taken off", "and asking for it again may work"
    ),
    HarnessCode.HAND_OFF_PENDING: FailureText(
        "This request is already waiting on your answer or on a plan"
    ),
    SeamCode.TRANSLATION_REPEATED: FailureText(
        "The plan for this failed the same way twice", "so ask for it another way"
    ),
    SeamCode.PLAN_NOT_READY: FailureText("A plan for what was asked could not be completed"),
    ProblemCode.QUESTION_FIXED: FailureText(
        "The plan on record already answers this question",
        "so ask the other question as its own message",
    ),
    ProblemCode.BUDGET_EXHAUSTED: FailureText(
        "This request used all the plan drafts it allows", "so ask again for a fresh start"
    ),
    ProblemCode.NOTHING_TO_PLAN: FailureText("The system has no calculation for what was asked"),
    # The orchestrator searches for the prices next, and plans again when it may.
    ProblemCode.PRICES_UNSERVED: FailureText(
        "The data found so far is not in the prices you chose", "so a search for them runs next"
    ),
}

_NARROW_SCOPE = FailureText(
    "That asks for more data than one request can fetch",
    "so ask about fewer places or a shorter period",
)

# The text of an `over_ceiling` refusal, by the `ceiling_kind` its `fix` names.
# Fewer places or a shorter period bring a request under a ceiling on its rows,
# bytes or upstream requests. The other two count the series the definition
# itself covers, which only a narrower question changes. An `over_ceiling` that
# names no kind here reads its class's text.
CEILING_SENTENCES: Mapping[str, FailureText] = {
    "rows": _NARROW_SCOPE,
    "bytes": _NARROW_SCOPE,
    "requests": _NARROW_SCOPE,
    "member_cardinality": FailureText(
        "That covers more series than one request can serve",
        "so ask about a narrower topic, such as one sector",
    ),
    "candidate_window": FailureText(
        "That covers too broad a family of series to list in one request",
        "so name the breakdown you want",
    ),
}

# The member of a refusal's `fix` that fills a code's `filled` cause.
_FILL_KEYS: Mapping[str, str] = {
    "binding_invalid": "value",
    HarnessCode.TITLE_QUOTES_ROLLING_PERIOD: "period",
}

# A fill-in the Activity log would cut the sentence at, or an identifier (a
# name joined by an underscore or a slash, or a run of hexadecimal digits).
_UNSAYABLE_FILL = re.compile(r"[.\n\r_/\\{}<>=:]|[0-9a-f]{8,}", re.IGNORECASE)

# The advice a refusal's retry mode gives a text that carries none. A refusal
# that is not to be retried gives none.
_RETRY_ADVICE: Mapping[RetryMode, str | None] = {
    RetryMode.RETRY_AFTER_DELAY: _TRY_AGAIN,
    RetryMode.REFRESH_THEN_RETRY: "so ask again to work from the latest data",
    RetryMode.CORRECT_AND_RETRY: "so change what you ask for and try again",
    RetryMode.DO_NOT_RETRY: None,
}

_NO_CAUSE = FAILURE_SENTENCES[FailureClass.STEP_FAILED].cause

# The sentence for a code in no class, by the next step its retry mode supports.
RETRY_MODE_SENTENCES: Mapping[RetryMode, str] = {
    RetryMode.RETRY_AFTER_DELAY: f"{_NO_CAUSE}, {_TRY_AGAIN}.",
    RetryMode.REFRESH_THEN_RETRY: (f"{_NO_CAUSE}, {_RETRY_ADVICE[RetryMode.REFRESH_THEN_RETRY]}."),
    RetryMode.CORRECT_AND_RETRY: f"{_NO_CAUSE}, {_RETRY_ADVICE[RetryMode.CORRECT_AND_RETRY]}.",
    RetryMode.DO_NOT_RETRY: f"{_NO_CAUSE}, and asking again the same way will not change that.",
}


def failure_sentence(
    code: object, retry_mode: object = None, fix: Mapping[str, object] | None = None
) -> str:
    """The sentence a user reads for a failure `code`.

    A code with a text of its own reads it, an `over_ceiling` reads the text
    of the ceiling its `fix` names, and any other reads its class's. The
    advice is the text's own, or else the retry mode's. A code in no class
    reads the sentence of its `retry_mode`, which names no cause, and is
    logged.

    Args:
        code: The failure's code.
        retry_mode: The refusal's `retry_mode`, when it states one.
        fix: The refusal's `fix`. Its `ceiling_kind` chooses the text of an
            `over_ceiling`, and the one member a code's text names fills its
            cause when the sentence still fits the Activity log with it.
    """
    known = isinstance(retry_mode, str) and retry_mode in set(RetryMode)
    mode = RetryMode(retry_mode) if known else None
    failure_class = FAILURE_CLASS.get(code) if isinstance(code, str) else None
    if failure_class is None:
        logger.warning("no failure sentence for code %r (retry_mode %r)", code, retry_mode)
        return RETRY_MODE_SENTENCES[mode] if mode else f"{_NO_CAUSE}."
    text = (
        CODE_SENTENCES.get(str(code))
        or _ceiling_text(str(code), fix)
        or FAILURE_SENTENCES[failure_class]
    )
    advice = text.advice or (_RETRY_ADVICE[mode] if mode else None)
    tail = f", {advice}." if advice else "."
    fill = _fill(str(code), fix)
    if text.filled and fill:
        filled = text.filled.format(fill=fill) + tail
        if len(filled) <= MAX_SENTENCE_CHARS:
            return filled
    return text.cause + tail


def _ceiling_text(code: str, fix: Mapping[str, object] | None) -> FailureText | None:
    """The text of the ceiling an `over_ceiling` refusal's `fix` names, when it names one."""
    kind = (fix or {}).get("ceiling_kind")
    if code != ToolErrorCode.OVER_CEILING or not isinstance(kind, str):
        return None
    return CEILING_SENTENCES.get(kind)


def _fill(code: str, fix: Mapping[str, object] | None) -> str | None:
    """The value of `fix` that `code`'s text takes, when it is one a sentence can say."""
    value = (fix or {}).get(_FILL_KEYS.get(code, ""))
    if isinstance(value, bool) or not isinstance(value, str | int):
        return None
    said = str(value).strip()
    return said if said and not _UNSAYABLE_FILL.search(said) else None
