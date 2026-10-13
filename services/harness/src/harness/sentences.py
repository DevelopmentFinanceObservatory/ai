"""The sentence a user reads for every reason a plan is not made, declined or cut short.

One table per vocabulary, each total over its enum, so a new code without a
sentence fails `tests/test_sentences.py`:

- `REFERRAL_REASONS`, per `translator.referral.ReferralKind`;
- `COMPILE_REASONS`, per `compiler.compile.CompileIssueCode`, which a
  `requirement_needs_repair` referral names as its `constraint`;
- `DECLINE_REASONS`, per `planner.plan.UnsupportedOperation`, the reason the
  plan card and `harness.graph.plan_done` give for an ask the plan declines;
- `SEAM_SENTENCES`, per `SeamCode` and each `planner.tool_results.ProblemCode`
  in `TERMINAL_PROPOSE_CODES`: the whole sentence `harness.planner_seam`
  answers when an entry into the planner ends without a plan, which
  `harness.graph.planning` hands the orchestrator to quote, and the final
  message `harness.turn_budget` writes when a request spends a bound.

`referral_sentence` states a referral's reason as a whole sentence, the lead of
the question a referral the user decides puts to them.

The first three hold reason clauses, lower case and without a full stop, so a
seam sentence can carry one (`{why}`) and a declined ask reads "not computed
because <reason>". A `SEAM_SENTENCES` entry is a template filled by
`refusal_sentence` from the refusal's own `detail`. The tables are separate
because the vocabularies share values: `requirement_unmeetable` is both a
referral kind and a compile issue code, and `discovery_unavailable` both a
referral kind and a seam code.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from enum import StrEnum
from typing import Any

from compiler.compile import CompileIssueCode
from planner.plan import MAX_PLAN_ROUNDS, UnsupportedOperation
from planner.tool_results import ProblemCode
from translator.referral import ReferralKind


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
    {ProblemCode.BUDGET_EXHAUSTED, ProblemCode.QUESTION_FIXED, ProblemCode.NOTHING_TO_PLAN}
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
    ReferralKind.SURFACE_UNAVAILABLE: "the calculation service could not be reached",
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
    CompileIssueCode.LEVEL_FILTER_ON_FLOW_AXIS: (
        "the dataset lists the other side of each flow without a place level, so it cannot "
        "be kept to countries or regions"
    ),
    CompileIssueCode.RANK_AXIS_REQUIRED: "a ranking needs to know what it ranks",
    CompileIssueCode.SHARE_AXIS_REQUIRED: "a share needs to know what it is a share of",
    CompileIssueCode.SHARE_KEY_UNKNOWN: "a share names a part the data does not hold",
    CompileIssueCode.QUOTIENT_AXIS_REQUIRED: "a division needs to know which values to pair",
    CompileIssueCode.QUOTIENT_KEY_UNKNOWN: "a division names a value the data does not hold",
    CompileIssueCode.KEYS_DISJOINT: "the two sides of a division share nothing to pair",
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
    CompileIssueCode.REQUIREMENT_UNMEETABLE: _NOT_COVERED,
    CompileIssueCode.WINDOW_OUTSIDE_PERIOD: (
        "the years asked for fall outside the period the plan covers"
    ),
    CompileIssueCode.ROLLING_NEEDS_OPEN_PERIOD: (
        "a rolling window needs a period that runs to the newest data"
    ),
    CompileIssueCode.ROLLING_POPULATION_UNUSABLE: (
        "the period that follows the newest data cannot be worked out for these series"
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
    CompileIssueCode.JOIN_NO_COMMON_KEY_COLUMNS: "the tables being joined share nothing to join on",
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

# Filled by `refusal_sentence`: `about` names the requirement a refusal is
# about ("the plan" when it names none), `at` is " for <about>" or nothing,
# `why` the reason clause of the referral it ended on, `asks` the declined asks,
# and `rounds`, `tool` and `reference` come straight from the refusal's detail.
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


def refusal_sentence(refused: Mapping[str, object]) -> str:
    """The sentence a refusal that ends a planner entry becomes for the user.

    `refused` is `{"code", "detail"}`, the shape `harness.planner_seam` stamps
    and `propose_plan` refuses with. A code in no table gives a plain sentence
    that claims nothing about why.
    """
    code = refused.get("code")
    template = SEAM_SENTENCES.get(code, _UNKNOWN_REFUSAL) if isinstance(code, str) else None
    detail = refused.get("detail")
    fields = _fields(detail) if isinstance(detail, Mapping) else _fields({})
    return (template or _UNKNOWN_REFUSAL).format_map(fields)


def _fields(detail: Mapping[Any, object]) -> dict[str, object]:
    """What a `SEAM_SENTENCES` template reads, from a refusal's `detail`."""
    requirement_id = detail.get("requirement_id")
    about = f"requirement {requirement_id!r}" if requirement_id else None
    declined = detail.get("declined")
    return {
        "about": about or "the plan",
        "at": f" for {about}" if about else "",
        "why": referral_reason(detail.get("kind"), detail.get("constraint")),
        "asks": "; ".join(
            str(decline.get("ask"))
            for decline in (declined if isinstance(declined, list) else [])
            if isinstance(decline, Mapping)
        ),
        "rounds": detail.get("rounds", MAX_PLAN_ROUNDS),
        "tool": detail.get("tool"),
        "reference": detail.get("reference"),
    }
