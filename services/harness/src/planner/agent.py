"""The planner agent: plan-lifecycle tool functions plus `create_planner_agent`.

`submit_plan` approves nothing and gates nothing itself: it runs
`_execution_blocked`'s checks (a rejected plan, a pending referral, an
uncompiled workflow, a stale version) and, past those, answers
`_approval_summary(plan)` (the version and revision that tie the answer to one
content hash, and the compiler's warnings, the one thing on it a caller cannot
read off the plan record), declaring the plan ready for the user's answer.
`harness.studio_translate` builds the plan card the user reads from the plan
record itself, through `harness.plan_display`, so nothing here restates a
decision, a requirement, a source or a step. Whatever runs this agent is what
actually withholds execution until that answer arrives; for the served path
that is the harness's own graph, keyed on the plan's revision rather than on
an ADK tool confirmation this agent would have to carry across its own
nested `Runner`.

The harness decides the depth policy (how much dialogue a question earns)
before the planner is invoked.

Built with a tool surface, the planner also carries `planner.skills`'
`get_skill` and `get_skill_file`, and its instruction carries `_SKILLS`, the
paragraph naming them. Built without one, it carries neither, so the
standalone eval agents run without a live server.

The planner settles methodology and states what evidence an answer rests on,
then stops. A proposed plan is a `DRAFT` carrying `decisions` and
`evidence_requirements`; finding the data, proposing the operations and
compiling the workflow belong to `translator.tool`'s `translate`, a function
tool the planner calls before the user ever sees the plan --
`create_planner_agent` takes that tool, and `harness.agents` builds it. So no
tool name, column or operation appears here, `PENDING_APPROVAL` is a status
only translation writes, and `submit_plan` requires a plan that has been
through that step.

Every tool below answers with `planner.tool_results`' typed shape -- `ok`,
`problem`, with a stable `code` and, where the next step is known, an
`action` -- rather than an unstructured `{"error": ...}` a caller has to
parse. When translation cannot carry a plan out, `translate` answers with
the `Referral` itself: this agent's instruction reads it straight off that
result, with nothing here to fetch it back a second time.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any

from compiler.surface import ToolSurface
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_core import to_jsonable_python
from translator.referral import ReferralKind, clear_referral, load_referral
from translator.tool import TRANSLATE_TOOL_NAME

from planner.declaration import PlanTool
from planner.parties import PARTY_ROLES, party_role
from planner.plan import (
    MAX_DECLINES,
    MAX_PLAN_ASKS,
    MAX_PLAN_DECISIONS,
    MAX_PLAN_REQUIREMENTS,
    MAX_PLAN_ROUNDS,
    AnswerMode,
    AskAnswer,
    AsksUnaccountedError,
    DatasetRequirement,
    Decision,
    DecisionDraft,
    DecisionFixes,
    Decline,
    EvidenceRequirement,
    Plan,
    PlanStatus,
    ResolvedBy,
    SeededSource,
    UnsettledFixError,
    UnsupportedPlanError,
    WindowedPin,
    cut_user_text,
    is_search_pin,
    load_asks,
    load_plan,
    load_plan_rounds,
    load_unsupported,
    names_dropped_seed,
    period_window,
    pin_sources,
    plan_revision,
    record_unsupported,
    recorded_identity,
    save_plan,
    save_plan_rounds,
    save_seeded_sources,
    save_windowed_pins,
    search_pin_slot,
    seeded_sources,
    settle_decisions,
    windowed_copy,
    windowed_pin,
    windowed_pins,
)
from planner.records import Identifier
from planner.settings import AGENT_THINKING_CONFIG, drop_thought_parts
from planner.skills import build_skill_tools
from planner.tool_results import Action, Outcome, ProblemCode, ok, problem

if TYPE_CHECKING:
    from harness.prefix_cache import PrefixCache

AGENT_NAME = "dfo_planner"


class _Patch(BaseModel):
    """The lists one `propose_plan` call states, checked before they reach a `Plan`.

    ADK hands `propose_plan` raw values wherever it could not convert them, and
    builds the tool's declaration from the signature with every `Annotated`
    limit stripped, so the count and shape of each list is enforced here. A
    removal names a slot or an id, so it is held to `Identifier`; each list
    holds no more items than a plan does.
    """

    decisions: Annotated[list[DecisionDraft], Field(max_length=MAX_PLAN_DECISIONS)]
    evidence_requirements: Annotated[
        list[EvidenceRequirement], Field(max_length=MAX_PLAN_REQUIREMENTS)
    ]
    removed_decisions: Annotated[list[Identifier], Field(max_length=MAX_PLAN_DECISIONS)]
    removed_evidence_requirements: Annotated[
        list[Identifier], Field(max_length=MAX_PLAN_REQUIREMENTS)
    ]
    declined: Annotated[list[Decline], Field(max_length=MAX_DECLINES)]
    answered: Annotated[list[AskAnswer], Field(max_length=MAX_PLAN_ASKS)]

    @model_validator(mode="after")
    def _roles_are_party_words(self) -> _Patch:
        """Reject an entity `role` outside `PARTY_ROLES`, naming the word it stands for when one does."""
        refused = []
        for requirement in self.evidence_requirements:
            if not isinstance(requirement, DatasetRequirement):
                continue
            for entity in requirement.entities:
                if entity.role is None or entity.role in PARTY_ROLES:
                    continue
                meant = party_role(entity.role)
                refused.append(
                    f"{requirement.id}/{entity.name}: {entity.role!r}"
                    + (f", write {meant!r}" if meant is not None else "")
                )
        if refused:
            raise ValueError(
                f"an entity's role is one of {', '.join(PARTY_ROLES)}, or unset for a place "
                f"the measure is about; {'; '.join(refused)}"
            )
        return self


_PLANNING = (
    "You are the Planner Agent for the Development Finance Observatory. "
    "Given a development finance question, draft a methodological plan: "
    "state each real decision (measure, scope, time window when "
    "the search results do not settle it, and so on), the "
    "alternatives you considered, and a one-line reason for your choice. "
    "A decision is recorded only where you genuinely weighed more than one "
    "alternative; a value the question already states outright -- the "
    "entity it names, the measure it names -- is a fact for a requirement's "
    "own fields to carry, and needs no `Decision` entry at all. The search "
    "results in your first message, a resolve envelope or numbered search "
    "readings, are what the conversation's searches established, as far as "
    "that message reports them; a reading from an earlier turn counts as "
    "much as one from this turn. With no search results, draft from the "
    "question alone. A "
    "`time_window_defaulted` warning means the question gave no time text; "
    "the resolver has already settled the answer's coverage, so that is not "
    "a decision -- never weigh candidate ranges, never call "
    "`adk_request_input` over it, and never invent a specific range from "
    "memory. The period follows observed coverage: when a result carries a "
    "`window`, state the dataset requirement's `period` from it verbatim -- "
    "`start` and `end` exactly as given -- and cite no decision that fixes "
    "it. An end year earlier than the latest year with data, or a start the "
    "question did not ask for, is a `Decision` whose `fixes` names "
    "`period` and whose rationale says why, such as a latest year whose "
    "counts fall far below the years before it. Never call a year the latest "
    "available when a result shows data for a later one. When no result "
    "carries a window, the period is a decision like any other. Source is "
    "different: when a result's own `spec_id` (a definite or chosen reading, "
    "or the envelope's `spec_id`) answers a dataset requirement, pin it as "
    "that requirement's `source_spec_id`, verbatim, and cite no decision "
    "for the pin itself -- the plan records why on its own. A candidate's "
    "`spec_id` is pinned only with a `Decision` of your own whose "
    "`fixes` names `source_spec_id`. Otherwise the source is a decision "
    "like any other. "
    "Ask the user directly only when a choice is "
    "genuinely ambiguous, settling other defaults the way a domain expert "
    "would. "
    "State what the answer rests on as evidence requirements: one dataset "
    "requirement per measure and source the answer needs, naming every "
    "entity the answer needs for that measure, and one derived requirement "
    "per calculation the answer needs. "
    "When a flow source spans a party the question "
    "leaves unnamed (aid from named donors to every recipient) and the "
    "measure is an additive amount, add an `aggregate` with "
    "`reduction='sum'` grouped by `entity` and period before any "
    "`growth`, `rank` or `comparison`: a group by `entity` keeps the parties "
    "the requirement names apart and sums its `across` party, so each named "
    "entity has one value per year. Grouped by period alone, it totals every "
    "party into one series per year, which a `share_of` reads as a whole "
    "side. Over a requirement that names no entity, a group by `entity` "
    "keeps nothing and is refused: group by period, or `rank` with a "
    "`reduction` to set each member's total beside the others. "
    "A rate, index or per-person figure is never summed. "
    'A party the question names no member of, such as the donors of "top '
    'donors", goes in `across` (see `across`). '
    "A result's open axes "
    "(`open_entity_axes`) are the columns left open: `origin` is the giving party "
    "(an entity `role` of donor, creditor or exporter), `destination` the receiving "
    "party (recipient, debtor or importer), `observationAbout` the place. "
    "A requirement whose `across` is the place, such as `'country'`, names "
    "no place, so its retrieval holds every level (see `PlaceType`). Before "
    "a `rank`, an `aggregate` across places or a `share_of` reads it, keep "
    "one level with a `filter` reading that requirement directly, whose "
    "predicate is `place_level`. Take the level the "
    'question names ("top 10 countries"). When it names none and reads as '
    'a comparison of countries ("top 10 by GDP"), choose `Country` in a '
    "`Decision` resolved by `planner_default` whose `fixes` names "
    "`predicates`. "
    'When its word covers several levels ("by region" is '
    "continents or subdivisions), ask the user. A question "
    "naming its places names them in `entities` and needs no level. State "
    "`change='absolute'` on a `growth` "
    "whenever its input is itself a rank position, since a rank moves by "
    "places, not by a percentage. Several entities of one measure are one "
    "dataset requirement, which under `answer_mode='series'` is itself the "
    "answer. Reach for "
    "`comparison` only when the question asks for a difference, for "
    "`share_of` only for a ratio whose two sources declare one unit or both "
    "declare none, and for `join` only across different measures, since "
    "none of them keeps the separate series a chart of one measure over "
    "several entities shows. "
    "A family of StatVars is read summed or separately: see `stat_vars`. "
    "When the question asks which members of a classification hold the most, "
    'such as "which sectors received the most aid", state `breakdown` on the '
    "dataset requirement, at `level` 0 unless the question names a finer "
    "one; then an `aggregate` with `reduction='sum'` and "
    "`group_by` period and breakdown, which totals the parties the question "
    "leaves open within each sector, and a `rank` with `axis='sector'` and a "
    "`limit`. To "
    "set that ranking beside another year, widen the period to cover both: "
    "state `per_period` on the rank for each year's ranking side by side, or "
    "read a `growth` over the aggregate between the two years for each "
    "sector's change. A "
    "derived requirement's figures are what a question about a value, a "
    "growth, a share or a rank asks for. When the answer is "
    "one total of two different sources, such as bilateral aid plus imputed "
    "multilateral aid, state a `sum_of`, naming each side's `entity` when "
    "its requirement names several on one side of the flow, such as two "
    "donors. Whether the two sources add up without "
    "counting a flow twice is a methodological fact, not an assumption: "
    "read it in the skill the sources' search results name, and cite a "
    "`Decision` recording it, with the skill in its `rationale`. A sum "
    "states the same `unit` on both inputs, or on neither. To total several "
    "entities of one series, `aggregate` it. A total over a span of years, "
    'such as "the total between 2010 and the latest", is one figure, never '
    "a figure per year: an `aggregate` with `reduction='sum'` grouped by "
    "`entity` alone sums the years, as a rank's `reduction` does for a "
    "ranking of each entity's total. A total of two sources over a span "
    "states the question's own period on both dataset requirements, adds "
    "the two yearly series with a `sum_of` first, and then sums the years "
    "of that one series with an `aggregate` grouped by `entity`; the "
    "`sum_of` adds the years both sources report and names any year one "
    "lacks. When a part of the question earns a decline (see `declined`), "
    "such as a calculation over a comparison or a join, state it in "
    "`declined` and plan what can still be answered, or propose `declined` "
    "alone when nothing can. "
    "Decide the "
    "methodology and leave the mechanics alone: which data and which "
    "operations carry your decisions out is worked out separately, after "
    "you propose, so name no tool. "
    "A rank's `among='population'` is a claim over "
    "entities nobody named, so cite a `Decision` whose `fixes` names "
    "`among`, as a `share_of` stating `denominator_scope='pooled'` cites "
    "one whose `fixes` names `denominator_scope`. "
    "State `per_period` when the question asks how a position "
    "changed over time rather than who is first; a rank over it is an "
    "ordinal series, not a value, so growing it needs a `filter` naming "
    "the one entity the question is about, placed between the rank and "
    "the growth -- the series carries the whole population, and a growth "
    "over everyone's position at once has no defined answer. State a rank's "
    "`limit` when the question asks for a top or bottom N, or for the "
    'members that hold the most, as "top donors" or "which sectors '
    'received the most" does, with a `Decision` fixing the N when the '
    "question names none. A growth or a per-entity `share_of` goes "
    "before a rank with `per_period` and a `limit`, and a top N by each "
    "entity's total is one rank stating the `reduction` and the `limit` "
    "with `per_period` unset. No rank reads another rank's output: put the "
    "`limit` on the one rank that reads the measure. "
)

# What each `propose_plan` call states about which plan it means. Held apart
# from `_PLANNING` because it is the one paragraph `question_fixed` rewrites:
# standalone and `harness_pair_v1` keep `_VERSIONING_FREE`, where a restart is
# a normal way to begin a second plan; the harness graph's own seam builds with
# `question_fixed=True`, where the turn's question is the graph's, not the
# planner's, to change. What each argument does is on the argument.
_VERSIONING_FREE = (
    "Call `propose_plan` once with the complete first draft, and again "
    "whenever the user asks for a change. Once a plan is recorded, decide "
    "deliberately whether the ask patches it (`amends_version`) or is a "
    "different question that needs a plan of its own (`starts_new_plan`); "
    "never let the wording of `question` stand in for the choice. "
)

_VERSIONING_FIXED = (
    "Call `propose_plan` once with the complete first draft, and again, "
    "patching it through `amends_version`, whenever the user asks for a "
    "change. This turn's question is fixed. When the user's reply asks a "
    "different question, call `propose_plan` with `starts_new_plan=true` "
    "and stop there: the harness tells the user to send that question as "
    "its own message, so nothing further is needed from you once you have "
    "made it. "
)

# How to read skills, when the skill tools are built. Held apart because the
# planner is also built without them, and an instruction naming a tool the
# agent lacks sends the model at one ADK raises on.
_SKILLS = (
    "The search results name the skills that apply to them by id, and an "
    "envelope's `skills` describes each. When a skill bears on the question, "
    "call `get_skill` with its id before you draft, and let what it says "
    "shape your decisions; read a file it links with `get_skill_file`. A "
    "question about DFO itself (what it covers, where its data comes from, "
    "how it works) starts with `get_skill`, though no search result names "
    "that skill. When a skill tool refuses, `capability_unavailable` "
    "included, draft the plan without that skill. "
)

# What to do once a plan is recorded, when the `translate` tool is attached to
# carry it out. Held apart because the planner is also built standalone -- its
# own eval set scores it that way -- and a standalone planner that is told to
# call `translate` calls a tool that is not there. ADK raises on an unknown
# tool name rather than answering the model, so the run dies mid-turn.
_TRANSLATING = (
    "Once `propose_plan` has recorded the plan, call `translate`. Once a "
    "referral is settled, by your own decision or the "
    "user's answer, propose with the `amends_version` the last "
    "`propose_plan` call answered with, so the patch lands on that plan, and "
    "call `translate` again. When `translate` answers that the plan "
    "compiled, call `submit_plan`. Otherwise it "
    "answers with the referral. `choice_required` "
    "(`source_ambiguous`, or `pinned_source_not_found` when the pinned source "
    "was not found) means a requirement's source is not settled: "
    "the referral's `candidates`, each with its `spec_id`, `sentence`, "
    "`coverage_window` and the measures, entities and unit it resolves, are "
    "the only sources that requirement may pin, and `propose_plan` refuses "
    "any other. A `pinned_source_not_found` referral also lists in "
    "`stale_pins` every other requirement whose pin was not found, each with "
    "its own `candidates`; repair them all in one patch. Where a requirement "
    "is offered one candidate, the fresh search settled it: pin that "
    "`spec_id` as its `source_spec_id` and cite no decision for the pin. "
    "Where several are offered, settle it yourself when the question, the brief or the "
    "candidates' coverage make one the clear answer, a source the question "
    "or the brief names included; when the choice is the user's to make, "
    "call `adk_request_input` with the candidates. Either way, record a "
    "`Decision` whose `options` name every candidate, whose `fixes` names "
    "`source_spec_id` and whose rationale says why, "
    "resolved by `planner_default` when you chose and by `user` when the "
    "user's answer did; pin that `spec_id` as the requirement's "
    "`source_spec_id`, and propose again. "
    "A `choice_required` referral with no candidates means the pinned source "
    "is gone: unpin it and change what the requirement asks for. By `kind`: "
    "`no_data` carries the nearest real sources search found instead and the "
    "reason code for why, each described without a spec_id -- read them for "
    "context, never pin one, since the requirement itself is what has to "
    "change; "
    "`unsupported_ask` or `requirement_unmeetable` means the requirement cannot be "
    "met as stated: tell the user what cannot be done and ask, through "
    "`adk_request_input`, before proposing anything, since what to change is "
    "theirs to choose; `ask_not_understood` means search dropped part of the ask and "
    "answered a narrower one, so reword the part the referral names; "
    "`seed_dropped` means remove that requirement; "
    "`requirement_needs_repair` names the `constraint` the requirement breaks and "
    "the `repair` you may make, so make it and propose again; "
    "`no_operation_covers` means no operation carries that requirement out; "
    "`unsupported_ask` and `no_operation_covers` are `unsupported` answers, "
    "which earn a decline (see `declined`); `discovery_budget_exhausted` means the plan asks "
    "for too much at once; `discovery_unavailable` or `surface_unavailable` means "
    "the data service "
    "could not answer just now, so nothing about the requirement needs to "
    "change and the same plan is worth trying again; `internal_failure` is a "
    "fault in the system that no change to the plan repairs, so tell the user, "
    "with its `reference`, and propose nothing further. A plan "
    "that cannot be carried out is not one to approve. "
)


_WITHOUT_A_TRANSLATOR = (
    "Stop once `propose_plan` has recorded the plan: having it translated, "
    "compiled and carried out belongs to a later step that is not attached "
    "here, so there is no translator to call and nothing to approve yet. "
)

_CLOSING = (
    "Every tool answers with an `outcome`, a `code` and sometimes an "
    "`action` -- follow the action rather than guessing at the message. "
    "Call `submit_plan` when the plan is ready for the user's answer; the "
    "harness asks them, and nothing runs until they say yes. Call "
    "`reject_plan` if they decline outright."
)


def planner_instruction(
    *, has_translator: bool, has_skills: bool = False, question_fixed: bool = False
) -> str:
    """The planner's instruction, for the agent it is actually part of.

    `has_translator` and `has_skills` each name something the agent carries:
    the `translate` tool, or the skill tools. An instruction naming
    either where the agent has none sends the model at a tool ADK raises on.

    `question_fixed` selects which paragraph explains `starts_new_plan`:
    `_VERSIONING_FREE` where a restart begins a second plan of its own
    (standalone, and `harness_pair_v1`'s multi-turn session), or
    `_VERSIONING_FIXED`, where the harness graph's own seam holds the turn's
    question and a restart is instead how the model tells it the user asked a
    different one. False, the default, selects `_VERSIONING_FREE`.
    """
    return (
        _PLANNING
        + (_VERSIONING_FIXED if question_fixed else _VERSIONING_FREE)
        + (_SKILLS if has_skills else "")
        + (_TRANSLATING if has_translator else _WITHOUT_A_TRANSLATOR)
        + _CLOSING
    )


_NO_PLAN = "No plan has been proposed yet."

# The most failing fields one refusal names. A plan can fail on dozens at once, and
# the model corrects the first few before it proposes again.
_MAX_REPORTED_ERRORS = 10

# The longest field name in an error location the refusal carries. A location names
# a field the plan declares, except where the model supplied an unknown key, which
# arrives as a name of any length.
_MAX_LOC_ELEMENT_CHARS = 80


def _cut_loc_element(element: str | int) -> str | int:
    """Return `element` with a string cut to `_MAX_LOC_ELEMENT_CHARS`, an index unchanged."""
    return cut_user_text(element, _MAX_LOC_ELEMENT_CHARS) if isinstance(element, str) else element


def _invalid_plan(message: str) -> dict[str, object]:
    """Return the `INVALID_PLAN` problem carrying `message`."""
    return problem(ProblemCode.INVALID_PLAN, message, outcome=Outcome.INVALID_INPUT)


def _invalid_data(exc: ValueError) -> dict[str, object]:
    """Return the `INVALID_PLAN` problem for data a plan's own rules refused.

    A `ValidationError` is reported as each failing field's path, the rule it
    broke and the rule's limit, for the first `_MAX_REPORTED_ERRORS` fields and
    a count of the rest. The refused value is left out: the response lands in
    the model's next request, so echoing it would spend the space the refusal
    exists to protect and repeat it in the history on every retry. A location
    can still carry model-supplied text (an unknown key on a requirement), so
    each string in it is cut at `_MAX_LOC_ELEMENT_CHARS`. A `Decision`
    validator's message echoes `chosen` and `options`, which their own
    length caps bound. Any other `ValueError` is a plan rule stated in words,
    so its text is the message. A rule that refused what the system cannot do
    (`UnsupportedPlanError`) is no malformed draft, so it answers `_unsupported`;
    nor is an ask left unaccounted for (`AsksUnaccountedError`), which answers
    `_asks_unaccounted`.

    `exc.errors()` carries a `loc` tuple and, for a `model_validator` failure,
    the raised exception itself in `ctx["error"]`. Neither is JSON on its own,
    so `to_jsonable_python(..., serialize_unknown=True)` renders both as plain
    JSON (a list, a string) rather than `problem()`'s own validation rejecting
    the detail it was just handed.
    """
    if not isinstance(exc, ValidationError):
        return _invalid_plan(f"Invalid plan data: {exc}")
    cause = _unsupported_cause(exc)
    if cause is not None:
        return _unsupported(cause)
    errors = exc.errors(include_url=False, include_input=False)
    for error in errors:
        unaccounted = error.get("ctx", {}).get("error")
        if isinstance(unaccounted, AsksUnaccountedError):
            return _asks_unaccounted(unaccounted)
    reported = [
        {**error, "loc": tuple(_cut_loc_element(element) for element in error["loc"])}
        for error in errors[:_MAX_REPORTED_ERRORS]
    ]
    lines = [
        f"{'.'.join(map(str, error['loc'])) or 'the plan'}: {error['msg']}" for error in reported
    ]
    if len(errors) > len(reported):
        lines.append(f"and {len(errors) - len(reported)} more")
    return problem(
        ProblemCode.INVALID_PLAN,
        f"Invalid plan data: {'; '.join(lines)}",
        outcome=Outcome.INVALID_INPUT,
        detail={"errors": to_jsonable_python(reported, serialize_unknown=True)},
    )


def _unsupported_cause(exc: ValueError) -> UnsupportedPlanError | None:
    """The `UnsupportedPlanError` a plan rule raised under `exc`, if one did."""
    if not isinstance(exc, ValidationError):
        return None
    for error in exc.errors(include_url=False, include_input=False):
        cause = error.get("ctx", {}).get("error")
        if isinstance(cause, UnsupportedPlanError):
            return cause
    return None


def _unsupported(exc: UnsupportedPlanError) -> dict[str, object]:
    """Return the `UNSUPPORTED` problem for a calculation no plan can state.

    The `action` carries the decline's `code` and leaves its `ask`, the user's
    own words, to the model.
    """
    code = exc.operation.value
    return problem(
        ProblemCode.OPERATION_UNSUPPORTED,
        f"Unsupported: {exc}. This is a limit of the system, not a fault in the plan, so no "
        f"reshaping makes it work. Record the part of the question that asks for it in "
        f"`declined` with code {code!r}, in the user's words and, when the brief lists asks, "
        "with the ask's number as `ask_ref`, and plan only what can still be answered, such "
        "as each series on its own; when nothing can, propose `declined` alone, with no "
        "requirements. The user is told that part was not computed.",
        outcome=Outcome.UNSUPPORTED,
        action=Action(
            call="propose_plan", arguments={"declined": [{"code": code}]}, needs=("declined.ask",)
        ),
        detail={"operation": code},
    )


def _asks_unaccounted(exc: AsksUnaccountedError) -> dict[str, object]:
    """Return the `ASKS_UNACCOUNTED` problem for a plan that drops or doubles an ask.

    The `action` names each ask left unaccounted for and leaves the outputs
    that answer it to the model.
    """
    return problem(
        ProblemCode.ASKS_UNACCOUNTED,
        f"{exc}. Answer an ask in `answered` with the ids of the requirements whose results "
        "answer it, or, when an `unsupported` answer earned its decline (see `declined`), "
        "decline it in `declined` with its number as `ask_ref`; an ask the plan computes in "
        "part takes one "
        "of each. A patch that answers or declines an ask replaces whatever accounted for it "
        "before, so state each once.",
        outcome=Outcome.INVALID_INPUT,
        action=Action(
            call="propose_plan",
            arguments={"answered": [{"ask": number} for number in exc.uncovered]},
            needs=("answered.outputs",),
        )
        if exc.uncovered
        else None,
        detail={"uncovered": list(exc.uncovered), "repeated": list(exc.repeated)},
    )


def _decline_missing(plan: Plan, tool_context: ToolContext) -> dict[str, object] | None:
    """Return the refusal for a plan that drops an ask answered `unsupported`.

    Each `unsupported` answer `propose_plan` or `translate` gave in this
    planner entry (`load_unsupported`) needs a `Decline` with its operation
    as `code`, so the ask stays on the record instead of vanishing from a
    reshaped plan.
    """
    declined = {decline.code for decline in plan.declined}
    missing = sorted(code.value for code in load_unsupported(tool_context.state) - declined)
    if not missing:
        return None
    return problem(
        ProblemCode.DECLINE_MISSING,
        f"An `unsupported` answer named {missing!r} earlier in this turn, and this plan does "
        "not say so. State each in `declined`, with the part of the question it leaves "
        "unanswered, in the user's words.",
        outcome=Outcome.INVALID_INPUT,
        action=Action(
            call="propose_plan",
            arguments={"declined": [{"code": code} for code in missing]},
            needs=("declined.ask",),
        ),
        detail={"operations": list(missing)},
    )


def _decline_unearned(
    plan: Plan, previous: Plan | None, tool_context: ToolContext
) -> dict[str, object] | None:
    """Return the refusal for a plan declining what no tool answered `unsupported`.

    A decline this plan adds needs its code among the `unsupported` answers
    recorded in this planner entry (`load_unsupported`): `propose_plan`'s or
    `translate`'s, or one a replan's run-time refusal stands for
    (`harness.planner_seam._earned_by_refusal`). A decline carried unchanged from `previous`, the
    plan this one amends, was earned when it was recorded; one restated with
    the same `Decline.key` in other words is the same decline.
    """
    earned = load_unsupported(tool_context.state)
    carried = {decline.key for decline in previous.declined} if previous is not None else set()
    unearned = sorted(
        {
            decline.code.value
            for decline in plan.declined
            if decline.key not in carried and decline.code not in earned
        }
    )
    if not unearned:
        return None
    return problem(
        ProblemCode.DECLINE_UNEARNED,
        f"`declined` names {unearned!r}, and nothing in this planner entry earned that "
        "code. A decline is earned when `propose_plan` or `translate` answers `unsupported` "
        "with its code, or when the refusal this entry replans from means no operation "
        "covers the step. To earn it, propose the computation the ask states; if nothing "
        "carries it out, that answer names the code.",
        outcome=Outcome.INVALID_INPUT,
        detail={"operations": unearned},
    )


def _nothing_to_plan(plan: Plan, tool_context: ToolContext) -> dict[str, object] | None:
    """Return the answer for a plan that declines every ask and states nothing else.

    Only declines this entry's `unsupported` answers required count: the
    question is wholly beyond what the system computes, so the entry ends
    on the declines, which the caller tells the user.
    """
    codes = {decline.code for decline in plan.declined}
    if not codes or not codes <= load_unsupported(tool_context.state):
        return None
    return problem(
        ProblemCode.NOTHING_TO_PLAN,
        "Nothing in the question can be planned once its unsupported asks are declined.",
        outcome=Outcome.REFUSED,
        detail={"declined": [decline.model_dump(mode="json") for decline in plan.declined]},
    )


def _approval_summary(plan: Plan) -> dict[str, object]:
    """Return what a next call needs to know about the compiled `plan`.

    The version and revision, which tie a later answer to one content hash,
    and every warning the compiler let through (`Workflow.warnings`). The
    plan's own decisions, requirements, sources and steps are deliberately
    absent. `harness.studio_translate` builds the card the user approves from
    the plan record, through `harness.plan_display`, and `translate` and
    `submit_plan` both read the record out of session state, so a copy here is
    a second description of one plan that can disagree with the first.

    Requires a plan that carries a workflow.
    """
    workflow = plan.workflow
    assert workflow is not None
    return {
        "version": plan.version,
        "revision": plan_revision(plan),
        "warnings": [warning.model_dump(mode="json") for warning in workflow.warnings],
    }


def _execution_blocked(
    plan: Plan, version: int, tool_context: ToolContext, *, has_translator: bool = True
) -> dict[str, object] | None:
    """Return the problem that stops `submit_plan` on `plan`, or `None` to let it proceed.

    Checked in order: a rejected plan; a pending referral, which the planner
    settles by proposing again, after asking the user where the referral is of
    a kind in `_USER_DECIDES` (the refusal's `action` names `adk_request_input`
    until they answer, `propose_plan` otherwise); a plan not yet
    translated and compiled; a stale version; and, last, an allowlist on
    `plan.status` itself -- only `PENDING_APPROVAL` may be submitted, so a
    plan already `APPROVED`, `EXECUTING`, `COMPLETED` or `FAILED` is refused
    too, even though none of those carries a message of its own above. A plan
    seeded back from a failed run (see `harness.graph`) still carries a
    workflow and the version it failed at, so it would otherwise pass every
    check above and reach the user as though it were still approvable. Split
    out of `submit_plan` itself only to keep that function's own branch count
    within this project's `PLR0911` budget.
    """
    if plan.status is PlanStatus.REJECTED:
        return problem(
            ProblemCode.PLAN_REJECTED,
            "The current plan was rejected. Propose a new one before executing.",
            outcome=Outcome.REFUSED,
            action=Action(call="propose_plan"),
        )
    referral = load_referral(tool_context.state)
    if referral is not None:
        asks_user = referral.kind in _USER_DECIDES and not _answered_since_referral(tool_context)
        return problem(
            ProblemCode.REFERRAL_PENDING,
            "The translator referred this plan back; settle the referral with the user "
            "and propose again before executing.",
            outcome=Outcome.REFUSED,
            detail={"referral": referral.model_dump(mode="json")},
            action=Action(call=REQUEST_INPUT_FUNCTION_CALL_NAME if asks_user else "propose_plan"),
        )
    if plan.status is PlanStatus.DRAFT or plan.workflow is None:
        # A draft states the methodology; what runs arrives with
        # translation, so approving one would approve nothing to run.
        return problem(
            ProblemCode.WORKFLOW_MISSING,
            "This plan still needs translating and compiling into a workflow "
            + (
                "before it can execute. Call `translate` first."
                if has_translator
                else "before anything can execute, and no translator is attached here."
            ),
            outcome=Outcome.REFUSED,
            action=Action(call=TRANSLATE_TOOL_NAME) if has_translator else None,
        )
    if plan.version != version:
        return problem(
            ProblemCode.VERSION_STALE,
            f"Plan version {version} is not the current version ({plan.version}) -- "
            "it was amended since. Review the latest version before executing.",
            outcome=Outcome.INVALID_INPUT,
            detail={"current": plan.version, "given": version},
        )
    if plan.status is not PlanStatus.PENDING_APPROVAL:
        return problem(
            ProblemCode.TRANSITION_REFUSED,
            f"This plan is {plan.status.value}, not awaiting approval, so it cannot be "
            "submitted. Propose a new plan before executing.",
            outcome=Outcome.REFUSED,
            detail={"status": plan.status.value},
            action=Action(call="propose_plan"),
        )
    return None


# The refusal for a `propose_plan` call after `MAX_PLAN_ROUNDS` proposals in one
# entry into the planner. The plan's version can exceed the cap, since only the
# count restarts on a new entry. It names what remains possible: approving the
# recorded version when it compiled, and the user's next message.
_BUDGET_EXHAUSTED = (
    f"{MAX_PLAN_ROUNDS} plans have been proposed in this planner run, each with its own "
    "discovery budget; no further version will be recorded."
)

# The same refusal for a `starts_new_plan` call. The call already claims to be a
# different question, so the text says a restart counts against the same bound.
_BUDGET_EXHAUSTED_RESTART = (
    f"A restart is a proposal like any other, and {MAX_PLAN_ROUNDS} have been made in this "
    "planner run, each with its own discovery budget; no further plan will be recorded, "
    "however the call is stated."
)

# What follows either refusal. The approval is offered only where the recorded plan
# compiled, since a plan that never did has no card for the user to approve.
_BUDGET_APPROVABLE = (
    "The last version that compiled can still be approved, and the user's next message "
    "opens a fresh budget."
)
_BUDGET_FRESH = "The user's next message opens a fresh budget."

# The refusal for a `starts_new_plan` call while `question_fixed`. Unlike
# `_BUDGET_EXHAUSTED_RESTART`, this is not about the discovery budget at all:
# it is how the model tells the harness the user's reply asks a different
# question, so the message says what happened and that nothing further is
# needed from it, the way `_after_planner_tool` -- see `harness.planner_seam`
# -- ends the turn on `BUDGET_EXHAUSTED` today.
_QUESTION_FIXED_RESTART = (
    "This turn's question is fixed, so `starts_new_plan` cannot open a plan for a "
    "different one; nothing is recorded, and the plan already on record stands "
    "unchanged. This is exactly how a different question is reported: the turn ends "
    "here, the user is told to send it as its own message, and nothing further is "
    "needed from you."
)


def _resolve_intent(
    existing: Plan | None,
    *,
    amends_version: int | None,
    starts_new_plan: bool,
    rounds_so_far: int,
    question: str | None = None,
    question_fixed: bool = False,
) -> tuple[Plan | None, int] | dict[str, object]:
    """Which plan a `propose_plan` call means, from what the call states.

    Returns `(previous, rounds)` (the plan to patch, and the proposal count
    this call brings the entry to) or, when the call does not state its
    intent unambiguously, the problem to answer with. Which of the two a call
    is comes from the call's own `amends_version`/`starts_new_plan` alone,
    regardless of how `question`'s wording compares to the recorded plan's. A
    model that rewords the question while meaning to amend would otherwise be
    read as starting a fresh plan and drop every item it did not restate.

    With nothing recorded, the call is a first draft. With a plan recorded,
    an `amends_version` naming that same version patches it. `starts_new_plan`
    begins a plan of its own. Anything else is refused, naming the version to
    pass so the next call is correct.

    `amends_version` and `starts_new_plan` are read in that order, so a call
    stating both is treated as an amendment, since a payload shaped as a
    patch is read as one. `rounds_so_far` counts the proposals made in this
    entry into the planner. `PlannerSeam` zeroes it on every entry that is not
    a mid-turn resume. An amendment or a restart increments it, since every
    version opens a fresh discovery budget, and past `MAX_PLAN_ROUNDS` the call
    is refused rather than recorded, with no `action` named, since any tool
    named there would restart the loop. A restart begins at version 1 and is
    counted like an amendment, so restarting does not escape the bound.

    `question_fixed` (the harness graph's own seam builds the planner with it
    set; standalone and `harness_pair_v1` leave it `False`) narrows both
    readings once a plan is recorded, against `existing.question` rather than
    the call's own wording of the question it is answering:

    - A restart (`starts_new_plan` with no `amends_version`) is refused with
      `QUESTION_FIXED` before the round cap is even checked, ahead of
      `BUDGET_EXHAUSTED` -- both refusals record nothing, so nothing is lost
      by checking this one first, and a restart can never succeed under a
      fixed question however fresh the budget is. Leading with a budget
      message here would read as an invitation to retry once the user's next
      message opens one, which is false.
    - An amendment (`amends_version` naming the recorded version) whose own
      `question` states something other than what is recorded is refused as
      `INVALID_PLAN`, checked after the round cap like the shape checks below
      it: a correctly-shaped amendment that merely lost the race for the last
      round in this entry is exactly what the cap exists to stop, so the round
      check still takes priority there, the opposite ordering from the restart
      case above.
    """
    if existing is None:
        return None, 1
    blocked = _question_fixed_restart_blocked(
        existing,
        amends_version=amends_version,
        starts_new_plan=starts_new_plan,
        question_fixed=question_fixed,
    ) or _round_cap_blocked(
        existing,
        amends_version=amends_version,
        starts_new_plan=starts_new_plan,
        rounds_so_far=rounds_so_far,
    )
    if blocked is not None:
        return blocked
    if amends_version is None and starts_new_plan:
        return None, rounds_so_far + 1
    invalid = _amendment_version_invalid(
        existing, amends_version
    ) or _question_fixed_amendment_invalid(
        existing, question=question, question_fixed=question_fixed
    )
    if invalid is not None:
        return invalid
    return existing, rounds_so_far + 1


def _question_fixed_restart_blocked(
    existing: Plan, *, amends_version: int | None, starts_new_plan: bool, question_fixed: bool
) -> dict[str, object] | None:
    """Return the `QUESTION_FIXED` refusal for a restart while `question_fixed`, or `None`.

    Split out of `_resolve_intent` to keep its own branch count within this
    project's `PLR0911` budget, the same reason `_execution_blocked` is split
    out of `submit_plan`.
    """
    if not (question_fixed and amends_version is None and starts_new_plan):
        return None
    return problem(
        ProblemCode.QUESTION_FIXED,
        _QUESTION_FIXED_RESTART,
        outcome=Outcome.REFUSED,
        detail={"version": existing.version},
    )


def _round_cap_blocked(
    existing: Plan, *, amends_version: int | None, starts_new_plan: bool, rounds_so_far: int
) -> dict[str, object] | None:
    """Return the `BUDGET_EXHAUSTED` refusal past `MAX_PLAN_ROUNDS`, or `None`.

    Split out of `_resolve_intent` for the same reason as
    `_question_fixed_restart_blocked`, above.
    """
    if rounds_so_far + 1 <= MAX_PLAN_ROUNDS:
        return None
    restart = amends_version is None and starts_new_plan
    return problem(
        ProblemCode.BUDGET_EXHAUSTED,
        f"{_BUDGET_EXHAUSTED_RESTART if restart else _BUDGET_EXHAUSTED} "
        f"{_BUDGET_APPROVABLE if existing.translation is not None else _BUDGET_FRESH}",
        outcome=Outcome.REFUSED,
        detail={
            "scope": "plan_rounds",
            "rounds": MAX_PLAN_ROUNDS,
            "version": existing.version,
        },
    )


def _amendment_version_invalid(
    existing: Plan, amends_version: int | None
) -> dict[str, object] | None:
    """Return the `INVALID_PLAN` refusal for an amendment naming no version, or the wrong
    one, or `None` when `amends_version` is exactly `existing.version`.

    Split out of `_resolve_intent` for the same reason as
    `_question_fixed_restart_blocked`, above.
    """
    if amends_version is None:
        return _invalid_plan(
            f"A plan is already recorded (version {existing.version}), so this call has "
            "to say which it is: pass "
            f"`amends_version={existing.version}` to patch that plan, or "
            "`starts_new_plan=true` to begin a plan of its own for a different question. "
            "A patch states only what changes; everything it does not mention is carried "
            "forward.",
        )
    if amends_version != existing.version:
        return _invalid_plan(
            f"This call amends version {amends_version}, but version {existing.version} is "
            f"the one recorded. Pass `amends_version={existing.version}`, or "
            "`starts_new_plan=true` to begin a plan of its own for a different question.",
        )
    return None


def _question_fixed_amendment_invalid(
    existing: Plan, *, question: str | None, question_fixed: bool
) -> dict[str, object] | None:
    """Return the `INVALID_PLAN` refusal for an amendment that rewords a fixed question, or `None`.

    Split out of `_resolve_intent` for the same reason as
    `_question_fixed_restart_blocked`, above.
    """
    if not (question_fixed and question is not None and question != existing.question):
        return None
    return _invalid_plan(
        f"This call amends version {existing.version}, but its `question` "
        f"({question!r}) does not match the one recorded ({existing.question!r}) -- "
        "the question is fixed for this turn, so a patch may not change it. Omit "
        "`question` to leave it as recorded, or restate it unchanged. Pass "
        "`starts_new_plan=true` instead if the user's reply genuinely asks a "
        "different question; that ends the turn rather than recording a plan for it.",
    )


def _offered_candidates(args: dict[str, Any]) -> frozenset[str]:
    """The candidates an `adk_request_input` call's own function-call event
    offered, from either shape that name occurs under:

    - the planner's own `ask_user` tool call: a plain `FunctionTool` call,
      so `args` is exactly the kwargs the model supplied, flat --
      `args["candidates"]`.
    - the graph's `clarify` interrupt (`harness.graph`'s `CLARIFY_INTERRUPT`):
      built through ADK's own `google.adk.workflow.utils.
      _workflow_hitl_utils.create_request_input_event`, which dumps the
      whole `RequestInput` into `args` -- the caller's own `payload` (where
      `clarify` puts `candidates`) nests under `args["payload"]`.
    """
    candidates = args.get("candidates")
    if not candidates:
        payload = args.get("payload")
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
    return frozenset(
        str(candidate["spec_id"])
        for candidate in candidates or []
        if isinstance(candidate, dict) and candidate.get("spec_id")
    )


def _elected_spec_ids(tool_context: ToolContext) -> frozenset[str]:
    """Every spec_id the user actually chose answering an `adk_request_input`
    call, read off this session's own event history -- the record of what
    the user was actually asked and what they actually answered, checked
    against a `resolved_by=user` source pin (`propose_plan`, below).

    Membership in a call's offered `candidates` is not by itself an
    election: an `adk_request_input` call that offers `{A, B}` records only
    a question, not a choice, and a plan pinning `B` after the user picked
    `A` -- or after no answer has arrived at all -- is not a user election
    either. What counts as one is a call's own function-call event, paired
    with the matching `FunctionResponse` event, if any, that answers it:
    `create_request_input_event` builds the call with `id=interrupt_id`, and
    `create_request_input_response` builds the answer with
    `FunctionResponse.id` set to that same `interrupt_id`; `harness.studio`'s
    `/interrupt` route is what actually sends it, shaped
    `{"result": {"spec_id": ...}}` for a chosen candidate. A call with no
    paired response yet -- the run is still paused there, or the answer
    hasn't arrived -- contributes nothing: its candidates were offered, not
    elected, so a pin naming one is refused rather than silently accepted.

    Every `adk_request_input` call the graph itself raises -- `harness.
    graph`'s `CLARIFY_INTERRUPT` -- shares the one fixed id `"clarify"`
    across every clarify round in a session, unlike the planner's own
    `ask_user` calls, each of which gets its own id. Pairing by id alone
    would let a round-2 offer, re-raised under the same id, retroactively
    read a round-1 answer as electing whatever round 2 happens to also
    offer. This walks the session once, in event order, and pairs each
    response with the call whose offer was seen most recently at that
    point -- the nearest preceding call sharing its id -- rather than
    building a dict of every call keyed by id first and reading it back
    after the fact.

    Not a state key this seam writes and reads back: a long-running tool
    call that pauses (returns `None`) never gets its own `FunctionResponse`
    event built for *that* call (`_tool_caller.py`'s
    `is_long_running`-and-falsy rule), so nothing written to
    `tool_context.state` -- or even straight to `tool_context.session.state`
    -- from inside that call survives the pause; there is no event for
    either write's delta to ride, and no commit to trigger for a plain
    mutation. The call's own function-call event, carrying its offered
    candidates in `args`, is built regardless (it is the model's own turn,
    not the tool's response) and is exactly what `ctx.run_node`'s own
    pause-and-resume already depends on surviving (`harness.planner_seam`'s
    module docstring) -- so reading it back here needs no bookkeeping of
    this seam's own. The matching `FunctionResponse`, once it exists, is
    built by ADK from the resumed run's own input message, not by anything
    in this package.

    Read across the whole session, not scoped to one invocation: the ask
    pauses the run, and the answer that resumes it can arrive on a later
    invocation than the one that asked, so there is no single "this turn"
    an `Event.invocation_id` could scope this to.

    `ToolContext` is `Context` (`google.adk.tools.tool_context`), so
    `tool_context.session` is the same live `Session` every other seam
    shares, `events` included.
    """
    offered_by_call: dict[str, frozenset[str]] = {}
    elected: set[str] = set()
    for event in tool_context.session.events:
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            call = part.function_call
            if (
                call is not None
                and call.name == REQUEST_INPUT_FUNCTION_CALL_NAME
                and call.id is not None
            ):
                offered_by_call[call.id] = _offered_candidates(call.args or {})
                continue
            response = part.function_response
            if response is None or response.name != REQUEST_INPUT_FUNCTION_CALL_NAME:
                continue
            offered = offered_by_call.get(response.id)
            if not offered:
                continue
            result = (response.response or {}).get("result")
            spec_id = result.get("spec_id") if isinstance(result, dict) else None
            if spec_id is not None and str(spec_id) in offered:
                elected.add(str(spec_id))
    return frozenset(elected)


# The referral kinds that hand a requirement back because what it asks for cannot be
# done, so what to change is the user's to choose. `propose_plan` holds a plan over
# one of these until the user has answered.
_USER_DECIDES = frozenset({ReferralKind.REQUIREMENT_UNMEETABLE, ReferralKind.UNSUPPORTED_ASK})


def _is_answer(body: dict[str, Any]) -> bool:
    """Whether an `adk_request_input` response body carries something the user said.

    A user's answer is `{"result": ...}` (`harness.studio`'s `/interrupt` route wraps
    it, as ADK wraps any non-dict value): free text, `{"text": ...}` for a reply
    outside the offered options, or `{"spec_id": ...}` for a chosen candidate. A
    body with no `result` is a call the tool or ADK rejected. `{"skip": true}` is the
    user declining to choose, and blank text says nothing.
    """
    result = body.get("result")
    if isinstance(result, dict):
        result = result.get("text") or result.get("spec_id")
    return isinstance(result, str) and bool(result.strip())


def _answered_since_referral(tool_context: ToolContext) -> bool:
    """Whether the user has answered an ask the planner raised after seeing the
    translator's referral, read off this session's event history as
    `_elected_spec_ids` reads it.

    The referral reaches the planner as the `FunctionResponse` for its own
    `translate` call, carrying its `kind` and `requirement_id`. This walks the
    session once, in event order. An `adk_request_input` call event after that
    response is an ask the model raised knowing the referral, and its `id` is
    remembered. A response counts as the user's answer when it pairs by `id` with
    one of those asks and `_is_answer` holds. An ask issued before the referral, a
    call the tool or ADK rejected, and a skip do not count. A skip is the user
    declining to choose, which the planner may not substitute for, so the planner
    ends the turn by telling them what cannot be done.

    A `translate` response naming a different referral than the last one starts
    the walk over. `translate` called again on a plan whose referral is still pending
    answers with that same referral, so the answer the user gave in between still
    counts. A `propose_plan` that recorded a version clears the referral in
    session state, so it ends the referral this walk is following too. A refused
    proposal leaves both alone, which keeps the user's answer standing for the
    retry.

    With no `translate` response on record the result is no answer. The guard
    refuses when it cannot say where the referral began, so an answer given for
    something else settles nothing.
    """
    answered = False
    referral: tuple[object, object] | None = None
    asks: set[str] = set()
    for event in tool_context.session.events:
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            call = part.function_call
            if (
                call is not None
                and call.name == REQUEST_INPUT_FUNCTION_CALL_NAME
                and call.id is not None
                and referral is not None
            ):
                asks.add(call.id)
            response = part.function_response
            if response is None:
                continue
            body = response.response or {}
            if response.name == REQUEST_INPUT_FUNCTION_CALL_NAME:
                answered = answered or (response.id in asks and _is_answer(body))
            elif response.name == TRANSLATE_TOOL_NAME:
                handed_back = (body.get("kind"), body.get("requirement_id"))
                if handed_back != referral:
                    referral, answered = handed_back, False
                    asks.clear()
            elif response.name == "propose_plan" and body.get("outcome") == Outcome.OK.value:
                referral, answered = None, False
                asks.clear()
    return answered


def _referral_unanswered(
    existing: Plan | None, tool_context: ToolContext
) -> dict[str, object] | None:
    """Return the refusal for a `propose_plan` call over a referral the user has not
    answered, or `None` to let the call through.

    A first draft, with no plan recorded, is never held.

    A patch replaces a requirement whole, and a proposal clears the referral, so a
    plan proposed over an `unsupported_ask` or `requirement_unmeetable`
    referral would drop what the referral names and go on to translate. Every
    other kind is the planner's to settle, or is settled through a source
    election that `_elected_spec_ids` checks. The refusal points at the
    tool that asks the user, since `propose_plan` is what it stops.
    """
    referral = load_referral(tool_context.state)
    if (
        existing is None
        or referral is None
        or referral.kind not in _USER_DECIDES
        or _answered_since_referral(tool_context)
    ):
        return None
    return problem(
        ProblemCode.REFERRAL_PENDING,
        f"The translator handed requirement `{referral.requirement_id}` back as "
        f"{referral.kind.value}, and the user has said nothing since. Tell them what "
        f"cannot be done and put the choice to them through `{REQUEST_INPUT_FUNCTION_CALL_NAME}` "
        "before proposing again.",
        outcome=Outcome.REFUSED,
        action=Action(call=REQUEST_INPUT_FUNCTION_CALL_NAME),
    )


def _election_untraceable(
    decisions: Sequence[Decision], tool_context: ToolContext
) -> dict[str, object] | None:
    """Return the refusal for a `resolved_by=user` source pin the user never chose, or `None`.

    A `resolved_by=user` source pin is model-self-reported, so it is checked
    against what the user actually chose: an answer to an `adk_request_input`
    call on record in this session (`_elected_spec_ids`), or a dataset the user
    handed to Studio from Search, or a copy of one over another window
    (`planner.plan.seeded_sources`, which only `harness.graph`'s `seed` node
    and `_window_pins` write). A pin naming a spec_id that was never
    offered, was offered but not chosen, or was offered with no answer on
    record yet is refused. `Plan`'s own validators have no session to read
    (see `planner.plan.DatasetRequirement`), so they cannot check this. A pin
    on a dataset the turn dropped is refused before this check
    (`_dropped_seed_pinned`), so the advice below to record a pin as
    `planner_default` is never given for one.
    """
    elected = _elected_spec_ids(tool_context) | {
        seeded.source.spec_id for seeded in seeded_sources(tool_context.state)
    }
    untraceable = sorted(
        {
            decision.fixes.source_spec_id
            for decision in decisions
            if decision.resolved_by is ResolvedBy.USER
            and decision.fixes is not None
            and decision.fixes.source_spec_id is not None
            and decision.fixes.source_spec_id not in elected
        }
    )
    if not untraceable:
        return None
    return problem(
        ProblemCode.ELECTION_UNTRACEABLE,
        f"decision(s) claim `resolved_by=user` for source(s) {untraceable!r}, but "
        "the user never actually chose them answering `adk_request_input` -- a user "
        "election has to trace to what the user was actually asked and picked, not "
        "merely offered. Call `adk_request_input` with that candidate and get the "
        "user's answer before pinning it this way, or record the decision as "
        "`resolved_by=planner_default` if you settled it yourself.",
        outcome=Outcome.REFUSED,
        detail={"spec_ids": untraceable},
    )


def _refound_spec_ids(tool_context: ToolContext) -> frozenset[str]:
    """The sources a pending `pinned_source_not_found` referral found in place of a stale pin.

    One per requirement whose offer is a single source: the fresh search
    settled on it, so re-pinning to it is no choice for the planner to
    record. A requirement offered several is a real choice, which a
    `Decision` the planner writes settles.
    """
    referral = load_referral(tool_context.state)
    if referral is None or referral.kind is not ReferralKind.PINNED_SOURCE_NOT_FOUND:
        return frozenset()
    return frozenset(
        candidates[0].spec_id for candidates in referral.offered().values() if len(candidates) == 1
    )


def _choice_not_offered(plan: Plan, tool_context: ToolContext) -> dict[str, object] | None:
    """Return the refusal for a pin a pending source choice did not offer, or `None`.

    A `choice_required` referral names the sources each requirement it is
    about may pin (`Referral.offered`): its `candidates`, and each stale
    pin's, which discovery recorded and no model wrote. While it is pending,
    each such requirement as `plan` states it either pins one of them or
    pins nothing. A requirement the plan no longer states passes.
    """
    referral = load_referral(tool_context.state)
    if referral is None or referral.outcome is not Outcome.CHOICE_REQUIRED:
        return None
    for requirement_id, candidates in referral.offered().items():
        try:
            requirement = plan.requirement(requirement_id)
        except KeyError:
            continue
        if not isinstance(requirement, DatasetRequirement) or requirement.source_spec_id is None:
            continue
        offered = [candidate.spec_id for candidate in candidates]
        if requirement.source_spec_id in offered:
            continue
        return problem(
            ProblemCode.SOURCE_NOT_OFFERED,
            f"{requirement.id!r} pins {requirement.source_spec_id!r}, which the "
            f"{referral.kind.value} referral did not offer. Pin one of its candidates, put "
            "them to the user, or unpin the requirement and change what it asks for.",
            outcome=Outcome.REFUSED,
            detail={"requirement_id": requirement.id, "candidates": offered},
        )
    return None


def _dropped_seed_pinned(
    decisions: Sequence[Decision],
    requirements: Sequence[EvidenceRequirement],
    tool_context: ToolContext,
) -> dict[str, object] | None:
    """Return the refusal for a pin on a dataset the turn dropped, or `None`.

    The turn dropped the dataset because its StatVars have left it, and the
    user is told the plan leaves it out. A decision or a dataset requirement
    whose pin `planner.plan.names_dropped_seed` reads as a dropped dataset
    (the seed's spec_id, or that of a copy of it over another window) is
    refused whatever `resolved_by` says.
    """

    def dropped(spec_id: str | None) -> bool:
        return spec_id is not None and names_dropped_seed(tool_context.state, spec_id=spec_id)

    slots = sorted(
        decision.slot
        for decision in decisions
        if decision.fixes is not None and dropped(decision.fixes.source_spec_id)
    )
    requirement_ids = sorted(
        requirement.id
        for requirement in requirements
        if isinstance(requirement, DatasetRequirement) and dropped(requirement.source_spec_id)
    )
    if not slots and not requirement_ids:
        return None
    return problem(
        ProblemCode.SEED_DROPPED,
        f"decision(s) {slots!r} and requirement(s) {requirement_ids!r} pin a dataset this "
        "turn dropped: the user is told that this plan leaves it out, why, and to search for "
        "it again. Do not plan that dataset again in this turn, pinned or unpinned. Name "
        "each requirement that retrieves or reads it in `removed_evidence_requirements` and "
        "each decision that pins it in `removed_decisions`, and plan the rest of the question.",
        outcome=Outcome.REFUSED,
        detail={"decisions": slots, "requirements": requirement_ids},
    )


def _unrecordable(
    plan: Plan, tool_context: ToolContext, *, previous: Plan | None
) -> dict[str, object] | None:
    """Return the refusal for a built plan `propose_plan` may not record, or `None`.

    A plan states at least one requirement, unless it declines every ask
    (`_nothing_to_plan`). A patch carries forward what it
    does not name, so the amended plan is checked whole for a pin the recorded
    plan made before the turn dropped its dataset (`_dropped_seed_pinned`), for
    a pin a pending source choice did not offer (`_choice_not_offered`), for a
    decline no `unsupported` answer earned (`_decline_unearned`, against
    `previous`, the plan it amends), and for a decline every `unsupported`
    answer this turn requires (`_decline_missing`).
    """
    if not plan.evidence_requirements:
        return _nothing_to_plan(plan, tool_context) or _invalid_plan(
            "A plan states at least one evidence requirement: what data the answer "
            "rests on, citing the decisions that fixed it. State the requirements this "
            "question needs, or name the ones a patch removes in "
            "`removed_evidence_requirements`."
        )
    return (
        _dropped_seed_pinned(plan.decisions, plan.evidence_requirements, tool_context)
        or _choice_not_offered(plan, tool_context)
        or _decline_unearned(plan, previous, tool_context)
        or _decline_missing(plan, tool_context)
    )


# The option a decision `propose_plan` writes for a search pin leaves the
# source unpinned by (`_synthesize_source_decisions`).
_UNPINNED_CHOICE = "leave the source unpinned"


def _pin_choice(spec_id: str) -> str:
    """The option a decision `propose_plan` writes for a search pin pins `spec_id` by."""
    return f"pin {spec_id}"


def _repinned(
    decisions: Sequence[Decision],
    requirements: Sequence[EvidenceRequirement],
    *,
    decision_pins: Mapping[str, str],
    requirement_pins: Mapping[str, str],
    slots: Mapping[str, str] = {},
) -> tuple[list[Decision], list[EvidenceRequirement]]:
    """`decisions` and `requirements` with each source pin moved as the maps say.

    `decision_pins` maps a decision's slot, `requirement_pins` a dataset
    requirement's id, to the spec_id it pins after the move; anything either
    map does not name is returned as it was. `slots` maps the slot of a
    decision `propose_plan` wrote for a search pin (`planner.plan.
    is_search_pin`) to the one its moved pin takes (`search_pin_slot`): that
    decision is rewritten for its new pin, and every citation of it cites the
    new slot, so it stays a search pin `harness.plan_display` leaves out.
    """
    moved_decisions: list[Decision] = []
    for decision in decisions:
        if decision.fixes is None or decision.slot not in decision_pins:
            moved_decisions.append(decision)
            continue
        pin = decision_pins[decision.slot]
        update: dict[str, object] = {
            "fixes": decision.fixes.model_copy(update={"source_spec_id": pin})
        }
        if is_search_pin(decision) and decision.slot in slots:
            update |= {
                "slot": slots[decision.slot],
                "options": (_pin_choice(pin), _UNPINNED_CHOICE),
                "chosen": _pin_choice(pin),
            }
        moved_decisions.append(decision.model_copy(update=update))
    moved_requirements: list[EvidenceRequirement] = []
    for requirement in requirements:
        update = {}
        if isinstance(requirement, DatasetRequirement) and requirement.id in requirement_pins:
            update["source_spec_id"] = requirement_pins[requirement.id]
        if slots.keys() & set(requirement.decision_ids):
            update["decision_ids"] = tuple(
                slots.get(slot, slot) for slot in requirement.decision_ids
            )
        moved_requirements.append(requirement.model_copy(update=update) if update else requirement)
    return moved_decisions, moved_requirements


def _unwindowed_pins(
    decisions: Sequence[Decision],
    requirements: Sequence[EvidenceRequirement],
    to_origin: Mapping[str, str],
) -> tuple[list[Decision], list[EvidenceRequirement]]:
    """`decisions` and `requirements` with every pin of a windowed copy moved back to the pin it copies.

    `to_origin` maps each copy's spec_id to the pin it copies over another
    period: a seed's (`planner.plan.SeededSource.windowed_from`) or a
    recorded pin's (`planner.plan.WindowedPin.windowed_from`).
    `propose_plan` reasons about a pin by the spec_id the planner was shown,
    whatever period an earlier call recorded it over (`_window_pins`), so a
    patch and the plan it amends always agree on what they pin.
    """
    return _repinned(
        decisions,
        requirements,
        decision_pins={
            decision.slot: origin
            for decision in decisions
            if decision.fixes is not None
            and (origin := to_origin.get(decision.fixes.source_spec_id or ""))
        },
        requirement_pins={
            requirement.id: origin
            for requirement in requirements
            if isinstance(requirement, DatasetRequirement)
            and (origin := to_origin.get(requirement.source_spec_id or ""))
        },
        slots={
            search_pin_slot(copy): search_pin_slot(origin) for copy, origin in to_origin.items()
        },
    )


def _copied_spec_id(copy: SeededSource | WindowedPin) -> str:
    """The spec_id a copy of a pin over another period retrieves."""
    return copy.source.spec_id if isinstance(copy, SeededSource) else copy.spec_id


def _window_pins(
    plan: Plan, seeded: Sequence[SeededSource]
) -> tuple[Plan, list[SeededSource], list[WindowedPin]]:
    """`plan` with each pin whose identity is on record moved to that identity over the requirement's own period.

    A pin's identity carries the years of the search that found it: the
    years the user saw in Search for a seed, often every year with data, or
    the window of the question an earlier search answered. A plan's
    retrieval mints the pinned spec_id as it is (`compiler.compile`'s
    `_retrieve_arguments`). So a dataset requirement whose pin's identity is
    over another period is pinned instead to that identity over its own
    (`data_mcp.definition_writes.windowed_identity`, the spec_id `get_data`'s
    `scope.window` would mint), and so is the decision that fixes the pin.

    The identity on record is a seed's own (`seeded`), or else the one a
    search in the conversation recorded (`planner.plan.recorded_identity`).
    One of a `latest` window is copied like any other: the copy replaces
    the window whole, as `get_data`'s scope surgery does, and the translator
    confirms the copy through Search before it is retrieved.

    Returns the moved plan and every copy it pins: of a seed, the one
    `seeded` holds for that spec_id or a new one (`planner.plan.
    windowed_copy`); of any other pin, a `planner.plan.WindowedPin`. Neither
    states what Search stated for the original's years; the translator
    reads those from Search's answer for the copy when it honours the pin
    (`translator.discovery`).

    One decision fixing the pin of requirements over different periods
    cannot name every copy with its one `source_spec_id`. Those
    requirements keep the pin as the planner wrote it, and the translator
    searches for each, unless the pin is a seed's.

    Raises:
        ValueError: If one decision fixes a seed's pin for requirements over
            different periods.
    """
    seeds = {item.source.spec_id: item for item in seeded if item.windowed_from is None}
    pinning: dict[str, str] = {}
    copies: dict[str, SeededSource | WindowedPin | None] = {}
    for requirement in plan.evidence_requirements:
        if not isinstance(requirement, DatasetRequirement) or requirement.source_spec_id is None:
            continue
        pin = requirement.source_spec_id
        seed = seeds.get(pin)
        recorded = None if seed is not None else recorded_identity(pin)
        if seed is not None:
            identity = seed.source.identity
        else:
            identity = recorded.identity if recorded is not None else None
        if identity is None:
            continue
        pinning[requirement.id] = pin
        window = period_window(requirement.period)
        if identity.get("requested_window") == window:
            copies[requirement.id] = None
        elif seed is not None:
            copies[requirement.id] = windowed_copy(seed, window)
        elif recorded is not None:
            copies[requirement.id] = windowed_pin(pin, recorded, requirement.period)
    targets: dict[str, set[str]] = {}
    for requirement in plan.evidence_requirements:
        if requirement.id not in copies:
            continue
        copy = copies[requirement.id]
        target = pinning[requirement.id] if copy is None else _copied_spec_id(copy)
        for slot in requirement.decision_ids:
            targets.setdefault(slot, set()).add(target)
    origins = set(pinning.values())
    decision_pins: dict[str, str] = {}
    slots: dict[str, str] = {}
    for decision in plan.decisions:
        pin = decision.fixes.source_spec_id if decision.fixes is not None else None
        if pin is None or pin not in origins:
            continue
        pinned = targets.get(decision.slot, set())
        if len(pinned) > 1:
            if pin in seeds:
                raise ValueError(
                    f"decision {decision.slot!r} fixes the pin of requirements over different "
                    "periods, which retrieve different spec_ids; give each of those requirements "
                    "its own decision"
                )
            for requirement in plan.evidence_requirements:
                if decision.slot in requirement.decision_ids:
                    copies.pop(requirement.id, None)
            continue
        if pinned and pinned != {pin}:
            decision_pins[decision.slot] = next(iter(pinned))
            if is_search_pin(decision):
                slots[decision.slot] = search_pin_slot(decision_pins[decision.slot])
    requirement_pins = {
        requirement_id: _copied_spec_id(copy)
        for requirement_id, copy in copies.items()
        if copy is not None
    }
    if not requirement_pins and not decision_pins:
        return plan, [], []
    seed_copies: dict[str, SeededSource] = {}
    recorded_copies: dict[str, WindowedPin] = {}
    for copy in copies.values():
        if isinstance(copy, SeededSource) and copy.source.spec_id not in seeds:
            # Two seeds over one dataset copy to one spec_id over a shared
            # period. One copy is recorded for both.
            seed_copies.setdefault(copy.source.spec_id, copy)
        elif isinstance(copy, WindowedPin):
            recorded_copies.setdefault(copy.spec_id, copy)
    decisions, requirements = _repinned(
        plan.decisions,
        plan.evidence_requirements,
        decision_pins=decision_pins,
        requirement_pins=requirement_pins,
        slots=slots,
    )
    moved = plan.model_copy(
        update={"decisions": tuple(decisions), "evidence_requirements": tuple(requirements)}
    )
    return (
        Plan.model_validate(moved.model_dump()),
        list(seed_copies.values()),
        list(recorded_copies.values()),
    )


def _synthesize_source_decisions(
    decisions: Sequence[Decision],
    evidence_requirements: Sequence[EvidenceRequirement],
    *,
    previous: Plan | None,
    settled_spec_ids: Collection[str],
    earlier_spec_ids: Collection[str] = (),
    refound_spec_ids: Collection[str] = (),
) -> tuple[list[Decision], list[EvidenceRequirement]]:
    """Add the `Decision` a verbatim pin of a settled search result needs, so citing it is all the model has to do.

    `harness.planner_seam` shows the planner the spec_ids this turn's
    searches settled on (`PinSource.settled`) as allowed pins, but
    `_check_pin_rests_on_a_fixing_decision` still requires a citing
    `Decision`. This writes one per pinned settled spec_id on the model's
    behalf, and likewise for a re-pin to the one source a
    `pinned_source_not_found` referral found in place of a stale pin
    (`refound_spec_ids`). A pin of anything else, a candidate among several
    included, needs a decision the model wrote. A requirement still citing a
    decision written here for a pin it no longer states has that citation
    dropped, so a re-pin needs no edit of `decision_ids` either.

    A synthesized `Decision`'s two `options` are pinning the id and leaving
    the source unpinned, the only two things true of a `DatasetRequirement`
    pinning it, and `resolved_by=planner_default` names the planner, not the
    user, as the one who settled it.

    Only a requirement in this call's own `evidence_requirements` is
    examined, never one carried unchanged from `previous`; `previous.decisions`
    is read only to check whether an existing citation already covers the
    pin, so a correctly-cited requirement never gets a redundant decision. A
    requirement whose citations do not yet cover the pin gets the citation
    added even when the synthesized decision's slot is already known
    elsewhere: only the append to `decisions` is skipped then, not the
    citation.

    Args:
        decisions: This `propose_plan` call's own `decisions` argument, before
            it is merged with any prior plan.
        evidence_requirements: Likewise, this call's own
            `evidence_requirements`.
        previous: The plan this call amends, or `None` on a first draft or a
            restart.
        settled_spec_ids: The settled spec_ids `harness.planner_seam` showed
            the model this entry (`pin_sources`). A pin of any other id is
            left for the ordinary validator to judge.
        earlier_spec_ids: Those of `settled_spec_ids` whose block came from
            an earlier turn (`PinSource.earlier_turn`); their decision's
            rationale names a search in this conversation, not this turn.
        refound_spec_ids: The sources a pending `pinned_source_not_found`
            referral offers as the one found in place of a requirement's
            stale pin (`_refound_spec_ids`). A pin of one is settled the
            same way, with a rationale saying so.

    Returns:
        `decisions`/`evidence_requirements`, unchanged unless some requirement
        needed a pin cited. When so, each such requirement's `decision_ids`
        cites the synthesized decision's slot, and each decision is appended
        to `decisions` unless a prior round already recorded it.
    """
    known = {decision.slot: decision for decision in (previous.decisions if previous else ())}
    known.update({decision.slot: decision for decision in decisions})
    pinnable = {*settled_spec_ids, *refound_spec_ids}

    def _uncited_pin(requirement: EvidenceRequirement) -> str | None:
        if not isinstance(requirement, DatasetRequirement):
            return None
        pin = requirement.source_spec_id
        if pin is None or pin not in pinnable:
            return None
        for slot in requirement.decision_ids:
            cited = known.get(slot)
            if cited is not None and cited.fixes is not None and cited.fixes.source_spec_id == pin:
                return None
        return pin

    def _stale_citation(requirement: EvidenceRequirement, slot: str) -> bool:
        """Whether `slot` is a decision this function wrote for a pin `requirement` no longer states."""
        cited = known.get(slot)
        return (
            isinstance(requirement, DatasetRequirement)
            and cited is not None
            and is_search_pin(cited)
            and cited.fixes is not None
            and cited.fixes.source_spec_id != requirement.source_spec_id
        )

    synthesized: list[Decision] = []
    patched_requirements: list[EvidenceRequirement] = []
    for stated in evidence_requirements:
        kept = tuple(slot for slot in stated.decision_ids if not _stale_citation(stated, slot))
        requirement = (
            stated
            if kept == stated.decision_ids
            else stated.model_copy(update={"decision_ids": kept})
        )
        pin = _uncited_pin(requirement)
        if pin is None:
            patched_requirements.append(requirement)
            continue
        slot = search_pin_slot(pin)
        patched_requirements.append(
            requirement.model_copy(update={"decision_ids": (*requirement.decision_ids, slot)})
        )
        if slot in known:
            continue
        known[slot] = Decision(
            slot=slot,
            description="Whether to pin the source a search settled on.",
            options=(_pin_choice(pin), _UNPINNED_CHOICE),
            chosen=_pin_choice(pin),
            resolved_by=ResolvedBy.PLANNER_DEFAULT,
            rationale=(
                "A fresh search found this source in place of the one pinned before."
                if pin in refound_spec_ids and pin not in settled_spec_ids
                else "A search in this conversation settled on this source for the measure."
                if pin in earlier_spec_ids
                else "A search this turn settled on this source for the measure."
            ),
            fixes=DecisionFixes(source_spec_id=pin),
        )
        synthesized.append(known[slot])
    return [*decisions, *synthesized], patched_requirements


def _settled_decisions(
    drafts: Sequence[DecisionDraft],
    requirements: Sequence[EvidenceRequirement],
    *,
    existing: Plan | None,
    requirements_after: Sequence[EvidenceRequirement],
    tool_context: ToolContext,
) -> list[Decision] | dict[str, object]:
    """This call's decisions built from `drafts`, or the refusal that stops the call first.

    These checks run before the plan is touched, so a refusal records nothing
    and spends no round: a referral the user has not answered
    (`_referral_unanswered`), a draft fixing a field its requirements do not
    settle (`planner.plan.settle_decisions`), a pin on a dataset the turn
    dropped (`_dropped_seed_pinned`), and a user election the user never made
    (`_election_untraceable`).

    Args:
        drafts: This call's `decisions` argument.
        requirements: This call's own requirements.
        existing: The plan recorded before this call.
        requirements_after: Every requirement the plan states once the call
            lands (`_requirements_after`), which a draft's fields are read off.
        tool_context: The call's tool context.
    """
    blocked = _referral_unanswered(existing, tool_context)
    if blocked is not None:
        return blocked
    try:
        decisions = settle_decisions(drafts, requirements_after)
    except UnsettledFixError as exc:
        return _fix_unsettled(exc)
    return (
        _dropped_seed_pinned(decisions, requirements, tool_context)
        or _election_untraceable(decisions, tool_context)
        or decisions
    )


def _uncited_source_decisions(
    previous: Plan | None,
    decisions: Sequence[Decision],
    requirements: Sequence[EvidenceRequirement],
    *,
    removed: Collection[str],
) -> list[str]:
    """The slot of each decision `_synthesize_source_decisions` wrote that no requirement of the patched plan cites.

    Such a decision exists only because a requirement pinned its spec_id, so
    when a patch unpins, re-pins or removes every requirement citing it, it
    goes with them. The model did not write it and would not remove it by
    name. A slot this call restates or removes is left as the call says.

    Args:
        previous: The plan this call amends, or `None`.
        decisions: This call's own decisions, synthesized ones included.
        requirements: Every requirement the patched plan states.
        removed: The slots this call removes.
    """
    if previous is None:
        return []
    restated = {decision.slot for decision in decisions} | set(removed)
    cited = {slot for requirement in requirements for slot in requirement.decision_ids}
    return [
        decision.slot
        for decision in previous.decisions
        if is_search_pin(decision) and decision.slot not in restated and decision.slot not in cited
    ]


def _requirements_after(
    previous: Plan | None,
    requirements: Sequence[EvidenceRequirement],
    removed: Collection[str],
) -> list[EvidenceRequirement]:
    """Every requirement the plan states once this call's patch lands, in `Plan.amend`'s order.

    A removal naming no requirement is left for `Plan.amend` to refuse.
    """
    by_id = {
        requirement.id: requirement
        for requirement in (previous.evidence_requirements if previous else ())
    }
    by_id.update({requirement.id: requirement for requirement in requirements})
    return [
        requirement
        for requirement_id, requirement in by_id.items()
        if requirement_id not in removed
    ]


def _fix_unsettled(exc: UnsettledFixError) -> dict[str, object]:
    """Return the refusal for a decision whose `fixes` names a field its requirements do not settle."""
    return problem(
        ProblemCode.FIX_DISAGREES if len(exc.stating) > 1 else ProblemCode.FIX_UNSTATED,
        str(exc),
        outcome=Outcome.INVALID_INPUT,
        detail={
            "decision": exc.slot,
            "field": exc.field,
            "citing": list(exc.citing),
            "stating": list(exc.stating),
        },
    )


# What each `propose_plan` argument means, as the model reads it on the
# argument itself (`planner.declaration.PlanTool`). `question` and
# `starts_new_plan` read differently under `question_fixed`, the way
# `_resolve_intent` enforces them; `_FIXED_ARGUMENTS` replaces those two.
_PROPOSE_PLAN_ARGUMENTS = {
    "question": (
        "The question this plan answers. Required on a first draft and with "
        "`starts_new_plan`, which starts at version 1 under it. A patch that "
        "states one records it; one that omits it keeps the recorded question."
    ),
    "answer_mode": (
        "Required on a first draft and with `starts_new_plan`. A patch that "
        "states one switches the plan's mode and re-validates its requirements "
        "under it; one that omits it keeps the recorded mode."
    ),
    "decisions": (
        "The decisions this call adds or replaces, keyed by `slot`: a slot the "
        "plan already states replaces that decision, and a new one adds it. A "
        f"first draft states every decision the plan makes, at most "
        f"{MAX_PLAN_DECISIONS}. Every decision is cited by at least one "
        "requirement."
    ),
    "evidence_requirements": (
        "The requirements this call adds or replaces, keyed by `id`, unique "
        "within the plan and tagged by `kind`: an id the plan already states "
        "replaces that requirement whole, so restate every field it keeps, "
        "its entities' roles included, and a new id adds one. A derived "
        "requirement names the requirement it reads in `input_id`. Each cites "
        "in `decision_ids` the decisions that fixed its fields, and none when "
        "the question or your instruction states them outright. A plan states "
        f"at most {MAX_PLAN_REQUIREMENTS} requirements."
    ),
    "removed_decisions": (
        "The `slot` of each decision this call deletes. Leaving a decision out of a patch keeps it."
    ),
    "removed_evidence_requirements": (
        "The `id` of each requirement this call deletes. Leaving a requirement "
        "out of a patch keeps it."
    ),
    "declined": (
        "Each part of the question no operation computes, one per `code` and "
        "`ask_ref`. Only an `unsupported` answer earns a decline, with that "
        "answer's code: one from `propose_plan` or `translate`, or a run-time "
        "refusal the plan is replanned from that names the code."
    ),
    "answered": (
        "When the brief lists asks, the requirements that answer each one, "
        "keyed by `ask`. Every ask is answered here or declined in `declined` "
        "with its number as `ask_ref`, exactly once, or both when the plan "
        "computes part of it."
    ),
    "amends_version": (
        "The `version` this call patches, as the last `propose_plan` call "
        'answered it and as the brief\'s own "version N" line states it. '
        "Required once a plan is recorded, unless `starts_new_plan` is set: "
        "a call stating neither is refused, and `question` never stands in "
        "for the choice."
    ),
    "starts_new_plan": (
        "True begins a plan of its own, at version 1, for a different question "
        "while a plan is recorded."
    ),
}

_FIXED_ARGUMENTS = {
    "question": (
        "The question this plan answers. Required on a first draft. This "
        "turn's question is fixed, so a patch either omits it or restates it "
        "unchanged: it has to match the one already recorded."
    ),
    "starts_new_plan": (
        "True says the user's reply asks a different question from the one "
        "fixed for this turn. Refused, recording nothing: the harness ends the "
        "turn and tells the user to send that question as its own message."
    ),
}


def build_planner_tools(
    *,
    has_translator: bool = True,
    question_fixed: bool = False,
) -> list[FunctionTool]:
    """Build the planner's tool set.

    `has_translator` says whether a translator is attached to the agent these
    tools belong to. A refusal names the tool to call next, and naming one the
    agent does not have sends the model at a tool ADK raises on.

    `question_fixed` closes over into `propose_plan`, which passes it straight
    to `_resolve_intent` on every call, and selects `_FIXED_ARGUMENTS`'
    descriptions of `question` and `starts_new_plan`, so those two arguments
    describe the mode this tool set actually enforces.
    """

    # Named `ask_user` here, for readability, but renamed below (once
    # defined) to `adk_request_input` -- the function-call name ADK's own
    # `google.adk.tools.request_input` uses for exactly this pausing shape
    # (`REQUEST_INPUT_FUNCTION_CALL_NAME`), reused rather than duplicated so
    # `harness.studio_translate._request_input_call` recognises this pause
    # the same way it already recognises the graph's own `clarify`/`approve`
    # interrupts, with no change to that recognition code. The model sees
    # and calls this tool under that reused name, not `ask_user`, so the
    # instruction above names it that way too.
    def ask_user(
        message: str,
        candidates: list[dict[str, str]] = [],  # noqa: B006
        recommended_spec_id: str | None = None,
        rationale: str | None = None,
    ) -> dict[str, object] | None:
        """Ask the user a question a referral leaves to them, pausing the turn.

        Returns nothing here: the user's answer arrives as this call's own
        response on your next turn, naming the `spec_id` they picked, or free
        text if they answered outside the candidates. A `resolved_by=user`
        decision may pin only a `spec_id` this call offered and the user
        picked.

        Args:
            message: The question to show the user, naming what is
                ambiguous and, briefly, what each candidate is, in plain
                words: never a requirement id, spec_id or code.
            candidates: The referral's own candidates, each
                `{"spec_id": ..., "label": ...}` -- `label` is the
                candidate's own `sentence`, not invented text. Omit it for
                a referral that carries none.
            recommended_spec_id: The candidate you favour, when you favour
                one -- must name one of `candidates`.
            rationale: Why you favour it, for the user to read, when
                `recommended_spec_id` is given.
        """
        del message, rationale  # carried through to the wire frame, not read back here
        offered = [str(c["spec_id"]) for c in candidates if c.get("spec_id")]
        if recommended_spec_id and recommended_spec_id not in offered:
            return problem(
                ProblemCode.INVALID_ARGUMENTS,
                f"recommended_spec_id {recommended_spec_id!r} names no candidate in {offered!r}",
                outcome=Outcome.INVALID_INPUT,
            )
        return None

    # `= []` rather than `None`. ADK builds this tool's declaration from the
    # signature, and an optional list declared `list[X] | None` reaches the
    # model as `anyOf: [array, null]`, which is not the flat array a patch is
    # written into. Nothing mutates these defaults.
    def propose_plan(
        tool_context: ToolContext,
        question: str | None = None,
        answer_mode: AnswerMode | None = None,
        decisions: list[DecisionDraft] = [],  # noqa: B006
        evidence_requirements: list[EvidenceRequirement] = [],  # noqa: B006
        removed_decisions: list[str] = [],  # noqa: B006
        removed_evidence_requirements: list[str] = [],  # noqa: B006
        declined: list[Decline] = [],  # noqa: B006
        answered: list[AskAnswer] = [],  # noqa: B006
        amends_version: int | None = None,
        starts_new_plan: bool = False,
    ) -> dict[str, object]:
        """Record a plan, or patch the one already recorded.

        Returns `ok` with `status` `"draft"`; the new `version`, which a
        patch passes as `amends_version` and `submit_plan` takes; the
        `previous_version` it amended, `None` on a first draft; and the
        `question` as stored.
        """
        try:
            # ADK's arg preprocessing silently falls back to raw dicts for the
            # *whole* list if any single item fails validation (e.g. a
            # Decision's `chosen` not in `options`), and skips conversion
            # entirely for a value that isn't a list at all. Re-validating here
            # turns either case into a retryable tool problem, keeping
            # unvalidated dicts out of a Plan and the exception out of the turn.
            patch = _Patch.model_validate(
                {
                    "decisions": decisions,
                    "evidence_requirements": evidence_requirements,
                    "removed_decisions": removed_decisions,
                    "removed_evidence_requirements": removed_evidence_requirements,
                    "declined": declined,
                    "answered": answered,
                }
            )
        except ValidationError as exc:
            return _invalid_data(exc)
        evidence_requirements = patch.evidence_requirements
        removed_decisions = patch.removed_decisions
        removed_evidence_requirements = patch.removed_evidence_requirements
        declined = patch.declined
        answered = patch.answered
        existing = load_plan(tool_context.state)
        seeded = seeded_sources(tool_context.state)
        resolved = _resolve_intent(
            existing,
            amends_version=amends_version,
            starts_new_plan=starts_new_plan,
            rounds_so_far=load_plan_rounds(tool_context.state),
            question=question,
            question_fixed=question_fixed,
        )
        if isinstance(resolved, dict):
            return resolved
        previous, rounds = resolved
        held_copies = windowed_pins(tool_context.state)
        to_origin = {
            **{copy.spec_id: copy.windowed_from for copy in held_copies.values()},
            **{item.source.spec_id: item.windowed_from for item in seeded if item.windowed_from},
        }
        if to_origin:
            _, evidence_requirements = _unwindowed_pins((), evidence_requirements, to_origin)
        if previous is not None and to_origin:
            unwindowed, unwindowed_requirements = _unwindowed_pins(
                previous.decisions, previous.evidence_requirements, to_origin
            )
            previous = previous.model_copy(
                update={
                    "decisions": tuple(unwindowed),
                    "evidence_requirements": tuple(unwindowed_requirements),
                }
            )
        settled = _settled_decisions(
            patch.decisions,
            evidence_requirements,
            existing=existing,
            requirements_after=_requirements_after(
                previous, evidence_requirements, removed_evidence_requirements
            ),
            tool_context=tool_context,
        )
        if isinstance(settled, dict):
            return settled
        # A verbatim pin of a spec_id a search settled on gets the citing
        # `Decision` it needs written here, on the model's behalf, so
        # `Plan`'s own validator never has to refuse it for lacking one.
        # See `_synthesize_source_decisions`.
        pins = pin_sources(tool_context.state)
        settled, evidence_requirements = _synthesize_source_decisions(
            settled,
            evidence_requirements,
            previous=previous,
            settled_spec_ids={spec_id for spec_id, source in pins.items() if source.settled},
            earlier_spec_ids={spec_id for spec_id, source in pins.items() if source.earlier_turn},
            refound_spec_ids=_refound_spec_ids(tool_context),
        )
        removed_decisions = [
            *removed_decisions,
            *_uncited_source_decisions(
                previous,
                settled,
                _requirements_after(previous, evidence_requirements, removed_evidence_requirements),
                removed=removed_decisions,
            ),
        ]
        try:
            if previous is not None:
                plan = previous.amend(
                    question=question,
                    answer_mode=answer_mode,
                    decisions=settled,
                    evidence_requirements=evidence_requirements,
                    removed_decisions=removed_decisions,
                    removed_evidence_requirements=removed_evidence_requirements,
                    declined=declined,
                    asks=load_asks(tool_context.state),
                    answered=answered,
                )
            elif question is None or answer_mode is None:
                missing = " and ".join(
                    clause
                    for clause, absent in (
                        ("the `question` it answers", question is None),
                        ("the `answer_mode` it answers under", answer_mode is None),
                    )
                    if absent
                )
                raise ValueError(
                    f"a first draft states {missing}"
                    if existing is None
                    else f"a `starts_new_plan` call states {missing}, for its new plan"
                )
            else:
                plan = Plan(
                    question=question,
                    answer_mode=answer_mode,
                    decisions=settled,
                    evidence_requirements=evidence_requirements,
                    declined=declined,
                    asks=load_asks(tool_context.state),
                    answered=answered,
                )
            plan, windows, copies = _window_pins(plan, seeded)
        except (ValidationError, ValueError) as exc:
            # The plan-level rules: unique slots and ids, every citation naming
            # a decision, every decision cited,
            # the length and count limits on each field, a question that is not
            # blank, and the size limit on the whole. A removal naming an item
            # the plan does not state is a `ValueError` from `Plan.amend` itself,
            # and a first draft or restart with no `question`/`answer_mode` is
            # one raised above. `_window_pins` raises one for a decision
            # fixing two seeded pins over different periods.
            cause = _unsupported_cause(exc)
            if cause is not None:
                record_unsupported(tool_context.state, cause.operation)
            return _invalid_data(exc)
        blocked = _unrecordable(plan, tool_context, previous=previous)
        if blocked is not None:
            return blocked
        if seeded:
            # The seeds and the copies this plan pins. A copy that only a
            # replaced plan pinned is let go, so however many drafts and
            # amendments the turn makes, the state holds at most one copy per
            # dataset requirement of the recorded plan.
            save_seeded_sources(
                tool_context.state,
                [*(item for item in seeded if item.windowed_from is None), *windows],
            )
        if copies or held_copies:
            # Likewise the copies of pins no seed names, which hold only those this plan pins.
            save_windowed_pins(tool_context.state, copies)
        save_plan(tool_context.state, plan)
        save_plan_rounds(tool_context.state, rounds)
        # A new proposal supersedes whatever the translator referred back.
        clear_referral(tool_context.state)
        # Only what the model cannot read back off its own call, and the
        # question as stored. The decisions and requirements the model sent are
        # echoed nowhere, because the plan is in session state and
        # `translate` reads it from there. `status` is always the
        # literal `"draft"` this call just wrote. `version` and its
        # predecessor are what the call did not already state.
        # `Plan.amend` bumps both and returns to `DRAFT`, and they are how
        # a later `submit_plan` version check is reasoned about.
        return ok(
            status=plan.status.value,
            version=plan.version,
            previous_version=plan.previous_version,
            question=plan.question,
        )

    def reject_plan(tool_context: ToolContext) -> dict[str, object]:
        """Record that the user rejected the current plan.

        Nothing in the plan may run afterwards: `submit_plan` refuses a
        rejected plan. Call `propose_plan` again if the user wants a
        revision rather than an outright stop.
        """
        plan = load_plan(tool_context.state)
        if plan is None:
            return problem(ProblemCode.PLAN_MISSING, _NO_PLAN, outcome=Outcome.STATE_MISSING)
        try:
            plan = plan.reject()
        except ValueError as exc:
            # A run that has started or finished cannot be withdrawn.
            return problem(
                ProblemCode.TRANSITION_REFUSED,
                str(exc),
                outcome=Outcome.REFUSED,
                detail={"status": plan.status.value},
            )
        save_plan(tool_context.state, plan)
        clear_referral(tool_context.state)
        return ok(status=plan.status.value)

    def submit_plan(version: int, tool_context: ToolContext) -> dict[str, object]:
        """Declare the current plan's compiled workflow ready for the user's answer.

        Approves nothing and asks nothing of the user itself: whatever runs
        this agent is what actually gates on their answer, keyed on the
        revision this call returns alongside the summary. Blocked the same
        way `_execution_blocked` blocks a plan that has been rejected, is
        awaiting a referral, has not yet compiled, or has been amended past
        the version given.

        Args:
            version: The plan version being submitted. Must match the
                current plan's version -- if the plan was amended since
                this version was reviewed, submission is blocked.
        """
        plan = load_plan(tool_context.state)
        if plan is None:
            return problem(ProblemCode.PLAN_MISSING, _NO_PLAN, outcome=Outcome.STATE_MISSING)
        blocked = _execution_blocked(plan, version, tool_context, has_translator=has_translator)
        if blocked is not None:
            return blocked
        # The submitted plan ends the run. The harness asks the user, so no
        # model call follows to restate the plan the user already sees.
        tool_context.actions.skip_summarization = True
        return ok(
            status=plan.status.value,
            version=plan.version,
            revision=plan_revision(plan),
            summary=_approval_summary(plan),
        )

    ask_user.__name__ = REQUEST_INPUT_FUNCTION_CALL_NAME
    return [
        LongRunningFunctionTool(ask_user),
        PlanTool(
            propose_plan,
            arguments=(
                _PROPOSE_PLAN_ARGUMENTS | _FIXED_ARGUMENTS
                if question_fixed
                else _PROPOSE_PLAN_ARGUMENTS
            ),
        ),
        FunctionTool(reject_plan),
        FunctionTool(submit_plan),
    ]


def create_planner_agent(
    *,
    model_name: str,
    translate: FunctionTool | None = None,
    surface: ToolSurface | None = None,
    question_fixed: bool = False,
    prefix_cache: PrefixCache | None = None,
) -> LlmAgent:
    """Build the planner `LlmAgent`.

    Args:
        model_name: The model this agent runs on.
        translate: The `translate` tool (`translator.tool.build_translate_tool`),
            which is how a draft plan gets translated and compiled. Omitted,
            the planner is the standalone agent its own eval set scores --
            which is also what keeps that set from needing a live tool
            surface.
        surface: The tool surface `get_skill` and `get_skill_file` call
            data-mcp through. Omitted, the planner has neither tool and its
            instruction does not name them.
        question_fixed: Whether the question this planner drafts for is fixed
            for the entry it is called in. The harness graph's own seam sets
            this when it builds the planner (`harness.planner_seam`), since the
            graph derives and narrates a turn's `resolve_envelope`, `depth`
            and `question` before the planner ever runs, and a restart would
            silently swap the question out from under that state -- see
            `_resolve_intent`. False, the default, is the standalone
            planner's and `harness_pair_v1`'s own behaviour: a restart begins
            a fresh plan.
        prefix_cache: The process's shared cache of static prompt prefixes.
            Given, and `model_name` naming a Gemini model, every call reads
            the instruction and tool declarations from it. The model keeps
            ADK's single attempt either way, so a transient error ends the
            generation inside the entry's deadline rather than spending it
            on retries.

    Returns:
        The configured `LlmAgent`.
    """
    return LlmAgent(
        name=AGENT_NAME,
        model=model_name if prefix_cache is None else prefix_cache.model_for(model_name),
        instruction=planner_instruction(
            has_translator=translate is not None,
            has_skills=surface is not None,
            question_fixed=question_fixed,
        ),
        tools=[
            *(build_skill_tools(surface) if surface is not None else []),
            *build_planner_tools(
                has_translator=translate is not None, question_fixed=question_fixed
            ),
            *([translate] if translate is not None else []),
        ],
        # Run as a workflow node (`harness.planner_seam.PlannerSeam._propose`'s
        # `ctx.run_node`), ADK defaults an agent with no explicit `mode` to
        # `single_turn`, and a `single_turn` agent with no explicit
        # `include_contents` to `'none'` (`run_llm_agent_as_node`,
        # `google.adk.workflow._llm_agent_wrapper`). `'none'` builds every
        # request from `_get_current_turn_contents`, which slices the session
        # from the latest `author='user'` event. Every `_propose` call opens
        # on a user message of its own, and a continuing one (a correction
        # round, an amendment, a replan) carries only what is new, so that
        # slice would drop the brief and every earlier tool call the
        # continuing message relies on. Pinning `'default'` here uses
        # `_get_contents` instead, which reads the whole history on this
        # agent's branch.
        include_contents="default",
        # `AGENT_THINKING_CONFIG` (see `planner.settings`) pins every
        # generation to LOW thinking.
        planner=BuiltInPlanner(thinking_config=AGENT_THINKING_CONFIG),
        # Drops this generation's thought-summary parts once the trace has
        # them, so `include_contents="default"` above never replays one into
        # a later request of the same turn. See `drop_thought_parts`.
        after_model_callback=drop_thought_parts,
    )
