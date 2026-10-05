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
compiling the workflow belong to `translator.agent`, which the planner calls
as a tool before the user ever sees the plan -- `create_planner_agent` takes
that agent as a sub-agent, and `harness.agents` builds the pair. So no tool
name, column or operation appears here, `PENDING_APPROVAL` is a status only
the translator writes, and `submit_plan` requires a plan that has been
through that step.

Every tool below answers with `planner.tool_results`' typed shape -- `ok`,
`problem`, with a stable `code` and, where the next step is known, an
`action` -- rather than an unstructured `{"error": ...}` a caller has to
parse. When the translator cannot carry a plan out, its `dfo_translator`
tool call answers with the `Referral` itself, structured by
`translator.agent.create_translator_agent`'s own `output_schema`: this
agent's instruction reads it straight off that result, with nothing here
to fetch it back a second time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from compiler.surface import ToolSurface
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from pydantic import BaseModel, Field, ValidationError
from pydantic_core import to_jsonable_python
from translator.agent import AGENT_NAME as TRANSLATOR_AGENT_NAME
from translator.referral import ReferralKind, clear_referral, load_referral

from planner.plan import (
    ENVELOPE_SPEC_STATE_KEY,
    MAX_PLAN_DECISIONS,
    MAX_PLAN_REQUIREMENTS,
    MAX_PLAN_ROUNDS,
    AnswerMode,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    EvidenceRequirement,
    Plan,
    PlanStatus,
    ResolvedBy,
    SeededSource,
    cut_user_text,
    load_plan,
    load_plan_rounds,
    names_dropped_seed,
    plan_revision,
    save_plan,
    save_plan_rounds,
    save_seeded_sources,
    seeded_sources,
    windowed_copy,
)
from planner.records import Identifier, Period
from planner.settings import AGENT_THINKING_CONFIG, drop_thought_parts
from planner.skills import DFO_FAQ_SKILL_ID, build_skill_tools
from planner.tool_results import Action, Outcome, ProblemCode, ok, problem

AGENT_NAME = "dfo_planner"


class _Patch(BaseModel):
    """The four lists one `propose_plan` call states, checked before they reach a `Plan`.

    ADK hands `propose_plan` raw values wherever it could not convert them, and
    builds the tool's declaration from the signature with every `Annotated`
    limit stripped, so the count and shape of each list is enforced here. A
    removal names a slot or an id, so it is held to `Identifier`; each list
    holds no more items than a plan does.
    """

    decisions: Annotated[list[Decision], Field(max_length=MAX_PLAN_DECISIONS)]
    evidence_requirements: Annotated[
        list[EvidenceRequirement], Field(max_length=MAX_PLAN_REQUIREMENTS)
    ]
    removed_decisions: Annotated[list[Identifier], Field(max_length=MAX_PLAN_DECISIONS)]
    removed_evidence_requirements: Annotated[
        list[Identifier], Field(max_length=MAX_PLAN_REQUIREMENTS)
    ]


_PLANNING = (
    "You are the Planner Agent for the Development Finance Observatory. "
    "Given a development finance question, draft a methodological plan: "
    "state each real decision (measure, scope, time window when "
    "the resolve envelope does not settle it, and so on), the "
    "alternatives you considered, and a one-line reason for your choice. "
    "A decision is recorded only where you genuinely weighed more than one "
    "alternative; a value the question already states outright -- the "
    "entity it names, the measure it names -- is a fact for a requirement's "
    "own fields to carry, and needs no `Decision` entry at all. A "
    "`time_window_defaulted` warning in the resolve envelope means the "
    "question gave no time text; the resolver has already settled the "
    "answer's coverage, so that is not a decision -- never weigh candidate "
    "ranges, never call `adk_request_input` over it, and never invent a "
    "specific range from memory. When the envelope carries a `window`, "
    "state the dataset requirement's `period` from it verbatim -- `start` "
    "and `end` exactly as given, the coverage the resolver already found -- "
    "and cite no decision that fixes it. When it carries none, the period "
    "is a decision like any other. Source is different: when the envelope "
    "names a `spec_id`, pin it as the `source_spec_id` of the plan's one "
    "dataset requirement for that measure, verbatim, and cite no decision "
    "for the pin itself -- the plan records why on its own. Otherwise the "
    "source is a decision like any other. If `propose_plan` rejects a "
    "decision for having only one "
    "option, delete that decision in your next call rather than adding a "
    "second option you did not actually consider. "
    "Ask the user directly only when a choice is "
    "genuinely ambiguous, settling other defaults the way a domain expert "
    "would. "
    "State what the answer rests on as evidence requirements: one dataset "
    "requirement per measure and source the answer needs, naming every "
    "entity the answer needs for that measure, and one derived requirement "
    "per calculation the answer needs -- `aggregate`, `growth`, "
    "`comparison`, `rank` (see below), `share_of` (see below), `join`, or "
    "`filter` -- each citing the decision that fixed it and naming the "
    "requirements it reads. Where the question or measure identifies an "
    "entity's party, set its `role`, such as `'recipient'`/`'donor'`, "
    "`'debtor'`/`'creditor'` or `'importer'`/`'exporter'`. Use party names, "
    "never directions such as `'from'` or `'to'`. Otherwise leave `role` "
    "unset. Carry roles forward in patches unless the amendment changes or "
    "removes the relationship. When a flow source spans a party the question "
    "leaves unnamed (aid from named donors to every recipient) and the "
    "measure is an additive amount, add an `aggregate` with "
    "`reduction='sum'` grouped by the named party and period before any "
    "`growth`, `rank` or `comparison`: it collapses the unnamed party alone "
    "and keeps every entity the question ranks, so each named entity has one "
    "value per year. A rate, index or per-person figure is never summed. "
    "When the question ranks a party it names no "
    'member of, such as "top donors" or "top 10 countries", or its answer is '
    "that party's total, state the party in `across` and name no stand-in "
    "entity such as `'World'`. `across` is the party word alone, such as "
    "`'donor'`, `'recipient'` or `'country'`, and covers every member the "
    "source holds: for a narrower population, such as a region, an income "
    "group or a donor type, name its members in `entities` instead, and name "
    "no entity in the party `across` covers. A world figure set beside a "
    "named entity still names `'World'`. The envelope's `open_entity_axes` "
    "lists the columns left open: `origin` is the giving party, "
    "`destination` the receiving party, `observationAbout` the place. "
    "A requirement whose `across` is the place, such as `'country'`, names "
    "no place, and its retrieval holds every place with data at every "
    "level: the world, regions, World Bank groups, countries and their "
    "subdivisions. Before a `rank`, an `aggregate` across places or a "
    "`share_of` reads it, keep one level with a `filter` reading that "
    "requirement directly, whose predicate is `place_level`, such as "
    '`{"kind": "place_level", "level": "Country"}`. Take the level the '
    'question names ("top 10 countries"). When it names none and reads as '
    'a comparison of countries ("top 10 by GDP"), choose `Country` in a '
    "`Decision` resolved by `planner_default` whose `fixes.predicates` "
    'states it. When its word covers several levels ("by region" is '
    "continents or subdivisions), ask the user. A question "
    "naming its places names them in `entities` and needs no level. State "
    "`change='absolute'` on a `growth` "
    "whenever its input is itself a rank position, since a rank moves by "
    "places, not by a percentage. Several entities of one measure are one "
    "dataset requirement, and that requirement can itself be the answer "
    "when the question asks to see the series -- the frame it names carries "
    "one series per entity, which is what a chart draws. Reach for "
    "`comparison` only when the question asks for a difference or a ratio, "
    "and for `join` only across different measures, since neither keeps the "
    "separate series a chart of one measure over several entities shows. "
    "A family of StatVars is read summed or separately: see `stat_vars`. A "
    "derived requirement's figures are what a question about a value, a "
    "growth, a share or a rank asks for, and a comparison feeds "
    "nothing further; it answers the question outright. Decide the "
    "methodology and leave the mechanics alone: which data and which "
    "operations carry your decisions out is worked out separately, after "
    "you propose, so name no tool. "
    "A rank naming `among='named'` needs a visible `filter` step narrowing "
    "to those entities first; `among='population'` is a claim over "
    "entities nobody named, so cite a `Decision` whose `fixes.among` "
    "states it. State `per_period` when the question asks how a position "
    "changed over time rather than who is first; a rank over it is an "
    "ordinal series, not a value, so growing it needs a `filter` naming "
    "the one entity the question is about, placed between the rank and "
    "the growth -- the series carries the whole population, and a growth "
    "over everyone's position at once has no defined answer. State a rank's "
    "`limit` when the question asks for a top or bottom N; with `per_period` "
    "it keeps each period's top or bottom N at their measured values, not "
    "their positions, and each entity only in the years it made the cut, so "
    "a growth or a per-entity `share_of` goes before that rank, and a top N "
    "by each entity's total is one rank stating the `reduction` and the "
    "`limit` with `per_period` unset. No rank reads another rank's output: "
    "put the `limit` on the one rank that reads the measure. A `share_of` "
    "cites a `Decision` whose `fixes.denominator_scope` states `pooled` "
    "whenever it uses that scope, the same population claim a rank's "
    "`among='population'` makes. "
    "Whenever a `Decision` settles a field one of these calculations reads "
    "-- `reduction`, `group_by`, a window, `direction`, `predicates`, "
    "`among`, `axis`, `denominator_scope`, `change`, `per_period`, or "
    "anything else a requirement states -- record that in the decision's "
    "own `fixes`, naming the field and the value it settles. A decision "
    "with nothing in `fixes` is a rationale a reviewer reads; one with "
    "`fixes` set is checked automatically against every requirement that "
    "cites it, so state it whenever what was decided is a field a "
    "requirement carries -- `among='population'` and "
    "`denominator_scope='pooled'` need it to validate at all. "
)

# What each `propose_plan` call states about which plan it means, and how a
# patch's own `question` is read. Held apart from `_PLANNING` because it is
# the one paragraph `question_fixed` rewrites: standalone and `harness_pair_v1`
# keep `_VERSIONING_FREE`, where a restart is a normal way to begin a second
# plan; the harness graph's own seam builds with `question_fixed=True`, where
# the turn's question is the graph's, not the planner's, to change.
_VERSIONING_FREE = (
    "Call `propose_plan` once with the complete first draft, and again "
    "whenever the user asks for a change -- every call produces a new plan "
    "version. Once a plan is recorded, every call states which of the two it "
    "is: pass `amends_version` naming the version `propose_plan` answered "
    "with to patch that plan, or `starts_new_plan=true` when the ask is a "
    "different question that needs a plan of its own. A call stating neither "
    "is refused, so decide deliberately; never let the wording of `question` "
    "stand in for the choice. A patch states only the decisions and "
    "requirements that change, matched by `slot` and by `id`, and everything "
    "you leave out is carried forward. Nothing is deleted by leaving it out, "
    "so name a decision in `removed_decisions` or a requirement in "
    "`removed_evidence_requirements` to delete it. "
)

_VERSIONING_FIXED = (
    "Call `propose_plan` once with the complete first draft, and again "
    "whenever the user asks for a change -- every call produces a new plan "
    "version. This turn's question is fixed: once a plan is recorded, pass "
    "`amends_version` naming the version `propose_plan` answered with, to "
    "patch that plan, and a patch's own `question`, if it states one, has to "
    "match the one already recorded -- it cannot be changed here. Pass "
    "`starts_new_plan=true` instead only to say the user's reply asks a "
    "different question: that call is refused and records nothing, and ends "
    "the turn there -- the harness tells the user to send the other question "
    "as its own message, so nothing further is needed from you once you have "
    "made it. A call stating neither `amends_version` nor `starts_new_plan` "
    "is refused, so decide deliberately. A patch states only the decisions "
    "and requirements that change, matched by `slot` and by `id`, and "
    "everything you leave out is carried forward. Nothing is deleted by "
    "leaving it out, so name a decision in `removed_decisions` or a "
    "requirement in `removed_evidence_requirements` to delete it. "
)

# How to read skills, when the skill tools are built. Held apart because the
# planner is also built without them, and an instruction naming a tool the
# agent lacks sends the model at one ADK raises on.
_SKILLS = (
    "The resolve envelope's `skills` names the skills that apply to its "
    "results, each by id with a description. A skill carries the "
    "definitions, caveats and methods an answer in its domain rests on. When "
    "a description bears on the question, call `get_skill` with its id "
    "before you draft, and let what it says shape your decisions. Read a "
    "file the skill links with `get_skill_file`, passing the path exactly "
    "as `get_skill` lists it. Both answer in plain text, a refusal "
    "included. A question about DFO itself (what it covers, where its data "
    "comes from, how it works) starts with "
    f'`get_skill("{DFO_FAQ_SKILL_ID}")`, a skill no search result names. When a skill '
    "tool refuses, `capability_unavailable` included, draft the plan "
    "without that skill. "
)

# What to do once a plan is recorded, when a translator is attached to carry
# it out. Held apart because the planner is also built standalone -- its own
# eval set scores it that way -- and a standalone planner that is told to call
# `dfo_translator` calls a tool that is not there. ADK raises on an unknown
# tool name rather than answering the model, so the run dies mid-turn.
_TRANSLATING = (
    "Once `propose_plan` has recorded the plan, call `dfo_translator` with a "
    'short `request` naming what you want it to do, such as "translate the '
    'plan you have recorded" -- `request` is your instruction to it, not a '
    "copy of the plan, and it is what the translator answers you on. "
    "`dfo_translator` reads the recorded plan itself, from session state, so "
    "never retype the plan's `question`, `decisions` or "
    "`evidence_requirements` into `request`: it already has them as "
    "`propose_plan` recorded them, and a copy you type by hand can differ "
    "from them. Once a referral is settled, by your own decision or the "
    "user's answer, propose with the `amends_version` the last "
    "`propose_plan` call answered with, so the patch lands on that plan. "
    "`dfo_translator`'s answer "
    'is either `{"outcome": "ok"}`, meaning the plan compiled and no '
    "referral is pending, or the referral itself. A compiled answer means: "
    "call `submit_plan`, then invite the user's approval in one or two "
    "sentences -- the plan itself is on their screen, so do not restate its "
    "decisions, requirements, sources or steps. Otherwise the referral "
    "is a source decision to settle. When the question itself states that "
    "the user elected one of its candidates for this requirement, record "
    "that as a `Decision`: `resolved_by=user`, `fixes.source_spec_id` set to "
    "the elected candidate's id, and `options` listing every candidate the "
    "referral gave you, not only the one chosen. When the referral carries a "
    "`recommendation`, read it first: its `spec_id` names the candidate to "
    "prefer, and its `rationale` is what your own choice should rest on, "
    "rather than one you invent -- it is the translator's own grounded "
    "reading of what discovery found, not a pick made for you. It is "
    "advice, not an instruction: choose a different candidate when your "
    "own reading of the question calls for it, and say so plainly, with "
    "your own reason for departing from what was recommended. Settle "
    "the referral yourself when the recommendation (or your own reading) "
    "makes the choice unambiguous: record a `Decision` resolved by "
    "`planner_default`, with the same `options`, `fixes.source_spec_id` set "
    "to the candidate you chose, and a rationale that rests on the "
    "recommendation when you follow it, or states your own reason when you "
    "don't. Call `adk_request_input` instead whenever it is genuinely "
    "ambiguous even with the recommendation in hand -- offer it the "
    "referral's own `candidates` (each `spec_id` and a `label`, the "
    "candidate's own `sentence`) and, when there is one, the "
    "recommendation's `spec_id` and `rationale`. Its answer names the "
    "`spec_id` the user picked; record that as a `Decision` the same way, "
    "`resolved_by=user`, `fixes.source_spec_id` set to what they picked, "
    "`options` listing every candidate `adk_request_input` offered -- a "
    "`resolved_by=user` source pin is checked against what the user actually "
    "answered `adk_request_input` with, not merely what it offered, so "
    "naming a spec_id nobody actually picked -- whether it was never "
    "offered, offered but not chosen, or offered with no answer on record "
    "yet -- gets the plan refused. A `Decision` needs at least two options "
    "to choose between either way. A source is pinned only through such a "
    "decision between the candidates the translator offered, never picked "
    "from the candidates directly, and never `resolved_by=user` for a "
    "choice the question does not say the user made or `adk_request_input` "
    "did not actually ask. That is how a `source_ambiguous` referral is settled: propose again once the decision is recorded; "
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
    "`pinned_source_not_found` means a decision pins a source this revision's "
    "discovery did not find, so settle that choice again against whichever "
    "sources the referral carries instead; `no_operation_covers` means the "
    "methodology has to change; `discovery_budget_exhausted` means the plan asks "
    "for too much at once; `discovery_unavailable` means the data service "
    "could not answer just now, so nothing about the requirement needs to "
    "change and the same plan is worth trying again. A plan "
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
    a sub-agent wrapped as a tool, or the skill tools. An instruction naming
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
    so its text is the message.

    `exc.errors()` carries a `loc` tuple and, for a `model_validator` failure,
    the raised exception itself in `ctx["error"]`. Neither is JSON on its own,
    so `to_jsonable_python(..., serialize_unknown=True)` renders both as plain
    JSON (a list, a string) rather than `problem()`'s own validation rejecting
    the detail it was just handed.
    """
    if not isinstance(exc, ValidationError):
        return _invalid_plan(f"Invalid plan data: {exc}")
    errors = exc.errors(include_url=False, include_input=False)
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


def _approval_summary(plan: Plan) -> dict[str, object]:
    """Return what a next call needs to know about the compiled `plan`.

    The version and revision, which tie a later answer to one content hash,
    and every warning the compiler let through (`Workflow.warnings`). The
    plan's own decisions, requirements, sources and steps are deliberately
    absent. `harness.studio_translate` builds the card the user approves from
    the plan record, through `harness.plan_display`, and `dfo_translator` and
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
                "before it can execute. Call `dfo_translator` first."
                if has_translator
                else "before anything can execute, and no translator is attached here."
            ),
            outcome=Outcome.REFUSED,
            action=Action(call="dfo_translator") if has_translator else None,
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

    The referral reaches the planner as the `FunctionResponse` for the
    translator's own call, carrying its `kind` and `requirement_id`. This walks the
    session once, in event order. An `adk_request_input` call event after that
    response is an ask the model raised knowing the referral, and its `id` is
    remembered. A response counts as the user's answer when it pairs by `id` with
    one of those asks and `_is_answer` holds. An ask issued before the referral, a
    call the tool or ADK rejected, and a skip do not count. A skip is the user
    declining to choose, which the planner may not substitute for, so the planner
    ends the turn by telling them what cannot be done.

    A translator response naming a different referral than the last one starts
    the walk over. A translator called again on a plan whose referral is still pending
    answers with that same referral, so the answer the user gave in between still
    counts. A `propose_plan` that recorded a version clears the referral in
    session state, so it ends the referral this walk is following too. A refused
    proposal leaves both alone, which keeps the user's answer standing for the
    retry.

    With no translator response on record the result is no answer. The guard
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
            elif response.name == TRANSLATOR_AGENT_NAME:
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
    and `_window_seeded_pins` write). A pin naming a spec_id that was never
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


def _unrecordable(plan: Plan, tool_context: ToolContext) -> dict[str, object] | None:
    """Return the refusal for a built plan `propose_plan` may not record, or `None`.

    A plan states at least one requirement. A patch carries forward what it
    does not name, so the amended plan is checked whole for a pin the recorded
    plan made before the turn dropped its dataset (`_dropped_seed_pinned`).
    """
    if not plan.evidence_requirements:
        return _invalid_plan(
            "A plan states at least one evidence requirement: what data the answer "
            "rests on, citing the decisions that fixed it. State the requirements this "
            "question needs, or name the ones a patch removes in "
            "`removed_evidence_requirements`."
        )
    return _dropped_seed_pinned(plan.decisions, plan.evidence_requirements, tool_context)


def _repinned(
    decisions: Sequence[Decision],
    requirements: Sequence[EvidenceRequirement],
    *,
    decision_pins: Mapping[str, str],
    requirement_pins: Mapping[str, str],
) -> tuple[list[Decision], list[EvidenceRequirement]]:
    """`decisions` and `requirements` with each source pin moved as the two maps say.

    `decision_pins` maps a decision's slot, `requirement_pins` a dataset
    requirement's id, to the spec_id it pins after the move; anything either map does
    not name is returned as it was.
    """
    moved_decisions = [
        decision.model_copy(
            update={
                "fixes": decision.fixes.model_copy(
                    update={"source_spec_id": decision_pins[decision.slot]}
                )
            }
        )
        if decision.fixes is not None and decision.slot in decision_pins
        else decision
        for decision in decisions
    ]
    moved_requirements = [
        requirement.model_copy(update={"source_spec_id": requirement_pins[requirement.id]})
        if isinstance(requirement, DatasetRequirement) and requirement.id in requirement_pins
        else requirement
        for requirement in requirements
    ]
    return moved_decisions, moved_requirements


def _seed_pins(
    decisions: Sequence[Decision],
    requirements: Sequence[EvidenceRequirement],
    seeded: Sequence[SeededSource],
) -> tuple[list[Decision], list[EvidenceRequirement]]:
    """`decisions` and `requirements` with every pin of a windowed seed moved back to the seed.

    `propose_plan` reasons about a seeded pin by the seed's own spec_id, the
    one the planner was shown, whatever window an earlier call recorded it
    over (`_window_seeded_pins`), so a patch and the plan it amends always
    agree on what they pin.
    """
    to_seed = {
        item.source.spec_id: item.windowed_from for item in seeded if item.windowed_from is not None
    }
    return _repinned(
        decisions,
        requirements,
        decision_pins={
            decision.slot: to_seed[decision.fixes.source_spec_id]
            for decision in decisions
            if decision.fixes is not None and decision.fixes.source_spec_id in to_seed
        },
        requirement_pins={
            requirement.id: to_seed[requirement.source_spec_id]
            for requirement in requirements
            if isinstance(requirement, DatasetRequirement) and requirement.source_spec_id in to_seed
        },
    )


def _period_window(period: Period) -> dict[str, object]:
    """`period` as an identity's `requested_window`."""
    return {"kind": "range", "start_year": period.start, "end_year": period.end}


def _window_seeded_pins(
    plan: Plan, seeded: Sequence[SeededSource]
) -> tuple[Plan, list[SeededSource]]:
    """`plan` with each seeded pin moved to the seed over the requirement's own period.

    A seed's identity carries the years the user saw in Search, often every
    year with data, and a plan's retrieval mints the pinned spec_id as it is
    (`compiler.compile`'s `_retrieve_arguments`). So a dataset requirement
    pinned to a seed over another period is pinned instead to the seed's
    identity over that period (`data_mcp.definition_writes.windowed_identity`,
    the spec_id `get_data`'s `scope.window` would mint), and so is the
    decision that fixes the pin.

    Returns the moved plan and every copy of a seed over another window it
    pins: the one `seeded` holds for that spec_id, or a new one
    (`planner.plan.windowed_copy`), which states nothing Search stated for
    the seed's own years. The translator reads those from Search's answer
    for the copy when it honours the pin (`translator.discovery`'s
    `_seeded_source`).

    Raises:
        ValueError: If one decision fixes the pin of two requirements over
            different periods, since its one `source_spec_id` cannot name both.
    """
    seeds = {item.source.spec_id: item for item in seeded if item.windowed_from is None}
    copies = {item.source.spec_id: item for item in seeded if item.windowed_from is not None}
    requirement_pins: dict[str, str] = {}
    targets: dict[str, set[str]] = {}
    windows: dict[str, SeededSource] = {}
    for requirement in plan.evidence_requirements:
        if not isinstance(requirement, DatasetRequirement):
            continue
        seed = seeds.get(requirement.source_spec_id or "")
        if seed is None or seed.source.identity is None:
            continue
        window = _period_window(requirement.period)
        target = seed.source.spec_id
        if seed.source.identity.get("requested_window") != window:
            copy = windowed_copy(seed, window)
            target = copy.source.spec_id
            requirement_pins[requirement.id] = target
            if target in copies:
                windows[target] = copies[target]
            elif target not in seeds and target not in windows:
                # Two seeds over one dataset copy to one spec_id over a
                # shared period; the first to pin it is the copy recorded.
                windows[target] = copy
        for slot in requirement.decision_ids:
            targets.setdefault(slot, set()).add(target)
    decision_pins: dict[str, str] = {}
    for decision in plan.decisions:
        if decision.fixes is None or decision.fixes.source_spec_id not in seeds:
            continue
        pinned = targets.get(decision.slot, set())
        if len(pinned) > 1:
            raise ValueError(
                f"decision {decision.slot!r} fixes the pin of requirements over different "
                "periods, which retrieve different spec_ids; give each of those requirements "
                "its own decision"
            )
        if pinned and pinned != {decision.fixes.source_spec_id}:
            decision_pins[decision.slot] = next(iter(pinned))
    if not requirement_pins and not decision_pins:
        return plan, []
    decisions, requirements = _repinned(
        plan.decisions,
        plan.evidence_requirements,
        decision_pins=decision_pins,
        requirement_pins=requirement_pins,
    )
    moved = plan.model_copy(
        update={"decisions": tuple(decisions), "evidence_requirements": tuple(requirements)}
    )
    return Plan.model_validate(moved.model_dump()), list(windows.values())


def _source_decision_slot(spec_id: str) -> str:
    """The `Decision.slot` `_synthesize_source_decisions` records a pin's decision under.

    Derived from `spec_id` rather than fixed, so two different pins never
    collide: this call names at most one, since `exposed_spec_id` is a single
    id, but a patch can still cite an old one carried over from a plan drafted
    against an earlier turn's envelope. `spec_id` already matches
    `Identifier`'s own pattern (`planner.records.Identifier`, a lower ASCII
    id), so prefixing it costs nothing in validity.
    """
    return f"source_{spec_id}"


def _synthesize_source_decisions(
    decisions: Sequence[Decision],
    evidence_requirements: Sequence[EvidenceRequirement],
    *,
    previous: Plan | None,
    exposed_spec_id: str | None,
) -> tuple[list[Decision], list[EvidenceRequirement]]:
    """Add the `Decision` a verbatim pin of the envelope's own `spec_id` needs, so citing it is all the model has to do.

    `harness.planner_seam` renders the resolve envelope's `spec_id` as an
    allowed pin for a single-part question with nothing left to resolve, but
    `_check_pin_rests_on_a_fixing_decision` still requires a citing
    `Decision`. This writes that decision on the model's behalf, the way
    `_ELECTED_SPEC_REASON` asks the model to write one when a clarify round
    elects an id, except here nothing was elected for the model to narrate.

    A synthesized `Decision`'s two `options` are pinning the envelope's id
    and leaving the source unpinned, the only two things genuinely true of a
    `DatasetRequirement` pinning it, and `resolved_by=planner_default` names
    the planner, not the user, as the one who settled it.

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
        exposed_spec_id: `ENVELOPE_SPEC_STATE_KEY`'s value for this session,
            the spec_id `harness.planner_seam` actually rendered to the model
            this entry. `None` when the envelope omitted it, in which case
            this is a no-op: a pin the model was never told was allowed is
            left for the ordinary validator to refuse.

    Returns:
        `decisions`/`evidence_requirements`, unchanged unless some requirement
        needed the pin cited. When so, each such requirement's `decision_ids`
        cites the synthesized decision's slot, and the decision itself is
        appended to `decisions` unless a prior round already recorded it.
    """
    if exposed_spec_id is None:
        return list(decisions), list(evidence_requirements)
    known = {decision.slot: decision for decision in (previous.decisions if previous else ())}
    known.update({decision.slot: decision for decision in decisions})

    def _already_fixes_the_pin(decision_id: str) -> bool:
        cited = known.get(decision_id)
        return (
            cited is not None
            and cited.fixes is not None
            and cited.fixes.source_spec_id == exposed_spec_id
        )

    def _needs_the_pin(requirement: EvidenceRequirement) -> bool:
        return (
            isinstance(requirement, DatasetRequirement)
            and requirement.source_spec_id == exposed_spec_id
            and not any(_already_fixes_the_pin(slot) for slot in requirement.decision_ids)
        )

    slot = _source_decision_slot(exposed_spec_id)
    if not any(_needs_the_pin(requirement) for requirement in evidence_requirements):
        return list(decisions), list(evidence_requirements)
    patched_requirements = [
        requirement.model_copy(update={"decision_ids": (*requirement.decision_ids, slot)})
        if _needs_the_pin(requirement)
        else requirement
        for requirement in evidence_requirements
    ]
    if slot in known:
        return list(decisions), patched_requirements
    synthesized = Decision(
        slot=slot,
        description="Whether to pin the source the resolve envelope already settled on.",
        options=(f"pin {exposed_spec_id}", "leave the source unpinned"),
        chosen=f"pin {exposed_spec_id}",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale=(
            "The resolve envelope named this as the question's definite source, with "
            "no other candidate on offer."
        ),
        fixes=DecisionFixes(source_spec_id=exposed_spec_id),
    )
    return [*decisions, synthesized], patched_requirements


# `propose_plan`'s own docstring under `question_fixed`, assigned over the
# function's definition-time docstring below once `build_planner_tools` knows
# the mode. ADK builds the tool's declaration -- including each argument's own
# `description` -- from this text, so the two arguments `question_fixed`
# actually changes the handling of, `question` and `starts_new_plan`, read
# here the way `_resolve_intent` enforces them; everything else is identical
# to the definition's own docstring.
_PROPOSE_PLAN_DOC_FIXED = """Record a plan, or patch the one already recorded.

The call states which of the two it is, and never has a question's
wording stand in for that. With nothing recorded, it is a first draft:
state every decision the plan makes and every requirement the answer
rests on. With a plan recorded, pass the `amends_version` that plan
answers with, and the call patches it. This turn's question is fixed, so
a patch's own `question`, if it states one, has to match the one already
recorded. Pass `starts_new_plan=true` instead only to say the user's
reply asks a different question -- that call is refused and records
nothing, ending the turn there for the harness to tell the user to send
it as its own message. A recorded plan and neither argument is refused.
The reading is not guessed from `question`, because a reworded question
that meant an amendment would otherwise drop every item the call did
not restate.

A patch names only what changes, keyed by `slot` for a decision and by
`id` for a requirement, and every item it does not mention is carried
forward. A patch's items are matched to the recorded plan by that key:
a key the plan already states replaces that item, and a key it does
not state adds one. Every call produces a new plan version.

Nothing is deleted by leaving it out. A decision or requirement is
removed only by naming it in `removed_decisions` (its `slot`) or
`removed_evidence_requirements` (its `id`), which is why those two
arguments exist: an item absent from a patch means unchanged.

Args:
    question: The question this plan answers. Required on a first
        draft. A patch may omit it, keeping the recorded question, or
        restate it unchanged -- it cannot be changed here, since the
        question is fixed for this turn.
    amends_version: The `version` this call patches, as the last
        `propose_plan` call answered it and as the brief's own
        "version N" line states it. Required whenever a plan is
        already recorded for this session, unless `starts_new_plan`
        is set. It must be the version actually recorded.
    starts_new_plan: Set true only to say the user's reply asks a
        different question from the one already recorded for this
        turn. Refused and records nothing: the harness ends the turn
        and tells the user to send the other question as its own
        message.
    decisions: The decisions this call adds or replaces, keyed by
        `slot`. On the first call, every decision the plan makes: at
        most 24. `options` records what was considered, and
        `rationale` is the one-line reason, and when `resolved_by` is
        the user, that can just say so (e.g. "user's explicit preference, no stated
        methodology reason").
    evidence_requirements: The requirements this call adds or
        replaces, keyed by `id`: a dataset requirement per series, and
        a derived requirement per calculation, each citing in
        `decision_ids` the decisions that fixed it. Every decision has
        to be cited by at least one, and every dataset requirement has
        to feed a calculation. A plan states at most 24
        requirements.
    removed_decisions: The `slot` of each decision this call deletes.
    removed_evidence_requirements: The `id` of each requirement this
        call deletes.

Returns:
    `ok` with `status`, the `"draft"` this call just recorded;
    `version`, the number to pass to `submit_plan`; `previous_version`,
    the version this amended, `None` on a first draft; and `question`,
    as stored. `previous_version` is why this is returned at all: it is
    the one thing `Plan.amend` changed that your own arguments did not
    state. Your decisions and requirements are in your own call, and
    `dfo_translator` reads the plan from session state.
"""


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
    to `_resolve_intent` on every call, and selects `propose_plan`'s own
    docstring (`_PROPOSE_PLAN_DOC_FIXED` in place of the definition's own),
    so the `starts_new_plan` and `question` arguments ADK declares from it
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
        """Ask the user a question a referral leaves to them.

        Call this when a `source_ambiguous` (or similarly candidate-
        carrying) referral's choice is genuinely ambiguous, the same
        standard that governs asking the user anything else, and for an
        `unsupported_ask` or `requirement_unmeetable` referral, to say what
        cannot be done and ask what to change. This is a long-running call:
        it suspends the turn and returns nothing here -- the user's own
        answer arrives as this call's own response, on your next turn,
        naming the `spec_id` they picked (or free text, if they answered
        outside the offered candidates). Record whichever `spec_id`
        comes back as a `Decision`: `resolved_by=user`,
        `fixes.source_spec_id` set to it, `options` listing every candidate
        you offered here -- naming a `spec_id` this call never offered, or
        one it offered but the user did not actually pick, gets the plan
        refused, since a `resolved_by=user` pin has to trace to what the
        user was actually asked and answered.

        Args:
            message: The question to show the user, naming what is
                ambiguous and, briefly, what each candidate is, in plain
                words: never a requirement id, spec_id or code.
            candidates: The referral's own candidates, each
                `{"spec_id": ..., "label": ...}` -- `label` is the
                candidate's own `sentence`, not invented text. Omit it for
                a referral that carries none.
            recommended_spec_id: The referral's `recommendation.spec_id`,
                when it carries one -- must name one of `candidates`.
            rationale: The referral's `recommendation.rationale`, when
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
        decisions: list[Decision] = [],  # noqa: B006
        evidence_requirements: list[EvidenceRequirement] = [],  # noqa: B006
        removed_decisions: list[str] = [],  # noqa: B006
        removed_evidence_requirements: list[str] = [],  # noqa: B006
        amends_version: int | None = None,
        starts_new_plan: bool = False,
    ) -> dict[str, object]:
        """Record a plan, or patch the one already recorded.

        The call states which of the two it is, and never has a question's
        wording stand in for that. With nothing recorded, it is a first draft:
        state every decision the plan makes and every requirement the answer
        rests on. With a plan recorded, pass the `amends_version` that plan
        answers with, and the call patches it. Pass `starts_new_plan=true`
        instead when the user's ask is a different question that calls for a
        plan of its own. A recorded plan and neither argument is refused. The
        reading is not guessed from `question`, because a reworded question
        that meant an amendment would otherwise drop every item the call did
        not restate.

        A patch names only what changes, keyed by `slot` for a decision and by
        `id` for a requirement, and every item it does not mention is carried
        forward. A patch's items are matched to the recorded plan by that key:
        a key the plan already states replaces that item, and a key it does
        not state adds one. Every call produces a new plan version.

        Nothing is deleted by leaving it out. A decision or requirement is
        removed only by naming it in `removed_decisions` (its `slot`) or
        `removed_evidence_requirements` (its `id`), which is why those two
        arguments exist: an item absent from a patch means unchanged.

        Args:
            question: The question this plan answers. Required on a first
                draft and with `starts_new_plan`, which starts at version 1
                under it. A patch that states one records it, and one that
                omits it keeps the recorded question.
            answer_mode: `series` when the question asks to see a level or a
                trend for named entities -- a lone dataset requirement,
                unread by anything else, is then itself the answer. `computed`
                when the question asks for a figure the series does not state
                on its own, such as a difference, growth, share, rank or
                total -- every dataset requirement then has to feed a
                calculation. Required on a first draft and with
                `starts_new_plan`. A patch that states one switches the
                amended plan's mode and re-validates the requirements under
                it; one that omits it keeps the recorded mode.
            amends_version: The `version` this call patches, as the last
                `propose_plan` call answered it and as the brief's own
                "version N" line states it. Required whenever a plan is
                already recorded for this session, unless `starts_new_plan`
                is set. It must be the version actually recorded.
            starts_new_plan: Set true to begin a plan of its own for a
                different question while a plan is already recorded.
            decisions: The decisions this call adds or replaces, keyed by
                `slot`. On the first call, every decision the plan makes: at
                most 24. `options` records what was considered, and
                `rationale` is the one-line reason, and when `resolved_by` is
                the user, that can just say so (e.g. "user's explicit preference, no stated
                methodology reason").
            evidence_requirements: The requirements this call adds or replaces,
                keyed by `id`: a dataset requirement per measure and source,
                naming every entity, and a derived requirement per calculation,
                each citing in `decision_ids` the decisions that fixed it,
                or none when the question states its fields outright.
                Leave a dataset requirement's `unit` unset unless the
                question asks for a currency or price basis: a source's own
                unit is not one the user asked for.
                Every decision has to be cited by at least one. Under
                `answer_mode="series"`, a dataset requirement no calculation
                reads is itself the answer; under `"computed"`, every one has
                to feed a calculation. A plan states at most 24 requirements.
            removed_decisions: The `slot` of each decision this call deletes.
            removed_evidence_requirements: The `id` of each requirement this
                call deletes.

        Returns:
            `ok` with `status`, the `"draft"` this call just recorded;
            `version`, the number to pass to `submit_plan`; `previous_version`,
            the version this amended, `None` on a first draft; and `question`,
            as stored. `previous_version` is why this is returned at all: it is
            the one thing `Plan.amend` changed that your own arguments did not
            state. Your decisions and requirements are in your own call, and
            `dfo_translator` reads the plan from session state.
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
                }
            )
        except ValidationError as exc:
            return _invalid_data(exc)
        decisions = patch.decisions
        evidence_requirements = patch.evidence_requirements
        removed_decisions = patch.removed_decisions
        removed_evidence_requirements = patch.removed_evidence_requirements
        existing = load_plan(tool_context.state)
        seeded = seeded_sources(tool_context.state)
        if seeded:
            decisions, evidence_requirements = _seed_pins(decisions, evidence_requirements, seeded)
        # These checks run before the plan is touched, so a refusal records nothing and
        # spends no round.
        blocked = (
            _referral_unanswered(existing, tool_context)
            or _dropped_seed_pinned(decisions, evidence_requirements, tool_context)
            or _election_untraceable(decisions, tool_context)
        )
        if blocked is not None:
            return blocked
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
        if previous is not None and seeded:
            unwindowed, unwindowed_requirements = _seed_pins(
                previous.decisions, previous.evidence_requirements, seeded
            )
            previous = previous.model_copy(
                update={
                    "decisions": tuple(unwindowed),
                    "evidence_requirements": tuple(unwindowed_requirements),
                }
            )
        # A verbatim pin of the envelope's own `spec_id` gets the citing
        # `Decision` it needs written here, on the model's behalf, so
        # `Plan`'s own validator never has to refuse it for lacking one.
        # See `_synthesize_source_decisions`.
        decisions, evidence_requirements = _synthesize_source_decisions(
            decisions,
            evidence_requirements,
            previous=previous,
            exposed_spec_id=tool_context.state.get(ENVELOPE_SPEC_STATE_KEY),
        )
        try:
            if previous is not None:
                plan = previous.amend(
                    question=question,
                    answer_mode=answer_mode,
                    decisions=decisions,
                    evidence_requirements=evidence_requirements,
                    removed_decisions=removed_decisions,
                    removed_evidence_requirements=removed_evidence_requirements,
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
                    decisions=decisions,
                    evidence_requirements=evidence_requirements,
                )
            plan, windows = _window_seeded_pins(plan, seeded)
        except (ValidationError, ValueError) as exc:
            # The plan-level rules: unique slots and ids, every citation naming
            # a decision, every decision cited,
            # the length and count limits on each field, a question that is not
            # blank, and the size limit on the whole. A removal naming an item
            # the plan does not state is a `ValueError` from `Plan.amend` itself,
            # and a first draft or restart with no `question`/`answer_mode` is
            # one raised above. `_window_seeded_pins` raises one for a decision
            # fixing two seeded pins over different periods.
            return _invalid_data(exc)
        blocked = _unrecordable(plan, tool_context)
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
        save_plan(tool_context.state, plan)
        save_plan_rounds(tool_context.state, rounds)
        # A new proposal supersedes whatever the translator referred back.
        clear_referral(tool_context.state)
        # Only what the model cannot read back off its own call, and the
        # question as stored. The decisions and requirements the model sent are
        # echoed nowhere, because the plan is in session state and
        # `dfo_translator` reads it from there. `status` is always the
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

    if question_fixed:
        propose_plan.__doc__ = _PROPOSE_PLAN_DOC_FIXED

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
        return ok(
            status=plan.status.value,
            version=plan.version,
            revision=plan_revision(plan),
            summary=_approval_summary(plan),
        )

    ask_user.__name__ = REQUEST_INPUT_FUNCTION_CALL_NAME
    return [
        LongRunningFunctionTool(ask_user),
        FunctionTool(propose_plan),
        FunctionTool(reject_plan),
        FunctionTool(submit_plan),
    ]


def create_planner_agent(
    *,
    model_name: str,
    translator: LlmAgent | None = None,
    surface: ToolSurface | None = None,
    question_fixed: bool = False,
) -> LlmAgent:
    """Build the planner `LlmAgent`.

    Args:
        model_name: The model this agent runs on.
        translator: The translator agent to attach as a sub-agent. Built with
            `mode="single_turn"`, ADK wraps it as one more tool on this agent,
            which is how a draft plan gets translated and compiled. Omitted,
            the planner is the standalone agent its own eval set scores --
            which is also what keeps that set from needing a live tool
            surface.
        surface: The tool surface `get_skill` and `get_skill_file` call
            data-mcp through. Omitted, the planner has neither tool and its
            instruction does not name them.
        question_fixed: Whether the question this planner drafts for is fixed
            for the entry it is called in. The harness graph's own seam sets
            this when it builds the pair (`harness.planner_seam`), since the
            graph derives and narrates a turn's `resolve_envelope`, `depth`
            and `question` before the planner ever runs, and a restart would
            silently swap the question out from under that state -- see
            `_resolve_intent`. False, the default, is the standalone
            planner's and `harness_pair_v1`'s own behaviour: a restart begins
            a fresh plan.

    Returns:
        The configured `LlmAgent`.
    """
    return LlmAgent(
        name=AGENT_NAME,
        model=model_name,
        instruction=planner_instruction(
            has_translator=translator is not None,
            has_skills=surface is not None,
            question_fixed=question_fixed,
        ),
        tools=[
            *(build_skill_tools(surface) if surface is not None else []),
            *build_planner_tools(
                has_translator=translator is not None, question_fixed=question_fixed
            ),
        ],
        sub_agents=[translator] if translator is not None else [],
        # Run as a workflow node (`harness.planner_seam.PlannerSeam._propose`'s
        # `ctx.run_node`), ADK defaults an agent with no explicit `mode` to
        # `single_turn`, and a `single_turn` agent with no explicit
        # `include_contents` to `'none'` (`run_llm_agent_as_node`,
        # `google.adk.workflow._llm_agent_wrapper`). `'none'` builds every
        # request from `_get_current_turn_contents`, which slices the session
        # from the latest `author='user'` or other-agent-reply event --
        # exactly what the translator sub-agent's own turn plants twice, once
        # for its node input and once for each of its own replies. That
        # slice boundary lands past this agent's own `dfo_translator` call,
        # orphaning its response and, with it, the seam's context message and
        # every earlier tool call of this agent's own turn, one orphaned
        # response per `dfo_translator` call. Pinning `'default'` here uses
        # `_get_contents` instead, which reads this agent's whole turn
        # regardless of what a sub-agent plants in between, so the next
        # generation carries the seam's context message and both tool
        # call/response pairs.
        include_contents="default",
        # `AGENT_THINKING_CONFIG` (see `planner.settings`) pins every
        # generation to LOW thinking.
        planner=BuiltInPlanner(thinking_config=AGENT_THINKING_CONFIG),
        # Drops this generation's thought-summary parts once the trace has
        # them, so `include_contents="default"` above never replays one into
        # a later request of the same turn. See `drop_thought_parts`.
        after_model_callback=drop_thought_parts,
    )
