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

Built with a tool surface, the planner also carries `planner.skills`'
`get_skill` and `get_skill_file`, and its instruction carries `_SKILLS`, the
paragraph naming them. Built without one, it carries neither, so the
standalone eval agents run without a live server.

The planner settles methodology and states what evidence an answer rests on,
then stops. A proposed plan is a `DRAFT` carrying `decisions` and
`evidence_requirements`; finding the data, proposing the operations and
compiling the workflow belong to `translator.tool`'s `build_translate`, the
translation `propose_plan` runs on every plan it records before the user ever
sees it -- `create_planner_agent` takes that function, and `harness.agents`
builds it. So no tool name, column or operation appears here,
`PENDING_APPROVAL` is a status only translation writes, and `submit_plan`
requires a plan that has been through that step.

Every tool below answers with `planner.tool_results`' typed shape -- `ok`,
`problem`, with a stable `code` and, where the next step is known, an
`action` -- rather than an unstructured `{"error": ...}` a caller has to
parse. When translation cannot carry a plan out, `propose_plan`'s
`translation` is the `Referral` itself: this agent's instruction reads it
straight off that answer, with nothing here to fetch it back a second time.
"""

from __future__ import annotations

import asyncio
import functools
import unicodedata
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple, cast

from compiler.sources import PROVIDER_ACCOUNT, block_groups, source_from_spec_block
from compiler.surface import ToolSurface
from data_mcp.definition_writes import windowed_identity
from data_mcp.settings import Settings as DataMcpSettings
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from orchestrator.agent import _retrying
from pydantic import BaseModel, Field, JsonValue, ValidationError, model_validator
from pydantic_core import to_jsonable_python
from translator.discovery import (
    DefinitionNotResolvedError,
    PartNotResolvedError,
    load_log,
    requirement_open_roles,
    requirement_revision,
    resolve_definition,
    resolved_spec,
)
from translator.referral import Referral, ReferralKind, clear_referral, load_referral
from translator.tool import Translate

from planner.declaration import PlanTool
from planner.parties import PARTY_ROLES, across_party, party_role
from planner.plan import (
    MAX_DECISION_OPTIONS,
    MAX_DECLINES,
    MAX_OPTION_CHARS,
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
    FullHistory,
    PinSource,
    Plan,
    PlanStatus,
    RecordedDefinition,
    ResolvedBy,
    SessionState,
    SumOfRequirement,
    UnsettledFixError,
    UnsupportedPlanError,
    WindowedPin,
    cut_user_text,
    full_histories,
    full_history,
    inputs_of,
    is_search_pin,
    load_asks,
    load_optional_declines,
    load_plan,
    load_plan_rounds,
    load_unsupported,
    open_window,
    period_window,
    pin_sources,
    plan_revision,
    record_unsupported,
    recorded_identity,
    rolling_readers,
    rolls_from_its_start,
    save_full_history,
    save_plan,
    save_plan_rounds,
    save_windowed_pins,
    search_pin_slot,
    settle_decisions,
    shown_identity,
    windowed_pin,
    windowed_pins,
)
from planner.records import (
    Identifier,
    OpenPeriodFault,
    ResolvedSourceMention,
    RollingIssue,
    RollingPlanError,
)
from planner.settings import AGENT_THINKING_CONFIG, drop_thought_parts
from planner.shapes import PLACE_PHRASE
from planner.skills import build_skill_tools
from planner.tool_results import Action, Outcome, ProblemCode, ok, problem
from planner.workflow import ResolvedGroup, SelectedSource

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
        """Reject an entity `role` outside `PARTY_ROLES`, naming the word it stands for when one does.

        A place word is no role: a place the measure is about states none,
        and an axis names its dimension by a place word all the same
        (`planner.shapes.roles_named`).
        """
        refused = []
        for requirement in self.evidence_requirements:
            if not isinstance(requirement, DatasetRequirement):
                continue
            for entity in requirement.entities:
                if entity.role is None or entity.role in PARTY_ROLES:
                    continue
                meant = party_role(entity.role)
                advice = ""
                if meant is not None:
                    advice = f", write {meant!r}"
                elif across_party(entity.role) == "place":
                    advice = f", leave it unset: an axis names such places {PLACE_PHRASE!r}"
                refused.append(f"{requirement.id}/{entity.name}: {entity.role!r}{advice}")
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
    "results in your first message, numbered search readings, are what the "
    "conversation's searches established, as far as "
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
    "carries a window, the period is a decision like any other. A period the "
    'question words against the newest data ("the last five years", "the '
    'latest year", "since 2015") also follows it: state the years from the '
    "`window` as above, set `period_end_open` on the dataset requirement, and "
    "state `rolling` on the growth, rank or period threshold that reads "
    "those years, the newest at offset 0 and a year five before it at -5. "
    '"Since 2015" keeps 2015 a fixed start and rolls only the end. Years '
    'the question names ("from 2015 to 2020") stay fixed, with no '
    "`rolling` and no `period_end_open`. Source is "
    "different: when a result's own `spec_id` (a definite or chosen reading) "
    "answers a dataset requirement, pin it as "
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
    "party into one series per year, which a `share_of` or `quotient_of` "
    "reads as a whole side. Over a requirement that names no entity, a group by `entity` "
    "keeps nothing and is refused: group by period, or `rank` with a "
    "`reduction` to set each member's total beside the others. "
    "A rate, index or per-person figure is never summed. "
    'A party the question names no member of, such as the donors of "top '
    'donors", goes in `across` (see `across`). '
    'A region or group named alone, such as "Africa", is one entity with its '
    'own figure; "African countries", or a figure for each member, names the '
    "set once with `members` set, however many members it has. "
    "A result's open axes "
    "(`open_entity_axes`) are the columns left open: `origin` is the giving party "
    "(an entity `role` of donor, creditor or exporter), `destination` the receiving "
    "party (recipient, debtor or importer), `observationAbout` the place. "
    "A requirement whose `across` is the place, such as `'country'`, names "
    "no place, so its retrieval holds every level (see `level` on "
    "`place_level`). Before "
    "a `rank`, an `aggregate` across places, a `share_of` or a "
    "`quotient_of` reads it, keep "
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
    "`share_of` only for a part of its whole, both in one unit, for "
    "`quotient_of` only for one measure per unit of another, such as a "
    'per-head or per-capita figure or "divide one by the other", and for '
    "`join` only across different measures, since "
    "none of them keeps the separate series a chart of one measure over "
    "several entities shows. An `axis` or a `denominator_axis` names a role "
    "its side's entities state, or `'country'` for places stating none. "
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
    "donors. Tables whose provenance states account `provider`, such as the "
    "OECD and ONE aid tables, add up as the skill the sources' search "
    "results name says: cite a `Decision` recording it, with the skill in its `rationale`. "
    "Whether any other two datasets add up without "
    "counting a flow twice is the user's to decide: read what that skill "
    "says about it; `propose_plan` refuses such a `sum_of` with the "
    "`options` to ask with. If the user picks adding them, cite a "
    "`Decision` resolved by `user` whose `chosen` is that option id; "
    "otherwise show the two series separately. ODA loans "
    'count in both aid and debt flows, so asking about a total of the two ("net '
    'flows" of aid and debt) names that overlap. A sum '
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
    "question names none. A growth or a per-entity `share_of` or "
    "`quotient_of` goes "
    "before a rank with `per_period` and a `limit`, and a top N by each "
    "entity's total is one rank stating the `reduction` and the `limit` "
    "with `per_period` unset. No rank reads another rank's output: put the "
    "`limit` on the one rank that reads the measure. "
)

# What each `propose_plan` call states about which plan it means. Held apart
# from `_PLANNING` because it is the one paragraph `question_fixed` rewrites:
# standalone and `harness_pair_v1` keep `_VERSIONING_FREE`, where a restart is
# a normal way to begin a second plan; `harness.planner_seam` builds with
# `question_fixed=True`, where the orchestrator's brief fixes the turn's
# question. What each argument does is on the argument.
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
    'The search readings close with a "Skills named" line listing, by id, '
    "the skills that apply to them. When a skill bears on the question, "
    "call `get_skill` with its id before you draft, and let what it says "
    "shape your decisions; read a file it links with `get_skill_file`. A "
    "question about DFO itself (what it covers, where its data comes from, "
    "how it works) starts with `get_skill`, though no search result names "
    "that skill. When a skill tool refuses, `capability_unavailable` "
    "included, draft the plan without that skill. "
)

# What to do once a plan is recorded, when a translator is attached to carry it
# out and `propose_plan` answers with its verdict under `translation`. Held
# apart because the planner is also built standalone -- its own eval set scores
# it that way -- and its `propose_plan` answers with no translation to read.
_TRANSLATING = (
    "`source_name_ambiguous` means Search could not choose which publisher or table a "
    "source wording names. Ask the user through `adk_request_input`, using the offered "
    "refs as candidate ids, then state the chosen ref in `source_ref` and keep the "
    "source wording and `source_exclude` on the requirement. "
    "`propose_plan` has every plan it records translated at once and answers "
    "with the verdict in `translation`. Once a "
    "referral is settled, by your own decision or the "
    "user's answer, propose with the `amends_version` the last "
    "`propose_plan` call answered with, so the patch lands on that plan and "
    "is translated in turn. When `translation` answers that the plan "
    "compiled, call `submit_plan`. When translation answers "
    "`named_in_place_of_across`, the pinned dataset covers only that group: the "
    "requirement now names it in `entities` and states "
    "no `across`. Amend from the plan as recorded and do not restore `across`. "
    "Otherwise `translation` "
    "is the referral. `choice_required` "
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
    "A source under the referral's `withheld` dropped the pinned source's "
    "scope, such as a named group, and is never pinned: when no candidate "
    "keeps that scope, make the referral's `repair` to the requirement, which "
    "names that scope again, and propose again; with no `repair`, ask the user. "
    "Where several are offered, settle it yourself when the question, the brief or the "
    "candidates' coverage make one the clear answer, a source the question "
    "or the brief names included; when the choice is the user's to make, "
    "call `adk_request_input` with the candidates. Either way, record a "
    "`Decision` whose `options` name every candidate, whose `fixes` names "
    "`source_spec_id` and whose rationale says why, "
    "resolved by `planner_default` when you chose and by `user` when the "
    "user's answer did; pin that `spec_id` as the requirement's "
    "`source_spec_id`, and propose again. "
    "A `choice_required` referral with no candidates and none withheld means "
    "the pinned source is gone: unpin it and change what the requirement asks for. By `kind`: "
    "`no_data` carries the nearest real sources search found instead and the "
    "reason code for why, each described without a spec_id -- read them for "
    "context, never pin one, since the requirement itself is what has to "
    "change; "
    "`unsupported_ask` or `requirement_unmeetable` means the requirement cannot be "
    "met as stated: tell the user what cannot be done and ask, through "
    "`adk_request_input`, before proposing anything, since what to change is "
    "theirs to choose; a `requirement_unmeetable` referral with `candidates` "
    "means the source the requirement states holds none of the data, and the "
    "candidates are the sources found instead, so offer them and pin the one "
    "the user picks as its `source_spec_id` under a `Decision` resolved by "
    "`user`, which replaces the stated source; `ask_not_understood` means "
    "search dropped part of the ask and answered a narrower one, so reword "
    "the part the referral names; "
    "`requirement_needs_repair` names the `constraint` the requirement breaks and "
    "the `repair` you may make, so make it and propose again; "
    "`no_operation_covers` means no operation carries that requirement out; "
    "`unsupported_ask` and `no_operation_covers` are `unsupported` answers, "
    "which earn a decline (see `declined`); a `requirement_unmeetable` whose "
    "`reason` says it earns a decline lets the plan make that decline once the "
    "user's answer is to see the two sides apart, and does not require it; "
    "`discovery_budget_exhausted` means the plan asks "
    "for too much at once; `discovery_unavailable` or `surface_unavailable` means "
    "the data service "
    "could not answer just now, so nothing about the requirement needs to "
    "change: propose the same plan again, with `amends_version` and no other "
    "change, to have that version translated again, which records no new "
    "version and spends no plan round; "
    "`internal_failure` is a "
    "fault in the system that no change to the plan repairs, so tell the user, "
    "with its `reference`, and propose nothing further. The options an ask "
    "offers come only from the referral's `candidates` or a refusal's "
    "`options`; with neither, the ask says what cannot be done and asks what "
    "to change. A plan "
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
    a translator `propose_plan` runs, or the skill tools. An instruction
    describing a translation the agent does not run, or naming a tool it
    lacks, sends the model after something that is not there.

    `question_fixed` selects which paragraph explains `starts_new_plan`:
    `_VERSIONING_FREE` where a restart begins a second plan of its own
    (standalone, and `harness_pair_v1`'s multi-turn session), or
    `_VERSIONING_FIXED`, where `harness.planner_seam` holds the turn's
    question and a restart is how the model tells it the user asked a
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
    `_asks_unaccounted`. A rolling period a rule refused (`RollingPlanError`)
    answers `_rolling_refused`, which names the rule.

    `exc.errors()` carries a `loc` tuple and, for a `model_validator` failure,
    the raised exception itself in `ctx["error"]`. Neither is JSON on its own,
    so `to_jsonable_python(..., serialize_unknown=True)` renders both as plain
    JSON (a list, a string) rather than `problem()`'s own validation rejecting
    the detail it was just handed.
    """
    if isinstance(exc, RollingPlanError):
        return _rolling_refused(exc)
    if not isinstance(exc, ValidationError):
        return _invalid_plan(f"Invalid plan data: {exc}")
    cause = _unsupported_cause(exc)
    if cause is not None:
        return _unsupported(cause)
    errors = exc.errors(include_url=False, include_input=False)
    for error in errors:
        raised = error.get("ctx", {}).get("error")
        if isinstance(raised, AsksUnaccountedError):
            return _asks_unaccounted(raised)
        if isinstance(raised, RollingPlanError):
            return _rolling_refused(raised)
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


def _rolling_refused(exc: RollingPlanError) -> dict[str, object]:
    """Return the `INVALID_PLAN` problem for a rolling period a plan rule refused.

    `detail.issue` is the rule (`planner.records.RollingIssue`),
    `detail.requirement_ids` the requirements to change, and `detail.fault`,
    under `rolling_needs_open_period`, which part of it holds
    (`planner.records.OpenPeriodFault`). `detail.spec_id` is the pinned
    dataset whose first year is missing, when that is the fault.
    """
    detail: dict[str, Any] = {
        "issue": exc.issue.value,
        "requirement_ids": list(exc.requirement_ids),
    }
    if exc.fault is not None:
        detail["fault"] = exc.fault.value
    if exc.spec_id is not None:
        detail["spec_id"] = exc.spec_id
    return problem(
        ProblemCode.INVALID_PLAN,
        f"Invalid plan data: {exc}",
        outcome=Outcome.INVALID_INPUT,
        detail=detail,
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
        "answer it, or, when an `unsupported` answer or a `requirement_unmeetable` whose "
        "reason names a decline earned it (see `declined`), "
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

    Each `unsupported` answer `propose_plan` or its translation gave in this
    planner entry (`load_unsupported`) needs a `Decline` with its operation
    as `code`, so the ask stays on the record instead of vanishing from a
    reshaped plan. An optional decline (`load_optional_declines`) is owed by
    no plan.
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
    its translation's, or one a replan's run-time refusal stands for
    (`harness.planner_seam._earned_by_refusal`); or among its optional
    declines (`load_optional_declines`). A decline carried unchanged from `previous`, the
    plan this one amends, was earned when it was recorded; one restated with
    the same `Decline.key` in other words is the same decline.
    """
    earned = load_unsupported(tool_context.state) | load_optional_declines(tool_context.state)
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
        "code. A decline is earned when `propose_plan` or its `translation` answers `unsupported` "
        "with its code, when a `requirement_unmeetable` translation's reason says it earns "
        "that code, or when the refusal this entry replans from means no operation "
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
    the plan record, through `harness.plan_display`, and translation and
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
            "This plan has not compiled into a workflow, so nothing can execute "
            + (
                "yet. Propose it again, with its `version` as `amends_version`, to have it "
                "translated."
                if has_translator
                else "yet, and no translator is attached here."
            ),
            outcome=Outcome.REFUSED,
            action=Action(call="propose_plan") if has_translator else None,
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

    `question_fixed` (`harness.planner_seam` builds the planner with it
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


# The `propose_plan` arguments that patch a plan. A call naming none of them
# changes nothing about the version it amends.
_PATCH_ARGUMENTS = (
    "decisions",
    "evidence_requirements",
    "removed_decisions",
    "removed_evidence_requirements",
    "declined",
    "answered",
)


def _transient_retry(arguments: Mapping[str, Any], state: SessionState) -> Plan | None:
    """The recorded plan a `propose_plan` call asks to have translated again, or `None`.

    A transient referral (the data service did not answer) leaves the
    recorded plan as it was. A call amending that version and changing
    nothing (`amends_version` alone, any restated `question` or
    `answer_mode` equal to the recorded one) asks only for that version to
    be translated again, so it records no new version and spends no plan
    round. Any other call is a proposal.
    """
    referral = load_referral(state)
    plan = load_plan(state)
    if (
        referral is None
        or referral.outcome is not Outcome.TRANSIENT
        or plan is None
        or arguments.get("amends_version") != plan.version
        or arguments.get("starts_new_plan")
        or any(arguments.get(name) for name in _PATCH_ARGUMENTS)
    ):
        return None
    if arguments.get("question") not in (None, plan.question):
        return None
    if arguments.get("answer_mode") not in (None, plan.answer_mode):
        return None
    return plan


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
    - the graph's `ask` node, under `harness.graph`'s `CLARIFY_INTERRUPT`:
      built through ADK's own `google.adk.workflow.utils.
      _workflow_hitl_utils.create_request_input_event`, which dumps the
      whole `RequestInput` into `args` -- the caller's own `payload` (where
      `ask` puts `candidates`) nests under `args["payload"]`.
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

    Every `adk_request_input` call the workflow itself raises (the `ask`
    node of `harness.graph`, under `CLARIFY_INTERRUPT`) shares the one fixed
    id `"clarify"` across every question in a session. The planner's own
    `ask_user` calls each get their own id. Pairing by id alone
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
_USER_DECIDES = frozenset(
    {
        ReferralKind.REQUIREMENT_UNMEETABLE,
        ReferralKind.UNSUPPORTED_ASK,
        ReferralKind.SOURCE_NAME_AMBIGUOUS,
    }
)


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
    translator's referral (`_answers_since_referral`)."""
    return _answers_since_referral(tool_context)[0]


def _answers_since_referral(tool_context: ToolContext) -> tuple[bool, frozenset[str]]:
    """Whether the user has answered an ask the planner raised after seeing the
    translator's referral, and the spec_ids they picked answering those asks,
    read off this session's event history as `_elected_spec_ids` reads it.

    The referral reaches the planner in the `translation` of the `propose_plan`
    response that recorded the plan, carrying its `kind` and `requirement_id`.
    This walks the session once, in event order. An `adk_request_input` call
    event after that response is an ask the model raised knowing the referral,
    and its `id` is remembered. A response counts as the user's answer when it
    pairs by `id` with one of those asks and `_is_answer` holds. An ask issued
    before the referral, a call the tool or ADK rejected, and a skip do not
    count. A skip is the user declining to choose, which the planner may not
    substitute for, so the planner ends the turn by telling them what cannot be
    done.

    A `propose_plan` that recorded a version clears the referral in session
    state and translates the new version, so it ends the referral this walk is
    following and starts the one its `translation` names, if any. A refused
    proposal leaves both alone, which keeps the user's answer standing for the
    retry.

    With no referral on record the result is no answer. The guard refuses when
    it cannot say where the referral began, so an answer given for something
    else settles nothing.

    A spec_id counts as picked when an answer names it and the ask it answers,
    the nearest preceding one with its `id`, offered it, as `_elected_spec_ids`
    pairs them. A pick made before the referral is not counted.
    """
    answered = False
    referral: tuple[object, object] | None = None
    asks: dict[str, frozenset[str]] = {}
    picked: set[str] = set()
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
                asks[call.id] = _offered_candidates(call.args or {})
            response = part.function_response
            if response is None:
                continue
            body = response.response or {}
            if response.name == REQUEST_INPUT_FUNCTION_CALL_NAME:
                if response.id not in asks or not _is_answer(body):
                    continue
                answered = True
                result = body.get("result")
                spec_id = result.get("spec_id") if isinstance(result, dict) else None
                if spec_id is not None and str(spec_id) in asks[response.id]:
                    picked.add(str(spec_id))
            elif response.name == "propose_plan" and body.get("outcome") == Outcome.OK.value:
                translation = body.get("translation")
                referral = (
                    (translation.get("kind"), translation.get("requirement_id"))
                    if isinstance(translation, dict) and "kind" in translation
                    else None
                )
                answered = False
                asks.clear()
                picked.clear()
    return answered, frozenset(picked)


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


# The option of a combination ask that adds the two datasets.
COMBINE_OPTION = "add them"

# The options a combination ask offers, as `_combination_unasked`'s refusal
# hands them to `adk_request_input`: each `id` is what the user's pick comes
# back as, and Studio shows each `label`.
COMBINATION_OPTIONS: tuple[dict[str, str], ...] = (
    {"id": COMBINE_OPTION, "label": "Add them into one total"},
    {"id": "show them separately", "label": "Show them side by side, without adding them"},
)


def _combination_ask() -> Action:
    """The `adk_request_input` call that asks the user whether to add two datasets.

    It carries `COMBINATION_OPTIONS` and needs only the question. What it asks
    about is the refusal that carries it (`_combination_picks`).
    """
    return Action(
        call=REQUEST_INPUT_FUNCTION_CALL_NAME,
        arguments={"options": [dict(option) for option in COMBINATION_OPTIONS]},
        needs=("message",),
    )


# What a combination ask asks about: the `sum_of` requirement's id and the
# datasets its sides read (`_sum_datasets`).
type _Combination = tuple[str, frozenset[str]]


def _refused_combination(body: dict[str, Any]) -> _Combination | None:
    """The combination a `combination_unasked` refusal names in its `detail`, or `None`."""
    if body.get("code") != ProblemCode.COMBINATION_UNASKED.value:
        return None
    detail = body.get("detail")
    if not isinstance(detail, dict):
        return None
    requirement_id = detail.get("requirement_id")
    datasets = detail.get("datasets")
    if not isinstance(requirement_id, str) or not isinstance(datasets, list):
        return None
    named = [dataset for dataset in datasets if isinstance(dataset, str)]
    if len(named) != len(datasets):
        return None
    return requirement_id, frozenset(named)


def _combination_picks(
    tool_context: ToolContext,
) -> tuple[dict[_Combination, str | None], bool]:
    """The option id the user last picked for each combination they were asked about, and a flag.

    A combination ask is an `adk_request_input` call whose `options` are
    `COMBINATION_OPTIONS`. It asks about the combination the latest
    `combination_unasked` refusal of `propose_plan` before it names, read off
    that refusal's own response event, so the model cannot point an ask at
    another sum. A refusal binds the first ask after it and no other, so an
    ask with no refusal of its own before it asks about nothing. Each
    response pairs with the nearest preceding call sharing its `id`, as
    `_elected_spec_ids` pairs them. Studio sends a picked option's id as
    `{"spec_id": id}`. Each answer about a combination replaces the one
    before it, and free text or a skip records `None`, so a user who changes
    their mind, or answers outside the options, ends with no pick for it.
    The flag says whether any answer to any combination ask, about a
    combination or about nothing, was other than `COMBINE_OPTION`.
    """
    offered = [dict(option) for option in COMBINATION_OPTIONS]
    ids = {option["id"] for option in COMBINATION_OPTIONS}
    refused: _Combination | None = None
    asks: dict[str, _Combination | None] = {}
    picks: dict[_Combination, str | None] = {}
    declined = False
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
                if (call.args or {}).get("options") == offered:
                    asks[call.id], refused = refused, None
                else:
                    asks.pop(call.id, None)
                continue
            response = part.function_response
            if response is None:
                continue
            body = response.response or {}
            if response.name == "propose_plan":
                refused = _refused_combination(body) or refused
            elif response.name == REQUEST_INPUT_FUNCTION_CALL_NAME and response.id in asks:
                result = body.get("result")
                option = result.get("spec_id") if isinstance(result, dict) else None
                declined = declined or option != COMBINE_OPTION
                combination = asks[response.id]
                if combination is not None:
                    picks[combination] = option if option in ids else None
    return picks, declined


def _election_untraceable(
    decisions: Sequence[Decision], tool_context: ToolContext
) -> dict[str, object] | None:
    """Return the refusal for a `resolved_by=user` decision the user never made, or `None`.

    A `resolved_by=user` source pin is model-self-reported, so it is checked
    against what the user actually chose: an answer to an `adk_request_input`
    call on record in this session (`_elected_spec_ids`). A pin naming a
    spec_id that was never offered, was offered but not chosen, or was
    offered with no answer on record yet is refused. `Plan`'s own validators
    have no session to read (see `planner.plan.DatasetRequirement`), so they
    cannot check this.
    """
    elected = _elected_spec_ids(tool_context)
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


def _stated_source_given_way(
    requirements: Sequence[EvidenceRequirement], tool_context: ToolContext
) -> list[EvidenceRequirement]:
    """`requirements`, with the stated `source` dropped from a requirement that pins the source the user chose.

    A pending `requirement_unmeetable` referral marked `source_mismatch`
    hands a requirement back because the source it states names none of the
    sources discovery found, and offers those. A pin of one of them that the
    user picked, answering an ask raised after the referral
    (`_answers_since_referral`), names the exact source they chose, so the
    stated source gives way to it. Left in place, it would refuse that source
    as `compiler.evidence`'s `SOURCE_MISMATCH` and refer the plan back the
    same way. Any other pin keeps the stated source, including one the planner
    settled itself after a free-text answer and one the user picked before the
    referral, so the compiler still refuses a source the user neither named
    nor chose.
    """
    referral = load_referral(tool_context.state)
    if referral is None or not referral.source_mismatch:
        return list(requirements)
    offered = {candidate.spec_id for candidate in referral.candidates}
    chosen = offered & _answers_since_referral(tool_context)[1]
    return [
        requirement.model_copy(update={"source": None})
        if isinstance(requirement, DatasetRequirement)
        and requirement.id == referral.requirement_id
        and requirement.source_spec_id in chosen
        else requirement
        for requirement in requirements
    ]


def _choice_not_offered(plan: Plan, tool_context: ToolContext) -> dict[str, object] | None:
    """Return the refusal for a pin a pending source choice did not offer, or `None`.

    A `choice_required` referral names the sources each requirement it is
    about may pin (`Referral.offered`): its `candidates`, and each stale
    pin's, which discovery recorded and no model wrote. While it is pending,
    each such requirement as `plan` states it either pins one of them or
    pins nothing. A requirement the plan no longer states passes.

    A requirement whose found sources the referral withholds, because they
    drop the scope of the pinned source it set aside, may pin nothing only
    when it also changes its `entities`, `across` or `source` from the plan the
    referral is about: stated unpinned and otherwise unchanged, it would
    have the translator's search find those sources again
    (`_withheld_scope_dropped`).
    """
    referral = load_referral(tool_context.state)
    if referral is None or referral.outcome is not Outcome.CHOICE_REQUIRED:
        return None
    for requirement_id, candidates in referral.offered().items():
        try:
            requirement = plan.requirement(requirement_id)
        except KeyError:
            continue
        if not isinstance(requirement, DatasetRequirement):
            continue
        if requirement.source_spec_id is None:
            refused = _withheld_scope_dropped(requirement, referral, tool_context)
            if refused is not None:
                return refused
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


def _withheld_scope_dropped(
    requirement: DatasetRequirement, referral: Referral, tool_context: ToolContext
) -> dict[str, object] | None:
    """Return the refusal for `requirement`, stated unpinned, that would retrieve a source `referral` withholds, or `None`.

    It is refused when `referral` withholds sources found in its place
    (`translator.referral.WithheldSource`) and it asks `search` for what the
    recorded plan the referral is about asked (`_search_scope`).
    """
    if requirement.id == referral.requirement_id:
        withheld, repair = referral.withheld, referral.repair
    else:
        stale = next(
            (pin for pin in referral.stale_pins if pin.requirement_id == requirement.id), None
        )
        withheld, repair = (stale.withheld, stale.repair) if stale is not None else ((), None)
    recorded = load_plan(tool_context.state)
    if not withheld or recorded is None:
        return None
    try:
        referred = recorded.requirement(requirement.id)
    except KeyError:
        return None
    if not isinstance(referred, DatasetRequirement) or (
        _search_scope(requirement) != _search_scope(referred)
    ):
        return None
    return problem(
        ProblemCode.SOURCE_NOT_OFFERED,
        f"{requirement.id!r} is unpinned but asks for what it asked before, so the search "
        "would find again the sources the referral withholds for dropping the pinned "
        "source's scope. "
        + (
            f"Make the referral's repair, {repair}, and propose again."
            if repair is not None
            else f"Ask the user through `{REQUEST_INPUT_FUNCTION_CALL_NAME}` what to change."
        ),
        outcome=Outcome.REFUSED,
        detail={"requirement_id": requirement.id, "repair": repair},
    )


def _search_scope(
    requirement: DatasetRequirement,
) -> tuple[
    frozenset[tuple[str, str | None, bool]], tuple[str, ...], tuple[str | None, str | None, bool]
]:
    """The scope `search` is asked for `requirement`: its entities, open roles and named source.

    The entities are a set of names, casefolded with runs of whitespace as
    one space, each with the `PARTY_ROLES` word its role stands for
    (`party_role`) and whether it asks for the set's members, and the open
    roles are `translator.discovery.requirement_open_roles`.
    Two requirements stating their entities in another order, or their
    `across` in other words for one role ("recipient", "recipients"), have
    one scope. The named source follows Search's word matching: casefold,
    punctuation as spaces, a leading "the" dropped and whitespace collapsed.
    """
    entities = frozenset(
        (
            " ".join(entity.name.casefold().split()),
            party_role(entity.role) if entity.role is not None else None,
            entity.members,
        )
        for entity in requirement.entities
    )
    source = None
    if requirement.source is not None:
        words = "".join(
            " " if unicodedata.category(char).startswith("P") else char
            for char in requirement.source.casefold()
        ).split()
        if len(words) > 1 and words[0] == "the":
            words = words[1:]
        source = " ".join(words)
    return (
        entities,
        tuple(requirement_open_roles(requirement)),
        (source, requirement.source_ref, requirement.source_exclude),
    )


def _pinned_source(spec_id: str | None, tool_context: ToolContext) -> SelectedSource | None:
    """The source `spec_id` pins, as the record that states it; `None` when nothing on record does.

    The source is a block's the planner was shown, or one the
    translator's discovery recorded, such as a referral's candidate
    (`translator.discovery.DiscoveryLog.recorded_source`).
    """
    if spec_id is None:
        return None
    shown = pin_sources(tool_context.state).get(spec_id)
    block = shown.spec_block(spec_id) if shown is not None else None
    if block is not None:
        try:
            return source_from_spec_block(block)
        except ValueError:
            return None
    return load_log(tool_context.state).recorded_source(spec_id)


class _SumDatasets(NamedTuple):
    """The datasets both sides of a `sum_of` read, and what their pins state of them.

    `stated` says a pin states each dataset's provenance. `provider_flows`
    says each pin also states its provenance's account is `PROVIDER_ACCOUNT`:
    the OECD and ONE aid tables, whose skill says how they add up.
    """

    datasets: frozenset[str]
    stated: bool
    provider_flows: bool


def _sum_datasets(
    plan: Plan, requirement: SumOfRequirement, tool_context: ToolContext
) -> _SumDatasets:
    """The datasets both sides of `requirement` read in `plan`, and what their pins state.

    A dataset requirement whose pin states its dataset (`_pinned_source`)
    reads that dataset's provenance ref. One whose dataset no pin states
    reads a dataset of its own: `spec:<source_spec_id>` for a pin whose
    dataset this turn cannot read, `requirement:<id>:<revision>` for an unpinned one, where the revision
    (`requirement_revision`) hashes what the requirement asks for. So two such
    sides read one dataset only when they share the pin, or the requirement
    with the content it was asked about.
    """
    datasets: set[str] = set()
    stated = True
    provider_flows = True
    pending = [requirement.left.input_id, requirement.right.input_id]
    while pending:
        read = plan.requirement(pending.pop())
        if isinstance(read, DatasetRequirement):
            source = _pinned_source(read.source_spec_id, tool_context)
            prior = pin_sources(tool_context.state).get(read.source_spec_id or "")
            dataset = (
                source.provenance_ref
                if source is not None
                else prior.provenance_ref
                if prior is not None
                else None
            )
            account = (
                source.provenance_account
                if source is not None
                else prior.provenance_account
                if prior is not None
                else None
            )
            if dataset is None:
                stated = False
                dataset = (
                    f"requirement:{read.id}:{requirement_revision(read)}"
                    if read.source_spec_id is None
                    else f"spec:{read.source_spec_id}"
                )
            if account != PROVIDER_ACCOUNT:
                provider_flows = False
            datasets.add(dataset)
        pending.extend(inputs_of(read))
    return _SumDatasets(frozenset(datasets), stated, provider_flows)


def _combination_unasked(
    plan: Plan, tool_context: ToolContext, *, previous: Plan | None
) -> dict[str, object] | None:
    """Return the refusal for a `sum_of` of two datasets the user was not asked to combine, or `None`.

    Whether two datasets add up without counting a flow twice is the
    user's to settle, since ODA loans, for one, count in both aid and debt
    flows. So a `sum_of` passes only when the user's latest pick for it,
    its id and the datasets its sides read (`_combination_picks`), is
    `COMBINE_OPTION`, and it cites a `Decision` resolved by `user` whose
    `chosen` is that option, unless its sides read one dataset, or only
    tables of a provider's flows (`_SumDatasets.provider_flows`), as the
    pins of their dataset requirements state. A `user` decision is the model's own claim,
    so citing one, even a source pin the user did elect, settles nothing
    without the pick. A pick for one sum, or for the same sum over other
    datasets, settles nothing for another.

    `plan` pins the sources the planner named, before `_window_pins` moves
    a pin to a copy over the requirement's period, which no store records a
    dataset for until the plan is recorded. A side with an unpinned dataset
    requirement, or a pin whose dataset this turn cannot read
    (`_pinned_source`), reads a dataset of its own, named by that
    requirement or pin (`_sum_datasets`). A Studio turn opens its own
    session, so a plan carried into a later turn has no pick on record.
    A `sum_of` that `previous`, the plan this call amends, states unchanged,
    over the same datasets and citing the same `user` decisions to add
    them, was checked when `previous` was recorded and keeps its sum,
    unless some combination ask in this session was answered with anything
    but `COMBINE_OPTION`. The refusal's action carries the `COMBINATION_OPTIONS`
    the ask offers, so the user picks one of them and the planner phrases
    only the question, and its `detail` names the sum and its datasets,
    which is what the ask that follows asks about.
    """
    picks, declined = _combination_picks(tool_context)
    for requirement in plan.evidence_requirements:
        if not isinstance(requirement, SumOfRequirement):
            continue
        datasets, stated, provider_flows = _sum_datasets(plan, requirement, tool_context)
        if stated and (len(datasets) == 1 or provider_flows):
            continue
        consent = _consenting_decisions(plan, requirement)
        pick = picks.get((requirement.id, datasets))
        if consent and pick == COMBINE_OPTION:
            continue
        if (
            consent
            and previous is not None
            and not declined
            and requirement in previous.evidence_requirements
            and _sum_datasets(previous, requirement, tool_context).datasets == datasets
            and _consenting_decisions(previous, requirement) == consent
        ):
            continue
        return problem(
            ProblemCode.COMBINATION_UNASKED,
            f"{requirement.id!r} adds two datasets, and whether they add up without "
            "counting a flow twice is the user's to decide: ODA loans, for one, count in "
            f"both aid and debt flows. Ask the user through `{REQUEST_INPUT_FUNCTION_CALL_NAME}` "
            "with this action's `options` unchanged and a `message` naming that overlap. "
            f"If they pick {COMBINE_OPTION!r}, cite a `Decision` resolved by `user` whose "
            "`options` are those option ids and whose `chosen` is the one they picked; "
            f"otherwise remove {requirement.id!r} and show the two series separately.",
            outcome=Outcome.REFUSED,
            action=_combination_ask(),
            detail={
                "requirement_id": requirement.id,
                "datasets": sorted(datasets),
            },
        )
    return None


def _consenting_decisions(plan: Plan, requirement: SumOfRequirement) -> dict[str, Decision]:
    """The decisions `requirement` cites in `plan` resolved by `user` to `COMBINE_OPTION`, by slot."""
    return {
        decision.slot: decision
        for decision in plan.decisions
        if decision.slot in requirement.decision_ids
        and decision.resolved_by is ResolvedBy.USER
        and decision.chosen == COMBINE_OPTION
    }


def _source_ref_unpicked(
    plan: Plan, tool_context: ToolContext, previous: Plan | None
) -> dict[str, object] | None:
    """A new source reference must be a source option the user actually picked."""
    picked = _answers_since_referral(tool_context)[1]
    referral = load_referral(tool_context.state)
    for requirement in plan.evidence_requirements:
        if not isinstance(requirement, DatasetRequirement) or requirement.source_ref is None:
            continue
        old = (
            next((r for r in previous.evidence_requirements if r.id == requirement.id), None)
            if previous
            else None
        )
        if (
            isinstance(old, DatasetRequirement)
            and old.source_ref == requirement.source_ref
            and old.source == requirement.source
            and old.source_exclude == requirement.source_exclude
        ):
            continue
        offered = (
            {
                candidate.ref
                for group in referral.ambiguous_sources
                if " ".join(group.text.casefold().split())
                == " ".join((requirement.source or "").casefold().split())
                for candidate in group.candidates
            }
            if referral
            and referral.kind is ReferralKind.SOURCE_NAME_AMBIGUOUS
            and referral.requirement_id == requirement.id
            else set()
        )
        if requirement.source_ref not in picked or requirement.source_ref not in offered:
            return _invalid_plan(
                "The source_ref must be a source reference Search offered and the user picked. Ask the user before resolving source ambiguity."
            )
    return None


def _unrecordable(
    plan: Plan, tool_context: ToolContext, *, stated: Plan, previous: Plan | None
) -> dict[str, object] | None:
    """Return the refusal for a built plan `propose_plan` may not record, or `None`.

    A plan states at least one requirement, unless it declines every ask
    (`_nothing_to_plan`). A patch carries forward what it
    does not name, so the amended plan is checked whole for a pin a pending
    source choice did not offer (`_choice_not_offered`), for a
    sum of two datasets the user was not asked to combine
    (`_combination_unasked`, against `stated`, the plan before `_window_pins`
    moved its pins), for a
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
        _source_ref_unpicked(plan, tool_context, previous)
        or _choice_not_offered(plan, tool_context)
        or _combination_unasked(stated, tool_context, previous=previous)
        or _decline_unearned(plan, previous, tool_context)
        or _decline_missing(plan, tool_context)
    )


# The option a decision `propose_plan` writes for a search pin leaves the
# source unpinned by (`_synthesize_source_decisions`).
_UNPINNED_CHOICE = "leave the source unpinned"


def _pin_choice(spec_id: str) -> str:
    """The option a decision `propose_plan` writes for a search pin pins `spec_id` by."""
    return f"pin {spec_id}"


@dataclass(frozen=True)
class _MeasureChoice:
    """How a flow is counted, as the decision `propose_plan` writes for a pin of one of its measures states it.

    `options` are the measure search elected, then each it offers counting
    the flow another way (`planner.workflow.MeasureAlternative`, axis
    `what`), by label, and `spec_ids` the spec_id of each in the same order.
    `chosen` is the label of the one pinned.
    """

    options: tuple[str, ...]
    spec_ids: tuple[str, ...]
    chosen: str


def _measure_choices(
    shown: Mapping[str, PinSource], *, compiled: Sequence[SelectedSource] = ()
) -> dict[str, _MeasureChoice]:
    """The measure choice a pin of each spec_id states, for every measure a settled source shown offers.

    Families from the prior compiled plan are retained on an amendment;
    a shown source with the same id takes precedence.

    A settled source whose block elects a measure (`SelectedSource.
    elected_measure`) and offers another way of counting it maps to the
    choice with its own label chosen, and each of those others to the same
    choice with that one's label chosen. A source settled only as another
    count of one (`PinSource.other_count_of`) takes its choice from the
    source that offered it, since its own block names its own measure as
    if search had elected it, and states none when that source is not
    shown. A label over `MAX_OPTION_CHARS`, a
    repeated one and any past `MAX_DECISION_OPTIONS` are left out, and a
    source left with one option offers no choice. Price alternatives (axis
    `unit`) count the flow the same way in other prices and offer no choice here.
    """
    elections = {source.spec_id: source.elected_measure for source in compiled}
    for spec_id, source in shown.items():
        block = source.spec_block(spec_id) if source.settled and not source.other_count_of else None
        if block is None:
            continue
        try:
            elected = source_from_spec_block(block).elected_measure
        except ValueError:
            continue
        elections[spec_id] = elected
    choices: dict[str, _MeasureChoice] = {}
    for spec_id, elected in elections.items():
        if elected is None or len(elected.label) > MAX_OPTION_CHARS:
            continue
        labels = {elected.label: spec_id}
        for alternative in elected.other_counts():
            if len(alternative.label) <= MAX_OPTION_CHARS and len(labels) < MAX_DECISION_OPTIONS:
                labels.setdefault(alternative.label, alternative.spec_id)
        if len(labels) < 2:
            continue
        options, spec_ids = tuple(labels), tuple(labels.values())
        choices[spec_id] = _MeasureChoice(options=options, spec_ids=spec_ids, chosen=elected.label)
        for label, measure in labels.items():
            # A source search elected states its own choice, whatever
            # another source offers it as.
            choices.setdefault(
                measure, _MeasureChoice(options=options, spec_ids=spec_ids, chosen=label)
            )
    return choices


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
    new slot, so it stays a search pin. A pin-or-unpin choice is rewritten to
    name the new pin; a measure choice (`_MeasureChoice`) keeps its options.
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
            update["slot"] = slots[decision.slot]
            # A measure choice reads the same over any period.
            if decision.chosen == _pin_choice(decision.fixes.source_spec_id or ""):
                update |= {
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
    period (`planner.plan.WindowedPin.windowed_from`).
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


def _first_year_on_record(
    pin: str,
    *,
    shown: Mapping[str, PinSource],
    read: Mapping[str, FullHistory | None],
) -> FullHistory | None:
    """What is on record about every year the dataset `pin` holds; `None` when nothing is.

    Read from the Search answer the planner was shown for `pin`
    (`planner.plan.full_history`), or else from what `propose_plan` read for
    it over no window (`read`, `planner.plan.full_histories`).
    """
    source = shown.get(pin)
    block = source.spec_block(pin) if source is not None else None
    if block is not None and (history := full_history(block)) is not None:
        return history
    return read.get(pin)


def _open_window(
    plan: Plan,
    requirement: DatasetRequirement,
    *,
    shown: Mapping[str, PinSource],
    read: Mapping[str, FullHistory | None],
) -> dict[str, object]:
    """The window with no end year a pinned `requirement` stating `period_end_open` is retrieved over.

    It starts at the requirement's own first year when every period that
    follows the newest data over it keeps a fixed start. When one rolls
    from its start (`planner.plan.rolls_from_its_start`), the year it reads
    can sit before the stated period, so the window starts at the first
    year the dataset holds (`_first_year_on_record`) and the frame holds
    its whole history. That frame is refused here when Search counted more
    rows for it than one frame may hold (data-mcp's `max_frame_rows`); an
    uncounted one is refused by `get_data` itself (`over_ceiling`) if it is
    too large.

    Raises:
        RollingPlanError: `RollingIssue.NEEDS_OPEN_PERIOD` when no first year
            is on record: `OpenPeriodFault.FIRST_YEAR_UNKNOWN` before the
            dataset was read over no window, `FIRST_YEAR_READ_FAILED` when
            that read's answer stated none (`read` holds the pin with
            `None`). `RollingIssue.FRAME_TOO_LARGE` when the history is past
            the ceiling.
    """
    if not _rolls_from_its_start(plan, requirement):
        return period_window(requirement.period, end_open=True)
    pin = requirement.source_spec_id or ""
    history = _first_year_on_record(pin, shown=shown, read=read)
    if history is None and pin in read:
        raise RollingPlanError(
            RollingIssue.NEEDS_OPEN_PERIOD,
            f"{requirement.id!r} is read from a start that follows the newest data, and "
            f"Search's answer for {pin} over all its years states no first year, so the "
            "retrieval cannot be started there. This is a limit of what was read, not a "
            "fault in the question. State the start as a fixed year and roll only the "
            "end, or drop `rolling` and `period_end_open` for fixed years",
            requirement_ids=[requirement.id],
            fault=OpenPeriodFault.FIRST_YEAR_READ_FAILED,
            spec_id=pin,
        )
    if history is None:
        raise RollingPlanError(
            RollingIssue.NEEDS_OPEN_PERIOD,
            f"{requirement.id!r} is read from a start that follows the newest data, which "
            f"needs the first year {pin} holds, and no search result over all its years "
            "states one. State the start as a fixed year and roll only the end, or drop "
            "`rolling` and `period_end_open` for fixed years",
            requirement_ids=[requirement.id],
            fault=OpenPeriodFault.FIRST_YEAR_UNKNOWN,
            spec_id=pin,
        )
    ceiling = DataMcpSettings().max_frame_rows
    if history.rows is not None and history.rows > ceiling:
        raise RollingPlanError(
            RollingIssue.FRAME_TOO_LARGE,
            f"{requirement.id!r} is read from a start that follows the newest data, so it "
            f"retrieves every year {pin} holds: {history.rows:,} rows, over the {ceiling:,} "
            "one retrieval may return. Name fewer entities, state the start as a fixed "
            "year and roll only the end, or drop `rolling` and `period_end_open`",
            requirement_ids=[requirement.id],
        )
    return open_window(history.first_year)


@dataclass(frozen=True)
class _ShownDefinition:
    """The definition of a dataset the planner was shown, as its block carries it,
    with the sets its places were asked for as."""

    identity: Mapping[str, Any]
    groups: tuple[ResolvedGroup, ...] = ()
    resolved_sources: tuple[ResolvedSourceMention, ...] = ()


def _identity_on_record(
    pin: str,
    *,
    shown: Mapping[str, PinSource],
    end_open: bool,
    definition_lookup: Callable[[str], RecordedDefinition | None] | None = None,
) -> RecordedDefinition | None:
    """The definition on record for `pin`; `None` when none is.

    The definition is the one a search in the conversation recorded
    (`planner.plan.recorded_identity`). For a requirement stating
    `period_end_open` it is otherwise the one the block the planner was
    shown carries (`planner.plan.shown_identity`): the translator can find
    a closed requirement's pin by searching its stated years, and no search
    finds a definition with no end year.
    """
    recorded: RecordedDefinition | None = recorded_identity(pin)
    if recorded is None and definition_lookup is not None:
        recorded = definition_lookup(pin)
    if recorded is None and end_open and pin in shown:
        identity = shown_identity(pin, shown[pin])
        block = shown[pin].spec_block(pin)
        recorded = (
            _ShownDefinition(
                identity,
                block_groups(block) if block is not None else (),
                source_from_spec_block(block).resolved_sources if block is not None else (),
            )
            if identity is not None
            else None
        )
    return recorded


def _rolls_from_its_start(plan: Plan, requirement: DatasetRequirement) -> bool:
    """Whether a period over `requirement` follows the newest data from its start (`planner.plan.rolls_from_its_start`)."""
    requirements = {item.id: item for item in plan.evidence_requirements}
    return any(
        rolls_from_its_start(reader) for reader in rolling_readers(requirements, requirement.id)
    )


def _needs_a_pin(plan: Plan, requirement: DatasetRequirement) -> bool:
    """Whether `requirement`, stating `period_end_open` with no identity on record, cannot be retrieved open.

    An unpinned one whose start is fixed can: the translator searches for it
    from its first year on (`translator.discovery`), and Search answers with
    a definition that has no end year. One read from a start that follows
    the newest data needs the first year its dataset holds, which only a
    dataset on record has. A pin with nothing on record is confirmed by a
    search over the stated years, which finds a fixed end.
    """
    return requirement.source_spec_id is not None or _rolls_from_its_start(plan, requirement)


def _window_pins(
    plan: Plan,
    *,
    shown: Mapping[str, PinSource] = {},
    read: Mapping[str, FullHistory | None] = {},
    definition_lookup: Callable[[str], RecordedDefinition | None] | None = None,
) -> tuple[Plan, list[WindowedPin]]:
    """`plan` with each pin whose identity is on record moved to that identity over the requirement's own period.

    A pin's identity carries the years of the search that found it, the
    window of the question that search answered. A plan's retrieval mints
    the pinned spec_id as it is (`compiler.compile`'s
    `_retrieve_arguments`). So a dataset requirement whose pin's identity is
    over another period is pinned instead to that identity over its own
    (`data_mcp.definition_writes.windowed_identity`, the spec_id `get_data`'s
    `scope.window` would mint), and so is the decision that fixes the pin.

    The identity on record is the one a search in the conversation recorded
    (`planner.plan.recorded_identity`). One of a `latest` window is copied
    like any other: the copy replaces the window whole, as `get_data`'s
    scope surgery does, and the translator confirms the copy through Search
    before it is retrieved.

    A requirement stating `period_end_open` is pinned to that identity over
    a range with no end year (`_open_window`), which dfo-apis ends at the
    current year at every mint, so its retrieval reaches years published
    after the plan. `shown` are the datasets the planner was shown this
    entry (`planner.plan.pin_sources`) and `read` what `propose_plan` read
    over no window (`planner.plan.full_histories`), both read for the first
    year one holds.
    Such a requirement needs an identity on record (`_identity_on_record`),
    unless it is unpinned and its start is fixed (`_needs_a_pin`).

    Returns the moved plan and every copy it pins, each a
    `planner.plan.WindowedPin`. A copy states nothing Search stated for the
    original's years; the translator reads those from Search's answer for
    the copy when it honours the pin (`translator.discovery`).

    One decision fixing the pin of requirements over different periods
    cannot name every copy with its one `source_spec_id`. Those
    requirements keep the pin as the planner wrote it, and the translator
    searches for each.

    Raises:
        ValueError: If one decision fixes the pin of a requirement stating
            `period_end_open` for requirements over different periods.
        RollingPlanError: If a requirement stating `period_end_open` needs an
            identity on record and has none, or `_open_window` refuses it.
    """
    pinning: dict[str, str] = {}
    copies: dict[str, WindowedPin | None] = {}
    open_ended: set[str] = set()
    for requirement in plan.evidence_requirements:
        if not isinstance(requirement, DatasetRequirement):
            continue
        pin = requirement.source_spec_id
        recorded = (
            _identity_on_record(
                pin,
                shown=shown,
                end_open=requirement.period_end_open,
                definition_lookup=definition_lookup,
            )
            if pin is not None
            else None
        )
        if requirement.period_end_open and recorded is None and _needs_a_pin(plan, requirement):
            raise RollingPlanError(
                RollingIssue.NEEDS_OPEN_PERIOD,
                f"{requirement.id!r} states `period_end_open`, which takes a dataset on "
                "record to retrieve with no end year: pin a search result's `spec_id` as its "
                "`source_spec_id`, or drop `period_end_open` and every `rolling` reading it "
                "for fixed years",
                requirement_ids=[requirement.id],
                fault=OpenPeriodFault.NO_SOURCE_ON_RECORD,
            )
        if pin is None or recorded is None:
            continue
        pinning[requirement.id] = pin
        if requirement.period_end_open:
            open_ended.add(requirement.id)
            window = _open_window(plan, requirement, shown=shown, read=read)
        else:
            window = period_window(requirement.period)
        if recorded.identity.get("requested_window") == window:
            copies[requirement.id] = None
        else:
            copies[requirement.id] = windowed_pin(pin, recorded, window)
    targets: dict[str, set[str]] = {}
    for requirement in plan.evidence_requirements:
        if requirement.id not in copies:
            continue
        copy = copies[requirement.id]
        target = pinning[requirement.id] if copy is None else copy.spec_id
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
            citing = {
                requirement.id
                for requirement in plan.evidence_requirements
                if decision.slot in requirement.decision_ids
            }
            if citing & open_ended:
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
        requirement_id: copy.spec_id for requirement_id, copy in copies.items() if copy is not None
    }
    if not requirement_pins and not decision_pins:
        return plan, []
    recorded_copies: dict[str, WindowedPin] = {}
    for copy in copies.values():
        if copy is not None:
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
    return Plan.model_validate(moved.model_dump()), list(recorded_copies.values())


def _synthesize_source_decisions(
    decisions: Sequence[Decision],
    evidence_requirements: Sequence[EvidenceRequirement],
    *,
    previous: Plan | None,
    settled_spec_ids: Collection[str],
    earlier_spec_ids: Collection[str] = (),
    refound_spec_ids: Collection[str] = (),
    measures: Mapping[str, _MeasureChoice] = {},
    user_settled: Collection[str] = (),
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

    A pin of a measure a shown source offers (`measures`) gets the measure
    decision instead: its options are the measure search elected and the
    others counting the flow another way, by label, and it chooses the
    pinned one's. A pin of one of those others is allowed as a settled pin
    is, so a re-pin to a sibling measure needs no decision of the model's.
    The user settled it (`resolved_by=user`) when the pin is in
    `user_settled`, else the planner did.

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
        measures: The measure choice a pin of each spec_id states
            (`_measure_choices`).
        user_settled: The spec_ids of `measures` whose measure the user
            chose (`_user_settled_measures`).

    Returns:
        `decisions`/`evidence_requirements`, unchanged unless some requirement
        needed a pin cited. When so, each such requirement's `decision_ids`
        cites the synthesized decision's slot, and each decision is appended
        to `decisions` unless a prior round already recorded it.
    """
    known = {decision.slot: decision for decision in (previous.decisions if previous else ())}
    known.update({decision.slot: decision for decision in decisions})
    pinnable = {*settled_spec_ids, *refound_spec_ids, *measures}

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
        known[slot] = (
            _measure_decision(slot, pin, measures[pin], user=pin in user_settled)
            if pin in measures
            else Decision(
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
        )
        synthesized.append(known[slot])
    return [*decisions, *synthesized], patched_requirements


def _measure_decision(slot: str, pin: str, choice: _MeasureChoice, *, user: bool) -> Decision:
    """The decision `_synthesize_source_decisions` writes for a pin of `pin`, one measure of `choice`."""
    elected = choice.options[0]
    if user:
        rationale = f"The user chose {choice.chosen}."
    elif choice.chosen == elected:
        rationale = f"Search elected {elected} for this source."
    else:
        rationale = f"Chosen in place of {elected}, the measure search elected."
    return Decision(
        slot=slot,
        description="How the flow is counted.",
        options=choice.options,
        chosen=choice.chosen,
        resolved_by=ResolvedBy.USER if user else ResolvedBy.PLANNER_DEFAULT,
        rationale=rationale,
        fixes=DecisionFixes(source_spec_id=pin),
    )


def _user_settled_measures(
    measures: Mapping[str, _MeasureChoice], tool_context: ToolContext
) -> frozenset[str]:
    """The spec_ids of `measures` whose measure the user picked answering an
    `adk_request_input` call: the orchestrator's `ask_user` offers each
    measure under its own spec_id, so a pick names it (`_elected_spec_ids`).
    A typed answer or an amendment at approval names no spec_id, so the
    measure it leads to stays the planner's."""
    return frozenset(_elected_spec_ids(tool_context) & measures.keys())


def _user_settled_rewrites(
    previous: Plan | None,
    decisions: Sequence[Decision],
    requirements: Sequence[EvidenceRequirement],
    *,
    measures: Mapping[str, _MeasureChoice],
    user_settled: Collection[str],
) -> list[Decision]:
    """Each measure decision `previous` records as the planner's that the user has since settled, as theirs.

    `_synthesize_source_decisions` writes a measure decision when a
    requirement first pins the measure, so a pick answering an ask raised
    after that leaves the planner's decision standing on the same pin. Only
    a decision some requirement of the patched plan (`requirements`) cites,
    and that this call's `decisions` do not restate, is rewritten.
    """
    if previous is None:
        return []
    restated = {decision.slot for decision in decisions}
    cited = {slot for requirement in requirements for slot in requirement.decision_ids}
    rewritten: list[Decision] = []
    for decision in previous.decisions:
        pin = decision.fixes.source_spec_id if decision.fixes is not None else None
        choice = measures.get(pin) if pin is not None else None
        if (
            pin is not None
            and choice is not None
            and pin in user_settled
            and is_search_pin(decision)
            and decision.resolved_by is ResolvedBy.PLANNER_DEFAULT
            and decision.slot in cited
            and decision.slot not in restated
        ):
            rewritten.append(_measure_decision(decision.slot, pin, choice, user=True))
    return rewritten


def _settled_decisions(
    drafts: Sequence[DecisionDraft],
    *,
    existing: Plan | None,
    requirements_after: Sequence[EvidenceRequirement],
    tool_context: ToolContext,
) -> list[Decision] | dict[str, object]:
    """This call's decisions built from `drafts`, or the refusal that stops the call first.

    These checks run before the plan is touched, so a refusal records nothing
    and spends no round: a referral the user has not answered
    (`_referral_unanswered`), a draft fixing a field its requirements do not
    settle (`planner.plan.settle_decisions`), and a user election the user
    never made (`_election_untraceable`).

    Args:
        drafts: This call's `decisions` argument.
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
    return _election_untraceable(decisions, tool_context) or decisions


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
        "What kind of answer the requirements add up to. `series` answers a "
        "question that asks to see a level or a trend: a lone dataset "
        "requirement, unread by anything else, is itself the answer, its frame "
        "carrying one series per entity for a chart to draw. `computed` answers "
        "a question that asks for a figure the series does not state on its "
        "own (a difference, growth, share, quotient, rank or total), so every "
        "dataset requirement has to feed a calculation, since one left unread "
        "would leave the plan without the figure the question asked for. "
        "Either way, `answered` may name a dataset requirement a calculation "
        "reads. "
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
        "answer's code: one `propose_plan` answered, itself or in its `translation`, "
        "or a run-time refusal the plan is replanned from that names the code. A "
        "`requirement_unmeetable` translation whose `reason` says it earns a "
        "decline earns that code too, which the plan may make or leave out."
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


def _first_year_wanted(result: Mapping[str, object]) -> str | None:
    """The pinned dataset a refusal says no first year is on record for; `None` for any other answer."""
    detail = result.get("detail")
    if (
        not isinstance(detail, Mapping)
        or detail.get("fault") != OpenPeriodFault.FIRST_YEAR_UNKNOWN.value
    ):
        return None
    spec_id = detail.get("spec_id")
    return spec_id if isinstance(spec_id, str) else None


# The seconds one first-year read may take (`_read_full_history`). It runs
# inside a `propose_plan` call, so inside the planner entry's remaining time,
# which `translator.discovery.resolve_definition` already holds it to; this
# keeps a slow Search from spending that whole remainder on one read.
FIRST_YEAR_READ_DEADLINE_SECONDS = 10.0


async def _read_full_history(
    surface: ToolSurface, identity: Mapping[str, Any]
) -> FullHistory | None:
    """What Search states about every year the dataset of `identity` holds, read now.

    One `resolve_definition` call for `identity` over no window, the same
    read the translator makes for a pin's copy
    (`translator.discovery.resolve_definition`): data-mcp posts the identity
    to Search once and stores the definition it answers. `None` when Search
    does not answer with that one dataset, or its answer states no first
    year (`planner.plan.full_history`). The window is read for its first
    year only and is never retrieved.

    Raises:
        DefinitionNotResolvedError: If the call did not complete, or did not
            answer within `FIRST_YEAR_READ_DEADLINE_SECONDS`, and can succeed
            later (`retryable`).
    """
    spec_id, unwindowed = windowed_identity(identity, {"kind": "all"})
    try:
        async with asyncio.timeout(FIRST_YEAR_READ_DEADLINE_SECONDS):
            part = await resolve_definition(surface, spec_id=spec_id, identity=unwindowed)
        return full_history(resolved_spec(part, spec_id=spec_id))
    except TimeoutError as exc:
        raise DefinitionNotResolvedError(
            f"Search did not answer in {FIRST_YEAR_READ_DEADLINE_SECONDS:.0f} s", retryable=True
        ) from exc
    except DefinitionNotResolvedError as exc:
        if exc.retryable:
            raise
        return None
    except PartNotResolvedError:
        return None


def _first_year_unreadable(
    refused: Mapping[str, object], exc: DefinitionNotResolvedError
) -> dict[str, object]:
    """Return the `SURFACE_UNAVAILABLE` problem for a first-year read that did not complete.

    `refused` is the `first_year_unknown` refusal the read was made for; its
    requirements and dataset carry over, under the fault
    `first_year_read_failed`. The same proposal can succeed later, so
    nothing in it needs changing.
    """
    detail = dict(cast("Mapping[str, JsonValue]", refused["detail"]))
    detail["fault"] = OpenPeriodFault.FIRST_YEAR_READ_FAILED.value
    return problem(
        ProblemCode.SURFACE_UNAVAILABLE,
        f"The first year {detail['spec_id']} holds could not be read: {exc}. The plan is "
        "sound and nothing was recorded; propose it again, or state the start as a fixed "
        "year and roll only the end.",
        outcome=Outcome.REFUSED,
        detail=detail,
    )


def build_planner_tools(
    *,
    translate: Translate | None = None,
    question_fixed: bool = False,
    surface: ToolSurface | None = None,
) -> list[FunctionTool]:
    """Build the planner's tool set.

    `translate` is the translation (`translator.tool.build_translate`) a
    `propose_plan` that records a plan runs on it, answering with the verdict
    under `translation` alongside its own `ok`. Given, `propose_plan` is a
    coroutine function. Omitted, `propose_plan` records the plan and stops,
    and `submit_plan`'s refusal of an uncompiled plan names no next call.

    `surface` is the tool surface `propose_plan` reads a pinned dataset's
    first year through, when a start that follows the newest data needs one
    no answer on record states (`_read_full_history`). Given, `propose_plan`
    is a coroutine function. Omitted, it reads nothing and refuses such a
    plan `first_year_unknown`.

    `question_fixed` closes over into `propose_plan`, which passes it straight
    to `_resolve_intent` on every call, and selects `_FIXED_ARGUMENTS`'
    descriptions of `question` and `starts_new_plan`, so those two arguments
    describe the mode this tool set actually enforces.
    """
    has_translator = translate is not None

    # Named `ask_user` here, for readability, but renamed below (once
    # defined) to `adk_request_input` -- the function-call name ADK's own
    # `google.adk.tools.request_input` uses for exactly this pausing shape
    # (`REQUEST_INPUT_FUNCTION_CALL_NAME`), reused rather than duplicated so
    # `harness.studio_translate._request_input_call` recognises this pause
    # the same way it already recognises the graph's own `ask`/`approve`
    # interrupts, with no change to that recognition code. The model sees
    # and calls this tool under that reused name, not `ask_user`, so the
    # instruction above names it that way too.
    def ask_user(
        message: str,
        candidates: list[dict[str, str]] = [],  # noqa: B006
        options: list[dict[str, str]] = [],  # noqa: B006
        recommended_spec_id: str | None = None,
        rationale: str | None = None,
    ) -> dict[str, object] | None:
        """Ask the user a question a referral or a refusal leaves to them, pausing the turn.

        Returns nothing here: the user's answer arrives as this call's own
        response on your next turn, naming under `spec_id` the candidate or
        option id they picked, or free text if they answered outside them. A
        `resolved_by=user` decision may pin only a `spec_id` this call offered
        and the user picked. The choices offered come from the referral's
        candidates, or from the `options` a refusal's action carries; with
        neither, the question says what cannot be done and asks what to change.

        Args:
            message: The question to show the user, naming what is
                ambiguous and, briefly, what each candidate is, in plain
                words: never a requirement id, spec_id or code.
            candidates: The referral's own candidates, each
                `{"spec_id": ..., "label": ...}` -- `label` is the
                candidate's own `sentence`, not invented text. Omit it for
                a referral that carries none. For source_name_ambiguous, use
                each offered source's ref as spec_id and its label as label;
                the chosen reference belongs in source_ref, never source_spec_id.
            options: The `options` a `combination_unasked` or
                `election_untraceable` refusal's action carries, unchanged.
                Omit it otherwise.
            recommended_spec_id: The candidate you favour, when you favour
                one -- must name one of `candidates`.
            rationale: Why you favour it, for the user to read, when
                `recommended_spec_id` is given.
        """
        del message, rationale  # carried through to the wire frame, not read back here
        if options and (candidates or options != [dict(o) for o in COMBINATION_OPTIONS]):
            return problem(
                ProblemCode.INVALID_ARGUMENTS,
                "`options` must be the ones a refusal's action carries, unchanged, "
                "and offered without `candidates`.",
                outcome=Outcome.INVALID_INPUT,
            )
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
        to_origin = {copy.spec_id: copy.windowed_from for copy in held_copies.values()}
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
        pins = pin_sources(tool_context.state)
        measures = _measure_choices(
            pins,
            compiled=previous.workflow.sources
            if previous is not None and previous.workflow
            else (),
        )
        user_settled = _user_settled_measures(measures, tool_context)
        settled = _settled_decisions(
            patch.decisions,
            existing=existing,
            requirements_after=_requirements_after(
                previous, evidence_requirements, removed_evidence_requirements
            ),
            tool_context=tool_context,
        )
        if isinstance(settled, dict):
            return settled
        evidence_requirements = _stated_source_given_way(evidence_requirements, tool_context)
        # A verbatim pin of a spec_id a search settled on gets the citing
        # `Decision` it needs written here, on the model's behalf, so
        # `Plan`'s own validator never has to refuse it for lacking one.
        # See `_synthesize_source_decisions`.
        settled, evidence_requirements = _synthesize_source_decisions(
            settled,
            evidence_requirements,
            previous=previous,
            settled_spec_ids={spec_id for spec_id, source in pins.items() if source.settled},
            earlier_spec_ids={spec_id for spec_id, source in pins.items() if source.earlier_turn},
            refound_spec_ids=_refound_spec_ids(tool_context),
            measures=measures,
            user_settled=user_settled,
        )
        requirements_after = _requirements_after(
            previous, evidence_requirements, removed_evidence_requirements
        )
        settled = [
            *settled,
            *_user_settled_rewrites(
                previous,
                settled,
                requirements_after,
                measures=measures,
                user_settled=user_settled,
            ),
        ]
        removed_decisions = [
            *removed_decisions,
            *_uncited_source_decisions(
                previous,
                settled,
                requirements_after,
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
            stated = plan
            plan, copies = _window_pins(
                stated,
                shown=pins,
                read=full_histories(tool_context.state),
                definition_lookup=load_log(tool_context.state).recorded_alternative,
            )
        except (ValidationError, ValueError) as exc:
            # The plan-level rules: unique slots and ids, every citation naming
            # a decision, every decision cited,
            # the length and count limits on each field, a question that is not
            # blank, and the size limit on the whole. A removal naming an item
            # the plan does not state is a `ValueError` from `Plan.amend` itself,
            # and a first draft or restart with no `question`/`answer_mode` is
            # one raised above. `_window_pins` raises one for a decision
            # fixing an open-ended pin of requirements over different periods.
            cause = _unsupported_cause(exc)
            if cause is not None:
                record_unsupported(tool_context.state, cause.operation)
            return _invalid_data(exc)
        blocked = _unrecordable(plan, tool_context, stated=stated, previous=previous)
        if blocked is not None:
            return blocked
        if copies or held_copies:
            # The copies this plan pins. A copy that only a replaced plan
            # pinned is let go, so however many drafts and amendments the
            # turn makes, the state holds at most one copy per dataset
            # requirement of the recorded plan.
            save_windowed_pins(tool_context.state, copies)
        save_plan(tool_context.state, plan)
        save_plan_rounds(tool_context.state, rounds)
        # A new proposal supersedes whatever the translator referred back.
        clear_referral(tool_context.state)
        # Only what the model cannot read back off its own call, and the
        # question as stored. The decisions and requirements the model sent are
        # echoed nowhere, because the plan is in session state and
        # translation reads it from there. `status` is always the
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

    @functools.wraps(propose_plan)
    async def propose_plan_reading(**arguments: Any) -> dict[str, object]:
        """`propose_plan`, reading each first year it refuses for want of.

        A refusal records nothing, so the proposal is made again once the
        read is held (`planner.plan.save_full_history`). One read serves
        every requirement, and every later proposal of the turn, that pins
        the dataset; an answer stating no first year is held too, and the
        next refusal says so (`first_year_read_failed`).
        """
        assert surface is not None
        state = arguments["tool_context"].state
        while True:
            result = propose_plan(**arguments)
            pin = _first_year_wanted(result)
            if pin is None or pin in full_histories(state):
                return result
            recorded = _identity_on_record(pin, shown=pin_sources(state), end_open=True)
            if recorded is None:
                return result
            try:
                history = await _read_full_history(surface, recorded.identity)
            except DefinitionNotResolvedError as exc:
                return _first_year_unreadable(result, exc)
            save_full_history(state, pin, history)

    @functools.wraps(propose_plan)
    async def propose_plan_translating(**arguments: Any) -> dict[str, object]:
        """`propose_plan`, translating the plan it records.

        Translation runs once, on the version the proposal recorded, after
        any first-year reads (`propose_plan_reading`). The proposal's own
        `outcome` stays `ok` whatever the verdict, since the version is
        recorded either way. A proposal that records nothing translates
        nothing. A bare re-proposal after a transient verdict
        (`_transient_retry`) translates the recorded version again and
        records nothing.
        """
        assert translate is not None
        tool_context = arguments["tool_context"]
        retried = _transient_retry(arguments, tool_context.state)
        if retried is not None:
            clear_referral(tool_context.state)
            return ok(
                status=retried.status.value,
                version=retried.version,
                previous_version=retried.previous_version,
                question=retried.question,
                translation=await translate(tool_context),
            )
        result = (
            propose_plan(**arguments)
            if surface is None
            else await propose_plan_reading(**arguments)
        )
        if result.get("outcome") != Outcome.OK.value:
            return result
        return {**result, "translation": await translate(arguments["tool_context"])}

    ask_user.__name__ = REQUEST_INPUT_FUNCTION_CALL_NAME
    proposing: Callable[..., Any] = propose_plan
    if translate is not None:
        proposing = propose_plan_translating
    elif surface is not None:
        proposing = propose_plan_reading
    return [
        LongRunningFunctionTool(ask_user),
        PlanTool(
            proposing,
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
    translate: Translate | None = None,
    surface: ToolSurface | None = None,
    question_fixed: bool = False,
    prefix_cache: PrefixCache | None = None,
) -> LlmAgent:
    """Build the planner `LlmAgent`.

    Args:
        model_name: The model this agent runs on.
        translate: The translation (`translator.tool.build_translate`)
            `propose_plan` runs on every plan it records, which is how a
            draft plan gets translated and compiled. It runs inside
            `propose_plan`, so the model spends no generation on it. Omitted, the
            planner is the standalone agent its own eval set scores --
            which is also what keeps that set from needing a live tool
            surface.
        surface: The tool surface `get_skill` and `get_skill_file` call
            data-mcp through, and `propose_plan` reads a pinned dataset's
            first year through. Omitted, the planner has neither skill tool,
            its instruction does not name them, and `propose_plan` reads
            nothing.
        question_fixed: Whether the question this planner drafts for is fixed
            for the entry it is called in. `harness.planner_seam` sets
            this when it builds the planner, since the orchestrator states
            the plan's question in its brief before the planner ever runs,
            and a restart would silently swap the question out from under
            that brief (see `_resolve_intent`). False, the default, is the standalone
            planner's and `harness_pair_v1`'s own behaviour: a restart begins
            a fresh plan.
        prefix_cache: The process's shared cache of static prompt prefixes.
            Given, and `model_name` naming a Gemini model, every call reads
            the instruction and tool declarations from it. Either way a
            Gemini model retries a transient error (408, 429, 5xx) up to
            three attempts, as the orchestrator's does
            (`orchestrator.agent._retrying`).

    Returns:
        The configured `LlmAgent`.
    """
    return LlmAgent(
        name=AGENT_NAME,
        model=_retrying(model_name, prefix_cache=prefix_cache),
        instruction=planner_instruction(
            has_translator=translate is not None,
            has_skills=surface is not None,
            question_fixed=question_fixed,
        ),
        tools=[
            *(build_skill_tools(surface) if surface is not None else []),
            *build_planner_tools(
                translate=translate,
                question_fixed=question_fixed,
                surface=surface,
            ),
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
