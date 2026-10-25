"""The planner agent: plan-lifecycle tool functions plus `create_planner_agent`.

A `propose_plan` whose translation compiles declares the plan ready for the
user's answer in code (`declare_ready`), with no model call. It runs
`_execution_blocked`'s checks (a rejected plan, a pending referral, an
uncompiled workflow, a status other than `PENDING_APPROVAL`) and, past those,
answers `_approval_summary(plan)` (the version and revision that tie the
answer to one content hash, and the compiler's warnings, the one thing on it a
caller cannot read off the plan record) under `summary`, and ends the run.
Declaring a plan ready approves nothing. `harness.studio_translate` builds the
plan card the user reads from the plan record itself, through `harness.plan_display`, so nothing here restates a
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
`PENDING_APPROVAL` is a status only translation writes, and `declare_ready`
requires a plan that has been through that step. The checks a proposal passes
before it is recorded are `submission.service`'s, which `propose_plan` runs
over its call's session (`ToolContextSession`).

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
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from compiler.surface import ToolSurface
from data_mcp.definition_writes import windowed_identity
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from harness.graph.orchestrate import counts_chosen
from harness.graph.state import ORCHESTRATOR_SPECS_KEY, resume_answers
from orchestrator.agent import _retrying
from pydantic import JsonValue
from submission.service import (
    COMBINATION_OPTIONS,
    COMBINE_OPTION,
    Admission,
    Combination,
    Proposal,
    identity_on_record,
    submit,
)
from submission.service import PROPOSE_PLAN as PROPOSE_PLAN_PROPOSER
from translator.discovery import (
    DefinitionNotResolvedError,
    PartNotResolvedError,
    resolve_definition,
    resolved_spec,
)
from translator.referral import USER_DECIDES, clear_referral, load_referral
from translator.tool import Translate, authored_answer

from planner.declaration import PlanTool
from planner.plan import (
    MAX_PLAN_DECISIONS,
    MAX_PLAN_REQUIREMENTS,
    AnswerMode,
    AskAnswer,
    DecisionDraft,
    Decline,
    EvidenceRequirement,
    FullHistory,
    PinSource,
    Plan,
    PlanStatus,
    SessionState,
    UnsupportedOperation,
    full_histories,
    full_history,
    load_plan,
    pin_sources,
    plan_revision,
    record_reduction_refused_asks,
    record_unsupported,
    save_full_history,
    save_plan,
    save_plan_rounds,
    save_windowed_pins,
    split_of,
)
from planner.records import (
    OpenPeriodFault,
)
from planner.settings import AGENT_THINKING_CONFIG, drop_thought_parts
from planner.skills import build_skill_tools
from planner.tool_results import Action, Outcome, ProblemCode, ok, problem

if TYPE_CHECKING:
    from harness.prefix_cache import PrefixCache

AGENT_NAME = "dfo_planner"

# The tool the planner proposes a plan through.
PROPOSE_PLAN = "propose_plan"


_PLANNING = (
    "You are the Planner Agent for the Development Finance Observatory. "
    "Given a development finance question, draft a methodological plan. "
    "State each real decision (measure, scope, time window when the "
    "search results do not settle it, and so on), the alternatives you "
    "considered, and a one-line reason for your choice. Record a decision "
    "only where you weighed more than one alternative. A value the "
    "question already states outright, such as the entity it names or the "
    "measure it names, is a fact for a requirement's own fields to carry "
    "and needs no `Decision` entry at all. The search results in your "
    "first message, numbered search readings, are what the conversation's "
    "searches established, as far as that message reports them. A reading "
    "from an earlier turn counts as much as one from this turn. With no "
    "search results, draft from the question alone.\n\n"
    "A `time_window_defaulted` warning means the question gave no time "
    "text, and the resolver has already settled the answer's coverage. "
    "That is not a decision. Do not weigh candidate ranges, do not call "
    "`adk_request_input` over it, and do not invent a specific range from "
    "memory.\n\n"
    "The period follows observed coverage. When a result carries a "
    "`window`, state the dataset requirement's `period` from it verbatim, "
    "with `start` and `end` exactly as given, and cite no decision that "
    "fixes it. An end year earlier than the latest year with data, or a "
    "start the question did not ask for, is a `Decision` whose `fixes` "
    "names `period` and whose rationale says why, such as a latest year "
    "whose counts fall far below the years before it. Do not call a year "
    "the latest available when a result shows data for a later one. When "
    "no result carries a window, the period is a decision like any other.\n\n"
    'A period the question words against the newest data ("the last five '
    'years", "the latest year", "since 2015") also follows it. State the '
    "years from the `window` as above, set `period_end_open` on the "
    "dataset requirement, and state `rolling` on the growth, rank or "
    "period threshold that reads those years, with the newest at offset 0 "
    'and a year five before it at -5. "Since 2015" keeps 2015 a fixed '
    'start and rolls only the end. Years the question names ("from 2015 '
    'to 2020") stay fixed, with no `rolling` and no `period_end_open`.\n\n'
    "A source follows the search readings the same way. When a result's "
    "own `spec_id` (a definite or chosen reading, chosen being one the "
    "conversation fetched) answers a dataset requirement, pin it as that "
    "requirement's `source_spec_id`, verbatim, and cite no decision for "
    "the pin. The plan retrieves that dataset as it was shown, with no "
    "new search. Pin a candidate's `spec_id` with a `Decision` naming the "
    "candidates you weighed. Otherwise the source is a decision like any "
    "other. State a dataset requirement's `unit` only when the question "
    "names a currency or price basis, and do not copy a source's own "
    "unit. Ask the user directly only when a choice is genuinely "
    "ambiguous, and settle other defaults the way a domain expert would.\n\n"
    "State what the answer rests on as evidence requirements: one dataset "
    "requirement per measure and source the answer needs, naming every "
    "entity the answer needs for that measure, and one derived "
    "requirement per calculation the answer needs.\n\n"
    "When a flow source spans a party the question leaves unnamed (aid "
    "from named donors to every recipient) and the measure is an additive "
    "amount, add an `aggregate` with `reduction='sum'` grouped by "
    "`entity` and period before any `growth`, `rank` or `comparison`. A "
    "group by `entity` keeps the parties the requirement names apart and "
    "sums its `across` party, so each named entity has one value per "
    "year. A result left per member of an `across` party is refused "
    "unless a step names that party as its axis (a `rank`, a keyed "
    "`share_of` or `quotient_of`, a `filter`) or the dataset cites a "
    "`Decision` whose `fixes` names `across`. Grouped by period alone, it "
    "totals every party into one series per year, which a `share_of` or "
    "`quotient_of` reads as a whole side. Over a requirement that names "
    "no entity, a group by `entity` keeps nothing and is refused. Group "
    "by period, or `rank` to set each member beside the others. Do not "
    "sum a rate, index or per-person figure.\n\n"
    'A party the question names no member of, such as the donors of "top '
    'donors", goes in `across` (see `across`). A region or group named '
    'alone, such as "Africa", is one entity with its own figure. "African '
    'countries", or a figure for each member, names the set once with '
    "`members` set, however many members it has.\n\n"
    "A result's open axes (`open_entity_axes`) are the columns left open: "
    "`origin` is the giving party (an entity `role` of donor, creditor or "
    "exporter), `destination` the receiving party (recipient, debtor or "
    "importer), `observationAbout` the place.\n\n"
    "A flow side's retrieval holds no group totals (DAC countries, the G7, "
    'regional totals), so a side left open, such as the donors of "top '
    'donors", sums and ranks every party, organisations included, with no '
    "level filter. When the question names a side's countries (\"recipient "
    'countries", "donor countries"), use `across` with that role, such as '
    "`across=['recipient']`, and a direct filter with `{kind:'place_level', "
    "role:'recipient', level:'Country', strict:true}` in its `predicates`, "
    "naming that role. Route calculations through it.\n\n"
    "A requirement whose `across` is the place, such as `'country'`, "
    "names no place, so its retrieval holds every level (see `level` on "
    "`place_level`). Before a `rank`, an `aggregate` across places, a "
    "`share_of` or a `quotient_of` reads it, keep one level with a "
    "`filter` reading that requirement directly, whose predicate is "
    '`place_level`. Take the level the question names ("top 10 '
    'countries"). When it names none and reads as a comparison of '
    'countries ("top 10 by GDP"), choose `Country` in a `Decision` '
    "resolved by `planner_default` whose `fixes` names `predicates`. When "
    'its word covers several levels ("by region" is continents or '
    "subdivisions), ask the user. A question naming its places names them "
    "in `entities` and needs no level.\n\n"
    "Several entities of one measure are one dataset requirement, which "
    "under `answer_mode='series'` is itself the answer. None of "
    "`comparison`, `share_of`, `quotient_of` or `join` keeps the separate "
    "series that a chart of one measure over several entities shows, so "
    "use each only in its case. Use `comparison` only when the question "
    "asks for a difference, `share_of` only for a part of its whole, both "
    "in one unit, `quotient_of` only for one measure per unit of another, "
    'such as a per-head or per-capita figure or "divide one by the '
    'other", and `join` only across different measures.\n\n'
    "An `axis` or a `denominator_axis` names a role its side's entities "
    "state, or `'country'` for places stating none. A family of StatVars "
    "is read summed or separately (see `stat_vars`).\n\n"
    "When the question asks which members of a classification hold the "
    'most, such as "which sectors received the most aid", state '
    "`breakdown` on the dataset requirement, at `level` 0 unless the "
    "question names a finer one. Then add an `aggregate` with "
    "`reduction='sum'` and `group_by` period and breakdown, which totals "
    "the parties the question leaves open within each sector, and a "
    "`rank` with `axis='sector'` and a `limit`, of one year unless the "
    "ask is marked (across years). To set that ranking beside another "
    "year, widen the period to cover both. State "
    "`per_period` on the rank for each year's ranking side by side, or "
    "read a `growth` over the aggregate between the two years for each "
    "sector's change. A derived requirement's figures are what a question "
    "about a value, a growth, a share or a rank asks for. Which members, "
    "or how many, meet a condition is a `filter` with a `value_threshold` "
    "over one figure per member, and a `period_threshold` for the year; "
    "the members it keeps are the answer and their count.\n\n"
    "When the answer is one total of two different sources, such as "
    "bilateral aid plus imputed multilateral aid, state a `sum_of`, "
    "naming each side's `entity` when its requirement names several on "
    "one side of the flow, such as two donors. Tables whose provenance "
    "states account `provider`, such as the OECD and ONE aid tables, add "
    "up as the skill the sources' search results name says, and a sum of "
    "them cites a `Decision` recording that, with the skill in its "
    "`rationale`. Whether any other two datasets add up without counting "
    "a flow twice is the user's to decide, so cite no decision of your "
    "own for it. `propose_plan` refuses such a `sum_of` with the `options` "
    "to ask with, and the skill says what to tell the user about the "
    "overlap. If the user picks adding them, cite a `Decision` resolved "
    "by `user` whose `chosen` is that option id. Otherwise show the two "
    "series separately. ODA loans count in both aid and debt flows, so "
    'asking about a total of the two ("net flows" of aid and debt) names '
    "that overlap.\n\n"
    "A sum states the same `unit` on both inputs, or on neither. To total "
    "several entities of one series, `aggregate` it.\n\n"
    "One figure or one ranking over several years taken together answers "
    "only an ask marked (across years), whatever years the question "
    "names, or the user's own words asking for it. When your message "
    'says "The user asks for this change to the plan" and the change asks '
    "for one figure or one ranking over the years, or no asks are listed "
    "and the question asks for one, name the step that combines them in "
    "`across_years`. For either, an "
    "`aggregate` with `reduction='sum'` grouped by `entity` alone sums "
    "the years, as a rank's `reduction` does for a ranking of each "
    "entity's total, an `aggregate` with an empty `group_by` sums the "
    "entities and the years into one figure, and a total of two sources states the question's own "
    "period on both dataset requirements, adds the two yearly series with "
    "a `sum_of` first, and then sums the years of that one series with an "
    "`aggregate` grouped by `entity`. The `sum_of` adds the years both "
    "sources report and names any year one lacks. Otherwise every "
    "`aggregate` keeps `period` in its `group_by`, and a rank over several "
    "years with `per_period` unset reads one year, the newest unless the "
    "question names another: state `window_start` and `window_end` both "
    "that year, with `rolling` at offset 0 on both when the question "
    "words it against the newest data.\n\n"
    "When a part of the question earns a decline (see `declined`), such "
    "as a calculation over a comparison or a join, state it in `declined` "
    "and plan what can still be answered, or propose `declined` alone "
    "when nothing can. Decide the methodology and leave the mechanics "
    "alone. Which data and which operations carry your decisions out is "
    "worked out separately, after you propose, so name no tool.\n\n"
    "A rank's `among='population'` is a claim over entities nobody named, "
    "so cite a `Decision` whose `fixes` names `among`, as a `share_of` "
    "stating `denominator_scope='pooled'` cites one whose `fixes` names "
    "`denominator_scope`.\n\n"
    "State a rank's `limit` when the question asks for a top or bottom N, "
    'or for the members that hold the most, as "top donors" or "which '
    'sectors received the most" does, with a `Decision` fixing the N when '
    "the question names none. For an ask marked (across years), a top N "
    "by each entity's total is one rank stating the `reduction` and the "
    "`limit` with `per_period` unset. No rank reads another rank's "
    "output, so put the `limit` on the one rank that reads the measure.\n\n"
    "To rank entities by a growth, a per-entity `share_of` or a "
    "`quotient_of`, compute that first and rank its result. A per-period "
    "rank with a `limit` keeps an entity only in the periods it made the "
    "cut, so a growth or share over its output would miss years the "
    "source holds.\n\n"
    "State `per_period` on a rank when the question asks how a position "
    "changed over time rather than who is first. Without a `limit`, its "
    "output is each entity's rank position in each period, an ordinal "
    "series and not a value. To grow that position, put a `filter` naming "
    "the one entity the question is about between the rank and the "
    "`growth`, and state `change='absolute'` on the growth, since a rank "
    "moves by places and not by a percentage. The rank's output carries "
    "the whole population, and a growth over everyone's position at once "
    "has no defined answer.\n\n"
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
    "never let the wording of `question` stand in for the choice.\n\n"
)

_VERSIONING_FIXED = (
    "Call `propose_plan` once with the complete first draft, and again, "
    "patching it through `amends_version`, whenever the user asks for a "
    "change. This turn's question is fixed, so when the user's reply asks "
    "a different question, call `propose_plan` with `starts_new_plan=true` "
    "and stop there.\n\n"
)

# How to read skills, when the skill tools are built. Held apart because the
# planner is also built without them, and an instruction naming a tool the
# agent lacks sends the model at one ADK raises on.
_SKILLS = (
    'The search readings close with a "Skills named" line listing, by id, '
    "the skills that apply to them. When a skill bears on the question, "
    "call `get_skill` with its id before you draft, and let what it says "
    "shape your decisions. Read a file it links with `get_skill_file`. A "
    "question about DFO itself (what it covers, where its data comes "
    "from, how it works) starts with `get_skill`, though no search result "
    "names that skill. When a skill tool refuses, "
    "`capability_unavailable` included, draft the plan without that "
    "skill.\n\n"
)

# What to do once a plan is recorded, when a translator is attached to carry it
# out and `propose_plan` answers with its verdict under `translation`. Held
# apart because the planner is also built standalone -- its own eval set scores
# it that way -- and its `propose_plan` answers with no translation to read.
_TRANSLATING = (
    "`source_name_ambiguous` means Search could not choose which "
    "publisher or table a source wording names. The user is asked which "
    "they meant, and their answer names one of the offered refs. State it "
    "in `source_ref` and keep the source wording and `source_exclude` on "
    "the requirement. When the wording is the requirement's "
    "`source_code`, held by more than one publisher, keep the code and "
    "state the picked publisher's label in `source` with its ref in "
    "`source_ref`.\n\n"
    "`propose_plan` has every plan it records translated at once and answers "
    "with the verdict in `translation`. Once a referral is settled, by "
    "your own decision or the user's answer, propose with the "
    "`amends_version` the last `propose_plan` call answered with, so the "
    "patch lands on that plan and is translated in turn. When "
    "`translation` answers that the plan compiled, it goes to the user "
    "for their answer and your turn ends there.\n\n"
    "When translation answers `restated`, the recorded plan counts each "
    "listed requirement's rolling years from `anchor`, the newest year at "
    "least half the series hold: `years_to` where you wrote `years_from`. Amend "
    "from the plan as recorded, and state those years.\n\n"
    "When translation answers `named_in_place_of_across`, the pinned "
    "dataset covers only that group, so the recorded requirement already "
    "names it in `entities` and has dropped that party from `across`. "
    "Amend from the plan as recorded and do not restore it. Otherwise "
    "`translation` is the referral.\n\n"
    "`choice_required` (`source_ambiguous`, or `pinned_source_not_found` "
    "when the pinned source was not found) means a requirement's source "
    "is not settled. The referral's `candidates`, each with its "
    "`spec_id`, `sentence`, `coverage_window` and the measures, entities "
    "and unit it resolves, are the only sources that requirement may pin, "
    "and `propose_plan` refuses any other. A `pinned_source_not_found` "
    "referral also lists in `stale_pins` every other requirement whose "
    "pin was not found, each with its own `candidates`. Repair them all "
    "in one patch.\n\n"
    "Where a requirement is offered one candidate, the fresh search "
    "settled it. Pin that `spec_id` as its `source_spec_id` and cite no "
    "decision for the pin. A source under the referral's `withheld` "
    "dropped the pinned source's scope, such as a named group, so do not "
    "pin it. When no candidate keeps that scope, make the referral's "
    "`repair` to the requirement, which names that scope again, and "
    "propose again. With no `repair`, ask the user.\n\n"
    "Where several are offered, settle it yourself when the question, the "
    "brief or the candidates' coverage make one the clear answer, a "
    "source the question or the brief names included. When the choice is "
    "the user's to make, call `adk_request_input` with the candidates. "
    "Either way, record a `Decision` whose `options` name every candidate "
    "and whose rationale says why, resolved by `planner_default` when you "
    "chose and by `user` when the user's answer did. Pin that `spec_id` "
    "as the requirement's `source_spec_id`, and propose again.\n\n"
    "A `choice_required` referral with no candidates and none withheld "
    "means the pinned source is gone. Unpin it and change what the "
    "requirement asks for.\n\n"
    "A referral's `kind` says what to do next.\n\n"
    "A `no_data` referral carries the nearest real sources search found "
    "instead and the reason code for why, each described without a "
    "spec_id. Read them for context and do not pin one, since the "
    "requirement itself is what has to change.\n\n"
    "An `unsupported_ask` or `requirement_unmeetable` referral means the "
    "requirement cannot be met as stated. Tell the user what cannot be "
    "done and ask, through `adk_request_input`, before you propose again, "
    "since what to change is theirs to choose.\n\n"
    "A `requirement_unmeetable` referral with `candidates` means the "
    "source the requirement states holds none of the data, and the "
    "candidates are the sources found instead. The user is asked to pick "
    "one, so pin the one they pick as its `source_spec_id` under a "
    "`Decision` resolved by `user`, which replaces the stated source.\n\n"
    "An `ask_not_understood` referral means search dropped part of the "
    "ask and answered a narrower one, so reword the part the referral "
    "names.\n\n"
    "A `requirement_needs_repair` referral names the `constraint` the "
    "requirement breaks and the `repair` you may make, so make it and "
    "propose again.\n\n"
    "A `no_operation_covers` referral means no operation carries that "
    "requirement out.\n\n"
    "`unsupported_ask` and `no_operation_covers` are `unsupported` "
    "answers, which earn a decline (see `declined`). Every plan you "
    "propose after one states that part in `declined`, a plan reshaped "
    "after the user's answer included. A `requirement_unmeetable` whose "
    "`reason` says it earns a decline lets the plan make that decline "
    "once the user's answer is to see the two sides apart, and does not "
    "require it.\n\n"
    "A `discovery_budget_exhausted` referral means the plan asks for too "
    "much at once.\n\n"
    "A `discovery_unavailable` or `surface_unavailable` referral means "
    "the data service could not answer just now, so nothing about the "
    "requirement needs to change. Propose the same plan again, with "
    "`amends_version` and no other change, to have that version "
    "translated again, which records no new version and spends no plan "
    "round.\n\n"
    "An `internal_failure` referral is a fault in the system that no "
    "change to the plan repairs, so tell the user, with its `reference`, "
    "and propose nothing further.\n\n"
    "The options an ask offers come only from the referral's `candidates` "
    "or a refusal's `options`. With neither, the ask says what cannot be "
    "done and asks what to change. A plan that cannot be carried out is "
    "not one to approve.\n\n"
)


_WITHOUT_A_TRANSLATOR = (
    "Stop once `propose_plan` has recorded the plan: having it translated, "
    "compiled and carried out belongs to a later step that is not "
    "attached here, so there is no translator to call and nothing to "
    "approve yet.\n\n"
)

_CLOSING = (
    "Every tool answers with an `outcome`, a `code` and sometimes an "
    "`action`. Follow the action and do not guess at the message. Nothing "
    "runs until the user approves a compiled plan. Call `reject_plan` "
    "when the user declines the plan and asks for nothing in its place."
)


def planner_instruction(
    *,
    has_translator: bool,
    has_skills: bool = False,
    question_fixed: bool = False,
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


def _approval_summary(plan: Plan) -> dict[str, object]:
    """Return what a next call needs to know about the compiled `plan`.

    The version and revision, which tie a later answer to one content hash,
    and every warning the compiler let through (`Workflow.warnings`). The
    plan's own decisions, requirements, sources and steps are deliberately
    absent. `harness.studio_translate` builds the card the user approves from
    the plan record, through `harness.plan_display`, and translation and
    `declare_ready` both read the record out of session state, so a copy here is
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


def _execution_blocked(plan: Plan, tool_context: ToolContext) -> dict[str, object] | None:
    """Return the problem that stops `declare_ready` on `plan`, or `None` to let it proceed.

    Checked in order: a rejected plan; a pending referral, which the planner settles by
    proposing again, after asking the user where the referral is of a kind in `USER_DECIDES`
    (the refusal's `action` names `adk_request_input` until they answer, `propose_plan`
    otherwise); a plan not yet translated and compiled; and, last, an allowlist on `plan.status`
    itself. Only `PENDING_APPROVAL` may be declared ready, so a plan already `APPROVED`,
    `EXECUTING`, `COMPLETED` or `FAILED` is refused too, even though none of those carries a
    message of its own above. A plan seeded back from a failed run (see `harness.graph`) still
    carries a workflow and the version it failed at, so it would otherwise pass every check
    above and reach the user as though it were still approvable.
    """
    if plan.status is PlanStatus.REJECTED:
        return problem(
            ProblemCode.PLAN_REJECTED,
            "The current plan was rejected. Propose a new one before executing.",
            outcome=Outcome.REFUSED,
            action=PROPOSE_PLAN_PROPOSER.action(),
        )
    referral = load_referral(tool_context.state)
    if referral is not None:
        asks_user = referral.kind in USER_DECIDES and not _answered_since_referral(tool_context)
        return problem(
            ProblemCode.REFERRAL_PENDING,
            "The translator referred this plan back; settle the referral with the user "
            "and propose again before executing.",
            outcome=Outcome.REFUSED,
            detail={"referral": authored_answer(referral.model_dump(mode="json"), split_of(plan))},
            action=Action(call=REQUEST_INPUT_FUNCTION_CALL_NAME)
            if asks_user
            else PROPOSE_PLAN_PROPOSER.action(),
        )
    if plan.status is PlanStatus.DRAFT or plan.workflow is None:
        # A draft states the methodology; what runs arrives with
        # translation, so approving one would approve nothing to run.
        return problem(
            ProblemCode.WORKFLOW_MISSING,
            "This plan has not compiled into a workflow, so nothing can execute yet. "
            "Propose it again, with its `version` as `amends_version`, to have it "
            "translated.",
            outcome=Outcome.REFUSED,
            action=PROPOSE_PLAN_PROPOSER.action(),
        )
    if plan.status is not PlanStatus.PENDING_APPROVAL:
        return problem(
            ProblemCode.TRANSITION_REFUSED,
            f"This plan is {plan.status.value}, not awaiting approval, so it cannot be "
            "declared ready. Propose a new plan before executing.",
            outcome=Outcome.REFUSED,
            detail={"status": plan.status.value},
            action=PROPOSE_PLAN_PROPOSER.action(),
        )
    return None


def declare_ready(
    answer: dict[str, object],
    tool_context: ToolContext,
) -> dict[str, object]:
    """Declare the plan on record ready for the user's answer, when `answer`'s translation compiled.

    `answer` is an accepted `propose_plan` answer carrying its `translation`. A verdict other
    than `ok` leaves it unchanged. Past `_execution_blocked`, it gains `summary`
    (`_approval_summary`), and the run ends there (`skip_summarization`). The harness asks the
    user, so no model call follows to restate the plan the user already sees. A plan the gate
    blocks gains the gate's problem under `submission` instead, its `action` naming
    `propose_plan`, and the run goes on. Approves nothing. Whatever runs this agent gates on the
    user's answer, keyed on the summary's revision.
    """
    translation = answer.get("translation")
    if not isinstance(translation, dict) or translation.get("outcome") != Outcome.OK.value:
        return answer
    plan = load_plan(tool_context.state)
    assert plan is not None, "a translation answers ok only for the plan on record"
    blocked = _execution_blocked(plan, tool_context)
    if blocked is not None:
        return {**answer, "submission": blocked}
    tool_context.actions.skip_summarization = True
    return {**answer, "summary": _approval_summary(plan)}


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


def _candidate_ids(candidates: Any) -> frozenset[str]:
    """The spec_ids of a list of `{"spec_id", "label"}` candidates."""
    return frozenset(
        str(candidate["spec_id"])
        for candidate in candidates or []
        if isinstance(candidate, dict) and candidate.get("spec_id")
    )


def _offered_candidates(args: dict[str, Any]) -> dict[str, frozenset[str]]:
    """The candidates an `adk_request_input` call's own function-call event
    offered, by the id of the question that offered them, from any shape that
    name occurs under:

    - the planner's own `ask_user` tool call: a plain `FunctionTool` call,
      so `args` is exactly the kwargs the model supplied, flat --
      `args["candidates"]`, the one question `q1`.
    - the graph's `ask` node, under `harness.graph`'s `CLARIFY_INTERRUPT`:
      built through ADK's own `google.adk.workflow.utils.
      _workflow_hitl_utils.create_request_input_event`, which dumps the
      whole `RequestInput` into `args`, so the caller's own `payload` nests
      under `args["payload"]`. Its `questions` each carry their own `id` and
      `candidates`, whose ids are scoped to that question. A `payload` with
      `candidates` and no `questions` holds the one question `q1`.
    """
    candidates = args.get("candidates")
    payload = args.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    if not candidates and payload.get("questions"):
        return {
            str(question["id"]): _candidate_ids(question.get("candidates"))
            for question in payload["questions"]
            if isinstance(question, dict) and question.get("id")
        }
    return {"q1": _candidate_ids(candidates or payload.get("candidates"))}


def _answers(result: object) -> dict[str, dict[str, Any]]:
    """An `adk_request_input` response's `result` as one answer per question id
    (`harness.graph.state.resume_answers`). A missing result and a dismissal of the
    whole pause hold none."""
    return {} if result is None else resume_answers(result) or {}


def _picks(result: object, offered: dict[str, frozenset[str]]) -> tuple[str, ...]:
    """The spec_ids `result` picks, in question order, each as an option its own
    question offered.

    A skipped or free-text answer picks nothing, and neither does an id another
    question offered."""
    return tuple(
        str(answer["spec_id"])
        for question_id, answer in _answers(result).items()
        if isinstance(answer, dict)
        and answer.get("kind") == "option"
        and str(answer.get("spec_id")) in offered.get(question_id, frozenset())
    )


def _answered_calls(tool_context: ToolContext) -> Iterator[tuple[dict[str, Any], object]]:
    """Each interrupt answer with the nearest preceding call sharing its id."""
    calls: dict[str, dict[str, Any]] = {}
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
                calls[call.id] = call.args or {}
                continue
            response = part.function_response
            if response is None or response.name != REQUEST_INPUT_FUNCTION_CALL_NAME:
                continue
            if response.id in calls:
                yield calls[response.id], (response.response or {}).get("result")


def _elected_spec_ids(tool_context: ToolContext) -> frozenset[str]:
    """Every spec_id the user actually chose answering an `adk_request_input`
    call, read off this session's own event history -- the record of what
    the user was asked and what they answered, which
    makes a measure decision the user's (`_user_settled_measures`).

    Membership in a call's offered `candidates` is not by itself an
    election: an `adk_request_input` call that offers `{A, B}` records only
    a question, not a choice, and a plan pinning `B` after the user picked
    `A` -- or after no answer has arrived at all -- is not a user election
    either. What counts as one is a call's own function-call event, paired
    with the matching `FunctionResponse` event, if any, that answers it:
    `create_request_input_event` builds the call with `id=interrupt_id`, and
    `create_request_input_response` builds the answer with
    `FunctionResponse.id` set to that same `interrupt_id`; `harness.studio`'s
    `/interrupt` route is what actually sends it, its `result` read as one
    answer per question id (`harness.graph.state.resume_answers`), a chosen
    candidate as `{"kind": "option", "spec_id": ...}`. A call with no
    paired response yet -- the run is still paused there, or the answer
    hasn't arrived -- contributes nothing: its candidates were offered, not
    elected.

    One response's picks elect the counts they choose, read against the
    turn's search blocks (`harness.graph.orchestrate.counts_chosen`): a pick
    of other prices alone chooses no count, and a count picked with other
    prices elects the listed reading measuring both, the one the `ask`
    node tells the agent to fetch.

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
    specs = tool_context.state.get(ORCHESTRATOR_SPECS_KEY) or {}
    elected: set[str] = set()
    for args, result in _answered_calls(tool_context):
        elected |= counts_chosen(_picks(result, _offered_candidates(args)), specs)
    return frozenset(elected)


def _said(answer: object) -> str:
    """What one answer says: the candidate picked or the user's own words, else nothing."""
    if not isinstance(answer, dict):
        return ""
    key = {"option": "spec_id", "text": "text"}.get(answer.get("kind"))
    value = answer.get(key) if key else None
    return value.strip() if isinstance(value, str) else ""


def _is_answer(body: dict[str, Any]) -> bool:
    """Whether an `adk_request_input` response body carries something the user said.

    A user's answer is `{"result": ...}` (`harness.studio`'s `/interrupt` route wraps
    it, as ADK wraps any non-dict value), read as `harness.graph.state.resume_answers`
    reads it: a pick of a candidate, or free text for a reply outside the offered
    options. A body with no `result` is a call the tool or ADK rejected. A dismissal of
    the pause and a question left to the planner are the user declining to choose, and
    blank text says nothing.
    """
    return any(_said(answer) for answer in _answers(body.get("result")).values())


def _is_amendment(body: dict[str, Any]) -> bool:
    """Whether an `adk_request_input` response body is the user amending the approval card.

    `harness.studio`'s `/interrupt` route answers the approval card
    (`harness.graph.APPROVE_INTERRUPT`) with `{"result": {"amendment": ...}}`
    when the user types a change there.
    """
    result = body.get("result")
    amendment = result.get("amendment") if isinstance(result, dict) else None
    return isinstance(amendment, str) and bool(amendment.strip())


def _user_message(event: Any) -> bool:
    """Whether `event` is a message the user typed into the session.

    A user-authored event with text and no `branch`. The planner seam sends
    its own messages (the brief, an amendment's or a refusal's ask) as
    user-role content under `harness.planner_seam._PLANNER_BRANCH`, so they
    are not the user's.
    """
    return (
        getattr(event, "author", None) == "user"
        and not getattr(event, "branch", None)
        and any((getattr(part, "text", None) or "").strip() for part in event.content.parts)
    )


def _user_spoke_since_recorded(tool_context: ToolContext, version: int) -> bool:
    """Whether the user has said something since plan `version` was recorded.

    Read off this session's event history. The first `propose_plan` response carrying `version`
    marks where the plan was recorded. A later response repeating the version records nothing
    (`_transient_retry` answers a bare retry so) and moves no mark. After the mark, the user
    speaks with an `adk_request_input` response they answered (`_is_answer`: a pick, free text),
    an amendment of the approval card (`_is_amendment`), or a message of their own
    (`_user_message`), as the ADK eval runner sends a case's later messages into one session. A
    refusal, a skip, a referral and the planner seam's own messages are not.

    `harness.studio_state.ConversationState.session_id` opens a new ADK
    session for every Studio turn, and every turn starts with the user's
    message. A plan this session holds no recording response for was
    recorded before that message, so this answers `True`. The planner seam
    seeds such a plan from the graph's record
    (`harness.planner_seam.PlannerSeam`).
    """
    last: object = None
    recorded = False
    spoke = False
    for event in tool_context.session.events:
        if not (event.content and event.content.parts):
            continue
        if recorded and _user_message(event):
            spoke = True
            continue
        for part in event.content.parts:
            response = part.function_response
            if response is None:
                continue
            body = response.response or {}
            if response.name == PROPOSE_PLAN and "version" in body:
                if body["version"] != last:
                    last = body["version"]
                    recorded, spoke = last == version, False
            elif response.name == REQUEST_INPUT_FUNCTION_CALL_NAME and recorded:
                spoke = spoke or _is_answer(body) or _is_amendment(body)
    return spoke or not recorded


def _answered_since_referral(tool_context: ToolContext) -> bool:
    """Whether the user has answered an ask the planner raised after seeing the
    translator's referral (`_answers_since_referral`)."""
    return _answers_since_referral(tool_context)[0]


def _answers_since_referral(tool_context: ToolContext) -> tuple[bool, frozenset[str]]:
    """Whether the user has answered an ask the planner raised after seeing the
    translator's referral, and the spec_ids they picked answering those asks,
    read off this session's event history as `_elected_spec_ids` reads it.

    The referral reaches the planner in the `translation` of the response that recorded the
    plan, `propose_plan`'s, carrying its `kind` and `requirement_id`. This walks the session
    once, in event order. An `adk_request_input` call event after that response is an ask the
    model raised knowing the referral, and its `id` is remembered. A response counts as the
    user's answer when it pairs by `id` with one of those asks and `_is_answer` holds. An ask
    issued before the referral, a call the tool or ADK rejected, and a skip do not count. A skip
    is the user declining to choose, which the planner may not substitute for, so the planner
    ends the turn by telling them what cannot be done.

    A response that recorded a version clears the referral in session
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
    asks: dict[str, dict[str, frozenset[str]]] = {}
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
                picked.update(_picks(body.get("result"), asks[response.id]))
            elif "translation" in body:
                translation = body["translation"]
                referral = (
                    (translation.get("kind"), translation.get("requirement_id"))
                    if isinstance(translation, dict) and "kind" in translation
                    else None
                )
                answered = False
                asks.clear()
                picked.clear()
    return answered, frozenset(picked)


def _refused_combination(body: dict[str, Any]) -> Combination | None:
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
) -> tuple[dict[Combination, str | None], bool]:
    """The option id the user last picked for each combination they were asked about, and a flag.

    A combination ask is an `adk_request_input` call whose `options` are `COMBINATION_OPTIONS`.
    It asks about the combination the latest `combination_unasked` refusal before it names, read
    off that refusal's own response event, so the model cannot point an ask at another sum. A
    refusal binds the first ask after it and no other, so an ask with no refusal of its own
    before it asks about nothing. Each response pairs with the nearest preceding call sharing
    its `id`, as `_elected_spec_ids` pairs them. The combination ask is one question, `q1`
    (`harness.graph.state.resume_answers`). Each answer about a combination replaces the one
    before it, and free text or a skip records `None`, so a user who changes their mind, or
    answers outside the options, ends with no pick for it. The flag says whether any answer to
    any combination ask, about a combination or about nothing, was other than `COMBINE_OPTION`.
    """
    offered = [dict(option) for option in COMBINATION_OPTIONS]
    ids = {option["id"] for option in COMBINATION_OPTIONS}
    refused: Combination | None = None
    asks: dict[str, Combination | None] = {}
    picks: dict[Combination, str | None] = {}
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
            if response.name != REQUEST_INPUT_FUNCTION_CALL_NAME:
                refused = _refused_combination(body) or refused
            elif response.id in asks:
                answer = _answers(body.get("result")).get("q1")
                option = (
                    answer.get("spec_id") if answer and answer.get("kind") == "option" else None
                )
                declined = declined or option != COMBINE_OPTION
                combination = asks[response.id]
                if combination is not None:
                    picks[combination] = option if option in ids else None
    return picks, declined


@dataclass(frozen=True)
class ToolContextSession:
    """The `submission.service.SessionView` over one ADK tool call.

    The state is the call's own. What the user was asked and answered is
    read off the session's event history (`_elected_spec_ids`,
    `_answers_since_referral`, `_combination_picks`,
    `_user_spoke_since_recorded`). Recording an admitted
    plan writes its windowed pins, the plan, its round count, and clears the
    referral the new proposal supersedes.
    """

    tool_context: ToolContext

    @property
    def state(self) -> SessionState:
        """The call's session state."""
        return self.tool_context.state

    def elected_spec_ids(self) -> frozenset[str]:
        """`_elected_spec_ids` over the call's session."""
        return _elected_spec_ids(self.tool_context)

    def answers_since_referral(self) -> tuple[bool, frozenset[str]]:
        """`_answers_since_referral` over the call's session."""
        return _answers_since_referral(self.tool_context)

    def combination_picks(self) -> tuple[dict[Combination, str | None], bool]:
        """`_combination_picks` over the call's session."""
        return _combination_picks(self.tool_context)

    def user_spoke_since(self, version: int) -> bool:
        """`_user_spoke_since_recorded` over the call's session."""
        return _user_spoke_since_recorded(self.tool_context, version)

    def record_unsupported(self, operation: UnsupportedOperation) -> None:
        """Record `operation` as answered `unsupported` in this planner entry."""
        record_unsupported(self.tool_context.state, operation)

    def record_reduction_refused(self, asks: Iterable[int]) -> None:
        """Record `asks` as named by a `PERIOD_REDUCTION_UNSTATED` refusal in this planner entry."""
        record_reduction_refused_asks(self.tool_context.state, asks)

    def record_admission(self, admission: Admission) -> None:
        """Record `admission` in the call's state."""
        state = self.tool_context.state
        if admission.windowed_pins is not None:
            save_windowed_pins(state, admission.windowed_pins)
        save_plan(state, admission.plan)
        save_plan_rounds(state, admission.rounds)
        # A new proposal supersedes whatever the translator referred back.
        clear_referral(state)


# What each `propose_plan` argument means, as the model reads it on the
# argument itself (`planner.declaration.PlanTool`). `question` and
# `starts_new_plan` read differently under `question_fixed`, the way
# `submission.service._resolve_intent` enforces them; `_FIXED_ARGUMENTS`
# replaces those two.
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
        "`ask_ref`. An `unsupported` answer earns a decline, with that "
        "answer's code: one `propose_plan` answered, itself or in its `translation`, "
        "or a run-time refusal the plan is replanned from that names the code. A "
        "`requirement_unmeetable` translation whose `reason` says it earns a "
        "decline earns that code too, which the plan may make or leave out. "
        "`breakdown_not_kept` is earned when an `ask_role_dropped` problem names "
        "the ask and the source its answer pins cannot hold that party open. A "
        "first plan may also decline as `no_operation_covers` an ask no "
        "requirement kind or chain of them states."
    ),
    "answered": (
        "When the brief lists asks, the requirements that answer each one, "
        "keyed by `ask`. Every ask is answered here or declined in `declined` "
        "with its number as `ask_ref`, exactly once, or both when the plan "
        "computes part of it."
    ),
    "across_years": (
        "The `id` of each step that combines its years, an `aggregate` whose `group_by` "
        "leaves out `period` or a `rank` with a `reduction` and `per_period` false, because "
        "the user's own words ask for one figure or one ranking over them: a change to the "
        "plan the user asks for, or, when no asks are listed, the question itself. A step "
        "named once keeps that standing in later patches while its kind and `reduction` "
        "stay, and loses it when an `aggregate`'s `group_by` gains `period` or a `rank` "
        "sets `per_period`. Leave it empty for a step that answers an ask marked "
        "(across years)."
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


# The seconds one first-year read may take (`read_full_history`). It runs
# inside a `propose_plan` call, so inside the planner entry's remaining time,
# which `translator.discovery.resolve_definition` already holds it to; this
# keeps a slow Search from spending that whole remainder on one read.
FIRST_YEAR_READ_DEADLINE_SECONDS = 10.0


def _first_year_identity(pin: str, shown: Mapping[str, PinSource]) -> Mapping[str, Any] | None:
    """The identity a first-year read of `pin` reads; `None` when none is on record."""
    recorded = identity_on_record(pin, shown=shown, end_open=True)
    return recorded.identity if recorded is not None else None


async def read_full_history(
    surface: ToolSurface,
    identity: Mapping[str, Any],
    *,
    deadline_s: float = FIRST_YEAR_READ_DEADLINE_SECONDS,
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
            answer within `deadline_s`, and can succeed later (`retryable`).
    """
    spec_id, unwindowed = windowed_identity(identity, {"kind": "all"})
    try:
        async with asyncio.timeout(deadline_s):
            part = await resolve_definition(surface, spec_id=spec_id, identity=unwindowed)
        return full_history(resolved_spec(part, spec_id=spec_id))
    except TimeoutError as exc:
        raise DefinitionNotResolvedError(
            f"Search did not answer in {deadline_s:.0f} s", retryable=True
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
        "sound and nothing was recorded; propose it again, or, where the window covers "
        "several years, state the start as a fixed year and roll only the end; a window of "
        "one year cannot, so drop `rolling` and `period_end_open` there for fixed years.",
        outcome=Outcome.REFUSED,
        detail=detail,
    )


def build_planner_tools(
    *,
    translate: Translate | None = None,
    question_fixed: bool = False,
    reads_user_words: bool = True,
    surface: ToolSurface | None = None,
) -> list[BaseTool]:
    """Build the planner's tool set.

    `translate` is the translation (`translator.tool.build_translate`) a
    `propose_plan` that records a plan runs on it, answering with the verdict
    under `translation` alongside its own `ok`. Given, `propose_plan` is a
    coroutine function, and a translation that compiles declares the plan
    ready (`declare_ready`). Omitted, `propose_plan` records the plan and
    stops, and nothing declares a plan ready.

    `surface` is the tool surface `propose_plan` reads a pinned dataset's
    first year through, when a start that follows the newest data needs one
    no answer on record states (`read_full_history`). Given, `propose_plan`
    is a coroutine function. Omitted, it reads nothing and refuses such a
    plan `first_year_unknown`.

    `question_fixed` closes over into `propose_plan`, which passes it straight
    to `submission.service.submit` on every call, and selects `_FIXED_ARGUMENTS`'
    descriptions of `question` and `starts_new_plan`, so those two arguments
    describe the mode this tool set actually enforces.

    `reads_user_words` closes over into `propose_plan` the same way, for
    `submission.service.admit`. It says whether the planner reads the
    user's own words in every entry, so `propose_plan`'s `across_years` may
    name a step in any of them.
    """

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
        option id they picked, or free text if they answered outside them.
        The choices offered come from the referral's
        candidates, or from the `options` a refusal's action carries; with
        neither, the question says what cannot be done and asks what to change.

        Args:
            message: The question to show the user, naming what is
                ambiguous and, briefly, what each candidate is, in plain
                words: never a requirement id, spec_id or code.
            candidates: The referral's own candidates, each
                `{"spec_id": ..., "label": ...}` -- `label` is the
                candidate's own `sentence`, not invented text. Omit it for
                a referral that carries none.
            options: The `options` a `combination_unasked` refusal's action
                carries, unchanged. Omit it otherwise.
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
        across_years: list[str] = [],  # noqa: B006
        amends_version: int | None = None,
        starts_new_plan: bool = False,
    ) -> dict[str, object]:
        """Record a plan, or patch the one already recorded.

        Returns `ok` with `status` `"draft"`; the new `version`, which a
        patch passes as `amends_version`; the `previous_version` it amended,
        `None` on a first draft; and the `question` as stored.
        """
        return submit(
            ToolContextSession(tool_context),
            Proposal(
                question=question,
                answer_mode=answer_mode,
                decisions=decisions,
                evidence_requirements=evidence_requirements,
                removed_decisions=removed_decisions,
                removed_evidence_requirements=removed_evidence_requirements,
                declined=declined,
                answered=answered,
                across_years=across_years,
                amends_version=amends_version,
                starts_new_plan=starts_new_plan,
            ),
            question_fixed=question_fixed,
            reads_user_words=reads_user_words,
            proposer=PROPOSE_PLAN_PROPOSER,
        )

    def reject_plan(tool_context: ToolContext) -> dict[str, object]:
        """Record that the user rejected the current plan.

        Nothing in the plan may run afterwards. Call `propose_plan` again if
        the user wants a revision rather than an outright stop.
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
            identity = _first_year_identity(pin, pin_sources(state))
            if identity is None:
                return result
            try:
                history = await read_full_history(surface, identity)
            except DefinitionNotResolvedError as exc:
                return _first_year_unreadable(result, exc)
            save_full_history(state, pin, history)

    @functools.wraps(propose_plan)
    async def propose_plan_translating(**arguments: Any) -> dict[str, object]:
        """`propose_plan`, translating the plan it records.

        Translation runs once, on the version the proposal recorded, after
        any first-year reads (`propose_plan_reading`). The proposal's own
        `outcome` stays `ok` whatever the verdict, since the version is
        recorded either way, and a verdict of `ok` declares the plan ready
        (`declare_ready`). A proposal that records nothing translates
        nothing. A bare re-proposal after a transient verdict
        (`_transient_retry`) translates the recorded version again and
        records nothing.
        """
        assert translate is not None
        tool_context = arguments["tool_context"]
        retried = _transient_retry(arguments, tool_context.state)
        if retried is not None:
            clear_referral(tool_context.state)
            answer = ok(
                status=retried.status.value,
                version=retried.version,
                previous_version=retried.previous_version,
                question=retried.question,
                translation=await translate(tool_context),
            )
            return declare_ready(answer, tool_context)
        result = (
            propose_plan(**arguments)
            if surface is None
            else await propose_plan_reading(**arguments)
        )
        if result.get("outcome") != Outcome.OK.value:
            return result
        return declare_ready(
            {**result, "translation": await translate(tool_context)},
            tool_context,
        )

    ask_user.__name__ = REQUEST_INPUT_FUNCTION_CALL_NAME
    proposing: Callable[..., Any] = propose_plan
    if translate is not None:
        proposing = propose_plan_translating
    elif surface is not None:
        proposing = propose_plan_reading
    arguments = (
        _PROPOSE_PLAN_ARGUMENTS | _FIXED_ARGUMENTS if question_fixed else _PROPOSE_PLAN_ARGUMENTS
    )
    return [
        LongRunningFunctionTool(ask_user),
        PlanTool(proposing, arguments=arguments),
        FunctionTool(reject_plan),
    ]


def create_planner_agent(
    *,
    model_name: str,
    translate: Translate | None = None,
    surface: ToolSurface | None = None,
    question_fixed: bool = False,
    reads_user_words: bool = True,
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
            that brief (see `submission.service._resolve_intent`). False, the default, is the standalone
            planner's and `harness_pair_v1`'s own behaviour: a restart begins
            a fresh plan.
        reads_user_words: Whether the planner reads the user's own words in
            every entry, so `propose_plan`'s `across_years` may name a step
            in any of them. True, the default, is the standalone planner's
            and `harness_pair_v1`'s, which plan the user's question
            themselves. `harness.planner_seam` sets it false, since there the
            orchestrator reads the user's words and marks its asks, and the
            planner reads them only in an entry the user's amendment opens
            (`planner.plan.amendment_entry`).
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
                reads_user_words=reads_user_words,
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
