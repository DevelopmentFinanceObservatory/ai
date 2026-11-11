"""The planner agent (`create_planner_agent`) and the ADK tool that serves a planner tool.

Each planner tool is a `planner.tool_spec.ToolSpec` (`planner.specs`), whose
body runs `planner.tools`' or `planner.skills`' code over a
`planner.call.PlannerCall` and no framework. Here `SpecTool` serves a spec to
ADK, and `AdkCall` is the `PlannerCall` an ADK tool call is. A `propose_plan`
whose translation compiles declares the plan ready for the user's answer in
code (`planner.tools.declare_ready`), and the run ends on it
(`skip_summarization`) with no model call after. `end_run_when_ready` does the
same for an answer the harness builds itself. Declaring a plan ready approves
nothing. Whatever runs this agent withholds execution until that answer
arrives, which for the served path is the harness's own graph, keyed on the
plan's revision, so no ADK tool confirmation has to cross this agent's own
nested `Runner`.

Built with a tool surface, the planner also carries `planner.skills`'
`get_skill`, and its instruction carries `_SKILLS`, the paragraph naming it.
Built without one, it carries neither, so the
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
over its call's session (`planner.call.CallSession` over `AdkCall`).

Every planner tool answers with `planner.tool_results`' typed shape, `ok` or
`problem` with a stable `code` and, where the next step is known, an
`action`. When translation cannot carry a plan out, `propose_plan`'s
`translation` is the `Referral` itself: this agent's instruction reads it
straight off that answer, with nothing here to fetch it back a second time.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
from collections.abc import (
    Iterable,
    Iterator,
    Sequence,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

from compiler.surface import ToolSurface
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event
from google.adk.models import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.planners import BuiltInPlanner
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import Client, types
from harness.turn_events import PLANNER_ASK, REQUEST_INPUT_FUNCTION_CALL_NAME, after_turn_question
from opentelemetry import trace
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Status, StatusCode
from orchestrator.agent import _retrying
from translator.tool import Translate

from planner.call import (
    ResponseCall,
    Step,
)
from planner.plan import SessionState
from planner.settings import AGENT_THINKING_CONFIG, drop_thought_parts
from planner.specs import planner_tool_specs, skill_tool_spec
from planner.tool_results import Outcome
from planner.tool_spec import ToolSpec, function_declaration, run_tool
from planner.tools import EndRun, ready_and_end

if TYPE_CHECKING:
    from harness.prefix_cache import PrefixCache

logger = logging.getLogger(__name__)

AGENT_NAME = "dfo_planner"

_PLANNING = (
    "You are the Planner Agent for the Development Finance Observatory. "
    "Given a development finance question, draft a methodological plan. "
    "State each real decision (a measure no pinned source settles, "
    "scope, time window when the search results do not settle it, and so "
    "on), the alternatives you considered, and a one-line reason for your "
    "choice. Record a decision "
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
    f"`{PLANNER_ASK}` over it, and do not invent a specific range from "
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
    "new search. A pinned source also fixes how its flow is counted, "
    "such as gross or net disbursements, and `propose_plan` records that "
    "choice itself, so write no `Decision` over it. To count the flow "
    "another way, pin that count's own `spec_id` from the reading's other "
    "counts. Pin a candidate's `spec_id` with a `Decision` naming the "
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
    "For a share of a figure in its own total over every sector, donor or "
    "recipient, state that total as a dataset requirement with `whole_of` "
    "naming the part and the `axis` it opens, and no `source` or "
    "`source_spec_id`: only that share reads it. US health aid over all US "
    "aid opens 'sector' (same entities and `across`); the US share of all "
    "aid to Kenya opens 'donor', Kenya's share of US aid 'recipient' (the "
    "part's entities less that party, which joins `across`). Leave it unset "
    "for a whole that binds a sector itself (health aid under malaria aid) "
    "or names a group (DAC donors' aid), a provider share (the multilateral "
    "share of US aid), or a total from a named source. Set "
    "`accept_provider_difference` on a share only when the user asked for "
    'another source\'s total, as "use the DAC1 total" does, with that total '
    "pinned, and never to get past a refusal.\n\n"
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
    "Bilateral plus imputed multilateral aid from the CRS or DAC2a is one "
    "dataset requirement with no `sum_of`: pin a reading whose `scope "
    "provider` is `provider:inclusive`, or leave it unpinned with a "
    "`measure` naming bilateral plus imputed multilateral aid.\n\n"
    "When the answer is one total of two different sources, such as "
    "bilateral aid from another table plus imputed multilateral aid, "
    "state a `sum_of`, "
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

# How to read skills, when the skill tool is built. Held apart because the
# planner is also built without it, and an instruction naming a tool the
# agent lacks sends the model at one ADK raises on.
_SKILLS = (
    'The search readings close with a "Skills named" line listing, by id, '
    "the skills that apply to them. When a skill bears on the question, "
    "call `get_skill` with its id before you draft, and let what it says "
    "shape your decisions. Read a file it links with `get_skill` and its `path`. A "
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
    f"the user's to make, call `{PLANNER_ASK}` with the candidates. "
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
    f"done and ask, through `{PLANNER_ASK}`, before you propose again, "
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
    a translator `propose_plan` runs, or the skill tool. An instruction
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


def _end_run(tool_context: ToolContext) -> EndRun:
    """The `EndRun` for the tool `tool_context` serves: ADK ends the run on its
    response (`skip_summarization`), with no model call after."""

    def end_run() -> None:
        tool_context.actions.skip_summarization = True

    return end_run


def end_run_when_ready(answer: dict[str, object], call: AdkCall) -> dict[str, object]:
    """`declare_ready` over the ADK `call`, ending the run once it is ready (`ready_and_end`)."""
    return ready_and_end(answer, call, _end_run(call.tool_context))


@dataclass(frozen=True)
class AdkCall:
    """The `planner.call.PlannerCall` an ADK tool call is, over the `ToolContext` serving it.

    `ToolContext` is `Context` (`google.adk.tools.tool_context`), so
    `tool_context.session` is the same live `Session` every other seam
    shares, `events` included. The steps are those events read as
    `session_steps` reads them.
    """

    tool_context: ToolContext

    @property
    def state(self) -> SessionState:
        """The call's session state."""
        return self.tool_context.state

    @property
    def call_id(self) -> str | None:
        """The id of the function call the tool serves."""
        return self.tool_context.function_call_id

    @functools.cached_property
    def _steps(self) -> tuple[Step, ...]:
        return tuple(session_steps(self.tool_context.session.events, serving=self.call_id))

    def steps(self) -> Sequence[Step]:
        """`session_steps` over the call's session events, read once per call.

        An `AdkCall` serves one tool call, and ADK appends a model response's
        events only after its tool calls finish, so the events a call sees do
        not change while it runs and the cached view cannot outlive the
        response it was read in.
        """
        return self._steps

    def response_calls(self) -> Sequence[ResponseCall]:
        """The calls of the planner model's response that holds the call's own
        (`_response_holding`)."""
        if self.call_id is None:
            return ()
        calls = _response_holding(self.tool_context.session.events, self.call_id)
        return tuple(ResponseCall(call.name, call.id) for call in calls)


def _function_calls(event: Any) -> list[Any]:
    """The function calls of `event`'s parts, in order."""
    if not (event.content and event.content.parts):
        return []
    return [part.function_call for part in event.content.parts if part.function_call]


def _response_holding(events: Sequence[Event], call_id: str) -> list[Any]:
    """The function calls of the latest model response in `events` that holds a
    call with `call_id`, in order, or none.

    The graph's planner node pauses on an accepted ask with an interrupt call
    (`REQUEST_INPUT_FUNCTION_CALL_NAME`) under the ask's own id, so a later
    event can hold the id without being the response that made the call. A
    call of that name under `call_id` is skipped.
    """
    for event in reversed(events):
        calls = _function_calls(event)
        if any(
            call.id == call_id and call.name != REQUEST_INPUT_FUNCTION_CALL_NAME for call in calls
        ):
            return calls
    return []


def _in_flight_calls(events: Sequence[Event], serving: str | None) -> frozenset[str]:
    """The ids of the calls in the model response that holds the call `serving`
    which no event answers yet.

    `serving` is a tool that reads steps, never an ask, which decides from its
    response's call list instead (`_ask_beside_ask`).

    ADK appends a response's merged answers after all its calls finish, so
    while one call of the response runs, its siblings have no answer in the
    session, whether they have finished or not.
    """
    if serving is None:
        return frozenset()
    answered = {
        part.function_response.id
        for event in events
        for part in (event.content.parts if event.content and event.content.parts else ())
        if part.function_response is not None
    }
    calls = _response_holding(events, serving)
    if not any(call.id == serving and call.name != PLANNER_ASK for call in calls):
        return frozenset()
    return frozenset(call.id for call in calls if call.id and call.id not in answered)


def session_steps(events: Sequence[Event], *, serving: str | None = None) -> Iterator[Step]:
    """The steps of a session's `events`, in event order (`planner.call.Step`).

    The only reader of ADK events in the planner. A message the user typed
    (`_user_message`) is one `message` step; the event's other parts are not
    read, and no producer in the harness sends a user-typed event that also
    carries a function part. Every other event yields a `call` step for each
    ask carrying an id, an `answer` step for each response to one, paired
    with the `args` of the nearest preceding call sharing its id, and a
    `response` step for each other tool's response.

    An ask is the planner's own (`PLANNER_ASK`) or one of the workflow's
    interrupts. The planner's call holds its question flat in `args`. A
    planner ask whose response is its tool's refusal, or ADK's rejection of
    the call, never reached the user: it yields no step at all, so every
    reader sees the session as if the ask had not been made. An accepted
    one's tool answers `ASK_ACCEPTED`, which says nothing the user did and is
    not read,
    and the graph's planner node pauses on it with a `RequestInput` under
    the call's own id, so the user's answer to that interrupt is the
    answer to the call, and the interrupt's own call repeats the question.
    A workflow interrupt is a `RequestInput` its node yields:
    `create_request_input_event` builds the call with `id=interrupt_id` and
    dumps the whole `RequestInput` into its `args`, so the node's own
    `payload` nests under `args["payload"]`, and
    `create_request_input_response`, which `harness.studio`'s `/interrupt`
    route sends, builds the answer with `FunctionResponse.id` set to that
    same `interrupt_id`.

    `serving` is the id of the tool call being served. The calls of its own
    model response that no event answers yet (`_in_flight_calls`) yield no
    step, so a call running beside an ask reads the session as it stood
    before that response, and the ask, which may still be refused or may be
    accepted, is not counted as put to the user.
    """
    refused = _refused_asks(events)
    in_flight = _in_flight_calls(events, serving)
    calls: dict[str, dict[str, Any]] = {}
    planner_asks: set[str] = set()
    for event in events:
        if not (event.content and event.content.parts):
            continue
        if _user_message(event):
            yield Step("message")
            continue
        for part in event.content.parts:
            call = part.function_call
            if call is not None and call.id is not None:
                if call.id in in_flight:
                    continue
                if call.name == PLANNER_ASK:
                    if call.id in refused:
                        continue
                    planner_asks.add(call.id)
                elif call.name != REQUEST_INPUT_FUNCTION_CALL_NAME or call.id in planner_asks:
                    continue
                calls[call.id] = call.args or {}
                yield Step("call", call.id, call.args or {})
                continue
            response = part.function_response
            if response is None:
                continue
            body = response.response or {}
            if response.name == PLANNER_ASK and "result" not in body:
                continue
            if response.name in (PLANNER_ASK, REQUEST_INPUT_FUNCTION_CALL_NAME):
                yield Step("answer", response.id, calls.get(response.id or ""), body)
            else:
                yield Step("response", body=body, name=response.name)


def _refused_asks(events: Iterable[Event]) -> frozenset[str]:
    """The ids of the planner asks that never reached the user: their response
    is neither `ASK_ACCEPTED` nor an answer (no `result`), so it is the tool's
    refusal or ADK's rejection of the call."""
    return frozenset(
        response.id
        for event in events
        for part in (event.content.parts if event.content and event.content.parts else ())
        if (response := part.function_response) is not None
        and response.id is not None
        and response.name == PLANNER_ASK
        and "result" not in (body := response.response or {})
        and body.get("outcome") != Outcome.OK.value
    )


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


# The `error.type` ADK records on a `FunctionTool`'s span when its answer holds
# an `error` key.
_TOOL_ERROR = "TOOL_ERROR"


def _env_flag(name: str) -> bool | None:
    """The environment variable `name` as google-genai reads a flag: `true` or `1`
    in any case is true, any other value false, and an unset variable `None`."""
    value = os.environ.get(name)
    return None if value is None else value.lower() in ("true", "1")


def refuse_vertex_ai(model_name: str, client: Client | None = None) -> None:
    """Refuse to build the planner when the google-genai client its model calls
    through uses Vertex AI.

    The planner's declarations are the Gemini developer API's form
    (`planner.tool_spec.function_declaration`). Vertex AI rejects their
    optional `anyOf` schemas, which ADK rewrites only for its own function
    tools. `create_planner_agent` calls this once, when the planner's model
    is ADK's `Gemini`, with that model's name and the client it calls through.
    Any other model calls through no google-genai client.
    `harness.planner_seam.PlannerSeam` calls it for the Pydantic AI planner's
    model and client.

    The decision is the one the client makes. A given client decides by its
    `vertexai`. With none, it is the client ADK's `Gemini` builds for
    `model_name` with no client options (`Gemini.api_client`): a name of the
    form `projects/...` selects Vertex AI, and otherwise google-genai takes
    the backend from the environment, `GOOGLE_GENAI_USE_ENTERPRISE` when set,
    else `GOOGLE_GENAI_USE_VERTEXAI`, each true when `true` or `1`. That rule
    is read here without building a client, since building one without an API
    key or credentials raises.

    Args:
        model_name: The Gemini model's name, as the client is sent it.
        client: The client the model calls through, or None for the one built
            for it.

    Raises:
        RuntimeError: The client uses Vertex AI.
    """
    if client is not None:
        vertexai = client.vertexai
    elif model_name.startswith("projects/"):
        vertexai = True
    else:
        enterprise = _env_flag("GOOGLE_GENAI_USE_ENTERPRISE")
        vertexai = enterprise if enterprise is not None else _env_flag("GOOGLE_GENAI_USE_VERTEXAI")
    if vertexai:
        raise RuntimeError(
            "The planner's google-genai client uses Vertex AI, which rejects the "
            "planner's tool declarations. Run the planner on a Gemini developer API "
            "key (GOOGLE_API_KEY): give a client built without vertexai, or name the "
            "model without a projects/ path and unset GOOGLE_GENAI_USE_ENTERPRISE and "
            "GOOGLE_GENAI_USE_VERTEXAI."
        )


class SpecTool(BaseTool):
    """A `planner.tool_spec.ToolSpec` served to an ADK `LlmAgent`.

    The model is shown `function_declaration(spec)`, and a call runs the
    spec's body over the call's `AdkCall` (`run_tool`), ending the run through
    `skip_summarization`. `_get_declaration` is the hook `BaseTool` reads the
    declaration through. Its `process_llm_request` adds the declaration to the
    request's one `types.Tool` of function declarations, creating it when the
    request has none, and registers the tool under its name.

    The declaration is the Gemini developer API's form, which Vertex AI
    rejects (`refuse_vertex_ai`).
    """

    def __init__(self, spec: ToolSpec) -> None:
        super().__init__(name=spec.name, description=spec.description)
        self.spec = spec

    @functools.cached_property
    def declaration(self) -> types.FunctionDeclaration:
        """The declaration the model is shown for the spec, built once."""
        return function_declaration(self.spec)

    @override
    def _get_declaration(self) -> types.FunctionDeclaration:
        # Return a copy, since the request holds what this returns and later
        # request processing may edit it in place.
        return self.declaration.model_copy(deep=True)

    def invoke(self, args: dict[str, Any], tool_context: ToolContext) -> object:
        """The spec's answer to a call with `args`, an awaitable when its body is async."""
        return run_tool(self.spec, args, call=AdkCall(tool_context), end_run=_end_run(tool_context))

    @override
    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        """The spec's answer to a call with `args`.

        An answer holding an `error` key, the refusal of a call missing a
        required argument (`planner.tool_spec.missing_arguments`), marks the
        tool's span, which ADK makes current while the tool runs, failed with
        `error.type` `TOOL_ERROR`, as ADK marks a `FunctionTool`'s.
        """
        answer = self.invoke(args, tool_context)
        if inspect.isawaitable(answer):
            answer = await answer
        if isinstance(answer, dict) and answer.get("error"):
            span = trace.get_current_span()
            span.set_attribute(ERROR_TYPE, _TOOL_ERROR)
            span.set_status(Status(StatusCode.ERROR, _TOOL_ERROR))
        return answer


# The brief for the planner's current run, which `prepend_brief` shows the model.
# `harness.planner_seam.PlannerSeam` renders it before every run, a resume
# included, so a `temp:` key that is never persisted is enough.
BRIEF_STATE_KEY = "temp:planner_brief"


def prepend_brief(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
    """Show the model the run's brief (`BRIEF_STATE_KEY`) right after the turn's question.

    Every request of the run carries it, the first request after a resume
    included, since ADK hands a node its input only on a live run. The brief
    sits ahead of the run's own calls and responses, so each request extends
    the one before it. Adds nothing when state holds no brief.
    """
    brief = callback_context.state.get(BRIEF_STATE_KEY)
    if not brief:
        return
    llm_request.contents.insert(
        after_turn_question(callback_context, llm_request.contents),
        types.Content(role="user", parts=[types.Part(text=brief)]),
    )


def planner_specs(
    *,
    translate: Translate | None = None,
    surface: ToolSurface | None = None,
    question_fixed: bool = False,
    reads_user_words: bool = True,
) -> list[ToolSpec]:
    """The planner's tools, in the order the model is shown them: `get_skill`
    when `surface` is given (`planner.specs.skill_tool_spec`), then
    `planner.specs.planner_tool_specs`, each argument as
    `create_planner_agent` documents it."""
    return [
        *([skill_tool_spec(surface)] if surface is not None else []),
        *planner_tool_specs(
            translate=translate,
            question_fixed=question_fixed,
            reads_user_words=reads_user_words,
            surface=surface,
        ),
    ]


def create_planner_agent(
    *,
    model_name: str,
    translate: Translate | None = None,
    surface: ToolSurface | None = None,
    question_fixed: bool = False,
    reads_user_words: bool = True,
    prefix_cache: PrefixCache | None = None,
    specs: Sequence[ToolSpec] | None = None,
    client: Client | None = None,
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
        surface: The tool surface `get_skill` calls
            data-mcp through, and `propose_plan` reads a pinned dataset's
            first year through. Omitted, the planner has no skill tool,
            its instruction does not name it, and `propose_plan` reads
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
        specs: The tools the agent serves. None gives `planner_specs` over the
            arguments above.
        client: The google-genai client a Gemini model calls through, used as
            it is, HTTP options included. None gives one ADK builds from the
            environment.

    Returns:
        The configured `LlmAgent`.

    Raises:
        RuntimeError: The model is ADK's `Gemini` and its google-genai client
            would use Vertex AI (`refuse_vertex_ai`).
        ValueError: A client is given for a model other than ADK's `Gemini`,
            which alone takes one.
    """
    model = _retrying(model_name, prefix_cache=prefix_cache)
    if isinstance(model, Gemini):
        refuse_vertex_ai(model.model, client)
        if client is not None:
            model = model.model_copy(update={"client": client})
    elif client is not None:
        raise ValueError(f"a client runs a Gemini planner model only, not {model_name!r}")
    return LlmAgent(
        name=AGENT_NAME,
        model=model,
        instruction=planner_instruction(
            has_translator=translate is not None,
            has_skills=surface is not None,
            question_fixed=question_fixed,
        ),
        tools=[
            SpecTool(spec)
            for spec in (
                specs
                if specs is not None
                else planner_specs(
                    translate=translate,
                    surface=surface,
                    question_fixed=question_fixed,
                    reads_user_words=reads_user_words,
                )
            )
        ],
        # Run as a workflow node (`harness.planner_seam.PlannerSeam._propose`'s
        # `ctx.run_node`), ADK runs an agent with no explicit `mode` as
        # `single_turn`. `'none'` would build every request from
        # `_get_current_turn_contents`, which slices the session from the
        # latest turn-starting event. `'default'` uses `_get_contents`, which
        # reads the turn's question and every event on this run's node path,
        # so `prepend_brief` finds the question to follow.
        include_contents="default",
        # A stream cut off mid-response resumes from a continuation token, one
        # extra hop instead of a full retry. google-genai defaults this to
        # true; it is set so a change of that default cannot turn the resume
        # off.
        generate_content_config=types.GenerateContentConfig(automatic_continuation=True),
        # `AGENT_THINKING_CONFIG` (see `planner.settings`) pins every
        # generation to LOW thinking.
        planner=BuiltInPlanner(thinking_config=AGENT_THINKING_CONFIG),
        # Drops this generation's thought-summary parts once the trace has
        # them, so `include_contents="default"` above never replays one into
        # a later request of the same turn. See `drop_thought_parts`.
        after_model_callback=drop_thought_parts,
        before_model_callback=prepend_brief,
    )
