"""The planner agent: the planner's ADK tools and `create_planner_agent`.

The tools' bodies live in `planner.tools` (`ask_user`, `propose`, `reject`) and
`planner.skills` (`read_skill_page`), over a `planner.call.PlannerCall` and no
framework. Here `AdkCall` is the `PlannerCall` an ADK tool call is, and each
tool is a thin function carrying the signature and docstring ADK declares to the
model. A `propose_plan` whose translation compiles declares the plan ready for
the user's answer in code (`planner.tools.declare_ready`), and
`end_run_when_ready` ends the ADK run on it (`skip_summarization`) with no
model call after. Declaring a plan ready approves nothing: whatever runs this
agent withholds execution until that answer arrives, which for the served path
is the harness's own graph, keyed on the plan's revision rather than on an ADK
tool confirmation this agent would have to carry across its own nested `Runner`.

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

Every tool below answers with `planner.tool_results`' typed shape -- `ok`,
`problem`, with a stable `code` and, where the next step is known, an
`action` -- rather than an unstructured `{"error": ...}` a caller has to
parse. When translation cannot carry a plan out, `propose_plan`'s
`translation` is the `Referral` itself: this agent's instruction reads it
straight off that answer, with nothing here to fetch it back a second time.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Sequence,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

from compiler.surface import ToolSurface
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.planners import BuiltInPlanner
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from harness.turn_events import PLANNER_ASK, REQUEST_INPUT_FUNCTION_CALL_NAME, after_turn_question
from orchestrator.agent import _retrying
from pydantic import Field
from submission.service import PROPOSE_PLAN as PROPOSE_PLAN_PROPOSER
from submission.service import Proposal
from translator.tool import Translate

from planner.call import (
    ResponseCall,
    Step,
)
from planner.declaration import PlanTool
from planner.plan import (
    MAX_PLAN_DECISIONS,
    MAX_PLAN_REQUIREMENTS,
    AnswerMode,
    AskAnswer,
    DecisionDraft,
    Decline,
    EvidenceRequirement,
    SessionState,
)
from planner.settings import AGENT_THINKING_CONFIG, drop_thought_parts
from planner.skills import DFO_FAQ_SKILL_ID, read_skill_page
from planner.tool_results import Outcome
from planner.tools import (
    EndRun,
    ask_user,
    propose,
    propose_and_translate,
    propose_with_reads,
    ready_and_end,
    reject,
)

if TYPE_CHECKING:
    from harness.prefix_cache import PrefixCache

logger = logging.getLogger(__name__)

AGENT_NAME = "dfo_planner"

# The tool the planner proposes a plan through.
PROPOSE_PLAN = PROPOSE_PLAN_PROPOSER.call

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
    `_session_steps` reads them.
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
        return tuple(_session_steps(self.tool_context.session.events, serving=self.call_id))

    def steps(self) -> Sequence[Step]:
        """`_session_steps` over the call's session events, read once per call.

        An `AdkCall` serves one tool call, and ADK appends a model response's
        events only after its tool calls finish, so the events a call sees do
        not change while it runs and the cached view cannot outlive the
        response it was read in.
        """
        return self._steps

    def response_calls(self) -> Sequence[ResponseCall]:
        """The calls of the session event that holds the call's own, the latest such event."""
        if self.call_id is None:
            return ()
        for event in reversed(self.tool_context.session.events):
            calls = _function_calls(event)
            if any(call.id == self.call_id for call in calls):
                return tuple(ResponseCall(call.name, call.id) for call in calls)
        return ()


def _function_calls(event: Any) -> list[Any]:
    """The function calls of `event`'s parts, in order."""
    if not (event.content and event.content.parts):
        return []
    return [part.function_call for part in event.content.parts if part.function_call]


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
    for event in reversed(events):
        calls = _function_calls(event)
        if any(call.id == serving and call.name != PLANNER_ASK for call in calls):
            return frozenset(call.id for call in calls if call.id and call.id not in answered)
    return frozenset()


def _session_steps(events: Sequence[Event], *, serving: str | None = None) -> Iterator[Step]:
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
        "computes part of it, unless `removed_asks` names it."
    ),
    "removed_asks": (
        "The number of each ask the user's amendment removes from the plan. Leave it out "
        "of `answered` and `declined`, and narrow or remove what answered it. Only the "
        "user's amendment removes an ask, and only a later one restores it, by answering it."
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


def ask_user_question(
    tool_context: ToolContext,
    message: str,
    candidates: list[dict[str, str]] = [],  # noqa: B006
    options: list[dict[str, str]] = [],  # noqa: B006
    recommended_spec_id: str | None = None,
    rationale: str | None = None,
) -> dict[str, object]:
    """Ask the user a question a referral or a refusal leaves to them, ending your turn.

    The user's answer arrives with your next brief, after the question it
    answers, naming the candidate or option they picked, or in their own
    words if they answered outside them. The choices offered come from the
    referral's candidates, or from the `options` a refusal's action carries;
    with neither, the question says what cannot be done and asks what to
    change.

    Args:
        message: The question to show the user, naming what is
            ambiguous and, briefly, what each candidate is, in plain
            words: never a requirement id, spec_id or code.
        candidates: The referral's own candidates, each
            `{"spec_id": ..., "label": ...}` -- `label` is the
            candidate's own `sentence`, not invented text. Omit it for
            a referral that carries none.
        options: The `options` a refusal's action carries, unchanged.
            Omit it otherwise.
        recommended_spec_id: The candidate you favour, when you favour
            one -- must name one of `candidates`.
        rationale: Why you favour it, for the user to read, when
            `recommended_spec_id` is given.
    """
    # `= []` defaults keep the declaration's arrays flat, as `propose_plan`'s do.
    return ask_user(
        AdkCall(tool_context),
        _end_run(tool_context),
        message=message,
        candidates=candidates,
        options=options,
        recommended_spec_id=recommended_spec_id,
        rationale=rationale,
    )


# Keep the description to one sentence saying what the call returns: a longer one
# makes Gemini reason at length before some calls (`tests/planner/test_skills.py`
# bounds its length). ADK sends a docstring's `Args:` block inside the description,
# so the argument text sits in the parameters' `Field` descriptions.
_GET_SKILL_DOC = (
    "Returns the SKILL.md of the skill with the given id and the list of its files, "
    "or one file of the skill, verbatim, when `path` is given."
)


def build_skill_tools(surface: ToolSurface) -> list[FunctionTool]:
    """Build `get_skill`, which calls data-mcp through `surface`."""

    async def get_skill(
        skill_id: Annotated[
            str, Field(description=f"A skill id, or `{DFO_FAQ_SKILL_ID}` for DFO itself.")
        ],
        tool_context: ToolContext,
        path: Annotated[
            str | None, Field(description="A file's path exactly as the list gives it.")
        ] = None,
        offset: Annotated[
            int, Field(description="Where a page starts: 0, or the offset a closing line names.")
        ] = 0,
        digest: Annotated[
            str | None, Field(description="The digest a closing line names, on a continuation.")
        ] = None,
    ) -> str:
        arguments: dict[str, object] = {"skill_id": skill_id}
        if path is not None:
            arguments["path"] = path
        if offset:
            arguments["offset"] = offset
        if digest is not None:
            arguments["digest"] = digest
        return await read_skill_page(surface, arguments, call=AdkCall(tool_context))

    get_skill.__doc__ = _GET_SKILL_DOC
    return [FunctionTool(get_skill)]


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
    ready (`end_run_when_ready`). Omitted, `propose_plan` records the plan and
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
        removed_asks: list[int] = [],  # noqa: B006
        across_years: list[str] = [],  # noqa: B006
        amends_version: int | None = None,
        starts_new_plan: bool = False,
    ) -> dict[str, object]:
        """Record a plan, or patch the one already recorded.

        Returns `ok` with `status` `"draft"`; the new `version`, which a
        patch passes as `amends_version`; the `previous_version` it amended,
        `None` on a first draft; and the `question` as stored.
        """
        return propose(
            AdkCall(tool_context),
            Proposal(
                question=question,
                answer_mode=answer_mode,
                decisions=decisions,
                evidence_requirements=evidence_requirements,
                removed_decisions=removed_decisions,
                removed_evidence_requirements=removed_evidence_requirements,
                declined=declined,
                answered=answered,
                removed_asks=removed_asks,
                across_years=across_years,
                amends_version=amends_version,
                starts_new_plan=starts_new_plan,
            ),
            question_fixed=question_fixed,
            reads_user_words=reads_user_words,
        )

    def reject_plan(tool_context: ToolContext) -> dict[str, object]:
        """Record that the user rejected the current plan.

        Nothing in the plan may run afterwards. Call `propose_plan` again if
        the user wants a revision rather than an outright stop.
        """
        return reject(AdkCall(tool_context))

    @functools.wraps(propose_plan)
    async def propose_plan_reading(**arguments: Any) -> dict[str, object]:
        """`propose_plan`, making each source read it refuses for want of (`propose_with_reads`)."""
        assert surface is not None
        return await propose_with_reads(
            AdkCall(arguments["tool_context"]),
            lambda: propose_plan(**arguments),
            surface=surface,
        )

    @functools.wraps(propose_plan)
    async def propose_plan_translating(**arguments: Any) -> dict[str, object]:
        """`propose_plan`, translating the plan it records (`propose_and_translate`)."""
        assert translate is not None
        return await propose_and_translate(
            AdkCall(arguments["tool_context"]),
            _end_run(arguments["tool_context"]),
            arguments=arguments,
            draft=lambda: propose_plan(**arguments),
            translate=translate,
            surface=surface,
        )

    proposing: Callable[..., Any] = propose_plan
    if translate is not None:
        proposing = propose_plan_translating
    elif surface is not None:
        proposing = propose_plan_reading
    arguments = (
        _PROPOSE_PLAN_ARGUMENTS | _FIXED_ARGUMENTS if question_fixed else _PROPOSE_PLAN_ARGUMENTS
    )
    return [
        FunctionTool(ask_user_question),
        PlanTool(proposing, arguments=arguments),
        FunctionTool(reject_plan),
    ]


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
