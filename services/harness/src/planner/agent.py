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
import copy
import functools
import logging
from collections.abc import Awaitable, Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast, override

from compiler.evidence import PriceBasis
from compiler.sources import block_group, party_class, readable_wire_groups
from compiler.surface import ToolSurface, ToolSurfaceUnavailableError
from data_mcp.definition_store import canonicalize, spec_id_for
from data_mcp.definition_writes import windowed_identity
from data_mcp.scope import ChoiceOrigin, ScopeEntry
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.planners import BuiltInPlanner
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from harness.graph.orchestrate import counts_chosen
from harness.graph.state import ORCHESTRATOR_SPECS_KEY, resume_answers
from harness.plan_display import DECISION_CHOICE_PREFIX, decision_option_id
from harness.scope_choices import (
    PROVIDER_AXIS,
    EntitiesBinding,
    PartScope,
    ScopeChoice,
    mark_planner_axes,
    option_axis,
    planner_axes,
    switch_entries,
)
from harness.tool_contracts import resolver_digest
from harness.turn_events import REQUEST_INPUT_FUNCTION_CALL_NAME, after_turn_question
from orchestrator.agent import _retrying
from pydantic import JsonValue
from submission.prices_choice import (
    PRICES_BY_LABEL,
    PRICES_OPTIONS,
    PRICES_PICKS,
    PRICES_SLOT_PREFIX,
    PricesChoice,
    measure_key,
    prices_slot,
)
from submission.service import (
    COMBINATION_OPTIONS,
    COMBINE_OPTION,
    WHOLE_COLUMNS,
    Admission,
    AlignedPin,
    Combination,
    Proposal,
    Realignment,
    ReferralAnswers,
    SearchedPin,
    Unpinned,
    aligned_pins,
    axis_slot,
    basis_unservable,
    block_parties,
    bound_parties,
    part_scope,
    realignments_wanted,
    save_aligned_pin,
    save_searched_pin,
    share_whole_underivable,
    submit,
    unpinned_wanted,
    unread_part,
    whole_axis_held,
)
from submission.service import PROPOSE_PLAN as PROPOSE_PLAN_PROPOSER
from translator.discovery import (
    SEARCH,
    DefinitionNotResolvedError,
    PartNotResolvedError,
    answered_once,
    call_within_entry,
    load_log,
    refusal_retryable,
    resolve_definition,
    resolved_spec,
)
from translator.referral import (
    USER_DECIDES,
    Referral,
    ReferralKind,
    clear_referral,
    load_referral,
)
from translator.tool import Translate, authored_answer

from planner.declaration import PlanTool
from planner.plan import (
    MAX_PLAN_DECISIONS,
    MAX_PLAN_REQUIREMENTS,
    AnswerMode,
    AskAnswer,
    DecisionDraft,
    DecisionPick,
    Decline,
    EvidenceRequirement,
    FullHistory,
    PinSource,
    Plan,
    PlanStatus,
    SessionState,
    UnsupportedOperation,
    decision_pick,
    full_histories,
    full_history,
    held_sources,
    identity_on_record,
    load_plan,
    pin_sources,
    plan_revision,
    record_optional_decline,
    record_reduction_refused_asks,
    record_unsupported,
    save_alignments,
    save_full_history,
    save_pin_sources,
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

logger = logging.getLogger(__name__)

AGENT_NAME = "dfo_planner"

# The tool the planner proposes a plan through.
PROPOSE_PLAN = "propose_plan"


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
    return {
        question_id: _candidate_ids(candidates)
        for question_id, candidates in _question_candidates(args).items()
    }


def _question_candidates(args: dict[str, Any]) -> dict[str, Any]:
    """The raw candidates list of each question an `adk_request_input` call's
    `args` offers, by question id, read as `_offered_candidates` reads them."""
    candidates = args.get("candidates")
    payload = args.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    if not candidates and payload.get("questions"):
        return {
            str(question["id"]): question.get("candidates")
            for question in payload["questions"]
            if isinstance(question, dict) and question.get("id")
        }
    return {"q1": candidates or payload.get("candidates")}


def _prices_chosen(tool_context: ToolContext) -> dict[str, PricesChoice]:
    """The prices the user picked answering `adk_request_input` calls, by the
    part each prices, read off this session's event history.

    The `ask` node marks each prices option's candidate with its `part` and
    `prices` (`harness.graph.state.ask_candidates`), so a pick states them
    whether or not the option carries a spec_id. Each response pairs with the
    nearest preceding call sharing its `id`, as `_elected_spec_ids` pairs
    them. A choice stands until the user answers the same question again,
    a question pricing the same part or another window of its measure
    (`submission.prices_choice.measure_key`): a pick of other prices
    replaces it, and any other answer, a skip aside, releases it, so one
    measure holds one choice however many asks priced it.

    A prices refusal that sends the planner to the user (`detail.part`,
    `submission.prices_choice`) binds the planner's next ask offering
    `PRICES_OPTIONS` to its part, and the user's pick there sets the part's
    prices, marked `asked` so the planner does not ask it again
    (`submission.prices_choice`). The user's own words there release the
    part's choice, as they do on the orchestrator's question, and a skip
    keeps it, marking a constant choice `asked`. A switch of the part's
    prices row on the plan card (`_prices_switched`) is a new answer for the
    part, marked `asked` too, while its replan is under way or once the plan
    on record states it (`_switch_stands`). Any other `propose_plan`
    response, the plan landing included, clears the refusal, so no later ask
    of another kind is read as that one. A message the user typed
    (`_user_message`) releases every choice and clears the refusal, and an
    amendment of the approval card (`_is_amendment`) releases every choice.
    An answer to any other question leaves them standing. A Studio turn is
    one session, so a choice holds for the turn it was made in.
    """
    measures = {
        spec_id: measure_key(block)
        for spec_id, source in pin_sources(tool_context.state).items()
        if (block := source.spec_block(spec_id)) is not None
    }
    chosen: dict[str, PricesChoice] = {}
    refused: str | None = None
    asked: dict[str, str] = {}
    offered = [dict(option) for option in PRICES_OPTIONS]
    for step in _session_steps(tool_context):
        if step.kind == "message":
            chosen.clear()
            refused = None
        elif step.kind == "response":
            if step.name == PROPOSE_PLAN:
                refused = _prices_refused(step.body)
        elif step.kind == "call":
            if refused is not None and (step.args or {}).get("options") == offered:
                asked[str(step.call_id)], refused = refused, None
        elif _is_amendment(dict(step.body)):
            chosen.clear()
        elif (switched := _prices_switched(step.body)) is not None:
            part, basis = switched
            if _switch_stands(tool_context.state, part, basis):
                _release_measure(chosen, part, measures)
                chosen[part] = PricesChoice(basis, asked=True)
        elif step.call_id in asked:
            part = asked.pop(step.call_id)
            answer = _answers(step.body.get("result")).get("q1") or {}
            basis = PRICES_PICKS.get(str(answer.get("spec_id")))
            if answer.get("kind") == "option" and basis is not None:
                _release_measure(chosen, part, measures)
                chosen[part] = PricesChoice(basis, asked=True)
            elif _said(answer):
                _release_measure(chosen, part, measures)
            elif part in chosen and chosen[part].prices == "constant":
                chosen[part] = chosen[part]._replace(asked=True)
        else:
            _priced_answers(step.args or {}, step.body, chosen, measures)
    return chosen


def _prices_switched(body: Mapping[str, Any]) -> tuple[str, PriceBasis] | None:
    """The part and prices a switch of a part's prices row on the plan card
    picks (`harness.studio`'s `{"result": {"switch": {"choice_id",
    "option_id"}}}` answer to the approval card), or `None` for any other answer."""
    result = body.get("result")
    switch = result.get("switch") if isinstance(result, dict) else None
    if not isinstance(switch, dict):
        return None
    slot = str(switch.get("choice_id")).removeprefix(DECISION_CHOICE_PREFIX)
    if not slot.startswith(PRICES_SLOT_PREFIX):
        return None
    basis = next(
        (
            basis
            for label, basis in PRICES_BY_LABEL.items()
            if decision_option_id(label) == switch.get("option_id")
        ),
        None,
    )
    return (slot.removeprefix(PRICES_SLOT_PREFIX), basis) if basis is not None else None


def _switch_stands(state: Any, part: str, basis: PriceBasis) -> bool:
    """Whether a switch of `part`'s prices row to `basis` holds: its replan is
    under way (`planner.plan.decision_pick` names the row), or the plan on
    record states the switched prices. A switch whose replan was refused or
    stopped holds neither, so the prices stay as the card shows them."""
    slot = prices_slot(part)
    pick = decision_pick(state)
    if pick is not None and pick.slot == slot:
        return True
    plan = load_plan(state)
    label = PricesChoice(basis).label
    return plan is not None and any(d.slot == slot and d.chosen == label for d in plan.decisions)


def _release_measure(
    chosen: dict[str, PricesChoice], part: str, measures: Mapping[str, str | None]
) -> None:
    """Take out of `chosen` the choice for `part` and for every part measuring
    what it measures (equal `measures`), as a new answer pricing it replaces them."""
    key = measures.get(part)
    for priced in [p for p in chosen if p == part or (key is not None and measures.get(p) == key)]:
        del chosen[priced]


def _prices_refused(body: Mapping[str, Any]) -> str | None:
    """The part a `propose_plan` refusal of the user's prices sending the planner
    to the user names (`detail.part`), or `None`."""
    detail = body.get("detail")
    action = body.get("action")
    if not (
        isinstance(detail, dict)
        and isinstance(action, dict)
        and action.get("call") == REQUEST_INPUT_FUNCTION_CALL_NAME
        and str(detail.get("slot")).startswith(PRICES_SLOT_PREFIX)
    ):
        return None
    part = detail.get("part")
    return part if isinstance(part, str) else None


def _priced_answers(
    args: dict[str, Any],
    body: Mapping[str, Any],
    chosen: dict[str, PricesChoice],
    measures: Mapping[str, str | None],
) -> None:
    """Record in `chosen` what one clarify response says of each part its call
    priced, each answer replacing the choices of its part's measure (`_prices_chosen`)."""
    answers = _answers(body.get("result"))
    for question_id, candidates in _question_candidates(args).items():
        marked = [c for c in candidates or () if isinstance(c, dict) and c.get("part")]
        answer = answers.get(question_id)
        if not marked or not isinstance(answer, dict) or not _said(answer):
            continue
        part = str(marked[0]["part"])
        picked = next(
            (
                choice
                for candidate in marked
                if answer.get("kind") == "option"
                and candidate.get("spec_id") == answer.get("spec_id")
                and (choice := PricesChoice.of_candidate(candidate)) is not None
            ),
            None,
        )
        _release_measure(chosen, part, measures)
        if picked is not None:
            chosen[part] = picked


class _Step(NamedTuple):
    """One step of a session's history, as the readers of what the user answered walk it.

    `message` is a message the user typed (`_user_message`). `call` is an
    `adk_request_input` call, with its id and `args`. `answer` is a response
    to one, with its id, the `args` of the nearest preceding call sharing
    that id (`None` when none precedes it) and the response body. `response`
    is any other tool's response, with its `name` and body.
    """

    kind: Literal["message", "call", "answer", "response"]
    call_id: str | None = None
    args: dict[str, Any] | None = None
    body: Mapping[str, Any] = MappingProxyType({})
    name: str | None = None


def _session_steps(tool_context: ToolContext) -> Iterator[_Step]:
    """The steps of `tool_context`'s session history, in event order (`_Step`)."""
    calls: dict[str, dict[str, Any]] = {}
    for event in tool_context.session.events:
        if not (event.content and event.content.parts):
            continue
        if _user_message(event):
            yield _Step("message")
            continue
        for part in event.content.parts:
            call = part.function_call
            if (
                call is not None
                and call.name == REQUEST_INPUT_FUNCTION_CALL_NAME
                and call.id is not None
            ):
                calls[call.id] = call.args or {}
                yield _Step("call", call.id, call.args or {})
                continue
            response = part.function_response
            if response is None:
                continue
            body = response.response or {}
            if response.name == REQUEST_INPUT_FUNCTION_CALL_NAME:
                yield _Step("answer", response.id, calls.get(response.id or ""), body)
            else:
                yield _Step("response", body=body, name=response.name)


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
    for step in _session_steps(tool_context):
        if step.kind == "answer" and step.args is not None:
            yield step.args, step.body.get("result")


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
    return _answers_since_referral(tool_context).answered


def _answers_since_referral(tool_context: ToolContext) -> ReferralAnswers:
    """Whether the user has answered an ask the planner raised after seeing the
    translator's referral, the spec_ids they picked answering those asks, and the
    spec_ids offered by those asks still unanswered, read off this session's event
    history as `_elected_spec_ids` reads it.

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
    pairs them. A pick made before the referral is not counted. Picks keep the
    order the answers came in. An ask with no answer, a skip included, leaves
    every id it offered unanswered. An ask whose response has no `result`,
    one the tool or ADK refused, never reached the user, so it offered
    nothing. A pending ask has no response and stays unanswered.

    `picked_ever` holds the user's standing picks since the first referral,
    which a later translation does not clear. An answered ask, picked or in
    the user's own words, replaces every earlier pick among the ids it
    offered, so the latest answer about a reading stands and a free-text
    answer releases it to the planner. A message the user typed
    (`_user_message`) or an amendment of the approval card (`_is_amendment`)
    releases them all.
    """
    answered = False
    referral: tuple[object, object] | None = None
    asks: dict[str, dict[str, frozenset[str]]] = {}
    replied: set[str] = set()
    picked: list[str] = []
    picked_ever: list[str] = []
    for event in tool_context.session.events:
        if not (event.content and event.content.parts):
            continue
        if _user_message(event):
            picked_ever.clear()
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
                if _is_amendment(body):
                    picked_ever.clear()
                if response.id not in asks:
                    continue
                if "result" not in body:
                    del asks[response.id]
                    continue
                if not _is_answer(body):
                    continue
                answered = True
                replied.add(response.id)
                offered = {spec_id for ids in asks[response.id].values() for spec_id in ids}
                picks = _picks(body.get("result"), asks[response.id])
                picked.extend(picks)
                picked_ever = [pick for pick in picked_ever if pick not in offered]
                picked_ever.extend(picks)
            elif "translation" in body:
                translation = body["translation"]
                referral = (
                    (translation.get("kind"), translation.get("requirement_id"))
                    if isinstance(translation, dict) and "kind" in translation
                    else None
                )
                answered = False
                asks.clear()
                replied.clear()
                picked.clear()
    unanswered = frozenset(
        spec_id
        for call_id, offered in asks.items()
        if call_id not in replied
        for spec_ids in offered.values()
        for spec_id in spec_ids
    )
    return ReferralAnswers(
        answered=answered,
        picked=tuple(picked),
        unanswered=unanswered,
        picked_ever=tuple(picked_ever),
    )


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
    `_user_spoke_since_recorded`, `_prices_chosen`). Recording an admitted
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

    def answers_since_referral(self) -> ReferralAnswers:
        """`_answers_since_referral` over the call's session."""
        return _answers_since_referral(self.tool_context)

    def combination_picks(self) -> tuple[dict[Combination, str | None], bool]:
        """`_combination_picks` over the call's session."""
        return _combination_picks(self.tool_context)

    def user_spoke_since(self, version: int) -> bool:
        """`_user_spoke_since_recorded` over the call's session."""
        return _user_spoke_since_recorded(self.tool_context, version)

    def prices_chosen(self) -> dict[str, PricesChoice]:
        """`_prices_chosen` over the call's session."""
        return _prices_chosen(self.tool_context)

    def decision_pick(self) -> DecisionPick | None:
        """The pick on the plan card the call's planner entry holds (`planner.plan.decision_pick`)."""
        return decision_pick(self.tool_context.state)

    def record_unsupported(self, operation: UnsupportedOperation) -> None:
        """Record `operation` as answered `unsupported` in this planner entry."""
        record_unsupported(self.tool_context.state, operation)

    def record_optional_decline(self, operation: UnsupportedOperation) -> None:
        """Permit `operation` as an optional decline in this planner entry."""
        record_optional_decline(self.tool_context.state, operation)

    def record_reduction_refused(self, asks: Iterable[int]) -> None:
        """Record `asks` as named by a `PERIOD_REDUCTION_UNSTATED` refusal in this planner entry."""
        record_reduction_refused_asks(self.tool_context.state, asks)

    def record_admission(self, admission: Admission) -> None:
        """Record `admission` in the call's state."""
        state = self.tool_context.state
        if admission.windowed_pins is not None:
            save_windowed_pins(state, admission.windowed_pins)
        save_alignments(state, admission.alignments)
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
        "computes part of it. When the user's amendment removes every requirement "
        "answering an ask, leave that ask out of both: the plan records that the "
        "user removed it."
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


def _first_year_identity(pin: str, state: SessionState) -> Mapping[str, Any] | None:
    """The identity a first-year read of `pin` reads; `None` when none is on record.

    It is on record as `submission.service.admit` finds it, the discovery
    log's definitions included, so a pin `propose_plan` admits is read.
    """
    recorded = identity_on_record(
        pin,
        shown=held_sources(state),
        end_open=True,
        definition_lookup=load_log(state).recorded_definition,
    )
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


# A read of a source, made: applied to the session state, it makes the read's
# writes and answers the refusal the proposal is given, or `None` once the
# read is held. `with_source_reads` makes the reads one refusal asks for at
# once and applies them in order, so a read after a refused one writes nothing.
type SourceRead = Callable[[SessionState], dict[str, object] | None]


def _refusing(refused: dict[str, object]) -> SourceRead:
    """The read answering `refused`, writing nothing."""
    return lambda _: refused


async def compile_on_basis(  # noqa: PLR0911 -- one distinct refusal per way the compile fails
    surface: ToolSurface,
    realignment: Realignment,
    *,
    state: SessionState,
) -> SourceRead:
    """Compile `realignment`'s pin on its calculation's options and hold the pin Search answers.

    Posts the pin's base identity with its scope entries and each option, on
    the axis it names (`harness.scope_choices.option_axis`), as an explicit
    entry from its choice in place of the entry on that axis, the resubmit
    a switch on the plan card posts (`harness.scope_choices.switch_entries`)
    once per option, with what the resubmit posts of the part's entity groups
    (`compiler.sources.block_wire_groups`), through data-mcp's
    `resolve_definition`, which stores the definition it answers. One
    resubmit carries every option, so an input moving on the measure basis
    and the provider scope is compiled once. The answer is held as a
    `planner.plan.AlignedPin` with the answer's digest
    (`harness.tool_contracts.resolver_digest`), its block marking those
    axes as the planner's, beside any axis the origin's block marks
    (`harness.scope_choices.mark_planner_axes`). The datasets the planner was
    shown (`planner.plan.pin_sources`) are left as shown, so a requirement
    pinning the answer's spec_id directly takes none of the origin's
    `planner.plan.PinSource` fields (settled, warnings, asked roles,
    provenance ref). Where it was not shown that spec_id it reads the
    compile's block (`planner.plan.held_sources`), choice provenance and
    planner marks included. The next proposal pins the input to it
    (`submission.service.admit`).

    The compile takes the time a resubmit has
    (`translator.discovery.call_within_entry`): the capability row's call
    timeout, above data-mcp's resubmit budget, within the planner entry's
    time left.

    `state` is read; the writes are made when the read returned is applied
    (`SourceRead`). It answers `None` once the pin is held, or the refusal
    the proposal answers: `SURFACE_UNAVAILABLE` when Search did not answer
    and can later, `OPERATION_UNSUPPORTED` when it refuses the options or
    answers with no one dataset, or when what it posts of the part's entity
    groups is not readable, since a resubmit posted without them can bind
    another population.

    A declared whole (`Realignment.whole_axis`) is compiled by `compile_whole`.
    """
    if realignment.whole_axis is not None:
        return await compile_whole(surface, realignment, state=state)
    source = held_sources(state).get(realignment.spec_id)
    block = source.spec_block(realignment.spec_id) if source is not None else None
    scope = part_scope(realignment.spec_id, block)
    choices: dict[str, ScopeChoice] = (
        {choice.axis: choice for choice in scope.choices} if scope is not None else {}
    )
    axes = [option_axis(option_id) for option_id in realignment.option_ids]

    def unservable(reason: str) -> SourceRead:
        return lambda into: basis_unservable(
            spec_id=realignment.spec_id,
            option_ids=realignment.option_ids,
            requirement_id=realignment.requirement_id,
            step_id=realignment.step_id,
            reason=reason,
            record=lambda operation: record_unsupported(into, operation),
            proposer=PROPOSE_PLAN_PROPOSER,
        )

    missing = [axis for axis in axes if axis not in choices]
    if block is None or scope is None or missing:
        return unservable(
            f"{realignment.requirement_id!r}'s dataset has no "
            f"{' or '.join(missing or axes)} choice on record"
        )
    switched = scope.entries
    for option_id in realignment.option_ids:
        switched = tuple(
            switch_entries(
                scope.model_copy(update={"entries": switched}),
                choice=choices[option_axis(option_id)],
                option_id=option_id,
            )
        )
    entries = [entry.model_dump(mode="json", exclude_none=True) for entry in switched]
    groups = readable_wire_groups(block)
    if groups is None:
        return unservable(
            f"Search's named sets for {realignment.requirement_id!r}'s dataset are not readable"
        )
    try:
        part = await resolve_definition(
            surface,
            spec_id=scope.base.spec_id,
            identity=scope.base.identity,
            groups=groups,
            scope_entries=entries,
        )
    except DefinitionNotResolvedError as exc:
        if exc.retryable:
            return _refusing(_basis_uncompiled(realignment, str(exc)))
        return unservable(
            f"Search does not count {realignment.requirement_id!r}'s dataset the way its "
            f"calculation's other inputs share ({exc.code or 'refused'})"
        )
    spec = part.get("spec") if part.get("status") == "definite" else None
    spec_id = spec.get("spec_id") if isinstance(spec, Mapping) else None
    if not isinstance(spec, Mapping) or not isinstance(spec_id, str):
        return unservable(
            f"Search holds no one dataset counting {realignment.requirement_id!r} the way "
            "its calculation's other inputs share"
        )
    digest = resolver_digest({"parts": [dict(part)]})
    mark_planner_axes(digest, planner_axes(block) | set(axes))

    def hold(into: SessionState) -> None:
        save_aligned_pin(
            into,
            AlignedPin(
                origin=realignment.spec_id,
                option_ids=realignment.option_ids,
                spec_id=spec_id,
                digest=digest,
            ),
        )

    return hold


async def compile_whole(  # noqa: PLR0911 -- one distinct refusal per way the compile fails
    surface: ToolSurface,
    realignment: Realignment,
    *,
    state: SessionState,
) -> SourceRead:
    """Compile the declared whole `realignment` names from its part, and hold it as a pin.

    The part is the pin `realignment.spec_id` and the axis left open
    `realignment.whole_axis` (`planner.plan.WholeOf.axis`). Posts the part's
    base identity with that axis opened, under the spec_id that identity
    hashes to, the part's entity groups, and one explicit entry from each of
    its other choices holding its active option, with the concept of the
    part's entry on that axis (`_opened`). The sector is opened by
    unbinding the slot the part's sector choice binds
    (`submission.service.axis_slot`). A donor or recipient is opened by
    dropping that role's column from the identity
    (`submission.service.WHOLE_COLUMNS`), with the set named on it and the
    choice adding entities to it, the region, so Search answers with the
    total over every party there that holds the part, and what population
    that is stays Search's answer. Search keeps the part's measure basis,
    and its region on another role, which an option the whole cannot serve
    would otherwise leave to Search's default. The provider scope of a
    donor whole is posted as a carried entry: Search counts the whole on it
    where the whole offers it, and otherwise drops it and counts the whole
    on its own default, since the scope names whose spending a named donor
    counts and Search offers the inclusive scope on no total naming no
    donor. The part keeps its own scope, and the whole's caveat says what
    each side counts (`submission.service._whole_shares`). A recipient
    whole names the part's donor, so Search can count it on the part's
    scope, and its provider scope is posted as an explicit entry that
    Search refuses where it cannot hold it, as the other choices are.
    No model call is made. The part is read off the blocks on record
    (`planner.plan.held_sources`), so a part alignment moved reads its
    compile. The whole takes its choices
    from its part. Each the whole holds on the part's option carries the
    part's provenance, and the block marks as the planner's
    (`harness.scope_choices.mark_planner_axes`) those of them the part's
    block marks, so the whole reads as settled the way the part was and a
    later search carries what the part's carries. It is held as a
    `planner.plan.AlignedPin` from the part on that axis, which every
    whole declaring the part and the axis pins. Like a compile from
    `compile_on_basis`, it is held there alone.

    `state` is read; the writes are made when the read returned is applied
    (`SourceRead`). It answers `None` once the pin is held,
    `SURFACE_UNAVAILABLE` when Search did not answer and can later, or
    `SHARE_WHOLE_UNDERIVABLE` under a cause
    naming why: the part's pin is not among the datasets shown
    (`part_not_shown`), is shown with no block (`part_block_missing`), holds
    scope choices this release cannot read (`scope_unreadable`), binds
    nothing on the axis to leave open (`axis_not_held`), or names a set on a
    role the whole keeps whose posted identity (`compiler.sources.block_wire_groups`)
    is not readable, which the whole would otherwise be posted without
    (`groups_unreadable`); or Search refuses
    the whole (as `requested_option_unavailable` where it cannot hold an
    option the part holds), or answers with no one dataset (`whole_refused`),
    or with a donor or recipient whole that Search shows does not hold a party
    the part names on that role (`part_outside_whole`), or bound to one total
    that states neither its members nor a whole class of party, so nothing
    shows it does (`whole_members_unstated`) (`_parties_outside`).
    """
    part_pin, axis = realignment.spec_id, cast(str, realignment.whole_axis)

    def underivable(reason: str, cause: str = "whole_refused") -> SourceRead:
        return _refusing(
            share_whole_underivable(
                whole_id=realignment.requirement_id,
                part_spec_id=part_pin,
                reason=reason,
                cause=cause,
            )
        )

    source = held_sources(state).get(part_pin)
    block = source.spec_block(part_pin) if source is not None else None
    scope = part_scope(part_pin, block)
    if source is None or block is None or scope is None or not whole_axis_held(scope, axis):
        return underivable(*unread_part(part_pin, axis, source=source, block=block, scope=scope))
    column = WHOLE_COLUMNS.get(axis)
    groups = readable_wire_groups(block, leaving_out=column)
    if groups is None:
        return underivable(
            f"Search's named sets for {part_pin} are not readable", "groups_unreadable"
        )
    identity, choices = _opened(scope, axis)
    concepts = {entry.axis: entry.concept for entry in scope.entries}
    entries = [
        ScopeEntry(
            axis=choice.axis,
            option_id=choice.active,
            provenance="carried"
            if axis == "donor" and choice.axis == PROVIDER_AXIS
            else "explicit",
            concept=concepts.get(choice.axis),
            origin=ChoiceOrigin(choice_id=choice.id),
        )
        for choice in choices
    ]
    try:
        part = await resolve_definition(
            surface,
            spec_id=spec_id_for(canonicalize(identity)),
            identity=identity,
            groups=groups,
            scope_entries=[entry.model_dump(mode="json", exclude_none=True) for entry in entries],
        )
    except DefinitionNotResolvedError as exc:
        if exc.retryable:
            return _refusing(_basis_uncompiled(realignment, str(exc)))
        return underivable(
            f"Search does not count {part_pin}'s dataset over every {axis} the way the part is "
            f"counted ({exc.code or 'refused'}): {exc}"
        )
    spec = part.get("spec") if part.get("status") == "definite" else None
    spec_id = spec.get("spec_id") if isinstance(spec, Mapping) else None
    if not isinstance(spec, Mapping) or not isinstance(spec_id, str):
        return underivable(f"Search holds no one dataset counting every {axis} of {part_pin}")
    held = (
        _parties_outside(scope, spec, column, part_block=block)
        if column is not None
        else _Containment()
    )
    if held.outside or held.unshown:
        return underivable(*_uncontained(part_pin, axis, held))
    answered = _answered_options(spec_id, spec)
    inherited: dict[str, str] = {
        choice.axis: choice.provenance
        for choice in choices
        if answered.get(choice.axis) == choice.active
    }
    digest = resolver_digest({"parts": [_as_whole(part, inherited, entries, answered)]})
    mark_planner_axes(digest, planner_axes(block) & inherited.keys())

    def hold(into: SessionState) -> None:
        save_aligned_pin(
            into,
            AlignedPin(
                origin=part_pin, option_ids=(), spec_id=spec_id, digest=digest, whole_axis=axis
            ),
        )

    return hold


async def run_together[T](calls: Iterable[Awaitable[T]]) -> list[T]:
    """Each of `calls`' results in order, run at once.

    The first to raise cancels the rest, which are waited for before it is
    raised, so none is left running.
    """
    tasks = [asyncio.ensure_future(call) for call in calls]
    try:
        return list(await asyncio.gather(*tasks))
    finally:
        await _cancel_and_drain(tasks)


async def _cancel_and_drain(tasks: Sequence[asyncio.Future[Any]]) -> None:
    """Cancel each of `tasks` still running, wait for every one, and retrieve each
    exception, so none is reported as never retrieved."""
    running = [task for task in tasks if not task.done()]
    for task in running:
        task.cancel()
    if running:
        await asyncio.wait(running)
    for task in tasks:
        if not task.cancelled():
            task.exception()


def _opened(scope: PartScope, axis: str) -> tuple[dict[str, Any], list[ScopeChoice]]:
    """The identity and scope choices the whole of the part `scope` records over
    every member of `axis` posts (`compile_whole`).

    On the sector, the base identity with the slot the sector choice binds
    unbound (`axis_slot`), and every choice but the sector. On a donor or
    recipient, the base identity without that role's column
    (`WHOLE_COLUMNS`), and every choice but one adding entities to that role,
    the region's. The whole posts the part's groups on other roles
    (`compiler.sources.block_wire_groups`).
    """
    identity = copy.deepcopy(scope.base.identity)
    column = WHOLE_COLUMNS.get(axis)
    if column is None:
        slot = axis_slot(scope, axis)
        for held in identity.get("slots") or ():
            if held.get("property") == slot:
                held["binding"] = {"state": "unbound"}
        return identity, [choice for choice in scope.choices if choice.axis != axis]
    identity["entities"] = {
        role: refs for role, refs in (identity.get("entities") or {}).items() if role != column
    }
    return identity, [
        choice
        for choice in scope.choices
        if not any(
            isinstance(option.binding, EntitiesBinding) and option.binding.role == column
            for option in choice.options
        )
    ]


def _answered_options(spec_id: str, spec: Mapping[str, Any]) -> dict[str, str]:
    """The option each scope choice of `spec`, the Search spec block of `spec_id`,
    holds active, by axis (`compile_whole`). Empty where it records no choices this
    release reads (`submission.service.part_scope`)."""
    scope = part_scope(spec_id, spec)
    return {choice.axis: choice.active for choice in scope.choices} if scope is not None else {}


class _Containment(NamedTuple):
    """The parties a part names on a donor or recipient whole's role that Search's
    answer shows are outside the whole (`outside`), and those it does not show are
    inside or outside (`unshown`) (`_parties_outside`)."""

    outside: tuple[str, ...] = ()
    unshown: tuple[str, ...] = ()


def _uncontained(part_pin: str, axis: str, held: _Containment) -> tuple[str, str]:
    """Why the whole over every `axis` of the part pinned `part_pin` does not contain
    it, `held` its containment (`_parties_outside`), as a clause and its
    `share_whole_underivable` cause: `part_outside_whole` where Search shows a
    party the part names is outside, else `whole_members_unstated`."""
    if held.outside:
        return (
            f"Search counts {part_pin}'s dataset over a {axis} total that does not hold "
            f"{', '.join(held.outside)}, which the part names, so the total would not "
            "contain it",
            "part_outside_whole",
        )
    return (
        f"Search counts {part_pin}'s dataset over a {axis} total that lists no members "
        f"and no class of party holding {', '.join(held.unshown)}, which the part "
        "names, so nothing shows the total holds it",
        "whole_members_unstated",
    )


# Search's cause for an entry dropped because the part has no choice on its axis.
_AXIS_ABSENT = "axis_absent"

# The class of party a total holding a whole population can leave out and still
# hold every party the part names: a residual names no party (`common.place_type`).
_NOT_A_PARTY_CLASS = "residual"


def _parties_outside(
    scope: PartScope, spec: Mapping[str, Any], column: str, *, part_block: Mapping[str, Any]
) -> _Containment:
    """Whether the whole Search answered with, `spec`, holds each party the part
    `scope` records names on the role `column` (`compile_whole`), `part_block`
    the part's spec block.

    A whole binding nothing on `column` counts every party its dataset holds
    there, the part's own dataset, so it holds every party the part names.
    One binding parties there holds those it binds, and of a total it binds,
    the members Search lists for the group naming it on `column`
    (`ResolvedGroup.members`, DAC countries' total). A total holding a whole
    population (`ResolvedGroup.whole_of`) holds every party of its classes and
    no other: the part's party where its ref shows it is of one
    (`compiler.sources.party_class`, a country official), and not where its
    ref shows another. It holds a party whose ref shows no class where its
    classes are every class the side publishes (`ResolvedGroup.side_classes`),
    residuals aside; where the side publishes another class, or Search states
    no classes for the side but residuals, nothing here says which class that
    party is, so it is unshown.
    Any other party is outside where Search binds parties or lists a total's
    members there, and unshown where the total it binds states neither its
    members nor a whole class. Where Search's group on `column` is not
    readable (`compiler.sources.block_group`), as one naming a class of party
    outside `PartyClass`, every party it does not bind is unshown; a group on
    another role is not read. The parties the part and the whole bind on
    `column` are those their scope records bind
    (`submission.service.bound_parties`), a choice's among them; where the
    whole's record does not read, those its definition binds
    (`submission.service.block_parties`).

    A party of the part the whole neither binds nor lists is read through the
    part's own group on `column` where it is that group's total (the G7's): the
    whole holds it where it holds every member Search lists for the group, does
    not where one is outside, and nothing shows it does where Search lists no
    members, or the part's group there is not readable.
    """
    named = bound_parties(scope, column)
    bound = block_parties(spec, column, scope=part_scope(str(spec.get("spec_id")), spec))
    if not bound:
        return _Containment()
    try:
        group = block_group(spec, column)
    except ValueError:
        return _Containment(unshown=tuple(party for party in named if party not in bound))
    members = set(group.members) if group is not None else set()

    def holds(party: str) -> bool | None:
        """Whether the whole holds `party` as itself, `None` where nothing shows it."""
        if party in bound or party in members:
            return True
        if group is not None and group.whole_of:
            if (held := party_class(party)) is not None:
                return held in group.whole_of
            classes = set(group.side_classes) - {_NOT_A_PARTY_CLASS}
            return True if classes and classes <= set(group.whole_of) else None
        if members or group is None or group.total not in bound:
            return False
        return None

    try:
        part_group = block_group(part_block, column)
    except ValueError:
        part_group, part_readable = None, False
    else:
        part_readable = True
    outside: list[str] = []
    unshown: list[str] = []
    for party in named:
        held = holds(party)
        if held:
            continue
        if not part_readable:
            # The party may be the total of a group nothing here reads.
            held = None
        elif part_group is not None and party == part_group.total:
            each = [holds(member) for member in part_group.members]
            if not each:
                held = None
            elif all(each):
                continue
            else:
                held = False if False in each else None
        (outside if held is False else unshown).append(party)
    return _Containment(outside=tuple(outside), unshown=tuple(unshown))


def _as_whole(
    part: Mapping[str, Any],
    provenance: Mapping[str, str],
    posted: Iterable[ScopeEntry],
    answered: Mapping[str, str],
) -> dict[str, Any]:
    """A copy of `part`, the `definite` Search part answering a whole, whose scope
    choice on each axis of `provenance` carries the provenance given for that axis,
    and whose dropped entries leave out each carried entry of `posted`, the entries
    the whole was posted with, that changed nothing (`compile_whole`). `answered` is
    the option the whole holds on each axis (`_answered_options`).

    A carried entry the harness posted changed nothing where the whole holds its
    option all the same, or where Search dropped it because the whole has no
    choice on its axis (`axis_absent`), as a DAC2a total naming no donor has no
    provider axis, so the user chose nothing there that failed to apply. One Search
    dropped for a whole that counts another option, as DAC countries' total ODA
    beside a part on bilateral ODA, stays, so the whole's reading says it did not
    apply. The whole's caveat says what each side counts
    (`submission.service._whole_shares`). Any other dropped entry stays."""
    copied = copy.deepcopy(dict(part))
    resolved = copied["spec"].get("resolved")
    scope = resolved.get("scope_choices") if isinstance(resolved, dict) else None
    if not isinstance(scope, dict):
        return copied
    for choice in scope.get("choices") or ():
        if isinstance(choice, dict) and choice.get("axis") in provenance:
            choice["provenance"] = provenance[choice["axis"]]
    carried = {
        (entry.axis, entry.option_id, entry.origin.choice_id)
        for entry in posted
        if entry.provenance == "carried" and isinstance(entry.origin, ChoiceOrigin)
    }
    if carried and isinstance(scope.get("dropped_entries"), list):
        scope["dropped_entries"] = [
            dropped
            for dropped in scope["dropped_entries"]
            if not _changed_nothing(dropped, carried, answered)
        ]
    return copied


def _changed_nothing(
    dropped: object, carried: Collection[tuple[str, str, str]], answered: Mapping[str, str]
) -> bool:
    """Whether `dropped`, one of Search's dropped entries, is a carried entry the
    harness posted, `carried` keying each by axis, option and choice, that changed
    nothing because the whole holds its option on its axis (`answered`) or has no
    choice there (`_as_whole`)."""
    if not isinstance(dropped, Mapping):
        return False
    entry = dropped.get("entry")
    if not isinstance(entry, Mapping) or entry.get("provenance") != "carried":
        return False
    origin = entry.get("origin")
    choice_id = origin.get("choice_id") if isinstance(origin, Mapping) else None
    axis, option_id = entry.get("axis"), entry.get("option_id")
    if (axis, option_id, choice_id) not in carried:
        return False
    return dropped.get("cause") == _AXIS_ABSENT or answered.get(str(axis)) == option_id


async def pin_by_search(surface: ToolSurface, unpinned: Unpinned) -> SourceRead:
    """Search for `unpinned`, a calculation's input the planner pinned no dataset for, and hold what Search settles on.

    Posts the `search` part the translator's own discovery sends for the
    requirement (`translator.discovery.search_part`), in the planner's words
    and with no message of the user's, so every axis an entry settled on
    Search's answer is marked as the planner's
    (`harness.scope_choices.mark_planner_axes`). A `definite` answer is
    shown to the planner as a source of its own
    (`planner.plan.save_pin_sources`) and held as a
    `submission.service.SearchedPin` with its digest, so the next proposal
    pins it and the planner seam stores it as a reading of the conversation
    (`harness.planner_seam.store_searched_pin_readings`). Any other answer is
    held as a search that settled on nothing, under Search's part status, or
    `refused` for a refusal, and the next proposal is refused
    (`submission.service._aligned_bases`). The search takes the time the
    translator's own discovery gives one
    (`translator.discovery.call_within_entry`), and is made once an
    invocation (`translator.discovery.answered_once`), so a read the
    proposal's other refused search discarded (`with_source_reads`) is not
    repeated.

    The holds are made when the read returned is applied (`SourceRead`). It
    answers `None` once the search is held, or `SURFACE_UNAVAILABLE` when
    Search did not answer, or refused in a way that can succeed later
    (`translator.discovery.refusal_retryable`); nothing is held then, so the
    next proposal searches again.
    """
    arguments = {"parts": [unpinned.part]}
    try:
        answer = await answered_once(
            SEARCH,
            arguments,
            lambda: call_within_entry(surface, tool=SEARCH, arguments=arguments),
        )
    except ToolSurfaceUnavailableError as exc:
        return _refusing(_search_unavailable(unpinned, str(exc)))
    if answer.is_error and refusal_retryable(answer):
        refusal = answer.structured_content or {}
        cause = refusal.get("message") or answer.text or "Search refused it for now"
        return _refusing(_search_unavailable(unpinned, str(cause)))
    envelope = None if answer.is_error else (answer.meta or {}).get(SEARCH_ENVELOPE_META_KEY)
    parts = envelope.get("parts") if isinstance(envelope, dict) else None
    part = parts[0] if isinstance(parts, list) and len(parts) == 1 else None
    status = part.get("status") if isinstance(part, dict) else None
    spec = part.get("spec") if isinstance(part, dict) and status == "definite" else None
    spec_id = spec.get("spec_id") if isinstance(spec, dict) else None
    searched = SearchedPin(
        requirement_key=unpinned.requirement_key,
        status="refused"
        if answer.is_error
        else status
        if isinstance(status, str)
        else "unreadable",
        asked_roles=unpinned.asked_roles,
        open_roles=unpinned.open_roles,
    )
    if not isinstance(spec_id, str) or not isinstance(envelope, dict):
        return lambda into: save_searched_pin(into, searched)
    digest = resolver_digest(cast(dict[str, Any], envelope))
    scope = part_scope(spec_id, digest["spec_block"])
    mark_planner_axes(
        digest,
        (choice.axis for choice in scope.choices if choice.provenance != "default")
        if scope is not None
        else (),
    )
    shown = PinSource(block=digest["spec_block"], settled=True, asked_roles=unpinned.asked_roles)

    def hold(into: SessionState) -> None:
        save_pin_sources(into, {**pin_sources(into), spec_id: shown})
        save_searched_pin(into, searched.model_copy(update={"spec_id": spec_id, "digest": digest}))

    return hold


def _search_unavailable(unpinned: Unpinned, cause: str) -> dict[str, object]:
    """Return the `SURFACE_UNAVAILABLE` problem for a search of `unpinned` that can succeed later.

    The same proposal can succeed later, so nothing in it needs changing.
    """
    return problem(
        ProblemCode.SURFACE_UNAVAILABLE,
        f"{unpinned.requirement_id!r}, an input of {unpinned.step_id!r}, pins no dataset, "
        f"and Search could not be asked for one: {cause}. The plan is sound and nothing was "
        "recorded; propose it again.",
        outcome=Outcome.REFUSED,
        detail={"search": unpinned.model_dump(mode="json")},
    )


def _basis_uncompiled(realignment: Realignment, cause: str) -> dict[str, object]:
    """Return the `SURFACE_UNAVAILABLE` problem for a compile on another basis or provider scope that did not complete.

    The same proposal can succeed later, so nothing in it needs changing.
    """
    return problem(
        ProblemCode.SURFACE_UNAVAILABLE,
        f"{realignment.requirement_id!r} could not be counted the way the other inputs of "
        f"{realignment.step_id!r} are: {cause}. The plan is sound and nothing was recorded; "
        "propose it again.",
        outcome=Outcome.REFUSED,
        detail={"basis": realignment.detail()},
    )


async def with_source_reads(
    propose: Callable[[], dict[str, object]], *, state: SessionState, surface: ToolSurface
) -> dict[str, object]:
    """`propose`'s answer, once every read of a source it was refused for is made.

    A refusal records nothing, so `propose` runs again once the read is held.
    Two refusals ask for one. A plan rolling from a dataset's first year
    needs that first year (`read_full_history`, held by
    `planner.plan.save_full_history`): one read serves every requirement, and
    every later proposal of the turn, that pins the dataset; an answer
    stating no first year is held too, and the next refusal says so
    (`first_year_read_failed`). A calculation one of whose inputs pins no
    dataset while another carries a basis choice needs that input searched
    for and pinned (`BASIS_UNKNOWN`, `pin_by_search`), as does the part of
    a declared whole; a search held
    for the requirement's content is made once. A calculation whose inputs are
    counted on different measure bases or provider scopes needs each input
    it names compiled on their shared options, every option of one input in
    one compile (`BASES_UNALIGNED`, `compile_on_basis`), and a declared
    whole needs its part compiled with the axis left open (`compile_whole`);
    a pin already compiled on a set of options, or a whole already compiled,
    is asked for once, so a refusal naming one
    is answered as it stands, and inputs sharing a pin and a set of options
    compile it once (`_alignment_key`).

    The reads one refusal asks for run at once. Their writes are made in the
    order the refusal names them, up to the first read refused, whose
    refusal is answered as soon as it and every read before it are made,
    and the reads after it are cancelled (`_held_in_order`), so the state is
    the one reading them one after another leaves. A search the refused read's writes discarded keeps
    its answer for the invocation (`pin_by_search`), so the next proposal
    searches only for what was refused.
    """
    while True:
        result = propose()
        pin = _first_year_wanted(result)
        if pin is not None:
            if pin in full_histories(state):
                return result
            identity = _first_year_identity(pin, state)
            if identity is None:
                return result
            try:
                history = await read_full_history(surface, identity)
            except DefinitionNotResolvedError as exc:
                return _first_year_unreadable(result, exc)
            save_full_history(state, pin, history)
            continue
        unpinned = unpinned_wanted(result)
        if unpinned:
            refused = await _held_in_order(
                (pin_by_search(surface, item) for item in unpinned), state=state
            )
            if refused is not None:
                return refused
            continue
        wanted = realignments_wanted(result)
        done = {
            _alignment_key(pin.origin, pin.option_ids, pin.whole_axis)
            for pin in aligned_pins(state)
        }
        keys = [_alignment_key(item.spec_id, item.option_ids, item.whole_axis) for item in wanted]
        if not wanted or any(key in done for key in keys):
            return result
        compiles: dict[tuple[str, tuple[str, ...], str | None], Realignment] = {}
        for key, item in zip(keys, wanted, strict=True):
            compiles.setdefault(key, item)
        refused = await _held_in_order(
            (compile_on_basis(surface, item, state=state) for item in compiles.values()),
            state=state,
        )
        if refused is not None:
            return refused


async def _held_in_order(
    reads: Iterable[Awaitable[SourceRead]], *, state: SessionState
) -> dict[str, object] | None:
    """Run `reads` at once and apply each to `state` in order, answering the first refused.

    Each read is applied once it and every read before it are made, so a
    refusal is answered without waiting for the reads after it, which are
    cancelled. A read that raises raises once every read before it is
    applied without a refusal, and cancels the reads after it. Every read
    takes `state` before its first wait, so none reads what another wrote.
    """
    tasks = [asyncio.ensure_future(read) for read in reads]
    try:
        for task in tasks:
            refused = (await task)(state)
            if refused is not None:
                return refused
        return None
    finally:
        await _cancel_and_drain(tasks)


def _alignment_key(
    spec_id: str, option_ids: Iterable[str], whole_axis: str | None
) -> tuple[str, tuple[str, ...], str | None]:
    """The compile that puts the pin `spec_id` on the options `option_ids`, or leaves its choice on `whole_axis` open, as a key.

    The options are sorted and deduplicated, so one compile naming the same
    options in another order is the same work.
    """
    return spec_id, tuple(sorted(set(option_ids))), whole_axis


def _entity_ask_unbound(referral: Referral | None, candidates: object) -> dict[str, object] | None:
    """The refusal for an ask offering anything but one reading's picks under an entity referral, or `None`.

    While an `entity_name_ambiguous` referral is pending, an ask that offers
    candidates offers those of one reading in its `ambiguous_entities`, each
    under its `pick` (`planner.records.entity_pick`), and offers all of them.
    Studio shows an ask as one single-select question, so one answer settles
    one entity, an id that names no pick would come back as an answer nothing
    reads, and an ask leaving a candidate out would make the planner's choice
    the user's. An ask with no candidates, as for free text or a combination,
    passes.
    """
    if referral is None or referral.kind is not ReferralKind.ENTITY_NAME_AMBIGUOUS:
        return None
    offered = _candidate_ids(candidates)
    readings = {
        candidate.pick: reading.entity
        for reading in referral.ambiguous_entities
        for candidate in reading.candidates
    }
    unknown = sorted(offered - readings.keys())
    if unknown:
        return problem(
            ProblemCode.INVALID_ARGUMENTS,
            f"spec_id {unknown[0]!r} is no candidate's `pick` in the pending referral's "
            "`ambiguous_entities`. Offer one reading's candidates, each under its `pick` "
            "unchanged.",
            outcome=Outcome.INVALID_INPUT,
        )
    touched = {readings[spec_id] for spec_id in offered}
    if len(touched) > 1:
        return problem(
            ProblemCode.INVALID_ARGUMENTS,
            "These candidates come from more than one reading in `ambiguous_entities`. Ask "
            "about each reading in an `adk_request_input` call of its own.",
            outcome=Outcome.INVALID_INPUT,
        )
    if touched and offered != {pick for pick, entity in readings.items() if entity in touched}:
        return problem(
            ProblemCode.INVALID_ARGUMENTS,
            "These candidates leave out some of their reading's candidates in "
            "`ambiguous_entities`. Offer every candidate of the reading, each under its `pick`.",
            outcome=Outcome.INVALID_INPUT,
        )
    return None


class _AskTool(LongRunningFunctionTool):
    """The planner's ask, refusing a call that would not bind its answer to one entity reading.

    The function and so the declaration the model reads are `ask_user`'s. The
    referral pending in the call's session decides what the ask may offer
    (`_entity_ask_unbound`), which the function, holding no `ToolContext`,
    cannot read.
    """

    @override
    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        referral = load_referral(tool_context.state)
        plan = load_plan(tool_context.state)
        if referral is not None and plan is not None and plan.rolling_copies:
            referral = Referral.model_validate(
                authored_answer(referral.model_dump(mode="json"), split_of(plan))
            )
        refusal = _entity_ask_unbound(referral, args.get("candidates"))
        if refusal is not None:
            return refusal
        return await super().run_async(args=args, tool_context=tool_context)


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
            options: The `options` a refusal's action carries, unchanged.
                Omit it otherwise.
            recommended_spec_id: The candidate you favour, when you favour
                one -- must name one of `candidates`.
            rationale: Why you favour it, for the user to read, when
                `recommended_spec_id` is given.
        """
        del message, rationale  # carried through to the wire frame, not read back here
        fixed = ([dict(o) for o in COMBINATION_OPTIONS], [dict(o) for o in PRICES_OPTIONS])
        if options and (candidates or options not in fixed):
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
        """`propose_plan`, making each source read it refuses for want of (`with_source_reads`).

        A refusal records nothing, so the proposal is made again once the
        read is held: a dataset's first year, or a calculation's input
        compiled on the basis its other inputs share.
        """
        assert surface is not None
        return await with_source_reads(
            lambda: propose_plan(**arguments),
            state=arguments["tool_context"].state,
            surface=surface,
        )

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
        _AskTool(ask_user),
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
        # `ctx.run_node`), ADK runs an agent with no explicit `mode` as
        # `single_turn`. `'none'` would build every request from
        # `_get_current_turn_contents`, which slices the session from the
        # latest turn-starting event. `'default'` uses `_get_contents`, which
        # reads the turn's question and every event on this run's node path,
        # so `prepend_brief` finds the question to follow.
        include_contents="default",
        # `AGENT_THINKING_CONFIG` (see `planner.settings`) pins every
        # generation to LOW thinking.
        planner=BuiltInPlanner(thinking_config=AGENT_THINKING_CONFIG),
        # Drops this generation's thought-summary parts once the trace has
        # them, so `include_contents="default"` above never replays one into
        # a later request of the same turn. See `drop_thought_parts`.
        after_model_callback=drop_thought_parts,
        before_model_callback=prepend_brief,
    )
