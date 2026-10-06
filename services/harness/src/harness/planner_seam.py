"""The graph's planner seam, backed by the planner agent.

An adapter over `planner.agent.create_planner_agent`, built with its
`translate` tool by `harness.agents.create_planner_with_translator`: the
planner runs as a dynamically-scheduled child of the graph's own node,
through `Context.run_node`, and this module reads what it leaves in session
state and translates it into `harness.graph.PlannerModel`'s contract.

The planner's own tools settle methodology (`propose_plan`), carry it to a
compiled workflow (`translate`, whose result is either the compiled outcome
or the `Referral` translation stopped on) and declare it ready for the
user's answer (`submit_plan`). This module never constructs a `Plan` itself.

`ctx.run_node` shares one session both directions with the graph's own
node: there is no second `Runner`, no second session service, and no LRU or
lease bookkeeping to keep one bounded, because there is only the one
session the graph already owns and already persists. `ctx.state["plan"]`
is the graph's own authoritative record; `planner.plan.save_plan`/
`load_plan` read and write a second, namespaced key
(`planner.plan.PLAN_STATE_KEY`, `"planner:plan"`) in that same `ctx.state`
for the planner's own drafting state -- one store, two keys, not two stores
kept in step. `create_planner_agent` also pins `include_contents="default"`
so every planner event already in a session is visible to the next call
before its own content is added.

Every call renders a message shaped by whether this session has already
been briefed: a session-scoped `_BRIEFED_STATE_KEY` boolean, written
through `ctx.session.state`, flips true the first time this seam sends
anything in a session. Session ids are minted fresh per `/turn`
(`studio_state.ConversationState.session_id`), so a flag that is merely
absent by default is automatically correct on a new turn with no reset
logic there -- a studio session never needs the clear described next.
`evals/agents/dfo_analyst_graph` (`just adk-web`) instead keeps one
session across many turns, so `__call__` itself clears this flag, along
with `planner.plan.PLAN_STATE_KEY`, on a genuine first-draft entry
(`prior_plan`, `refusal` and `amendment` all `None`, the shape the
graph's planner node passes only on the `choose_depth` edge): without it,
a later turn's own first-draft entry
would read as a continuing call on a session that already holds an
earlier turn's plan, get only the continuing message instead of the full
brief, and be pushed to amend that stale plan or have `starts_new_plan`
refused as answering a different question. `_persist_durable` (see
`__call__`) is what makes both this flag's flip and its clear survive a
pause-and-resume across `adk_request_input`: a plain `ctx.session.state`
write is visible in-process but never reaches a session event's
`state_delta`, so a fresh `Runner.run_async` call after a resume would
otherwise read a write back as it was before this call, undoing it. The
first call of a session (or a session's first entry after such a clear)
renders the full brief --
envelope, canvas digest, conversation digest, the plan being changed, the
question. Every later call in that same session -- a correction round
inside `_propose`'s own loop, the `amend` edge (`approve_n -> planner_n`),
and the `replan` edge (`execute_n -> planner_n`) -- renders only what is
new: the amendment or refusal ask, or the correction's problem nudge.
Resending the full brief on every call would restate a history the model
can already see with `include_contents` pinned; a continuing call sends
only the new thing.

`prior_plan` is dropped from every continuing render too: on the `amend`
edge it is the plan the model itself drafted and submitted last call,
untouched since (`approve_n`'s own `"amend"` route never calls
`Plan.approve`), already in its own tool-call history. On the `replan`
edge it is `execute`'s own copy of that same plan (`graph.py:1190`),
which by then differs from what the model drafted in exactly two fields:
`status` and `approved_revision`. Both come from upstream of `execute`
itself -- `approve_n`'s `"approved"` route calls `Plan.approve()`
(`planner/plan.py:1277-1292`), which sets both before `execute_approved_plan`
ever runs; `execute_approved_plan` then only ever advances `status` further,
through `start_execution`, `complete`/`fail`, each a bare
`model_copy(update={"status": ...})` that touches nothing else. Dropping
both from the continuing render is still right: `status` states where the
plan sits in the approve/execute lifecycle, which the refusal text this
seam sends already says in its own words; `approved_revision` is
`plan_revision(self)`, a content hash of the very decisions and
requirements the model itself just drafted, not a new fact about the
world. Neither is something the refusal text does not already restate.

`_plan_context` and `_render_envelope` render the brief's two largest
sections as digests. With a realistic nine-requirement plan, compiled, and a
`candidates` envelope at every cap, the plan section is 8,438 characters where
its raw JSON is 22,609, and the envelope is 2,357 where its raw JSON is 3,326.
The whole brief renders 11,437 characters, and `test_context_budget.py` holds
it to a `SEAM_BRIEF_CEILING` of 14,300, 25% above that. These are character
counts of the brief alone.
"""

from __future__ import annotations

import contextvars
import json
import logging
import random
import time
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from common.telemetry import record_span
from compiler.surface import ToolSurface
from google.adk import Context, Event
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.genai import types
from planner.plan import (
    MAX_PLAN_ROUNDS,
    MAX_QUESTION_CHARS,
    PIN_SOURCES_STATE_KEY,
    PLAN_STATE_KEY,
    UNSUPPORTED_STATE_KEY,
    PinSource,
    Plan,
    PlanStatus,
    SessionState,
    UnsupportedOperation,
    clear_unsupported,
    current_discovery_time_left,
    cut_user_text,
    load_asks,
    load_plan,
    load_plan_rounds,
    pin_sources,
    record_unsupported,
    save_pin_sources,
    save_plan,
    save_plan_rounds,
    seed_of,
    seeded_sources,
)
from planner.settings import (
    THINKING_SAMPLE_STATE_KEY,
    AgentSettings,
    override_thinking_on_sampled_turns,
)
from planner.tool_results import Outcome, ProblemCode
from translator.discovery import stat_var_count_text
from translator.mechanical import COMPILE_FAILURES_STATE_KEY, clear_compile_failures
from translator.referral import ReferralKind
from translator.tool import TRANSLATE_TOOL_NAME

from harness.agents import create_planner_with_translator
from harness.graph import (
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPEC_WARNINGS_KEY,
    ORCHESTRATOR_SPECS_KEY,
    frame_row_ceiling,
)
from harness.prefix_cache import PrefixCache
from harness.settings import Settings
from harness.turn_records import (
    SearchReading,
    named_spec_ids,
    newest_per_spec_id,
    render_readings,
)
from harness.usage import log_usage
from harness.wire import MAX_SEEDS

logger = logging.getLogger(__name__)

# Corrective rounds after the first run. Past this we raise rather than
# keep paying for a model that is not converging on `submit_plan`.
_MAX_CORRECTIONS = 2

# Insurance against a runaway tool loop inside one agent run. A healthy plan
# takes well under this many tool calls (skill search, propose, translate,
# submit).
_MAX_EVENTS = 60

# Session-state slot marking that this seam has already sent a message in
# this session, so a later call knows to render a continuing message
# instead of the full brief. Namespaced alongside `planner.plan.
# PLAN_STATE_KEY`'s own key.
_BRIEFED_STATE_KEY = "planner:seam_briefed"

# The branch every planner run is stamped with, so ADK's own branch filter
# keeps the planner's events and another agent's apart. `ctx.run_node` is
# called without one otherwise, and an event carrying no branch belongs to
# every branch (`flows/llm_flows/contents._is_event_belongs_to_branch`), so
# the orchestrator, which runs on its own branch (`harness.graph`), would
# read the planner's calls and results back as fenced user text
# (`flows/llm_flows/_fencing._present_other_agent_message`).
#
# The name is a fixed root segment. Each `_propose` call is a separate
# dynamic run, and `use_sub_branch` would give each its own (`dfo_planner@1`,
# `dfo_planner@2`), excluding the planner's own earlier events from its later
# request and splitting a multi-round turn's history. A fixed branch keeps
# the planner's events on one path across every call. It also sits outside
# the graph's own names (`planner`, not `dfo_planner`), so anything matching
# on the branch can tell an event the planner node itself authors apart from
# a planner event.
_PLANNER_BRANCH = "dfo_planner"

_ENVELOPE_CONTEXT = """\
Context for this question. The search step has already run, so what is \
already known about the dataset is named below. A candidate a referral \
names for a requirement is an allowed source for that requirement -- pin \
it from its own discovery. Never invent a spec_id or a source that \
neither the envelope nor a referral has named.{reasons}

Resolve envelope:
{envelope}
"""

# The orchestrator engine's counterpart to `_ENVELOPE_CONTEXT`. No probe runs
# before its planner, so the conversation's search readings stand where the
# envelope does.
_READINGS_CONTEXT = """\
Context for this question. These are the readings of the searches this \
conversation ran, earlier turns' included, oldest first: each part's \
status and spec_id, what it resolved \
to, its window, the observations in its most recent years, its warnings \
(! marks one that dropped part of the ask) and what a fetch of it \
returned. A candidate a referral names for a requirement is an allowed \
source for that requirement -- pin it from its own discovery. Never invent \
a spec_id or a source that neither these readings nor a referral has named.

{readings}
"""

# The characters the search readings may take in the planner's first
# message. The largest recorded turn, four readings with whole resolved
# sentences, renders in 2,124 characters (`tests/fixtures/
# turn_records_recorded.json`, `kenya_sectors`), and each definite reading's
# recent-year counts add about 80 more. A turn of that size renders whole,
# and a longer one sheds sentences before it folds or drops any reading
# (`render_readings`).
_READINGS_BUDGET = 3_000

# Up to this many envelope candidates are named, and the rest are only
# counted. `graph_envelope`'s `candidates` branch caps its own list upstream,
# but a no_data part's nearest-real suggestions come back uncapped, and a
# list long enough to double the brief is worth a trailing count instead.
_ENVELOPE_CANDIDATE_CAP = 6

# A seeded envelope names at most `MAX_SEEDS` seeds, the Studio wire's own
# cap, and, per seed, this many operations, the number of verbs, and counts
# the rest.
_ENVELOPE_OPERATION_CAP = 7

# Each candidate's `label` is the resolve's own `rendered_sentence`, which
# can run to a paragraph. Capped so six of them cannot outweigh the question
# they were resolved for.
_ENVELOPE_LABEL_CAP = 120

# Each of the envelope's other fields renders at most this many characters: the
# ids, counts and flags `graph_envelope` builds from a search result, a
# candidate's `spec_id`, a warning's `severity` and `code`, and the list of
# entity roles. A real value is a few dozen characters at most. A longer one is
# cut.
_ENVELOPE_FIELD_CAP = 200

# Up to this many skills render, and the rest are counted, as data-mcp's search
# digest does. `harness.tool_contracts.graph_envelope` passes every skill in
# that digest's order, so the first five here are the digest's first five.
_ENVELOPE_SKILL_CAP = 5

# Each skill's description is cut here, the cut data-mcp's search digest makes.
# A description is written to be matched against, and its first sentence says
# what the skill covers.
_ENVELOPE_SKILL_DESCRIPTION_CAP = 160

# The envelope fields `_render_envelope` names. Anything else on the envelope
# renders under its own name, after these.
_ENVELOPE_KEYS = frozenset(
    {
        "status",
        "spec_id",
        "parts",
        "choice_points",
        "wide",
        "stat_var_shape_kind",
        "stat_var_item_count",
        "stat_var_member_count_exact",
        "entity_roles",
        "stat_var_served_count",
        "open_entity_axes",
        "candidates",
        "warnings",
        "skills",
        "seeds",
    }
)

# At most this many envelope warnings render one by one; the rest are
# counted. The resolve's own diagnostics ride here, and a degraded search
# carries more of them than are worth reading individually.
_ENVELOPE_WARNING_CAP = 5

# Each rendered warning message is cut here. A warning says why an ask was
# dropped or a default was fallen back to, which one sentence does.
_ENVELOPE_WARNING_TEXT_CAP = 200

# A decision's `description` and `rationale` are prose no validator compares,
# so they are the two fields this render may shorten. Every other plan field
# is rendered in full. `chosen`, `options` and every requirement field are
# what a restated plan is checked against, and shortening any of them would
# change the plan itself, not only the text the model reads.
_PLAN_DESCRIPTION_CAP = 200
_PLAN_RATIONALE_CAP = 300

# A compiled source's `sentence` is the resolved reading in search's own
# words. Capped so a nine-source plan cannot let one verbose sentence carry
# the section.
_PLAN_SOURCE_SENTENCE_CAP = 160

# The envelope's resolve runs once, question-wide, before `propose_plan`
# has invented any requirement (`graph.py`'s `probe -> route_status ->
# choose_depth -> planner` order). Its `spec_id` can therefore be right for
# at most one requirement of a `parts > 1` question, so it is left out of
# what the planner sees rather than merely told not to use it -- a field
# present and forbidden is the shape a model reliably tries anyway once it
# has no other candidate to reach for.
#
# The one exception is a `spec_id` the user actually elected answering a
# `clarify` round (`graph.py`'s `clarify`/`probe`, `spec_id_elected`): that
# choice is kept and named as the source for the part it matches, in
# `_ELECTED_SPEC_REASON` below, in place of this omission.
_MULTI_PART_REASON = """\


This question has {parts} parts. The envelope's spec_id ran once, \
question-wide, before any part existed, so it can be right for at most one \
of them and is left out of the envelope below. Pin each part from its own \
discovery."""

# The carve-out named above: the user chose this source answering which
# reading they meant, so it is kept and named for the part it matches,
# instead of `_MULTI_PART_REASON`'s omission.
_ELECTED_SPEC_REASON = """\


This question has {parts} parts. The user chose {label!r} when asked which \
reading they meant, so its spec_id, {spec_id!r}, is kept below as the \
source for the part it answers -- pin it as the requirement's \
source_spec_id, citing a Decision with resolved_by=user, fixes naming \
source_spec_id, and options listing every candidate offered. Pin every other part from its own \
discovery."""

# A turn seeded from Search (`graph.py`'s `seed` node) arrives with the
# datasets the user chose, one per part, each with the operations the
# question asked of it, its window, the year and rows Search showed when the
# question named no year, and its coverage counts. The envelope's own
# `spec_id` names only the first, so it is left out and each seed is named
# in its own block (`_render_seeds`). The pin is the user's choice, recorded
# as such; `propose_plan` traces it to the seed and pins the seed over the
# requirement's period (`planner.agent._window_pins`), so the period
# is the window the plan retrieves. `ceiling` is data-mcp's row ceiling as
# this process configures it (`harness.graph.frame_row_ceiling`). The series
# cap is the platform's: a chart or table component draws at most 24
# series. An operation over a party runs over an axis its dataset leaves
# open, which a requirement states in `across`, and a rank's `limit` keeps
# its top or bottom rows.
_SEEDED_REASON = """\


This question comes from Search with its data already chosen: the user saw \
each dataset under "Seeded datasets" below and handed it to Studio with the \
operations listed under it. Plan from these datasets, and plan any part of \
the question none of them covers as usual. Give each its own dataset \
requirement, pinned to its spec_id and citing a Decision with resolved_by=user, \
fixes naming source_spec_id, and options naming that spec_id and an unpinned \
source. The requirement's period is the years the plan \
retrieves, and one retrieval returns at most {ceiling:,} rows: choose \
the fewest years the question needs. A rank over place or origin needs one \
year, the one Search showed when the question named none; growth needs the \
range it runs over. A dataset's "showed" line counts the rows of one year, \
which bounds how many years fit. If a retrieval stopped at that ceiling, \
keep the dataset pinned and narrow its period. Turn each rank, aggregate, \
growth or share_of operation into the calculation it names. An operation \
"across" a party runs over every member of an axis the dataset leaves open, \
listed on its open_entity_axes line: state that party in the requirement's \
`across` and name no member of it. State a rank's `limit` when its phrase \
asks for a top or bottom N. An explain, forecast or report operation is what \
the answer argues from the data, and a forecast's "to" year is the year it \
projects to. A component shows at most 24 series."""

# A `RediscoveryNeeded` replan (`graph.py`'s `_replan_refusal`) sends the
# planner straight back here over the `execute -> planner` edge with no
# re-probe, so the envelope this seam renders still names the exact spec_id
# the executor just refused, and every spec_id a wave sibling's own
# refusal named (`also_stopped`) -- the one thing this prompt must not
# offer as pinnable again, on any `parts` count, with no carve-out: the
# executor's refusal is a fact about the spec, while an election only says
# which reading the user meant.
_REFUSED_SPEC_REASON = """\


The source {spec_id!r} this envelope would otherwise name was just \
refused when the approved plan ran ({detail}), so it is left out of the \
envelope below. Propose a different source -- naming this one again \
reproduces the same refusal."""

_REFUSED_SPECS_REASON = """\


These sources this envelope would otherwise name were just refused when \
the approved plan ran, so they are left out of the envelope below: \
{named}. Propose different sources -- naming any of them again reproduces \
the same refusal."""

# A refusal naming a seed, or its copy over a plan's period, keeps the seed
# pinned: the user chose the dataset in Search, and another source would
# replace that choice. A retrieval stopped at the row ceiling names no
# spec_id; `_SEEDED_REASON` covers it.
_KEEP_SEED_SENTENCE = (
    " The seeded dataset {seed!r} could not be retrieved ({detail}). The user chose it "
    "in Search, so keep it pinned and propose it again."
)
_KEEP_SEEDS_SENTENCE = (
    " These seeded datasets could not be retrieved: {named}. The user chose them in "
    "Search, so keep each pinned and propose it again."
)

# The run-time refusal codes that mean no operation computes what the refused
# step was asked to, each with the `unsupported` answer it stands for. A replan
# entry over one starts with that answer recorded (`_earned_by_refusal`), so a
# plan may decline the ask it served, as it may after `propose_plan` or
# `translate` answers `unsupported`. `cross_frame_unit_unbound`: one side of a
# `share_of`, `compare` or `sum_of` declares no unit while the other does, so
# no operation combines the two. A source declaring a unit would also clear
# it, and the decline leaves the planner free to pin one instead. Codes whose
# `fix` offers a choice (a common unit, a facet, matching periods, a window, a
# denominator, an entity) are method choices, and a stop
# (`executor.classify.STOP_CODES`) never replans.
_UNSUPPORTED_BY_RUN_TIME_CODE: Mapping[str, UnsupportedOperation] = {
    "cross_frame_unit_unbound": UnsupportedOperation.NO_OPERATION_COVERS,
}

_NO_SUBMIT_NUDGE = (
    "You have not called submit_plan. Propose the plan with propose_plan "
    "(and have it translated with translate, if it is not yet compiled), "
    "then call submit_plan with the current version."
)

# The `refused["code"]` `_before_planner_model` stamps on a deadline refusal
# it builds itself, never a `planner.tool_results.ProblemCode` read off a
# `propose_plan` response. `_refusal_text` compares against this to route to
# `_deadline_text`, and nothing outside this module ever sees it.
_DEADLINE_REFUSAL_CODE = "planner_deadline_exceeded"

# The seconds of an entry's deadline a discovery call leaves for the planner's
# next model call, which reads what the call answered. A search that runs into
# this margin is cut off (`planner.plan.discovery_time_left`), so a search that
# succeeds still leaves the planner time to use it before the deadline.
_NEXT_MODEL_CALL_MARGIN_S = 10.0

# The `refused["code"]` `_track_tool_call` stamps once `translate` answers
# with an `internal` referral, a fault no correction round repairs, so it
# ends the entry the way a deadline does, carrying the referral's reference.
_INTERNAL_REFUSAL_CODE = "translation_internal_failure"

# The `refused["code"]` `_track_tool_call` stamps once `translate` answers
# with a referral marked `repeated`. The plan failed to translate the way
# it failed before, unchanged where it failed, so a correction round would
# translate it a third time to the same stop.
_REPEATED_REFUSAL_CODE = "translation_repeated"

# The `refused["code"]` `_track_tool_call` stamps once `translate` answers
# with a referral marked `budget_spent`. A discovery budget is spent and the
# most recent search that spent it timed out, so translating again in this
# entry is refused before it searches.
_UNAVAILABLE_REFUSAL_CODE = "discovery_unavailable"

# What the break `_before_planner_model` returns says, per closing refusal code.
_CLOSING_REASONS = {
    _INTERNAL_REFUSAL_CODE: "translation failed internally",
    _REPEATED_REFUSAL_CODE: "the plan failed to translate the same way again",
    _UNAVAILABLE_REFUSAL_CODE: "the data service is unavailable",
    ProblemCode.NOTHING_TO_PLAN: "every ask in the question is declined",
}

# What a stalled referral's message tells the planner to do, per kind.
# `{repair}` and `{constraint}` read the referral's own fields. A transient
# kind never stalls, and an internal one ends the entry first.
_STALLED_ACT_DEFAULT = "Change what the requirement asks for"
_STALLED_ACTS: Mapping[str, str] = {
    ReferralKind.SOURCE_AMBIGUOUS: "Settle on one of the named candidates",
    ReferralKind.PINNED_SOURCE_NOT_FOUND: (
        "Settle on one of the named candidates, or unpin the source and change what "
        "the requirement asks for"
    ),
    ReferralKind.NO_DATA: (
        "Change what the requirement asks for; the nearest sources are context, not pins"
    ),
    ReferralKind.REQUIREMENT_UNMEETABLE: "Tell the user what cannot be done and ask what to change",
    ReferralKind.ASK_NOT_UNDERSTOOD: "Reword the part of the ask the referral names",
    ReferralKind.DISCOVERY_BUDGET_EXHAUSTED: "Ask for less at once",
    ReferralKind.SEED_DROPPED: "Remove that requirement",
    ReferralKind.REQUIREMENT_NEEDS_REPAIR: "Make the permitted repair ({repair}) for {constraint}",
    ReferralKind.UNSUPPORTED_ASK: "Tell the user what cannot be done and ask what to change",
    ReferralKind.NO_OPERATION_COVERS: (
        "Record the ask in `declined`, in the user's words, and plan only what can "
        "still be answered"
    ),
}

# Gemini requires a `thought_signature` on a generation's first `functionCall`
# part and answers a request replaying one without it with a 400. This is
# the documented placeholder that makes the API skip validation for a call
# injected outside a model turn. `_before_planner_model`'s other synthetic
# response is a text part, which the rule does not cover.
_SKIP_THOUGHT_SIGNATURE = b"skip_thought_signature_validator"


class PlannerSeamError(RuntimeError):
    """The planner produced no plan ready for the user's answer.

    Raised so that a plan the user approves is a plan `submit_plan` itself
    declared ready, never one this seam assembled from a partial run.

    `refusal` is the shape `harness.studio_translate._stream_fatal` reads
    (`code`, `retry_mode`, `fix.detail`), so this reaches the wire as the
    real reason rather than as a bare exception the trace id has to stand
    in for. Every way this seam fails on its own account replays the same
    way against the same state, so the default is `do_not_retry` -- a
    caller told to retry a deterministic failure just pays for the same
    stall again.
    """

    def __init__(self, message: str, *, refusal: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.refusal = refusal or {
            "code": "capability_unavailable",
            "retry_mode": "do_not_retry",
            "fix": {"detail": message},
        }


class _RunTracker:
    """Per-`_propose` scratch state the planner's callbacks read and mutate
    through `_active_call` (see `_CallState`).

    Never a session field and never shared across calls: each `_propose`
    call builds a fresh tracker, hands it to `_active_call.set(...)` for
    the duration of its own `await ctx.run_node(...)` call, and resets
    that var in a `finally` -- deterministically visible to the code
    right after that `await` returns, no queue needed to get the count or
    the stall back out, and never reachable from a callback firing for a
    different call (see `PlannerSeam.__init__` and `PlannerSeam._propose`
    for why the callbacks themselves are assigned once, not per call, and
    therefore cannot close over their own call's tracker directly). See
    `_track_tool_call`.
    """

    def __init__(self, deadline_at: float) -> None:
        # The wall-clock instant (`PlannerSeam._clock()`) this entry into the
        # planner has to stop generating by, computed once in
        # `PlannerSeam.__call__` and shared by every `_propose` call of the
        # same entry; a resumed entry gets a fresh one since `__call__` runs
        # again to produce it. Checked in `_before_planner_model`.
        self.deadline_at = deadline_at
        self.referral_counts: dict[tuple[str, str | None], int] = {}
        self.stalled: str | None = None
        self.refused: dict[str, Any] | None = None
        self.ready_to_break = False
        self.submitted: dict[str, Any] | None = None
        self.events = 0
        # The referral the last `translate` response with an `outcome`
        # answered, or `None` once one of those responses answers something
        # other than a referral -- see `_track_tool_call`. `_before_planner_
        # model`'s deadline branch reads this for its own candidates, since
        # the run can hit its deadline on a referral that has not yet
        # repeated.
        self.last_referral: dict[str, Any] | None = None
        # The clock a progress report's `duration_ms` measures against:
        # reset every time one fires, so each report covers everything
        # since the last one, including the thinking the model call that
        # produced this one spent before ever emitting a function call --
        # not a measurement of the call in isolation, which nothing here
        # ever sees start.
        self.last_progress = time.monotonic()


@dataclass(frozen=True, slots=True)
class _CallState:
    """The one `_propose` call currently in flight through this seam's
    permanently assigned callbacks, threaded through `_active_call`
    instead of a closure.

    Bundled in one object, not two separate `ContextVar`s, so one
    `.set()`/reset pair covers both -- `tracker` and `on_progress` always
    belong to the same call and are never meaningful apart.
    """

    tracker: _RunTracker
    on_progress: Callable[[dict[str, Any]], None] | None


# The call this seam's callbacks read their state from. A plain instance
# attribute would not do: `self._agent` is shared by every conversation
# this process serves (see `PlannerSeam.__init__`), so its callbacks are
# assigned once and have no per-call closure to read state from. A
# `ContextVar` instead: `_propose` sets it for the
# duration of its own `await ctx.run_node(...)` call and resets it in a
# `finally`; the seam's node is awaited directly by the graph, with no
# `asyncio.ensure_future` wrapper, so the callbacks ADK invokes run in the
# same task as the `_propose` frame that set the var, and any task ADK
# spawns beneath inherits that task's own copy -- never a sibling
# conversation's. Module-level rather than a `PlannerSeam` attribute:
# there is exactly one production `PlannerSeam`, so it makes no
# observable difference, and a module-level var is what the seam's
# callbacks -- themselves free functions in every sense but their `self`
# argument -- need no instance to reach.
_active_call: contextvars.ContextVar[_CallState | None] = contextvars.ContextVar(
    "planner_seam_active_call", default=None
)


class PlannerSeam:
    """`harness.graph.PlannerModel`, implemented by `src/planner` and its `translate` tool."""

    def __init__(
        self,
        *,
        settings: Settings,
        agent_settings: AgentSettings | None = None,
        surface: ToolSurface | None = None,
        clock: Callable[[], float] = time.monotonic,
        thinking_sample_rng: Callable[[], float] = random.random,
        prefix_cache: PrefixCache | None = None,
    ) -> None:
        # `planner_entry_deadline_seconds`/`thinking_sample_rate` are read once
        # here rather than per call, matching the default
        # `create_planner_with_translator` resolves for the models it builds
        # (`None` -> `AgentSettings()`). `clock` and `thinking_sample_rng`
        # stand in for `time.monotonic`/`random.random` in a test driving the
        # deadline or the sampling draw without a real sleep or RNG, see
        # `__call__` and `_before_planner_model`.
        resolved_agent_settings = agent_settings if agent_settings is not None else AgentSettings()
        self._planner_deadline_seconds = resolved_agent_settings.planner_entry_deadline_seconds
        self._planner_deadline_seconds_sampled = (
            resolved_agent_settings.planner_entry_deadline_seconds_sampled
        )
        self._thinking_sample_rate = resolved_agent_settings.thinking_sample_rate
        self._clock = clock
        self._thinking_sample_rng = thinking_sample_rng
        self._agent = create_planner_with_translator(
            settings=settings,
            agent_settings=agent_settings,
            surface=surface,
            # The graph derives and narrates a turn's `resolve_envelope`,
            # `depth` and `question` before the planner ever runs (see
            # `planner.agent.create_planner_agent`'s own `question_fixed`
            # doc), so this seam's planner never gets to restart a plan for a
            # different question -- `starts_new_plan` is refused with
            # `QUESTION_FIXED` instead, handled below the same way
            # `BUDGET_EXHAUSTED` is.
            question_fixed=True,
            prefix_cache=prefix_cache,
        )
        # The callbacks are assigned exactly once, here, as bound methods,
        # and read the call they belong to off `_active_call` (a
        # `ContextVar` `_propose` sets for the duration of its own
        # `ctx.run_node` call) instead of closing over it, since one
        # `PlannerSeam` serves every conversation this process runs.
        self._agent.after_tool_callback = self._after_planner_tool
        # `drop_thought_parts` (`planner.settings`) returns a modified
        # response whenever it drops a part, and ADK's callback list stops at
        # the first truthy return, so the usage callback goes first: it
        # always returns `None`, and `drop_thought_parts` still runs after it.
        self._agent.after_model_callback = [
            self._after_planner_model,
            *self._agent.canonical_after_model_callbacks,
        ]
        # `override_thinking_on_sampled_turns` always returns `None`, so it
        # never short-circuits this list: `_before_planner_model` still
        # decides every response. A single assignment here would silently
        # drop the override the moment a deadline check or a stalled-referral
        # stop replaces it.
        self._agent.before_model_callback = [
            override_thinking_on_sampled_turns,
            self._before_planner_model,
        ]

    async def __call__(
        self,
        *,
        ctx: Context,
        question: str,
        envelope: dict[str, Any],
        prior_plan: dict[str, Any] | None = None,
        refusal: dict[str, Any] | None = None,
        amendment: str | None = None,
        canvas_digest: dict[str, Any] | None = None,
        conversation_digest: dict[str, Any] | None = None,
        spec_id_elected: bool = False,
        spec_id_elected_label: str | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        readings: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # `readings`, set on the orchestrator engine only, is the search
        # readings of the conversation's earlier turns and then this turn's,
        # as `harness.orchestrator_tools` stores them. The first message
        # renders them where the classic engine's renders the envelope.
        #
        # `propose_plan` amends from the graph's record, not whatever this
        # session's own drafting key already holds, reseeded every call --
        # unconditionally, regardless of `fresh` below: this is tool
        # bookkeeping `propose_plan` reads, not message content, so it stays
        # exactly as it was before the fresh/continuing split.
        #
        # Written through `ctx.session.state`, not `ctx.state`, and so is
        # every other write in this method: `ctx.state[key] = value` records
        # the write in `ctx.actions.state_delta` for the framework to attach
        # to this *node's own* event once it finishes, which -- since this
        # write happens before `ctx.run_node` runs the planner and the node's
        # own event does not finish until well after -- would sit there
        # holding this stale seed until that later event re-applies it,
        # overwriting whatever the dynamically-run pair wrote via
        # `ctx.run_node` in between. Writing straight to the live
        # `Session.state` dict skips that delta entirely: visible
        # immediately to the planner's own tool calls (the same live dict,
        # never copied mid-run), but not on its own durable across a
        # `Runner.run_async` boundary -- a resume included. `_persist_durable`,
        # below, is what closes that gap without reopening the stale-seed
        # one: it mirrors whatever this method's own writes and the planner's
        # now-settled ones actually left in the live dict, read fresh at the
        # very end, into `ctx.state` -- so the delta this node's own event
        # eventually carries is never older than what `ctx.run_node` itself
        # already produced.
        if prior_plan is not None:
            save_plan(cast(SessionState, ctx.session.state), Plan.model_validate(prior_plan))
        if not ctx.resume_inputs:
            # One `__call__` is one entry into the planner, EXCEPT when it
            # re-enters through a `RequestInput` this seam itself raised
            # partway through (the ask-the-user interrupt) and is resumed with
            # that interrupt's answer in `ctx.resume_inputs` -- the graph's own
            # "amend" and "replan" edges both re-enter `__call__` fresh, with
            # nothing in `resume_inputs`, so they get a new budget here.
            # Resetting unconditionally would zero the budget partway through
            # an entry on the resume path, and the refer-and-repropose loop
            # `MAX_PLAN_ROUNDS` bounds would run unbounded for the rest of it.
            save_plan_rounds(cast(SessionState, ctx.session.state), 0)
            clear_unsupported(cast(SessionState, ctx.session.state))
            for operation in _earned_by_refusal(refusal):
                record_unsupported(cast(SessionState, ctx.session.state), operation)
            clear_compile_failures(ctx.session.state)
            if prior_plan is None and refusal is None and amendment is None:
                # This shape -- no `prior_plan`, no `refusal`, no
                # `amendment` -- names a genuine first-draft entry: the
                # graph's planner node passes exactly this on the
                # `choose_depth` edge, never on `amend`/`replan`, which
                # always carry the graph's own record and their own ask
                # (`harness.graph`'s own `planner` node). `prior_plan is
                # None` alone is not enough to gate this on: a caller of
                # this seam directly can omit it on a real amendment too
                # (`amendment` set, still nothing to seed `propose_plan`
                # from but the session's own last draft -- see
                # `test_a_sixth_amendment_turn_still_records_its_own_
                # version`), and that call must not have its own
                # in-progress draft wiped out from under it. Gated the same
                # way the round reset above is, by `not ctx.resume_inputs`:
                # a first-draft entry that pauses mid-draft (a
                # planner-side ask) re-enters `__call__` with this same
                # shape on its resume, and clearing again there would
                # throw away the very draft the ask paused, not a stale
                # one.
                #
                # A studio turn never has anything to clear here -- each
                # turn opens a brand-new ADK session, so both keys are
                # already unset. `evals/agents/dfo_analyst_graph`
                # (`just adk-web`) keeps one session across turns, so
                # without this, turn 2's own first-draft entry would find
                # turn 1's plan still on record and, with this seam's pair
                # built with `question_fixed=True`, could only amend turn
                # 1's plan, since `starts_new_plan` is refused. Clearing `_BRIEFED_STATE_KEY` alongside makes the
                # very next read of it, two lines below, compute a fresh
                # `True` for this call, so turn 2 gets the full brief
                # rather than the continuing-call message a session that
                # has already been briefed gets.
                ctx.session.state[PLAN_STATE_KEY] = None
                ctx.session.state[_BRIEFED_STATE_KEY] = False
                # A first-draft entry draws fresh rather than reusing an
                # earlier turn's draw on the same persistent session: popped
                # here so the write below sees it absent and draws again.
                ctx.session.state.pop(THINKING_SAMPLE_STATE_KEY, None)
        try:
            # Read once, at the top of `__call__`: `fresh` answers "has this
            # session ever been briefed," not "is this the first `_propose`
            # call of this `__call__`" -- a correction round below builds its
            # own message with `fresh=False` unconditionally, never by
            # re-reading this flag, because by the time a correction round
            # runs, the first `_propose` call of this same `__call__` has
            # already put a message in the session's history.
            fresh = not ctx.session.state.get(_BRIEFED_STATE_KEY, False)
            ctx.session.state[_BRIEFED_STATE_KEY] = True
            # Drawn once per turn and shared with every generation through session
            # state (`THINKING_SAMPLE_STATE_KEY`,
            # `override_thinking_on_sampled_turns`): a correction round or a
            # resume within this turn reads the draw already on record.
            if THINKING_SAMPLE_STATE_KEY not in ctx.session.state:
                sampled = self._thinking_sample_rng() < self._thinking_sample_rate
                ctx.session.state[THINKING_SAMPLE_STATE_KEY] = sampled
                if sampled:
                    record_span("planner.thinking_sampled", {"dfo.thinking_sampled": True})
            # Anchored once here and carried unchanged into every `_propose`
            # call this entry makes, through `_RunTracker.deadline_at`: a
            # correction round is still the same entry into the planner, so
            # it shares this deadline, while a resume after a user answer
            # re-enters `__call__` and gets a fresh one. Checked in
            # `_before_planner_model`. A sampled turn (the draw just above,
            # read back off session state so a correction round or a resume
            # within this turn picks the same budget its own first entry
            # did) gets its own, larger deadline: it runs the planner at the
            # model's own default thinking level, deliberately slower than
            # the ordinary `LOW` turn this deadline is otherwise sized for.
            deadline_seconds = (
                self._planner_deadline_seconds_sampled
                if ctx.session.state[THINKING_SAMPLE_STATE_KEY]
                else self._planner_deadline_seconds
            )
            deadline_at = self._clock() + deadline_seconds
            seed_of = _seed_of(cast(SessionState, ctx.session.state))
            # Every spec_id the planner's message shows it, written on every
            # entry so a correction round inside this same `__call__` reads
            # it (`PIN_SOURCES_STATE_KEY`). Only a shown one reuses its
            # block, and only a shown, settled one gets a synthesized source
            # decision. Any other pin needs the model's own decision and is
            # confirmed by a fresh search. Readings render on a fresh
            # message only, so a later entry keeps the ones it rendered, less
            # any the executor has since refused.
            ledger = None
            if readings is None:
                sources = _envelope_pin_sources(
                    envelope, refusal=refusal, spec_id_elected=spec_id_elected
                )
            elif fresh:
                ledger, shown = _readings_ledger(readings)
                # An earlier turn's spec_id pins from the block its own turn
                # stored; this turn's search of one replaces it.
                prior = ctx.state.get(ORCHESTRATOR_PRIOR_LEDGER_KEY) or {}
                blocks = ctx.state.get(ORCHESTRATOR_SPECS_KEY) or {}
                searched = {*_named(ctx.state.get(ORCHESTRATOR_SEARCHES_KEY) or ()), *blocks}
                sources = _reading_pin_sources(
                    readings,
                    shown,
                    blocks=blocks,
                    earlier=_named(prior.get("readings", ())) - searched,
                    warnings={
                        **prior.get("warnings", {}),
                        **(ctx.state.get(ORCHESTRATOR_SPEC_WARNINGS_KEY) or {}),
                    },
                )
            else:
                sources = pin_sources(cast(SessionState, ctx.session.state))
            if readings is not None:
                refused = {spec_id for spec_id, _ in _refused_specs(refusal)}
                sources = {k: v for k, v in sources.items() if k not in refused}
            save_pin_sources(cast(SessionState, ctx.session.state), sources)
            message = self._message(
                fresh=fresh,
                question=question,
                envelope=envelope,
                prior_plan=prior_plan,
                refusal=refusal,
                amendment=amendment,
                canvas_digest=canvas_digest,
                conversation_digest=conversation_digest,
                spec_id_elected=spec_id_elected,
                spec_id_elected_label=spec_id_elected_label,
                seed_of=seed_of,
                ledger=ledger,
                asks=load_asks(cast(SessionState, ctx.session.state)),
            )
            submitted, plan, refused = await self._propose(
                ctx=ctx, message=message, on_progress=on_progress, deadline_at=deadline_at
            )
            if refused is not None:
                return self._refused(refused)
            problem = self._problem(submitted)
            corrections = 0
            while problem is not None and corrections < _MAX_CORRECTIONS:
                corrections += 1
                logger.info("planner seam correction %d: %s", corrections, problem)
                # `fresh=False` by construction, not by re-reading the flag:
                # the first `_propose` call above already briefed this
                # session, so this round sends only the problem to fix. It is
                # still its own `ctx.run_node` call, but `ctx.run_node`
                # shares the graph's own session (see the module docstring),
                # so the model still sees everything the first call rendered
                # -- it does not need it repeated.
                message = self._message(
                    fresh=False,
                    question=question,
                    envelope=envelope,
                    prior_plan=prior_plan,
                    refusal=refusal,
                    amendment=amendment,
                    canvas_digest=canvas_digest,
                    conversation_digest=conversation_digest,
                    spec_id_elected=spec_id_elected,
                    spec_id_elected_label=spec_id_elected_label,
                    seed_of=seed_of,
                    problem=problem,
                )
                submitted, plan, refused = await self._propose(
                    ctx=ctx, message=message, on_progress=on_progress, deadline_at=deadline_at
                )
                if refused is not None:
                    # A correction round's own `propose_plan` call can still
                    # hit the cap (the problem being corrected may have
                    # burned the last round). Return the same way the first
                    # call does: never retry a budget refusal, since a retry
                    # replays the same call against the same exhausted count.
                    return self._refused(refused)
                problem = self._problem(submitted)
            if problem is not None:
                raise PlannerSeamError(problem)
            assert submitted is not None and plan is not None, (
                "a None problem means submit_plan answered ok, which only happens once "
                "load_plan(state) found a plan (see planner.agent.submit_plan)"
            )
            if plan.status is not PlanStatus.PENDING_APPROVAL:
                # `submit_plan` itself enforces this (`planner.agent.
                # _execution_blocked`); repeated here so a future gap in that
                # allowlist reaches this seam as a loud `PlannerSeamError`
                # instead of a plan the graph shows the user as approvable.
                raise PlannerSeamError(
                    f"submit_plan answered ok for a plan that is {plan.status.value}, "
                    "not pending_approval; refusing to report it ready"
                )
            return {"plan": plan.model_dump(mode="json"), "summary": submitted["summary"]}
        finally:
            # Every exit of this method runs through here -- a normal
            # return, a raised `PlannerSeamError`, and a `NodeInterruptedError`
            # propagating straight out of `_propose`'s own `await
            # ctx.run_node(...)` (a planner-side `adk_request_input` ask)
            # alike. `finally` is what makes the last of those work: that
            # exception is never caught anywhere in this method, so only code
            # that always runs on the way out, regardless of how the method
            # is leaving, ever gets a chance to run before the flush this
            # node's own `NodeRunner` performs on the way to reporting the
            # pause (see `_persist_durable`'s own docstring).
            self._persist_durable(ctx)

    @staticmethod
    def _persist_durable(ctx: Context) -> None:
        """Mirror this call's live `ctx.session.state` writes into `ctx.state`,
        so a real event carries them and they survive past this
        `Runner.run_async` call -- a resume included.

        Read fresh off `ctx.session.state` right here, at the very end of
        `__call__` (see its own `finally`), never restated from what
        `__call__` wrote at its own top: `ctx.run_node`, in between, may have
        replaced `planner:plan` and `planner:rounds` several times over with
        the planner's own, already-durable writes (`propose_plan`, through
        `tool_context.state`). Mirroring the live value here, instead of
        writing through `ctx.state` at the point each one is first set, is
        what keeps this node's own delta -- which only ever flushes onto a
        real event once this node's own run finishes or pauses, never
        immediately -- from reapplying a stale seed over that fresher write
        when it finally does.

        `_BRIEFED_STATE_KEY` has no such collision to guard against for what
        this mirrors: `__call__`'s own first-draft clear (see there) writes
        it `False` only transiently, always flipping it back to `True`
        again, synchronously, before any `await` -- so it is always `True`
        again by the time this runs -- and nothing else in this session ever
        writes it `False`. Mirroring it here is only about durability, never
        about staleness: reapplying `True` a second time is a no-op either
        way.

        `PLAN_STATE_KEY` is the one exception to "only mirror what changed":
        `__call__`'s own first-draft clear can leave the live plan `None` on
        a session whose *durable* `ctx.state` still holds an earlier turn's
        plan (`evals/agents/dfo_analyst_graph`'s multi-turn session) -- a
        real value there that has to be overwritten, not merely left
        unmirrored, or a later durable flush would reapply that stale plan
        over the clear. `PLAN_STATE_KEY in ctx.session.state` (`__contains__`
        on the plain `dict` ADK actually hands back for `ctx.session.state`,
        true even holding `None`) is what tells that clear apart from a
        session that has simply never held a plan, where there is nothing to
        overwrite.
        """
        session_state = cast(SessionState, ctx.session.state)
        plan = load_plan(session_state)
        if plan is not None:
            save_plan(cast(SessionState, ctx.state), plan)
        elif PLAN_STATE_KEY in ctx.session.state:
            ctx.state[PLAN_STATE_KEY] = None
        save_plan_rounds(cast(SessionState, ctx.state), load_plan_rounds(session_state))
        if ctx.session.state.get(_BRIEFED_STATE_KEY, False):
            ctx.state[_BRIEFED_STATE_KEY] = True
        # `PIN_SOURCES_STATE_KEY` needs `PLAN_STATE_KEY`'s treatment, not
        # `_BRIEFED_STATE_KEY`'s: `__call__` writes it, empty included, on
        # every entry, so a resume after a planner-side ask sees this turn's
        # own value once `propose_plan` runs again. The `in` check only
        # guards a direct caller of this seam's pair that bypassed `__call__`.
        if PIN_SOURCES_STATE_KEY in ctx.session.state:
            ctx.state[PIN_SOURCES_STATE_KEY] = ctx.session.state[PIN_SOURCES_STATE_KEY]
        # `UNSUPPORTED_STATE_KEY` and `COMPILE_FAILURES_STATE_KEY` too. The
        # declines an ask-the-user resume still owes, and the failures it may
        # repeat, are this entry's own.
        for key in (UNSUPPORTED_STATE_KEY, COMPILE_FAILURES_STATE_KEY):
            if key in ctx.session.state:
                ctx.state[key] = ctx.session.state[key]
        # `THINKING_SAMPLE_STATE_KEY` needs `PIN_SOURCES_STATE_KEY`'s
        # treatment: a drawn `False` survives a resume the same as a drawn
        # `True`, so `__call__`'s "already set" check never draws twice for
        # the same turn.
        if THINKING_SAMPLE_STATE_KEY in ctx.session.state:
            ctx.state[THINKING_SAMPLE_STATE_KEY] = ctx.session.state[THINKING_SAMPLE_STATE_KEY]

    # -- one agent run ------------------------------------------------------

    async def _propose(
        self,
        *,
        ctx: Context,
        message: str,
        on_progress: Callable[[dict[str, Any]], None] | None,
        deadline_at: float,
    ) -> tuple[dict[str, Any] | None, Plan | None, dict[str, Any] | None]:
        """Run the planner once through `ctx.run_node`, in the graph's own
        session.

        The planner's own tool loop is opaque to this call: `ctx.run_node`
        returns only once the whole turn has settled (no
        streaming/async-generator variant of it exists in the installed
        2.9.1 package). Progress, the stall watchdog's counting and its
        early stop all happen from *inside* that one turn, through
        `LlmAgent` callbacks that ADK runs synchronously as part of the
        same `ctx.run_node` call: `after_tool_callback` fires right after
        each tool call, and
        `before_model_callback` on the planner substitutes a synthetic,
        terminal `LlmResponse` once a stall is confirmed, ending the
        run cleanly rather than mid-generation.

        The substitution is deliberately not a cancellation: it ends the
        planner's current turn the way a normal turn ends (a model
        response with no further function call), so the session it leaves
        behind has a `FunctionResponse` recorded for every call. No
        half-finished turn is left for the next `_propose` call's fresh
        message to resume instead of seeing.

        The planner's callbacks themselves are assigned once, in `__init__`
        (see its comment for why). What makes them
        *this* call's own is `_active_call`, set here for the duration of
        `ctx.run_node` and reset in a `finally` so a call that raises still
        clears it -- never left dangling for some later call's own
        callback firing to misread, and never visible to a sibling call
        running in a different task in the meantime.

        The run carries `_PLANNER_BRANCH` (see its own comment), because a
        `ctx.run_node` call made without one leaves the planner's events
        unbranched, and an unbranched event belongs to every branch. The
        branch filters on ADK's side. Session state
        is untouched by it and `ctx.session.state` still reads the child's
        write back, below.

        What this run appended to the session is checked afterward
        (`_check_branches`), because the branch is what makes that filtering
        hold and ADK's own `_is_event_belongs_to_branch` returns True for any
        event whose branch is empty. The assertion covers both ways the
        guarantee can lapse: a dropped `override_branch` here, and a future
        ADK release that stops honouring it. The check runs when the run
        raises as well, so a run that suspends for a planner-side ask has the
        events it appended checked before the resume reads them.
        """
        tracker = _RunTracker(deadline_at)
        token = _active_call.set(_CallState(tracker=tracker, on_progress=on_progress))
        time_left_token = current_discovery_time_left.set(
            lambda: deadline_at - self._clock() - _NEXT_MODEL_CALL_MARGIN_S
        )
        before = len(ctx.session.events)
        try:
            content = types.Content(role="user", parts=[types.Part(text=message)])
            await ctx.run_node(self._agent, content, override_branch=_PLANNER_BRANCH)
        finally:
            current_discovery_time_left.reset(time_left_token)
            _active_call.reset(token)
            # Also checked when the run raised. A planner-side ask suspends it
            # by raising through `ctx.run_node`.
            _check_branches(ctx.session.events[before:], planner=self._agent.name)

        # Not `ctx.state`: `Context.state` is a `State` wrapper holding a
        # reference to `invocation_context.session.state` captured once, at
        # this node's own `Context` construction -- correct for what this
        # node itself writes before calling `ctx.run_node`, but it does not
        # pick up a write the *child* makes during that call, on a session
        # this node's own turn did not start. `ctx.session.state`, read
        # fresh off the live `Session` object, is what actually reflects
        # what the run just left behind.
        plan = load_plan(cast(SessionState, ctx.session.state))
        return tracker.submitted, plan, tracker.refused

    # -- the seam's own callbacks, assigned once in __init__ ----------------
    #
    # Each reads `_active_call` for the call it belongs to. A callback that
    # fires with nothing set there is not this seam's own call -- there is
    # no other source, since `__init__` is the only place these are ever
    # assigned, but a defensive no-op is still the right answer over a bare
    # `None` dereference: a stray or unexpected firing should drop one
    # progress report or skip one usage log, never crash a run some other
    # conversation is mid-turn on.

    def _after_planner_tool(
        self, tool: BaseTool, args: dict[str, Any], tool_context: Any, tool_response: dict[str, Any]
    ) -> None:
        del args, tool_context
        call = _active_call.get()
        if call is None:
            logger.warning("planner after_tool_callback fired with no active call")
            return
        self._track_tool_call(
            call.tracker,
            call.on_progress,
            agent_name=self._agent.name,
            tool=tool,
            tool_response=tool_response,
        )

    def _after_planner_model(self, callback_context: Any, llm_response: LlmResponse) -> None:
        del callback_context
        if llm_response.usage_metadata is not None:
            log_usage(self._agent.name, llm_response.usage_metadata)

    def _before_planner_model(self, callback_context: Any, llm_request: Any) -> LlmResponse | None:
        """Substitute a synthetic response once this call's own tracker is
        ready to break (see `_propose`'s docstring) or its deadline has
        passed. With no active call, see the class-level note above, there is
        nothing to check against, so `None` here means "generate normally,"
        the same as every ordinary turn this callback lets through.

        The deadline is checked first, and only while the tracker has not
        already been told to break some other way: a `ready_to_break` already
        set (the version cap, a repeating referral) names its own reason
        instead.
        """
        del llm_request
        call = _active_call.get()
        if call is None:
            logger.warning("planner before_model_callback fired with no active call")
            return None
        tracker = call.tracker
        if not tracker.ready_to_break and self._clock() >= tracker.deadline_at:
            return self._deadline_response(callback_context, tracker)
        if not tracker.ready_to_break:
            return None
        refused_code = (tracker.refused or {}).get("code")
        if refused_code in _CLOSING_REASONS:
            reason = _CLOSING_REASONS[refused_code]
        else:
            reason = tracker.stalled or "the plan's version cap was reached"
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"Stopping this turn: {reason}")],
            )
        )

    @staticmethod
    def _deadline_response(callback_context: Any, tracker: _RunTracker) -> LlmResponse:
        """The response substituted once a planner entry's wall-clock deadline
        (`tracker.deadline_at`) passes, before this call would otherwise
        reach the model for one.

        A referral naming pinnable candidates (`tracker.last_referral`,
        replaced by the outcome of every later `translate` response that
        carries one -- an `ok` clears it -- and left unchanged by one that
        carries none, see `_track_tool_call`) becomes the same
        `adk_request_input` long-running call `planner.agent.ask_user` issues,
        built here from those candidates since the model has run out of time
        to compose it itself.
        `populate_client_function_call_id` gives the call an id, so the
        resume pairs with the user's answer the way a model-issued ask does
        (`planner.agent._elected_spec_ids`), and a `resolved_by=user` pin of
        the chosen candidate validates. This branch needs no tracker write:
        the long-running call pauses `ctx.run_node` before any later callback
        of this tracker could fire.

        With no referral, or one with nothing pinnable (a `no_data`
        referral's nearest-series candidates carry no `spec_id`, see
        `_pinnable_candidates`), this refuses instead: `tracker.refused` is
        set the way a `propose_plan` `BUDGET_EXHAUSTED` refusal is, so
        `_propose` and `__call__` report it immediately, with no correction
        round. `submit_plan`'s ok answer ends the planner's run
        (`skip_summarization`), so no model call, and so no deadline, follows
        a submitted plan.
        """
        candidates = _pinnable_candidates(tracker.last_referral)
        if candidates:
            assert tracker.last_referral is not None
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name=REQUEST_INPUT_FUNCTION_CALL_NAME,
                                args=_clarifying_question_args(candidates),
                            ),
                            thought_signature=_SKIP_THOUGHT_SIGNATURE,
                        )
                    ],
                )
            )
        tracker.ready_to_break = True
        tracker.refused = {
            "code": _DEADLINE_REFUSAL_CODE,
            "detail": {
                "requirement_ids": _requirements_without_discovery(
                    cast(SessionState, callback_context.session.state)
                )
            },
        }
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="Stopping this turn: this turn's time budget ran out")],
            )
        )

    @staticmethod
    def _track_tool_call(
        tracker: _RunTracker,
        on_progress: Callable[[dict[str, Any]], None] | None,
        *,
        agent_name: str,
        tool: BaseTool,
        tool_response: dict[str, Any] | None,
    ) -> None:
        """The planner's `after_tool_callback`: progress, then whichever of
        `submit_plan` / `propose_plan` / `translate` this tool call is, if any.
        """
        tracker.events += 1
        if tracker.events > _MAX_EVENTS:
            raise PlannerSeamError(
                f"the planner seam ran past {_MAX_EVENTS} events without settling"
            )
        if on_progress is not None:
            now = time.monotonic()
            on_progress(
                {
                    "agent": agent_name,
                    "tools": [tool.name],
                    "duration_ms": int((now - tracker.last_progress) * 1000),
                }
            )
            tracker.last_progress = now
        if tool.name == "submit_plan":
            tracker.submitted = tool_response
        elif tool.name == "propose_plan" and PlannerSeam._is_terminal_refusal(tool_response):
            # `propose_plan` refused with a code that records nothing and
            # leaves nothing correctable within this entry: the round cap
            # (`planner.plan.MAX_PLAN_ROUNDS`, `BUDGET_EXHAUSTED`), or a
            # restart while the graph holds the question fixed
            # (`QUESTION_FIXED`, `planner.agent`'s `question_fixed` mode --
            # this seam's own planner is always built that way, see
            # `PlannerSeam.__init__`). The cap bounds the proposals within
            # one entry into the planner, since `PlannerSeam.__call__`
            # resets the count on each entry (see `__call__`); a restart is
            # refused however fresh that budget is. Stop the planner's turn
            # here either way, before the model gets a chance to call
            # `submit_plan` on the stale plan the refusal names as still
            # approvable.
            tracker.ready_to_break = True
            tracker.refused = tool_response
        elif tool.name == TRANSLATE_TOOL_NAME:
            PlannerSeam._track_translation(tracker, tool_response)

    @staticmethod
    def _track_translation(tracker: _RunTracker, tool_response: dict[str, Any] | None) -> None:
        """Read one `translate` response: the referral it answered, and
        whether it ends the entry.

        `last_referral` is written from every response that carries an
        `outcome`, referral or not: `translate` runs the whole plan, so each
        outcome-bearing response states what is open across it right now, and
        replaces whatever the last one recorded. `ok` means every
        requirement compiled, which clears a referral the same way. A response
        with no `outcome` settles nothing and is left alone.
        `_before_planner_model`'s deadline branch reads `last_referral`
        whichever way it landed: a deadline can expire on a referral's first
        appearance, before it ever repeats, and still needs its candidates.

        An internal, repeated or budget-spent failure (`_closing_refusal`)
        ends the entry with no correction round, since the next translation
        would stop where this one did. A referral that recurs with nothing new
        (`_stalled_referral`) ends the run for `__call__`'s correction round.
        Either way the planner's turn stops before its next generation.
        """
        if isinstance(tool_response, dict) and "outcome" in tool_response:
            tracker.last_referral = PlannerSeam._referral_dict(tool_response)
        closing = PlannerSeam._closing_refusal(tracker.last_referral)
        if closing is not None:
            tracker.ready_to_break = True
            tracker.refused = closing
            return
        stalled = PlannerSeam._stalled_referral(tool_response, tracker.referral_counts)
        if stalled is not None:
            tracker.stalled = stalled
            tracker.ready_to_break = True
            tracker.submitted = {"outcome": "referral_stalled", "message": stalled}

    # `propose_plan` codes that record nothing and end the planner's turn
    # rather than opening a correction round -- see `_is_terminal_refusal`.
    _TERMINAL_PROPOSE_CODES = frozenset(
        {ProblemCode.BUDGET_EXHAUSTED, ProblemCode.QUESTION_FIXED, ProblemCode.NOTHING_TO_PLAN}
    )

    # Refusal codes after which no other plan for the same ask can succeed
    # in this turn: a fault in the system, a plan that failed the same way
    # twice, a question whose every ask is declined, or a data service that
    # is not answering.
    _FINAL_CODES = frozenset(
        {
            _INTERNAL_REFUSAL_CODE,
            _REPEATED_REFUSAL_CODE,
            _UNAVAILABLE_REFUSAL_CODE,
            ProblemCode.NOTHING_TO_PLAN,
        }
    )

    @staticmethod
    def _refused(refused: dict[str, Any]) -> dict[str, Any]:
        """What `__call__` answers for a refusal that ends the entry: its plain
        sentence, and whether another plan for the same ask could succeed."""
        return {
            "refused": {
                "text": PlannerSeam._refusal_text(refused),
                "final": refused.get("code") in PlannerSeam._FINAL_CODES,
            }
        }

    @staticmethod
    def _is_terminal_refusal(tool_response: dict[str, Any] | None) -> bool:
        """Whether a `propose_plan` response is one of `_TERMINAL_PROPOSE_CODES`
        (`planner.agent.propose_plan`) -- a `propose_plan` outcome that records
        no new version and must not be treated as a correctable problem.
        """
        return (
            isinstance(tool_response, dict)
            and tool_response.get("outcome") == "refused"
            and tool_response.get("code") in PlannerSeam._TERMINAL_PROPOSE_CODES
        )

    @staticmethod
    def _refusal_text(refused: dict[str, Any]) -> str:
        """The plain sentence a terminal `propose_plan` refusal becomes for the
        user, per `code` (see `_TERMINAL_PROPOSE_CODES`).

        This seam has no view of the graph's own record (`ctx.state["plan"]`),
        so neither text makes a claim about what can still be approved:
        `harness.graph`'s planner node adds that, naming its own version, when
        the record it holds supports one.
        """
        if refused.get("code") == ProblemCode.QUESTION_FIXED:
            return (
                "The plan already on record answers this turn's question, so this "
                "was not recorded as a change to it. Ask the other question as its "
                "own message."
            )
        if refused.get("code") == _DEADLINE_REFUSAL_CODE:
            return PlannerSeam._deadline_text(refused)
        detail = refused.get("detail") or {}
        if refused.get("code") == ProblemCode.NOTHING_TO_PLAN:
            asks = "; ".join(str(decline.get("ask")) for decline in detail.get("declined") or [])
            return f"This cannot be calculated by this system, so nothing was planned: {asks}."
        translated = PlannerSeam._translation_refusal_text(refused.get("code"), detail)
        if translated is not None:
            return translated
        return PlannerSeam._budget_exhausted_text(refused)

    @staticmethod
    def _translation_refusal_text(code: object, detail: dict[str, Any]) -> str | None:
        """The plain sentence a refusal `_closing_refusal` stamped becomes for
        the user, or `None` for any other code."""
        if code == _INTERNAL_REFUSAL_CODE:
            return (
                f"Translation failed internally (reference {detail.get('reference')}), so "
                "nothing was recorded. The fault is in the system, not the request, and "
                "planning the same request again fails the same way."
            )
        if code == _UNAVAILABLE_REFUSAL_CODE:
            return (
                "The data service did not answer while finding data for requirement "
                f"{detail.get('requirement_id')!r}, so nothing was recorded. The service is "
                "unavailable; try again shortly."
            )
        if code == _REPEATED_REFUSAL_CODE:
            return (
                f"The plan failed to translate the same way twice: requirement "
                f"{detail.get('requirement_id')!r} came back as {detail.get('kind')!r} "
                f"({detail.get('reason')}), and the plan did not change there, so nothing was "
                "recorded. Planning the same request again fails the same way; what that "
                "requirement asks for has to change."
            )
        return None

    @staticmethod
    def _budget_exhausted_text(refused: dict[str, Any]) -> str:
        """The plain sentence a `BUDGET_EXHAUSTED` refusal becomes for the
        user: the per-entry budget it hit, read off its own `detail`, and that
        the user's next message starts with a fresh one.

        The budget counts the proposals one entry into the planner makes,
        restarts included (`planner.plan.MAX_PLAN_ROUNDS`). Every entry (a new
        message, an amendment, a replan) starts it over, while the plan's
        version keeps climbing across entries, so the text names no version.
        """
        detail = refused.get("detail") or {}
        return (
            f"This request has used its budget of {detail.get('rounds', MAX_PLAN_ROUNDS)} plan "
            "drafts, so nothing was recorded. Your next message starts with a fresh "
            "budget."
        )

    @staticmethod
    def _deadline_text(refused: dict[str, Any]) -> str:
        """The plain sentence a deadline refusal becomes for the user: which
        requirement(s), if any, had no source discovered yet when this
        entry's wall-clock budget ran out (`_before_planner_model`'s
        `detail.requirement_ids`, see `_requirements_without_discovery`).

        Reached only on the branch with no referral carrying pinnable
        candidates, see `_before_planner_model`. A plan with every
        requirement bound is one `submit_plan` would already have recorded,
        ending the entry before its deadline is checked again, so this is
        never reached for one.

        Says nothing was recorded rather than naming a change:
        this entry can be a first draft, with no prior plan for anything to
        be a change to, as readily as an amendment, and `tracker.submitted`
        being unset (the condition that puts this refusal on the tracker at
        all, see `_before_planner_model`) means the same thing either way --
        this entry recorded nothing new.
        """
        detail = refused.get("detail") or {}
        requirement_ids = detail.get("requirement_ids") or []
        if requirement_ids:
            named = ", ".join(str(requirement_id) for requirement_id in requirement_ids)
            pronoun = "it" if len(requirement_ids) == 1 else "them"
            about = f"requirement(s) {named} had a source discovered for {pronoun}"
        else:
            about = "this plan had a source discovered for it"
        return (
            f"This request ran out of time before {about}, so nothing was recorded. Your "
            "next message starts with a fresh budget."
        )

    @staticmethod
    def _referral_dict(response: dict[str, Any] | None) -> dict[str, Any] | None:
        """The referral a `translate` response answers with, or `None` for any
        other answer. It is the one check `_stalled_referral` and
        `_track_translation`'s own `last_referral` bookkeeping both need, kept
        here so the two never drift apart. A referral is the one answer that
        names a `kind`, and `ok` and a problem name none.
        """
        if not isinstance(response, dict) or "kind" not in response:
            return None
        return response

    @staticmethod
    def _closing_refusal(referral: dict[str, Any] | None) -> dict[str, Any] | None:
        """The refusal a referral ends the planner entry on, or `None` when a
        correction round may still settle it: an `internal` referral, with
        its reference; one marked `repeated`, with what it is about; or one
        marked `budget_spent`, with the requirement it is about."""
        if referral is None:
            return None
        if referral.get("outcome") == Outcome.INTERNAL.value:
            return {
                "code": _INTERNAL_REFUSAL_CODE,
                "detail": {"reference": str(referral.get("reference"))},
            }
        if referral.get("budget_spent"):
            return {
                "code": _UNAVAILABLE_REFUSAL_CODE,
                "detail": {"requirement_id": referral.get("requirement_id")},
            }
        if referral.get("repeated"):
            return {
                "code": _REPEATED_REFUSAL_CODE,
                "detail": {key: referral.get(key) for key in ("kind", "requirement_id", "reason")},
            }
        return None

    @staticmethod
    def _stalled_referral(
        response: dict[str, Any] | None, counts: dict[tuple[str, str | None], int]
    ) -> str | None:
        """Count one referral's `(kind, requirement_id)` pair, and answer the
        message to stop the run on once that exact pair recurs a second time.
        A third repetition would carry nothing a model could act on that the
        second one did not already say. A
        referral marked `repeated` ends the entry instead
        (`_closing_refusal`). A transient referral is never counted: the
        same plan is worth translating again. The message names what to
        do per kind (`_STALLED_ACTS`).

        Reads `response` defensively. `MAY_CARRY_CANDIDATES`
        (`translator/referral.py`) admits `source_ambiguous`,
        `pinned_source_not_found` and `no_data`, and a kind outside it
        answers an empty tuple here, so this count holds whichever kinds
        that set names.
        """
        referral = PlannerSeam._referral_dict(response)
        if referral is None or referral.get("outcome") == Outcome.TRANSIENT.value:
            return None
        kind = referral.get("kind")
        if kind is None:
            return None
        requirement_id = referral.get("requirement_id")
        pair = (kind, requirement_id)
        count = counts.get(pair, 0) + 1
        counts[pair] = count
        if count < 2:
            return None
        candidates = referral.get("candidates") or []
        named = ", ".join(
            str(candidate["spec_id"])
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("spec_id")
        )
        about = f"requirement {requirement_id!r}" if requirement_id else "the plan"
        have = f"; candidates on record: {named}" if named else "; no candidates are on record"
        act = _STALLED_ACTS.get(kind, _STALLED_ACT_DEFAULT).format(
            repair=referral.get("repair"), constraint=referral.get("constraint")
        )
        return (
            f"Translation referred {about} back as {kind!r} for the second time with "
            f"nothing new{have}. {act} -- calling translate again on the same plan "
            "will not converge."
        )

    @staticmethod
    def _problem(submitted: dict[str, Any] | None) -> str | None:
        """What is wrong with the run, as a message the planner can act on."""
        if submitted is None:
            return _NO_SUBMIT_NUDGE
        if submitted.get("outcome") != "ok":
            return str(submitted.get("message") or _NO_SUBMIT_NUDGE)
        return None

    # -- messages -----------------------------------------------------------

    def _message(
        self,
        *,
        fresh: bool,
        question: str,
        envelope: dict[str, Any],
        prior_plan: dict[str, Any] | None,
        refusal: dict[str, Any] | None,
        amendment: str | None,
        canvas_digest: dict[str, Any] | None,
        conversation_digest: dict[str, Any] | None,
        spec_id_elected: bool = False,
        spec_id_elected_label: str | None = None,
        seed_of: Mapping[str, str] | None = None,
        problem: str | None = None,
        ledger: str | None = None,
        asks: Sequence[str] = (),
    ) -> str:
        """The message for one `_propose` call.

        `fresh=True` -- the first call this seam makes in a session --
        renders the full brief: the envelope, the canvas and conversation
        digests, the plan being changed, and the question. `fresh=False` --
        a correction round inside `__call__`'s own loop, or a later
        `__call__` re-entering over the `amend`/`replan` edge -- renders
        only the new thing: the `amendment`/`refusal` ask, or the
        correction's `problem` nudge. `prior_plan` is never rendered when
        `fresh` is `False`, even though it is still accepted here and still
        seeded into `propose_plan`'s own drafting state unconditionally
        (`__call__`'s own `save_plan` call, which does not read `fresh`).
        `spec_id_elected`/`spec_id_elected_label` only affect the envelope
        section, so they too are inert once `fresh` is `False`.

        `seed_of` maps each seeded spec_id, a seed or its copy over a plan's
        period, to the seed (`_seed_of`); the envelope's own seeds map to
        themselves whether or not it names them. A refusal naming one keeps
        the seed pinned (`_keep_seeds_sentence`) where any other refused spec
        is to be avoided.

        `ledger`, the orchestrator engine's rendered search readings
        (`_readings_ledger`), stands in place of the envelope section when
        set. `asks`, the outputs the orchestrator engine's brief lists
        (`planner.plan.ASKS_STATE_KEY`), render numbered after the question.

        See the module docstring for why a continuing call needs no more
        than this: `ctx.run_node` shares the graph's own session, so the
        model already has the full brief in its own history by the time a
        continuing call runs, and `prior_plan` on both edges restates
        nothing the ask does not already say.

        The question and the amendment are the user's own words, and a refusal
        carries text from the run that failed. Nothing bounds their length on
        the wire, so each renders cut to `MAX_QUESTION_CHARS`
        (`planner.plan.cut_user_text`), the size `Plan.question` holds.

        The envelope and `prior_plan` render as digests, through
        `_render_envelope` and `_plan_context`. The instruction above the
        envelope keeps its own wording. It is what stops the model inventing
        a `spec_id`, and a bare reference to the envelope would need a fetch
        tool neither agent has.
        """
        seed_of = {
            **{seed["spec_id"]: seed["spec_id"] for seed in envelope.get("seeds") or []},
            **(seed_of or {}),
        }
        ask = ""
        if amendment:
            ask = (
                "\nThe user asks for this change to the plan: "
                f"{cut_user_text(amendment, MAX_QUESTION_CHARS)}\n"
                "Call propose_plan again with the revised decisions and evidence "
                "requirements -- the change has to alter what a requirement "
                "actually states, not only a decision's rationale -- then have "
                "it translated and compiled again before calling submit_plan.\n"
            )
        elif refusal:
            refused, kept = _split_seeded(_refused_specs(refusal), seed_of)
            avoid = (
                _avoid_sentence(refused)
                + _keep_seeds_sentence(kept)
                + _earned_decline_sentence(_earned_by_refusal(refusal))
            )
            stop = {key: value for key, value in refusal.items() if key != "computed_asks"}
            ask = (
                f"\n{_failed_run_sentence(refusal.get('computed_asks'))} "
                f"Refusal: {cut_user_text(json.dumps(stop), MAX_QUESTION_CHARS)}\n"
                f"Call propose_plan again with a plan that avoids it.{avoid}\n"
            )
        nudge = f"\n{problem}\n" if problem is not None else ""
        if not fresh:
            return "".join([ask, nudge])
        return "".join(
            [
                _READINGS_CONTEXT.format(readings=ledger)
                if ledger is not None
                else _envelope_context(
                    envelope,
                    refusal=refusal,
                    spec_id_elected=spec_id_elected,
                    spec_id_elected_label=spec_id_elected_label,
                    seed_of=seed_of,
                ),
                _section("What is already on the user's canvas", canvas_digest),
                _section("Earlier turns in this conversation", conversation_digest),
                _plan_context(prior_plan),
                f"\nQuestion: {cut_user_text(question, MAX_QUESTION_CHARS)}\n",
                _asks_section(asks),
                ask,
                nudge,
            ]
        )


def _failed_run_sentence(computed: Sequence[int] | None) -> str:
    """What a replan's failed run computed: the asks its finished terminal
    steps answer (`harness.graph._computed_asks`), by number, or `None` for
    a plan held to no asks. The new plan runs from the start, so it still
    answers every ask."""
    if computed is None:
        return "Executing the approved plan failed, and the new plan runs from the start."
    if not computed:
        return "Executing the approved plan failed before it computed any ask."
    numbers = ", ".join(str(number) for number in computed)
    return (
        f"Executing the approved plan failed after it computed asks {numbers}. The new plan "
        "runs from the start, so it still answers every ask."
    )


def _asks_section(asks: Sequence[str]) -> str:
    """The asks the answer is held to, numbered as `propose_plan` names them. Empty when none."""
    if not asks:
        return ""
    lines = "".join(f"{number}. {ask}\n" for number, ask in enumerate(asks))
    return (
        "The answer is held to these asks. Account for each exactly once: answer it in "
        "`answered`, or, when an `unsupported` answer earned its decline (see `declined`), "
        "decline it in `declined` with its number as `ask_ref`. When the plan computes "
        "part of an ask, do both: answer it with what is computed and decline the rest, "
        "naming that part in the decline's `ask`.\n" + lines
    )


def _check_branches(events: Sequence[Event], *, planner: str) -> None:
    """Raise unless every planner event this run appended carries `_PLANNER_BRANCH`.

    ADK's `_is_event_belongs_to_branch` (`google.adk.flows.llm_flows.contents`)
    returns True when either side's branch is empty, and otherwise when the
    invocation's branch path equals the event's or descends from it. An
    unbranched planner event therefore belongs to every branch, the
    orchestrator's included, which then reads the planner's calls and results
    as another agent's turns. The planner's `ctx.run_node` call passes
    `override_branch=_PLANNER_BRANCH` to prevent that, and this checks the
    effect rather than the argument: a dropped keyword and a future ADK
    release that stops honouring it both surface here.

    An empty slice, or one with no planner event, passes. A run resumed by the
    user's answer to a planner-side ask finishes the model turn the answer
    completed and may append no planner event, and a run that raised may have
    appended nothing yet.

    Raises:
        PlannerSeamError: If a planner event carries another branch. Nothing
            about the run is reported back. A planner whose events reach
            every branch has also read every unbranched event, so the plan it
            produced cannot be trusted to have been drafted from its own
            context alone.
    """
    strays = sorted(
        str(event.branch)
        for event in events
        if event.author == planner and event.branch != _PLANNER_BRANCH
    )
    if strays:
        raise PlannerSeamError(
            f"{len(strays)} planner event(s) this run appended carry a branch other than "
            f"{_PLANNER_BRANCH!r}: {strays!r}. ADK treats an unbranched event as belonging "
            "to every branch and filters a branched one by path, so the planner's events "
            f"stay out of other agents' requests only on exactly {_PLANNER_BRANCH!r}. The "
            f"planner's `ctx.run_node` call has to pass override_branch={_PLANNER_BRANCH!r}."
        )


def _pinnable_candidates(referral: dict[str, Any] | None) -> list[dict[str, str]]:
    """The `spec_id`-bearing candidates a referral names, shaped as
    `planner.agent.ask_user`'s own `candidates` argument
    (`{"spec_id": ..., "label": ...}`, `label` the candidate's `sentence`).

    A `no_data` referral's candidates (`translator.discovery.compact_source
    (selectable=False)`) carry no `spec_id`, since they describe what search
    found nearby rather than a source pinnable without a fresh search, so
    they never appear here. `referral` is `None`, or carries no candidates,
    for a `translate` response this entry never saw, or one whose kind is
    not among `translator.referral.MAY_CARRY_CANDIDATES`.
    """
    if not isinstance(referral, dict):
        return []
    candidates = referral.get("candidates")
    if not isinstance(candidates, list):
        return []
    return [
        {
            "spec_id": str(candidate["spec_id"]),
            "label": _truncate(str(candidate.get("sentence") or ""), _ENVELOPE_LABEL_CAP),
        }
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("spec_id")
    ]


def _clarifying_question_args(candidates: list[dict[str, str]]) -> dict[str, Any]:
    """The `adk_request_input` call args for a deadline that runs out on a
    referral carrying pinnable candidates: the shape `planner.agent.ask_user`
    declares, built here from the referral's candidates since the model has run out of
    time to compose it. It recommends no candidate, since nothing has weighed them.
    The message reaches the user, so it names no requirement id, referral kind
    or spec_id; each candidate's label says what it is.
    """
    message = (
        "This turn ran out of time before the plan settled on a source. "
        "Which of these did you mean?"
    )
    return {"message": message, "candidates": candidates}


def _requirements_without_discovery(state: SessionState) -> list[str]:
    """The current draft's evidence requirement ids with no
    `Workflow.evidence_bindings` entry, every one of them when the draft has
    no `workflow` at all since translation has not run.

    Read fresh off session state: `PlannerSeam._deadline_response` calls this
    at the moment the deadline passes, from whatever `propose_plan` last
    recorded (`planner.plan.load_plan`), which may be `None` when the
    deadline runs out before any draft exists.
    """
    plan = load_plan(state)
    if plan is None:
        return []
    bound = (
        {binding.requirement_id for binding in plan.workflow.evidence_bindings}
        if plan.workflow is not None
        else set()
    )
    return [
        str(requirement.id)
        for requirement in plan.evidence_requirements
        if requirement.id not in bound
    ]


def _readings_ledger(readings: Sequence[Mapping[str, Any]]) -> tuple[str, tuple[str, ...]]:
    """The conversation's search readings as the planner's first message
    renders them, each spec_id by its newest reading only, and every spec_id
    that text names (`render_readings`). Over `_READINGS_BUDGET` the oldest
    readings are folded, then dropped and counted."""
    if not readings:
        return "(no search has run in this conversation)", ()
    rendered = render_readings(
        newest_per_spec_id([SearchReading.model_validate(reading) for reading in readings]),
        budget=_READINGS_BUDGET,
    )
    return rendered.text, rendered.spec_ids


# The reading statuses whose own spec_id a search settled on: a `definite`
# part, and a candidate `fetch` read (`chosen`).
_SETTLED_STATUSES = frozenset({"definite", "chosen"})


def _named(readings: Iterable[Mapping[str, Any]]) -> set[str]:
    """Every spec_id `readings` name, as their own or as a candidate."""
    return {
        spec_id
        for reading in readings
        for spec_id in named_spec_ids(SearchReading.model_validate(reading))
    }


def _reading_pin_sources(
    readings: Sequence[Mapping[str, Any]],
    shown: Sequence[str],
    *,
    blocks: Mapping[str, Any],
    earlier: Collection[str],
    warnings: Mapping[str, Any],
) -> dict[str, PinSource]:
    """A `PinSource` for each of `shown`, from the block and warnings `search` stored for it.

    A spec_id in `earlier`, which an earlier turn's reading names and no
    search this turn returned, is marked `earlier_turn` and holds no block:
    a pin of it reads the one that turn stored (`PinSource.spec_block`), so
    the turn's session never re-stores it. Any other takes this turn's block
    (`blocks`). Settled when a reading with a `_SETTLED_STATUSES` status
    holds it as its own spec_id; a spec_id only ever offered as a candidate
    is not.
    """
    settled = {
        reading.get("spec_id") for reading in readings if reading.get("status") in _SETTLED_STATUSES
    }
    sources: dict[str, PinSource] = {}
    for spec_id in shown:
        from_earlier = spec_id in earlier
        sources[spec_id] = PinSource(
            block=None if from_earlier else blocks.get(spec_id),
            warnings=_warning_codes(warnings.get(spec_id)),
            settled=spec_id in settled,
            earlier_turn=from_earlier,
        )
    return sources


def _envelope_pin_sources(
    envelope: dict[str, Any], *, refusal: dict[str, Any] | None, spec_id_elected: bool
) -> dict[str, PinSource]:
    """The resolve envelope's own spec_id as a settled `PinSource`, when the
    planner was shown it as a pin (`_exposed_spec_id`); empty otherwise."""
    spec_id = _exposed_spec_id(envelope, refusal=refusal, spec_id_elected=spec_id_elected)
    if spec_id is None:
        return {}
    block = envelope.get("spec_block")
    return {
        spec_id: PinSource(
            block=block if isinstance(block, dict) else None,
            warnings=_warning_codes(envelope.get("warnings")),
            settled=True,
        )
    }


def _warning_codes(warnings: object) -> tuple[str, ...]:
    """The `code` of each warning dict in `warnings`, a list as search returns it."""
    if not isinstance(warnings, list):
        return ()
    codes = (warning.get("code") for warning in warnings if isinstance(warning, dict))
    return tuple(str(code) for code in codes if code)


def _envelope_spec_disposition(
    envelope: dict[str, Any], *, refusal: dict[str, Any] | None, spec_id_elected: bool
) -> tuple[bool, bool]:
    """`(omit, elected)`: whether `_envelope_context` leaves the envelope's
    own `spec_id` out of what the planner sees, and whether that is because
    the user elected it answering a `clarify` round.

    Shared with `_exposed_spec_id`, below, so the two can never disagree
    about what the model was shown: `_exposed_spec_id` uses `elected` to
    decide which pin `propose_plan` may synthesize a `Decision` for. See
    `_envelope_context` for why each omission reason fires. A seeded
    envelope always omits it: its seeds name every spec_id, and the model
    records each pin as the user's (`_SEEDED_REASON`).
    """
    if envelope.get("seeds"):
        return True, False
    parts = envelope.get("parts", 1)
    refused = bool(_refused_specs(refusal))
    elected = spec_id_elected and not refused and envelope.get("spec_id") is not None
    omit = refused or (isinstance(parts, int) and parts > 1 and not elected)
    return omit, elected


def _exposed_spec_id(
    envelope: dict[str, Any], *, refusal: dict[str, Any] | None, spec_id_elected: bool
) -> str | None:
    """The envelope's own `spec_id` as `_envelope_context` rendered it to the
    model, or `None` when it was omitted, including on the elected branch
    where `_envelope_context` still renders it.

    `PlannerSeam.__call__` records this in session state
    (`PIN_SOURCES_STATE_KEY`) so `propose_plan` (`planner.agent`) can
    synthesize a `resolved_by=planner_default` `Decision` for a verbatim pin
    of it (see `_synthesize_source_decisions`) without ever manufacturing one
    for an id the model was not shown.

    The elected branch (`parts > 1`, `spec_id_elected`, not refused) shows
    the id through `_ELECTED_SPEC_REASON`, but returns `None` here: that text
    tells the model to record the pin itself with `resolved_by=user`, since
    the user chose it answering a `clarify` round. A synthesized
    `planner_default` decision would mislabel a choice the user made.
    """
    omit, elected = _envelope_spec_disposition(
        envelope, refusal=refusal, spec_id_elected=spec_id_elected
    )
    parts = envelope.get("parts", 1)
    if omit or (isinstance(parts, int) and parts > 1 and elected):
        return None
    spec_id = envelope.get("spec_id")
    return spec_id if isinstance(spec_id, str) else None


def _envelope_context(
    envelope: dict[str, Any],
    *,
    refusal: dict[str, Any] | None = None,
    spec_id_elected: bool = False,
    spec_id_elected_label: str | None = None,
    seed_of: Mapping[str, str] | None = None,
) -> str:
    """Render the resolve envelope for the planner's prompt.

    Two independent reasons omit the envelope's own `spec_id` from what
    the planner sees, rather than merely warning against it -- a field
    present and forbidden is the shape a model reliably tries anyway once
    it has no other candidate to reach for. Either can fire alone, or
    both together (`_envelope_spec_disposition` is the one test behind both
    this and `_exposed_spec_id`):

    - `parts > 1` and not elected: `parts` is on the wire before any
      requirement exists (`graph.py`'s `choose_depth`, read a node earlier
      by `route_status`), so it is the test this uses rather than a
      requirement count. The envelope's one `spec_id` is a question-wide
      answer and can be right for at most one part of a `parts > 1`
      question -- unless `spec_id_elected` says the user actually chose it
      answering a `clarify` round, in which case it is kept and named for
      the part it matches (`_ELECTED_SPEC_REASON`) instead of omitted.
    - `refusal` names this envelope's own `spec_id`, or a wave sibling's, as
      refused (see `_refused_specs`): a `RediscoveryNeeded` replan over the
      `execute -> planner` edge, which does not re-probe. This omission wins
      over an election on any `parts` count: the executor's refusal is a
      fact about the spec, while an election only says which reading the
      user meant.

    A refused spec_id that `seed_of` maps to a seed is not treated as
    refused here: the seed stays listed, and `PlannerSeam._message`'s ask
    tells the planner to keep it pinned (`_keep_seeds_sentence`).
    """
    parts = envelope.get("parts", 1)
    reasons = ""
    omit, elected = _envelope_spec_disposition(
        envelope, refusal=refusal, spec_id_elected=spec_id_elected
    )
    seeds = envelope.get("seeds")
    refused, _ = _split_seeded(_refused_specs(refusal), seed_of or {})
    if seeds:
        reasons += _SEEDED_REASON.format(ceiling=frame_row_ceiling())
    elif isinstance(parts, int) and parts > 1:
        if elected:
            reasons += _ELECTED_SPEC_REASON.format(
                parts=parts,
                spec_id=envelope.get("spec_id"),
                label=spec_id_elected_label or envelope.get("spec_id"),
            )
        else:
            reasons += _MULTI_PART_REASON.format(parts=parts)
    if refused:
        reasons += _refused_spec_reason(refused)
    rendered = (
        {key: value for key, value in envelope.items() if key != "spec_id"} if omit else envelope
    )
    text = _render_envelope(rendered)
    if seeds:
        text += "\n" + _render_seeds(seeds)
    return _ENVELOPE_CONTEXT.format(envelope=text, reasons=reasons)


# The facts of one seed's identity `_render_seeds` prints under it, as
# `_render_envelope` prints a searched envelope's own.
_SEED_FACT_LINES = (
    "wide",
    "stat_var_shape_kind",
    "stat_var_item_count",
    "entity_roles",
    "stat_var_served_count",
)


def _render_seeds(seeds: list[dict[str, Any]]) -> str:
    """A seeded turn's datasets, one block each: spec_id, the sentence the
    user saw, its window, the year and rows Search showed, its coverage
    counts, its width and the facts behind it, the entity axes it leaves open,
    and the operations asked of it (`harness.seeds`'s `_envelope_seed`). An operation over a party reads
    "across" that party (`_operation_text`).

    Every seed stays listed, a refused one too: the user chose it, so the
    plan keeps it pinned (`_keep_seeds_sentence`). Seeds and each seed's
    operations are capped and counted, and every field is cut, like the
    envelope's own.
    """
    lines = [f"Seeded datasets ({len(seeds)}):"]
    for seed in seeds[:MAX_SEEDS]:
        sentence = _truncate(str(seed.get("sentence") or ""), _ENVELOPE_LABEL_CAP)
        lines.append(
            f"- spec_id: {_envelope_field(seed['spec_id'])}"
            + (f" -- {sentence}" if sentence else "")
        )
        lines.extend(
            f"  {key}: {_envelope_field(seed[key])}"
            for key in ("window", "showed", "coverage")
            if seed.get(key)
        )
        lines.extend(
            f"  {key}: {_envelope_field(_fact(seed, key))}"
            for key in _SEED_FACT_LINES
            if seed.get(key) is not None
        )
        if seed.get("open_entity_axes"):
            axes = _open_entity_axes_text(seed["open_entity_axes"])
            lines.append(f"  open_entity_axes: {_envelope_field(axes)}")
        operations = seed.get("operations") or []
        if not operations:
            lines.append("  operations: (none)")
        lines.extend(
            f"  operation: {_operation_text(operation)}"
            for operation in operations[:_ENVELOPE_OPERATION_CAP]
        )
        if len(operations) > _ENVELOPE_OPERATION_CAP:
            lines.append(f"  and {len(operations) - _ENVELOPE_OPERATION_CAP} more")
    if len(seeds) > MAX_SEEDS:
        lines.append(f"- and {len(seeds) - MAX_SEEDS} more")
    return "\n".join(lines)


# The party each entity axis of a Search operation's `over` names, with the
# column `open_entity_axes` lists it under. `category` and `year` are no
# party: an operation over one names its axis alone.
_OPERATION_PARTIES = {
    "place": "observationAbout, the place",
    "origin": "origin, the giving party",
    "destination": "destination, the receiving party",
}


def _operation_text(operation: Mapping[str, Any]) -> str:
    """One seed operation as the planner reads it.

    `rank across origin, the giving party ("top 5 donors")` for an operation
    over an entity axis, which a requirement states in `across`;
    `growth over year ("since 2010")` for any other axis; and
    `forecast to 2050 ("...")` for a forecast with a target year.
    """
    text = _envelope_field(operation["verb"])
    over = operation.get("over")
    if over in _OPERATION_PARTIES:
        text += f" across {_OPERATION_PARTIES[over]}"
    elif over:
        text += f" over {_envelope_field(over)}"
    if operation.get("target_year") is not None:
        text += f" to {_envelope_field(operation['target_year'])}"
    phrase = _truncate(str(operation.get("phrase") or ""), _ENVELOPE_LABEL_CAP)
    return f'{text} ("{phrase}")'


def _render_envelope(envelope: dict[str, Any]) -> str:
    """The resolve envelope as a digest, its verdicts rendered one to a line.

    Every field `graph_envelope` (`harness.tool_contracts`) puts on the
    envelope is rendered, because each one is an input to a decision this
    prompt asks the model to make:

    - `status` and `spec_id`: what the resolve settled on, and the one
      source the requirement may be pinned to when the question has one part.
    - `choice_points` and `wide`: the methodological forks search settled
      by default, and whether anything downstream can narrow the identity to
      one series. Both are the depth policy's own inputs
      (`harness.depth.decide_depth`), so a plan that proposes a different
      reading from them is proposing against the reading the user's turn was
      routed on.
    - `parts`: how many variables the question decomposed into.
    - The pin-predicate facts (`stat_var_shape_kind`, `stat_var_item_count`,
      `entity_roles`, `stat_var_served_count`) ride alongside `wide` for the same
      reason `graph_envelope` carries them, sparing a later reader of `wide`
      another trip into the raw envelope.
    - `open_entity_axes`, each entity axis `search` left open with its member count, so the
      planner states the party a rank or total covers in `across`. An empty
      list renders nothing.
    - `candidates`, each by `spec_id` and `label`, capped and counted.
    - `skills`, each by `id` and `description`, capped and counted: the
      skills the resolve names for its results, which the planner loads with
      `get_skill` when a description bears on the question.
    - `warnings`, each by `severity`, `code` and `message`, capped and
      counted. `severity` leads the line because `harness.graph` reads it
      (`graph.py`'s `degraded=any(w.get("severity") == "degraded" ...)`) to
      tell a resolve that degraded from one that merely noted something, and
      a warning rendered without it leaves `info`, `warn` and `degraded`
      reading the same.

    An unrecognized field renders under its own name. The envelope grows
    fields upstream of this module, and a silent omission would hand the
    model a summary it could not tell was incomplete.
    """
    lines: list[str] = []
    status = envelope.get("status")
    if status is not None:
        lines.append(f"- status: {_envelope_field(status)}")
    spec_id = envelope.get("spec_id")
    if spec_id is not None:
        lines.append(f"- spec_id: {_envelope_field(spec_id)}")
    lines.extend(
        f"- {key}: {_envelope_field(envelope[key])}"
        for key in ("parts", "choice_points", "wide")
        if key in envelope
    )
    lines.extend(
        f"- {key}: {_envelope_field(_fact(envelope, key))}"
        for key in (
            "stat_var_shape_kind",
            "stat_var_item_count",
            "entity_roles",
            "stat_var_served_count",
        )
        if envelope.get(key) is not None
    )
    open_entity_axes = envelope.get("open_entity_axes")
    if open_entity_axes:
        lines.append(
            f"- open_entity_axes: {_envelope_field(_open_entity_axes_text(open_entity_axes))}"
        )

    # `candidates`, `skills` and `warnings` are read with `is not None`, like
    # every other field here, not with `or []`: an absent key renders nothing,
    # while a key the envelope carries and leaves empty renders as a
    # counted empty line. `graph_envelope` sets all three keys on every branch
    # it returns, so today the two cases are always identical, but a producer
    # that omitted any one would read to the model exactly like a resolve
    # that found nothing.
    candidates = envelope.get("candidates")
    if isinstance(candidates, list):
        lines.append(f"- candidates ({len(candidates)}):")
        for candidate in candidates[:_ENVELOPE_CANDIDATE_CAP]:
            if not isinstance(candidate, dict):
                lines.append(f"  - {_envelope_field(candidate)}")
                continue
            label = _truncate(str(candidate.get("label") or ""), _ENVELOPE_LABEL_CAP)
            pinned = f" -- {label}" if label else ""
            wide = " (wide)" if candidate.get("wide") else ""
            lines.append(f"  - {_envelope_field(candidate.get('spec_id'))}{wide}{pinned}")
        if len(candidates) > _ENVELOPE_CANDIDATE_CAP:
            lines.append(f"  - and {len(candidates) - _ENVELOPE_CANDIDATE_CAP} more")
    elif candidates is not None:
        lines.append(f"- candidates: {_envelope_field(candidates)}")

    skills = envelope.get("skills")
    if isinstance(skills, list):
        lines.append(f"- skills ({len(skills)}):")
        for skill in skills[:_ENVELOPE_SKILL_CAP]:
            if not isinstance(skill, dict):
                lines.append(f"  - {_envelope_field(skill)}")
                continue
            description = _truncate(
                str(skill.get("description") or ""), _ENVELOPE_SKILL_DESCRIPTION_CAP
            )
            about = f" -- {description}" if description else ""
            lines.append(f"  - {_envelope_field(skill.get('id'))}{about}")
        if len(skills) > _ENVELOPE_SKILL_CAP:
            lines.append(f"  - and {len(skills) - _ENVELOPE_SKILL_CAP} more")
    elif skills is not None:
        lines.append(f"- skills: {_envelope_field(skills)}")

    warnings = envelope.get("warnings")
    if isinstance(warnings, list):
        lines.append(f"- warnings ({len(warnings)}):")
        for warning in warnings[:_ENVELOPE_WARNING_CAP]:
            if not isinstance(warning, dict):
                lines.append(f"  - {_envelope_field(warning)}")
                continue
            code = _envelope_field(warning.get("code"))
            severity = _envelope_field(warning.get("severity"))
            message = _truncate(str(warning.get("message") or ""), _ENVELOPE_WARNING_TEXT_CAP)
            lines.append(
                f"  - [{severity}] {code}: {message}" if message else f"  - [{severity}] {code}"
            )
        if len(warnings) > _ENVELOPE_WARNING_CAP:
            lines.append(f"  - and {len(warnings) - _ENVELOPE_WARNING_CAP} more")
    elif warnings is not None:
        lines.append(f"- warnings: {_envelope_field(warnings)}")

    lines.extend(
        f"- {key}: {_envelope_field(envelope[key])}"
        for key in sorted(set(envelope) - _ENVELOPE_KEYS)
    )
    return "\n".join(lines) if lines else "(the envelope carried nothing)"


def _plan_context(prior_plan: dict[str, Any] | None) -> str:
    """The plan being changed, rendered field by field as a digest.

    The title above it comes from `_section`, matching the name `_message`'s
    docstring and the tests that read the brief use for that section. This
    function supplies only the body.

    Every field of the plan is rendered. This renderer shrinks only the text.
    The planner's model decides an amendment from what it reads here, and it
    cannot read session state, so a field dropped here is a field the next
    proposal cannot preserve. `Decision.description` and `Decision.rationale`
    are the only two cut, and they are the only two `planner.plan` compares
    against nothing. `chosen` is checked against `options`, and every
    requirement field is checked against the decisions it cites. A
    decision's `fixes` renders as the names of the fields it fixes, the form
    `propose_plan` takes; their values are on the requirements citing it. A
    shortened `description` or `rationale` is prose the model is being asked
    to rewrite anyway. A shortened `measure` would be a different plan.
    """
    if not prior_plan:
        return ""
    # The leading newline is what `_section` puts before its own title, so
    # this section is spaced off whatever digest precedes it the same way.
    lines = ["\nThe plan being changed:"]
    question = prior_plan.get("question")
    if question:
        lines.append(f"Question: {question}")
    version = prior_plan.get("version")
    previous = prior_plan.get("previous_version")
    lineage = f"version {version}" if version is not None else "a plan"
    if previous is not None:
        lineage += f" (amended from version {previous})"
    lines.append(lineage)
    lines.extend(
        f"{key}: {_scalar(prior_plan[key])}"
        for key in ("status", "approved_revision")
        if prior_plan.get(key) is not None
    )

    decisions = prior_plan.get("decisions") or []
    lines.append(f"Decisions ({len(decisions)}):")
    for decision in decisions:
        if not isinstance(decision, dict):
            lines.append(f"- {_scalar(decision)}")
            continue
        lines.append(f"- {_scalar(decision.get('slot'))}: {_scalar(decision.get('chosen'))}")
        description = _truncate(str(decision.get("description") or ""), _PLAN_DESCRIPTION_CAP)
        if description:
            lines.append(f"  description: {description}")
        options = decision.get("options") or []
        lines.append("  options: " + _list_text(options))
        rationale = _truncate(str(decision.get("rationale") or ""), _PLAN_RATIONALE_CAP)
        if rationale:
            lines.append(f"  rationale: {rationale}")
        lines.append(f"  resolved_by: {_scalar(decision.get('resolved_by'))}")
        fixes = decision.get("fixes")
        if isinstance(fixes, dict) and (named := [k for k, v in fixes.items() if v is not None]):
            lines.append("  fixes: " + _list_text(named))

    requirements = prior_plan.get("evidence_requirements") or []
    lines.append(f"Evidence requirements ({len(requirements)}):")
    for requirement in requirements:
        if not isinstance(requirement, dict):
            lines.append(f"- {_scalar(requirement)}")
            continue
        kind = _scalar(requirement.get("kind"))
        req_id = _scalar(requirement.get("id"))
        cites = _list_text(requirement.get("decision_ids") or [])
        lines.append(f"- {req_id} ({kind}) cites {cites}")
        lines.append("  " + _field_line(requirement))

    # `translation` and `workflow` render as a summary. Together they are
    # 13,883 chars of a nine-requirement plan, more than every other field
    # combined. `_workflow_lines` keeps the two parts a later proposal has a
    # decision to make about: `workflow.sources`, the datasets the user
    # approved, on which a requirement's `source_spec_id` pin has to rest;
    # and `workflow.evidence_bindings`, which say which requirement each
    # source answers. The compiler's own mechanics are left out: each step's
    # tool, its literal `args` and `bindings`, and the operations they were
    # built from. `_PLANNING` tells the planner to "decide the methodology and
    # leave the mechanics alone: which data and which operations carry your
    # decisions out is worked out separately, after you propose, so name no
    # tool", and `Plan.amend` sets both fields to `None`, so the translator
    # recompiles them from whatever this prompt produces.
    workflow = prior_plan.get("workflow")
    if isinstance(workflow, dict):
        lines.extend(_workflow_lines(workflow))

    known = {
        "question",
        "version",
        "previous_version",
        "status",
        "approved_revision",
        "decisions",
        "evidence_requirements",
        "translation",
        "workflow",
    }
    lines.extend(f"{key}: {_scalar(prior_plan[key])}" for key in sorted(set(prior_plan) - known))
    return "\n".join(lines) + "\n"


def _field_line(requirement: dict[str, Any]) -> str:
    """Every field of one requirement but `id`, `kind` and its citation list."""
    skip = {"id", "kind", "decision_ids"}
    return " ".join(
        f"{key}={_scalar(value)}" for key, value in requirement.items() if key not in skip
    )


def _workflow_lines(workflow: dict[str, Any]) -> list[str]:
    """A compiled workflow's selected sources and its evidence bindings.

    `SelectedSource.sentence` is kept. It is the resolved reading, in the
    words search answered, and a proposal that changes a requirement's
    measure has to see what the approved source said it was. `searched_as`
    and `has_data` are dropped. Both are discovery's own bookkeeping, and the
    step and binding wiring is the mechanic above.
    """
    sources = workflow.get("sources")
    if not isinstance(sources, list) or not sources:
        lines = ["Compiled sources: (none)"]
    else:
        named = []
        for source in sources:
            if not isinstance(source, dict):
                named.append(_scalar(source))
                continue
            sentence = _truncate(str(source.get("sentence") or ""), _PLAN_SOURCE_SENTENCE_CAP)
            spec_id = _scalar(source.get("spec_id"))
            named.append(f"{spec_id} -- {sentence}" if sentence else spec_id)
        lines = ["Compiled sources:", *(f"  {name}" for name in named)]
    bindings = workflow.get("evidence_bindings")
    if isinstance(bindings, list):
        lines.extend(_binding_line(binding) for binding in bindings)
    return lines


def _binding_line(binding: Any) -> str:
    """One evidence binding, naming which requirement's evidence came from which source.

    A pinned dataset requirement carries its own `source_spec_id`, which
    `_field_line` already renders, so its source is readable without this
    line. An unpinned requirement states `source_spec_id` as null, and
    `Workflow.sources` is in the compiler's canonical retrieval order
    (`compiler.compile`'s `_index_operations`) rather than the order
    requirements are declared in, so for those nothing else relates the two.
    """
    if not isinstance(binding, dict):
        return f"- evidence: {_scalar(binding)}"
    requirement_id = _scalar(binding.get("requirement_id"))
    spec_id = _scalar(binding.get("spec_id"))
    step_id = _scalar(binding.get("step_id"))
    return f"- evidence: {requirement_id} from {spec_id} (step {step_id})"


def _scalar(value: Any) -> str:
    """One value as compact text, JSON for anything with structure.

    Non-ASCII characters are written as they are, so a rendered value has as
    many characters as `Plan` counted for it
    (`planner.plan.MAX_PLAN_RECORD_CHARS`).
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, separators=(",", ":"), ensure_ascii=False)


def _open_entity_axes_text(open_entity_axes: Any) -> Any:
    """`open_entity_axes` as `origin (55 members)`, comma-joined; anything else as it came."""
    if not isinstance(open_entity_axes, list) or not all(
        isinstance(axis, dict) for axis in open_entity_axes
    ):
        return open_entity_axes
    parts = []
    for axis in open_entity_axes:
        members = axis.get("members")
        count = (
            "members not counted"
            if not isinstance(members, int)
            else f"{members} member{'' if members == 1 else 's'}"
        )
        parts.append(f"{_scalar(axis.get('column'))} ({count})")
    return ", ".join(parts)


def _fact(reading: Mapping[str, Any], key: str) -> Any:
    """`reading[key]`, except that a `stat_var_item_count` which
    `stat_var_member_count_exact` marks as a floor reads "at least" that many
    StatVars, as `compact_source` counts a source's
    (`translator.discovery.stat_var_count_text`)."""
    value = reading[key]
    floor = reading.get("stat_var_member_count_exact") is False
    if key == "stat_var_item_count" and floor and isinstance(value, int):
        return stat_var_count_text(value, floor=True)
    return value


def _envelope_field(value: Any) -> str:
    """One envelope field as text, cut to `_ENVELOPE_FIELD_CAP` characters."""
    return _truncate(_scalar(value), _ENVELOPE_FIELD_CAP)


def _list_text(values: Any) -> str:
    """A list as a comma-joined run of scalars, JSON for anything nested."""
    if not isinstance(values, list):
        return _scalar(values)
    return ", ".join(_scalar(value) for value in values) if values else "(none)"


def _truncate(text: str, cap: int) -> str:
    """`text` cut to `cap` chars with a trailing ellipsis, and nothing else.

    Same convention as `harness.studio_translate._clipped`. The ellipsis is
    inside the cap, so the rendered length is the cap and not the cap plus one.
    """
    return text if len(text) <= cap else text[: cap - 1] + "…"


def _earned_by_refusal(refusal: dict[str, Any] | None) -> set[UnsupportedOperation]:
    """The declines a replan's `refusal` earns: one per stop in it, the
    primary's or an `also_stopped` sibling's, whose `code` is in
    `_UNSUPPORTED_BY_RUN_TIME_CODE`.

    Only `MethodChoiceNeeded`'s shape (`graph.py`'s `_refusal_shape`)
    carries a tool's `code` at its top level.
    """
    if not isinstance(refusal, dict):
        return set()
    also_stopped = refusal.get("also_stopped")
    payloads = [refusal, *(also_stopped if isinstance(also_stopped, list) else ())]
    return {
        _UNSUPPORTED_BY_RUN_TIME_CODE[code]
        for payload in payloads
        if isinstance(payload, dict)
        and isinstance(code := payload.get("code"), str)
        and code in _UNSUPPORTED_BY_RUN_TIME_CODE
    }


def _earned_decline_sentence(earned: Collection[UnsupportedOperation]) -> str:
    """What a replan's message says about the declines its refusal earned. Empty when none."""
    if not earned:
        return ""
    codes = ", ".join(f"`{operation.value}`" for operation in sorted(earned))
    return (
        f" The refusal means no operation computes what that step was asked to, so the "
        f"plan may decline the ask it served in `declined` with code {codes}."
    )


def _refused_specs(refusal: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Every `spec_id` a refusal names as refused, and why: the primary's own,
    then each of its `also_stopped` siblings' (`graph.py`'s `_replan_refusal`),
    in that order.

    Reads `refusal` and each `also_stopped` entry as the opaque per-kind dict
    `_message`'s `elif refusal:` branch already treats them as: only
    `RediscoveryNeeded`'s shape carries a `spec_id` today, at the top
    level, with why in its nested `refusal.message`. `MethodChoiceNeeded` and
    `Refused` name neither. `RediscoveryNeeded` fires for
    `executor.classify.REDISCOVERY_CODES` (`upstream_version_changed`,
    `definition_not_held`, `open_role_unavailable`) alone;
    `over_ceiling`/`member_cardinality` is a
    `Refused` `STOP` (`classify.py`'s own `STOP_CODES` default) and so
    carries no `spec_id` here. This does not assume that shape survives
    unchanged.
    """
    if not isinstance(refusal, dict):
        return []
    also_stopped = refusal.get("also_stopped")
    payloads = [refusal, *(also_stopped if isinstance(also_stopped, list) else ())]
    return [spec for payload in payloads for spec in _one_refused_spec(payload)]


def _one_refused_spec(payload: object) -> list[tuple[str, str]]:
    """The `(spec_id, why)` one refusal payload names as refused, if any."""
    if not isinstance(payload, dict):
        return []
    spec_id = payload.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id:
        return []
    nested = payload.get("refusal")
    message = nested.get("message") if isinstance(nested, dict) else None
    detail = message if isinstance(message, str) and message else "refused"
    return [(spec_id, detail)]


def _refused_spec_reason(refused: list[tuple[str, str]]) -> str:
    """The reason text naming every refused spec left out of the envelope."""
    if len(refused) == 1:
        spec_id, detail = refused[0]
        return _REFUSED_SPEC_REASON.format(
            spec_id=spec_id, detail=cut_user_text(detail, MAX_QUESTION_CHARS)
        )
    named = "; ".join(f"{spec_id!r} ({detail})" for spec_id, detail in refused)
    return _REFUSED_SPECS_REASON.format(named=cut_user_text(named, MAX_QUESTION_CHARS))


def _avoid_sentence(refused: list[tuple[str, str]]) -> str:
    """The sentence naming every refused spec the planner must not propose again."""
    if not refused:
        return ""
    if len(refused) == 1:
        spec_id, detail = refused[0]
        detail = cut_user_text(detail, MAX_QUESTION_CHARS)
        return f" Do not propose {spec_id!r} as the source again -- it was just refused ({detail})."
    named = "; ".join(f"{spec_id!r} ({detail})" for spec_id, detail in refused)
    named = cut_user_text(named, MAX_QUESTION_CHARS)
    return f" Do not propose any of these sources again -- each was just refused: {named}."


def _seed_of(state: SessionState) -> dict[str, str]:
    """Each seeded spec_id this turn holds, a seed or its copy over a plan's
    period (`planner.plan.seeded_sources`), mapped to the seed's own."""
    return {item.source.spec_id: seed_of(item) for item in seeded_sources(state)}


def _split_seeded(
    refused: list[tuple[str, str]], seed_of: Mapping[str, str]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """`refused` split into the specs to avoid and the seeds to keep pinned.

    A refused seeded spec_id is replaced by its seed's (`seed_of`), once per
    seed, with the first reason given for it.
    """
    avoid: list[tuple[str, str]] = []
    keep: dict[str, str] = {}
    for spec_id, detail in refused:
        seed = seed_of.get(spec_id)
        if seed is None:
            avoid.append((spec_id, detail))
        else:
            keep.setdefault(seed, detail)
    return avoid, list(keep.items())


def _keep_seeds_sentence(kept: list[tuple[str, str]]) -> str:
    """The sentence telling the planner to keep each refused seed pinned."""
    if not kept:
        return ""
    if len(kept) == 1:
        seed, detail = kept[0]
        return _KEEP_SEED_SENTENCE.format(
            seed=seed, detail=cut_user_text(detail, MAX_QUESTION_CHARS)
        )
    named = "; ".join(f"{seed!r} ({detail})" for seed, detail in kept)
    return _KEEP_SEEDS_SENTENCE.format(named=cut_user_text(named, MAX_QUESTION_CHARS))


def _section(title: str, body: dict[str, Any] | None) -> str:
    return f"\n{title}:\n{json.dumps(body)}\n" if body else ""


__all__ = ["PlannerSeam", "PlannerSeamError"]
