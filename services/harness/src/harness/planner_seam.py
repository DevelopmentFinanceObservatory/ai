"""The graph's planner seam, backed by the planner agent.

An adapter over `planner.agent.create_planner_agent`, built with
translation attached by `harness.agents.create_planner_with_translator`: the
planner runs as a dynamically-scheduled child of the graph's own node,
through `Context.run_node`, and this module reads what it leaves in session
state and translates it into `harness.graph.seams.PlannerModel`'s contract.

The planner's own tools settle methodology and carry it to a compiled
workflow (`propose_plan`, whose `translation` is either the compiled outcome
or the `Referral` translation stopped on); a translation that compiles
declares the plan ready for the user's answer in code
(`planner.agent.declare_ready`), and ends the planner's run. This module never
constructs a `Plan` itself. An entry that ends without a plan ready ends on a
refusal whose sentence comes from `harness.sentences`.

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
(`studio_state.ConversationState.session_id`), so the flag starts every
turn unset. One turn can still enter the planner for a first draft more
than once: the orchestrator's agent may call `plan` again after an earlier
entry handed back (`harness.graph`'s `plan_done`). So `__call__` clears
this flag, along with `planner.plan.PLAN_STATE_KEY`, on a genuine
first-draft entry (`prior_plan`, `refusal` and `amendment` all `None`,
the shape the graph's planner node passes only after `plan_start`):
without it, the later first-draft entry would read as a continuing call
on a session that already holds the earlier entry's plan, get only the
continuing message instead of the full brief, and be pushed to amend that
plan or have `starts_new_plan` refused as answering a different question.
`_persist_durable` (see `__call__`) is what makes both this flag's flip and its clear survive a
pause-and-resume across `adk_request_input`: a plain `ctx.session.state`
write is visible in-process but never reaches a session event's
`state_delta`, so a fresh `Runner.run_async` call after a resume would
otherwise read a write back as it was before this call, undoing it. The
first call of a session (or a session's first entry after such a clear)
renders the full brief --
the search readings, canvas digest, conversation digest, the plan being
changed, the question. Every later call in that same session -- the `amend`
edge (`approve_n -> planner_n`) and the `replan` edge
(`execute_n -> planner_n`) -- renders only what is new: the amendment or
refusal ask.
Resending the full brief on every call would restate a history the model
can already see with `include_contents` pinned; a continuing call sends
only the new thing.

`prior_plan` is dropped from every continuing render too: on the `amend`
edge it is the plan the model itself drafted and submitted last call,
untouched since (`approve_n`'s own `"amend"` route never calls
`Plan.approve`), already in its own tool-call history. On the `replan`
edge it is `execute`'s own copy of that same plan (`harness.graph.execute`),
which by then differs from what the model drafted in exactly three fields:
`status`, `approved_revision` and `approved_at`. All three come from
upstream of `execute` itself -- `approve_n`'s `"approved"` route calls
`Plan.approve`, which sets them before `execute_approved_plan` ever runs;
`execute_approved_plan` then only ever advances `status` further, through
`start_execution`, `complete`/`fail`, each a bare
`model_copy(update={"status": ...})` that touches nothing else. Dropping
them from the continuing render is still right: `status` states where the
plan sits in the approve/execute lifecycle, which the refusal text this
seam sends already says in its own words; `approved_revision` is
`plan_revision(self)`, a content hash of the very decisions and
requirements the model itself just drafted, not a new fact about the
world; and `approved_at` is when the user answered, which no proposal
acts on. None is something the refusal text does not already restate.

`_plan_context` renders the plan being changed as a digest, and
`_READINGS_BUDGET` holds the search readings. With a realistic
nine-requirement plan, compiled, and readings at their whole budget, the plan
section is 8,438 characters where its raw JSON is 22,609, and the readings
section is 3,486. The whole brief renders 12,627 characters, and
`test_context_budget.py` holds it to a `SEAM_BRIEF_CEILING` of 15,800, 25%
above that. These are character counts of the brief alone.
"""

from __future__ import annotations

import contextvars
import copy
import json
import logging
import time
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from common.telemetry import current_trace_id
from compiler.compile import CompileIssueCode
from compiler.surface import ToolSurface
from executor.classify import ExecutorCode
from google.adk import Context, Event
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.genai import types
from planner.agent import PROPOSE_PLAN, ToolContextSession, declare_ready, with_source_reads
from planner.operations import OperationKind
from planner.plan import (
    AMENDMENT_ENTRY_STATE_KEY,
    MAX_QUESTION_CHARS,
    OPTIONAL_DECLINES_STATE_KEY,
    PIN_SOURCES_STATE_KEY,
    PLAN_STATE_KEY,
    REDUCTION_REFUSED_ASKS_STATE_KEY,
    TERMINAL_DECLINES_STATE_KEY,
    UNSUPPORTED_STATE_KEY,
    AskBreakdown,
    PinSource,
    Plan,
    PlanStatus,
    SessionState,
    UnsupportedOperation,
    clear_unsupported,
    current_discovery_time_left,
    cut_user_text,
    discovery_time_left,
    load_ask_breakdowns,
    load_asks,
    load_asks_across_years,
    load_plan,
    load_plan_rounds,
    pin_sources,
    record_optional_decline,
    record_unsupported,
    save_amendment_entry,
    save_pin_sources,
    save_plan,
    save_plan_rounds,
)
from planner.settings import AgentSettings
from planner.tool_results import Outcome, ProblemCode
from planner.workflow import ElectedMeasure
from submission.service import (
    ALIGNED_PINS_STATE_KEY,
    SEARCHED_PINS_STATE_KEY,
    Proposal,
    aligned_pins,
    searched_pins,
    submit,
)
from submission.service import PROPOSE_PLAN as PROPOSE_PLAN_PROPOSER
from translator.discovery import CutCalls, current_cut_calls
from translator.mechanical import COMPILE_FAILURES_STATE_KEY, clear_compile_failures
from translator.referral import USER_DECIDES
from translator.tool import Translate, build_translate, permitted_repair

from harness import plan_display
from harness.agents import build_tool_surface, create_planner_with_translator
from harness.graph.state import (
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPEC_WARNINGS_KEY,
    ORCHESTRATOR_SPECS_KEY,
    store_search,
)
from harness.prefix_cache import PrefixCache
from harness.sentences import (
    TERMINAL_PROPOSE_CODES,
    SeamCode,
    referral_sentence,
    refusal_sentence,
)
from harness.settings import Settings
from harness.turn_budget import CLOSED_BY_KEY, current_turn_budget
from harness.turn_records import (
    SETTLED_STATUSES,
    CandidateReading,
    RenderedReadings,
    SearchReading,
    measure_choice_text,
    named_spec_ids,
    newest_per_spec_id,
    render_readings,
)
from harness.usage import log_usage

logger = logging.getLogger(__name__)

# What `PlannerSeam._stalled_referral` counts a referral by: its kind,
# requirement, constraint, and the (name, role) of each entity it leaves
# unsettled.
type StallKey = tuple[str, str | None, str | None, tuple[tuple[str, str | None], ...]]

# Insurance against a runaway tool loop inside one agent run. A healthy plan
# takes well under this many tool calls (skill reads and proposals).
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

# The opening of the planner's first message: the search readings of the
# conversation, which name every spec_id a pin may take without a referral.
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
# The ledger a call renders without readings: a continuing call, which
# renders no readings section, or a caller that passes none.
_NO_READINGS = RenderedReadings("", ())

# A decision's `description` and `rationale` are prose no validator compares,
# so they are the two fields this render may shorten. Every other plan field
# is rendered in full. `chosen`, `options` and every requirement field are
# what a restated plan is checked against, and shortening any of them would
# change the plan itself, not only the text the model reads.
_PLAN_DESCRIPTION_CAP = 200
_PLAN_RATIONALE_CAP = 300

# A referral candidate's `label` is the resolve's own `rendered_sentence`,
# which can run to a paragraph. Capped where `_pinnable_candidates` shapes
# it for a question to the user.
_CANDIDATE_LABEL_CAP = 120

# A compiled source's `sentence` is the resolved reading in search's own
# words. Capped so a nine-source plan cannot let one verbose sentence carry
# the section.
_PLAN_SOURCE_SENTENCE_CAP = 160
# A compiled source's measure and the other ways of counting it, each with
# its spec_id (`_measure_choice_text`). An alternative past the cap is left
# out whole, since a cut spec_id pins nothing.
_PLAN_MEASURE_CHOICE_CAP = 240

# Gemini requires a `thought_signature` on a generation's first `functionCall`
# part and answers a request replaying one without it with a 400. This is
# the documented placeholder that makes the API skip validation for a call
# injected outside a model turn.
_SKIP_THOUGHT_SIGNATURE = b"skip_thought_signature_validator"

# The run-time refusal codes that mean no operation computes what the refused
# step was asked to, each with the `unsupported` answer it stands for. A replan
# entry over one starts with that answer recorded (`_earned_by_refusal`), so a
# plan may decline the ask it served, as it may after `propose_plan` or its
# translation answers `unsupported`. `cross_frame_unit_unbound`: one side of a
# `compare` or `sum_of` declares no unit while the other does, so no operation
# combines the two. A source declaring a unit would also clear it, and the
# decline leaves the planner free to pin one instead. Codes whose `fix` offers
# a choice (a common unit, a facet, matching periods, a window, a denominator,
# an entity) are method choices, and a stop (`executor.classify.STOP_CODES`)
# never replans.
_UNSUPPORTED_BY_RUN_TIME_CODE: Mapping[str, UnsupportedOperation] = {
    "cross_frame_unit_unbound": UnsupportedOperation.NO_OPERATION_COVERS,
}

# The run-time refusal codes a `share_of` step answers for two sides whose
# units it does not pair. `quotient_of` divides the same two sides, so a replan
# over one carries the repair translation gives `SHARE_UNITS_DIFFER` and earns
# no decline.
_SHARE_UNIT_CODES = frozenset({"cross_frame_unit_unbound", "cross_frame_unit_mismatch"})

# The run-time refusal code a `share_of` or `compare` step answers for two
# sides that read one measure from two sources. analysis-mcp's message speaks
# of frames. The repair in plan terms is the planner's.
_FACET_MISMATCH_CODE = "cross_frame_facet_mismatch"


class PlannerSeamError(RuntimeError):
    """The planner produced no plan ready for the user's answer.

    Raised so that a plan the user approves is a plan
    `planner.agent.declare_ready` declared ready.

    `refusal` is the shape `harness.studio_translate._stream_fatal` reads
    (`code`, `retry_mode`), so the user reads the sentence of its code. The
    default code is `internal_error`, because the seam stopping on its own
    account is a fault of the system, whatever the question. The wire
    grammar has no member for it, so the frame states
    `capability_unavailable`. Every way this seam fails on its own account
    replays the same way against the same state, so the default is
    `do_not_retry`. A caller told to retry a deterministic failure pays for
    the same stall again. `fix.detail` holds the message for the log.
    """

    def __init__(self, message: str, *, refusal: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.refusal = refusal or {
            "code": ExecutorCode.INTERNAL_ERROR.value,
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

    def __init__(self) -> None:
        self.referral_counts: dict[StallKey, int] = {}
        self.refused: dict[str, Any] | None = None
        self.ready_to_break = False
        # The `adk_request_input` call args the next generation is replaced
        # with, for a referral the user decides (`_user_decided_ask`).
        self.ask: dict[str, Any] | None = None
        self.submitted: dict[str, Any] | None = None
        self.events = 0
        # The referral the last translation with an `outcome` answered,
        # or `None` once one of those translations answers something
        # other than a referral -- see `_track_tool_call`. An entry that ends
        # with no plan ready carries it (`_not_ready`).
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

# A caller's callback on each request the planner sends its model, and on each
# response the model answers, as `LlmAgent`'s callback lists run them: one that
# answers a response ends its list there.
ModelRequestCallback = Callable[[CallbackContext, LlmRequest], LlmResponse | None]
ModelResponseCallback = Callable[[CallbackContext, LlmResponse], LlmResponse | None]


@dataclass(frozen=True)
class _RevisionSession(ToolContextSession):
    """The session a revision's proposal is checked against (`PlannerSeam.revise`).

    The user has spoken since the plan it amends was recorded: their switch
    is the revision. A dataset that replaces one the user picked answering
    an ask (`inherits`, the new spec_id to the one it replaces) counts as
    picked too, so the measure decision a pin of it states stays theirs.
    """

    inherits: Mapping[str, str] = field(default_factory=dict)

    def user_spoke_since(self, version: int) -> bool:
        """Always: the revision answers the user's own switch."""
        del version
        return True

    def elected_spec_ids(self) -> frozenset[str]:
        """The spec_ids the user picked, and each that replaces one of them."""
        elected = super().elected_spec_ids()
        return elected | {new for new, old in self.inherits.items() if old in elected}


class PlannerSeam:
    """`harness.graph.seams.PlannerModel`, implemented by `src/planner` and the translation it runs."""

    def __init__(
        self,
        *,
        settings: Settings,
        agent_settings: AgentSettings | None = None,
        surface: ToolSurface | None = None,
        prefix_cache: PrefixCache | None = None,
        translate: Translate | None = None,
        model: BaseLlm | None = None,
        before_model_callbacks: Sequence[ModelRequestCallback] = (),
        after_model_callbacks: Sequence[ModelResponseCallback] = (),
    ) -> None:
        # `translate` replaces the translation `propose_plan` runs, and `model`
        # the model `agent_settings` names. `before_model_callbacks` run after
        # the seam's own, which answer a request themselves once the entry
        # has to stop, so they see only the requests the model is sent.
        # `after_model_callbacks` run before the seam's own, so they see each
        # response as the model answered it, before `drop_thought_parts`
        # replaces one. `evals/planner_entry` runs the served planner this way.
        # The surface and translation `propose_plan` runs, kept for `revise`,
        # which records a proposal of its own with no model call.
        self._surface = surface if surface is not None else build_tool_surface(settings=settings)
        self._translate = (
            translate if translate is not None else build_translate(surface=self._surface)
        )
        self._agent = create_planner_with_translator(
            settings=settings,
            agent_settings=agent_settings,
            surface=self._surface,
            # The orchestrator states the plan's question in its brief
            # (`harness.graph.state.ORCHESTRATOR_BRIEF_KEY`) before the planner ever
            # runs (see `planner.agent.create_planner_agent`'s own
            # `question_fixed` doc), so this seam's planner never gets to
            # restart a plan for a different question -- `starts_new_plan` is
            # refused with
            # `QUESTION_FIXED` instead, handled below the same way
            # `BUDGET_EXHAUSTED` is.
            question_fixed=True,
            # The orchestrator reads the user's words and marks its asks
            # (across years). The planner reads them only in an entry the
            # user's amendment opens, which `save_amendment_entry` records
            # below, and only there may `propose_plan`'s `across_years`
            # name a step.
            reads_user_words=False,
            prefix_cache=prefix_cache,
            translate=self._translate,
        )
        if model is not None:
            self._agent.model = model
        # The callbacks are assigned exactly once, here, as bound methods,
        # and read the call they belong to off `_active_call` (a
        # `ContextVar` `_propose` sets for the duration of its own
        # `ctx.run_node` call) instead of closing over it, since one
        # `PlannerSeam` serves every conversation this process runs.
        self._agent.after_tool_callback = self._after_planner_tool
        self._agent.on_tool_error_callback = self._on_planner_tool_error
        # `drop_thought_parts` (`planner.settings`) returns a modified
        # response whenever it drops a part, and ADK's callback list stops at
        # the first truthy return, so the usage callback goes first: it
        # always returns `None`, and `drop_thought_parts` still runs after it.
        self._agent.after_model_callback = [
            *after_model_callbacks,
            self._after_planner_model,
            *self._agent.canonical_after_model_callbacks,
        ]
        self._agent.before_model_callback = [
            self._before_planner_model,
            *before_model_callbacks,
        ]

    async def __call__(
        self,
        *,
        ctx: Context,
        question: str,
        readings: Sequence[Mapping[str, Any]],
        prior_plan: dict[str, Any] | None = None,
        refusal: dict[str, Any] | None = None,
        amendment: str | None = None,
        canvas_digest: dict[str, Any] | None = None,
        conversation_digest: dict[str, Any] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        # `readings` is the search readings of the conversation's earlier
        # turns and then this turn's, as `harness.orchestrator_tools` stores
        # them. The first message renders them.
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
        seeded = Plan.model_validate(prior_plan) if prior_plan is not None else None
        if seeded is not None:
            save_plan(cast(SessionState, ctx.session.state), seeded)
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
            # Whether the user's own words open this entry, which
            # `propose_plan` reads to accept the steps its `across_years`
            # names (`submission.service.admit`). `_message` renders the
            # amendment on the same test.
            save_amendment_entry(cast(SessionState, ctx.session.state), bool(amendment))
            clear_unsupported(cast(SessionState, ctx.session.state))
            for operation in _earned_by_refusal(refusal):
                record_unsupported(cast(SessionState, ctx.session.state), operation)
            clear_compile_failures(ctx.session.state)
            if prior_plan is None and refusal is None and amendment is None:
                # This shape -- no `prior_plan`, no `refusal`, no
                # `amendment` -- names a genuine first-draft entry: the
                # graph's planner node passes exactly this after
                # `plan_start`, never on `amend`/`replan`, which
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
                # Each turn opens a brand-new ADK session, so a turn's
                # first entry has nothing to clear. A later `plan` call in
                # the same turn does: without this, its first-draft entry
                # would find the earlier entry's plan still on record and,
                # with this seam's pair built with `question_fixed=True`,
                # could only amend it, since `starts_new_plan` is refused.
                # Clearing `_BRIEFED_STATE_KEY` alongside makes the very
                # next read of it, two lines below, compute a fresh `True`
                # for this call, so the entry gets the full brief rather
                # than the continuing-call message a session that has
                # already been briefed gets.
                ctx.session.state[PLAN_STATE_KEY] = None
                ctx.session.state[_BRIEFED_STATE_KEY] = False
                # A first draft may decline an ask no requirement kind states,
                # which no tool's answer could ever earn, except one a
                # `period_reduction_unstated` refusal named
                # (`submission.service._declinable`). A replan or an
                # amendment earns each decline from a tool's answer.
                record_optional_decline(
                    cast(SessionState, ctx.session.state),
                    UnsupportedOperation.NO_OPERATION_COVERS,
                    terminal=True,
                )
        # A `resolve_definition` the request's wall-clock bound cuts keeps running here
        # for a later proposal in the same entry to be served, and is
        # cancelled on the way out (`translator.discovery.CutCalls`).
        cut_calls = CutCalls()
        cut_calls_token = current_cut_calls.set(cut_calls)
        try:
            # `fresh` answers "has this session ever been briefed".
            fresh = not ctx.session.state.get(_BRIEFED_STATE_KEY, False)
            ctx.session.state[_BRIEFED_STATE_KEY] = True
            # Every spec_id the planner's message shows it, written on every
            # entry so `propose_plan` reads it (`PIN_SOURCES_STATE_KEY`). A
            # pin of a shown one reuses its block, and one of a shown,
            # settled measure gets the measure decision `propose_plan`
            # writes. Readings render on a fresh message only, so a later
            # entry keeps the ones it rendered, less any the executor has
            # since refused.
            ledger = _NO_READINGS
            if fresh:
                ledger = _readings_ledger(readings)
                # An earlier turn's spec_id pins from the block its own turn
                # stored; this turn's search of one replaces it.
                prior = ctx.state.get(ORCHESTRATOR_PRIOR_LEDGER_KEY) or {}
                blocks = ctx.state.get(ORCHESTRATOR_SPECS_KEY) or {}
                searched = {*_named(ctx.state.get(ORCHESTRATOR_SEARCHES_KEY) or ()), *blocks}
                sources = _reading_pin_sources(
                    readings,
                    ledger.spec_ids,
                    blocks=blocks,
                    earlier=_named(prior.get("readings", ())) - searched,
                    warnings={
                        **prior.get("warnings", {}),
                        **(ctx.state.get(ORCHESTRATOR_SPEC_WARNINGS_KEY) or {}),
                    },
                )
            else:
                sources = pin_sources(cast(SessionState, ctx.session.state))
            refused = {spec_id for spec_id, _ in _refused_specs(refusal)}
            sources = {k: v for k, v in sources.items() if k not in refused}
            save_pin_sources(cast(SessionState, ctx.session.state), sources)
            message = self._message(
                fresh=fresh,
                question=question,
                prior_plan=_as_authored(seeded),
                refusal=refusal,
                amendment=amendment,
                canvas_digest=canvas_digest,
                conversation_digest=conversation_digest,
                ledger=ledger,
                asks=load_asks(cast(SessionState, ctx.session.state)),
                ask_breakdowns=load_ask_breakdowns(cast(SessionState, ctx.session.state)),
                asks_across_years=load_asks_across_years(cast(SessionState, ctx.session.state)),
            )
            submitted, plan, refused = await self._propose(
                ctx=ctx, message=message, on_progress=on_progress
            )
            if refused is not None:
                return self._refused(refused)
            assert submitted is not None and plan is not None, (
                "_propose answers a refusal unless a proposal declared the plan on record "
                "ready (see planner.agent.declare_ready)"
            )
            if plan.status is not PlanStatus.PENDING_APPROVAL:
                # `declare_ready` enforces this (`planner.agent.
                # _execution_blocked`); repeated here so a future gap in that
                # allowlist reaches this seam as a loud `PlannerSeamError`
                # instead of a plan the graph shows the user as approvable.
                raise PlannerSeamError(
                    f"a plan that is {plan.status.value} was declared ready, not "
                    "pending_approval; refusing to report it ready"
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
            try:
                store_aligned_readings(ctx, cast(SessionState, ctx.session.state))
                self._persist_durable(ctx)
            finally:
                current_cut_calls.reset(cut_calls_token)
                await cut_calls.close()

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
        a session whose *durable* `ctx.state` still holds the plan of an
        earlier `plan` call in the same turn -- a real value there that has to be overwritten, not merely left
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
        # `UNSUPPORTED_STATE_KEY`, `OPTIONAL_DECLINES_STATE_KEY`,
        # `TERMINAL_DECLINES_STATE_KEY`, `REDUCTION_REFUSED_ASKS_STATE_KEY`,
        # `COMPILE_FAILURES_STATE_KEY`, `AMENDMENT_ENTRY_STATE_KEY`,
        # `ALIGNED_PINS_STATE_KEY` and `SEARCHED_PINS_STATE_KEY` too.
        # The declines an ask-the-user resume still owes or may make, the
        # failures it may repeat, the amendment that opened it and the pins
        # compiled for its proposals are this entry's own.
        for key in (
            UNSUPPORTED_STATE_KEY,
            OPTIONAL_DECLINES_STATE_KEY,
            TERMINAL_DECLINES_STATE_KEY,
            REDUCTION_REFUSED_ASKS_STATE_KEY,
            COMPILE_FAILURES_STATE_KEY,
            AMENDMENT_ENTRY_STATE_KEY,
            ALIGNED_PINS_STATE_KEY,
            SEARCHED_PINS_STATE_KEY,
        ):
            if key in ctx.session.state:
                ctx.state[key] = ctx.session.state[key]

    async def revise(
        self,
        *,
        ctx: Context,
        prior_plan: dict[str, Any],
        evidence_requirements: Sequence[Mapping[str, Any]],
        inherits: Mapping[str, str],
    ) -> dict[str, Any]:
        """Amend `prior_plan` with `evidence_requirements` and translate it, calling no model.

        `harness.graph.PlannerModel.revise`'s contract. The proposal runs
        the chain `propose_plan` runs (`submission.service.submit`), reading
        each first year it needs and compiling each input a calculation needs
        on another basis or provider scope (`planner.agent.with_source_reads`), then
        the translation, and the plan is ready when `declare_ready` says so.
        A dataset requirement re-pinned to another spec_id cites the
        decisions that pin states and drops those of the one it left, which
        go with it. The proposal opens its own round count, as a planner
        entry does. `inherits` maps each new spec_id to the one it
        replaces (`_RevisionSession`).

        Returns `{"plan", "summary"}` once the revised plan awaits approval,
        or `{"refused": ...}` as `__call__` refuses, the revision recording
        a version or not; the caller restores what it keeps.
        """
        state = cast(SessionState, ctx.state)
        prior = Plan.model_validate(prior_plan)
        save_plan(state, prior)
        save_plan_rounds(state, 0)
        view = _RevisionSession(tool_context=ctx, inherits=dict(inherits))
        proposal = Proposal(
            evidence_requirements=list(evidence_requirements), amends_version=prior.version
        )
        recorded = await with_source_reads(
            lambda: submit(
                view,
                proposal,
                question_fixed=True,
                reads_user_words=False,
                proposer=PROPOSE_PLAN_PROPOSER,
            ),
            state=state,
            surface=self._surface,
        )
        store_aligned_readings(ctx, state)
        if recorded.get("outcome") != Outcome.OK.value:
            return self._refused(recorded)
        answer = declare_ready({**recorded, "translation": await self._translate(ctx)}, ctx)
        plan = load_plan(state)
        if "summary" not in answer or plan is None:
            blocked = answer.get("submission") or answer.get("translation")
            return self._refused(
                cast(dict[str, Any], blocked if isinstance(blocked, dict) else answer)
            )
        return {"plan": plan.model_dump(mode="json"), "summary": answer["summary"]}

    # -- one agent run ------------------------------------------------------

    async def _propose(
        self,
        *,
        ctx: Context,
        message: str,
        on_progress: Callable[[dict[str, Any]], None] | None,
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
        half-finished turn is left for a later entry's message to resume
        instead of seeing.

        Answers the declared-ready `propose_plan` answer and the plan, or the
        refusal the run ended on. A run that ends with no plan declared ready
        and no refusal of its own ends on the bound that closed one of its
        model calls (`harness.turn_budget.TurnBudget.closed`), or else on
        `SeamCode.PLAN_NOT_READY`, carrying the referral the last translation
        answered, if any (`_not_ready`).

        A discovery call inside the run may take what is left of the
        request's wall-clock bound (`planner.plan.discovery_time_left`); with
        no request budget set it takes its own timeout alone.

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
        tracker = _RunTracker()
        token = _active_call.set(_CallState(tracker=tracker, on_progress=on_progress))
        budget = current_turn_budget.get()
        time_left_token = current_discovery_time_left.set(
            budget.wall_left if budget is not None else None
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
        if tracker.submitted is None and tracker.refused is None:
            closed = budget.closed if budget is not None else None
            tracker.refused = closed or _not_ready(tracker.last_referral)
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

    def _on_planner_tool_error(
        self, tool: BaseTool, args: dict[str, Any], tool_context: Any, error: Exception
    ) -> dict[str, Any] | None:
        """The planner's `on_tool_error_callback`: an `Exception` one of its
        own tools raised ends the entry on `SeamCode.PLANNER_TOOL_FAILED`, a
        final refusal, instead of failing the turn.

        A name that is none of the planner's tools answers `None`, so ADK's
        own tool-not-found response, which lists the tools the model can
        call, stands. The dict returned for a real tool becomes that call's
        function response, and `_after_planner_tool` still runs on it. The
        dict carries no `outcome`, so `_track_tool_call` reads it as neither a
        terminal refusal nor a translation. The model never reads it, since
        `_before_planner_model` stops the run before the next generation.
        """
        del args
        if tool.name not in {t.name for t in self._agent.tools if isinstance(t, BaseTool)}:
            return None
        error_class = type(error).__name__
        logger.error(
            "planner tool %s raised %s session=%s invocation=%s trace_id=%s",
            tool.name,
            error_class,
            tool_context.session.id,
            tool_context.invocation_id,
            current_trace_id(),
            exc_info=error,
        )
        call = _active_call.get()
        if call is None:
            logger.warning("planner on_tool_error_callback fired with no active call")
        elif call.tracker.refused is None:
            call.tracker.ready_to_break = True
            call.tracker.refused = {
                "code": SeamCode.PLANNER_TOOL_FAILED,
                "detail": {"tool": tool.name},
            }
        return {"error": f"{tool.name} failed inside DFO ({error_class}). Do not repeat this call."}

    def _after_planner_model(
        self, callback_context: Any, llm_response: LlmResponse
    ) -> LlmResponse | None:
        """Log the generation's usage, and answer a call the request's budget
        closed (`harness.turn_budget.CLOSED_BY_KEY`) with a question to the
        user when the last referral names sources to choose from.

        The question is the same `adk_request_input` long-running call
        `planner.agent.ask_user` issues, built here from the referral's
        candidates (`_pinnable_candidates`), since no model call is left to
        compose it. `populate_client_function_call_id` gives the call an id,
        so the resume pairs with the user's answer the way a model-issued ask
        does (`planner.agent._elected_spec_ids`), and the answer arrives in a
        request of its own, with a fresh budget. With nothing pinnable (no
        referral, or a `no_data` referral, whose nearest-series candidates
        carry no `spec_id`) the closing text stands: the tracker refuses on
        the bound and the run ends on it.

        Answers `None` for every other response, so `drop_thought_parts`,
        listed after this callback, still runs on it.
        """
        del callback_context
        if llm_response.usage_metadata is not None:
            log_usage(self._agent.name, llm_response.usage_metadata)
        metadata = llm_response.custom_metadata or {}
        call = _active_call.get()
        if CLOSED_BY_KEY not in metadata or llm_response.partial or call is None:
            return None
        tracker = call.tracker
        candidates = _pinnable_candidates(tracker.last_referral)
        if not candidates:
            tracker.ready_to_break = True
            tracker.refused = {"code": SeamCode(metadata[CLOSED_BY_KEY]), "detail": {}}
            return None
        return _ask_response(_clarifying_question_args(candidates, lead=_BUDGET_CLOSED_LEAD))

    def _before_planner_model(self, callback_context: Any, llm_request: Any) -> LlmResponse | None:
        """Substitute a synthetic response once this call's own tracker is
        ready to break (see `_propose`'s docstring). With no active call, see
        the class-level note above, there is nothing to check against, so
        `None` here means "generate normally," the same as every ordinary turn
        this callback lets through.

        A referral the user decides (`_user_decided_ask`) is answered first,
        with the `adk_request_input` call it asks, which suspends the run in
        place of a model turn composing the question. Every break carries a
        refusal, and the substituted text is that refusal's sentence
        (`harness.sentences.refusal_sentence`). The request's budget is kept
        by the model itself (`harness.turn_budget.BoundedGemini`).
        """
        del callback_context, llm_request
        call = _active_call.get()
        if call is None:
            logger.warning("planner before_model_callback fired with no active call")
            return None
        tracker = call.tracker
        if tracker.ask is not None:
            ask, tracker.ask = tracker.ask, None
            return _ask_response(ask)
        if not tracker.ready_to_break:
            return None
        assert tracker.refused is not None, "every break sets the refusal it ends on"
        return _stop_response(tracker.refused)

    @staticmethod
    def _track_tool_call(
        tracker: _RunTracker,
        on_progress: Callable[[dict[str, Any]], None] | None,
        *,
        agent_name: str,
        tool: BaseTool,
        tool_response: dict[str, Any] | None,
    ) -> None:
        """The planner's `after_tool_callback`: progress, then what the answer settles.

        A terminal refusal (`_is_terminal_refusal`) ends the turn, a `translation` is the
        verdict on the plan the call recorded, and `summary` says the call declared that plan
        ready (`planner.agent.declare_ready`).
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
        if PlannerSeam._is_terminal_refusal(tool_response):
            # The proposal refused with a code that records nothing and
            # leaves nothing correctable within this entry: the round cap
            # (`planner.plan.MAX_PLAN_ROUNDS`, `BUDGET_EXHAUSTED`), or a
            # restart while the graph holds the question fixed
            # (`QUESTION_FIXED`, `planner.agent`'s `question_fixed` mode --
            # this seam's own planner is always built that way, see
            # `PlannerSeam.__init__`). The cap bounds the proposals within
            # one entry into the planner, since `PlannerSeam.__call__`
            # resets the count on each entry (see `__call__`); a restart is
            # refused however fresh that budget is. Stop the planner's turn
            # here either way.
            tracker.ready_to_break = True
            tracker.refused = tool_response
        elif isinstance(tool_response, dict):
            translation = tool_response.get("translation")
            if isinstance(translation, dict):
                PlannerSeam._track_translation(tracker, translation)
            if "summary" in tool_response:
                tracker.submitted = tool_response

    @staticmethod
    def _track_translation(tracker: _RunTracker, tool_response: dict[str, Any] | None) -> None:
        """Read one translation, the `translation` of an accepted `propose_plan`:
        the referral it answered, and whether it ends the entry.

        `last_referral` is written from every translation that carries an
        `outcome`, referral or not: translation runs the whole plan, so each
        outcome-bearing verdict states what is open across it right now, and
        replaces whatever the last one recorded. `ok` means every
        requirement compiled, which clears a referral the same way. A response
        with no `outcome` settles nothing and is left alone.
        An entry that ends with no plan ready carries `last_referral`
        whichever way it landed (`_not_ready`).

        An internal, repeated, budget-spent or never-issued failure, or a
        transient one with no discovery time left (`_closing_refusal`), ends
        the entry on its own refusal, since the next translation would stop
        where this one did. A referral that recurs with nothing new
        (`_stalled_referral`) ends it on `SeamCode.PLAN_NOT_READY`, carrying
        that referral. Either way the planner's turn stops before its next
        generation. A referral the user decides, with candidates to pick from
        (`_user_decided_ask`), replaces that next generation with the
        question it asks instead.
        """
        if isinstance(tool_response, dict) and "outcome" in tool_response:
            tracker.last_referral = PlannerSeam._referral_dict(tool_response)
        closing = PlannerSeam._closing_refusal(tracker.last_referral)
        if closing is not None:
            tracker.ready_to_break = True
            tracker.refused = closing
            return
        ask = _user_decided_ask(PlannerSeam._referral_dict(tool_response))
        if ask is not None:
            tracker.ask = ask
            return
        if PlannerSeam._stalled_referral(tool_response, tracker.referral_counts):
            tracker.ready_to_break = True
            tracker.refused = _not_ready(tracker.last_referral)

    # Refusal codes after which no other plan for the same ask can succeed
    # in this turn: a fault in the system, a plan that failed the same way
    # twice, a question whose every ask is declined, a data service that is
    # not answering, or a request that spent a bound of its budget.
    _FINAL_CODES = frozenset(
        {
            SeamCode.TURN_MODEL_TIME_SPENT,
            SeamCode.TURN_MODEL_CALLS_SPENT,
            SeamCode.TURN_WALL_TIME_SPENT,
            SeamCode.TRANSLATION_INTERNAL_FAILURE,
            SeamCode.TRANSLATION_REPEATED,
            SeamCode.DISCOVERY_UNAVAILABLE,
            SeamCode.PLANNER_TOOL_FAILED,
            ProblemCode.NOTHING_TO_PLAN,
        }
    )

    @staticmethod
    def _refused(refused: dict[str, Any]) -> dict[str, Any]:
        """What `__call__` answers for a refusal that ends the entry: its plain
        sentence (`harness.sentences.refusal_sentence`), whether another plan
        for the same ask could succeed, and its code."""
        return {
            "refused": {
                "text": refusal_sentence(refused),
                "final": refused.get("code") in PlannerSeam._FINAL_CODES,
                "code": refused.get("code"),
            }
        }

    @staticmethod
    def _is_terminal_refusal(tool_response: dict[str, Any] | None) -> bool:
        """Whether a tool response refuses with one of
        `harness.sentences.TERMINAL_PROPOSE_CODES`, which only a proposal (`propose_plan`)
        answers: an outcome that records no new version and ends the entry.
        """
        return (
            isinstance(tool_response, dict)
            and tool_response.get("outcome") == "refused"
            and tool_response.get("code") in TERMINAL_PROPOSE_CODES
        )

    @staticmethod
    def _referral_dict(response: dict[str, Any] | None) -> dict[str, Any] | None:
        """The referral a translation answers with, or `None` for any
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
        """The refusal a referral ends the planner entry on, or `None` when the
        planner may still settle it in this entry.

        An `internal` referral ends it with its reference. One marked
        `budget_spent` ends it as the data service unavailable, whatever else
        it carries. One marked `not_issued`, for a call the request's
        wall-clock bound left no time to make, ends it as that bound spent.
        Any other `transient` referral once no discovery time is left
        (`planner.plan.discovery_time_left`) ends it as the data service
        unavailable: the call was made and did not answer, and no discovery
        can run again in this request. Each of these
        carries the requirement the referral is about and its reason. One
        marked `repeated` ends it with what it is about, and the compile issue
        a `requirement_needs_repair` one names as its `constraint`.
        """
        if referral is None:
            return None
        if referral.get("outcome") == Outcome.INTERNAL.value:
            return {
                "code": SeamCode.TRANSLATION_INTERNAL_FAILURE,
                "detail": {"reference": str(referral.get("reference"))},
            }
        requirement_id = referral.get("requirement_id")
        reason = referral.get("reason")
        if referral.get("not_issued") and not referral.get("budget_spent"):
            return {
                "code": SeamCode.TURN_WALL_TIME_SPENT,
                "detail": {"requirement_id": requirement_id, "reason": reason},
            }
        time_left = discovery_time_left()
        out_of_time = (
            referral.get("outcome") == Outcome.TRANSIENT.value
            and time_left is not None
            and time_left <= 0
        )
        if referral.get("budget_spent") or out_of_time:
            return {
                "code": SeamCode.DISCOVERY_UNAVAILABLE,
                "detail": {"requirement_id": requirement_id, "reason": reason},
            }
        if referral.get("repeated"):
            return {
                "code": SeamCode.TRANSLATION_REPEATED,
                "detail": _referral_detail(referral),
            }
        return None

    @staticmethod
    def _stalled_referral(response: dict[str, Any] | None, counts: dict[StallKey, int]) -> bool:
        """Count one referral by its kind, its requirement, the compile
        issue it names as its `constraint` and the entities its
        `ambiguous_entities` leave unsettled, and answer whether that exact
        referral has now come back a second time, which stops the run. Two
        different repairs on one requirement count apart, so a planner that
        fixes one compile issue per round keeps going, and an
        `entity_name_ambiguous` referral covering fewer entities counts
        apart, so a planner that settles one end of "aid from Congo to Congo"
        per round keeps going too. A third repetition
        would carry nothing a model could act on that the second one did not
        already say. A referral marked `repeated` ends the entry instead
        (`_closing_refusal`). A transient referral is never counted: the same
        plan is worth proposing again.
        """
        referral = PlannerSeam._referral_dict(response)
        if referral is None or referral.get("outcome") == Outcome.TRANSIENT.value:
            return False
        kind = referral.get("kind")
        if kind is None:
            return False
        unsettled = tuple(
            (reading["entity"]["name"], reading["entity"].get("role"))
            for reading in referral.get("ambiguous_entities") or ()
        )
        key = (kind, referral.get("requirement_id"), referral.get("constraint"), unsettled)
        counts[key] = counts.get(key, 0) + 1
        return counts[key] >= 2

    # -- messages -----------------------------------------------------------

    def _message(
        self,
        *,
        fresh: bool,
        question: str,
        prior_plan: dict[str, Any] | None,
        refusal: dict[str, Any] | None,
        amendment: str | None,
        canvas_digest: dict[str, Any] | None,
        conversation_digest: dict[str, Any] | None,
        ledger: RenderedReadings = _NO_READINGS,
        asks: Sequence[str] = (),
        ask_breakdowns: Sequence[AskBreakdown] = (),
        asks_across_years: Sequence[int] = (),
    ) -> str:
        """The message for one `_propose` call.

        `fresh=True` -- the first call this seam makes in a session --
        renders the full brief: the search readings, the canvas and
        conversation digests, the plan being changed, and the question.
        `fresh=False` -- a later `__call__` re-entering over the
        `amend`/`replan` edge -- renders only the new thing: the
        `amendment`/`refusal` ask. `prior_plan` is never rendered when
        `fresh` is `False`, even though it is still accepted here and still
        seeded into `propose_plan`'s own drafting state unconditionally
        (`__call__`'s own `save_plan` call, which does not read `fresh`).

        `ledger` is the rendered search readings (`_readings_ledger`), the
        fresh brief's opening section, and a compiled source of the plan
        being changed states no measure choice the ledger already names
        (`_plan_context`). `asks`, the outputs the orchestrator's brief lists
        (`planner.plan.ASKS_STATE_KEY`), render numbered after the question,
        each with the parties `ask_breakdowns` says it breaks down by and
        marked when `asks_across_years` numbers it.

        See the module docstring for why a continuing call needs no more
        than this: `ctx.run_node` shares the graph's own session, so the
        model already has the full brief in its own history by the time a
        continuing call runs, and `prior_plan` on both edges restates
        nothing the ask does not already say.

        The question and the amendment are the user's own words, and a refusal
        carries text from the run that failed. Nothing bounds their length on
        the wire, so each renders cut to `MAX_QUESTION_CHARS`
        (`planner.plan.cut_user_text`), the size `Plan.question` holds.

        `prior_plan` is the plan being changed as the model wrote it
        (`_as_authored`), and renders as a digest, through `_plan_context`. The
        instruction above the readings keeps its own wording. It is what
        stops the model inventing a `spec_id`.
        """
        ask = ""
        if amendment:
            ask = (
                "\nThe user asks for this change to the plan: "
                f"{cut_user_text(amendment, MAX_QUESTION_CHARS)}\n"
                f"Call {PROPOSE_PLAN} again with the revised decisions and evidence "
                "requirements -- the change has to alter what a requirement "
                "actually states, not only a decision's rationale -- which has "
                "it translated and compiled again.\n"
            )
        elif refusal:
            avoid = (
                _avoid_sentence(_refused_specs(refusal))
                + _earned_decline_sentence(_earned_by_refusal(refusal))
                + _share_units_repair_sentence(refusal)
                + _facet_repair_sentence(refusal)
            )
            stop = {key: value for key, value in refusal.items() if key != "computed_asks"}
            ask = (
                f"\n{_failed_run_sentence(refusal.get('computed_asks'))} "
                f"Refusal: {cut_user_text(json.dumps(stop), MAX_QUESTION_CHARS)}\n"
                f"Call {PROPOSE_PLAN} again with a plan that avoids it.{avoid}\n"
            )
        if not fresh:
            return ask
        return "".join(
            [
                _READINGS_CONTEXT.format(readings=ledger.text),
                _section("What is already on the user's canvas", canvas_digest),
                _section("Earlier turns in this conversation", conversation_digest),
                _plan_context(prior_plan, stated=ledger.other_counts),
                f"\nQuestion: {cut_user_text(question, MAX_QUESTION_CHARS)}\n",
                _asks_section(asks, ask_breakdowns, asks_across_years),
                ask,
            ]
        )


def _referral_detail(referral: Mapping[str, Any]) -> dict[str, Any]:
    """What a refusal ending on `referral` carries for its sentence and its reader."""
    return {key: referral.get(key) for key in ("kind", "requirement_id", "reason", "constraint")}


def _not_ready(referral: Mapping[str, Any] | None) -> dict[str, Any]:
    """The refusal for an entry that ended with no plan ready, carrying `referral` if any."""
    return {
        "code": SeamCode.PLAN_NOT_READY,
        "detail": _referral_detail(referral) if referral is not None else {},
    }


def _stop_response(refused: Mapping[str, Any]) -> LlmResponse:
    """The terminal text `_before_planner_model` substitutes for a generation,
    stating the refusal the run ends on in its user sentence."""
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text=f"Stopping this turn: {refusal_sentence(refused)}")],
        )
    )


def _failed_run_sentence(computed: Sequence[int] | None) -> str:
    """What a replan's failed run computed: the asks its finished terminal
    steps answer (`harness.graph.execute._computed_asks`), by number, or `None` for
    a plan held to no asks. The new plan runs from the start, so it still
    answers every ask. The list of numbers is cut at `MAX_QUESTION_CHARS`."""
    if computed is None:
        return "Executing the approved plan failed, and the new plan runs from the start."
    if not computed:
        return "Executing the approved plan failed before it computed any ask."
    numbers = cut_user_text(", ".join(str(number) for number in computed), MAX_QUESTION_CHARS)
    return (
        f"Executing the approved plan failed after it computed asks {numbers}. The new plan "
        "runs from the start, so it still answers every ask."
    )


def _asks_section(
    asks: Sequence[str],
    breakdowns: Sequence[AskBreakdown] = (),
    across_years: Sequence[int] = (),
) -> str:
    """The asks the answer is held to, numbered as `propose_plan` names them. Empty when none.

    An ask in `breakdowns` names the parties it breaks down by (`by`), and
    the section then states the rule `planner.plan.Plan` holds its answer
    to. The parties an ask totals across (`over`) are not shown: the plan
    holds a share's whole to them, and its refusal names the party.
    An ask in `across_years` is marked as one figure over its years taken
    together, and the section then says no other ask is.
    """
    if not asks:
        return ""
    by = {breakdown.ask: breakdown.by for breakdown in breakdowns if breakdown.by}
    lines = "".join(
        f"{number}. {ask}"
        + (f" (by {', '.join(by[number])})" if number in by else "")
        + (" (across years)" if number in across_years else "")
        + "\n"
        for number, ask in enumerate(asks)
    )
    rule = (
        "An ask marked with a party, as in (by creditor), lists each member of that party "
        "one by one, so its answer keeps the party: name its members as `entities` with "
        "that role when the question names them; otherwise pin a source whose reading above "
        "holds it open and name the party in the dataset requirement's `across`. Carry it "
        "to the answer through a rank whose `axis` is that party or no aggregate that sums "
        "it away. Decline that part with code 'breakdown_not_kept' when the source this "
        "ask's answer pins cannot hold the party open.\n"
        + (
            "An ask marked with several parties is one result: one output keeps them all.\n"
            if any(len(roles) > 1 for roles in by.values())
            else ""
        )
        if by
        else ""
    )
    if across_years:
        rule += (
            "An ask marked (across years) asks for one figure, or one ranking, over its "
            "years taken together. No other ask does: its answer keeps each year or reads "
            "one year.\n"
        )
    return (
        "The answer is held to these asks. Account for each exactly once: answer it in "
        "`answered`, or, when `declined` permits its decline, "
        "decline it in `declined` with its number as `ask_ref`. When the plan computes "
        "part of an ask, do both: answer it with what is computed and decline the rest, "
        "naming that part in the decline's `ask`.\n" + rule + lines
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
    for a translation this entry never saw, or one whose kind is
    not among `translator.referral.MAY_CARRY_CANDIDATES`.
    """
    if not isinstance(referral, dict):
        return []
    if referral.get("kind") == "source_name_ambiguous":
        return [
            {"spec_id": str(candidate["ref"]), "label": str(candidate["label"])}
            for group in referral.get("ambiguous_sources") or []
            for candidate in group.get("candidates") or []
        ]
    candidates = referral.get("candidates")
    if not isinstance(candidates, list):
        return []
    return [
        {
            "spec_id": str(candidate["spec_id"]),
            "label": _truncate(str(candidate.get("sentence") or ""), _CANDIDATE_LABEL_CAP),
        }
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("spec_id")
    ]


# What the question a request's closing budget asks opens with.
_BUDGET_CLOSED_LEAD = "This request reached its limit before the plan settled on a source."


# What the question a referral's candidates put to the user asks: which one
# they meant, or, when the source they named holds none of the data, which
# other source to use.
_WHICH_MEANT = "Which of these did you mean?"
_WHICH_INSTEAD = "Which of these should be used instead?"


def _clarifying_question_args(
    candidates: list[dict[str, str]], *, lead: str, question: str = _WHICH_MEANT
) -> dict[str, Any]:
    """The `adk_request_input` call args that put a referral's pinnable
    candidates to the user: the shape `planner.agent.ask_user` declares, built
    here from the candidates with no model call composing it. `lead` says why
    the user is asked and `question` what they are asked. It recommends no
    candidate, since nothing has weighed them. The message reaches the user, so
    it names no requirement id, referral kind or spec_id; each candidate's
    label says what it is.
    """
    return {"message": f"{lead} {question}", "candidates": candidates}


def _user_decided_ask(referral: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The `adk_request_input` call args for a referral the user decides, or
    `None` for any other answer.

    A referral of a kind in `translator.referral.USER_DECIDES` carrying
    pinnable candidates (`_pinnable_candidates`) leaves the pick to the user,
    so the seam asks it, led by the referral's sentence
    (`harness.sentences.referral_sentence`); one marked `source_mismatch`
    asks which other source to use instead. One with no candidates leaves
    the planner to say what cannot be done and ask what to change.
    """
    if referral is None or referral.get("kind") not in USER_DECIDES:
        return None
    candidates = _pinnable_candidates(dict(referral))
    if not candidates:
        return None
    source_mismatch = bool(referral.get("source_mismatch"))
    lead = referral_sentence(
        referral.get("kind"), referral.get("constraint"), source_mismatch=source_mismatch
    )
    return _clarifying_question_args(
        candidates, lead=lead, question=_WHICH_INSTEAD if source_mismatch else _WHICH_MEANT
    )


def _ask_response(args: dict[str, Any]) -> LlmResponse:
    """A model turn that is one `adk_request_input` call with `args`.

    `populate_client_function_call_id` gives the call an id, so the resume
    pairs with the user's answer the way a model-issued ask does
    (`planner.agent._elected_spec_ids`).
    """
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name=REQUEST_INPUT_FUNCTION_CALL_NAME, args=args
                    ),
                    thought_signature=_SKIP_THOUGHT_SIGNATURE,
                )
            ],
        )
    )


def _readings_ledger(readings: Sequence[Mapping[str, Any]]) -> RenderedReadings:
    """The conversation's search readings as the planner's first message
    renders them, each spec_id by its newest reading only, with the spec_ids
    that text names (`render_readings`). Over `_READINGS_BUDGET` the oldest
    readings are folded, then dropped and counted."""
    if not readings:
        return RenderedReadings("(no search has run in this conversation)", ())
    return render_readings(
        newest_per_spec_id([SearchReading.model_validate(reading) for reading in readings]),
        budget=_READINGS_BUDGET,
    )


def store_aligned_readings(ctx: Context, planner_state: SessionState) -> None:
    """Store each pin compiled to put a calculation's inputs on one basis or provider scope as a reading of the conversation.

    `planner_state` holds the pins (`submission.service.aligned_pins`), each
    with Search's answer as a digest. A pin no reading of the conversation
    names yet joins this turn's readings through `store_search`, as the
    `switch` node stores a switched one: under the words of the reading it
    was compiled from, keeping that reading's asked, held-open and share
    roles, with its compiled scope on `SearchReading.scope`. Its block marks
    the axis it was compiled on as the planner's (`harness.scope_choices.PLANNER_AXES_BLOCK_KEY`),
    so the reading reads it in `SearchReading.planner_axes` and
    `harness.scope_choices.carried_entries` leaves that entry out.
    Later turns then find it among the readings, and a switch on its
    plan-card row finds its scope (`harness.graph.switch`).

    Each dataset Search settled on for a calculation's input the planner
    pinned none for (`submission.service.searched_pins`) joins them the same
    way, under the words it was searched as, with the parties the search
    named and held open, its block marking every axis those words settled
    as the planner's.
    """
    held = aligned_pins(planner_state)
    searched = [pin for pin in searched_pins(planner_state) if pin.spec_id and pin.digest]
    if not held and not searched:
        return
    readings = [
        SearchReading.model_validate(reading)
        for reading in (
            *(ctx.state.get(ORCHESTRATOR_PRIOR_LEDGER_KEY) or {}).get("readings", []),
            *(ctx.state.get(ORCHESTRATOR_SEARCHES_KEY) or []),
        )
    ]
    stored = {reading.spec_id for reading in readings}
    for pin in held:
        if pin.spec_id in stored:
            continue
        origin = next(
            (reading for reading in reversed(readings) if reading.spec_id == pin.origin), None
        )
        digest = copy.deepcopy(pin.digest)
        if origin is not None:
            (part,) = digest["part_digests"]
            part["variable"] = origin.text
        store_search(
            cast(SessionState, ctx.state),
            digest,
            asked=[origin.asked_roles] if origin is not None else (),
            held_open=[origin.open_roles] if origin is not None else (),
            share_sides=[origin.share_over] if origin is not None else (),
        )
        stored.add(pin.spec_id)
    for found in searched:
        if found.spec_id in stored or found.digest is None:
            continue
        store_search(
            cast(SessionState, ctx.state),
            found.digest,
            asked=[found.asked_roles],
            held_open=[found.open_roles],
        )
        stored.add(found.spec_id)


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
    (`blocks`). Settled when a reading with a `SETTLED_STATUSES` status
    holds it as its own spec_id; a spec_id only ever offered as a candidate
    is not. A spec_id that only readings of another count settle
    (`SearchReading.other_count_of`) names the source that offered it. Each
    keeps the parties its settling reading asked for. A fetched price sibling
    (`SearchReading.sibling_of` on axis `unit`) settles as its own source,
    since its block elects the count search elected for the source that
    offered it.
    """
    settled = [
        parsed
        for parsed in (SearchReading.model_validate(reading) for reading in readings)
        if parsed.status in SETTLED_STATUSES
    ]
    elected = {reading.spec_id for reading in settled if reading.other_count_of is None}
    counts_of = {
        reading.spec_id: reading.other_count_of
        for reading in settled
        if reading.other_count_of is not None
    }
    sources: dict[str, PinSource] = {}
    for spec_id in shown:
        from_earlier = spec_id in earlier
        reading = next((r for r in reversed(settled) if r.spec_id == spec_id), None)
        sources[spec_id] = PinSource(
            provenance_ref=reading.provenance_ref if reading else None,
            provenance_account=reading.provenance_account if reading else None,
            block=None if from_earlier else blocks.get(spec_id),
            warnings=_warning_codes(warnings.get(spec_id)),
            settled=spec_id in elected or spec_id in counts_of,
            earlier_turn=from_earlier,
            other_count_of=None if spec_id in elected else counts_of.get(spec_id),
            asked_roles=tuple(reading.asked_roles) if reading else (),
        )
    return sources


def _warning_codes(warnings: object) -> tuple[str, ...]:
    """The `code` of each warning dict in `warnings`, a list as search returns it."""
    if not isinstance(warnings, list):
        return ()
    codes = (warning.get("code") for warning in warnings if isinstance(warning, dict))
    return tuple(str(code) for code in codes if code)


def _as_authored(plan: Plan | None) -> dict[str, Any] | None:
    """`plan` as the model wrote it, for `_plan_context` to render.

    A stored plan holds the copies the split of whole-history reads added
    (`Plan.rolling_copies`), which the model did not write and cannot
    patch by name. The render shows the plan without them
    (`plan_display.shown_plan`), and the stored plan is saved as it is.
    """
    if plan is None:
        return None
    return plan_display.shown_plan(plan).model_dump(mode="json")


def _plan_context(prior_plan: dict[str, Any] | None, *, stated: Collection[str] = ()) -> str:
    """The plan being changed, rendered field by field as a digest.

    `stated` are the spec_ids of other counts the search readings above
    already name (`RenderedReadings.other_counts`); a compiled source
    offering only those states no measure choice of its own.

    The title above it comes from `_section`, matching the name `_message`'s
    docstring and the tests that read the brief use for that section. This
    function supplies only the body.

    Every field the model states is rendered. This renderer shrinks only the
    text. The planner's model decides an amendment from what it reads here,
    and it cannot read session state, so a field dropped here is a field the
    next proposal cannot preserve. `Decision.description` and
    `Decision.rationale` are the only two cut, and they are the only two
    `planner.plan` compares against nothing; `chosen` is checked against
    `options`. A decision's `fixes` renders as the names of the fields it
    fixes, the form `propose_plan` takes; their values are on the
    requirements citing it. `source_spec_id` is left out, since code writes
    it. A shortened `description` or `rationale` is prose the model is being
    asked to rewrite anyway. A shortened `measure` would be a different plan.
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
        if isinstance(fixes, dict) and (
            named := [k for k, v in fixes.items() if v is not None and k != "source_spec_id"]
        ):
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
        lines.extend(_workflow_lines(workflow, stated=stated))

    known = {
        "question",
        "version",
        "previous_version",
        "status",
        "approved_revision",
        "approved_at",
        "decisions",
        "evidence_requirements",
        "translation",
        "workflow",
        # Code writes both (`Plan.rolling_copies`, `Plan.first_years`), so the
        # digest leaves them out.
        "rolling_copies",
        "first_years",
    }
    lines.extend(f"{key}: {_scalar(prior_plan[key])}" for key in sorted(set(prior_plan) - known))
    return "\n".join(lines) + "\n"


def _field_line(requirement: dict[str, Any]) -> str:
    """Every field of one requirement but `id`, `kind` and its citation list."""
    skip = {"id", "kind", "decision_ids"}
    return " ".join(
        f"{key}={_scalar(value)}" for key, value in requirement.items() if key not in skip
    )


def _workflow_lines(workflow: dict[str, Any], *, stated: Collection[str] = ()) -> list[str]:
    """A compiled workflow's selected sources and its evidence bindings.

    `SelectedSource.sentence` is kept. It is the resolved reading, in the
    words search answered, and a proposal that changes a requirement's
    measure has to see what the approved source said it was. `searched_as`
    and `has_data` are dropped. Both are discovery's own bookkeeping, and the
    step and binding wiring is the mechanic above.

    A source whose measure offers other ways of counting the flow or other
    prices names each with its spec_id (`_measure_choice_text`), so an
    amendment asking for one pins it, unless every one is in `stated`. A windowed copy carries
    its origin's `elected_measure`, so the choice a reading of the origin
    states is not stated again.
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
            name = f"{spec_id} -- {sentence}" if sentence else spec_id
            named.append(name + _measure_choice_text(source.get("elected_measure"), stated))
        lines = ["Compiled sources:", *(f"  {name}" for name in named)]
    bindings = workflow.get("evidence_bindings")
    if isinstance(bindings, list):
        lines.extend(_binding_line(binding) for binding in bindings)
    return lines


def _measure_choice_text(elected: Any, stated: Collection[str]) -> str:
    """A source's `elected_measure`, each other count and each other prices,
    with its spec_id (`measure_choice_text`).

    For example "; counted as gross disbursements; other counts: net
    disbursements [spec_…]; other prices: constant prices [spec_…]", the
    choice at most `_PLAN_MEASURE_CHOICE_CAP` characters. Empty when it
    offers none, none fits, or `stated` holds every one.
    """
    if not isinstance(elected, dict):
        return ""
    measure = ElectedMeasure.model_validate(elected)
    counts = measure.other_counts()
    prices = [
        CandidateReading(spec_id=other.spec_id, label=other.label)
        for other in measure.alternatives
        if other.axis == "unit"
    ]
    if all(other.spec_id in stated for other in [*counts, *prices]):
        return ""
    text = measure_choice_text(
        measure.label, counts, other_prices=prices, cap=_PLAN_MEASURE_CHOICE_CAP
    )
    return f"; {text}" if text else ""


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


def _stops(refusal: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Each stop in a replan's `refusal`: the primary's, then its `also_stopped` siblings'."""
    if not isinstance(refusal, dict):
        return []
    also_stopped = refusal.get("also_stopped")
    payloads = [refusal, *(also_stopped if isinstance(also_stopped, list) else ())]
    return [payload for payload in payloads if isinstance(payload, dict)]


def _share_units_refused(stop: dict[str, Any]) -> bool:
    """Whether `stop` is a `share_of` step refusing its two sides on their units.

    `MethodChoiceNeeded`'s shape (`harness.graph.execute`'s `_refusal_shape`) names the
    refusing step's tool by its name on analysis-mcp, which is the verb.
    """
    return stop.get("tool") == OperationKind.SHARE_OF and stop.get("code") in _SHARE_UNIT_CODES


def _earned_by_refusal(refusal: dict[str, Any] | None) -> set[UnsupportedOperation]:
    """The declines a replan's `refusal` earns: one per stop in it, the
    primary's or an `also_stopped` sibling's, whose `code` is in
    `_UNSUPPORTED_BY_RUN_TIME_CODE`. A `share_of` step's unit refusal earns
    none (`_share_units_refused`).

    Only `MethodChoiceNeeded`'s shape (`harness.graph.execute`'s `_refusal_shape`)
    carries a tool's `code` at its top level.
    """
    return {
        _UNSUPPORTED_BY_RUN_TIME_CODE[code]
        for stop in _stops(refusal)
        if isinstance(code := stop.get("code"), str)
        and code in _UNSUPPORTED_BY_RUN_TIME_CODE
        and not _share_units_refused(stop)
    }


def _share_units_repair_sentence(refusal: dict[str, Any] | None) -> str:
    """The repair a replan's message gives each `share_of` step that refused
    its sides on their units, as translation gives it for `SHARE_UNITS_DIFFER`.
    Empty when none did. The list of steps is cut at `MAX_QUESTION_CHARS`, as
    `_avoid_sentence` cuts its list of specs."""
    steps = [stop.get("step_id") for stop in _stops(refusal) if _share_units_refused(stop)]
    if not steps:
        return ""
    named = cut_user_text(", ".join(f"{step!r}" for step in steps), MAX_QUESTION_CHARS)
    return (
        f" To repair the share {named} refused: "
        f"{permitted_repair(CompileIssueCode.SHARE_UNITS_DIFFER)}."
    )


def _facet_repair_sentence(refusal: dict[str, Any] | None) -> str:
    """The repair a replan's message gives each step that refused its sides
    as one measure read from two sources (`_FACET_MISMATCH_CODE`). Empty when
    none did. The list of steps is cut at `MAX_QUESTION_CHARS`, as
    `_share_units_repair_sentence` cuts its own."""
    steps = [
        stop.get("step_id") for stop in _stops(refusal) if stop.get("code") == _FACET_MISMATCH_CODE
    ]
    if not steps:
        return ""
    named = cut_user_text(", ".join(f"{step!r}" for step in steps), MAX_QUESTION_CHARS)
    return (
        f" The step {named} read one measure from two sources on its two sides. To repair "
        "it, give every requirement on that measure the same source, by `source` or "
        "`source_spec_id`, the one either side already holds, and leave every other "
        "requirement's source alone."
    )


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
    then each of its `also_stopped` siblings' (`harness.graph.execute`'s `_replan_refusal`),
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


def _section(title: str, body: dict[str, Any] | None) -> str:
    return f"\n{title}:\n{json.dumps(body)}\n" if body else ""


__all__ = ["PlannerSeam", "PlannerSeamError"]
