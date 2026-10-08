"""Minimal root workflow graph for the DFO harness.

The graph is the deterministic spine of the architecture -- routing, depth,
retries, and execution are code; only converse / planner / compose / narrate
/ refusal generate. Each generative node has an injectable model seam
(`GenerativeNodes`, implementations in `harness.generative`). `converse`,
`compose`, `narrate` and `refusal` fall back to a scripted one-liner when
their seam is left unset; `planner` has no such fallback -- a plan is too
structured a thing to improvise -- so every caller supplies a
`PlannerModel`, scripted mode's being
`harness.scripted.ScriptedPlannerModel`. The graph still constructs and runs
offline with no model and no MCP server, since that seam is deterministic
Python, not a call to anything.

Everything external is injected (`resolver`, `executor`), the same pattern
`harness.agents.create_planner_with_translator`'s injectable `surface` uses.
The seams are async: ADK runs a sync function node inline on the event
loop, and the real implementations (`harness.mcp_client`) call the loopback
MCP mounts over HTTP on that same loop -- a sync seam would block the loop
and deadlock the loopback request. The depth-hint extractor stays sync (its google.genai client is
sync) and is offloaded to a worker thread for the same reason.

Behavioral guarantees the topology encodes:

* The four search statuses are four control flows. `unsupported` routes to
  a dead end with no outgoing edge -- structurally unretryable.
* The clarify loop is a conditional back-edge bounded by our own counter
  (ADK's graph engine has no iteration cap). Past the bound, the top-ranked
  candidate wins: the ranking is deterministic and another identical
  question is not progress.
* The depth policy (`harness.depth.decide_depth`) is a pure function whose
  `none` verdict skips the planner -- and therefore the approval gate --
  entirely. It never hands `none` to a resolved identity `graph_envelope`
  marks `wide`: a described-arm identity nothing downstream can pin to a
  single series (no elected item, or a role-bearing shape the pin
  deliberately leaves alone), or one the resolver itself calls truncated.
  That identity gets at least one clarifying question first.
* Approval is a graph interrupt with a stable id. Nothing downstream of
  `approve` exists on a path that skips it, and approval is version-bound:
  a plan amended since the reviewed version waits for a fresh approval.
* The `execute` node runs the approved plan's compiled workflow through
  `executor.run.execute_approved_plan`, which owns per-step retry, the
  per-capability transport policy, and the same `executor.classify.
  classify_refusal` policy `executor.run` uses for its own engine; the node
  itself only maps what comes back onto two edges. A refusal that carries a
  choice -- for the user or for a fresh discovery -- routes back to the
  planner with the diagnosis attached, so choosing the correction is a plan
  decision that goes back through the approval gate, never a retry the
  executor makes for itself. A refusal `classify_refusal` calls `STOP` is a
  dead end no replan reaches: the executor may not correct the call itself,
  and the user could not pick a narrower one from offered candidates
  either, so `execute` closes the turn immediately with the refusal's own
  remedy rather than spending a replan reproducing the identical failing
  call.
* Execution-triggered replanning is bounded: after `MAX_REPLANS` rounds the
  turn closes with what ran, naming the step that could not.

One approval gate: the graph's own `approve` interrupt. The planner's
`submit_plan` tool (`planner.agent`) declares the plan ready for the user's
answer and gates nothing itself -- only this graph's `execute` node, reached
downstream of `approve`, runs a step against a real server.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import (
    AsyncIterator,
    Callable,
    Collection,
    Coroutine,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Literal, Protocol, cast

import httpx2
from analysis_mcp import digest as analysis_digest
from common.materialisation import FrameRef
from common.telemetry import record_event, record_span
from compiler.evidence import errors
from compiler.frame_columns import FRAME_PARAMETER
from compiler.surface import ProgressCallback, ProgressReport, ToolSurface
from data_mcp import get_data_digest
from data_mcp.search_tool import MAX_PARTS
from data_mcp.settings import Settings as DataMcpSettings
from data_mcp.tool_errors import ToolErrorCode
from dfo_frame_grammar.grammar import FACET_IDENTITY_COLUMNS, UNIT_COLUMN
from executor.artifacts import declared_shape, parse_manifest
from executor.classify import (
    Disposition,
    ToolFailure,
    candidates_of,
    classify_refusal,
    parse_tool_failure,
)
from executor.outcomes import (
    Accepted,
    MethodChoiceNeeded,
    RediscoveryNeeded,
    RefusalKind,
    Refused,
    RunOutcome,
    StepRecord,
)
from executor.run import (
    FetchSeriesStats,
    StepProgressCallback,
    current_step_id,
    default_fetch_series_stats,
    execute_approved_plan,
)
from google.adk import Context, Event
from google.adk.agents import LlmAgent
from google.adk.events import RequestInput
from google.adk.workflow import START, Edge, Workflow, node
from google.adk.workflow._errors import NodeInterruptedError
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_event
from google.genai import types
from mcp import MCPError
from planner.plan import (
    ASKS_STATE_KEY,
    DROPPED_SEEDS_STATE_KEY,
    SEEDED_SOURCES_STATE_KEY,
    SEEDS_STATE_KEY,
    DatasetRequirement,
    Decision,
    EvidenceRequirement,
    PlanStatus,
    QuotientRequirement,
    ResolvedBy,
    SeededSource,
    dropped_seeds,
    load_asks,
    names_dropped_seed,
    plan_revision,
    save_dropped_seeds,
    save_seeded_sources,
    seeded_source,
    seeded_sources,
    stored_plan,
)
from planner.plan import Plan as PlannerPlan
from planner.workflow import ReadingSetBy, SelectedSource, WorkflowStep
from planner.workflow import Workflow as PlanWorkflow
from pydantic import BaseModel, Field, TypeAdapter
from translator.discovery import (
    DefinitionNotResolvedError,
)

from harness import plan_display
from harness.depth import DepthSignal, decide_depth
from harness.depth_hints import DepthHintExtractor
from harness.provenance import BOTH_SIDES_VERBS, summed_stat_var_count
from harness.seeds import (
    NO_OPERATIONS_ON_DOWNLOAD,
    ONE_DATASET_PER_DOWNLOAD,
    SEED_DROPPED,
    SEED_DUPLICATE,
    SEED_REFERENCE_UNUSED,
    SEED_STEP_PREFIX,
    SEED_STEP_TOOL,
    SEEDS_STEP_ID,
    DroppedSeed,
    ReadSeed,
    StaleSeedError,
    dropped_seeds_note,
    dropped_seeds_text,
    read_answer,
    refuse_seed,
    resolve_seed,
    seed_name,
    seed_step,
    seeded_envelope,
    seeds_received,
    take_seeds,
    unplanned_seeds_text,
)
from harness.tool_contracts import (
    DENOMINATOR_NOT_AVAILABLE,
    REFUSAL_ORIGIN_HARNESS,
    REFUSAL_ORIGIN_SERVICE,
    UNIT_NOT_RESOLVED,
    ResolverError,
    harness_refusal,
    platform_trace_id_of,
)
from harness.turn_records import PlanAsk, PlanRecord
from harness.wire import (
    usable_component,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "dfo_analyst_graph"

# Interrupt ids are stable strings: ADK matches a
# resuming function response to its interrupt by id, and both loops reuse
# their id across rounds.
CLARIFY_INTERRUPT = "clarify"
APPROVE_INTERRUPT = "approve"


MAX_CLARIFY_ROUNDS = 2

# Execution-triggered replans per turn. Each one is a new plan card and a new
# approval; past this the turn closes and names the step that could not run.
MAX_REPLANS = 1

# The `custom_metadata` key a relayed step progress report travels under on
# `execute`'s and `fetch`'s events (`_step_progress_event`), read by
# `harness.studio_translate._Translator.translate`.
STEP_PROGRESS_KEY = "step_progress"

# The `custom_metadata` key the `planner` node's first event travels under,
# as it starts a draft: `{"revising": bool}`, true when the draft revises a
# plan already on record, and `"untold_drops"`, the sentences naming the
# seeds the turn dropped that the user has not been told of, when there are
# any: a question the planner asks during the draft opens with them. Read by
# `harness.studio_translate._Translator`.
PLANNER_STARTED_KEY = "planner_started"

# The step id `fetch`'s lone `get_data` call runs under (`executor.run.
# current_step_id`), so its progress reports and its executor record name
# the same step. No plan runs on that path.
FETCH_STEP_ID = "fetch"

# The orchestrator engine's session state (`create_root_workflow`'s
# `orchestrator`), written by its tools (`harness.orchestrator_tools`) and
# read by the nodes below. Prefixed, so `GraphState` never validates it.
ORCHESTRATOR_FRAMES_KEY = "orchestrator:frames"  # frame id -> {"ref", "view"}
ORCHESTRATOR_SPECS_KEY = "orchestrator:specs"  # spec_id -> its search spec block
ORCHESTRATOR_PENDING_KEY = "orchestrator:pending"  # the turn's one pending action
ORCHESTRATOR_CALLS_KEY = "orchestrator:calls"  # tool name -> calls this turn
ORCHESTRATOR_BRIEF_KEY = "orchestrator:brief"  # the planner's question on this engine
ORCHESTRATOR_ASKED_KEY = "orchestrator:asked"  # the question `ask` is waiting on
ORCHESTRATOR_READINGS_KEY = "orchestrator:readings"  # spec_id -> a plan step's stat_vars reading
# Every search part this turn read, oldest first, as `SearchReading` JSON
# (`harness.turn_records`), failed parts included; `fetch` adds its outcome
# to the readings of its spec_id.
ORCHESTRATOR_SEARCHES_KEY = "orchestrator:searches"
# Every plan this turn ran, oldest first, as `PlanRecord` JSON, written by
# `plan_done`; `harness.studio` keeps it with the turn's record.
ORCHESTRATOR_PLAN_RECORDS_KEY = "orchestrator:plan_records"
# spec_id -> the warnings of the search part that returned it, as `search`
# stores its block under `ORCHESTRATOR_SPECS_KEY`.
ORCHESTRATOR_SPEC_WARNINGS_KEY = "orchestrator:spec_warnings"
# The search readings of the conversation's earlier turns, with the warnings
# of the spec_ids they name and the spec_ids whose block an earlier turn
# stored, seeded by `harness.studio` from `ConversationState.
# prior_ledger_state` on the orchestrator engine; unset when no earlier turn
# read any. The blocks themselves resolve through `planner.plan.
# earlier_block`.
ORCHESTRATOR_PRIOR_LEDGER_KEY = "orchestrator:prior_ledger"
# spec_id -> the refusal message of a fetch or a plan step that read it and
# was refused `capability_unavailable` this turn; `fetch` refuses it again
# without a call.
ORCHESTRATOR_UNAVAILABLE_KEY = "orchestrator:unavailable"
# Ask number -> the terminal frames of the latest run of this plan call that
# computed that ask (`_kept_asks`), written by `execute` when a run stops and
# read by `plan_done`; cleared by `plan_start` and `plan_done`.
ORCHESTRATOR_KEPT_ASKS_KEY = "orchestrator:kept_asks"
# The `custom_metadata` key an orchestrator node's own status line travels
# under ("Reading your question"), read by `_OrchestratorTranslator`.
ORCHESTRATOR_STATUS_KEY = "orchestrator_status"
# The status a plan hand-back carries when the planner made no plan, and
# when the approved plan ran; `_OrchestratorTranslator` shows the first as
# a failed step and the second as nothing.
ORCHESTRATOR_PLAN_FAILED_STATUS = "Reading the planner's answer"
ORCHESTRATOR_PLAN_RAN_STATUS = "Reading the plan's results"
# Characters of planner text one hand-back to the orchestrator may carry per
# field, and characters the plan record may take. The two recorded plans in
# `tests/fixtures/turn_records_recorded.json` render whole in 1,347 and 1,638
# characters. A larger plan folds steps first and loses decisions last
# (`PlanRecord.render`).
_HAND_BACK_CAP = 800
_PLAN_RECORD_CHARS = 2_000
# The branch the approval and `ask` interrupts are stamped with. No agent runs
# on it or under it, so ADK's branch filter keeps the interrupt's payload out
# of every model request; the browser reads the interrupt off the event stream.
_INTERRUPT_BRANCH = "dfo_interrupts"
# Lines of a step summary addressed to a tool caller handling frame references
# by hand: the frame reference and execution identity a digest ends with, and
# the note on a mint that ran as a queued operation, by the prefixes their
# renderers write them with. The orchestrator charts the frames `plan_done`
# registers under ids and fetches by a step's spec_id.
_SUMMARY_NOISE = (
    get_data_digest.FRAME_REFERENCE_PREFIX,
    get_data_digest.OPERATION_POLL_PREFIX,
    analysis_digest.FRAME_REFERENCE_PREFIX,
    analysis_digest.EXECUTION_IDENTITY_PREFIX,
)


def _interrupt(request: RequestInput) -> Event:
    """`request` as its interrupt event, on `_INTERRUPT_BRANCH`."""
    event = create_request_input_event(request)
    event.branch = _INTERRUPT_BRANCH
    return event


def _without_noise(summary: str) -> str:
    """`summary` without its `_SUMMARY_NOISE` lines and the blank lines they leave."""
    kept = "\n".join(line for line in summary.split("\n") if not line.startswith(_SUMMARY_NOISE))
    return re.sub(r"\n{3,}", "\n\n", kept).strip()


def _give_plan_back(ctx: Context) -> bool:
    """The first plan that could not be made or run gives the orchestrator
    its plan call back (`harness.orchestrator_tools.BUDGETS`), for one brief
    that avoids the failure; a second failure does not. Answers whether it
    gave the call back. The graph's own replan cap is untouched."""
    calls = dict(ctx.state.get(ORCHESTRATOR_CALLS_KEY) or {})
    if calls.get("plan_given_back"):
        return False
    calls["plan"] = 0
    calls["plan_given_back"] = True
    ctx.state[ORCHESTRATOR_CALLS_KEY] = calls
    return True


def _ask_status(
    ask: PlanAsk,
    ran: Mapping[str, dict[str, Any]],
    failed: Mapping[str, Any] | None,
    stopped: Collection[str],
) -> dict[str, Any]:
    """One ask as `plan_done` reports it: declined, with the plain reason, or
    each step answering it with its status, and for an ask answered in part
    the part not computed with its reason. `ran` holds each final step that
    produced a frame. `data` and `empty` name the frame, and `empty` means it
    holds no series. `not_chartable` gives the reason no view was prepared.
    A step outside `ran` is `failed` when it is in `stopped` (the failed step
    and every step whose inputs trace back to it, `_stopped_steps`), naming
    the failed step (`failed_at`) and its refusal `code`, and `not_run`
    otherwise."""
    if ask.declined is not None and not ask.outputs:
        return {"ask": ask.ask, "declined": plan_display.DECLINE_REASONS[ask.declined]}
    failure = {"status": "failed"}
    if failed is not None:
        failure["failed_at"] = str(failed.get("step_id"))
        if failed.get("code"):
            failure["code"] = str(failed["code"])
    status: dict[str, Any] = {
        "ask": ask.ask,
        "answered_by": [
            {
                "step": step,
                **ran.get(step, failure if step in stopped else {"status": "not_run"}),
            }
            for step in ask.outputs
        ],
    }
    if ask.declined is not None:
        status["not_computed"] = {
            "part": ask.not_computed or ask.ask,
            "reason": plan_display.DECLINE_REASONS[ask.declined],
        }
    return status


# The `_ask_status` statuses of a step that produced no frame.
_NOT_COMPUTED = frozenset({"failed", "not_run"})


def _computed(ask: Mapping[str, Any]) -> bool:
    """Whether every step answering `ask` (an `_ask_status` entry) produced its frame."""
    entries = ask.get("answered_by") or ()
    return bool(entries) and not any(entry["status"] in _NOT_COMPUTED for entry in entries)


def _partial_result(asks: list[dict[str, Any]] | None, *, frames: bool) -> str:
    """What a stopped plan computed, as `plan_done` states it: the asks whose
    every answering frame holds series (`data`) and can be charted, the asks
    computed with a frame that holds none or cannot be drawn, those an
    earlier plan computed, and the asks a failed or unrun step leaves not
    computed. An ask computed in part is named with the part not computed
    and its reason. Without a computed ask, whether any frame is listed."""
    computed = [ask for ask in asks or () if _computed(ask)]

    def computed_part(ask: Mapping[str, Any]) -> str:
        """`ask` quoted, with the part its plan declined and the reason."""
        declined = ask.get("not_computed")
        if declined is None:
            return json.dumps(ask["ask"])
        return (
            f"{json.dumps(ask['ask'])} except {json.dumps(declined['part'])}, "
            f"not computed because {declined['reason']}"
        )

    charted = [
        ask for ask in computed if all(entry["status"] == "data" for entry in ask["answered_by"])
    ]
    uncharted = [ask for ask in computed if ask not in charted]
    earlier = [ask["ask"] for ask in computed if ask.get("earlier_plan")]
    missing = [ask["ask"] for ask in asks or () if ask.get("answered_by") and not _computed(ask)]
    sentences = []
    if charted:
        sentences.append(
            "The frames listed compute these asks and can be charted: "
            + "; ".join(computed_part(ask) for ask in charted)
            + "."
        )
    if uncharted:
        sentences.append(
            "Computed, with a frame that holds no series or cannot be drawn, as each "
            "step's status says: " + "; ".join(computed_part(ask) for ask in uncharted) + "."
        )
    if earlier:
        sentences.append(
            "Of these, an earlier plan that stopped computed "
            + "; ".join(json.dumps(ask) for ask in earlier)
            + "; their steps are not in the plan record."
        )
    if not computed and frames:
        sentences.append("The frames listed are the results computed before the stop.")
    if missing:
        sentences.append("Not computed: " + "; ".join(json.dumps(ask) for ask in missing) + ".")
    return " ".join(sentences) or "Its result is not available."


def _stopped_steps(workflow: PlanWorkflow | None, failed_step: str | None) -> set[str]:
    """`failed_step` and every step of `workflow` whose bindings trace back to
    it. Steps are in run order, so one pass sees each step's inputs first."""
    if failed_step is None:
        return set()
    stopped = {failed_step}
    for step in workflow.steps if workflow is not None else ():
        if any(binding.from_step in stopped for binding in step.bindings.values()):
            stopped.add(step.id)
    return stopped


class PendingAsk(BaseModel):
    """`ask_user` asked: the workflow pauses with the question."""

    kind: Literal["ask"] = "ask"
    question: str
    options: list[str]


class PendingPlan(BaseModel):
    """`plan` asked: the workflow runs the plan path on the brief, holding the
    plan to `asks`, the outputs the user asked for."""

    kind: Literal["plan"] = "plan"
    brief: str
    asks: list[str] = []


_PENDING_ACTION = TypeAdapter(Annotated[PendingAsk | PendingPlan, Field(discriminator="kind")])


def pending_action(state: Any) -> PendingAsk | PendingPlan | None:
    """The one action the orchestrator asked the workflow for this turn, or None.
    `state` is any mapping-like read of session state (`Context.state`, a dict)."""
    raw = state.get(ORCHESTRATOR_PENDING_KEY)
    return _PENDING_ACTION.validate_python(raw) if raw else None


def frame_id_for(ref: dict[str, Any]) -> str:
    """The short id the orchestrator knows a frame by, from its reference, so
    the same frame gets the same id however many times it is registered."""
    digest = hashlib.sha1(json.dumps(ref, sort_keys=True).encode()).hexdigest()
    return f"f{digest[:6]}"


def register_frames(
    frames: dict[str, Any], described: list[tuple[dict[str, Any], dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Add each (frame ref, `describe_frame` view) to `frames` under its id,
    and return the entries as the orchestrator reads them."""
    entries = []
    for ref, view in described:
        frame_id = frame_id_for(ref)
        frames[frame_id] = {"ref": ref, "view": view}
        entries.append({"frame": frame_id, "view": view})
    return entries


def frame_row_ceiling() -> int:
    """The most rows one `get_data` retrieval returns: data-mcp's `max_frame_rows`
    as this process configures it (`MAX_FRAME_ROWS`). data-mcp runs in this
    process and refuses a frame past it (`over_ceiling`)."""
    return DataMcpSettings().max_frame_rows


def _clear_seed_slot(ctx: Context) -> None:
    """Clear the seeds slot (`GraphState.seeds`) on a route `seed` does not
    take, so up to `harness.wire.MAX_SEEDS` identities ride no later state
    write of the turn."""
    if ctx.state.get(SEEDS_STATE_KEY) is not None:
        ctx.state[SEEDS_STATE_KEY] = None


def _state_json(ctx: Context, key: str, default: Any = None) -> Any:
    """Decode a JSON blob out of session state, `default` when absent."""
    raw = ctx.state.get(key)
    return json.loads(raw) if raw else default


def _validated_revises(
    revises: Any, *, composed_ids: list[str] | None, auto_add: bool
) -> str | None:
    """The card id converse named as the one to change, when it really is a
    block on the canvas right now and this turn can replace one at all.
    Anything else is dropped with an info line and the turn proceeds as an
    append:

    - not a string, or empty;
    - `auto_add` is off (delta 11): the recorder never composes a block for
      anything this turn, so there is no block for any component to swap;
    - the id is not in `composed_ids`, the conversation's actual composed
      block ids. This is narrower than the canvas digest the model reads: a
      card-scoped follow-up's digest also carries its focus card even when
      that card was never composed, and naming it here must not read as a
      request to replace a block that does not exist.

    `land_component`'s own replace (`harness.studio_state._replace_block`)
    relies on this check, but a validated id here is not guaranteed to land.
    `/composition` (`harness.studio`'s route of that name) mutates the
    canvas without the turn lock, deliberately, since planner turns run
    40-60s and the canvas must stay editable, so the target can vanish
    between this check and the write, or the write itself can turn out
    unusable. `narrate`'s `revises_requested` fact still names what this
    turn's question asked for, set from this same validated value. Whether
    the change actually landed is a separate fact, `components_replaced`,
    read from the executor's own `components_replaced` capability
    (`_RecordingExecutor.components_replaced`, `harness.studio_translate`),
    which reports the drain-time replace-or-append decision
    (`harness.studio_state.land_component`, called from
    `_component_frames`), closing this gap entirely rather than
    narrowing it.

    Card identity alone decides the match. The model reads the digest,
    which names the variable a card shows. "add Guatemala to the chart" and
    "now show me Guatemala" share a variable and differ in intent, so
    structure cannot decide which the user meant.
    """
    if not isinstance(revises, str) or not revises:
        return None
    if not auto_add:
        logger.info(
            "converse named revises=%r, but auto_add is off this turn -- no block on the "
            "canvas to replace; appending instead",
            revises,
        )
        return None
    if revises not in (composed_ids or []):
        logger.info(
            "converse named revises=%r, which is no block on this canvas; appending instead",
            revises,
        )
        return None
    return revises


def _top_candidate_spec_id(node_input: dict[str, Any]) -> str | None:
    """The resolution's default reading: the top-ranked candidate."""
    candidates = node_input.get("candidates") or [{}]
    return candidates[0].get("spec_id")


def _candidate_wide(candidates: list[dict[str, Any]], spec_id: str | None) -> bool:
    """The chosen candidate's own `wide` verdict (`harness.tool_contracts.
    graph_envelope`'s `candidates` branch computes one per offered spec, the
    same predicate it applies to a `definite` reading). `clarify` reads this
    off the un-resumed envelope's candidates before it overwrites
    `ctx.state["spec_id"]`, and stores it alongside so `probe`'s re-entry
    resolve call -- `spec_id` set, no upstream `search` -- knows the width
    of the identity it is re-entering with, instead of defaulting narrow."""
    return next(
        (bool(c.get("wide", False)) for c in candidates if c.get("spec_id") == spec_id),
        False,
    )


def _candidate_label(candidates: list[dict[str, Any]], spec_id: str | None) -> str | None:
    """The chosen candidate's own `label`, the reading text the user saw and
    picked -- carried forward the same way `_candidate_wide` carries `wide`,
    so `harness.planner_seam` can name what the user chose in the planner's
    own prompt instead of a bare spec_id."""
    return next(
        (str(c["label"]) for c in candidates if c.get("spec_id") == spec_id and c.get("label")),
        None,
    )


def _fetched_seed(ctx: Context) -> SeededSource | None:
    """The seeded dataset `fetch` retrieves (the state's `spec_id`), on a seeded turn."""
    spec_id = ctx.state.get("spec_id")
    return seeded_source(seeded_sources(ctx.state), spec_id)


class GraphState(BaseModel):
    """Session-state schema for the workflow (validated on every delta)."""

    question: str = ""
    surface: str | None = None
    depth_hint: str | None = None
    search_parts: str | None = None
    # This turn's auto-add preference (`studio.py`'s `body.auto_add`,
    # mirroring `ConversationState.auto_add`) and the component ids
    # actually composed on the canvas right now (`conversation.composition.
    # blocks`), both seeded fresh per turn (`StudioRunner`'s per-turn ADK
    # session). Distinct from `canvas_digest`, which the model reads and
    # which also carries a card-scoped follow-up's focus card even when
    # that card was never composed. `_validated_revises` checks these, not
    # the digest, so a `revises` target only survives when the recorder can
    # really swap it in.
    auto_add: bool = True
    composed_ids: str | None = None
    # The canvas block this turn's question asks to change, when converse
    # named one that really is on the canvas. See `converse`. Read by the
    # component recorder, which swaps that block in place.
    revises: str | None = None
    resolve_envelope: str | None = None
    # Every search part's summary (`harness.tool_contracts.graph_envelope`'s
    # `part_digests`), kept apart from `resolve_envelope`, which the planner
    # reads. See `probe`.
    part_digests: str | None = None
    # Whether `probe` has searched again for a part search refused for its
    # denominator or its unit (`split_denominators`). It does so once for
    # each question `converse` drafts: `clarify` clears it when a free-text
    # answer sends the question back to be drafted again.
    denominator_split_tried: bool = False
    canvas_digest: str | None = None
    conversation_digest: str | None = None
    # The window a window anaphor in this turn's question should resolve
    # to (`"<start> to <end>"`), from `ConversationState.anaphor_window`;
    # see `converse`. Host-seeded fresh per turn, like `canvas_digest`.
    anaphor_window: str | None = None
    # The Search results a "... in Studio" action handed over, as JSON: each
    # seed as `seeds_slot` keeps it, or the one reference `TurnRequest.seed_spec`
    # also takes. Host-seeded fresh per turn under `planner.plan.SEEDS_STATE_KEY`;
    # `seed` reads it once and clears it.
    seeds: str | None = None
    spec_id: str | None = None
    # The `spec_id`'s own `wide` verdict, carried forward from the
    # `candidates` reading `clarify` chose it from -- see
    # `_candidate_wide`. Stale once `spec_id` changes for any other reason,
    # but every path that sets `spec_id` also sets this alongside it.
    spec_id_wide: bool = False
    # Whether `spec_id` is the user's own pick, answering a `clarify`
    # round, as opposed to a system default (skip, the round bound, or no
    # clarify at all) -- see `clarify` and `probe`. `spec_id_elected_label`
    # is the picked candidate's own `label`, the reading text the user saw.
    spec_id_elected: bool = False
    spec_id_elected_label: str | None = None
    parts: int = 1
    clarify_rounds: int = 0
    awaiting_clarify: bool = False
    awaiting_approve: bool = False
    depth: str | None = None
    depth_reason: str | None = None
    depth_notice: str | None = None
    plan: str | None = None
    replans: int = 0
    # The seeds this turn dropped that the user has not been told of yet, as
    # JSON `[{"name": ..., "reason": ...}]` (`_record_drops`, `DroppedSeed`).
    dropped_untold: str | None = None


# Every `GraphState` field a node derives from *this turn's* question, as
# opposed to a field the host seeds fresh per turn and no node ever writes
# (`surface`, `auto_add`, `composed_ids`, `canvas_digest`,
# `conversation_digest`, `anaphor_window`, `seeds`, see `studio.py`'s
# `body`). A studio turn opens a brand-new ADK session per turn
# (`ConversationState.session_id = f"{conversation_id}#t{turn_index}"`), so
# every key below starts unset there. `evals/agents/dfo_analyst_graph`
# (`just adk-web`) keeps one session across many turns: a stale `spec_id`
# makes `McpResolver.__call__` treat it as a clarify re-entry and skip
# searching, and `clarify_rounds`/`replans` start already spent. `converse`
# clears every key here, back to its own `GraphState` default, on the edge
# that starts a genuinely new turn (see `converse`).
PER_TURN_STATE_KEYS: tuple[str, ...] = (
    "question",
    "depth_hint",
    "search_parts",
    "revises",
    "resolve_envelope",
    "part_digests",
    "denominator_split_tried",
    "spec_id",
    "spec_id_wide",
    "spec_id_elected",
    "spec_id_elected_label",
    "parts",
    "awaiting_clarify",
    "clarify_rounds",
    "awaiting_approve",
    "depth",
    "depth_reason",
    "depth_notice",
    "plan",
    "replans",
    "dropped_untold",
)


# What the plan card opens with when the plan the user held was stored in a
# shape this release does not read, and the question was planned again.
UNREADABLE_PLAN_NOTE = (
    "The plan you were shown could not be read after an update, so this is a new plan for "
    "the same question. Ask again for any change you wanted."
)


# What the closing text opens with when the plan the user held was stored in
# a shape this release does not read, and the planner drafted no plan in its
# place. The planner's own refusal follows it.
UNREADABLE_PLAN_UNPLANNED_NOTE = (
    "The plan you were shown could not be read after an update, and the question could not "
    "be planned again."
)


def _readable_plan(ctx: Context, raw: str | None) -> PlannerPlan | None:
    """The plan `raw` holds, `ctx.state["plan"]` as a node read it; `None` when it holds none.

    A plan stored in a shape `planner.plan.Plan` rejects is none
    (`planner.plan.stored_plan`), and the slot is cleared here, by the node
    reading it, so no later node reads it again. `planner` then drafts the
    question from the start: its own entry, or `approve`'s and `execute`'s
    `{"unreadable_plan": True}` output, says so.
    """
    plan = stored_plan(raw, state_key="plan")
    if raw and plan is None:
        ctx.state["plan"] = None
    return plan


def _reset_turn_state(ctx: Context) -> None:
    """Clear every `PER_TURN_STATE_KEYS` entry back to its own `GraphState`
    default, never a bare `None`. `google.adk.sessions.state.State.
    __setitem__` validates each write against `GraphState` (a `state_schema`
    on `create_root_workflow`'s `Workflow`), and a non-optional field
    (`clarify_rounds`, `replans`, `awaiting_clarify`, `awaiting_approve`,
    `spec_id_wide`, `spec_id_elected`, `parts`) rejects `None` outright."""
    for key in PER_TURN_STATE_KEYS:
        ctx.state[key] = GraphState.model_fields[key].default
    # Only `seed` writes the user's datasets, on the turn they start, and
    # only `seed` drops one.
    ctx.state[SEEDED_SOURCES_STATE_KEY] = None
    ctx.state[DROPPED_SEEDS_STATE_KEY] = None


class Resolver(Protocol):
    """Stands in for data-mcp `search`: question -> resolve envelope.

    The envelope carries the decision inputs a search digest exposes:
    `status` (definite / candidates / no_data / unsupported), `spec_id`
    when resolved, `candidates`, `warnings`, `choice_points`, `parts`, and
    -- on the `definite` route only -- `wide` and `spec_block`, the raw
    `spec` object the search that settled `spec_id` carried, for
    `harness.planner_seam` to pass on to a pinned requirement that names the
    same id (`None` on a clarify re-entry, which re-searches nothing).

    `wide` is also an INPUT, not only an output: a clarify re-entry calls
    this with `spec_id` set, and a real resolver (`harness.mcp_client.
    McpResolver`) answers that call with no upstream `search` at all, since
    re-searching would only reproduce the fork the user's choice just
    settled. It still needs a `wide` verdict for the identity it is
    re-entering with, and `probe` passes it the one `clarify` carried
    forward from the candidate the user picked (`_candidate_wide`) so the
    depth gate downstream (`choose_depth`) sees the real width instead of
    defaulting narrow.
    """

    async def __call__(
        self,
        *,
        question: str,
        spec_id: str | None = None,
        wide: bool = False,
        parts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]: ...


class Executor(Protocol):
    """Runs one plan step against the data layer (data-mcp / analysis-mcp).

    Returns `{"outcome": "ok", "frame_refs": [...], "summary": str?}` (the
    optional summary is a one-line human result for the Activity feed) or
    `{"outcome": "refusal", "refusal": {code, retry_mode, fix, message,
    origin?, advice?, retry_after_s?}}` -- the refusal grammar's shape, minus
    transport. `harness.mcp_client.McpExecutor` retries a `retry_after_delay`
    refusal itself before returning it.
    `origin` is `"service"` when an MCP service refused and `"harness"` when
    the executor itself withheld or misrouted the call; an executor that
    states none is read as unattributed.

    A concrete executor may also define `land_component(self, component:
    dict[str, Any], *, land: bool, revises: str | None) -> None`, an
    optional, synchronous capability `compose` probes with `getattr`
    (never required by this Protocol, matching `_RecordingExecutor.
    aclose`'s own optional-capability pattern). `compose` calls it
    immediately after the write it belongs to returns, with no `await` in
    between, to stamp (never decide or mutate) that write's own record:
    `land=True` for every component that passes `harness.wire.
    usable_component`, `revises` on the first usable component only. The
    actual replace-or-append decision, and the canvas mutation itself,
    happen later, at drain time (`harness.studio_translate`'s
    `_component_frames`), which ADK's own event-processing order
    guarantees runs before `narrate` starts. Its absence means no write
    this turn ever lands.

    An executor may also define `components_replaced(self) -> bool`,
    another optional, synchronous capability, this one probed by `narrate`.
    It reports whether a component landed by THIS turn actually replaced
    an existing block, the true outcome of the drain-time decision above.
    `narrate` calls it after `compose`, by which point every write in the
    turn has already been drained. Its absence means `narrate`'s
    `components_replaced` fact is always `False`.
    """

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        """Run `tool`; `on_progress`, when given, hears the call's progress reports."""
        ...


class ConverseModel(Protocol):
    """Generates the turn's opening text and, optionally, structured search
    parts (`data_mcp.search_tool.SearchPart`-shaped dicts) parsed from the
    question, plus `about` ("data" | "canvas") when a canvas digest is
    supplied. Returns `{"text": str, "search_parts": list[dict] | None,
    "about": str | None, "revises": str | None, "recall": {"scope",
    "query"} | None}` -- `recall` asks the harness for context beyond the
    digests (older turns, full card facts); the node answers it mechanically
    and re-runs the call once. `revises` names an existing canvas card whose
    content the question asks to change. The node validates it against the
    conversation's own canvas before storing it, and drops an id that names
    no block there."""

    async def __call__(
        self,
        *,
        question: str,
        canvas_digest: dict[str, Any] | None = None,
        conversation_digest: dict[str, Any] | None = None,
        recall_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class RecallFn(Protocol):
    """Serves a converse recall request from the conversation store --
    mechanical (no model): digests and fact summaries, never rows."""

    async def __call__(self, *, scope: str, query: str) -> dict[str, Any]: ...


class RefusalModel(Protocol):
    """Writes the plain-language close for a turn that cannot be answered,
    from the resolver's diagnostics alone. Returns `{"text": str}`."""

    async def __call__(
        self, *, question: str, status: str, warnings: list[str]
    ) -> dict[str, Any]: ...


class DenominatorSplitModel(Protocol):
    """Names the two measured variables whose division gives a search part
    that search refused for its denominator (a per-head figure with no such
    series). `part` is the refused part as converse drafted it and `message`
    what search said of it. Returns `{"measure": str, "measure_unit": str |
    None, "denominator": str, "denominator_unit": str | None,
    "denominator_entities": list[int], "denominator_keeps_roles": bool}`,
    where `denominator_entities` are positions in `part`'s entities. `probe` builds the two search parts
    from `part` and that draft (`_measure_and_denominator`), so no draft
    changes an entity or the time."""

    async def __call__(
        self, *, question: str, part: dict[str, Any], message: str
    ) -> dict[str, Any]: ...


class PlannerModel(Protocol):
    """Drafts (or redrafts) a compiled plan from the resolve envelope.

    Returns `{"plan": <planner.plan.Plan as JSON>, "summary": <approval summary>}`
    on a draft, or `{"refused": {"text": <plain sentence>, "final": <bool>}}`
    when the seam ends the entry without one (`harness.planner_seam.
    PlannerSeam._refused`); `final` says no other plan for the same ask can
    succeed. The node validates a drafted
    plan itself, so a malformed one fails here rather than at the approve
    gate; a refusal never writes `ctx.state["plan"]`, so whatever plan was
    already there is exactly what this call leaves behind. Where that
    record is a plan still `PENDING_APPROVAL` -- an amendment that hit the
    cap trying to redraft it -- the node routes back to `approve`, offering
    that same, unmodified plan again rather than ending the turn on a
    version the user could no longer act on; only a refusal with nothing
    `PENDING_APPROVAL` on record (the cap hit on a first draft, or on a
    replan against an already-approved plan) ends the turn at `deadend`.
    `prior_plan` is the record the graph holds, which the seam seeds its
    own session from before drafting. `refusal` / `amendment` are set on
    the replan and amend re-entries.

    `on_progress`, when given, may be called any number of times before this
    returns, each call naming one round trip the underlying model pair made
    (a shape this protocol treats as opaque -- the node only forwards it as
    an activity update). A seam with nothing to report, scripted mode's
    included, simply never calls it; the parameter is additive so every
    existing `PlannerModel` keeps working unchanged.

    `ctx` is the planner node's own `Context`, passed through so a seam
    that hosts its models as a dynamically-scheduled child of this node
    (`ctx.run_node`) can do so in the graph's own session, rather than
    building a second `Runner`/session service of its own
    (`harness.planner_seam.PlannerSeam`). A seam with nothing to run
    through it, scripted mode's included, simply never touches it.

    `spec_id_elected`/`spec_id_elected_label` mirror the `GraphState` fields
    of the same name (see there for what they mean; `clarify` and `probe`,
    in this module, set them) and exist so a seam can keep that pin visible
    on a `parts > 1` question, where an unelected `spec_id` is otherwise
    left out as not necessarily right for every part (`harness.
    planner_seam.PlannerSeam`). A seam with no use for the distinction,
    scripted mode's included, simply ignores both.

    `readings` is passed on the orchestrator engine only: the search
    readings of the conversation's earlier turns
    (`ORCHESTRATOR_PRIOR_LEDGER_KEY`) and then this turn's
    (`ORCHESTRATOR_SEARCHES_KEY`), oldest first, which a seam shows its
    planner where the classic engine shows the envelope.
    """

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
    ) -> dict[str, Any]: ...


class ComposeModel(Protocol):
    """Chooses canvas component intents for the produced frames. Receives one
    describe_frame digest per frame ref, shaped `{"available": True, "view":
    {...}}` on success or `{"available": False, "reason": str}` when
    describe_frame refused -- a gap the model must name, never paper over by
    drafting from a different ref's view. `frame_provenance` tags each ref,
    in the same order, with the plan step index and tool that produced it
    and whether that step is FINAL -- the plan's terminal step, stated by
    the producer rather than inferred from position, since a plan can
    answer through more than one terminal step -- so the model can tell
    which refs are the question's actual answer instead of picking whichever
    ref happens to carry a view. Returns `{"intents": [...]}` where each
    intent is a platform write intent (type/frame_ref/select/title). Both
    rules above are instructions to the model, not the enforcement of them:
    the compose node's `_admissible_intents` re-checks every intent this
    seam returns -- available view, final-step frame -- before writing
    anything, so an implementation that ignores either rule writes nothing
    rather than the wrong frame."""

    async def __call__(
        self,
        *,
        question: str,
        frame_refs: list[dict[str, Any]],
        frame_views: list[dict[str, Any]],
        frame_provenance: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


class NarrateModel(Protocol):
    """Writes the closing message from bounded facts only (component fact
    summaries and frame refs -- never rows). Returns `{"text": str}`."""

    async def __call__(self, *, question: str, facts: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True, kw_only=True)
class GenerativeNodes:
    """The generative nodes' model seams. `converse`, `compose`, `narrate`
    and `refusal` left `None` keep that node's scripted stand-in: the
    feature is absent. `planner` left `None` means the graph cannot reach
    its `planner` node at all -- the node raises rather than drafting a
    different kind of plan. `denominator_split` left `None` means `probe`
    searches once, and a part refused for its denominator closes the turn."""

    converse: ConverseModel | None = None
    planner: PlannerModel | None = None
    compose: ComposeModel | None = None
    narrate: NarrateModel | None = None
    refusal: RefusalModel | None = None
    denominator_split: DenominatorSplitModel | None = None


# Closed vocabulary for a describe_frame refusal, bucketed by retry_mode so
# no platform string -- not `detail`, not `message`, not the refusal `code`
# itself -- ever reaches a generative seam. Those carry internal tokens
# (literal spec_id/content_version values, upstream refusal codes) that
# `_NARRATE_INSTRUCTION` and `_COMPOSE_INSTRUCTION` were never taught to
# withhold.
_FRAME_UNAVAILABLE_REASONS: dict[str, str] = {
    "retry_after_delay": "this step's data view is temporarily unavailable",
    "do_not_retry": "this step's data view could not be prepared",
}


def _frame_unavailable_reason(refusal: dict[str, Any]) -> str:
    """A safe, closed-vocabulary reason a `describe_frame` refusal produced
    no view. Never the raw refusal text: see the module comment above."""
    retry_mode = str(refusal.get("retry_mode"))
    return _FRAME_UNAVAILABLE_REASONS.get(retry_mode, _FRAME_UNAVAILABLE_REASONS["do_not_retry"])


# The same closed-vocabulary discipline as `_FRAME_UNAVAILABLE_REASONS`,
# for a `write_component` refusal (or a written-but-unusable component):
# no platform string -- not `message`, not `advice`, not the refusal `code`
# itself -- reaches narrate through `refused_writes`, since
# `_NARRATE_INSTRUCTION` forbids internal jargon (spec ids, content versions,
# status codes) even when a fact contains it, and a platform
# `detail`/`message` routinely does. The trace event `compose.write_refused`
# records the raw code, origin, retry mode, advice and the platform's trace
# id; the platform's message is logged (`_log_refusal`) and not recorded on
# the trace. The message never reaches `refused_writes`, the narrate facts or
# the compose output. It appears in the Activity step detail and the Edit
# Card closing text, which read the refusal directly.
#
# Each string reads after "A component was not added: ". Only a refusal
# stamped `origin == "service"` can name the platform. A card the harness
# withheld says so (`_RW_UNSOURCED`, `_RW_TOO_MANY_SERIES`, `_RW_INTERNAL`), and a refusal with no
# recognised `origin` gets `_RW_UNKNOWN`, which names nobody.
# dfo-apis's `MAX_SERIES_PER_COMPONENT`: the assembler refuses a component
# whose selection's estimated series count (`_series_estimate`) exceeds this.
MAX_SERIES_PER_COMPONENT = 24
# The catalogue grammar's `maxItems` on a `line_chart` select's one array
# value (the release document's `intent_schemas_tolerant.line_chart`). The
# platform refuses a longer list as `schema_invalid`.
LINE_CHART_MAX_VALUES = 8
# dfo-apis's `facet_dimension_names` (dfo-data-tools' `assembler/frame_view.py`):
# the facet columns a select may leave unbound. The platform refuses one only
# when it splits a series, and its estimate counts an unbound facet as 1.
_FACET_DIMENSIONS = frozenset({UNIT_COLUMN, *FACET_IDENTITY_COLUMNS})

_RW_SELECTION = "the selection did not identify a single series"
_RW_SHAPE = "the component request was not a shape the canvas accepts"
_RW_RULES = "the platform's chart rules do not allow that chart for this data"
_RW_TOO_LARGE = "the chart asked for more than the platform allows on one card"
_RW_REJECTED = "the platform could not accept the component request"
_RW_DATA_VIEW = "the platform could not load this data view"
_RW_DATA_VIEW_REFRESH = "the platform could not load this data view; asking again re-runs the query"
_RW_TEMPORARY = "the platform was temporarily unavailable; asking again may work"
_RW_BUSY = "the assistant was busy with other large requests; asking again shortly may work"
_RW_UNCONFIRMED = "the platform did not confirm that the card was added"
_RW_NOT_CONFIGURED = "this deployment is not set up to present that kind of result"
_RW_UNSOURCED = "the card was withheld because no citable source could be tied to this data view"
_RW_TOO_MANY_SERIES = (
    "the card was withheld because its selection covers more than "
    f"{MAX_SERIES_PER_COMPONENT} series, the most one card draws"
)
_RW_INTERNAL = "the assistant could not build the card"
_RW_UNKNOWN = "the card could not be added"

REFUSED_WRITE_VOCABULARY: frozenset[str] = frozenset(
    {
        _RW_SELECTION,
        _RW_SHAPE,
        _RW_RULES,
        _RW_TOO_LARGE,
        _RW_REJECTED,
        _RW_DATA_VIEW,
        _RW_DATA_VIEW_REFRESH,
        _RW_TEMPORARY,
        _RW_BUSY,
        _RW_UNCONFIRMED,
        _RW_NOT_CONFIGURED,
        _RW_UNSOURCED,
        _RW_TOO_MANY_SERIES,
        _RW_INTERNAL,
        _RW_UNKNOWN,
    }
)

# Codes a presentation service returns, by what the user can act on. A code
# not listed (`not_found`, `internal_error`, `unspecified`, or any other)
# reads as `_RW_UNKNOWN`, which names nobody.
_REFUSED_WRITE_BY_CODE: dict[str, str] = {
    "binding_invalid": _RW_SELECTION,
    "schema_invalid": _RW_SHAPE,
    "constraint_violation": _RW_RULES,
    "resource_exhausted": _RW_TOO_LARGE,
    "request_invalid": _RW_REJECTED,
    "unsupported_schema_version": _RW_REJECTED,
    "frame_unavailable": _RW_DATA_VIEW,
    "spec_not_found": _RW_DATA_VIEW,
    "snapshot_not_found": _RW_DATA_VIEW,
    "snapshot_revoked": _RW_DATA_VIEW,
    "artifact_expired": _RW_DATA_VIEW,
    "artifact_not_found": _RW_DATA_VIEW,
    # Only reached for a `do_not_retry` mode: the create may have reached the
    # platform before the transport failed. A retry-safe failure is `_RW_TEMPORARY`.
    "transient_unavailable": _RW_UNCONFIRMED,
    "publish_not_configured": _RW_NOT_CONFIGURED,
    "publish_not_permitted": _RW_NOT_CONFIGURED,
}

# Codes the harness mints itself (`REFUSAL_ORIGIN_HARNESS`); any other
# harness code is an internal defect.
_SERIES_LIMIT_EXCEEDED = "series_limit_exceeded"
_SELECT_INVALID = "select_invalid"
_REFUSED_WRITE_BY_HARNESS_CODE: dict[str, str] = {
    "provenance_unavailable": _RW_UNSOURCED,
    _SERIES_LIMIT_EXCEEDED: _RW_TOO_MANY_SERIES,
}


def _refused_write_reason(refusal: Mapping[str, Any]) -> str:
    """A safe, closed-vocabulary reason a `write_component` intent produced
    no card. The raw refusal `message`, `fix` and `advice` stay out of it: see
    the comment above.

    The `origin` decides who can be named. A refusal whose `origin` is neither
    `REFUSAL_ORIGIN_SERVICE` nor `REFUSAL_ORIGIN_HARNESS` (absent, or a value
    no producer stamps) names nobody and reads as `_RW_UNKNOWN` for every code
    and `retry_mode`. Precedence for a harness refusal: the harness code table,
    then `_RW_INTERNAL`. For a service refusal: `retry_mode ==
    "retry_after_delay"` (any code), then the code table (a data-view code
    reads as `_RW_DATA_VIEW_REFRESH` when `retry_mode` is `refresh_then_retry`
    and as `_RW_DATA_VIEW` otherwise), then `_RW_UNKNOWN`. A `retry_after_delay`
    refusal whose `fix` carries `rows_requested` came from the frame row budget
    of the process serving the call (`common.row_budget`), which every service
    refuses with that field, and reads as `_RW_BUSY`; any other, such as
    presentation-mcp's `transient_unavailable` for a platform it could not
    reach, reads as `_RW_TEMPORARY`.
    """
    origin = refusal.get("origin")
    code = str(refusal.get("code"))
    retry_mode = refusal.get("retry_mode")
    if origin == REFUSAL_ORIGIN_HARNESS:
        return _REFUSED_WRITE_BY_HARNESS_CODE.get(code, _RW_INTERNAL)
    if origin != REFUSAL_ORIGIN_SERVICE:
        return _RW_UNKNOWN
    if retry_mode == "retry_after_delay":
        fix = refusal.get("fix")
        return _RW_BUSY if isinstance(fix, Mapping) and "rows_requested" in fix else _RW_TEMPORARY
    reason = _REFUSED_WRITE_BY_CODE.get(code, _RW_UNKNOWN)
    if reason == _RW_DATA_VIEW and retry_mode == "refresh_then_retry":
        return _RW_DATA_VIEW_REFRESH
    return reason


def _unusable_component_refusal(component: dict[str, Any] | None) -> dict[str, Any]:
    """The refusal `compose` records for a `write_component` that answered
    `ok` with nothing to draw a card from: no component at all, or one
    that fails the `component.created` wire contract (`harness.wire.
    usable_component`, the same check the drain re-runs right before a
    write actually lands); a missing `fact_summary` is one instance of
    that, never checked on its own."""
    if component is None:
        return harness_refusal(
            code="component_missing", message="write_component returned ok with no component"
        )
    return harness_refusal(
        code="component_invalid",
        message="the written component failed the component.created wire contract",
    )


def _log_refusal(summary: str, refusal: Mapping[str, Any], **context: object) -> None:
    """Log `refusal` at WARNING as `summary: k=v ...`: `context`, then the raw
    `code`, `origin`, `retry_mode` and `message`, then the platform's own
    `platform_trace_id` when the refusal carries one."""
    fields = {
        **context,
        "code": refusal.get("code"),
        "origin": refusal.get("origin"),
        "retry_mode": refusal.get("retry_mode"),
        "message": refusal.get("message"),
    }
    platform_trace_id = platform_trace_id_of(refusal)
    if platform_trace_id:
        fields["platform_trace_id"] = platform_trace_id
    logger.warning("%s: %s", summary, " ".join(f"{key}={value!r}" for key, value in fields.items()))


def _record_refusal_event(
    name: str, refusal: Mapping[str, Any], *, reason: str, **context: str | None
) -> None:
    """Record `refusal` on the current trace as the event `name`: the raw
    `code`, `origin`, `retry_mode`, the `reason` the model was given, `advice`
    (cut to 300 characters; the only carrier of the violated constraint id),
    the platform's own `platform_trace_id` when the refusal carries a
    non-empty one, plus `context`. The platform `message` is left off, and so
    is any field with no value. Of the fields above, `compose` puts only
    `reason` into the `refused_writes` it returns, and `platform_trace_id` appears in no
    Activity row."""
    advice = refusal.get("advice")
    metadata: dict[str, str | int | float | bool | None] = {
        "code": refusal.get("code"),
        "origin": refusal.get("origin"),
        "retry_mode": refusal.get("retry_mode"),
        "reason": reason,
        "platform_trace_id": platform_trace_id_of(refusal),
        "advice": str(advice)[:300] if advice else None,
        **context,
    }
    record_event(
        name, metadata={key: value for key, value in metadata.items() if value is not None}
    )


def _plural_dimension_names(view: dict[str, Any]) -> set[str]:
    """The view's non-facet dimension names marked `plural: true` -- the ones
    a `stat_card` select must pin to a single value or the platform resolves
    more than one series. A facet column (`_FACET_DIMENSIONS`) is exempt: the
    platform refuses an unbound facet only when it splits a series, and a
    plural `unit` such as `{USDollar, null}` often names one series with a
    unit on some rows. A dimension entry that is not a `{name, plural,
    values}` mapping (older scripted fixtures list bare dimension names)
    carries no plurality signal, so it is skipped rather than guessed at."""
    return {
        str(dim["name"])
        for dim in view.get("dimensions") or []
        if isinstance(dim, dict)
        and dim.get("plural")
        and dim.get("name")
        and dim["name"] not in _FACET_DIMENSIONS
    }


def _select_binding_ok(
    *, type_: str, select: dict[str, Any], view: dict[str, Any]
) -> dict[str, Any] | None:
    """The binding shapes the platform certainly refuses (measured live
    against a 7-series frame, and against the catalogue grammar's stated
    per-type caps): a select binding a dimension the view does not
    declare; a `stat_card` select that leaves a non-facet plural dimension
    unpinned; for `line_chart` and `bar_chart`, a select that arrays more
    than one dimension; a `line_chart` array longer than
    `LINE_CHART_MAX_VALUES`; and for every other type, a select whose
    `_series_estimate` exceeds `MAX_SERIES_PER_COMPONENT` (24), the cap
    dfo-apis's assembler enforces although the catalogue grammar's
    `maxItems` allows an array of 30. `table` may array more than one
    dimension; the catalogue grammar restricts the single-array-dimension
    rule to `line_chart` and `bar_chart`. A non-mapping `select`
    (a model that emitted a JSON array or scalar for select_json)
    is a rejection reason here rather than an `AttributeError` on
    `.items()` -- the parse site in `harness.generative.GeminiCompose`
    already skips this shape, but this guard does not assume its only
    caller is disciplined. Returns `None` when the binding is fine, else
    the harness refusal that says why it is not: `series_limit_exceeded`
    for a select over the series cap and `select_invalid` for every other
    shape, each with the reason as its `message`. It is caught here, before
    a `write_component` round trip, the same discipline
    `_admissible_intents` already applies to a hallucinated frame_ref or a
    non-final frame."""
    if not isinstance(select, dict):
        return _select_invalid(f"{type_} select is not a JSON object: {select!r}")
    declared = _declared_dimensions(view)
    undeclared = sorted(select.keys() - declared.keys()) if declared else []
    if undeclared:
        return _select_invalid(
            f"{type_} select binds {undeclared}, which the frame does not declare"
        )
    array_dims = sorted(name for name, value in select.items() if isinstance(value, list))
    if type_ == "stat_card":
        reason = _stat_card_binding_reason(select=select, view=view, array_dims=array_dims)
        return _select_invalid(reason) if reason is not None else None
    array_issue = _chart_array_refusal(type_=type_, select=select, array_dims=array_dims)
    if array_issue is not None:
        return array_issue
    series = _series_estimate(select=select, declared=declared)
    if series > MAX_SERIES_PER_COMPONENT:
        return harness_refusal(
            code=_SERIES_LIMIT_EXCEEDED,
            message=(
                f"{type_} select expands to an estimated {series} series, "
                f"over the platform's maximum of {MAX_SERIES_PER_COMPONENT}"
            ),
        )
    return None


def _chart_array_refusal(
    *, type_: str, select: dict[str, Any], array_dims: list[str]
) -> dict[str, Any] | None:
    """The refusal of a `line_chart` or `bar_chart` select that arrays more
    than one dimension, or of a `line_chart` array longer than
    `LINE_CHART_MAX_VALUES`; None for any other select."""
    if type_ not in ("line_chart", "bar_chart"):
        return None
    if len(array_dims) > 1:
        return _select_invalid(
            f"{type_} select carries array values for more than one dimension: {array_dims}"
        )
    listed = [len(select[name]) for name in array_dims]
    if type_ == "line_chart" and listed and listed[0] > LINE_CHART_MAX_VALUES:
        return harness_refusal(
            code=_SERIES_LIMIT_EXCEEDED,
            message=(
                f"line_chart select lists {listed[0]} values on {array_dims[0]!r}, more than "
                f"the {LINE_CHART_MAX_VALUES} one line_chart draws: list at most "
                f"{LINE_CHART_MAX_VALUES}, or show them all as a bar_chart or a table"
            ),
        )
    return None


def _select_invalid(message: str) -> dict[str, Any]:
    """The harness refusal for a select the platform would refuse for its shape."""
    return harness_refusal(code=_SELECT_INVALID, message=message)


def _stat_card_binding_reason(
    *, select: dict[str, Any], view: dict[str, Any], array_dims: list[str]
) -> str | None:
    """Why a `stat_card` select resolves to more than one series: an array
    value, or a non-facet plural dimension left unpinned. `None` when it
    names one."""
    if array_dims:
        return f"stat_card select carries array values for {array_dims}"
    unpinned = sorted(_plural_dimension_names(view) - set(select))
    if unpinned:
        return f"stat_card select leaves plural dimension(s) {unpinned} unpinned"
    return None


def _declared_dimensions(view: dict[str, Any]) -> dict[str, int]:
    """Each dimension the view lists, with its `total_count`. An entry with
    no `total_count`, or a bare dimension name (older scripted fixtures),
    counts as 1, so a view missing the figure never refuses."""
    declared: dict[str, int] = {}
    for dim in view.get("dimensions") or []:
        if isinstance(dim, str) and dim:
            declared[dim] = 1
        elif isinstance(dim, dict) and dim.get("name"):
            count = dim.get("total_count")
            declared[str(dim["name"])] = count if isinstance(count, int) else 1
    return declared


def _series_estimate(*, select: dict[str, Any], declared: dict[str, int]) -> int:
    """dfo-apis's own upper bound on the series `select` resolves to
    (`_check_series_estimate` in dfo-data-tools' `assembler/resolve.py`):
    the product, over every declared dimension, of the bound array's
    length, 1 for a single bound value or an unbound facet column, and
    `max(total_count, 1)` for any other dimension `select` leaves unbound.
    `_select_binding_ok` has already refused a select binding an undeclared
    dimension, so the dimensions `select` binds count only when the view
    declares none."""
    estimate = 1
    for name in declared.keys() | select.keys():
        if name in select:
            estimate *= len(select[name]) if isinstance(select[name], list) else 1
        elif name not in _FACET_DIMENSIONS:
            estimate *= max(declared[name], 1)
    return estimate


def _admissible_intents(
    intents: list[dict[str, Any]],
    *,
    frame_refs: list[dict[str, Any]],
    frame_views: list[dict[str, Any]],
    frame_provenance: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any]]]]:
    """The deterministic gate every compose seam's output passes through
    before a single `write_component` call is made -- a model instruction is
    a request, not a guarantee, and this is enforced in code instead. An
    intent survives only if its `frame_ref` names one of this turn's actual
    produced frames (never a hallucinated one), that frame's view was
    available (a `describe_frame` refusal never gets papered over by
    presenting a different frame in its place), that frame belongs to
    the plan's final step (an intermediate step's frame is that step's raw
    input, never the question's answer -- see `ComposeModel`), and its
    select binds to a shape the platform will accept (`_select_binding_ok`).
    A step that produced more than one frame keeps every one of them
    eligible, since they are all equally final; only a *different, earlier*
    step is excluded, so a legitimate one-card-per-frame turn from a single
    final step is never wrongly refused. A rejection is logged rather than
    dropped silently, since a silent drop here is the exact failure mode
    this guard exists to end.

    Returns the intents to write, and each intent `_select_binding_ok`
    refused beside its refusal, which `compose` records in `refused_writes`
    the way it records a refused `write_component`."""
    admissible: list[dict[str, Any]] = []
    refused: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for intent in intents:
        frame_ref = intent.get("frame_ref")
        index = next(
            (i for i, ref in enumerate(frame_refs) if ref == frame_ref),
            None,
        )
        view = frame_views[index] if index is not None else None
        provenance = frame_provenance[index] if index is not None else None
        if (
            index is None
            or view is None
            or not view.get("available")
            or provenance is None
            or not provenance.get("is_final_step")
        ):
            logger.warning(
                "compose seam rejected an intent before write_component: "
                "type=%r title=%r frame_index=%s available=%s is_final_step=%s",
                intent.get("type"),
                intent.get("title"),
                index,
                view.get("available") if view else None,
                provenance.get("is_final_step") if provenance else None,
            )
            continue
        refusal = _select_binding_ok(
            type_=str(intent.get("type")),
            select=intent.get("select") or {},
            view=view.get("view") or {},
        )
        if refusal is not None:
            refused.append((intent, refusal))
            continue
        admissible.append(intent)
    return admissible, refused


def _refused_write(
    intent: Mapping[str, Any], refusal: Mapping[str, Any], *, summary: str
) -> dict[str, Any]:
    """Log `refusal` under `summary`, record it on the trace, and return the
    `refused_writes` record for `intent`: the intent's own `type` and
    `title`, and `_refused_write_reason`'s closed-vocabulary `reason`, the
    only part of the refusal copied in."""
    reason = _refused_write_reason(refusal)
    _log_refusal(summary, refusal, type=intent.get("type"), title=intent.get("title"))
    frame_ref = intent.get("frame_ref") or {}
    _record_refusal_event(
        "compose.write_refused",
        refusal,
        reason=reason,
        component_type=intent.get("type"),
        title=intent.get("title"),
        spec_id=frame_ref.get("spec_id"),
        content_version=frame_ref.get("content_version"),
    )
    return {"type": intent.get("type"), "title": intent.get("title"), "reason": reason}


def _question_text(node_input: Any) -> str:
    """Pull text out of the user Content a START successor receives."""
    if isinstance(node_input, str):
        return node_input
    parts = getattr(node_input, "parts", None)
    if parts:
        return " ".join(p.text for p in parts if getattr(p, "text", None))
    if isinstance(node_input, dict):
        return str(node_input.get("question", ""))
    return ""


def _replan_refusal(outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused) -> dict[str, Any]:
    """The refusal payload the planner node reads on a "replan" route:
    each outcome kind's own identity plus its diagnosis, never a numeric step
    index -- the compiled workflow steps are named, not ordinal. `also_stopped`
    names the same wave's other stops (`outcome.also_stopped`), each in this
    same per-kind shape, empty when the wave surfaced only this one."""
    return {
        **_refusal_shape(outcome),
        "also_stopped": [_refusal_shape(sibling) for sibling in outcome.also_stopped],
    }


def _is_dead_end(outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused) -> bool:
    """Whether `outcome` is a tool's refusal `classify_refusal` calls `Disposition.STOP`."""
    return (
        isinstance(outcome, Refused)
        and outcome.payload is not None
        and classify_refusal(outcome.payload) is Disposition.STOP
    )


def _settle_depth(ctx: Context, envelope: Mapping[str, Any], *, surface: str) -> DepthSignal:
    """The depth verdict for a turn that resolved to `envelope`, written to the state's
    `depth`, `depth_reason` and `depth_notice`.

    `choose_depth` settles it when the turn resolves, so the three keys
    describe the envelope the state holds. `surface` is the workflow's own,
    which the turn's `surface` state overrides.
    """
    signal = decide_depth(
        status=str(envelope.get("status", "")),
        choice_points=int(envelope.get("choice_points", 0)),
        degraded=any(w.get("severity") == "degraded" for w in envelope.get("warnings", [])),
        part_count=int(envelope.get("parts", 1)),
        # Per-turn surface wins over the construction default: a
        # download-surface turn earns no dialogue.
        surface=ctx.state.get("surface") or surface,
        depth_hint=ctx.state.get("depth_hint"),
        # `graph_envelope`'s own structural reading of the resolved
        # identity's width -- absent on every non-`definite` route.
        wide=bool(envelope.get("wide", False)),
        # A seeded turn's operations (`seed`); absent on a searched one.
        operations=sum(len(item["operations"]) for item in envelope.get("seeds") or []),
    )
    ctx.state["depth"] = signal.depth
    ctx.state["depth_reason"] = signal.reason
    ctx.state["depth_notice"] = signal.notice
    return signal


def _record_drops(
    ctx: Context, sources: Sequence[SeededSource], drops: Sequence[DroppedSeed]
) -> None:
    """Record `sources` as datasets the turn dropped, and `drops` as drops the user is yet to be told of.

    `DROPPED_SEEDS_STATE_KEY` makes `propose_plan` refuse a plan that pins
    one, `discover_source` take no source from a search that finds it again,
    and `probe` take it out of the question's own search
    (`planner.plan.names_dropped_seed`). `GraphState.dropped_untold` holds
    the drops until the turn's next words to the user name them: a question
    `clarify` or the planner asks, the `planner` node's plan card note or
    closing text, or else the `narrate` or `deadend` closing
    (`_untold_drops`).
    """
    save_dropped_seeds(ctx.state, [*dropped_seeds(ctx.state), *sources])
    untold = _untold_drops(ctx)
    ctx.state["dropped_untold"] = json.dumps([*untold, *drops])


def _untold_drops(ctx: Context) -> list[DroppedSeed]:
    """The drops `_record_drops` recorded that the user has not been told of yet."""
    return _state_json(ctx, "dropped_untold", default=[])


def _with_untold_drops(ctx: Context, text: str) -> str:
    """`text` after the drops the user has not been told of (`_untold_drops`), which are told now."""
    drops = _untold_drops(ctx)
    if not drops:
        return text
    ctx.state["dropped_untold"] = None
    return " ".join(part for part in (dropped_seeds_text(drops), text) if part)


def _without_dropped_datasets(ctx: Context, envelope: dict[str, Any]) -> dict[str, Any]:
    """`envelope` with no reading of a dataset the turn dropped (`planner.plan.names_dropped_seed`).

    `probe` reads every search answer into the turn through this, a clarify
    re-entry's included, so neither `fetch` nor a plan drafted from the
    envelope retrieves a dropped dataset and `clarify` never offers one. A
    `definite` reading of one, by its spec_id or by the identity its
    `spec_block` carries, becomes `unsupported`, and so does a candidate list
    that offers nothing else; `deadend` then closes with the drops and
    `text`. A candidate list otherwise loses each candidate whose spec_id
    names one: a candidate carries no identity.
    """
    if not dropped_seeds(ctx.state):
        return envelope
    left_out = {
        "code": SEED_DROPPED,
        "severity": "warn",
        "message": "Search resolves the question to a dataset this turn left out",
    }
    if envelope.get("status") == "definite":
        block = envelope.get("spec_block") or {}
        identity = (block.get("definition") or {}).get("identity")
        if not names_dropped_seed(ctx.state, spec_id=envelope.get("spec_id"), identity=identity):
            return envelope
        candidates = []
    else:
        offered = envelope.get("candidates") or []
        candidates = [
            candidate
            for candidate in offered
            if not names_dropped_seed(ctx.state, spec_id=candidate.get("spec_id"))
        ]
        if len(candidates) == len(offered):
            return envelope
    warnings = [*(envelope.get("warnings") or []), left_out]
    if candidates:
        return {**envelope, "candidates": candidates, "warnings": warnings}
    return {
        "status": "unsupported",
        "spec_id": None,
        "choice_points": 0,
        "parts": envelope.get("parts", 1),
        "warnings": warnings,
        "candidates": [],
        "text": "The question resolves to a dataset this turn left out, so nothing was retrieved.",
    }


# What a `Resolver` raises when a search gives no answer: `ResolverError` for
# one the tool refused or answered without an envelope, and the MCP client's
# own errors for a call that ran past its deadline or lost its transport.
_SEARCH_FAILURES = (ResolverError, MCPError, httpx2.TransportError)


def _refused_for(
    envelope: Mapping[str, Any], parts: Sequence[Mapping[str, Any]], reason: str
) -> list[int]:
    """The positions of the search parts `envelope` answered `no_data` with
    `reason`, the `no_data.reason` code a part digest carries. None when the
    digests do not line up with `parts` one to one, since a position then
    names no part."""
    digests = envelope.get("part_digests") or []
    if len(digests) != len(parts):
        return []
    return [index for index, digest in enumerate(digests) if digest.get("reason") == reason]


def _unit_in_variable(part: Mapping[str, Any]) -> dict[str, Any]:
    """`part` with its unit phrase moved to the end of its variable text."""
    folded = {key: value for key, value in part.items() if key != "unit"}
    folded["variable"] = f"{part.get('variable')} {part['unit']}"
    return folded


def _measure_and_denominator(
    part: Mapping[str, Any], draft: Mapping[str, Any]
) -> list[dict[str, Any]] | None:
    """The two search parts whose division gives `part`, from a
    `DenominatorSplitModel` draft.

    The draft supplies each side's variable text and unit, which of
    `part`'s entities the denominator is about, by position, and whether
    they keep their roles. Everything else is `part`'s: the measure takes
    its entities, roles and time unchanged, and the denominator the
    entities at the drafted positions, in `part`'s order, and the same
    time, with each role kept or left out as the draft says.

    `None` when the draft names no variable for a side, or when `part`
    has entities and the draft's positions are none, repeated, or outside
    the list."""
    entities = list(part.get("entities") or [])
    drafted = draft.get("denominator_entities")
    chosen: list[int] = drafted if isinstance(drafted, list) else []
    if entities and not (
        chosen
        and all(type(at) is int and 0 <= at < len(entities) for at in chosen)
        and len(set(chosen)) == len(chosen)
    ):
        return None
    sides: list[dict[str, Any]] = []
    for side in ("measure", "denominator"):
        variable = draft.get(side)
        if not isinstance(variable, str) or not variable.strip():
            return None
        built: dict[str, Any] = {"variable": variable.strip()}
        if entities and side == "measure":
            built["entities"] = [dict(entity) for entity in entities]
        elif entities:
            keeps_roles = draft.get("denominator_keeps_roles") is True
            built["entities"] = [
                {key: value for key, value in entities[at].items() if keeps_roles or key != "role"}
                for at in sorted(chosen)
            ]
        if time := part.get("time"):
            built["time"] = time
        unit = draft.get(f"{side}_unit")
        if isinstance(unit, str) and unit.strip():
            built["unit"] = unit.strip()
        sides.append(built)
    return sides


def _split_search(
    parts: Sequence[dict[str, Any]], drafts: Mapping[int, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[int]] | None:
    """The search parts that put a measure and its denominator in place of
    each part `drafts` has a `DenominatorSplitModel` draft for, by position
    in `parts`, and where that search holds the parts built, in its order.

    Each pair stands where its part stood (`_measure_and_denominator`). A
    built part equal to one already in the search, or to a part of `parts`
    no draft replaces, is not added, so parts that share a denominator hold
    it at one position.

    `None`, with the reason logged, when a draft builds no pair or the
    search would hold more than `MAX_PARTS` parts, the most `search` takes."""
    untouched = [part for index, part in enumerate(parts) if index not in drafts]
    search_parts: list[dict[str, Any]] = []
    built: list[dict[str, Any]] = []
    for index, part in enumerate(parts):
        if index not in drafts:
            search_parts.append(part)
            continue
        pair = _measure_and_denominator(part, drafts[index])
        if pair is None:
            logger.info(
                "a denominator split names no variable for a side or no usable "
                "entity positions, so the refusal stands: part=%s draft=%s",
                json.dumps(part),
                json.dumps(drafts[index]),
            )
            return None
        built.extend(pair)
        search_parts.extend(
            side for side in pair if side not in search_parts and side not in untouched
        )
    if len(search_parts) > MAX_PARTS:
        logger.info(
            "a denominator split makes %d parts and search takes %d, so the "
            "refusal stands: parts=%s",
            len(search_parts),
            MAX_PARTS,
            json.dumps(search_parts),
        )
        return None
    return search_parts, sorted({search_parts.index(side) for side in built})


def _resolves(digest: Mapping[str, Any]) -> bool:
    """Whether a part digest names a dataset or offers a choice of them."""
    return digest.get("status") == "definite" or bool(digest.get("candidates"))


def _refusal_shape(outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused) -> dict[str, Any]:
    """One outcome's own identity and diagnosis, with no `also_stopped` of its own."""
    if isinstance(outcome, MethodChoiceNeeded):
        return {
            "kind": outcome.kind,
            "step_id": outcome.step_id,
            "tool": outcome.tool.name,
            "code": outcome.code,
            "candidates": outcome.candidates,
            "refusal": outcome.refusal.model_dump(mode="json"),
        }
    if isinstance(outcome, RediscoveryNeeded):
        return {
            "kind": outcome.kind,
            "step_id": outcome.step_id,
            "tool": outcome.tool.name,
            "requirement_id": outcome.requirement_id,
            "spec_id": outcome.spec_id,
            # Whatever the refusal itself offered, verbatim -- same helper
            # `MethodChoiceNeeded.candidates` is minted from at
            # `executor/run.py`. A rediscovery with nothing attached sends
            # the planner back to search with nothing to choose from.
            "candidates": candidates_of(outcome.refusal),
            "refusal": outcome.refusal.model_dump(mode="json"),
        }
    return {
        "kind": outcome.kind,
        "refusal": outcome.refusal.value,
        "detail": outcome.detail,
        "step_id": outcome.step_id,
        "requirement_id": outcome.requirement_id,
        "issues": [issue.model_dump(mode="json") for issue in outcome.issues],
    }


def _upstream_retrievals(
    step_id: str,
    *,
    steps_by_id: dict[str, WorkflowStep],
    retrieval_step_ids: frozenset[str],
    memo: dict[str, frozenset[str]],
) -> frozenset[str]:
    """The retrieval step ids that transitively feed `step_id`, via `WorkflowStep.bindings`.

    A retrieval step (one `Workflow.evidence_bindings` names) feeds itself;
    any other step's set is the union of what feeds each step its own
    bindings point `from_step` at -- so a two-hop chain (retrieve ->
    aggregate -> filter) still resolves to the one retrieval underneath it,
    and a join resolves to every retrieval either side rests on. Memoized
    per call: a DAG's shared ancestor is walked once per descendant it
    feeds, not once per edge on the path to it.
    """
    if step_id in memo:
        return memo[step_id]
    if step_id in retrieval_step_ids:
        result = frozenset({step_id})
    else:
        step = steps_by_id.get(step_id)
        upstream: set[str] = set()
        for binding in step.bindings.values() if step is not None else ():
            upstream |= _upstream_retrievals(
                binding.from_step,
                steps_by_id=steps_by_id,
                retrieval_step_ids=retrieval_step_ids,
                memo=memo,
            )
        result = frozenset(upstream)
    memo[step_id] = result
    return result


def _carried_frame_parameters(step: WorkflowStep) -> list[str]:
    """The arguments naming the frames whose StatVars `step`'s frame carries:
    `FRAME_PARAMETER`, and the second frame's for a verb that carries both
    sides (`harness.provenance.BOTH_SIDES_VERBS`)."""
    names = [FRAME_PARAMETER]
    if (verb := BOTH_SIDES_VERBS.get(step.tool.name)) is not None:
        names.append(verb.second_frame)
    return names


def _writes_own_stat_var(step: WorkflowStep) -> bool:
    """Whether `step`'s verb writes its result under a StatVar of its own."""
    verb = BOTH_SIDES_VERBS.get(step.tool.name)
    return verb is not None and verb.own_stat_var is not None


def _carried_retrievals(
    step_id: str,
    *,
    steps_by_id: dict[str, WorkflowStep],
    retrieval_step_ids: frozenset[str],
) -> frozenset[str]:
    """The retrieval step ids whose StatVars `step_id`'s frame carries.

    A verb writes the StatVars of the frame bound to `FRAME_PARAMETER`, and
    `join` also those of its right side. `sum_of`, `quotient_of` and
    `share_of` each write a StatVar of their own that stands for both sides'
    StatVars, so they carry both too (`_carried_frame_parameters`).
    `compare`'s right side supplies values only, so a frame resting on it
    carries none of its StatVars. A subset of `_upstream_retrievals`.
    """
    if step_id in retrieval_step_ids:
        return frozenset({step_id})
    step = steps_by_id.get(step_id)
    if step is None:
        return frozenset()
    carried: set[str] = set()
    for name in _carried_frame_parameters(step):
        if (binding := step.bindings.get(name)) is not None:
            carried |= _carried_retrievals(
                binding.from_step, steps_by_id=steps_by_id, retrieval_step_ids=retrieval_step_ids
            )
    return frozenset(carried)


def _carried_parts(
    step_id: str,
    *,
    steps_by_id: dict[str, WorkflowStep],
    retrieval_step_ids: frozenset[str],
) -> tuple[frozenset[str], frozenset[str]]:
    """The retrieval steps, and the steps writing a StatVar of their own,
    whose StatVars `step_id`'s frame carries.

    The walk `_carried_retrievals` takes, stopping at each `sum_of`,
    `quotient_of` or `share_of` (`_writes_own_stat_var`). The frame carries
    that step's one StatVar, which stands for every StatVar under its sides,
    in place of the retrievals beneath it.
    """
    if step_id in retrieval_step_ids:
        return frozenset({step_id}), frozenset()
    step = steps_by_id.get(step_id)
    if step is None:
        return frozenset(), frozenset()
    if _writes_own_stat_var(step):
        return frozenset(), frozenset({step_id})
    retrievals: set[str] = set()
    writers: set[str] = set()
    for name in _carried_frame_parameters(step):
        if (binding := step.bindings.get(name)) is not None:
            found, written = _carried_parts(
                binding.from_step, steps_by_id=steps_by_id, retrieval_step_ids=retrieval_step_ids
            )
            retrievals |= found
            writers |= written
    return frozenset(retrievals), frozenset(writers)


def _ran(outcome: RunOutcome) -> tuple[StepRecord, ...]:
    """Every step of `outcome`'s run that finished, in the plan's own step order."""
    return outcome.steps if isinstance(outcome, Accepted) else outcome.completed


def _answer_frames(outcome: RunOutcome) -> dict[str, FrameRef]:
    """Each terminal frame that stands as an answer, by step id, in the
    workflow's terminal order: every terminal step's on an accepted run
    (`Accepted.results`), and on a stopped run each finished terminal step's
    that passed the same answer check (`answers`, `executor.run._Run.
    _answer_refusal`)."""
    return dict(outcome.results if isinstance(outcome, Accepted) else outcome.answers)


def _link_evidence_to_terminal_frames(
    tool_surface: ToolSurface | None, outcome: RunOutcome, *, workflow: PlanWorkflow
) -> None:
    """Link each terminal frame's provenance to every retrieval that feeds it.

    A terminal step is either a bare `get_data` -- a plan answering a
    question that asks to see a series names the retrieved frame as its own
    terminal -- or an analysis-mcp transform (`aggregate`, `filter`, `join`,
    ...) chained onto one or more `get_data` steps. A transform mints its own
    `ops_` frame identity, distinct from the retrieval's `spec_` one that
    `harness.provenance.ProvenanceStore` was written under
    (`harness.studio_translate._RecordingSurface._remember_manifest`).
    Without this, `McpExecutor._assemble_component` would never find a
    terminal frame's provenance on the approved-plan path.

    A stopped run links the terminal frames it finished (`_answer_frames`)
    the same way, from the evidence its finished retrievals recorded, so a
    partial answer's cards keep their sources.

    Attribution is per frame: `_upstream_retrievals` walks each terminal
    step's own `bindings` back to the retrieval(s) under it, so two
    independent branches in the same run (a debt-service step and a
    population step, each aggregated on its own) each link to their own
    source, and a frame fed by several retrievals (a join, a compare, a
    share of two datasets) links to all of them, in `workflow.steps` order.
    The frame's lineage names exactly the sources its card rests on. The
    link also carries the distinct StatVars the terminal step declared its
    frame carries (`StepRecord.stat_var_count`), which the card counts its
    indicators from, and the retrievals and sums whose StatVars the frame
    carries (`_carried_parts`), each sum with the retrievals under it, so a
    summed family's ref counts as the StatVars it sums and a sum's ref as the
    distinct StatVars under its sides, and a quotient's ref the same way.
    Each `sum_of`, `quotient_of` or `share_of` step's own StatVar ref goes
    with every link, beside its tool name and the parts of each of its sides
    (`_own_stat_vars`), so the card can name that ref by what it adds or
    divides wherever it is drawn.

    All-or-nothing: a frame with no retrieval under it (a workflow defect,
    never a valid one), or one whose retrieval has no `EvidenceRecord`, is
    not linked and a warning names it. `McpExecutor._assemble_component`
    withholds the card for a frame with no lineage rather than state a
    provenance narrower than the frame has. `link_evidence` is looked up via
    `getattr`, the same duck-typed opt-in `_RecordingExecutor.aclose` uses.
    """
    link = getattr(tool_surface, "link_evidence", None)
    if link is None:
        return
    retrieval_step_ids = frozenset(binding.step_id for binding in workflow.evidence_bindings)
    steps_by_id = {step.id: step for step in workflow.steps}
    step_order = {step.id: index for index, step in enumerate(workflow.steps)}
    evidence_by_step = {record.step_id: record for record in outcome.evidence}
    stat_var_counts = {record.step_id: record.stat_var_count for record in _ran(outcome)}
    own_stat_vars = _own_stat_vars(
        outcome,
        steps_by_id=steps_by_id,
        retrieval_step_ids=retrieval_step_ids,
        step_order=step_order,
    )
    memo: dict[str, frozenset[str]] = {}
    for step_id, frame_ref in _answer_frames(outcome).items():
        sources = sorted(
            _upstream_retrievals(
                step_id,
                steps_by_id=steps_by_id,
                retrieval_step_ids=retrieval_step_ids,
                memo=memo,
            ),
            key=lambda source: step_order[source],
        )
        unrecorded = [source for source in sources if source not in evidence_by_step]
        if unrecorded or not sources:
            logger.warning(
                "provenance: frame for step %r not linked; retrieval step(s) without an "
                "evidence record: %s",
                step_id,
                unrecorded or "no retrieval feeds it",
            )
            continue
        carried, writers = _carried_parts(
            step_id, steps_by_id=steps_by_id, retrieval_step_ids=retrieval_step_ids
        )
        link(
            frame_ref=frame_ref.model_dump(),
            evidence_frame_refs=[
                evidence_by_step[source].frame_ref.model_dump() for source in sources
            ],
            stat_var_count=stat_var_counts.get(step_id),
            carried_frame_refs=[
                evidence_by_step[source].frame_ref.model_dump()
                for source in sources
                if source in carried
            ],
            summed_frame_refs=[
                [
                    evidence_by_step[source].frame_ref.model_dump()
                    for source in sources
                    if source in under
                ]
                for under in (
                    _carried_retrievals(
                        writer, steps_by_id=steps_by_id, retrieval_step_ids=retrieval_step_ids
                    )
                    for writer in sorted(writers, key=step_order.__getitem__)
                )
            ],
            own_stat_vars=own_stat_vars,
        )


def _own_stat_vars(
    outcome: RunOutcome,
    *,
    steps_by_id: dict[str, WorkflowStep],
    retrieval_step_ids: frozenset[str],
    step_order: dict[str, int],
) -> dict[str, tuple[str, list[dict[str, Any] | str], list[dict[str, Any] | str]]]:
    """Each StatVar ref a step wrote for its own result (`_writes_own_stat_var`),
    with the step's tool name and the parts of its first and second side.

    analysis-mcp writes a sum's, a quotient's or a share's rows under
    `dcid:<its spec_id>`, a ref no graph node describes. A side's parts are
    what its frame carries (`_carried_parts`), in workflow order: a
    retrieval as its frame ref, and an earlier sum, quotient or share as the
    ref it wrote, which the card names by that verb's own sides. A share of
    two quotients therefore lists one quotient's ref on each side. A step
    whose second side reads its first frame again names no second frame, so
    both sides list the same parts.
    """
    evidence_by_step = {record.step_id: record for record in outcome.evidence}
    written_by_step = {
        record.step_id: f"dcid:{record.frame_ref.spec_id}"
        for record in _ran(outcome)
        if (step := steps_by_id.get(record.step_id)) is not None and _writes_own_stat_var(step)
    }

    def side_parts(from_step: str) -> list[dict[str, Any] | str]:
        retrievals, writers = _carried_parts(
            from_step, steps_by_id=steps_by_id, retrieval_step_ids=retrieval_step_ids
        )
        parts: list[dict[str, Any] | str] = []
        for part in sorted(retrievals | writers, key=step_order.__getitem__):
            if part in evidence_by_step:
                parts.append(evidence_by_step[part].frame_ref.model_dump())
            elif part in written_by_step:
                parts.append(written_by_step[part])
        return parts

    written: dict[str, tuple[str, list[dict[str, Any] | str], list[dict[str, Any] | str]]] = {}
    for step_id, ref in written_by_step.items():
        step = steps_by_id[step_id]
        left = step.bindings[FRAME_PARAMETER]
        right = step.bindings.get(BOTH_SIDES_VERBS[step.tool.name].second_frame, left)
        written[ref] = (step.tool.name, side_parts(left.from_step), side_parts(right.from_step))
    return written


# The flow axis a recipient's role column sits on (`get_data`'s coverage
# dimensions).
_RECIPIENT_AXIS = "destination"


def _stat_var_totals(
    tool_surface: ToolSurface | None,
    outcome: RunOutcome,
    *,
    plan: PlannerPlan,
    terminals: Collection[str] | None = None,
) -> list[str]:
    """One sentence per summed family an answer frame carries, for `narrate`,
    over every answer frame (`_answer_frames`) or only those of `terminals`.

    Walks the retrievals whose StatVars each answer frame carries
    (`_carried_retrievals`, the walk the card counts its indicators by), so
    a `compare`'s right side gets no sentence. A retrieval is summed when
    its recorded `get_data` result (`_RecordingSurface.retrieval_manifest`,
    looked up via `getattr` like `link_evidence`) names the StatVars dfo-apis added into the family total
    (`summed_stat_var_count`). The sentence names the family, states that
    count and who set the reading (`_reading_set_by`), and says when the
    figures add up recipients the question did not name
    (`_totals_across_recipients`). Identical sentences are stated once.
    """
    manifest_of = getattr(tool_surface, "retrieval_manifest", None)
    workflow = plan.workflow
    if manifest_of is None or workflow is None:
        return []
    retrieval_step_ids = frozenset(binding.step_id for binding in workflow.evidence_bindings)
    steps_by_id = {step.id: step for step in workflow.steps}
    step_order = {step.id: index for index, step in enumerate(workflow.steps)}
    requirement_ids = {b.step_id: b.requirement_id for b in workflow.evidence_bindings}
    requirements = {requirement.id: requirement for requirement in plan.evidence_requirements}
    decisions = {decision.slot: decision for decision in plan.decisions}
    evidence_by_step = {record.step_id: record for record in outcome.evidence}
    role_keys_by_step = {record.step_id: record.role_keys for record in _ran(outcome)}
    sentences: list[str] = []
    for terminal_id in _answer_frames(outcome):
        if terminals is not None and terminal_id not in terminals:
            continue
        carried = _carried_retrievals(
            terminal_id, steps_by_id=steps_by_id, retrieval_step_ids=retrieval_step_ids
        )
        for retrieval_id in sorted(carried, key=step_order.__getitem__):
            record = evidence_by_step.get(retrieval_id)
            manifest = manifest_of(frame_ref=record.frame_ref.model_dump()) if record else None
            count = summed_stat_var_count(manifest) if manifest is not None else None
            if manifest is None or count is None:
                continue
            source = workflow.source_of(retrieval_id)
            sentence = _total_sentence(
                _family_name(manifest, source),
                count,
                set_by=_reading_set_by(
                    source, requirements.get(requirement_ids[retrieval_id]), decisions
                ),
                across_recipients=_totals_across_recipients(
                    manifest, source, answer_role_keys=role_keys_by_step.get(terminal_id)
                ),
            )
            if sentence not in sentences:
                sentences.append(sentence)
    return sentences


def _answer_notes(
    tool_surface: ToolSurface | None,
    outcome: RunOutcome,
    *,
    plan: PlannerPlan,
    terminals: Collection[str] | None = None,
) -> list[str]:
    """The sentences `narrate` states whatever its draft says, over every
    answer frame (`_answer_frames`) or only those of `terminals`: one per
    summed family (`_stat_var_totals`), then one per step that left keys out
    (`_dropped_key_sentences`)."""
    return [
        *_stat_var_totals(tool_surface, outcome, plan=plan, terminals=terminals),
        *_dropped_key_sentences(outcome, plan=plan, terminals=terminals),
    ]


def _dropped_key_sentences(
    outcome: RunOutcome,
    *,
    plan: PlannerPlan,
    terminals: Collection[str] | None = None,
) -> list[str]:
    """One sentence per step under an answer frame that left keys out of its
    result (`StepRecord.dropped_key_count`), over every answer frame
    (`_answer_frames`) or only those of `terminals`.

    A keyed `quotient_of` drops each numerator key it has nothing to divide
    by, and every step reading its frame, a `rank` among them, works over
    the keys that remain. So the walk covers every step an answer frame
    rests on (`_upstream_steps`), in the plan's step order. Identical
    sentences are stated once.
    """
    workflow = plan.workflow
    if workflow is None:
        return []
    dropped = {
        record.step_id: record.dropped_key_count
        for record in _ran(outcome)
        if record.dropped_key_count
    }
    if not dropped:
        return []
    steps_by_id = {step.id: step for step in workflow.steps}
    sentences: list[str] = []
    for terminal_id in _answer_frames(outcome):
        if terminals is not None and terminal_id not in terminals:
            continue
        under = _upstream_steps(terminal_id, steps_by_id=steps_by_id)
        for step_id, count in dropped.items():
            if step_id not in under:
                continue
            sentence = _dropped_key_sentence(count, *_dropped_key_nouns(step_id, plan))
            if sentence not in sentences:
                sentences.append(sentence)
    return sentences


def _upstream_steps(step_id: str, *, steps_by_id: dict[str, WorkflowStep]) -> frozenset[str]:
    """`step_id` and every step that transitively feeds it, via `WorkflowStep.bindings`."""
    found: set[str] = set()
    pending = [step_id]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.add(current)
        if (step := steps_by_id.get(current)) is not None:
            pending.extend(binding.from_step for binding in step.bindings.values())
    return frozenset(found)


def _dropped_key_nouns(step_id: str, plan: PlannerPlan) -> tuple[str | None, str | None]:
    """The plan's own words for the keys step `step_id` divides and for what
    it divides them by: the quotient requirement's `axis`, and the measure
    of the dataset requirement under its denominator. `None` for either the
    plan does not state.

    A step carries out the operation it is named for (`WorkflowStep.id`),
    and that operation names its requirement.
    """
    operations = plan.translation.operations if plan.translation is not None else ()
    requirement_id = next(
        (operation.requirement_id for operation in operations if operation.id == step_id), None
    )
    requirements = {requirement.id: requirement for requirement in plan.evidence_requirements}
    requirement = requirements.get(requirement_id or "")
    if not isinstance(requirement, QuotientRequirement):
        return None, None
    divisor = requirements.get(requirement.denominator.input_id)
    # A denominator read through a filter or an aggregate keeps its
    # dataset's measure.
    while divisor is not None and not isinstance(divisor, DatasetRequirement):
        divisor = requirements.get(getattr(divisor, "input_id", None) or "")
    return requirement.axis, divisor.measure if divisor is not None else None


def _dropped_key_sentence(count: int, axis: str | None, divisor: str | None) -> str:
    """The sentence for `count` keys a quotient left out: how many, of which
    role when the plan names one (`axis`), and that each had nothing to
    divide by, named by its measure when the plan states one (`divisor`).

    A key is left out when the denominator holds no entry for it, none in a
    period the key reports, or only zeros where it does, and the one wording
    is true of all three.

    The role phrase is the planner's own word and is not inflected: it
    qualifies "entry" or "entries", which carries the number.
    """
    noun = "entry" if count == 1 else "entries"
    subject = f"{count:,} {axis} {noun}" if axis else f"{count:,} {noun}"
    lacked = f"no {divisor} figure to divide by" if divisor else "no figure to divide by"
    verb = "was" if count == 1 else "were"
    return f"{subject} had {lacked} and {verb} left out."


def _family_name(manifest: Mapping[str, Any], source: SelectedSource | None) -> str:
    """The name narration gives a summed family: the label dfo-apis gave the
    family's ref (`summed_stat_var_label`), else the name of its source."""
    label = manifest.get("summed_stat_var_label")
    if isinstance(label, str) and label:
        return label
    if source is not None and source.provenance_label:
        return source.provenance_label
    return parse_manifest(manifest).citation.text


def _totals_across_recipients(
    manifest: Mapping[str, Any],
    source: SelectedSource | None,
    *,
    answer_role_keys: tuple[str, ...] | None,
) -> bool:
    """Whether the answer adds up recipients of a summed retrieval the question did not name.

    True when the retrieval's frame carries a role column on the recipient
    axis (its declared `role_keys`, each placed on its axis by the
    manifest's coverage dimensions), the answer frame's declared role keys
    omit that column, and the source resolves no recipient, so
    the frame held every recipient the source reports. False when either
    frame's role keys are unknown.
    """
    _rows, role_keys, _stat_var_count = declared_shape(manifest)
    if role_keys is None or answer_role_keys is None:
        return False
    if source is not None and any(entity.role == _RECIPIENT_AXIS for entity in source.entities):
        return False
    return any(
        dimension.axis == _RECIPIENT_AXIS
        and dimension.label in role_keys
        and dimension.label not in answer_role_keys
        for dimension in parse_manifest(manifest).coverage.dimensions
    )


def _reading_set_by(
    source: SelectedSource | None,
    requirement: EvidenceRequirement | None,
    decisions: Mapping[str, Decision],
) -> ResolvedBy | ReadingSetBy | None:
    """Who set a summed retrieval's reading, when narration names them.

    `ReadingSetBy.RULE` when the compiler applied the source's default
    (`compiler.sources.default_reading`); `ResolvedBy.USER` when the
    requirement cites a decision the user settled that fixes the reading.
    `None` when the planner read it off the question, which needs no
    mention.
    """
    if source is not None and source.stat_var_reading_set_by is ReadingSetBy.RULE:
        return ReadingSetBy.RULE
    slots = requirement.decision_ids if requirement is not None else ()
    cited = (decisions[slot] for slot in slots if slot in decisions)
    if any(
        decision.resolved_by is ResolvedBy.USER
        and decision.fixes is not None
        and decision.fixes.stat_vars is not None
        for decision in cited
    ):
        return ResolvedBy.USER
    return None


def _total_sentence(
    name: str, count: int, *, set_by: ResolvedBy | ReadingSetBy | None, across_recipients: bool
) -> str:
    """The sentence `narrate` states for one summed family.

    `count` is the StatVars dfo-apis summed, named indicators as the card's
    attribution names them. It describes the family, so it stays true of
    figures a later filter or verb drew from the sum.
    """
    sentence = f'the figures for "{name}" sum the family\'s {count:,} indicators.'
    if set_by is ReadingSetBy.RULE:
        sentence = f"By default, {sentence}"
    elif set_by is ResolvedBy.USER:
        sentence = f"As you asked, {sentence}"
    else:
        sentence = sentence[0].upper() + sentence[1:]
    if across_recipients:
        sentence += (
            " They include regional and unallocated flows as well as flows to individual countries."
        )
    return sentence


@dataclass(frozen=True)
class _StepReport:
    """One progress report, tagged with the step it describes."""

    step_id: str
    report: ProgressReport


class _ReportRelay:
    """Holds the latest progress report per step, for one reader to drain.

    `put` replaces any report still pending for the same step rather than
    queueing another one, so a producer faster than the reader grows this
    at most one entry per distinct step, never per report -- only the
    latest report says where a step is now. `latest` wakes whenever there
    is something pending, yields it as one batch, and returns once
    `finish` has been called and nothing is left to deliver.
    """

    def __init__(self) -> None:
        self._pending: dict[str, _StepReport] = {}
        self._ready = asyncio.Event()
        self._finished = False

    def put(self, step_id: str, progress: ProgressReport) -> None:
        self._pending[step_id] = _StepReport(step_id=step_id, report=progress)
        self._ready.set()

    def finish(self) -> None:
        self._finished = True
        self._ready.set()

    async def latest(self) -> AsyncIterator[_StepReport]:
        while True:
            await self._ready.wait()
            self._ready.clear()
            pending, self._pending = self._pending, {}
            for item in pending.values():
                yield item
            if self._finished and not self._pending:
                return


def _start_relayed[T](
    call: Callable[[StepProgressCallback], Coroutine[Any, Any, T]],
) -> tuple[asyncio.Task[T], _ReportRelay]:
    """Run `call` as its own task, relaying its progress reports.

    A node awaiting a long call directly could yield nothing until the call
    returns; run as a task, the call reports while the node relays
    (`_ReportRelay.latest`). The relay is told the task finished, however it
    finishes, so `latest` always returns. The caller awaits the task for its
    result and stops it (`_stop`) if the node exits first.
    """
    relay = _ReportRelay()
    task = asyncio.create_task(call(relay.put))
    task.add_done_callback(lambda _: relay.finish())
    return task, relay


async def _stop(task: asyncio.Task[Any]) -> None:
    """Cancel `task` if it is still running, and wait until it has stopped."""
    if not task.done():
        task.cancel()
        await asyncio.wait({task})


def _step_progress_event(report: _StepReport, *, tool: str) -> Event:
    """A relayed progress report as its own event, carried on `custom_metadata`.

    Never on `output`: a node run may set `output` once (`_node_runner.py`'s
    `ctx.output`), and the node's result needs it.
    """
    return Event(
        custom_metadata={
            STEP_PROGRESS_KEY: {
                "step_id": report.step_id,
                "tool": tool,
                "message": report.report.message,
            }
        }
    )


def _mark_masked_step_failed(
    tool_surface: ToolSurface | None,
    outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused,
    *,
    tools: Mapping[str, str],
) -> None:
    """Correct the activity record of every step a wave's stop touches --
    the primary outcome and each sibling in `outcome.also_stopped` -- when
    its raw call succeeded but the executor's own post-call validation
    (shape, evidence, frame) refused it afterward, and record one for a
    step whose budget ran out before its call returned. `tools` names each
    plan step's tool, by step id, for that record.

    `harness.studio_translate._RecordingSurface.call_tool` records a step's
    call the moment it returns, before `executor.run._Run` can run those
    checks on it, so a masked failure would otherwise read "complete" on the
    Activity feed forever, and a call cut off mid-flight would leave its
    running line open. A wave can surface more than one post-call
    refusal at once (`executor.run._Run.execute`), and each sibling in
    `also_stopped` carries its own `step_id` and gets its own record
    corrected. `mark_step_failed` is looked up via `getattr` (`tool_surface`
    is a bare `ToolSurface` in tests with no recorder around it, and a step
    whose own call already failed is a no-op there), the same duck-typed
    opt-in `_RecordingExecutor.aclose` uses. Each stop's own `_failed_step`
    supplies the `code`/`message` the Activity feed's refusal shows, naming
    the actual check that fired (e.g. `source_pin_mismatch`).
    """
    mark = getattr(tool_surface, "mark_step_failed", None)
    if mark is None:
        return
    for stopped in (outcome, *outcome.also_stopped):
        if stopped.step_id is None:
            continue
        failed = _failed_step(stopped)
        mark(
            step_id=stopped.step_id,
            tool=tools.get(stopped.step_id, ""),
            code=failed["code"],
            detail=failed["message"],
        )


def _computed_asks(plan: PlannerPlan, answers: Collection[str]) -> list[int]:
    """The number of each ask `plan` answers whose answering steps all
    finished as terminal steps (`answers`, `_answer_frames`'s keys)."""
    return [
        number
        for number, row in enumerate(plan_display.ask_rows(plan))
        if row["answered_by"]
        and all(step in answers for step in cast("list[str]", row["answered_by"]))
    ]


def _kept_asks(
    kept: Mapping[str, Any] | None,
    tool_surface: ToolSurface | None,
    outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused,
    *,
    plan: PlannerPlan,
) -> dict[str, Any]:
    """`kept` with each ask `outcome`'s run computed (`_computed_asks`) set to
    that run's answer frames for it: per answering step its id, tool, frame
    ref and digest, the notes on how those frames were computed
    (`_answer_notes`: family totals and keys left out), and for an ask
    `plan` answers in part, the part not computed with its reason, as
    `_ask_status` reports it. A plan call's asks are fixed (`plan_start`), so
    an ask's number names the same ask in every plan the call runs."""
    answers = _answer_frames(outcome)
    records = {record.step_id: record for record in outcome.completed}
    rows = plan_display.ask_rows(plan)
    kept = dict(kept or {})
    for number in _computed_asks(plan, answers):
        row = rows[number]
        steps = cast("list[str]", row["answered_by"])
        kept[str(number)] = {
            "steps": [
                {
                    "step_id": step,
                    "tool": records[step].tool.name,
                    "frame_ref": answers[step].model_dump(),
                    "digest": records[step].digest,
                }
                for step in steps
            ],
            "totals": _answer_notes(tool_surface, outcome, plan=plan, terminals=steps),
        }
        if row["declined"] is not None:
            kept[str(number)]["not_computed"] = {
                "part": row["not_computed"],
                "reason": row["reason"],
            }
    return kept


def _failed_step(outcome: MethodChoiceNeeded | RediscoveryNeeded | Refused) -> dict[str, Any]:
    """`compose`/`narrate`'s `failed_step`, once the replan budget is spent."""
    if isinstance(outcome, MethodChoiceNeeded):
        return {
            "step_id": outcome.step_id,
            "tool": outcome.tool.name,
            "code": outcome.code,
            "message": outcome.refusal.message,
        }
    if isinstance(outcome, RediscoveryNeeded):
        return {
            "step_id": outcome.step_id,
            "tool": outcome.tool.name,
            "code": "rediscovery_needed",
            "message": outcome.refusal.message
            or f"the source for requirement {outcome.requirement_id!r} needs discovering again",
        }
    # `outcome.detail` wraps the tool's own text ("'aggregate' refused:
    # ..."), but only when the call left one -- with no `result.text`,
    # `_refused` falls back to `failure.code` instead of `failure.message`
    # (`executor.run._Run._stopped_by_refusal`), so `detail` is not a
    # reliable stand-in for what the tool actually said. `payload`, set on
    # every `STEP_REFUSED` stop, carries the tool's own `code` and
    # `message` straight from its `ToolFailure`; a `Refused` raised for any
    # other reason (`EMPTY_ANSWER`, `SHAPE_MISMATCH`, ...) has no tool call
    # to report and keeps today's `refusal.value`/`detail` fallback -- unless
    # it carries `issues` (`executor.run._Run._check_evidence`'s
    # `REQUIREMENT_UNMET`, today's only source of them): `outcome.detail`
    # there is the generic "requirement X is not met by what Y returned",
    # naming neither which of `check_evidence`'s ten sub-checks fired nor
    # what it found versus what the plan required, so the specific
    # `EvidenceIssueCode` and its own expected/actual detail lead instead.
    if outcome.payload is not None:
        return _refused_step(step_id=outcome.step_id, tool=None, failure=outcome.payload)
    if outcome.issues:
        failing = errors(list(outcome.issues)) or list(outcome.issues)
        return {
            "step_id": outcome.step_id,
            "tool": None,
            "code": failing[0].code.value,
            "message": "; ".join(issue.detail for issue in failing),
        }
    return {
        "step_id": outcome.step_id,
        "tool": None,
        "code": outcome.refusal.value,
        "message": outcome.detail,
    }


def _refused_step(*, step_id: str | None, tool: str | None, failure: ToolFailure) -> dict[str, Any]:
    """`compose`/`narrate`'s `failed_step` for a step the tool itself refused.

    A tool's `fix` can carry its own `remedy`: one sentence written for the
    end user (`data_mcp.tool_errors`: an `over_ceiling` on
    `member_cardinality`, and a mint dfo-apis refuses `member_not_resolvable`),
    not the code/detail prose the rest of `fix` holds for a retrying caller. `narrate` appends it to the closing text by
    code (`_with_remedy`) rather than trusting a narrator model to restate it.
    """
    failed: dict[str, Any] = {
        "step_id": step_id,
        "tool": tool,
        "code": failure.code,
        "message": failure.message,
    }
    remedy = failure.fix.get("remedy")
    if isinstance(remedy, str) and remedy:
        failed["remedy"] = remedy
    return failed


def _with_remedy(text: str, failed_step: dict[str, Any] | None) -> str:
    """`text` with `failed_step`'s own `remedy` appended verbatim, when it
    carries one and `text` does not already state it.

    `_NARRATE_INSTRUCTION` asks the narrator model to state a failed step's
    own fix, but an instruction is not enforcement -- a model can summarize
    the remedy away or drop it under length pressure. This is what actually
    guarantees it reaches the user: called on both `narrate`'s model path
    (the model's own drafted text, likely a paraphrase that does not
    already contain the sentence verbatim) and its scripted fallback (which
    already echoes `failed_step["message"]` in full, `remedy` included, so
    the `in text` check is what keeps this a no-op there instead of a
    duplicate).
    """
    remedy = (failed_step or {}).get("remedy")
    if not remedy or remedy in text:
        return text
    return f"{text} {remedy}" if text else remedy


def _with_totals(text: str, totals: list[str]) -> str:
    """`text` with each `_answer_notes` sentence it does not already state
    appended verbatim, the guarantee `_with_remedy` gives a failed step's
    remedy: a reader is told the figure is a total, or that keys were left
    out of it, whatever the draft says."""
    for sentence in totals:
        if sentence not in text:
            text = f"{text} {sentence}" if text else sentence
    return text


# Phrases anaphoric to a window a prior card already resolved. "The same
# years" after a 2015-2020 card means 2015 to 2020. Left unresolved, the
# resolver reads it as a fresh, unbounded time expression.
_WINDOW_ANAPHORS: tuple[str, ...] = (
    "the same years",
    "the same period",
    "the same time frame",
)


def _resolve_window_anaphors(text: str, window: str) -> str:
    """`text` with every `_WINDOW_ANAPHORS` phrase replaced by `window`,
    case-insensitively, so `probe`'s resolver sees the resolved span."""
    resolved = text
    for phrase in _WINDOW_ANAPHORS:
        resolved = re.sub(re.escape(phrase), window, resolved, flags=re.IGNORECASE)
    return resolved


def _resolve_window_anaphors_in_parts(
    parts: list[dict[str, Any]], window: str
) -> list[dict[str, Any]]:
    """The same substitution as `_resolve_window_anaphors`, walked over
    every string field of every search part. `entities` nests
    `{"text": ...}` dicts, so a flat field-by-field pass would miss it."""

    def _walk(value: Any) -> Any:
        if isinstance(value, str):
            return _resolve_window_anaphors(value, window)
        if isinstance(value, list):
            return [_walk(item) for item in value]
        if isinstance(value, dict):
            return {key: _walk(item) for key, item in value.items()}
        return value

    return [_walk(part) for part in parts]


def create_root_workflow(
    *,
    resolver: Resolver,
    executor: Executor,
    tool_surface: ToolSurface | None,
    definition_surface: ToolSurface | None = None,
    depth_hint_extractor: DepthHintExtractor | None = None,
    generative: GenerativeNodes | None = None,
    recall: RecallFn | None = None,
    surface: str = "studio",
    max_concurrent_steps: int = 4,
    fetch_series_stats: FetchSeriesStats = default_fetch_series_stats,
    orchestrator: LlmAgent | None = None,
) -> Workflow:
    """Build the root graph with its external seams injected.

    `depth_hint_extractor` classifies the user's stated depth preference
    (`harness.depth_hints`); omitting it means the depth policy runs on structural
    signals alone: the feature is absent. `generative` carries the generative
    nodes' model seams the same way: `converse`/`compose`/`narrate`/`refusal`
    left unset keep their scripted stand-in, `planner` left unset means the
    graph cannot reach its planner node. `recall` serves converse's ask for
    context beyond the seeded digests (mechanical, conversation-scoped);
    omitting it disables the recall round. `tool_surface` is the servers the
    `execute` node runs an approved workflow against; it carries no default,
    so a caller states its absence (`None`, refused loudly at `execute`)
    rather than falling into it unnoticed. `definition_surface` reaches the
    data-mcp the `seed` node resolves each seed's identity again through
    (`translator.discovery.resolve_definition`), a call the Activity feed
    does not show; a seeded turn without one is refused loudly at `seed`.
    `max_concurrent_steps` bounds how many of one plan's wave's tool calls
    the `execute` node's run may hold in flight against `tool_surface` at
    once (`executor.run.execute_approved_plan`'s own keyword of the same
    name). `fetch_series_stats`
    is that same call's own keyword: the real signed-URL fetch by default,
    injectable for a caller whose `tool_surface` is itself fully virtualised
    (`harness.scripted`'s "no upstream" deterministic mode supplies its own).
    `orchestrator` is the orchestrator engine's agent (`orchestrator.agent`,
    with `harness.orchestrator_tools`): given, the turn opens with it in
    place of converse, probe, clarify, depth, fetch, compose and narrate,
    and its `plan` tool reaches the same `planner`, `approve` and `execute`
    nodes, edges and all, through `plan_start` and `plan_done` below.
    """
    models = generative or GenerativeNodes()

    async def converse(ctx: Context, node_input: Any) -> AsyncIterator[Any]:
        """The turn's opening generative node: restate the question, extract
        structured search parts, produce the opening text -- and route:
        a canvas question (the on-canvas "Ask follow up") goes
        straight to canvas_answer instead of the data search, and a message
        that asks for no data (a greeting, a question about DFO itself) goes
        to reply with converse's own answer. Scripted
        stand-in when no model seam is injected.

        The hint extraction is a real (small-)model call behind the injected
        seam, offloaded to a worker thread because its client is sync, and
        run concurrently with `models.converse`'s draft when both seams are
        present: the two calls take the same `question` and nothing else,
        so neither needs the other's output.

        `question` and a drafted `search_parts` entry can both name a
        window anaphor ("the same years" and siblings, `_WINDOW_ANAPHORS`)
        rather than their own time expression. State's `anaphor_window`,
        seeded per turn by `studio.py` from `ConversationState.
        anaphor_window` (a card-scoped follow-up's own focus card's window,
        else the most recently landed card's, across the full conversation
        history), names the span, and
        `_resolve_window_anaphors`/`_resolve_window_anaphors_in_parts`
        substitute it in, so `probe`'s resolver only ever sees a concrete
        span. No window on record leaves the text untouched.

        Reached twice: fresh from START (`node_input` carries the user's
        Content) and re-entered on clarify's "refine" edge (`node_input` is
        the prior resolve envelope, which carries no question text) -- so
        the question falls back to the state clarify already refined,
        rather than being clobbered by an empty extraction. A non-empty
        `_question_text(node_input)` means a genuinely new turn, since
        only `START` hands this node a user `Content`. `converse` then
        clears every `PER_TURN_STATE_KEYS` entry before this turn's own
        writes below run. A refine re-entry carries no question text, so
        this turn's own `spec_id`, `clarify_rounds` and `replans` stay as
        they are. It is mid-turn, not a new one.
        """
        fresh_question = _question_text(node_input)
        if fresh_question:
            _reset_turn_state(ctx)
        question = fresh_question or str(ctx.state.get("question", ""))
        digest = _state_json(ctx, "canvas_digest")
        history = _state_json(ctx, "conversation_digest")
        window = ctx.state.get("anaphor_window")
        if window:
            question = _resolve_window_anaphors(question, window)
        ctx.state["question"] = question
        hint_call = (
            asyncio.to_thread(depth_hint_extractor, question=question)
            if depth_hint_extractor is not None
            else None
        )
        output: dict[str, Any] = {"question": question}
        about = None
        parts: list[dict[str, Any]] | None = None
        if models.converse is not None:
            converse_call = models.converse(
                question=question, canvas_digest=digest, conversation_digest=history
            )
            if hint_call is not None:
                hint, drafted = await asyncio.gather(hint_call, converse_call)
            else:
                hint = None
                drafted = await converse_call
            requested = drafted.get("recall")
            if recall is not None and isinstance(requested, dict) and requested.get("query"):
                # One bounded round: the model asked for context beyond the
                # digests; the fetch is mechanical (store lookup, no model),
                # then converse redrafts with the slice in hand. A second
                # request is ignored -- the round cap is structural.
                fetched = await recall(
                    scope=str(requested.get("scope") or "conversation"),
                    query=str(requested["query"]),
                )
                drafted = await models.converse(
                    question=question,
                    canvas_digest=digest,
                    conversation_digest=history,
                    recall_result=fetched,
                )
            parts = drafted.get("search_parts")
            if parts and window:
                parts = _resolve_window_anaphors_in_parts(parts, window)
            if parts:
                ctx.state["search_parts"] = json.dumps(parts)
            # Model output naming another conversation's card, a card that
            # is not there, or nothing at all is dropped, so the turn
            # appends the card it would have appended anyway. This runs on
            # every entry to the node, including clarify's "refine" edge,
            # so a refined question that names no card clears the earlier
            # draft's target for the recorder.
            ctx.state["revises"] = _validated_revises(
                drafted.get("revises"),
                composed_ids=_state_json(ctx, "composed_ids", default=[]),
                auto_add=bool(ctx.state.get("auto_add", True)),
            )
            about = drafted.get("about")
            if drafted.get("text"):
                # A reply's text is the whole answer, which `reply` closes the
                # turn with; carried under its own key so it is not also
                # streamed here as an opening line.
                output["reply" if about == "reply" else "text"] = drafted["text"]
        else:
            hint = await hint_call if hint_call is not None else None
        ctx.state["depth_hint"] = hint
        output["depth_hint"] = hint
        if about not in ("canvas", "reply") and models.converse is not None and not parts:
            # A genuine decomposition failure: the model produced no
            # search_parts at all. Close honestly instead of handing the
            # resolver an unscoped `{"variable": question}` part.
            _clear_seed_slot(ctx)
            yield Event(
                output={
                    "status": "unsupported",
                    "warnings": [
                        {
                            "message": (
                                "couldn't identify a specific measure or subject "
                                "to search for in that question"
                            )
                        }
                    ],
                },
                route="deadend",
            )
            return
        if about in ("canvas", "reply"):
            route = about
        elif fresh_question and ctx.state.get(SEEDS_STATE_KEY):
            # Only on the question's own entry: a refine re-entry comes from
            # `clarify`, which a seeded turn reaches only after `seed` fell
            # back to the question and cleared the slot.
            route = "seeded"
        else:
            route = "data"
        if route != "seeded":
            _clear_seed_slot(ctx)
        yield Event(output=output, route=route)

    def reply(ctx: Context, node_input: dict[str, Any]) -> dict[str, Any]:
        """Close a turn that asks for no data -- a greeting, thanks, a question
        about DFO itself -- with converse's own answer. No search runs."""
        del ctx
        return {"text": str(node_input.get("reply") or "")}

    async def canvas_answer(ctx: Context, node_input: Any) -> Any:
        """Answer about the canvas from the digest alone, the same
        bounded-facts rule as narrate. Reached even on an EMPTY canvas (the
        digest is then absent): "what's on my canvas" gets "nothing yet"."""
        del node_input
        digest = _state_json(ctx, "canvas_digest", default={})
        if not digest.get("blocks"):
            return {"text": "There is nothing on your canvas yet."}
        if models.narrate is not None:
            drafted = await models.narrate(
                question=ctx.state.get("question", ""),
                # components_created gates canvas references in the narrate
                # prompt; for a canvas question the canvas is the subject,
                # so it counts the blocks that are really there.
                facts={"canvas": digest, "components_created": len(digest["blocks"])},
            )
            return {"text": drafted.get("text", "")}
        titles = [block.get("title", "untitled") for block in digest.get("blocks", [])]
        return {"text": f"On the canvas: {'; '.join(titles)}."}

    async def seed(ctx: Context, node_input: Any) -> AsyncIterator[Any]:
        """Start a turn from the datasets the user handed over from Search.

        The seeds (`GraphState.seeds`) are the Search results the user
        handed to Studio, with the operations their question asked for, as
        `seeds_slot` keeps them: the host reads each posted Spec block as a
        `harness.wire.Seed`, which keeps its spec_id, identity and
        operations and drops the rest of the block, and checks its
        identity against its spec_id, before the session exists. This node
        reads the slot once and clears it, so the identities ride no later
        state write. Each seed is taken once (`take_seeds`): a seed that
        cannot be taken fails its own `SEED_STEP_PREFIX-<n>` step and is left
        out, and the others go on.

        The browser's block is never read for what it says of the dataset.
        The node resolves each taken seed's identity again through Search's
        `dataset_resubmit` door, all at once, one data-mcp
        `resolve_definition` call per seed through `definition_surface`, and
        reads every label, sentence, population fact, slot and StatVar the
        turn uses from Search's answer (`read_answer`). data-mcp stores the
        definition as it answers, so the data-mcp that `get_data` reaches
        holds it wherever it runs. The node mints nothing: the plan retrieves
        each seed over the years it chooses, and `fetch` over the year Search
        showed when the question named none.

        A seed data-mcp or Search could not be reached for fails its own
        step as `service_unavailable`. A seed whose identity Search refuses,
        no longer resolves to a definite part, resolves to another spec_id,
        or answers with a block the turn cannot read is dropped: its own step
        fails as `SEED_DROPPED`, with the reason, and it is recorded as a
        drop (`_record_drops`), so the plan card's note, or
        the turn's closing text, names it and why, and no plan in the turn
        pins it or takes it from a search. The turn plans from the other
        seeds. The `SEEDS_STEP_ID` step counts the seeds kept.

        With the seeds read, it records each as a
        `planner.plan.SeededSource` (`SEEDED_SOURCES_STATE_KEY`), which the
        translator pins from and `propose_plan` counts as the user's own
        election, and sets the state `probe` sets -- `spec_id`,
        `spec_id_elected`, `parts`, `resolve_envelope` -- so the rest of the
        graph proceeds as if search had resolved the question to these
        datasets. The envelope (`seeded_envelope`) is built from the same
        `graph_envelope` reading `probe`'s resolver gives a search of each
        seed: its top level states the turn, and `seeds` states each kept
        seed as the planner reads it (`_envelope_seed`).

        A download retrieves one dataset and plans nothing
        (`harness.depth.decide_depth`), so on the `download` surface the
        first kept seed is the turn's and every other kept seed fails its
        own step as `ONE_DATASET_PER_DOWNLOAD`. The seed it retrieves keeps its
        data and loses its operations, which a `SEED_STEP_PREFIX-<n>-operations`
        step reports as `NO_OPERATIONS_ON_DOWNLOAD`.

        A slot holding one `{spec_id, content_version}` reference, the shape
        `seed_spec` was reserved with, has no block to read: the
        `SEEDS_STEP_ID` step fails as `SEED_REFERENCE_UNUSED`.

        No seed taken, or none kept, routes to `probe`, and the turn
        proceeds on the question alone, less any dataset it dropped
        (`_without_dropped_datasets`). The `SEEDS_STEP_ID` step is recorded on
        every turn that posted a list of seeds: when data-mcp or Search could
        not be reached for a seed and none was kept, it fails with the first
        such failure; otherwise it counts the seeds kept, none included.
        """
        del node_input
        raw = _state_json(ctx, SEEDS_STATE_KEY, default=[])
        ctx.state[SEEDS_STATE_KEY] = None
        record_step = getattr(executor, "record_step", None)
        if isinstance(raw, dict):
            seed_step(
                record_step,
                step_id=SEEDS_STEP_ID,
                args={"seeds": 0},
                result={
                    "outcome": "refusal",
                    "refusal": {
                        "code": SEED_REFERENCE_UNUSED,
                        "retry_mode": "do_not_retry",
                        "fix": None,
                        "message": (
                            "seed_spec holds one {spec_id, content_version} reference, which "
                            "carries no Spec block to start from, so the turn uses the question; "
                            "send a list of seeds to start from Search results"
                        ),
                    },
                },
            )
        taken = take_seeds(raw, record_step=record_step)
        if not taken:
            if isinstance(raw, list) and raw:
                seeds_received(record_step, posted=len(raw), taken=0, kept=0)
            yield Event(output={"seeded": False}, route="fallback")
            return
        if definition_surface is None:
            raise RuntimeError(
                "the seed node has no definition surface: create_root_workflow("
                "definition_surface=...) was not given one, so no seed can be resolved"
            )
        answers = await asyncio.gather(
            *(resolve_seed(definition_surface, item) for item in taken),
            return_exceptions=True,
        )
        stored: list[ReadSeed] = []
        unreachable: list[DefinitionNotResolvedError] = []
        left_out: list[SeededSource] = []
        drops: list[DroppedSeed] = []
        for item, answer in zip(taken, answers, strict=True):
            if isinstance(answer, DefinitionNotResolvedError) and answer.retryable:
                logger.warning("data-mcp could not resolve seed %d", item.index, exc_info=answer)
                unreachable.append(answer)
                refuse_seed(
                    record_step,
                    item.index,
                    code="service_unavailable",
                    retry_mode="retry_after_delay",
                    detail=f"data-mcp could not resolve seed {item.index}: {answer}",
                )
                continue
            if isinstance(answer, DefinitionNotResolvedError):
                reason = str(answer)
            elif isinstance(answer, BaseException):
                raise answer
            else:
                try:
                    stored.append(read_answer(item, answer))
                    continue
                except StaleSeedError as exc:
                    reason = str(exc)
                except Exception as exc:
                    logger.warning(
                        "Search's answer for seed %d cannot be read", item.index, exc_info=True
                    )
                    reason = f"Search's answer for it cannot be read ({type(exc).__name__}: {exc})"
            reason = reason.rstrip(".")
            refuse_seed(
                record_step,
                item.index,
                code=SEED_DROPPED,
                detail=f"seed {item.index} is left out: {reason}. Search for it again to add it.",
            )
            name = f"dataset {item.index + 1} from Search"
            spec = item.seed.spec
            left_out.append(
                SeededSource(
                    source=SelectedSource(
                        spec_id=spec.spec_id,
                        sentence=name,
                        has_data=None,
                        identity=spec.definition.identity,
                    )
                )
            )
            drops.append(DroppedSeed(name=name, reason=reason))
        if drops:
            # Named by the seed's position: Search gave no name the turn can trust.
            _record_drops(ctx, left_out, drops)
        if not stored:
            seeds_received(
                record_step, posted=len(raw), taken=len(taken), kept=0, unreachable=unreachable
            )
            yield Event(output={"seeded": False}, route="fallback")
            return
        if (ctx.state.get("surface") or surface) == "download":
            for read in stored[1:]:
                refuse_seed(
                    record_step,
                    read.index,
                    code=ONE_DATASET_PER_DOWNLOAD,
                    detail=(
                        f"a download retrieves one dataset, so seed {read.index} is left "
                        "out; download it on its own"
                    ),
                )
            stored = stored[:1]
            if operations := stored[0].planned["operations"]:
                asked = ", ".join(f'{op["verb"]} ("{op["phrase"]}")' for op in operations)
                seed_step(
                    record_step,
                    step_id=f"{SEED_STEP_PREFIX}-{stored[0].index}-operations",
                    args={"seed": stored[0].index},
                    result={
                        "outcome": "refusal",
                        "refusal": {
                            "code": NO_OPERATIONS_ON_DOWNLOAD,
                            "retry_mode": "do_not_retry",
                            "fix": None,
                            "message": (
                                "a download retrieves the data and computes nothing, so "
                                f"seed {stored[0].index} is retrieved without its operations: "
                                f"{asked}; ask for them in Studio"
                            ),
                        },
                    },
                )
        seeds_received(record_step, posted=len(raw), taken=len(taken), kept=len(stored))
        save_seeded_sources(ctx.state, [read.source for read in stored])
        envelope = seeded_envelope([read.planned for read in stored])
        ctx.state["spec_id"] = envelope["spec_id"]
        ctx.state["spec_id_wide"] = bool(envelope.get("wide", False))
        ctx.state["spec_id_elected"] = True
        ctx.state["spec_id_elected_label"] = stored[0].planned["sentence"]
        ctx.state["parts"] = len(stored)
        ctx.state["resolve_envelope"] = json.dumps(envelope)
        yield Event(output=envelope, route="seeded")

    async def split_denominators(
        ctx: Context, parts: list[dict[str, Any]], envelope: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Search again for the parts `envelope` refused for their denominator.

        The triggers are reason codes alone (`_refused_for`). A part refused
        as `DENOMINATOR_NOT_AVAILABLE` is searched as two, its measure and
        its denominator: the `denominator_split` seam names them, the parts
        are built from the refused part (`_measure_and_denominator`), and
        one search resolves `parts` with each pair in its part's place,
        under the question as the user asked it. A built part equal to one
        already in that search, or to a part of `parts` no split replaces,
        is not added: several measures per head of one population share one
        denominator part (`_split_search`). When the search would still hold
        more than `MAX_PARTS` parts, the most `search` takes, it is not made.

        A part that states a unit and is refused as `UNIT_NOT_RESOLVED` may
        hold a per-head phrase as its unit. Search decides: one search asks
        for those parts with the unit phrase moved into the variable
        (`_unit_in_variable`). When search refuses every one of them for its
        denominator, each is split too, from the part as first drafted and
        that answer's message, so a currency stated beside the phrase stays
        the measure's unit. Any other outcome discards the search, so a
        unit search could not read is never dropped or read as something
        else.

        Both searches are attempts to recover a refusal `probe` already
        holds. One that search refuses or that does not complete
        (`_SEARCH_FAILURES`) is logged and makes no further search.

        Nothing is searched again on a turn that would not plan the two
        sources: the depth policy is asked what a definite answer with the
        split's parts would get on this turn, and `none`, which goes to
        `fetch` and retrieves the first part alone, leaves the refusal
        standing. That is every turn on the `download` surface.

        All of it runs once for each question `converse` drafts
        (`GraphState.denominator_split_tried`): at most two searches and one
        model call per refused part beyond the first search, and at most
        `MAX_CLARIFY_ROUNDS` redrafts a turn.

        `None` when no search was made beyond the first. Otherwise `probed`,
        the variables searched with their unit moved in; `replaced`, the
        variables of the parts split; `searched_as`, how many parts stand
        for them in the split's search; `parts`, the variable of each of
        those and its status in the answer, in the search's order; and
        `kept`, whether every one of them resolves (`_resolves`), with that
        answer as `envelope` and the parts searched as `search_parts` when
        it is. `replaced` is empty and `searched_as` zero when no split was
        searched, and `parts` is empty when none was answered part for
        part."""
        refused = _refused_for(envelope, parts, DENOMINATOR_NOT_AVAILABLE)
        unread = [
            index
            for index in _refused_for(envelope, parts, UNIT_NOT_RESOLVED)
            if parts[index].get("unit")
        ]
        planned = decide_depth(
            status="definite",
            choice_points=0,
            degraded=False,
            part_count=len(parts) + len(refused) + len(unread),
            surface=ctx.state.get("surface") or surface,
            depth_hint=ctx.state.get("depth_hint"),
        )
        if (
            not (refused or unread)
            or models.denominator_split is None
            or ctx.state.get("denominator_split_tried")
            or planned.depth == "none"
        ):
            return None
        ctx.state["denominator_split_tried"] = True
        question = str(ctx.state.get("question", ""))
        digests = envelope["part_digests"]
        messages = {index: str(digests[index].get("message") or "") for index in refused}
        outcome: dict[str, Any] = {
            "probed": [],
            "replaced": [],
            "searched_as": 0,
            "parts": [],
            "kept": False,
        }

        async def search_again(asked: list[dict[str, Any]]) -> dict[str, Any] | None:
            try:
                return await resolver(question=question, parts=asked)
            except _SEARCH_FAILURES:
                logger.warning(
                    "a search for a part refused for its denominator or its unit "
                    "failed, so the refusal stands: parts=%s",
                    json.dumps(asked),
                    exc_info=True,
                )
                return None

        if unread:
            moved = [
                _unit_in_variable(part) if index in unread else part
                for index, part in enumerate(parts)
            ]
            outcome["probed"] = [moved[index]["variable"] for index in unread]
            logger.info("searching again with the unit in the variable: %s", outcome["probed"])
            probed = await search_again(moved)
            if probed is None:
                messages = {}
            elif set(unread) <= set(_refused_for(probed, moved, DENOMINATOR_NOT_AVAILABLE)):
                answered = probed["part_digests"]
                messages.update(
                    {index: str(answered[index].get("message") or "") for index in unread}
                )
        if not messages:
            return outcome
        split_at = sorted(messages)
        drafts = await asyncio.gather(
            *(
                models.denominator_split(
                    question=question, part=parts[index], message=messages[index]
                )
                for index in split_at
            )
        )
        split = _split_search(parts, dict(zip(split_at, drafts, strict=True)))
        if split is None:
            return outcome if outcome["probed"] else None
        search_parts, positions = split
        outcome["replaced"] = [str(parts[index].get("variable")) for index in split_at]
        outcome["searched_as"] = len(positions)
        logger.info(
            "searching again for parts refused for their denominator: replaced=%s parts=%s",
            json.dumps([parts[index] for index in split_at]),
            json.dumps([search_parts[at] for at in positions]),
        )
        again = await search_again(search_parts)
        if again is None:
            return outcome
        found = again.get("part_digests") or []
        if len(found) != len(search_parts):
            logger.info(
                "the search of a denominator split answered %d parts for the %d asked, "
                "so the refusal stands: parts=%s",
                len(found),
                len(search_parts),
                json.dumps(search_parts),
            )
            return outcome
        outcome["parts"] = [
            {"variable": search_parts[at].get("variable"), "status": found[at].get("status")}
            for at in positions
        ]
        outcome["kept"] = all(_resolves(found[at]) for at in positions)
        if outcome["kept"]:
            outcome["envelope"] = again
            outcome["search_parts"] = search_parts
        return outcome

    async def probe(ctx: Context, node_input: Any) -> Any:
        """Resolve the question via the injected resolver (data-mcp search).

        A part search refused for its denominator, a per-head figure with no
        such series, is searched once more as two parts, the measure and its
        denominator (`split_denominators`), and so is one whose unit holds
        the per-head phrase. Parts that share a denominator are searched
        with it once. When every such part resolves, that search's answer
        is the turn's, `search_parts` holds the parts it asked, and the turn
        goes on as a question that named each measure does. Otherwise the
        first answer stands, as it does when that search fails or would ask
        more parts than `search` takes. Either way the output says so under
        `denominator_split`, which the Activity feed reads and
        `resolve_envelope`, the planner's reading, does not carry.

        A reading of a dataset the turn dropped is taken out of the answer
        (`_without_dropped_datasets`)."""
        del node_input  # reads state: clarify re-entry must not depend on edge payload
        parts = _state_json(ctx, "search_parts")
        envelope = await resolver(
            question=ctx.state.get("question", ""),
            spec_id=ctx.state.get("spec_id"),
            # Only load-bearing on the `spec_id`-set (clarify re-entry)
            # branch -- see `_candidate_wide` and `McpResolver.__call__`.
            wide=bool(ctx.state.get("spec_id_wide", False)),
            **({"parts": parts} if parts else {}),
        )
        split = await split_denominators(ctx, parts or [], envelope)
        if split is not None and split["kept"]:
            envelope = split.pop("envelope")
            ctx.state["search_parts"] = json.dumps(split.pop("search_parts"))
        envelope = _without_dropped_datasets(ctx, envelope)
        # An election survives only while the envelope still names the id
        # `clarify` set: a fresh probe with no clarify carries no election
        # to survive, and a resolver that comes back naming a different
        # spec_id has moved past whatever the user picked.
        elected = bool(ctx.state.get("spec_id_elected")) and envelope.get(
            "spec_id"
        ) == ctx.state.get("spec_id")
        ctx.state["spec_id_elected"] = elected
        if not elected:
            # Keep the label consistent with the flag it describes, even
            # though every reader gates on `spec_id_elected` alone.
            ctx.state["spec_id_elected_label"] = None
        if envelope.get("spec_id"):
            ctx.state["spec_id"] = envelope["spec_id"]
        ctx.state["parts"] = int(envelope.get("parts", 1))
        if "part_digests" in envelope:
            # Each part's spec block rides on its digest for a caller that
            # fetches every part (`harness.orchestrator_tools`); the graph
            # fetches the first, whose block is `spec_block`, so state keeps
            # the summaries alone.
            part_digests = [
                {
                    key: value
                    for key, value in part.items()
                    if key not in ("spec_block", "candidate_blocks")
                }
                for part in envelope["part_digests"]
            ]
        else:
            # A clarify re-entry resolves the first part alone (the one
            # clarify asked about); the other parts keep the first search's
            # summary.
            part_digests = _state_json(ctx, "part_digests", default=[])
            if part_digests and envelope.get("status") == "definite":
                first = part_digests[0]
                chosen = next(
                    (c for c in first["candidates"] if c["spec_id"] == envelope["spec_id"]), {}
                )
                first = {
                    **first,
                    "status": "definite",
                    "spec_id": envelope["spec_id"],
                    "resolved": chosen.get("label"),
                    "wide": bool(envelope.get("wide", False)),
                    "candidates": [],
                }
                part_digests = [first, *part_digests[1:]]
        ctx.state["part_digests"] = json.dumps(part_digests)
        envelope = {key: value for key, value in envelope.items() if key != "part_digests"}
        ctx.state["resolve_envelope"] = json.dumps(envelope)
        output = {**envelope, "part_digests": part_digests}
        if split is not None:
            output["denominator_split"] = split
        return output

    def route_status(ctx: Context, node_input: dict[str, Any]) -> Iterator[Any]:
        """Branch on the resolve outcome. Output and route ride one event."""
        del ctx
        status = str(node_input.get("status", "unsupported"))
        route = status
        if status in ("candidates", "no_data") and not node_input.get("candidates"):
            # Nothing to choose between, for any status that would
            # otherwise reach clarify: an elicitation here would carry zero
            # options, which the frontend cannot answer, and past
            # `MAX_CLARIFY_ROUNDS` clarify would otherwise loop "again"
            # back through probe forever (ADK has no iteration cap of its
            # own -- module docstring). Close the turn.
            #
            # The route to `deadend` is "unsupported" either way, but the
            # output status keeps `no_data` when that was the finding: an
            # empty `candidates` status would mislead the refusal seam into
            # implying there were options, while `no_data` is already the
            # honest reason.
            route = "unsupported"
            if status == "candidates":
                status = "unsupported"
        output = (
            node_input if status == node_input.get("status") else {**node_input, "status": status}
        )
        yield Event(output=output, route=route)

    def clarify(ctx: Context, node_input: dict[str, Any]) -> Iterator[Any]:
        """Ask the user to choose a reading, then re-resolve. Bounded.

        `awaiting_clarify` distinguishes a fresh entry from a resumed one:
        the interrupt id is stable, so a stale answer from a previous round
        must not be consumed twice.

        Sets `ctx.state["spec_id_elected"]`/`["spec_id_elected_label"]`
        alongside `spec_id` on every branch below (see `GraphState` for what
        they mean): only the branch where the user actually names a
        candidate sets `spec_id_elected` true, with the label to match; the
        skip, round-bound, and open-text-refine branches all set a system
        default instead. `probe`'s re-entry re-derives the flag from
        whether the resolver still names the same id (see `probe`).

        The question opens with the seeds the turn dropped that no earlier
        words named (`_with_untold_drops`).
        """
        answer = (ctx.resume_inputs or {}).get(CLARIFY_INTERRUPT)
        rounds = int(ctx.state.get("clarify_rounds", 0))
        if ctx.state.get("awaiting_clarify") and answer is not None:
            ctx.state["awaiting_clarify"] = False
            ctx.state["clarify_rounds"] = rounds + 1
            candidates = node_input.get("candidates", [])
            if isinstance(answer, dict) and answer.get("skip"):
                # Declined to choose: proceed on the default reading,
                # exactly as the round bound below does. A system default,
                # not an election.
                chosen = _top_candidate_spec_id(node_input)
                ctx.state["spec_id"] = chosen
                ctx.state["spec_id_wide"] = _candidate_wide(candidates, chosen)
                ctx.state["spec_id_elected"] = False
                ctx.state["spec_id_elected_label"] = None
                yield Event(output=node_input, route="again")
                return
            if isinstance(answer, dict) and answer.get("text"):
                # An open-ended reply refines the question rather than
                # picking a candidate: it needs re-decomposing, not a bare
                # re-search on stale (or absent) parts, so this re-enters
                # through converse ("refine") instead of probe directly.
                ctx.state["spec_id"] = None
                ctx.state["spec_id_wide"] = False
                ctx.state["spec_id_elected"] = False
                ctx.state["spec_id_elected_label"] = None
                ctx.state["search_parts"] = None
                ctx.state["denominator_split_tried"] = False
                ctx.state["question"] = (
                    f"{ctx.state.get('question', '')} -- {answer['text']}".strip(" -")
                )
                yield Event(output=node_input, route="refine")
                return
            # The one genuine election: the user named a candidate.
            chosen = answer.get("spec_id") if isinstance(answer, dict) else str(answer)
            ctx.state["spec_id"] = chosen
            ctx.state["spec_id_wide"] = _candidate_wide(candidates, chosen)
            ctx.state["spec_id_elected"] = True
            ctx.state["spec_id_elected_label"] = _candidate_label(candidates, chosen)
            yield Event(output=node_input, route="again")
            return
        if rounds >= MAX_CLARIFY_ROUNDS:
            # The round bound took the top-ranked candidate; no one chose
            # it, so it is a system default, not an election.
            chosen = _top_candidate_spec_id(node_input)
            ctx.state["spec_id"] = chosen
            ctx.state["spec_id_wide"] = _candidate_wide(node_input.get("candidates", []), chosen)
            ctx.state["spec_id_elected"] = False
            ctx.state["spec_id_elected_label"] = None
            yield Event(output=node_input, route="again")
            return
        ctx.state["awaiting_clarify"] = True
        yield RequestInput(
            interrupt_id=CLARIFY_INTERRUPT,
            message=_with_untold_drops(ctx, "Which reading did you mean?"),
            payload={
                "candidates": node_input.get("candidates", []),
                "warnings": node_input.get("warnings", []),
                "round": rounds + 1,
                "max_rounds": MAX_CLARIFY_ROUNDS,
            },
        )

    async def deadend(ctx: Context, node_input: dict[str, Any]) -> Any:
        """`unsupported` (or `no_data` with nothing to choose): never retried.

        The user-facing close is PLAIN language: with a refusal seam the
        model rewrites the resolver's diagnostics into one or two honest
        sentences (what failed, in the user's terms, and an invitation to
        rephrase); without one, a fixed sentence. The raw diagnostics ride
        on the resolve step's `detail` (the Activity panel) instead.
        Plainness of the model's own wording rests on the instruction and
        the live eval's refusal rubric alone.

        Two callers arrive with `text` already drafted, which this node
        renders verbatim rather than routing it through a second, unrelated
        drafting step: `planner_n`'s own `"refused"` route -- a
        plan-version-cap refusal has nothing in common with the resolver's
        diagnostics this node otherwise reads from -- and a search that
        resolves only to datasets the turn dropped
        (`_without_dropped_datasets`).

        Seeds the turn dropped that no earlier words named are told first
        (`_with_untold_drops`)."""
        supplied = node_input.get("text")
        if isinstance(supplied, str) and supplied:
            return {"text": _with_untold_drops(ctx, supplied)}
        status = str(node_input.get("status", ""))
        warnings = [
            str(w.get("message", "")) for w in node_input.get("warnings", []) if w.get("message")
        ]
        if status == "no_data":
            fallback = "I couldn't find data matching that question in the published datasets."
        else:
            fallback = (
                "That request is understood but cannot be expressed against the data as published."
            )
        if models.refusal is not None:
            drafted = await models.refusal(
                question=str(ctx.state.get("question", "")), status=status, warnings=warnings
            )
            text = str(drafted.get("text") or "").strip()
            if text:
                return {"text": _with_untold_drops(ctx, text)}
        return {"text": _with_untold_drops(ctx, fallback)}

    def choose_depth(ctx: Context, node_input: dict[str, Any]) -> Iterator[Any]:
        """Apply the pure depth policy and route on its verdict."""
        signal = _settle_depth(ctx, node_input, surface=surface)
        yield Event(output=signal.model_dump(), route=signal.depth)

    async def fetch(ctx: Context, node_input: Any) -> AsyncIterator[Any]:
        """The no-planner path: resolved spec straight to data.

        The call runs as its own task under `FETCH_STEP_ID`, so the node
        relays its progress reports as they arrive, the same way `execute`
        relays a plan's. A seed whose question named no year is retrieved
        over the year Search showed (`SeededSource.seen_year`), as the call's
        `scope.window`; every other spec over its own window.

        A refused call yields no frame refs and a `failed_step` in the shape
        `execute` gives a stopped plan (`_refused_step`), so `narrate` names
        the refusal and appends its remedy. No plan runs here, so nothing
        replans.
        """
        del node_input
        args: dict[str, Any] = {"spec_id": ctx.state.get("spec_id")}
        fetched = _fetched_seed(ctx)
        if fetched is not None and fetched.seen_year is not None:
            year = fetched.seen_year
            args["scope"] = {"window": {"kind": "range", "start_year": year, "end_year": year}}

        async def get_data(on_progress: StepProgressCallback) -> dict[str, Any]:
            # The task runs in its own copy of the context, so this setting
            # never reaches the node's.
            current_step_id.set(FETCH_STEP_ID)
            return await executor(
                tool="get_data", args=args, on_progress=partial(on_progress, FETCH_STEP_ID)
            )

        task, relay = _start_relayed(get_data)
        try:
            async for report in relay.latest():
                yield _step_progress_event(report, tool="get_data")
            result = await task
        finally:
            await _stop(task)
        summaries = [str(result["summary"])] if result.get("summary") else []
        refused = result.get("outcome") == "refusal"
        frame_refs = [] if refused else result.get("frame_refs", [])
        output: dict[str, Any] = {
            "frame_refs": frame_refs,
            # No plan, so this one get_data call is the only (and thus
            # final) step -- same shape compose expects from `execute`.
            "frame_ref_steps": [
                {"step": 0, "tool": "get_data", "is_final_step": True} for _ in frame_refs
            ],
            "step_summaries": summaries,
        }
        if refused:
            output["failed_step"] = _refused_step(
                step_id=FETCH_STEP_ID,
                tool="get_data",
                failure=parse_tool_failure(result.get("refusal") or {}),
            )
        yield Event(output=output)

    async def planner(ctx: Context, node_input: Any) -> AsyncIterator[Any]:
        """The planner node.

        A `PlannerModel` seam drafts a compiled `planner.plan.Plan` from the
        resolve envelope and this node only validates it (`PlannerModel`'s
        contract): versioning, `previous_version` lineage and the
        DRAFT/PENDING_APPROVAL transitions are enforced by `planner.plan.Plan`
        itself, inside the seam, never here. Every caller supplies a seam --
        scripted mode's is `harness.scripted.ScriptedPlannerModel` -- so a
        graph reaching this node with none configured is a construction
        defect, refused loudly rather than answered with a different plan
        shape, the same discipline `execute` uses for a missing tool surface.

        A draft can run long (the planner loads skills, proposes,
        translates, and corrects) with nothing to show for it until
        this returns. So this node queues each `on_progress` report as it
        arrives and relays the queued reports as their own `Event`s once the
        await settles (see the direct-await discussion below). `execute`,
        below, relays its reports live instead, because its run never calls
        `ctx.run_node` and so can run as a task of its own. Carried on
        `custom_metadata`, never `output`: `ctx.output` (set
        by any yielded event whose `output` is not `None`,
        `_node_runner.py`'s `_track_event_in_context`) accepts exactly one
        write per node run and raises on a second, so only the final yield,
        the drafted plan (or the refusal, below), may use it.
        `custom_metadata` is untouched by that rule, so every progress event
        passes straight through as a plain activity update and
        `studio_translate._Translator.translate` reads
        `event.custom_metadata["progress"]` instead of `event.output` to
        tell the two apart.

        Three routes out, all explicit on the final yield: `"ok"`, to
        `approve_n`, for a drafted plan; `"pending_refusal"`, also to
        `approve_n`, for a seam that refuses to draft one at all -- either
        `PlannerSeam`'s own `MAX_PLAN_ROUNDS` cap, or `starts_new_plan`
        refused as `QUESTION_FIXED` because this seam's pair is built with
        the question fixed (`harness.planner_seam.PlannerSeam.__init__`) --
        surfaced as `{"refused": {"text": ...}}` rather than a raised
        `PlannerSeamError` -- while `ctx.state["plan"]` still holds a plan
        `PENDING_APPROVAL`; and `"refused"`, to `deadend_n`, for the same
        two refusals when it does not. Raising `PlannerSeamError` here
        instead would end the turn through `_stream_fatal`'s own
        `close_turn(error=...)`, which `studio_state.begin_turn` documents
        as abandoning the paused session and its plan -- exactly the plan
        a refusal that still has one pending needs the user to be able to
        act on. `ctx.state["plan"]` is never written on either refusal
        route, so it is unchanged either way; what differs is only whether
        that unchanged record is worth routing back to `approve_n` for.
        `"refused"` covers a refusal on a first draft (nothing was ever
        submitted this turn) and on a replan (the record on file is already
        `APPROVED`, not something a fresh approval interrupt could act on)
        -- both end the turn the same plain way `converse`'s and
        `route_status`'s own dead ends do. When the turn dropped seeds the
        user has not been told of (`_untold_drops`: `seed` dropped them as it
        took them), the `"ok"` route's plan carries `dropped_seeds_note` as
        `refusal_note`, which
        `harness.studio_translate` shows before the plan card, and the
        `"refused"` route's text is `unplanned_seeds_text`, which names the
        dropped seeds and why, the seeds left unplanned, and the planner's
        reason. A question the planner asks during the draft opens with them
        instead (`PLANNER_STARTED_KEY`). A seam that raises fails the turn.
        On the `"refused"` route `plan_done` lists the asks an earlier run
        of the plan call computed (`ORCHESTRATOR_KEPT_ASKS_KEY`).

        An amendment or a replan of a plan stored in a shape this release
        does not read (`_readable_plan`) is drafted as a first draft, and
        the `"ok"` route's plan carries `UNREADABLE_PLAN_NOTE` as
        `refusal_note`, ahead of any `dropped_seeds_note`, and
        `"restarted": True`, since its version counts from 1 again. The
        `"refused"` route's text opens with `UNREADABLE_PLAN_UNPLANNED_NOTE`.

        `models.planner(...)` is awaited directly here, with no
        `asyncio.ensure_future` wrapper of its own: ADK's own docstring
        warns against wrapping a `ctx.run_node`-calling call in a task like
        that, and this node has two reasons that matters. A real
        `PlannerModel` seam (`harness.planner_seam.PlannerSeam`) calls
        `ctx.run_node` from inside it, and that call can genuinely pause
        mid-turn -- a planner-side ask (`planner.agent`'s
        `adk_request_input` tool) suspends the whole run by raising
        `NodeInterruptedError` straight out of the awaited call, a
        `BaseException` `ctx.run_node` uses exclusively for this
        (`google.adk.workflow._errors`); wrapped in a `Task`, that exception
        would sit on the `Task` instead of propagating up through this
        node's own coroutine the way its `rerun_on_resume=True` re-entry
        depends on. The direct await also closes a real turn more simply:
        `studio.py`'s `_sse_stream` -> `_unwind` cancels the event stream on
        a stop or a disconnect, throwing into whichever call is suspended --
        with no separate `Task` to orphan, the throw propagates straight
        through `ctx.run_node` on its own, with no explicit `.cancel()`
        needed here. The cost is `on_progress`: a report queues rather than
        relaying as its own `Event` the instant it arrives, since yielding
        one while a nested plain `await` is still pending would need a
        second concurrent task to rendezvous with, which this design
        avoids creating. Reports queue as they happen and relay the moment
        the `await` returns. A pause drops them instead: ADK streams the
        ask before the pause reaches this node, so anything relayed then
        would follow the ask.
        """
        if models.planner is None:
            raise RuntimeError(
                "the planner node has no PlannerModel seam: create_root_workflow(generative=...) "
                "must supply one, scripted mode included"
            )
        raw = ctx.state.get("plan")
        question = ctx.state.get(ORCHESTRATOR_BRIEF_KEY if orchestrator else "question", "")
        payload = node_input if isinstance(node_input, dict) else {}
        refusal = payload.get("refusal")
        # Seeds `seed` dropped as it took them that the user has not been
        # told of. The plan card's note names them, or the closing text when
        # no plan is drafted.
        dropped = _untold_drops(ctx)
        amendment_answer = payload.get("answer")
        amendment = (
            amendment_answer.get("amendment") if isinstance(amendment_answer, dict) else None
        )
        # A first-draft entry -- the `choose_depth` edge, the only one of
        # the three that lands here with neither a `"plan"` key (`approve`'s
        # own "amend" route, whether the answer carries amendment text or
        # merely names a stale revision -- `_resume`d with an
        # unrecognised one, say) nor a `"refusal"` key (`execute`'s own
        # "replan" route) -- passes no `prior_plan`, even when
        # `ctx.state["plan"]` already holds one. Keyed on which of those the
        # payload carries, not on whether `refusal`/`amendment` themselves
        # are `None`: a stale-revision "amend" carries no amendment text
        # either, and reading that shape as a first draft would wipe out a
        # plan still `PENDING_APPROVAL` right before `approve_n` re-offers
        # it. Studio never has one to withhold: each turn opens a brand-new
        # session, so `existing` is already `None` there. `evals/agents/
        # dfo_analyst_graph` (`just adk-web`) keeps one session across
        # turns, so on a later turn's first draft `existing` is the turn
        # before's own plan -- withholding it here is what lets
        # `harness.planner_seam.PlannerSeam.__call__` tell this is a genuine
        # first draft and clear its own `planner:plan` and briefed-flag
        # state for it (see that module's docstring), rather than reading
        # as a continuing call on an already-briefed session that would
        # need `starts_new_plan` -- refused, this seam's pair being built
        # with the question fixed -- to draft anything but an amendment to
        # that stale plan.
        envelope = _state_json(ctx, "resolve_envelope", default={})
        existing = _readable_plan(ctx, raw)
        is_first_draft = "plan" not in payload and "refusal" not in payload
        # The plan this entry amends or replans could not be read: found so
        # by `approve` or `execute`, which cleared it, or here, on the resume
        # of a question asked while amending it. Nothing is left to amend or
        # to replan from, so the question is drafted from the start and the
        # card says why. A first draft never read the plan on record, and
        # says nothing of one it cannot read.
        unreadable = bool(payload.get("unreadable_plan")) or bool(
            not is_first_draft and raw and existing is None
        )
        if unreadable:
            refusal = amendment = None
            is_first_draft = True
        started: dict[str, Any] = {"revising": not is_first_draft}
        if dropped:
            started["untold_drops"] = dropped_seeds_text(dropped)
        yield Event(custom_metadata={PLANNER_STARTED_KEY: started})

        progress: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        try:
            drafted = await models.planner(
                ctx=ctx,
                question=question,
                envelope=envelope,
                prior_plan=(
                    None
                    if is_first_draft
                    else (existing.model_dump(mode="json") if existing else None)
                ),
                refusal=refusal,
                amendment=amendment,
                canvas_digest=_state_json(ctx, "canvas_digest"),
                conversation_digest=_state_json(ctx, "conversation_digest"),
                spec_id_elected=bool(ctx.state.get("spec_id_elected")),
                spec_id_elected_label=ctx.state.get("spec_id_elected_label"),
                on_progress=progress.put_nowait,
                # The classic engine passes no `readings`; its seams render the envelope.
                **(
                    {
                        "readings": [
                            *(ctx.state.get(ORCHESTRATOR_PRIOR_LEDGER_KEY) or {}).get(
                                "readings", []
                            ),
                            *(ctx.state.get(ORCHESTRATOR_SEARCHES_KEY) or []),
                        ]
                    }
                    if orchestrator
                    else {}
                ),
            )
        except NodeInterruptedError:
            # A planner-side ask paused the draft mid-turn, and ADK has
            # already streamed the ask, so the reports queued so far are
            # dropped with it: relayed now they would follow the ask. The
            # question opened with the drops (`PLANNER_STARTED_KEY`), so they
            # are told.
            ctx.state["dropped_untold"] = None
            raise
        ctx.state["dropped_untold"] = None
        while not progress.empty():
            yield Event(custom_metadata={"progress": progress.get_nowait()})
        refused = drafted.get("refused") if isinstance(drafted, dict) else None
        if refused is not None:
            text = str(refused.get("text", ""))
            if existing is not None and existing.status is PlanStatus.PENDING_APPROVAL:
                # The plan the user was already looking at is untouched
                # (see the docstring: this refusal never writes
                # `ctx.state["plan"]`) and still `PENDING_APPROVAL`, so it
                # is a real thing to re-offer, not a stale claim -- route
                # back to `approve_n` with that same plan, carrying the
                # refusal's own sentence so the user reads why their
                # change did not land before seeing the unchanged plan
                # again.
                note = (
                    f"{text} Version {existing.version} is still there to "
                    "approve; anything past that needs a new question."
                )
                yield Event(
                    output={
                        **existing.model_dump(mode="json"),
                        "summary": note,
                        "revision": plan_revision(existing),
                        "refusal_note": note,
                        "depth": ctx.state.get("depth"),
                        "depth_notice": ctx.state.get("depth_notice"),
                    },
                    route="pending_refusal",
                )
                return
            if dropped:
                kept = [seed_name(seed) for seed in envelope.get("seeds") or []]
                text = unplanned_seeds_text(dropped, kept, text)
            if unreadable:
                text = f"{UNREADABLE_PLAN_UNPLANNED_NOTE} {text}".rstrip()
            yield Event(output={"text": text, "final": bool(refused.get("final"))}, route="refused")
            return
        planner_plan = PlannerPlan.model_validate(drafted["plan"])
        ctx.state["plan"] = planner_plan.model_dump_json()
        notes = [
            *([UNREADABLE_PLAN_NOTE] if unreadable else []),
            *([dropped_seeds_note(dropped)] if dropped else []),
        ]
        yield Event(
            output={
                **planner_plan.model_dump(mode="json"),
                "summary": drafted["summary"],
                "revision": plan_revision(planner_plan),
                "depth": ctx.state.get("depth"),
                "depth_notice": ctx.state.get("depth_notice"),
                **({"refusal_note": " ".join(notes)} if notes else {}),
                # The draft's version counts from 1 again, on a line of its
                # own: no earlier card of the turn is a version of this plan.
                **({"restarted": True} if unreadable else {}),
            },
            route="ok",
        )

    def approve(ctx: Context, node_input: dict[str, Any]) -> Iterator[Any]:
        """The approval gate, as a graph interrupt. Revision-bound.

        This is the only gate: nothing but `execute`, below, reaches the
        executor.

        `ctx.state["plan"]` always holds a `planner.plan.Plan`, since every
        `PlannerModel` seam drafts one. A plan is keyed on `plan_revision(plan)`,
        a content hash that changes with every amendment. A plan stored in a
        shape this release does not read (`_readable_plan`) has no revision
        for the answer to name: it routes `"amend"` with
        `{"unreadable_plan": True}`, and `planner` drafts the question from
        the start.

        The `"approved"` route records a `plan.approved` span (nesting under
        the turn span `harness.studio_state.begin_turn` opens) and a matching
        log line with the same facts: `depth`, `depth_reason` and `parts`
        come straight off `ctx.state`; `choice_points` and `degraded` come
        off the resolve envelope stored there; `plan_version` and `amended`
        come off the `Plan` just approved. This is what a query over
        compound-floored approvals answers with, since `depth`, `parts` and
        `choice_points` together say whether the plan a compound question
        reached was ever anything but a formality.
        """
        answer = (ctx.resume_inputs or {}).get(APPROVE_INTERRUPT)
        if not (ctx.state.get("awaiting_approve") and answer is not None):
            ctx.state["awaiting_approve"] = True
            yield _interrupt(
                RequestInput(
                    interrupt_id=APPROVE_INTERRUPT,
                    message="Approve this plan before anything is retrieved?",
                    payload=node_input,
                )
            )
            return
        ctx.state["awaiting_approve"] = False
        approved_answer = answer.get("approved") if isinstance(answer, dict) else None
        plan = _readable_plan(ctx, ctx.state["plan"])
        if plan is None:
            # The card the user answered names a plan this release does not
            # read, so the answer approves nothing and amends nothing.
            yield Event(output={"unreadable_plan": True}, route="amend")
            return
        key: Any = plan_revision(plan)
        if approved_answer == key:
            approved = plan.approve()
            ctx.state["plan"] = approved.model_dump_json()
            envelope = _state_json(ctx, "resolve_envelope", default={})
            depth = ctx.state.get("depth")
            depth_reason = ctx.state.get("depth_reason")
            parts = ctx.state.get("parts")
            choice_points = int(envelope.get("choice_points", 0))
            degraded = any(w.get("severity") == "degraded" for w in envelope.get("warnings", []))
            plan_version = approved.version
            amended = plan_version > 1
            record_span(
                "plan.approved",
                {
                    "dfo.depth": depth,
                    "dfo.depth_reason": depth_reason,
                    "dfo.parts": parts,
                    "dfo.choice_points": choice_points,
                    "dfo.degraded": degraded,
                    "dfo.plan_version": plan_version,
                    "dfo.amended": amended,
                },
            )
            logger.info(
                "plan approved: depth=%s depth_reason=%r parts=%s choice_points=%s "
                "degraded=%s plan_version=%s amended=%s",
                depth,
                depth_reason,
                parts,
                choice_points,
                degraded,
                plan_version,
                amended,
            )
            yield Event(output=approved.model_dump(mode="json"), route="approved")
            return
        yield Event(
            output={"plan": plan.model_dump(mode="json"), "answer": answer},
            route="amend",
        )

    async def execute(ctx: Context, node_input: Any) -> AsyncIterator[Any]:
        """Run the approved plan's compiled workflow through the tool surface.

        `execute_approved_plan` owns the run: per-step retry, the
        per-capability transport policy, and the checks that turn a step's
        own declared shape into a refusal. This node only reads what came
        back and maps it onto the graph's two outgoing edges.
        An `Accepted` run first aliases each terminal frame's provenance to
        the retrieval(s) that actually feed it, per frame
        (`_link_evidence_to_terminal_frames`), then routes "ok" with the
        terminal steps' frame refs (`compose` never sees an intermediate
        one, only `workflow.terminal_steps`). `MethodChoiceNeeded` and
        `RediscoveryNeeded` route "replan" while the budget allows and "ok"
        with `failed_step` once it is spent, the same as any other
        `Refused`. A stopped run's answer frames (`_answer_frames`) have
        their provenance linked as an accepted run's are, and the asks they
        compute are kept for the plan call (`_kept_asks`), so `plan_done`
        can list them whichever later run or planner refusal ends the call.
        The "ok" route carries those frames as final; the "replan" refusal
        names the asks they compute (`computed_asks`), since the new plan
        runs from the start.
        The exception is a plain `Refused` whose own `payload` is a
        `ToolFailure` `executor.classify.classify_refusal` calls
        `Disposition.STOP`: that is a dead end no replan reaches, so the
        turn closes right there with the refusal's own remedy, no replan
        spent, rather than spending the budget reproducing the identical
        failing call. `Refused(NOT_APPROVED)` and
        `Refused(STALE_REVISION)` mean the graph's own `approve` node let
        through a plan it should not have -- a replan here would loop on the
        same defect, so those raise instead. Any other outcome also corrects
        that step's own activity record, and every sibling's in
        `outcome.also_stopped`, in case a raw call had already succeeded
        before a post-call check refused it (`_mark_masked_step_failed`).

        The run goes on as its own task (`_start_relayed`) while this node
        relays each step's progress reports as their own events
        (`_step_progress_event`), one per report as it arrives. The task is
        awaited last, so its exceptions propagate here, and cancelled if the
        node exits first (a stop or a disconnect closing the stream).

        A plan stored in a shape this release does not read
        (`_readable_plan`) runs nothing and spends no replan: it routes
        `"replan"` with `{"unreadable_plan": True}`, and `planner` drafts the
        question from the start.
        """
        del node_input
        if tool_surface is None:
            raise RuntimeError(
                "the execute node has no tool surface: create_root_workflow(tool_surface=...) "
                "was not given one, so the approved plan cannot be run"
            )
        plan = _readable_plan(ctx, ctx.state["plan"])
        if plan is None:
            # Nothing to run, and no replan is spent on it.
            yield Event(output={"unreadable_plan": True}, route="replan")
            return
        tools = {step.id: step.tool.name for step in plan.workflow.steps} if plan.workflow else {}
        task, relay = _start_relayed(
            lambda on_progress: execute_approved_plan(
                plan,
                surface=tool_surface,
                max_concurrent_steps=max_concurrent_steps,
                fetch_series_stats=fetch_series_stats,
                on_progress=on_progress,
            )
        )
        try:
            async for report in relay.latest():
                yield _step_progress_event(report, tool=tools.get(report.step_id, ""))
            finished, outcome = await task
        finally:
            await _stop(task)
        ctx.state["plan"] = finished.model_dump_json()
        workflow = plan.workflow
        assert workflow is not None, "an approved plan always carries a workflow"
        if isinstance(outcome, Accepted):
            _link_evidence_to_terminal_frames(tool_surface, outcome, workflow=workflow)
            step_order = {record.step_id: index for index, record in enumerate(outcome.steps)}
            yield Event(
                output={
                    "frame_refs": [ref.model_dump() for ref in outcome.results.values()],
                    # Parallel to frame_refs above: which plan step (index +
                    # tool) produced each ref, and that it is final. compose
                    # reads `is_final_step` as stated here rather than
                    # inferring it from position -- `outcome.results` is
                    # keyed over `workflow.terminal_steps`
                    # (`executor.run._Run.execute`), so every entry here
                    # really is a real terminal step's, whatever its
                    # position among `outcome.steps`; a DAG can have more
                    # than one, and a later step's own index proves nothing
                    # about an earlier terminal step's finality.
                    "frame_ref_steps": [
                        {
                            "step": step_order[step_id],
                            "step_id": step_id,
                            "tool": outcome.steps[step_order[step_id]].tool.name,
                            "is_final_step": True,
                        }
                        for step_id in outcome.results
                    ],
                    "plan_version": finished.version,
                    "step_summaries": [step.digest for step in outcome.steps if step.digest],
                    "stat_var_totals": _answer_notes(tool_surface, outcome, plan=finished),
                },
                route="ok",
            )
            return
        _mark_masked_step_failed(tool_surface, outcome, tools=tools)
        if isinstance(outcome, Refused) and outcome.refusal in (
            RefusalKind.NOT_APPROVED,
            RefusalKind.STALE_REVISION,
        ):
            raise RuntimeError(
                "the approve node let a plan reach execute that execute_approved_plan refused "
                f"as {outcome.refusal.value}: {outcome.detail}"
            )
        # A dead end: no replan can fix this, so none is spent, and the
        # planner's re-proposal (which would reproduce this exact failing
        # call) never runs.
        dead_end = _is_dead_end(outcome)
        answers = _answer_frames(outcome)
        _link_evidence_to_terminal_frames(tool_surface, outcome, workflow=workflow)
        if finished.asks:
            ctx.state[ORCHESTRATOR_KEPT_ASKS_KEY] = _kept_asks(
                ctx.state.get(ORCHESTRATOR_KEPT_ASKS_KEY), tool_surface, outcome, plan=finished
            )
        replans = int(ctx.state.get("replans", 0))
        if not dead_end and replans < MAX_REPLANS:
            ctx.state["replans"] = replans + 1
            refusal = _replan_refusal(outcome)
            if finished.asks:
                refusal["computed_asks"] = _computed_asks(finished, answers)
            output: dict[str, Any] = {"refusal": refusal, "plan_version": finished.version}
            yield Event(output=output, route="replan")
            return
        completed = outcome.completed
        yield Event(
            output={
                "frame_refs": [step.frame_ref.model_dump() for step in completed],
                # Parallel to frame_refs above: which plan step (index, id,
                # tool) produced each ref, and whether it is final. A
                # terminal step that finished before the stop is final
                # (`_answer_frames`): its frame answers the asks it serves,
                # so `compose` and `plan_done` take it as they would on an
                # accepted run. Any other step's frame is an intermediate,
                # and `_admissible_intents` admits no card of it.
                "frame_ref_steps": [
                    {
                        "step": index,
                        "step_id": step.step_id,
                        "tool": step.tool.name,
                        "is_final_step": step.step_id in answers,
                    }
                    for index, step in enumerate(completed)
                ],
                "plan_version": finished.version,
                "step_summaries": [step.digest for step in completed if step.digest],
                "stat_var_totals": _answer_notes(tool_surface, outcome, plan=finished),
                "failed_step": _failed_step(outcome),
            },
            route="ok",
        )

    async def compose(ctx: Context, node_input: dict[str, Any]) -> Any:
        """The compose node. With a model seam: describe each produced frame
        (presentation-mcp `describe_frame` through the executor), let the
        model choose component intents, filter them through
        `_admissible_intents` (below), and write each survivor
        (`write_component`) -- a select `_admissible_intents` refused, a
        refused write (the platform's, or the executor withholding the
        card), or an `ok` outcome that carries no usable component, skips
        that intent, is logged and recorded on the trace, and is collected
        into `refused_writes`, so `narrate` can name the gap instead of the
        turn silently drawing a smaller canvas. Scripted
        pass-through when no seam is injected.

        A refused describe_frame becomes a NAMED gap
        (`{"available": False, "reason": ...}`), never a bare `{}` that a
        populated view for a different ref would outshine -- the model
        drafts with the hole rather than silently substituting another
        step's frame. `frame_provenance` tells it which ref is which plan
        step's output and whether it is the question's actual answer --
        `is_final_step`, stated by the producer (`fetch`/`execute`) on each
        `frame_ref_steps` entry, never inferred here from step position: a
        DAG can answer with more than one terminal step, and no single
        "last" index names all of them. The model is instructed to respect
        both facts, but an instruction is not enforcement: `_admissible_intents`
        re-checks both, in code, on every seam's output (the real prompt,
        a scripted stand-in, a test fake) before a single `write_component`
        call is made, and logs whatever it rejects.

        Every write that returns a component usable enough to satisfy the
        `component.created` wire contract (`harness.wire.usable_component`)
        is stamped `land=True` here, through the executor's optional
        `land_component` capability (`_RecordingExecutor.land_component`,
        `harness.studio_translate`), called with no `await` between the
        write returning and this call (the same event-loop step
        `/composition`'s lockless route cannot interleave with). The stamp
        also carries the turn's revise target, but only for the first
        usable component: a second one in the same turn is never checked
        against a target that is no longer its business. Neither the
        stamp nor this node ever mutates the canvas; that, and the
        real replace-or-append decision, happen later, at drain time
        (`harness.studio_translate`'s `_component_frames`), which runs
        before `narrate` in every case ADK's own event ordering allows.
        A component that fails validation is never stamped, never
        consumes the revise target, and is recorded in `refused_writes`
        instead. A bare executor with no `land_component` capability (a
        unit test with no store to bind) means no write this turn ever
        lands.
        """
        output = {
            "frame_refs": node_input.get("frame_refs", []),
            "plan_version": node_input.get("plan_version"),
            "depth": ctx.state.get("depth"),
            "step_summaries": node_input.get("step_summaries", []),
            "stat_var_totals": node_input.get("stat_var_totals", []),
            **({"failed_step": node_input["failed_step"]} if node_input.get("failed_step") else {}),
        }
        if models.compose is None:
            return output
        frame_refs = output["frame_refs"]
        frame_ref_steps = node_input.get("frame_ref_steps", [])
        if len(frame_ref_steps) != len(frame_refs):
            raise RuntimeError(
                f"compose got {len(frame_refs)} frame_refs but {len(frame_ref_steps)} "
                "frame_ref_steps entries; every producer (fetch, execute) supplies one "
                "frame_ref_steps entry, with is_final_step stated, per frame ref"
            )
        frame_provenance: list[dict[str, Any]] = []
        for entry in frame_ref_steps:
            if "is_final_step" not in entry:
                # `max(step)` cannot infer finality once a DAG's terminal
                # steps do not all share the highest index (see `execute`'s
                # Accepted branch), so a producer that omits `is_final_step`
                # is a construction defect, refused loudly rather than
                # papered over by defaulting every ref to final.
                raise RuntimeError(
                    f"frame_ref_steps entry {entry!r} states no is_final_step: the producer, "
                    "not compose, must say whether this ref is the plan's answer"
                )
            frame_provenance.append(
                {
                    "step": entry.get("step"),
                    "tool": entry.get("tool"),
                    "is_final_step": bool(entry["is_final_step"]),
                }
            )
        frame_views: list[dict[str, Any]] = []
        for ref in frame_refs:
            described = await executor(tool="describe_frame", args={"frame_ref": ref})
            if described.get("outcome") == "ok":
                frame_views.append({"available": True, "view": described.get("view") or {}})
            else:
                refusal = described.get("refusal") or {}
                reason = _frame_unavailable_reason(refusal)
                _log_refusal(
                    "describe_frame refused",
                    refusal,
                    spec_id=ref.get("spec_id"),
                    content_version=ref.get("content_version"),
                )
                _record_refusal_event(
                    "compose.describe_frame_refused",
                    refusal,
                    reason=reason,
                    spec_id=ref.get("spec_id"),
                    content_version=ref.get("content_version"),
                )
                frame_views.append({"available": False, "reason": reason})
        drafted = await models.compose(
            question=ctx.state.get("question", ""),
            frame_refs=frame_refs,
            frame_views=frame_views,
            frame_provenance=frame_provenance,
        )
        fact_summaries: list[dict[str, Any]] = []
        admissible, unbindable = _admissible_intents(
            drafted.get("intents", []),
            frame_refs=frame_refs,
            frame_views=frame_views,
            frame_provenance=frame_provenance,
        )
        refused_writes = [
            _refused_write(
                intent, refusal, summary="compose seam rejected an intent before write_component"
            )
            for intent, refusal in unbindable
        ]
        # The turn's revise target, stamped onto the first USABLE write's
        # record only. `_component_frames`, at drain time, is what
        # actually lands it (or not); this is only which target that write
        # gets to try against. A second usable component in the same turn
        # is never checked against a target that is no longer its
        # business, and an unusable write never sees it at all.
        revises_target = ctx.state.get("revises")
        stamp = getattr(executor, "land_component", None)
        for intent in admissible:
            written = await executor(tool="write_component", args={"intent": intent})
            component = written.get("component") if written.get("outcome") == "ok" else None
            if component is not None and usable_component(component):
                if stamp is not None:
                    stamp(component, land=True, revises=revises_target)
                revises_target = None
                fact_summaries.append(component["fact_summary"])
                continue
            # Either the platform refused the write (a real
            # `POST /api/presentation/components` rejection, surfaced to the
            # executor as `{"outcome": "refusal", "refusal": {...}}`), or the
            # executor withheld the card itself, or it answered `ok` with a
            # component that fails the `component.created` wire contract
            # (`harness.wire.usable_component`, a missing `fact_summary`
            # among the possible reasons); none of these leave anything to
            # draw a card from. Named here the same way `_admissible_intents`
            # names its own rejections: logged and recorded on the trace, and
            # carried forward so narrate can say a write did not land instead
            # of a chart quietly not appearing.
            if written.get("outcome") == "ok":
                refusal = _unusable_component_refusal(component)
            else:
                refusal = written.get("refusal") or {}
            # The record reaches narrate's facts and its scripted fallback text.
            refused_writes.append(
                _refused_write(
                    intent,
                    refusal,
                    summary="write_component did not produce a usable component",
                )
            )
        return {
            **output,
            "fact_summaries": fact_summaries,
            "frame_views": frame_views,
            "refused_writes": refused_writes,
        }

    async def narrate(ctx: Context, node_input: dict[str, Any]) -> Any:
        """The closing generative node: the turn's final text, written from
        bounded facts only (fact summaries and frame refs, never rows).
        Scripted one-liner when no seam is injected. Seeds the turn dropped
        that no earlier words named are told first (`_with_untold_drops`)."""
        refs = node_input.get("frame_refs", [])
        version = node_input.get("plan_version")
        if models.narrate is not None:
            fact_summaries = node_input.get("fact_summaries", [])
            # The executor's optional capability, probed the same way
            # `compose` probes `land_component`; see `Executor`'s
            # docstring. By the time this node runs, ADK has already
            # translated compose's event (`harness.studio_translate`'s
            # `_component_frames`), so every write this turn attempted to
            # land has already been decided.
            components_replaced = getattr(executor, "components_replaced", None) or (lambda: False)
            drafted = await models.narrate(
                question=ctx.state.get("question", ""),
                facts={
                    "frame_refs": refs,
                    "plan_version": version,
                    "fact_summaries": fact_summaries,
                    # each summary is one component that made it to the canvas;
                    # narrate must not claim a canvas when this is 0
                    "components_created": len(fact_summaries),
                    "step_summaries": node_input.get("step_summaries", []),
                    # one sentence per summed family the answer rests on:
                    # how many StatVars it totals and who set the reading,
                    # and one per step that left keys out of its result;
                    # appended by code below when the draft leaves it out
                    "stat_var_totals": node_input.get("stat_var_totals", []),
                    "frame_views": node_input.get("frame_views", []),
                    # set when a plan step could not run: narrate names it
                    "failed_step": node_input.get("failed_step"),
                    # one entry per compose intent that produced no card --
                    # a refused write (the platform's, or one the harness
                    # withheld) or a component missing its fact_summary;
                    # narrate names the reason rather than letting a smaller
                    # canvas pass without explanation
                    "refused_writes": node_input.get("refused_writes", []),
                    # True when this turn's question named a canvas card to
                    # change (converse's `revises`, still on state, since
                    # nothing after converse clears it). States the request,
                    # not the outcome: `/composition` can remove the named
                    # block mid-turn, so a true value here does not mean the
                    # swap landed. Pair with `components_replaced` for that.
                    "revises_requested": bool(ctx.state.get("revises")),
                    # True only when a component this turn actually swapped
                    # in for the card `revises_requested` named: the
                    # executor's own optional `components_replaced`
                    # capability, which reads the turn's actual landed
                    # actions, decided at drain time and always before this
                    # node runs, not this turn-start intent. False with
                    # components_created >= 1 means a new card was drawn
                    # instead, because the target vanished before the write
                    # landed, or no such capability exists. narrate must
                    # describe it as added, the same as any other new
                    # component, never as an update.
                    "components_replaced": bool(components_replaced()),
                },
            )
            text = _with_remedy(drafted.get("text", ""), node_input.get("failed_step"))
            text = _with_totals(text, node_input.get("stat_var_totals", []))
            return {"text": _with_untold_drops(ctx, text), **node_input}
        via = f"plan v{version}" if version is not None else "direct fetch"
        text = f"Prepared {len(refs)} frame(s) via {via}."
        failed = node_input.get("failed_step")
        if failed:
            named = f" ({failed['tool']})" if failed.get("tool") else ""
            text += (
                f" Step {failed['step_id']!r}{named} could not run:"
                f" {failed.get('code')} -- {failed.get('message')}"
            )
        for refused in node_input.get("refused_writes", []):
            title = f" {refused['title']!r}" if refused.get("title") else ""
            text += f" A component{title} was not added: {refused.get('reason')}."
        text = _with_totals(_with_remedy(text, failed), node_input.get("stat_var_totals", []))
        return {"text": _with_untold_drops(ctx, text), **node_input}

    # --- The orchestrator engine's own nodes (`orchestrator` given) ---------

    async def orchestrate(ctx: Context, node_input: Any) -> AsyncIterator[Event]:
        """The orchestrator's turn: the agent runs with its tools on its own
        branch, so it sees the whole turn so far -- the user's message, its
        own calls, every hand-back -- and none of the planner's internal
        events. Reached from START with the user's message (already in the
        session), and again after `ask` and `plan_done` with text to hand
        back, which goes in as a user message of its own so it is saved
        with the session and still in view after a later pause.

        The hand-back is authored `user` on the agent's branch: ADK quotes an
        event another author wrote as that agent's data, and an unbranched
        event reaches every agent, the planner on a later `plan` call
        included. The author is restored before the agent runs, since a
        child context inherits it. An event's explicit branch also becomes
        its node's branch, which ADK hands to every node downstream; the
        status event that follows the hand-back carries the empty branch,
        which ADK reads as "clear it", so this node's output and every later
        node of the turn stay unbranched, as they are before any hand-back."""
        assert orchestrator is not None
        if isinstance(node_input, dict):
            author, ctx.event_author = ctx.event_author, "user"
            try:
                yield Event(
                    content=types.Content(
                        role="user", parts=[types.Part.from_text(text=node_input["text"])]
                    ),
                    branch=orchestrator.name,
                )
            finally:
                ctx.event_author = author
            yield Event(custom_metadata={ORCHESTRATOR_STATUS_KEY: node_input["status"]}, branch="")
        else:
            yield Event(custom_metadata={ORCHESTRATOR_STATUS_KEY: "Reading your question"})
        answer = await ctx.run_node(orchestrator, None, override_branch=orchestrator.name)
        yield Event(output=answer)

    def route(ctx: Context, node_input: Any) -> Iterator[Event]:
        """The one action the agent asked for, or none: it answered."""
        del node_input
        pending = pending_action(ctx.state)
        ctx.state[ORCHESTRATOR_PENDING_KEY] = None
        if pending is None:
            yield Event(output=None, route="answered")
        else:
            yield Event(output=pending.model_dump(), route=pending.kind)

    def ask(ctx: Context, node_input: dict[str, Any]) -> Iterator[Any]:
        """Put the agent's question to the user through the clarify
        interrupt, its options as candidates under generated ids, and hand
        the answer back in the option's own words. The question waits
        under `ORCHESTRATOR_ASKED_KEY` while the interrupt is open, so a
        resume is told from a fresh entry the way `clarify` tells them."""
        answer = (ctx.resume_inputs or {}).get(CLARIFY_INTERRUPT)
        asked = ctx.state.get(ORCHESTRATOR_ASKED_KEY)
        if not (asked and answer is not None):
            ctx.state[ORCHESTRATOR_ASKED_KEY] = node_input
            yield _interrupt(
                RequestInput(
                    interrupt_id=CLARIFY_INTERRUPT,
                    message=node_input["question"],
                    payload={
                        "candidates": [
                            {"spec_id": f"o{index + 1}", "label": option}
                            for index, option in enumerate(node_input["options"])
                        ]
                    },
                )
            )
            return
        ctx.state[ORCHESTRATOR_ASKED_KEY] = None
        options = {f"o{index + 1}": option for index, option in enumerate(asked["options"])}
        if isinstance(answer, dict) and answer.get("skip"):
            reply = (
                "The user skipped the question. Do not answer it or search for it: close "
                "the turn in a sentence and ask what else you can help with."
            )
        elif isinstance(answer, dict) and answer.get("text"):
            reply = f"The user answered in their own words: {answer['text']}"
        else:
            chosen = answer.get("spec_id") if isinstance(answer, dict) else str(answer)
            reply = f"The user chose: {options.get(chosen, chosen)}"
        yield Event(
            output={
                "status": "Reading your answer",
                "text": f"You asked the user: {asked['question']}\n{reply}",
            }
        )

    def plan_start(ctx: Context, node_input: dict[str, Any]) -> dict[str, Any]:
        """Hand the agent's brief to the planner under the orchestrator's own
        key, leaving the graph's `question` alone. This call's asks go under
        `ASKS_STATE_KEY`, cleared when it lists none so no plan is held to an
        earlier call's asks, and no earlier call's kept asks are listed. The
        planner's first-draft entry is an empty payload."""
        ctx.state[ORCHESTRATOR_BRIEF_KEY] = node_input["brief"]
        ctx.state[ASKS_STATE_KEY] = node_input.get("asks") or None
        ctx.state[ORCHESTRATOR_KEPT_ASKS_KEY] = None
        return {}

    async def plan_done(ctx: Context, node_input: dict[str, Any]) -> dict[str, Any]:
        """What the agent reads once its plan has run, or could not be made:
        the final-step frames, described and registered under ids it can
        chart (never an intermediate step's, the rule `compose` applies; on
        a stopped plan, each terminal step that finished before the stop),
        the step summaries, the notes on how the figures were computed
        (`totals`, `_answer_notes`), what failed, each cut, and
        each ask the brief listed with its outputs' status or its decline
        (`_ask_status`), then the plan record
        (`harness.turn_records.PlanRecord`). A step refused
        `capability_unavailable` marks its spec_id unavailable to `fetch`
        for the rest of the turn and does not give the plan call back.

        When the plan call's last run stopped, or the planner refused after
        a run stopped, each ask that run did not compute but an earlier run
        of the call did is listed from that earlier run's kept frames
        (`ORCHESTRATOR_KEPT_ASKS_KEY`), marked `earlier_plan`, with their
        step summaries and those notes, and an ask that earlier run
        computed in part with the part not computed and its reason."""
        kept: dict[str, Any] = ctx.state.get(ORCHESTRATOR_KEPT_ASKS_KEY) or {}
        ctx.state[ORCHESTRATOR_KEPT_ASKS_KEY] = None
        described: list[tuple[dict[str, Any], dict[str, Any]]] = []
        not_chartable: list[dict[str, Any]] = []
        summaries: list[str] = list(node_input.get("step_summaries", []))
        totals: list[str] = list(node_input.get("stat_var_totals", []))
        frames: dict[str, Any] = dict(ctx.state.get(ORCHESTRATOR_FRAMES_KEY) or {})

        async def describe(ref: dict[str, Any], *, tool: str) -> dict[str, Any]:
            """Describe one final frame for listing, and return its status as
            `_ask_status` reports it."""
            view = await executor(tool="describe_frame", args={"frame_ref": ref})
            if view.get("outcome") == "ok":
                described.append((ref, view.get("view") or {}))
                empty = (view.get("view") or {}).get("series_count") == 0
                return {"status": "empty" if empty else "data", "frame": frame_id_for(ref)}
            # The same closed vocabulary `compose` hands narrate: never the
            # platform's own text.
            reason = _frame_unavailable_reason(view.get("refusal") or {})
            not_chartable.append({"step": tool, "reason": reason})
            return {"status": "not_chartable", "reason": reason}

        async def earlier(number: int, ask: str) -> dict[str, Any]:
            """Ask `number` as an earlier run of this plan call computed it,
            from its kept frames, adding their summaries and notes."""
            entry = kept[str(number)]
            summaries.extend(step["digest"] for step in entry["steps"] if step["digest"])
            totals.extend(total for total in entry["totals"] if total not in totals)
            status: dict[str, Any] = {
                "ask": ask,
                "answered_by": [
                    {
                        "step": step["step_id"],
                        **await describe(step["frame_ref"], tool=step["tool"]),
                    }
                    for step in entry["steps"]
                ],
            }
            if "not_computed" in entry:
                status["not_computed"] = entry["not_computed"]
            status["earlier_plan"] = True
            return status

        def listing(asks: list[dict[str, Any]] | None) -> dict[str, Any]:
            """The frames, summaries, computation notes (`totals`) and asks
            the agent reads, the described frames registered under their ids."""
            result: dict[str, Any] = {
                "frames": register_frames(frames, described),
                "step_summaries": [
                    _without_noise(str(summary))[:_HAND_BACK_CAP] for summary in summaries
                ],
                "totals": totals,
            }
            if not_chartable:
                result["frames_not_chartable"] = not_chartable
            if asks is not None:
                result["asks"] = asks
            ctx.state[ORCHESTRATOR_FRAMES_KEY] = frames
            return result

        if "text" in node_input:
            # A final refusal (an internal or repeated failure, or a question
            # declined whole) fails the same way for any brief.
            again = not node_input.get("final") and _give_plan_back(ctx)
            computed = listed = ""
            if kept:
                brief_asks = load_asks(ctx.state)
                result = listing(
                    [
                        await earlier(int(number), brief_asks[int(number)])
                        for number in sorted(kept, key=int)
                    ]
                )
                computed = (
                    f"{_partial_result(result['asks'], frames=bool(result['frames']))} "
                    "Every other ask was not computed. "
                )
                listed = f"\n{json.dumps(result)}"
            return {
                "status": ORCHESTRATOR_PLAN_FAILED_STATUS,
                "text": "The planner could not make a plan: "
                f"{str(node_input['text'])[:_HAND_BACK_CAP]}\n{computed}Answer with search "
                "and fetch where you can, and say plainly what could not be calculated."
                + (" A brief that avoids the problem may be planned once more." if again else "")
                + listed,
            }
        # Final step id -> its output's status, as `_ask_status` reports it.
        ran: dict[str, dict[str, Any]] = {}
        steps = node_input.get("frame_ref_steps", [])
        for ref, step in zip(node_input.get("frame_refs", []), steps, strict=True):
            if step.get("is_final_step"):
                ran[step.get("step_id", "")] = await describe(ref, tool=step.get("tool", ""))
        asks: list[dict[str, Any]] | None = None
        failed = node_input.get("failed_step")
        missing_capability = bool(
            failed and failed.get("code") == ToolErrorCode.CAPABILITY_UNAVAILABLE.value
        )
        record = ""
        plan = _readable_plan(ctx, ctx.state.get("plan"))
        if plan is not None:
            # The plan record names every source by spec_id, so the agent can
            # fetch a part the failure did not touch without searching again,
            # and the readings let `fetch` read such a spec the way the plan
            # did.
            readings = dict(ctx.state.get(ORCHESTRATOR_READINGS_KEY) or {})
            unavailable = dict(ctx.state.get(ORCHESTRATOR_UNAVAILABLE_KEY) or {})
            for row in plan_display.step_rows(plan):
                args = cast("dict[str, Any]", row["args"])
                if args.get("spec_id") and args.get("stat_vars"):
                    readings[args["spec_id"]] = args["stat_vars"]
                if (
                    missing_capability
                    and args.get("spec_id")
                    and row["id"] == failed.get("step_id")
                ):
                    unavailable[args["spec_id"]] = str(failed.get("message") or "")
            ctx.state[ORCHESTRATOR_READINGS_KEY] = readings
            ctx.state[ORCHESTRATOR_UNAVAILABLE_KEY] = unavailable
            plan_record = PlanRecord.from_plan(
                plan,
                # A source an earlier turn's search returned is a searched
                # one too, and keeps its block in that turn's record.
                searched={
                    *(ctx.state.get(ORCHESTRATOR_SPECS_KEY) or {}),
                    *(ctx.state.get(ORCHESTRATOR_PRIOR_LEDGER_KEY) or {}).get("spec_ids", []),
                },
                seeded={seeded.source.spec_id for seeded in seeded_sources(ctx.state)},
                failed_step=failed.get("step_id") if failed else None,
            )
            ctx.state[ORCHESTRATOR_PLAN_RECORDS_KEY] = [
                *(ctx.state.get(ORCHESTRATOR_PLAN_RECORDS_KEY) or []),
                plan_record.model_dump(mode="json"),
            ]
            record = "\n" + plan_record.render(budget=_PLAN_RECORD_CHARS)
            if plan_record.asks:
                stopped = _stopped_steps(plan.workflow, failed.get("step_id") if failed else None)
                asks = [_ask_status(ask, ran, failed, stopped) for ask in plan_record.asks]
        if failed and asks:
            asks = [
                await earlier(number, status["ask"])
                if str(number) in kept and not _computed(status)
                else status
                for number, status in enumerate(asks)
            ]
        result = listing(asks)
        if failed:
            result["failed_step"] = {
                key: str(failed[key])[:_HAND_BACK_CAP]
                for key in ("step_id", "tool", "code", "message", "remedy")
                if failed.get(key)
            }
            partial = _partial_result(result.get("asks"), frames=bool(result["frames"]))
            if missing_capability:
                # No brief avoids a capability the question needs, so the
                # plan call is not given back.
                lead = (
                    "The approved plan stopped at the step marked failed: this deployment "
                    "cannot do what that step needs, so the part of the question it serves "
                    "is unavailable. Its data fetched without the refused option, or a chart "
                    "of other data, does not answer it. Say that this part is unavailable "
                    f"and why, in the user's terms. {partial}"
                )
            else:
                lead = (
                    f"The approved plan stopped at the step marked failed. {partial} Some "
                    "other parts of the question may still be retrievable on their own with "
                    "fetch, by the spec_id a step names: show what can be shown, labelled as "
                    "what it is, never as the derived result; do not repeat the failed "
                    "request unchanged; and say plainly what remains unanswered and why, in "
                    "the user's terms."
                )
                if _give_plan_back(ctx):
                    lead += " A brief that avoids the failure may be planned once more."
        elif not_chartable and not described:
            lead = (
                "The approved plan ran, but its result cannot be drawn: tell the user what "
                "the plan computed, in words, from the step summaries, and that the chart "
                "could not be prepared. Only a frame listed under frames can be charted."
            )
        else:
            lead = "The approved plan ran. Chart its frames and answer."
        return {
            "status": ORCHESTRATOR_PLAN_RAN_STATUS,
            "text": f"{lead}\n{json.dumps(result)}{record}",
        }

    converse_n = node(converse, name="converse")
    reply_n = node(reply, name="reply")
    canvas_answer_n = node(canvas_answer, name="canvas_answer")
    seed_n = node(seed, name="seed")
    probe_n = node(probe, name="probe")
    route_status_n = node(route_status, name="route_status")
    clarify_n = node(clarify, name="clarify", rerun_on_resume=True)
    deadend_n = node(deadend, name="deadend")
    choose_depth_n = node(choose_depth, name="choose_depth")
    fetch_n = node(fetch, name="fetch")
    # `rerun_on_resume=True`: a real `PlannerModel` seam
    # (`harness.planner_seam.PlannerSeam`) calls `ctx.run_node` from inside
    # this node, and ADK raises `ValueError` at that call unless the
    # *calling* node itself carries this flag (`agents/context.py`'s
    # `_run_node_internal`), matching `clarify_n`/`approve_n`'s own
    # convention.
    planner_n = node(planner, name="planner", rerun_on_resume=True)
    approve_n = node(approve, name="approve", rerun_on_resume=True)
    execute_n = node(execute, name="execute")
    compose_n = node(compose, name="compose")
    narrate_n = node(narrate, name="narrate")

    # The plan path's own edges, shared by both engines: the approval gate,
    # amendments and the replan loop.
    plan_edges = [
        Edge(from_node=planner_n, to_node=approve_n, route=["ok", "pending_refusal"]),
        Edge(from_node=approve_n, to_node=execute_n, route="approved"),
        Edge(from_node=approve_n, to_node=planner_n, route="amend"),
        Edge(from_node=execute_n, to_node=planner_n, route="replan"),
    ]
    if orchestrator is not None:
        orchestrate_n = node(orchestrate, name="orchestrate", rerun_on_resume=True)
        route_n = node(route, name="route")
        ask_n = node(ask, name="ask", rerun_on_resume=True)
        plan_start_n = node(plan_start, name="plan_start")
        plan_done_n = node(plan_done, name="plan_done")
        return Workflow(
            name=AGENT_NAME,
            state_schema=GraphState,
            edges=[
                (START, orchestrate_n),
                (orchestrate_n, route_n),
                Edge(from_node=route_n, to_node=ask_n, route="ask"),
                (ask_n, orchestrate_n),
                Edge(from_node=route_n, to_node=plan_start_n, route="plan"),
                (plan_start_n, planner_n),
                Edge(from_node=planner_n, to_node=plan_done_n, route="refused"),
                Edge(from_node=execute_n, to_node=plan_done_n, route="ok"),
                (plan_done_n, orchestrate_n),
                *plan_edges,
            ],
        )
    return Workflow(
        name=AGENT_NAME,
        state_schema=GraphState,
        edges=[
            (START, converse_n),
            Edge(from_node=converse_n, to_node=probe_n, route="data"),
            Edge(from_node=converse_n, to_node=reply_n, route="reply"),
            Edge(from_node=converse_n, to_node=seed_n, route="seeded"),
            Edge(from_node=seed_n, to_node=route_status_n, route="seeded"),
            Edge(from_node=seed_n, to_node=probe_n, route="fallback"),
            Edge(from_node=converse_n, to_node=canvas_answer_n, route="canvas"),
            Edge(from_node=converse_n, to_node=deadend_n, route="deadend"),
            (probe_n, route_status_n),
            Edge(from_node=route_status_n, to_node=choose_depth_n, route="definite"),
            Edge(from_node=route_status_n, to_node=clarify_n, route=["candidates", "no_data"]),
            Edge(from_node=route_status_n, to_node=deadend_n, route="unsupported"),
            Edge(from_node=clarify_n, to_node=probe_n, route="again"),
            Edge(from_node=clarify_n, to_node=converse_n, route="refine"),
            Edge(from_node=choose_depth_n, to_node=fetch_n, route="none"),
            Edge(
                from_node=choose_depth_n,
                to_node=planner_n,
                route=["shallow", "standard", "deep"],
            ),
            Edge(from_node=planner_n, to_node=deadend_n, route="refused"),
            Edge(from_node=execute_n, to_node=compose_n, route="ok"),
            (fetch_n, compose_n),
            (compose_n, narrate_n),
            *plan_edges,
        ],
    )


# The public name of the select check `make_charts` runs
# (`harness.orchestrator_tools`); the graph's compose node uses it as well.
select_binding_ok = _select_binding_ok


__all__ = [
    "AGENT_NAME",
    "APPROVE_INTERRUPT",
    "CLARIFY_INTERRUPT",
    "LINE_CHART_MAX_VALUES",
    "MAX_CLARIFY_ROUNDS",
    "MAX_REPLANS",
    "MAX_SERIES_PER_COMPONENT",
    "NO_OPERATIONS_ON_DOWNLOAD",
    "ONE_DATASET_PER_DOWNLOAD",
    "ORCHESTRATOR_BRIEF_KEY",
    "ORCHESTRATOR_CALLS_KEY",
    "ORCHESTRATOR_FRAMES_KEY",
    "ORCHESTRATOR_PENDING_KEY",
    "ORCHESTRATOR_PLAN_FAILED_STATUS",
    "ORCHESTRATOR_PLAN_RAN_STATUS",
    "ORCHESTRATOR_READINGS_KEY",
    "ORCHESTRATOR_SPECS_KEY",
    "ORCHESTRATOR_STATUS_KEY",
    "SEEDS_STEP_ID",
    "SEED_DUPLICATE",
    "SEED_REFERENCE_UNUSED",
    "SEED_STEP_PREFIX",
    "SEED_STEP_TOOL",
    "ComposeModel",
    "ConverseModel",
    "Executor",
    "GenerativeNodes",
    "GraphState",
    "NarrateModel",
    "PendingAsk",
    "PendingPlan",
    "PlannerModel",
    "Resolver",
    "create_root_workflow",
    "frame_id_for",
    "frame_row_ceiling",
    "pending_action",
    "register_frames",
    "select_binding_ok",
]
