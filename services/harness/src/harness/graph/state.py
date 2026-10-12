"""The orchestrator workflow's session-state contract.

The keys and statuses the workflow's nodes and the orchestrator's tools
(`harness.orchestrator_tools`) read and write, the session-state schema
(`GraphState`), the one pending action the agent leaves for `route`, the
tools' budgets, and the helpers more than one node reads: the stored plan
(`readable_plan`), a JSON slot (`state_json`) and the interrupt event
(`interrupt_event`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import (
    Mapping,
    Sequence,
)
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from data_mcp.search_tool import MAX_PARTS
from google.adk import Context, Event
from google.adk.events import RequestInput
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_event
from planner.plan import Plan as PlannerPlan
from planner.plan import (
    SessionState,
    stored_plan,
)
from pydantic import BaseModel, BeforeValidator, Field, TypeAdapter

from harness.tool_contracts import (
    reading_facts,
)
from harness.turn_records import Origin, SearchReading

# The ADK app name every stored conversation's sessions are keyed under.
AGENT_NAME = "dfo_analyst_graph"

# Interrupt ids are stable strings: ADK matches a resuming function response
# to its interrupt by id, and `ask` and `approve` reuse their id across rounds.
CLARIFY_INTERRUPT = "clarify"
APPROVE_INTERRUPT = "approve"


# The `custom_metadata` key a relayed step progress report travels under on
# `execute`'s events (`_step_progress_event`), read by
# `harness.studio_translate._Translator.translate`.
STEP_PROGRESS_KEY = "step_progress"

# The `custom_metadata` key the `planner` node's first event travels under,
# as it starts a draft: `{"revising": bool}`, true when the draft revises a
# plan already on record. Read by `harness.studio_translate._Translator`.
PLANNER_STARTED_KEY = "planner_started"

# The orchestrator's session state (`create_root_workflow`'s
# `orchestrator`), written by its tools (`harness.orchestrator_tools`) and
# read by the workflow's nodes. Prefixed, so `GraphState` never validates it.
ORCHESTRATOR_FRAMES_KEY = "orchestrator:frames"  # frame id -> {"ref", "view"}
ORCHESTRATOR_SPECS_KEY = "orchestrator:specs"  # spec_id -> its search spec block
ORCHESTRATOR_PENDING_KEY = "orchestrator:pending"  # the turn's one pending action
ORCHESTRATOR_CALLS_KEY = "orchestrator:calls"  # tool name -> calls this turn
ORCHESTRATOR_BRIEF_KEY = "orchestrator:brief"  # the question the planner drafts for
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
# prior_ledger_state`; unset when no earlier turn read any. The blocks
# themselves resolve through `planner.plan.earlier_block`.
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
# under ("Reading your question"), read by `studio_translate._Translator`.
ORCHESTRATOR_STATUS_KEY = "orchestrator_status"
# The status a plan hand-back carries when the planner made no plan, and
# when the approved plan ran; `studio_translate._Translator` shows the first as
# a failed step and the second as nothing.
ORCHESTRATOR_PLAN_FAILED_STATUS = "Reading the planner's answer"
ORCHESTRATOR_PLAN_RAN_STATUS = "Reading the plan's results"


# The branch the approval and `ask` interrupts are stamped with. No agent runs
# on it or under it, so ADK's branch filter keeps the interrupt's payload out
# of every model request; the browser reads the interrupt off the event stream.
_INTERRUPT_BRANCH = "dfo_interrupts"


def interrupt_event(request: RequestInput) -> Event:
    """`request` as its interrupt event, on `_INTERRUPT_BRANCH`."""
    event = create_request_input_event(request)
    event.branch = _INTERRUPT_BRANCH
    return event


class AskOption(BaseModel):
    """One answer `ask_user` offers: its wording and, for a count, its spec_id."""

    label: str = Field(description="the answer as the user reads it")
    spec_id: str | None = Field(
        default=None,
        description=(
            "for a way of counting the result names, the spec_id it gives that count; "
            "leave it out for any other answer"
        ),
    )


def ask_options(options: list[Any]) -> list[dict[str, Any]]:
    """Persisted `ask_user` options as `AskOption` dicts. A question paused
    before options carried a spec_id stored each option as its bare label,
    which reads here as an option without one."""
    return [{"label": option} if isinstance(option, str) else option for option in options]


class PendingAsk(BaseModel):
    """`ask_user` asked: the workflow pauses with the question."""

    kind: Literal["ask"] = "ask"
    question: str
    options: Annotated[list[AskOption], BeforeValidator(ask_options)]


class PendingPlan(BaseModel):
    """`plan` asked: the workflow runs the plan path on the brief, holding the
    plan to `asks`, the outputs the user asked for."""

    kind: Literal["plan"] = "plan"
    brief: str
    asks: list[str] = []


def ask_candidates(options: list[dict[str, Any]]) -> list[dict[str, str]]:
    """`PendingAsk.options` as the clarify interrupt's candidates: a count under
    its own spec_id, so a pick returns it, and any other answer under `o{n}`,
    its place in the list."""
    return [
        {"spec_id": option.get("spec_id") or f"o{index + 1}", "label": option["label"]}
        for index, option in enumerate(ask_options(options))
    ]


_PENDING_ACTION = TypeAdapter(Annotated[PendingAsk | PendingPlan, Field(discriminator="kind")])


# Calls one turn may make of each of the orchestrator's tools
# (`harness.orchestrator_tools`). `search` counts every resolver call, `fetch`
# every dataset read; a plan is one per turn, and a plan that failed gives it
# back (`plan_done`) for a brief that avoids the failure. `make_charts` counts
# the calls that drew at least one chart, `make_charts_failed` the calls that
# drew none.
# `invalid_arguments` counts the calls of any tool refused for malformed
# arguments, which spend nothing of the tool's own budget.
BUDGETS = {
    # The first search, a corrected place, and one raw-text search for each
    # part the first left unanswered.
    "search": MAX_PARTS + 2,
    "fetch": 6,
    "make_charts": 5,
    "make_charts_failed": 3,
    "ask_user": 2,
    "plan": 1,
    "get_skill": 2,
    # A file a skill links, or the next page of a long one.
    "get_skill_file": 4,
    "recall": 3,
    # recall(part="canvas") with no component: the canvas listing.
    "recall_canvas": 2,
    "remove_cards": 3,
    "invalid_arguments": 3,
}
# The download surface fetches one dataset and never plans.
DOWNLOAD_BUDGETS = {**BUDGETS, "fetch": 1, "plan": 0}


def turn_budgets(state: Any) -> dict[str, int]:
    """The calls this turn may make of each orchestrator tool: `DOWNLOAD_BUDGETS`
    on the download surface, `BUDGETS` otherwise. `state` is any mapping-like
    read of session state (`Context.state`, a dict)."""
    return DOWNLOAD_BUDGETS if state.get("surface") == "download" else BUDGETS


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


def store_search(
    state: SessionState,
    digest: Mapping[str, Any],
    *,
    asked: Sequence[Sequence[str]] = (),
    origin: Origin = "search",
) -> list[dict[str, Any]]:
    """Store what a `resolver_digest` read in the orchestrator's session state,
    and return its parts as the model reads them, without their blocks.

    Each part's spec blocks, its own and its candidates', go under
    `ORCHESTRATOR_SPECS_KEY` and the part's warnings under
    `ORCHESTRATOR_SPEC_WARNINGS_KEY`, where `fetch` finds them, and each part
    joins `ORCHESTRATOR_SEARCHES_KEY` as a `SearchReading` from `origin`.
    `asked` holds each part's asked flow parties by position, and is empty
    when no part's are known.
    """
    specs = dict(state.get(ORCHESTRATOR_SPECS_KEY) or {})
    spec_warnings = dict(state.get(ORCHESTRATOR_SPEC_WARNINGS_KEY) or {})
    searches = list(state.get(ORCHESTRATOR_SEARCHES_KEY) or [])
    skill_ids = [skill["id"] for skill in digest.get("skills") or []]
    observed_at = datetime.now(UTC)
    readings = []
    for index, part in enumerate(digest.get("part_digests") or []):
        reading = {
            key: value
            for key, value in part.items()
            if key not in ("spec_block", "candidate_blocks")
        }
        blocks = dict(part.get("candidate_blocks") or {})
        facts: dict[str, Any] = {}
        if part.get("spec_block") and part.get("spec_id"):
            blocks[part["spec_id"]] = part["spec_block"]
            facts = reading_facts(part["spec_block"])
        if mentions := reading.get("resolved_sources"):
            blocks = {
                spec_id: {**block, "_resolved_sources": mentions}
                for spec_id, block in blocks.items()
            }
        specs.update(blocks)
        spec_warnings.update(dict.fromkeys(blocks, reading.get("warnings") or []))
        if asked:
            facts["asked_roles"] = list(asked[index])
        stored = SearchReading.from_part(
            {**reading, **facts}, skills=skill_ids, origin=origin, observed_at=observed_at
        )
        searches.append(stored.model_dump(mode="json"))
        readings.append(reading)
    state[ORCHESTRATOR_SPECS_KEY] = specs
    state[ORCHESTRATOR_SPEC_WARNINGS_KEY] = spec_warnings
    state[ORCHESTRATOR_SEARCHES_KEY] = searches
    return readings


def state_json(ctx: Context, key: str, default: Any = None) -> Any:
    """Decode a JSON blob out of session state, `default` when absent."""
    raw = ctx.state.get(key)
    return json.loads(raw) if raw else default


class GraphState(BaseModel):
    """Session-state schema for the workflow (validated on every delta)."""

    surface: str | None = None
    canvas_digest: str | None = None
    conversation_digest: str | None = None
    # The Search results a "... in Studio" action handed over, as JSON: each
    # seed as `seeds_slot` keeps it, or the one reference `TurnRequest.seed_spec`
    # also takes. Host-seeded fresh per turn under `planner.plan.SEEDS_STATE_KEY`;
    # `seed` reads it once and clears it.
    seeds: str | None = None
    awaiting_approve: bool = False
    plan: str | None = None
    replans: int = 0


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


def readable_plan(ctx: Context, raw: str | None) -> PlannerPlan | None:
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
