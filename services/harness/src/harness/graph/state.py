"""The orchestrator workflow's session-state contract.

The keys and statuses the workflow's nodes and the orchestrator's tools
(`harness.orchestrator_tools`) read and write, the session-state schema
(`GraphState`), the one pending action the agent leaves for `route`, the
tools' budgets, and the helpers more than one node reads: the stored plan
(`readable_plan`), a JSON slot (`state_json`), the interrupt event
(`interrupt_event`) and the answers to a clarify interrupt (`resume_answers`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import (
    Mapping,
    Sequence,
)
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, NamedTuple

from data_mcp.search_tool import MAX_PARTS
from google.adk import Context, Event
from google.adk.events import RequestInput
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_event
from planner.plan import AskBreakdown, SessionState, stored_plan
from planner.plan import Plan as PlannerPlan
from pydantic import BaseModel, BeforeValidator, Field, TypeAdapter, model_validator

from harness.tool_contracts import (
    reading_facts,
)
from harness.turn_records import Origin, SearchReading, open_prices

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
ORCHESTRATOR_FRAMES_KEY = "orchestrator:frames"  # frame id -> {"ref", "view", "store_ref"}
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
# The chip a turn applies, as `{"choice_id", "option_id"}` JSON
# (`harness.wire.ApplyChoice`), seeded by `harness.studio` for a turn
# requested with `apply_choice`; `seed` applies it
# (`harness.graph.switch.apply_stored_choice`) and `turn_budgets` reads it.
APPLY_CHOICE_STATE_KEY = "apply_choice"
# The chip `seed` applied this turn: `{"choice_id", "option_id",
# "from_option_id", "spec_id"}`, the option it moved from and the spec_id
# Search compiled. `harness.studio` keeps it with the turn's record
# (`ConversationState.turn_applied_choice`), where the next turn reads whether
# the user kept it.
ORCHESTRATOR_APPLIED_KEY = "orchestrator:applied_choice"
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
ORCHESTRATOR_PLAN_FAILED_STATUS = "Reading why no plan was made"
# The `custom_metadata` key under which the planner's refusal travels beside
# that status. It holds `{"code"}`, which selects the failed step's sentence.
ORCHESTRATOR_STATUS_FAILURE_KEY = "orchestrator_status_failure"
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
    """One answer `ask_user` offers: its wording and, for a count or prices, its spec_id."""

    label: str = Field(description="the answer as the user reads it")
    spec_id: str | None = Field(
        default=None,
        description=(
            "for a count or prices the result names, the spec_id it gives that reading; "
            "leave it out for any other answer"
        ),
    )


def ask_options(options: list[Any]) -> list[dict[str, Any]]:
    """Persisted `ask_user` options as `AskOption` dicts. A question paused
    before options carried a spec_id stored each option as its bare label,
    which reads here as an option without one."""
    return [{"label": option} if isinstance(option, str) else option for option in options]


class AskQuestion(BaseModel):
    """One question `ask_user` puts to the user, with the answers it offers."""

    question: str = Field(description="the question as the user reads it")
    options: Annotated[list[AskOption], BeforeValidator(ask_options)] = Field(
        description="the likely answers, at least one; the user may also write their own"
    )


class PendingAsk(BaseModel):
    """`ask_user` asked: the workflow pauses with its questions, in order."""

    kind: Literal["ask"] = "ask"
    questions: list[AskQuestion]

    @model_validator(mode="before")
    @classmethod
    def _one_question(cls, data: Any) -> Any:
        """A stored single-question action, `{kind, question, options}`, reads
        as a call with that one question."""
        if isinstance(data, dict) and "questions" not in data and "question" in data:
            return {
                "kind": data.get("kind", "ask"),
                "questions": [{"question": data["question"], "options": data.get("options")}],
            }
        return data


class PendingPlan(BaseModel):
    """`plan` asked: the workflow runs the plan path on the brief, holding the
    plan to `asks`, the outputs the user asked for, to `ask_breakdowns`, the
    parties those asks break down by, and to `asks_across_years`, the numbers
    of the asks put to one figure over several years."""

    kind: Literal["plan"] = "plan"
    brief: str
    asks: list[str] = []
    ask_breakdowns: list[AskBreakdown] = []
    asks_across_years: list[int] = []


class PricedOption(NamedTuple):
    """The prices an `ask_user` option offers, as code reads them off the turn's
    search results (`harness.graph.orchestrate.priced_options`): `part`, the
    spec_id of the reading whose `open_prices` the option is one of, and
    `prices`, `current` or `constant`."""

    part: str
    prices: Literal["current", "constant"]


def ask_candidates(
    options: list[AskOption], priced: Mapping[int, PricedOption] | None = None
) -> list[dict[str, str]]:
    """One question's options as the clarify interrupt's candidates: a count
    under its own spec_id, so a pick returns it, and any other answer under
    `o{n}`, its place in the question's list. An option of `priced`, by
    place, carries its `part` and `prices`, which the planner reads off the
    answered interrupt (`planner.agent._prices_chosen`)."""
    priced = priced or {}
    return [
        {
            "spec_id": option.spec_id or f"o{index + 1}",
            "label": option.label,
            **(priced[index]._asdict() if index in priced else {}),
        }
        for index, option in enumerate(options)
    ]


def ask_questions(
    questions: list[AskQuestion], priced: Sequence[Mapping[int, PricedOption]] = ()
) -> list[dict[str, Any]]:
    """`PendingAsk.questions` as the clarify interrupt's payload entries: each
    under `q{n}`, its place in the call, with its candidates (`ask_candidates`),
    each question's priced options read from `priced` by place."""
    return [
        {
            "id": f"q{index + 1}",
            "question": question.question,
            "candidates": ask_candidates(
                question.options, priced[index] if index < len(priced) else None
            ),
        }
        for index, question in enumerate(questions)
    ]


def resume_answers(result: Any) -> dict[str, dict[str, Any]] | None:
    """The clarify interrupt's resume value as one answer per question id, or
    None when the user dismissed the whole pause (`{"skip": True}`).

    The value is `{"answers": {question id: answer}}`, each answer
    `{"kind": "option", "spec_id"}`, `{"kind": "text", "text"}` or
    `{"kind": "skipped"}`. A single-question reply is the answer for `q1`:
    `{"spec_id"}` for a pick, `{"text"}` for the user's own words, or a bare
    candidate id.
    """
    if not isinstance(result, dict):
        return {"q1": {"kind": "option", "spec_id": str(result)}}
    if result.get("skip"):
        return None
    if "answers" in result:
        return dict(result["answers"] or {})
    if result.get("text"):
        return {"q1": {"kind": "text", "text": result["text"]}}
    if result.get("spec_id"):
        return {"q1": {"kind": "option", "spec_id": result["spec_id"]}}
    return {"q1": {"kind": "skipped"}}


_PENDING_ACTION = TypeAdapter(Annotated[PendingAsk | PendingPlan, Field(discriminator="kind")])


# Calls one turn may make of each of the orchestrator's tools
# (`harness.orchestrator_tools`). `search` counts every resolver call, `fetch`
# every dataset read; a plan is one per turn, and a plan that failed gives it
# back (`plan_done`) for a brief that avoids the failure. `make_charts` counts
# the calls that drew at least one chart, `make_charts_failed` the calls that
# drew none since a frame last reached the turn. A registration that adds a
# frame id clears it (`register_frames`). Each clear costs a spent `fetch` or
# a plan hand-back (at most 2 with the plan given back), so a turn makes at
# most 3 x (1 + 6 + 2) = 27 calls that draw nothing.
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
# A turn applying a chip answers from the data `seed` fetched for it, so it
# neither searches nor plans.
APPLY_CHOICE_BUDGETS = {**BUDGETS, "search": 0, "plan": 0}


def turn_budgets(state: Any) -> dict[str, int]:
    """The calls this turn may make of each orchestrator tool:
    `APPLY_CHOICE_BUDGETS` on a turn applying a chip
    (`APPLY_CHOICE_STATE_KEY`), `DOWNLOAD_BUDGETS` on the download surface,
    `BUDGETS` otherwise. `state` is any mapping-like read of session state
    (`Context.state`, a dict)."""
    if state.get(APPLY_CHOICE_STATE_KEY):
        return APPLY_CHOICE_BUDGETS
    return DOWNLOAD_BUDGETS if state.get("surface") == "download" else BUDGETS


def count_call(state: Any, tool: str) -> None:
    """Count one more of `tool`'s calls this turn (`ORCHESTRATOR_CALLS_KEY`),
    reading the counts afresh. Synchronous, so no other tool call runs
    between a check of a budget and the change."""
    calls = dict(state.get(ORCHESTRATOR_CALLS_KEY) or {})
    calls[tool] = int(calls.get(tool, 0)) + 1
    state[ORCHESTRATOR_CALLS_KEY] = calls


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


type DescribedFrame = tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]


def register_frames(
    state: SessionState,
    described: Sequence[DescribedFrame],
    *,
    fetch_args: dict[str, Any] | None = None,
    wholes: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Add each (frame ref, `describe_frame` view, store ref) to the frames in
    `state` (`ORCHESTRATOR_FRAMES_KEY`) under its id, and return the entries
    as the orchestrator reads them.

    The frame ref is the pair the frame is described and drawn by; the store
    ref is the pair the materialisation store holds it under, kept as the
    entry's `store_ref`. It is None for a frame whose executor states no
    store pair. The entry then carries no `store_ref` and is read by its
    frame ref (`frame_store_ref`). `fetch_args` are the arguments of the
    `fetch` that retrieved the frames, kept on each entry because
    `make_charts` asks the same retrieval again for the frame's members' kinds.
    `wholes` maps the id of a frame holding shares to the whole they divide
    by (`harness.plan_display.share_wholes`), kept as the entry's `whole`,
    which `make_charts` states in the card's title.

    A frame id the frames did not hold clears the turn's count of calls that
    drew nothing (`make_charts_failed`), so a chart of a frame that arrived
    after those calls is not refused for them. Synchronous, so no other tool
    call runs between the read of the frames and the writes.
    """
    frames = dict(state.get(ORCHESTRATOR_FRAMES_KEY) or {})
    added = False
    entries = []
    for ref, view, store_ref in described:
        frame_id = frame_id_for(ref)
        added = added or frame_id not in frames
        frames[frame_id] = {"ref": ref, "view": view}
        if store_ref is not None:
            frames[frame_id]["store_ref"] = store_ref
        if fetch_args is not None:
            frames[frame_id]["fetch_args"] = fetch_args
        if wholes and frame_id in wholes:
            frames[frame_id]["whole"] = wholes[frame_id]
        entries.append({"frame": frame_id, "view": view})
    state[ORCHESTRATOR_FRAMES_KEY] = frames
    calls = state.get(ORCHESTRATOR_CALLS_KEY) or {}
    if added and calls.get("make_charts_failed"):
        state[ORCHESTRATOR_CALLS_KEY] = {
            tool: count for tool, count in calls.items() if tool != "make_charts_failed"
        }
    return entries


def frame_store_ref(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The pair to open the materialisation store with for `entry`, an entry
    of `ORCHESTRATOR_FRAMES_KEY`: its `store_ref`, or its `ref` when the
    entry states no store pair or one with a member missing."""
    store_ref = entry.get("store_ref")
    if (
        isinstance(store_ref, Mapping)
        and store_ref.get("spec_id") is not None
        and store_ref.get("content_version") is not None
    ):
        return dict(store_ref)
    return entry["ref"]


def store_search(
    state: SessionState,
    digest: Mapping[str, Any],
    *,
    asked: Sequence[Sequence[str]] = (),
    held_open: Sequence[Sequence[str]] = (),
    share_sides: Sequence[Sequence[str]] = (),
    origin: Origin = "search",
) -> list[dict[str, Any]]:
    """Store what a `resolver_digest` read in the orchestrator's session state,
    and return its parts as the model reads them, without their blocks.

    Each part's spec blocks, its own and its candidates', go under
    `ORCHESTRATOR_SPECS_KEY` and the part's warnings under
    `ORCHESTRATOR_SPEC_WARNINGS_KEY`, where `fetch` finds them, and each part
    joins `ORCHESTRATOR_SEARCHES_KEY` as a `SearchReading` from `origin`.
    `asked` holds each part's asked flow parties by position, and
    `held_open` those it asked to hold open, and `share_sides` the sides
    its `share_of` operations run over; each is empty when no part's are
    known. A `definite` part whose prices are open over several years reads
    with `open_prices`, the options of its prices question
    (`harness.turn_records.open_prices`).
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
            if prices := open_prices(
                part["spec_block"],
                window=part.get("window"),
                other_prices=part.get("other_prices") or (),
            ):
                reading["open_prices"] = prices
        if mentions := reading.get("resolved_sources"):
            blocks = {
                spec_id: {**block, "_resolved_sources": mentions}
                for spec_id, block in blocks.items()
            }
        specs.update(blocks)
        spec_warnings.update(dict.fromkeys(blocks, reading.get("warnings") or []))
        if asked:
            facts["asked_roles"] = list(asked[index])
        if held_open:
            facts["open_roles"] = list(held_open[index])
        if share_sides:
            facts["share_over"] = list(share_sides[index])
        stored = SearchReading.from_part(
            {**reading, **facts}, skills=skill_ids, origin=origin, observed_at=observed_at
        )
        searches.append(stored.model_dump(mode="json"))
        readings.append(reading)
    state[ORCHESTRATOR_SPECS_KEY] = specs
    state[ORCHESTRATOR_SPEC_WARNINGS_KEY] = spec_warnings
    state[ORCHESTRATOR_SEARCHES_KEY] = searches
    return readings


def conversation_readings(state: SessionState) -> list[dict[str, Any]]:
    """The conversation's stored search readings as JSON, oldest first: the
    earlier turns' (`ORCHESTRATOR_PRIOR_LEDGER_KEY`), then this turn's
    (`ORCHESTRATOR_SEARCHES_KEY`)."""
    return [
        *(state.get(ORCHESTRATOR_PRIOR_LEDGER_KEY) or {}).get("readings", []),
        *(state.get(ORCHESTRATOR_SEARCHES_KEY) or []),
    ]


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
    # The chip a turn applies, as JSON, under `APPLY_CHOICE_STATE_KEY`.
    apply_choice: str | None = None
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
