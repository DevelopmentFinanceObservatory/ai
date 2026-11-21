"""The turn's state contract.

The keys and statuses the turn's nodes and the orchestrator's tools
(`harness.orchestrator_tools`) read and write, the inputs a turn opens with
(`TurnInput`), the one pending action the agent's hand-off records, the
tools' budgets, and the helpers more than one node reads: the stored plan
(`readable_plan`), the answers to the agent's question (`resume_answers`),
and the read-only reads of the large containers (`stored_mapping`,
`stored_sequence`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import (
    Mapping,
    Sequence,
)
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, NamedTuple

from data_mcp.search_tool import MAX_PARTS
from planner.plan import AskBreakdown, SessionState, scope_unread, stored_plan, with_blocks
from planner.plan import Plan as PlannerPlan
from pydantic import BaseModel, BeforeValidator, Field, TypeAdapter

from harness.tool_contracts import (
    reading_facts,
)
from harness.turn.store import SPECS_KEY, FrameKey, FrameKeys
from harness.turn_records import Origin, SearchReading, open_prices

if TYPE_CHECKING:
    from harness.turn.node import NodeCtx

# The kind of the pause the `approve` node puts the plan card to the user
# under, read by `harness.studio_translate` to tell the plan card from a
# question.
APPROVE_PAUSE: Final = "approve"

# The orchestrator's turn state, written by its tools
# (`harness.orchestrator_tools`) and read by the turn's nodes.
ORCHESTRATOR_FRAMES_KEY = "orchestrator:frames"  # frame id -> {"ref", "view", "store_ref"}
# spec_id -> its search spec block, kept by the turn store as rows of their own.
ORCHESTRATOR_SPECS_KEY = SPECS_KEY
# The action the agent's `plan` or `ask_user` call recorded, which the
# `orchestrate` node takes once the run hands off (`pending_action`).
ORCHESTRATOR_PENDING_KEY = "orchestrator:pending"
ORCHESTRATOR_CALLS_KEY = "orchestrator:calls"  # tool name -> calls this turn
ORCHESTRATOR_BRIEF_KEY = "orchestrator:brief"  # the question the planner drafts for
ORCHESTRATOR_READINGS_KEY = "orchestrator:readings"  # spec_id -> a plan step's stat_vars reading
# Every search part this turn read, oldest first, as `SearchReading` JSON
# (`harness.turn_records`), failed parts included; `fetch` adds its outcome
# to the readings of its spec_id.
ORCHESTRATOR_SEARCHES_KEY = "orchestrator:searches"
# Every plan this turn ran, oldest first, as `PlanRecord` JSON, written by
# `plan_done`; `harness.studio` keeps it with the turn's record.
ORCHESTRATOR_PLAN_RECORDS_KEY = "orchestrator:plan_records"
# Compile spec_id -> how a plan step's retrieval of it was served, as
# `harness.scope_choices.served_facts` JSON, for each compile alignment moved
# inputs onto (`planner.plan.aligned_pins`), which no search reading holds.
# Written by `execute` (`record_served_scopes`), read by `plan_done`
# (`record_plan`) into the plan record's `PlanSource.aligned_basis`.
ORCHESTRATOR_ALIGNED_SERVED_KEY = "orchestrator:aligned_served"
# spec_id -> the warnings of the search part that returned it, as `search`
# stores its block under `ORCHESTRATOR_SPECS_KEY`.
ORCHESTRATOR_SPEC_WARNINGS_KEY = "orchestrator:spec_warnings"
# The search readings of the conversation's earlier turns, with the warnings
# of the spec_ids they name and the spec_ids whose block an earlier turn
# stored, seeded by `harness.studio` from `ConversationState.
# prior_ledger_state`; unset when no earlier turn read any. The blocks
# themselves resolve through `planner.plan.earlier_block`.
ORCHESTRATOR_PRIOR_LEDGER_KEY = "orchestrator:prior_ledger"
# The orchestrator's conversation while its run waits on a hand-back, as
# `orchestrator.run.StoredHistory` JSON, written and taken by the
# `orchestrate` node.
ORCHESTRATOR_HISTORY_KEY = "orchestrator:history"
# spec_id -> the refusal message of a fetch or a plan step that read it and
# was refused `capability_unavailable` this turn; `fetch` refuses it again
# without a call.
ORCHESTRATOR_UNAVAILABLE_KEY = "orchestrator:unavailable"
# The chip a turn applies, `{"choice_id", "option_id", "spec_id"}`
# (`harness.wire.ApplyChoice`), in the opening state of a turn requested
# with `apply_choice`; `seed` applies it
# (`harness.graph.switch.apply_stored_choice`) and `turn_budgets` reads it,
# the orchestrator's tools included, so it is state rather than an input.
APPLY_CHOICE_STATE_KEY = "apply_choice"
# The chip `seed` applied this turn: `{"choice_id", "option_id",
# "from_option_id", "spec_id"}`, the option it moved from and the spec_id
# Search compiled; for a count chip, the spec_id of the count it moved from
# and of the count it moved to. `harness.studio` keeps it with the turn's record
# (`ConversationState.turn_applied_choice`), where the next turn reads whether
# the user kept it.
ORCHESTRATOR_APPLIED_KEY = "orchestrator:applied_choice"
# Ask number -> the terminal frames of the latest run of this plan call that
# computed that ask (`_kept_asks`), written by `execute` when a run stops and
# read by `plan_done`; cleared by `plan_start` and `plan_done`.
ORCHESTRATOR_KEPT_ASKS_KEY = "orchestrator:kept_asks"
# The run the studio's interrupt route admitted with the approval of the
# plan `approve` last approved (`harness.studio`), or None when it admitted
# none: written by `approve.decide` on every approval and read by `execute`,
# which then finds the run and its stored plan already there.
APPROVED_RUN_KEY = "orchestrator:approved_run"
# The user's words for each amendment of this plan call's plan that the
# planner drafted, in order, each cut to `planner.plan.MAX_QUESTION_CHARS`.
# Written by the `planner` node, cleared by `plan_start`, and handed to the
# agent by `plan_done`, since the amendment reaches the planner alone.
ORCHESTRATOR_AMENDMENTS_KEY = "orchestrator:amendments"
# The state a planner entry replanning for a pick on the plan card puts back
# when the replan does not land (`harness.graph.planning`), written when the
# replan pauses on a planner-side ask and cleared when the entry ends, so it
# is set only while such a replan is paused. Outside the drafting prefixes
# the restore resets (`harness.graph.switch._DRAFTING_PREFIXES`).
PICK_REPLAN_KEY = "switch:pick_replan"
# The status a plan hand-back carries when the planner made no plan, and
# when the approved plan ran; `studio_translate._Translator` shows the first as
# a failed step and the second as nothing.
ORCHESTRATOR_PLAN_FAILED_STATUS = "Reading why no plan was made"
ORCHESTRATOR_PLAN_RAN_STATUS = "Reading the plan's results"


class TurnInput(BaseModel):
    """What a turn opens with, fixed for the turn: the user's `message`, the
    canvas and conversation digests, as the JSON text the conversation wrote
    and the agent reads, and the seeds slot, the Search results a "... in
    Studio" action handed over (each seed as `harness.seeds.seeds_slot`
    keeps it, or the one reference `harness.wire.TurnRequest.seed_spec` also
    takes). The rest of what a turn opens with, which the orchestrator's
    tools also read, is opening state (`turn_budgets`)."""

    message: str
    canvas_digest: str | None = None
    conversation_digest: str | None = None
    seeds: list[Any] | dict[str, Any] | None = None


class AskOption(BaseModel):
    """One answer `ask_user` offers: its wording and, for a part's candidate, its spec_id."""

    label: str = Field(description="the answer as the user reads it")
    spec_id: str | None = Field(
        default=None,
        description=(
            "for one of a part's candidates, the spec_id the search result gives it; "
            "leave it out for any other answer"
        ),
    )


def ask_options(options: list[Any]) -> list[dict[str, Any]]:
    """The `ask_user` options the model sent as `AskOption` dicts. Gemini
    does not hold a call's arguments to the declaration, so an option sent
    as its bare label reads here as an option without a spec_id."""
    return [{"label": option} if isinstance(option, str) else option for option in options]


class AskQuestion(BaseModel):
    """One question `ask_user` puts to the user, with the answers it offers."""

    question: str = Field(description="the question as the user reads it")
    options: Annotated[list[AskOption], BeforeValidator(ask_options)] = Field(
        default_factory=list,
        description=(
            "the likely answers, at least one; the user may also write their own. Leave it "
            "empty for a prices question, whose options are filled in for you"
        ),
    )
    prices_of: str | None = Field(
        default=None,
        description="for the prices question of a part that lists `open_prices`, that part's spec_id",
    )


class PendingAsk(BaseModel):
    """`ask_user` asked: the workflow pauses with its questions, in order."""

    kind: Literal["ask"] = "ask"
    questions: list[AskQuestion]


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


def stored_mapping(state: SessionState, key: str) -> Mapping[str, Any]:
    """The dict `state` holds under `key` as a read-only view, empty when
    the key is unset.

    For the large containers a turn reads many times (frames, spec blocks,
    readings, the prior ledger). An assignment through the view raises
    `TypeError`, so every change is an assignment to the key, which the
    orchestrator's `harness.gemini_agent.RecordingState` records. Its
    entries are the stored objects, so a change is built on a copy
    (`dict(view)`, `{**entry, ...}`) and assigned to the key.
    """
    return MappingProxyType(state.get(key) or {})


def stored_sequence(state: SessionState, key: str) -> tuple[Any, ...]:
    """The list `state` holds under `key` as a tuple, empty when the key is
    unset. A tuple has no method that changes it in place, as
    `stored_mapping`'s view has none."""
    return tuple(state.get(key) or ())


def open_parts(state: SessionState) -> dict[str, list[dict[str, Any]]]:
    """The `open_prices` entries of each reading of the turn whose prices are open,
    by its spec_id, each with its `prices`: the first the prices search read,
    current, and each after it constant (`harness.turn_records.open_prices`).

    Read over the readings of `ORCHESTRATOR_SEARCHES_KEY` and the blocks of
    `ORCHESTRATOR_SPECS_KEY`. Only code reads `prices`: the readings the model
    reads carry the entries without it (`store_search`).
    """
    specs = stored_mapping(state, ORCHESTRATOR_SPECS_KEY)
    parts: dict[str, list[dict[str, Any]]] = {}
    for stored in stored_sequence(state, ORCHESTRATOR_SEARCHES_KEY):
        reading = SearchReading.model_validate(stored)
        block = specs.get(reading.spec_id) if reading.spec_id is not None else None
        if block is None or reading.spec_id is None:
            continue
        entries = open_prices(
            block,
            window=reading.window.model_dump() if reading.window is not None else None,
            other_prices=[offer.model_dump() for offer in reading.other_prices],
        )
        if entries:
            parts[reading.spec_id] = [
                {**entry, "prices": "current" if index == 0 else "constant"}
                for index, entry in enumerate(entries)
            ]
    return parts


def prices_question(
    question: AskQuestion, parts: Mapping[str, list[dict[str, Any]]]
) -> AskQuestion:
    """`question` with, when it prices a part of `parts` (`AskQuestion.prices_of`),
    that part's `open_prices` entries as its options."""
    entries = parts.get(question.prices_of) if question.prices_of is not None else None
    if entries is None:
        return question
    return question.model_copy(
        update={
            "options": [
                AskOption(label=entry["label"], spec_id=entry.get("spec_id")) for entry in entries
            ]
        }
    )


def priced_options(
    questions: list[AskQuestion], state: SessionState
) -> list[dict[int, PricedOption]]:
    """For each of `questions`, the place of each option offering prices, with
    the part it prices and the prices: each option of a question pricing a
    part (`AskQuestion.prices_of`) that is one of the part's `open_prices`
    entries (`open_parts`), matched by spec_id, else by label among the
    entries carrying none. A question pricing no part has none."""
    parts = open_parts(state)
    priced: list[dict[int, PricedOption]] = []
    for question in questions:
        entries = parts.get(question.prices_of) if question.prices_of is not None else None
        marks: dict[int, PricedOption] = {}
        for index, option in enumerate(question.options):
            entry = next(
                (
                    entry
                    for entry in entries or ()
                    if entry.get("spec_id") == option.spec_id
                    and (option.spec_id is not None or entry["label"] == option.label)
                ),
                None,
            )
            if entry is not None and question.prices_of is not None:
                marks[index] = PricedOption(question.prices_of, entry["prices"])
        priced.append(marks)
    return priced


class PricedOption(NamedTuple):
    """The prices an `ask_user` option offers, as code reads them off the turn's
    search results (`priced_options`): `part`, the
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
    answered interrupt (`planner.call.prices_chosen`)."""
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

    The value `harness.studio`'s `/interrupt` route resumes a clarify with
    is `{"answers": {question id: answer}}`, each answer
    `{"kind": "option", "spec_id"}`, `{"kind": "text", "text"}` or
    `{"kind": "skipped"}`. Any other value, such as the plan card's answer,
    answers no question.
    """
    if not isinstance(result, dict):
        return {}
    if result.get("skip"):
        return None
    return dict(result.get("answers") or {})


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
    # The first search, a corrected place, and one `text` search for each
    # part whose result carried `retry_with_text`.
    "search": MAX_PARTS + 2,
    "fetch": 6,
    "make_charts": 5,
    "make_charts_failed": 3,
    "ask_user": 2,
    "plan": 1,
    "get_skill": 2,
    # `get_skill` with a `path`: a file a skill links, or the next page of a long one.
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
    `BUDGETS` otherwise. `state` is the turn's state."""
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
    """The one action the agent's run recorded (`ORCHESTRATOR_PENDING_KEY`),
    or None. `state` is the turn's state."""
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
    frames = dict(stored_mapping(state, ORCHESTRATOR_FRAMES_KEY))
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


def frame_store_keys(state: Mapping[str, Any]) -> FrameKeys:
    """The lineage store keys of the frames in `state`
    (`ORCHESTRATOR_FRAMES_KEY`): each entry's `ref` and `store_ref`, in order
    and each once, which the turn store keeps the lineage of
    (`harness.turn.store.FrameLineage`). A pair with a member missing names
    no key, and so does a frames value or entry that is not a mapping, which
    `FrameKeys.skipped` describes for the caller to report."""
    frames = state.get(ORCHESTRATOR_FRAMES_KEY) or {}
    if not isinstance(frames, Mapping):
        return FrameKeys(keys=[], skipped=[], container=type(frames).__name__)
    keys: dict[FrameKey, None] = {}
    skipped: list[str] = []
    for frame_id, entry in frames.items():
        if not isinstance(entry, Mapping):
            skipped.append(f"{frame_id}:{type(entry).__name__}")
            continue
        for ref in (entry.get("ref"), entry.get("store_ref")):
            if isinstance(ref, Mapping):
                spec_id, content_version = ref.get("spec_id"), ref.get("content_version")
                if isinstance(spec_id, str) and isinstance(content_version, str):
                    keys[(spec_id, content_version)] = None
    return FrameKeys(keys=list(keys), skipped=skipped)


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
    `ORCHESTRATOR_SPECS_KEY`, each candidate's marked as one whose scope
    choices Search has not stated (`planner.plan.scope_unread`), and the
    part's warnings under `ORCHESTRATOR_SPEC_WARNINGS_KEY`, where `fetch`
    finds them. The part holds each of its spec_ids with its own record and
    warnings, and an earlier block of the spec_id, of an earlier part or
    already stored, that says more about its scope choices keeps its scope
    content standing (`planner.plan.with_blocks`). The parts' blocks are
    merged first and joined to the store once, which copies it once. Each
    part joins `ORCHESTRATOR_SEARCHES_KEY` as a `SearchReading` from `origin`.
    `asked` holds each part's asked flow parties by position, and
    `held_open` those it asked to hold open, and `share_sides` the sides
    its `share_of` operations run over; each is empty when no part's are
    known. A `definite` part whose prices are open over several years reads
    with `open_prices`, the options of its prices question
    (`harness.turn_records.open_prices`).

    A spec_id a part holds as its own keeps that part's block and warnings
    against a later part offering it as a candidate, since a pin of it is
    one of the search that settled on it (`harness.planner_adapter`, which
    reads both for the pin). Between blocks no search settled, or two
    searches that each settled it, `planner.plan.with_blocks` decides.
    """
    specs = stored_mapping(state, ORCHESTRATOR_SPECS_KEY)
    spec_warnings = dict(state.get(ORCHESTRATOR_SPEC_WARNINGS_KEY) or {})
    searches = list(stored_sequence(state, ORCHESTRATOR_SEARCHES_KEY))
    own = {stored["spec_id"] for stored in searches if stored.get("spec_id")}
    skill_ids = [skill["id"] for skill in digest.get("skills") or []]
    observed_at = datetime.now(UTC)
    readings = []
    found: dict[str, Mapping[str, Any]] = {}
    for index, part in enumerate(digest.get("part_digests") or []):
        reading = {
            key: value
            for key, value in part.items()
            if key not in ("spec_block", "candidate_blocks")
        }
        blocks = {
            spec_id: scope_unread(block)
            for spec_id, block in (part.get("candidate_blocks") or {}).items()
            if spec_id not in own
        }
        facts: dict[str, Any] = {}
        if part.get("spec_block") and part.get("spec_id"):
            blocks[part["spec_id"]] = part["spec_block"]
            own.add(part["spec_id"])
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
        found = with_blocks(found, blocks)
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
    state[ORCHESTRATOR_SPECS_KEY] = with_blocks(specs, found)
    state[ORCHESTRATOR_SPEC_WARNINGS_KEY] = spec_warnings
    state[ORCHESTRATOR_SEARCHES_KEY] = searches
    return readings


def conversation_readings(state: SessionState) -> list[dict[str, Any]]:
    """The conversation's stored search readings as JSON, oldest first: the
    earlier turns' (`ORCHESTRATOR_PRIOR_LEDGER_KEY`), then this turn's
    (`ORCHESTRATOR_SEARCHES_KEY`)."""
    return [
        *stored_mapping(state, ORCHESTRATOR_PRIOR_LEDGER_KEY).get("readings", []),
        *stored_sequence(state, ORCHESTRATOR_SEARCHES_KEY),
    ]


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


def readable_plan(ctx: NodeCtx[Any], raw: str | None) -> PlannerPlan | None:
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
