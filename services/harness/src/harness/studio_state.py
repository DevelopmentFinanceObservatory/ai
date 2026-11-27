"""Per-conversation studio state and the composition manifest rules.

`ConversationState` is one conversation's wire bookkeeping: frame `seq`,
pending interrupt ids, plan version lineage, the composition manifest, the
turn records, and the digests the models see. `StudioStore`
(`harness.studio_store`) keys those states on (principal, conversation_id)
and stores their durable fields. The manifest helpers here are the one place a
composition changes: `_apply_operations` for the user's mutations,
`_next_manifest` for every new revision.

`Segment` is a durable lease on a conversation's active stream, the
mechanism that keeps at most one instance streaming a conversation at a
time: `StudioStore.claim_segment` takes it, `renew` pushes its TTL forward,
`close_segment` writes the rest of the conversation's state to storage
at end of segment in one transaction, and
`request_cancel` signals it from any instance. See `Segment`,
`SegmentBusy`, `SegmentLost` and `DatabaseUnavailable`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from common.held_note import (
    drawn_window,
    held_period,
    held_runs,
    holds_projections,
    period_year,
    reaches,
)
from common.telemetry import end_turn_segment, start_turn_segment
from compiler.sources import alternative_definitions, source_from_spec_block
from compiler.surface import SpecUse
from data_mcp.definition_store.identity import IdentityMismatch
from data_mcp.definition_writes import IdentityUnusable, caller_definition, windowed_identity
from executor.ledger import RunStatus
from fastapi.responses import JSONResponse
from planner.plan import PlanStatus, cut_user_text
from pydantic import BaseModel, ValidationError

from harness.approval import ApprovalRecord
from harness.card_unit import CARD_UNIT_KEY
from harness.conversation import current_conversation_key
from harness.reply_figures import drawn_series
from harness.turn_records import (
    TURN_CARD_PROJECTED,
    TURN_CARD_RUNS,
    TURN_CARD_SPECS,
    PlanRecord,
    PlanSource,
    RecordedIdentity,
    SearchReading,
    named_spec_ids,
    newest_per_spec_id,
    render_readings,
)
from harness.wire import (
    AddOp,
    ClientFeature,
    ComponentType,
    CompositionActor,
    CompositionBlock,
    CompositionCounts,
    CompositionManifest,
    ElicitationInput,
    ErrorCode,
    LayoutHint,
    LayoutHintType,
    PinOp,
    PlanApproved,
    PlanLink,
    PlanProposed,
    RemoveOp,
    ReorderOp,
    RetryMode,
    SetCardMetaOp,
    SetLayoutHintsOp,
    StudioFrame,
    ToolError,
    UnitOffer,
)

if TYPE_CHECKING:
    from harness.provenance import CardLineage
    from harness.studio_store import StudioStore

logger = logging.getLogger(__name__)

# The canvas views' bounds: up to `_DIGEST_BLOCK_CAP` cards in full, each
# with a capped id, type, title and fact summary, and a focus id that names a
# stored component. Caps are serialised JSON bytes (`json.dumps`, so a
# non-ASCII character counts as its escape). A cut field ends in `_CUT_MARKER`.
# A fact summary over its cap keeps its leading series in display order and
# says so (`_bounded_fact_summary`),
# and one that does not fit even without series is replaced by a `{"cut": ...}`
# stub. Factored (`_factored_fact_summary`), a one-series fact summary runs
# about 600 bytes and each further series adds about 150, so the cap holds a
# chart of about twenty series. Titles run about 40 bytes.
_DIGEST_BLOCK_CAP = 16
_TITLE_BYTE_CAP = 200
_FACT_SUMMARY_BYTE_CAP = 4_096
# The wire forwards a unit object whole, so a large one is replaced by
# `_CUT_MARKER` before any series is cut.
_UNIT_OBJECT_BYTE_CAP = 512
_CUT_MARKER = "[cut]"
# Around its title and fact summary a block carries the keys, a type name of up
# to `_BLOCK_TYPE_BYTES` (the longest `ComponentType` is 16), the revision (an
# integer the platform mints, taken to be at most ten digits), and a component
# id of up to `_BLOCK_ID_BYTES` (the platform mints `cmp_` and 16 hex digits).
# `_BLOCK_FRAME_BYTES` covers the keys, the type and the revision, and the id
# adds its own cap. The id and the type are cut to their caps, so the bound
# holds for any input except a revision of more than ten digits.
_BLOCK_ID_BYTES = 64
_BLOCK_TYPE_BYTES = 32
_BLOCK_FRAME_BYTES = 128
# What a block shows (`_digest_shows`): up to `_DIGEST_SHOW_CAP` entity names,
# the widest select a component admits (`line_chart`'s 8 values), each cut to
# `_DIGEST_NAME_BYTE_CAP`, and a time span cut to `_DIGEST_TIME_BYTE_CAP`.
# `_BLOCK_SHOWS_BYTES` covers the list with its key, brackets and separators,
# and the time span with its key.
# A list cut to its cap carries an `entities_truncated` label of up to
# `_SUBSET_LABEL_BYTES`, key included (`_subset_label`).
_DIGEST_SHOW_CAP = 8
_DIGEST_NAME_BYTE_CAP = 64
_DIGEST_TIME_BYTE_CAP = 24
_SUBSET_LABEL_BYTES = 96
_BLOCK_SHOWS_BYTES = (
    16
    + _DIGEST_SHOW_CAP * (_DIGEST_NAME_BYTE_CAP + 2)
    + _SUBSET_LABEL_BYTES
    + 12
    + _DIGEST_TIME_BYTE_CAP
)
# The rule that chooses which items a cut list keeps: its leading items, in the
# order the card displays them.
_SUBSET_RULE = "display order"
# `recall(part="canvas")`'s listing (`ConversationState.canvas_listing_state`)
# names cards in the order `ConversationState._canvas_rank` ranks them by the
# turn that drew each card: this turn's cards on the canvas, the pinned cards,
# the earlier cards still on the canvas, newest turn first, and then the cards
# off the canvas, this turn's first and then newest turn first. The
# manifest's block order plays no part, and no card states a place on the
# canvas: Studio keeps where each card sits on the client, so the manifest's
# order is not the order the user sees. `_DIGEST_BLOCK_CAP` cards on the
# canvas come in full, and past them each of the three on-canvas groups names up to
# `_LISTING_COMPACT_CAP` cards compactly (`_compact_entry`: id, type and title
# inside `_COMPACT_FRAME_BYTES` of keys). The cards off the canvas are named
# compactly a whole turn at a time while the turns fit `_LISTING_COMPACT_CAP`.
# A full or compact entry carries up to `_VIEW_FLAG_BYTES` of pinned,
# on_canvas and turn flags. A turn draws at most
# `BUDGETS["make_charts"] * MAX_CHARTS_PER_CALL` cards (40), fewer than
# `_LISTING_COMPACT_CAP`, so every card of this turn is named. The cards left
# are counted by turn: the newest `_EARLIER_TURN_CAP` turns, each within
# `_EARLIER_ENTRY_BYTES`, and one more entry for the rest. The keys and counts
# around the lists take up to `_LISTING_WRAPPER_BYTES`.
_LISTING_COMPACT_CAP = 48
_COMPACT_FRAME_BYTES = 48
_VIEW_FLAG_BYTES = 56
_EARLIER_TURN_CAP = 16
_EARLIER_ENTRY_BYTES = 80
_LISTING_WRAPPER_BYTES = 256
_FULL_ENTRY_BYTES = (
    _BLOCK_FRAME_BYTES
    + _VIEW_FLAG_BYTES
    + _BLOCK_ID_BYTES
    + _TITLE_BYTE_CAP
    + _FACT_SUMMARY_BYTE_CAP
    + _BLOCK_SHOWS_BYTES
)
_COMPACT_ENTRY_BYTES = (
    _COMPACT_FRAME_BYTES + _VIEW_FLAG_BYTES + _BLOCK_ID_BYTES + _BLOCK_TYPE_BYTES + _TITLE_BYTE_CAP
)
CANVAS_LISTING_MAX_BYTES = (
    _LISTING_WRAPPER_BYTES
    + _DIGEST_BLOCK_CAP * _FULL_ENTRY_BYTES
    + 4 * _LISTING_COMPACT_CAP * _COMPACT_ENTRY_BYTES
    + (_EARLIER_TURN_CAP + 1) * _EARLIER_ENTRY_BYTES
)
# The canvas summary every turn opens with (`ConversationState.canvas_summary_state`):
# three counts and, for a card-scoped follow-up, the focus card's id and the
# card in full, the counts and keys within `_SUMMARY_WRAPPER_BYTES`.
_SUMMARY_WRAPPER_BYTES = 192
CANVAS_SUMMARY_MAX_BYTES = _SUMMARY_WRAPPER_BYTES + _BLOCK_ID_BYTES + _FULL_ENTRY_BYTES
# The conversation digest's bounds: newest turns, each with its ask whole,
# an older turn's answer cut to whole sentences, its plans each as one line
# cut by characters, its cards by count and title bytes, and its error by
# bytes. Over the byte cap on the serialised digest, older answers shrink
# through `_HISTORY_ANSWER_CAPS`, then the oldest turn drops, and finally the
# newest turn shrinks. Past any bound, the orchestrator `recall`s.
_HISTORY_TURN_CAP = 10
_HISTORY_TEXT_CAP = 280
_HISTORY_ANSWER_CAPS = (_HISTORY_TEXT_CAP, _HISTORY_TEXT_CAP // 2, 0)
_HISTORY_PLAN_CAP = 1_000
_HISTORY_CARD_CAP = 8
_HISTORY_CARD_TITLE_BYTE_CAP = 100
_HISTORY_ERROR_BYTE_CAP = 100
CONVERSATION_DIGEST_MAX_BYTES = 5_000
# What a newest turn's ask and answer are cut to once the newest turn alone
# is over the digest's cap, halved each further try.
_HISTORY_NEWEST_TEXT_CAP = CONVERSATION_DIGEST_MAX_BYTES // 2
type RecallPart = Literal["plan", "readings", "canvas"]
# What `recall_part` renders of a turn's plan records, or of its readings, in
# characters, which a turn's plans share. `_RECALL_ID_CAP` is the most component
# ids a refusal lists.
_RECALL_TEXT_BUDGET = 4_000
_RECALL_ID_CAP = 20
# Where a sentence ends: its closing mark, when whitespace follows.
_SENTENCE_END = re.compile(r"[.!?](?=\s)")

_COUNT_KEYS: dict[ComponentType, str] = {
    ComponentType.STAT_CARD: "stats",
    ComponentType.LINE_CHART: "charts",
    ComponentType.BAR_CHART: "charts",
    ComponentType.SCATTER_CHART: "charts",
    ComponentType.TABLE: "tables",
    ComponentType.NOTE_USER: "notes",
    ComponentType.NOTE_NARRATION: "notes",
}


def _wire_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _apply_operations(
    components: Mapping[str, dict[str, Any]],
    manifest: CompositionManifest | None,
    operations: list[Any],
) -> JSONResponse | tuple[list[CompositionBlock], Any]:
    """Apply mutations to LOCAL copies; the first invalid operation refuses
    the whole request with nothing applied (mutations are transactional).
    `components` is the conversation's digest entries by component id, the
    cards an add may put back."""
    blocks = [block.model_copy(deep=True) for block in manifest.blocks] if manifest else []
    layout_hints = manifest.layout_hints if manifest else None
    for operation in operations:
        if isinstance(operation, SetLayoutHintsOp):
            layout_hints = operation.layout_hints
            continue
        if isinstance(operation, AddOp):
            # Delta 11: append a block for a component this conversation
            # already streamed; the server rebuilds it from its digest.
            entry = components.get(operation.component_id)
            if entry is None:
                return _tool_error_response(
                    ToolError(
                        error=ErrorCode.NOT_FOUND,
                        retry=RetryMode.DO_NOT_RETRY,
                        detail=(
                            f"no component {operation.component_id!r} was streamed "
                            "on this conversation"
                        ),
                        pointer="/operations/component_id",
                    ),
                    status_code=404,
                )
            if any(b.component_id == operation.component_id for b in blocks):
                continue  # idempotent: already on the canvas
            blocks = [
                *blocks,
                CompositionBlock(
                    component_id=operation.component_id,
                    revision=int(entry.get("revision") or 1),
                    type=entry.get("type") or "stat_card",
                    title=entry.get("title") or "component",
                    pinned=False,
                ),
            ]
            continue
        if isinstance(operation, ReorderOp):
            by_id = {block.component_id: block for block in blocks}
            reordered = [by_id.pop(cid) for cid in operation.order if cid in by_id]
            blocks = reordered + list(by_id.values())
            continue
        target = next((b for b in blocks if b.component_id == operation.component_id), None)
        if target is None:
            return _tool_error_response(
                ToolError(
                    error=ErrorCode.NOT_FOUND,
                    retry=RetryMode.REFRESH_THEN_RETRY,
                    detail=f"no block {operation.component_id!r} in the composition",
                    pointer="/operations/component_id",
                ),
                status_code=404,
            )
        if isinstance(operation, RemoveOp):
            blocks = [b for b in blocks if b.component_id != operation.component_id]
        elif isinstance(operation, SetCardMetaOp):
            # Delta 13: presentation-only card fields the platform intent
            # cannot hold; harness-owned, client-rendered.
            target.meta = operation.meta
        else:
            target.pinned = isinstance(operation, PinOp)
    return blocks, layout_hints


def _tool_error_response(error: ToolError, *, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content=error.model_dump(mode="json", exclude_none=True)
    )


# ---------------------------------------------------------------------------
# Conversation bookkeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CanvasRank:
    """The conversation's cards as `ConversationState._canvas_rank` ranks them
    for the canvas views. `this_turn` holds the cards on the canvas the open
    turn drew, in the order it drew them; `pinned` the other pinned cards,
    newest drawing turn first; `on_canvas` the other cards on the canvas,
    newest drawing turn first and each turn's in the order it drew them, with
    the cards of no recorded turn last; `off_canvas` the cards off the canvas
    by the turn that drew them, the open turn's first, then newest turn first,
    then the cards of no recorded turn under `None`. `turn_of` maps a card to
    the turn that last drew or revised it, and `composed` holds the cards on
    the canvas."""

    focus: str | None
    this_turn: list[str]
    pinned: list[str]
    on_canvas: list[str]
    off_canvas: list[tuple[int | None, list[str]]]
    turn_of: dict[str, int]
    composed: set[str]
    pinned_ids: set[str]


@dataclass(frozen=True)
class _PendingQuestion:
    """One question of a pending elicitation, as `/interrupt` checks an
    answer against it and writes it into the turn's record."""

    id: str
    question: str
    # Each offered option id to its label. A label is `None` on a question
    # restored from before questions were kept (`legacy`).
    options: dict[str, str | None] = field(default_factory=dict)
    skippable: bool = False
    # How the question is answered, which names the answer kinds it takes.
    input: ElicitationInput = ElicitationInput.SINGLE_SELECT
    # A unit question's offer, which a unit answer must lie inside. A unit
    # question sent as a `single_select` keeps it too: its options are the
    # offer's whole units (`harness.unit_offer.whole_units`).
    unit: UnitOffer | None = None
    # The form the question takes for a client that renders every form, when
    # the frame sent another because the client declared it does not render
    # this one (`ConversationState.client_features`): `unit` or
    # `multi_select` for a question sent as a `single_select`. `None` when
    # `input` is the question's own form.
    native_input: ElicitationInput | None = None


@dataclass(frozen=True)
class _PendingOption:
    """One option of a scope choice the plan card shows, as `/interrupt`
    checks a switch against it: its label and whether it can be served
    (`available`, `partial`, `unavailable`)."""

    label: str
    status: str


@dataclass(frozen=True)
class _PendingChoice:
    """A scope choice the plan card shows, as `/interrupt` checks a switch
    against it: its active option and each option by id."""

    active: str
    options: dict[str, _PendingOption] = field(default_factory=dict)


@dataclass
class _PendingInterrupt:
    """One wire interrupt the conversation is paused on, with what
    `/interrupt` checks an answer to it against. `adk_id` is the wire
    interrupt id, under the key stored conversations hold it by."""

    adk_id: str
    kind: str  # "plan" | "elicitation"
    plan_version: int | None = None
    # The plan revision this interrupt is keyed on. The resumed answer is
    # compared against this rather than against `plan_version`, so a stale
    # plan card cannot approve a later plan sharing its version number.
    plan_revision: str | None = None
    # An elicitation's questions, in frame order. Empty for a plan.
    questions: tuple[_PendingQuestion, ...] = ()
    # The elicitation frame's top-level `skippable`. When true, the whole
    # pause may be dismissed with `decision: "skip"`.
    skippable: bool = False
    # An elicitation stored before questions were kept: its one question is
    # lifted from the flat option ids, and `/interrupt` also takes the
    # single-answer `payload` for it.
    legacy: bool = False
    # A "plan" interrupt's workflow step order at the moment `plan.proposed`
    # was built (`_RecordingSurface._step_order`, `studio_translate.py`).
    # Restored before a resumed segment starts so activity steps stay
    # numbered by plan order, whichever instance answers the interrupt.
    step_order: tuple[str, ...] = ()
    # The same plan's label for each step, by step id, restored beside
    # `step_order` so a resumed segment's Activity lines carry the plan's
    # own step names. Empty for an interrupt saved before labels were kept.
    step_labels: dict[str, str] = field(default_factory=dict)
    # The plan card's step each workflow step runs for
    # (`harness.plan_display.card_steps`), restored beside `step_labels` so a
    # resumed segment's Activity lines name the card's steps. Empty for an
    # interrupt saved before it was kept.
    card_steps: dict[str, str] = field(default_factory=dict)
    # Set by `/interrupt` to the id of the segment resuming this interrupt,
    # instead of deleting the entry outright: a crash between the answer
    # being appended and the resumed segment's own closing write leaves a
    # durable, retryable record. Cleared by `ConversationState.drop_resuming`
    # once that segment ends, win or lose; a crashed segment leaves it set,
    # and the same answer resumes again on a fresh segment.
    resuming_segment: str | None = None
    # The decision, payload and answers `/interrupt` was called with,
    # recorded alongside `resuming_segment` so a retried resume can be
    # verified against what actually resumed.
    answer: dict[str, Any] | None = None
    # A plan's scope choices the card shows, by choice id: what a switch on
    # this card may name. Empty for an elicitation, a plan with none, and a
    # plan paused before choices were kept.
    choices: dict[str, _PendingChoice] = field(default_factory=dict)
    # A plan's `plan.proposed` frame as the stream sent it: the card a
    # refused stale answer gets back (`PlanRevisionStaleError.current_plan`).
    # None for an elicitation and a plan paused before cards were kept.
    card: PlanProposed | None = None
    # The run the save that recorded this plan interrupt's approval admitted
    # and left for `execute` to compute, `admitted` or `running`
    # (`harness.studio._approval_admission`); None when it admitted none or
    # the key named a finished run. A retry resuming the same answer names
    # it to `approve` as the first did.
    admitted_run: str | None = None
    # Every frame the pause was sent as, in order, its card or questions
    # last: a plan's refusal note and `plan.versioned` come first. A replay
    # of the pause to a client that never read it sends these again under
    # fresh `seq`s (`harness.studio_translate._Translator._interrupt_frames`),
    # so it mints, supersedes and versions nothing. Empty for an entry
    # stored before the frames were kept, which a replay translates afresh.
    frames: tuple[StudioFrame, ...] = ()


# A conversation's active stream may run on any instance; this is the
# lease that keeps at most one running at a time. TTL 45s means a crashed
# owner blocks the conversation for at most that long before another
# instance can take over.
SEGMENT_TTL_S = 45
# How often a stream's heartbeat renews the live segment's lease
# (`harness.turn.loop.run_segment`, `harness.studio.StudioRunner._heartbeat`),
# well inside the TTL so a slow renewal still lands before it lapses.
HEARTBEAT_INTERVAL_S = 5


class SegmentBusy(Exception):  # noqa: N818 -- named for the wire 409, not an error condition
    """Raised by `StudioStore.claim_segment` when a live segment already
    owns this conversation. `retry_after_seconds` is the ceiling of its
    remaining TTL, the wait before a fresh claim can succeed."""

    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__(
            f"conversation already has a live segment; retry in {retry_after_seconds}s"
        )
        self.retry_after_seconds = retry_after_seconds


class SegmentLost(Exception):  # noqa: N818 -- named for the wire condition, not an error kind
    """Raised by `StudioStore`'s fenced writes (`change_canvas`) when the
    stored live segment is no longer the one the call writes on behalf of. Another
    instance took over after this one's TTL lapsed."""


class DatabaseUnavailable(Exception):  # noqa: N818 -- named for the condition the stream reports
    """Raised by `StudioStore`'s retried write
    (`change_canvas`) when the database stayed unreachable
    until the lease's deadline. The stream ends as a lost segment does,
    leaving the turn as stored, since no close could be saved either."""


@dataclass(frozen=True)
class Segment:
    """One durable claim on a conversation's active stream, the lease
    columns of its stored row. `id` is the fencing token
    `StudioStore.claim_segment` mints; every later write on behalf of this
    segment (`renew`, `change_canvas`) is fenced against it, never against
    `owner`. `owner` is the claiming instance's id, carried for logs only.
    `expires_at` is a wall-clock epoch second past which the claim is dead
    and takeable. The Stop requests made of the conversation live in columns
    of their own (`harness.studio_store.StopState`)."""

    id: str
    owner: str
    expires_at: float


# The message, and the `event` field, of the log line `StudioStore` writes
# for each turn close and each pause one of its writes commits.
TURN_LOG_EVENT = "studio.turn"

# A turn record's `ended_by` for a turn the user ended by typing a different
# question on its plan card (`ConversationState.close_turn`).
NEW_QUESTION = "new_question"
# The reply such a turn closes on, which a client that does not read
# `done.new_question` still shows.
SET_ASIDE_REPLY = "The plan was set aside, since your message asks a new question."
# What the conversation digest says of such a turn (`_history_turn`).
_SET_ASIDE = "The user set this plan aside unapproved and asked a new question."
# A turn record's `ended_by` for a turn whose plan card the user set aside for
# a new question no message of theirs states, which closes asking them for it
# (`harness.graph.state.ASK_FOR_QUESTION_KEY`).
ASKED_FOR_QUESTION = "asked_for_question"
# The reply such a turn closes on.
ASK_FOR_QUESTION_REPLY = "The plan was set aside. What would you like to ask?"


# What the conversation digest says of such a turn (`_history_turn`).
_ASKED_FOR = (
    "The user set this plan aside unapproved to ask a new question, and was asked to send it."
)


@dataclass
class TurnUsage:
    """Who asked the current turn and which specs it used, for its
    `TURN_LOG_EVENT` line and its telemetry spans.

    `caller` is the `conversation:` caller the harness's MCP clients send for
    the conversation (`harness.conversation.current_conversation_caller`).
    `account` is the website's digest of a signed-in account, `visitor` the
    website's `ip:` caller, and `surface` the turn's `harness.wire.Surface`
    value; each is `None` when the request named none. `prompt_source` is
    the `harness.wire.PromptSource` value of a turn whose question came from
    an example the page offered, and `None` for a typed one or a value outside those three. `fetched_specs` holds
    each spec a foreground `get_data` call retrieved and `drawn_specs` each
    spec a card drawn this turn rests on, once each, in the order first used
    (`compiler.surface.current_spec_uses`). `trace_id` is the turn's
    `ConversationState.trace_id`, so a stored usage naming another turn's
    reads as unset (`harness.studio_store._turn_usage`). `started_at` is when
    `begin_turn` began the turn, as an ISO 8601 UTC timestamp, so a turn
    closed days later still counts on the day it ran.
    """

    trace_id: str | None = None
    started_at: str | None = None
    caller: str | None = None
    account: str | None = None
    visitor: str | None = None
    surface: str | None = None
    prompt_source: str | None = None
    fetched_specs: list[str] = field(default_factory=list)
    drawn_specs: list[str] = field(default_factory=list)

    @property
    def user_id(self) -> str | None:
        """The Langfuse user of the turn: the account, else the visitor."""
        return self.account or self.visitor

    @property
    def tags(self) -> list[str]:
        """The Langfuse tags of the turn: its surface, when it names one."""
        return [self.surface] if self.surface else []

    @property
    def trace_metadata(self) -> dict[str, str]:
        """The Langfuse trace metadata of the turn: its `prompt_source`, when
        it has one."""
        return {"prompt_source": self.prompt_source} if self.prompt_source else {}

    def copy(self) -> TurnUsage:
        """A copy whose spec lists are lists of their own."""
        return dataclasses.replace(
            self, fetched_specs=list(self.fetched_specs), drawn_specs=list(self.drawn_specs)
        )


@dataclass
class ConversationState:
    """Wire bookkeeping for one (principal, conversation_id) pair."""

    conversation_id: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    turn_index: int = 0
    # The next frame number. A segment's claim sets it to the start of the
    # block it reserved (`StudioStore.claim_segment`), and the stream's
    # writes store it.
    seq: int = 0
    trace_id: str = ""
    pending: dict[str, _PendingInterrupt] = field(default_factory=dict)
    plan_version_ids: dict[int, str] = field(default_factory=dict)
    latest_plan_version: int | None = None
    composition: CompositionManifest | None = None
    # The durable lease on this conversation's active stream. None when no
    # segment is live. Set by `claim_segment`, refreshed by `renew`,
    # cleared by `close_segment`.
    segment: Segment | None = None
    # Set while a stream runs on this instance. Setting it makes the
    # stream's heartbeat renew the lease at once, so a Stop posted here
    # (`POST /api/studio/cancel`) or a lease another instance took reaches
    # the stream without waiting for the next beat (`renew_lease_now`).
    renew_now: asyncio.Event | None = None
    # The current turn's auto-add preference (delta 11); reset per turn.
    auto_add: bool = True
    # The question forms the latest turn or interrupt request declared the
    # client renders (`harness.wire.TurnRequest.client_features`). Stored, so
    # a resume on another instance asks in the same forms. A conversation
    # stored before features were kept holds none.
    client_features: frozenset[ClientFeature] = field(default_factory=frozenset)
    # Digest entries for every component this conversation streamed,
    # keyed by component_id: {type, title, revision, fact_summary}.
    # Feeds the models' canvas context and the composition "add" op.
    components: dict[str, dict[str, Any]] = field(default_factory=dict)
    # The write token each component this conversation created was issued,
    # keyed by component_id (`common.mcp_telemetry.WRITE_TOKEN_KEY`). A
    # bearer secret: kept apart from `components`, which feeds the digests,
    # recall and telemetry, and out of repr so a logged state never shows
    # one. A component with no entry here was not created by this
    # conversation, so the Edit Card route recreates it instead of revising.
    write_tokens: dict[str, str] = field(default_factory=dict, repr=False)
    # One record per user answer to a plan interrupt (approve, amend or
    # reject), appended by `/interrupt` regardless of what the graph does
    # with it next. Survives across turns within this conversation; lost
    # with the rest of this state on restart (`ApprovalRecord`'s docstring).
    approvals: list[ApprovalRecord] = field(default_factory=list)
    # One record per completed turn: {user, assistant, cards, error, plans,
    # readings, specs, identities}, where `plans` are the `PlanRecord`s of
    # the plans the turn ran, `readings` the `SearchReading`s it read, as
    # JSON, `specs` the spec block `search` returned for each source those
    # plans used, and `identities` the `RecordedIdentity` of every source the
    # turn's searches returned or its plans used, as JSON, by spec_id
    # (`recorded_identity`). A record `close_turn` builds also names the
    # turn's `trace_id` and its `TurnUsage` as `usage`; a record stored
    # without them holds neither. Every key but `user`
    # and `assistant` is optional, and a record without `plans`, `readings`,
    # `specs` or `identities` holds none. Feeds the conversation digest (newest
    # window), the recall function (everything older) and the planner's ledger
    # (`prior_ledger_state`). None of them reads `identities`.
    turns: list[dict[str, Any]] = field(default_factory=list)
    # MCP calls that have timed out at least once in this conversation, keyed
    # by tool + the call's own arguments (`mcp_client._McpTimeoutError.
    # timeout_key`) -- not by tool alone, so two DIFFERENT calls to the same
    # tool timing out (the service is generally slow) reads differently from
    # the SAME call timing out twice (the wait didn't help THIS query). Not
    # reset per turn: a repeat, even turns apart, still means the earlier
    # wait bought nothing (`_stream_fatal`'s timeout handling reads and
    # updates this).
    timed_out_calls: set[str] = field(default_factory=set)
    # The user's licence to read partly reported and projected years, as
    # JSON (`planner.stated_scope.LicenceRecord`), `None` before any turn's
    # words set one. It is not reset per turn. `harness.studio` puts it into
    # each turn's graph state (`planner.stated_scope.LICENCE_STATE_KEY`), the
    # turn's read of the user's words updates it there, and
    # `harness.studio_store.keep_turn_records` keeps it back here. No model
    # reads it.
    licence: dict[str, Any] | None = None
    # The current turn's record-in-progress; reset by begin_turn.
    # `turn_user_message` is the history record's text: the message, each
    # amendment and answer, and each switch on a plan card (`add_to_turn`).
    # `turn_user_words` holds the user's own words alone, the switches left
    # out, and is what the turn's searches send Search as `user_message`.
    turn_user_message: str = ""
    turn_user_words: str = ""
    turn_final_text: str = ""
    turn_components: list[dict[str, Any]] = field(default_factory=list)
    # The turn's plan records and search readings as JSON, and the spec
    # blocks its searches returned by spec_id, each the whole value the
    # graph last wrote (`harness.graph.state.ORCHESTRATOR_PLAN_RECORDS_KEY`,
    # `ORCHESTRATOR_SEARCHES_KEY`, `ORCHESTRATOR_SPECS_KEY`), set by
    # `harness.studio` as events stream. `turn_specs` is process-local: the
    # open turn's `turn_specs` rows hold the blocks
    # (`harness.turn.store.TurnStore.specs`), so storage keeps only the
    # blocks `close_turn` keeps.
    turn_plans: list[dict[str, Any]] = field(default_factory=list)
    # Whether the turn's own words asked about projections, as the read of
    # them at the turn's opening found (`planner.stated_scope`), whatever
    # the conversation's licence carries; read only by the turn's closing
    # line (`harness.turn_records.held_back_line`).
    turn_projection_asked: bool = False
    turn_readings: list[dict[str, Any]] = field(default_factory=list)
    turn_specs: dict[str, Any] = field(default_factory=dict)
    # The chip the turn was asked to apply, `{"choice_id", "option_id"}`, set
    # by `harness.studio`; once the graph's `seed` node applied it, as that
    # node wrote it, with `from_option_id` and `spec_id`
    # (`harness.graph.state.ORCHESTRATOR_APPLIED_KEY`). `close_turn` keeps it
    # with the turn's record as `applied_choice`.
    turn_applied_choice: dict[str, Any] | None = None
    # The card a card change (`apply_unit`, `apply_change`) named and what it
    # changes, `{"component_id", "kind", "role"?}` as `harness.wire.
    # CardChangeMark` states it, set by `harness.studio`; `close_turn` keeps
    # it with the turn's record as `card_change`.
    turn_card_change: dict[str, Any] | None = None
    # The option label each scope choice of the turn's plan held on the card
    # the user first switched it from, by the row's choice id, kept from the
    # first switch of the turn on (`/interrupt`). `plan.approved` names it
    # as `changed_from` for a choice that holds another label now.
    turn_choice_baseline: dict[str, str] = field(default_factory=dict)
    # Each `plan.approved` frame the turn sent or a resume restored, one per
    # plan revision, as JSON, in the order they were first recorded
    # (`record_approved`). `close_turn` keeps them with the turn's record as
    # `approved`, and the state read hands them to a page that never read
    # the frame.
    turn_approved: list[dict[str, Any]] = field(default_factory=list)
    # The plan and plan step each frame the turn's plan runs produced came
    # from, by the frame's id (`harness.graph.state.frame_id_for`), written as
    # the `execute` node hands a run back (`harness.studio_translate`). A card
    # drawn on one of these frames names that plan step in its
    # `created_from`. Kept for the turn only, since a frame id names the frame's
    # content, which a later turn can fetch outside any plan. No model reads it.
    turn_plan_frames: dict[str, PlanLink] = field(default_factory=dict)
    turn_recorded: bool = False
    # Who asked the current turn and the specs it used; set by `begin_turn`.
    turn_usage: TurnUsage = field(default_factory=TurnUsage)
    # Whether a `remove_cards` of this turn changed the canvas since the
    # stream last reported one (`take_unreported_removal`). Not durable.
    # Reset by `begin_turn`.
    _removal_unreported: bool = field(default=False, repr=False, compare=False)

    def take_unreported_removal(self) -> bool:
        """Whether `remove_cards` changed the canvas since the last call: the
        stream then owes one `composition.updated`, which carries the
        manifest stored when it is sent."""
        unreported, self._removal_unreported = self._removal_unreported, False
        return unreported

    def _canvas_rank(self, focus: str | None = None) -> _CanvasRank:
        """Rank the conversation's cards for the canvas views by the turn that
        drew each one: a card drawn, recreated or revised in turn n belongs to
        turn n (`turns[n - 1]["cards"]`), and one the open turn touched to the
        open turn (`turn_components`). The manifest's block order plays no
        part. `focus`, when it names a
        stored card, is held apart from every group for the summary to show
        whole."""
        drawn_in: dict[str, int] = {}
        records = [record.get("cards") or [] for record in self.turns]
        if not self.turn_recorded:
            records.append(self.turn_components)
        for number, cards in enumerate(records, start=1):
            for card in cards:
                card_id = card.get("component_id") if isinstance(card, dict) else None
                if isinstance(card_id, str):
                    # Re-inserted, so the dict runs in the order cards were last touched.
                    drawn_in.pop(card_id, None)
                    drawn_in[card_id] = number
        open_turn = len(self.turns) + 1 if not self.turn_recorded else None
        composed: set[str] = set()
        pinned_ids: set[str] = set()
        if self.composition is not None:
            composed = {block.component_id for block in self.composition.blocks}
            pinned_ids = {block.component_id for block in self.composition.blocks if block.pinned}
        focused = focus if focus is not None and focus in self.components else None
        # Newest turn first, each turn's cards in the order it drew them, then
        # the cards of no recorded turn, newest stored first.
        by_turn: dict[int, list[str]] = {}
        for cid, turn in drawn_in.items():
            if cid in self.components and cid != focused:
                by_turn.setdefault(turn, []).append(cid)
        ranked = [cid for turn in sorted(by_turn, reverse=True) for cid in by_turn[turn]]
        ranked += [
            cid for cid in reversed(self.components) if cid not in drawn_in and cid != focused
        ]
        drawn_now = by_turn.get(open_turn, []) if open_turn is not None else []
        this_turn = [cid for cid in drawn_now if cid in composed]
        drawn_ids = set(drawn_now)
        earlier = [cid for cid in ranked if cid not in drawn_ids]
        off: dict[int | None, list[str]] = {}
        for cid in [*drawn_now, *earlier]:
            if cid not in composed:
                off.setdefault(drawn_in.get(cid), []).append(cid)
        return _CanvasRank(
            focus=focused,
            this_turn=this_turn,
            pinned=[cid for cid in earlier if cid in pinned_ids],
            on_canvas=[cid for cid in earlier if cid in composed and cid not in pinned_ids],
            off_canvas=list(off.items()),
            turn_of=drawn_in,
            composed=composed,
            pinned_ids=pinned_ids,
        )

    def _view_flags(self, rank: _CanvasRank, component_id: str) -> dict[str, Any]:
        """What a canvas view says of a card beside its own fields: pinned, not
        on the canvas, and the turn that drew it, each only when it applies."""
        flags: dict[str, Any] = {}
        if component_id in rank.pinned_ids:
            flags["pinned"] = True
        if component_id not in rank.composed:
            flags["on_canvas"] = False
        if component_id in rank.turn_of:
            flags["turn"] = rank.turn_of[component_id]
        return flags

    def _view_block(self, rank: _CanvasRank, component_id: str) -> dict[str, Any]:
        """One card of a canvas view in full, with its flags (`_view_flags`)."""
        return _digest_block(component_id, self.components[component_id]) | self._view_flags(
            rank, component_id
        )

    def _compact_entry(self, rank: _CanvasRank, component_id: str) -> dict[str, Any]:
        """One card of a canvas view by id, type and title, with its flags."""
        entry = self.components[component_id]
        return {
            "component_id": _cut_text(component_id, _BLOCK_ID_BYTES),
            "type": _cut_text(str(entry.get("type")), _BLOCK_TYPE_BYTES),
            "title": _cut_text(str(entry.get("title")), _TITLE_BYTE_CAP),
        } | self._view_flags(rank, component_id)

    def canvas_summary_state(self, *, focus: str | None = None) -> dict[str, Any] | None:
        """The canvas line every turn opens with: how many cards are on the
        canvas, how many of those the user pinned (when any), and how many are
        hidden. The cards themselves are a `recall(part="canvas")` away. A
        card-scoped follow-up's focus card comes whole, with its id (#348).
        None when the conversation has no cards yet. Bounded by
        `CANVAS_SUMMARY_MAX_BYTES`."""
        if self.composition is None and not self.components:
            return None
        rank = self._canvas_rank(focus)
        summary: dict[str, Any] = {"cards_on_canvas": len(rank.composed)}
        if rank.pinned_ids:
            summary["cards_pinned"] = len(rank.pinned_ids)
        summary["cards_hidden"] = sum(1 for cid in self.components if cid not in rank.composed)
        # `focus` is the client's `component_ref.component_id`, an unbounded
        # string that need not name anything, so only a stored component's id
        # is named, and cut like the block that carries it.
        if rank.focus is not None:
            summary["focus_component_id"] = _cut_text(rank.focus, _BLOCK_ID_BYTES)
            summary["blocks"] = [self._view_block(rank, rank.focus)]
        return summary

    def canvas_listing_state(self) -> dict[str, Any]:
        """`recall(part="canvas")`'s listing, in `_canvas_rank`'s order, so a
        card can be recalled or removed by id, each with its flags
        (`_view_flags`). This turn's cards on the canvas, the pinned ones and
        then the earlier cards on the canvas take the `_DIGEST_BLOCK_CAP` full
        places (`blocks`). Past them each of those groups is named compactly
        up to `_LISTING_COMPACT_CAP` (`titles`). The cards off the canvas are
        named compactly a whole turn at a time while the turns fit
        `_LISTING_COMPACT_CAP` (`off_canvas`). Every card no list holds is
        counted by the turn that drew it (`counted`: turn, on_canvas,
        off_canvas), the newest `_EARLIER_TURN_CAP` turns apart and the rest,
        older turns' cards and those of no recorded turn, in one count
        (`counted_rest`). Bounded by `CANVAS_LISTING_MAX_BYTES`."""
        rank = self._canvas_rank()
        places = _DIGEST_BLOCK_CAP
        full: list[str] = []
        titled: list[str] = []
        on_canvas = (rank.this_turn, rank.pinned, rank.on_canvas)
        for group in on_canvas:
            in_full = group[:places]
            places -= len(in_full)
            full += in_full
            titled += group[len(in_full) :][:_LISTING_COMPACT_CAP]
        listing: dict[str, Any] = {"blocks": [self._view_block(rank, cid) for cid in full]}
        if titled:
            listing["titles"] = [self._compact_entry(rank, cid) for cid in titled]
        turns_off = list(rank.off_canvas)
        listed_off: list[str] = []
        while turns_off and len(listed_off) + len(turns_off[0][1]) <= _LISTING_COMPACT_CAP:
            listed_off += turns_off.pop(0)[1]
        if listed_off:
            listing["off_canvas"] = [self._compact_entry(rank, cid) for cid in listed_off]
        named = {*full, *titled}
        counted: dict[int | None, dict[str, int]] = {}
        for cid in (cid for group in on_canvas for cid in group):
            if cid not in named:
                count = counted.setdefault(rank.turn_of.get(cid), {})
                count["on_canvas"] = count.get("on_canvas", 0) + 1
        for turn, ids in turns_off:
            count = counted.setdefault(turn, {})
            count["off_canvas"] = count.get("off_canvas", 0) + len(ids)
        turns = sorted((turn for turn in counted if turn is not None), reverse=True)
        if newest := turns[:_EARLIER_TURN_CAP]:
            listing["counted"] = [{"turn": turn} | counted[turn] for turn in newest]
        rest: dict[str, int] = {}
        for turn in [*turns[_EARLIER_TURN_CAP:], *([None] if None in counted else [])]:
            for where, number in counted[turn].items():
                rest[where] = rest.get(where, 0) + number
        if rest:
            listing["counted_rest"] = rest
        return listing

    def awaited_runs(self) -> set[str]:
        """The run ids of the stored plan records that state `EXECUTING`,
        whose status the run ledger may since have settled
        (`PlanRecord.with_run_status`)."""
        return {
            plan["run_id"]
            for turn in self.turns
            for plan in turn.get("plans") or ()
            if isinstance(plan, dict)
            and plan.get("status") == PlanStatus.EXECUTING
            and isinstance(plan.get("run_id"), str)
        }

    def conversation_digest_state(self) -> dict[str, Any] | None:
        """The newest turns as compact {user, assistant, plans, cards}
        records, the models' cross-turn memory.

        Each ask is whole and so is the newest turn's answer. An older
        turn's answer keeps the leading sentences that fit
        `_HISTORY_TEXT_CAP` characters. Each plan a turn ran is one line
        (`PlanRecord.digest_line`) cut to `_HISTORY_PLAN_CAP` characters.
        Over `CONVERSATION_DIGEST_MAX_BYTES` serialised, the digest sheds in
        a fixed order until it fits: older answers to the next of
        `_HISTORY_ANSWER_CAPS`, then the oldest turn, one at a time, then the
        newest turn's own caps, halved each try (`_history_turn`). The
        orchestrator can `recall` past all of these bounds."""
        if not self.turns:
            return None
        newest = self.turns[-_HISTORY_TURN_CAP:]

        def digest(window: list[dict[str, Any]]) -> dict[str, Any]:
            return {"turns": window, "total_turns": len(self.turns)}

        def fits(window: list[dict[str, Any]]) -> bool:
            return len(json.dumps(digest(window))) <= CONVERSATION_DIGEST_MAX_BYTES

        window: list[dict[str, Any]] = []
        for answer_cap in _HISTORY_ANSWER_CAPS:
            window = [_history_turn(turn, answer_cap=answer_cap) for turn in newest[:-1]]
            window.append(_history_turn(newest[-1]))
            if fits(window):
                return digest(window)
        while len(window) > 1 and not fits(window):
            window.pop(0)
        shrink = 0
        while not fits(window) and shrink < _HISTORY_NEWEST_TEXT_CAP.bit_length():
            shrink += 1
            window[-1] = _history_turn(newest[-1], shrink=shrink)
        return digest(window)

    def canvas_series_ids(self) -> list[str]:
        """The id of every series the cards on the canvas draw, in canvas
        order: each card's stored `_SERIES_IDS`, or for an entry stored
        without them, the ids its fact summary lists, which the platform
        caps at twelve a card."""
        ids: list[str] = []
        for block in self.composition.blocks if self.composition is not None else []:
            entry = self.components.get(block.component_id) or {}
            held = entry.get(_SERIES_IDS)
            if not isinstance(held, list):
                summary = entry.get("fact_summary")
                listed = summary.get("series") if isinstance(summary, dict) else None
                held = [
                    series["id"]
                    for series in listed or []
                    if isinstance(series, dict) and series.get("id")
                ]
            ids += [str(series_id) for series_id in held]
        return ids

    def canvas_cards(self) -> list[dict[str, Any]]:
        """The cards on the canvas that `make_charts` drew, in canvas order,
        each its id with the intent it was written from and the revision it
        was drawn as. A card the canvas holds at another revision, revised
        since it was drawn, is not listed."""
        cards: list[dict[str, Any]] = []
        for block in self.composition.blocks if self.composition is not None else []:
            drawn = (self.components.get(block.component_id) or {}).get(_DRAWN_CARD)
            if isinstance(drawn, dict) and drawn.get("revision") == block.revision:
                cards.append({"component_id": block.component_id, **drawn})
        return cards

    def recall_part(
        self,
        *,
        part: RecallPart,
        turn: int | None = None,
        component_id: str | None = None,
        runs: Mapping[str, RunStatus] | None = None,
    ) -> dict[str, Any]:
        """One earlier part of this conversation, in full: the plan records
        (`plan`) or search readings (`readings`) of turn number `turn`
        (1-based; the newest turn that has any when `turn` is None), each
        rendered to at most `_RECALL_TEXT_BUDGET` characters, a plan record's
        status read from its run in `runs`, the run ledger's statuses by run
        id (`PlanRecord.with_run_status`), the stored
        entry of the canvas component `component_id`, or, with no
        `component_id`, the canvas listing (`canvas_listing_state`). Raises
        `RecallRefusedError` naming the legal values when the target does not
        exist."""
        if part == "canvas" and component_id is None:
            return {"part": part, **self.canvas_listing_state()}
        if part == "canvas":
            entry = self.components.get(component_id or "")
            if entry is None:
                ids = list(self.components)[-_RECALL_ID_CAP:]
                raise RecallRefusedError(
                    f"no canvas component {component_id!r}", {"component_ids": ids}
                )
            composed = self.composition is not None and any(
                block.component_id == component_id for block in self.composition.blocks
            )
            return {
                "part": part,
                "component_id": component_id,
                "on_canvas": composed,
                **_model_entry(entry),
            }
        key = "plans" if part == "plan" else "readings"
        holding = [number for number, t in enumerate(self.turns, start=1) if t.get(key)]
        if turn is None and holding:
            turn = holding[-1]
        if turn not in holding:
            raise RecallRefusedError(
                f"{'no turn' if turn is None else f'turn {turn}'} of this conversation "
                f"holds a {part}",
                {"turns": holding},
            )
        record = self.turns[turn - 1]
        if part == "plan":
            plans = [
                plan.with_run_status(runs or {})
                for plan in _validated(
                    PlanRecord, record["plans"], reader="recall", what="plan record"
                )
            ]
            budget = _RECALL_TEXT_BUDGET // max(len(plans), 1)
            text = [plan.render(budget=budget) for plan in plans]
        else:
            readings = _validated(
                SearchReading, record["readings"], reader="recall", what="search reading"
            )
            text = [render_readings(readings, budget=_RECALL_TEXT_BUDGET).text]
        return {"part": part, "turn": turn, "user": record["user"], "text": text}

    def add_to_turn(self, line: str, *, user_words: bool) -> None:
        """Add `line` to the turn's record as `(then: line)`, and to
        `turn_user_words` too when the user wrote it (an amendment, an
        answer). A switch's line names an option's label, which Search would
        read as scope the user stated, so it goes in the record only."""
        self.turn_user_message += f"\n(then: {line})"
        if user_words:
            self.turn_user_words += f"\n(then: {line})"

    def turn_cards_shown(self) -> list[dict[str, Any]]:
        """The cards the open turn made that the user still sees, once each, in landing
        order: each `turn_components` entry whose block is on the canvas, or every
        entry where the conversation holds its cards off the canvas (`auto_add`)."""
        on_canvas = set(_block_ids(self.composition)) if self.auto_add else None
        shown: dict[str, dict[str, Any]] = {}
        for card in self.turn_components:
            component_id = card.get("component_id")
            if component_id and (on_canvas is None or component_id in on_canvas):
                shown.setdefault(str(component_id), card)
        return list(shown.values())

    def cards_answer(self) -> str:
        """The answer the record keeps for an open turn its cards answered, the agent
        closing with no text (`harness.studio_translate._Translator.finish`): the cards
        the user still sees (`turn_cards_shown`), by title, or by type where a card has
        none, within the conversation digest's bounds on a turn's cards
        (`_HISTORY_CARD_CAP`, `_HISTORY_CARD_TITLE_BYTE_CAP`), for a later turn's model
        to read."""
        cards = self.turn_cards_shown()
        names = [
            _cut_text(
                str(card.get("title") or f"a {card.get('type') or 'card'}"),
                _HISTORY_CARD_TITLE_BYTE_CAP,
            )
            for card in cards[:_HISTORY_CARD_CAP]
        ]
        more = len(cards) - len(names)
        listed = "; ".join(names) + (f"; and {more} more" if more else "")
        return f"Answered with the cards drawn: {listed}."

    def close_turn(
        self,
        *,
        final_text: str,
        error: str | None = None,
        match_notes: str | None = None,
        basis_line: str | None = None,
        held_line: str | None = None,
        new_question: str | None = None,
        asked_for_question: bool = False,
        reply_shown: bool = True,
        user_display: str | None = None,
        shown_answer: str | None = None,
    ) -> None:
        """Append the turn's record, once, since the resume protocol can end a
        turn from more than one code path. A turn recorded with an `error`
        drops its pending interrupts: no pause is left to answer, so
        `/interrupt` answers 404. The record names the turn's `trace_id` and
        its `usage` as they stand at the close, so the store, which writes the
        turn's `TURN_LOG_EVENT` line and `turn` outcome once the write that
        commits the record lands (`harness.studio_store.StudioStore`), names
        this turn even after `begin_turn` has moved on. A record that cannot
        be built is replaced by a minimal one (`turn`, `user`, `assistant`,
        the close's own `error` code when it has one, `trace_id` and `usage`).

        `match_notes` (`harness.turn_records.loose_match_notes`),
        `basis_line` (`harness.turn_records.basis_line`) and `held_line`
        (`harness.turn_records.held_back_line`) are the text the code
        appends to the reply after the model's `final_text`, in that order.
        The record keeps each under its own key, beside the answer, so the
        conversation digest (`_history_turn`) shows a later turn's model the
        answer alone, and that model has no text of the code's own to copy.

        `new_question` is the user's words for a turn that ended with its
        plan card set aside, since the user typed a different question on
        it. The record keeps them as `new_question`, with `ended_by` set to
        `NEW_QUESTION`, and the conversation digest tells a later turn's
        model the plan was set aside (`_history_turn`). `asked_for_question`
        marks a turn whose plan card was set aside for a new question no
        message of the user's states: its `ended_by` is `ASKED_FOR_QUESTION`.

        A turn whose stream sent no `message.complete` (`reply_shown` false)
        keeps `final_text` for later turns' models and is marked
        `reply_shown: false`, so a state read gives its reply as empty.
        `shown_answer` is the text `message.complete` began with where it differs
        from the answer a later turn's model reads, as for a turn its cards answered
        (`ConversationState.cards_answer`): a state read gives it in place of
        `final_text`.
        `user_display` is the line a state read shows as the user's message
        where the record's `user` line is written for a model, such as a
        unit change naming its spec_id (`harness.studio_edit.unit_events`)."""
        if self.turn_recorded:
            return
        record: dict[str, Any]
        try:
            record = self._turn_record(
                final_text=final_text,
                error=error,
                match_notes=match_notes,
                basis_line=basis_line,
                held_line=held_line,
            )
        except Exception as exc:
            # A record that cannot be built must not leave the turn open, since
            # every later turn would try to close it again and fail the same
            # way. The minimal record names no sources and keeps the close's
            # own code, so a turn that ended cleanly is not marked abandoned.
            logger.warning(
                "studio turn: the record of turn %d of conversation %s could not be built "
                "(%s); a minimal record with error=%s is kept",
                len(self.turns) + 1,
                self.conversation_id,
                type(exc).__name__,
                error,
                exc_info=True,
            )
            record = {
                # The turn's number as the digest counts turns.
                "turn": len(self.turns) + 1,
                "user": self.turn_user_message,
                "assistant": final_text,
            }
            if error:
                record["error"] = error
        if new_question:
            record["ended_by"] = NEW_QUESTION
            record["new_question"] = new_question
        elif asked_for_question:
            record["ended_by"] = ASKED_FOR_QUESTION
        if not reply_shown:
            record["reply_shown"] = False
        if user_display:
            record["user_display"] = user_display
        if shown_answer is not None:
            record["shown_answer"] = shown_answer
        record["trace_id"] = self.trace_id
        record["usage"] = dataclasses.asdict(self.turn_usage)
        self.turns.append(record)
        # Only once a record is appended, the minimal one included.
        self.turn_recorded = True
        if error:
            self.pending.clear()

    def _turn_record(
        self,
        *,
        final_text: str,
        error: str | None,
        match_notes: str | None,
        basis_line: str | None,
        held_line: str | None,
    ) -> dict[str, Any]:
        """The turn's full record, which `close_turn` appends. May raise, on a
        stored plan record that no longer validates, say."""
        record: dict[str, Any] = {
            "user": self.turn_user_message,
            "assistant": final_text,
        }
        if match_notes:
            record["match_notes"] = match_notes
        if basis_line:
            record["basis_line"] = basis_line
        if held_line:
            record["held_line"] = held_line
        if self.turn_components:
            record["cards"] = list(self.turn_components)
        if self.turn_approved:
            record["approved"] = list(self.turn_approved)
        if error:
            record["error"] = error
        if self.turn_plans:
            record["plans"] = list(self.turn_plans)
        if self.turn_applied_choice:
            record["applied_choice"] = dict(self.turn_applied_choice)
        if self.turn_card_change:
            record["card_change"] = dict(self.turn_card_change)
        # A source only the translator or the canvas found is read here too,
        # so a later turn's planner is shown it beside the ones a search
        # returned (`prior_ledger_state`).
        observed_at = datetime.now(UTC)
        plans = _validated(PlanRecord, self.turn_plans, reader="close turn", what="plan record")
        sourced = [
            reading for plan in plans for reading in plan.source_readings(observed_at=observed_at)
        ]
        readings = [*self.turn_readings, *(reading.model_dump(mode="json") for reading in sourced)]
        if readings:
            record["readings"] = readings
        # Keep the blocks of the sources the turn's plans used, of the source
        # each windowed copy among them copies, and of the pin each input
        # alignment moved states (`PlanSource.aligned_from`): a later turn can
        # pin or fetch that source through `earlier_block`, and of the spec each
        # card the turn drew or redrew reads when its unit can change
        # (`harness.card_unit`), which a later turn's unit change reads. A block
        # can run to about 120 KB and the record is kept for the whole
        # conversation, so no other is kept.
        identities = _turn_identities(self.turn_specs, plans)
        copied = _copied_sources(
            [source for plan in plans for source in plan.sources],
            {**self._earlier_identities(), **identities},
        )
        used = (
            {
                spec_id
                for plan in plans
                for source in plan.sources
                for spec_id in (source.spec_id, *source.aligned_from)
            }
            | copied
            | self._unit_card_specs()
        )
        specs = {spec_id: block for spec_id, block in self.turn_specs.items() if spec_id in used}
        if specs:
            record["specs"] = specs
        if identities:
            record["identities"] = {
                spec_id: recorded.model_dump(mode="json")
                for spec_id, recorded in identities.items()
            }
        return record

    def _unit_card_specs(self) -> set[str]:
        """The spec_id of each card the open turn drew or redrew whose unit
        can change (`CARD_UNIT_KEY`)."""
        specs: set[str] = set()
        for card in self.turn_components:
            entry = self.components.get(card.get("component_id") or "") or {}
            stored = entry.get(CARD_UNIT_KEY)
            if isinstance(stored, dict) and isinstance(stored.get("spec_id"), str):
                specs.add(stored["spec_id"])
        return specs

    def take_identity(
        self,
        *,
        account: str | None,
        visitor: str | None,
        caller: str | None,
        surface: str | None = None,
    ) -> None:
        """Fill in who asks the turn where it recorded no one: `caller` for a
        turn with no caller, `surface` for a turn with no surface, and
        `account` and `visitor` for a turn that names neither, which the
        turn's later segments and its outcome observation then name as their
        user (`TurnUsage.user_id`)."""
        usage = self.turn_usage
        if usage.caller is None:
            usage.caller = caller
        if usage.surface is None:
            usage.surface = surface
        if usage.account is not None or usage.visitor is not None:
            return
        usage.account, usage.visitor = account, visitor

    def note_spec_use(self, use: SpecUse, spec_id: str) -> None:
        """Add `spec_id` to the turn's `fetched_specs` or `drawn_specs`, once."""
        specs = self.turn_usage.fetched_specs if use == "fetched" else self.turn_usage.drawn_specs
        if spec_id not in specs:
            specs.append(spec_id)

    def recorded_identity(self, spec_id: str) -> RecordedIdentity | None:
        """The identity `spec_id` hashes from, as this turn or an earlier one
        recorded it. None when none did.

        This turn's are those of the spec blocks its searches have returned
        so far and of the sources its plans have used. An earlier turn's
        are the ones `close_turn` stored. Every identity hashes to its
        spec_id (`_turn_identities`), so which turn recorded it does not
        change it.
        """
        plans = _validated(PlanRecord, self.turn_plans, reader="identity", what="plan record")
        recorded = _turn_identities(self.turn_specs, plans).get(spec_id)
        if recorded is not None:
            return recorded
        for turn in reversed(self.turns):
            stored = (turn.get("identities") or {}).get(spec_id)
            if stored is not None:
                return RecordedIdentity.model_validate(stored)
        return None

    def _earlier_identities(self) -> dict[str, RecordedIdentity]:
        """The identities earlier turns stored (`close_turn`), by spec_id."""
        return {
            spec_id: RecordedIdentity.model_validate(stored)
            for turn in self.turns
            for spec_id, stored in (turn.get("identities") or {}).items()
        }

    def prior_ledger_state(self) -> dict[str, Any] | None:
        """The search readings of every earlier turn, for this turn's planner
        to render beside its own and pin from. None when no turn read any.

        `readings` holds them oldest first, each spec_id held by its newest
        reading only (`newest_per_spec_id`), as `SearchReading` JSON.
        `warnings` holds, for each spec_id a reading names as its own or as
        a candidate, the warnings of the part that named it, as `harness.
        graph.ORCHESTRATOR_SPEC_WARNINGS_KEY` holds a turn's own; the newest
        reading that names a spec_id decides. `spec_ids` lists the spec_ids
        `earlier_block` resolves. No block is included. A pin or a fetch
        reads one through `earlier_block`.
        """
        readings, blocks, warnings = self._earlier_ledger()
        if not readings:
            return None
        return {
            "readings": [r.model_dump(mode="json") for r in newest_per_spec_id(readings)],
            "warnings": warnings,
            "spec_ids": list(blocks),
        }

    def earlier_block(self, spec_id: str) -> dict[str, Any] | None:
        """The spec block an earlier turn stored for `spec_id`, decided by the
        newest reading that names it; None when that reading's turn stored
        none.

        A reading whose requested window is `latest` contributes no block,
        since the latest year may have moved and a pin of it is confirmed by
        a fresh search. A reading with no stored block, such as a source
        only the translator found or a spec_id no plan of its turn used,
        contributes none either.
        """
        return self._earlier_ledger()[1].get(spec_id)

    def _earlier_ledger(
        self,
    ) -> tuple[list[SearchReading], dict[str, Any], dict[str, list[dict[str, Any]]]]:
        """Every earlier turn's readings oldest first, and the block and
        warnings the newest reading naming each spec_id decides
        (`prior_ledger_state`, `earlier_block`)."""
        readings: list[SearchReading] = []
        blocks: dict[str, Any] = {}
        warnings: dict[str, list[dict[str, Any]]] = {}
        for turn in self.turns:
            stored = turn.get("specs") or {}
            for reading in _validated(
                SearchReading, turn.get("readings") or (), reader="ledger", what="search reading"
            ):
                readings.append(reading)
                for spec_id in named_spec_ids(reading):
                    block = stored.get(spec_id)
                    if block is None or reading.requested_window_kind == "latest":
                        blocks.pop(spec_id, None)
                    else:
                        blocks[spec_id] = block
                    warnings[spec_id] = [w.model_dump(mode="json") for w in reading.warnings]
        return readings, blocks, warnings

    @property
    def awaiting_answer(self) -> bool:
        """Whether the turn is paused on an interrupt that `/interrupt` can
        still answer. Excludes an entry already resuming under a segment: a
        turn whose client left mid-resume runs, and a retry of the same
        answer carries it on."""
        return any(p.resuming_segment is None for p in self.pending.values())

    def drop_resuming(self, segment_id: str) -> None:
        """Remove every pending entry marked as resuming under
        `segment_id`. Called at that segment's end, win or lose (done,
        stream-fatal error, cancel, reject). A crash, and a client that
        left while the segment ran, leave a resuming entry behind, for a
        retry of the same answer to carry the turn on."""
        self.pending = {
            interrupt_id: p
            for interrupt_id, p in self.pending.items()
            if p.resuming_segment != segment_id
        }

    def record_approved(self, frame: PlanApproved) -> None:
        """Keep `frame` among the turn's approved plans (`turn_approved`),
        unless the turn already holds one of its plan revision."""
        if any(kept.get("revision") == frame.revision for kept in self.turn_approved):
            return
        self.turn_approved.append(frame.model_dump(mode="json", exclude_none=True))

    def start_plan_lineage(self) -> None:
        """Forget the plan version ids given out and the choice labels the
        turn's switches kept. The next plan card is the first of a new line, as
        at a turn's start and at a second `plan` call in one turn."""
        self.plan_version_ids.clear()
        self.latest_plan_version = None
        self.turn_choice_baseline = {}

    def begin_turn(self, usage: TurnUsage | None = None, *, trace_id: str | None = None) -> None:
        """Reset per-turn wire state. Abandons any pending interrupt: a fresh
        turn.request on a paused conversation ends the pause. `usage` names
        who asks the new turn, and `trace_id` is its id, minted here when not
        given. A previous turn this closes as `abandoned`
        keeps its own trace id and usage, and its record rides the new turn's
        claim save, whose commit writes its line
        (`harness.studio_store.StudioStore.save`). It opens no span. The owner
        of the segment the claim starts opens that segment's
        (`harness.studio_stream`).

        A turn with a cursor whose stream's closing write failed is recorded
        before this, by the load that finds its cursor closed
        (`harness.studio_store.StudioStore.load`). An Edit Card turn has no
        cursor, so one whose stream sent its ending frame and whose closing
        write then failed is recorded `abandoned` here, where its stream
        said otherwise."""
        self.turn_index += 1
        self.seq = 0
        self.pending.clear()
        self.start_plan_lineage()
        self._removal_unreported = False
        if self.turn_index > 1 and not self.turn_recorded:
            # The previous turn never reached a terminal frame (an abandoned
            # pause, or a death no code path recorded). Its record keeps
            # what was asked, with the error marked as abandoned. It closes
            # before the new trace id is minted, so it names its own.
            self.close_turn(final_text=self.turn_final_text, error="abandoned")
        self.trace_id = trace_id if trace_id is not None else _wire_id("tr")
        self.turn_user_message = ""
        self.turn_user_words = ""
        self.turn_final_text = ""
        self.turn_components = []
        self.turn_plans = []
        self.turn_projection_asked = False
        self.turn_readings = []
        self.turn_specs = {}
        self.turn_applied_choice = None
        self.turn_card_change = None
        self.turn_approved = []
        self.turn_plan_frames = {}
        self.turn_recorded = False
        self.turn_usage = usage if usage is not None else TurnUsage()
        self.turn_usage.trace_id = self.trace_id
        self.turn_usage.started_at = datetime.now(UTC).isoformat()

    def next_seq(self) -> int:
        value = self.seq
        self.seq += 1
        return value


class RecallRefusedError(Exception):
    """`ConversationState.recall_part`'s target does not exist. `legal` maps
    what it can be set to, by argument, to the values it accepts."""

    def __init__(self, message: str, legal: dict[str, list[Any]]) -> None:
        super().__init__(message)
        self.legal = legal


def make_recall_part(store: StudioStore) -> Any:
    """The orchestrator's `recall` function, bound to a store and resolved to the
    conversation at CALL time through the request-scoped contextvar, because
    the workflow is built once and shared across every conversation."""

    async def recall_part(
        *, part: RecallPart, turn: int | None, component_id: str | None
    ) -> dict[str, Any]:
        key = current_conversation_key.get()
        conversation = store.current(*key) if key is not None else None
        if key is None or conversation is None:
            raise RecallRefusedError("there is no conversation to recall from", {})
        runs = await store.run_statuses(*key, conversation=conversation) if part == "plan" else {}
        return conversation.recall_part(part=part, turn=turn, component_id=component_id, runs=runs)

    return recall_part


def make_canvas_series(store: StudioStore) -> Any:
    """The callable `make_charts` and `remove_cards` read the canvas's series ids
    through (`ConversationState.canvas_series_ids`), bound to a store and
    resolved to the conversation at CALL time, as `make_recall_part` is.
    Empty when there is no conversation."""

    def canvas_series() -> list[str]:
        key = current_conversation_key.get()
        conversation = store.current(*key) if key is not None else None
        return conversation.canvas_series_ids() if conversation is not None else []

    return canvas_series


def make_canvas_cards(store: StudioStore) -> Any:
    """The callable `make_charts` reads the cards it drew on the canvas through
    (`ConversationState.canvas_cards`), bound to a store and resolved to the
    conversation at CALL time, as `make_recall_part` is. Empty when there is
    no conversation."""

    def canvas_cards() -> list[dict[str, Any]]:
        key = current_conversation_key.get()
        conversation = store.current(*key) if key is not None else None
        return conversation.canvas_cards() if conversation is not None else []

    return canvas_cards


def renew_lease_now(conversation: ConversationState) -> None:
    """Have the stream running on `conversation` renew its lease at once
    (`ConversationState.renew_now`), so the renewal reports a Stop or a lost
    lease now. A no-op when no stream runs here."""
    if conversation.renew_now is not None:
        conversation.renew_now.set()


# How a request that let go of its claim before its stream started let go
# (`record_unstreamed_span`).
type UnstreamedStatus = Literal["not_started", "failed", "cancelled"]


def record_unstreamed_span(
    conversation_id: str,
    *,
    trace_id: str,
    status: UnstreamedStatus,
    usage: TurnUsage | None = None,
) -> None:
    """Record a zero-length `turn.segment` span on `trace_id` with `status`,
    for a request that let go of its claim before its stream started:
    refused before or after its claim save committed (`not_started`, level
    `WARNING`), or failed (`failed`, level `ERROR`) or cancelled
    (`cancelled`, level `WARNING`) after it. It carries the user and tags of
    `usage` and, as every segment span, no input or output. A stream's own
    span is its owner's (`harness.studio_stream`)."""
    end_turn_segment(
        start_turn_segment(
            trace_id,
            session_id=conversation_id,
            user_id=usage.user_id if usage is not None else None,
            tags=usage.tags if usage is not None else (),
            trace_metadata=usage.trace_metadata if usage is not None else None,
        ),
        status=status,
        level="ERROR" if status == "failed" else "WARNING",
    )


def _recorded_reply(record: Mapping[str, Any]) -> str:
    """The reply a turn record holds, as its `message.complete` sent it: the answer
    the user was shown and the text the code appended to it, in that order
    (`ConversationState.close_turn`)."""
    shown = record["shown_answer"] if "shown_answer" in record else record.get("assistant")
    parts = [shown, record.get("match_notes"), record.get("basis_line"), record.get("held_line")]
    return "\n\n".join(part for part in parts if part)


def make_remove_cards(store: StudioStore) -> Any:
    """The orchestrator's `remove_cards` function, bound to a store and resolved
    to the conversation at CALL time, as `make_recall_part` is. The removal
    is written as it happens, on the canvas as stored
    (`StudioStore.change_canvas`), which runs it once per committed write.

    A `SegmentLost` or `DatabaseUnavailable` there has the stream renew its
    lease at once (`renew_lease_now`) before it propagates. Raised inside
    the tool, it reaches the model as a failed call. The renewal then ends
    the turn's segment as lost, whether it finds the lease gone or cannot
    reach the database before the lease's deadline."""

    async def remove(*, component_ids: list[str], all_cards: bool) -> CardRemoval:
        key = current_conversation_key.get()
        conversation = store.current(*key) if key is not None else None
        if key is None or conversation is None:
            return CardRemoval([], list(component_ids), [], {})

        def change(work: ConversationState) -> CardRemoval:
            return remove_cards(work, component_ids, all_cards=all_cards)

        try:
            return await store.change_canvas(*key, change)
        except (SegmentLost, DatabaseUnavailable):
            renew_lease_now(conversation)
            raise

    return remove


# ---------------------------------------------------------------------------
# Executor recording -- the progress channel
# ---------------------------------------------------------------------------


def _cut_text(text: str, cap: int) -> str:
    """`text` when its serialised form fits `cap` bytes, else a prefix ending
    in `_CUT_MARKER` that does. Empty when `cap` cannot hold the marker."""
    if len(text) < cap and len(json.dumps(text)) <= cap:
        return text
    # Each character serialises to at least one byte, and the quotes and the
    # marker take `len(_CUT_MARKER) + 2` more, so a longer prefix cannot fit.
    # Slicing first keeps a long text cheap to cut.
    kept = text[: max(cap - len(_CUT_MARKER) - 2, 0)]
    while kept and len(json.dumps(kept + _CUT_MARKER)) > cap:
        kept = kept[:-1]
    cut = kept + _CUT_MARKER
    return cut if len(json.dumps(cut)) <= cap else ""


def _subset_label(*, showing: int, total: int) -> dict[str, Any]:
    """The label a cut list carries, so a model reads `showing` items as a
    subset of `total` chosen by `_SUBSET_RULE`."""
    return {"showing": showing, "total": total, "rule": _SUBSET_RULE}


def _factored_fact_summary(summary: Any) -> dict[str, Any] | None:
    """A wire fact summary with each coordinate that every series holds at
    the same value stated once, as `shared_coordinates`. Every other
    coordinate stays on the series that holds it. The component id and
    revision, which the block carries, are left out. None for a summary that
    is not a dict whose series are dicts."""
    if not isinstance(summary, dict) or not isinstance(summary.get("series"), list):
        return None
    series = [item for item in summary["series"] if isinstance(item, dict)]
    if len(series) != len(summary["series"]):
        return None
    coordinates = [item.get("coordinates") or {} for item in series]
    first, *rest = coordinates or [{}]
    shared = {
        key: value
        for key, value in first.items()
        if all(key in other and other[key] == value for other in rest)
    }
    factored_series = []
    for item, own in zip(series, coordinates, strict=True):
        entry = {key: value for key, value in item.items() if key != "coordinates"}
        if kept := {key: value for key, value in own.items() if key not in shared}:
            entry["coordinates"] = kept
        factored_series.append(entry)
    factored = {
        key: value
        for key, value in summary.items()
        if key not in ("component_id", "revision", "series")
    }
    if shared:
        factored["shared_coordinates"] = shared
    factored["series"] = factored_series
    return factored


def _bounded_fact_summary(summary: Any) -> Any:
    """`summary`, factored where it can be (`_factored_fact_summary`), when
    that fits `_FACT_SUMMARY_BYTE_CAP`. Over it, a factored summary keeps the
    most leading series that fit, and `series_truncated` labels them a subset
    of the card's series chosen by `_SUBSET_RULE`. A summary the platform
    already cut (its own `series_truncated`) states the platform's total in
    that label. Anything else over the cap becomes a `{"cut": ...}` stub
    stating the summary's size and the limit. A summary over the cap first has
    any `unit` object over `_UNIT_OBJECT_BYTE_CAP` replaced by `_CUT_MARKER`,
    so it cannot crowd out the series."""
    factored = _factored_fact_summary(summary)
    platform_total = 0
    if factored is not None and isinstance(cut := factored.get("series_truncated"), dict):
        platform_total = cut.get("total", 0)
        factored["series_truncated"] = _subset_label(
            showing=len(factored["series"]), total=max(platform_total, len(factored["series"]))
        )
    shown = summary if factored is None else factored
    if len(json.dumps(shown)) <= _FACT_SUMMARY_BYTE_CAP:
        return shown
    if (
        isinstance(summary, dict)
        and isinstance(unit := summary.get("unit"), dict)
        and len(json.dumps(unit)) > _UNIT_OBJECT_BYTE_CAP
    ):
        return _bounded_fact_summary({**summary, "unit": _CUT_MARKER})
    if factored is not None:
        series = factored["series"]
        total = max(platform_total, len(series))

        def leading(count: int) -> dict[str, Any]:
            return {
                **factored,
                "series": series[:count],
                "series_truncated": _subset_label(showing=count, total=total),
            }

        # The largest count that fits, by bisection, since each series kept adds bytes.
        low, high = -1, len(series) - 1
        while low < high:
            middle = (low + high + 1) // 2
            if len(json.dumps(leading(middle))) <= _FACT_SUMMARY_BYTE_CAP:
                low = middle
            else:
                high = middle - 1
        if low >= 0:
            return leading(low)
    size = len(json.dumps(summary))
    return {"cut": f"a {size}-byte fact summary exceeds the {_FACT_SUMMARY_BYTE_CAP}-byte limit"}


def _digest_block(component_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    """One card of a canvas view in full: the stored entry with its id, type,
    title, fact summary and what it shows bounded. An `entities` list
    shorter than the names the card draws carries an `entities_truncated`
    label. The stored entry stays whole for `recall_part`
    and the composition "add" op. The keys `_model_entry` drops are left out."""
    block: dict[str, Any] = {
        "component_id": _cut_text(component_id, _BLOCK_ID_BYTES),
        **_model_entry(entry),
    }
    if isinstance(block.get("type"), str):
        block["type"] = _cut_text(block["type"], _BLOCK_TYPE_BYTES)
    if block.get("title") is not None:
        block["title"] = _cut_text(str(block["title"]), _TITLE_BYTE_CAP)
    if "fact_summary" in block:
        block["fact_summary"] = _bounded_fact_summary(block["fact_summary"])
    if "entities" in block:
        drawn = _entity_names(entry.get("fact_summary"))
        names = block["entities"] if isinstance(block["entities"], list) else [block["entities"]]
        block["entities"] = [
            _cut_text(str(name), _DIGEST_NAME_BYTE_CAP) for name in names[:_DIGEST_SHOW_CAP]
        ]
        total = max(len(names), len(set(drawn)))
        if total > len(block["entities"]):
            block["entities_truncated"] = _subset_label(showing=len(block["entities"]), total=total)
    if "time" in block:
        block["time"] = _cut_text(str(block["time"]), _DIGEST_TIME_BYTE_CAP)
    return block


def _history_turn(
    turn: dict[str, Any], *, answer_cap: int | None = None, shrink: int = 0
) -> dict[str, Any]:
    """One turn of the conversation digest.

    The ask is whole, and the answer whole unless `answer_cap` holds it to
    its leading sentences (`_leading_sentences`). `shrink` tightens the
    newest turn when it alone is over the digest's cap: the ask and answer
    are cut to `_HISTORY_NEWEST_TEXT_CAP` bytes and every other cap is
    halved, `shrink` times each. Past the cards' cap, `cards_omitted` counts
    the cards left out."""
    cards = turn.get("cards") or []
    kept = [
        {
            **{
                key: value
                for key, value in card.items()
                if key not in (TURN_CARD_RUNS, TURN_CARD_PROJECTED, TURN_CARD_SPECS)
            },
            **(
                {"title": _cut_text(card["title"], _HISTORY_CARD_TITLE_BYTE_CAP >> shrink)}
                if isinstance(card.get("title"), str)
                else {}
            ),
        }
        for card in cards[: _HISTORY_CARD_CAP >> shrink]
    ]
    user, assistant = turn["user"], turn["assistant"]
    if answer_cap is not None:
        assistant = _leading_sentences(assistant, answer_cap)
    if shrink:
        user = _cut_text(user, _HISTORY_NEWEST_TEXT_CAP >> shrink)
        assistant = _cut_text(assistant, _HISTORY_NEWEST_TEXT_CAP >> shrink)
    record: dict[str, Any] = {"user": user, "assistant": assistant}
    plans = [
        _cut_text(plan.digest_line(), _HISTORY_PLAN_CAP >> shrink)
        for plan in _validated(
            PlanRecord, turn.get("plans") or (), reader="conversation digest", what="plan record"
        )
    ]
    if plans:
        record["plans"] = plans
    if kept:
        record["cards"] = kept
    if len(cards) > len(kept):
        record["cards_omitted"] = len(cards) - len(kept)
    if turn.get("error"):
        record["error"] = _cut_text(turn["error"], _HISTORY_ERROR_BYTE_CAP >> shrink)
    if turn.get("ended_by") == NEW_QUESTION:
        record["ended"] = _SET_ASIDE
    elif turn.get("ended_by") == ASKED_FOR_QUESTION:
        record["ended"] = _ASKED_FOR
    return record


def _turn_identities(
    specs: dict[str, Any], plans: Iterable[PlanRecord]
) -> dict[str, RecordedIdentity]:
    """The identity of each spec block in `specs` and of each source of
    `plans`, by spec_id.

    A block is read as a source (`source_from_spec_block`) for its
    `definition.identity`, and for the identity of each measure it offers
    beside its own (`alternative_definitions`), with the block's named sets,
    so a pin of one is read from Search's answer for it. A block of that
    measure's own wins. A plan source states its identity (`PlanSource`). An
    identity data-mcp's `resolve_definition` would refuse
    (`caller_definition`: one that does not hash to its spec_id, or one past
    `MAX_IDENTITY_BYTES` as compact JSON) is left out, so each entry is one
    a later pin can resolve, and none takes more than that cap. The rest of
    a block is dropped."""
    offered: dict[str, RecordedIdentity] = {}
    found: dict[str, RecordedIdentity] = {}
    for spec_id, block in specs.items():
        if not isinstance(block, dict):
            continue
        try:
            source = source_from_spec_block(block)
        except ValueError:
            continue
        if source.identity is not None:
            found[spec_id] = RecordedIdentity(
                identity=source.identity,
                groups=source.groups,
                resolved_sources=source.resolved_sources,
            )
        for alternative, identity in alternative_definitions(block).items():
            offered.setdefault(
                alternative,
                RecordedIdentity(
                    identity=identity,
                    groups=source.groups,
                    resolved_sources=source.resolved_sources,
                ),
            )
    for plan in plans:
        for source in plan.sources:
            if source.identity is not None:
                found[source.spec_id] = RecordedIdentity(
                    identity=source.identity,
                    groups=source.groups,
                    resolved_sources=source.resolved_sources,
                )
            if source.elected_measure is not None:
                for alternative in source.elected_measure.alternatives:
                    if alternative.identity is not None:
                        offered.setdefault(
                            alternative.spec_id,
                            RecordedIdentity(
                                identity=alternative.identity,
                                groups=source.groups,
                                resolved_sources=source.resolved_sources,
                            ),
                        )
    found = {**offered, **found}
    identities: dict[str, RecordedIdentity] = {}
    for spec_id, recorded in found.items():
        try:
            caller_definition(
                {"schema_version": 1, "spec_id": spec_id, "identity": recorded.identity}
            )
        except (IdentityMismatch, IdentityUnusable):
            logger.warning("not recording the identity of %s, which does not resolve", spec_id)
            continue
        identities[spec_id] = recorded
    return identities


def _copied_sources(
    sources: Iterable[PlanSource], candidates: dict[str, RecordedIdentity]
) -> set[str]:
    """The spec_ids of `candidates` that one of `sources` copies over another window.

    A copy's identity is its source's with `requested_window` replaced
    (`data_mcp.definition_writes.windowed_identity`), so a candidate whose
    identity over the copy's window hashes to the copy's spec_id is the one
    it copies.
    """
    copied: set[str] = set()
    for source in sources:
        window = (source.identity or {}).get("requested_window")
        if not isinstance(window, dict):
            continue
        for spec_id, recorded in candidates.items():
            if (
                spec_id != source.spec_id
                and windowed_identity(recorded.identity, window)[0] == source.spec_id
            ):
                copied.add(spec_id)
                break
    return copied


def _validated[M: BaseModel](
    model: type[M], stored: Iterable[Any], *, reader: str, what: str
) -> list[M]:
    """Each of `stored` as a `model`. A record that fails validation is logged
    and skipped, so a turn stored under an older schema still reads as the
    records that do validate."""
    records: list[M] = []
    for raw in stored:
        try:
            records.append(model.model_validate(raw))
        except ValidationError:
            logger.warning("%s: skipping a stored %s", reader, what, exc_info=True)
    return records


def _leading_sentences(text: str, cap: int) -> str:
    """`text` when it is at most `cap` characters, else its leading sentences
    that fit `cap` ending in `_CUT_MARKER`, or, when its first sentence is
    already longer, a prefix cut to `cap` (`planner.plan.cut_user_text`)."""
    if len(text) <= cap:
        return text
    end = max((match.end() for match in _SENTENCE_END.finditer(text, 0, cap)), default=0)
    return text[:end] + _CUT_MARKER if end else cut_user_text(text, cap)


# The key of a stored component entry holding the id of every series the card
# draws. No digest, listing or recall states it, and `canvas_series_ids` reads it.
_SERIES_IDS = "series_ids"
# The key of a stored entry holding, for a card `make_charts` drew, the intent
# it was written from and the revision it was drawn as, so a later turn's
# `make_charts` of the same intent reads the card's figures rather than
# drawing it again while the canvas holds that revision (`canvas_cards`). No
# digest, listing or recall states it either.
_DRAWN_CARD = "drawn_card"
# The key of a stored entry holding the id of the card whose canvas slot this
# card took (`ComponentCreated.replaces`), set by the swap that landed it
# (`harness.studio_edit._swapped`). The state read gives it on the card's
# `StoredCard`. No digest, listing or recall states it.
REPLACED_CARD_KEY = "replaces"
# The key of a stored entry holding the plan and plan step whose frame the
# card draws (`harness.wire.PlanLink`), which its `component.created` names
# in `created_from`, kept so a card that takes its slot names them too
# (`harness.studio_edit._swapped`). No digest, listing or recall states it.
PLAN_LINK_KEY = "plan_link"


def _digest_entry(component: dict[str, Any]) -> dict[str, Any]:
    """The stored entry of a component: the per-component slice of the
    models' canvas views, and the ids of the series it draws
    (`drawn_series_ids`). The digest slice states what the card is and what
    it shows, as names only, no rows: the entities it draws and the span it
    covers. The resolved revision in `payload` contributes only its title
    (`_block_title`) and its series ids. The model never sees the rest of
    it."""
    return {
        "type": str(component.get("type", "")),
        "title": _block_title(component),
        "revision": int(component.get("revision") or 1),
        "fact_summary": component.get("fact_summary"),
        **_digest_shows(component),
        **drawn_series_ids(component),
    }


def drawn_series_ids(component: dict[str, Any]) -> dict[str, list[str]]:
    """The `_SERIES_IDS` member of a stored entry: the id of every series
    the resolved revision in `component`'s `payload` draws
    (`harness.reply_figures.drawn_series`). Empty for a component that
    carries no resolved revision, so its entry states no ids."""
    drawn = (component.get("payload") or {}).get("component")
    if not isinstance(drawn, dict):
        return {}
    series = drawn_series(drawn, component.get("fact_summary") or {})
    return {_SERIES_IDS: [one.id for one in series if one.id]}


def _model_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """A stored entry as a model may read it, without its `_SERIES_IDS`,
    `_DRAWN_CARD`, `CARD_UNIT_KEY`, `REPLACED_CARD_KEY` and `PLAN_LINK_KEY`."""
    return {
        key: value
        for key, value in entry.items()
        if key not in {_SERIES_IDS, _DRAWN_CARD, CARD_UNIT_KEY, REPLACED_CARD_KEY, PLAN_LINK_KEY}
    }


def _turn_component_entry(
    *,
    component_id: str,
    type: str | None,
    title: str | None,
    action: str,
    window: Any,
    runs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One `conversation.turn_components` entry, `window` included only when
    the digest entry resolved one (`_digest_shows`' `time`), and the card's
    partial and projected runs (`runs`, `card_runs`) where it states any.
    See `land_component`'s docstring for what reads it back."""
    entry: dict[str, Any] = {
        "component_id": component_id,
        "type": type,
        "title": title,
        "action": action,
    }
    if window:
        entry["window"] = window
    entry.update(runs or {})
    return entry


def card_runs(component: Mapping[str, Any], lineage: CardLineage | None) -> dict[str, Any]:
    """The keys a turn card entry records for the partial and projected runs
    of `component`, a card drawn from the frame of `lineage`: every run its
    manifests state (`TURN_CARD_RUNS`, `common.held_note.held_runs`), with
    its `source` and marked `reached` where the card's window, or its
    manifests' where it states none, reaches it (`common.held_note.reaches`,
    `drawn_window`). A run keeps its first year, which names the latest
    reported year before it. A drawn (`served`) run ends at the card's drawn
    end where that falls inside it (`_drawn_end`), as the card's label does
    (`harness.mcp_client._held_periods`), and a run left out keeps its whole
    span. The keys also record whether a manifest states a projected run at
    all (`TURN_CARD_PROJECTED`, `common.held_note.holds_projections`) and the
    spec_ids the card reads (`TURN_CARD_SPECS`). The reply's closing line and
    projection chip read them (`harness.turn_records.held_back_line`,
    `projection_chip`).

    A card with no lineage, or a lineage holding no manifest, records no
    runs, since its freshness lists runs already cut to the window it draws,
    which would word a cut run's first year as the run's. Whether its data
    holds projections is then unknown, so a card reading any spec_id records
    `TURN_CARD_PROJECTED` as `None`, and the reply does not say the data
    holds none. Any other card stating no run records none of these keys.
    """
    manifests = [member.manifest for member in lineage.lineage] if lineage is not None else []
    runs: list[dict[str, Any]] = []
    projected: bool | None = None if _card_specs(component) else False
    if manifests:
        summary = component.get("fact_summary")
        drawn = summary.get("window") if isinstance(summary, Mapping) else None
        drawn = drawn_window(drawn if isinstance(drawn, Mapping) else {}, manifests)
        periods = [(held_period(fact), fact.source) for fact in held_runs(manifests)]
        runs = [
            {
                **_drawn_end(period, drawn=drawn),
                "source": source,
                "reached": reaches(period, drawn=drawn),
            }
            for period, source in periods
        ]
        projected = any(holds_projections(manifest) for manifest in manifests)
    keys: dict[str, Any] = {}
    if runs:
        keys[TURN_CARD_RUNS] = runs
    if projected is not False:
        keys[TURN_CARD_PROJECTED] = projected
    if keys:
        keys[TURN_CARD_SPECS] = _card_specs(component)
    return keys


def _drawn_end(period: dict[str, Any], *, drawn: Mapping[str, Any]) -> dict[str, Any]:
    """`period`, a wire `HeldPeriod` dict, with a drawn (`served`) run's end
    cut to the end of `drawn`, the card's `{start, end}` window, where that
    falls inside the run."""
    last = period_year(drawn.get("end"))
    start, end = period_year(period.get("start")), period_year(period.get("end"))
    if not period.get("served") or last is None or start is None or end is None:
        return period
    return {**period, "end": str(last)} if start <= last < end else period


def _card_specs(component: Mapping[str, Any]) -> list[str]:
    """The spec_ids a component's wire provenance reads (`Provenance.inputs`)."""
    provenance = component.get("provenance")
    inputs = provenance.get("inputs") if isinstance(provenance, Mapping) else None
    return [
        str(ref["spec_id"])
        for ref in inputs or ()
        if isinstance(ref, Mapping) and ref.get("spec_id")
    ]


def _digest_shows(component: dict[str, Any]) -> dict[str, Any]:
    """The `entities` / `time` keys of a digest entry, from the component's
    wire fact summary: each series carries the entity's label and `window`
    the resolved span. The summary is decoded for its names only, and a key
    the summary cannot answer is left out. The list is capped at
    `_DIGEST_SHOW_CAP`, keeping the entry's size bounded however many series
    the underlying frame holds.
    """
    summary = component.get("fact_summary") or {}
    names = _entity_names(summary)

    shows: dict[str, Any] = {}
    if names:
        shows["entities"] = _bounded_names(names)
    window = summary.get("window") or {}
    start = window.get("start") if isinstance(window, dict) else None
    end = window.get("end") if isinstance(window, dict) else None
    if start and end:
        shows["time"] = f"{start}-{end}"
    return shows


def _entity_names(summary: Any) -> list[str]:
    """The series labels a wire fact summary draws, in display order. Empty
    for a summary that is not a dict."""
    if not isinstance(summary, dict):
        return []
    return [
        str(series["label"])
        for series in summary.get("series") or []
        if isinstance(series, dict) and series.get("label")
    ]


def _bounded_names(names: list[str]) -> list[str]:
    """Names in first-seen order, deduplicated and capped at
    `_DIGEST_SHOW_CAP`, so a stored entry's size does not follow the frame's
    series count. Deduplication runs on the full name, so two distinct labels
    that share a long prefix both survive; `_digest_block` cuts each name to
    its byte cap."""
    unique = list(dict.fromkeys(names))
    return unique[:_DIGEST_SHOW_CAP]


def _block_title(component: dict[str, Any]) -> str:
    """A block's display title, from the resolved revision's
    `component.title.text`, or the component type when that member carries
    no title (a note or a source card)."""
    title = (component.get("payload") or {}).get("component", {}).get("title")
    if isinstance(title, dict) and isinstance(title.get("text"), str):
        return title["text"]
    return component.get("type", "component")


_INHERIT: Any = object()


def _next_manifest(
    previous: CompositionManifest | None,
    blocks: list[CompositionBlock],
    *,
    actor: CompositionActor,
    layout_hints: Any = _INHERIT,
) -> CompositionManifest:
    """The next composition revision: `previous`'s own id and revision+1
    when there is a previous manifest; otherwise revision 1 under a freshly
    minted id. `layout_hints` defaults to inheriting the previous
    manifest's; pass a value (including None) to overwrite, since the user
    route must distinguish "preserve" from "clear".
    """
    if layout_hints is _INHERIT:
        layout_hints = previous.layout_hints if previous else None
    composition_id = previous.composition_id if previous else _wire_id("cps")
    return CompositionManifest(
        composition_id=composition_id,
        revision=(previous.revision + 1) if previous else 1,
        actor=actor,
        blocks=blocks,
        counts=_recount(blocks),
        layout_hints=layout_hints,
    )


def _block_ids(composition: CompositionManifest | None) -> list[str]:
    return [block.component_id for block in composition.blocks] if composition else []


def _block_index(composition: CompositionManifest | None, component_id: str) -> int | None:
    """The position of `component_id`'s block in `composition`, or `None`
    when there is no composition or no block there names it. The
    membership check `_replace_block` makes."""
    if composition is None:
        return None
    return next(
        (i for i, block in enumerate(composition.blocks) if block.component_id == component_id),
        None,
    )


def land_component(
    conversation: ConversationState,
    component: dict[str, Any],
    *,
    intent: dict[str, Any] | None = None,
    unit_card: dict[str, Any] | None = None,
    lineage: CardLineage | None = None,
    plan: PlanLink | None = None,
) -> Literal["created", "held"]:
    """Land `component` on `conversation`'s canvas. `harness.studio_translate`
    calls this at drain time for a write `make_charts` stamped through
    `_RecordingExecutor.land_component`, inside the store's canvas write
    (`StudioStore.change_canvas`), where `conversation` is a copy of the live
    state whose `composition` is the canvas as stored.

    Always records the component's digest entry
    (`conversation.components`), so a card is findable by `recall` and the
    canvas views whether or not it lands a block, and for a card
    `make_charts` wrote from `intent`, that intent with the revision it
    was drawn as (`canvas_cards`), and for a card whose unit can change, its
    `unit_card` (`harness.card_unit.UnitCard.as_json`), and for a card drawn
    on a plan step's frame, its `plan` (`PLAN_LINK_KEY`). `auto_add=False` means
    there is no block to append: the component still streams to the
    client, but nothing here composes for it, and the return is `"held"`.
    Otherwise a new block is appended and the return is `"created"`.

    Either way a `turn_components` entry records it, carrying `window` when
    the digest entry resolved one (`_digest_shows`' `time`) and the partial
    and projected runs of the card and of the manifests of its frame's
    `lineage` (`card_runs`), which the reply's closing line and projection
    chip read (`harness.turn_records.held_back_line`).
    `close_turn` folds it into `conversation.turns`, in landing order, where
    the conversation digest reads it without the runs.

    A canvas that already holds the component's block gets no second one
    and the return is `"created"`. `StudioStore.change_canvas` runs it once
    per committed write.
    """
    entry = _digest_entry(component)
    if intent is not None:
        entry[_DRAWN_CARD] = {"intent": intent, "revision": int(component["revision"])}
    if unit_card is not None:
        entry[CARD_UNIT_KEY] = unit_card
    if plan is not None:
        entry[PLAN_LINK_KEY] = dict(plan)
    conversation.components[component["component_id"]] = entry
    conversation.turn_components.append(
        _turn_component_entry(
            component_id=component["component_id"],
            type=component["type"],
            title=_block_title(component),
            action="created",
            window=entry.get("time"),
            runs=card_runs(component, lineage),
        )
    )
    if not conversation.auto_add:
        return "held"
    previous = conversation.composition
    if _block_index(previous, component["component_id"]) is not None:
        return "created"
    blocks = [block.model_copy(deep=True) for block in previous.blocks] if previous else []
    blocks.append(
        CompositionBlock(
            component_id=component["component_id"],
            revision=component["revision"],
            type=component["type"],
            title=_block_title(component),
            pinned=False,
        )
    )
    conversation.composition = _next_manifest(previous, blocks, actor=CompositionActor.MODEL)
    return "created"


class CardRemoval(NamedTuple):
    """What `remove_cards` did: the ids it took off, in canvas order; the
    named ids the conversation never held; the pinned ids it left on the
    canvas, in canvas order; and the title of each card taken off or kept."""

    removed: list[str]
    missing: list[str]
    kept_pinned: list[str]
    titles: dict[str, str]


def remove_cards(
    conversation: ConversationState, component_ids: list[str], *, all_cards: bool
) -> CardRemoval:
    """Take unpinned blocks off `conversation`'s canvas: the named ones, or
    every unpinned block when `all_cards`. A pinned block stays, since only
    the user unpins a card, and is named back in `kept_pinned` when the call
    named it or set `all_cards`. A card already off the canvas is neither
    removed nor missing. A removal marks the stream as owing a report
    (`ConversationState.take_unreported_removal`). `make_remove_cards` runs
    it inside the store's canvas write, on the canvas as stored."""
    previous = conversation.composition
    blocks_now = previous.blocks if previous is not None else []
    named = set(component_ids)
    targeted = [b for b in blocks_now if all_cards or b.component_id in named]
    removed = [b.component_id for b in targeted if not b.pinned]
    kept_pinned = [b.component_id for b in targeted if b.pinned]
    titles = {b.component_id: _cut_text(b.title, _TITLE_BYTE_CAP) for b in targeted}
    present = _block_ids(previous)
    missing = [
        cid
        for cid in dict.fromkeys(component_ids)
        if cid not in present and cid not in conversation.components
    ]
    if removed and previous is not None:
        blocks = [
            block.model_copy(deep=True)
            for block in previous.blocks
            if block.component_id not in removed
        ]
        conversation.composition = _next_manifest(previous, blocks, actor=CompositionActor.MODEL)
        conversation._removal_unreported = True
    return CardRemoval(removed, missing, kept_pinned, titles)


def _replace_block(
    conversation: ConversationState,
    *,
    old_component_id: str,
    component: dict[str, Any],
    rewrite: Callable[[str, str], str | None] | None = None,
    unit: bool = False,
) -> bool:
    """Swap the named card's block for `component`, in place. The new block
    keeps the old one's `pinned` flag and card meta (`CardMeta`, the user's
    note and labels), and a layout hint naming the old card names the new one
    in its place, so nothing on the canvas stays on the old id.

    `rewrite` rewrites the kept meta's subtitle, description and axis labels,
    each by name, for a card changed in place: a unit change names the new
    unit where they named the old (`harness.card_unit.retitled`), and a
    change of years or places drops the words that may name the old ones
    (`harness.studio_edit._changed_meta`). `unit` marks a unit change: a
    `comparison` hint, which aligns its members' units, drops the card, and a
    hint left with fewer than two members goes with it. A `row` hint keeps
    the card.

    True iff the canvas shows `component` in the slot: a block was
    replaced, or the canvas already held `component`'s block. A card that
    was never composed (auto_add=false, or removed since) has no block to
    swap. The Edit Card route's recreate and a card's unit change
    (`harness.studio_edit`) name the existing component_id whose slot the
    new component takes over, inside the store's canvas write.
    """
    previous = conversation.composition
    index = _block_index(previous, old_component_id)
    if previous is None or index is None:
        # A canvas that already shows the new block reads the swap done.
        return _block_index(previous, component["component_id"]) is not None
    new_id = component["component_id"]
    blocks = [block.model_copy(deep=True) for block in previous.blocks]
    meta = blocks[index].meta
    if meta is not None and rewrite is not None:
        meta = meta.model_copy(
            update={
                name: rewrite(name, text)
                for name in ("subtitle", "description", "x_label", "y_label")
                if isinstance(text := getattr(meta, name), str)
            }
        )
    blocks[index] = CompositionBlock(
        component_id=new_id,
        revision=int(component["revision"]),
        type=component["type"],
        title=_block_title(component),
        pinned=blocks[index].pinned,
        meta=meta,
    )
    hints = previous.layout_hints
    if hints is not None:
        hints = [
            hint.model_copy(update={"members": members})
            for hint in hints
            if len(members := _swapped_members(hint, old_component_id, new_id, unit=unit)) >= 2
        ]
    conversation.composition = _next_manifest(
        previous, blocks, actor=CompositionActor.MODEL, layout_hints=hints
    )
    return True


def _swapped_members(hint: LayoutHint, old_id: str, new_id: str, *, unit: bool) -> list[str]:
    """`hint`'s members with `old_id` named `new_id`, or left out of a
    `comparison` hint when `unit` marks a unit change."""
    if unit and hint.type is LayoutHintType.COMPARISON:
        return [member for member in hint.members if member != old_id]
    return [new_id if member == old_id else member for member in hint.members]


def _recount(blocks: list[CompositionBlock]) -> CompositionCounts:
    tallies = {"stats": 0, "charts": 0, "tables": 0, "notes": 0}
    for block in blocks:
        key = _COUNT_KEYS.get(block.type)
        if key is not None:
            tallies[key] += 1
    return CompositionCounts(all=len(blocks), **tallies)
