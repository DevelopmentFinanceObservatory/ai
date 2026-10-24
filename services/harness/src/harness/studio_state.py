"""Per-conversation studio state and the composition manifest rules.

`ConversationState` is one conversation's wire bookkeeping: frame `seq`,
pending interrupt ids, plan version lineage, the composition manifest, the
turn records, and the digests the models see. `StudioStore` keys those
states on (principal, conversation_id). The manifest helpers here are the
one place a composition changes: `_apply_operations` for the user's
mutations, `_next_manifest` for every new revision.

Every durable field of a conversation, everything but the identity field
`conversation_id` and the process-local `lock`, `cancel_event`,
`cancel_reason`, `_telemetry_span` and `_composition_ops`, lives in a
per-conversation ADK "ledger" session, the same session service the runner
already uses, keyed `(app_name, user_id=principal,
session_id=conversation_id)`. Any instance can load a conversation's ledger
and continue it; `StudioStore` keeps only runtime objects and each
instance's own live `ConversationState` in process, like
`harness.provenance.ProvenanceStore`. `create`/`load`/`save`/`release` are
the ledger-backed API every route in `harness.studio` goes through;
`current` is the held-in-process accessor for callers that run mid-turn,
after a route has already loaded or created the conversation on this
instance (`make_recall_part`, `_RecordingExecutor._keep_write_token`).

`studio:segment` is a durable lease on a conversation's active stream, the
mechanism that keeps at most one instance streaming a conversation at a
time: `claim_segment` takes it, `renew` pushes its TTL forward,
`checkpoint` reconciles the rest of the conversation's state into the
ledger mid-stream (and at end of segment, through `release`), and
`request_cancel` signals it from any instance. See `Segment`,
`SegmentBusy` and `SegmentLost`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import json
import logging
import math
import re
import time
import uuid
from collections.abc import Callable, Coroutine, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from common.telemetry import end_turn_span, start_turn_span, use_turn_span
from compiler.sources import alternative_definitions, source_from_spec_block
from data_mcp.definition_store.identity import IdentityMismatch
from data_mcp.definition_writes import IdentityUnusable, caller_definition, windowed_identity
from executor.ledger import RunStatus
from fastapi.responses import JSONResponse
from google.adk.errors import StaleSessionError
from google.adk.events import Event, EventActions
from google.adk.sessions import BaseSessionService, Session
from google.adk.sessions.base_session_service import GetSessionConfig
from planner.plan import PlanStatus, cut_user_text
from pydantic import BaseModel, ValidationError

from harness.approval import ApprovalRecord
from harness.conversation import current_conversation_key
from harness.graph.state import ORCHESTRATOR_SPECS_KEY
from harness.reply_figures import drawn_series
from harness.turn_records import (
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
    ComponentType,
    CompositionActor,
    CompositionBlock,
    CompositionCounts,
    CompositionManifest,
    ErrorCode,
    PinOp,
    RemoveOp,
    ReorderOp,
    RetryMode,
    SetCardMetaOp,
    SetLayoutHintsOp,
    ToolError,
)

logger = logging.getLogger(__name__)

# The canvas digest's bounds: the newest blocks, each with a capped id, type,
# title and fact summary, and a focus id that names a stored component. Caps are
# serialised JSON bytes (`json.dumps`, so a non-ASCII character counts as its
# escape). A cut field ends in `_CUT_MARKER`. A fact summary over its cap keeps
# its leading series in display order and says so (`_bounded_fact_summary`),
# and one that does not fit even without series is replaced by a `{"cut": ...}`
# stub. Factored (`_factored_fact_summary`), a one-series fact summary runs
# about 600 bytes and each further series adds about 150, so the cap holds a
# chart of about twenty series. Titles run about 40 bytes.
_DIGEST_BLOCK_CAP = 16
_TITLE_BYTE_CAP = 200
_FACT_SUMMARY_BYTE_CAP = 4_096
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
_DIGEST_WRAPPER_BYTES = 160
# `recall(part="canvas")`'s listing (`canvas_listing_state`): per list, the
# newest `_DIGEST_BLOCK_CAP` cards in full (with up to `_VIEW_FLAG_BYTES` of
# pinned / on_canvas flags each) and `_LISTING_COMPACT_CAP` older ones as
# compact entries, plus the keys and counts in `_LISTING_WRAPPER_BYTES`.
_LISTING_COMPACT_CAP = 48
_VIEW_FLAG_BYTES = 40
_LISTING_WRAPPER_BYTES = 256
# The conversation's cards not on the canvas (`hidden`): the newest
# `_DIGEST_HIDDEN_CAP`, each its id, type and title cut to their caps inside
# `_HIDDEN_FRAME_BYTES` of keys, plus the list's own key and cut count.
_DIGEST_HIDDEN_CAP = 16
_HIDDEN_FRAME_BYTES = 64
_HIDDEN_WRAPPER_BYTES = 64
CANVAS_DIGEST_MAX_BYTES = (
    _DIGEST_WRAPPER_BYTES
    + _DIGEST_BLOCK_CAP
    * (
        _BLOCK_FRAME_BYTES
        + _BLOCK_ID_BYTES
        + _TITLE_BYTE_CAP
        + _FACT_SUMMARY_BYTE_CAP
        + _BLOCK_SHOWS_BYTES
    )
    + _HIDDEN_WRAPPER_BYTES
    + _DIGEST_HIDDEN_CAP
    * (_HIDDEN_FRAME_BYTES + _BLOCK_ID_BYTES + _BLOCK_TYPE_BYTES + _TITLE_BYTE_CAP)
)
CANVAS_LISTING_MAX_BYTES = _LISTING_WRAPPER_BYTES + 2 * (
    _DIGEST_BLOCK_CAP
    * (
        _BLOCK_FRAME_BYTES
        + _VIEW_FLAG_BYTES
        + _BLOCK_ID_BYTES
        + _TITLE_BYTE_CAP
        + _FACT_SUMMARY_BYTE_CAP
        + _BLOCK_SHOWS_BYTES
    )
    + _LISTING_COMPACT_CAP
    * (_HIDDEN_FRAME_BYTES + _BLOCK_ID_BYTES + _BLOCK_TYPE_BYTES + _TITLE_BYTE_CAP)
)
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
    conversation: ConversationState,
    manifest: CompositionManifest | None,
    operations: list[Any],
) -> JSONResponse | tuple[list[CompositionBlock], Any]:
    """Apply mutations to LOCAL copies; the first invalid operation refuses
    the whole request with nothing applied (mutations are transactional)."""
    blocks = [block.model_copy(deep=True) for block in manifest.blocks] if manifest else []
    layout_hints = manifest.layout_hints if manifest else None
    for operation in operations:
        if isinstance(operation, SetLayoutHintsOp):
            layout_hints = operation.layout_hints
            continue
        if isinstance(operation, AddOp):
            # Delta 11: append a block for a component this conversation
            # already streamed; the server rebuilds it from its digest.
            entry = conversation.components.get(operation.component_id)
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
class _CanvasView:
    """What `ConversationState._canvas_view` decides a canvas view names."""

    ordered: list[str]
    composed: set[str]
    pinned: set[str]
    not_shown: int
    hidden: list[str]


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


@dataclass
class _PendingInterrupt:
    """One wire interrupt id, mapped back to its ADK interrupt."""

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
    # Set by `/interrupt` to the id of the segment resuming this interrupt,
    # instead of deleting the entry outright: a crash between the answer
    # being appended and the resumed segment's own checkpoint leaves a
    # durable, retryable record. Cleared by `ConversationState.drop_resuming`
    # once that segment ends, win or lose; a crashed segment leaves it set,
    # and the same answer resumes again on a fresh segment.
    resuming_segment: str | None = None
    # The decision, payload and answers `/interrupt` was called with,
    # recorded alongside `resuming_segment` so a retried resume can be
    # verified against what actually resumed.
    answer: dict[str, Any] | None = None


# A conversation's active stream may run on any instance; this is the
# lease that keeps at most one running at a time. TTL 45s means a crashed
# owner blocks the conversation for at most that long before another
# instance can take over.
SEGMENT_TTL_S = 45
# How often `StudioRunner._locked`'s heartbeat task renews the live
# segment, well inside the TTL so a slow renewal still lands before it
# lapses.
HEARTBEAT_INTERVAL_S = 5
# This process's id, carried on every segment it claims or renews for logs
# only. Fencing compares segment ids, never owners.
_OWNER_ID = uuid.uuid4().hex[:12]


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
    """Raised by `StudioStore.checkpoint` when the ledger's live segment is
    no longer the one this call is checkpointing on behalf of. Another
    instance took over after this one's TTL lapsed."""


@dataclass(frozen=True)
class Segment:
    """One durable claim on a conversation's active stream,
    `studio:segment` in the ledger. `id` is the fencing token
    `StudioStore.claim_segment` mints; every later write on behalf of this
    segment (`renew`, `checkpoint`) is fenced against it, never against
    `owner`. `owner` is the claiming instance's id, carried for logs only.
    `expires_at` is a wall-clock epoch second past which the claim is dead
    and takeable. `cancel` is set by `StudioStore.request_cancel`, from any
    instance, and observed by the owner's own heartbeat."""

    id: str
    owner: str
    expires_at: float
    cancel: bool = False


def _serialize_segment(segment: Segment) -> dict[str, Any]:
    return {
        "id": segment.id,
        "owner": segment.owner,
        "expires_at": segment.expires_at,
        "cancel": segment.cancel,
    }


def _deserialize_segment(value: Any) -> Segment | None:
    if not value:
        return None
    return Segment(
        id=value["id"],
        owner=value["owner"],
        expires_at=value["expires_at"],
        cancel=bool(value.get("cancel", False)),
    )


@dataclass
class ConversationState:
    """Wire bookkeeping for one (principal, conversation_id) pair."""

    conversation_id: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    turn_index: int = 0
    seq: int = 0
    trace_id: str = ""
    pending: dict[str, _PendingInterrupt] = field(default_factory=dict)
    plan_version_ids: dict[int, str] = field(default_factory=dict)
    latest_plan_version: int | None = None
    composition: CompositionManifest | None = None
    # The durable lease on this conversation's active stream. None when no
    # segment is live. Set by `claim_segment`, refreshed by `renew`,
    # cleared by `release`; a `checkpoint` writes whatever value it holds
    # like any other durable field.
    segment: Segment | None = None
    # Armed while a segment streams; POST /api/studio/cancel and
    # `StudioRunner._heartbeat` set it.
    cancel_event: asyncio.Event | None = None
    # Set alongside `cancel_event`, naming why it was armed: "cancelled" (a
    # client's Stop, from this instance or another) or "segment_lost" (this
    # stream's segment was taken over, or its own renewal could not keep it
    # alive). `_sse_stream`'s cancel branch reads it once to pick the
    # frame; reset to `None` alongside `cancel_event` when the stream ends.
    cancel_reason: str | None = None
    # The current turn's auto-add preference (delta 11); reset per turn.
    auto_add: bool = True
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
    # (`recorded_identity`). Every key but `user` and `assistant` is
    # optional, and a record without `plans`, `readings`, `specs` or
    # `identities` holds none. Feeds the conversation digest (newest
    # window), the recall seam (everything older) and the planner's ledger
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
    # The current turn's record-in-progress; reset by begin_turn.
    turn_user_message: str = ""
    turn_final_text: str = ""
    turn_components: list[dict[str, Any]] = field(default_factory=list)
    # The turn's plan records and search readings as JSON, and the spec
    # blocks its searches returned by spec_id, each the whole value the
    # graph last wrote (`harness.graph.state.ORCHESTRATOR_PLAN_RECORDS_KEY`,
    # `ORCHESTRATOR_SEARCHES_KEY`, `ORCHESTRATOR_SPECS_KEY`), set by
    # `harness.studio` as events stream. `turn_specs` is process-local: the
    # turn's own ADK session holds the blocks, and `StudioStore.load` reads
    # them back from it, so the ledger stores only the blocks `close_turn`
    # keeps.
    turn_plans: list[dict[str, Any]] = field(default_factory=list)
    turn_readings: list[dict[str, Any]] = field(default_factory=list)
    turn_specs: dict[str, Any] = field(default_factory=dict)
    turn_recorded: bool = False
    # The current turn's telemetry root span, opened by `begin_turn` on the
    # freshly minted `trace_id`, kept open across a pause for approval, and
    # closed by `close_turn`. Streams make it current with `in_turn_span`.
    # Excluded from repr so a log of the state doesn't dump SDK internals.
    _telemetry_span: Any = field(default=None, repr=False, compare=False)
    # This turn's composition mutations, in order, each a pure function of
    # the manifest before it to the manifest after (`land_component`,
    # `_replace_block`). Not durable: replayed against a freshly reloaded
    # manifest when a release save's `StaleSessionError` retry finds that
    # `/composition` (which runs without the turn lock) changed it in the
    # meantime, so that write is kept rather than overwritten. Reset by
    # `begin_turn`.
    _composition_ops: list[Callable[[CompositionManifest | None], CompositionManifest | None]] = (
        field(default_factory=list, repr=False, compare=False)
    )
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

    def _canvas_view(self, focus: str | None = None) -> _CanvasView:
        """The cards a canvas view names: the composed blocks the cap keeps,
        in canvas order, with the focus card kept whatever its age in place of
        the oldest; which blocks are pinned; how many composed blocks the cap
        left out; and the conversation's other cards, oldest first."""
        composed: list[str] = []
        pinned: set[str] = set()
        if self.composition is not None:
            composed = [block.component_id for block in self.composition.blocks]
            pinned = {block.component_id for block in self.composition.blocks if block.pinned}
        # Bounded: the last blocks in canvas order win on a very large canvas,
        # the rest are counted, and `_digest_block` cuts each block's fields,
        # so a view stays within `CANVAS_DIGEST_MAX_BYTES` however long the
        # conversation runs.
        ordered = composed[-_DIGEST_BLOCK_CAP:]
        if focus and focus in self.components and focus not in ordered:
            ordered = [*ordered, focus][-_DIGEST_BLOCK_CAP:]
        return _CanvasView(
            ordered=[cid for cid in ordered if cid in self.components],
            composed=set(composed),
            pinned=pinned,
            not_shown=len(set(composed) - set(ordered)),
            hidden=[cid for cid in self.components if cid not in set(composed)],
        )

    def _view_block(self, view: _CanvasView, component_id: str) -> dict[str, Any]:
        """One card of a canvas view in full; a pinned one says so, and a
        focus card off the canvas says it is not on it."""
        return (
            _digest_block(component_id, self.components[component_id])
            | ({"pinned": True} if component_id in view.pinned else {})
            | ({} if component_id in view.composed else {"on_canvas": False})
        )

    def canvas_digest_state(self, *, focus: str | None = None) -> dict[str, Any] | None:
        """The graph engine's digest: composed blocks in canvas order (plus
        the focused card for a card-scoped follow-up), titles and fact
        summaries only -- references and digests, never rows. A pinned block
        says so; a canvas whose blocks were all removed is `{"blocks": []}`,
        and None means the conversation has no canvas yet. `hidden` names the
        conversation's other cards (a new turn hid them, or they were
        removed), id, type and title only: the user can put one back."""
        view = self._canvas_view(focus)
        blocks = [self._view_block(view, cid) for cid in view.ordered]
        hidden = [
            _hidden_entry(cid, self.components[cid]) for cid in view.hidden[-_DIGEST_HIDDEN_CAP:]
        ]
        if not blocks and not hidden:
            return None if self.composition is None else {"blocks": []}
        digest: dict[str, Any] = {"blocks": blocks}
        if view.not_shown:
            digest["blocks_not_shown"] = view.not_shown
        if hidden:
            digest["hidden"] = hidden
        if len(view.hidden) > _DIGEST_HIDDEN_CAP:
            digest["hidden_not_shown"] = len(view.hidden) - _DIGEST_HIDDEN_CAP
        # `focus` is the client's `component_ref.component_id`, an unbounded
        # string that need not name anything, so only a stored component's id
        # is named, and cut like the block that carries it.
        if focus in self.components:
            digest["focus_component_id"] = _cut_text(str(focus), _BLOCK_ID_BYTES)
        return digest

    def canvas_summary_state(self, *, focus: str | None = None) -> dict[str, Any] | None:
        """The orchestrator's always-on canvas line: how many cards are on
        the canvas and how many are hidden. The cards themselves are a
        `recall(part="canvas")` away. A card-scoped follow-up's focus card
        comes whole, with its id (#348). None when the conversation has no
        cards yet."""
        if self.composition is None and not self.components:
            return None
        view = self._canvas_view(focus)
        summary: dict[str, Any] = {
            "cards_on_canvas": len(view.composed),
            "cards_hidden": len(view.hidden),
        }
        if focus in self.components:
            summary["focus_component_id"] = _cut_text(str(focus), _BLOCK_ID_BYTES)
            summary["blocks"] = [self._view_block(view, str(focus))]
        return summary

    def canvas_listing_state(self) -> dict[str, Any]:
        """`recall(part="canvas")`'s listing: the cards on the canvas, in
        canvas order, then the conversation's hidden cards. The newest
        `_DIGEST_BLOCK_CAP` of each come in full (id, type, title, fact
        summary, pinned); the next `_LISTING_COMPACT_CAP` older ones by id,
        type and title (`<list>_older`), so each can still be recalled or
        removed by id; the rest are counted (`<list>_not_shown`). Bounded by
        `CANVAS_LISTING_MAX_BYTES`."""
        view = self._canvas_view()
        composed = (
            [block.component_id for block in self.composition.blocks] if self.composition else []
        )
        listing: dict[str, Any] = {}
        for name, ids in (
            ("on_canvas", [cid for cid in composed if cid in self.components]),
            ("hidden", view.hidden),
        ):
            full = ids[-_DIGEST_BLOCK_CAP:]
            older = ids[:-_DIGEST_BLOCK_CAP][-_LISTING_COMPACT_CAP:]
            listing[name] = [self._view_block(view, cid) for cid in full]
            if older:
                listing[f"{name}_older"] = [
                    _hidden_entry(cid, self.components[cid]) for cid in older
                ]
            if rest := len(ids) - len(full) - len(older):
                listing[f"{name}_not_shown"] = rest
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

    def close_turn(self, *, final_text: str, error: str | None = None) -> None:
        """Append the turn's record, once -- the resume protocol can end a
        turn from more than one code path. The turn's telemetry span ends
        with the user's message and `final_text` as its input and output, and
        at level `ERROR` naming the `error` code when there is one."""
        if self.turn_recorded:
            return
        self.turn_recorded = True
        end_turn_span(
            self._telemetry_span,
            request=self.turn_user_message,
            response=final_text,
            error=error,
        )
        self._telemetry_span = None
        record: dict[str, Any] = {
            "user": self.turn_user_message,
            "assistant": final_text,
        }
        if self.turn_components:
            record["cards"] = list(self.turn_components)
        if error:
            record["error"] = error
        if self.turn_plans:
            record["plans"] = list(self.turn_plans)
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
        # Keep the blocks of the sources the turn's plans used and of the source
        # each windowed copy among them copies: a later turn can pin or fetch
        # that source through `earlier_block`. A block can run to about 120 KB
        # and the record is kept for the whole conversation, so no other is kept.
        identities = _turn_identities(self.turn_specs, plans)
        copied = _copied_sources(
            [source for plan in plans for source in plan.sources],
            {**self._earlier_identities(), **identities},
        )
        used = {source.spec_id for plan in plans for source in plan.sources} | copied
        specs = {spec_id: block for spec_id, block in self.turn_specs.items() if spec_id in used}
        if specs:
            record["specs"] = specs
        if identities:
            record["identities"] = {
                spec_id: recorded.model_dump(mode="json")
                for spec_id, recorded in identities.items()
            }
        self.turns.append(record)

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
        still answer. A paused turn keeps its telemetry span open until the
        resumed segment ends it, `/cancel` closes it, or the next `begin_turn`
        abandons it. Excludes an entry already resuming under a segment: a
        disconnect mid-resume closes the turn as cancelled rather than
        looking paused for `/interrupt` to answer again."""
        return any(p.resuming_segment is None for p in self.pending.values())

    def drop_resuming(self, segment_id: str) -> None:
        """Remove every pending entry marked as resuming under
        `segment_id`. Called at that segment's end, win or lose (done,
        stream-fatal error, cancel, disconnect, reject). Only a crash
        leaves a resuming entry behind, for the next segment to retry."""
        self.pending = {
            interrupt_id: p
            for interrupt_id, p in self.pending.items()
            if p.resuming_segment != segment_id
        }

    @property
    def session_id(self) -> str:
        """The current turn's ADK session id. Scoped by conversation_id so two
        conversations of one principal never share a graph session (ADK keys
        sessions on (app_name, user_id, session_id) alone)."""
        return f"{self.conversation_id}#t{self.turn_index}"

    def begin_turn(self) -> None:
        """Reset per-turn wire state. Abandons any pending interrupt: a fresh
        turn.request on a paused conversation ends the pause, and the paused
        ADK invocation stays parked in its own per-turn session."""
        self.turn_index += 1
        self.seq = 0
        self.trace_id = _wire_id("tr")
        self.pending.clear()
        self.plan_version_ids.clear()
        self.latest_plan_version = None
        self._composition_ops = []
        self._removal_unreported = False
        if self.turn_index > 1 and not self.turn_recorded:
            # The previous turn never reached a terminal frame (an abandoned
            # pause, or a death no code path recorded). Its record keeps
            # what was asked, with the error marked as abandoned.
            self.close_turn(final_text=self.turn_final_text, error="abandoned")
        self.turn_user_message = ""
        self.turn_final_text = ""
        self.turn_components = []
        self.turn_plans = []
        self.turn_readings = []
        self.turn_specs = {}
        self.turn_recorded = False
        self._telemetry_span = start_turn_span(self.trace_id, session_id=self.conversation_id)

    def ensure_turn_span(self) -> None:
        """Open a turn span on this conversation's `trace_id` when it holds
        none. `begin_turn` opens the span for the instance that starts a
        turn; an instance that resumes a paused turn through `/interrupt`
        after loading it fresh from the ledger holds no span, since
        `_telemetry_span` is process-local and never persists, so the
        resumed segment would otherwise export on no trace. A no-op once
        the instance already holds a span."""
        if self._telemetry_span is None:
            self._telemetry_span = start_turn_span(self.trace_id, session_id=self.conversation_id)

    def in_turn_span(self) -> AbstractContextManager[None]:
        """Make this turn's telemetry span current for a `with` block. Every
        stream that runs graph or executor work for the turn (the `/turn`
        stream and the `/interrupt` stream that resumes it) enters this, so
        their spans and events share the one `turn` span and its trace. A
        no-op when telemetry is off or the turn's span has closed."""
        return use_turn_span(self._telemetry_span)

    def next_seq(self) -> int:
        value = self.seq
        self.seq += 1
        return value


# ---------------------------------------------------------------------------
# Ledger-backed persistence: a conversation's durable fields, mirrored into
# a per-conversation ADK session (see the module docstring)
# ---------------------------------------------------------------------------

# Every durable ConversationState field but the growing collections
# (`turns`, `components`, `write_tokens`, `approvals`, keyed per item
# below) gets one scalar key. `studio:` namespaces the ledger session's
# state so nothing here collides with a key ADK or another caller might add.
_KEY_TURN_INDEX = "studio:turn_index"
_KEY_SEQ = "studio:seq"
_KEY_TRACE_ID = "studio:trace_id"
_KEY_PENDING = "studio:pending"
_KEY_PLAN_VERSION_IDS = "studio:plan_version_ids"
_KEY_LATEST_PLAN_VERSION = "studio:latest_plan_version"
_KEY_COMPOSITION = "studio:composition"
_KEY_SEGMENT = "studio:segment"
_KEY_AUTO_ADD = "studio:auto_add"
_KEY_TIMED_OUT_CALLS = "studio:timed_out_calls"
_KEY_TURN_USER_MESSAGE = "studio:turn_user_message"
_KEY_TURN_FINAL_TEXT = "studio:turn_final_text"
_KEY_TURN_COMPONENTS = "studio:turn_components"
_KEY_TURN_RECORDED = "studio:turn_recorded"
_KEY_TURN_PLANS = "studio:turn_plans"
_KEY_TURN_READINGS = "studio:turn_readings"

# `StudioStore.checkpoint`'s fetch-merge-append cycles, capping how long a
# conversation under sustained contention retries before it raises.
_CHECKPOINT_RETRY_ATTEMPTS = 3
# `StudioStore.renew` and `StudioStore.request_cancel`'s fetch-append
# cycles for their single-key `studio:segment` write.
_SEGMENT_WRITE_RETRY_ATTEMPTS = 3


async def _run_to_completion[T](work: Coroutine[Any, Any, T]) -> T:
    """Run `work` in a task of its own and wait for it to finish even when
    the caller is cancelled, then re-raise the caller's cancellation.

    A ledger write the stream makes from a `finally` (`release`) or from a
    task the stream cancels (`renew`, on the heartbeat) is cancelled with
    `Task.cancel`, which no anyio shield stops. Cancelled mid-statement, the
    write never lands, and an aiosqlite connection cancelled mid-statement
    can leave the database file locked."""
    task = asyncio.ensure_future(work)
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = exc
    if cancelled is not None:
        raise cancelled
    return task.result()


# `StudioStore.mutate_composition`'s fetch-rebuild-mutate-append cycles for
# its single-key `studio:composition` write.
_COMPOSITION_MUTATE_RETRY_ATTEMPTS = 3

_NO_BASELINE_VALUE = object()


def _changed_since(full: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """The subset of `full` that `baseline` either lacks or holds a
    different value for. `save` and `checkpoint` both diff a freshly
    serialized state against a baseline this way."""
    return {
        key: value for key, value in full.items() if baseline.get(key, _NO_BASELINE_VALUE) != value
    }


# Per-item keys for the collections that grow across a conversation's whole
# life: every ledger append re-stores its own state_delta as a new event
# row, so a whole-list key would re-store the entire list on every turn.
# `turns` and `approvals` are append-only, indexed by their position at
# append time; `components` and `write_tokens` are keyed by the component
# id they are already keyed by.
_TURN_ITEM_PREFIX = "studio:turn:"
_COMPONENT_ITEM_PREFIX = "studio:component:"
_WRITE_TOKEN_ITEM_PREFIX = "studio:write_token:"
_APPROVAL_ITEM_PREFIX = "studio:approval:"

# The run ledger's items (`harness.run_ledger`): one record per run, one
# idempotency-key index entry per key, one canonical submission and one set of
# sealed retrievals per run, and one compiling plan per revision.
# They live in the ledger session but are no part of `ConversationState`;
# only `StudioStore.append_run_items` writes them.
RUN_ITEM_PREFIX = "studio:run:"
RUN_KEY_ITEM_PREFIX = "studio:run_key:"
RUN_SUBMISSION_ITEM_PREFIX = "studio:run_submission:"
RUN_RETRIEVALS_ITEM_PREFIX = "studio:run_retrievals:"
RUN_PLAN_ITEM_PREFIX = "studio:run_plan:"
_RUN_ITEM_PREFIXES = (
    RUN_ITEM_PREFIX,
    RUN_KEY_ITEM_PREFIX,
    RUN_SUBMISSION_ITEM_PREFIX,
    RUN_RETRIEVALS_ITEM_PREFIX,
    RUN_PLAN_ITEM_PREFIX,
)
# `StudioStore.append_run_items`' fetch-fence-append cycles.
_RUN_ITEMS_RETRY_ATTEMPTS = 3

# Identity, the five process-local runtime objects, and the turn's spec
# blocks (`turn_specs`, read back from the turn's own ADK session by
# `StudioStore.load`), which `_serialize_durable`/`_apply_delta` skip.
_PROCESS_LOCAL_FIELDS = frozenset(
    {
        "conversation_id",
        "lock",
        "cancel_event",
        "cancel_reason",
        "_telemetry_span",
        "_composition_ops",
        "_removal_unreported",
        "turn_specs",
    }
)


def _json_safe(value: Any) -> Any:
    """`value`, proven JSON-serialisable by round-tripping it through
    `json.dumps`/`json.loads`. Raises on a value `json.dumps` itself
    refuses -- ADK's own database path would silently coerce it to its
    `repr` instead and log only a warning (`_session_util.
    make_json_safe_state`), which would hide a bad field on that path
    alone."""
    return json.loads(json.dumps(value))


def _serialize_pending(pending: dict[str, _PendingInterrupt]) -> dict[str, Any]:
    return {
        interrupt_id: {
            "adk_id": p.adk_id,
            "kind": p.kind,
            "plan_version": p.plan_version,
            "plan_revision": p.plan_revision,
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "options": dict(q.options),
                    "skippable": q.skippable,
                }
                for q in p.questions
            ],
            "skippable": p.skippable,
            "legacy": p.legacy,
            "step_order": list(p.step_order),
            "step_labels": dict(p.step_labels),
            "resuming_segment": p.resuming_segment,
            "answer": p.answer,
        }
        for interrupt_id, p in pending.items()
    }


def _deserialize_entry(entry: dict[str, Any]) -> _PendingInterrupt:
    pending = _PendingInterrupt(
        adk_id=entry["adk_id"],
        kind=entry["kind"],
        plan_version=entry.get("plan_version"),
        plan_revision=entry.get("plan_revision"),
        step_order=tuple(entry.get("step_order") or ()),
        step_labels=dict(entry.get("step_labels") or {}),
        resuming_segment=entry.get("resuming_segment"),
        answer=entry.get("answer"),
    )
    if "questions" in entry:
        pending.questions = tuple(
            _PendingQuestion(
                id=q["id"],
                question=q["question"],
                options=dict(q["options"]),
                skippable=q["skippable"],
            )
            for q in entry["questions"]
        )
        pending.skippable = entry["skippable"]
        pending.legacy = entry["legacy"]
    elif pending.kind == "elicitation":
        # Saved before questions were kept. The flat `option_ids` lift into
        # one question `q1` whose labels are unknown, and every such frame
        # was skippable as a whole.
        options: dict[str, str | None] = dict.fromkeys(entry.get("option_ids") or ())
        pending.questions = (_PendingQuestion(id="q1", question="", options=options),)
        pending.skippable = True
        pending.legacy = True
    return pending


def _deserialize_pending(data: dict[str, Any]) -> dict[str, _PendingInterrupt]:
    return {interrupt_id: _deserialize_entry(entry) for interrupt_id, entry in data.items()}


def _serialize_durable(state: ConversationState) -> dict[str, Any]:
    """Every durable field of `state`, keyed for the ledger and proven
    JSON-safe. The dict this returns is both a save's candidate
    `state_delta` (before diffing against a baseline) and the snapshot
    `save` diffs future calls against."""
    data: dict[str, Any] = {
        _KEY_TURN_INDEX: state.turn_index,
        _KEY_SEQ: state.seq,
        _KEY_TRACE_ID: state.trace_id,
        _KEY_PENDING: _serialize_pending(state.pending),
        # A JSON object's keys are always strings; `plan_version_ids`' int
        # keys are recovered on load.
        _KEY_PLAN_VERSION_IDS: {
            str(version): value for version, value in state.plan_version_ids.items()
        },
        _KEY_LATEST_PLAN_VERSION: state.latest_plan_version,
        _KEY_COMPOSITION: (
            state.composition.model_dump(mode="json") if state.composition is not None else None
        ),
        _KEY_SEGMENT: _serialize_segment(state.segment) if state.segment is not None else None,
        _KEY_AUTO_ADD: state.auto_add,
        _KEY_TIMED_OUT_CALLS: sorted(state.timed_out_calls),
        _KEY_TURN_USER_MESSAGE: state.turn_user_message,
        _KEY_TURN_FINAL_TEXT: state.turn_final_text,
        _KEY_TURN_COMPONENTS: state.turn_components,
        _KEY_TURN_RECORDED: state.turn_recorded,
        _KEY_TURN_PLANS: state.turn_plans,
        _KEY_TURN_READINGS: state.turn_readings,
    }
    for index, turn in enumerate(state.turns):
        data[f"{_TURN_ITEM_PREFIX}{index:04d}"] = turn
    for component_id, component_entry in state.components.items():
        data[f"{_COMPONENT_ITEM_PREFIX}{component_id}"] = component_entry
    for component_id, token in state.write_tokens.items():
        data[f"{_WRITE_TOKEN_ITEM_PREFIX}{component_id}"] = token
    for index, approval in enumerate(state.approvals):
        data[f"{_APPROVAL_ITEM_PREFIX}{index:04d}"] = approval.model_dump(mode="json")
    return {key: _json_safe(value) for key, value in data.items()}


def _set_list_item(items: list[Any], index: int, value: Any) -> None:
    """`items[index] = value`, extending `items` first when `index` is one
    past its end -- the shape every per-item key's index takes, since
    `turns` and `approvals` are append-only."""
    if index < len(items):
        items[index] = value
    else:
        items.extend([None] * (index - len(items)))
        items.append(value)


def _apply_delta(state: ConversationState, delta: dict[str, Any]) -> None:
    """Merge `delta` (a subset of `_serialize_durable`'s keys) onto `state`
    in place. Used both to rebuild a `ConversationState` from a ledger's
    full state and to re-apply one request's own changed keys onto a
    freshly reloaded state after a release save's `StaleSessionError`."""
    for key, value in delta.items():
        if key == _KEY_TURN_INDEX:
            state.turn_index = value
        elif key == _KEY_SEQ:
            state.seq = value
        elif key == _KEY_TRACE_ID:
            state.trace_id = value
        elif key == _KEY_PENDING:
            state.pending = _deserialize_pending(value)
        elif key == _KEY_PLAN_VERSION_IDS:
            state.plan_version_ids = {int(version): v for version, v in value.items()}
        elif key == _KEY_LATEST_PLAN_VERSION:
            state.latest_plan_version = value
        elif key == _KEY_COMPOSITION:
            state.composition = (
                CompositionManifest.model_validate(value) if value is not None else None
            )
        elif key == _KEY_SEGMENT:
            state.segment = _deserialize_segment(value)
        elif key == _KEY_AUTO_ADD:
            state.auto_add = value
        elif key == _KEY_TIMED_OUT_CALLS:
            state.timed_out_calls = set(value)
        elif key == _KEY_TURN_USER_MESSAGE:
            state.turn_user_message = value
        elif key == _KEY_TURN_FINAL_TEXT:
            state.turn_final_text = value
        elif key == _KEY_TURN_COMPONENTS:
            state.turn_components = list(value)
        elif key == _KEY_TURN_RECORDED:
            state.turn_recorded = value
        elif key == _KEY_TURN_PLANS:
            state.turn_plans = list(value)
        elif key == _KEY_TURN_READINGS:
            state.turn_readings = list(value)
        elif key.startswith(_TURN_ITEM_PREFIX):
            _set_list_item(state.turns, int(key.removeprefix(_TURN_ITEM_PREFIX)), value)
        elif key.startswith(_COMPONENT_ITEM_PREFIX):
            state.components[key.removeprefix(_COMPONENT_ITEM_PREFIX)] = value
        elif key.startswith(_WRITE_TOKEN_ITEM_PREFIX):
            state.write_tokens[key.removeprefix(_WRITE_TOKEN_ITEM_PREFIX)] = value
        elif key.startswith(_APPROVAL_ITEM_PREFIX):
            index = int(key.removeprefix(_APPROVAL_ITEM_PREFIX))
            _set_list_item(state.approvals, index, ApprovalRecord.model_validate(value))
        elif key.startswith(_RUN_ITEM_PREFIXES):
            pass  # the run ledger's, read through `StudioStore.run_items`
        else:
            logger.warning("studio ledger: ignoring unknown state key %r", key)


def _deserialize_durable(conversation_id: str, ledger_state: dict[str, Any]) -> ConversationState:
    """A fresh `ConversationState` from a ledger session's full `state`
    dict -- everything `_serialize_durable` wrote, reassembled. A key
    `_serialize_durable` never wrote (some other prefix a future version
    added) is ignored rather than failing the whole load."""
    state = ConversationState(conversation_id=conversation_id)
    _apply_delta(state, {k: v for k, v in ledger_state.items() if k.startswith("studio:")})
    return state


@dataclass
class _RuntimeEntry:
    """The process-local anchor for one (principal, conversation_id).

    `lock` serializes same-instance requests before any ledger data exists,
    including the pair racing to create a new conversation. `session` is
    the one ledger `Session` object every save on this instance goes
    through, set the first time `create` or `load` finds the conversation
    (or refreshed by `renew`/`checkpoint`/`request_cancel`'s own fetch),
    so a `StaleSessionError` means another instance, or this conversation's
    lock-free `/composition` route, wrote first. `state` is the
    `ConversationState` this instance keeps live between requests; `load`
    refreshes its durable fields in place, so `lock`, `cancel_event` and
    `_telemetry_span` keep their identity across a reload. `baseline` is
    the last full durable snapshot known to be in the ledger, the reference
    a save diffs against for a delta-only append. `composition_lock`
    serializes same-instance `/composition` calls on the shared `state`
    object, separate from `lock` since a canvas edit must land while a turn
    streams. `write_lock` serializes every ledger append this instance
    makes on this conversation: `save`, `renew`, `checkpoint`, `release`,
    `request_cancel` and `mutate_composition` all hold it from their own
    ledger fetch through the append and the `state`/`baseline` update that
    follows it, so two of this instance's own writers can never interleave
    their fetch and append and silently overwrite one another's `baseline`.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    composition_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    session: Session | None = None
    state: ConversationState | None = None
    baseline: dict[str, Any] = field(default_factory=dict)


class StudioStore:
    """Conversation state, keyed (principal, conversation_id).

    `create`/`load`/`save`/`release` are the ledger-backed API: durable
    fields round-trip through the ADK session service given at
    construction, so any instance holding it can load and continue a
    conversation another instance started. `session_service` is required --
    every conversation this store holds lives in that service's ledger, and
    a store with nowhere to persist to has nothing to fall back to.
    `claim_segment`/`renew`/`checkpoint`/`request_cancel` are the segment
    lease API (module docstring): together they keep at most one instance
    streaming a conversation at a time.

    `app_name` is the runner's app name, the same one per-turn ADK sessions
    (`{conversation_id}#t{n}`) are keyed under. The ledger keeps its own,
    derived app name (`f"{app_name}.ledger"`) so a client-chosen
    `conversation_id` can never collide with a per-turn session id that
    happens to look like `"{conversation_id}#t{n}"` -- both live in the
    same session service, but under disjoint `(app_name, user_id,
    session_id)` keyspaces.

    `clock` is a `time.time`-shaped callable, overridable so a test can
    move a segment's TTL without sleeping.
    """

    def __init__(
        self,
        session_service: BaseSessionService,
        *,
        app_name: str = "studio",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._session_service = session_service
        self._turn_app_name = app_name
        self._app_name = f"{app_name}.ledger"
        self._runtime: dict[tuple[str, str], _RuntimeEntry] = {}
        self._clock = clock

    def now(self) -> float:
        """The injected clock's current reading. Every TTL comparison this
        store makes goes through it; a caller outside the store that needs
        to compare against a segment's `expires_at` (the heartbeat's
        renewal-failure deadline) reads it through here, not the wall
        clock, so a test can move it without sleeping."""
        return self._clock()

    def current(self, principal: str, conversation_id: str) -> ConversationState | None:
        """This instance's live `ConversationState` for `(principal,
        conversation_id)`, if `create` or `load` has already fetched one --
        no ledger access. The held-in-process accessor `make_recall_part` and
        the recorder read a conversation's in-flight durable state
        through."""
        entry = self._runtime.get((principal, conversation_id))
        return entry.state if entry is not None else None

    def lock_for(self, principal: str, conversation_id: str) -> asyncio.Lock:
        """This (principal, conversation_id) pair's lock, creating its
        runtime entry on first use. Callers acquire it BEFORE `load` or
        `create` touch any durable state, so ownership settles while the
        request handler still controls the response."""
        return self._entry(principal, conversation_id).lock

    def composition_lock_for(self, principal: str, conversation_id: str) -> asyncio.Lock:
        """This (principal, conversation_id) pair's composition lock,
        creating its runtime entry on first use. `/composition` serializes
        same-instance calls on this lock, separate from `lock_for`'s turn
        lock, so a canvas edit can land while a turn streams."""
        return self._entry(principal, conversation_id).composition_lock

    def _entry(self, principal: str, conversation_id: str) -> _RuntimeEntry:
        return self._runtime.setdefault((principal, conversation_id), _RuntimeEntry())

    async def _fetch_session(self, principal: str, conversation_id: str) -> Session | None:
        """A raw, uncached read of `conversation_id`'s ledger session.
        Unlike `load`, never touches `entry.state`: `renew`, `checkpoint`
        and `request_cancel` use this to inspect the ledger's current
        truth without overwriting whatever the live stream is doing to
        the held `ConversationState` while this await is in flight."""
        return await self._session_service.get_session(
            app_name=self._app_name,
            user_id=principal,
            session_id=conversation_id,
            config=GetSessionConfig(num_recent_events=0),
        )

    async def create(self, principal: str, conversation_id: str) -> ConversationState:
        """Create `conversation_id`'s ledger session -- the only path that
        creates one (`/turn` on a conversation absent from `load`). Raises
        whatever the session service raises (`AlreadyExistsError` when
        another instance created it first) rather than papering over a
        race the caller must answer for."""
        entry = self._entry(principal, conversation_id)
        session = await self._session_service.create_session(
            app_name=self._app_name, user_id=principal, session_id=conversation_id
        )
        entry.session = session
        entry.baseline = {}
        entry.state = ConversationState(conversation_id=conversation_id)
        entry.state.lock = entry.lock
        return entry.state

    async def load(self, principal: str, conversation_id: str) -> ConversationState | None:
        """Reload `conversation_id`'s durable fields from its ledger
        session. `None` when the ledger holds nothing for it -- the store
        does not hold the conversation. Reads with `num_recent_events=0`:
        the ledger's own event history is never needed, only its merged
        `state`, on both `InMemorySessionService` and
        `DatabaseSessionService`."""
        entry = self._entry(principal, conversation_id)
        session = await self._session_service.get_session(
            app_name=self._app_name,
            user_id=principal,
            session_id=conversation_id,
            config=GetSessionConfig(num_recent_events=0),
        )
        if session is None:
            return None
        entry.session = session
        durable = _deserialize_durable(conversation_id, session.state)
        if entry.state is None:
            durable.lock = entry.lock
            entry.state = durable
        else:
            for f in dataclasses.fields(ConversationState):
                if f.name not in _PROCESS_LOCAL_FIELDS:
                    setattr(entry.state, f.name, getattr(durable, f.name))
        entry.baseline = _serialize_durable(entry.state)
        if entry.state.turn_index and not entry.state.turn_recorded:
            entry.state.turn_specs = await self._turn_specs(principal, entry.state.session_id)
        return entry.state

    async def _turn_specs(self, principal: str, session_id: str) -> dict[str, Any]:
        """The spec blocks the open turn's own ADK session holds
        (`ORCHESTRATOR_SPECS_KEY`), for `ConversationState.close_turn` to keep
        those of the sources its plans used. Empty when the session holds
        none or does not exist yet."""
        session = await self._session_service.get_session(
            app_name=self._turn_app_name,
            user_id=principal,
            session_id=session_id,
            config=GetSessionConfig(num_recent_events=0),
        )
        return dict((session.state.get(ORCHESTRATOR_SPECS_KEY) or {}) if session else {})

    def claim_segment(self, principal: str, conversation_id: str) -> str:
        """Claim a fresh segment on the LIVE state already loaded on this
        instance (`create` or `load` must have run first). Raises
        `SegmentBusy` when a segment with a still-future `expires_at` is
        already there, else writes a fresh one into `entry.state` and
        returns its id. Does no ledger I/O and does not save: the caller's
        own claim save (the `/turn` claim or the `/interrupt` save)
        persists it under the ledger's usual compare-and-swap."""
        entry = self._runtime[(principal, conversation_id)]
        assert entry.state is not None
        now = self._clock()
        live = entry.state.segment
        if live is not None and live.expires_at > now:
            raise SegmentBusy(retry_after_seconds=math.ceil(live.expires_at - now))
        segment = Segment(id=_wire_id("seg"), owner=_OWNER_ID, expires_at=now + SEGMENT_TTL_S)
        entry.state.segment = segment
        return segment.id

    async def save(self, principal: str, conversation_id: str) -> None:
        """Append this instance's changed durable fields as one ledger
        event, through the one `Session` object this runtime entry holds.
        Only the keys whose serialised value changed since the last load
        or save on this instance are sent -- a no-op event appends
        nothing. Raises `StaleSessionError` when another instance's write
        has advanced the ledger past what this instance last saw; the
        caller maps that to the existing 409 for a `/turn` claim or
        `/interrupt` (`/composition` never calls this:
        `mutate_composition` is its own fetch-rebuild-mutate-append cycle,
        immune to whatever this instance's own `entry.session` marker
        says). Held under `write_lock` like every other ledger
        append this instance makes; when this save folds the composition
        into its delta, the `_composition_ops` recorded for it are
        consumed (by prefix, since more may accumulate while this call
        awaits the append) once the append lands.
        """
        entry = self._runtime[(principal, conversation_id)]
        assert entry.session is not None and entry.state is not None
        async with entry.write_lock:
            full = _serialize_durable(entry.state)
            delta = _changed_since(full, entry.baseline)
            if not delta:
                return
            ops_folded = len(entry.state._composition_ops) if _KEY_COMPOSITION in delta else 0
            event = Event(author="studio", actions=EventActions(state_delta=delta))
            await self._session_service.append_event(entry.session, event)
            entry.baseline.update(delta)
            if ops_folded:
                del entry.state._composition_ops[:ops_folded]

    async def mutate_composition(
        self,
        principal: str,
        conversation_id: str,
        mutate: Callable[[CompositionManifest | None], CompositionManifest | None | JSONResponse],
    ) -> CompositionManifest | None | JSONResponse:
        """`/composition`'s only path to the ledger: fetch, rebuild, mutate,
        append the composition key alone, retrying the whole cycle a
        bounded number of times on a stale append. Immune to whatever
        `entry.session`'s own marker says: unlike `save`, every attempt
        fetches the ledger fresh, so a renewal this instance's own
        heartbeat made in the meantime (which advances `entry.session`
        without this route ever seeing it) cannot hide a write another
        instance made to the composition key.

        Each attempt: fetch the ledger's current `studio:composition`
        value: rebuild it as that fetched value with this instance's own
        unsaved `_composition_ops` replayed on top, assign the rebuild to
        `entry.state.composition`, and set `entry.baseline`'s composition
        entry to the FETCHED value, not the rebuild, the same
        fetch-then-diff shape `checkpoint` uses, so a later `save` or
        `checkpoint` on this instance still diffs against what the ledger
        actually holds. Then call `mutate` on the rebuild. Its return
        says what happens next:

        * The exact object passed in (a no-op): nothing to persist,
          return it.
        * A `JSONResponse` (a refusal): nothing to persist, return it.
        * Any other `CompositionManifest` (or `None`): append it under
          the fetched value's compare-and-swap. Landing cleanly assigns it
          to `entry.state.composition` and `entry.baseline`, refreshes
          `entry.session`, consumes the replayed ops by prefix (more may
          have accumulated since this attempt started its rebuild, and
          those stay for the next call), and returns it. A stale append
          retries the whole cycle from a fresh fetch.

        On every path but a stale append exhausting its retries,
        `entry.state.composition` ends the call holding the rebuild, or,
        on a landed mutation, the appended value with any ops the live turn
        recorded during the append replayed on top (they stay unsaved), not
        this instance's pre-call copy, which may already be behind the
        ledger. Raises
        `StaleSessionError` once `_COMPOSITION_MUTATE_RETRY_ATTEMPTS`
        appends in a row land stale; `entry.state.composition` is left
        holding the last rebuild, without the caller's edit. Never raises
        for a refusal or a no-op.
        """
        entry = self._runtime[(principal, conversation_id)]
        assert entry.session is not None and entry.state is not None
        last_error: StaleSessionError | None = None
        async with entry.write_lock:
            for _attempt in range(1, _COMPOSITION_MUTATE_RETRY_ATTEMPTS + 1):
                session = await self._fetch_session(principal, conversation_id)
                assert session is not None, (
                    f"conversation {conversation_id!r} vanished from the ledger mid-mutation"
                )
                fetched_raw = session.state.get(_KEY_COMPOSITION)
                fetched = CompositionManifest.model_validate(fetched_raw) if fetched_raw else None
                ops = list(entry.state._composition_ops)
                ops_folded = len(ops)
                rebuilt = fetched
                for op in ops:
                    rebuilt = op(rebuilt)
                entry.state.composition = rebuilt
                entry.baseline[_KEY_COMPOSITION] = fetched_raw
                result = mutate(rebuilt)
                if isinstance(result, JSONResponse) or result is rebuilt:
                    return result
                delta = {
                    _KEY_COMPOSITION: result.model_dump(mode="json") if result is not None else None
                }
                event = Event(author="studio", actions=EventActions(state_delta=delta))
                try:
                    await self._session_service.append_event(session, event)
                except StaleSessionError as exc:
                    last_error = exc
                    continue
                entry.session = session
                entry.baseline.update(delta)
                if ops_folded:
                    del entry.state._composition_ops[:ops_folded]
                # Ops the live turn recorded during the append stay unsaved
                # and are replayed on top, so the turn's card or removal is
                # not overwritten by this result.
                composition = result
                for op in entry.state._composition_ops:
                    composition = op(composition)
                entry.state.composition = composition
                return composition
        assert last_error is not None
        raise last_error

    async def renew(
        self, principal: str, conversation_id: str, segment_id: str, *, timeout: float = 5.0
    ) -> Literal["ok", "cancelled", "lost"]:
        """Push `segment_id`'s TTL forward by `SEGMENT_TTL_S`, writing only
        the `studio:segment` key under compare-and-swap. Fetches the
        ledger fresh each attempt (`_fetch_session`, not `load`), so it
        never disturbs whatever the live stream is doing to `entry.state`
        in the meantime, and never reconciles the rest of the conversation.
        That is `checkpoint`'s job.

        `"lost"` when the fetched segment is no longer `segment_id`
        (another instance took over after this one's TTL lapsed); the
        caller ends the stream fatally. `"cancelled"` when it is still
        `segment_id` but its `cancel` flag is set, still renewed, so the
        flag keeps being observed until the caller reacts to it. `"ok"`
        otherwise. A `StaleSessionError` still standing after
        `_SEGMENT_WRITE_RETRY_ATTEMPTS` attempts, or any other exception
        the session service raises, propagates, as does a `TimeoutError`
        when the renewal takes longer than `timeout`. The caller's own
        heartbeat decides how long to keep trying before giving up on the
        renewal altogether.

        Runs to the end even when the caller is cancelled
        (`_run_to_completion`), since the heartbeat that calls it is
        cancelled whenever its stream ends.
        """
        return await _run_to_completion(
            self._renew(principal, conversation_id, segment_id, timeout=timeout)
        )

    async def _renew(
        self, principal: str, conversation_id: str, segment_id: str, *, timeout: float
    ) -> Literal["ok", "cancelled", "lost"]:
        """`renew`'s write, run as its own task."""
        async with asyncio.timeout(timeout):
            entry = self._entry(principal, conversation_id)
            last_error: StaleSessionError | None = None
            async with entry.write_lock:
                for _attempt in range(1, _SEGMENT_WRITE_RETRY_ATTEMPTS + 1):
                    session = await self._fetch_session(principal, conversation_id)
                    segment = (
                        _deserialize_segment(session.state.get(_KEY_SEGMENT)) if session else None
                    )
                    if session is None or segment is None or segment.id != segment_id:
                        return "lost"
                    renewed = dataclasses.replace(segment, expires_at=self._clock() + SEGMENT_TTL_S)
                    event = Event(
                        author="studio",
                        actions=EventActions(
                            state_delta={_KEY_SEGMENT: _serialize_segment(renewed)}
                        ),
                    )
                    try:
                        await self._session_service.append_event(session, event)
                    except StaleSessionError as exc:
                        last_error = exc
                        continue
                    entry.session = session
                    entry.baseline[_KEY_SEGMENT] = _serialize_segment(renewed)
                    live_state = entry.state
                    if (
                        live_state is not None
                        and live_state.segment is not None
                        and live_state.segment.id == renewed.id
                    ):
                        live_state.segment = renewed
                    return "cancelled" if renewed.cancel else "ok"
            assert last_error is not None
            raise last_error

    async def checkpoint(self, principal: str, conversation_id: str, segment_id: str) -> None:
        """The reconciling full-state save: this instance's own changed
        keys, merged onto the ledger's current truth, appended as one
        event. Runs before every actionable pause frame and before `done`
        (never `message.complete`), and once more, with the segment
        cleared, at the end of the segment (`release`).

        Fences first: raises `SegmentLost` and writes nothing when the
        fetched ledger's live segment is not `segment_id` (another
        instance's claim outlived this one's TTL). Otherwise fetches the
        ledger BEFORE computing this request's changed keys, so a
        mutation the live stream makes to `entry.state` while that fetch
        is in flight is captured in the diff. A naive "diff first, then
        reload" order has the opposite bug: the reload's wholesale
        overwrite of `entry.state` would drop anything written during its
        own await. The merge and the write to `entry.state`/`baseline`
        that follow happen with no further await before the append, so
        nothing new can slip in between.

        The composition manifest merges by replaying this request's own
        recorded operations (`ConversationState._composition_ops`) onto
        the freshly fetched manifest, so a concurrent `/composition` edit
        survives; every other changed key by its own final value, since
        every durable key but the composition has exactly one writer (a
        non-composition key changed on both sides logs a warning naming
        it, and this request's value still wins). Once the append lands,
        the ops folded into it are consumed by prefix. More may have
        accumulated meanwhile, and those stay for the next checkpoint.

        Retries `_CHECKPOINT_RETRY_ATTEMPTS` times on `StaleSessionError`
        (re-fencing and re-fetching each time); a retry that stays stale
        re-raises the last one.
        """
        entry = self._runtime[(principal, conversation_id)]
        assert entry.session is not None and entry.state is not None
        last_error: StaleSessionError | None = None
        async with entry.write_lock:
            for _attempt in range(1, _CHECKPOINT_RETRY_ATTEMPTS + 1):
                session = await self._fetch_session(principal, conversation_id)
                if session is None:
                    raise SegmentLost(
                        f"conversation {conversation_id!r} no longer has a ledger session"
                    )
                fenced = _deserialize_segment(session.state.get(_KEY_SEGMENT))
                if fenced is None or fenced.id != segment_id:
                    raise SegmentLost(
                        f"conversation {conversation_id!r} segment {segment_id!r} was taken over"
                    )
                original_baseline = entry.baseline
                changed = _changed_since(_serialize_durable(entry.state), original_baseline)
                ops = list(entry.state._composition_ops)
                ops_folded = len(ops)
                reloaded = _deserialize_durable(conversation_id, session.state)
                entry.baseline = _serialize_durable(reloaded)
                if not changed:
                    # Nothing of ours to persist; still pull in whatever
                    # the ledger holds now so `entry.state` does not fall
                    # behind another writer.
                    for f in dataclasses.fields(ConversationState):
                        if f.name not in _PROCESS_LOCAL_FIELDS:
                            setattr(entry.state, f.name, getattr(reloaded, f.name))
                    return
                conflicts = sorted(
                    key
                    for key in changed
                    if key != _KEY_COMPOSITION
                    and entry.baseline.get(key) != original_baseline.get(key)
                )
                if conflicts:
                    logger.warning(
                        "studio ledger checkpoint: conversation %s had %s changed by another "
                        "writer; this request's value wins",
                        conversation_id,
                        conflicts,
                    )
                if _KEY_COMPOSITION in changed:
                    manifest = reloaded.composition
                    for op in ops:
                        manifest = op(manifest)
                    reloaded.composition = manifest
                    changed[_KEY_COMPOSITION] = (
                        manifest.model_dump(mode="json") if manifest is not None else None
                    )
                for f in dataclasses.fields(ConversationState):
                    if f.name not in _PROCESS_LOCAL_FIELDS:
                        setattr(entry.state, f.name, getattr(reloaded, f.name))
                _apply_delta(
                    entry.state,
                    {key: value for key, value in changed.items() if key != _KEY_COMPOSITION},
                )
                event = Event(author="studio", actions=EventActions(state_delta=changed))
                try:
                    await self._session_service.append_event(session, event)
                except StaleSessionError as exc:
                    last_error = exc
                    continue
                entry.session = session
                entry.baseline.update(changed)
                if ops_folded:
                    del entry.state._composition_ops[:ops_folded]
                return
        assert last_error is not None
        raise last_error

    async def release(
        self, principal: str, conversation_id: str, segment_id: str, *, timeout: float = 5.0
    ) -> None:
        """The end-of-segment save: drop this segment's still-resuming
        pending entries, clear the segment if it is still ours, then
        `checkpoint`. Runs to the end even when the caller is cancelled
        (`_run_to_completion`), since it runs in a `finally` on
        client-disconnect and generator-finaliser paths, where a disconnect
        cancels the task running it (`_unwind`'s `Task.cancel`). Cancelled
        mid-checkpoint, the segment would stay live in the ledger until its
        TTL. The timeout cancels the checkpoint once (`asyncio.timeout`).
        anyio's `move_on_after` re-cancels at every await until its scope
        exits, cutting short the session service's own rollback and
        connection close.

        Never raises anything but the caller's own cancellation: a
        `checkpoint` that cannot land (fenced out by a takeover, a DB
        error, or the timeout above) is logged and swallowed. The segment
        then simply expires by its own TTL, which is what makes it safe to
        give up here.
        """
        entry = self._runtime.get((principal, conversation_id))
        if entry is None or entry.state is None:
            return
        await _run_to_completion(
            self._save_release(entry.state, principal, conversation_id, segment_id, timeout=timeout)
        )

    async def _save_release(
        self,
        state: ConversationState,
        principal: str,
        conversation_id: str,
        segment_id: str,
        *,
        timeout: float,
    ) -> None:
        """`release`'s save, run as its own task."""
        try:
            async with asyncio.timeout(timeout):
                state.drop_resuming(segment_id)
                if state.segment is not None and state.segment.id == segment_id:
                    state.segment = None
                try:
                    await self.checkpoint(principal, conversation_id, segment_id)
                except SegmentLost:
                    logger.warning(
                        "studio ledger release: conversation %s segment %s was already "
                        "taken over; nothing saved",
                        conversation_id,
                        segment_id,
                    )
                except Exception:
                    logger.exception(
                        "studio ledger release: conversation %s failed to persist",
                        conversation_id,
                    )
        except TimeoutError:
            logger.warning(
                "studio ledger release on conversation %s did not complete within %ss",
                conversation_id,
                timeout,
            )

    async def request_cancel(self, principal: str, conversation_id: str) -> bool:
        """Signal cancellation on `conversation_id`'s live segment,
        whoever owns it, including this instance's own, which covers
        the gap between a claim landing and the stream arming its local
        `cancel_event`. Fetches the ledger fresh and writes only the
        `studio:segment` key under compare-and-swap, retrying up to
        `_SEGMENT_WRITE_RETRY_ATTEMPTS` times on `StaleSessionError`
        before re-raising the last one.

        Returns whether a live segment was found and signalled: `False`
        when there is none, or the one on record has already expired.
        The caller falls back to whatever cancellation means with no
        active segment.
        """
        entry = self._entry(principal, conversation_id)
        last_error: StaleSessionError | None = None
        async with entry.write_lock:
            for _attempt in range(1, _SEGMENT_WRITE_RETRY_ATTEMPTS + 1):
                session = await self._fetch_session(principal, conversation_id)
                segment = _deserialize_segment(session.state.get(_KEY_SEGMENT)) if session else None
                if session is None or segment is None or segment.expires_at <= self._clock():
                    return False
                if segment.cancel:
                    return True
                cancelled = dataclasses.replace(segment, cancel=True)
                event = Event(
                    author="studio",
                    actions=EventActions(state_delta={_KEY_SEGMENT: _serialize_segment(cancelled)}),
                )
                try:
                    await self._session_service.append_event(session, event)
                except StaleSessionError as exc:
                    last_error = exc
                    continue
                entry.session = session
                entry.baseline[_KEY_SEGMENT] = _serialize_segment(cancelled)
                live_state = entry.state
                if (
                    live_state is not None
                    and live_state.segment is not None
                    and live_state.segment.id == cancelled.id
                ):
                    live_state.segment = cancelled
                return True
        assert last_error is not None
        raise last_error

    async def run_items(self, principal: str, conversation_id: str) -> dict[str, Any]:
        """The run ledger's items in `conversation_id`'s ledger session, read
        fresh. Empty when the conversation has no ledger session."""
        session = await self._fetch_session(principal, conversation_id)
        if session is None:
            return {}
        return {k: v for k, v in session.state.items() if k.startswith(_RUN_ITEM_PREFIXES)}

    async def run_statuses(
        self, principal: str, conversation_id: str, *, conversation: ConversationState
    ) -> dict[str, RunStatus]:
        """The run ledger's status of each run `conversation`'s plan records
        await (`ConversationState.awaited_runs`), by run id, read off the
        stored run items without validating the rest. The ledger is read
        once, and not at all when no record awaits a run."""
        awaited = conversation.awaited_runs()
        if not awaited:
            return {}
        items = await self.run_items(principal, conversation_id)
        statuses: dict[str, RunStatus] = {}
        for run_id in awaited:
            raw = items.get(f"{RUN_ITEM_PREFIX}{run_id}")
            status = raw.get("status") if isinstance(raw, dict) else None
            if status in RunStatus:
                statuses[run_id] = RunStatus(status)
        return statuses

    async def append_run_items(
        self,
        principal: str,
        conversation_id: str,
        segment_id: str,
        items_for: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Append the run-ledger items `items_for` derives from the ledger's
        current run items, as one event fenced on `segment_id`.

        Each attempt fetches the ledger fresh and fences first, the way
        `checkpoint` does: raises `SegmentLost` and writes nothing when the
        live segment is not `segment_id`. `items_for` then sees the fetched
        run items and returns the keys to write (each under one of the
        `RUN_*_ITEM_PREFIX`es), or an empty
        dict to write nothing; whatever it raises propagates with nothing written. A stale
        append retries the whole cycle, so `items_for` decides again against
        what the other writer landed, up to `_RUN_ITEMS_RETRY_ATTEMPTS`
        times before the last `StaleSessionError` is raised.
        """
        entry = self._entry(principal, conversation_id)
        last_error: StaleSessionError | None = None
        async with entry.write_lock:
            for _attempt in range(1, _RUN_ITEMS_RETRY_ATTEMPTS + 1):
                session = await self._fetch_session(principal, conversation_id)
                if session is None:
                    raise SegmentLost(
                        f"conversation {conversation_id!r} no longer has a ledger session"
                    )
                fenced = _deserialize_segment(session.state.get(_KEY_SEGMENT))
                if fenced is None or fenced.id != segment_id:
                    raise SegmentLost(
                        f"conversation {conversation_id!r} segment {segment_id!r} was taken over"
                    )
                current = {
                    k: v for k, v in session.state.items() if k.startswith(_RUN_ITEM_PREFIXES)
                }
                delta = items_for(current)
                if not delta:
                    return
                outside = sorted(key for key in delta if not key.startswith(_RUN_ITEM_PREFIXES))
                if outside:
                    raise ValueError(f"run ledger items outside its prefixes: {outside}")
                event = Event(
                    author="studio",
                    actions=EventActions(
                        state_delta={key: _json_safe(value) for key, value in delta.items()}
                    ),
                )
                try:
                    await self._session_service.append_event(session, event)
                except StaleSessionError as exc:
                    last_error = exc
                    continue
                entry.session = session
                return
        assert last_error is not None
        raise last_error


class RecallRefusedError(Exception):
    """`ConversationState.recall_part`'s target does not exist. `legal` maps
    what it can be set to, by argument, to the values it accepts."""

    def __init__(self, message: str, legal: dict[str, list[Any]]) -> None:
        super().__init__(message)
        self.legal = legal


def make_recall_part(store: StudioStore) -> Any:
    """The orchestrator's `recall` seam, bound to a store and resolved to the
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
    """The seam `make_charts` and `remove_cards` read the canvas's series ids
    through (`ConversationState.canvas_series_ids`), bound to a store and
    resolved to the conversation at CALL time, as `make_recall_part` is.
    Empty when there is no conversation."""

    def canvas_series() -> list[str]:
        key = current_conversation_key.get()
        conversation = store.current(*key) if key is not None else None
        return conversation.canvas_series_ids() if conversation is not None else []

    return canvas_series


def make_canvas_cards(store: StudioStore) -> Any:
    """The seam `make_charts` reads the cards it drew on the canvas through
    (`ConversationState.canvas_cards`), bound to a store and resolved to the
    conversation at CALL time, as `make_recall_part` is. Empty when there is
    no conversation."""

    def canvas_cards() -> list[dict[str, Any]]:
        key = current_conversation_key.get()
        conversation = store.current(*key) if key is not None else None
        return conversation.canvas_cards() if conversation is not None else []

    return canvas_cards


def make_remove_cards(store: StudioStore) -> Any:
    """The orchestrator's `remove_cards` seam, bound to a store and resolved
    to the conversation at CALL time, as `make_recall_part` is."""

    async def remove(*, component_ids: list[str], all_cards: bool) -> tuple[list[str], list[str]]:
        key = current_conversation_key.get()
        conversation = store.current(*key) if key is not None else None
        if conversation is None:
            return [], list(component_ids)
        return remove_cards(conversation, component_ids, all_cards=all_cards)

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
    stating the summary's size and the limit."""
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
    """One block of the canvas digest: the stored entry with its id, type,
    title, fact summary and what it shows bounded. An `entities` list
    shorter than the names the card draws carries an `entities_truncated`
    label. The stored entry stays whole for `recall_part`
    and the composition "add" op. Its `_SERIES_IDS` and `_DRAWN_CARD` are
    left out."""
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


def _hidden_entry(component_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    """A card of the conversation that is not on the canvas, as the digest
    names it: its id, type and title, each cut to its cap."""
    return {
        "component_id": _cut_text(component_id, _BLOCK_ID_BYTES),
        "type": _cut_text(str(entry.get("type")), _BLOCK_TYPE_BYTES),
        "title": _cut_text(str(entry.get("title")), _TITLE_BYTE_CAP),
    }


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
            **card,
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


def _digest_entry(component: dict[str, Any]) -> dict[str, Any]:
    """The stored entry of a component: the per-component slice of the
    models' canvas digest, and the ids of the series it draws
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
    """A stored entry as a model may read it, without its `_SERIES_IDS` and
    `_DRAWN_CARD`."""
    return {key: value for key, value in entry.items() if key not in {_SERIES_IDS, _DRAWN_CARD}}


def _turn_component_entry(
    *, component_id: str, type: str | None, title: str | None, action: str, window: Any
) -> dict[str, Any]:
    """One `conversation.turn_components` entry, `window` included only when
    the digest entry resolved one (`_digest_shows`' `time`). See
    `land_component`'s docstring for what reads it back."""
    entry: dict[str, Any] = {
        "component_id": component_id,
        "type": type,
        "title": title,
        "action": action,
    }
    if window:
        entry["window"] = window
    return entry


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
    composition_id: str | None = None,
    layout_hints: Any = _INHERIT,
) -> CompositionManifest:
    """The next composition revision: `previous`'s own id and revision+1
    when there is a previous manifest; otherwise revision 1 under
    `composition_id`, or a freshly minted one when the caller passes none,
    the ordinary case for a one-shot call building a manifest that has no
    earlier state to carry an id from. `layout_hints` defaults to
    inheriting the previous manifest's; pass a value (including None) to
    overwrite, since the user route must distinguish "preserve" from
    "clear".

    A recorded composition op that can replay with `previous=None`
    (`_append_block`) passes the id it minted the first time it ran live
    instead of leaving this default, so the replay reproduces the exact
    manifest it built then, without minting a second, different one.
    """
    if layout_hints is _INHERIT:
        layout_hints = previous.layout_hints if previous else None
    composition_id = previous.composition_id if previous else (composition_id or _wire_id("cps"))
    return CompositionManifest(
        composition_id=composition_id,
        revision=(previous.revision + 1) if previous else 1,
        actor=actor,
        blocks=blocks,
        counts=_recount(blocks),
        layout_hints=layout_hints,
    )


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
) -> Literal["created", "held"]:
    """Land `component` on `conversation`'s canvas. `harness.studio_translate`
    calls this at drain time for a write `make_charts` stamped through
    `_RecordingExecutor.land_component`.

    Always records the component's digest entry
    (`conversation.components`), so a card is findable by `recall` and the
    canvas digest whether or not it lands a block, and for a card
    `make_charts` wrote from `intent`, that intent with the revision it
    was drawn as (`canvas_cards`). `auto_add=False` means
    there is no block to append: the component still streams to the
    client, but nothing here composes for it, and the return is `"held"`.
    Otherwise a new block is appended and the return is `"created"`.

    Either way a `turn_components` entry records it, carrying `window` when
    the digest entry resolved one (`_digest_shows`' `time`). `close_turn`
    folds it into `conversation.turns`, in landing order, where the
    conversation digest reads it.
    """
    entry = _digest_entry(component)
    if intent is not None:
        entry[_DRAWN_CARD] = {"intent": intent, "revision": int(component["revision"])}
    conversation.components[component["component_id"]] = entry
    conversation.turn_components.append(
        _turn_component_entry(
            component_id=component["component_id"],
            type=component["type"],
            title=_block_title(component),
            action="created",
            window=entry.get("time"),
        )
    )
    if not conversation.auto_add:
        return "held"
    # Minted here, once, live (never inside `_append_block` itself) and
    # bound into the op, so a later replay of that op against a freshly
    # reloaded manifest that is ALSO still None (the ledger has not
    # seen this composition either) reproduces this exact id, not a
    # second, different one.
    composition_id = (
        conversation.composition.composition_id
        if conversation.composition is not None
        else _wire_id("cps")
    )
    op = functools.partial(
        _append_block,
        component_id=component["component_id"],
        revision=component["revision"],
        type=component["type"],
        title=_block_title(component),
        composition_id=composition_id,
    )
    conversation.composition = op(conversation.composition)
    conversation._composition_ops.append(op)
    return "created"


def remove_cards(
    conversation: ConversationState, component_ids: list[str], *, all_cards: bool
) -> tuple[list[str], list[str]]:
    """Take blocks off `conversation`'s canvas: the named ones, or every
    block when `all_cards`. Returns the ids removed, in canvas order, and the
    named ids the conversation never held; a named card already off the
    canvas is neither. Recorded as a replayable op, like `_append_block`, so
    a concurrent `/composition` reconciles with it. The stream is told a
    removal is unreported (`ConversationState.take_unreported_removal`)."""
    previous = conversation.composition
    present = [block.component_id for block in previous.blocks] if previous else []
    named = set(component_ids)
    removed = [cid for cid in present if all_cards or cid in named]
    missing = [
        cid for cid in component_ids if cid not in present and cid not in conversation.components
    ]
    if removed:
        op = functools.partial(_remove_blocks, component_ids=frozenset(removed))
        conversation.composition = op(previous)
        conversation._composition_ops.append(op)
        conversation._removal_unreported = True
    return removed, missing


def _remove_blocks(
    previous: CompositionManifest | None, *, component_ids: frozenset[str]
) -> CompositionManifest | None:
    """The next manifest without `component_ids`' blocks, or `previous`
    unchanged when it holds none of them (a replay after `/composition`
    removed them first)."""
    if previous is None or not any(b.component_id in component_ids for b in previous.blocks):
        return previous
    blocks = [
        block.model_copy(deep=True)
        for block in previous.blocks
        if block.component_id not in component_ids
    ]
    return _next_manifest(previous, blocks, actor=CompositionActor.MODEL)


def _append_block(
    previous: CompositionManifest | None,
    *,
    component_id: str,
    revision: int,
    type: str,
    title: str,
    composition_id: str,
) -> CompositionManifest:
    """The next manifest with a new block for `component_id` appended.

    A pure function of `previous` and `composition_id` (`land_component`'s
    docstring explains why the id is bound here, not minted), so it
    doubles as a release save's stale-retry replay step: called again
    against a freshly reloaded manifest, it reproduces this append on top
    of whatever `/composition` changed in between, and, when that manifest
    is still None, the exact one this call built the first time.
    """
    blocks = [block.model_copy(deep=True) for block in previous.blocks] if previous else []
    blocks.append(
        CompositionBlock(
            component_id=component_id, revision=revision, type=type, title=title, pinned=False
        )
    )
    return _next_manifest(
        previous, blocks, actor=CompositionActor.MODEL, composition_id=composition_id
    )


def _replace_block(
    conversation: ConversationState,
    *,
    old_component_id: str,
    component: dict[str, Any],
) -> bool:
    """Swap the named card's block for `component`, in place.

    True iff a block was replaced. A card that was never composed
    (auto_add=false, or removed since) has no block to swap: the caller's
    component still streams as component.created, but must not sneak onto
    the canvas here. The Edit Card route's recreate (`harness.studio_edit`)
    names the existing component_id whose slot the new component takes
    over.
    """
    previous = conversation.composition
    index = _block_index(previous, old_component_id)
    if previous is None or index is None:
        return False
    op = functools.partial(
        _replace_block_at,
        old_component_id=old_component_id,
        component_id=component["component_id"],
        revision=int(component["revision"]),
        type=component["type"],
        title=_block_title(component),
    )
    conversation.composition = op(previous)
    conversation._composition_ops.append(op)
    return True


def _replace_block_at(
    previous: CompositionManifest | None,
    *,
    old_component_id: str,
    component_id: str,
    revision: int,
    type: str,
    title: str,
) -> CompositionManifest | None:
    """The next manifest with `old_component_id`'s block swapped for the
    named component, or `previous` unchanged when it no longer has one (a
    release save's replay against a freshly reloaded manifest where
    `/composition` removed the target in the meantime finds nothing to
    swap, the same outcome `_replace_block` itself gives up front).

    A pure function of `previous`, replayed the same way `_append_block`
    is. Reads the swapped block's `pinned` flag off `previous` rather than
    carrying the flag captured at the original call, so a pin `/composition`
    applied in the meantime survives the replay.
    """
    index = _block_index(previous, old_component_id)
    if previous is None or index is None:
        return previous
    blocks = [block.model_copy(deep=True) for block in previous.blocks]
    blocks[index] = CompositionBlock(
        component_id=component_id,
        revision=revision,
        type=type,
        title=title,
        pinned=blocks[index].pinned,
    )
    return _next_manifest(previous, blocks, actor=CompositionActor.MODEL)


def _recount(blocks: list[CompositionBlock]) -> CompositionCounts:
    tallies = {"stats": 0, "charts": 0, "tables": 0, "notes": 0}
    for block in blocks:
        key = _COUNT_KEYS.get(block.type)
        if key is not None:
            tallies[key] += 1
    return CompositionCounts(all=len(blocks), **tallies)
