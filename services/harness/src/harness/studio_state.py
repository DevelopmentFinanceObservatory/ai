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
instance (`make_recall`, `_RecordingExecutor.components_replaced`).

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
import time
import uuid
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Literal

import anyio
from common.telemetry import end_turn_span, start_turn_span, use_turn_span
from fastapi.responses import JSONResponse
from google.adk.errors import StaleSessionError
from google.adk.events import Event, EventActions
from google.adk.sessions import BaseSessionService, Session
from google.adk.sessions.base_session_service import GetSessionConfig

from harness.approval import ApprovalRecord
from harness.conversation import current_conversation_key
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
# escape). A cut field ends in `_CUT_MARKER`, and a cut summary is replaced by a
# `{"cut": ...}` stub.
# A one-series fact summary runs 430 to 960 bytes and each further series adds
# about 550, so the cap holds a chart of about six series. Titles run about 40
# bytes.
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
# What a block shows (`_digest_shows`): up to `_DIGEST_SHOW_CAP` variable names
# and as many entity names, the widest select a component admits (`line_chart`'s
# 8 values), each cut to `_DIGEST_NAME_BYTE_CAP`, and a time span cut to
# `_DIGEST_TIME_BYTE_CAP`. `_BLOCK_SHOWS_BYTES` covers both lists with their keys,
# brackets and separators, and the time span with its key.
_DIGEST_SHOW_CAP = 8
_DIGEST_NAME_BYTE_CAP = 64
_DIGEST_TIME_BYTE_CAP = 24
_BLOCK_SHOWS_BYTES = (
    2 * (16 + _DIGEST_SHOW_CAP * (_DIGEST_NAME_BYTE_CAP + 2)) + 12 + _DIGEST_TIME_BYTE_CAP
)
_DIGEST_WRAPPER_BYTES = 160
CANVAS_DIGEST_MAX_BYTES = _DIGEST_WRAPPER_BYTES + _DIGEST_BLOCK_CAP * (
    _BLOCK_FRAME_BYTES
    + _BLOCK_ID_BYTES
    + _TITLE_BYTE_CAP
    + _FACT_SUMMARY_BYTE_CAP
    + _BLOCK_SHOWS_BYTES
)
# The conversation digest's bounds: newest turns, each turn's text cut by
# characters, its cards by count and title bytes, and its error by bytes, then
# a byte cap on the serialised digest that drops the oldest turn and finally
# shrinks the newest. Past any bound, converse `recall`s.
_HISTORY_TURN_CAP = 10
_HISTORY_TEXT_CAP = 280
_HISTORY_CARD_CAP = 8
_HISTORY_CARD_TITLE_BYTE_CAP = 100
_HISTORY_ERROR_BYTE_CAP = 100
CONVERSATION_DIGEST_MAX_BYTES = 5_000
_RECALL_TURN_CAP = 5
_RECALL_CARD_CAP = 3

_COUNT_KEYS: dict[ComponentType, str] = {
    ComponentType.STAT_CARD: "stats",
    ComponentType.LINE_CHART: "charts",
    ComponentType.BAR_CHART: "charts",
    ComponentType.SCATTER_CHART: "charts",
    ComponentType.TABLE: "tables",
    ComponentType.NOTE_USER: "notes",
    ComponentType.NOTE_NARRATION: "notes",
}


_DETERMINISTIC_COUNTER: dict[str, int] = {}


def deterministic_ids(enabled: bool) -> None:
    """Scripted test mode only: wire ids become sequential so two identical
    runs produce identical streams. Never enable in a real deployment --
    interrupt ids must be unguessable there."""
    _DETERMINISTIC_COUNTER.clear()
    if enabled:
        _DETERMINISTIC_COUNTER["next"] = 1


def _wire_id(prefix: str) -> str:
    if "next" in _DETERMINISTIC_COUNTER:
        value = _DETERMINISTIC_COUNTER["next"]
        _DETERMINISTIC_COUNTER["next"] = value + 1
        return f"{prefix}_{value:016d}"
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
    option_ids: frozenset[str] = frozenset()
    # A "plan" interrupt's workflow step order at the moment `plan.proposed`
    # was built (`_RecordingSurface._step_order`, `studio_translate.py`).
    # Restored before a resumed segment starts so activity steps stay
    # numbered by plan order, whichever instance answers the interrupt.
    step_order: tuple[str, ...] = ()
    # Set by `/interrupt` to the id of the segment resuming this interrupt,
    # instead of deleting the entry outright: a crash between the answer
    # being appended and the resumed segment's own checkpoint leaves a
    # durable, retryable record. Cleared by `ConversationState.drop_resuming`
    # once that segment ends, win or lose; a crashed segment leaves it set,
    # and the same answer resumes again on a fresh segment.
    resuming_segment: str | None = None
    # The decision and payload `/interrupt` was called with, recorded
    # alongside `resuming_segment` so a retried resume can be verified
    # against what actually resumed.
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
    # One record per completed turn: {user, assistant, cards, error}.
    # Feeds the conversation digest (newest window) and the recall seam
    # (everything older). In-process, like the composition manifest.
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

    def canvas_digest_state(self, *, focus: str | None = None) -> dict[str, Any] | None:
        """The digest the models see: composed blocks in canvas order (plus
        the focused card for a card-scoped follow-up), titles and fact
        summaries only -- references and digests, never rows."""
        ordered: list[str] = []
        if self.composition is not None:
            ordered = [block.component_id for block in self.composition.blocks]
        if focus and focus in self.components and focus not in ordered:
            ordered.append(focus)
        # Bounded: the newest blocks win on a very large canvas, and
        # `_digest_block` cuts each block's fields, so the digest stays within
        # `CANVAS_DIGEST_MAX_BYTES` however long the conversation runs.
        ordered = ordered[-_DIGEST_BLOCK_CAP:]
        blocks = [
            _digest_block(cid, self.components[cid]) for cid in ordered if cid in self.components
        ]
        if not blocks:
            return None
        digest: dict[str, Any] = {"blocks": blocks}
        # `focus` is the client's `component_ref.component_id`, an unbounded
        # string that need not name anything, so only a stored component's id
        # is named, and cut like the block that carries it.
        if focus in self.components:
            digest["focus_component_id"] = _cut_text(str(focus), _BLOCK_ID_BYTES)
        return digest

    def conversation_digest_state(self) -> dict[str, Any] | None:
        """The newest turns as compact {user, assistant, cards} records --
        the models' cross-turn memory for statements that made no card.
        Bounded three times: a turn-count window with each turn's text,
        cards and error truncated (`_history_turn`), then a byte cap that
        drops the oldest of the window, then, when the newest turn alone
        still exceeds it, that turn's own truncation tightens until it fits.
        The serialised digest never exceeds `CONVERSATION_DIGEST_MAX_BYTES`
        (converse can `recall` past all of these bounds)."""
        if not self.turns:
            return None
        newest = self.turns[-_HISTORY_TURN_CAP:]
        window = [_history_turn(turn) for turn in newest]

        def size() -> int:
            return len(json.dumps({"turns": window, "total_turns": len(self.turns)}))

        while len(window) > 1 and size() > CONVERSATION_DIGEST_MAX_BYTES:
            window.pop(0)
        shrink = 0
        while size() > CONVERSATION_DIGEST_MAX_BYTES and shrink < _HISTORY_TEXT_CAP.bit_length():
            shrink += 1
            window[-1] = _history_turn(newest[-1], shrink=shrink)
        return {"turns": window, "total_turns": len(self.turns)}

    def anaphor_window(self, focus: str | None) -> str | None:
        """The window a window anaphor ("the same years" and siblings,
        `harness.graph._WINDOW_ANAPHORS`) should resolve to for this turn,
        as `"<start> to <end>"`.

        A card-scoped follow-up (`focus` names a stored component) reads
        that card's own `time` -- the field `_digest_entry` puts on its
        digest entry when `land_component` stores it -- so an older focus
        card's window wins over a newer card elsewhere on the canvas.
        Falls back to the most recently landed card's window when `focus`
        names nothing, or names a card with no window of its own: `turns`,
        walked newest turn first and, within a turn, newest card first,
        the same order `land_component`/`close_turn` build it in, over the
        FULL history -- not `conversation_digest_state`'s capped window,
        which drops turns past `_HISTORY_TURN_CAP` and, within the newest
        turn, cards past `_HISTORY_CARD_CAP`, and so can miss the very
        card this should read. None when no turn recorded a card with a
        window.
        """
        if focus:
            window = _window_phrase(self.components.get(focus, {}).get("time"))
            if window:
                return window
        for turn in reversed(self.turns):
            for card in reversed(turn.get("cards") or []):
                window = _window_phrase(card.get("window"))
                if window:
                    return window
        return None

    def recall_state(self, *, scope: str, query: str) -> dict[str, Any]:
        """Serve a converse recall request, mechanically: matching turn
        records (conversation scope) or full digest entries of matching
        cards (canvas scope). Digests and fact summaries only, never rows."""
        terms = [term for term in query.lower().split() if len(term) > 2]

        def matches(text: str) -> bool:
            # A query of only short words carries no usable terms; matching
            # everything then returns the newest slice, which is the most
            # useful honest answer to a vague recall.
            lowered = text.lower()
            return not terms or any(term in lowered for term in terms)

        if scope == "canvas":
            cards = [
                {"component_id": component_id, **entry}
                for component_id, entry in self.components.items()
                if matches(f"{entry.get('title', '')} {json.dumps(entry.get('fact_summary'))}")
            ]
            return {
                "scope": "canvas",
                "cards": cards[-_RECALL_CARD_CAP:],
                "total_cards": len(self.components),
            }
        hits = [
            {"turn": index, **turn}
            for index, turn in enumerate(self.turns, start=1)
            if matches(f"{turn['user']} {turn['assistant']}")
        ]
        return {
            "scope": "conversation",
            "turns": hits[-_RECALL_TURN_CAP:],
            "total_turns": len(self.turns),
        }

    def close_turn(self, *, final_text: str, error: str | None = None) -> None:
        """Append the turn's record, once -- the resume protocol can end a
        turn from more than one code path."""
        if self.turn_recorded:
            return
        self.turn_recorded = True
        end_turn_span(self._telemetry_span)
        self._telemetry_span = None
        record: dict[str, Any] = {
            "user": self.turn_user_message,
            "assistant": final_text,
        }
        if self.turn_components:
            record["cards"] = list(self.turn_components)
        if error:
            record["error"] = error
        self.turns.append(record)

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
        if self.turn_index > 1 and not self.turn_recorded:
            # The previous turn never reached a terminal frame (an abandoned
            # pause, or a death no code path recorded). Its record keeps
            # what was asked, with the error marked as abandoned.
            self.close_turn(final_text=self.turn_final_text, error="abandoned")
        self.turn_user_message = ""
        self.turn_final_text = ""
        self.turn_components = []
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

# `StudioStore.checkpoint`'s fetch-merge-append cycles, capping how long a
# conversation under sustained contention retries before it raises.
_CHECKPOINT_RETRY_ATTEMPTS = 3
# `StudioStore.renew` and `StudioStore.request_cancel`'s fetch-append
# cycles for their single-key `studio:segment` write.
_SEGMENT_WRITE_RETRY_ATTEMPTS = 3
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

# Identity and the five process-local runtime objects that
# `_serialize_durable`/`_apply_delta` skip.
_PROCESS_LOCAL_FIELDS = frozenset(
    {
        "conversation_id",
        "lock",
        "cancel_event",
        "cancel_reason",
        "_telemetry_span",
        "_composition_ops",
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
            "option_ids": sorted(p.option_ids),
            "step_order": list(p.step_order),
            "resuming_segment": p.resuming_segment,
            "answer": p.answer,
        }
        for interrupt_id, p in pending.items()
    }


def _deserialize_pending(data: dict[str, Any]) -> dict[str, _PendingInterrupt]:
    return {
        interrupt_id: _PendingInterrupt(
            adk_id=entry["adk_id"],
            kind=entry["kind"],
            plan_version=entry.get("plan_version"),
            plan_revision=entry.get("plan_revision"),
            option_ids=frozenset(entry.get("option_ids") or ()),
            step_order=tuple(entry.get("step_order") or ()),
            resuming_segment=entry.get("resuming_segment"),
            answer=entry.get("answer"),
        )
        for interrupt_id, entry in data.items()
    }


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
        elif key.startswith(_TURN_ITEM_PREFIX):
            _set_list_item(state.turns, int(key.removeprefix(_TURN_ITEM_PREFIX)), value)
        elif key.startswith(_COMPONENT_ITEM_PREFIX):
            state.components[key.removeprefix(_COMPONENT_ITEM_PREFIX)] = value
        elif key.startswith(_WRITE_TOKEN_ITEM_PREFIX):
            state.write_tokens[key.removeprefix(_WRITE_TOKEN_ITEM_PREFIX)] = value
        elif key.startswith(_APPROVAL_ITEM_PREFIX):
            index = int(key.removeprefix(_APPROVAL_ITEM_PREFIX))
            _set_list_item(state.approvals, index, ApprovalRecord.model_validate(value))
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
        no ledger access. The held-in-process accessor `make_recall` and
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
        return entry.state

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

    async def save(
        self, principal: str, conversation_id: str, *, keys: Iterable[str] | None = None
    ) -> None:
        """Append this instance's changed durable fields as one ledger
        event, through the one `Session` object this runtime entry holds.
        Only the keys whose serialised value changed since the last load
        or save on this instance are sent -- a no-op event appends
        nothing. `keys`, when given, restricts the delta to that set even
        when other fields also changed locally. Raises `StaleSessionError`
        when another instance's write has advanced the ledger past what
        this instance last saw; the caller maps that to the existing 409
        for a `/turn` claim or `/interrupt` (`/composition` never calls
        this: `mutate_composition` is its own fetch-rebuild-mutate-append
        cycle, immune to whatever this instance's own `entry.session`
        marker says). Held under `write_lock` like every other ledger
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
            if keys is not None:
                delta = {key: value for key, value in delta.items() if key in keys}
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
        on a landed mutation, the appended value, not this instance's
        pre-call copy, which may already be behind the ledger. Raises
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
                entry.state.composition = result
                if ops_folded:
                    del entry.state._composition_ops[:ops_folded]
                return result
        assert last_error is not None
        raise last_error

    async def renew(
        self, principal: str, conversation_id: str, segment_id: str
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
        the session service raises, propagates. The caller's own
        heartbeat decides how long to keep trying before giving up on the
        renewal altogether.
        """
        entry = self._entry(principal, conversation_id)
        last_error: StaleSessionError | None = None
        async with entry.write_lock:
            for _attempt in range(1, _SEGMENT_WRITE_RETRY_ATTEMPTS + 1):
                session = await self._fetch_session(principal, conversation_id)
                segment = _deserialize_segment(session.state.get(_KEY_SEGMENT)) if session else None
                if session is None or segment is None or segment.id != segment_id:
                    return "lost"
                renewed = dataclasses.replace(segment, expires_at=self._clock() + SEGMENT_TTL_S)
                event = Event(
                    author="studio",
                    actions=EventActions(state_delta={_KEY_SEGMENT: _serialize_segment(renewed)}),
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
        `checkpoint`. Runs shielded from the caller's own cancellation,
        with a timeout, since it runs in a `finally` on client-disconnect
        and generator-finaliser paths where an unshielded await can
        itself be cancelled. The timeout cancels the checkpoint once
        (`asyncio.timeout`). anyio's `move_on_after` re-cancels at every
        await until its scope exits, cutting short the session service's
        own rollback and connection close.

        Never raises: a `checkpoint` that cannot land (fenced out by a
        takeover, a DB error, or the timeout above) is logged and
        swallowed. The segment then simply expires by its own TTL, which
        is what makes it safe to give up here.
        """
        entry = self._runtime.get((principal, conversation_id))
        if entry is None or entry.state is None:
            return
        with anyio.CancelScope(shield=True):
            try:
                async with asyncio.timeout(timeout):
                    entry.state.drop_resuming(segment_id)
                    if entry.state.segment is not None and entry.state.segment.id == segment_id:
                        entry.state.segment = None
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


def make_recall(store: StudioStore) -> Any:
    """The graph's recall seam, bound to a store. The conversation is
    resolved at CALL time through the request-scoped contextvar -- the same
    pattern that routes the executor's presentation client -- because the
    workflow is constructed once and shared across every conversation."""

    async def recall(*, scope: str, query: str) -> dict[str, Any]:
        key = current_conversation_key.get()
        if key is None:
            return {}
        conversation = store.current(*key)
        if conversation is None:
            return {}
        return conversation.recall_state(scope=scope, query=query)

    return recall


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


def _bounded_fact_summary(summary: Any) -> Any:
    """`summary` when its serialised form fits `_FACT_SUMMARY_BYTE_CAP`, else
    a `{"cut": ...}` stub stating its size and the limit. The block already
    carries the component id and revision."""
    size = len(json.dumps(summary))
    if size <= _FACT_SUMMARY_BYTE_CAP:
        return summary
    return {"cut": f"a {size}-byte fact summary exceeds the {_FACT_SUMMARY_BYTE_CAP}-byte limit"}


def _digest_block(component_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    """One block of the canvas digest: the stored entry with its id, type,
    title, fact summary and what it shows bounded. The stored entry stays
    whole for `recall_state` and the composition "add" op."""
    block: dict[str, Any] = {"component_id": _cut_text(component_id, _BLOCK_ID_BYTES), **entry}
    if isinstance(block.get("type"), str):
        block["type"] = _cut_text(block["type"], _BLOCK_TYPE_BYTES)
    if block.get("title") is not None:
        block["title"] = _cut_text(str(block["title"]), _TITLE_BYTE_CAP)
    if "fact_summary" in block:
        block["fact_summary"] = _bounded_fact_summary(block["fact_summary"])
    for key in ("variable", "entities"):
        if key in block:
            names = block[key] if isinstance(block[key], list) else [block[key]]
            block[key] = [
                _cut_text(str(name), _DIGEST_NAME_BYTE_CAP) for name in names[:_DIGEST_SHOW_CAP]
            ]
    if "time" in block:
        block["time"] = _cut_text(str(block["time"]), _DIGEST_TIME_BYTE_CAP)
    return block


def _window_phrase(display: Any) -> str | None:
    """`"<start>-<end>"` as `"<start> to <end>"`, the form
    `harness.graph._resolve_window_anaphors` substitutes in; None when
    `display` is not a string or carries no `-`-separated span."""
    if isinstance(display, str) and "-" in display:
        start, _, end = display.partition("-")
        if start and end:
            return f"{start} to {end}"
    return None


def _history_turn(turn: dict[str, Any], *, shrink: int = 0) -> dict[str, Any]:
    """One turn of the conversation digest. Each cap is halved `shrink` times,
    so the newest turn can be tightened until the whole digest fits; past the
    cards' cap, `cards_omitted` counts the cards left out."""
    text_cap = _HISTORY_TEXT_CAP >> shrink
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
    record: dict[str, Any] = {
        "user": turn["user"][:text_cap],
        "assistant": turn["assistant"][:text_cap],
    }
    if kept:
        record["cards"] = kept
    if len(cards) > len(kept):
        record["cards_omitted"] = len(cards) - len(kept)
    if turn.get("error"):
        record["error"] = _cut_text(turn["error"], _HISTORY_ERROR_BYTE_CAP >> shrink)
    return record


def _digest_entry(component: dict[str, Any]) -> dict[str, Any]:
    """The per-component slice of the models' canvas digest. It states what
    the card is and what it shows, as names only, no rows: the variable
    name(s), the entities it draws, and the span it covers. The resolved
    revision in `payload` contributes only its title (`_block_title`). The
    model never sees the rest of it."""
    return {
        "type": str(component.get("type", "")),
        "title": _block_title(component),
        "revision": int(component.get("revision") or 1),
        "fact_summary": component.get("fact_summary"),
        **_digest_shows(component),
    }


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
    """The `variable` / `entities` / `time` keys of a digest entry, from the
    component's wire fact summary: each series carries the entity's label,
    each indicator the variable's name, and `window` the resolved span. The
    summary is decoded for its names only, and a key the summary cannot
    answer is left out. Each list is capped at `_DIGEST_SHOW_CAP`, keeping
    the entry's size bounded however many series the underlying frame
    holds.
    """
    summary = component.get("fact_summary") or {}

    entities = [
        str(series["label"])
        for series in summary.get("series") or []
        if isinstance(series, dict) and series.get("label")
    ]
    variables = [
        str(indicator["name"])
        for indicator in summary.get("indicators") or []
        if isinstance(indicator, dict) and indicator.get("name")
    ]

    shows: dict[str, Any] = {}
    if variables:
        shows["variable"] = _bounded_names(variables)
    if entities:
        shows["entities"] = _bounded_names(entities)
    window = summary.get("window") or {}
    start = window.get("start") if isinstance(window, dict) else None
    end = window.get("end") if isinstance(window, dict) else None
    if start and end:
        shows["time"] = f"{start}-{end}"
    return shows


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
    membership check `_replace_block` and `land_component` share."""
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
    revises: str | None,
) -> Literal["replaced", "created", "held"]:
    """Land `component` on `conversation`'s canvas: the one place the
    replace-or-append decision is made and carried out, together, with no
    `await` between deciding and mutating. `harness.graph`'s `compose` node
    calls this (through `_RecordingExecutor.land_component`,
    `harness.studio_translate`) synchronously, right after `write_component`
    returns, for the first component this turn's write actually lands --
    `revises` is the target `compose` passes only that once. That
    single call decides `narrate`'s `components_replaced` fact and performs
    the real mutation together, so nothing downstream re-derives a second,
    possibly different, answer from a canvas `/composition` (which runs
    without the turn lock) could have changed in between.

    Always records the component's digest entry
    (`conversation.components`), so a card is findable by `recall` and the
    canvas digest whether or not it lands a block. `auto_add=False` means
    there is no block for anything to swap or append: the component still
    streams to the client, but nothing here composes for it, and the
    return is `"held"`. Otherwise, `revises` naming a block still on the
    canvas swaps it in place (`_replace_block`) and returns `"replaced"`;
    anything else appends a new block and returns `"created"`.

    The `turn_components` entry also carries `window` when the digest entry
    resolved one (`_digest_shows`' `time`). `close_turn` folds it into
    `conversation.turns`, in landing order, so `anaphor_window` can read
    "the previous turn's resolved window" off turn order instead of canvas
    position, correct even when this turn revises an earlier-positioned
    block in place.
    """
    entry = _digest_entry(component)
    conversation.components[component["component_id"]] = entry
    window = entry.get("time")
    if not conversation.auto_add:
        conversation.turn_components.append(
            _turn_component_entry(
                component_id=component["component_id"],
                type=component["type"],
                title=_block_title(component),
                action="created",
                window=window,
            )
        )
        return "held"
    replaced = bool(revises) and _replace_block(
        conversation, old_component_id=revises, component=component
    )
    conversation.turn_components.append(
        _turn_component_entry(
            component_id=component["component_id"],
            type=component["type"],
            title=_block_title(component),
            action="replaced" if replaced else "created",
            window=window,
        )
    )
    if not replaced:
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
    return "replaced" if replaced else "created"


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
    the canvas here. Shared by the Edit Card route's recreate
    (`harness.studio_edit`) and the message path's "add X to the chart"
    (`land_component`, above), both of which name an existing component_id
    whose slot the new component takes over.
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
