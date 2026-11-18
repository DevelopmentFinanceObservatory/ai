"""The Edit Card route: `component_ref.intent: "edit"`.

No resolve, no plan, no agent run: the card is described, the model
decides between a revise patch (title / select / order) and a full recreate
(type, window, frame_ref), and the platform write streams back as
`component.revised` or `component.created`. A recreated component takes
the replaced block's slot on the canvas.

A card on a frame whose period follows the latest data stands on the frame of
the series it draws (`harness.rolling_cards`, applied by the executor to every
write). A revise cannot move a card to another frame, so when the executor
refuses one `rolling_select_needs_new_frame` the edit is written as a new card
from the stored intent with the patch merged in, on the frame the executor
picks, and takes the old block's slot.

A revise needs the write token the platform issued when this conversation
created the card (`ConversationState.write_tokens`). A card the
conversation holds no token for (one it did not create, or one created
before tokens existed) is rebuilt as a new card from its stored intent
with the patch merged in, so the edit still lands in the card's slot. The
new card has a new id and a fresh revision history, and the reply says so.

Failures serialise into the stream as a stream-fatal error frame, as
they do in a turn.

The edit runs as two halves (`harness.studio_stream.Work`). `edit_events`
makes the edit's calls in the work's drive and reads the conversation. It
emits what the edit sends and changes as events, with its frames unnumbered,
and leaves every write to the other half. `edit_event_frames` turns each
event into frames in the stream owner's task, which numbers them, records
the turn, and writes the card to the canvas (`StudioStore.change_canvas`).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any

from common.mcp_telemetry import WRITE_TOKEN_KEY
from pydantic import BaseModel

from harness.conversation import current_conversation_key
from harness.provenance import CardLineage
from harness.studio_state import (
    ConversationState,
    _block_title,
    _digest_entry,
    _next_manifest,
    _replace_block,
    _turn_component_entry,
    drawn_series_ids,
)
from harness.studio_store import StudioStore
from harness.studio_translate import (
    _RecordingExecutor,
    _stream_fatal,
    closed_on_reply,
    failure_detail,
)
from harness.tool_contracts import HarnessCode, harness_refusal
from harness.wire import (
    ActivityStep,
    CompositionActor,
    CompositionUpdated,
    StepProduced,
    StepStatus,
    component_created,
    component_revised,
)

logger = logging.getLogger(__name__)

# Intent fields a revise patch may not name, as apis
# `assembler.merge.merge_intent` refuses them.
_CREATE_ONLY_FIELDS = ("type", "frame_ref")
# The component types whose intent takes an `order`.
_ORDERED_TYPES = frozenset({"line_chart", "bar_chart", "table"})

# What a closing message says before the sentence for a refused edit's code.
_EDIT_REFUSED = "The edit was refused."

# What a closing message says before the sentence for a rebuild whose stored
# card the harness cannot read. The canvas keeps the card it showed.
_UNREADABLE_LEAD = "The card on the canvas is unchanged."

# The same for a revise. The platform's head holds the change, the canvas shows
# the earlier version, and the next edit starts from the saved change.
_UNREADABLE_REVISE_LEAD = (
    "Your change was saved, but the card cannot be shown with it, so the canvas still "
    "shows the earlier version. Your next edit builds on the saved change."
)

_REBUILT_TEXT = (
    "This conversation holds no edit access to this card, so I could not update it in "
    "place. I rebuilt it as a new card with your change: it has a new id and starts a "
    "fresh revision history, and the original card is unchanged."
)


@dataclass(frozen=True)
class EditClosed:
    """The edit ends on `text`, its reply, and `done`."""

    text: str


@dataclass(frozen=True)
class EditFailed:
    """The edit ends on the stream-fatal frame for `error`."""

    error: Exception


@dataclass(frozen=True)
class CanvasChange:
    """A card write. `change` runs on the canvas as stored and returns
    whether it changed the canvas (`StudioStore.change_canvas`), and
    `lineage`, the written card's frame lineage, commits with it."""

    change: Callable[[ConversationState], bool]
    lineage: CardLineage | None = None


# An edit emits a frame to number or one of the events above.
type EditEvent = BaseModel | EditClosed | EditFailed | CanvasChange


async def edit_events(
    *,
    recorder: _RecordingExecutor,
    edit_model: Any,
    conversation: ConversationState,
    component_id: str,
    message: str,
) -> AsyncGenerator[EditEvent]:
    """The Edit Card route's events: no resolve, no plan.

    Failures serialise into the stream as a stream-fatal error frame, as
    they do in a turn, so a malformed model draft cannot end the transport
    without an error frame."""
    try:
        async for event in _edit_flow(
            recorder=recorder,
            edit_model=edit_model,
            conversation=conversation,
            component_id=component_id,
            message=message,
        ):
            yield event
    except Exception as exc:
        logger.exception("studio edit turn failed (trace_id=%s)", conversation.trace_id)
        yield EditFailed(exc)


async def edit_event_frames(
    event: EditEvent, *, conversation: ConversationState, store: StudioStore
) -> list[BaseModel]:
    """The frames of `event`, numbered, with what it changes written: the
    turn's record for its end, and the card for a card write. A card write
    fenced out by another instance's takeover, or one that cannot reach the
    database before the lease's deadline, raises `SegmentLost` or
    `DatabaseUnavailable`, and the stream ends as a lost lease does."""
    c = conversation
    match event:
        case EditClosed(text=text):
            return [*closed_on_reply(c, text, suggestions=True)]
        case EditFailed(error=error):
            return [_stream_fatal(c, error)]
        case CanvasChange(change=change, lineage=lineage):
            key = current_conversation_key.get()
            assert key is not None, "an edit runs only inside a conversation's stream"
            if not await store.change_canvas(*key, change, lineage=lineage):
                return []
            return [
                CompositionUpdated(
                    event="composition.updated", seq=c.next_seq(), manifest=c.composition
                )
            ]
        case _:
            return [event.model_copy(update={"seq": c.next_seq()})]


async def _edit_flow(
    *,
    recorder: _RecordingExecutor,
    edit_model: Any,
    conversation: ConversationState,
    component_id: str,
    message: str,
) -> AsyncGenerator[EditEvent]:
    """Verified platform rules (2026-08-28, hosted dev): a revision may
    change only title/select; type/window/frame_ref edits recreate the
    component instead, and the studio replaces the block in place. The
    old component stays readable at its prior revisions.

    Its frames carry `seq` 0 until `edit_event_frames` numbers them, in the
    order this yields them.
    """
    c = conversation

    def _step(
        step_id: str,
        label: str,
        *,
        failed: str | None = None,
        produced: StepProduced | None = None,
    ) -> ActivityStep:
        """A finished step, failed when `failed` gives the sentence it shows."""
        return ActivityStep(
            event="activity.step",
            seq=0,
            step_id=step_id,
            label=label,
            status=StepStatus.FAILED if failed else StepStatus.COMPLETE,
            detail=failed,
            produced=produced,
        )

    def _close(text: str) -> list[EditEvent]:
        return [EditClosed(text)]

    def _refused(step_id: str, label: str, lead: str, result: dict[str, Any]) -> list[EditEvent]:
        """The failed step and the closing message of a call that was refused.
        Both give the sentence for the refusal's code. Its message, which is
        written for a model, goes to the log."""
        refusal = result.get("refusal")
        refusal = refusal if isinstance(refusal, dict) else {}
        logger.info("card edit refused (%s): %s", refusal.get("code"), refusal.get("message"))
        sentence = failure_detail(refusal)
        return [
            _step(step_id, label, failed=sentence),
            *_close(" ".join(part for part in (lead, sentence) if part)),
        ]

    described = await recorder(tool="describe_component", args={"component_id": component_id})
    if described.get("outcome") != "ok":
        for frame in _refused("e1", "Read the card", "I couldn't open that card.", described):
            yield frame
        return
    body = described.get("component_body") or {}
    stored = body.get("stored_intent") or {}
    current_revision = int(body.get("revision") or 1)
    yield _step("e1", "Read the card")

    frame_view: dict[str, Any] = {}
    if stored.get("frame_ref"):
        viewed = await recorder(tool="describe_frame", args={"frame_ref": stored["frame_ref"]})
        if viewed.get("outcome") == "ok":
            frame_view = viewed.get("view") or {}

    if edit_model is None:
        for frame in _close("Card edits are not enabled on this deployment."):
            yield frame
        return
    drafted = await edit_model(message=message, stored_intent=stored, frame_view=frame_view)
    patch = drafted.get("patch")
    recreate = drafted.get("recreate")
    text = str(drafted.get("text") or "")
    if patch:
        patch = _ordered_patch(stored, patch)
        if not patch:
            # The patch named only an order the card cannot take.
            text = ""
    if not patch and not recreate:
        for frame in _close(text or "That edit isn't possible for this card's data."):
            yield frame
        return

    write_token = c.write_tokens.get(component_id)
    # Set when the revise was refused because the new select needs another frame.
    moved = False
    if patch and write_token:
        written = await recorder(
            tool="write_component",
            args={"intent": patch, "component_id": component_id, WRITE_TOKEN_KEY: write_token},
        )
        moved = (written.get("refusal") or {}).get(
            "code"
        ) == HarnessCode.ROLLING_SELECT_NEEDS_NEW_FRAME
    if patch and write_token and not moved:
        if written.get("outcome") != "ok" or not written.get("component"):
            frames = _refused(
                "e2",
                "Apply the edit",
                _refused_lead(written, unreadable=_UNREADABLE_REVISE_LEAD),
                written,
            )
        elif _is_no_op(written["component"], current_revision) and not _canvas_is_behind(
            conversation, component_id, written["component"]
        ):
            # The platform treats an effectively-equal patch as a no-op.
            yield _step("e2", "Apply the edit")
            frames = _close("The card already looks like that; nothing changed.")
        else:
            component = written["component"]
            token = recorder.write_token(component)
            revised = component_revised(component, component_id, patch)
            yield _step(
                "e2",
                "Apply the edit",
                produced=StepProduced(component_id=component_id, revision=revised.revision),
            )
            caught_up = _is_no_op(component, current_revision)
            frames: list[EditEvent] = [revised]
            title = _block_title(component) if caught_up else patch.get("title")

            # Whether an earlier attempt of the card's write changed the
            # canvas: a retry after a commit whose acknowledgement was lost
            # finds the block already refreshed, and owes the client the
            # `composition.updated` that commit made.
            refreshed = False

            def revise(work: ConversationState) -> bool:
                nonlocal refreshed
                if token:
                    work.write_tokens = {**work.write_tokens, component_id: token}
                _record_revision(
                    work,
                    component_id=component_id,
                    component=component,
                    revision=revised.revision,
                    title=title,
                )
                refreshed = (
                    _refresh_block(
                        work, component_id=component_id, revision=revised.revision, title=title
                    )
                    or refreshed
                )
                return refreshed

            frames.append(CanvasChange(revise, lineage=recorder.card_lineage(component)))
            frames.extend(_close(text or "Done -- the card was updated."))
        for frame in frames:
            yield frame
        return

    if patch:
        named = next((field for field in _CREATE_ONLY_FIELDS if field in patch), None)
        if named is not None:
            create_only = harness_refusal(
                code=HarnessCode.FIELD_CREATE_ONLY,
                message=f"{named!r} is create-only and cannot be revised",
            )
            for frame in _refused("e2", "Apply the edit", _EDIT_REFUSED, {"refusal": create_only}):
                yield frame
            return
        intent = _merge_patch(stored, patch)
        if moved:
            step_label = "Rebuild the card"
            closing = text or "Done -- the card was rebuilt with your change."
        else:
            step_label = "Rebuild as a new card"
            # The model drafted `text` for an in-place revise; this is a new card.
            closing = _REBUILT_TEXT
    else:
        intent = _recreated(stored, recreate)
        step_label = "Rebuild the card"
        closing = text or "Done -- the card was rebuilt with your change."
    written = await recorder(tool="write_component", args={"intent": intent})
    if written.get("outcome") != "ok" or not written.get("component"):
        lead = _refused_lead(written, unreadable=_UNREADABLE_LEAD)
        for frame in _refused("e2", step_label, lead, written):
            yield frame
        return
    created = component_created(written["component"], step_id="e2")
    component = written["component"]
    token = recorder.write_token(component)
    # The step frame is yielded first. Frames are numbered in the order they
    # are yielded, and the step precedes the card it produced on the wire.
    yield _step(
        "e2",
        step_label,
        produced=StepProduced(component_id=created.component_id, revision=created.revision),
    )
    yield created

    def recreate(work: ConversationState) -> bool:
        if token:
            work.write_tokens = {**work.write_tokens, created.component_id: token}
        entry = _digest_entry(component)
        work.components[created.component_id] = entry
        work.turn_components.append(
            _turn_component_entry(
                component_id=created.component_id,
                type=created.type,
                title=_block_title(component),
                action="recreated",
                window=entry.get("time"),
            )
        )
        return _replace_block(work, old_component_id=component_id, component=component)

    yield CanvasChange(recreate, lineage=recorder.card_lineage(component))
    for frame in _close(closing):
        yield frame


def _is_no_op(component: dict[str, Any], current_revision: int) -> bool:
    """Whether a revise left the card at the revision it held."""
    revision = component.get("revision")
    return isinstance(revision, int) and revision <= current_revision


def _canvas_is_behind(
    conversation: ConversationState, component_id: str, head: dict[str, Any]
) -> bool:
    """Whether the conversation shows an older revision than `head`, a readable
    head the platform stored for an earlier edit. That edit's reply never
    reached the harness (a timeout), or its canvas write failed. The card is
    behind when its digest entry, or its canvas block where it has one, is
    below the head. A no-op write over such a head is applied as a revise, so
    the card catches up. A card with no block (held, or removed) is judged by
    its digest entry alone."""
    head_revision = head["revision"]
    recorded = (conversation.components.get(component_id) or {}).get("revision")
    if isinstance(recorded, int) and recorded < head_revision:
        return True
    composition = conversation.composition
    blocks = composition.blocks if composition is not None else []
    return any(
        block.component_id == component_id and block.revision < head_revision for block in blocks
    )


def _refused_lead(result: dict[str, Any], *, unreadable: str) -> str:
    """What a closing message says before the sentence for a refused write. It is
    `unreadable` when the platform stored a card the harness cannot read."""
    code = (result.get("refusal") or {}).get("code")
    return unreadable if code == HarnessCode.STORED_CARD_UNREADABLE else _EDIT_REFUSED


def _legal_order(order: Any, *, type_: Any) -> dict[str, Any] | None:
    """`order` as a `type_` card's intent takes it: None on a type that takes
    no order and for an order whose shape the platform refuses, which is
    logged, and with no `period` on a `bar_chart`."""
    if type_ not in _ORDERED_TYPES:
        return None
    valid = (
        isinstance(order, dict)
        and set(order) <= {"by", "direction", "period"}
        and order.get("by") == "value"
        and order.get("direction") in {"ascending", "descending"}
        and ("period" not in order or (isinstance(order["period"], str) and order["period"]))
    )
    if not valid:
        logger.warning("card edit dropped an invalid order: %r", order)
        return None
    return _without_period(order) if type_ == "bar_chart" else order


def _without_period(order: dict[str, Any]) -> dict[str, Any]:
    """`order` ranking at the platform's own period."""
    return {key: value for key, value in order.items() if key != "period"}


def _ordered_patch(stored: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """`patch` with an `order` the stored card's type takes: the model's own,
    made legal (`_legal_order`) or left out, `null` clearing it. A patch that
    changes the select or window and leaves the order alone ranks at the
    platform's own period, since the stored card's period may hold no value
    for a series the new card draws, and the platform refuses an order at
    such a period."""
    patch = dict(patch)
    if patch.get("order") is not None:
        order = _legal_order(patch.pop("order"), type_=stored.get("type"))
        if order is not None:
            patch["order"] = order
    held = stored.get("order")
    if (
        "order" not in patch
        and isinstance(held, dict)
        and "period" in held
        and any(key in patch and patch[key] != stored.get(key) for key in ("select", "window"))
    ):
        patch["order"] = _without_period(held)
    return patch


def _recreated(stored: dict[str, Any], recreate: dict[str, Any]) -> dict[str, Any]:
    """The intent a recreate writes: the model's `recreate` on the stored
    card's frame, with its `order` made legal for its type (`_legal_order`).
    A recreate with no order draws the card unordered. The stored card's own
    period is kept only on its select and window, among the series it ranked,
    since the platform refuses an order at a period no drawn series holds; a
    period the model names afresh stands."""
    intent = {key: value for key, value in recreate.items() if key != "order"}
    intent["frame_ref"] = stored.get("frame_ref")
    if recreate.get("order") is None:
        return intent
    order = _legal_order(recreate["order"], type_=intent.get("type"))
    if order is None:
        return intent
    held = stored.get("order")
    carried = isinstance(held, dict) and "period" in order and order["period"] == held.get("period")
    if carried and any(intent.get(key) != stored.get(key) for key in ("select", "window")):
        order = _without_period(order)
    return intent | {"order": order}


def _merge_patch(stored: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """`patch` applied to `stored` as apis `assembler.merge.merge_intent`
    applies a revise patch (RFC 7386 as the platform names it), at the top
    level only: absent keys survive, `null` deletes, and any other value
    replaces whole, objects and arrays included."""
    merged = dict(stored)
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def _record_revision(
    conversation: ConversationState,
    *,
    component_id: str,
    component: dict[str, Any],
    revision: int,
    title: str | None,
) -> None:
    """Bring a landed revise's digest entry up to `revision` and add its turn
    record. `title` is the card's new title, None when the edit left it. The
    entry is a new dict, since the store runs this on a one-level copy of the
    live state (`StudioStore.change_canvas`)."""
    entry = dict(conversation.components.get(component_id) or _digest_entry(component))
    entry["revision"] = revision
    entry["fact_summary"] = component.get("fact_summary") or entry.get("fact_summary")
    entry.update(drawn_series_ids(component))
    if title:
        entry["title"] = title
    conversation.components[component_id] = entry
    conversation.turn_components.append(
        _turn_component_entry(
            component_id=component_id,
            type=entry.get("type"),
            title=entry.get("title"),
            action="revised",
            window=entry.get("time"),
        )
    )


def _refresh_block(
    conversation: ConversationState,
    *,
    component_id: str,
    revision: int,
    title: str | None,
) -> bool:
    """Bring the edited component's block up to `revision`, and `title` when
    given; True iff a block changed. A card that is not composed, or whose
    block already shows `revision` and `title`, mints no empty manifest
    revision. The store retries a write whose commit landed but whose
    acknowledgement was lost, and that retry reads the block the commit
    wrote."""
    previous = conversation.composition
    if previous is None:
        return False
    blocks = [block.model_copy(deep=True) for block in previous.blocks]
    changed = False
    for block in blocks:
        if block.component_id == component_id and (
            block.revision != revision or (title and block.title != title)
        ):
            block.revision = revision
            if title:
                block.title = title
            changed = True
    if not changed:
        return False
    conversation.composition = _next_manifest(previous, blocks, actor=CompositionActor.MODEL)
    return True
