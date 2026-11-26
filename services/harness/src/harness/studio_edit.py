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
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from common.mcp_telemetry import WRITE_TOKEN_KEY
from compiler.surface import ToolSurface
from pydantic import BaseModel

from harness.card_unit import (
    CARD_UNIT_KEY,
    UNIT_GONE_NOTE,
    UnitMove,
    card_unit_change,
    retitled,
    stored_block,
    unit_card,
)
from harness.conversation import current_conversation_key
from harness.graph.state import ORCHESTRATOR_SEARCHES_KEY, store_search
from harness.provenance import CARD_FRAME_KEY, CardLineage
from harness.scope_choices import mark_planner_axes
from harness.studio_state import (
    REPLACED_CARD_KEY,
    ConversationState,
    _block_index,
    _block_title,
    _digest_entry,
    _next_manifest,
    _replace_block,
    _turn_component_entry,
    card_runs,
    drawn_series_ids,
)
from harness.studio_store import StudioStore
from harness.studio_translate import (
    _RecordingExecutor,
    _stream_fatal,
    closed_on_reply,
    failure_detail,
)
from harness.tool_contracts import HarnessCode, harness_refusal, resolver_digest
from harness.turn_records import TURN_CARD_RUNS, held_back_line
from harness.unit_offer import unit_words
from harness.unit_switch import UnitRefused, UnitResubmitted, resubmit_unit
from harness.wire import (
    ActivityStep,
    ComponentCreated,
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
    """The edit ends on `text`, its reply, and `done`. A reply that is not
    `shown` is kept in the turn's record only, and the stream sends no
    `message.complete`. `user_display` is the line a state read shows as the
    user's message, when the record's own line is written for a model.
    `held_line` is the code's sentences on the partial and projected years
    the card draws or left out (`harness.turn_records.held_back_line`),
    shown after `text` and recorded under its own key."""

    text: str
    shown: bool = True
    user_display: str | None = None
    held_line: str | None = None


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
    # Search blocks the turn read, by spec_id, joined to the conversation's
    # `turn_specs` once the write commits: process-local, so outside the write.
    specs: Mapping[str, Any] | None = None


class SwapLanded(Enum):
    """Where a card written to take another's slot (`CardSwap`) landed."""

    # In the old card's slot, which it now holds.
    SWAPPED = "swapped"
    # Nowhere: the old card holds no block (held, or removed), and the new
    # card is kept off the canvas.
    HELD = "held"
    # Nowhere and not kept: the old card left the canvas and the swap needed its slot.
    GONE = "gone"


@dataclass(frozen=True)
class CardSwap:
    """A card written to take card `replaces`' slot, the last event of the
    edit that wrote it. `swap` runs on the canvas as stored and makes the
    write (`StudioStore.change_canvas`), with `lineage` and `specs` as
    `CanvasChange` has them. Only once it commits are `step`, the step that
    produced the card, and `created`, its `component.created`, published,
    `created` naming `replaces` when the card took its slot, and the edit
    ends on `closed`. A swap that finds the slot gone, when `gone` is given,
    writes nothing and the edit ends on `gone`, a failed step and its
    closing, with nothing published."""

    step: ActivityStep
    created: ComponentCreated
    replaces: str
    swap: Callable[[ConversationState], SwapLanded]
    lineage: CardLineage | None
    specs: Mapping[str, Any] | None
    closed: EditClosed
    gone: tuple[ActivityStep, EditClosed] | None = None


# An edit emits a frame to number or one of the events above.
type EditEvent = BaseModel | EditClosed | EditFailed | CanvasChange | CardSwap


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
    turn's record for its end, and the card for a card write. A card swap's
    frames follow its write (`_swap_frames`). A card write
    fenced out by another instance's takeover, or one that cannot reach the
    database before the lease's deadline, raises `SegmentLost` or
    `DatabaseUnavailable`, and the stream ends as a lost lease does."""
    c = conversation
    match event:
        case EditClosed():
            return _closed_frames(c, event)
        case EditFailed(error=error):
            return [_stream_fatal(c, error)]
        case CanvasChange(change=change, lineage=lineage, specs=specs):
            key = current_conversation_key.get()
            assert key is not None, "an edit runs only inside a conversation's stream"
            changed = await store.change_canvas(*key, change, lineage=lineage)
            if specs:
                c.turn_specs = {**c.turn_specs, **specs}
            if not changed:
                return []
            return [
                CompositionUpdated(
                    event="composition.updated", seq=c.next_seq(), manifest=c.composition
                )
            ]
        case CardSwap():
            return await _swap_frames(event, conversation=c, store=store)
        case _:
            return [event.model_copy(update={"seq": c.next_seq()})]


async def _swap_frames(
    event: CardSwap, *, conversation: ConversationState, store: StudioStore
) -> list[BaseModel]:
    """The frames of `event`, a card swap, once its write commits: none of the
    card's before it, so a client never shows a card the canvas does not
    hold. A write that raises publishes nothing."""
    c = conversation
    key = current_conversation_key.get()
    assert key is not None, "an edit runs only inside a conversation's stream"
    landed = await store.change_canvas(
        *key,
        event.swap,
        lineage=event.lineage,
        landed=lambda landed: landed is SwapLanded.SWAPPED,
    )
    if landed is SwapLanded.GONE:
        assert event.gone is not None, "only a swap that needs its slot finds it gone"
        step, closed = event.gone
        return [step.model_copy(update={"seq": c.next_seq()}), *_closed_frames(c, closed)]
    if event.specs:
        c.turn_specs = {**c.turn_specs, **event.specs}
    replaces = event.replaces if landed is SwapLanded.SWAPPED else None
    frames: list[BaseModel] = [
        event.step.model_copy(update={"seq": c.next_seq()}),
        event.created.model_copy(update={"seq": c.next_seq(), "replaces": replaces}),
    ]
    if landed is SwapLanded.SWAPPED:
        frames.append(
            CompositionUpdated(
                event="composition.updated", seq=c.next_seq(), manifest=c.composition
            )
        )
    return [*frames, *_closed_frames(c, event.closed)]


def _closed_frames(conversation: ConversationState, closed: EditClosed) -> list[BaseModel]:
    """The frames that end the edit on `closed`, with the turn recorded."""
    return [
        *closed_on_reply(
            conversation,
            closed.text,
            suggestions=True,
            say=closed.shown,
            user_display=closed.user_display,
            held_line=closed.held_line,
        )
    ]


async def unit_events(
    *,
    recorder: _RecordingExecutor,
    surface: ToolSurface | None,
    conversation: ConversationState,
    move: UnitMove,
) -> AsyncGenerator[EditEvent]:
    """The events of a card's unit change (`TurnRequest.apply_unit`), in code
    with no model call: Search reads the card's spec in the new unit
    (`harness.unit_switch.resubmit_unit`, through `surface`), `get_data`
    fetches it with the arguments that shaped the card's frame
    (`harness.card_unit.UnitCard.args`), and the card is written again from
    its stored intent on the new frame and takes its own slot, as an Edit
    Card recreate does. Its title says the new unit where it named the old
    (`harness.card_unit.retitled`). The new card carries a fresh
    `unit_change` starting on the new unit.

    The turn's record names the change and the new spec_id
    (`ConversationState.turn_user_message`), and the new reading joins the
    turn's readings with its block, so the next turn's model reads the card
    in the new unit. A step Search or the platform refuses ends the turn on
    its sentence, the card unchanged. Failures serialise as `edit_events`'
    do."""
    try:
        async for event in _unit_flow(
            recorder=recorder, surface=surface, conversation=conversation, move=move
        ):
            yield event
    except Exception as exc:
        logger.exception("studio unit change failed (trace_id=%s)", conversation.trace_id)
        yield EditFailed(exc)


# What a closing message says before the reason for a unit change that did not land.
_UNIT_UNCHANGED = "The card is unchanged."


async def _unit_flow(
    *,
    recorder: _RecordingExecutor,
    surface: ToolSurface | None,
    conversation: ConversationState,
    move: UnitMove,
) -> AsyncGenerator[EditEvent]:
    """`unit_events`' steps, their frames numbered by `edit_event_frames`."""
    words = unit_words(move.unit, move.offer)
    in_sentence = words[:1].lower() + words[1:]
    if surface is None:
        for event in _close("Changing a card's unit is not enabled on this deployment."):
            yield event
        return
    described = await recorder(tool="describe_component", args={"component_id": move.component_id})
    if described.get("outcome") != "ok":
        for event in _refused("u1", "Read the card", _UNIT_UNCHANGED, described):
            yield event
        return
    stored = (described.get("component_body") or {}).get("stored_intent") or {}
    yield _step("u1", "Read the card")

    label = f"Read the data in {in_sentence}"
    resubmitted = await resubmit_unit(
        surface,
        reading=move.reading,
        block=move.block,
        unit=move.unit.model_dump(mode="json", exclude_none=True),
        label=move.title,
    )
    if isinstance(resubmitted, UnitRefused):
        yield _step("u2", label, failed=resubmitted.reason)
        for event in _close(f"{_UNIT_UNCHANGED} {resubmitted.reason}"):
            yield event
        return
    fetched = await recorder(
        tool="get_data", args={"spec_id": resubmitted.spec_id, **move.card.args}
    )
    frame_refs = fetched.get("frame_refs") or []
    if fetched.get("outcome") != "ok" or not frame_refs:
        for event in _refused("u2", label, _UNIT_UNCHANGED, fetched):
            yield event
        return
    yield _step("u2", label)

    # The card left the canvas while its new unit was read: the change is
    # refused as the route refuses a card off the canvas, and nothing is
    # written.
    gone = (_step("u3", "Redraw the card", failed=UNIT_GONE_NOTE), EditClosed(UNIT_GONE_NOTE))
    if _block_index(conversation.composition, move.component_id) is None:
        for event in gone:
            yield event
        return

    def retitle(text: str) -> str:
        return retitled(text, old=move.served, new=move.unit, offer=move.offer)

    intent = {**stored, "frame_ref": frame_refs[0]}
    if isinstance(stored.get("title"), str):
        intent["title"] = retitle(stored["title"])
    held = intent.get("order")
    if isinstance(held, dict) and "period" in held:
        # The new unit can hold values at other periods than the old.
        intent["order"] = _without_period(held)
    written = await recorder(tool="write_component", args={"intent": intent})
    if written.get("outcome") != "ok" or not written.get("component"):
        for event in _refused("u3", "Redraw the card", _UNIT_UNCHANGED, written):
            yield event
        return
    block = dict(resubmitted.spec)
    readings = _read_again(move, resubmitted)
    record_line = f"Changed {move.title} to {in_sentence} ({resubmitted.spec_id})."
    display_line = f"Change {move.title} to {in_sentence}."

    def record(work: ConversationState) -> None:
        work.turn_user_message = record_line
        work.turn_readings = [*work.turn_readings, *readings]

    # The card shows the change; the record keeps the sentence and the
    # spec_id for the next turn's model, and a state read shows neither.
    closed = EditClosed(
        f"The card now shows {in_sentence}.", shown=False, user_display=display_line
    )
    yield _swapped(
        recorder,
        conversation,
        component=written["component"],
        old_component_id=move.component_id,
        step_id="u3",
        step_label="Redraw the card",
        closed=closed,
        specs={resubmitted.spec_id: block},
        also=record,
        retitle=retitle,
        # The card left the canvas while it was written.
        gone=gone,
    )


def _read_again(move: UnitMove, resubmitted: UnitResubmitted) -> list[dict[str, Any]]:
    """The readings, as JSON, of the part Search answered for the card's spec
    in the new unit, named and asked as the reading it replaces."""
    digest = resolver_digest({"parts": [dict(resubmitted.part)]})
    reading = move.reading
    for read in digest.get("part_digests") or ():
        read["variable"] = (reading.text if reading is not None else None) or move.title
    if reading is not None:
        mark_planner_axes(digest, reading.planner_axes)
    state: dict[str, Any] = {}
    store_search(
        state,
        digest,
        asked=[reading.asked_roles] if reading is not None else (),
        held_open=[reading.open_roles] if reading is not None else (),
        share_sides=[reading.share_over] if reading is not None else (),
    )
    return list(state.get(ORCHESTRATOR_SEARCHES_KEY) or ())


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
        revise_args: dict[str, Any] = {
            "intent": patch,
            "component_id": component_id,
            WRITE_TOKEN_KEY: write_token,
        }
        if stored.get("frame_ref"):
            # The executor reads the card's lineage by this frame, which the
            # card's stored intent read above names.
            revise_args[CARD_FRAME_KEY] = stored["frame_ref"]
        written = await recorder(tool="write_component", args=revise_args)
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
            lineage = recorder.card_lineage(component)
            runs = card_runs(component, lineage)

            def revise(work: ConversationState) -> bool:
                if token:
                    work.write_tokens = {**work.write_tokens, component_id: token}
                _record_revision(
                    work,
                    component_id=component_id,
                    component=component,
                    revision=revised.revision,
                    title=title,
                    runs=runs,
                )
                return _refresh_block(
                    work, component_id=component_id, revision=revised.revision, title=title
                )

            frames.append(CanvasChange(revise, lineage=lineage))
            held_line = _held_line(
                conversation,
                component_id=component_id,
                runs=runs,
                reframed=_reframes(stored, _merge_patch(stored, patch)),
            )
            frames.append(EditClosed(text or "Done -- the card was updated.", held_line=held_line))
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
    yield _swapped(
        recorder,
        c,
        component=written["component"],
        old_component_id=component_id,
        step_id="e2",
        step_label=step_label,
        closed=EditClosed(closing),
        reframed=_reframes(stored, intent),
    )


def _swapped(
    recorder: _RecordingExecutor,
    conversation: ConversationState,
    *,
    component: dict[str, Any],
    old_component_id: str,
    step_id: str,
    step_label: str,
    closed: EditClosed,
    specs: Mapping[str, Any] | None = None,
    also: Callable[[ConversationState], None] | None = None,
    retitle: Callable[[str], str] | None = None,
    gone: tuple[ActivityStep, EditClosed] | None = None,
    reframed: bool = False,
) -> CardSwap:
    """The swap of `component`, a card written to take card
    `old_component_id`'s slot, ending the edit on `closed`: the step
    `step_id`, labelled `step_label`, that produced it, its
    `component.created` naming that step, and the write that swaps it in,
    with its digest entry, turn line and write token. The entry of a card
    that took the slot names the old card (`REPLACED_CARD_KEY`), as its
    `component.created` does in `replaces`. A card whose unit can
    change carries its `unit_change` and keeps its `UnitCard` with its entry
    (`harness.card_unit`), its Search block read from `specs` and else from
    the conversation. `also` makes the write's other changes, and `specs`
    joins the turn's blocks once it commits. `retitle` marks a unit change,
    which `_replace_block` carries into the slot's card meta and layout
    hints. Given `gone`, the swap needs the old card's slot: a canvas that
    no longer holds it, read under the write's lock, takes only the new
    card's write token, so the conversation keeps the card it wrote off the
    canvas, and the edit ends on `gone`. A `closed` that is shown carries the
    held line (`_held_line`), `reframed` saying whether the new card draws
    another window or frame than the old."""
    created = component_created(component, step_id=step_id)
    token = recorder.write_token(component)
    lineage = recorder.card_lineage(component)
    card = unit_card(component, lineage)
    if card is not None:
        block = (specs or {}).get(card.spec_id) or stored_block(
            conversation.turn_specs, conversation.turns, card.spec_id
        )
        created.unit_change = card_unit_change(card, block)
    kept = card.as_json() if card is not None and created.unit_change is not None else None
    runs = card_runs(component, lineage)
    if closed.shown:
        held_line = _held_line(
            conversation, component_id=old_component_id, runs=runs, reframed=reframed
        )
        closed = replace(closed, held_line=held_line)

    def swap(work: ConversationState) -> SwapLanded:
        if token:
            work.write_tokens = {**work.write_tokens, created.component_id: token}
        if (
            gone is not None
            and _block_index(work.composition, old_component_id) is None
            and _block_index(work.composition, created.component_id) is None
        ):
            return SwapLanded.GONE
        entry = _digest_entry(component)
        if kept is not None:
            entry[CARD_UNIT_KEY] = kept
        work.components[created.component_id] = entry
        work.turn_components.append(
            _turn_component_entry(
                component_id=created.component_id,
                type=created.type,
                title=_block_title(component),
                action="recreated",
                window=entry.get("time"),
                runs=runs,
            )
        )
        if also is not None:
            also(work)
        swapped = _replace_block(
            work, old_component_id=old_component_id, component=component, retitle=retitle
        )
        if not swapped:
            return SwapLanded.HELD
        entry[REPLACED_CARD_KEY] = old_component_id
        return SwapLanded.SWAPPED

    return CardSwap(
        step=_step(
            step_id,
            step_label,
            produced=StepProduced(component_id=created.component_id, revision=created.revision),
        ),
        created=created,
        replaces=old_component_id,
        swap=swap,
        lineage=lineage,
        specs=specs,
        closed=closed,
        gone=gone,
    )


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


def _held_line(
    conversation: ConversationState,
    *,
    component_id: str,
    runs: Mapping[str, Any],
    reframed: bool,
) -> str | None:
    """The sentences on the partial and projected years the edited card
    draws or left out (`harness.turn_records.held_back_line` over its
    `runs`, `harness.studio_state.card_runs`), as a turn drawing the card
    closes. They are said only where the edit changed what the card reads:
    its window or frame (`reframed`), or the runs it states, against the
    latest turn entry of card `component_id`. A title-only edit reads no
    data and says none."""
    if not reframed and runs.get(TURN_CARD_RUNS) == _entry_runs(conversation, component_id):
        return None
    return held_back_line([runs], projection_asked=False)


def _entry_runs(conversation: ConversationState, component_id: str) -> Any:
    """The runs (`TURN_CARD_RUNS`) of the latest turn entry of card
    `component_id`, the open turn's last; `None` when no entry states any."""
    records = [record.get("cards") or [] for record in conversation.turns]
    entries = [
        card
        for cards in [*records, conversation.turn_components]
        for card in cards
        if isinstance(card, Mapping) and card.get("component_id") == component_id
    ]
    return entries[-1].get(TURN_CARD_RUNS) if entries else None


def _reframes(stored: Mapping[str, Any], intent: Mapping[str, Any]) -> bool:
    """Whether `intent` draws another window or frame than the card's
    `stored` intent."""
    return any(intent.get(key) != stored.get(key) for key in ("window", "frame_ref"))


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
    runs: Mapping[str, Any] | None = None,
) -> None:
    """Bring a landed revise's digest entry up to `revision` and add its turn
    record, with the card's partial and projected runs (`runs`,
    `harness.studio_state.card_runs`). `title` is the card's new title, None
    when the edit left it. The
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
            runs=runs,
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
    revision. `StudioStore.change_canvas` runs it once per committed
    write."""
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
