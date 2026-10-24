"""The Edit Card route: `component_ref.intent: "edit"`.

No resolve, no plan, no ADK session: the card is described, the model
decides between a revise patch (title / select) and a full recreate (type,
window, frame_ref), and the platform write streams back as
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
"""

from __future__ import annotations

import functools
import logging
from collections.abc import AsyncGenerator
from typing import Any

from common.mcp_telemetry import WRITE_TOKEN_KEY

from harness.conversation import current_conversation_id
from harness.studio_state import (
    ConversationState,
    _block_title,
    _digest_entry,
    _next_manifest,
    _replace_block,
    _turn_component_entry,
    drawn_series_ids,
    forget_drawn_card,
)
from harness.studio_translate import _RecordingExecutor, _stream_fatal, failure_detail
from harness.tool_contracts import HarnessCode, harness_refusal
from harness.wire import (
    ActivityStep,
    ComponentCreated,
    ComponentRevised,
    CompositionActor,
    CompositionUpdated,
    Done,
    MessageComplete,
    StepProduced,
    StepStatus,
    Suggestions,
)

logger = logging.getLogger(__name__)

# Intent fields a revise patch may not name, as apis
# `assembler.merge.merge_intent` refuses them.
_CREATE_ONLY_FIELDS = ("type", "frame_ref")

# What a closing message says before the sentence for a refused edit's code.
_EDIT_REFUSED = "The edit was refused."

_REBUILT_TEXT = (
    "This conversation holds no edit access to this card, so I could not update it in "
    "place. I rebuilt it as a new card with your change: it has a new id and starts a "
    "fresh revision history, and the original card is unchanged."
)


async def edit_frames(
    *,
    recorder: _RecordingExecutor,
    edit_model: Any,
    conversation: ConversationState,
    component_id: str,
    message: str,
) -> AsyncGenerator[Any]:
    """The Edit Card route: no resolve, no plan.

    Failures serialise INTO the stream as a stream-fatal error frame,
    exactly like `_frames` -- a malformed model draft must not kill the
    transport with no error frame.
    """
    try:
        async for frame in _edit_flow(
            recorder=recorder,
            edit_model=edit_model,
            conversation=conversation,
            component_id=component_id,
            message=message,
        ):
            yield frame
    except Exception as exc:
        logger.exception("studio edit turn failed (trace_id=%s)", conversation.trace_id)
        yield _stream_fatal(conversation, exc)


async def _edit_flow(
    *,
    recorder: _RecordingExecutor,
    edit_model: Any,
    conversation: ConversationState,
    component_id: str,
    message: str,
) -> AsyncGenerator[Any]:
    """Verified platform rules (2026-08-28, hosted dev): a revision may
    change only title/select; type/window/frame_ref edits recreate the
    component instead, and the studio replaces the block in place. The
    old component stays readable at its prior revisions.
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
            seq=c.next_seq(),
            step_id=step_id,
            label=label,
            status=StepStatus.FAILED if failed else StepStatus.COMPLETE,
            detail=failed,
            produced=produced,
        )

    def _close(text: str) -> list[Any]:
        c.turn_final_text = text
        c.close_turn(final_text=text)
        return [
            MessageComplete(event="message.complete", seq=c.next_seq(), text=text),
            Suggestions(event="suggestions", seq=c.next_seq(), items=[]),
            Done(event="done", seq=c.next_seq(), trace_id=c.trace_id, composition=c.composition),
        ]

    def _refused(step_id: str, label: str, lead: str, result: dict[str, Any]) -> list[Any]:
        """The failed step and the closing message of a call that was refused.
        Both give the sentence for the refusal's code. Its message, which is
        written for a model, goes to the log."""
        refusal = result.get("refusal")
        refusal = refusal if isinstance(refusal, dict) else {}
        logger.info("card edit refused (%s): %s", refusal.get("code"), refusal.get("message"))
        sentence = failure_detail(refusal)
        return [_step(step_id, label, failed=sentence), *_close(f"{lead} {sentence}")]

    described = await recorder(tool="describe_component", args={"component_id": component_id})
    recorder.drain(current_conversation_id.get())
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
        recorder.drain(current_conversation_id.get())
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
        recorder.drain(current_conversation_id.get())
        moved = (written.get("refusal") or {}).get(
            "code"
        ) == HarnessCode.ROLLING_SELECT_NEEDS_NEW_FRAME
    if patch and write_token and not moved:
        if written.get("outcome") != "ok" or not written.get("component"):
            frames = _refused("e2", "Apply the edit", _EDIT_REFUSED, written)
        elif int(written["component"]["revision"]) <= current_revision:
            # The platform treats an effectively-equal patch as a no-op.
            yield _step("e2", "Apply the edit")
            frames = _close("The card already looks like that; nothing changed.")
        else:
            component = written["component"]
            revision = int(component["revision"])
            yield _step(
                "e2",
                "Apply the edit",
                produced=StepProduced(component_id=component_id, revision=revision),
            )
            frames = [
                *_revised_frames(
                    conversation, component_id=component_id, component=component, patch=patch
                ),
                *_close(text or "Done -- the card was updated."),
            ]
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
        intent = {**recreate, "frame_ref": stored.get("frame_ref")}
        step_label = "Rebuild the card"
        closing = text or "Done -- the card was rebuilt with your change."
    written = await recorder(tool="write_component", args={"intent": intent})
    recorder.drain(current_conversation_id.get())
    if written.get("outcome") != "ok" or not written.get("component"):
        for frame in _refused("e2", step_label, _EDIT_REFUSED, written):
            yield frame
        return
    component = written["component"]
    # The step frame is yielded FIRST and its seq allocated first --
    # constructing the created frame earlier gave it a lower seq than
    # the step that precedes it on the wire (broke delta 2).
    yield _step(
        "e2",
        step_label,
        produced=StepProduced(
            component_id=component["component_id"], revision=int(component["revision"])
        ),
    )
    created = ComponentCreated.model_validate(
        {
            **component,
            "event": "component.created",
            "seq": c.next_seq(),
            "created_from": {"step_id": "e2"},
        }
    )
    yield created
    entry = _digest_entry(component)
    conversation.components[created.component_id] = entry
    conversation.turn_components.append(
        _turn_component_entry(
            component_id=created.component_id,
            type=created.type,
            title=_block_title(component),
            action="recreated",
            window=entry.get("time"),
        )
    )
    if _replace_block(conversation, old_component_id=component_id, component=component):
        yield CompositionUpdated(
            event="composition.updated", seq=c.next_seq(), manifest=c.composition
        )
    for frame in _close(closing):
        yield frame


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


def _revised_frames(
    conversation: ConversationState,
    *,
    component_id: str,
    component: dict[str, Any],
    patch: dict[str, Any],
) -> list[Any]:
    revision = int(component["revision"])
    frames: list[Any] = [
        ComponentRevised(
            event="component.revised",
            seq=conversation.next_seq(),
            component_id=component_id,
            revision=revision,
            changed=sorted(patch),
            fact_summary=component["fact_summary"],
            payload=component.get("payload"),
        )
    ]
    entry = conversation.components.setdefault(component_id, _digest_entry(component))
    entry["revision"] = revision
    entry["fact_summary"] = component.get("fact_summary") or entry.get("fact_summary")
    entry.update(drawn_series_ids(component))
    forget_drawn_card(entry)
    if patch.get("title"):
        entry["title"] = patch["title"]
    conversation.turn_components.append(
        _turn_component_entry(
            component_id=component_id,
            type=entry.get("type"),
            title=entry.get("title"),
            action="revised",
            window=entry.get("time"),
        )
    )
    if _refresh_block(
        conversation,
        component_id=component_id,
        revision=revision,
        title=patch.get("title"),
    ):
        frames.append(
            CompositionUpdated(
                event="composition.updated",
                seq=conversation.next_seq(),
                manifest=conversation.composition,
            )
        )
    return frames


def _refresh_block(
    conversation: ConversationState,
    *,
    component_id: str,
    revision: int,
    title: str | None,
) -> bool:
    """Update the edited component's block; True iff a block changed (a
    not-composed card mints no empty manifest revision). Recorded as a
    composition op (`ConversationState._composition_ops`), the same
    contract `harness.studio_state.land_component`/`_replace_block` follow:
    a checkpoint reconciling this segment's composition against a freshly
    fetched ledger manifest replays this revise onto it, rather than
    discarding it in favour of the fetch (`StudioStore.checkpoint`'s
    docstring)."""
    previous = conversation.composition
    if previous is None:
        return False
    op = functools.partial(
        _refresh_block_at, component_id=component_id, revision=revision, title=title
    )
    updated = op(previous)
    if updated is previous:
        return False
    conversation.composition = updated
    conversation._composition_ops.append(op)
    return True


def _refresh_block_at(previous: Any, *, component_id: str, revision: int, title: str | None) -> Any:
    """The next manifest with `component_id`'s block revised, a pure
    function of `previous`. This is `_refresh_block`'s replay step,
    reproducing this edit on top of whatever the ledger holds when a
    checkpoint reconciles it."""
    if previous is None:
        return None
    blocks = [block.model_copy(deep=True) for block in previous.blocks]
    changed = False
    for block in blocks:
        if block.component_id == component_id:
            block.revision = revision
            if title:
                block.title = title
            changed = True
    if not changed:
        return previous
    return _next_manifest(previous, blocks, actor=CompositionActor.MODEL)
