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

import asyncio
import functools
import json
import logging
import re
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from common.mcp_telemetry import WRITE_TOKEN_KEY
from compiler.surface import ToolId, ToolSurface, ToolSurfaceUnavailableError
from pydantic import BaseModel

from harness.card_unit import (
    CARD_UNIT_KEY,
    UNIT_GONE_NOTE,
    ChangeMove,
    ConversationBlocks,
    ScopeYears,
    SummedFrame,
    UnitCard,
    UnitMove,
    bound_facets,
    card_changes,
    card_unit_change,
    drawn_places,
    encoded,
    joined_words,
    place_labels,
    retitled,
    root_facets,
    scope_needs_years,
    select_entry,
    stored_block,
    strayed_facets,
    unit_card,
    unit_pattern,
    unlike_series,
)
from harness.conversation import current_conversation_id, current_conversation_key
from harness.graph.state import ORCHESTRATOR_SEARCHES_KEY, store_search
from harness.provenance import CARD_FRAME_KEY, CardLineage, ProvenanceStore
from harness.scope_choices import mark_planner_axes
from harness.studio_state import (
    PLAN_LINK_KEY,
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
from harness.unit_offer import served_unit, unit_words
from harness.unit_switch import UnitRefused, UnitResubmitted, resubmit_unit, scope_years
from harness.wire import (
    ActivityStep,
    CardChanges,
    ComponentCreated,
    CompositionActor,
    CompositionUpdated,
    PlacesChange,
    StepProduced,
    StepStatus,
    WindowChange,
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
    places: bool = False,
) -> AsyncGenerator[EditEvent]:
    """The Edit Card route's events: no resolve, no plan. A recreated card
    offers its places where `places` says Search's places door is on.

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
            places=places,
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


async def change_events(
    *,
    recorder: _RecordingExecutor,
    surface: ToolSurface | None,
    conversation: ConversationState,
    move: UnitMove | ChangeMove,
    places: bool = False,
    provenance: ProvenanceStore | None = None,
) -> AsyncGenerator[EditEvent]:
    """The events of a card's change in code with no model call: its unit
    (`TurnRequest.apply_unit`, a `UnitMove`), or its years or places
    (`TurnRequest.apply_change`, a `ChangeMove`).

    Every change reads the card, reads its data and redraws it
    (`_change_flow`). A unit change has Search read the card's spec in the
    new unit (`harness.unit_switch.resubmit_unit`, through `surface`). A
    change of years or places reads the card's spec over the new `scope`.
    Either way `get_data` fetches the spec with the arguments that shaped the
    card's frame (`harness.card_unit.UnitCard.args`) and the scope the card
    carries, and the card is written again from its stored intent on the new
    frame and takes its own slot, as an Edit Card recreate does. A unit
    change names the new unit in its title and meta where they named the old
    (`harness.card_unit.retitled`). A change of years or places builds its
    title again from its parts and drops the meta words that may name the
    old ones (`_redrawn_intent`). The new card carries fresh offers, places
    among them where `places` says Search's places door is on, and the scope
    it was fetched over, so a later change keeps this one.

    A card on a sum of its retrieval (`harness.card_unit.UnitCard.total`)
    changes its years or places the same way, with one more call: once the
    retrieval is fetched, analysis-mcp's `aggregate` is called through
    `surface` with the parameters the sum's recipe recorded and the new
    frame, and the card is redrawn on that sum. The sum's lineage is written
    into `provenance` from the new retrieval's, so the card's write finds its
    sources. A sum refused, or one whose lineage cannot be written, ends the
    change as a refused fetch does.

    The turn's record (`ConversationState.turn_user_message`) names the
    change and the spec_id, with the scope a change of years or places
    fetched over, which the new card's entry keeps too (`FETCHED_KEY`). A
    unit change's new reading joins the turn's readings with its block, so
    the next turn's model reads the card as it is. A step Search or the
    platform refuses, and a retrieval that holds no figures, ends the turn on
    its sentence, the card unchanged. Failures serialise as `edit_events`'
    do."""
    try:
        async for event in _change_flow(
            recorder=recorder,
            surface=surface,
            conversation=conversation,
            move=move,
            places=places,
            provenance=provenance,
        ):
            yield event
    except Exception as exc:
        logger.exception("studio card change failed (trace_id=%s)", conversation.trace_id)
        yield EditFailed(exc)


# The key of a card's digest entry naming the `get_data` arguments a change
# fetched it with, which the model reads, so a follow-up fetch draws what the
# card shows.
FETCHED_KEY = "fetched_with"

# The key of the digest entry of a card on a sum naming the analysis step that
# made its frame from that fetch, with the arguments it was called with
# besides the frame, which the model reads beside `FETCHED_KEY`.
SUMMED_KEY = "summed_with"


# What a closing message says before the reason for a change that did not land.
_UNIT_UNCHANGED = "The card is unchanged."
# The step and closing sentence of a retrieval that holds no figures for the
# years or places asked.
_NO_DATA = "The data holds no figures for that choice."
# The closing sentence of a redraw whose data counts another source, method or
# unit than the card did, which binding the card's facets rules out.
_OTHER_BASIS = "The data for that choice is counted another way."
# The step and closing sentence of a sum whose new retrieval holds two units,
# sources or methods, which the sum would add together.
_MIXED_BASIS = "The data for that choice is counted more than one way, so it cannot be totalled."
# The step and closing sentence of a sum that could not be worked out again
# over the new data: analysis-mcp unreachable, or its answer or the
# retrieval's lineage unread.
_SUM_UNREAD = "The total for that choice could not be worked out."

# The analysis-mcp tool a card on a sum is summed again by.
_AGGREGATE = ToolId(server="analysis-mcp", name="aggregate")


@dataclass(frozen=True)
class _Read:
    """What a change reads its data as: the spec and scope `get_data`
    fetches, the Search blocks and readings it adds to the turn, and its words
    for the step label and the record ("in constant 2023 US dollars",
    "for 2010 to 2024")."""

    spec_id: str
    scope: dict[str, Any] | None
    specs: dict[str, Any]
    readings: list[dict[str, Any]]
    phrase: str


async def _read_change(surface: ToolSurface, move: UnitMove | ChangeMove) -> _Read | UnitRefused:
    """What `move` fetches: a unit change resubmits the card's spec in the new
    unit and keeps the card's scope. A change of years or places reads the
    card's spec over its new scope."""
    if isinstance(move, ChangeMove):
        return _Read(
            spec_id=move.card.spec_id,
            scope=move.scope,
            specs={},
            readings=[],
            phrase=f"for {move.words}",
        )
    words = unit_words(move.unit, move.offer)
    resubmitted = await resubmit_unit(
        surface,
        reading=move.reading,
        block=move.block,
        unit=move.unit.model_dump(mode="json", exclude_none=True),
        label=move.title,
        facets=move.card.bindable_facets,
    )
    if isinstance(resubmitted, UnitRefused):
        return resubmitted
    return _Read(
        spec_id=resubmitted.spec_id,
        scope=move.card.scope,
        specs={resubmitted.spec_id: dict(resubmitted.spec)},
        readings=_read_again(move, resubmitted),
        phrase=f"in {words[:1].lower()}{words[1:]}",
    )


def _redrawn_intent(
    stored: Mapping[str, Any],
    frame_ref: Any,
    move: UnitMove | ChangeMove,
    *,
    family: tuple[Any, Any] | None = None,
) -> tuple[dict[str, Any], Callable[[str, str], str | None]]:
    """The intent `move` redraws its card from, on `frame_ref`, and how it
    rewrites each word of the card's meta, by name (`None` drops it).

    `family` is the spec the old retrieval was served under and the spec the
    new one is. apis names a summed retrieval's series by its family ref,
    `dcid:<spec_id>` of that spec (`data_mcp.frame_materialisation
    .holds_family_ref`), so a `stat_var` entry naming the old one names the
    new one, which the new frame holds in its place.

    Every change drops the `order`'s period, since the new frame can hold
    values at other periods. A unit change names the new unit in the title
    and the meta where they named the old (`harness.card_unit.retitled`,
    built from the unit offer's own words). A change of years or places
    builds the title again from its parts (`_changed_title`) and drops the
    model's meta words that may name the old years or places
    (`_changed_meta`). A change of years drops the intent's `window`, a
    period pinned on the old frame or a count of latest periods that would
    override the new years. A change of places keeps it, since the years do
    not change, and names the new places in the role's own `select` entry
    where it named the role's old ones (`_reselected`); an entry of another
    role keeps its places, even one naming the same place, since the new
    frame keeps that role's places."""
    intent = {**stored, "frame_ref": frame_ref}
    select = intent.get("select")
    old_spec, new_spec = family or (None, None)
    if isinstance(select, Mapping) and isinstance(old_spec, str) and isinstance(new_spec, str):
        named = select.get("stat_var")
        if named is not None:
            reselected = _reselected(named, {old_spec}, [new_spec])
            intent["select"] = {**select, "stat_var": reselected}
    title = stored.get("title")
    match move:
        case UnitMove():

            def retitle(text: str) -> str:
                return retitled(text, old=move.served, new=move.unit, offer=move.offer)

            if isinstance(title, str):
                intent["title"] = retitle(title)

            def rewrite(_name: str, text: str) -> str | None:
                return retitle(text)

        case ChangeMove(change=WindowChange()):
            intent.pop("window", None)
            intent["title"] = _changed_title(title, move)
            rewrite = functools.partial(
                _changed_meta,
                shown=_shown_labels(move),
                unit=_unit_said(move),
                summed=move.card.total is not None,
            )
        case ChangeMove(change=PlacesChange() as change):
            places = next(p for p in move.offer.places or () if p.role == change.role)
            old = {o.id for o in places.chosen}
            select = intent.get("select")
            if isinstance(select, Mapping):
                # A select is keyed by the frame's columns, so a flow role's
                # places sit under its party column.
                entry = select_entry(select, change.role)
                intent["select"] = {
                    key: _reselected(value, old, change.entities) if key == entry else value
                    for key, value in select.items()
                }
            intent["title"] = _changed_title(title, move)
            rewrite = functools.partial(
                _changed_meta,
                shown=_shown_labels(move),
                unit=_unit_said(move),
                summed=move.card.total is not None,
            )
    held = intent.get("order")
    if isinstance(held, dict) and "period" in held:
        intent["order"] = _without_period(held)
    return intent, rewrite


def _reselected(value: Any, old: set[str], new: list[str]) -> Any:
    """`value`, a `select` entry, naming the places `new` where it names any of
    `old`, the places a role showed: a list for a list or several places, in
    the entry's own `dcid:` form."""
    values = value if isinstance(value, list) else [value]
    if not any(isinstance(v, str) and v.removeprefix("dcid:") in old for v in values):
        return value
    marked = any(isinstance(v, str) and v.startswith("dcid:") for v in values)
    named = [f"dcid:{ref}" if marked else ref for ref in new]
    return named if isinstance(value, list) or len(named) > 1 else named[0]


# The period `orchestrator_tools._card_title` ends a title on: ", 2015 to 2023",
# ", 2023", ", 2015/16", ", latest year", ", last 5 years", and the dashed
# range a model writes, ", 1960-2024" or with an en dash.
_PERIOD_TAIL = re.compile(
    r",\s+(?:\d{4}(?:/\d{2,4})?(?:(?:\s+to\s+|\s*[-\u2013]\s*)\d{4}(?:/\d{2,4})?)?"
    r"|latest \w+|last \d+ \w+)$"
)
_ANY_YEAR = re.compile(r"(?<!\d)\d{4}(?!\d)")
# The longest title `orchestrator_tools._card_title` writes.
_TITLE_CHARS = 120


def period_words(start: int, end: int) -> str:
    """A period as a card title states it: "2023", "2010 to 2024"."""
    return str(start) if start == end else f"{start} to {end}"


def _unit_said(move: ChangeMove) -> str | None:
    """The pattern of the phrases that name the unit the card is served in
    (`harness.card_unit.unit_pattern`), so the year in "constant 2023 US
    dollars" reads as no period. It names the unit by the offer's words where
    the unit can change, else by the served unit's (`served_unit`) with its
    currency by code. `None` where the block states no unit."""
    change = card_unit_change(move.card, move.block)
    if change is not None:
        return unit_pattern(change.offer.start, change.offer)
    served = served_unit(move.block)
    return unit_pattern(served, None) if served is not None else None


def _names_a_year(text: str, unit: str | None) -> bool:
    """Whether `text` names a four-digit year outside the unit's own words."""
    if unit is not None:
        text = re.sub(unit, " ", text, flags=re.IGNORECASE)
    return _ANY_YEAR.search(text) is not None


def _title_subject(title: str, unit: str | None = None) -> str | None:
    """`title` without the period `make_charts` ended it on, or the whole title
    when it ends on none. `None` when what is left still names a year outside
    the unit's words (`unit`, `_unit_said`), such as a cut's suffix or a
    period the title words in its own way."""
    subject = _PERIOD_TAIL.sub("", title, count=1)
    return None if _names_a_year(subject, unit) else subject


def _cut_words(text: str, room: int) -> str:
    """`text` cut at a word boundary to at most `room` characters."""
    if len(text) <= room:
        return text
    cut = text[: max(room, 0) + 1].rsplit(" ", 1)[0].rstrip(" ,")
    return cut if len(cut) <= room else text[: max(room, 0)]


def _changed_title(title: Any, move: ChangeMove) -> str:
    """The title of the card `move` redraws, so it names no year or place the
    card no longer shows.

    A change of years keeps the subject the title opened on
    (`_title_subject`) and states the new period after it, whole. A change
    of places renames the places in the title where it can
    (`_renamed_places`), else builds the title from the Search block's
    facts (`_composed_title`) with the period the title ended on, or the
    card's years where it ended on none (its offer's, else the ones it draws). A title whose subject cannot be
    read, or a card with no title, is built from the block's facts too.

    A card on a sum (`harness.card_unit.UnitCard.total`) whose title counts
    the members it adds up (`_counts_members`, "from 32 donors") is built
    from the block's facts, with the new period or the one its title ended
    on, since other years or places can add up another number."""
    text = title if isinstance(title, str) else ""
    card = replace(move.card, scope=move.scope)
    summed = move.card.total is not None and _counts_members(text, move)
    if isinstance(move.change, WindowChange):
        period = period_words(move.change.start_year, move.change.end_year)
        subject = _title_subject(text, _unit_said(move)) if text and not summed else None
        if subject:
            return f"{_cut_words(subject, _TITLE_CHARS - len(period) - 2)}, {period}"
        return _composed_title(move.block, card, period)
    renamed = None if summed else _renamed_places(text, move)
    if renamed is not None:
        return renamed
    tail = _PERIOD_TAIL.search(text)
    window = move.offer.window
    if tail is not None:
        period = tail.group(0).removeprefix(",").strip()
    elif window is not None:
        period = period_words(window.start_year, window.end_year)
    elif move.drawn is not None:
        period = period_words(*move.drawn)
    else:
        period = None
    return _composed_title(move.block, card, period)


def _counts_members(title: str, move: ChangeMove) -> bool:
    """Whether `title`, before the period it ends on and outside the unit's
    words (`_unit_said`, so "constant 2023 US dollars" counts nothing), holds
    a number other than a year the card's offer spans: a count ("32 donors",
    "8 largest of 36") that another sum may not share."""
    subject = _PERIOD_TAIL.sub("", title)
    if (unit := _unit_said(move)) is not None:
        subject = re.sub(unit, " ", subject, flags=re.IGNORECASE)
    window = move.offer.window
    years = range(window.first_year, window.last_year + 1) if window is not None else range(0)
    return any(int(number) not in years for number in re.findall(r"\b\d+\b", subject))


def _named(label: str) -> str:
    """The pattern of `label` as a whole word or phrase."""
    return rf"(?<!\w){re.escape(label)}(?!\w)"


def _renamed_places(title: str, move: ChangeMove) -> str | None:
    """`title` naming the places the role now shows where it named the ones it
    showed (`move.old`, `move.new`): their words as a reader lists them
    (`joined_words`) replaced whole; else, for a change swapping places one
    for one, each place it left named once in the title replaced by the one
    that took its place. `None` when neither holds, when the title already
    names a place the change brings, or when a place the card left is still
    named after it."""
    old, new = list(move.old), list(move.new)
    if not title or not old:
        return None
    left = [label for label in old if label not in new]
    arrived = [label for label in new if label not in old]
    if any(re.search(_named(label), title) for label in arrived):
        # "Aid to Kenya compared with Uganda" would read "Uganda compared
        # with Uganda".
        return None
    joined = joined_words(old)
    renamed: str | None = None
    if re.search(_named(joined), title):
        renamed = re.sub(_named(joined), lambda _: joined_words(new), title, count=1)
    elif (
        len(old) == len(new)
        and len(left) == len(arrived)
        and all(len(re.findall(_named(label), title)) == 1 for label in left)
    ):
        renamed = title
        for gone, came in zip(left, arrived, strict=True):
            renamed = re.sub(_named(gone), lambda _, came=came: came, renamed, count=1)
    if renamed is None or len(renamed) > _TITLE_CHARS:
        return None
    if any(re.search(_named(label), renamed) for label in left):
        return None
    return renamed


def _composed_title(block: Mapping[str, Any], card: UnitCard, period: str | None) -> str:
    """A title from `block`'s facts: what it measures (the words its sentence
    opens on, its `sentence_plan`'s `what` slot, else `concept_label`, else
    `measure_label`, else its shape's label), the places `card` shows, and
    `period`, whole and last, in at most `_TITLE_CHARS`. The places give way
    before the subject does: listed as far as they fit, then counted ("12
    places"), then left out, as they are when a place has no name
    (`_places_words`). The subject is cut, at a word, only when it alone
    leaves no room for the period."""
    resolved = block.get("resolved") or {}
    heading = next(
        (
            part.get("text")
            for part in resolved.get("sentence_plan") or ()
            if isinstance(part, Mapping) and part.get("axis") == "what"
        ),
        None,
    )
    measure = (
        (heading if isinstance(heading, str) and heading.strip() else None)
        or resolved.get("concept_label")
        or resolved.get("measure_label")
        or (resolved.get("shape") or {}).get("label")
    )
    measure = str(measure) if measure else "The data"
    subject = measure[:1].upper() + measure[1:]
    tail = f", {period}" if period else ""
    room = _TITLE_CHARS - len(tail)
    places = _places_words(card, block, room=room - len(subject) - 2)
    head = f"{subject}, {places}" if places else _cut_words(subject, room) or subject[:room]
    return head + tail


def _fitting(labels: list[str], room: int) -> str | None:
    """`labels` as a reader lists them in at most `room` characters: every one
    where they fit, else the first ones and "and N others". `None` when not
    even one fits."""
    whole = joined_words(labels)
    if len(whole) <= room:
        return whole
    for kept in range(len(labels) - 1, 0, -1):
        rest = len(labels) - kept
        words = joined_words([*labels[:kept], f"{rest} other" + ("s" if rest > 1 else "")])
        if len(words) <= room:
            return words
    return None


def _places_words(
    card: UnitCard, block: Mapping[str, Any], *, room: int = _TITLE_CHARS
) -> str | None:
    """The places `card` shows as a title states them in at most `room`
    characters: "France to Kenya" for a flow from one party to another, else
    every role's places in a list, each list cut to what fits (`_fitting`);
    where that does not fit, their count ("12 places"). `None` where even the
    count does not fit, or where a place has no name, since a title never
    shows a place by its ref."""
    labels = {**place_labels(block), **(card.labels or {})}
    shown = {role: refs for role, refs in drawn_places(card, block).items() if refs}
    if any(not labels.get(ref) for refs in shown.values() for ref in refs):
        return None
    named = {role: [labels[ref] for ref in refs] for role, refs in shown.items()}
    words: str | None
    whole_flow = (
        f"{joined_words(named.get('origin', []))} to {joined_words(named.get('destination', []))}"
    )
    if set(named) == {"origin", "destination"} and len(whole_flow) <= room:
        words = whole_flow
    elif set(named) == {"origin", "destination"}:
        origin = _fitting(named["origin"], max((room - 4) // 2, 0))
        destination = (
            _fitting(named["destination"], room - 4 - len(origin)) if origin is not None else None
        )
        words = f"{origin} to {destination}" if origin and destination else None
    else:
        words = _fitting([label for labels_ in named.values() for label in labels_], room)
    if words is not None and len(words) <= room:
        return words
    count = sum(len(labels_) for labels_ in named.values())
    counted = f"{count} place" + ("s" if count != 1 else "")
    return counted if count and len(counted) <= room else None


def _shown_labels(move: ChangeMove) -> set[str]:
    """The labels of every place the card showed before `move`."""
    return {o.label for p in move.offer.places or () for o in p.chosen} | set(move.old)


def _changed_meta(
    name: str, text: str, *, shown: set[str], unit: str | None = None, summed: bool = False
) -> str | None:
    """A word of the card's meta after a change of years or places: the
    subtitle and the description, which the model wrote about the old years
    and places, are dropped, and an axis label is kept only when it names no
    year outside the unit's words (`unit`, `_unit_said`) and none of the
    places in `shown`, the ones the card showed. On a card on a sum
    (`summed`) an axis label naming any number outside the unit's words is
    dropped too, since it can count the members the old sum added up."""
    if name in {"subtitle", "description"}:
        return None
    bare = re.sub(unit, " ", text, flags=re.IGNORECASE) if unit is not None else text
    if summed and re.search(r"\d", bare):
        return None
    if _names_a_year(text, unit) or any(
        re.search(rf"(?<!\w){re.escape(label)}(?!\w)", text) for label in shown if label
    ):
        return None
    return text


async def _change_flow(  # noqa: PLR0911 -- one ending per step that can refuse the change
    *,
    recorder: _RecordingExecutor,
    surface: ToolSurface | None,
    conversation: ConversationState,
    move: UnitMove | ChangeMove,
    places: bool,
    provenance: ProvenanceStore | None,
) -> AsyncGenerator[EditEvent]:
    """`change_events`' steps, their frames numbered by `edit_event_frames`."""
    if surface is None:
        for event in _close("Changing a card in place is not enabled on this deployment."):
            yield event
        return
    described = await recorder(tool="describe_component", args={"component_id": move.component_id})
    if described.get("outcome") != "ok":
        for event in _refused("u1", "Read the card", _UNIT_UNCHANGED, described):
            yield event
        return
    stored = (described.get("component_body") or {}).get("stored_intent") or {}
    yield _step("u1", "Read the card")

    read = await _read_change(surface, move)
    if isinstance(read, UnitRefused):
        words = unit_words(move.unit, move.offer) if isinstance(move, UnitMove) else ""
        label = f"Read the data in {words[:1].lower()}{words[1:]}"
        yield _step("u2", label, failed=read.reason)
        for event in _close(f"{_UNIT_UNCHANGED} {read.reason}"):
            yield event
        return
    label = f"Read the data {read.phrase}"
    # A unit change binds the card's other facets on the spec Search answered
    # (`resubmit_unit`); a change of years or places binds them all on the
    # fetch, so the data API elects none of them again.
    card = _changed_card(move, read)
    # The years held over the new places depend on the spec and its scope
    # alone, so Search reads them while the data is fetched. A change refused
    # meanwhile drops them with the card.
    years = asyncio.create_task(_with_years(surface, conversation, card, read))
    years.add_done_callback(lambda task: task.cancelled() or task.exception())
    try:
        if card.total is not None:
            # A card on a sum keeps the facets its retrieval was counted in,
            # read from that retrieval's frame, since the sum's own series may
            # not name them. The frame is described by the pair it is served
            # under, which the store's pair need not be.
            root = {"spec_id": card.total.served[0], "content_version": card.total.served[1]}
            viewed = await recorder(tool="describe_frame", args={"frame_ref": root})
            view = viewed.get("view") if viewed.get("outcome") == "ok" else None
            kept = root_facets(card, view) if isinstance(view, Mapping) else None
            if kept is None:
                refused = _SUM_UNREAD if not isinstance(view, Mapping) else _OTHER_BASIS
                yield _step("u2", label, failed=refused)
                for event in _close(f"{_UNIT_UNCHANGED} {refused}"):
                    yield event
                return
            card = replace(card, facets=kept or None)
        facets = (
            bound_facets(
                card, stored_block(conversation.turn_specs, conversation.turns, card.spec_id)
            )
            if isinstance(move, ChangeMove)
            else None
        )
        fetched = await recorder(
            tool="get_data", args=card.fetch_args(read.spec_id, read.scope, facets=facets)
        )
        frame_refs = fetched.get("frame_refs") or []
        if fetched.get("outcome") != "ok" or not frame_refs:
            for event in _refused("u2", label, _UNIT_UNCHANGED, fetched):
                yield event
            return
        if fetched.get("rows") == 0:
            yield _step("u2", label, failed=_NO_DATA)
            for event in _close(f"{_UNIT_UNCHANGED} {_NO_DATA}"):
                yield event
            return
        frame_ref = frame_refs[0]
        # The spec each retrieval is served under, whose family ref a summed
        # retrieval's series carry (`_redrawn_intent`).
        family = (
            move.card.total.served[0]
            if move.card.total is not None
            else (stored.get("frame_ref") or {}).get("spec_id"),
            frame_ref.get("spec_id"),
        )
        if card.total is not None:
            # The new retrieval is summed only when its series share the unit,
            # source and method the card's sum was drawn with, since `aggregate`
            # would add two methods' figures, or label a sum of another unit as the
            # card's.
            viewed = await recorder(tool="describe_frame", args={"frame_ref": frame_ref})
            view = viewed.get("view") if viewed.get("outcome") == "ok" else None
            unlike = unlike_series(card, view) if isinstance(view, Mapping) else "unread"
            summed = (
                {"mixed": _MIXED_BASIS, "other": _OTHER_BASIS, "unread": _SUM_UNREAD}[unlike]
                if unlike is not None
                else await _summed_again(surface, provenance, card.total, fetched)
            )
            if isinstance(summed, str):
                yield _step("u2", label, failed=summed)
                for event in _close(f"{_UNIT_UNCHANGED} {summed}"):
                    yield event
                return
            frame_ref, total = summed
            card = replace(card, total=total)
        card = replace(card, years=(await years).years)
    finally:
        years.cancel()
    yield _step("u2", label)

    # The card left the canvas while its data was read. The change is
    # refused as the route refuses a card off the canvas, and nothing is
    # written.
    gone = (_step("u3", "Redraw the card", failed=UNIT_GONE_NOTE), EditClosed(UNIT_GONE_NOTE))
    if _block_index(conversation.composition, move.component_id) is None:
        for event in gone:
            yield event
        return

    intent, retitle = _redrawn_intent(stored, frame_ref, move, family=family)
    written = await recorder(tool="write_component", args={"intent": intent})
    if written.get("outcome") != "ok" or not written.get("component"):
        for event in _refused("u3", "Redraw the card", _UNIT_UNCHANGED, written):
            yield event
        return
    if strayed_facets(card, written["component"].get("fact_summary")):
        yield _step("u3", "Redraw the card", failed=_OTHER_BASIS)
        for event in _close(f"{_UNIT_UNCHANGED} {_OTHER_BASIS}"):
            yield event
        return
    shown = read.phrase.removeprefix("in ").removeprefix("for ")
    # A change of years or places names the scope the card now shows, so the
    # next turn's model fetches what the card draws.
    summed_with = (
        ""
        if card.total is None
        else f", then aggregate {json.dumps(card.total.parameters, sort_keys=True)}"
    )
    record_line = (
        f"Changed {move.title} to {shown} ({read.spec_id})."
        if read.scope is None
        else f"Changed {move.title} to {shown} "
        "(get_data "
        f"{json.dumps(card.fetch_args(read.spec_id, read.scope, facets=facets), sort_keys=True)}"
        f"{summed_with})."
    )
    display_line = f"Change {move.title} to {shown}."

    def record(work: ConversationState) -> None:
        work.turn_user_message = record_line
        work.turn_readings = [*work.turn_readings, *read.readings]

    # The card shows the change. The record keeps the sentence and the
    # spec_id for the next turn's model, and a state read shows neither.
    closed = EditClosed(f"The card now shows {shown}.", shown=False, user_display=display_line)
    yield _swapped(
        recorder,
        conversation,
        component=written["component"],
        old_component_id=move.component_id,
        step_id="u3",
        step_label="Redraw the card",
        closed=closed,
        specs=read.specs or None,
        also=record,
        rewrite=retitle,
        unit=isinstance(move, UnitMove),
        places=places,
        # The card left the canvas while it was written.
        gone=gone,
        card=card,
        intent=intent,
    )


async def _summed_again(  # noqa: PLR0911 -- one sentence per way the sum fails
    surface: ToolSurface,
    provenance: ProvenanceStore | None,
    total: SummedFrame,
    fetched: Mapping[str, Any],
) -> tuple[dict[str, str], SummedFrame] | str:
    """The sum `total` over the retrieval `fetched` gave: analysis-mcp's
    `aggregate` called with the parameters `total` recorded and the
    retrieval's frame, as the store holds it (`store_refs`), in place of the
    one it read. Its frame comes back with the sum it is, and its lineage is
    written into `provenance` as the new retrieval's, with the count of
    StatVars the old sum's lineage holds. Otherwise the sentence the change
    ends on: no figures for a sum with no rows, else `_SUM_UNREAD`, or the
    sentence for the code `aggregate` refused with."""
    refs = fetched.get("store_refs") or fetched.get("frame_refs") or []
    root = refs[0] if refs and isinstance(refs[0], Mapping) else {}
    root_pair = (root.get("spec_id"), root.get("content_version"))
    shown = (fetched.get("frame_refs") or [{}])[0]
    served = (shown.get("spec_id"), shown.get("content_version"))
    if not all(isinstance(part, str) and part for part in (*root_pair, *served)):
        return _SUM_UNREAD
    try:
        answer = await surface.call_tool(
            tool=_AGGREGATE, arguments={**total.parameters, "frame_ref": dict(root)}
        )
    except ToolSurfaceUnavailableError as exc:
        logger.warning("a card's sum was not worked out again: %s", exc)
        return _SUM_UNREAD
    result = answer.structured_content or {}
    if answer.is_error:
        logger.info("a card's sum was refused (%s): %s", result.get("code"), result.get("message"))
        return failure_detail(result)
    spec_id, content_version = result.get("spec_id"), result.get("content_version")
    if not isinstance(spec_id, str) or not isinstance(content_version, str):
        return _SUM_UNREAD
    if result.get("rows") == 0:
        return _NO_DATA
    if provenance is not None:
        conversation = current_conversation_id.get()
        old_spec_id, old_content_version = total.frame
        linked = provenance.alias(
            conversation=conversation,
            spec_id=spec_id,
            content_version=content_version,
            sources=[(str(root_pair[0]), str(root_pair[1]))],
            stat_var_count=provenance.stat_var_count(
                conversation=conversation,
                spec_id=old_spec_id,
                content_version=old_content_version,
            ),
        )
        if not linked:
            return _SUM_UNREAD
    summed = SummedFrame(
        frame=(spec_id, content_version),
        root=(str(root_pair[0]), str(root_pair[1])),
        served=(str(served[0]), str(served[1])),
        parameters=dict(total.parameters),
    )
    return {"spec_id": spec_id, "content_version": content_version}, summed


def _changed_card(move: UnitMove | ChangeMove, read: _Read) -> UnitCard:
    """The card `move` redraws, reading `read`'s spec over its scope. A unit
    change drops the unit the card was drawn in from its bound facets, which
    the new card's own drawn facets replace (`encoded`)."""
    facets = move.card.facets
    if isinstance(move, UnitMove) and facets:
        facets = {axis: ref for axis, ref in facets.items() if axis != "unit"} or None
    return replace(move.card, spec_id=read.spec_id, scope=read.scope, facets=facets)


async def _with_years(
    surface: ToolSurface, conversation: ConversationState, card: UnitCard, read: _Read
) -> UnitCard:
    """`card` with the years its spec holds over the places its scope names,
    read from Search (`harness.unit_switch.scope_years`) when its scope names
    other places than the spec's block and the card holds none for them: one
    Search read, made only after a change that leaves the card on other
    places than its block's. A read that fails leaves the card with no years,
    and it offers none (`harness.card_unit.card_changes`)."""
    block = read.specs.get(read.spec_id) or stored_block(
        conversation.turn_specs, conversation.turns, read.spec_id
    )
    if block is None:
        return card
    entities = scope_needs_years(card, block)
    if entities is None:
        return replace(card, years=None)
    held = card.years
    if (
        held is not None
        and held.spec_id == card.spec_id
        and {r: sorted(v) for r, v in held.entities.items()}
        == {r: sorted(v) for r, v in entities.items()}
    ):
        return card
    years = await scope_years(surface, block=block, entities=entities)
    if years is None:
        return replace(card, years=None)
    first, last, gaps = years
    return replace(
        card,
        years=ScopeYears(
            spec_id=card.spec_id, entities=entities, first=first, last=last, gaps=gaps
        ),
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
    places: bool,
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
            title = _block_title(component) if caught_up else patch.get("title")
            lineage = recorder.card_lineage(component)
            runs = card_runs(component, lineage)
            offered = _revised_card(
                c, component_id, component, lineage, _merge_patch(stored, patch), places=places
            )
            revised = revised.model_copy(update={"card_changes": offered.changes})
            frames: list[EditEvent] = [revised]

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
                entry = work.components[component_id]
                if offered.card is None:
                    entry.pop(CARD_UNIT_KEY, None)
                else:
                    entry[CARD_UNIT_KEY] = offered.card
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
        places=places,
        reframed=_reframes(stored, intent),
        # The rebuild draws the old card's frame, so it reads what the old
        # card read: its scope, the labels a change set, and its places.
        card=UnitCard.from_json((c.components.get(component_id) or {}).get(CARD_UNIT_KEY)),
        intent=intent,
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
    rewrite: Callable[[str, str], str | None] | None = None,
    unit: bool = False,
    places: bool = False,
    gone: tuple[ActivityStep, EditClosed] | None = None,
    reframed: bool = False,
    card: UnitCard | None = None,
    intent: Mapping[str, Any] | None = None,
) -> CardSwap:
    """The swap of `component`, a card written to take card
    `old_component_id`'s slot, ending the edit on `closed`: the step
    `step_id`, labelled `step_label`, that produced it, its
    `component.created` naming that step, and the write that swaps it in,
    with its digest entry, turn line and write token. The entry of a card
    that took the slot names the old card (`REPLACED_CARD_KEY`), as its
    `component.created` does in `replaces`. The new card names the plan step
    the old card's entry names (`PLAN_LINK_KEY`), in its `created_from` and
    its entry, since it draws that step's result in a new form. A card that
    can change carries its `unit_change` and `card_changes` and keeps its
    `UnitCard` with its entry (`harness.card_unit`), its Search block read
    from `specs` and else from the conversation, places among its offers
    where `places` says Search's places door is on. `card` is the `UnitCard` a
    change fetched the new card as, or the old card's for an Edit Card
    rebuild, which draws the same frame. Without it the card is read off its
    frame (`unit_card`). Its encoding follows `intent`, the new card's
    written intent (`harness.card_unit.encoded`). A card that a change
    fetched over a scope names its `get_data` arguments in its entry
    (`FETCHED_KEY`), which the next turn's model reads. `also` makes the
    write's other changes, and `specs` joins the turn's blocks once it
    commits. `rewrite` rewrites the words of the slot's card meta for a
    change of the card in place, and `unit` marks a unit change, which drops
    the card from a `comparison` hint (`_replace_block`). Given `gone`, the
    swap needs the old card's slot: a canvas that no longer holds it, read
    under the write's lock, takes only the new card's write token, so the
    conversation keeps the card it wrote off the canvas, and the edit ends on
    `gone`. A `closed` that is shown carries the held line (`_held_line`),
    `reframed` saying whether the new card draws another window or frame than
    the old."""
    plan = (conversation.components.get(old_component_id) or {}).get(PLAN_LINK_KEY)
    created = component_created(component, step_id=step_id, plan=plan)
    token = recorder.write_token(component)
    lineage = recorder.card_lineage(component)
    blocks = ConversationBlocks(conversation.turn_specs, conversation.turns, specs)
    # A card on a sum reads its sum off the card it was fetched or rebuilt
    # as, which names the frame the sum gave and the retrieval it read: the
    # lineage of the written card must name both.
    drawn = unit_card(component, lineage, blocks, total=card.total if card is not None else None)
    card = card if card is not None and drawn is not None else drawn
    if card is not None:
        block = blocks.get(card.spec_id)
        card = encoded(card, block, intent=intent, component=component)
        created.unit_change = card_unit_change(card, block)
        created.card_changes = card_changes(
            card, block, fact_summary=component.get("fact_summary"), places=places
        )
    # The facets a later fetch of the card binds, which the next turn's model
    # reads with the rest of its `get_data` arguments.
    fetched_facets = bound_facets(card, blocks.get(card.spec_id)) if card is not None else None
    changes = created.unit_change is not None or created.card_changes is not None
    kept = card.as_json() if card is not None and changes else None
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
        if card is not None and card.scope:
            entry[FETCHED_KEY] = card.fetch_args(card.spec_id, card.scope, facets=fetched_facets)
        if card is not None and card.total is not None:
            entry[SUMMED_KEY] = {"tool": "aggregate", "arguments": dict(card.total.parameters)}
        if plan is not None:
            entry[PLAN_LINK_KEY] = plan
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
            work,
            old_component_id=old_component_id,
            component=component,
            rewrite=rewrite,
            unit=unit,
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


@dataclass(frozen=True)
class _RevisedOffers:
    """The offers of a card revised in place, worked out again from the
    revision: `card` is the `UnitCard` to store for it, `None` when it offers
    no change, and `changes` the `card_changes` it offers."""

    card: dict[str, Any] | None
    changes: CardChanges | None


def _revised_card(
    conversation: ConversationState,
    component_id: str,
    component: dict[str, Any],
    lineage: CardLineage | None,
    intent: Mapping[str, Any],
    *,
    places: bool,
) -> _RevisedOffers:
    """The offers of `component_id` after a revise in place drew `component`
    from `intent`, read as `_swapped` reads a new card: the card the old one
    was stored as, encoded again from the revised card, so its offers name
    the places and years the revision draws. A revised card that cannot be
    read as one that offers changes offers none, since the offers the old
    card held describe a card the canvas no longer shows."""
    held = UnitCard.from_json((conversation.components.get(component_id) or {}).get(CARD_UNIT_KEY))
    blocks = ConversationBlocks(conversation.turn_specs, conversation.turns)
    drawn = unit_card(component, lineage, blocks, total=held.total if held is not None else None)
    if drawn is None:
        return _RevisedOffers(card=None, changes=None)
    card = held if held is not None else drawn
    block = blocks.get(card.spec_id)
    card = encoded(card, block, intent=intent, component=component)
    changes = card_changes(card, block, fact_summary=component.get("fact_summary"), places=places)
    if changes is None and card_unit_change(card, block) is None:
        return _RevisedOffers(card=None, changes=None)
    return _RevisedOffers(card=card.as_json(), changes=changes)


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
