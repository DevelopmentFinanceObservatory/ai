"""Whether a card's unit, years or places can change, and the pieces that change it.

A card can change when it draws the frame of one `get_data` retrieval: its
frame is that retrieval's own, not a frame an analysis step derived from it.
The card's `provenance.inputs[0]` names the spec the retrieval served. A
retrieval that named a `scope` is served under a spec of its own, and the
card reads the spec Search answered (its Search block's) with that scope
(`UnitCard.scope`), read back from the served definition (`unit_card`).

The unit can change when Search converts the spec (`resolved.unit_conversion`,
read by `harness.unit_offer.unit_offer`). The years can change when the card
draws more than one and the spec holds data over more than one
(`card_changes`). The places of a role can change when the role binds places;
the places it can change to are read from Search only when asked
(`harness.place_options`).

A unit change resubmits the spec in the new unit
(`harness.unit_switch.resubmit_unit`); a change of years or places names the
new `scope`. Either fetches the spec with the arguments that shaped the card's
frame and its scope, and redraws the card on the new frame in its slot
(`harness.studio_edit.change_events`).
Those arguments are read off the retrieval's manifest, whose data artifact
names the rendering apis minted (`frame_arguments`). A rendering this module
cannot map back to `get_data`'s arguments leaves the card without a change.

`UnitCard` is what the conversation keeps with such a card, under
`CARD_UNIT_KEY` on its stored entry: the spec_id it reads, those arguments and
its scope. Each change carries the scope on, so a change of one axis keeps
what an earlier change set on another. No model reads it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from common.materialisation.frame_name import KIND_COLUMNS, WITHOUT_PARTY_TOTALS
from common.materialisation.rendering import DEFAULT_MINT_OPTIONS
from data_mcp.frame_materialisation import complete_rendering

from harness.place_options import (
    PlaceOptionsReader,
    PlacesNotServedError,
    PlacesUnavailableError,
    RolePlaces,
    read_role_places,
)
from harness.provenance import CardLineage, data_artifact
from harness.turn_records import SearchReading
from harness.unit_offer import offered_unit_refusal, served_unit, unit_offer, unit_words
from harness.wire import (
    ApplyChange,
    ApplyUnit,
    CardChanges,
    PlaceOption,
    PlacesChange,
    PlacesOffer,
    UnitChange,
    UnitChoice,
    UnitOffer,
    UnitPrices,
    WindowChange,
    WindowOffer,
)

if TYPE_CHECKING:
    from harness.studio_state import ConversationState

# The key of a stored component entry holding its `UnitCard` as JSON. No
# digest, listing or recall states it (`harness.studio_state._model_entry`).
CARD_UNIT_KEY = "unit_card"

# The `options.groupby` `get_data` sends for `stat_vars: "sum"`
# (`data_mcp.get_data_tool._SUM_STAT_VARS_GROUPBY`).
_SUM_GROUPBY = {"by": ["time_period"], "op": "sum"}


@dataclass(frozen=True)
class UnitCard:
    """A card that can change: the spec_id its Search block answers, the
    `get_data` arguments, `spec_id` and `scope` aside, that shaped its frame,
    and the `scope` its retrieval named over that spec (`get_data`'s own
    shape, `entities` naming each place `dcid:`-marked), `None` for none,
    with the label of each place a change of places set (`labels`, by ref),
    which the spec's Search block does not name.

    `single`, `roles` and `cut` are the card's encoding, fixed when the card
    is drawn (`encoded`): whether it shows one value per series, so its years
    change one at a time; the roles whose places it can change, each with
    whether it takes several; and whether its members are the largest ones
    `make_charts` ranked at one period, so its years cannot change, since
    another year has other largest members. A card stored before they were
    kept has `roles` `None`, and reads its roles off the places it shows,
    and `cut` false."""

    spec_id: str
    args: dict[str, Any]
    scope: dict[str, Any] | None = None
    labels: dict[str, str] | None = None
    single: bool = False
    roles: dict[str, bool] | None = None
    cut: bool = False

    def as_json(self) -> dict[str, Any]:
        stored: dict[str, Any] = {"spec_id": self.spec_id, "args": dict(self.args)}
        if self.scope:
            stored["scope"] = dict(self.scope)
        if self.labels:
            stored["labels"] = dict(self.labels)
        if self.single:
            stored["single"] = True
        if self.roles is not None:
            stored["roles"] = dict(self.roles)
        if self.cut:
            stored["cut"] = True
        return stored

    @classmethod
    def from_json(cls, stored: Any) -> UnitCard | None:
        """The `UnitCard` `as_json` wrote, or `None` for anything else."""
        if not isinstance(stored, Mapping):
            return None
        spec_id, args = stored.get("spec_id"), stored.get("args")
        if not isinstance(spec_id, str) or not spec_id or not isinstance(args, Mapping):
            return None
        scope, labels, roles = stored.get("scope"), stored.get("labels"), stored.get("roles")
        return cls(
            spec_id=spec_id,
            args=dict(args),
            scope=dict(scope) if isinstance(scope, Mapping) and scope else None,
            labels={str(k): str(v) for k, v in labels.items()}
            if isinstance(labels, Mapping) and labels
            else None,
            single=stored.get("single") is True,
            cut=stored.get("cut") is True,
            roles={str(k): v is True for k, v in roles.items()}
            if isinstance(roles, Mapping)
            else None,
        )

    def fetch_args(self, spec_id: str, scope: Mapping[str, Any] | None) -> dict[str, Any]:
        """The `get_data` arguments that fetch `spec_id` as this card's frame
        was fetched, over `scope`."""
        return {"spec_id": spec_id, **self.args, **({"scope": dict(scope)} if scope else {})}


def one_value(intent: Mapping[str, Any] | None) -> bool:
    """Whether a card drawn from `intent` shows one value per series: a stat
    card, or a bar chart whose select lists one of its dimensions, which
    `make_charts` draws at one period (`orchestrator_tools._card_period`)."""
    if not isinstance(intent, Mapping):
        return False
    select = intent.get("select")
    return intent.get("type") == "stat_card" or (
        intent.get("type") == "bar_chart"
        and isinstance(select, Mapping)
        and any(isinstance(value, list) for value in select.values())
    )


def _naming(select: Mapping[str, Any], refs: Sequence[str]) -> set[str]:
    """The `select` entries naming any of `refs`, `dcid:`-marked or not."""
    shown = set(refs)
    return {
        name
        for name, value in select.items()
        if any(
            isinstance(v, str) and v.removeprefix("dcid:") in shown
            for v in (value if isinstance(value, list) else [value])
        )
    }


def takes_several(
    intent: Mapping[str, Any] | None,
    refs: Sequence[str],
    others: Sequence[Sequence[str]] = (),
) -> bool:
    """Whether a role showing places `refs` on the card drawn from `intent`
    can show several, each drawn as its own series, as dfo-apis' assembler
    draws a select (dfo-data-tools' `assembler/resolve.py` and
    `pipeline/candidate.py`): a select entry naming a role's places as a list
    draws one series per place, and so does a place column the select leaves
    unbound. A stat card draws one value; a table draws any select. A line
    chart takes several unless its select lists another dimension, since the
    platform refuses a line or bar chart listing two, or another place role
    (`others`, the places each shows) already draws several series, being
    unbound or listed over several places. A bar chart takes several only
    where its listed dimension is the role's, one bar per place: a bar chart
    of one series draws its periods, and a second place would turn it into
    one bar per place at one period."""
    if not isinstance(intent, Mapping):
        return True
    kind = intent.get("type")
    if kind == "stat_card":
        return False
    if kind == "table":
        return True
    select = intent.get("select")
    select = select if isinstance(select, Mapping) else {}
    naming = _naming(select, refs)
    listed = {name for name, value in select.items() if isinstance(value, list)}
    for other in others:
        named = _naming(select, other)
        if len(other) > 1 and (not named or named & listed):
            return False
    if kind == "bar_chart":
        return bool(listed) and listed <= naming
    return listed <= naming


def encoded(
    card: UnitCard,
    block: Mapping[str, Any] | None,
    *,
    intent: Mapping[str, Any] | None,
    cut: bool | None = None,
) -> UnitCard:
    """`card` with its encoding as the card drawn from `intent` fixes it:
    `single` when it shows one value per series (`one_value`), and as `roles`
    each role whose places it shows, the ones `card` already kept, else
    those `block` and its scope bind, each marked with whether it can show
    several places (`takes_several`). `cut` says whether `make_charts` cut
    the card to its largest members; `None` keeps the card's own."""
    shown = chosen_places(card, block) if block is not None else {}
    if card.roles is not None:
        bound: list[str] = list(card.roles)
    else:
        bound = [role for role, refs in shown.items() if refs]
    roles = {
        role: takes_several(
            intent,
            shown.get(role) or (),
            [refs for other, refs in shown.items() if other != role and refs],
        )
        for role in bound
    }
    return replace(
        card, single=one_value(intent), roles=roles, cut=card.cut if cut is None else cut
    )


def frame_arguments(rendering: Mapping[str, Any]) -> dict[str, Any] | None:
    """The `get_data` arguments, `spec_id` and `scope` aside, that mint
    `rendering`, a retrieval's data artifact rendering; `None` for one no call
    of `get_data` sends.

    `get_data` sends the default options (`DEFAULT_MINT_OPTIONS`), a groupby
    for `stat_vars: "sum"` or for a `breakdown`, the kind columns for
    `without_party_totals` and that option itself, and an enrichment for
    `place_types`, which the rendering names by the columns apis resolved it
    to: any column beyond the kind columns. A rendering apis linked after a
    failed metadata read (`metadata: "incomplete"`) maps as its complete one."""
    rest = complete_rendering(rendering)
    for key, value in DEFAULT_MINT_OPTIONS.items():
        if rest.pop(key, None) != value:
            return None
    args: dict[str, Any] = {}
    groupby = rest.pop("groupby", None)
    if groupby == _SUM_GROUPBY:
        args["stat_vars"] = "sum"
    elif groupby is not None:
        breakdown = _breakdown(groupby)
        if breakdown is None:
            return None
        args["breakdown"] = breakdown
    without_totals = rest.pop(WITHOUT_PARTY_TOTALS, False)
    if without_totals not in (True, False):
        return None
    if without_totals:
        args["without_party_totals"] = True
    columns = rest.pop("columns", None)
    if columns is not None:
        if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
            return None
        if not set(columns) <= set(KIND_COLUMNS):
            args["place_types"] = True
        elif not without_totals:
            return None
    return None if rest else args


def _breakdown(groupby: Any) -> dict[str, Any] | None:
    """The `breakdown` argument whose `options.groupby` is `groupby`
    (`data_mcp.get_data_tool.Breakdown.groupby`), or `None`."""
    if not isinstance(groupby, Mapping) or set(groupby) != {"by", "op", "levels"}:
        return None
    levels = groupby["levels"]
    if groupby["op"] != "sum" or not isinstance(levels, Mapping) or len(levels) != 1:
        return None
    ((code_system, level),) = levels.items()
    if not isinstance(level, int) or groupby["by"] != sorted(["time_period", code_system]):
        return None
    return {"code_system": code_system, "level": level}


def unit_card(  # noqa: PLR0911 -- one `None` per way a card fails to draw one retrieval
    component: Mapping[str, Any],
    lineage: CardLineage | None,
    blocks: Mapping[str, Mapping[str, Any]] | None = None,
) -> UnitCard | None:
    """The `UnitCard` of `component`, a written card, drawn from `lineage`;
    `None` when the card does not draw one retrieval's own frame, or its
    arguments cannot be read.

    The card reads the served spec when `blocks`, the Search blocks the
    conversation keeps by spec_id (`known_blocks`), hold none for it but
    hold one whose identity the served definition's matches outside its
    places and window: the card reads that block's spec with the places and
    window the served definition names as its scope (`_scope_over`), the
    places as Search resolved them, under the block's own roles. The served
    identity is the one source, since the model's `get_data` arguments name
    places as spoken, under its own role names, and may name facets the
    scope cannot carry; a served identity that differs from every block's in
    anything else, its facets among them, offers no change."""
    if lineage is None or len(lineage.lineage) != 1:
        return None
    (member,) = lineage.lineage
    served = member.served_ref
    if lineage.frame not in {
        (member.spec_id, member.content_version),
        (served["spec_id"], served["content_version"]),
    }:
        return None
    inputs = (component.get("provenance") or {}).get("inputs") or []
    spec_id = served["spec_id"]
    if len(inputs) != 1 or not spec_id or inputs[0].get("spec_id") != spec_id:
        return None
    rendering = data_artifact(member.manifest.get("artifacts") or []).get("rendering")
    args = frame_arguments(rendering) if isinstance(rendering, Mapping) else None
    if args is None:
        return None
    if blocks is None or spec_id in blocks:
        return UnitCard(spec_id=spec_id, args=args)
    served_identity = (member.manifest.get("definition") or {}).get("identity")
    for base_id, block in blocks.items():
        scope = _scope_over((block.get("definition") or {}).get("identity"), served_identity)
        if scope:
            return UnitCard(spec_id=base_id, args=args, scope=scope)
    return UnitCard(spec_id=spec_id, args=args)


def _scope_over(base: Any, served: Any) -> dict[str, Any] | None:
    """The `get_data` scope that serves identity `served` over identity
    `base`: their places (`entities`) and window (`requested_window`) where
    they differ, `None` where the two differ elsewhere or either is unread.
    Slots compare in any order."""
    if not isinstance(base, Mapping) or not isinstance(served, Mapping):
        return None

    def rest(identity: Mapping[str, Any]) -> dict[str, Any]:
        kept = {k: v for k, v in identity.items() if k not in {"entities", "requested_window"}}
        slots = kept.get("slots")
        if isinstance(slots, list):
            kept["slots"] = sorted(
                slots,
                key=lambda slot: (
                    str(slot.get("property")) if isinstance(slot, Mapping) else str(slot)
                ),
            )
        return kept

    if rest(base) != rest(served):
        return None
    scope: dict[str, Any] = {}
    entities = served.get("entities")
    if entities != base.get("entities") and isinstance(entities, Mapping):
        scope["entities"] = {
            str(role): [f"dcid:{ref}" for ref in refs] for role, refs in entities.items()
        }
    window = served.get("requested_window")
    if window != base.get("requested_window") and isinstance(window, Mapping):
        scope["window"] = dict(window)
    return scope


def known_blocks(
    turn_specs: Mapping[str, Any], turns: list[dict[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    """Every Search block the conversation keeps, by spec_id: the open turn's,
    else the newest an earlier turn's record kept, as `stored_block` reads one."""
    blocks: dict[str, Mapping[str, Any]] = {}
    for turn in turns:
        blocks.update(
            {k: v for k, v in (turn.get("specs") or {}).items() if isinstance(v, Mapping)}
        )
    blocks.update({k: v for k, v in turn_specs.items() if isinstance(v, Mapping)})
    return blocks


def card_unit_change(card: UnitCard, block: Mapping[str, Any] | None) -> UnitChange | None:
    """The `unit_change` of a card reading `card`'s spec, whose Search block is
    `block`: the units the block converts into, starting on the one it is
    served in; `None` when it converts into none or its served unit lies
    outside them."""
    if block is None:
        return None
    offer = unit_offer([block])
    served = served_unit(block)
    if offer is None or served is None or offered_unit_refusal(offer, served) is not None:
        return None
    return UnitChange(spec_id=card.spec_id, offer=offer.model_copy(update={"start": served}))


def stored_unit_change(
    entry: Mapping[str, Any], turn_specs: Mapping[str, Any], turns: list[dict[str, Any]]
) -> UnitChange | None:
    """The `unit_change` of the card whose stored entry is `entry`: its
    `UnitCard` with the Search block a turn kept for its spec, read as
    `unit_move` reads them. `None` for a card stored with no `UnitCard`, or
    whose spec's block is gone or converts into none."""
    card = UnitCard.from_json(entry.get(CARD_UNIT_KEY))
    if card is None:
        return None
    return card_unit_change(card, stored_block(turn_specs, turns, card.spec_id))


def card_changes(
    card: UnitCard,
    block: Mapping[str, Any] | None,
    *,
    fact_summary: Mapping[str, Any] | None,
    places: bool = False,
) -> CardChanges | None:
    """The `card_changes` of a card reading `card`'s spec, whose Search block
    is `block`, that draws the years its `fact_summary` states; `None` when it
    offers no change.

    `window` is offered when the card draws one year or a range and its data
    holds more than one: Search's `years_with_data`, else the window its
    answer serves (`download.window`, then `coverage.window`), widened to the
    years the card draws, and never on a card cut to its largest members at
    one period (`UnitCard.cut`). With `places` (Search's places door is on),
    `places` holds each role whose places the card names."""
    if block is None:
        return None
    window = None if card.cut else _window_offer(block, fact_summary, single=card.single)
    roles = _places_offers(card, block) if places else []
    if window is None and not roles:
        return None
    return CardChanges(spec_id=card.spec_id, window=window, places=roles or None)


def _year(value: Any) -> int | None:
    """The calendar year `value`, a fact summary's period, names: its first
    four digits when it opens on them, else `None`."""
    text = str(value) if isinstance(value, int | str) else ""
    return int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else None


def _years_held(block: Mapping[str, Any]) -> tuple[int, int, list[int]] | None:
    """The first and last year `block`'s spec holds data, with the years
    between them it holds none of: `years_with_data` where Search states it,
    else the window its answer serves, with none missing."""
    resolved = block.get("resolved") or {}
    held = resolved.get("years_with_data")
    if isinstance(held, Mapping):
        start, end = held.get("start_year"), held.get("end_year")
        if isinstance(start, int) and isinstance(end, int):
            gaps = [year for year in held.get("gap_years") or () if isinstance(year, int)]
            return start, end, gaps
    for window in (
        (resolved.get("download") or {}).get("window"),
        (resolved.get("coverage") or {}).get("window"),
    ):
        if isinstance(window, Mapping):
            start, end = window.get("start_year"), window.get("end_year")
            if isinstance(start, int) and isinstance(end, int):
                return start, end, []
    return None


def _window_offer(
    block: Mapping[str, Any], fact_summary: Mapping[str, Any] | None, *, single: bool
) -> WindowOffer | None:
    drawn = (fact_summary or {}).get("window") or {}
    start, end = _year(drawn.get("start")), _year(drawn.get("end"))
    held = _years_held(block)
    if start is None or end is None or end < start or held is None:
        return None
    first, last, gaps = min(held[0], start), max(held[1], end), held[2]
    if first == last:
        return None
    missing = [year for year in gaps if first < year < last]
    return WindowOffer(
        start_year=start,
        end_year=end,
        first_year=first,
        last_year=last,
        missing_years=missing or None,
        single=True if single else None,
    )


def _places_offers(card: UnitCard, block: Mapping[str, Any]) -> list[PlacesOffer]:
    """One offer per role of the card's encoding (`UnitCard.roles`), taking
    several places where it says so, whatever number it shows now; a card
    stored with no encoding offers each role it shows places for, taking
    several where it shows several."""
    labels = {**place_labels(block), **(card.labels or {})}
    shown = chosen_places(card, block)
    roles = card.roles if card.roles is not None else {r: len(v) > 1 for r, v in shown.items()}
    # In the order Search lists the block's places, so every redraw lists
    # the roles alike.
    listed = [
        str(entity.get("role"))
        for entity in (block.get("resolved") or {}).get("entities") or ()
        if isinstance(entity, Mapping)
    ]
    roles = dict(
        sorted(
            roles.items(), key=lambda kv: listed.index(kv[0]) if kv[0] in listed else len(listed)
        )
    )
    return [
        PlacesOffer(
            role=role,
            multiple=several,
            chosen=[PlaceOption(id=ref, label=labels.get(ref, ref)) for ref in shown[role]],
        )
        for role, several in roles.items()
        if shown.get(role)
    ]


def chosen_places(card: UnitCard, block: Mapping[str, Any]) -> dict[str, list[str]]:
    """The places each role of `card` shows, by ref: its scope's where the
    scope names places, else its spec's identity's."""
    scoped = (card.scope or {}).get("entities")
    if isinstance(scoped, Mapping):
        return {
            str(role): [str(name).removeprefix("dcid:") for name in names]
            for role, names in scoped.items()
        }
    identity = (block.get("definition") or {}).get("identity") or {}
    entities = identity.get("entities") or {}
    return {str(role): [str(ref) for ref in refs] for role, refs in entities.items()}


def place_labels(block: Mapping[str, Any]) -> dict[str, str]:
    """The label Search gives each place `block` binds, by ref."""
    return {
        str(entity["ref"]): str(entity.get("label") or entity["ref"])
        for entity in (block.get("resolved") or {}).get("entities") or ()
        if isinstance(entity, Mapping) and entity.get("ref")
    }


def stored_card_changes(
    entry: Mapping[str, Any],
    turn_specs: Mapping[str, Any],
    turns: list[dict[str, Any]],
    *,
    places: bool = False,
) -> CardChanges | None:
    """The `card_changes` of the card whose stored entry is `entry`, read as
    `change_move` reads them. `None` for a card stored with no `UnitCard`,
    or whose spec's block is gone or offers no change."""
    card = UnitCard.from_json(entry.get(CARD_UNIT_KEY))
    if card is None:
        return None
    return card_changes(
        card,
        stored_block(turn_specs, turns, card.spec_id),
        fact_summary=entry.get("fact_summary"),
        places=places,
    )


@dataclass(frozen=True)
class UnitMove:
    """A unit change the route admitted: the card (`component_id`, titled
    `title`), what it reads (`card`), the unit it is served in (`served`) and
    the one it moves to (`unit`), within `offer`, with the Search block of its
    spec and the newest reading of that spec, `None` when no reading names it."""

    component_id: str
    title: str
    card: UnitCard
    served: UnitChoice
    unit: UnitChoice
    offer: UnitOffer
    block: dict[str, Any]
    reading: SearchReading | None


# The `detail` of a unit change refused because the card moved on: the client
# shows it in the reply.
UNIT_STALE_NOTE = "This card has changed since its unit menu opened, so nothing changed."
UNIT_GONE_NOTE = "This card is no longer on the canvas, so nothing changed."


def _admitted(
    conversation: ConversationState, component_id: str, spec_id: str | None, *, stale: str
) -> tuple[UnitCard, dict[str, Any], Mapping[str, Any], str] | tuple[str, str]:
    """Card `component_id` of `conversation` as a unit change or a change of
    years or places reads it: what it reads, its spec's Search block, its
    digest entry and its title; or the `scope_choice_stale` refusal's
    constraint and detail, `UNIT_GONE_NOTE` when the card is off the canvas
    and `stale` when it reads a spec other than `spec_id` (any, with `None`)
    or no turn kept its spec's block."""
    composition = conversation.composition
    blocks = composition.blocks if composition is not None else []
    on_canvas = next((b for b in blocks if b.component_id == component_id), None)
    if on_canvas is None:
        return "scope_choice_stale", UNIT_GONE_NOTE
    entry = conversation.components.get(component_id) or {}
    card = UnitCard.from_json(entry.get(CARD_UNIT_KEY))
    if card is None or (spec_id is not None and card.spec_id != spec_id):
        return "scope_choice_stale", stale
    block = stored_block(conversation.turn_specs, conversation.turns, card.spec_id)
    if block is None:
        return "scope_choice_stale", stale
    return card, block, entry, str(entry.get("title") or on_canvas.title or "The card")


def unit_move(conversation: ConversationState, pick: ApplyUnit) -> UnitMove | tuple[str, str]:
    """The change `pick` asks of `conversation`, or its refusal's constraint
    and detail: `scope_choice_stale` when the card is off the canvas, when its
    `provenance.inputs[0]` names a spec other than `pick.spec_id`, or when no
    turn kept that spec's Search block; `unit_not_offered` for a unit the
    card's offer does not hold or the one it already shows. `pick.turn` plays
    no part."""
    held = _admitted(conversation, pick.component_id, pick.spec_id, stale=UNIT_STALE_NOTE)
    if len(held) == 2:
        return held
    card, block, _, title = held
    change = card_unit_change(card, block)
    if change is None:
        return "scope_choice_stale", UNIT_STALE_NOTE
    offer, served = change.offer, change.offer.start
    if (refused := offered_unit_refusal(offer, pick.unit)) is not None:
        return "unit_not_offered", f"That unit is not offered for this card: {refused}."
    if pick.unit == served:
        return "unit_not_offered", f"The card already shows {unit_words(served, offer)}."
    return UnitMove(
        component_id=pick.component_id,
        title=title,
        card=card,
        served=served,
        unit=pick.unit,
        offer=offer,
        block=block,
        reading=stored_reading(conversation.turn_readings, conversation.turns, card.spec_id),
    )


@dataclass(frozen=True)
class ChangeMove:
    """A change of a card's years or places the route admitted: the card
    (`component_id`, titled `title`), what it reads (`card`), the change, the
    `scope` its new retrieval names (the card's own with the change's member
    replaced), its words (`words`: "2010 to 2024", "Uganda"), the offer it
    was checked against (`offer`) and the Search block of its spec. A change
    of places carries the labels of the places it leaves and shows (`old` and
    `new`) and the places Search listed for its role (`listed`)."""

    component_id: str
    title: str
    card: UnitCard
    change: WindowChange | PlacesChange
    scope: dict[str, Any]
    words: str
    offer: CardChanges
    block: dict[str, Any]
    old: tuple[str, ...] = ()
    new: tuple[str, ...] = ()
    listed: RolePlaces | None = None


CHANGE_STALE_NOTE = "This card has changed since its menu opened, so nothing changed."


# The refusal of a change of places on a deployment whose apis serves no
# places door (`harness.place_options.PlacesNotServedError`), which the route
# answers as `/places` does, 503 `capability_unavailable`.
PLACES_NOT_SERVED = "places_not_served"


# The `detail` of a change refused because Search's places door could not be read.
PLACES_UNREAD_NOTE = "The places this card can show cannot be read now; try again."


def changeable(
    conversation: ConversationState, component_id: str, spec_id: str | None, *, places: bool
) -> tuple[UnitCard, dict[str, Any], CardChanges, str] | tuple[str, str]:
    """The card `component_id` of `conversation` as a change reads it: what it
    reads, its spec's Search block, its offer and its title; or the
    `scope_choice_stale` refusal's constraint and detail when the card is off
    the canvas, reads a spec other than `spec_id` (any, with `None`), no turn
    kept its spec's block or it offers no change. `places` says whether
    Search's places door is on."""
    held = _admitted(conversation, component_id, spec_id, stale=CHANGE_STALE_NOTE)
    if len(held) == 2:
        return held
    card, block, entry, title = held
    offer = card_changes(card, block, fact_summary=entry.get("fact_summary"), places=places)
    if offer is None:
        return "scope_choice_stale", CHANGE_STALE_NOTE
    return card, block, offer, title


async def change_move(  # noqa: PLR0911 -- one refusal per way a change cannot apply
    conversation: ConversationState,
    pick: ApplyChange,
    *,
    reader: PlaceOptionsReader | None,
    listed: RolePlaces | None = None,
) -> ChangeMove | tuple[str, str]:
    """The change `pick` asks of `conversation`, or its refusal's constraint
    and detail: `scope_choice_stale` as `changeable` refuses;
    `change_not_offered` for years or places outside the card's offer, or the
    ones it shows; `change_no_data` for years its data holds none of. A change
    of places reads the places its role can change to from Search through
    `reader` (the places door, `None` when off), or takes them from `listed`
    when the route read them already, and is refused
    `place_options_unavailable` when they cannot be read."""
    held = changeable(conversation, pick.component_id, pick.spec_id, places=reader is not None)
    if len(held) == 2:
        return held
    card, block, offer, title = held
    change = pick.change
    if isinstance(change, WindowChange):
        scope, words = _window_move(card, offer, change)
        if isinstance(scope, str):
            return scope, words
        return ChangeMove(pick.component_id, title, card, change, scope, words, offer, block)
    refused = _places_refusal(offer, change)
    if refused is not None:
        return refused
    assert reader is not None, "a places offer is made only with the door on"
    if listed is None:
        try:
            listed = await read_role_places(
                reader,
                identity=(block.get("definition") or {}).get("identity") or {},
                role=change.role,
                conversation_id=conversation.conversation_id,
                component_id=pick.component_id,
            )
        except PlacesNotServedError:
            return PLACES_NOT_SERVED, PLACES_UNREAD_NOTE
        except PlacesUnavailableError:
            return "place_options_unavailable", PLACES_UNREAD_NOTE
    return _places_move(pick.component_id, title, card, block, offer, change, listed)


def _window_move(
    card: UnitCard, offer: CardChanges, change: WindowChange
) -> tuple[dict[str, Any], str] | tuple[str, str]:
    window = offer.window
    if window is None:
        return "change_not_offered", "This card's years cannot change."
    start, end = change.start_year, change.end_year
    if window.single and start != end:
        return "change_not_offered", "This card shows one year at a time."
    if start < window.first_year or end > window.last_year:
        return (
            "change_not_offered",
            f"The data runs from {window.first_year} to {window.last_year}.",
        )
    if (start, end) == (window.start_year, window.end_year):
        return "change_not_offered", f"The card already shows {start} to {end}."
    missing = set(window.missing_years or ())
    if all(year in missing for year in range(start, end + 1)):
        return "change_no_data", f"The data holds no figures from {start} to {end}."
    held = (card.scope or {}).get("window") or {}
    kept = {"include": held["include"]} if isinstance(held, Mapping) and "include" in held else {}
    scope = {
        **(card.scope or {}),
        "window": {"kind": "range", "start_year": start, "end_year": end, **kept},
    }
    return scope, f"{start} to {end}"


def _places_refusal(offer: CardChanges, change: PlacesChange) -> tuple[str, str] | None:
    """Why `change` is refused before Search is read: a role the card does not
    offer, a count its `multiple` does not allow, or the places it shows."""
    places = next((p for p in offer.places or () if p.role == change.role), None)
    if places is None:
        return "change_not_offered", "The places of this card cannot change there."
    picked = list(dict.fromkeys(change.entities))
    if not places.multiple and len(picked) != 1:
        return "change_not_offered", "This card shows one place at a time."
    if set(picked) == {o.id for o in places.chosen}:
        return "change_not_offered", "The card already shows those places."
    return None


def _places_move(
    component_id: str,
    title: str,
    card: UnitCard,
    block: Mapping[str, Any],
    offer: CardChanges,
    change: PlacesChange,
    listed: RolePlaces,
) -> ChangeMove | tuple[str, str]:
    places = next(p for p in offer.places or () if p.role == change.role)
    labels = {**dict(listed.places), **{o.id: o.label for o in places.chosen}}
    picked = list(dict.fromkeys(change.entities))
    if any(ref not in labels for ref in picked):
        return "change_not_offered", "That place is not offered for this card."
    kept = {**(card.labels or {}), **{ref: labels[ref] for ref in picked}}
    entities = {**chosen_places(card, block), change.role: picked}
    scope = {
        **(card.scope or {}),
        "entities": {role: [f"dcid:{ref}" for ref in refs] for role, refs in entities.items()},
    }
    new = tuple(labels[ref] for ref in picked)
    return ChangeMove(
        component_id,
        title,
        replace(card, labels=kept),
        change,
        scope,
        joined_words(list(new)),
        offer,
        dict(block),
        old=tuple(o.label for o in places.chosen),
        new=new,
        listed=listed,
    )


def joined_words(words: list[str]) -> str:
    """`words` as a reader lists them: "A", "A and B", "A, B and C"."""
    if len(words) <= 2:
        return " and ".join(words)
    return f"{', '.join(words[:-1])} and {words[-1]}"


def stored_block(
    turn_specs: Mapping[str, Any], turns: list[dict[str, Any]], spec_id: str
) -> dict[str, Any] | None:
    """The Search block of `spec_id`: the open turn's (`turn_specs`), else the
    newest one an earlier turn's record kept; `None` when none did."""
    block = turn_specs.get(spec_id)
    if isinstance(block, dict):
        return block
    for turn in reversed(turns):
        block = (turn.get("specs") or {}).get(spec_id)
        if isinstance(block, dict):
            return block
    return None


def stored_reading(
    turn_readings: list[dict[str, Any]], turns: list[dict[str, Any]], spec_id: str
) -> SearchReading | None:
    """The newest reading whose own spec_id is `spec_id`, the open turn's first."""
    newest_first = [
        *reversed(turn_readings),
        *(r for turn in reversed(turns) for r in reversed(turn.get("readings") or ())),
    ]
    for stored in newest_first:
        if isinstance(stored, Mapping) and stored.get("spec_id") == spec_id:
            return SearchReading.model_validate(stored)
    return None


def unit_pattern(unit: UnitChoice, offer: UnitOffer) -> str:
    """The pattern, read case-insensitively, of each phrase that names `unit`
    (`retitled` lists them): the currency by its code or by the name `offer`
    gives it, after an optional price basis; a price basis followed by
    "prices"; or, for a constant unit, "constant <base year>" alone."""
    names = {currency.code: currency.name for currency in offer.currencies}
    name = names.get(unit.currency)
    if unit.prices is UnitPrices.CONSTANT:
        year = re.escape(str(unit.base_year))
        basis = rf"constant(?:\s+{year})?"
        priced = rf"{basis}|{year}"
        alone = rf"|(?P<alone>constant\s+{year})"
    else:
        basis = priced = "current"
        alone = ""
    currencies = [rf"(?P<code>(?-i:{re.escape(unit.currency)}))"]
    if name:
        currencies.append(rf"(?P<name>{re.escape(name)})")
    return (
        rf"(?<!\w)(?:(?:(?P<basis>{basis})\s+)?(?:{'|'.join(currencies)})"
        rf"|(?P<prices>{priced})\s+prices{alone})(?!\w)"
    )


def retitled(title: str, *, old: UnitChoice, new: UnitChoice, offer: UnitOffer) -> str:
    """`title` with each phrase that names `old` naming `new` in the same shape.

    A phrase is the currency `old` is served in, by its code or by the name
    `offer` gives it, after an optional price basis; a price basis followed by
    "prices"; or, when `old` is constant, "constant <base year>" alone. The
    basis is the one `old` is served in: "constant" or "constant <base
    year>" when `old` is constant, "current" when it is current, and "<base
    year>" before "prices" when it is constant. The whole phrase is written
    again from `new`, its basis in the words `unit_words` gives it
    ("constant 2023", "current"): the basis and the new currency, by code or
    by name as the phrase named the old one, the basis and "prices", or the
    basis alone. A currency named with no basis is named alone, and a year
    before "prices" stays a bare year when `new` is constant. Any other words
    stand, another basis and a year alone among them.
    """
    names = {currency.code: currency.name for currency in offer.currencies}
    old_name = names.get(old.currency)
    new_name = names.get(new.currency, new.currency)
    words = unit_words(new, offer)
    new_basis = words[: len(words) - len(new_name) - 1].lower()
    pattern = unit_pattern(old, offer)

    def swap(match: re.Match[str]) -> str:
        said = match.group(0)
        if match.groupdict().get("alone") is not None:
            phrase = new_basis
        elif match.group("prices") is not None:
            bare_year = match.group("prices")[:1].isdigit() and new.base_year is not None
            phrase = f"{new.base_year if bare_year else new_basis} prices"
        elif match.group("basis") is not None:
            currency = new.currency if match.group("code") is not None else new_name
            phrase = f"{new_basis} {currency}"
        elif match.group("code") is not None:
            return new.currency
        else:
            # A name opens on a capital only where it opens the title, or
            # where the title capitalises a name written lower-case.
            assert old_name is not None
            if match.start() == 0 or (said[:1].isupper() and old_name[:1].islower()):
                return new_name[:1].upper() + new_name[1:]
            return new_name
        opener = phrase[:1].upper() if said[:1].isupper() else phrase[:1]
        return opener + phrase[1:]

    return re.sub(pattern, swap, title, flags=re.IGNORECASE)
