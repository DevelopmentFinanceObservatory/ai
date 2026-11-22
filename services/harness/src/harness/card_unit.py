"""Whether a card's unit can change, and the pieces that change it.

A card can change unit when it draws the frame of one `get_data` retrieval of a
spec Search converts (`resolved.unit_conversion`, read by
`harness.unit_offer.unit_offer`): its frame is that retrieval's own, not a
frame an analysis step derived from it, and the retrieval named no `scope`,
so its spec_id is the one Search answered. The card's `provenance.inputs[0]`
then names that spec_id.

A change resubmits the spec in the new unit (`harness.unit_switch.resubmit_unit`),
fetches the new spec with the arguments that shaped the card's frame, and
redraws the card on the new frame in its slot (`harness.studio_edit.unit_events`).
Those arguments are read off the retrieval's manifest, whose data artifact
names the rendering apis minted (`frame_arguments`). A rendering this module
cannot map back to `get_data`'s arguments leaves the card without a change.

`UnitCard` is what the conversation keeps with such a card, under
`CARD_UNIT_KEY` on its stored entry: the spec_id it reads and those
arguments. No model reads it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from common.materialisation.frame_name import KIND_COLUMNS, WITHOUT_PARTY_TOTALS
from common.materialisation.rendering import DEFAULT_MINT_OPTIONS
from data_mcp.frame_materialisation import complete_rendering

from harness.provenance import CardLineage, data_artifact
from harness.turn_records import SearchReading
from harness.unit_offer import offered_unit_refusal, served_unit, unit_offer, unit_words
from harness.wire import ApplyUnit, UnitChange, UnitChoice, UnitOffer, UnitPrices

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
    """A card whose unit can change: the spec_id its frame was retrieved from
    and the `get_data` arguments, `spec_id` aside, that shaped the frame."""

    spec_id: str
    args: dict[str, Any]

    def as_json(self) -> dict[str, Any]:
        return {"spec_id": self.spec_id, "args": dict(self.args)}

    @classmethod
    def from_json(cls, stored: Any) -> UnitCard | None:
        """The `UnitCard` `as_json` wrote, or `None` for anything else."""
        if not isinstance(stored, Mapping):
            return None
        spec_id, args = stored.get("spec_id"), stored.get("args")
        if not isinstance(spec_id, str) or not spec_id or not isinstance(args, Mapping):
            return None
        return cls(spec_id=spec_id, args=dict(args))


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


def unit_card(component: Mapping[str, Any], lineage: CardLineage | None) -> UnitCard | None:
    """The `UnitCard` of `component`, a written card, drawn from `lineage`;
    `None` when the card does not draw one retrieval's own frame, or its
    arguments cannot be read."""
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
    return UnitCard(spec_id=spec_id, args=args)


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


def unit_move(conversation: ConversationState, pick: ApplyUnit) -> UnitMove | tuple[str, str]:
    """The change `pick` asks of `conversation`, or its refusal's constraint
    and detail: `scope_choice_stale` when the card is off the canvas, when its
    `provenance.inputs[0]` names a spec other than `pick.spec_id`, or when no
    turn kept that spec's Search block; `unit_not_offered` for a unit the
    card's offer does not hold or the one it already shows. `pick.turn` plays
    no part."""
    composition = conversation.composition
    blocks = composition.blocks if composition is not None else []
    on_canvas = next((b for b in blocks if b.component_id == pick.component_id), None)
    if on_canvas is None:
        return "scope_choice_stale", UNIT_GONE_NOTE
    entry = conversation.components.get(pick.component_id) or {}
    card = UnitCard.from_json(entry.get(CARD_UNIT_KEY))
    if card is None or card.spec_id != pick.spec_id:
        return "scope_choice_stale", UNIT_STALE_NOTE
    block = stored_block(conversation.turn_specs, conversation.turns, card.spec_id)
    change = card_unit_change(card, block)
    if block is None or change is None:
        return "scope_choice_stale", UNIT_STALE_NOTE
    offer, served = change.offer, change.offer.start
    if (refused := offered_unit_refusal(offer, pick.unit)) is not None:
        return "unit_not_offered", f"That unit is not offered for this card: {refused}."
    if pick.unit == served:
        return "unit_not_offered", f"The card already shows {unit_words(served, offer)}."
    return UnitMove(
        component_id=pick.component_id,
        title=str(entry.get("title") or on_canvas.title or "The card"),
        card=card,
        served=served,
        unit=pick.unit,
        offer=offer,
        block=block,
        reading=stored_reading(conversation.turn_readings, conversation.turns, card.spec_id),
    )


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


def retitled(title: str, *, old: UnitChoice, new: UnitChoice, offer: UnitOffer) -> str:
    """`title` with the words that name `old` naming `new`: the whole unit as
    `unit_words` writes it, then the price basis ("constant 2022 prices",
    "2022 prices", "constant 2022", "current prices", "current USD") and the
    currency, by its code and by the name `offer` gives it, written as
    `offer` writes it. Any other words stand."""
    names = {currency.code: currency.name for currency in offer.currencies}
    old_name, new_name = names.get(old.currency), names.get(new.currency)
    retitled = _replace(title, unit_words(old, offer), unit_words(new, offer))
    constant = new.prices is UnitPrices.CONSTANT
    if old.prices is UnitPrices.CONSTANT:
        year = old.base_year
        prices = f"{new.base_year} prices" if constant else "current prices"
        for phrase, replacement in (
            (f"constant {year} prices", f"constant {prices}" if constant else prices),
            (f"{year} prices", prices),
            (f"constant {year}", f"constant {new.base_year}" if constant else "current"),
        ):
            retitled = _replace(retitled, phrase, replacement)
    elif constant:
        retitled = _replace(retitled, "current prices", f"constant {new.base_year} prices")
        for word in dict.fromkeys(w for w in (old.currency, old_name) if w):
            retitled = _replace(retitled, f"current {word}", f"constant {new.base_year} {word}")
    if old.currency != new.currency:
        retitled = _replace(retitled, old.currency, new.currency, keep_case=False)
        if old_name and new_name:
            retitled = _replace_name(retitled, old_name, new_name)
    return retitled


def _replace(text: str, phrase: str, replacement: str, *, keep_case: bool = True) -> str:
    """`text` with each whole-word `phrase`, in any case, replaced; with
    `keep_case`, a match opening on a capital opens the replacement on one."""

    def swap(match: re.Match[str]) -> str:
        if keep_case and match.group(0)[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        if keep_case:
            return replacement[:1].lower() + replacement[1:]
        return replacement

    pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
    return re.sub(pattern, swap, text, flags=0 if not keep_case else re.IGNORECASE)


def _replace_name(text: str, old_name: str, new_name: str) -> str:
    """`text` with each whole-word `old_name`, in any case, replaced by
    `new_name` as Search writes it ("US dollars", "euros"). The first letter
    is raised only where the match opens `text`, or where it is capitalised
    while `old_name` is written lower-case."""

    def swap(match: re.Match[str]) -> str:
        opens_capital = match.group(0)[:1].isupper() and old_name[:1].islower()
        if match.start() == 0 or opens_capital:
            return new_name[:1].upper() + new_name[1:]
        return new_name

    pattern = rf"(?<!\w){re.escape(old_name)}(?!\w)"
    return re.sub(pattern, swap, text, flags=re.IGNORECASE)
