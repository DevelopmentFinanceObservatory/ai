"""The prices the user chose for a part answering an ask, held on the plan as their decision.

A part is a search reading whose prices were open (`harness.turn_records.
open_prices`). The orchestrator's unit question names its part (`unit_of`):
code either offers a unit, or, when Search serves constant prices only as
another reading, gives it the part's `open_prices` entries as options, each
marked with the part and its prices (`harness.graph.state.priced_options`).
The planner reads the user's answers off the answered interrupt, one per
part (`planner.call.prices_chosen`).

`prices_decisions` writes each pick as a `resolved_by=user` decision fixing
the price basis alone (`DecisionFixes.prices`), cited by every dataset
requirement the planner pinned to one of the part's series (`Parts.series`),
so each keeps its own currency and base year. A requirement alignment moves
is read on the pin it states (`submission.service.admit` derives every
decision before it pins the compile). It refuses a plan in which
one of those leaves its unit out, states other prices or pins a series in
other prices, naming the unit to state: the base year comes from the
requirement's own pin, or from the series it is told to move to. With no
series shown in constant prices the user kept, the planner's entry ends, so
the orchestrator can search for one. Once recorded, the decision stands as
every user decision does (`submission.service._user_decision_refused`).

A unit question's answer names a whole unit, and the series the agent
fetches for the part in it (`PricesChoice.unit`): the chosen unit, or the
unit the series is served in when Search refused the chosen one. A
requirement pinning a series measuring what that series measures states
that unit whole and pins a series served in it, or is refused, told to
state the unit and pin that series.

`base_year_refusal` gives the same instruction for a unit in constant prices
with no base year on such a requirement. A requirement reading no part the
user priced is left as it was.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from types import MappingProxyType
from typing import Any, Literal, NamedTuple

from compiler.evidence import PriceBasis, UnitReading, read_unit_currency, read_unit_dcid
from compiler.evidence import requested_unit as requested_unit_of
from compiler.sources import readable_source
from harness.turn_events import PLANNER_ASK
from planner.phrasing import CONSTANT_PRICES, CURRENT_PRICES
from planner.plan import (
    BaseYearUnstatedError,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    EvidenceRequirement,
    PinSource,
    Plan,
    ResolvedBy,
    SessionState,
    Unit,
)
from planner.tool_results import Action, Outcome, ProblemCode, problem
from planner.workflow import SelectedSource
from pydantic import ValidationError

_LABELS: dict[PriceBasis, str] = {"current": CURRENT_PRICES, "constant": CONSTANT_PRICES}
# Each measurement qualifier a series states its prices by.
BASES: dict[str, PriceBasis] = {"Nominal": "current", "RealValue": "constant"}
_SPEC_ID = re.compile(r"spec_[0-9a-f]{32}")

# What the decision for a part's prices is recorded under, the part's spec_id after it.
PRICES_SLOT_PREFIX = "user_prices_"
# What the decision for the whole unit a unit question's answer holds its
# parts to is recorded under, the question's lead part after it (`unit_slot`).
UNIT_SLOT_PREFIX = "user_unit_"
# The prefixes of every decision recording the prices the user chose, which code alone writes.
USER_PRICES_SLOT_PREFIXES = (PRICES_SLOT_PREFIX, UNIT_SLOT_PREFIX)

# The fixed choices of the planner's ask a prices refusal sends it to, by
# the prices each picks. A pick sets the part's prices
# (`planner.call.prices_chosen`).
PRICES_OPTIONS: tuple[dict[str, str], ...] = tuple(
    {"id": f"prices:{basis}", "label": f"{label[:1].upper()}{label[1:]}"}
    for basis, label in _LABELS.items()
)
PRICES_PICKS: dict[str, PriceBasis] = {f"prices:{basis}": basis for basis in _LABELS}
# The prices each option of a part's prices decision states, by its label
# (`Decision.options`), as a switch of the card row picks them.
PRICES_BY_LABEL: dict[str, PriceBasis] = {label: basis for basis, label in _LABELS.items()}


def prices_slot(part: str) -> str:
    """The `Decision.slot` of the prices the user chose for `part`."""
    return f"{PRICES_SLOT_PREFIX}{part}"


def unit_slot(group: str) -> str:
    """The `Decision.slot` of the whole unit a unit question's answer holds
    its parts to, the question led by part `group` (`PricesChoice.group`)."""
    return f"{UNIT_SLOT_PREFIX}{group}"


def decision_slots(choices: Mapping[str, PricesChoice]) -> dict[str, str]:
    """The slot of the decision recording each part's choice in `choices`, by the part.

    The parts on one unit row (`PricesChoice.group`), every one served in the
    whole unit its question chose, share one decision under `unit_slot`.
    Every other choice, a part Search refused to read in that unit among
    them, keeps a decision of its own under `prices_slot`.
    """
    return {
        part: (
            unit_slot(choice.group)
            if choice.group is not None and choice.unit is not None
            else prices_slot(part)
        )
        for part, choice in choices.items()
    }


def whole_unit_slots(choices: Mapping[str, PricesChoice]) -> frozenset[str]:
    """The slots of the unit rows recording a whole unit the user chose
    (`decision_slots`): each choice's that names a `unit` and sits on a row
    (`PricesChoice.group`). A bare prices answer, current or constant, names
    no unit, and a part Search refused to read in the unit chosen keeps a
    prices row of its own stating the unit it is served in."""
    slots = decision_slots(choices)
    return frozenset(
        slots[part]
        for part, choice in choices.items()
        if choice.unit is not None and choice.group is not None
    )


def plan_unit(plan: Plan, slot: str) -> Unit | None:
    """The whole unit `plan` holds the parts of the unit row under `slot`
    (`unit_slot`) to: the one `unit` every dataset requirement citing the
    slot states. `None` when none cites it, one states no unit, or they
    state different units."""
    units = {
        requirement.unit
        for requirement in plan.evidence_requirements
        if isinstance(requirement, DatasetRequirement) and slot in requirement.decision_ids
    }
    if len(units) != 1:
        return None
    (unit,) = units
    return unit


# The session-state slot holding, by each unit row's slot (`unit_slot`),
# the parts of the unit question's answer the row has absorbed, each with
# the ids of the dataset requirements that read it when admission wrote the
# row (`absorbed_rows`, `absorbed`). Admission alone writes it, and only
# adds to it: a switch of the card moves pins and keeps requirement ids, so
# what a row absorbed stays true across switches.
UNIT_ROWS_STATE_KEY = "planner:unit_rows"

# Each unit row's absorbed parts, by the row's slot, each with the ids of
# the requirements that read it.
type AbsorbedRows = Mapping[str, Mapping[str, tuple[str, ...]]]


def absorbed_rows(state: SessionState) -> dict[str, dict[str, tuple[str, ...]]]:
    """What `UNIT_ROWS_STATE_KEY` holds; empty when unset. An entry not in
    its shape is left out."""
    stored = state.get(UNIT_ROWS_STATE_KEY)
    rows: dict[str, dict[str, tuple[str, ...]]] = {}
    for slot, parts in (stored or {}).items() if isinstance(stored, Mapping) else ():
        if not isinstance(parts, Mapping):
            continue
        rows[str(slot)] = {
            str(part): tuple(str(rid) for rid in ids)
            for part, ids in parts.items()
            if isinstance(ids, list | tuple)
        }
    return rows


def save_absorbed_rows(state: SessionState, rows: AbsorbedRows) -> None:
    """Record `rows` as `UNIT_ROWS_STATE_KEY`."""
    state[UNIT_ROWS_STATE_KEY] = {
        slot: {part: list(ids) for part, ids in parts.items()} for slot, parts in rows.items()
    }


def absorbed(
    rows: AbsorbedRows,
    answered: Mapping[str, PricesChoice],
    readings: Mapping[str, tuple[str, ...]],
) -> dict[str, dict[str, tuple[str, ...]]]:
    """`rows` with each part of a unit question's answer (`answered`, the
    choices `prices_chosen` gives, `PricesChoice.group`) that some
    requirement reads (`readings`, `part_readings`) absorbed by its row,
    with the ids of the requirements reading it. A part read again is
    recorded with this reading's ids, and no part leaves its row."""
    merged = {slot: dict(parts) for slot, parts in rows.items()}
    for part, ids in readings.items():
        choice = answered.get(part)
        if not ids or choice is None or choice.group is None or choice.unit is None:
            continue
        merged.setdefault(unit_slot(choice.group), {})[part] = ids
    return merged


class AnsweredPart(NamedTuple):
    """A part a unit question's answer put on a unit row: `group`, the row's
    lead part (`unit_slot`), `spec_id`, the series the agent fetches for the
    part, and `unit`, the unit that series is served in
    (`harness.graph.state.ServedUnit`)."""

    group: str
    spec_id: str
    unit: Unit | None


def plan_held_choices(
    choices: Mapping[str, PricesChoice],
    plan: Plan,
    *,
    units: Mapping[str, Unit],
    answered: Mapping[str, AnsweredPart] = MappingProxyType({}),
    rows: AbsorbedRows = MappingProxyType({}),
    fresh: Collection[str] = (),
) -> dict[str, PricesChoice]:
    """`choices` with every unit row `plan` states holding its members to the row's unit.

    A unit row's choices (`PricesChoice.group`) seed the plan that first
    states the row. Once `plan` states the row (slot `unit_slot`), its unit
    U is the one `units` holds for the slot, the unit a switch of the row
    picked, else the one its citing requirements state (`plan_unit`), and
    every member of the row holds U, whether or not `choices` still holds a
    choice for it, as after a message the user typed:

    - each distinct pin of the requirements citing the row, as that pin in U;
    - each part the unit question's answer put on the row (`answered`) that
      the row has not absorbed, as the series fetched for it, in U, which
      that series is served in or not (`PricesChoice.in_unit`). A part the
      row absorbed is one
      `rows` records under the slot whose requirements still cite the row,
      every part on state recorded before rows were (no entry for the
      slot); its requirements' pins stand for it.

    Two answers stand over the row: a part answered since the plan landed
    (`fresh`) holds its row's choices as answered, and a pin a choice
    outside the row holds, as a bare prices answer the user gave on it
    after a refusal, keeps that choice. A row `plan` does not state, or
    whose requirements state no one unit, keeps the choices `choices` gives it.
    """
    held = dict(choices)
    for decision in plan.decisions:
        slot = decision.slot
        if not slot.startswith(UNIT_SLOT_PREFIX):
            continue
        group = slot.removeprefix(UNIT_SLOT_PREFIX)
        live = [part for part, choice in held.items() if choice.group == group]
        members = [part for part, answer in answered.items() if answer.group == group]
        if any(part in fresh for part in (*live, *members)):
            continue
        unit = units.get(slot) or plan_unit(plan, slot)
        if unit is None:
            continue
        citing = [
            requirement
            for requirement in plan.evidence_requirements
            if isinstance(requirement, DatasetRequirement) and slot in requirement.decision_ids
        ]
        pins = list(dict.fromkeys(r.source_spec_id for r in citing if r.source_spec_id is not None))
        recorded = rows.get(slot)
        cited = {requirement.id for requirement in citing}
        absorbed_parts = {
            part
            for part in members
            if recorded is None or cited.intersection(recorded.get(part, ()))
        }
        held = {part: choice for part, choice in held.items() if choice.group != group}
        holds = [
            *((pin, pin, True) for pin in pins),
            *(
                (part, answered[part].spec_id, answered[part].unit == unit)
                for part in members
                if part not in absorbed_parts
            ),
        ]
        for key, spec_id, in_unit in holds:
            if key not in held:
                held[key] = PricesChoice(
                    unit.prices, spec_id, unit=unit, group=group, in_unit=in_unit
                )
    return held


class PricesChoice(NamedTuple):
    """The prices the user picked for a part, the spec_id of the series they
    picked, when the option carried one, and whether they answered the
    planner's own prices question on the part (`asked`), which is not asked
    twice.

    A unit question's answer also names the whole `unit`, currency and base
    year with the prices, that the series `spec_id` is served in: the series
    the agent fetches for the part (`harness.graph.state.units_served`), and
    the `group`, the lead part of the unit row the part sits on
    (`harness.graph.state.ServedUnit.row`), `None` for a part on none.
    `in_unit` is false for a part a switch of its row moved to a unit its
    series `spec_id` is not served in, the series no longer read
    (`plan_held_choices`): the part has no series in `unit`."""

    prices: PriceBasis
    spec_id: str | None = None
    asked: bool = False
    unit: Unit | None = None
    group: str | None = None
    in_unit: bool = True

    @classmethod
    def of_candidate(cls, candidate: Mapping[str, Any]) -> PricesChoice | None:
        """The choice a picked clarify candidate states, `None` when it prices no part.

        A candidate's id is its option's spec_id, or `o{n}` for an option
        carrying none (`harness.graph.state.ask_candidates`).
        """
        prices = candidate.get("prices")
        if prices not in _LABELS or not isinstance(candidate.get("part"), str):
            return None
        offered = candidate.get("spec_id")
        spec_id = offered if isinstance(offered, str) and _SPEC_ID.fullmatch(offered) else None
        return cls(prices, spec_id)

    @property
    def label(self) -> str:
        """The prices in words, as the prices question offered them."""
        return _LABELS[self.prices]

    @property
    def held_currency(self) -> str | None:
        """The currency the choice holds the part to: a whole `unit`'s; `None`
        when it names none."""
        return self.unit.currency if self.unit is not None else None

    @property
    def words(self) -> str:
        """What the user chose in words: the prices, with the base year and
        currency of a whole `unit`, "constant 2020 prices in EUR". The prices
        decision's `chosen`."""
        if self.unit is None:
            return self.label
        year = f" {self.unit.base_year}" if self.unit.base_year is not None else ""
        return f"{self.unit.prices}{year} prices in {self.unit.currency}"


@dataclass(frozen=True)
class Parts:
    """What the prices rules read of the datasets on record.

    `source_of` and `block_of` answer a spec_id's source and block, `None`
    when none is held. `shown` is every spec_id the planner was shown, and
    `searched` those a search settled on as readings of their own, not
    fetched as another count of one (`of_shown`); `None` reads every shown
    series so. `priced` are the parts the user chose prices for,
    `fetched` the part each series fetched for a part in a currency the
    user chose is fetched for, by that series, and `held` the currency a
    whole unit the user chose holds each part to (`priced_by`).

    A part's series are the part, each series fetched for it, each series
    its block offers in its place, another count or other prices, and each
    shown series measuring what one of those measures in the part's
    currency, or in the currency the user's whole unit holds it to, in any
    prices (`same_measure`), such as one a later search found, each less
    any series another reading owns (`owner_of`).
    """

    source_of: Callable[[str], SelectedSource | None]
    block_of: Callable[[str], Mapping[str, Any] | None]
    shown: Collection[str] = ()
    searched: Collection[str] | None = None
    priced: Collection[str] = ()
    fetched: Mapping[str, str] = field(default_factory=dict)
    held: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def of_shown(
        cls,
        shown: Mapping[str, PinSource],
        source_of: Callable[[str], SelectedSource | None],
        block_of: Callable[[str], Mapping[str, Any] | None],
        *,
        pinned: Collection[str] = (),
    ) -> Parts:
        """The parts of the datasets `shown` to the planner, read through
        `source_of` and `block_of`: each settled one is a reading of its own
        unless it was fetched as another count of one (`PinSource.other_count_of`).

        A dataset a requirement pins (`pinned`) that the planner was not shown,
        a pin compiled for another input that the planner pins directly, or a
        source translation's discovery found that a referral offers in place
        of a pin (`planner.plan.held_sources`), is a reading of its own too,
        its measure read off the block `block_of` holds for it. So the
        user's prices choice on that measure holds it."""
        compiled = [
            spec_id
            for spec_id in dict.fromkeys(pinned)
            if spec_id not in shown and block_of(spec_id) is not None
        ]
        return cls(
            source_of,
            block_of,
            shown=[*shown.keys(), *compiled],
            searched=[
                *(
                    spec_id
                    for spec_id, source in shown.items()
                    if source.settled and source.other_count_of is None
                ),
                *compiled,
            ],
        )

    def priced_by(self, choices: Mapping[str, PricesChoice]) -> Parts:
        """These parts, with the parts `choices` prices marked, and the series
        fetched for each part a choice holds to a currency
        (`PricesChoice.held_currency`), whose currency may differ from the part's,
        with that currency. A fetched series that is itself a part `choices`
        prices stays that part's alone, so no series belongs to two parts."""
        return replace(
            self,
            priced=frozenset(choices),
            held={
                part: choice.held_currency
                for part, choice in choices.items()
                if choice.held_currency is not None
            },
            fetched={
                choice.spec_id: part
                for part, choice in choices.items()
                if choice.held_currency is not None
                and choice.spec_id is not None
                and choice.spec_id not in choices
            },
        )

    def unit_of(self, spec_id: str) -> UnitReading:
        """What the unit of the series `spec_id` states (`source_unit`), nothing when no source is held."""
        source = self.source_of(spec_id)
        return source_unit(source) if source is not None else UnitReading()

    def alternatives(self, part: str) -> list[Mapping[str, Any]]:
        """The series `part`'s block offers in its place, another count or other prices."""
        resolved = (self.block_of(part) or {}).get("resolved") or {}
        offered = resolved.get("alternative_measures") or ()
        return [alt for alt in offered if isinstance(alt, Mapping)]

    def offered(self, part: str, *, counts_only: bool = False) -> list[str]:
        """`part` and the spec_id of each series its block offers in its place,
        leaving out those in other prices (axis `unit`) when `counts_only`."""
        return [
            part,
            *(
                spec_id
                for alt in self.alternatives(part)
                if isinstance(spec_id := (alt.get("definition") or {}).get("spec_id"), str)
                and not (counts_only and alt.get("axis") == "unit")
            ),
        ]

    def series(self, part: str) -> frozenset[str]:
        """`part`, each series fetched for it (`fetched`), and every series it
        reaches (`_reached`) that no other reading owns (`owner_of`)."""
        return frozenset(
            {
                part,
                *(one for one, owner in self.fetched.items() if owner == part),
                *(one for one in self._reached(part) if self.owner_of(one) in (part, None)),
            }
        )

    def _reached(self, reading: str) -> list[str]:
        """Every series `reading` offers (`offered`), and every shown series
        measuring what one of those measures in its own currency, or in the
        currency a whole unit the user chose holds it to (`held`), as
        `same_measure` reads it."""
        currencies = [self.unit_of(reading).currency]
        if (held := self.held.get(reading)) is not None and held not in currencies:
            currencies.append(held)
        offered = self.offered(reading)
        return [
            *offered,
            *(
                found
                for currency in currencies
                for one in offered
                for found in self.same_measure(one, currency)
            ),
        ]

    def _offers_in_prices(self, reading: str) -> list[str]:
        """The series `reading`'s block offers as its count in other prices (axis `unit`)."""
        return [one for one in self.offered(reading) if self.unit_offered(reading, one)]

    def _measures(self, reading: str) -> list[str]:
        """Every shown series measuring what `reading` measures in its currency,
        any window or prices (`same_measure`)."""
        return self.same_measure(reading, self.unit_of(reading).currency)

    @cached_property
    def _readings(self) -> tuple[str, ...]:
        """The series a search settled on (`searched`) not in constant prices.
        A constant series is one found for a reading's prices."""
        searched = self.shown if self.searched is None else self.searched
        return tuple(one for one in searched if self.unit_of(one).prices != "constant")

    def _part_of(self, reading: str) -> str:
        """The reading whose part `reading` is: the first priced reading
        measuring what it measures (`_measures`), whatever the shown order,
        else the first such reading. Readings differing only in window are
        one part; another count or currency is another measure, save the
        currency a whole unit the user chose holds a part to (`held`), a
        part in constant prices included: a reading measuring what that part
        measures in that currency is that part's, so the user's unit holds it."""
        holder = next(
            (
                one
                for one, currency in self.held.items()
                if one != reading and reading in self.same_measure(one, currency)
            ),
            None,
        )
        if holder is not None:
            return holder
        same = [reading, *(one for one in self._measures(reading) if one in self._readings)]
        ordered = [one for one in self._readings if one in same]
        return next((one for one in ordered if one in self.priced), ordered[0])

    def owner_of(self, spec_id: str) -> str | None:
        """The one reading whose part `spec_id` belongs to: a priced reading
        itself, any other reading the part it is one of (`_part_of`); else
        the readings that offer it in other prices, then those measuring what
        it measures, then those reaching it through another count they offer
        (`_reached`), each time a priced one of them first. `None` when no
        reading reaches it. So a gross part and a net part of one flow pair
        never claim each other's series, and a constant series goes to the
        reading the user priced. A series fetched for a part in a currency
        the user chose (`fetched`) is that part's."""
        if spec_id in self.fetched:
            return self.fetched[spec_id]
        if spec_id in self._readings:
            return spec_id if spec_id in self.priced else self._part_of(spec_id)
        for claims in (self._offers_in_prices, self._measures, self._reached):
            claimants = [one for one in self._readings if spec_id in claims(one)]
            if claimants:
                return next(
                    (one for one in claimants if one in self.priced), self._part_of(claimants[0])
                )
        return None

    def current_of(self, part: str, pin: str) -> str | None:
        """The series of `part` measuring what `pin` measures in current prices:
        the part when its block offers `pin` as its count in other prices, else
        the part or the count it offers, and no other reading owns
        (`owner_of`), whose identity, prices left out (`_priceless`), equals
        the pin's; `None` when none does."""
        if self.unit_offered(part, pin):
            return part
        wanted = _priceless(self.definition_of(pin))
        if wanted is None:
            return None
        return next(
            (
                one
                for one in self.offered(part, counts_only=True)
                if one != pin
                and self.owner_of(one) in (part, None)
                and _priceless(self.definition_of(one)) == wanted
            ),
            None,
        )

    def measures_alike(self, one: str, other: str) -> bool:
        """Whether `one` and `other` measure the same thing, window, unit and
        prices left out (`_priceless`): their identities are both known and equal."""
        identity = _priceless(self.definition_of(one))
        return identity is not None and identity == _priceless(self.definition_of(other))

    def definition_of(self, spec_id: str) -> Mapping[str, Any] | None:
        """What states `spec_id`'s definition: its own block, or else the entry a
        shown block offers it under (`alternatives`); `None` when neither holds it."""
        block = self.block_of(spec_id)
        if block is not None:
            return block
        return next(
            (
                alt
                for shown in self.shown
                for alt in self.alternatives(shown)
                if (alt.get("definition") or {}).get("spec_id") == spec_id
            ),
            None,
        )

    def unit_offered(self, part: str, spec_id: str) -> bool:
        """Whether `part`'s block offers `spec_id` as its own count in other prices (axis `unit`)."""
        return any(
            (alt.get("definition") or {}).get("spec_id") == spec_id and alt.get("axis") == "unit"
            for alt in self.alternatives(part)
        )

    def same_measure(self, series: str, currency: str | None) -> list[str]:
        """Each shown series but `series` measuring what `series` measures, in
        `currency` and any prices: its identity equals the one of `series`
        (`definition_of`) with the prices left out (`_priceless`), so
        provenance and measurement method match, and its unit states `currency`."""
        wanted = _priceless(self.definition_of(series))
        if wanted is None:
            return []
        return [
            spec_id
            for spec_id, identity in self._identities.items()
            if spec_id != series
            and identity == wanted
            and self.unit_of(spec_id).currency == currency
        ]

    @cached_property
    def _identities(self) -> dict[str, dict[str, Any] | None]:
        """Each shown series' identity with its prices left out (`_priceless`)."""
        return {spec_id: _priceless(self.block_of(spec_id)) for spec_id in self.shown}

    def constant_found(
        self, series: str, currency: str | None, *, part: str
    ) -> tuple[str, UnitReading] | None:
        """A shown series of `part` (`series`) measuring what `series` measures
        in `currency` at constant prices of a stated base year (`same_measure`),
        such as one a later search found, with its unit; `None` when none is shown."""
        held = self.series(part)
        for spec_id in self.same_measure(series, currency):
            if spec_id not in held:
                continue
            reading = self.unit_of(spec_id)
            if reading.prices == "constant" and reading.base_year is not None:
                return spec_id, reading
        return None


def measure_key(block: Mapping[str, Any]) -> str | None:
    """What `block` measures, its window and prices left out, with its currency,
    as one comparable key: two parts with equal keys measure the same thing,
    as `Parts.same_measure` reads it, and take one prices choice. `None` when
    the block states no identity."""
    identity = _priceless(block)
    if identity is None:
        return None
    source = readable_source(block)
    currency = source_unit(source).currency if source is not None else None
    return json.dumps([identity, currency], sort_keys=True)


def _priceless(block: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A block's identity less its window, its unit facet and its measurement
    qualifier, which state its prices; `None` when it states none. No facets
    but the unit reads as no facets, since stating a unit by request gives an
    identity with no facets a `facets` holding the unit alone
    (`data_mcp.definition_writes.unit_identity`)."""
    identity = ((block or {}).get("definition") or {}).get("identity")
    if not isinstance(identity, Mapping):
        return None
    kept = {k: v for k, v in identity.items() if k != "requested_window"}
    facets = kept.pop("facets", None)
    if isinstance(facets, Mapping) and (others := {k: v for k, v in facets.items() if k != "unit"}):
        kept["facets"] = others
    shape = kept.get("shape")
    if isinstance(shape, Mapping):
        kept["shape"] = {k: v for k, v in shape.items() if k != "measurement_qualifier"}
    return kept


def source_unit(source: SelectedSource) -> UnitReading:
    """What `source`'s unit states outside its label: a unit it binds by request,
    else its unit id's structure, else its measure's `measurement_qualifier`
    for the prices and its label for the currency."""
    requested = requested_unit_of(source.identity)
    if requested is not None:
        return UnitReading(requested.currency, requested.prices, requested.base_year)
    reading = read_unit_dcid(source.unit_ref)
    if reading.prices is None:
        shape = (source.identity or {}).get("shape")
        qualifier = shape.get("measurement_qualifier") if isinstance(shape, dict) else None
        reading = reading._replace(
            prices=BASES.get(qualifier) if isinstance(qualifier, str) else None
        )
    if reading.currency is None:
        reading = reading._replace(currency=read_unit_currency(source.unit_ref, source.unit_label))
    return reading


class _Wanted(NamedTuple):
    """What a requirement reading a priced part states to honour the user's
    choice: `unit`, and `pin` when it moves to another series of the part;
    `unit` is `None` when no series shown serves those prices. `exact` when
    the requirement states `unit` whole, currency and base year included.
    `unserved` when the requirement's pin is not served in `unit` and no
    series of the part is known in it, so the pin cannot stand."""

    unit: Unit | None
    pin: str | None = None
    exact: bool = False
    unserved: bool = False


def _wanted(
    pin: str | None, currency: str | None, part: str, choice: PricesChoice, parts: Parts
) -> _Wanted:
    """The unit, and the series, a requirement pinning `pin` states in `choice`'s prices.

    The pin's own unit gives the currency, when the requirement states none,
    else the part's, and the base year of constant prices. A requirement
    pinned to a series of the part not in constant prices moves, for constant
    prices, to the series the user picked when it pins the part itself, or
    else to a series a search found measuring what the pin measures in those
    prices and that currency (`Parts.constant_found`), with that series' base
    year. One pinned to a series in constant prices moves for current prices
    to the series of the part measuring the same (`Parts.current_of`).

    A choice naming a whole unit holds some pins to it whole (`_whole_unit`).
    """
    whole = _whole_unit(pin, part, choice, parts)
    return whole if whole is not None else _prices_wanted(pin, currency, part, choice, parts)


def _prices_wanted(
    pin: str | None, currency: str | None, part: str, choice: PricesChoice, parts: Parts
) -> _Wanted:
    """`_wanted` for the prices of `choice` alone."""
    own = parts.unit_of(pin) if pin is not None else UnitReading()
    currency = choice.held_currency or currency or own.currency or parts.unit_of(part).currency
    if currency is None:
        return _Wanted(None)
    if choice.prices == "current":
        if own.prices != "constant":
            return _Wanted(Unit(currency=currency, prices="current"))
        moved = parts.current_of(part, pin) if pin is not None else None
        return _Wanted(Unit(currency=currency, prices="current"), moved) if moved else _Wanted(None)
    if own.prices == "constant" and own.base_year is not None:
        return _Wanted(Unit(currency=currency, prices="constant", base_year=own.base_year))
    found = _constant_series(pin, part, choice, parts, currency) if pin is not None else None
    if found is None:
        return _Wanted(None)
    return _Wanted(Unit(currency=currency, prices="constant", base_year=found[1]), found[0])


def _whole_unit(pin: str | None, part: str, choice: PricesChoice, parts: Parts) -> _Wanted | None:
    """What a requirement pinning `pin` states for a choice on `part` naming a
    whole unit (`PricesChoice.unit`), `None` when the choice names none or
    `pin` is none of the part's series (`Parts.series`) and does not measure
    what the series fetched for the part measures (`Parts.measures_alike`).

    The requirement states the unit whole. The pin stands when it is the
    series fetched or is served in the unit; it moves to that series when it
    is the part itself, which that series reads in the unit, or measures
    what that series measures; and it stands, held to the unit, when it is
    another of the part's series. The part's own pin needs no block for the
    series fetched, which a re-read may leave unheld.

    A part with no series in the unit (`PricesChoice.in_unit` false) moves
    nowhere: a pin served in the unit stands, and one that is the series
    fetched, the part, or measures what they measure, or another of the
    part's series, cannot stand (`_Wanted.unserved`)."""
    if choice.unit is None or choice.spec_id is None or pin is None:
        return None
    if not choice.in_unit:
        return _unserved_unit(pin, choice.spec_id, part, choice.unit, parts)
    if pin == choice.spec_id or _served_in(parts.unit_of(pin), choice.unit):
        return _Wanted(choice.unit, exact=True)
    if pin == part or parts.measures_alike(pin, choice.spec_id):
        return _Wanted(choice.unit, choice.spec_id, exact=True)
    if pin in parts.series(part):
        return _Wanted(choice.unit, exact=True)
    return None


def _unserved_unit(pin: str, fetched: str, part: str, unit: Unit, parts: Parts) -> _Wanted | None:
    """`_whole_unit` for a part with no series in `unit`, whose series
    `fetched` is served in another."""
    if _served_in(parts.unit_of(pin), unit):
        return _Wanted(unit, exact=True)
    if pin in (fetched, part) or parts.measures_alike(pin, fetched) or pin in parts.series(part):
        return _Wanted(unit, exact=True, unserved=True)
    return None


def _served_in(reading: UnitReading, unit: Unit) -> bool:
    """Whether a series whose unit reads `reading` is served in `unit`."""
    return (reading.currency, reading.prices, reading.base_year) == (
        unit.currency,
        unit.prices,
        unit.base_year,
    )


def _constant_series(
    pin: str, part: str, choice: PricesChoice, parts: Parts, currency: str
) -> tuple[str, int] | None:
    """The series in constant prices a requirement pinning `pin` moves to, with
    its base year: the one the user picked, when `pin` is the part itself, else
    one of the part's series a search found measuring what `pin` measures in
    `currency` (`Parts.constant_found`)."""
    reading = parts.unit_of(choice.spec_id) if choice.spec_id is not None else UnitReading()
    if (
        pin == part
        and choice.spec_id is not None
        and reading.prices == "constant"
        and reading.base_year is not None
    ):
        return choice.spec_id, reading.base_year
    found = parts.constant_found(pin, currency, part=part)
    if found is None or found[1].base_year is None:
        return None
    return found[0], found[1].base_year


def _refusal(
    code: ProblemCode,
    lead: str,
    *,
    slot: str,
    part: str,
    reading: object,
    choice: PricesChoice,
    requirement_id: object,
    wanted: _Wanted,
    propose: str,
    outcome: Literal[Outcome.REFUSED, Outcome.INVALID_INPUT],
) -> dict[str, object]:
    """The refusal naming what a requirement reading `part` states to honour `choice`.

    `detail` names the decision's `slot` (`decision_slots`), what the user `chose`, the `part`
    and the `requirement_id`, and the `unit` to state and the series to
    `pin`, when one is named. With no unit to state, no series shown serves
    the prices, and the planner asks the user which prices to go on in,
    offering `PRICES_OPTIONS`. Once the user has answered that question on
    the part and kept constant prices, only a search can find a series in
    them, and the planner has none: `PRICES_UNSERVED` ends the planner's
    entry, naming the series the requirement pins as `reading`, and the
    orchestrator reads why (`harness.sentences`).
    """
    detail: dict[str, Any] = {
        "slot": slot,
        "chosen": choice.label,
        "part": part,
        "requirement_id": requirement_id,
    }
    if wanted.unit is None and choice.asked and choice.prices == "constant":
        return problem(
            ProblemCode.PRICES_UNSERVED,
            f"{lead} No series shown serves it in {choice.label}, and the user kept them "
            "when asked, so a search for them has to find one first.",
            outcome=Outcome.REFUSED,
            detail={**detail, "reading": reading if isinstance(reading, str) else part},
        )
    if wanted.unit is None:
        return problem(
            code,
            f"{lead} No series shown serves it in {choice.label}. Ask the user through "
            f"`{PLANNER_ASK}`, with the action's `options`, which prices "
            "to go on in; once they pick, a patch states those.",
            outcome=outcome,
            action=Action(
                call=PLANNER_ASK,
                arguments={"options": [dict(option) for option in PRICES_OPTIONS]},
                needs=("message",),
            ),
            detail=detail,
        )
    detail["unit"] = wanted.unit.model_dump(mode="json", exclude_none=True)
    moved = ""
    if wanted.pin is not None:
        detail["pin"] = wanted.pin
        moved = f" and pin {wanted.pin}, the series of this part in {choice.words}"
    return problem(
        code,
        f"{lead} State `unit` {detail['unit']} on it{moved}.",
        outcome=outcome,
        action=Action(call=propose, needs=("evidence_requirements",)),
        detail=detail,
    )


def prices_decisions(
    choices: Mapping[str, PricesChoice],
    patch: Sequence[EvidenceRequirement],
    requirements: Sequence[EvidenceRequirement],
    *,
    parts: Parts,
    propose: str,
) -> tuple[list[Decision], list[EvidenceRequirement]] | dict[str, object]:
    """The user's prices `choices`, by part, as their decisions, with `patch` citing them, or the refusal of the plan.

    `requirements` is every requirement of the patched plan, `patch` this
    call's own. A dataset requirement reads a part when the pin the planner
    states, or a declared whole's part's (`DatasetRequirement.whole_of`),
    whose dataset admission compiles the whole from, is the part's series
    or one the part offers in its place (`Parts.series`). Each part that some requirement reads states
    one decision fixing `prices`, which each of those cites, its `chosen`
    what the user chose in words (`PricesChoice.words`). The parts on one
    unit row state one decision between them (`decision_slots`), cited by
    every requirement reading any of them, which includes one no part reads
    by its series that cites the row already or measures what a part of it
    measures (`_row_readers`). A
    requirement citing one that `patch` does not state is appended, so
    `Plan.amend` replaces it. A part no requirement reads states nothing.

    The refusal is `USER_DECISION_DROPPED` when such a requirement states no
    unit, and `USER_DECISION_CONTRADICTED` when it states other prices,
    another currency than the choice holds (`PricesChoice.held_currency`),
    another base year than a whole unit the user chose, or pins a series
    that cannot serve them (`_refusal`, `_wanted`). `propose`
    is the tool a refusal naming a unit calls.
    """
    parts = parts.priced_by(choices)
    slot_of = decision_slots(choices)
    read = _Readings.of(choices, requirements, parts=parts)
    decisions: dict[str, Decision] = {}
    slots: dict[str, list[str]] = {}
    for part, choice in choices.items():
        slot = slot_of[part]
        reading = read.by_part[part]
        for requirement in reading:
            unit = requirement.unit
            wanted = _wanted(
                read.pin(requirement),
                unit.currency if unit is not None else None,
                part,
                choice,
                parts,
            )
            if read.row_of.get(requirement.id) == (part, True) and not wanted.exact:
                # It cites the row: the plan holds its pin to the row's unit.
                assert choice.unit is not None
                wanted = _Wanted(choice.unit, exact=True)
            if unit is None:
                code, opening = ProblemCode.USER_DECISION_DROPPED, "states no `unit`"
            elif unit.prices != choice.prices:
                code = ProblemCode.USER_DECISION_CONTRADICTED
                opening = f"states `unit` in {unit.prices} prices"
            elif choice.held_currency not in (None, unit.currency):
                code = ProblemCode.USER_DECISION_CONTRADICTED
                opening = f"states `unit` in {unit.currency}"
            elif wanted.exact and unit != wanted.unit:
                code = ProblemCode.USER_DECISION_CONTRADICTED
                opening = f"states `unit` {unit.model_dump(mode='json', exclude_none=True)}"
            elif wanted.unit is None or wanted.pin is not None or wanted.unserved:
                code = ProblemCode.USER_DECISION_CONTRADICTED
                other = "another unit" if wanted.exact else "other prices"
                opening = f"pins {requirement.source_spec_id}, a series in {other}"
            else:
                continue
            return _refusal(
                code,
                f"The user chose {choice.words} for part {part} answering an ask, and "
                f"{requirement.id!r} {opening}.",
                slot=slot,
                part=part,
                reading=read.pin(requirement),
                choice=choice,
                requirement_id=requirement.id,
                wanted=wanted,
                propose=propose,
                outcome=Outcome.REFUSED,
            )
        if not reading:
            continue
        decisions.setdefault(
            slot,
            Decision(
                slot=slot,
                description="Which prices the figures are in.",
                options=tuple(
                    choice.words if label == choice.label else label
                    for label in (CURRENT_PRICES, CONSTANT_PRICES)
                ),
                chosen=choice.words,
                resolved_by=ResolvedBy.USER,
                rationale=f"The user chose {choice.words}.",
                fixes=DecisionFixes(prices=choice.prices),
            ),
        )
        for requirement in reading:
            if slot not in (*requirement.decision_ids, *slots.get(requirement.id, ())):
                slots.setdefault(requirement.id, []).append(slot)
    citing = {
        requirement.id: requirement.model_copy(
            update={"decision_ids": (*requirement.decision_ids, *slots[requirement.id])}
        )
        for requirement in requirements
        if requirement.id in slots
    }
    stated = {requirement.id for requirement in patch}
    return list(decisions.values()), [
        *(citing.get(requirement.id, requirement) for requirement in patch),
        *(requirement for rid, requirement in citing.items() if rid not in stated),
    ]


@dataclass(frozen=True)
class _Readings:
    """Which dataset requirements read each part `prices_decisions` prices.

    `pin` is the series a requirement reads: the pin it states, or a declared
    whole's part's (`DatasetRequirement.whole_of`), whose dataset admission
    compiles the whole from. `row_of` holds the unit row part each
    requirement no part reads by its series reads (`_row_readers`), and
    `by_part` every requirement each part reads, by its series or so.
    """

    pin: Callable[[DatasetRequirement], str | None]
    row_of: Mapping[str, tuple[str, bool]]
    by_part: Mapping[str, list[DatasetRequirement]]

    @classmethod
    def of(
        cls,
        choices: Mapping[str, PricesChoice],
        requirements: Sequence[EvidenceRequirement],
        *,
        parts: Parts,
    ) -> _Readings:
        """The readings of `requirements` by the parts `choices` prices, read
        through `parts` with those choices marked (`Parts.priced_by`)."""
        by_id = {requirement.id: requirement for requirement in requirements}

        def read_pin(requirement: DatasetRequirement) -> str | None:
            whole = (
                by_id.get(requirement.whole_of.part) if requirement.whole_of is not None else None
            )
            return (
                whole.source_spec_id
                if isinstance(whole, DatasetRequirement)
                else requirement.source_spec_id
            )

        datasets = [r for r in requirements if isinstance(r, DatasetRequirement)]
        series_of = {part: parts.series(part) for part in choices}
        row_of = _row_readers(
            datasets,
            choices,
            read_pin=read_pin,
            series_of=series_of,
            slot_of=decision_slots(choices),
            parts=parts,
        )
        return cls(
            pin=read_pin,
            row_of=row_of,
            by_part={
                part: [
                    requirement
                    for requirement in datasets
                    if read_pin(requirement) in series_of[part]
                    or row_of.get(requirement.id, ("",))[0] == part
                ]
                for part in choices
            },
        )


def part_readings(
    choices: Mapping[str, PricesChoice],
    requirements: Sequence[EvidenceRequirement],
    *,
    parts: Parts,
) -> dict[str, tuple[str, ...]]:
    """The id of each dataset requirement of `requirements` that reads each
    part `choices` prices, by the part, as `prices_decisions` reads them."""
    read = _Readings.of(choices, requirements, parts=parts.priced_by(choices))
    return {
        part: tuple(requirement.id for requirement in reading)
        for part, reading in read.by_part.items()
    }


def _row_readers(
    requirements: Sequence[DatasetRequirement],
    choices: Mapping[str, PricesChoice],
    *,
    read_pin: Callable[[DatasetRequirement], str | None],
    series_of: Mapping[str, Collection[str]],
    slot_of: Mapping[str, str],
    parts: Parts,
) -> dict[str, tuple[str, bool]]:
    """The part of a unit row (`PricesChoice.group`) each of `requirements`
    no part reads by its series (`series_of`) reads, by the requirement's id,
    with whether the requirement cites the row.

    One already citing the row's slot (`slot_of`) reads its first part,
    whatever series a count or scope switch of the plan card moved it to,
    and states the row's unit whole on that pin.
    One citing none reads the first part of a row whose series measures what
    its pin measures (`Parts.measures_alike`), such as the part pinned again
    in the unit it is served in, which `_whole_unit` moves to the series in
    the row's unit. A requirement reading no row is left out.
    """
    rows = {
        part: choice
        for part, choice in choices.items()
        if choice.group is not None and choice.unit is not None
    }
    readers: dict[str, tuple[str, bool]] = {}
    for requirement in requirements:
        pin = read_pin(requirement)
        if any(pin in series for series in series_of.values()):
            continue
        cited = next((part for part in rows if slot_of[part] in requirement.decision_ids), None)
        if cited is not None:
            readers[requirement.id] = (cited, True)
            continue
        part = next(
            (
                part
                for part, choice in rows.items()
                if pin is not None
                and choice.spec_id is not None
                and parts.measures_alike(pin, choice.spec_id)
            ),
            None,
        )
        if part is not None:
            readers[requirement.id] = (part, False)
    return readers


def base_year_refusal(
    exc: ValidationError,
    requirements: Sequence[Any],
    *,
    choices: Mapping[str, PricesChoice],
    parts: Parts,
    propose: str,
) -> dict[str, object] | None:
    """The refusal of a requirement reading a part the user priced whose `unit`
    states constant prices with no base year, or `None`.

    `requirements` are the requirements the proposal states, as the call
    passed them. The refusal names what `prices_decisions` would for the
    requirement (`_wanted`): the unit with its own pin's base year, the
    series to move to, or, with neither, the question for the user. A
    requirement reading no part the user priced answers `None`, so its
    proposal is refused as any invalid plan is.
    """
    parts = parts.priced_by(choices)
    for error in exc.errors(include_url=False, include_input=False):
        if not isinstance(error.get("ctx", {}).get("error"), BaseYearUnstatedError):
            continue
        loc = error["loc"]
        if len(loc) < 2 or loc[0] != "evidence_requirements" or not isinstance(loc[1], int):
            continue
        if loc[1] >= len(requirements) or not isinstance(raw := requirements[loc[1]], dict):
            continue
        pin = raw.get("source_spec_id")
        if not isinstance(pin, str):
            continue
        part = next((part for part in choices if pin in parts.series(part)), None)
        if part is None:
            continue
        choice = choices[part]
        unit = raw.get("unit")
        currency = unit.get("currency") if isinstance(unit, dict) else None
        return _refusal(
            ProblemCode.INVALID_PLAN,
            f"The user chose {choice.words} for part {part} answering an ask, and "
            f"{raw.get('id')!r} states constant prices with no `base_year`.",
            slot=decision_slots(choices)[part],
            part=part,
            reading=pin,
            choice=choice,
            requirement_id=raw.get("id"),
            wanted=_wanted(
                pin, currency if isinstance(currency, str) else None, part, choice, parts
            ),
            propose=propose,
            outcome=Outcome.INVALID_INPUT,
        )
    return None
