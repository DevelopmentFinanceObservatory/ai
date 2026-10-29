"""The prices the user chose for a part answering an ask, held on the plan as their decision.

A part is a search reading whose prices were open (`harness.turn_records.
open_prices`). The orchestrator's prices question names its part, and code
gives it the part's `open_prices` entries as options, each marked with the
part and its prices (`harness.graph.state.priced_options`). The planner
reads the user's picks off the answered interrupt, one per part
(`planner.agent._prices_chosen`).

`prices_decisions` writes each pick as a `resolved_by=user` decision fixing
the price basis alone (`DecisionFixes.prices`), cited by every dataset
requirement the planner pinned to one of the part's series (`Parts.series`),
so each keeps its own currency and base year. A pin alignment moved is read
as the planner wrote it (`Parts.origin_of`). It refuses a plan in which
one of those leaves its unit out, states other prices or pins a series in
other prices, naming the unit to state: the base year comes from the
requirement's own pin, or from the series it is told to move to. With no
series shown in constant prices the user kept, the planner's entry ends, so
the orchestrator can search for one. Once recorded, the decision stands as
every user decision does (`submission.service._user_decision_refused`).

`base_year_refusal` gives the same instruction for a unit in constant prices
with no base year on such a requirement. A requirement reading no part the
user priced is left as it was.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

from compiler.evidence import PriceBasis, UnitReading, read_unit_currency, read_unit_dcid
from compiler.evidence import requested_unit as requested_unit_of
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from harness.turn_records import CONSTANT_PRICES, CURRENT_PRICES
from planner.plan import (
    BaseYearUnstatedError,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    EvidenceRequirement,
    ResolvedBy,
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

# The fixed choices of the planner's ask a prices refusal sends it to, by
# the prices each picks. A pick sets the part's prices
# (`planner.agent._prices_chosen`).
PRICES_OPTIONS: tuple[dict[str, str], ...] = tuple(
    {"id": f"prices:{basis}", "label": f"{label[:1].upper()}{label[1:]}"}
    for basis, label in _LABELS.items()
)
PRICES_PICKS: dict[str, PriceBasis] = {f"prices:{basis}": basis for basis in _LABELS}


def prices_slot(part: str) -> str:
    """The `Decision.slot` of the prices the user chose for `part`."""
    return f"{PRICES_SLOT_PREFIX}{part}"


class PricesChoice(NamedTuple):
    """The prices the user picked for a part, the spec_id of the series they
    picked, when the option carried one, and whether they answered the
    planner's own prices question on the part (`asked`), which is not asked
    twice."""

    prices: PriceBasis
    spec_id: str | None = None
    asked: bool = False

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


@dataclass(frozen=True)
class Parts:
    """What the prices rules read of the datasets on record.

    `source_of` and `block_of` answer a spec_id's source and block, `None`
    when none is held. `shown` is every spec_id the planner was shown.
    `aligned` maps each pin alignment compiled to the pin it was compiled
    from (`submission.service.AlignedPin`).

    A part's series are the part, each series its block offers in its place,
    and each shown series measuring what it measures in its currency, in any
    prices (`same_measure`), such as one a later search found.
    """

    source_of: Callable[[str], SelectedSource | None]
    block_of: Callable[[str], Mapping[str, Any] | None]
    shown: Collection[str] = ()
    aligned: Mapping[str, str] | None = None

    def origin_of(self, pin: str | None) -> str | None:
        """The pin as the planner wrote it, before alignment moved it: each
        compiled pin is followed to the one it was compiled from until a pin
        alignment did not compile, so a whole compiled from an aligned part's
        pin reads as the part."""
        aligned = self.aligned or {}
        seen: set[str] = set()
        while pin is not None and pin in aligned and pin not in seen:
            seen.add(pin)
            pin = aligned[pin]
        return pin

    def unit_of(self, spec_id: str) -> UnitReading:
        """What the unit of the series `spec_id` states (`source_unit`), nothing when no source is held."""
        source = self.source_of(spec_id)
        return source_unit(source) if source is not None else UnitReading()

    def alternatives(self, part: str) -> list[Mapping[str, Any]]:
        """The series `part`'s block offers in its place, another count or other prices."""
        resolved = (self.block_of(part) or {}).get("resolved") or {}
        offered = resolved.get("alternative_measures") or ()
        return [alt for alt in offered if isinstance(alt, Mapping)]

    def series(self, part: str) -> frozenset[str]:
        """`part`, the spec_id of every series its block offers in its place, and
        every shown series measuring what it measures in its own currency (`same_measure`)."""
        offered = ((alt.get("definition") or {}).get("spec_id") for alt in self.alternatives(part))
        return frozenset(
            {
                part,
                *(spec_id for spec_id in offered if isinstance(spec_id, str)),
                *self.same_measure(part, self.unit_of(part).currency),
            }
        )

    def unit_offered(self, part: str, spec_id: str) -> bool:
        """Whether `part`'s block offers `spec_id` as its own count in other prices (axis `unit`)."""
        return any(
            (alt.get("definition") or {}).get("spec_id") == spec_id and alt.get("axis") == "unit"
            for alt in self.alternatives(part)
        )

    def same_measure(self, part: str, currency: str | None) -> list[str]:
        """Each shown series but `part` measuring what `part` measures, in
        `currency` and any prices: its identity equals the part's with the
        prices left out (`_priceless`), so provenance and measurement method
        match, and its unit states `currency`."""
        wanted = _priceless(self.block_of(part))
        if wanted is None:
            return []
        return [
            spec_id
            for spec_id in self.shown
            if spec_id != part
            and _priceless(self.block_of(spec_id)) == wanted
            and self.unit_of(spec_id).currency == currency
        ]

    def constant_found(self, part: str, currency: str | None) -> tuple[str, UnitReading] | None:
        """A shown series measuring what `part` measures in `currency` at constant
        prices of a stated base year (`same_measure`), such as one a later search
        found, with its unit; `None` when none is shown."""
        for spec_id in self.same_measure(part, currency):
            reading = self.unit_of(spec_id)
            if reading.prices == "constant" and reading.base_year is not None:
                return spec_id, reading
        return None


def _priceless(block: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A block's identity less its window, its unit facet and its measurement
    qualifier, which state its prices; `None` when it states none."""
    identity = ((block or {}).get("definition") or {}).get("identity")
    if not isinstance(identity, Mapping):
        return None
    kept = {k: v for k, v in identity.items() if k != "requested_window"}
    facets = kept.get("facets")
    if isinstance(facets, Mapping):
        kept["facets"] = {k: v for k, v in facets.items() if k != "unit"}
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
    `unit` is `None` when no series shown serves those prices."""

    unit: Unit | None
    pin: str | None = None


def _wanted(
    pin: str | None, currency: str | None, part: str, choice: PricesChoice, parts: Parts
) -> _Wanted:
    """The unit, and the series, a requirement pinning `pin` states in `choice`'s prices.

    The pin's own unit gives the currency, when the requirement states none,
    and the base year of constant prices. A requirement pinned to the part's
    own series moves, for constant prices, to the series the user picked, or
    else to a series a search found in those prices and that currency
    (`Parts.constant_found`), with that series' base year. One pinned to the
    part's count in constant prices, offered or found, moves back to the part
    for current prices.
    """
    own = parts.unit_of(pin) if pin is not None else UnitReading()
    currency = currency or own.currency
    if currency is None:
        return _Wanted(None)
    if choice.prices == "current":
        if own.prices != "constant":
            return _Wanted(Unit(currency=currency, prices="current"))
        moved = pin is not None and (
            parts.unit_offered(part, pin) or pin in parts.same_measure(part, currency)
        )
        return _Wanted(Unit(currency=currency, prices="current"), part) if moved else _Wanted(None)
    if own.prices == "constant" and own.base_year is not None:
        return _Wanted(Unit(currency=currency, prices="constant", base_year=own.base_year))
    found = _constant_series(part, choice, parts, currency) if pin == part else None
    if found is None:
        return _Wanted(None)
    return _Wanted(Unit(currency=currency, prices="constant", base_year=found[1]), found[0])


def _constant_series(
    part: str, choice: PricesChoice, parts: Parts, currency: str
) -> tuple[str, int] | None:
    """The series of `part` in constant prices to move to, with its base year:
    the one the user picked, else one a search found in `currency` (`Parts.constant_found`)."""
    reading = parts.unit_of(choice.spec_id) if choice.spec_id is not None else UnitReading()
    if (
        choice.spec_id is not None
        and reading.prices == "constant"
        and reading.base_year is not None
    ):
        return choice.spec_id, reading.base_year
    found = parts.constant_found(part, currency)
    if found is None or found[1].base_year is None:
        return None
    return found[0], found[1].base_year


def _refusal(
    code: ProblemCode,
    lead: str,
    *,
    part: str,
    choice: PricesChoice,
    requirement_id: object,
    wanted: _Wanted,
    propose: str,
    outcome: Literal[Outcome.REFUSED, Outcome.INVALID_INPUT],
) -> dict[str, object]:
    """The refusal naming what a requirement reading `part` states to honour `choice`.

    `detail` names the decision's `slot`, what the user `chose`, the `part`
    and the `requirement_id`, and the `unit` to state and the series to
    `pin`, when one is named. With no unit to state, no series shown serves
    the prices, and the planner asks the user which prices to go on in,
    offering `PRICES_OPTIONS`. Once the user has answered that question on
    the part and kept constant prices, only a search can find a series in
    them, and the planner has none: `PRICES_UNSERVED` ends the planner's
    entry, and the orchestrator reads why (`harness.sentences`).
    """
    detail: dict[str, Any] = {
        "slot": prices_slot(part),
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
            detail=detail,
        )
    if wanted.unit is None:
        return problem(
            code,
            f"{lead} No series shown serves it in {choice.label}. Ask the user through "
            f"`{REQUEST_INPUT_FUNCTION_CALL_NAME}`, with the action's `options`, which prices "
            "to go on in; once they pick, a patch states those.",
            outcome=outcome,
            action=Action(
                call=REQUEST_INPUT_FUNCTION_CALL_NAME,
                arguments={"options": [dict(option) for option in PRICES_OPTIONS]},
                needs=("message",),
            ),
            detail=detail,
        )
    detail["unit"] = wanted.unit.model_dump(mode="json", exclude_none=True)
    moved = ""
    if wanted.pin is not None:
        detail["pin"] = wanted.pin
        moved = f" and pin {wanted.pin}, the series of this part in {choice.label}"
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
    wrote (`Parts.origin_of`) is the part's series or one the part offers in
    its place (`Parts.series`). Each part that some requirement reads states
    one decision fixing `prices`, which each of those cites. A requirement
    citing one that `patch` does not state is appended, so `Plan.amend`
    replaces it. A part no requirement reads states nothing.

    The refusal is `USER_DECISION_DROPPED` when such a requirement states no
    unit, and `USER_DECISION_CONTRADICTED` when it states other prices or
    pins a series that cannot serve them (`_refusal`, `_wanted`). `propose`
    is the tool a refusal naming a unit calls.
    """
    decisions: list[Decision] = []
    slots: dict[str, list[str]] = {}
    for part, choice in choices.items():
        series = parts.series(part)
        reading = [
            requirement
            for requirement in requirements
            if isinstance(requirement, DatasetRequirement)
            and parts.origin_of(requirement.source_spec_id) in series
        ]
        for requirement in reading:
            unit = requirement.unit
            wanted = _wanted(
                parts.origin_of(requirement.source_spec_id),
                unit.currency if unit is not None else None,
                part,
                choice,
                parts,
            )
            if unit is None:
                code, opening = ProblemCode.USER_DECISION_DROPPED, "states no `unit`"
            elif unit.prices != choice.prices:
                code = ProblemCode.USER_DECISION_CONTRADICTED
                opening = f"states `unit` in {unit.prices} prices"
            elif wanted.unit is None or wanted.pin is not None:
                code = ProblemCode.USER_DECISION_CONTRADICTED
                opening = f"pins {requirement.source_spec_id}, a series in other prices"
            else:
                continue
            return _refusal(
                code,
                f"The user chose {choice.label} for part {part} answering an ask, and "
                f"{requirement.id!r} {opening}.",
                part=part,
                choice=choice,
                requirement_id=requirement.id,
                wanted=wanted,
                propose=propose,
                outcome=Outcome.REFUSED,
            )
        if not reading:
            continue
        slot = prices_slot(part)
        decisions.append(
            Decision(
                slot=slot,
                description="Which prices the figures are in.",
                options=(CURRENT_PRICES, CONSTANT_PRICES),
                chosen=choice.label,
                resolved_by=ResolvedBy.USER,
                rationale=f"The user chose {choice.label}.",
                fixes=DecisionFixes(prices=choice.prices),
            )
        )
        for requirement in reading:
            if slot not in requirement.decision_ids:
                slots.setdefault(requirement.id, []).append(slot)
    citing = {
        requirement.id: requirement.model_copy(
            update={"decision_ids": (*requirement.decision_ids, *slots[requirement.id])}
        )
        for requirement in requirements
        if requirement.id in slots
    }
    stated = {requirement.id for requirement in patch}
    return decisions, [
        *(citing.get(requirement.id, requirement) for requirement in patch),
        *(requirement for rid, requirement in citing.items() if rid not in stated),
    ]


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
            f"The user chose {choice.label} for part {part} answering an ask, and "
            f"{raw.get('id')!r} states constant prices with no `base_year`.",
            part=part,
            choice=choice,
            requirement_id=raw.get("id"),
            wanted=_wanted(
                pin, currency if isinstance(currency, str) else None, part, choice, parts
            ),
            propose=propose,
            outcome=Outcome.INVALID_INPUT,
        )
    return None
