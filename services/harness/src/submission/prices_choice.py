"""The prices the user chose for a part answering an ask, held on the plan as their decision.

A part is a search reading whose prices were open (`harness.turn_records.
open_prices`). The `ask` node marks each option of a prices question with
the part it prices and the prices it offers, read in code off the turn's
search results (`harness.graph.orchestrate.priced_options`), and the planner
reads the user's picks off the answered interrupt, one per part
(`planner.agent._prices_chosen`).

`prices_decisions` writes each pick as a `resolved_by=user` decision fixing
the price basis alone (`DecisionFixes.prices`), cited by every dataset
requirement pinning the part's series or a series the part offers, so each
keeps its own currency and base year. It refuses a plan in which one of
those leaves its unit out or states other prices, naming the unit to state:
the base year comes from the requirement's own pin, or from the series the
user picked once the requirement is moved to it. Once recorded, the decision
stands as every user decision does (`submission.service._user_decision_refused`).

`base_year_refusal` gives the same instruction for a unit in constant prices
with no base year on such a requirement. A requirement reading no part the
user priced is left as it was.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
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
_BASES: dict[str, PriceBasis] = {"Nominal": "current", "RealValue": "constant"}
_SPEC_ID = re.compile(r"spec_[0-9a-f]{32}")

# What the decision for a part's prices is recorded under, the part's spec_id after it.
PRICES_SLOT_PREFIX = "user_prices_"

# The source a spec_id's block states, and the block itself; `None` when none is on record.
SourceOf = Callable[[str], SelectedSource | None]
BlockOf = Callable[[str], Mapping[str, Any] | None]


def prices_slot(part: str) -> str:
    """The `Decision.slot` of the prices the user chose for `part`."""
    return f"{PRICES_SLOT_PREFIX}{part}"


class PricesChoice(NamedTuple):
    """The prices the user picked for a part, and the spec_id of the series
    they picked, when the option carried one."""

    prices: PriceBasis
    spec_id: str | None = None

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
            prices=_BASES.get(qualifier) if isinstance(qualifier, str) else None
        )
    if reading.currency is None:
        reading = reading._replace(currency=read_unit_currency(source.unit_ref, source.unit_label))
    return reading


def _alternatives(part: str, block_of: BlockOf) -> list[Mapping[str, Any]]:
    """The series `part`'s block offers in its place, another count or other prices."""
    resolved = (block_of(part) or {}).get("resolved") or {}
    return [alt for alt in resolved.get("alternative_measures") or () if isinstance(alt, Mapping)]


def _series(part: str, block_of: BlockOf) -> frozenset[str]:
    """`part` and the spec_id of every series its block offers in its place."""
    offered = (
        (alt.get("definition") or {}).get("spec_id") for alt in _alternatives(part, block_of)
    )
    return frozenset({part, *(spec_id for spec_id in offered if isinstance(spec_id, str))})


def _unit_offered(part: str, spec_id: str, block_of: BlockOf) -> bool:
    """Whether `part`'s block offers `spec_id` as its own count in other prices (axis `unit`)."""
    return any(
        (alt.get("definition") or {}).get("spec_id") == spec_id and alt.get("axis") == "unit"
        for alt in _alternatives(part, block_of)
    )


class _Wanted(NamedTuple):
    """What a requirement reading a priced part states to honour the user's
    choice: `unit`, and `pin` when it moves to another series of the part;
    `unit` is `None` when no series shown serves those prices."""

    unit: Unit | None
    pin: str | None = None


def _wanted(
    pin: str | None,
    currency: str | None,
    part: str,
    choice: PricesChoice,
    *,
    source_of: SourceOf,
    block_of: BlockOf,
) -> _Wanted:
    """The unit, and the series, a requirement pinning `pin` states in `choice`'s prices.

    The pin's own unit gives the currency, when the requirement states none,
    and the base year of constant prices. A requirement pinned to the part's
    own series moves, for constant prices, to the series the user picked,
    with that series' base year. One pinned to the part's count in constant
    prices moves back to the part for current prices.
    """
    source = source_of(pin) if pin is not None else None
    own = source_unit(source) if source is not None else UnitReading()
    currency = currency or own.currency
    if currency is None:
        return _Wanted(None)
    if choice.prices == "current":
        if own.prices != "constant":
            return _Wanted(Unit(currency=currency, prices="current"))
        moved = pin is not None and _unit_offered(part, pin, block_of)
        return _Wanted(Unit(currency=currency, prices="current"), part) if moved else _Wanted(None)
    if own.prices == "constant" and own.base_year is not None:
        return _Wanted(Unit(currency=currency, prices="constant", base_year=own.base_year))
    picked = source_of(choice.spec_id) if pin == part and choice.spec_id is not None else None
    reading = source_unit(picked) if picked is not None else UnitReading()
    if reading.prices != "constant" or reading.base_year is None:
        return _Wanted(None)
    return _Wanted(
        Unit(currency=currency, prices="constant", base_year=reading.base_year), choice.spec_id
    )


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
    the prices, and the planner asks the user whether to go on in the other.
    """
    detail: dict[str, Any] = {
        "slot": prices_slot(part),
        "chosen": choice.label,
        "part": part,
        "requirement_id": requirement_id,
    }
    if wanted.unit is None:
        other = _LABELS["current" if choice.prices == "constant" else "constant"]
        return problem(
            code,
            f"{lead} No series shown serves it in {choice.label}. Ask the user through "
            f"`{REQUEST_INPUT_FUNCTION_CALL_NAME}` whether to go on in {other}; once they "
            "answer, a patch may state those.",
            outcome=outcome,
            action=Action(call=REQUEST_INPUT_FUNCTION_CALL_NAME),
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
    source_of: SourceOf,
    block_of: BlockOf,
    propose: str,
) -> tuple[list[Decision], list[EvidenceRequirement]] | dict[str, object]:
    """The user's prices `choices`, by part, as their decisions, with `patch` citing them, or the refusal of the plan.

    `requirements` is every requirement of the patched plan, `patch` this
    call's own. A dataset requirement reads a part when it pins the part's
    series or one the part offers in its place (`_series`). Each part that
    some requirement reads states one decision fixing `prices`, which each
    of those cites. A requirement citing one that `patch` does not state is
    appended, so `Plan.amend` replaces it. A part no requirement reads
    states nothing.

    The refusal is `USER_DECISION_DROPPED` when such a requirement states no
    unit, and `USER_DECISION_CONTRADICTED` when it states other prices or
    pins a series that cannot serve them (`_refusal`, `_wanted`). `propose` is the tool a refusal naming a unit calls.
    """
    decisions: list[Decision] = []
    slots: dict[str, list[str]] = {}
    for part, choice in choices.items():
        series = _series(part, block_of)
        reading = [
            requirement
            for requirement in requirements
            if isinstance(requirement, DatasetRequirement) and requirement.source_spec_id in series
        ]
        for requirement in reading:
            unit = requirement.unit
            wanted = _wanted(
                requirement.source_spec_id,
                unit.currency if unit is not None else None,
                part,
                choice,
                source_of=source_of,
                block_of=block_of,
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
    source_of: SourceOf,
    block_of: BlockOf,
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
        part = next((part for part in choices if pin in _series(part, block_of)), None)
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
                pin,
                currency if isinstance(currency, str) else None,
                part,
                choice,
                source_of=source_of,
                block_of=block_of,
            ),
            propose=propose,
            outcome=Outcome.INVALID_INPUT,
        )
    return None
