"""The prices the user chose answering an ask, held on the plan as their decision.

The orchestrator's prices question offers each `open_prices` entry with its
`prices` (`harness.turn_records.open_prices`), and the planner reads the
user's pick off the answered interrupt (`planner.agent._prices_chosen`).
`prices_decision` writes that pick as a `resolved_by=user` decision fixing
the unit of every dataset requirement that measures money, and refuses a
plan leaving a unit out or stating it in other prices. Once recorded, the
decision stands as every user decision does
(`submission.service._user_decision_refused`).

`base_year_refusal` answers a unit stating constant prices with no base
year with the base year the requirement's source states, so the planner
restates the unit and keeps it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

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

# The slot `prices_decision` records the user's prices under.
PRICES_SLOT = "user_prices"

_LABELS: dict[PriceBasis, str] = {"current": CURRENT_PRICES, "constant": CONSTANT_PRICES}
_SPEC_ID = re.compile(r"spec_[0-9a-f]{32}")

# The source a pin of a spec_id states, `None` when none is on record.
SourceOf = Callable[[str], SelectedSource | None]


class PricesChoice(NamedTuple):
    """The prices the user picked answering an ask, and the spec_id of the
    series they picked, when the option carried one."""

    prices: PriceBasis
    spec_id: str | None = None

    @classmethod
    def of_candidate(cls, candidate: dict[str, Any]) -> PricesChoice | None:
        """The choice a picked clarify candidate states, `None` when it names no prices.

        A candidate's id is its option's spec_id, or `o{n}` for an option
        carrying none (`harness.graph.state.ask_candidates`).
        """
        prices = candidate.get("prices")
        if prices not in _LABELS:
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
    else its unit id's structure, with the currency its label names when neither does."""
    requested = requested_unit_of(source.identity)
    if requested is not None:
        return UnitReading(requested.currency, requested.prices, requested.base_year)
    reading = read_unit_dcid(source.unit_ref)
    if reading.currency is not None:
        return reading
    currency = read_unit_currency(source.unit_ref, source.unit_label)
    return reading._replace(currency=currency)


def _base_year(
    pin: str | None, choice: PricesChoice | None, source_of: SourceOf
) -> tuple[int, str] | None:
    """The base year of constant prices the pinned source states, else the series
    the user picked, with the spec_id stating it; `None` when neither does."""
    for spec_id in (pin, choice.spec_id if choice is not None else None):
        if spec_id is None or (source := source_of(spec_id)) is None:
            continue
        reading = source_unit(source)
        if reading.prices == "constant" and reading.base_year is not None:
            return reading.base_year, spec_id
    return None


def _currency(requirement: DatasetRequirement, source_of: SourceOf) -> str | None:
    """The currency `requirement` states, else the one its pinned source states."""
    if requirement.unit is not None:
        return requirement.unit.currency
    pin = requirement.source_spec_id
    source = source_of(pin) if pin is not None else None
    return source_unit(source).currency if source is not None else None


def prices_decision(
    choice: PricesChoice | None,
    patch: Sequence[EvidenceRequirement],
    requirements: Sequence[EvidenceRequirement],
    *,
    source_of: SourceOf,
    propose: str,
) -> tuple[Decision, list[EvidenceRequirement]] | dict[str, object] | None:
    """The user's prices `choice` as their decision, with `patch` citing it, or the refusal of the plan.

    `requirements` is every requirement of the patched plan, `patch` this
    call's own. Every dataset requirement stating a unit or pinning a source
    in a currency measures money, and states `unit` in the prices the user
    chose; the decision fixes the first one's unit, and each cites it. A
    requirement citing it that `patch` does not state is appended, so
    `Plan.amend` replaces it. `None` when the user chose no prices.

    The refusal is `USER_DECISION_DROPPED` when such a requirement states no
    unit or none measures money, and `USER_DECISION_CONTRADICTED` when one
    states other prices. It names the unit to state, with the base year of
    constant prices the pinned source or the series the user picked states.
    When neither states one the plan cannot serve constant prices, and the
    refusal sends the planner to ask the user. `propose` is the tool a
    refusal that names the unit calls.
    """
    if choice is None:
        return None
    money = [
        requirement
        for requirement in requirements
        if isinstance(requirement, DatasetRequirement)
        and _currency(requirement, source_of) is not None
    ]
    if not money:
        return problem(
            ProblemCode.USER_DECISION_DROPPED,
            f"The user chose {choice.label} answering an ask, and no requirement states a "
            "`unit` or pins a source in a currency. State `unit` in those prices on the "
            "dataset requirement they apply to.",
            outcome=Outcome.REFUSED,
            action=Action(call=propose, needs=("evidence_requirements",)),
            detail={"slot": PRICES_SLOT, "chosen": choice.label, "requirement_id": None},
        )
    for requirement in money:
        if requirement.unit is None or requirement.unit.prices != choice.prices:
            return _unit_refused(choice, requirement, source_of=source_of, propose=propose)
    decision = Decision(
        slot=PRICES_SLOT,
        description="Which prices the figures are in.",
        options=(CURRENT_PRICES, CONSTANT_PRICES),
        chosen=choice.label,
        resolved_by=ResolvedBy.USER,
        rationale=f"The user chose {choice.label}.",
        fixes=DecisionFixes(unit=money[0].unit),
    )
    citing = {
        requirement.id: requirement.model_copy(
            update={"decision_ids": (*requirement.decision_ids, PRICES_SLOT)}
        )
        for requirement in money
        if PRICES_SLOT not in requirement.decision_ids
    }
    stated = {requirement.id for requirement in patch}
    return decision, [
        *(citing.get(requirement.id, requirement) for requirement in patch),
        *(requirement for rid, requirement in citing.items() if rid not in stated),
    ]


def _unit_refused(
    choice: PricesChoice,
    requirement: DatasetRequirement,
    *,
    source_of: SourceOf,
    propose: str,
) -> dict[str, object]:
    """The refusal of `requirement`, which measures money, for stating no unit or one in other prices."""
    stated = requirement.unit
    code, opening = (
        (ProblemCode.USER_DECISION_DROPPED, f"{requirement.id!r} states no `unit`")
        if stated is None
        else (
            ProblemCode.USER_DECISION_CONTRADICTED,
            f"{requirement.id!r} states `unit` in {stated.prices} prices",
        )
    )
    lead = f"The user chose {choice.label} answering an ask, and {opening}."
    detail: dict[str, Any] = {
        "slot": PRICES_SLOT,
        "chosen": choice.label,
        "requirement_id": requirement.id,
    }
    currency = _currency(requirement, source_of)
    based = _base_year(requirement.source_spec_id, choice, source_of)
    if choice.prices == "constant" and based is None:
        detail["base_year"] = None
        return problem(
            code,
            f"{lead} No source shown states the base year of these figures in constant "
            "prices, so the plan cannot serve them. Ask the user through "
            f"`{REQUEST_INPUT_FUNCTION_CALL_NAME}` whether to go on in current prices; once "
            "they answer, a patch may state those.",
            outcome=Outcome.REFUSED,
            action=Action(call=REQUEST_INPUT_FUNCTION_CALL_NAME),
            detail=detail,
        )
    base_year = based[0] if choice.prices == "constant" and based is not None else None
    if base_year is not None:
        detail["base_year"] = base_year
    if currency is None:
        wanted = f"in {choice.label}" + (f" of {base_year}" if base_year is not None else "")
        return problem(
            code,
            f"{lead} State its `unit` {wanted}.",
            outcome=Outcome.REFUSED,
            action=Action(call=propose, needs=("evidence_requirements",)),
            detail=detail,
        )
    unit = Unit(currency=currency, prices=choice.prices, base_year=base_year)
    detail["unit"] = unit.model_dump(mode="json", exclude_none=True)
    repin = (
        f", and pin {based[1]}, the series in those prices"
        if based is not None and based[1] != requirement.source_spec_id
        else ""
    )
    return problem(
        code,
        f"{lead} State `unit` {detail['unit']} on it{repin}.",
        outcome=Outcome.REFUSED,
        action=Action(call=propose, needs=("evidence_requirements",)),
        detail=detail,
    )


def base_year_refusal(
    exc: ValidationError,
    requirements: Sequence[Any],
    *,
    choice: PricesChoice | None,
    source_of: SourceOf,
    propose: str,
) -> dict[str, object] | None:
    """The refusal of a requirement whose `unit` states constant prices with no base year, or `None`.

    `requirements` are the requirements the proposal states, as the call
    passed them. The refusal names the base year the requirement's pinned
    source states, else the one of the series the user picked
    (`PricesChoice.spec_id`), and the unit to state. With neither, constant
    prices cannot be served: a user who chose them is asked whether to go
    on in current prices, and a planner who chose them states current prices.
    """
    for error in exc.errors(include_url=False, include_input=False):
        if not isinstance(error.get("ctx", {}).get("error"), BaseYearUnstatedError):
            continue
        loc = error["loc"]
        if len(loc) < 2 or loc[0] != "evidence_requirements" or not isinstance(loc[1], int):
            continue
        if loc[1] >= len(requirements) or not isinstance(raw := requirements[loc[1]], dict):
            continue
        return _base_year_unstated(raw, choice=choice, source_of=source_of, propose=propose)
    return None


def _base_year_unstated(
    raw: dict[str, Any], *, choice: PricesChoice | None, source_of: SourceOf, propose: str
) -> dict[str, object]:
    """`base_year_refusal`'s answer for `raw`, the requirement as the call stated it."""
    requirement_id = raw.get("id")
    pin = raw.get("source_spec_id")
    unit = raw.get("unit")
    currency = unit.get("currency") if isinstance(unit, dict) else None
    lead = f"{requirement_id!r} states constant prices with no `base_year`."
    detail: dict[str, Any] = {"requirement_id": requirement_id}
    based = _base_year(pin if isinstance(pin, str) else None, choice, source_of)
    if based is None:
        detail["base_year"] = None
        if choice is not None and choice.prices == "constant":
            return problem(
                ProblemCode.INVALID_PLAN,
                f"{lead} No source shown states the base year of these figures in constant "
                "prices, so the plan cannot serve the constant prices the user chose. Ask "
                f"them through `{REQUEST_INPUT_FUNCTION_CALL_NAME}` whether to go on in "
                "current prices.",
                outcome=Outcome.INVALID_INPUT,
                action=Action(call=REQUEST_INPUT_FUNCTION_CALL_NAME),
                detail=detail,
            )
        return problem(
            ProblemCode.INVALID_PLAN,
            f"{lead} No source shown states the base year of these figures in constant "
            "prices: state `unit` in current prices.",
            outcome=Outcome.INVALID_INPUT,
            action=Action(call=propose, needs=("evidence_requirements",)),
            detail=detail,
        )
    base_year, spec_id = based
    detail["base_year"] = base_year
    stated = (
        {"currency": currency, "prices": "constant", "base_year": base_year}
        if isinstance(currency, str)
        else {"prices": "constant", "base_year": base_year}
    )
    detail["unit"] = stated
    return problem(
        ProblemCode.INVALID_PLAN,
        f"{lead} {spec_id} states constant {base_year} prices: state `unit` {stated}.",
        outcome=Outcome.INVALID_INPUT,
        action=Action(call=propose, needs=("evidence_requirements",)),
        detail=detail,
    )
