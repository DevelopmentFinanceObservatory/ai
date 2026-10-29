"""The option a user picks for a planner decision on the plan card.

The card offers the options of a planner decision it can switch
(`harness.plan_display.switchable_decision`) under the row id
`"decision/<slot>"`. A `switch` naming such a row resumes the paused
`approve` node, which replans with the pick held: `decision_pick` reads the
pick off the plan the card showed, as the planner states it.
"""

from __future__ import annotations

from collections.abc import Mapping

from planner.plan import DecisionPick, Plan, authored, slot_reading

from harness.plan_display import (
    DECISION_CHOICE_PREFIX,
    decision_option_id,
    switchable_decision,
)


def is_decision_choice(choice_id: str) -> bool:
    """Whether `choice_id` names a planner decision's row."""
    return choice_id.startswith(DECISION_CHOICE_PREFIX)


def decision_pick(
    plan: Plan, *, choice_id: str, option_id: str, to_origin: Mapping[str, str]
) -> DecisionPick | None:
    """The pick a switch of row `choice_id` to option `option_id` makes on `plan`.

    The pick records the decision's `chosen` as `plan` states it, and what
    `plan`'s requirements citing it read as the model wrote them
    (`planner.plan.slot_reading` over `planner.plan.authored`, with
    `to_origin` mapping each windowed copy's spec_id to the pin it copies):
    the origin every proposal of the replan is held against. `None` when
    `choice_id` names no switchable planner decision of `plan`, or
    `option_id` none of its options.
    """
    if not is_decision_choice(choice_id):
        return None
    slot = choice_id.removeprefix(DECISION_CHOICE_PREFIX)
    decision = next((d for d in plan.decisions if d.slot == slot), None)
    if decision is None or decision.fixes is None or not switchable_decision(decision):
        return None
    return next(
        (
            DecisionPick(
                slot=slot,
                chosen=option,
                origin_chosen=decision.chosen,
                origin_reading=slot_reading(
                    authored(plan).evidence_requirements, slot, to_origin=to_origin
                ),
            )
            for option in decision.options
            if decision_option_id(option) == option_id
        ),
        None,
    )
