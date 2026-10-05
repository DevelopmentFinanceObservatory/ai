"""Tests for `planner.tool_results`: the one result shape every tool answers with."""

from __future__ import annotations

import pytest
from planner.tool_results import Action, Outcome, ProblemCode, ok, problem, referred


def test_ok_sets_the_outcome_and_carries_the_fields() -> None:
    assert ok(nodes=[], budget_remaining=3) == {"outcome": "ok", "nodes": [], "budget_remaining": 3}


def test_ok_refuses_a_caller_that_passes_its_own_outcome() -> None:
    # Every caller is code in this repository, so an override is a programmer
    # error, not
    # something to paper over.
    with pytest.raises(TypeError, match="outcome"):
        ok(outcome="refused")


def test_problem_and_referred_set_their_outcomes() -> None:
    refused = problem(ProblemCode.BUDGET_EXHAUSTED, "spent", outcome=Outcome.REFUSED)

    assert (refused["outcome"], refused["code"]) == ("refused", "budget_exhausted")
    assert "action" not in refused
    assert referred({"kind": "no_data"})["outcome"] == "referred"


def test_an_actions_needs_round_trips_when_set() -> None:
    action = Action(
        call="return_to_planner",
        arguments={"kind": "no_operation_covers", "reason": "r"},
        needs=("decision_id",),
    )

    dumped = action.model_dump(mode="json", exclude_none=True)

    assert dumped["needs"] == ["decision_id"]


def test_an_actions_needs_is_omitted_when_not_set() -> None:
    # Scoped to `Action` alone (its own `model_serializer`), not to
    # `exclude_defaults` on the enclosing dump: plain `exclude_none=True` is
    # enough here, same as everywhere else in this module.
    action = Action(call="discover_source", arguments={"requirement_id": "x"})

    dumped = action.model_dump(mode="json", exclude_none=True)

    assert "needs" not in dumped


def test_a_problem_carrying_an_action_with_needs_surfaces_it() -> None:
    action = Action(
        call="return_to_planner",
        arguments={"kind": "no_operation_covers"},
        needs=("decision_id",),
    )

    result = problem(ProblemCode.COMPILE_FAILED, "m", outcome=Outcome.INVALID_INPUT, action=action)

    assert result["action"]["needs"] == ["decision_id"]
