"""`skills_loaded_before_plan`: a custom ADK eval metric reading the recorded tool calls.

A case states the skills a turn needs as `get_skill` entries in the turn's
`intermediate_data.tool_uses`, one per skill id, and the skills it forbids in
`cases.CASES` (an `Invocation` has no field to carry them, so the metric finds
the case by its brief). A turn passes when the agent called `propose_plan`
and, before its first `propose_plan` call, called `get_skill` with every
required skill id and with no forbidden one. A named skill in neither set
may be loaded or skipped, and a load after the first `propose_plan` is not
judged: the plan was drafted without it.

ADK's trajectory metric cannot make this assertion. It compares a call's
arguments exactly, and `propose_plan`'s arguments are the model's own drafted
decisions, so it can state the `get_skill` calls but not that they come
before the plan. This metric reads both from the same ordered call list
(`get_all_tool_calls`, which reads a live run's `invocation_events` in event
order) and needs no judge model.

Registered in `test_config.json` as `harness_planner_skills_v1.skill_metric.
skills_loaded_before_plan`; this package's `__init__` puts `evals/agents` on
`sys.path` so that dotted path resolves under `adk eval`.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

from google.adk.evaluation.eval_case import Invocation, get_all_tool_calls
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.evaluator import EvalStatus, EvaluationResult, PerInvocationResult
from google.genai import types as genai_types

from harness_planner_skills_v1.cases import case_for

GET_SKILL = "get_skill"
PROPOSE_PLAN = "propose_plan"


def required_skills(expected: Invocation) -> set[str]:
    """The skill ids `expected` names in its `get_skill` tool uses."""
    return {
        str((call.args or {}).get("skill_id"))
        for call in get_all_tool_calls(expected.intermediate_data)
        if call.name == GET_SKILL
    }


def skills_loaded_before_plan_of(calls: Sequence[genai_types.FunctionCall]) -> set[str] | None:
    """The skill ids `get_skill` loaded before the first `propose_plan`, or `None` with no plan."""
    names = [call.name for call in calls]
    if PROPOSE_PLAN not in names:
        return None
    first_plan = names.index(PROPOSE_PLAN)
    return {
        str((call.args or {}).get("skill_id"))
        for call in calls[:first_plan]
        if call.name == GET_SKILL
    }


def turn_score(actual: Invocation, expected: Invocation, *, forbidden: Collection[str]) -> float:
    """1.0 when the first `propose_plan` follows every required load and no forbidden one."""
    loaded = skills_loaded_before_plan_of(get_all_tool_calls(actual.intermediate_data))
    if loaded is None:
        return 0.0
    return 1.0 if required_skills(expected) <= loaded and loaded.isdisjoint(forbidden) else 0.0


def skills_loaded_before_plan(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: list[Invocation] | None,
    conversation_scenario: Any = None,
) -> EvaluationResult:
    """Score each turn with `turn_score`; the case passes only when every turn does.

    Args:
        eval_metric: This metric's configuration. Unused: the score is a
            pass-or-fail gate with no threshold to tune.
        actual_invocations: The turns the agent produced.
        expected_invocations: The case's own turns, naming the required
            skills in `intermediate_data.tool_uses`; each is matched to its
            `cases.CASES` entry for the forbidden skills.
        conversation_scenario: Unused; the set's conversations are static.

    Returns:
        `overall_score` 1.0 when every turn scored 1.0, else 0.0.

    Raises:
        ValueError: If `expected_invocations` is `None`.
        LookupError: If a turn's user message is no case's brief.
    """
    if expected_invocations is None:
        raise ValueError("skills_loaded_before_plan needs the case's expected invocations.")
    per_invocation_results = []
    for actual, expected in zip(actual_invocations, expected_invocations, strict=True):
        score = turn_score(actual, expected, forbidden=case_for(expected).forbidden)
        per_invocation_results.append(
            PerInvocationResult(
                actual_invocation=actual,
                expected_invocation=expected,
                score=score,
                eval_status=EvalStatus.PASSED if score == 1.0 else EvalStatus.FAILED,
            )
        )
    passed = all(result.score == 1.0 for result in per_invocation_results)
    return EvaluationResult(
        overall_score=1.0 if passed else 0.0,
        overall_eval_status=EvalStatus.PASSED if passed else EvalStatus.FAILED,
        per_invocation_results=per_invocation_results,
    )
