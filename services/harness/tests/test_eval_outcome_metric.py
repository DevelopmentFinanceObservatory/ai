"""Unit tests for `harness_translator_v1.outcome_metric.workflow_outcome`.

Fabricated `Invocation`s, model-free: this pins the metric's own matching
logic (subset comparison, `sources` in any order, exclusivity between a
proposal and a referral and between a rejection and a proposal that
declared its plan ready, "last response with the stated outcome wins") independently
of whether the eval sets that configure it are internally consistent -- that
is `test_eval_sets.py`'s job, and `test_eval_outcomes.py` proves each case's
expected outcome is reachable.

`evals/agents` is not on `sys.path` for a normal `pytest` run -- `adk eval`
and this suite's other eval-adjacent tests insert it themselves, and so does
this module, since it imports `harness_translator_v1.outcome_metric` by the
same dotted path `_get_metric_function` resolves it by.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_AGENTS_DIR = REPO_ROOT / "evals" / "agents"
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

from google.adk.evaluation.eval_case import IntermediateData, Invocation  # noqa: E402
from google.adk.evaluation.eval_metrics import BaseCriterion, EvalMetric  # noqa: E402
from google.adk.evaluation.evaluator import EvalStatus  # noqa: E402
from google.genai import types as genai_types  # noqa: E402
from harness_translator_v1.outcome_metric import matches_subset, workflow_outcome  # noqa: E402

DEBT_SPEC = "spec_" + "d" * 32
OTHER_SPEC = "spec_" + "e" * 32

_USER_CONTENT = genai_types.Content(role="user", parts=[genai_types.Part.from_text(text="go")])


def _metric() -> EvalMetric:
    return EvalMetric(
        metric_name="workflow_outcome",
        threshold=None,
        criterion=BaseCriterion(threshold=1.0),
    )


def _invocation(*responses: genai_types.FunctionResponse) -> Invocation:
    return Invocation(
        invocation_id="turn",
        user_content=_USER_CONTENT,
        intermediate_data=IntermediateData(tool_responses=list(responses)),
    )


def _ok_proposal(
    steps: list[dict[str, str]], sources: list[dict[str, str]]
) -> genai_types.FunctionResponse:
    """The `translation` a case expects, or one the planner's proposal came back with."""
    return genai_types.FunctionResponse(
        name="translation",
        response={"outcome": "ok", "steps": steps, "sources": sources},
    )


def _referral(kind: str, requirement_id: str) -> genai_types.FunctionResponse:
    outcome = "choice_required" if kind == "source_ambiguous" else "invalid_input"
    return genai_types.FunctionResponse(
        name="translation",
        response={"outcome": outcome, "kind": kind, "requirement_id": requirement_id},
    )


def _carried(translation: genai_types.FunctionResponse) -> genai_types.FunctionResponse:
    """The accepted `propose_plan` response a run records, carrying `translation`'s verdict.

    A case states its expectation as a `translation` response; a run has no
    such response, only the verdict under an accepted proposal's own key.
    """
    return genai_types.FunctionResponse(
        name="propose_plan",
        response={
            "outcome": "ok",
            "status": "draft",
            "version": 1,
            "translation": translation.response,
        },
    )


def _plan_proposal(**fields: object) -> genai_types.FunctionResponse:
    return genai_types.FunctionResponse(
        name="propose_plan", response={"outcome": "ok", "status": "draft", **fields}
    )


def _compiled_proposal(**fields: object) -> genai_types.FunctionResponse:
    """A `propose_plan` whose translation compiled, with what
    `planner.agent.declare_ready` added: `summary` or a blocked `submission`."""
    return _plan_proposal(version=1, translation={"outcome": "ok"}, **fields)


_COMPILE_FAILED = genai_types.FunctionResponse(
    name="translation", response={"outcome": "invalid_input", "code": "compile_failed"}
)
_EXPECTED_STEPS = [{"tool": "data-mcp.get_data"}, {"tool": "analysis-mcp.aggregate"}]
_EXPECTED_SOURCES = [{"spec_id": DEBT_SPEC}]
_BLOCKED_SUBMISSION = {"outcome": "refused", "code": "plan_rejected"}
_SUMMARY = {"version": 1, "revision": "r1", "warnings": []}


def test_matches_subset_ignores_keys_and_extra_fields_the_expectation_does_not_name() -> None:
    expected = {"outcome": "ok", "steps": [{"tool": "data-mcp.get_data"}]}
    actual = {
        "outcome": "ok",
        "version": 1,
        "steps": [{"id": "op-1", "tool": "data-mcp.get_data"}],
    }

    assert matches_subset(expected, actual)


def test_matches_subset_rejects_a_list_of_a_different_length() -> None:
    expected = {"steps": [{"tool": "data-mcp.get_data"}, {"tool": "analysis-mcp.aggregate"}]}
    actual = {"steps": [{"tool": "data-mcp.get_data"}]}

    assert not matches_subset(expected, actual)


def test_matches_subset_rejects_a_differing_value() -> None:
    assert not matches_subset(
        {"sources": [{"spec_id": DEBT_SPEC}]}, {"sources": [{"spec_id": OTHER_SPEC}]}
    )


def test_matches_subset_compares_sources_in_any_order_but_steps_in_order() -> None:
    # The compiler emits sources in a canonical order of its own; a case
    # states which were selected, not how the compiler sorts them. Steps
    # are the tool sequence under test, so their order still counts.
    expected = {
        "steps": [{"tool": "data-mcp.get_data"}, {"tool": "analysis-mcp.aggregate"}],
        "sources": [{"spec_id": DEBT_SPEC}, {"spec_id": OTHER_SPEC}],
    }
    sources_swapped = {
        "steps": [{"tool": "data-mcp.get_data"}, {"tool": "analysis-mcp.aggregate"}],
        "sources": [{"spec_id": OTHER_SPEC}, {"spec_id": DEBT_SPEC}],
    }
    steps_swapped = {
        "steps": [{"tool": "analysis-mcp.aggregate"}, {"tool": "data-mcp.get_data"}],
        "sources": [{"spec_id": DEBT_SPEC}, {"spec_id": OTHER_SPEC}],
    }

    assert matches_subset(expected, sources_swapped)
    assert not matches_subset(expected, steps_swapped)


def test_matches_subset_still_counts_sources() -> None:
    # Unordered is not "a subset of": every actual source has to be one the
    # case named, and each expected one has to match a distinct actual one.
    assert not matches_subset(
        {"sources": [{"spec_id": DEBT_SPEC}]},
        {"sources": [{"spec_id": DEBT_SPEC}, {"spec_id": OTHER_SPEC}]},
    )
    assert not matches_subset(
        {"sources": [{"spec_id": DEBT_SPEC}, {"spec_id": DEBT_SPEC}]},
        {"sources": [{"spec_id": DEBT_SPEC}, {"spec_id": OTHER_SPEC}]},
    )


def test_a_matching_proposal_scores_one() -> None:
    expected = [
        _invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES)),
    ]
    actual = [
        _invocation(_carried(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))),
    ]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.overall_score == 1.0
    assert result.overall_eval_status == EvalStatus.PASSED
    assert result.per_invocation_results[0].score == 1.0


def test_a_mismatched_proposal_scores_zero() -> None:
    expected = [_invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))]
    wrong_steps = [{"tool": "data-mcp.get_data"}, {"tool": "analysis-mcp.growth"}]
    actual = [_invocation(_carried(_ok_proposal(wrong_steps, _EXPECTED_SOURCES)))]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.overall_score == 0.0
    assert result.overall_eval_status == EvalStatus.FAILED
    assert result.per_invocation_results[0].score == 0.0


def test_a_missing_response_scores_zero() -> None:
    expected = [_invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))]
    actual = [_invocation()]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 0.0


def test_a_referral_where_a_proposal_was_expected_scores_zero() -> None:
    expected = [_invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))]
    actual = [_invocation(_carried(_referral("source_ambiguous", "annual_debt_service")))]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 0.0


def test_a_written_referral_alongside_the_expected_proposal_scores_zero() -> None:
    # The mapping case's exclusivity check: a translation that compiled and
    # another that referred, on the same turn, is hedging. Both are
    # `translation` verdicts distinguished only by `outcome`.
    expected = [_invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))]
    actual = [
        _invocation(
            _carried(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES)),
            _carried(_referral("source_ambiguous", "annual_debt_service")),
        )
    ]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 0.0


def test_a_refused_referral_alongside_the_expected_proposal_does_not_hedge() -> None:
    # A translation that answered a problem wrote no referral, so the plan
    # is exactly as the compiled one left it: there is nothing to hedge
    # with. Only a referral, the one answer naming a `kind`, counts.
    expected = [_invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))]
    problem = genai_types.FunctionResponse(
        name="translation", response={"outcome": "invalid_input", "code": "referral_unearned"}
    )
    actual = [
        _invocation(_carried(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES)), _carried(problem))
    ]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 1.0


def test_an_accepted_proposal_alongside_the_expected_referral_scores_zero() -> None:
    # The referral case's exclusivity check: a translation that refers and
    # another that compiled on the same turn is hedging, not answering the
    # referral case's question.
    expected = [_invocation(_referral("source_ambiguous", "annual_aid"))]
    actual = [
        _invocation(
            _carried(_referral("source_ambiguous", "annual_aid")),
            _carried(_ok_proposal(_EXPECTED_STEPS, [{"spec_id": OTHER_SPEC}])),
        )
    ]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 0.0


def test_a_matching_referral_scores_one() -> None:
    expected = [_invocation(_referral("requirement_unmeetable", "annual_debt_service"))]
    actual = [
        _invocation(
            _carried(_referral("requirement_unmeetable", "annual_debt_service")),
        )
    ]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.overall_score == 1.0
    assert result.per_invocation_results[0].score == 1.0


def test_the_last_matching_actual_response_is_the_one_compared() -> None:
    # A first proposal whose translation failed, then a second that compiled:
    # the metric reads the retry, not the failed first try.
    expected = [_invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))]
    actual = [
        _invocation(
            _carried(_COMPILE_FAILED),
            _carried(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES)),
        )
    ]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 1.0


def test_a_failed_translation_after_the_compiled_one_does_not_hide_it() -> None:
    # The compiled translation, then one more proposal whose translation
    # failed: the plan keeps the compiled workflow, so that is the turn's
    # outcome. Grading the last response by name alone scored this zero;
    # preferring the last response with the stated outcome grades the
    # compiled one -- and the hedge checks above still read every response.
    expected = [_invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))]
    actual = [
        _invocation(
            _carried(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES)),
            _carried(_COMPILE_FAILED),
        )
    ]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 1.0


def test_a_proposal_the_planner_refused_carries_no_translation_to_grade() -> None:
    # A refused proposal ran no translation, so it neither stands in for the
    # compiled verdict nor hides it; one that never reached translation
    # leaves the case with nothing to match.
    expected = [_invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))]
    refused = genai_types.FunctionResponse(
        name="propose_plan", response={"outcome": "invalid_input", "code": "invalid_plan"}
    )
    compiled = _carried(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))

    after = workflow_outcome(_metric(), [_invocation(refused, compiled)], expected, None)
    alone = workflow_outcome(_metric(), [_invocation(refused)], expected, None)

    assert after.per_invocation_results[0].score == 1.0
    assert alone.per_invocation_results[0].score == 0.0


def test_a_matching_plan_proposal_scores_one() -> None:
    expected = [_invocation(_plan_proposal())]
    actual = [_invocation(_plan_proposal(version=1, plan={"question": "q"}))]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.overall_score == 1.0
    assert result.per_invocation_results[0].score == 1.0


def _submitted(outcome: str, **fields: object) -> genai_types.FunctionResponse:
    """A `submit_analysis` answer, as a compact-delivery run records it."""
    return genai_types.FunctionResponse(
        name="submit_analysis", response={"outcome": outcome, "status": "draft", **fields}
    )


@pytest.mark.parametrize(
    ("expected", "actual", "score"),
    [
        (_plan_proposal(version=2), _submitted("recorded", version=2), 1.0),
        (_plan_proposal(), _submitted("ready_for_review", version=1), 1.0),
        (_plan_proposal(), _submitted("needs_fields"), 0.0),
        (
            _ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES),
            _submitted(
                "ready_for_review",
                version=1,
                translation=_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES).response,
            ),
            1.0,
        ),
    ],
    ids=["recorded", "ready-for-review", "needs-fields", "translation"],
)
def test_a_submit_analysis_answer_is_graded_as_the_proposal_it_records(
    expected: genai_types.FunctionResponse, actual: genai_types.FunctionResponse, score: float
) -> None:
    result = workflow_outcome(_metric(), [_invocation(actual)], [_invocation(expected)], None)

    assert result.per_invocation_results[0].score == score


def test_a_plan_amendment_is_graded_by_the_version_the_case_states() -> None:
    # `version: 2` asserts the revision amended the first plan rather than
    # starting a fresh one, which no rubric can read off the trace.
    expected = [_invocation(_plan_proposal(version=2))]
    fresh = [_invocation(_plan_proposal(version=1))]
    amended = [_invocation(_plan_proposal(version=2))]

    assert workflow_outcome(_metric(), fresh, expected, None).per_invocation_results[0].score == 0.0
    assert (
        workflow_outcome(_metric(), amended, expected, None).per_invocation_results[0].score == 1.0
    )


def test_a_proposal_declared_ready_matches_the_expected_plan_proposal() -> None:
    # Declaring a plan ready approves nothing and executes nothing itself,
    # so a plan-proposal case has no hedge to check against it.
    expected = [_invocation(_plan_proposal())]
    actual = [_invocation(_compiled_proposal(summary=_SUMMARY))]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 1.0


def test_a_matching_rejection_scores_one() -> None:
    expected = [
        _invocation(
            genai_types.FunctionResponse(
                name="reject_plan", response={"outcome": "ok", "status": "rejected"}
            )
        )
    ]
    actual = [
        _invocation(
            _plan_proposal(version=1),
            genai_types.FunctionResponse(
                name="reject_plan", response={"outcome": "ok", "status": "rejected"}
            ),
        )
    ]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 1.0


def test_a_proposal_declared_ready_alongside_the_expected_rejection_scores_zero() -> None:
    # The reject case's exclusivity check: an agent cannot both record a
    # rejection and declare the same plan ready for the user's answer.
    expected = [
        _invocation(
            genai_types.FunctionResponse(
                name="reject_plan", response={"outcome": "ok", "status": "rejected"}
            )
        )
    ]
    actual = [
        _invocation(
            genai_types.FunctionResponse(
                name="reject_plan", response={"outcome": "ok", "status": "rejected"}
            ),
            _compiled_proposal(summary=_SUMMARY),
        )
    ]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 0.0


def test_a_blocked_declaration_alongside_the_expected_rejection_does_not_hedge() -> None:
    # A compiled proposal whose readiness the gate refused -- here blocked by
    # the very rejection it is paired with -- declared nothing, so there is
    # nothing to hedge with. Only an answer carrying `summary` counts.
    expected = [
        _invocation(
            genai_types.FunctionResponse(
                name="reject_plan", response={"outcome": "ok", "status": "rejected"}
            )
        )
    ]
    actual = [
        _invocation(
            genai_types.FunctionResponse(
                name="reject_plan", response={"outcome": "ok", "status": "rejected"}
            ),
            _compiled_proposal(submission=_BLOCKED_SUBMISSION),
        )
    ]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 1.0


def test_a_response_for_a_tool_the_metric_does_not_score_states_no_expectation() -> None:
    # A `tool_responses` entry naming `discover_source` is not an expectation
    # this metric can grade: the turn scores `None`, which is why
    # `test_eval_sets.py` refuses such a turn in a set that turns it on.
    expected = [
        _invocation(
            genai_types.FunctionResponse(name="discover_source", response={"outcome": "ok"})
        )
    ]
    actual = [_invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score is None


def test_a_turn_with_no_expected_response_is_not_evaluated_and_excluded_from_the_average() -> None:
    expected = [_invocation()]
    actual = [_invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score is None
    assert result.per_invocation_results[0].eval_status == EvalStatus.NOT_EVALUATED
    assert result.overall_eval_status == EvalStatus.NOT_EVALUATED


def test_one_failing_turn_fails_the_whole_case() -> None:
    expected = [
        _invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES)),
        _invocation(_referral("source_ambiguous", "annual_aid")),
    ]
    actual = [
        _invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES)),
        _invocation(),  # the second turn never made the expected referral
    ]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 1.0
    assert result.per_invocation_results[1].score == 0.0
    assert result.overall_score == 0.0
    assert result.overall_eval_status == EvalStatus.FAILED


def test_workflow_outcome_needs_expected_invocations() -> None:
    with pytest.raises(ValueError, match="expected_invocations"):
        workflow_outcome(_metric(), [_invocation()], None, None)
