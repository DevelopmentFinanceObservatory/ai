"""Unit tests for `harness_translator_v1.outcome_metric.workflow_outcome`.

Fabricated `Invocation`s, model-free: this pins the metric's own matching
logic (subset comparison, `sources` in any order, exclusivity between a
proposal and a referral and between a rejection and an accepted
`submit_plan`, "last response with the stated outcome wins") independently
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
    return genai_types.FunctionResponse(
        name="translate",
        response={"outcome": "ok", "steps": steps, "sources": sources},
    )


def _referral(kind: str, requirement_id: str) -> genai_types.FunctionResponse:
    outcome = "choice_required" if kind == "source_ambiguous" else "invalid_input"
    return genai_types.FunctionResponse(
        name="translate",
        response={"outcome": outcome, "kind": kind, "requirement_id": requirement_id},
    )


def _plan_proposal(**fields: object) -> genai_types.FunctionResponse:
    return genai_types.FunctionResponse(
        name="propose_plan", response={"outcome": "ok", "status": "draft", **fields}
    )


def _submit_plan(response: dict[str, object]) -> genai_types.FunctionResponse:
    return genai_types.FunctionResponse(name="submit_plan", response=response)


_EXPECTED_STEPS = [{"tool": "data-mcp.get_data"}, {"tool": "analysis-mcp.aggregate"}]
_EXPECTED_SOURCES = [{"spec_id": DEBT_SPEC}]
_REFUSED_SUBMISSION = {"outcome": "refused", "code": "workflow_missing"}
_ACCEPTED_SUBMISSION = {"outcome": "ok", "version": 1, "revision": "r1", "summary": "..."}


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
        _invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES)),
    ]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.overall_score == 1.0
    assert result.overall_eval_status == EvalStatus.PASSED
    assert result.per_invocation_results[0].score == 1.0


def test_a_mismatched_proposal_scores_zero() -> None:
    expected = [_invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))]
    wrong_steps = [{"tool": "data-mcp.get_data"}, {"tool": "analysis-mcp.growth"}]
    actual = [_invocation(_ok_proposal(wrong_steps, _EXPECTED_SOURCES))]

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
    actual = [_invocation(_referral("source_ambiguous", "annual_debt_service"))]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 0.0


def test_a_written_referral_alongside_the_expected_proposal_scores_zero() -> None:
    # The mapping case's exclusivity check: an accepted proposal and a
    # referral that was written on the same turn is hedging. Both are
    # `translate` responses distinguished only by `outcome` -- before
    # `_hedged` read `expected`'s outcome, `translate` fell to its `case _:
    # return False` and this scored 1.0.
    expected = [_invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))]
    actual = [
        _invocation(
            _ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES),
            _referral("source_ambiguous", "annual_debt_service"),
        )
    ]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 0.0


def test_a_refused_referral_alongside_the_expected_proposal_does_not_hedge() -> None:
    # A `translate` call the tool refused wrote no referral, so the plan
    # is exactly as the accepted proposal left it: there is nothing to hedge
    # with. Only a referral, the one answer naming a `kind`, counts.
    expected = [_invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))]
    refused = genai_types.FunctionResponse(
        name="translate", response={"outcome": "invalid_input", "code": "referral_unearned"}
    )
    actual = [_invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES), refused)]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 1.0


def test_an_accepted_proposal_alongside_the_expected_referral_scores_zero() -> None:
    # The referral case's exclusivity check: a translator that refers and
    # also has an accepted proposal on the same turn is hedging, not
    # answering the referral case's question. Both are `translate`
    # responses, the same dead branch as the mapping case above.
    expected = [_invocation(_referral("source_ambiguous", "annual_aid"))]
    actual = [
        _invocation(
            _referral("source_ambiguous", "annual_aid"),
            _ok_proposal(_EXPECTED_STEPS, [{"spec_id": OTHER_SPEC}]),
        )
    ]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 0.0


def test_a_matching_referral_scores_one() -> None:
    expected = [_invocation(_referral("requirement_unmeetable", "annual_debt_service"))]
    actual = [
        _invocation(
            _referral("requirement_unmeetable", "annual_debt_service"),
        )
    ]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.overall_score == 1.0
    assert result.per_invocation_results[0].score == 1.0


def test_the_last_matching_actual_response_is_the_one_compared() -> None:
    # A first attempt that failed, then a second that succeeded: the metric
    # reads the retry, not the failed first try.
    expected = [_invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))]
    failed = genai_types.FunctionResponse(
        name="translate", response={"outcome": "invalid_input", "code": "compile_failed"}
    )
    actual = [_invocation(failed, _ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 1.0


def test_a_refused_call_after_the_accepted_proposal_does_not_hide_it() -> None:
    # The accepted proposal, then one more `translate` call the compiler
    # refused: the plan keeps the accepted workflow, so that is the turn's
    # outcome. Grading the last response by name alone scored this zero;
    # preferring the last response with the stated outcome grades the
    # accepted one -- and the hedge checks above still read every response.
    expected = [_invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES))]
    refused = genai_types.FunctionResponse(
        name="translate", response={"outcome": "invalid_input", "code": "compile_failed"}
    )
    actual = [_invocation(_ok_proposal(_EXPECTED_STEPS, _EXPECTED_SOURCES), refused)]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 1.0


def test_a_matching_plan_proposal_scores_one() -> None:
    expected = [_invocation(_plan_proposal())]
    actual = [_invocation(_plan_proposal(version=1, plan={"question": "q"}))]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.overall_score == 1.0
    assert result.per_invocation_results[0].score == 1.0


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


def test_a_submit_plan_answer_alongside_the_expected_plan_proposal_does_not_hedge() -> None:
    # `submit_plan` approves nothing and executes nothing itself -- calling
    # it in the same turn as `propose_plan` is the trajectory a plan's
    # readiness is keyed on, not a way to have it both ways, so a
    # plan-proposal case has no hedge to check against it.
    expected = [_invocation(_plan_proposal())]
    actual = [_invocation(_plan_proposal(version=1), _submit_plan(_ACCEPTED_SUBMISSION))]

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


def test_an_accepted_submit_plan_alongside_the_expected_rejection_scores_zero() -> None:
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
            _submit_plan(_ACCEPTED_SUBMISSION),
        )
    ]

    result = workflow_outcome(_metric(), actual, expected, None)

    assert result.per_invocation_results[0].score == 0.0


def test_a_refused_submit_plan_alongside_the_expected_rejection_does_not_hedge() -> None:
    # A `submit_plan` the tool refused -- here blocked by the very rejection
    # it is paired with -- declared nothing, so there is nothing to hedge
    # with. Only a response whose outcome is `ok` counts.
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
            _submit_plan(_REFUSED_SUBMISSION),
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
