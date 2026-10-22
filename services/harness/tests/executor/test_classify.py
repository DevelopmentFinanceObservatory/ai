"""Tests for `executor.classify`: what a refusal calls for."""

from __future__ import annotations

import math
import random
from typing import cast, get_args

import pytest
from analysis_mcp.tool_errors import ToolErrorCode as AnalysisCode
from data_mcp.errors import UpstreamError
from data_mcp.tool_errors import CeilingKind, ToolRefusal, from_upstream_error, to_tool_result
from data_mcp.tool_errors import ToolErrorCode as DataCode
from executor.classify import (
    BACKOFF_SECONDS,
    COMPILER_PROVABLE_CODES,
    MAX_ATTEMPTS_PER_STEP,
    MAX_RETRY_AFTER_SECONDS,
    METHOD_CHOICE_CODES,
    REDISCOVERY_CODES,
    RETRY_JITTER_FACTOR,
    STOP_CODES,
    TRANSIENT_CODES,
    Disposition,
    ExecutorCode,
    ToolFailure,
    candidates_of,
    classify_refusal,
    delay_before_retry,
    is_anchor_fault,
    jittered_retry_delay,
    parse_tool_failure,
    retry_delay,
)

_PLACED = (
    METHOD_CHOICE_CODES,
    REDISCOVERY_CODES,
    TRANSIENT_CODES,
    COMPILER_PROVABLE_CODES,
    STOP_CODES,
)


def test_every_code_either_server_can_answer_with_is_placed_exactly_once() -> None:
    # A new server code must be placed here deliberately, not fall into the
    # stop bucket by omission: this fails until someone picks a side.
    served = {code.value for code in DataCode} | {code.value for code in AnalysisCode}
    placed = frozenset().union(*_PLACED)

    assert served == placed, {
        "unplaced": sorted(served - placed),
        "stale": sorted(placed - served),
    }
    assert sum(len(codes) for codes in _PLACED) == len(placed), "a code sits in two sets"


def test_a_refusal_missing_its_code_parses_into_a_failure_naming_the_defect() -> None:
    # A server that cannot even refuse properly is exactly when the executor
    # still has to report something back to the plan, rather than crash the
    # run on a malformed `structured_content`.
    failure = parse_tool_failure({"retry_mode": "correct_and_retry", "message": "oops"})

    assert failure.code == ExecutorCode.MALFORMED_REFUSAL
    assert "ToolFailure" in failure.message


def test_a_well_formed_refusal_does_not_parse_as_malformed() -> None:
    # Proves the test above is failable: a payload naming a real code is not
    # swept into the malformed bucket by the same assertion shape.
    failure = parse_tool_failure({"code": "internal_error", "retry_mode": "do_not_retry"})

    assert failure.code == "internal_error"
    assert failure.code != ExecutorCode.MALFORMED_REFUSAL


@pytest.mark.parametrize(
    "code",
    sorted(frozenset().union(*_PLACED)),
)
def test_every_placed_code_round_trips_through_tool_failure(code: str) -> None:
    failure = parse_tool_failure(
        {"code": code, "retry_mode": "correct_and_retry", "message": "refused", "fix": {}}
    )

    assert failure.code == code
    assert failure.message == "refused"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"code": "service_unavailable", "retry_mode": "retry_after_delay"},
            Disposition.RETRY_LATER,
        ),
        (
            {"code": "operation_still_processing", "retry_mode": "retry_after_delay"},
            Disposition.RETRY_LATER,
        ),
        (
            {"code": "upstream_version_changed", "retry_mode": "refresh_then_retry"},
            Disposition.REDISCOVER,
        ),
        (
            {"code": "definition_not_held", "retry_mode": "correct_and_retry"},
            Disposition.REDISCOVER,
        ),
        (
            {"code": "open_role_unavailable", "retry_mode": "correct_and_retry"},
            Disposition.REDISCOVER,
        ),
        (
            {"code": "unit_heterogeneous", "retry_mode": "correct_and_retry"},
            Disposition.METHOD_CHOICE,
        ),
        (
            {"code": "entity_ambiguous", "retry_mode": "correct_and_retry"},
            Disposition.METHOD_CHOICE,
        ),
        (
            {"code": "cross_frame_unit_mismatch", "retry_mode": "correct_and_retry"},
            Disposition.METHOD_CHOICE,
        ),
        (
            {"code": "coverage_insufficient", "retry_mode": "correct_and_retry"},
            Disposition.METHOD_CHOICE,
        ),
        (
            {"code": "share_of_key_unmatched", "retry_mode": "correct_and_retry"},
            Disposition.METHOD_CHOICE,
        ),
        # A quotient that found nothing to divide by, in every matched cell
        # or for every key, goes to the planner as a share's does.
        (
            {"code": "quotient_of_zero_denominator", "retry_mode": "correct_and_retry"},
            Disposition.METHOD_CHOICE,
        ),
        (
            {"code": "quotient_of_no_key_matched", "retry_mode": "correct_and_retry"},
            Disposition.METHOD_CHOICE,
        ),
        # A side with no unit to compose names a fact about the frame, as
        # `derived_input_ordinal` does: dividing something else is a
        # different plan.
        (
            {"code": "quotient_of_side_unit_unsupported", "retry_mode": "correct_and_retry"},
            Disposition.STOP,
        ),
        ({"code": "role_ambiguous", "retry_mode": "correct_and_retry"}, Disposition.STOP),
        ({"code": "frame_not_held", "retry_mode": "correct_and_retry"}, Disposition.STOP),
        ({"code": "internal_error", "retry_mode": "do_not_retry"}, Disposition.STOP),
        # apis cannot sum the family `get_data` asked to sum. It stops by
        # its code too, whatever mode it arrives under: reading the family
        # separately is a different plan.
        ({"code": "rollup_not_summable", "retry_mode": "do_not_retry"}, Disposition.STOP),
        ({"code": "rollup_not_summable", "retry_mode": "correct_and_retry"}, Disposition.STOP),
        # A partial `denominator_by` mapping is a compiler bug -- the compiler
        # writes it either empty or total over `numerator_by` -- so it stops
        # like any other `COMPILER_PROVABLE_CODES` member.
        (
            {"code": "share_of_key_unknown", "retry_mode": "correct_and_retry"},
            Disposition.STOP,
        ),
        # The planner refuses a sum over a quotient, so the engine's refusal
        # of one stops as a compiler-provable code does.
        (
            {"code": "composed_unit_sum_unsupported", "retry_mode": "correct_and_retry"},
            Disposition.STOP,
        ),
        # `derived_input_ordinal`'s own fix names a reason, not a set of
        # candidates, and its correction (a different frame_ref, or
        # `change="absolute"`) is a different plan, like
        # `derived_input_unusable`/`derived_input_unrecoverable`.
        (
            {"code": "derived_input_ordinal", "retry_mode": "correct_and_retry"},
            Disposition.STOP,
        ),
        # A method-choice code under a mode that says not to retry is a stop:
        # the mode is the tool's word on whether anything can be done.
        ({"code": "unit_heterogeneous", "retry_mode": "do_not_retry"}, Disposition.STOP),
        # `over_ceiling` is ceiling-kind-dependent rather than a flat stop:
        # `rows`/`bytes` narrow with scope, so they're a method choice.
        (
            {
                "code": "over_ceiling",
                "retry_mode": "correct_and_retry",
                "fix": {"ceiling_kind": "rows"},
            },
            Disposition.METHOD_CHOICE,
        ),
        (
            {
                "code": "over_ceiling",
                "retry_mode": "correct_and_retry",
                "fix": {"ceiling_kind": "bytes"},
            },
            Disposition.METHOD_CHOICE,
        ),
        # A request ceiling narrows with `scope.entities` too.
        (
            {
                "code": "over_ceiling",
                "retry_mode": "correct_and_retry",
                "fix": {"ceiling_kind": "requests"},
            },
            Disposition.METHOD_CHOICE,
        ),
        # A candidate window is fixed by the spec_id's identity, like a member
        # ceiling, so it stops.
        (
            {
                "code": "over_ceiling",
                "retry_mode": "correct_and_retry",
                "fix": {"ceiling_kind": "candidate_window"},
            },
            Disposition.STOP,
        ),
        # `member_cardinality` is fixed by the spec_id itself and the index
        # version behind it -- scope can't narrow it, and rediscovering the
        # same spec reproduces the identical refusal -- so it stops.
        (
            {
                "code": "over_ceiling",
                "retry_mode": "correct_and_retry",
                "fix": {"ceiling_kind": "member_cardinality"},
            },
            Disposition.STOP,
        ),
        # No `ceiling_kind` at all -- an older upstream, or a hand-built
        # payload -- stops: the executor cannot guess whether an unnamed
        # ceiling narrows with scope or names a different spec entirely, and
        # guessing wrong either way would spend a replan reproducing the
        # identical failing call.
        (
            {"code": "over_ceiling", "retry_mode": "correct_and_retry", "fix": {"detail": "x"}},
            Disposition.STOP,
        ),
        ({"code": "over_ceiling", "retry_mode": "correct_and_retry"}, Disposition.STOP),
        ({}, Disposition.STOP),
    ],
    ids=[
        "delay",
        "operation-still-processing-delay",
        "refresh",
        "definition-gone",
        "open-role-gone",
        "unit-choice",
        "entity-choice",
        "cross-frame-choice",
        "coverage-choice",
        "share-of-key-unmatched-choice",
        "quotient-of-zero-denominator-choice",
        "quotient-of-no-key-matched-choice",
        "quotient-of-side-unit-unsupported-stop",
        "compiler-provable",
        "genuine-stop",
        "final",
        "rollup-not-summable-final",
        "rollup-not-summable-stop",
        "share-of-key-unknown-stop",
        "composed-unit-sum-unsupported-stop",
        "derived-input-ordinal-stop",
        "choice-under-final-mode",
        "over-ceiling-rows-choice",
        "over-ceiling-bytes-choice",
        "over-ceiling-requests-choice",
        "over-ceiling-candidate-window-stop",
        "over-ceiling-member-cardinality-stop",
        "over-ceiling-unrecognised-kind-stop",
        "over-ceiling-no-fix-stop",
        "empty",
    ],
)
def test_a_refusal_is_retried_referred_or_stopped_on_by_its_mode_and_code(
    payload: dict[str, object], expected: Disposition
) -> None:
    # `role_ambiguous` is a correction the executor never makes: the compiler
    # designs it out, so at run time it is a stop, not a choice.
    # `definition_not_held` is `correct_and_retry` on the wire, but the
    # approved spec_id is gone from the store; only a new discovery fixes it.
    # `classify_refusal` reads the typed model, never the raw payload.
    assert classify_refusal(parse_tool_failure(payload)) is expected


# One row ceiling as apis reports it: refused inline by the mint door, and as a
# failed operation whose poll carries the stored refusal's ceiling fields.
_ROW_CEILING_DETAIL = (
    "materialised coverage of 120000 rows exceeds the sync materialisation ceiling of 100000 rows"
)
_ROW_CEILING_FIELDS = {"ceiling_kind": "rows", "count": 120000, "cap": 100000}
_OVER_CEILING_CAUSE = {"cause": {"code": "over_ceiling", "entities": [], "truncated": False}}


@pytest.mark.parametrize(
    ("status", "code", "extensions", "expected"),
    [
        (422, "over_ceiling", _ROW_CEILING_FIELDS, Disposition.METHOD_CHOICE),
        (
            500,
            "operation_failed",
            _ROW_CEILING_FIELDS | _OVER_CEILING_CAUSE,
            Disposition.METHOD_CHOICE,
        ),
        # An operation failed before apis stored its refusal names the code alone.
        (500, "operation_failed", _OVER_CEILING_CAUSE, Disposition.STOP),
    ],
    ids=["inline", "operation", "operation-without-stored-refusal"],
)
def test_a_row_ceiling_is_a_method_choice_when_its_body_names_the_kind(
    status: int, code: str, extensions: dict[str, object], expected: Disposition
) -> None:
    refusal = from_upstream_error(
        UpstreamError(status=status, code=code, detail=_ROW_CEILING_DETAIL, extensions=extensions)
    )
    payload = to_tool_result(refusal).structured_content
    assert payload is not None

    assert classify_refusal(parse_tool_failure(payload)) is expected


@pytest.mark.parametrize("kind", get_args(CeilingKind))
def test_a_ceiling_is_a_method_choice_exactly_when_data_mcp_offers_no_remedy(kind: str) -> None:
    """data-mcp gives an end-user remedy to the ceilings scope cannot narrow, and
    the executor offers a narrower request for every other kind."""
    refusal = ToolRefusal.over_ceiling(detail="over", ceiling_kind=cast(CeilingKind, kind))
    payload = to_tool_result(refusal).structured_content
    assert payload is not None

    disposition = classify_refusal(parse_tool_failure(payload))

    assert (disposition is Disposition.METHOD_CHOICE) is ("remedy" not in refusal.fix)


def test_the_delay_honours_the_refusals_own_and_otherwise_doubles() -> None:
    assert retry_delay({}, 0) == BACKOFF_SECONDS
    assert retry_delay({}, 2) == BACKOFF_SECONDS * 4
    assert retry_delay({"retry_after_s": 30}, 0) == 30.0
    assert retry_delay({"retry_after_s": 0.1}, 3) == BACKOFF_SECONDS * 8


def test_a_named_delay_is_honoured_no_further_than_the_cap() -> None:
    assert retry_delay({"retry_after_s": 3600}, 0) == MAX_RETRY_AFTER_SECONDS
    assert retry_delay({"retry_after_s": MAX_RETRY_AFTER_SECONDS + 0.5}, 0) == (
        MAX_RETRY_AFTER_SECONDS
    )
    assert retry_delay({"retry_after_s": MAX_RETRY_AFTER_SECONDS - 1}, 0) == (
        MAX_RETRY_AFTER_SECONDS - 1
    )


@pytest.mark.parametrize(
    "named",
    [-1, -0.5, math.inf, -math.inf, math.nan, True, "7", None],
    ids=["negative-int", "negative-float", "inf", "-inf", "nan", "bool", "string", "null"],
)
def test_a_delay_that_is_not_a_finite_non_negative_number_is_ignored(named: object) -> None:
    assert retry_delay({"retry_after_s": named}, 1) == BACKOFF_SECONDS * 2


def _bounds(low: float, high: float) -> float:
    """A jitter that returns its upper bound, so a test reads the widest wait."""
    del low
    return high


def test_a_jittered_wait_spreads_from_the_delay_up_to_twice_it() -> None:
    draws = [
        jittered_retry_delay({"retry_after_s": 2}, 0, jitter=random.Random(seed).uniform)
        for seed in range(20)
    ]

    assert all(2.0 <= draw <= 2.0 * RETRY_JITTER_FACTOR for draw in draws)
    assert len(set(draws)) == len(draws)


def test_a_jittered_wait_never_passes_the_cap() -> None:
    assert jittered_retry_delay({"retry_after_s": 20}, 0, jitter=_bounds) == (
        MAX_RETRY_AFTER_SECONDS
    )
    assert jittered_retry_delay({"retry_after_s": 3600}, 0, jitter=_bounds) == (
        MAX_RETRY_AFTER_SECONDS
    )


def test_only_a_retry_after_delay_refusal_gets_a_wait_before_the_last_attempt() -> None:
    refused = {"code": "service_unavailable", "retry_mode": "retry_after_delay"}

    assert delay_before_retry(refused, 0, jitter=_bounds) is not None
    assert delay_before_retry(refused, MAX_ATTEMPTS_PER_STEP - 2, jitter=_bounds) is not None
    assert delay_before_retry(refused, MAX_ATTEMPTS_PER_STEP - 1, jitter=_bounds) is None


@pytest.mark.parametrize("retry_mode", ["correct_and_retry", "do_not_retry", "refresh_then_retry"])
def test_no_other_retry_mode_gets_a_wait(retry_mode: str) -> None:
    refused = {"code": "service_unavailable", "retry_mode": retry_mode, "retry_after_s": 2}

    assert delay_before_retry(refused, 0, jitter=_bounds) is None


def test_candidates_are_read_from_the_fix_payload() -> None:
    fix = {"candidates": [{"id": "usd", "label": "US dollars"}, {"id": "eur", "label": "Euro"}]}

    assert candidates_of(_failure(fix=fix)) == fix["candidates"]
    assert candidates_of(_failure(fix={"window": [2016, 2020]})) == {"window": [2016, 2020]}
    assert candidates_of(_failure()) is None


def _failure(**overrides: object) -> ToolFailure:
    fields: dict[str, object] = {
        "code": "unit_heterogeneous",
        "message": "refused",
        "retry_mode": "correct_and_retry",
        "fix": {},
    }
    fields.update(overrides)
    return ToolFailure.model_validate(fields)


@pytest.mark.parametrize(
    ("payload", "fault"),
    [
        # No year is held by half of the series: a fact of the data.
        (
            {"code": "rolling_anchor_unresolved", "fix": {"detail": "no_qualifying_period"}},
            False,
        ),
        # Every other detail, and a refusal stating none, says what the run
        # itself sent does not fit.
        ({"code": "rolling_anchor_unresolved", "fix": {"detail": "literal_differs"}}, True),
        ({"code": "rolling_anchor_unresolved", "fix": {"detail": "populations_differ"}}, True),
        ({"code": "rolling_anchor_unresolved"}, True),
        ({"code": "rolling_grain_unsupported", "fix": {"detail": "literal_differs"}}, False),
    ],
)
def test_an_unresolved_anchor_is_the_runs_fault_unless_no_period_qualified(
    payload: dict[str, object], fault: bool
) -> None:
    failure = parse_tool_failure({"retry_mode": "correct_and_retry", **payload})

    assert is_anchor_fault(failure) is fault
    # Either way no replan changes it.
    assert classify_refusal(failure) is Disposition.STOP
