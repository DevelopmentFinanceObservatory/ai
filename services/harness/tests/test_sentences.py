"""Tests for `harness.sentences`: every code the user can be told about has its sentence."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from enum import StrEnum

import pytest
from compiler.compile import CompileIssueCode
from harness.sentences import (
    COMPILE_REASONS,
    DECLINE_REASONS,
    REFERRAL_REASONS,
    SEAM_SENTENCES,
    TERMINAL_PROPOSE_CODES,
    SeamCode,
    referral_reason,
    refusal_sentence,
)
from planner.plan import UnsupportedOperation
from translator.referral import ReferralKind


@pytest.mark.parametrize(
    ("table", "codes"),
    [
        (REFERRAL_REASONS, set(ReferralKind)),
        (COMPILE_REASONS, set(CompileIssueCode)),
        (DECLINE_REASONS, set(UnsupportedOperation)),
        (SEAM_SENTENCES, set(SeamCode) | TERMINAL_PROPOSE_CODES),
    ],
    ids=["referral", "compile", "decline", "seam"],
)
def test_every_code_has_exactly_one_sentence(
    table: Mapping[StrEnum, str], codes: Collection[StrEnum]
) -> None:
    assert set(table) == set(codes)


@pytest.mark.parametrize(
    "clause", [*REFERRAL_REASONS.values(), *COMPILE_REASONS.values(), *DECLINE_REASONS.values()]
)
def test_a_reason_is_a_clause_another_sentence_can_carry(clause: str) -> None:
    assert clause[0].islower()
    assert not clause.endswith(".")


@pytest.mark.parametrize("code", [*SeamCode, *TERMINAL_PROPOSE_CODES])
def test_every_seam_sentence_fills_from_an_empty_detail(code: StrEnum) -> None:
    text = refusal_sentence({"code": code})

    assert "{" not in text
    assert text.endswith(".")


def test_a_code_no_table_names_claims_no_budget() -> None:
    text = refusal_sentence({"code": "no_such_code", "detail": {"rounds": 4}})

    assert "budget" not in text
    assert text not in {refusal_sentence({"code": code}) for code in SEAM_SENTENCES}


def test_a_repair_referral_reads_the_compile_issue_it_names() -> None:
    repair = ReferralKind.REQUIREMENT_NEEDS_REPAIR

    assert (
        referral_reason(repair, "family_unsummable")
        == (COMPILE_REASONS[CompileIssueCode.FAMILY_UNSUMMABLE])
    )
    assert referral_reason(repair, None) == REFERRAL_REASONS[repair]
    assert referral_reason(repair, "no_such_issue") == REFERRAL_REASONS[repair]
    assert referral_reason("no_data") == REFERRAL_REASONS[ReferralKind.NO_DATA]
    assert referral_reason("no_such_kind") == referral_reason(None)
    unmeetable = ReferralKind.REQUIREMENT_UNMEETABLE
    assert referral_reason(unmeetable, source_mismatch=True) != REFERRAL_REASONS[unmeetable]
    assert referral_reason(unmeetable) == REFERRAL_REASONS[unmeetable]


def test_a_plan_not_ready_sentence_carries_the_referral_it_ended_on() -> None:
    text = refusal_sentence(
        {
            "code": SeamCode.PLAN_NOT_READY,
            "detail": {
                "kind": "requirement_needs_repair",
                "requirement_id": "aid_by_donor",
                "constraint": "level_filter_on_flow_axis",
            },
        }
    )

    assert "'aid_by_donor'" in text
    assert COMPILE_REASONS[CompileIssueCode.LEVEL_FILTER_ON_FLOW_AXIS] in text


def test_the_budget_refusal_is_stated_as_a_per_entry_budget_in_the_user_s_terms() -> None:
    """`MAX_PLAN_ROUNDS` counts the proposals of one entry into the planner, restarts included.

    The plan's own version can read higher than the count (it climbs across
    entries), so the sentence names the budget. It makes no claim about what
    can still be approved: the seam has no view of the graph's record, and the
    graph adds that claim where its record supports it.
    """
    text = refusal_sentence(
        {"code": "budget_exhausted", "detail": {"scope": "plan_rounds", "rounds": 4, "version": 9}}
    )

    assert text == (
        "This request has used its budget of 4 plan drafts, so nothing was "
        "recorded. Your next message starts with a fresh budget."
    )


def test_the_question_fixed_refusal_is_stated_as_asking_the_other_question_separately() -> None:
    """A restart while a plan is already recorded spends no budget, so the
    sentence says the other question goes in its own message."""
    text = refusal_sentence({"code": "question_fixed", "outcome": "refused"})

    assert "own message" in text
    assert "budget" not in text
