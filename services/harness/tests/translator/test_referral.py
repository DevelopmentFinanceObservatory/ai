"""Tests for `translator.referral`: a referral names what it is about, and is saved."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from planner.workflow import SelectedSource
from pydantic import ValidationError
from translator.referral import (
    REFERRAL_STATE_KEY,
    Referral,
    ReferralKind,
    SourceRecommendation,
    clear_referral,
    load_referral,
    save_referral,
)

SourceFactory = Callable[..., SelectedSource]


@pytest.mark.parametrize(
    "kind",
    [
        ReferralKind.NO_DATA,
        ReferralKind.UNSUPPORTED_ASK,
        ReferralKind.REQUIREMENT_UNMEETABLE,
        ReferralKind.PINNED_SOURCE_NOT_FOUND,
        ReferralKind.ASK_NOT_UNDERSTOOD,
        ReferralKind.DISCOVERY_UNAVAILABLE,
        ReferralKind.SEED_DROPPED,
    ],
)
def test_a_referral_about_a_requirement_names_it(kind: ReferralKind) -> None:
    assert Referral(kind=kind, reason="r", requirement_id="annual_debt_service").kind is kind
    with pytest.raises(ValidationError, match="names the requirement it is about"):
        Referral(kind=kind, reason="r")


def test_a_referral_about_a_decision_names_it() -> None:
    assert Referral(kind=ReferralKind.NO_OPERATION_COVERS, reason="r", decision_id="normalisation")
    with pytest.raises(ValidationError, match="names the decision it is about"):
        Referral(kind=ReferralKind.NO_OPERATION_COVERS, reason="r")
    with pytest.raises(ValidationError, match="names no decision"):
        Referral(kind=ReferralKind.NO_DATA, reason="r", requirement_id="a", decision_id="b")


def test_a_spent_budget_names_nothing() -> None:
    assert Referral(kind=ReferralKind.DISCOVERY_BUDGET_EXHAUSTED, reason="r").requirement_id is None
    with pytest.raises(ValidationError, match="names no requirement"):
        Referral(kind=ReferralKind.DISCOVERY_BUDGET_EXHAUSTED, reason="r", requirement_id="a")


def test_an_ambiguous_source_carries_its_candidates(source: SourceFactory) -> None:
    two = (source(), source(spec_id="spec_" + "b" * 32))

    assert (
        len(
            Referral(
                kind=ReferralKind.SOURCE_AMBIGUOUS, reason="r", requirement_id="a", candidates=two
            ).candidates
        )
        == 2
    )
    with pytest.raises(ValidationError, match="carries the candidate sources"):
        Referral(
            kind=ReferralKind.SOURCE_AMBIGUOUS, reason="r", requirement_id="a", candidates=two[:1]
        )
    with pytest.raises(ValidationError, match="carries no candidates"):
        Referral(kind=ReferralKind.UNSUPPORTED_ASK, reason="r", requirement_id="a", candidates=two)


def test_a_no_data_referral_carries_its_nearest_sources_and_reason(source: SourceFactory) -> None:
    nearest = (source(),)

    referral = Referral(
        kind=ReferralKind.NO_DATA,
        reason="r",
        requirement_id="a",
        candidates=nearest,
        no_data_reason="no_observations",
    )

    assert referral.candidates == nearest
    assert referral.no_data_reason == "no_observations"
    # As few as none is fine: the nearest-series relaxation can find nothing.
    assert Referral(kind=ReferralKind.NO_DATA, reason="r", requirement_id="a").candidates == ()


def test_only_a_no_data_referral_carries_a_reason_code() -> None:
    with pytest.raises(ValidationError, match="carries no reason code"):
        Referral(
            kind=ReferralKind.UNSUPPORTED_ASK,
            reason="r",
            requirement_id="a",
            no_data_reason="no_observations",
        )


def test_a_pinned_source_not_found_may_carry_candidates_but_need_not(
    source: SourceFactory,
) -> None:
    two = (source(), source(spec_id="spec_" + "b" * 32))

    assert (
        Referral(
            kind=ReferralKind.PINNED_SOURCE_NOT_FOUND,
            reason="r",
            requirement_id="a",
            candidates=two,
        ).candidates
        == two
    )
    assert (
        Referral(
            kind=ReferralKind.PINNED_SOURCE_NOT_FOUND, reason="r", requirement_id="a"
        ).candidates
        == ()
    )


def test_an_ambiguous_source_may_carry_a_recommendation_naming_one_of_its_candidates(
    source: SourceFactory,
) -> None:
    two = (source(), source(spec_id="spec_" + "b" * 32))

    referral = Referral(
        kind=ReferralKind.SOURCE_AMBIGUOUS,
        reason="r",
        requirement_id="a",
        candidates=two,
        recommendation=SourceRecommendation(spec_id=two[0].spec_id, rationale="closer match"),
    )

    assert referral.recommendation is not None
    assert referral.recommendation.spec_id == two[0].spec_id
    assert referral.recommendation.rationale == "closer match"
    with pytest.raises(ValidationError, match="must name one of the referral's own candidates"):
        Referral(
            kind=ReferralKind.SOURCE_AMBIGUOUS,
            reason="r",
            requirement_id="a",
            candidates=two,
            recommendation=SourceRecommendation(spec_id="spec_" + "c" * 32, rationale="r"),
        )


def test_only_an_ambiguous_source_carries_a_recommendation(source: SourceFactory) -> None:
    with pytest.raises(ValidationError, match="carries no recommendation"):
        Referral(
            kind=ReferralKind.NO_DATA,
            reason="r",
            requirement_id="a",
            candidates=(source(),),
            recommendation=SourceRecommendation(spec_id=source().spec_id, rationale="r"),
        )


def test_a_recommendation_needs_a_rationale(source: SourceFactory) -> None:
    with pytest.raises(ValidationError, match="at least 1 character"):
        SourceRecommendation(spec_id=source().spec_id, rationale="")


def test_a_referral_gives_a_reason() -> None:
    with pytest.raises(ValidationError, match="at least 1 character"):
        Referral(kind=ReferralKind.DISCOVERY_BUDGET_EXHAUSTED, reason="")


def test_save_load_and_clear_round_trip_through_session_state() -> None:
    state: dict[str, Any] = {}
    referral = Referral(kind=ReferralKind.NO_DATA, reason="no series matches", requirement_id="a")

    assert load_referral(state) is None

    save_referral(state, referral)

    assert state[REFERRAL_STATE_KEY]
    loaded = load_referral(state)
    assert loaded == referral

    clear_referral(state)

    assert load_referral(state) is None
