"""Tests for `translator.referral`: a referral names what it is about, and is saved."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import pytest
from compiler.compile import CompileIssueCode
from planner.plan import UnsupportedOperation
from planner.tool_results import Outcome
from planner.workflow import SelectedSource
from pydantic import ValidationError
from translator.referral import (
    OUTCOME_OF,
    REFERRAL_STATE_KEY,
    Referral,
    ReferralKind,
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
        ReferralKind.NO_OPERATION_COVERS,
    ],
)
def test_a_referral_about_a_requirement_names_it(kind: ReferralKind) -> None:
    assert Referral(kind=kind, reason="r", requirement_id="annual_debt_service").kind is kind
    with pytest.raises(ValidationError, match="names the requirement it is about"):
        Referral(kind=kind, reason="r")


def test_every_kind_lands_on_one_outcome() -> None:
    assert set(OUTCOME_OF) == set(ReferralKind)


def test_every_unsupported_kind_is_a_code_a_plan_can_decline() -> None:
    unsupported = {kind for kind, outcome in OUTCOME_OF.items() if outcome is Outcome.UNSUPPORTED}

    assert {kind.value for kind in unsupported} <= {code.value for code in UnsupportedOperation}


def test_a_repair_names_its_constraint_and_permitted_change() -> None:
    repair = Referral(
        kind=ReferralKind.REQUIREMENT_NEEDS_REPAIR,
        reason="r",
        requirement_id="a",
        constraint=CompileIssueCode.ENTITY_REQUIRED,
        repair="name the entity",
    )
    assert repair.outcome is Outcome.INVALID_INPUT
    with pytest.raises(ValidationError, match="names the constraint and the permitted repair"):
        Referral(kind=ReferralKind.REQUIREMENT_NEEDS_REPAIR, reason="r", requirement_id="a")
    with pytest.raises(ValidationError, match="names no constraint or repair"):
        Referral(kind=ReferralKind.NO_DATA, reason="r", requirement_id="a", repair="x")


def test_an_internal_failure_carries_its_reference() -> None:
    internal = Referral(kind=ReferralKind.INTERNAL_FAILURE, reason="r", reference="0123456789ab")
    assert internal.outcome is Outcome.INTERNAL
    with pytest.raises(ValidationError, match="carries its diagnostic reference"):
        Referral(kind=ReferralKind.INTERNAL_FAILURE, reason="r")
    with pytest.raises(ValidationError, match="carries no diagnostic reference"):
        Referral(kind=ReferralKind.SURFACE_UNAVAILABLE, reason="r", reference="0123456789ab")


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


def test_a_referral_stored_with_member_lists_is_no_referral(
    source: SourceFactory,
    stored_with_member_list: Callable[[SelectedSource], dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A session paused on the planner's question holds the referral that
    # prompted it. Its candidates were stored with fields `SelectedSource`
    # no longer has, so the referral is dropped and the plan is translated
    # again.
    two = (source(), source(spec_id="spec_" + "b" * 32))
    referral = Referral(
        kind=ReferralKind.SOURCE_AMBIGUOUS, reason="r", requirement_id="a", candidates=two
    )
    stored = referral.model_dump(mode="json")
    stored["candidates"] = [stored_with_member_list(candidate) for candidate in two]
    state: dict[str, Any] = {REFERRAL_STATE_KEY: json.dumps(stored)}

    with caplog.at_level(logging.WARNING, logger="translator.referral"):
        assert load_referral(state) is None

    (logged,) = caplog.records
    assert logged.levelno == logging.WARNING
    assert logged.args == (
        REFERRAL_STATE_KEY,
        1,
        "candidates.pinned, candidates.stat_var_truncated, candidates.stat_vars",
    )
