"""Tests for `translator.referral`: a referral names what it is about, and is saved."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import pytest
from compiler.compile import CompileIssueCode
from planner.plan import UnsupportedOperation
from planner.records import AmbiguousEntityReading, EntityCandidate, EntityKey
from planner.tool_results import Outcome
from planner.workflow import OpenEntityAxis, ResolvedEntity, ResolvedGroup, SelectedSource
from pydantic import ValidationError
from translator.referral import (
    ENTITY_CHOICE_REPAIR,
    OUTCOME_OF,
    REFERRAL_STATE_KEY,
    Referral,
    ReferralKind,
    WithheldSource,
    clear_referral,
    dropped_scope,
    load_referral,
    save_referral,
    scope_repair,
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
    with pytest.raises(ValidationError, match="names a repair only for a scope it withholds"):
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


def test_an_ambiguous_entity_carries_the_graph_nodes_it_could_name() -> None:
    def group(*refs: str) -> AmbiguousEntityReading:
        return AmbiguousEntityReading(
            text="Congo",
            candidates=tuple(EntityCandidate(ref=ref, label=ref) for ref in refs),
            entity=EntityKey(name="Congo", role="donor"),
        )

    two = (group("country/COD", "country/COG"),)
    referral = Referral(
        kind=ReferralKind.ENTITY_NAME_AMBIGUOUS,
        reason="r",
        requirement_id="a",
        ambiguous_entities=two,
        repair=ENTITY_CHOICE_REPAIR,
    )

    assert referral.outcome is Outcome.CHOICE_REQUIRED
    # Each candidate's pick names the requirement and the reading, a restated
    # group by its name, and reading the stored referral back keeps them.
    restated = Referral(
        kind=ReferralKind.ENTITY_NAME_AMBIGUOUS,
        reason="r",
        requirement_id="a",
        ambiguous_entities=(
            AmbiguousEntityReading(
                text="Central Africa",
                candidates=(
                    EntityCandidate(ref="region/UN", label="UN", restate_as="Central Africa (UN)"),
                    EntityCandidate(ref="region/AU", label="AU"),
                ),
                entity=EntityKey(name="Central Africa", role="recipient"),
            ),
            *two,
        ),
        repair=ENTITY_CHOICE_REPAIR,
    )
    assert [c.pick for r in restated.ambiguous_entities for c in r.candidates] == [
        "a|recipient|Central Africa|Central Africa (UN)",
        "a|recipient|Central Africa|region/AU",
        "a|donor|Congo|country/COD",
        "a|donor|Congo|country/COG",
    ]
    assert Referral.model_validate_json(restated.model_dump_json()) == restated
    with pytest.raises(ValidationError, match="carries the offered graph nodes"):
        Referral(
            kind=ReferralKind.ENTITY_NAME_AMBIGUOUS,
            reason="r",
            requirement_id="a",
            repair=ENTITY_CHOICE_REPAIR,
        )
    with pytest.raises(ValidationError, match="carries the offered graph nodes"):
        Referral(
            kind=ReferralKind.ENTITY_NAME_AMBIGUOUS,
            reason="r",
            requirement_id="a",
            ambiguous_entities=(group("country/COD"),),
            repair=ENTITY_CHOICE_REPAIR,
        )
    # The settle-or-ask guidance travels with the referral, and only with it.
    with pytest.raises(ValidationError, match="carries its own guidance as repair"):
        Referral(
            kind=ReferralKind.ENTITY_NAME_AMBIGUOUS,
            reason="r",
            requirement_id="a",
            ambiguous_entities=two,
        )
    with pytest.raises(ValidationError, match="only an entity_name_ambiguous referral"):
        Referral(kind=ReferralKind.NO_DATA, reason="r", requirement_id="a", ambiguous_entities=two)
    # A referral with no ambiguity is stored as it was before the field existed.
    plain = Referral(kind=ReferralKind.NO_DATA, reason="r", requirement_id="a")
    assert "ambiguous_entities" not in json.loads(plain.model_dump_json())


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


def test_an_unmeetable_requirement_carries_candidates_only_for_a_stated_source_not_found(
    source: SourceFactory,
) -> None:
    found = (source(),)

    assert (
        Referral(
            kind=ReferralKind.REQUIREMENT_UNMEETABLE,
            reason="r",
            requirement_id="a",
            candidates=found,
            source_mismatch=True,
        ).candidates
        == found
    )
    with pytest.raises(ValidationError, match="carries candidates only for a stated source"):
        Referral(
            kind=ReferralKind.REQUIREMENT_UNMEETABLE,
            reason="r",
            requirement_id="a",
            candidates=found,
        )
    with pytest.raises(ValidationError, match="never marked source_mismatch"):
        Referral(kind=ReferralKind.NO_DATA, reason="r", requirement_id="a", source_mismatch=True)


_FRANCE = ResolvedEntity(role="origin", ref="country/FRA", label="France")
_GHANA = ResolvedEntity(role="destination", ref="country/GHA", label="Ghana")
_KENYA = ResolvedEntity(role="destination", ref="country/KEN", label="Kenya")
_AFRICA = ResolvedGroup(role="destination", label="Africa", ref="africa")
_EVERY_RECIPIENT = (OpenEntityAxis(column="destination", member_count=266),)


@pytest.mark.parametrize(
    ("held", "found", "dropped"),
    [
        (
            {"entities": (_FRANCE, _GHANA, _KENYA), "groups": (_AFRICA,)},
            {"entities": (_FRANCE,), "open_entity_axes": _EVERY_RECIPIENT},
            "'destination': the pinned source holds it to Africa; this one leaves it open "
            "over 266 members",
        ),
        (
            {"entities": (_FRANCE, _GHANA, _KENYA)},
            {"entities": (_FRANCE, _GHANA)},
            "'destination': the pinned source holds it to 2 named members; this one binds 1 "
            "of those 2 members",
        ),
        (
            {"entities": (_FRANCE,), "groups": (_AFRICA,)},
            {"entities": (_FRANCE,)},
            "'destination': the pinned source holds it to Africa; this one names no Africa there",
        ),
        (
            {"entities": (_FRANCE, _GHANA, _KENYA), "groups": (_AFRICA,)},
            {"entities": (_FRANCE, _GHANA, _KENYA)},
            None,
        ),
        (
            {"entities": (_FRANCE,), "open_entity_axes": _EVERY_RECIPIENT},
            {"entities": (_FRANCE, _GHANA)},
            None,
        ),
    ],
    ids=["opens-a-group", "drops-named-members", "drops-a-group", "keeps-members", "narrows"],
)
def test_dropped_scope_names_each_column_the_found_source_widens_or_drops(
    source: SourceFactory,
    held: dict[str, Any],
    found: dict[str, Any],
    dropped: str | None,
) -> None:
    assert dropped_scope(source(**held), source(**found)) == ((dropped,) if dropped else ())


@pytest.mark.parametrize(
    ("held", "across", "repair"),
    [
        (
            {"entities": (_FRANCE, _GHANA, _KENYA), "groups": (_AFRICA,)},
            "recipient",
            "name 'Africa' as recipient in `entities` and drop 'recipient' from `across`",
        ),
        (
            {"entities": (_FRANCE, _GHANA)},
            "recipients",
            "name 'Ghana' as recipient in `entities` and drop 'recipients' from `across`",
        ),
        ({"entities": (_FRANCE, _GHANA, _KENYA)}, "recipient", None),
        ({"entities": (_FRANCE,), "groups": (_AFRICA,)}, "country", None),
    ],
    ids=["a-group", "one-member", "several-members", "no-side"],
)
def test_scope_repair_names_the_one_member_the_pinned_source_held(
    source: SourceFactory, held: dict[str, Any], across: str, repair: str | None
) -> None:
    assert scope_repair(source(**held), across) == repair


def test_a_withheld_source_is_never_offered(source: SourceFactory) -> None:
    kept, withheld = source(), source(spec_id="spec_" + "b" * 32)
    referral = Referral(
        kind=ReferralKind.PINNED_SOURCE_NOT_FOUND,
        reason="r",
        requirement_id="a",
        candidates=(kept,),
        withheld=(WithheldSource(source=withheld, dropped=("scope",)),),
    )

    assert referral.offered() == {"a": (kept,)}
    with pytest.raises(ValidationError, match="withholds no sources"):
        Referral(
            kind=ReferralKind.SOURCE_AMBIGUOUS,
            reason="r",
            requirement_id="a",
            candidates=(kept, kept),
            withheld=referral.withheld,
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
