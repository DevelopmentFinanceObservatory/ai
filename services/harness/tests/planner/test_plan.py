"""Tests for the `Plan` object: schema, versioning, status transitions, revisions."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from planner.operations import (
    AggregateOperation,
    GrowthOperation,
    RankOperation,
    RetrieveOperation,
    ShareOfOperation,
    Side,
    Translation,
)
from planner.plan import (
    CUT_MARKER,
    MAX_DECISION_DESCRIPTION_CHARS,
    MAX_DECISION_OPTIONS,
    MAX_DECISION_RATIONALE_CHARS,
    MAX_ENTITIES,
    MAX_ENTITY_NAME_CHARS,
    MAX_ENTITY_ROLE_CHARS,
    MAX_GROUP_BY,
    MAX_MEASURE_CHARS,
    MAX_OPTION_CHARS,
    MAX_PLAN_DECISIONS,
    MAX_PLAN_RECORD_CHARS,
    MAX_PREDICATES,
    MAX_QUESTION_CHARS,
    MAX_SOURCE_CHARS,
    PLAN_STATE_KEY,
    AggregateRequirement,
    AnswerMode,
    ComparisonRequirement,
    Coverage,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    EvidenceRequirement,
    FilterRequirement,
    GrowthRequirement,
    JoinRequirement,
    Period,
    Plan,
    PlanStatus,
    RankRequirement,
    RequirementSide,
    ResolvedBy,
    ShareOfRequirement,
    Unit,
    cut_user_text,
    load_plan,
    plan_revision,
)
from planner.proof import mismatches
from planner.records import (
    MAX_RANK_LIMIT,
    DenominatorScope,
    EntityIn,
    EvidenceIssue,
    EvidenceIssueCode,
    GroupBy,
    GrowthChange,
    PeriodThreshold,
    PlaceLevel,
    PlaceType,
    RankAmong,
    RankDirection,
    Reduction,
    Severity,
    TaggedModel,
    ValueThreshold,
)
from planner.workflow import (
    ArtifactKind,
    Binding,
    EvidenceBinding,
    SelectedSource,
    ToolId,
    Workflow,
    WorkflowStep,
)
from pydantic import TypeAdapter, ValidationError

from ..context_probe import field_limits_record, plan_record, record_chars, unbounded_paths

# The `decision`, `requirement`, `derived_requirement` and `requirements`
# fixtures, from the harness suite's conftest.py. Annotated structurally
# rather than imported: a test module importing a conftest by name is
# ambiguous when the workspace root has one too.
DecisionFactory = Callable[..., Decision]
RequirementFactory = Callable[..., DatasetRequirement]
DerivedRequirementFactory = Callable[..., AggregateRequirement]

_PIN = "spec_" + "a" * 32


def _plan(decision: Decision, **overrides: Any) -> Plan:
    """A plan awaiting the user, as the translator leaves it; a draft where a test says so."""
    fields: dict[str, Any] = {
        "question": "How dependent is Kenya on Chinese lending?",
        "decisions": [decision],
        "status": PlanStatus.PENDING_APPROVAL,
    }
    return Plan(**(fields | overrides))


def _at(plan: Plan, status: PlanStatus) -> Plan:
    """Place `plan` at `status` directly, for tests about what a status permits."""
    return plan.model_copy(update={"status": status})


def _every_kind_plan(decision: Decision) -> Plan:
    """Build a plan stating one requirement of every kind, wired into a valid DAG.

    `dataset_a` feeds `totals_a` (aggregate), which feeds `growth_a` (growth),
    and also feeds `rank_a` (rank, among='population', reducing the decade to
    one value per entity -- `decision` has to cite `fixes.among='population'`
    for `rank_a`'s population claim to validate). `dataset_b` feeds
    `filtered_b` (filter). `joined` (join), `share_a` (share_of) and
    `comparison_a` (comparison) each read both `totals_a` and `filtered_b`;
    `comparison_a` is terminal.
    """
    dataset_a = DatasetRequirement(
        id="dataset_a",
        decision_ids=["measure"],
        measure="external debt service paid",
        entities=[Entity(name="Kenya", role="debtor")],
        period=Period(start=2015, end=2024),
    )
    dataset_b = DatasetRequirement(
        id="dataset_b",
        decision_ids=["measure"],
        measure="government revenue",
        entities=[Entity(name="Kenya", role="creditor")],
        period=Period(start=2015, end=2024),
    )
    totals_a = AggregateRequirement(
        id="totals_a",
        decision_ids=["measure"],
        input_id="dataset_a",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    growth_a = GrowthRequirement(
        id="growth_a",
        decision_ids=["measure"],
        input_id="totals_a",
        window_start=2016,
        window_end=2020,
    )
    filtered_b = FilterRequirement(
        id="filtered_b",
        decision_ids=["measure"],
        input_id="dataset_b",
        predicates=(ValueThreshold(op="gt", value=0),),
    )
    rank_a = RankRequirement(
        id="rank_a",
        decision_ids=["measure"],
        input_id="dataset_a",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        reduction=Reduction.MEAN,
    )
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id="filtered_b", right_id="totals_a"
    )
    share_a = ShareOfRequirement(
        id="share_a",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="totals_a"),
        denominator=RequirementSide(input_id="filtered_b"),
    )
    comparison_a = ComparisonRequirement(
        id="comparison_a",
        decision_ids=["measure"],
        left=RequirementSide(input_id="totals_a"),
        right=RequirementSide(input_id="filtered_b"),
    )
    return _plan(
        decision,
        evidence_requirements=[
            dataset_a,
            dataset_b,
            totals_a,
            growth_a,
            filtered_b,
            rank_a,
            joined,
            share_a,
            comparison_a,
        ],
    )


def test_new_plan_defaults_to_a_draft_at_version_one(decision: DecisionFactory) -> None:
    # A plan starts where the planner leaves it: decisions recorded, nothing
    # to run yet.
    plan = Plan(question="q", decisions=[decision()])

    assert plan.version == 1
    assert plan.previous_version is None
    assert plan.status == PlanStatus.DRAFT
    assert plan.approved_revision is None
    assert plan.workflow is None


def test_decision_records_who_resolved_it(decision: DecisionFactory) -> None:
    user_resolved = decision(chosen="total debt stock", resolved_by=ResolvedBy.USER)

    assert user_resolved.resolved_by == ResolvedBy.USER
    assert user_resolved.chosen in user_resolved.options


# --- Immutability -----------------------------------------------------------


def test_plan_parts_are_frozen(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    # A revision is a value: `plan_revision` hashes it, so nothing may change
    # in place.
    plan = _plan(decision(), evidence_requirements=list(requirements))

    for frozen, field, value in [
        (plan, "question", "another question"),
        (plan.decisions[0], "chosen", "total debt stock"),
        (plan.evidence_requirements[0], "measure", "debt stock"),
        (plan.evidence_requirements[0].period, "end", 2030),
    ]:
        with pytest.raises(ValidationError, match="frozen"):
            setattr(frozen, field, value)


@pytest.mark.parametrize(
    ("build", "extra"),
    [
        (lambda d, r: Plan(question="q", decisions=[d], notes="x"), "notes"),
        (lambda d, r: d.model_copy(update={}).__class__(**d.model_dump(), weight=1), "weight"),
        (lambda d, r: Entity(name="Kenya", iso3="KEN"), "iso3"),
    ],
    ids=["plan", "decision", "entity"],
)
def test_plan_parts_reject_fields_they_do_not_declare(
    build: Callable[[Decision, DatasetRequirement], object],
    extra: str,
    decision: DecisionFactory,
    requirement: RequirementFactory,
) -> None:
    # Nothing rides along unnamed: a field the schema does not declare is a
    # mistake, never data to carry.
    with pytest.raises(ValidationError, match=extra):
        build(decision(), requirement())


# --- Versioning ---------------------------------------------------------------


def test_amend_returns_new_version_without_mutating_original(decision: DecisionFactory) -> None:
    original = _plan(decision())
    amended_decision = decision(chosen="total debt stock", resolved_by=ResolvedBy.USER)

    amended = original.amend(decisions=[amended_decision])

    # The original is untouched -- amend() never mutates `self`.
    assert original.version == 1
    assert original.decisions[0].chosen == "annual debt service"

    # The amendment is a distinct object with an incremented version and a
    # back-reference to what it amended.
    assert amended.version == 2
    assert amended.previous_version == 1
    assert amended.decisions[0].chosen == "total debt stock"
    assert amended.status == PlanStatus.DRAFT


def test_amend_preserves_unchanged_fields(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    original = _plan(decision(), evidence_requirements=list(requirements))

    amended = original.amend()

    assert amended.decisions == original.decisions
    assert amended.evidence_requirements == original.evidence_requirements


def test_repeated_amendment_chains_versions(decision: DecisionFactory) -> None:
    v1 = _plan(decision())
    v2 = v1.amend(decisions=[decision(chosen="total debt stock", resolved_by=ResolvedBy.USER)])
    v3 = v2.amend(decisions=[decision(chosen="annual debt service")])

    assert (v1.version, v2.version, v3.version) == (1, 2, 3)
    assert (v2.previous_version, v3.previous_version) == (1, 2)


def test_amend_revives_a_rejected_plan_as_a_new_version(decision: DecisionFactory) -> None:
    # Rejection stops this plan from running; it doesn't end
    # the lineage. A user who rejects and then asks for a change gets a new
    # version pointing back at the rejected one.
    rejected = _plan(decision()).reject()

    amended = rejected.amend(decisions=[decision(chosen="total debt stock")])

    assert rejected.status == PlanStatus.REJECTED
    assert amended.status == PlanStatus.DRAFT
    assert amended.previous_version == 1


def test_amend_clears_the_approved_revision(decision: DecisionFactory) -> None:
    # The user approved the previous version's content; the new content has
    # not been looked at.
    approved = _plan(decision()).approve()

    amended = approved.amend()

    assert approved.approved_revision is not None
    assert amended.approved_revision is None


def test_amend_validates_the_amended_plan_in_full(decision: DecisionFactory) -> None:
    # An amendment builds a new plan through the same validators as a first
    # draft, so it cannot introduce what a first draft could not. A patch
    # cannot even state a repeated slot. Two items under one key replace each
    # other, and the validator that refuses a repeat is reached by
    # constructing a `Plan` directly.
    plan = _plan(decision())

    amended = plan.amend(decisions=[decision(), decision(chosen="total debt stock")])

    assert [d.slot for d in amended.decisions] == ["measure"]
    with pytest.raises(ValidationError, match="decision slot must be unique"):
        Plan(question="q", decisions=[decision(), decision()])


def test_amend_replaces_a_decision_by_its_slot(decision: DecisionFactory) -> None:
    # A patch names the decision it changes by `slot`. The item under that
    # slot is replaced rather than appended, so patching one field of one
    # decision leaves the plan the same size.
    plan = _plan(decision())

    amended = plan.amend(
        decisions=[decision(chosen="total debt stock", resolved_by=ResolvedBy.USER)]
    )

    assert len(amended.decisions) == len(plan.decisions)
    assert amended.decisions[0].slot == "measure"
    assert amended.decisions[0].chosen == "total debt stock"


def test_amend_adds_a_decision_the_plan_does_not_state(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    plan = _plan(decision(), evidence_requirements=list(requirements))
    added = decision(
        slot="time_grain",
        options=["annual totals", "quarterly totals"],
        chosen="annual totals",
    )
    # Every decision has to be cited, so the patch cites the new one from a
    # requirement that already cites the first.
    dataset, derived = plan.evidence_requirements
    citing = [dataset.model_copy(update={"decision_ids": ("measure", "time_grain")}), derived]

    amended = plan.amend(decisions=[added], evidence_requirements=citing)

    assert [d.slot for d in amended.decisions] == ["measure", "time_grain"]


def test_amend_carries_forward_every_item_the_patch_does_not_name(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    # Removals have their own channel because an item left out of a patch
    # means unchanged, not deleted.
    plan = _plan(decision(), evidence_requirements=list(requirements))

    amended = plan.amend(decisions=[decision(rationale="One line, reworded.")])

    assert amended.decisions[0].slot == "measure"
    assert amended.evidence_requirements == plan.evidence_requirements


def test_amend_with_no_removals_keeps_every_decision_and_requirement_it_does_not_restate(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # Two independent chains. The patch restates one decision and one
    # requirement, names no removal, and the other chain is carried forward
    # whole and unchanged, in its original order.
    plan = _plan(
        decision(),
        decisions=[
            decision(),
            decision(slot="aid_measure", options=["gross ODA", "net ODA"], chosen="gross ODA"),
        ],
        evidence_requirements=[
            requirement(),
            derived_requirement(),
            requirement(id="aid", decision_ids=["aid_measure"], measure="aid disbursements"),
            derived_requirement(id="aid_total", decision_ids=["aid_measure"], input_id="aid"),
        ],
    )
    restated = requirement(measure="external debt service paid, all creditors")

    amended = plan.amend(
        decisions=[decision(rationale="One line, reworded.")], evidence_requirements=[restated]
    )

    assert [d.slot for d in amended.decisions] == ["measure", "aid_measure"]
    assert amended.decisions[1] == plan.decisions[1]
    assert [r.id for r in amended.evidence_requirements] == [
        "annual_debt_service",
        "annual_totals",
        "aid",
        "aid_total",
    ]
    assert amended.evidence_requirements[0] == restated
    assert amended.evidence_requirements[1:] == plan.evidence_requirements[1:]


def test_amend_records_the_question_it_is_given(decision: DecisionFactory) -> None:
    # The stored question has to agree with the requirements an amendment
    # restates, since the planner reads both back when it patches next.
    plan = _plan(decision())

    amended = plan.amend(question="How dependent is Kenya on lending from China since 2015?")

    assert amended.question == "How dependent is Kenya on lending from China since 2015?"
    assert plan.question == "How dependent is Kenya on Chinese lending?"
    assert plan_revision(amended) != plan_revision(plan.amend())


def test_amend_keeps_the_question_when_it_is_given_none(decision: DecisionFactory) -> None:
    plan = _plan(decision())

    assert plan.amend().question == plan.question


def test_amend_keeps_the_answer_mode_when_it_is_given_none(decision: DecisionFactory) -> None:
    plan = _plan(decision(), answer_mode=AnswerMode.SERIES)

    assert plan.amend().answer_mode == AnswerMode.SERIES


def test_amend_switches_the_answer_mode_when_it_is_given(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    plan = _plan(
        decision(), answer_mode=AnswerMode.COMPUTED, evidence_requirements=list(requirements)
    )
    assert plan.answer_mode == AnswerMode.COMPUTED

    amended = plan.amend(answer_mode=AnswerMode.SERIES)

    assert amended.answer_mode == AnswerMode.SERIES


def test_amend_switching_to_computed_revalidates_and_rejects_an_unread_requirement(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # A series plan with a lone, unread dataset requirement is legal; amending
    # it to computed re-validates the requirements under the new mode, and the
    # same unread requirement is rejected.
    unread = requirement()
    plan = _plan(decision(), answer_mode=AnswerMode.SERIES, evidence_requirements=[unread])

    with pytest.raises(ValidationError, match="unread dataset requirements"):
        plan.amend(answer_mode=AnswerMode.COMPUTED)


def test_amend_removes_a_named_decision_and_its_requirements(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    plan = _plan(decision(), evidence_requirements=list(requirements))
    dataset, derived = plan.evidence_requirements
    other = decision(
        slot="window",
        options=["since 2015", "since 2018"],
        chosen="since 2018",
    )
    without = [
        dataset.model_copy(update={"decision_ids": ("window",)}),
        derived.model_copy(update={"decision_ids": ("window",)}),
    ]

    amended = plan.amend(
        decisions=[other],
        evidence_requirements=without,
        removed_decisions=["measure"],
    )

    assert [d.slot for d in amended.decisions] == ["window"]


def test_amend_refuses_a_removal_naming_no_slot(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    # A removal naming nothing is a mistake to report, not a line silently
    # dropped from the patch. A model that mistypes a slot would otherwise
    # delete nothing and believe it had.
    plan = _plan(decision(), evidence_requirements=list(requirements))

    with pytest.raises(
        ValueError, match=r"every removed slot must name one the plan states; unknown: \['wndow'\]"
    ):
        plan.amend(removed_decisions=["wndow"])  # codespell:ignore wndow


# --- Requirements -------------------------------------------------------------


def test_a_plan_states_the_evidence_its_decisions_require(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    plan = _plan(decision(), evidence_requirements=list(requirements))

    dataset, derived = plan.evidence_requirements
    assert dataset.decision_ids == ("measure",)
    assert dataset.entities[0].name == "Kenya"
    assert dataset.period == Period(start=2015, end=2024)
    assert dataset.frequency == "annual"
    assert dataset.coverage.on_missing == "report_gaps"
    assert derived.input_id == dataset.id


def test_a_requirement_can_demand_every_year_or_fail(requirement: RequirementFactory) -> None:
    strict = requirement(coverage=Coverage(on_missing="fail"))

    assert strict.coverage.on_missing == "fail"
    assert DatasetRequirement.model_validate_json(strict.model_dump_json()) == strict


def test_a_plan_without_requirements_leaves_its_decisions_uncited(
    decision: DecisionFactory,
) -> None:
    # The citation rule applies once a plan states any requirement; a plan
    # that states none is still a plan.
    plan = _plan(decision())

    assert plan.evidence_requirements == ()


def test_a_requirement_must_cite_a_decision_the_plan_makes(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    with pytest.raises(ValidationError, match=r"does not make: \['time_grain'\]"):
        _plan(decision(), evidence_requirements=[requirement(decision_ids=["time_grain"])])


def test_every_decision_is_cited_once_requirements_are_stated(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # A decision no requirement rests on is a decision about nothing, or a
    # requirement the planner left out.
    uncited = decision(
        slot="time_grain", options=["annual totals", "quarterly totals"], chosen="annual totals"
    )

    with pytest.raises(ValidationError, match=r"no evidence requirement cites .*time_grain"):
        _plan(decision(), decisions=[decision(), uncited], evidence_requirements=[requirement()])


def test_decision_slots_must_be_unique(decision: DecisionFactory) -> None:
    # A requirement cites a decision by slot; two decisions sharing one would
    # leave the citation ambiguous.
    with pytest.raises(ValidationError, match=r"decision slot must be unique.*measure"):
        _plan(decision(), decisions=[decision(), decision(chosen="total debt stock")])


def test_requirement_ids_must_be_unique(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    with pytest.raises(
        ValidationError, match=r"requirement id must be unique.*annual_debt_service"
    ):
        _plan(decision(), evidence_requirements=[requirement(), requirement()])


@pytest.mark.parametrize("name", ["Measure", "time grain", "1st", ""], ids=repr)
def test_identifiers_are_lower_case_words(name: str, decision: DecisionFactory) -> None:
    # Slots and requirement ids are compared exactly, so their spelling is
    # closed down to what compares well.
    with pytest.raises(ValidationError, match="pattern"):
        decision(slot=name)


def test_a_period_cannot_end_before_it_starts() -> None:
    with pytest.raises(ValidationError, match=r"ends \(2014\) before it starts \(2015\)"):
        Period(start=2015, end=2014)


def test_a_period_may_be_a_single_year() -> None:
    assert Period(start=2020, end=2020).end == 2020


def test_constant_prices_need_a_base_year() -> None:
    with pytest.raises(ValidationError, match="constant prices need a base_year"):
        Unit(currency="USD", prices="constant")


def test_current_prices_take_no_base_year() -> None:
    with pytest.raises(ValidationError, match="current prices take no base_year"):
        Unit(currency="USD", prices="current", base_year=2015)


def test_a_currency_is_a_three_letter_code() -> None:
    with pytest.raises(ValidationError, match="pattern"):
        Unit(currency="dollars", prices="current")


def test_a_requirement_needs_an_entity_or_an_across(requirement: RequirementFactory) -> None:
    with pytest.raises(ValidationError, match="names at least one entity, or states in `across`"):
        requirement(entities=[])


def test_a_requirement_the_question_states_outright_cites_no_decision(
    requirement: RequirementFactory,
) -> None:
    """A measure and entity the question names are facts, not choices, so a
    requirement resting on nothing weighed cites no decision -- rather than
    a decision invented to be cited."""
    plan = Plan(
        question="Show me Kenya's external debt service.",
        answer_mode=AnswerMode.SERIES,
        evidence_requirements=[requirement(decision_ids=[])],
    )

    assert plan.evidence_requirements[0].decision_ids == ()


def test_a_pinned_source_citing_no_decision_is_still_refused(
    requirement: RequirementFactory,
) -> None:
    with pytest.raises(
        ValidationError, match="cites no decision that fixes that same source_spec_id"
    ):
        Plan(
            question="Show me Kenya's external debt service.",
            answer_mode=AnswerMode.SERIES,
            evidence_requirements=[requirement(decision_ids=[], source_spec_id=_PIN)],
        )


def test_a_requirement_may_cover_a_whole_party_and_name_no_entity() -> None:
    dataset = DatasetRequirement(
        id="health_spending",
        decision_ids=["measure"],
        measure="health spending per person",
        across="country",
        period=Period(start=2022, end=2022),
    )

    assert dataset.entities == ()
    assert dataset.across == "country"


@pytest.mark.parametrize(
    ("role", "across"),
    [
        ("recipient", "recipient"),
        ("Recipient", "recipient"),
        ("recipients", "recipient"),
        ("recipient", "recipients"),
        ("debtor", "recipient"),
        ("recipient country", "recipient"),
        ("country", "countries"),
    ],
)
def test_a_requirement_names_no_entity_in_the_party_across_covers(
    requirement: RequirementFactory, role: str, across: str
) -> None:
    # Kenya as a recipient, across every recipient, claims two populations at
    # once. The two are compared by the party each names, not by spelling.
    every = "country" if across == "countries" else "recipient"
    with pytest.raises(
        ValidationError,
        match=rf"across='{across}' covers every {every}, so no entity may also be named in "
        r"that party; named: \['Kenya'\]",
    ):
        requirement(entities=[Entity(name="Kenya", role=role)], across=across)


@pytest.mark.parametrize("across", ["country", "countries", "economy"])
def test_a_requirement_names_no_unroled_entity_beside_every_place(
    requirement: RequirementFactory, across: str
) -> None:
    # "Where does Kenya rank among countries": Kenya binds the place axis,
    # so the source leaves no population open for `across` to cover.
    with pytest.raises(
        ValidationError,
        match=r"\['Kenya'\] carry no `role`, so each names one of them and leaves no "
        r"population open\. To rank every member, drop them; to rank them against every "
        r"member, drop `across` and state the rank's among='population'",
    ):
        requirement(entities=[Entity(name="Kenya")], across=across)


def test_an_unroled_entity_may_sit_beside_a_flow_party(requirement: RequirementFactory) -> None:
    dataset = requirement(entities=[Entity(name="Kenya")], across="donor")

    assert dataset.across == "donor"


def test_a_requirement_names_an_entity_beside_the_party_across_covers(
    requirement: RequirementFactory,
) -> None:
    dataset = requirement(entities=[Entity(name="Kenya", role="recipient")], across="donor")

    assert dataset.across == "donor"


# --- Derived requirement validation --------------------------------------------


def test_dataset_entities_must_be_unique_by_name_and_role(requirement: RequirementFactory) -> None:
    with pytest.raises(ValidationError, match="unique by name and role"):
        requirement(
            entities=[
                Entity(name="Kenya", role="debtor"),
                Entity(name="Kenya", role="debtor"),
            ]
        )


def test_growth_window_must_end_after_it_starts() -> None:
    with pytest.raises(ValidationError, match="must end after it starts"):
        GrowthRequirement(
            id="growth",
            decision_ids=["measure"],
            input_id="annual_debt_service",
            window_start=2020,
            window_end=2020,
        )


def test_rank_window_needs_both_years_or_neither() -> None:
    with pytest.raises(ValidationError, match="needs both window_start and window_end, or neither"):
        RankRequirement(
            id="rank",
            decision_ids=["measure"],
            input_id="annual_debt_service",
            direction=RankDirection.HIGHEST_FIRST,
            among=RankAmong.POPULATION,
            window_start=2020,
        )


def test_a_rank_needs_among_stated() -> None:
    with pytest.raises(ValidationError, match="among"):
        RankRequirement(
            id="rank",
            decision_ids=["measure"],
            input_id="annual_debt_service",
            direction=RankDirection.HIGHEST_FIRST,
        )


def test_comparison_sides_must_differ() -> None:
    with pytest.raises(ValidationError, match="left and right sides must differ"):
        ComparisonRequirement(
            id="excess",
            decision_ids=["measure"],
            left=RequirementSide(input_id="annual_debt_service"),
            right=RequirementSide(input_id="annual_debt_service"),
        )


def test_share_of_sides_must_differ() -> None:
    with pytest.raises(ValidationError, match="numerator and denominator must differ"):
        ShareOfRequirement(
            id="share",
            decision_ids=["measure"],
            numerator=RequirementSide(input_id="annual_debt_service"),
            denominator=RequirementSide(input_id="annual_debt_service"),
        )


def test_join_ids_must_be_distinct() -> None:
    with pytest.raises(ValidationError, match="left_id and right_id must differ"):
        JoinRequirement(id="joined", decision_ids=["measure"], left_id="a", right_id="a")


def test_filter_needs_at_least_one_predicate() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        FilterRequirement(
            id="filtered",
            decision_ids=["measure"],
            input_id="annual_debt_service",
            predicates=(),
        )


# --- Requirement graph validation -----------------------------------------------


def test_a_derived_requirement_must_read_a_requirement_the_plan_states(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    orphan = AggregateRequirement(
        id="totals",
        decision_ids=["measure"],
        input_id="missing_requirement",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )

    with pytest.raises(ValidationError, match=r"does not state: \['missing_requirement'\]"):
        _plan(decision(), evidence_requirements=[dataset, orphan])


def test_evidence_requirements_must_form_a_dag(decision: DecisionFactory) -> None:
    agg_a = AggregateRequirement(
        id="agg_a",
        decision_ids=["measure"],
        input_id="agg_b",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    agg_b = AggregateRequirement(
        id="agg_b",
        decision_ids=["measure"],
        input_id="agg_a",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )

    with pytest.raises(ValidationError, match="do not form a DAG"):
        _plan(decision(), evidence_requirements=[agg_a, agg_b])


def test_nothing_may_read_a_comparison(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    dataset = requirement()
    agg = derived_requirement()
    comparison = ComparisonRequirement(
        id="excess",
        decision_ids=["measure"],
        left=RequirementSide(input_id=dataset.id),
        right=RequirementSide(input_id=agg.id),
    )
    reads_comparison = AggregateRequirement(
        id="totals_of_excess",
        decision_ids=["measure"],
        input_id="excess",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )

    with pytest.raises(ValidationError, match="feeds nothing further"):
        _plan(decision(), evidence_requirements=[dataset, agg, comparison, reads_comparison])


def test_a_computed_plan_rejects_an_unread_dataset_requirement(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # Under answer_mode=computed, a retrieved series carries a reference and
    # statistics, not the figure a computed answer asks for, so a dataset
    # requirement nothing reads leaves the plan without one.
    unread = requirement()

    with pytest.raises(ValidationError, match="unread dataset requirements"):
        _plan(decision(), answer_mode=AnswerMode.COMPUTED, evidence_requirements=[unread])


def test_a_series_plan_accepts_a_lone_dataset_requirement(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # Under answer_mode=series, the requirement itself is the answer. A
    # question that asks to see a series for named entities is satisfied by
    # the retrieval, whose frame carries one series per entity.
    unread = requirement()

    plan = _plan(decision(), answer_mode=AnswerMode.SERIES, evidence_requirements=[unread])

    assert plan.requirement(unread.id).id == unread.id


def test_a_series_plan_accepts_an_unread_retrieval_beside_a_derived_terminal(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # A plan can pair one calculation over one retrieval with a second,
    # separate retrieval nothing reads under answer_mode=series -- such as
    # asking for Kenya's total debt service and its population series in one
    # turn.
    dataset = requirement()
    derived = derived_requirement()
    population = requirement(id="population", measure="population")

    plan = _plan(
        decision(),
        answer_mode=AnswerMode.SERIES,
        evidence_requirements=[dataset, derived, population],
    )

    assert {r.id for r in plan.evidence_requirements} == {dataset.id, derived.id, population.id}


def test_a_requirement_side_entity_must_name_an_upstream_entity(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    dataset = requirement()  # entities=[Entity(name="Kenya", role="debtor")]
    # Grouped by entity too, so its shape keeps the "debtor" role dimension
    # the check below reads names off of; grouped by period alone would pool
    # entities away and leave nothing to compare against.
    agg = derived_requirement(group_by=(GroupBy.ENTITY, GroupBy.PERIOD))
    other_agg = AggregateRequirement(
        id="other_totals",
        decision_ids=["measure"],
        input_id=dataset.id,
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    comparison = ComparisonRequirement(
        id="excess",
        decision_ids=["measure"],
        left=RequirementSide(input_id=agg.id, entity="Nigeria"),
        right=RequirementSide(input_id=other_agg.id),
    )

    with pytest.raises(ValidationError, match=r"not one of \['Kenya'\]"):
        _plan(decision(), evidence_requirements=[dataset, agg, other_agg, comparison])


def test_a_comparison_reading_a_join_is_rejected(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # `join` is terminal by contract: nothing reads one, including as one
    # side of a comparison. The read itself is rejected first, before any
    # entity check runs.
    dataset_a = requirement()
    dataset_b = requirement(id="other_dataset", entities=[Entity(name="Nigeria", role="creditor")])
    agg_a = derived_requirement()  # reads dataset_a, id "annual_totals"
    agg_b = AggregateRequirement(
        id="other_totals",
        decision_ids=["measure"],
        input_id=dataset_b.id,
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id=agg_a.id, right_id=agg_b.id
    )
    comparison = ComparisonRequirement(
        id="excess",
        decision_ids=["measure"],
        left=RequirementSide(input_id=joined.id, entity="Not a real entity"),
        right=RequirementSide(input_id=agg_a.id),
    )

    with pytest.raises(ValidationError, match=r"feeds nothing further.*\('joined', 'join'\)"):
        _plan(
            decision(),
            evidence_requirements=[dataset_a, dataset_b, agg_a, agg_b, joined, comparison],
        )


def test_growth_window_must_fit_the_upstream_dataset_period(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    dataset = requirement()  # period 2015-2024
    agg = derived_requirement()  # reads dataset, id "annual_totals"
    growth = GrowthRequirement(
        id="growth",
        decision_ids=["measure"],
        input_id=agg.id,
        window_start=2010,
        window_end=2020,
    )

    with pytest.raises(ValidationError, match="falls outside"):
        _plan(decision(), evidence_requirements=[dataset, agg, growth])


def test_a_growth_reading_a_join_is_rejected(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # `join` is terminal by contract: nothing reads one. The read itself is
    # rejected first, before the window is even checked.
    dataset_a = requirement()
    dataset_b = requirement(id="other_dataset", entities=[Entity(name="Nigeria", role="creditor")])
    agg_a = derived_requirement()
    agg_b = AggregateRequirement(
        id="other_totals",
        decision_ids=["measure"],
        input_id=dataset_b.id,
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id=agg_a.id, right_id=agg_b.id
    )
    growth = GrowthRequirement(
        id="growth",
        decision_ids=["measure"],
        input_id=joined.id,
        window_start=1900,
        window_end=2100,
    )

    with pytest.raises(ValidationError, match=r"feeds nothing further.*\('joined', 'join'\)"):
        _plan(
            decision(), evidence_requirements=[dataset_a, dataset_b, agg_a, agg_b, joined, growth]
        )


def test_a_growth_window_is_checked_through_a_share_of(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # `shape.period` carries the two sides' period intersection through
    # every single-input hop that follows, so a growth several hops past a
    # `share_of` (through an aggregate and a filter) has its window checked
    # against that intersection, with no upstream walk and nothing skipped.
    numerator = requirement()  # period 2015-2024
    denominator = requirement(
        id="other_dataset",
        entities=[Entity(name="Kenya", role="creditor")],
        period=Period(start=2010, end=2022),
    )
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=numerator.id),
        denominator=RequirementSide(input_id=denominator.id),
    )
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["measure"],
        input_id="share",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    recent = FilterRequirement(
        id="recent",
        decision_ids=["measure"],
        input_id="totals",
        predicates=(PeriodThreshold(op="gt", year=2016),),
    )
    growth = GrowthRequirement(
        id="growth",
        decision_ids=["measure"],
        input_id="recent",
        window_start=2010,
        window_end=2020,
        change=GrowthChange.ABSOLUTE,  # a share_of reads as RATIO; relative would refuse first
    )

    # The share_of's period is 2015-2022 (the two sides' intersection), so a
    # window starting in 2010 falls outside it.
    with pytest.raises(ValidationError, match="falls outside"):
        _plan(
            decision(),
            evidence_requirements=[numerator, denominator, share, totals, recent, growth],
        )


def _rank(input_id: str, **overrides: Any) -> RankRequirement:
    fields: dict[str, Any] = {
        "id": "ranked",
        "decision_ids": ["measure"],
        "input_id": input_id,
        "direction": RankDirection.HIGHEST_FIRST,
        "among": RankAmong.POPULATION,
    }
    return RankRequirement(**(fields | overrides))


def _population_decision(decision: DecisionFactory, **overrides: Any) -> Decision:
    """A `measure` decision that also fixes `among='population'`, for `_rank`'s default.

    Every reduction test below ranks a single-entity dataset with no
    narrowing filter, so `among='population'` (rather than `named`, which
    needs a narrowed axis) is the reading that leaves the reduction rule
    the only thing under test.
    """
    return decision(fixes=DecisionFixes(among=RankAmong.POPULATION), **overrides)


def test_a_rank_over_several_years_per_entity_states_a_reduction(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # Ten rows per entity and no reduction leaves nothing to rank an entity by.
    dataset = requirement()  # period 2015-2024

    with pytest.raises(ValidationError, match="no entity has a single value to rank on"):
        _plan(_population_decision(decision), evidence_requirements=[dataset, _rank(dataset.id)])


def test_a_rank_window_spanning_several_years_states_a_reduction(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()

    with pytest.raises(ValidationError, match="over 2016-2018 with no reduction"):
        _plan(
            _population_decision(decision),
            evidence_requirements=[dataset, _rank(dataset.id, window_start=2016, window_end=2018)],
        )


@pytest.mark.parametrize(
    ("period", "window"),
    [
        pytest.param(Period(start=2024, end=2024), None, id="one-year-input"),
        pytest.param(
            Period(start=2015, end=2024), Period(start=2024, end=2030), id="window-keeps-one-year"
        ),
    ],
)
def test_a_rank_over_a_single_year_needs_no_reduction(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    period: Period,
    window: Period | None,
) -> None:
    # run_rank with no reduction ranks each entity's one row in the years
    # the window keeps of its input, 2024 under a 2024-2030 window.
    dataset = requirement(period=period)
    bounds = {} if window is None else {"window_start": window.start, "window_end": window.end}

    plan = _plan(
        _population_decision(decision),
        evidence_requirements=[dataset, _rank(dataset.id, **bounds)],
    )

    assert plan.requirement("ranked").reduction is None


@pytest.mark.parametrize("limit", [None, 5])
def test_a_rank_over_an_entity_grouped_aggregate_ranks_its_span(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    limit: int | None,
) -> None:
    # An aggregate grouped by entity alone holds one total per entity,
    # labelled with the span of years it collapsed, so a rank orders those
    # totals with no reduction of its own.
    dataset = requirement()
    per_entity = derived_requirement(group_by=(GroupBy.ENTITY,))

    plan = _plan(
        _population_decision(decision),
        evidence_requirements=[dataset, per_entity, _rank(per_entity.id, limit=limit)],
    )

    assert plan.requirement("ranked").input_id == per_entity.id


_DECADE = Period(start=2015, end=2024)


@pytest.mark.parametrize(
    ("reduction", "right_is_peak", "right_period", "refused"),
    [
        pytest.param(Reduction.MAX, False, _DECADE, None, id="max-vs-series"),
        pytest.param(Reduction.MIN, False, _DECADE, None, id="min-vs-series"),
        pytest.param(
            Reduction.MAX,
            False,
            Period(start=2015, end=2019),
            "no year to match",
            id="max-vs-a-series-missing-picked-years",
        ),
        pytest.param(Reduction.SUM, False, _DECADE, "carry single years", id="sum-labels-the-span"),
        pytest.param(Reduction.MAX, True, _DECADE, "both sides hold", id="max-vs-max"),
    ],
)
def test_a_comparison_reads_an_extreme_by_entity_as_single_years(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    reduction: Reduction,
    right_is_peak: bool,
    right_period: Period,
    refused: str | None,
) -> None:
    # analysis-mcp labels a max or min row with the year its value came from,
    # so the peak over 2018-2022 compares against another source's years,
    # which have to hold every year a peak may fall in; a sum is labelled
    # 2018/2022, which matches no single year, and two peaks meet only where
    # both fell in the same year.
    dataset = requirement(period=Period(start=2018, end=2022))
    peak = derived_requirement(reduction=reduction, group_by=(GroupBy.ENTITY,))
    population = requirement(id="population", measure="population", period=right_period)
    requirements: list[EvidenceRequirement] = [dataset, peak, population]
    right_id = population.id
    if right_is_peak:
        population_peak = derived_requirement(
            id="population_peak",
            input_id=population.id,
            reduction=reduction,
            group_by=(GroupBy.ENTITY,),
        )
        requirements.append(population_peak)
        right_id = population_peak.id
    comparison = ComparisonRequirement(
        id="excess",
        decision_ids=["measure"],
        left=RequirementSide(input_id=peak.id),
        right=RequirementSide(input_id=right_id),
    )
    requirements.append(comparison)

    if refused is not None:
        with pytest.raises(ValidationError, match=refused):
            _plan(decision(), evidence_requirements=requirements)
    else:
        plan = _plan(decision(), evidence_requirements=requirements)
        assert plan.requirement("excess").left.input_id == peak.id


def _after_peak(kind: str) -> EvidenceRequirement:
    """A `kind` step reading `peak`, the debt service's max by entity over 2018-2022."""
    common: dict[str, Any] = {"id": "reader", "decision_ids": ["measure"], "input_id": "peak"}
    if kind == "growth":
        return GrowthRequirement(**common, window_start=2018, window_end=2022)
    if kind == "per-period-rank":
        return _rank("peak", id="reader", per_period=True)
    if kind == "collapsing-rank":
        return _rank("peak", id="reader")
    if kind == "aggregate-by-period":
        return AggregateRequirement(**common, reduction=Reduction.SUM, group_by=(GroupBy.PERIOD,))
    return FilterRequirement(**common, predicates=(PeriodThreshold(op="gte", year=2020),))


@pytest.mark.parametrize(
    ("kind", "refused"),
    [
        pytest.param("growth", "growth reads each entity's series", id="growth-after-max"),
        pytest.param("per-period-rank", "rank counts every year", id="per-period-rank-after-max"),
        pytest.param("collapsing-rank", "rank counts every year", id="collapsing-rank-after-max"),
        pytest.param(
            "aggregate-by-period", "aggregate by period", id="aggregate-by-period-after-max"
        ),
        pytest.param("year-filter", None, id="year-filter-after-max"),
    ],
)
def test_a_reader_of_an_extreme_by_entity_follows_its_picked_years(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    kind: str,
    refused: str | None,
) -> None:
    # analysis-mcp holds a max by entity as one row per entity, at the year
    # its peak fell in: no series to grow, no year whose entities a per-year
    # reader could set side by side, and one year of each entity's window for
    # a rank's coverage to count.
    dataset = requirement(period=Period(start=2018, end=2022))
    peak = derived_requirement(id="peak", reduction=Reduction.MAX, group_by=(GroupBy.ENTITY,))
    requirements = [dataset, peak, _after_peak(kind)]

    if refused is not None:
        with pytest.raises(ValidationError, match=refused):
            _plan(_population_decision(decision), evidence_requirements=requirements)
    else:
        plan = _plan(_population_decision(decision), evidence_requirements=requirements)
        assert plan.requirement("reader").input_id == peak.id


@pytest.mark.parametrize(
    ("window", "limit"),
    [
        pytest.param({}, None, id="no-window"),
        pytest.param({}, 5, id="no-window-limited"),
        pytest.param({"window_start": 2014, "window_end": 2024}, 5, id="the-growth-window"),
    ],
)
def test_a_rank_over_a_growth_needs_no_reduction(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    window: dict[str, int],
    limit: int | None,
) -> None:
    # Growth states one figure per entity over its window, so the decade the
    # dataset spans is already behind it -- "which grew fastest" ranks that,
    # with no window or with the growth's own.
    dataset = requirement(period=Period(start=2014, end=2024))
    growth = GrowthRequirement(
        id="grew",
        decision_ids=("measure",),
        input_id=dataset.id,
        window_start=2014,
        window_end=2024,
    )

    plan = _plan(
        _population_decision(decision),
        evidence_requirements=[dataset, growth, _rank(growth.id, limit=limit, **window)],
    )

    assert plan.requirement("ranked").input_id == growth.id


def test_a_rank_window_cutting_a_growths_span_is_refused(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # Every growth row carries the span 2014/2024; a 2018-2024 window holds no
    # row of it, so analysis-mcp would refuse the rank.
    dataset = requirement(period=Period(start=2014, end=2024))
    growth = GrowthRequirement(
        id="grew",
        decision_ids=("measure",),
        input_id=dataset.id,
        window_start=2014,
        window_end=2024,
    )
    cut = _rank(growth.id, window_start=2018, window_end=2024)

    with pytest.raises(ValidationError, match="a 2018-2024 window cuts"):
        _plan(_population_decision(decision), evidence_requirements=[dataset, growth, cut])


def test_a_rank_over_a_filter_pinning_one_year_needs_no_reduction(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    one_year = FilterRequirement(
        id="in_2020",
        decision_ids=("measure",),
        input_id=dataset.id,
        predicates=(PeriodThreshold(op="eq", year=2020),),
    )

    plan = _plan(
        _population_decision(decision),
        evidence_requirements=[dataset, one_year, _rank(one_year.id)],
    )

    assert plan.requirement("ranked").input_id == one_year.id


def test_a_rank_over_a_filter_closing_on_one_year_needs_no_reduction(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # `gte 2020` with `lte 2020` admits 2020 and nothing else, so the span is
    # as collapsed as an `eq` would leave it.
    dataset = requirement()
    one_year = FilterRequirement(
        id="in_2020",
        decision_ids=("measure",),
        input_id=dataset.id,
        predicates=(
            PeriodThreshold(op="gte", year=2020),
            PeriodThreshold(op="lte", year=2020),
        ),
    )

    plan = _plan(
        _population_decision(decision),
        evidence_requirements=[dataset, one_year, _rank(one_year.id)],
    )

    assert plan.requirement("ranked").input_id == one_year.id


def test_a_rank_over_a_filter_that_narrows_without_pinning_a_year_states_a_reduction(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # Later than 2018 is still several years, so an entity has no one value.
    dataset = requirement()
    narrowed = FilterRequirement(
        id="recent",
        decision_ids=("measure",),
        input_id=dataset.id,
        predicates=(PeriodThreshold(op="gt", year=2018),),
    )

    with pytest.raises(ValidationError, match="no entity has a single value to rank on"):
        _plan(
            _population_decision(decision),
            evidence_requirements=[dataset, narrowed, _rank(narrowed.id)],
        )


def test_a_rank_with_a_reduction_ranks_any_span(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()

    plan = _plan(
        _population_decision(decision),
        evidence_requirements=[dataset, _rank(dataset.id, reduction=Reduction.MEAN)],
    )

    assert plan.requirement("ranked").reduction == Reduction.MEAN


@pytest.mark.parametrize("axis", ["country", None])
def test_a_rank_over_the_party_across_covers_validates(
    axis: str | None, decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # "Top countries by health spending": no country is named, and the lone
    # role dimension is the one `across` states. The retrieval holds every
    # level of place, so a filter keeps the countries first.
    dataset = requirement(entities=[], across="country", period=Period(start=2022, end=2022))

    plan = _plan(
        _population_decision(decision),
        evidence_requirements=[dataset, _countries(dataset.id), _rank("countries", axis=axis)],
    )

    assert plan.requirement("ranked").axis == axis


# --- A requirement that names no place holds every level of place ------------------


def _countries(input_id: str, **overrides: Any) -> FilterRequirement:
    fields: dict[str, Any] = {
        "id": "countries",
        "decision_ids": ["measure"],
        "input_id": input_id,
        "predicates": (PlaceLevel(level=PlaceType.COUNTRY),),
    }
    return FilterRequirement(**(fields | overrides))


def _every_place(requirement: RequirementFactory, **overrides: Any) -> DatasetRequirement:
    """GDP with no place named, over one year unless `overrides` says otherwise."""
    fields: dict[str, Any] = {
        "entities": [],
        "across": "country",
        "period": Period(start=2022, end=2022),
    }
    return requirement(**(fields | overrides))


_EVERY_LEVEL = "holds every place with data at every level"


@pytest.mark.parametrize("across", ["country", "countries", "place", "economy"])
def test_a_rank_over_a_requirement_naming_no_place_needs_a_level_first(
    across: str, decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # "Top 10 by GDP": the retrieval holds World, regions and World Bank
    # groups beside the countries, so a rank over it would place them together.
    dataset = _every_place(requirement, across=across)

    with pytest.raises(ValidationError, match=f"'ranked': .*{_EVERY_LEVEL}.*place_level"):
        _plan(
            _population_decision(decision),
            evidence_requirements=[dataset, _rank(dataset.id, limit=10)],
        )


def test_a_total_across_places_needs_a_level_first_and_a_series_per_place_does_not(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # "Total population": summing every row counts a country beside its own
    # subdivisions and the world. One total per place sets nothing side by side.
    dataset = _every_place(requirement, period=Period(start=2015, end=2024))
    total = AggregateRequirement(
        id="total",
        decision_ids=["measure"],
        input_id=dataset.id,
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )

    with pytest.raises(ValidationError, match=f"'total': .*{_EVERY_LEVEL}.*a total or an average"):
        _plan(decision(), evidence_requirements=[dataset, total])

    by_country = total.model_copy(update={"input_id": "countries"})
    plan = _plan(decision(), evidence_requirements=[dataset, _countries(dataset.id), by_country])
    assert plan.requirement("total").input_id == "countries"

    per_place = total.model_copy(update={"group_by": (GroupBy.ENTITY,)})
    plan = _plan(decision(), evidence_requirements=[dataset, per_place])
    assert plan.requirement("total").group_by == (GroupBy.ENTITY,)


def test_a_rank_over_each_places_growth_still_needs_a_level(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # "Fastest-growing by GDP": a growth per place keeps every level, and it
    # drops the type each place carries, so the level filter has to read the
    # retrieval itself.
    dataset = _every_place(requirement, period=Period(start=2015, end=2024))
    growth = GrowthRequirement(
        id="growth",
        decision_ids=["measure"],
        input_id=dataset.id,
        window_start=2015,
        window_end=2024,
    )

    with pytest.raises(ValidationError, match=f"'ranked': .*{_EVERY_LEVEL}"):
        _plan(
            _population_decision(decision),
            evidence_requirements=[dataset, growth, _rank("growth")],
        )
    with pytest.raises(ValidationError, match="'countries': a place_level filter reads"):
        _plan(
            _population_decision(decision),
            evidence_requirements=[dataset, growth, _countries("growth"), _rank("countries")],
        )

    growth_of_countries = growth.model_copy(update={"input_id": "countries"})
    plan = _plan(
        _population_decision(decision),
        evidence_requirements=[
            dataset,
            _countries(dataset.id),
            growth_of_countries,
            _rank("growth"),
        ],
    )
    assert plan.requirement("ranked").input_id == "growth"


def test_a_share_over_a_requirement_naming_no_place_needs_a_level_first(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = _every_place(requirement)
    world = requirement(id="world", entities=[Entity(name="World")], period=dataset.period)
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=dataset.id),
        denominator=RequirementSide(input_id="world"),
        axis="country",
        denominator_scope=DenominatorScope.POOLED,
    )

    with pytest.raises(ValidationError, match=f"'share': .*{_EVERY_LEVEL}.*a share"):
        _plan(
            decision(fixes=DecisionFixes(denominator_scope=DenominatorScope.POOLED)),
            evidence_requirements=[dataset, world, share],
        )


def test_a_level_filter_reads_through_another_filter(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = _every_place(requirement, period=Period(start=2015, end=2024))
    recent = FilterRequirement(
        id="recent",
        decision_ids=["measure"],
        input_id=dataset.id,
        predicates=(PeriodThreshold(op="eq", year=2024),),
    )

    plan = _plan(
        _population_decision(decision),
        evidence_requirements=[dataset, recent, _countries("recent"), _rank("countries")],
    )

    assert plan.requirement("countries").input_id == "recent"


def test_a_level_filter_over_named_places_is_refused(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # "GDP of the states of India": the states are named, so there is no
    # level to choose and the frame carries no type to choose it by.
    dataset = requirement()

    with pytest.raises(
        ValidationError, match="a place_level filter reads a requirement that names no place"
    ):
        _plan(decision(), evidence_requirements=[dataset, _countries(dataset.id)])


def test_a_filter_keeps_one_level() -> None:
    with pytest.raises(ValidationError, match=r"one place_level; got \['Country', 'Continent'\]"):
        _countries(
            "gdp",
            predicates=(
                PlaceLevel(level=PlaceType.COUNTRY),
                PlaceLevel(level=PlaceType.CONTINENT),
            ),
        )


def test_a_series_of_every_place_needs_no_level(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    plan = _plan(decision(), evidence_requirements=[_every_place(requirement)])

    assert plan.answer_mode is AnswerMode.SERIES


def test_a_decision_fixing_the_level_agrees_with_the_filter_citing_it(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = _every_place(requirement)
    level = decision(
        slot="level",
        options=["countries", "continents"],
        chosen="countries",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        fixes=DecisionFixes(predicates=(PlaceLevel(level=PlaceType.CONTINENT),)),
    )

    with pytest.raises(ValidationError, match="predicates"):
        Plan(
            question="top 10 by GDP",
            decisions=[_population_decision(decision), level],
            evidence_requirements=[
                dataset,
                _countries(dataset.id, decision_ids=["level"]),
                _rank("countries"),
            ],
        )


def test_a_rank_over_across_beside_a_named_entity_states_a_reduction(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # "Top donors to Kenya": Kenya's recipient role and every donor are two
    # role dimensions, so ranking donors collapses the recipient rows first.
    dataset = requirement(
        entities=[Entity(name="Kenya", role="recipient")],
        across="donor",
        period=Period(start=2022, end=2022),
    )

    with pytest.raises(ValidationError, match="more than one role, which needs a reduction"):
        _plan(
            _population_decision(decision),
            evidence_requirements=[dataset, _rank(dataset.id, axis="donor")],
        )
    with pytest.raises(ValidationError, match=r"roles are \['donor', 'recipient'\]"):
        _plan(
            _population_decision(decision),
            evidence_requirements=[
                dataset,
                _rank(dataset.id, axis="creditor", reduction=Reduction.SUM),
            ],
        )
    plan = _plan(
        _population_decision(decision),
        evidence_requirements=[
            dataset,
            _rank(dataset.id, axis="donor", reduction=Reduction.SUM),
        ],
    )

    assert plan.requirement("ranked").reduction == Reduction.SUM


def test_a_requirement_reading_a_join_is_rejected(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # `join` is terminal by contract: nothing reads one. The read is
    # rejected before any shape is even built.
    dataset_a = requirement()
    dataset_b = requirement(id="other_dataset", entities=[Entity(name="Nigeria", role="creditor")])
    agg_a = derived_requirement()
    agg_b = AggregateRequirement(
        id="other_totals",
        decision_ids=["measure"],
        input_id=dataset_b.id,
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id=agg_a.id, right_id=agg_b.id
    )

    with pytest.raises(ValidationError, match=r"feeds nothing further.*\('joined', 'join'\)"):
        _plan(
            decision(),
            evidence_requirements=[dataset_a, dataset_b, agg_a, agg_b, joined, _rank(joined.id)],
        )


# --- The frame-shape walk (planner.shapes, driven from requirements) -----------


def test_a_rank_reading_a_share_of_validates(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    fixed = decision(
        fixes=DecisionFixes(denominator_scope=DenominatorScope.POOLED, among=RankAmong.POPULATION)
    )
    numerator = requirement()  # role="debtor"
    denominator = requirement(id="other_dataset", entities=[Entity(name="Kenya", role="creditor")])
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=numerator.id),
        denominator=RequirementSide(input_id=denominator.id),
        axis="debtor",
        denominator_scope=DenominatorScope.POOLED,
    )
    rank = _rank(share.id, reduction=Reduction.MEAN)

    plan = _plan(fixed, evidence_requirements=[numerator, denominator, share, rank])

    assert plan.requirement("ranked").input_id == "share"


def test_an_axis_naming_a_role_the_input_does_not_carry_fails(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()  # role="debtor"

    with pytest.raises(ValidationError, match="does not name a role this input carries"):
        _plan(
            _population_decision(decision),
            evidence_requirements=[
                dataset,
                _rank(dataset.id, axis="creditor", reduction=Reduction.MEAN),
            ],
        )


def test_among_named_with_no_narrowing_filter_fails(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()

    with pytest.raises(ValidationError, match="carries no narrowing filter"):
        _plan(
            decision(),
            evidence_requirements=[
                dataset,
                _rank(dataset.id, among=RankAmong.NAMED, reduction=Reduction.MEAN),
            ],
        )


def test_among_named_with_a_narrowing_filter_validates(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    named = FilterRequirement(
        id="named_entities",
        decision_ids=["measure"],
        input_id=dataset.id,
        predicates=(EntityIn(names=["Kenya"], role="debtor"),),
    )

    plan = _plan(
        decision(),
        evidence_requirements=[
            dataset,
            named,
            _rank(named.id, among=RankAmong.NAMED, reduction=Reduction.MEAN),
        ],
    )

    assert plan.requirement("ranked").among == RankAmong.NAMED


def test_among_population_with_no_decision_fixing_it_fails(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()

    with pytest.raises(ValidationError, match=r"cites no decision whose fixes\.among"):
        _plan(
            decision(),  # fixes=None
            evidence_requirements=[
                dataset,
                _rank(dataset.id, among=RankAmong.POPULATION, reduction=Reduction.MEAN),
            ],
        )


def test_a_keyed_share_of_needs_a_denominator_scope() -> None:
    with pytest.raises(ValidationError, match="denominator_scope is stated exactly when axis is"):
        ShareOfRequirement(
            id="share",
            decision_ids=["measure"],
            numerator=RequirementSide(input_id="a"),
            denominator=RequirementSide(input_id="b"),
            axis="donor",
        )


def test_an_unkeyed_share_of_cannot_state_a_denominator_scope() -> None:
    with pytest.raises(ValidationError, match="denominator_scope is stated exactly when axis is"):
        ShareOfRequirement(
            id="share",
            decision_ids=["measure"],
            numerator=RequirementSide(input_id="a"),
            denominator=RequirementSide(input_id="b"),
            denominator_scope=DenominatorScope.PER_AXIS,
        )


def test_denominator_scope_pooled_with_no_decision_fixing_it_fails(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    numerator = requirement()
    denominator = requirement(id="other_dataset", entities=[Entity(name="Kenya", role="creditor")])
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=numerator.id),
        denominator=RequirementSide(input_id=denominator.id),
        axis="debtor",
        denominator_scope=DenominatorScope.POOLED,
    )

    with pytest.raises(ValidationError, match=r"cites no decision whose fixes\.denominator_scope"):
        _plan(decision(), evidence_requirements=[numerator, denominator, share])


def test_a_growth_over_a_share_of_with_relative_change_fails(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    numerator = requirement()
    denominator = requirement(id="other_dataset", entities=[Entity(name="Kenya", role="creditor")])
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=numerator.id),
        denominator=RequirementSide(input_id=denominator.id),
    )
    growth = GrowthRequirement(
        id="growth", decision_ids=["measure"], input_id="share", window_start=2016, window_end=2020
    )

    with pytest.raises(ValidationError, match="state change='absolute'"):
        _plan(decision(), evidence_requirements=[numerator, denominator, share, growth])


def test_a_growth_over_a_share_of_with_absolute_change_validates(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    numerator = requirement()
    denominator = requirement(id="other_dataset", entities=[Entity(name="Kenya", role="creditor")])
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=numerator.id),
        denominator=RequirementSide(input_id=denominator.id),
    )
    growth = GrowthRequirement(
        id="growth",
        decision_ids=["measure"],
        input_id="share",
        window_start=2016,
        window_end=2020,
        change=GrowthChange.ABSOLUTE,
    )

    plan = _plan(decision(), evidence_requirements=[numerator, denominator, share, growth])

    assert plan.requirement("growth").change == GrowthChange.ABSOLUTE


def _share_numerator(kind: str, dataset_id: str) -> list[EvidenceRequirement]:
    """The steps a `kind` share_of numerator adds over `dataset_id`, the last one its input."""
    if kind == "series":
        return []
    if kind == "totals":
        return [
            AggregateRequirement(
                id="numerator",
                decision_ids=["measure"],
                input_id=dataset_id,
                reduction=Reduction.SUM,
                group_by=(GroupBy.ENTITY,),
            )
        ]
    return [
        _rank(
            dataset_id, id="numerator", reduction=Reduction.SUM, window_start=2010, window_end=2030
        )
    ]


@pytest.mark.parametrize(
    ("numerator_kind", "refused"),
    [
        pytest.param("series", "no entity has a single value to rank on", id="series"),
        pytest.param("totals", None, id="totals"),
        pytest.param("ranked-under-a-wider-window", None, id="ranked-under-a-wider-window"),
    ],
)
def test_a_rank_with_no_reduction_over_a_share_of_needs_one_ratio_per_entity(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    numerator_kind: str,
    refused: str | None,
) -> None:
    # run_share_of writes one ratio per numerator entity per period. A share
    # of each entity's series holds several; a share of each entity's total
    # over 2015-2024 against Kenya's total over the same years holds one. A rank
    # under a 2010-2030 window labels its rows with the years it ranked,
    # 2015/2024, which the total matches.
    fixed = decision(
        fixes=DecisionFixes(denominator_scope=DenominatorScope.POOLED, among=RankAmong.POPULATION)
    )
    numerator = requirement()
    numerator_steps = _share_numerator(numerator_kind, numerator.id)
    denominator = requirement(id="other_dataset", entities=[Entity(name="Kenya", role="creditor")])
    denominator_steps: list[EvidenceRequirement] = []
    if numerator_steps:
        denominator_steps = [
            AggregateRequirement(
                id="denominator",
                decision_ids=["measure"],
                input_id=denominator.id,
                reduction=Reduction.SUM,
                group_by=(GroupBy.ENTITY,),
            )
        ]
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=(numerator_steps or [numerator])[-1].id),
        denominator=RequirementSide(input_id=(denominator_steps or [denominator])[-1].id),
        axis="debtor",
        denominator_scope=DenominatorScope.POOLED,
    )
    rank = _rank(share.id)  # among=POPULATION (default), no reduction, no window
    requirements = [numerator, *numerator_steps, denominator, *denominator_steps, share, rank]

    if refused is not None:
        with pytest.raises(ValidationError, match=refused):
            _plan(fixed, evidence_requirements=requirements)
    else:
        plan = _plan(fixed, evidence_requirements=requirements)
        assert plan.requirement("ranked").input_id == share.id


# --- The per-period rank ---------------------------------------------------------


def test_a_growth_over_a_collapsing_rank_fails(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    rank = _rank(dataset.id, reduction=Reduction.MEAN)  # per_period=False (default)
    growth = GrowthRequirement(
        id="growth",
        decision_ids=["measure"],
        input_id="ranked",
        window_start=2016,
        window_end=2020,
        change=GrowthChange.ABSOLUTE,
    )

    with pytest.raises(ValidationError, match="not a span already collapsed"):
        _plan(_population_decision(decision), evidence_requirements=[dataset, rank, growth])


def test_a_growth_over_a_per_period_rank_needs_a_narrowing_filter_first(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    rank = RankRequirement(
        id="ranked",
        decision_ids=["measure"],
        input_id=dataset.id,
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        per_period=True,
    )
    growth = GrowthRequirement(
        id="growth",
        decision_ids=["measure"],
        input_id="ranked",
        window_start=2016,
        window_end=2020,
        change=GrowthChange.ABSOLUTE,
    )

    # The rank series carries the whole population; a growth reads it as one
    # entity's move, so it has to be filtered to one first.
    with pytest.raises(ValidationError, match="add a filter narrowing"):
        _plan(_population_decision(decision), evidence_requirements=[dataset, rank, growth])


def test_a_growth_over_a_rank_needs_per_period_a_filter_and_absolute(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()  # Kenya, role="debtor"
    rank = RankRequirement(
        id="ranked",
        decision_ids=["measure"],
        input_id=dataset.id,
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        per_period=True,
    )
    narrowed = FilterRequirement(
        id="named_entities",
        decision_ids=["measure"],
        input_id="ranked",
        predicates=(EntityIn(names=["Kenya"], role="debtor"),),
    )
    growth = GrowthRequirement(
        id="growth",
        decision_ids=["measure"],
        input_id="named_entities",
        window_start=2016,
        window_end=2020,
        change=GrowthChange.ABSOLUTE,
    )

    plan = _plan(
        _population_decision(decision), evidence_requirements=[dataset, rank, narrowed, growth]
    )

    assert plan.requirement("growth").change == GrowthChange.ABSOLUTE


def test_a_relative_growth_over_a_rank_fails_even_when_narrowed(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    rank = RankRequirement(
        id="ranked",
        decision_ids=["measure"],
        input_id=dataset.id,
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        per_period=True,
    )
    narrowed = FilterRequirement(
        id="named_entities",
        decision_ids=["measure"],
        input_id="ranked",
        predicates=(EntityIn(names=["Kenya"], role="debtor"),),
    )
    growth = GrowthRequirement(
        id="growth",
        decision_ids=["measure"],
        input_id="named_entities",
        window_start=2016,
        window_end=2020,
        # change defaults to relative
    )

    with pytest.raises(ValidationError, match="state change='absolute'"):
        _plan(
            _population_decision(decision), evidence_requirements=[dataset, rank, narrowed, growth]
        )


def _top_five_each_year(input_id: str) -> RankRequirement:
    return _rank(input_id, per_period=True, limit=5)


def _pooled_decision(decision: DecisionFactory) -> Decision:
    return decision(
        fixes=DecisionFixes(denominator_scope=DenominatorScope.POOLED, among=RankAmong.POPULATION)
    )


def test_a_keyed_share_of_over_a_limited_per_period_rank_fails_naming_the_order_that_works(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # Each debtor keeps only the years it made the top five, so its periods
    # differ from the denominator's and `run_share_of` refuses the call.
    dataset = requirement()
    rank = _top_five_each_year(dataset.id)
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=rank.id),
        denominator=RequirementSide(input_id=dataset.id),
        axis="debtor",
        denominator_scope=DenominatorScope.POOLED,
    )

    with pytest.raises(ValidationError, match="compute the share_of first, then rank the shares"):
        _plan(_pooled_decision(decision), evidence_requirements=[dataset, rank, share])


def _kenya_share_of(input_id: str, *, denominator_id: str) -> ShareOfRequirement:
    return ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=input_id, entity="Kenya"),
        denominator=RequirementSide(input_id=denominator_id),
    )


def test_a_share_of_narrowed_to_one_entity_of_a_limited_per_period_rank_fails(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    rank = _top_five_each_year(dataset.id)
    share = _kenya_share_of(rank.id, denominator_id=dataset.id)

    with pytest.raises(
        ValidationError,
        match="read the share_of's side from the frame that rank reads, keeping its entity",
    ):
        _plan(_population_decision(decision), evidence_requirements=[dataset, rank, share])


def test_a_share_of_narrowed_to_one_entity_of_the_frame_a_limited_rank_reads_validates(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # The plan the refusal above advises, built from the refused one: the
    # numerator reads the rank's input instead of the rank and keeps Kenya,
    # so it holds every year of Kenya's series.
    dataset = requirement()
    rank = _top_five_each_year(dataset.id)
    share = _kenya_share_of(rank.input_id, denominator_id=dataset.id)

    plan = _plan(_population_decision(decision), evidence_requirements=[dataset, rank, share])

    assert plan.requirement("share").numerator.input_id == dataset.id
    assert plan.requirement("share").numerator.entity == "Kenya"


def test_a_share_of_ranked_per_period_with_a_limit_afterwards_validates(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    numerator = requirement()
    denominator = requirement(id="other_dataset", entities=[Entity(name="Kenya", role="creditor")])
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=numerator.id),
        denominator=RequirementSide(input_id=denominator.id),
        axis="debtor",
        denominator_scope=DenominatorScope.POOLED,
    )
    rank = _top_five_each_year(share.id)

    plan = _plan(
        _pooled_decision(decision), evidence_requirements=[numerator, denominator, share, rank]
    )

    assert plan.requirement("ranked").input_id == "share"


def test_a_per_axis_share_of_over_a_limited_denominator_fails_naming_the_order_that_works(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # One year leaves the limited rank uncut, but every debtor outside the
    # top five in the numerator has no denominator group, so `run_share_of`
    # refuses the call at the first one.
    total = requirement(period=Period(start=2022, end=2022))
    bilateral = requirement(
        id="bilateral", measure="bilateral debt service paid", period=Period(start=2022, end=2022)
    )
    top = _top_five_each_year(total.id)
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=bilateral.id),
        denominator=RequirementSide(input_id=top.id),
        axis="debtor",
        denominator_scope=DenominatorScope.PER_AXIS,
    )

    with pytest.raises(
        ValidationError,
        match="compute the share_of over the unlimited denominator, then rank the shares",
    ):
        _plan(_population_decision(decision), evidence_requirements=[total, bilateral, top, share])


def test_the_top_fives_share_of_the_total_each_year_validates(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    rank = _top_five_each_year(dataset.id)
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=rank.id),
        denominator=RequirementSide(input_id=dataset.id),
    )

    plan = _plan(_population_decision(decision), evidence_requirements=[dataset, rank, share])

    assert plan.requirement("share").numerator.input_id == "ranked"


def test_a_growth_over_a_limited_per_period_rank_fails_naming_the_order_that_works(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    rank = _top_five_each_year(dataset.id)
    growth = GrowthRequirement(
        id="growth", decision_ids=["measure"], input_id=rank.id, window_start=2016, window_end=2020
    )

    with pytest.raises(ValidationError, match="compute the growth first, then rank its result"):
        _plan(_population_decision(decision), evidence_requirements=[dataset, rank, growth])


_RANK_THE_SOURCE_REDUCED = (
    "rank the source once, with the reduction, the limit and per_period=false"
)


@pytest.mark.parametrize(
    ("group_by", "refused"),
    [
        ((GroupBy.ENTITY,), True),
        ((GroupBy.PERIOD,), False),
        ((GroupBy.ENTITY, GroupBy.PERIOD), False),
    ],
)
def test_an_aggregate_over_a_limited_per_period_rank_fails_only_by_entity_alone(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    group_by: tuple[GroupBy, ...],
    refused: bool,
) -> None:
    # By entity alone, each debtor's total covers only the years it made the
    # cut; by period, each year totals the five it kept.
    dataset = requirement()
    rank = _top_five_each_year(dataset.id)
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["measure"],
        input_id=rank.id,
        reduction=Reduction.SUM,
        group_by=group_by,
    )
    requirements = [dataset, rank, totals]

    if refused:
        with pytest.raises(ValidationError, match=_RANK_THE_SOURCE_REDUCED):
            _plan(_population_decision(decision), evidence_requirements=requirements)
    else:
        plan = _plan(_population_decision(decision), evidence_requirements=requirements)
        assert plan.requirement("totals").group_by == group_by


def test_the_plan_the_aggregate_refusal_advises_validates(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # "The five debtors that paid most over the decade": one rank over the
    # source states the reduction and the limit and collapses the window.
    dataset = requirement()
    top_five = _rank(dataset.id, reduction=Reduction.SUM, limit=5)

    plan = _plan(_population_decision(decision), evidence_requirements=[dataset, top_five])

    ranked = plan.requirement("ranked")
    assert (ranked.reduction, ranked.per_period, ranked.limit) == (Reduction.SUM, False, 5)


def test_the_plan_the_growth_refusal_advises_validates(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # "Compute the growth first, then rank its result with the limit."
    dataset = requirement()
    growth = GrowthRequirement(
        id="growth",
        decision_ids=["measure"],
        input_id=dataset.id,
        window_start=2016,
        window_end=2020,
    )
    top_five = _rank(growth.id, limit=5)

    plan = _plan(_population_decision(decision), evidence_requirements=[dataset, growth, top_five])

    assert plan.requirement("ranked").input_id == growth.id


def test_the_plan_the_per_axis_share_of_refusals_advise_validates(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # "Compute the share_of over the unlimited denominator, then rank the
    # shares with the limit", per period or collapsed to one year.
    total = requirement(period=Period(start=2022, end=2022))
    bilateral = requirement(
        id="bilateral", measure="bilateral debt service paid", period=Period(start=2022, end=2022)
    )
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=bilateral.id),
        denominator=RequirementSide(input_id=total.id),
        axis="debtor",
        denominator_scope=DenominatorScope.PER_AXIS,
    )

    for rank in (_top_five_each_year(share.id), _rank(share.id, limit=5)):
        plan = _plan(
            _population_decision(decision), evidence_requirements=[total, bilateral, share, rank]
        )
        assert plan.requirement("ranked").input_id == share.id


_RANK_ONCE = "rank the source once with the limit the question needs"


def test_no_rank_reads_a_limited_per_period_rank(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    each_year = _top_five_each_year(dataset.id)
    overall = _rank(each_year.id, id="overall", reduction=Reduction.SUM)
    top_three = _rank(each_year.id, id="top_three", per_period=True, limit=3)

    for rank in (overall, top_three):
        with pytest.raises(ValidationError, match=_RANK_ONCE):
            _plan(_population_decision(decision), evidence_requirements=[dataset, each_year, rank])


def test_the_bottom_three_of_each_years_top_ten_fails(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # "Among each year's top 10 debtors, which 3 paid least?" The second rank
    # would hold the 8th to 10th largest, and its card would claim it ranked
    # every entity the source holds.
    dataset = requirement()
    top_ten = _rank(dataset.id, id="top_ten", per_period=True, limit=10)
    bottom_three = _rank(
        top_ten.id,
        id="bottom_three",
        direction=RankDirection.LOWEST_FIRST,
        per_period=True,
        limit=3,
    )

    with pytest.raises(ValidationError, match=_RANK_ONCE):
        _plan(
            _population_decision(decision), evidence_requirements=[dataset, top_ten, bottom_three]
        )


def test_a_rank_over_a_limited_rank_of_one_year_fails(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # One year leaves the limited rank uncut, but its kept entities are still
    # neither named nor the source's population.
    dataset = requirement(period=Period(start=2022, end=2022))
    top_ten = _rank(dataset.id, id="top_ten", per_period=True, limit=10)
    bottom_three = _rank(
        top_ten.id, id="bottom_three", direction=RankDirection.LOWEST_FIRST, limit=3
    )

    with pytest.raises(ValidationError, match=_RANK_ONCE):
        _plan(
            _population_decision(decision), evidence_requirements=[dataset, top_ten, bottom_three]
        )


@pytest.mark.parametrize(
    "second",
    [
        {"per_period": True, "limit": 1},
        {"per_period": True},
        {"reduction": Reduction.MEAN, "limit": 1},
        {"reduction": Reduction.MEAN},
    ],
)
def test_no_rank_reads_an_unlimited_per_period_rank(
    decision: DecisionFactory, requirement: RequirementFactory, second: dict[str, Any]
) -> None:
    # The first rank writes positions, so a highest-first rank over it would
    # keep the debtor at the last position as "the top debtor".
    dataset = requirement()
    each_year = _rank(dataset.id, id="each_year", per_period=True)
    again = _rank(each_year.id, id="again", **second)

    with pytest.raises(ValidationError, match=f"rank position.*{_RANK_ONCE}"):
        _plan(_population_decision(decision), evidence_requirements=[dataset, each_year, again])


def test_the_limit_on_the_first_per_period_rank_validates(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # The plan the refusal above advises: "the top debtor each year" is one
    # per-period rank carrying the limit.
    dataset = requirement()
    top = _rank(dataset.id, per_period=True, limit=1)

    plan = _plan(_population_decision(decision), evidence_requirements=[dataset, top])

    assert plan.requirement("ranked").limit == 1


def test_no_rank_reads_a_change_in_rank_position(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # analysis-mcp carries its ordinal marker through the growth and refuses
    # the rank, so the plan fails here first.
    dataset = requirement()
    each_year = _rank(dataset.id, id="each_year", per_period=True)
    named = FilterRequirement(
        id="named_entities",
        decision_ids=["measure"],
        input_id=each_year.id,
        predicates=(EntityIn(names=["Kenya"], role="debtor"),),
    )
    places = GrowthRequirement(
        id="places",
        decision_ids=["measure"],
        input_id=named.id,
        window_start=2016,
        window_end=2020,
        change=GrowthChange.ABSOLUTE,
    )
    climbed = _rank(places.id, id="climbed", decision_ids=["scope"], among=RankAmong.NAMED)

    with pytest.raises(ValidationError, match=f"rank position.*{_RANK_ONCE}"):
        _plan(
            _population_decision(decision),
            decisions=[_population_decision(decision), decision(slot="scope")],
            evidence_requirements=[dataset, each_year, named, places, climbed],
        )


@pytest.mark.parametrize(
    ("predicates", "refused"),
    [
        ((PeriodThreshold(op="gte", year=2022),), False),
        ((PeriodThreshold(op="gt", year=2021),), False),
        ((PeriodThreshold(op="lt", year=2016),), False),
        ((PeriodThreshold(op="gt", year=2017), PeriodThreshold(op="lt", year=2019)), False),
        ((PeriodThreshold(op="gte", year=2021),), True),
        ((PeriodThreshold(op="gt", year=2020),), True),
        ((PeriodThreshold(op="gte", year=2021), PeriodThreshold(op="ne", year=2022)), False),
        ((PeriodThreshold(op="ne", year=2022),), True),
    ],
)
def test_a_year_filter_that_leaves_one_year_of_a_limited_rank_clears_its_cut(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    predicates: tuple[PeriodThreshold, ...],
    refused: bool,
) -> None:
    # Over 2015-2022, `year >= 2022`, `year > 2021`, `year < 2016` and
    # `2017 < year < 2019` each leave one year, so no debtor's total misses a
    # year, as does `year >= 2021` without 2022; `year >= 2021` and
    # `year > 2020` leave two, and `year != 2022` leaves seven.
    dataset = requirement(period=Period(start=2015, end=2022))
    rank = _top_five_each_year(dataset.id)
    recent = FilterRequirement(
        id="recent",
        decision_ids=["measure"],
        input_id=rank.id,
        predicates=predicates,
    )
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["measure"],
        input_id=recent.id,
        reduction=Reduction.SUM,
        group_by=(GroupBy.ENTITY,),
    )
    requirements = [dataset, rank, recent, totals]

    if refused:
        with pytest.raises(ValidationError, match=_RANK_THE_SOURCE_REDUCED):
            _plan(_population_decision(decision), evidence_requirements=requirements)
        # The plan the refusal advises, over the same years of the source.
        years = FilterRequirement(
            id="years", decision_ids=["measure"], input_id=dataset.id, predicates=predicates
        )
        top_five = _rank(years.id, reduction=Reduction.SUM, limit=5)
        advised = _plan(
            _population_decision(decision), evidence_requirements=[dataset, years, top_five]
        )
        assert advised.requirement("ranked").input_id == years.id
    else:
        plan = _plan(_population_decision(decision), evidence_requirements=requirements)
        assert plan.requirement("totals").input_id == recent.id


@pytest.mark.parametrize(
    "predicates",
    [
        (PeriodThreshold(op="gt", year=2022),),
        (PeriodThreshold(op="lt", year=2015),),
        (PeriodThreshold(op="gt", year=2017), PeriodThreshold(op="lt", year=2018)),
        (PeriodThreshold(op="gte", year=2023),),
        (PeriodThreshold(op="lte", year=2014),),
        (PeriodThreshold(op="eq", year=2030),),
        (PeriodThreshold(op="eq", year=2018), PeriodThreshold(op="eq", year=2019)),
        (PeriodThreshold(op="eq", year=2018), PeriodThreshold(op="gt", year=2020)),
        (PeriodThreshold(op="eq", year=2018), PeriodThreshold(op="ne", year=2018)),
    ],
)
def test_a_year_filter_that_leaves_no_year_fails_naming_the_filter(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    predicates: tuple[PeriodThreshold, ...],
) -> None:
    dataset = requirement(period=Period(start=2015, end=2022))
    recent = FilterRequirement(
        id="recent", decision_ids=["measure"], input_id=dataset.id, predicates=predicates
    )

    with pytest.raises(
        ValidationError, match="'recent': the year filter leaves no year of 2015-2022"
    ):
        _plan(decision(), evidence_requirements=[dataset, recent])


def test_a_year_filter_excluding_the_one_year_an_earlier_filter_kept_fails_naming_it(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement(period=Period(start=2015, end=2022))
    one_year = FilterRequirement(
        id="one_year",
        decision_ids=["measure"],
        input_id=dataset.id,
        predicates=(PeriodThreshold(op="eq", year=2020),),
    )
    others = FilterRequirement(
        id="others",
        decision_ids=["measure"],
        input_id=one_year.id,
        predicates=(PeriodThreshold(op="ne", year=2020),),
    )

    with pytest.raises(
        ValidationError, match="'others': the year filter leaves no year of 2020-2020"
    ):
        _plan(decision(), evidence_requirements=[dataset, one_year, others])


@pytest.mark.parametrize(
    "predicates",
    [
        (PeriodThreshold(op="eq", year=2018),),
        (PeriodThreshold(op="eq", year=2018), PeriodThreshold(op="eq", year=2018)),
        (PeriodThreshold(op="eq", year=2018), PeriodThreshold(op="gt", year=2017)),
        (PeriodThreshold(op="eq", year=2018), PeriodThreshold(op="ne", year=2019)),
    ],
)
def test_an_eq_year_filter_that_leaves_its_year_validates(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    predicates: tuple[PeriodThreshold, ...],
) -> None:
    dataset = requirement(period=Period(start=2015, end=2022))
    recent = FilterRequirement(
        id="recent", decision_ids=["measure"], input_id=dataset.id, predicates=predicates
    )

    plan = _plan(decision(), evidence_requirements=[dataset, recent])

    assert plan.requirement("recent").input_id == dataset.id


def test_a_population_rank_over_a_prenarrowed_axis_fails_naming_both_readings(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    named = FilterRequirement(
        id="named_entities",
        decision_ids=["measure"],
        input_id=dataset.id,
        predicates=(EntityIn(names=["Kenya"], role="debtor"),),
    )
    rank = _rank(named.id, among=RankAmong.POPULATION, reduction=Reduction.MEAN)

    with pytest.raises(ValidationError, match="state among='named'"):
        _plan(_population_decision(decision), evidence_requirements=[dataset, named, rank])


# --- Requirements honour the decisions they cite ---------------------------------


def test_a_requirement_cannot_contradict_what_a_cited_decision_fixes(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # `chosen` is prose, so it cannot be held against the requirement; `fixes`
    # can be.
    fixed = decision(fixes=DecisionFixes(period=Period(start=2018, end=2024)))

    with pytest.raises(
        ValidationError,
        match=r"'annual_debt_service' states period .*2015.*'measure' it cites fixes",
    ):
        _plan(fixed, evidence_requirements=[requirement(), derived_requirement()])


def test_a_requirement_cannot_contradict_the_across_a_cited_decision_fixes(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    fixed = decision(fixes=DecisionFixes(across="donor"))

    with pytest.raises(
        ValidationError,
        match=r"'annual_debt_service' states across 'creditor', but the decision 'measure'",
    ):
        _plan(fixed, evidence_requirements=[requirement(across="creditor"), derived_requirement()])
    plan = _plan(fixed, evidence_requirements=[requirement(across="donor"), derived_requirement()])

    assert plan.requirement("annual_debt_service").across == "donor"


def test_a_decision_holds_a_requirement_only_to_the_fields_it_fixes(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    fixed = decision(fixes=DecisionFixes(measure="external debt service paid"))
    other_period = requirement(period=Period(start=2000, end=2001))

    plan = _plan(fixed, evidence_requirements=[other_period, derived_requirement()])

    assert plan.requirement(other_period.id) == other_period


def test_a_requirement_stating_no_unit_contradicts_a_decision_fixing_one(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    fixed = decision(fixes=DecisionFixes(unit=Unit(currency="USD", prices="current")))

    with pytest.raises(ValidationError, match="states unit None"):
        _plan(fixed, evidence_requirements=[requirement(unit=None), derived_requirement()])


def test_an_aggregate_contradicting_its_decision_is_rejected(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    fixed = decision(fixes=DecisionFixes(reduction=Reduction.MEAN))
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["measure"],
        input_id=dataset.id,
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )

    with pytest.raises(
        ValidationError, match=r"states reduction .*sum.*'measure' it cites fixes .*mean"
    ):
        _plan(fixed, evidence_requirements=[dataset, totals])


def test_a_growth_window_contradicting_its_decision_is_rejected(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    fixed = decision(fixes=DecisionFixes(window_start=2018))
    growth = GrowthRequirement(
        id="growth",
        decision_ids=["measure"],
        input_id=dataset.id,
        window_start=2016,
        window_end=2020,
    )

    with pytest.raises(
        ValidationError, match=r"states window_start 2016, but the decision .*fixes 2018"
    ):
        _plan(fixed, evidence_requirements=[dataset, growth])


def test_a_rank_direction_contradicting_its_decision_is_rejected(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    fixed = decision(fixes=DecisionFixes(direction=RankDirection.LOWEST_FIRST))

    with pytest.raises(
        ValidationError, match=r"states direction .*highest_first.*fixes .*lowest_first"
    ):
        _plan(
            fixed,
            evidence_requirements=[
                dataset,
                _rank(dataset.id, reduction=Reduction.MEAN, among=RankAmong.POPULATION),
            ],
        )


def test_a_population_rank_with_a_limit_validates(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    rank = _rank(dataset.id, reduction=Reduction.MEAN, limit=10)

    plan = _plan(_population_decision(decision), evidence_requirements=[dataset, rank])

    assert plan.requirement(rank.id) == rank
    assert _rank(dataset.id).limit is None


@pytest.mark.parametrize("limit", [0, -1, MAX_RANK_LIMIT + 1])
def test_a_rank_limit_outside_one_to_the_maximum_is_refused(limit: int) -> None:
    with pytest.raises(ValidationError, match="limit"):
        _rank("annual_debt_service", limit=limit)
    with pytest.raises(ValidationError, match="limit"):
        DecisionFixes(limit=limit)


def test_a_rank_limit_at_the_maximum_validates() -> None:
    assert _rank("annual_debt_service", limit=MAX_RANK_LIMIT).limit == MAX_RANK_LIMIT


def test_a_rank_limit_contradicting_its_decision_is_rejected(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    fixed = decision(fixes=DecisionFixes(among=RankAmong.POPULATION, limit=5))

    with pytest.raises(ValidationError, match=r"states limit 10, but the decision .* fixes 5"):
        _plan(
            fixed,
            evidence_requirements=[dataset, _rank(dataset.id, reduction=Reduction.MEAN, limit=10)],
        )
    with pytest.raises(ValidationError, match=r"states limit None, but the decision .* fixes 5"):
        _plan(fixed, evidence_requirements=[dataset, _rank(dataset.id, reduction=Reduction.MEAN)])

    agreeing = _rank(dataset.id, reduction=Reduction.MEAN, limit=5)
    plan = _plan(fixed, evidence_requirements=[dataset, agreeing])
    assert plan.requirement(agreeing.id) == agreeing


def test_filter_predicates_contradicting_its_decision_is_rejected(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    fixed = decision(fixes=DecisionFixes(predicates=(ValueThreshold(op="gte", value=0),)))
    filtered = FilterRequirement(
        id="filtered",
        decision_ids=["measure"],
        input_id=dataset.id,
        predicates=(ValueThreshold(op="gt", value=0),),
    )

    with pytest.raises(ValidationError, match=r"states predicates .*'measure' it cites fixes"):
        _plan(fixed, evidence_requirements=[dataset, filtered])


def test_a_group_by_fix_agreeing_in_a_different_order_is_accepted(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    fixed = decision(fixes=DecisionFixes(group_by=(GroupBy.ENTITY, GroupBy.PERIOD)))
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["measure"],
        input_id=dataset.id,
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD, GroupBy.ENTITY),
    )

    plan = _plan(fixed, evidence_requirements=[dataset, totals])

    assert plan.requirement("totals").group_by == (GroupBy.PERIOD, GroupBy.ENTITY)


def test_a_predicates_fix_agreeing_in_a_different_order_is_accepted(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement()
    gt, lte = ValueThreshold(op="gt", value=0), ValueThreshold(op="lte", value=100)
    fixed = decision(fixes=DecisionFixes(predicates=(lte, gt)))
    filtered = FilterRequirement(
        id="filtered", decision_ids=["measure"], input_id=dataset.id, predicates=(gt, lte)
    )

    plan = _plan(fixed, evidence_requirements=[dataset, filtered])

    assert plan.requirement("filtered").predicates == (gt, lte)


def test_a_period_fix_cited_by_a_dataset_and_an_aggregate_is_accepted(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # `period` is a `DatasetRequirement` field only -- an `AggregateRequirement`
    # citing the same decision is not held to a field its own class does not
    # declare.
    fixed = decision(fixes=DecisionFixes(period=Period(start=2015, end=2024)))
    dataset = requirement(period=Period(start=2015, end=2024))
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["measure"],
        input_id=dataset.id,
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )

    plan = _plan(fixed, evidence_requirements=[dataset, totals])

    assert plan.requirement("totals").decision_ids == ("measure",)


def test_fixed_entities_compare_regardless_of_order(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    kenya, nigeria = Entity(name="Kenya", role="debtor"), Entity(name="Nigeria", role="debtor")
    fixed = decision(fixes=DecisionFixes(entities=(nigeria, kenya)))

    plan = _plan(
        fixed, evidence_requirements=[requirement(entities=[kenya, nigeria]), derived_requirement()]
    )

    assert plan.decisions[0].fixes == DecisionFixes(entities=(nigeria, kenya))


def test_derived_requirements_are_not_compared_with_what_a_decision_fixes(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # The aggregate cites `measure` too and carries no period of its own.
    fixed = decision(fixes=DecisionFixes(period=Period(start=2015, end=2024)))

    plan = _plan(fixed, evidence_requirements=[requirement(), derived_requirement()])

    assert plan.requirement("annual_totals").decision_ids == ("measure",)


def test_a_pinned_source_needs_a_decision_that_fixes_it(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # `decision()`'s default carries no `fixes` at all, so citing it pins
    # nothing -- regardless of who resolved it.
    with pytest.raises(
        ValidationError, match="cites no decision that fixes that same source_spec_id"
    ):
        _plan(
            decision(),
            evidence_requirements=[requirement(source_spec_id=_PIN), derived_requirement()],
        )


def test_a_pin_needs_a_decision_that_fixes_that_exact_source(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # `resolved_by=USER` alone is not enough: the cited decision has to fix
    # this exact pin structurally, not merely be a user's choice about
    # something else.
    source = decision(
        slot="source",
        options=["OECD DAC", "World Bank IDS"],
        chosen="OECD DAC",
        resolved_by=ResolvedBy.USER,
    )
    pinned = requirement(decision_ids=["measure", "source"], source_spec_id=_PIN)

    with pytest.raises(
        ValidationError, match="cites no decision that fixes that same source_spec_id"
    ):
        _plan(
            decision(),
            decisions=[decision(), source],
            evidence_requirements=[pinned, derived_requirement()],
        )


def test_a_pinned_source_backed_by_a_user_decision_validates(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # The user chooses "OECD DAC", not a hash; `chosen` states the prose
    # choice while `fixes.source_spec_id` is what the pin is checked against.
    source = decision(
        slot="source",
        options=["OECD DAC", "World Bank IDS"],
        chosen="OECD DAC",
        resolved_by=ResolvedBy.USER,
        fixes=DecisionFixes(source_spec_id=_PIN),
    )
    pinned = requirement(decision_ids=["measure", "source"], source_spec_id=_PIN)

    plan = _plan(
        decision(),
        decisions=[decision(), source],
        evidence_requirements=[pinned, derived_requirement()],
    )

    assert plan.requirement(pinned.id).source_spec_id == _PIN


def test_a_pinned_source_backed_by_a_planner_default_decision_validates(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # Attribution is not this validator's business: a decision the planner
    # settled itself pins just as well, as long as `fixes` names the pin.
    source = decision(
        slot="source",
        options=["OECD DAC", "World Bank IDS"],
        chosen="OECD DAC",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        fixes=DecisionFixes(source_spec_id=_PIN),
    )
    pinned = requirement(decision_ids=["measure", "source"], source_spec_id=_PIN)

    plan = _plan(
        decision(),
        decisions=[decision(), source],
        evidence_requirements=[pinned, derived_requirement()],
    )

    assert plan.requirement(pinned.id).source_spec_id == _PIN


def test_a_pinned_source_matches_the_one_the_cited_decision_fixes(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    source = decision(
        slot="source",
        options=["OECD DAC", "World Bank IDS"],
        chosen="OECD DAC",
        resolved_by=ResolvedBy.USER,
        fixes=DecisionFixes(source_spec_id="spec_" + "b" * 32),
    )
    pinned = requirement(decision_ids=["measure", "source"], source_spec_id=_PIN)

    with pytest.raises(ValidationError, match=r"states source_spec_id .*'source' it cites fixes"):
        _plan(
            decision(),
            decisions=[decision(), source],
            evidence_requirements=[pinned, derived_requirement()],
        )


# --- Requirement lookup ---------------------------------------------------------


def test_requirement_returns_the_named_requirement(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    plan = _plan(decision(), evidence_requirements=list(requirements))
    dataset, derived = requirements

    assert plan.requirement(dataset.id) == dataset
    assert plan.requirement(derived.id) == derived


def test_requirement_raises_for_an_unknown_id(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    plan = _plan(decision(), evidence_requirements=list(requirements))

    with pytest.raises(KeyError, match="states no requirement 'unknown'"):
        plan.requirement("unknown")


# --- Translation ---------------------------------------------------------------

SPEC = "spec_" + "a" * 32


def _translation() -> Translation:
    # Mirrors the `requirements` fixture's pair: a dataset requirement
    # ("annual_debt_service") read by an aggregate one ("annual_totals").
    return Translation(
        operations=[
            RetrieveOperation(id="debt", requirement_id="annual_debt_service", spec_id=SPEC),
            AggregateOperation(
                id="totals",
                requirement_id="annual_totals",
                input="debt",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
        ],
        terminal=["totals"],
    )


def _workflow(
    *, step_id: str = "debt", requirement_id: str = "annual_debt_service", spec_id: str = SPEC
) -> Workflow:
    return Workflow(
        steps=[
            WorkflowStep(
                id=step_id,
                operation_id=step_id,
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": spec_id},
                input_schema_hash="hash",
            ),
            WorkflowStep(
                id="totals",
                operation_id="totals",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                args={"aggregation": "sum", "group_by": ["time_period"]},
                bindings={"frame_ref": Binding(from_step=step_id, output=ArtifactKind.FRAME_REF)},
                input_schema_hash="hash",
            ),
        ],
        sources=[SelectedSource(spec_id=spec_id, sentence="external debt service", has_data=True)],
        evidence_bindings=[
            EvidenceBinding(requirement_id=requirement_id, step_id=step_id, spec_id=spec_id)
        ],
        terminal_steps=["totals"],
    )


def test_with_translation_completes_a_draft_and_holds_its_version(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    draft = _plan(decision(), evidence_requirements=list(requirements), status=PlanStatus.DRAFT)

    translated = draft.with_translation(translation=_translation(), workflow=_workflow())

    # Same version: translating a draft completes it, so one version stays
    # one thing the user reviewed.
    assert translated.version == draft.version
    assert translated.status == PlanStatus.PENDING_APPROVAL
    assert translated.translation == _translation()
    assert translated.workflow == _workflow()
    assert draft.status == PlanStatus.DRAFT
    assert draft.workflow is None


def test_a_workflow_caveat_changes_the_plan_revision(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    # Approving a plan approves its caveats: two plans alike in every call
    # but one carrying a warning must not share a revision.
    caveat = EvidenceIssue(
        code=EvidenceIssueCode.WINDOW_SHORT,
        severity=Severity.WARNING,
        requirement_id="annual_debt_service",
        detail="the source covers 2010-2020, the requirement asks for 2010-2024",
    )
    draft = _plan(decision(), evidence_requirements=list(requirements), status=PlanStatus.DRAFT)

    quiet = draft.with_translation(translation=_translation(), workflow=_workflow())
    caveated = draft.with_translation(
        translation=_translation(),
        workflow=_workflow().model_copy(update={"warnings": (caveat,)}),
    )

    assert plan_revision(quiet) != plan_revision(caveated)


@pytest.mark.parametrize(
    "status", [PlanStatus.PENDING_APPROVAL, PlanStatus.APPROVED, PlanStatus.REJECTED]
)
def test_with_translation_refuses_a_plan_that_is_not_a_draft(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...], status: PlanStatus
) -> None:
    # Translating anything else would overwrite what the user already saw.
    plan = _plan(decision(), evidence_requirements=list(requirements), status=status)

    with pytest.raises(ValueError, match="cannot be translated"):
        plan.with_translation(translation=_translation(), workflow=_workflow())


def test_a_translation_and_its_workflow_arrive_together(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    with pytest.raises(ValidationError, match="together"):
        _plan(decision(), evidence_requirements=list(requirements), translation=_translation())


def test_a_workflow_compiles_only_the_translations_operations(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    with pytest.raises(ValidationError, match="operations the translation lacks: \\['ghost'\\]"):
        _plan(
            decision(),
            evidence_requirements=list(requirements),
            translation=_translation(),
            workflow=_workflow(step_id="ghost"),
        )


def test_evidence_is_bound_to_requirements_the_plan_makes(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    with pytest.raises(ValidationError, match="requirements the plan lacks: \\['population'\\]"):
        _plan(
            decision(),
            evidence_requirements=list(requirements),
            translation=_translation(),
            workflow=_workflow(requirement_id="population"),
        )


def test_an_operation_must_name_a_requirement_the_plan_states(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    bogus = Translation(
        operations=[RetrieveOperation(id="debt", requirement_id="ghost", spec_id=SPEC)],
        terminal=["debt"],
    )

    with pytest.raises(
        ValidationError, match=r"operations name requirements this plan does not state: \['debt'\]"
    ):
        _plan(
            decision(),
            evidence_requirements=list(requirements),
            translation=bogus,
            workflow=_workflow(),
        )


def test_every_requirement_must_be_carried_out_by_an_operation(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    only_retrieve = Translation(
        operations=[
            RetrieveOperation(id="debt", requirement_id="annual_debt_service", spec_id=SPEC)
        ],
        terminal=["debt"],
    )

    with pytest.raises(
        ValidationError, match=r"requirements no operation carries out: \['annual_totals'\]"
    ):
        _plan(
            decision(),
            evidence_requirements=list(requirements),
            translation=only_retrieve,
            workflow=_workflow(),
        )


def test_a_requirement_carried_out_twice_is_rejected(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    twice = Translation(
        operations=[
            RetrieveOperation(id="debt", requirement_id="annual_debt_service", spec_id=SPEC),
            RetrieveOperation(id="debt_again", requirement_id="annual_debt_service", spec_id=SPEC),
            AggregateOperation(
                id="totals",
                requirement_id="annual_totals",
                input="debt",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
        ],
        terminal=["totals"],
    )

    with pytest.raises(
        ValidationError,
        match=r"carried out by more than one operation: \['annual_debt_service'\]",
    ):
        _plan(
            decision(),
            evidence_requirements=list(requirements),
            translation=twice,
            workflow=_workflow(),
        )


def test_an_operation_must_agree_with_the_requirement_it_carries_out(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    # `annual_totals` asks for `sum`; this operation proposes `mean`.
    wrong_reduction = Translation(
        operations=[
            RetrieveOperation(id="debt", requirement_id="annual_debt_service", spec_id=SPEC),
            AggregateOperation(
                id="totals",
                requirement_id="annual_totals",
                input="debt",
                reduction=Reduction.MEAN,
                group_by=(GroupBy.PERIOD,),
            ),
        ],
        terminal=["totals"],
    )

    with pytest.raises(ValidationError, match="operations disagree with what they carry out"):
        _plan(
            decision(),
            evidence_requirements=list(requirements),
            translation=wrong_reduction,
            workflow=_workflow(),
        )


def test_mismatches_reports_every_new_calculation_field_by_name() -> None:
    # `proof.mismatches` compares `among`/`axis`/`per_period` on a rank,
    # `change` on a growth, and `axis`/`denominator_scope` on a share_of --
    # the calculation fields each operation kind carries on both sides of
    # the comparison.
    rank_requirement = _rank("input", among=RankAmong.NAMED, axis="donor", reduction=Reduction.MEAN)
    rank_operation = RankOperation(
        id="ranked",
        requirement_id="ranked",
        input="input",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="recipient",
        reduction=Reduction.MEAN,
        per_period=True,
    )
    assert set(mismatches(rank_requirement, rank_operation, {"input": "input"})) == {
        "among",
        "axis",
        "per_period",
    }

    growth_requirement = GrowthRequirement(
        id="g",
        decision_ids=["measure"],
        input_id="input",
        window_start=2015,
        window_end=2020,
        change=GrowthChange.ABSOLUTE,
    )
    growth_operation = GrowthOperation(
        id="g",
        requirement_id="g",
        input="input",
        window_start=2015,
        window_end=2020,
        change=GrowthChange.RELATIVE,
    )
    assert mismatches(growth_requirement, growth_operation, {"input": "input"}) == ["change"]

    share_requirement = ShareOfRequirement(
        id="s",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="num"),
        denominator=RequirementSide(input_id="den"),
        axis="donor",
        denominator_scope=DenominatorScope.PER_AXIS,
    )
    share_operation = ShareOfOperation(
        id="s",
        requirement_id="s",
        numerator=Side(input="num"),
        denominator=Side(input="den"),
        axis="recipient",
        denominator_scope=DenominatorScope.POOLED,
    )
    assert set(mismatches(share_requirement, share_operation, {"num": "num", "den": "den"})) == {
        "axis",
        "denominator_scope",
    }


def test_evidence_is_bound_to_a_spec_id_the_workflow_selected(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    unpinned = _workflow().model_copy(
        update={
            "evidence_bindings": (
                EvidenceBinding(
                    requirement_id="annual_debt_service",
                    step_id="debt",
                    spec_id="spec_" + "c" * 32,
                ),
            )
        }
    )

    with pytest.raises(ValidationError, match=r"a spec_id the workflow never selected: \['debt'\]"):
        _plan(
            decision(),
            evidence_requirements=list(requirements),
            translation=_translation(),
            workflow=unpinned,
        )


def test_amend_clears_the_translation_and_the_workflow(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    # The operations were translated from decisions the new version may no
    # longer make, so the amended plan is a draft again.
    translated = _plan(
        decision(), evidence_requirements=list(requirements), status=PlanStatus.DRAFT
    ).with_translation(translation=_translation(), workflow=_workflow())

    amended = translated.amend()

    assert amended.translation is None
    assert amended.workflow is None


def test_plan_revision_covers_the_workflow(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    # Approval binds the exact calls, so a different workflow is a different
    # revision even when every decision is the same.
    draft = _plan(decision(), evidence_requirements=list(requirements), status=PlanStatus.DRAFT)
    one = draft.with_translation(translation=_translation(), workflow=_workflow())
    other = draft.with_translation(
        translation=_translation(), workflow=_workflow(spec_id="spec_" + "b" * 32)
    )

    assert plan_revision(one) != plan_revision(other)
    assert plan_revision(one) != plan_revision(draft)


# --- Approval and execution ---------------------------------------------------


def test_approve_marks_status_without_mutating_original(decision: DecisionFactory) -> None:
    plan = _plan(decision())

    approved = plan.approve()

    assert plan.status == PlanStatus.PENDING_APPROVAL
    assert approved.status == PlanStatus.APPROVED
    assert approved.decisions == plan.decisions


def test_approve_records_the_revision_approved(decision: DecisionFactory) -> None:
    plan = _plan(decision())

    approved = plan.approve()

    assert approved.approved_revision == plan_revision(plan)


@pytest.mark.parametrize(
    "status",
    [PlanStatus.DRAFT, PlanStatus.APPROVED, PlanStatus.REJECTED, PlanStatus.EXECUTING],
)
def test_approve_refuses_a_plan_not_awaiting_approval(
    decision: DecisionFactory, status: PlanStatus
) -> None:
    # A draft has nothing to run; anything past PENDING_APPROVAL already
    # carries the user's answer, which a second approval would overwrite.
    plan = _at(_plan(decision()), status)

    with pytest.raises(
        ValueError, match="cannot be approved; that takes a plan that is pending_approval"
    ):
        plan.approve()


def test_reject_marks_status_without_mutating_original(decision: DecisionFactory) -> None:
    plan = _plan(decision())

    rejected = plan.reject()

    assert plan.status == PlanStatus.PENDING_APPROVAL
    assert rejected.status == PlanStatus.REJECTED


@pytest.mark.parametrize("status", [PlanStatus.EXECUTING, PlanStatus.COMPLETED, PlanStatus.FAILED])
def test_reject_refuses_a_plan_whose_run_has_started(
    decision: DecisionFactory, status: PlanStatus
) -> None:
    plan = _at(_plan(decision()), status)

    with pytest.raises(ValueError, match="cannot be rejected"):
        plan.reject()


def test_reject_holds_on_a_plan_already_rejected(decision: DecisionFactory) -> None:
    # Declining twice is still one answer; the second call changes nothing.
    rejected = _plan(decision()).reject()

    assert rejected.reject().status == PlanStatus.REJECTED


def test_execution_follows_approval_and_ends_completed_or_failed(
    decision: DecisionFactory,
) -> None:
    executing = _plan(decision()).approve().start_execution()

    assert executing.status == PlanStatus.EXECUTING
    assert executing.complete().status == PlanStatus.COMPLETED
    assert executing.fail().status == PlanStatus.FAILED


def test_start_execution_requires_an_approved_plan(decision: DecisionFactory) -> None:
    with pytest.raises(ValueError, match="cannot be executed; that takes a plan that is approved"):
        _plan(decision()).start_execution()


def test_start_execution_refuses_content_that_is_not_what_was_approved(
    decision: DecisionFactory,
) -> None:
    # The recorded revision is the guard: a plan whose content hashes to
    # anything else was not the plan the user said yes to.
    approved = _plan(decision()).approve()
    tampered = approved.model_copy(update={"question": "a question nobody approved"})

    with pytest.raises(ValueError, match="is not what was approved"):
        tampered.start_execution()


@pytest.mark.parametrize(
    ("transition", "refusal"),
    [("complete", "cannot be completed"), ("fail", "cannot be marked failed")],
)
def test_only_a_running_plan_completes_or_fails(
    transition: str, refusal: str, decision: DecisionFactory
) -> None:
    approved = _plan(decision()).approve()

    with pytest.raises(ValueError, match=refusal):
        getattr(approved, transition)()


# --- Revisions -----------------------------------------------------------------


def test_plan_revision_ignores_status_and_approval(decision: DecisionFactory) -> None:
    # Approving must not change the value that was approved.
    plan = _plan(decision())

    assert plan_revision(plan.approve()) == plan_revision(plan)
    assert plan_revision(plan.reject()) == plan_revision(plan)


def test_plan_revision_changes_with_any_reviewed_content(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    plan = _plan(decision(), evidence_requirements=list(requirements))
    dataset, derived = requirements
    changed_dataset = dataset.model_copy(update={"period": Period(start=2015, end=2023)})
    changed = [
        plan.amend(),
        plan.model_copy(update={"question": "another question"}),
        plan.model_copy(update={"evidence_requirements": (changed_dataset, derived)}),
    ]

    assert len({plan_revision(other) for other in changed} | {plan_revision(plan)}) == 4


def test_plan_revision_changes_when_a_source_spec_id_changes(
    decision: DecisionFactory,
    pinning_decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    plan = _plan(decision(), evidence_requirements=[requirement(), derived_requirement()])
    repinned = _plan(
        pinning_decision(_PIN),
        evidence_requirements=[requirement(source_spec_id=_PIN), derived_requirement()],
    )

    assert plan_revision(repinned) != plan_revision(plan)


def test_plan_revision_is_stable_across_serialisation(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    # The hash is what session state and the executor compare, so it has to
    # survive a JSON round trip.
    plan = _plan(decision(), evidence_requirements=list(requirements))

    restored = Plan.model_validate_json(plan.model_dump_json())

    assert plan_revision(restored) == plan_revision(plan)


def test_a_plan_with_every_requirement_kind_round_trips_through_json(
    decision: DecisionFactory,
) -> None:
    plan = _every_kind_plan(decision(fixes=DecisionFixes(among=RankAmong.POPULATION)))

    restored = Plan.model_validate_json(plan.model_dump_json())

    assert restored == plan
    assert plan_revision(restored) == plan_revision(plan)


def test_json_without_kind_is_rejected(decision: DecisionFactory) -> None:
    plan = _every_kind_plan(decision(fixes=DecisionFixes(among=RankAmong.POPULATION)))
    raw = plan.model_dump(mode="json")
    del raw["evidence_requirements"][0]["kind"]

    with pytest.raises(ValidationError):
        Plan.model_validate(raw)


def test_json_without_answer_mode_loads_as_series(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # A plan persisted before `answer_mode` existed carries no such key. Its
    # terminal step, a raw retrieval of a dataset requirement nothing reads,
    # is exactly what `answer_mode=computed` would reject, proving the
    # missing key resolves to `series` rather than to the field's own prior
    # default.
    dataset = requirement()
    derived = derived_requirement()
    population = requirement(id="population", measure="population")
    plan = _plan(
        decision(),
        answer_mode=AnswerMode.SERIES,
        evidence_requirements=[dataset, derived, population],
    )
    raw = plan.model_dump(mode="json")
    del raw["answer_mode"]
    text = json.dumps(raw)

    restored = Plan.model_validate_json(text)
    assert restored.answer_mode == AnswerMode.SERIES

    state: dict[str, Any] = {PLAN_STATE_KEY: text}
    loaded = load_plan(state)
    assert loaded is not None
    assert loaded.answer_mode == AnswerMode.SERIES


@pytest.mark.parametrize(
    "tagged",
    [
        DatasetRequirement,
        AggregateRequirement,
        GrowthRequirement,
        ComparisonRequirement,
        RankRequirement,
        ShareOfRequirement,
        JoinRequirement,
        FilterRequirement,
        ValueThreshold,
        PeriodThreshold,
        EntityIn,
    ],
    ids=lambda cls: cls.__name__,
)
def test_the_json_schema_advertises_kind_as_required(tagged: type[TaggedModel]) -> None:
    # `kind` has a default so code constructing one need not state it, but a
    # model proposing from the schema has to: the union picks a member by it.
    assert tagged.model_json_schema()["required"][0] == "kind"


def test_the_plan_schema_advertises_kind_as_required_for_every_tagged_part() -> None:
    # Eight requirements, the four predicates a filter carries, and the eight
    # operations a plan's `translation` holds.
    definitions = Plan.model_json_schema()["$defs"]
    tagged = {
        name: schema
        for name, schema in definitions.items()
        if "kind" in schema.get("properties", {})
    }

    assert len(tagged) == 20
    assert all(schema["required"][0] == "kind" for schema in tagged.values())


# --- Decisions -----------------------------------------------------------------


def test_decision_rejects_a_chosen_value_not_in_options(decision: DecisionFactory) -> None:
    with pytest.raises(ValidationError, match="not one of the stated options"):
        decision(chosen="total debt service")  # typo: not one of the two options


def test_decision_rejects_options_edited_without_updating_chosen(
    decision: DecisionFactory,
) -> None:
    with pytest.raises(ValidationError, match="not one of the stated options"):
        decision(options=["debt service ratio", "debt-to-GNI ratio"])  # chosen unchanged


def test_a_decision_states_at_least_two_options(decision: DecisionFactory) -> None:
    # A decision whose only option is what was chosen records no decision.
    with pytest.raises(ValidationError, match="at least two options"):
        decision(options=["annual debt service"])


def test_the_single_option_rejection_says_to_delete_not_pad(decision: DecisionFactory) -> None:
    # A retry that pads `options` with an uninvestigated second option
    # rather than dropping the decision would record a choice that reads as
    # considered but never was, so the message has to tell a caller what to
    # do, not only restate the rule.
    with pytest.raises(ValidationError, match="delete this decision"):
        decision(options=["annual debt service"])


def test_a_decision_does_not_list_the_same_option_twice(decision: DecisionFactory) -> None:
    with pytest.raises(
        ValidationError, match=r"decision option must be unique.*annual debt service"
    ):
        decision(options=["annual debt service", "annual debt service"])


@pytest.mark.parametrize("field", ["description", "chosen", "rationale"])
def test_a_decision_does_not_leave_its_prose_blank(field: str, decision: DecisionFactory) -> None:
    with pytest.raises(ValidationError, match="at least 1 character"):
        decision(**{field: ""})


def test_a_decision_may_state_what_it_fixes(decision: DecisionFactory) -> None:
    fixed = decision(fixes=DecisionFixes(measure="external debt service paid"))

    restored = Decision.model_validate_json(fixed.model_dump_json())

    assert restored == fixed
    assert restored.fixes is not None
    assert restored.fixes.period is None


def test_every_decision_fixes_field_is_declared_by_a_requirement_class() -> None:
    # A static check over the requirement classes' own `model_fields`
    # catches a new `DecisionFixes` field going inert, so a harmless,
    # uncompared field never becomes a plan rejection through a runtime
    # `_check_every_fix_is_compared` rule.
    requirement_field_names = {
        name
        for cls in (
            DatasetRequirement,
            AggregateRequirement,
            GrowthRequirement,
            ComparisonRequirement,
            RankRequirement,
            ShareOfRequirement,
            JoinRequirement,
            FilterRequirement,
        )
        for name in cls.model_fields
    }

    undeclared = sorted(set(DecisionFixes.model_fields) - requirement_field_names)

    assert undeclared == []


# --- Limits on the plan's text and collections -----------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _recorded_plans() -> list[tuple[str, dict[str, Any]]]:
    """Every plan the checked-in eval sets record, as `(where, plan)`.

    An eval case seeds the planner's session with a plan as JSON text under
    `planner:plan`. These are plans a model wrote, or a maintainer wrote to
    look like one, and a limit must leave them alone.
    """
    found: list[tuple[str, dict[str, Any]]] = []
    for path in sorted((_REPO_ROOT / "evals").rglob("*.evalset.json")):
        for index, case in enumerate(json.loads(path.read_text())["eval_cases"]):
            recorded = (case.get("session_input") or {}).get("state", {}).get("planner:plan")
            if recorded is not None:
                found.append((f"{path.name}[{index}]", json.loads(recorded)))
    return found


def test_every_recorded_plan_validates_under_the_limits() -> None:
    # The count is asserted so a moved eval directory cannot leave the loop
    # below empty and the test green.
    recorded = _recorded_plans()
    assert len(recorded) >= 12, f"found only {len(recorded)} recorded plans"

    for where, plan in recorded:
        try:
            Plan.model_validate(plan)
        except ValidationError as error:
            pytest.fail(f"{where} no longer validates: {error}")


def _step_into(record: Any, path: tuple[str | int, ...]) -> Any:
    for step in path:
        record = record[step]
    return record


def test_every_decision_and_requirement_validates_at_its_limits() -> None:
    record = field_limits_record()

    assert len(record["decisions"]) == MAX_PLAN_DECISIONS
    assert len(record["evidence_requirements"]) == 24
    assert {requirement["kind"] for requirement in record["evidence_requirements"]} == {
        "dataset",
        "aggregate",
        "growth",
        "comparison",
        "rank",
        "share_of",
        "join",
        "filter",
    }
    for decision in record["decisions"]:
        Decision.model_validate(decision)
    requirement_type = TypeAdapter(EvidenceRequirement)
    for requirement in record["evidence_requirements"]:
        requirement_type.validate_python(requirement)


def test_a_plan_with_every_field_at_its_limit_is_refused_for_its_size_alone() -> None:
    # The field limits multiply through the nesting, so a plan at all of them is
    # far past what a brief can carry. The size limit is the only error, since
    # every part of it validates.
    record = field_limits_record()
    assert record_chars(record) > 100 * MAX_PLAN_RECORD_CHARS

    with pytest.raises(ValidationError) as refused:
        Plan.model_validate(record)

    assert [error["type"] for error in refused.value.errors()] == ["value_error"]
    assert "characters as serialised" in str(refused.value)


# Every free-text field and collection of a plan, as the path to it in a plan's
# JSON and the limit it carries. A dataset is `dataset_0` and a filter
# `filter_0`. The other kinds are named by their own id. `str` limits are in
# characters and `list` limits in items. Every path is checked to sit at its
# limit in `field_limits_record` before one more is added, so a path that
# drifted off its field cannot pass on a value that was never at the limit.
_LIMITS: list[tuple[str, tuple[str | int, ...], int]] = [
    ("Plan.question", ("question",), MAX_QUESTION_CHARS),
    (
        "Decision.description",
        ("decisions", 0, "description"),
        MAX_DECISION_DESCRIPTION_CHARS,
    ),
    ("Decision.rationale", ("decisions", 0, "rationale"), MAX_DECISION_RATIONALE_CHARS),
    ("Decision.chosen", ("decisions", 0, "chosen"), MAX_OPTION_CHARS),
    ("Decision.options", ("decisions", 0, "options"), MAX_DECISION_OPTIONS),
    ("Decision.options[]", ("decisions", 0, "options", 1), MAX_OPTION_CHARS),
    ("Fixes.entities", ("decisions", 0, "fixes", "entities"), MAX_ENTITIES),
    (
        "Fixes.entities[].name",
        ("decisions", 0, "fixes", "entities", 0, "name"),
        MAX_ENTITY_NAME_CHARS,
    ),
    (
        "Fixes.entities[].role",
        ("decisions", 0, "fixes", "entities", 0, "role"),
        MAX_ENTITY_ROLE_CHARS,
    ),
    ("Fixes.across", ("decisions", 0, "fixes", "across"), MAX_ENTITY_ROLE_CHARS),
    ("Fixes.measure", ("decisions", 0, "fixes", "measure"), MAX_MEASURE_CHARS),
    ("Fixes.axis", ("decisions", 0, "fixes", "axis"), MAX_ENTITY_ROLE_CHARS),
    ("Fixes.group_by", ("decisions", 0, "fixes", "group_by"), MAX_GROUP_BY),
    ("Fixes.predicates", ("decisions", 0, "fixes", "predicates"), MAX_PREDICATES),
    (
        "Fixes.predicates[].names",
        ("decisions", 0, "fixes", "predicates", 0, "names"),
        MAX_ENTITIES,
    ),
    (
        "Fixes.predicates[].names[]",
        ("decisions", 0, "fixes", "predicates", 0, "names", 0),
        MAX_ENTITY_NAME_CHARS,
    ),
    (
        "Fixes.predicates[].role",
        ("decisions", 0, "fixes", "predicates", 0, "role"),
        MAX_ENTITY_ROLE_CHARS,
    ),
    (
        "Dataset.decision_ids",
        ("evidence_requirements", 0, "decision_ids"),
        MAX_PLAN_DECISIONS,
    ),
    ("Dataset.measure", ("evidence_requirements", 0, "measure"), MAX_MEASURE_CHARS),
    ("Dataset.entities", ("evidence_requirements", 0, "entities"), MAX_ENTITIES),
    (
        "Dataset.entities[].name",
        ("evidence_requirements", 0, "entities", 0, "name"),
        MAX_ENTITY_NAME_CHARS,
    ),
    (
        "Dataset.entities[].role",
        ("evidence_requirements", 0, "entities", 0, "role"),
        MAX_ENTITY_ROLE_CHARS,
    ),
    ("Dataset.across", ("evidence_requirements", 0, "across"), MAX_ENTITY_ROLE_CHARS),
    ("Dataset.source", ("evidence_requirements", 0, "source"), MAX_SOURCE_CHARS),
    ("Filter.predicates", ("evidence_requirements", 8, "predicates"), MAX_PREDICATES),
    (
        "Filter.predicates[].names",
        ("evidence_requirements", 8, "predicates", 0, "names"),
        MAX_ENTITIES,
    ),
    (
        "Filter.predicates[].names[]",
        ("evidence_requirements", 8, "predicates", 0, "names", 0),
        MAX_ENTITY_NAME_CHARS,
    ),
    (
        "Filter.predicates[].role",
        ("evidence_requirements", 8, "predicates", 0, "role"),
        MAX_ENTITY_ROLE_CHARS,
    ),
    ("Aggregate.group_by", ("evidence_requirements", 18, "group_by"), MAX_GROUP_BY),
    ("Rank.axis", ("evidence_requirements", 20, "axis"), MAX_ENTITY_ROLE_CHARS),
    ("ShareOf.axis", ("evidence_requirements", 21, "axis"), MAX_ENTITY_ROLE_CHARS),
    (
        "ShareOf.numerator.entity",
        ("evidence_requirements", 21, "numerator", "entity"),
        MAX_ENTITY_NAME_CHARS,
    ),
    (
        "ShareOf.denominator.entity",
        ("evidence_requirements", 21, "denominator", "entity"),
        MAX_ENTITY_NAME_CHARS,
    ),
    (
        "Comparison.left.entity",
        ("evidence_requirements", 22, "left", "entity"),
        MAX_ENTITY_NAME_CHARS,
    ),
    (
        "Comparison.right.entity",
        ("evidence_requirements", 22, "right", "entity"),
        MAX_ENTITY_NAME_CHARS,
    ),
    ("Join.decision_ids", ("evidence_requirements", 23, "decision_ids"), MAX_PLAN_DECISIONS),
]

# Discriminator tags pydantic adds to an error's location inside a union.
_UNION_TAGS = frozenset(
    {
        "dataset",
        "aggregate",
        "growth",
        "comparison",
        "rank",
        "share_of",
        "join",
        "filter",
        "value_threshold",
        "period_threshold",
        "entity_in",
    }
)


@pytest.mark.parametrize(("label", "path", "limit"), _LIMITS, ids=[item[0] for item in _LIMITS])
def test_one_more_than_a_limit_is_refused_naming_the_field(
    label: str, path: tuple[str | int, ...], limit: int
) -> None:
    record = field_limits_record()
    # The paths above index into the maximal plan's requirements by position.
    ids = [requirement["id"] for requirement in record["evidence_requirements"]]
    assert (ids[0], ids[8], ids[18], ids[20], ids[21], ids[22], ids[23]) == (
        "dataset_0",
        "filter_0",
        "totals",
        "rank",
        "share",
        "comparison",
        "joined",
    )
    value = _step_into(record, path)
    assert len(value) == limit, f"{label} is {len(value)} in the maximal plan, not its {limit}"

    over = copy.deepcopy(record)
    if isinstance(value, str):
        _step_into(over, path[:-1])[path[-1]] = value + "x"
    else:
        _step_into(over, path).append(copy.deepcopy(value[-1]))

    with pytest.raises(ValidationError) as refused:
        Plan.model_validate(over)

    at_the_field = [
        error
        for error in refused.value.errors()
        if tuple(step for step in error["loc"] if step not in _UNION_TAGS) == path
    ]
    assert [error["type"] for error in at_the_field if "too_long" in error["type"]], (
        f"{label}: {limit + 1} was refused, but not as a length error at {path}: "
        f"{[(error['loc'], error['type']) for error in refused.value.errors()]}"
    )
    assert f"at most {limit}" in str(refused.value)


def test_a_hundred_kilobyte_measure_is_refused_at_the_schema(
    requirement: RequirementFactory,
) -> None:
    with pytest.raises(ValidationError, match=r"measure\n\s+String should have at most 200"):
        requirement(measure="m" * 100_000)


def test_a_plan_at_the_size_limit_validates_and_one_character_over_is_refused() -> None:
    at_the_limit = plan_record(MAX_PLAN_RECORD_CHARS)
    assert record_chars(at_the_limit) == MAX_PLAN_RECORD_CHARS

    plan = Plan.model_validate(at_the_limit)

    assert len(plan.decisions) == MAX_PLAN_DECISIONS
    with pytest.raises(ValidationError) as refused:
        Plan.model_validate(plan_record(MAX_PLAN_RECORD_CHARS + 1))
    assert f"the plan is {MAX_PLAN_RECORD_CHARS + 1:,} characters" in str(refused.value)


def test_the_size_refusal_names_the_parts_that_take_the_most() -> None:
    with pytest.raises(ValidationError) as refused:
        Plan.model_validate(plan_record(MAX_PLAN_RECORD_CHARS + 500))

    message = str(refused.value)
    assert f"over the {MAX_PLAN_RECORD_CHARS:,} a plan may take" in message
    # The padding sits on the first dataset's entities, which is the largest part.
    assert "requirement 'dataset_0'" in message
    assert "Shorten the longest parts" in message


def test_the_size_limit_counts_what_a_model_writes_and_not_what_the_compiler_adds(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    draft = _plan(decision(), evidence_requirements=list(requirements), status=PlanStatus.DRAFT)
    bloated = _workflow().model_copy(
        update={"sources": (SelectedSource(spec_id=SPEC, sentence="S" * 50_000, has_data=True),)}
    )
    compiled = draft.with_translation(translation=_translation(), workflow=bloated)

    dumped = compiled.model_dump(mode="json")
    reloaded = Plan.model_validate(dumped)

    assert reloaded == compiled
    assert len(json.dumps(dumped)) > 3 * MAX_PLAN_RECORD_CHARS


def test_an_amendment_that_pushes_the_plan_past_the_size_limit_is_refused() -> None:
    plan = Plan.model_validate(plan_record(MAX_PLAN_RECORD_CHARS - 20))
    longer = plan.decisions[0].model_copy(update={"rationale": "r" * 60})

    with pytest.raises(ValidationError, match="characters as serialised"):
        plan.amend(decisions=[longer])


# The `Plan` fields a model does not write: the compiler fills `translation` and
# `workflow` from search and tool results, and `approve` records the revision
# hash. `propose_plan` accepts none of them, so their text is not model output
# and the limits above do not apply to it.
_SYSTEM_AUTHORED = frozenset({"translation", "workflow", "approved_revision"})


def test_every_string_and_collection_a_model_writes_into_a_plan_is_bounded() -> None:
    # The schema is what the model reads and what pydantic enforces, so
    # walking it catches a field added later without a limit, which no list of
    # fields in a test would.
    # JSON schema cannot state the limit on the plan's total size, which
    # `Plan._record_fits_the_size_limit` enforces and
    # `test_a_plan_at_the_size_limit_validates_and_one_character_over_is_refused`
    # checks.
    schema = Plan.model_json_schema()
    assert set(schema["properties"]) >= _SYSTEM_AUTHORED
    authored = {
        **schema,
        "properties": {
            name: field
            for name, field in schema["properties"].items()
            if name not in _SYSTEM_AUTHORED
        },
    }

    unbounded = unbounded_paths(authored, schema["$defs"], "Plan")

    assert unbounded == []


def test_the_bound_check_reports_a_string_an_array_and_a_map_with_no_limit() -> None:
    # The control for the check above. It walks references, unions and arrays
    # and reports each kind of gap, so an empty result on the plan comes from
    # a bounded plan and a walk that reaches every field.
    schema = {
        "type": "object",
        "properties": {
            "bare": {"type": "string"},
            "listed": {"type": "array", "items": {"$ref": "#/$defs/Inner"}},
            "either": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "keyed": {"type": "object", "additionalProperties": {"type": "string"}},
            "fine": {"type": "string", "maxLength": 3},
            "named": {"type": "string", "enum": ["a"]},
            "id": {"type": "string", "pattern": r"^[a-z][a-z0-9_]{0,63}$"},
            "loose": {"type": "string", "pattern": r"^[a-z]+$"},
        },
    }
    definitions = {"Inner": {"type": "object", "properties": {"text": {"type": "string"}}}}

    found = unbounded_paths(schema, definitions, "Root")

    assert found == [
        "Root.bare: a string with no maxLength",
        "Root.listed: an array with no maxItems",
        "Root.listed[]<Inner>.text: a string with no maxLength",
        "Root.either: a string with no maxLength",
        "Root.keyed: an object whose keys are unbounded",
        "Root.keyed[]: a string with no maxLength",
        "Root.loose: a string with no maxLength",
    ]


# --- The size limit counts characters ------------------------------------------------


def _chars_with_escapes(record: dict[str, Any]) -> int:
    return len(json.dumps(record, separators=(",", ":")))


def _chars_without_escapes(record: dict[str, Any]) -> int:
    return len(json.dumps(record, separators=(",", ":"), ensure_ascii=False))


def test_the_size_limit_counts_characters_and_not_their_json_escapes() -> None:
    # The seam writes free text into the brief as it is, so an accented letter is one
    # character there and six once JSON-escaped. Six decisions and the question at their
    # limits are under the limit as written and over it as escaped.
    record = plan_record()
    for decision in record["decisions"][:6]:
        decision["description"] = "é" * MAX_DECISION_DESCRIPTION_CHARS
    record["question"] = "é" * MAX_QUESTION_CHARS
    assert _chars_with_escapes(record) > MAX_PLAN_RECORD_CHARS
    assert _chars_without_escapes(record) < MAX_PLAN_RECORD_CHARS

    plan = Plan.model_validate(record)

    assert plan.question == "é" * MAX_QUESTION_CHARS


def test_the_size_refusal_reports_the_size_in_characters_as_written() -> None:
    record = plan_record()
    for decision in record["decisions"]:
        decision["description"] = "é" * MAX_DECISION_DESCRIPTION_CHARS
    assert _chars_without_escapes(record) > MAX_PLAN_RECORD_CHARS

    with pytest.raises(ValidationError) as refused:
        Plan.model_validate(record)

    assert f"the plan is {_chars_without_escapes(record):,} characters" in str(refused.value)


# --- A plan states its question ---------------------------------------------------------


@pytest.mark.parametrize("blank", ["", " ", "  \n\t "])
def test_a_plan_without_a_question_is_refused(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...], blank: str
) -> None:
    with pytest.raises(ValidationError) as refused:
        Plan(question=blank, decisions=[decision()], evidence_requirements=list(requirements))

    assert [error["loc"] for error in refused.value.errors()] == [("question",)]


@pytest.mark.parametrize("blank", ["", " ", "  \n\t "])
def test_an_amendment_cannot_blank_the_question(decision: DecisionFactory, blank: str) -> None:
    plan = _plan(decision())

    with pytest.raises(ValidationError) as refused:
        plan.amend(question=blank)

    assert [error["loc"] for error in refused.value.errors()] == [("question",)]
    assert plan.question == "How dependent is Kenya on Chinese lending?"


def test_a_blank_question_tells_the_model_what_to_state(decision: DecisionFactory) -> None:
    with pytest.raises(ValidationError, match="state the question this plan answers"):
        Plan(question="   ", decisions=[decision()])


# --- The limits on entities, options and the question -----------------------------------


def test_a_requirement_for_a_bloc_of_twenty_seven_countries_validates(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
) -> None:
    # The EU has 27 members. data-mcp answers 50 entity names in one call, so a
    # limit below 27 could not state a requirement the tool can carry out.
    countries = [Entity(name=f"Member state {index:02d}", role="recipient") for index in range(27)]

    plan = Plan(
        question="How much aid did EU member states receive?",
        decisions=[decision()],
        evidence_requirements=[requirement(entities=countries), derived_requirement()],
    )

    assert len(plan.evidence_requirements[0].entities) == 27  # type: ignore[union-attr]


def test_the_entity_limit_is_the_number_of_names_data_mcp_accepts_in_one_call() -> None:
    data_mcp_limit = 50

    assert data_mcp_limit == MAX_ENTITIES
    EntityIn(names=tuple(f"n{index}" for index in range(data_mcp_limit)))
    with pytest.raises(ValidationError, match=f"at most {data_mcp_limit} items"):
        EntityIn(names=tuple(f"n{index}" for index in range(data_mcp_limit + 1)))


def test_a_decision_may_list_every_candidate_a_search_returns(decision: DecisionFactory) -> None:
    options = [f"candidate {index:02d}" for index in range(MAX_DECISION_OPTIONS)]

    listed = decision(options=options, chosen=options[0])

    assert MAX_DECISION_OPTIONS == 24
    assert listed.options == tuple(options)


# --- Cutting user text ----------------------------------------------------------------------


def test_text_within_its_cap_is_left_whole() -> None:
    assert cut_user_text("a question", 10) == "a question"
    assert cut_user_text("", 5) == ""


def test_text_over_its_cap_is_cut_to_a_prefix_ending_in_the_marker() -> None:
    cut = cut_user_text("q" * 5_000, MAX_QUESTION_CHARS)

    assert len(cut) == MAX_QUESTION_CHARS
    assert cut == "q" * (MAX_QUESTION_CHARS - len(CUT_MARKER)) + CUT_MARKER


@pytest.mark.parametrize("cap", [0, 1, len(CUT_MARKER) - 1, len(CUT_MARKER), len(CUT_MARKER) + 1])
def test_a_cap_smaller_than_the_marker_still_bounds_the_result(cap: int) -> None:
    assert len(cut_user_text("x" * 100, cap)) <= cap


def test_a_cut_question_is_a_valid_plan_question(
    decision: DecisionFactory, requirements: tuple[EvidenceRequirement, ...]
) -> None:
    question = cut_user_text("q" * 100_000, MAX_QUESTION_CHARS)

    plan = Plan(question=question, decisions=[decision()], evidence_requirements=list(requirements))

    assert plan.question == question
    assert MAX_QUESTION_CHARS == 2_000
