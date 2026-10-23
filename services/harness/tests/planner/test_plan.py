"""Tests for the `Plan` object: schema, versioning, status transitions, revisions."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

import pytest
from common.hashed_fields import revision_content
from compiler.sources import source_from_spec_block
from data_mcp.search_tool import SearchSource
from planner.operations import (
    AggregateOperation,
    FilterOperation,
    GrowthOperation,
    RankOperation,
    RetrieveOperation,
    ShareOfOperation,
    Side,
    Translation,
)
from planner.parties import PartyRole
from planner.plan import (
    CUT_MARKER,
    MAX_ACROSS_WORDS,
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
    AskAnswer,
    AskBreakdown,
    AskRoleDroppedError,
    ComparisonRequirement,
    Coverage,
    DatasetRequirement,
    Decision,
    DecisionDraft,
    DecisionFixes,
    Decline,
    Entity,
    EvidenceRequirement,
    FilterRequirement,
    FirstYear,
    FixedField,
    FullHistory,
    GrowthRequirement,
    JoinRequirement,
    Period,
    Plan,
    PlanStatus,
    QuotientRequirement,
    RankRequirement,
    RequirementSide,
    ResolvedBy,
    ShareOfRequirement,
    SumOfRequirement,
    Unit,
    UnsupportedOperation,
    UnsupportedPlanError,
    _build_shapes,
    authored_text,
    cut_user_text,
    frame_shapes,
    full_history,
    load_plan,
    period_window,
    plan_revision,
    retrieves_period,
    rolls_from_its_start,
    settle_decisions,
    windowed_pin,
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
    OpenPeriodFault,
    PeriodThreshold,
    PlaceLevel,
    PlaceType,
    Predicate,
    RankAmong,
    RankDirection,
    Reduction,
    RelativePeriod,
    RollingIssue,
    RollingPlanError,
    RollingWindow,
    Severity,
    TaggedModel,
    ValueThreshold,
)
from planner.shapes import FrameShape, TimeLabel
from planner.workflow import (
    ArtifactKind,
    Binding,
    EvidenceBinding,
    ResolvedGroup,
    SelectedSource,
    ToolId,
    Workflow,
    WorkflowStep,
)
from pydantic import TypeAdapter, ValidationError
from submission.service import _Patch

from ..context_probe import field_limits_record, plan_record, record_chars, unbounded_paths

# The `decision`, `requirement`, `derived_requirement` and `requirements`
# fixtures, from the harness suite's conftest.py. Annotated structurally
# rather than imported: a test module importing a conftest by name is
# ambiguous when the workspace root has one too.
DecisionFactory = Callable[..., Decision]
RequirementFactory = Callable[..., DatasetRequirement]
DerivedRequirementFactory = Callable[..., AggregateRequirement]

_PIN = "spec_" + "a" * 32


# The first year on record `_plan` gives every open dataset requirement.
_FIRST_YEAR = 1990


def _plan(decision: Decision, **overrides: Any) -> Plan:
    """A plan awaiting the user, as the translator leaves it; a draft where a test says so.

    Each dataset requirement stating `period_end_open` has its dataset's
    first year on record, `_FIRST_YEAR`, as `propose_plan` reads it for a
    pinned one, unless the test states `first_years`.
    """
    fields: dict[str, Any] = {
        "question": "How dependent is Kenya on Chinese lending?",
        "decisions": [decision],
        "status": PlanStatus.PENDING_APPROVAL,
    } | overrides
    fields.setdefault(
        "first_years",
        [
            FirstYear(requirement_id=requirement.id, year=_FIRST_YEAR)
            for requirement in fields.get("evidence_requirements", ())
            if isinstance(requirement, DatasetRequirement) and requirement.period_end_open
        ],
    )
    return Plan(**fields)


def _unsupported_operation(exc: ValidationError) -> UnsupportedOperation | None:
    """The operation an `UnsupportedPlanError` behind `exc` names, or `None` for a malformed draft."""
    for error in exc.errors():
        cause = error.get("ctx", {}).get("error")
        if isinstance(cause, UnsupportedPlanError):
            return cause.operation
    return None


def _at(plan: Plan, status: PlanStatus) -> Plan:
    """Place `plan` at `status` directly, for tests about what a status permits."""
    return plan.model_copy(update={"status": status})


def _every_kind_plan(decision: Decision) -> Plan:
    """Build a plan stating one requirement of every kind, wired into a valid DAG.

    `dataset_a` feeds `totals_a` (aggregate), which feeds `growth_a` (growth),
    and also feeds `rank_a` (rank, among='population', reducing the decade to
    one value per entity -- `decision` has to cite `fixes.among='population'`
    for `rank_a`'s population claim to validate). `dataset_b` feeds
    `filtered_b` (filter). `joined` (join), `share_a` (share_of),
    `quotient_a` (quotient_of), `comparison_a` (comparison) and `sum_a`
    (sum_of) each read both `totals_a` and `filtered_b`; `comparison_a` is
    terminal.
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
    quotient_a = QuotientRequirement(
        id="quotient_a",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="totals_a"),
        denominator=RequirementSide(input_id="filtered_b"),
    )
    comparison_a = ComparisonRequirement(
        id="comparison_a",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="filtered_b"),
        subtrahend=RequirementSide(input_id="totals_a"),
    )
    sum_a = SumOfRequirement(
        id="sum_a",
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
            quotient_a,
            comparison_a,
            sum_a,
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


def test_amend_clears_the_approval(decision: DecisionFactory) -> None:
    # The user approved the previous version's content; the new content has
    # not been looked at.
    approved = _plan(decision()).approve(at=datetime.now(UTC))

    amended = approved.amend()

    assert approved.approved_revision is not None
    assert approved.approved_at is not None
    assert amended.approved_revision is None
    assert amended.approved_at is None


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


def test_a_requirement_may_cover_a_whole_party_and_name_no_entity() -> None:
    dataset = DatasetRequirement(
        id="health_spending",
        decision_ids=["measure"],
        measure="health spending per person",
        across=("country",),
        period=Period(start=2022, end=2022),
    )

    assert dataset.entities == ()
    assert dataset.across == ("country",)


@pytest.mark.parametrize(
    ("role", "across", "word"),
    [
        ("recipient", ("recipient",), "recipient"),
        ("Recipient", ("recipient",), "recipient"),
        ("recipients", ("recipient",), "recipient"),
        ("recipient", ("recipients",), "recipients"),
        ("debtor", ("recipient",), "recipient"),
        ("recipient country", ("recipient",), "recipient"),
        ("country", ("countries",), "countries"),
        ("recipient", ("donor", "recipient"), "recipient"),
    ],
)
def test_a_requirement_names_no_entity_in_the_party_across_covers(
    requirement: RequirementFactory, role: str, across: tuple[str, ...], word: str
) -> None:
    # Kenya as a recipient, across every recipient, claims two populations at
    # once. The two are compared by the party each names, not by spelling.
    every = "country" if word == "countries" else "recipient"
    with pytest.raises(
        ValidationError,
        match=rf"across word '{word}' covers every {every}, so no entity may also be named "
        r"in that party; named: \['Kenya'\]",
    ):
        requirement(entities=[Entity(name="Kenya", role=role)], across=across)


@pytest.mark.parametrize(
    "across", [("donor", "donors"), ("donor", "provider"), ("country", "economy")]
)
def test_a_requirement_states_each_party_once_in_across(
    requirement: RequirementFactory, across: tuple[str, ...]
) -> None:
    with pytest.raises(ValidationError, match=r"`across` states each party once"):
        requirement(entities=[], across=across)


def test_a_requirement_may_cover_both_sides_of_a_flow(requirement: RequirementFactory) -> None:
    dataset = requirement(entities=[], across=("donor", "recipient"))

    assert dataset.across == ("donor", "recipient")


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
        r"member, drop the word from `across` and state the rank's among='population'",
    ):
        requirement(entities=[Entity(name="Kenya")], across=(across,))


def test_an_unroled_entity_may_sit_beside_a_flow_party(requirement: RequirementFactory) -> None:
    dataset = requirement(entities=[Entity(name="Kenya")], across=("donor",))

    assert dataset.across == ("donor",)


def test_a_requirement_names_an_entity_beside_the_party_across_covers(
    requirement: RequirementFactory,
) -> None:
    dataset = requirement(entities=[Entity(name="Kenya", role="recipient")], across=("donor",))

    assert dataset.across == ("donor",)


# --- Derived requirement validation --------------------------------------------


def test_dataset_entities_must_be_unique_by_name_and_role(requirement: RequirementFactory) -> None:
    with pytest.raises(ValidationError, match="unique by name and role"):
        requirement(
            entities=[
                Entity(name="Kenya", role="debtor"),
                Entity(name="Kenya", role="debtor"),
            ]
        )


def test_a_rank_window_may_be_one_year_and_refuses_an_end_before_its_start() -> None:
    fields: dict[str, Any] = {
        "id": "ranked",
        "direction": RankDirection.HIGHEST_FIRST,
        "among": RankAmong.POPULATION,
        "window_start": 2020,
    }
    requirement_fields = fields | {"decision_ids": ["measure"], "input_id": "debt"}
    operation_fields = fields | {"requirement_id": "ranked", "input": "debt"}
    last_year = RollingWindow(
        window_start=RelativePeriod(offset=0), window_end=RelativePeriod(offset=0)
    )

    for model, stated in ((RankRequirement, requirement_fields), (RankOperation, operation_fields)):
        one_year = model(**stated, window_end=2020, rolling=last_year)
        assert (one_year.window_start, one_year.window_end) == (2020, 2020)
        with pytest.raises(ValidationError, match="ends before it starts"):
            model(**stated, window_end=2019)


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
    with pytest.raises(ValidationError, match="minuend and subtrahend must differ"):
        ComparisonRequirement(
            id="excess",
            decision_ids=["measure"],
            minuend=RequirementSide(input_id="annual_debt_service"),
            subtrahend=RequirementSide(input_id="annual_debt_service"),
        )


def test_sum_of_sides_must_read_different_requirements() -> None:
    # Two entities of one series are an aggregate's to total; a sum adds two sources.
    with pytest.raises(ValidationError, match="its sides read different requirements"):
        SumOfRequirement(
            id="total",
            decision_ids=["measure"],
            left=RequirementSide(input_id="aid", entity="France"),
            right=RequirementSide(input_id="aid", entity="Germany"),
        )


@pytest.mark.parametrize("two_sided", [ShareOfRequirement, QuotientRequirement])
def test_share_of_and_quotient_of_sides_must_differ(
    two_sided: type[ShareOfRequirement | QuotientRequirement],
) -> None:
    with pytest.raises(ValidationError, match="numerator and denominator must differ"):
        two_sided(
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
        minuend=RequirementSide(input_id=agg.id),
        subtrahend=RequirementSide(input_id=dataset.id),
    )
    reads_comparison = AggregateRequirement(
        id="totals_of_excess",
        decision_ids=["measure"],
        input_id="excess",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )

    with pytest.raises(ValidationError) as refused:
        _plan(decision(), evidence_requirements=[dataset, agg, comparison, reads_comparison])
    assert _unsupported_operation(refused.value) is UnsupportedOperation.CALCULATION_OVER_COMBINED


def test_a_sum_of_feeds_a_further_calculation_as_one_series(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    bilateral = requirement(id="bilateral", period=Period(start=2010, end=2024))
    multilateral = requirement(
        id="multilateral",
        measure="imputed multilateral aid",
        period=Period(start=2011, end=2023),
    )
    total = SumOfRequirement(
        id="total",
        decision_ids=["measure"],
        left=RequirementSide(input_id=bilateral.id),
        right=RequirementSide(input_id=multilateral.id),
    )
    growth = GrowthRequirement(
        id="growth",
        decision_ids=["measure"],
        input_id=total.id,
        window_start=2012,
        window_end=2022,
    )

    plan = _plan(
        decision(),
        answer_mode=AnswerMode.COMPUTED,
        evidence_requirements=[bilateral, multilateral, total, growth],
    )

    shape = frame_shapes(plan)[total.id]
    assert shape.roles == ()
    assert shape.measures == 1
    assert shape.period == Period(start=2011, end=2023)


def test_a_calculation_the_question_states_outright_cites_no_decision(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # A total the question asks for, or the instruction prescribes, is not a
    # choice weighed among alternatives, so its requirement cites nothing.
    dataset = requirement()
    total = AggregateRequirement(
        id="total", input_id=dataset.id, reduction=Reduction.SUM, group_by=(GroupBy.PERIOD,)
    )

    plan = _plan(decision(), status=PlanStatus.DRAFT, evidence_requirements=[dataset, total])

    assert plan.requirement("total").decision_ids == ()


def test_a_draft_names_every_field_a_decision_fixes_but_the_pin() -> None:
    assert set(get_args(FixedField)) == set(DecisionFixes.model_fields) - {"source_spec_id"}


@pytest.mark.parametrize(
    ("periods", "fixed"),
    [
        ([Period(start=2015, end=2024)] * 2, Period(start=2015, end=2024)),
        ([Period(start=2015, end=2024), Period(start=2018, end=2024)], None),
        ([], None),
    ],
    ids=["agreeing", "disagreeing", "unstated"],
)
def test_settling_a_draft_reads_a_named_field_off_the_requirements_citing_it(
    requirement: RequirementFactory, periods: list[Period], fixed: Period | None
) -> None:
    """A value the citing requirements agree on is recorded; any other is left out, never refused."""
    draft = DecisionDraft(
        slot="years",
        description="Which years to read.",
        options=["2015-2024", "2018-2024"],
        chosen="2015-2024",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="r",
        fixes=("period",),
    )
    citing = [
        requirement(id=f"r{index}", decision_ids=["years"], period=period)
        for index, period in enumerate(periods)
    ]

    (settled,) = settle_decisions([draft], citing)

    assert (settled.fixes.period if settled.fixes else None) == fixed


@pytest.mark.parametrize("multilateral_start", [2010, 2011])
def test_a_total_of_two_sources_over_a_span_sums_the_yearly_series_first(
    decision: DecisionFactory, requirement: RequirementFactory, multilateral_start: int
) -> None:
    # Both sides over the question's own period, added year by year, then
    # the sum's one series totalled by an aggregate grouped by entity. The
    # sum_of keeps the years both report, so the plan holds whatever year
    # one source lacks without moving either period.
    bilateral = requirement(id="bilateral", period=Period(start=2010, end=2024))
    multilateral = requirement(
        id="multilateral",
        measure="imputed multilateral aid",
        period=Period(start=multilateral_start, end=2024),
    )
    total = SumOfRequirement(
        id="total",
        decision_ids=["measure"],
        left=RequirementSide(input_id=bilateral.id),
        right=RequirementSide(input_id=multilateral.id),
    )
    span = AggregateRequirement(
        id="span", input_id=total.id, reduction=Reduction.SUM, group_by=(GroupBy.ENTITY,)
    )

    plan = _plan(
        decision(),
        answer_mode=AnswerMode.COMPUTED,
        evidence_requirements=[bilateral, multilateral, total, span],
    )

    shape = frame_shapes(plan)[span.id]
    assert shape.time is TimeLabel.SPAN
    assert shape.roles == ()
    assert shape.period == Period(start=multilateral_start, end=2024)


@pytest.mark.parametrize(
    ("multilateral_start", "refused"), [(2011, None), (2012, "add the two yearly series")]
)
def test_a_total_of_two_sources_over_a_span_sums_each_sides_years_first(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    multilateral_start: int,
    refused: str | None,
) -> None:
    # Each side's years summed first, then the two totals added: the two
    # spans match only over one stated period, and a sum_of over spans that
    # differ is refused naming the order that holds, the yearly series
    # added first.
    bilateral = requirement(id="bilateral", period=Period(start=2011, end=2024))
    multilateral = requirement(
        id="multilateral",
        measure="imputed multilateral aid",
        period=Period(start=multilateral_start, end=2024),
    )
    totals = [
        AggregateRequirement(
            id=f"{side.id}_total",
            decision_ids=["measure"],
            input_id=side.id,
            reduction=Reduction.SUM,
            group_by=(GroupBy.ENTITY,),
        )
        for side in (bilateral, multilateral)
    ]
    total = SumOfRequirement(
        id="total",
        decision_ids=["measure"],
        left=RequirementSide(input_id=totals[0].id),
        right=RequirementSide(input_id=totals[1].id),
    )
    requirements = [bilateral, multilateral, *totals, total]

    if refused is not None:
        with pytest.raises(ValidationError, match=refused):
            _plan(decision(), answer_mode=AnswerMode.COMPUTED, evidence_requirements=requirements)
        return
    plan = _plan(decision(), answer_mode=AnswerMode.COMPUTED, evidence_requirements=requirements)

    shape = frame_shapes(plan)[total.id]
    assert shape.time is TimeLabel.SPAN
    assert shape.period == Period(start=2011, end=2024)


def test_a_sum_of_side_holding_a_share_is_refused(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    bilateral = requirement(id="bilateral")
    multilateral = requirement(id="multilateral", measure="imputed multilateral aid")
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=bilateral.id),
        denominator=RequirementSide(input_id=multilateral.id),
    )
    total = SumOfRequirement(
        id="total",
        decision_ids=["measure"],
        left=RequirementSide(input_id=share.id),
        right=RequirementSide(input_id=multilateral.id),
    )

    with pytest.raises(ValidationError, match="does not add up"):
        _plan(decision(), evidence_requirements=[bilateral, multilateral, share, total])


@pytest.mark.parametrize(
    "answered",
    [(), (AskAnswer(ask=0, outputs=("annual_debt_service",)),)],
    ids=["unanswered", "answered"],
)
def test_a_computed_plan_rejects_an_unread_dataset_requirement(
    decision: DecisionFactory, requirement: RequirementFactory, answered: tuple[AskAnswer, ...]
) -> None:
    # Under answer_mode=computed, a retrieved series carries a reference and
    # statistics, not the figure a computed answer asks for, so a dataset
    # requirement nothing reads leaves the plan without one, whether or not
    # an ask names it.
    unread = requirement()

    with pytest.raises(ValidationError, match="unread dataset requirements") as refused:
        _plan(
            decision(),
            answer_mode=AnswerMode.COMPUTED,
            evidence_requirements=[unread],
            asks=["Kenya's debt service"] if answered else [],
            answered=answered,
        )
    assert _unsupported_operation(refused.value) is None


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


@pytest.mark.parametrize(
    ("narrowed", "entity", "match"),
    [
        ("annual_totals", "Nigeria", r"not one of \['Kenya'\]"),
        ("other_totals", "Kenya", r"carries no entity: it is a total, so the side names none"),
    ],
    ids=["unnamed-entity", "total"],
)
def test_a_requirement_side_entity_must_name_an_upstream_entity(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    narrowed: str,
    entity: str,
    match: str,
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
        minuend=RequirementSide(input_id=({agg.id, other_agg.id} - {narrowed}).pop()),
        subtrahend=RequirementSide(input_id=narrowed, entity=entity),
    )

    with pytest.raises(ValidationError, match=match):
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
        minuend=RequirementSide(input_id=agg_a.id),
        subtrahend=RequirementSide(input_id=joined.id, entity="Not a real entity"),
    )

    with pytest.raises(ValidationError) as refused:
        _plan(
            decision(),
            evidence_requirements=[dataset_a, dataset_b, agg_a, agg_b, joined, comparison],
        )
    assert _unsupported_operation(refused.value) is UnsupportedOperation.CALCULATION_OVER_COMBINED


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

    with pytest.raises(ValidationError) as refused:
        _plan(
            decision(), evidence_requirements=[dataset_a, dataset_b, agg_a, agg_b, joined, growth]
        )
    assert _unsupported_operation(refused.value) is UnsupportedOperation.CALCULATION_OVER_COMBINED


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

    with pytest.raises(
        ValidationError,
        match=r"over 2016-2018 with no reduction.*window_start and window_end to that year",
    ):
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
        pytest.param(
            Period(start=2015, end=2024), Period(start=2020, end=2020), id="window-is-one-year"
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
    ("reduction", "right_is_peak", "right_period", "refused", "open_to"),
    [
        pytest.param(Reduction.MAX, False, _DECADE, None, (), id="max-vs-series"),
        pytest.param(
            Reduction.MAX,
            False,
            _DECADE,
            "newer data moves the last year",
            ("debt",),
            id="max-open-to-the-data-vs-a-fixed-series",
        ),
        pytest.param(
            Reduction.MAX,
            False,
            _DECADE,
            "newer data moves the last year",
            ("debt", "population"),
            id="max-vs-a-series-open-to-other-data",
        ),
        pytest.param(Reduction.MIN, False, _DECADE, None, (), id="min-vs-series"),
        pytest.param(
            Reduction.MAX,
            False,
            Period(start=2015, end=2019),
            "no year to match",
            (),
            id="max-vs-a-series-missing-picked-years",
        ),
        pytest.param(
            Reduction.SUM, False, _DECADE, "carry single years", (), id="sum-labels-the-span"
        ),
        pytest.param(Reduction.MAX, True, _DECADE, "both sides hold", (), id="max-vs-max"),
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
    open_to: tuple[str, ...],
) -> None:
    # analysis-mcp labels a max or min row with the year its value came from,
    # so the peak over 2018-2022 compares against another source's years,
    # which have to hold every year a peak may fall in, today and once newer
    # data adds years to either. A sum is labelled 2018/2022, which matches
    # no single year, and two peaks meet only where both fell in the same
    # year.
    dataset = requirement(period=Period(start=2018, end=2022), period_end_open="debt" in open_to)
    peak = derived_requirement(reduction=reduction, group_by=(GroupBy.ENTITY,))
    population = requirement(
        id="population",
        measure="population",
        period=right_period,
        period_end_open="population" in open_to,
    )
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
        minuend=RequirementSide(input_id=right_id),
        subtrahend=RequirementSide(input_id=peak.id),
    )
    requirements.append(comparison)

    if refused is not None:
        with pytest.raises(ValidationError, match=refused):
            _plan(decision(), evidence_requirements=requirements)
    else:
        plan = _plan(decision(), evidence_requirements=requirements)
        assert plan.requirement("excess").subtrahend.input_id == peak.id


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
    return FilterRequirement(**common, predicates=(_PEAK_YEAR_FILTERS[kind],))


def _years_back(op: str, offset: int) -> PeriodThreshold:
    """A year threshold `offset` years back from 2022, the newest year, as it follows the newest data."""
    return PeriodThreshold(op=op, year=2022 + offset, rolling=RelativePeriod(offset=offset))


# Filters over the peak: a fixed first year, a fixed last year, and each of
# them following the newest data, a first year one year later than the
# peak's own first year as it follows the newest data, and one on an entity
# alone.
_PEAK_YEAR_FILTERS = {
    "year-filter": PeriodThreshold(op="gte", year=2020),
    "earlier-year-filter": PeriodThreshold(op="gte", year=2017),
    "upper-year-filter": PeriodThreshold(op="lte", year=2021),
    "rolling-upper-year-filter": _years_back("lte", 0),
    "rolling-year-filter": _years_back("gte", -2),
    "two-years-back-filter": _years_back("lte", -2),
    "six-years-back-filter": _years_back("gte", -6),
    "five-years-back-filter": _years_back("gte", -5),
    "after-ten-years-back-filter": _years_back("gt", -10),
    "ten-years-back-filter": _years_back("gte", -10),
    "entity-filter": EntityIn(names=("France",)),
}
# Filters in front of the peak: a first year six years back behind a fixed
# 2018, a first year ten years back, and a last year before the newest.
_PEAK_INPUT_FILTERS = {
    "trailing-start": (_years_back("gte", -6), PeriodThreshold(op="gte", year=2018)),
    "ten-years-back": (_years_back("gte", -10),),
    "before-the-newest": (_years_back("lt", 0),),
}


_FIRST_PASSED = "can move past the first year the filter keeps"
_LAST_PASSED = "can move past the last year the filter keeps"


@pytest.mark.parametrize(
    ("kind", "open_", "before", "refused"),
    [
        pytest.param(
            "growth", False, None, "growth reads each entity's series", id="growth-after-max"
        ),
        pytest.param(
            "per-period-rank", False, None, "rank counts every year", id="per-period-rank-after-max"
        ),
        pytest.param(
            "collapsing-rank", False, None, "rank counts every year", id="collapsing-rank-after-max"
        ),
        pytest.param(
            "aggregate-by-period",
            False,
            None,
            "aggregate by period",
            id="aggregate-by-period-after-max",
        ),
        pytest.param("year-filter", False, None, None, id="year-filter-after-max"),
        pytest.param("upper-year-filter", False, None, None, id="upper-year-filter-after-max"),
        pytest.param("year-filter", True, None, None, id="year-filter-after-max-of-open-years"),
        pytest.param(
            "upper-year-filter",
            True,
            None,
            _LAST_PASSED,
            id="upper-year-filter-after-max-of-open-years",
        ),
        pytest.param(
            "rolling-upper-year-filter",
            True,
            None,
            _LAST_PASSED,
            id="rolling-upper-year-filter-after-max-of-open-years",
        ),
        pytest.param(
            "rolling-year-filter",
            True,
            None,
            _FIRST_PASSED,
            id="rolling-year-filter-after-max-of-open-years",
        ),
        pytest.param(
            "entity-filter", True, "trailing-start", None, id="entity-filter-after-a-trailing-start"
        ),
        pytest.param(
            "year-filter", True, "trailing-start", None, id="year-filter-after-a-trailing-start"
        ),
        pytest.param(
            "earlier-year-filter",
            True,
            "trailing-start",
            None,
            id="earlier-year-filter-after-a-trailing-start",
        ),
        pytest.param(
            "six-years-back-filter",
            True,
            "trailing-start",
            None,
            id="the-trailing-years-back-after-a-trailing-start",
        ),
        pytest.param(
            "five-years-back-filter",
            True,
            "trailing-start",
            _FIRST_PASSED,
            id="fewer-years-back-after-a-trailing-start",
        ),
        pytest.param(
            "ten-years-back-filter",
            True,
            "ten-years-back",
            None,
            id="the-same-years-back-after-a-following-start",
        ),
        pytest.param(
            "after-ten-years-back-filter",
            True,
            "ten-years-back",
            _FIRST_PASSED,
            id="a-later-year-back-after-a-following-start",
        ),
        pytest.param(
            "two-years-back-filter",
            True,
            "before-the-newest",
            _LAST_PASSED,
            id="an-earlier-last-year-after-one-before-the-newest",
        ),
    ],
)
def test_a_reader_of_an_extreme_by_entity_follows_its_picked_years(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    kind: str,
    open_: bool,
    before: str | None,
    refused: str | None,
) -> None:
    # analysis-mcp holds a max by entity as one row per entity, at the year
    # its peak fell in: no series to grow, no year whose entities a per-year
    # reader could set side by side, and one year of each entity's window for
    # a rank's coverage to count. A year filter keeps the entities whose peak
    # falls in its years, which over years open to the data a later peak can
    # leave: past a last year, or behind a first year that follows the newest
    # further ahead than the peak's own. A peak's year moves only later, so a
    # fixed first year, or one following the newest no further ahead than
    # the peak's own first year, keeps every entity it keeps today.
    dataset = requirement(period=Period(start=2018, end=2022), period_end_open=open_)
    requirements: list[EvidenceRequirement] = [dataset]
    if before is not None:
        requirements.append(
            FilterRequirement(
                id="before",
                decision_ids=["measure"],
                input_id=dataset.id,
                predicates=_PEAK_INPUT_FILTERS[before],
            )
        )
    peak = derived_requirement(
        id="peak",
        input_id=requirements[-1].id,
        reduction=Reduction.MAX,
        group_by=(GroupBy.ENTITY,),
    )
    requirements += [peak, _after_peak(kind)]

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


@pytest.mark.parametrize(
    ("predicates", "window"),
    [
        pytest.param((PeriodThreshold(op="eq", year=2020),), {}, id="an-eq"),
        # The window's first year is one the filter removed, so it ranks 2023.
        pytest.param(
            (PeriodThreshold(op="ne", year=2022),),
            {"window_start": 2022, "window_end": 2023},
            id="a-window-starting-on-a-removed-year",
        ),
        pytest.param(
            (PeriodThreshold(op="ne", year=2022), PeriodThreshold(op="ne", year=2024)),
            {"window_start": 2022, "window_end": 2024},
            id="a-window-ending-on-removed-years",
        ),
    ],
)
def test_a_rank_over_a_filter_pinning_one_year_needs_no_reduction(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    predicates: tuple[PeriodThreshold, ...],
    window: dict[str, int],
) -> None:
    dataset = requirement()
    one_year = FilterRequirement(
        id="in_2020",
        decision_ids=("measure",),
        input_id=dataset.id,
        predicates=predicates,
    )

    plan = _plan(
        _population_decision(decision),
        evidence_requirements=[dataset, one_year, _rank(one_year.id, **window)],
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
    dataset = requirement(entities=[], across=("country",), period=Period(start=2022, end=2022))

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
        "across": ["country"],
        "period": Period(start=2022, end=2022),
    }
    return requirement(**(fields | overrides))


@pytest.mark.parametrize("across", ["country", "countries", "place", "economy"])
def test_a_plan_over_a_requirement_naming_no_place_leaves_the_level_to_compile(
    across: str, decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # The place word covers `observationAbout`, which mixes levels, or a flow
    # axis, which does not; only compile knows which (`compile._check_level_filters`).
    dataset = _every_place(requirement, across=(across,))

    ranked = _plan(
        _population_decision(decision),
        evidence_requirements=[dataset, _rank(dataset.id, limit=10)],
    )

    assert ranked.requirement("ranked").input_id == dataset.id


def test_a_total_per_place_over_a_requirement_naming_no_place_is_refused(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # A group by entity keeps only the places the requirement names, and it
    # names none.
    dataset = _every_place(requirement, period=Period(start=2015, end=2024))
    total = AggregateRequirement(
        id="total",
        decision_ids=["measure"],
        input_id=dataset.id,
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )

    by_country = total.model_copy(update={"input_id": "countries"})
    plan = _plan(decision(), evidence_requirements=[dataset, _countries(dataset.id), by_country])
    assert plan.requirement("total").input_id == "countries"

    per_place = total.model_copy(update={"group_by": (GroupBy.ENTITY,)})
    with pytest.raises(ValidationError, match=r"'total': .*this input names no entity"):
        _plan(decision(), evidence_requirements=[dataset, per_place])


def test_a_level_filter_reads_the_retrieval_not_each_places_growth(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # "Fastest-growing by GDP": a growth per place drops the type each place
    # carries, so the level filter has to read the retrieval itself.
    dataset = _every_place(requirement, period=Period(start=2015, end=2024))
    growth = GrowthRequirement(
        id="growth",
        decision_ids=["measure"],
        input_id=dataset.id,
        window_start=2015,
        window_end=2024,
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


@pytest.mark.parametrize(
    ("predicates", "admitted"),
    [
        pytest.param(
            (PeriodThreshold(op="gte", year=2024), PeriodThreshold(op="gte", year=2024)),
            False,
            id="a-year-threshold-twice",
        ),
        pytest.param(
            (ValueThreshold(op="gt", value=1.5), ValueThreshold(op="gt", value=1.5)),
            False,
            id="a-value-threshold-twice",
        ),
        pytest.param(
            (EntityIn(names=("France", "Kenya")), EntityIn(names=("Kenya", "France"))),
            False,
            id="one-entity-list-twice-in-another-order",
        ),
        pytest.param(
            (
                PeriodThreshold(op="gte", year=2024, rolling=RelativePeriod(offset=0)),
                PeriodThreshold(op="gte", year=2024, rolling=RelativePeriod(offset=0)),
            ),
            False,
            id="a-rolling-year-threshold-twice",
        ),
        pytest.param(
            (
                PeriodThreshold(op="gte", year=2024),
                PeriodThreshold(op="gte", year=2024, rolling=RelativePeriod(offset=0)),
            ),
            True,
            id="a-fixed-and-a-rolling-threshold-on-one-year",
        ),
    ],
)
def test_a_filter_states_each_predicate_once(
    predicates: tuple[Predicate, ...], admitted: bool
) -> None:
    """analysis-mcp's filter op spec refuses a predicate stated twice, by its
    canonical form. A rolling threshold carries its reference there, so it
    differs from a fixed one on the same year."""
    if admitted:
        _countries("gdp", predicates=predicates)
        return
    with pytest.raises(ValidationError, match="every predicate must be unique"):
        _countries("gdp", predicates=predicates)


def test_a_series_of_every_place_needs_no_level(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    plan = _plan(decision(), evidence_requirements=[_every_place(requirement)])

    assert plan.answer_mode is AnswerMode.SERIES


def test_a_rank_over_across_beside_a_named_entity_states_a_reduction(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # "Top donors to Kenya": Kenya's recipient role and every donor are two
    # role dimensions, so ranking donors collapses the recipient rows first.
    dataset = requirement(
        entities=[Entity(name="Kenya", role="recipient")],
        across=("donor",),
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

    with pytest.raises(ValidationError) as refused:
        _plan(
            decision(),
            evidence_requirements=[dataset_a, dataset_b, agg_a, agg_b, joined, _rank(joined.id)],
        )
    assert _unsupported_operation(refused.value) is UnsupportedOperation.CALCULATION_OVER_COMBINED


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

    with pytest.raises(ValidationError, match="cites no decision whose `fixes` names `among`"):
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


def test_a_quotient_of_states_a_denominator_axis_only_with_an_axis() -> None:
    with pytest.raises(ValidationError, match="stated only with axis"):
        QuotientRequirement(
            id="per_head",
            decision_ids=["measure"],
            numerator=RequirementSide(input_id="a"),
            denominator=RequirementSide(input_id="b"),
            denominator_axis="country",
        )


def _aid_per_head(
    requirement: RequirementFactory, **quotient: Any
) -> tuple[DatasetRequirement, DatasetRequirement, QuotientRequirement]:
    """Aid by recipient over population by country, the same two places on each side."""
    aid = requirement(
        id="aid",
        measure="aid received",
        entities=[Entity(name="Kenya", role="recipient"), Entity(name="Ghana", role="recipient")],
    )
    population = requirement(
        id="population",
        measure="population",
        entities=[Entity(name="Kenya", role="country"), Entity(name="Ghana", role="country")],
        unit=None,
    )
    per_head = QuotientRequirement(
        id="per_head",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=aid.id),
        denominator=RequirementSide(input_id=population.id),
        **quotient,
    )
    return aid, population, per_head


@pytest.mark.parametrize(
    ("quotient", "refusal"),
    [
        pytest.param({}, None, id="un-keyed"),
        pytest.param({"axis": "recipient"}, None, id="every-key-over-one-total"),
        pytest.param(
            {"axis": "recipient", "denominator_axis": "country"},
            None,
            id="each-key-over-its-counterpart",
        ),
        pytest.param(
            {"axis": "country"},
            "axis 'country' does not name a role the numerator carries",
            id="an-axis-the-numerator-lacks",
        ),
        pytest.param(
            {"axis": "recipient", "denominator_axis": "recipient"},
            "denominator_axis 'recipient' does not name a role the denominator carries",
            id="a-denominator-axis-the-denominator-lacks",
        ),
    ],
)
def test_a_quotient_of_keys_each_side_by_its_own_role_and_cites_no_decision_for_it(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    quotient: dict[str, str],
    refusal: str | None,
) -> None:
    requirements = list(_aid_per_head(requirement, **quotient))

    if refusal is None:
        # `decision()` fixes nothing: neither form is a claim a decision has to state.
        plan = _plan(decision(), evidence_requirements=requirements)
        assert plan.requirement("per_head") == requirements[-1]
    else:
        with pytest.raises(ValidationError, match=refusal):
            _plan(decision(), evidence_requirements=requirements)


@pytest.mark.parametrize(
    ("axis", "refusal"),
    [
        pytest.param("recipient", None, id="the-numerators-phrase"),
        pytest.param(None, None, id="the-lone-role"),
        pytest.param(
            "country", "does not name a role this input carries", id="the-denominators-phrase"
        ),
    ],
)
def test_a_rank_reads_a_keyed_quotient_of_by_the_numerators_phrase(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    axis: str | None,
    refusal: str | None,
) -> None:
    aid, population, per_head = _aid_per_head(
        requirement, axis="recipient", denominator_axis="country"
    )
    rank = _rank(per_head.id, axis=axis, reduction=Reduction.MEAN)
    requirements = [aid, population, per_head, rank]

    if refusal is None:
        plan = _plan(_population_decision(decision), evidence_requirements=requirements)
        assert plan.requirement("ranked").input_id == "per_head"
    else:
        with pytest.raises(ValidationError, match=refusal):
            _plan(_population_decision(decision), evidence_requirements=requirements)


def _places(*names: str) -> list[Entity]:
    """Places the measure is about: entities stating no role."""
    return [Entity(name=name) for name in names]


def _over_places(
    requirement: RequirementFactory, derived: str, *, numerator_role: str | None, **keys: Any
) -> list[EvidenceRequirement]:
    """A measure of three places divided, by `derived`, by a second measure of the same places.

    The numerator names its places under `numerator_role`, or with no role.
    The denominator names them with none.
    """
    names = ("Kenya", "Uganda", "Tanzania")
    numerator = requirement(
        id="numerator",
        measure="aid received",
        entities=[Entity(name=name, role=numerator_role) for name in names],
    )
    denominator = requirement(
        id="denominator",
        measure="population",
        entities=_places(*names),
        unit=None if derived == "quotient_of" else numerator.unit,
    )
    fields: dict[str, Any] = {
        "id": "divided",
        "decision_ids": ["measure"],
        "numerator": RequirementSide(input_id=numerator.id),
        "denominator": RequirementSide(input_id=denominator.id),
        **keys,
    }
    divided = (
        QuotientRequirement(**fields) if derived == "quotient_of" else ShareOfRequirement(**fields)
    )
    return [numerator, denominator, divided]


@pytest.mark.parametrize(
    ("derived", "numerator_role", "keys"),
    [
        pytest.param(
            "quotient_of",
            "recipient",
            {"axis": "recipient", "denominator_axis": "country"},
            id="quotient-of-a-role-over-places",
        ),
        pytest.param(
            "quotient_of",
            None,
            {"axis": "country", "denominator_axis": "country"},
            id="quotient-of-places-over-places",
        ),
        pytest.param(
            "quotient_of",
            None,
            {"axis": "countries", "denominator_axis": "place"},
            id="any-place-word",
        ),
        pytest.param(
            "share_of",
            None,
            {"axis": "country", "denominator_scope": DenominatorScope.PER_AXIS},
            id="share-of-places-over-places",
        ),
    ],
)
def test_a_place_word_keys_a_side_whose_places_state_no_role(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived: str,
    numerator_role: str | None,
    keys: dict[str, Any],
) -> None:
    # A place the measure is about states no role, so no role phrase names
    # its dimension; a place word does. The draft passes `propose_plan`'s own
    # check of an entity's role as well as the plan's.
    requirements = _over_places(requirement, derived, numerator_role=numerator_role, **keys)

    _patch_of(requirements)
    plan = _plan(decision(), evidence_requirements=requirements)

    assert plan.requirement("divided") == requirements[-1]


def _patch_of(requirements: list[EvidenceRequirement]) -> None:
    """Validate `requirements` as the lists one `propose_plan` call states."""
    _Patch.model_validate(
        {
            "decisions": [],
            "evidence_requirements": [r.model_dump(mode="json") for r in requirements],
            "removed_decisions": [],
            "removed_evidence_requirements": [],
            "declined": [],
            "answered": [],
            "across_years": [],
        }
    )


def test_a_rank_reads_role_less_places_by_a_place_word(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    dataset = requirement(entities=_places("Kenya", "Uganda"))
    rank = _rank(dataset.id, axis="country", reduction=Reduction.MEAN)

    plan = _plan(_population_decision(decision), evidence_requirements=[dataset, rank])

    assert plan.requirement("ranked").axis == "country"


def test_a_filter_narrows_role_less_places_by_a_place_word(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # A named rank needs its axis narrowed, and the filter names the place
    # dimension as the rank's axis does.
    dataset = requirement(entities=_places("Kenya", "Uganda"))
    named = FilterRequirement(
        id="named",
        decision_ids=["measure"],
        input_id=dataset.id,
        predicates=(EntityIn(names=["Kenya", "Uganda"], role="country"),),
    )
    rank = _rank(named.id, axis="place", among=RankAmong.NAMED, reduction=Reduction.MEAN)

    plan = _plan(decision(), evidence_requirements=[dataset, named, rank])

    assert plan.requirement("ranked").among == RankAmong.NAMED


@pytest.mark.parametrize("derived", ["quotient_of", "share_of", "rank"])
def test_a_place_word_naming_two_dimensions_of_a_side_is_refused(
    decision: DecisionFactory, requirement: RequirementFactory, derived: str
) -> None:
    # One place under a place role and one with no role are two dimensions,
    # and "country" names both.
    mixed = [Entity(name="Kenya", role="country"), Entity(name="Uganda")]
    requirements: list[EvidenceRequirement]
    if derived == "rank":
        dataset = requirement(entities=mixed)
        requirements = [dataset, _rank(dataset.id, axis="country", reduction=Reduction.MEAN)]
    else:
        keys: dict[str, Any] = (
            {"axis": "country"}
            if derived == "quotient_of"
            else {"axis": "country", "denominator_scope": DenominatorScope.PER_AXIS}
        )
        requirements = _over_places(requirement, derived, numerator_role=None, **keys)
        requirements[0] = requirements[0].model_copy(update={"entities": tuple(mixed)})

    with pytest.raises(ValidationError, match="names more than one dimension"):
        _plan(_population_decision(decision), evidence_requirements=requirements)


@pytest.mark.parametrize(
    ("reduction", "group_by", "refused"),
    [
        (Reduction.SUM, (GroupBy.PERIOD,), True),
        (Reduction.SUM, (GroupBy.ENTITY,), True),
        (Reduction.SUM, (GroupBy.ENTITY, GroupBy.PERIOD), True),
        (Reduction.MEAN, (GroupBy.PERIOD,), False),
        (Reduction.MEAN, (GroupBy.ENTITY,), False),
        (Reduction.MAX, (GroupBy.ENTITY,), False),
        (Reduction.MIN, (GroupBy.PERIOD,), False),
    ],
)
def test_an_aggregate_over_a_quotient_takes_any_reduction_but_a_sum(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    reduction: Reduction,
    group_by: tuple[GroupBy, ...],
    refused: bool,
) -> None:
    # Two countries' dollars per person add up to no per-person figure.
    per_head = _aid_per_head(requirement, axis="recipient", denominator_axis="country")
    reduced = AggregateRequirement(
        id="reduced",
        decision_ids=["measure"],
        input_id="per_head",
        reduction=reduction,
        group_by=group_by,
    )
    requirements = [*per_head, reduced]

    if refused:
        with pytest.raises(ValidationError, match="'reduced'"):
            _plan(decision(), evidence_requirements=requirements)
    else:
        plan = _plan(decision(), evidence_requirements=requirements)
        assert plan.requirement("reduced") == reduced


def test_a_rank_over_a_quotient_refuses_a_summing_reduction(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    per_head = _aid_per_head(requirement, axis="recipient", denominator_axis="country")
    rank = _rank("per_head", reduction=Reduction.SUM)

    with pytest.raises(ValidationError, match="'ranked'"):
        _plan(_population_decision(decision), evidence_requirements=[*per_head, rank])


@pytest.mark.parametrize("derived", ["comparison", "share_of"])
@pytest.mark.parametrize("rate_side", ["first", "second", "both"])
def test_a_quotient_pairs_only_with_another_quotient(
    decision: DecisionFactory, requirement: RequirementFactory, derived: str, rate_side: str
) -> None:
    # A composed unit equals no source's own unit, so a quotient beside a
    # measured side is refused before analysis-mcp refuses it.
    aid = requirement(id="aid", entities=[Entity(name="Kenya", role="recipient")])
    other = requirement(id="other", entities=[Entity(name="Uganda", role="recipient")])
    population = requirement(id="population", entities=_places("Kenya"), unit=None)

    def quotient(name: str, numerator: str) -> QuotientRequirement:
        return QuotientRequirement(
            id=name,
            decision_ids=["measure"],
            numerator=RequirementSide(input_id=numerator),
            denominator=RequirementSide(input_id="population"),
        )

    first = "aid_per_head" if rate_side in ("first", "both") else "aid"
    second = "other_per_head" if rate_side in ("second", "both") else "other"
    sides = (RequirementSide(input_id=first), RequirementSide(input_id=second))
    paired: EvidenceRequirement = (
        ComparisonRequirement(
            id="paired", decision_ids=["measure"], minuend=sides[1], subtrahend=sides[0]
        )
        if derived == "comparison"
        else ShareOfRequirement(
            id="paired", decision_ids=["measure"], numerator=sides[0], denominator=sides[1]
        )
    )
    requirements = [
        aid,
        other,
        population,
        quotient("aid_per_head", "aid"),
        quotient("other_per_head", "other"),
        paired,
    ]

    if rate_side == "both":
        plan = _plan(decision(), evidence_requirements=requirements)
        assert plan.requirement("paired") == paired
    else:
        with pytest.raises(ValidationError, match="'paired'"):
            _plan(decision(), evidence_requirements=requirements)


def _growth(name: str, input_id: str, change: GrowthChange) -> GrowthRequirement:
    return GrowthRequirement(
        id=name,
        decision_ids=["measure"],
        input_id=input_id,
        change=change,
        window_start=2015,
        window_end=2024,
    )


@pytest.mark.parametrize(
    ("change", "reader", "refused"),
    [
        # An absolute change in dollars per person is still in dollars per person.
        pytest.param(GrowthChange.ABSOLUTE, "sum", True, id="absolute-summed"),
        pytest.param(GrowthChange.ABSOLUTE, "mean", False, id="absolute-averaged"),
        pytest.param(GrowthChange.ABSOLUTE, "quotient_of", True, id="absolute-divided-again"),
        pytest.param(GrowthChange.ABSOLUTE, "comparison", True, id="absolute-beside-a-measure"),
        pytest.param(GrowthChange.ABSOLUTE, "share_of", True, id="absolute-over-a-measure"),
        pytest.param(
            GrowthChange.ABSOLUTE, "comparison_of_two", False, id="absolute-beside-its-like"
        ),
        # A relative change has no unit, so it reads as any ratio does.
        pytest.param(GrowthChange.RELATIVE, "sum", False, id="relative-summed"),
        pytest.param(GrowthChange.RELATIVE, "quotient_of", True, id="relative-divided-again"),
        pytest.param(GrowthChange.RELATIVE, "comparison", False, id="relative-beside-a-ratio"),
    ],
)
def test_a_change_in_a_quotient_keeps_its_composed_unit_only_when_absolute(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    change: GrowthChange,
    reader: str,
    refused: bool,
) -> None:
    aid, population, per_head = _aid_per_head(
        requirement, axis="recipient", denominator_axis="country"
    )
    # The change in aid per head, and the same change in each plain measure.
    grew = _growth("grew", "per_head", change)
    aid_grew = _growth("aid_grew", "aid", change)
    people_grew = _growth("people_grew", "population", change)
    sides = (RequirementSide(input_id="grew"), RequirementSide(input_id="aid_grew"))
    reading: EvidenceRequirement
    if reader in ("sum", "mean"):
        reading = AggregateRequirement(
            id="reading",
            decision_ids=["measure"],
            input_id="grew",
            reduction=Reduction(reader),
            group_by=(GroupBy.PERIOD,),
        )
    elif reader == "quotient_of":
        reading = QuotientRequirement(
            id="reading",
            decision_ids=["measure"],
            numerator=sides[0],
            denominator=RequirementSide(input_id="people_grew"),
        )
    elif reader == "share_of":
        reading = ShareOfRequirement(
            id="reading", decision_ids=["measure"], numerator=sides[0], denominator=sides[1]
        )
    else:
        right = "grew_again" if reader == "comparison_of_two" else "aid_grew"
        reading = ComparisonRequirement(
            id="reading",
            decision_ids=["measure"],
            minuend=RequirementSide(input_id=right),
            subtrahend=sides[0],
        )
    requirements = [
        aid,
        population,
        per_head,
        grew,
        _growth("grew_again", "per_head", change),
        aid_grew,
        people_grew,
        reading,
    ]

    if refused:
        with pytest.raises(ValidationError, match="'reading'"):
            _plan(decision(), evidence_requirements=requirements)
    else:
        plan = _plan(decision(), evidence_requirements=requirements, answer_mode=AnswerMode.SERIES)
        assert plan.requirement("reading") == reading


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

    with pytest.raises(
        ValidationError, match="cites no decision whose `fixes` names `denominator_scope`"
    ):
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
    # leaves each of them out.
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


_RANK_THE_SOURCE_ONCE = "rank the source once with the limit and per_period=false instead"


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
        with pytest.raises(ValidationError, match=_RANK_THE_SOURCE_ONCE):
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
        with pytest.raises(ValidationError, match=_RANK_THE_SOURCE_ONCE):
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
        (PeriodThreshold(op="eq", year=2018), PeriodThreshold(op="gte", year=2018)),
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


# --- A rank's limit ---------------------------------------------------------------


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


# --- A decision's `fixes` binds the requirements citing it ----------------------


@pytest.mark.parametrize(
    ("fixes", "stated", "agreeing"),
    [
        (
            DecisionFixes(period=Period(start=2018, end=2024)),
            {"period": Period(start=2015, end=2024)},
            {"period": Period(start=2018, end=2024)},
        ),
        (DecisionFixes(across=("donor",)), {"across": ["creditor"]}, {"across": ["donor"]}),
        (
            DecisionFixes(unit=Unit(currency="USD", prices="current")),
            {"unit": None},
            {"unit": Unit(currency="USD", prices="current")},
        ),
    ],
    ids=["period", "across", "unit"],
)
def test_a_requirement_contradicting_a_value_its_decision_fixes_is_refused(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    fixes: DecisionFixes,
    stated: dict[str, object],
    agreeing: dict[str, object],
) -> None:
    """The backstop under `propose_plan`, which drops or refuses a carried decision a patch contradicts."""
    cited = decision(fixes=fixes)
    plan = _plan(cited, evidence_requirements=[requirement(**agreeing), derived_requirement()])
    assert plan.decisions == (cited,)

    with pytest.raises(ValidationError, match="but the decision 'measure' it cites fixes"):
        _plan(cited, evidence_requirements=[requirement(**stated), derived_requirement()])


def test_a_requirement_keeps_its_own_pin_whatever_a_cited_decision_fixes(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    """Code writes and moves a decision's `source_spec_id` with the pin, so no requirement is held to it."""
    dataset = requirement(source_spec_id=_PIN)

    plan = _plan(
        decision(fixes=DecisionFixes(source_spec_id="spec_" + "b" * 32)),
        evidence_requirements=[dataset, derived_requirement()],
    )

    assert plan.requirement(dataset.id) == dataset


def test_a_pinned_source_needs_no_decision(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    pinned = requirement(decision_ids=[], source_spec_id=_PIN)

    plan = _plan(decision(), evidence_requirements=[pinned, derived_requirement()])

    assert plan.requirement(pinned.id).source_spec_id == _PIN
    assert plan.requirement(pinned.id).decision_ids == ()


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

    approved = plan.approve(at=datetime.now(UTC))

    assert plan.status == PlanStatus.PENDING_APPROVAL
    assert approved.status == PlanStatus.APPROVED
    assert approved.decisions == plan.decisions


def test_approve_records_the_revision_approved_and_when(decision: DecisionFactory) -> None:
    plan = _plan(decision())
    at = datetime(2026, 10, 1, 9, 30, tzinfo=UTC)

    approved = plan.approve(at=at)

    assert approved.approved_revision == plan_revision(plan)
    assert approved.approved_at == at
    assert Plan.model_validate_json(approved.model_dump_json()).approved_at == at


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
        plan.approve(at=datetime.now(UTC))


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
    executing = _plan(decision()).approve(at=datetime.now(UTC)).start_execution()

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
    approved = _plan(decision()).approve(at=datetime.now(UTC))
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
    approved = _plan(decision()).approve(at=datetime.now(UTC))

    with pytest.raises(ValueError, match=refusal):
        getattr(approved, transition)()


# --- Revisions -----------------------------------------------------------------


def test_plan_revision_ignores_status_and_approval(decision: DecisionFactory) -> None:
    # Approving must not change the value that was approved.
    plan = _plan(decision())

    assert plan_revision(plan.approve(at=datetime.now(UTC))) == plan_revision(plan)
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
        QuotientRequirement,
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
    # Ten requirements, the four predicates a filter carries, and the ten
    # operations a plan's `translation` holds.
    definitions = Plan.model_json_schema()["$defs"]
    tagged = {
        name: schema
        for name, schema in definitions.items()
        if "kind" in schema.get("properties", {})
    }

    assert len(tagged) == 24
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
        "quotient_of",
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
    ("Fixes.across", ("decisions", 0, "fixes", "across"), MAX_ACROSS_WORDS),
    ("Fixes.across[]", ("decisions", 0, "fixes", "across", 0), MAX_ENTITY_ROLE_CHARS),
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
    ("Dataset.across", ("evidence_requirements", 0, "across"), MAX_ACROSS_WORDS),
    (
        "Dataset.across[]",
        ("evidence_requirements", 0, "across", 0),
        MAX_ENTITY_ROLE_CHARS,
    ),
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
    ("Aggregate.group_by", ("evidence_requirements", 17, "group_by"), MAX_GROUP_BY),
    ("Rank.axis", ("evidence_requirements", 19, "axis"), MAX_ENTITY_ROLE_CHARS),
    ("ShareOf.axis", ("evidence_requirements", 20, "axis"), MAX_ENTITY_ROLE_CHARS),
    (
        "ShareOf.numerator.entity",
        ("evidence_requirements", 20, "numerator", "entity"),
        MAX_ENTITY_NAME_CHARS,
    ),
    (
        "ShareOf.denominator.entity",
        ("evidence_requirements", 20, "denominator", "entity"),
        MAX_ENTITY_NAME_CHARS,
    ),
    ("Quotient.axis", ("evidence_requirements", 21, "axis"), MAX_ENTITY_ROLE_CHARS),
    (
        "Quotient.denominator_axis",
        ("evidence_requirements", 21, "denominator_axis"),
        MAX_ENTITY_ROLE_CHARS,
    ),
    (
        "Quotient.numerator.entity",
        ("evidence_requirements", 21, "numerator", "entity"),
        MAX_ENTITY_NAME_CHARS,
    ),
    (
        "Quotient.denominator.entity",
        ("evidence_requirements", 21, "denominator", "entity"),
        MAX_ENTITY_NAME_CHARS,
    ),
    (
        "Comparison.minuend.entity",
        ("evidence_requirements", 22, "minuend", "entity"),
        MAX_ENTITY_NAME_CHARS,
    ),
    (
        "Comparison.subtrahend.entity",
        ("evidence_requirements", 22, "subtrahend", "entity"),
        MAX_ENTITY_NAME_CHARS,
    ),
    ("Join.decision_ids", ("evidence_requirements", 23, "decision_ids"), MAX_PLAN_DECISIONS),
]


def test_a_plans_source_fits_the_wording_search_takes() -> None:
    (limit,) = (
        meta.max_length
        for meta in SearchSource.model_fields["text"].metadata
        if getattr(meta, "max_length", None) is not None
    )

    assert limit == MAX_SOURCE_CHARS


# Discriminator tags pydantic adds to an error's location inside a union.
_UNION_TAGS = frozenset(
    {
        "dataset",
        "aggregate",
        "growth",
        "comparison",
        "rank",
        "share_of",
        "quotient_of",
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
    assert (ids[0], ids[8], ids[17], ids[19], ids[20], ids[21], ids[22], ids[23]) == (
        "dataset_0",
        "filter_0",
        "totals",
        "rank",
        "share",
        "quotient",
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
# hash and the time of approval. `propose_plan` accepts none of them, so their
# text is not model output and the limits above do not apply to it.
_SYSTEM_AUTHORED = frozenset({"translation", "workflow", "approved_revision", "approved_at"})


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


# --- periods that follow the newest data ---------------------------------------


def _rolling_error(exc: ValidationError) -> RollingPlanError:
    """The `RollingPlanError` behind `exc`, as `planner.agent` reads it."""
    (cause,) = [
        error["ctx"]["error"]
        for error in exc.errors()
        if isinstance(error.get("ctx", {}).get("error"), RollingPlanError)
    ]
    return cause


_LAST_FIVE = RollingWindow(
    window_start=RelativePeriod(offset=-5), window_end=RelativePeriod(offset=0)
)


def _rolling_growth(input_id: str, **overrides: Any) -> GrowthRequirement:
    fields: dict[str, Any] = {
        "id": "growth",
        "decision_ids": ["measure"],
        "input_id": input_id,
        "window_start": 2019,
        "window_end": 2024,
        "rolling": _LAST_FIVE,
    }
    return GrowthRequirement(**(fields | overrides))


@pytest.mark.parametrize("offset", [1, -201, -1.0, "-1", True])
def test_a_relative_period_counts_whole_years_back_from_the_newest(offset: object) -> None:
    assert RelativePeriod(offset=-5).model_dump() == {"anchor": "latest", "offset": -5}

    with pytest.raises(ValidationError):
        RelativePeriod.model_validate({"anchor": "latest", "offset": offset})


def test_a_rolling_window_names_at_least_one_bound() -> None:
    with pytest.raises(ValidationError, match="names window_start, window_end or both"):
        RollingWindow()


def test_a_window_rolling_at_one_end_keeps_any_fixed_year_at_the_other() -> None:
    since = RollingWindow(window_end=RelativePeriod(offset=0))

    growth = _rolling_growth("d", window_start=2015, rolling=since)

    assert (growth.window_start, growth.window_end) == (2015, 2024)


def _windowed_rank(**overrides: Any) -> RankRequirement:
    fields: dict[str, Any] = {
        "id": "ranked",
        "input_id": "d",
        "direction": RankDirection.HIGHEST_FIRST,
        "among": RankAmong.POPULATION,
        "reduction": Reduction.SUM,
        "window_start": 2019,
        "window_end": 2024,
        "rolling": _LAST_FIVE,
    }
    return RankRequirement(**(fields | overrides))


def test_a_rank_rolling_a_window_it_does_not_state_is_refused() -> None:
    assert _windowed_rank().rolling == _LAST_FIVE

    with pytest.raises(ValidationError) as refused:
        _windowed_rank(window_start=None, window_end=None)

    assert _rolling_error(refused.value).issue is RollingIssue.YEARS_DISAGREE


_DATASET = "annual_debt_service"
_OFFSET = {offset: RelativePeriod(offset=offset) for offset in (0, -1, -4, -5, -6, -8)}


def _years(step_id: str, input_id: str, *thresholds: PeriodThreshold) -> FilterRequirement:
    return FilterRequirement(
        id=step_id, decision_ids=["measure"], input_id=input_id, predicates=thresholds
    )


def _gte(year: int, offset: int | None = None) -> PeriodThreshold:
    return PeriodThreshold(op="gte", year=year, rolling=None if offset is None else _OFFSET[offset])


def _lte(year: int, offset: int | None = None) -> PeriodThreshold:
    return PeriodThreshold(op="lte", year=year, rolling=None if offset is None else _OFFSET[offset])


def _eq(year: int, offset: int | None = None) -> PeriodThreshold:
    return PeriodThreshold(op="eq", year=year, rolling=None if offset is None else _OFFSET[offset])


# 2015-2024 retrieved with no end year, read as it is or below a fixed last year.
_CAP = _years("cap", _DATASET, _lte(2024))
_ROLLING_START = RollingWindow(window_start=_OFFSET[0])
_ON_THE_NEWEST = RollingWindow(window_start=_OFFSET[0], window_end=_OFFSET[0])


@pytest.mark.parametrize(
    ("steps", "refused"),
    [
        pytest.param(
            [
                _windowed_rank(
                    decision_ids=["measure"],
                    input_id=_DATASET,
                    window_start=2024,
                    rolling=_ROLLING_START,
                )
            ],
            "ranked",
            id="a-window-of-one-year-whose-start-alone-follows",
        ),
        pytest.param(
            [
                _windowed_rank(
                    decision_ids=["measure"],
                    input_id=_DATASET,
                    rolling=RollingWindow(window_start=_OFFSET[-5]),
                )
            ],
            "ranked",
            id="a-window-of-six-years-whose-start-alone-follows",
        ),
        pytest.param(
            [_rolling_growth(_DATASET, rolling=RollingWindow(window_start=_OFFSET[-5]))],
            "growth",
            id="a-growth-whose-start-alone-follows",
        ),
        pytest.param(
            [_years("recent", _DATASET, _gte(2019, -5), _lte(2024))],
            "recent",
            id="a-filter-from-a-following-year-to-a-fixed-one",
        ),
        pytest.param(
            [_years("recent", _DATASET, _gte(2019, -5), _eq(2022))],
            "recent",
            id="a-filter-from-a-following-year-keeping-a-fixed-one",
        ),
        pytest.param(
            [_CAP, _years("recent", "cap", _gte(2019, -5))],
            "recent",
            id="a-following-first-year-over-a-fixed-cap",
        ),
        pytest.param(
            [_CAP, _years("recent", "cap", _eq(2024, 0))],
            "recent",
            id="the-newest-year-over-a-fixed-cap",
        ),
        pytest.param(
            [_years("cap", _DATASET, _lte(2026)), _years("recent", "cap", _eq(2024, 0))],
            "recent",
            id="the-newest-year-over-a-cap-the-data-has-yet-to-reach",
        ),
        pytest.param(
            [
                _CAP,
                _windowed_rank(
                    decision_ids=["measure"],
                    input_id="cap",
                    window_start=2024,
                    rolling=_ON_THE_NEWEST,
                ),
            ],
            "ranked",
            id="a-window-on-the-newest-year-over-a-fixed-cap",
        ),
        pytest.param(
            [_CAP, _rolling_growth("cap")],
            "growth",
            id="a-growth-over-the-last-five-years-over-a-fixed-cap",
        ),
        pytest.param(
            [_CAP, _rolling_growth("cap", rolling=RollingWindow(window_end=_OFFSET[0]))],
            "growth",
            id="a-growth-whose-end-alone-follows-over-a-fixed-cap",
        ),
        pytest.param(
            [
                _years("recent", _DATASET, _gte(2019, -5)),
                _rolling_growth(
                    "recent", window_start=2020, rolling=RollingWindow(window_end=_OFFSET[0])
                ),
            ],
            "growth",
            id="a-growth-from-a-fixed-year-over-a-following-first-year",
        ),
        pytest.param(
            [
                _years("cap", _DATASET, _gte(2022), _lte(2024)),
                _years("recent", "cap", _gte(2019, -5)),
            ],
            "recent",
            id="a-following-first-year-behind-a-fixed-one-over-a-fixed-cap",
        ),
        pytest.param(
            [
                _years("recent", _DATASET, _gte(2019, -5)),
                _CAP,
                ComparisonRequirement(
                    id="excess",
                    decision_ids=["measure"],
                    minuend=RequirementSide(input_id="recent"),
                    subtrahend=RequirementSide(input_id="cap"),
                ),
            ],
            "excess",
            id="a-following-first-year-against-a-fixed-cap-on-the-other-side",
        ),
    ],
)
def test_a_first_year_following_the_newest_data_below_a_fixed_last_year_is_refused(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    steps: list[EvidenceRequirement],
    refused: str,
) -> None:
    """Each new year moves the first year while the last stays fixed, so the
    years shrink until none is left, whichever step states either bound."""
    population = decision(fixes=DecisionFixes(among=RankAmong.POPULATION))

    with pytest.raises(ValidationError) as raised:
        _plan(population, evidence_requirements=[requirement(period_end_open=True), *steps])

    error = _rolling_error(raised.value)
    assert error.issue is RollingIssue.YEARS_RUN_OUT
    assert error.requirement_ids == (refused,)


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param([_years("recent", _DATASET, _gte(2019, -5))], id="a-following-first-year"),
        pytest.param(
            [_CAP, _years("recent", "cap", _gte(2019))], id="a-fixed-first-year-over-a-cap"
        ),
        pytest.param(
            [_CAP, _years("recent", "cap", _lte(2023, -1))], id="a-following-last-year-over-a-cap"
        ),
        pytest.param(
            [
                _windowed_rank(
                    decision_ids=["measure"],
                    input_id=_DATASET,
                    window_start=2024,
                    rolling=_ON_THE_NEWEST,
                )
            ],
            id="a-window-on-the-newest-year",
        ),
        pytest.param([_rolling_growth(_DATASET)], id="a-growth-over-the-last-five-years"),
        pytest.param(
            [
                _years("recent", _DATASET, _gte(2019, -5)),
                _rolling_growth("recent"),
            ],
            id="a-growth-over-the-last-five-years-over-a-following-first-year",
        ),
    ],
)
def test_a_first_year_following_the_newest_data_with_a_last_year_that_follows_is_admitted(
    decision: DecisionFactory, requirement: RequirementFactory, steps: list[EvidenceRequirement]
) -> None:
    population = decision(fixes=DecisionFixes(among=RankAmong.POPULATION))

    assert _plan(population, evidence_requirements=[requirement(period_end_open=True), *steps])


_PEAK = AggregateRequirement(
    id="peak",
    decision_ids=["measure"],
    input_id=_DATASET,
    reduction=Reduction.MAX,
    group_by=(GroupBy.ENTITY,),
)
_SINCE_2015 = _years("since", _DATASET, _gte(2015))
_WITH_THE_PEAK = SumOfRequirement(
    id="pair",
    decision_ids=["measure"],
    left=RequirementSide(input_id="since"),
    right=RequirementSide(input_id="peak"),
)


_WITH_THE_RECENT_PEAK = _WITH_THE_PEAK.model_copy(
    update={"left": RequirementSide(input_id="recent")}
)
# The dataset's whole history, read from its first year on record for the
# rolling-start readers alone, beside the dataset its other readers read.
_HISTORY = f"{_DATASET}_history"
_WHOLE = {_HISTORY: 1990}


def _with_history(
    dataset: DatasetRequirement, steps: list[EvidenceRequirement]
) -> list[EvidenceRequirement]:
    """`dataset`, its whole-history read `_HISTORY`, and `steps`."""
    return [dataset, dataset.model_copy(update={"id": _HISTORY}), *steps]


def _shapes_read(
    requirements: list[EvidenceRequirement], *, whole_history: dict[str, int]
) -> dict[str, FrameShape]:
    """The shapes of `requirements` with the whole-history reads named by hand."""
    return _build_shapes(
        {requirement.id: requirement for requirement in requirements},
        whole_history=whole_history,
        origins={},
    )


@pytest.mark.parametrize(
    ("steps", "split", "refused"),
    [
        pytest.param(
            [_years("recent", _HISTORY, _gte(2019, -5)), _PEAK, _SINCE_2015, _WITH_THE_PEAK],
            True,
            None,
            id="a-sibling-following-first-year",
        ),
        pytest.param(
            [_PEAK, _SINCE_2015, _WITH_THE_PEAK, _years("recent", "pair", _gte(2019, -5))],
            False,
            "^'recent': ",
            id="a-following-first-year-above-the-pair",
        ),
        pytest.param(
            [
                _years("recent", _HISTORY, _gte(2019, -5)),
                _PEAK.model_copy(update={"input_id": "since"}),
                _SINCE_2015,
                _WITH_THE_PEAK,
            ],
            True,
            None,
            id="a-max-bounded-from-the-series-first-year",
        ),
        pytest.param(
            [_years("recent", _HISTORY, _gte(2019, -5)), _PEAK, _WITH_THE_RECENT_PEAK],
            True,
            "any year of 2015-2024, while the other side holds 2019-2024",
            id="a-series-from-a-following-first-year",
        ),
    ],
)
def test_a_max_by_entity_beside_a_whole_history_reads_the_stated_years(
    requirement: RequirementFactory,
    steps: list[EvidenceRequirement],
    split: bool,
    refused: str | None,
) -> None:
    """A max by entity reads the dataset's stated 2015-2024 whatever a rolling
    start beside it reads from the first year on record, and a following start
    above the pair reads the stated years too. So the max pairs with any
    series covering 2015-2024, and a series from a following first year
    covers too few. A filter whose first year follows the newest data above
    the pair is refused on its own account, since newer data moves its first
    year past a year the max fell in (`planner.shapes._picked_year_leaves`)."""
    dataset = requirement(period_end_open=True)
    requirements = _with_history(dataset, steps) if split else [dataset, *steps]
    whole = _WHOLE if split else {}

    if refused is not None:
        with pytest.raises(ValueError, match=refused):
            _shapes_read(requirements, whole_history=whole)
    else:
        shapes = _shapes_read(requirements, whole_history=whole)
        assert shapes["peak"].period == Period(start=2015, end=2024)


def test_a_max_by_entity_over_a_whole_history_needs_a_series_from_as_early_a_first_year(
    requirement: RequirementFactory,
) -> None:
    """A max by donor read whole from 1990 against a series read whole from
    2000, both up to 2024, can fall in a year before 2000, in either side
    order; against a series read whole from 1990 it pairs."""
    debt = requirement(id="debt", period_end_open=True)
    revenue = requirement(id="revenue", measure="government revenue", period_end_open=True)
    peak = _PEAK.model_copy(update={"input_id": "debt_capped"})
    capped = [
        _years("debt_capped", "debt", _lte(2024)),
        _years("revenue_capped", "revenue", _lte(2024)),
    ]
    rolling = [
        _years("recent_debt", "debt", _gte(2019, -5)),
        _years("recent", "revenue", _gte(2019, -5)),
    ]
    for left, right in (("peak", "revenue_capped"), ("revenue_capped", "peak")):
        pair = _WITH_THE_PEAK.model_copy(
            update={
                "left": RequirementSide(input_id=left),
                "right": RequirementSide(input_id=right),
            }
        )
        requirements = [debt, revenue, *capped, peak, *rolling, pair]

        with pytest.raises(
            ValueError, match="any year of 1990-2024, while the other side holds 2000"
        ):
            _shapes_read(requirements, whole_history={"debt": 1990, "revenue": 2000})
        shapes = _shapes_read(requirements, whole_history={"debt": 1990, "revenue": 1990})
        assert shapes["pair"].period == Period(start=1990, end=2024)


def _growth_over_both(numerator: str, denominator: str) -> list[EvidenceRequirement]:
    """`numerator` over `denominator`, then its growth over the last five years."""
    window = RollingWindow(window_start=_OFFSET[-4], window_end=_OFFSET[0])
    return [
        QuotientRequirement(
            id="pair",
            numerator=RequirementSide(input_id=numerator),
            denominator=RequirementSide(input_id=denominator),
        ),
        _rolling_growth("pair", window_start=2020, rolling=window),
    ]


def test_a_growth_over_two_whole_histories_reads_their_first_years_in_either_side_order(
    requirement: RequirementFactory,
) -> None:
    """Two datasets read whole from 1990, one stating 2020-2024 and one
    restated to 2024 alone. Each starts at its first year, so their quotient
    holds 1990-2024 and a growth over its last five years fits, whichever side
    comes first."""
    aid = requirement(id="aid", period=Period(start=2020, end=2024), period_end_open=True)
    imputed = requirement(
        id="imputed",
        measure="imputed multilateral aid",
        period=Period(start=2024, end=2024),
        period_end_open=True,
    )
    for numerator, denominator in (("aid", "imputed"), ("imputed", "aid")):
        shapes = _shapes_read(
            [aid, imputed, *_growth_over_both(numerator, denominator)],
            whole_history={"aid": 1990, "imputed": 1990},
        )

        assert shapes["pair"].period == Period(start=1990, end=2024)


_FROM_EIGHT_BACK = RollingWindow(window_start=_OFFSET[-8], window_end=_OFFSET[0])


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param(
            [_rolling_growth(_DATASET, window_start=2016, rolling=_FROM_EIGHT_BACK)],
            id="a-growth",
        ),
        pytest.param(
            [
                _windowed_rank(
                    decision_ids=["measure"],
                    input_id=_DATASET,
                    window_start=2016,
                    rolling=_FROM_EIGHT_BACK,
                )
            ],
            id="a-rank",
        ),
        pytest.param(
            [
                _years("recent", _DATASET, _gte(2016, -8)),
                _rolling_growth("recent", window_start=2016, rolling=_FROM_EIGHT_BACK),
            ],
            id="a-growth-over-a-following-first-year",
        ),
    ],
)
def test_a_rolling_start_before_the_stated_period_reads_a_whole_history(
    requirement: RequirementFactory, steps: list[EvidenceRequirement]
) -> None:
    """2020-2024 read whole from 1990 holds the years eight back of the newest."""
    dataset = requirement(period=Period(start=2020, end=2024), period_end_open=True)

    shapes = _shapes_read([dataset, *steps], whole_history={_DATASET: 1990})

    assert shapes[_DATASET].period == Period(start=1990, end=2024)


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param([_years("since", _DATASET, _eq(2018, -6))], id="one-year"),
        pytest.param(
            [
                _years("since", _DATASET, _eq(2018, -6)),
                AggregateRequirement(
                    id="total",
                    decision_ids=["measure"],
                    input_id="since",
                    reduction=Reduction.SUM,
                    group_by=(GroupBy.ENTITY,),
                ),
            ],
            id="one-year-totalled",
        ),
        pytest.param(
            [_years("since", _DATASET, _gte(2016, -8), _lte(2018, -6))],
            id="years-up-to-one",
        ),
    ],
)
def test_a_rolling_year_before_the_first_year_on_record_is_refused(
    requirement: RequirementFactory, steps: list[EvidenceRequirement]
) -> None:
    """The dataset holds nothing before 2020, its first year on record, so a
    year six back of the newest leaves no year of it."""
    dataset = requirement(period=Period(start=2020, end=2024), period_end_open=True)

    with pytest.raises(ValueError, match="leaves no year of 2020-2024"):
        _shapes_read([dataset, *steps], whole_history={_DATASET: 2020})


_SINCE = PeriodThreshold(op="gte", year=2019, rolling=RelativePeriod(offset=-5))
_UNTIL = PeriodThreshold(op="lte", year=2024, rolling=RelativePeriod(offset=0))


@pytest.mark.parametrize(
    ("steps", "refused"),
    [
        pytest.param(
            [_rolling_growth(_DATASET, window_start=2020)],
            ("growth",),
            id="a-growth-whose-offsets-span-other-years",
        ),
        pytest.param(
            [_windowed_rank(decision_ids=["measure"], input_id=_DATASET, window_start=2021)],
            ("ranked",),
            id="a-rank-window-whose-offsets-span-other-years",
        ),
        pytest.param(
            [_years("recent", _DATASET, _SINCE, _UNTIL.model_copy(update={"year": 2023}))],
            ("recent",),
            id="two-thresholds-of-one-filter",
        ),
        pytest.param(
            [_years("since", _DATASET, _gte(2024, -1)), _years("recent", "since", _lte(2024, 0))],
            ("since", "recent"),
            id="a-filter-over-a-filter",
        ),
        pytest.param(
            [
                _years("since", _DATASET, _gte(2019, -5)),
                _windowed_rank(
                    decision_ids=["measure"],
                    input_id="since",
                    window_start=2023,
                    window_end=2023,
                    rolling=_ON_THE_NEWEST,
                ),
            ],
            ("since", "ranked"),
            id="a-rank-window-over-a-filter",
        ),
        pytest.param(
            [
                _years("since", _DATASET, _gte(2019, -5)),
                _years("until", _DATASET, _lte(2023, 0)),
                ComparisonRequirement(
                    id="excess",
                    decision_ids=["measure"],
                    minuend=RequirementSide(input_id="since"),
                    subtrahend=RequirementSide(input_id="until"),
                ),
            ],
            ("since", "until"),
            id="the-two-sides-of-a-comparison",
        ),
    ],
)
def test_rolling_years_one_run_reads_from_two_newest_years_are_refused(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    steps: list[EvidenceRequirement],
    refused: tuple[str, ...],
) -> None:
    """The run writes every rolling year a result reads as one newest year
    moved by its offset, so years stated under two newest years are not the
    years it reads, in one step or across the steps one result reads."""
    population = decision(fixes=DecisionFixes(among=RankAmong.POPULATION))

    with pytest.raises(ValidationError) as raised:
        _plan(population, evidence_requirements=[requirement(period_end_open=True), *steps])

    error = _rolling_error(raised.value)
    assert error.issue is RollingIssue.YEARS_DISAGREE
    assert error.requirement_ids == refused


@pytest.mark.parametrize(
    ("steps", "ends", "refused"),
    [
        pytest.param(
            [_years("recent", _DATASET, _gte(2021, -5))],
            {_DATASET: 2024},
            ("recent",),
            id="a-filter-whose-newest-year-follows-the-data",
        ),
        pytest.param(
            [_years("since", _DATASET, _gte(2020, -5)), _years("recent", "since", _lte(2025, 0))],
            {_DATASET: 2024},
            ("since", "recent"),
            id="a-filter-over-a-filter-agreeing-on-a-later-year",
        ),
        pytest.param(
            [_rolling_growth(_DATASET, window_start=2020, window_end=2025)],
            {_DATASET: 2024},
            ("growth",),
            id="a-growth-whose-newest-year-follows-the-data",
        ),
        pytest.param(
            [
                _years("since", _DATASET, _gte(2019, -5)),
                _years("until", "other", _lte(2024, 0)),
                ComparisonRequirement(
                    id="excess",
                    decision_ids=["measure"],
                    minuend=RequirementSide(input_id="since"),
                    subtrahend=RequirementSide(input_id="until"),
                ),
            ],
            {_DATASET: 2024, "other": 2023},
            ("since", "until"),
            id="one-side-of-a-comparison-ends-before-the-newest-year",
        ),
    ],
)
def test_rolling_years_whose_newest_year_follows_the_data_are_refused(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    steps: list[EvidenceRequirement],
    ends: dict[str, int],
    refused: tuple[str, ...],
) -> None:
    """The newest year a group's rolling years state is one every dataset
    they rest on holds today, so a later one restates the period Search
    found."""
    datasets = [
        requirement(id=dataset_id, period=Period(start=2015, end=end), period_end_open=True)
        for dataset_id, end in ends.items()
    ]

    with pytest.raises(ValidationError) as raised:
        _plan(decision(), evidence_requirements=[*datasets, *steps])

    error = _rolling_error(raised.value)
    assert error.issue is RollingIssue.YEARS_DISAGREE
    assert error.requirement_ids == refused


def test_rolling_years_no_result_reads_together_may_state_two_newest_years(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    """The run resolves a newest year for each group of rolling steps one
    result reads (`compiler.rolling.anchor_groups`), so two results over one
    dataset count from newest years of their own."""
    steps = [_years("since", _DATASET, _gte(2019, -5)), _years("until", _DATASET, _lte(2023, 0))]

    assert _plan(decision(), evidence_requirements=[requirement(period_end_open=True), *steps])


@pytest.mark.parametrize(
    ("text", "named"),
    [
        pytest.param("step 'aid_history' failed.", "step 'aid' failed.", id="quoted"),
        pytest.param("aid_history: no data", "aid: no data", id="before-a-colon"),
        pytest.param("read aid_history.", "read aid.", id="ending-a-sentence"),
        pytest.param("aid_history_2 ran", "aid_history_2 ran", id="a-longer-id"),
        pytest.param("dc/topic/aid_history", "dc/topic/aid_history", id="a-dcid-segment"),
        pytest.param("/aid_history.csv", "/aid_history.csv", id="a-path"),
        pytest.param("aid_history.csv", "aid_history.csv", id="a-file-name"),
        pytest.param("dc:aid_history", "dc:aid_history", id="a-prefixed-name"),
        pytest.param("aid_history/rows", "aid_history/rows", id="a-path-head"),
    ],
)
def test_authored_text_renames_a_copy_named_in_prose_and_keeps_refs(text: str, named: str) -> None:
    assert authored_text(text, {"aid_history": "aid"}) == named


def test_rolling_years_refused_on_a_copy_name_the_requirement_the_model_wrote(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    """A total per year and a growth rolling from its start both read 'f', so
    the growth reads a copy of 'f' over the dataset's whole history
    ('f_history'). The refusal of the years the copy and the growth state
    names 'f', in its requirements and in its text."""
    steps = [
        _years("f", _DATASET, _lte(2023, 0)),
        AggregateRequirement(
            id="total",
            decision_ids=["measure"],
            input_id="f",
            reduction=Reduction.SUM,
            group_by=(GroupBy.PERIOD,),
        ),
        _rolling_growth("f"),
    ]

    with pytest.raises(ValidationError) as raised:
        _plan(decision(), evidence_requirements=[requirement(period_end_open=True), *steps])

    error = _rolling_error(raised.value)
    assert (error.issue, error.requirement_ids) == (RollingIssue.YEARS_DISAGREE, ("f", "growth"))
    assert "('f' [2023]; 'growth' [2024])" in str(error)


def _sum(step_id: str, left: str, right: str) -> SumOfRequirement:
    return SumOfRequirement(
        id=step_id,
        decision_ids=["measure"],
        left=RequirementSide(input_id=left),
        right=RequirementSide(input_id=right),
    )


@pytest.mark.parametrize(
    ("steps", "refused"),
    [
        pytest.param(
            [_years("newest", "other", _eq(2024, 0)), _sum("both", _DATASET, "newest")],
            "both",
            id="a-sum-of-an-open-side-and-a-newest-year-of-the-other",
        ),
        pytest.param(
            [
                _years("newest", "other", _eq(2024, 0)),
                QuotientRequirement(
                    id="both",
                    decision_ids=["measure"],
                    numerator=RequirementSide(input_id=_DATASET),
                    denominator=RequirementSide(input_id="newest"),
                ),
            ],
            "both",
            id="a-quotient-of-an-open-side-over-a-newest-year-of-the-other",
        ),
        pytest.param(
            [_years("before", "other", _lte(2023, -1)), _sum("both", _DATASET, "before")],
            "both",
            id="a-sum-of-an-open-side-and-years-before-the-newest-of-the-other",
        ),
        pytest.param(
            [_sum("both", _DATASET, "other"), _years("newest", "both", _eq(2024, 0))],
            None,
            id="the-newest-year-of-the-sum",
        ),
        pytest.param(
            [
                _years("since", _DATASET, _gte(2019, -5)),
                _years("newest", "other", _eq(2024, 0)),
                _sum("both", "since", "newest"),
            ],
            None,
            id="an-open-side-whose-own-first-year-follows-the-newest",
        ),
        pytest.param(
            [
                _years("newest_one", _DATASET, _eq(2024, 0)),
                _years("newest", "other", _eq(2024, 0)),
                _sum("both", "newest_one", "newest"),
            ],
            None,
            id="a-newest-year-on-each-side",
        ),
    ],
)
def test_a_two_sided_step_holds_an_open_side_to_a_newest_year_read_over_it(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    steps: list[EvidenceRequirement],
    refused: str | None,
) -> None:
    """The run reads the newest year over the datasets the rolling steps
    counting from it reach (`compiler.rolling.anchor_groups`). An open side
    no such step reaches holds its own newest year, which can lag the other
    side's, so the pair keeps no year once the other side reports one it
    lacks. A rolling bound reading both sides counts both."""
    datasets = [
        requirement(id=dataset_id, period=Period(start=2015, end=2024), period_end_open=True)
        for dataset_id in (_DATASET, "other")
    ]

    if refused is None:
        assert _plan(decision(), evidence_requirements=[*datasets, *steps])
        return
    with pytest.raises(ValidationError) as raised:
        _plan(decision(), evidence_requirements=[*datasets, *steps])

    error = _rolling_error(raised.value)
    assert error.issue is RollingIssue.SIDE_UNANCHORED
    assert error.requirement_ids == (refused,)


_NOT_2022 = PeriodThreshold(op="ne", year=2022)


@pytest.mark.parametrize(
    ("steps", "refused"),
    [
        pytest.param(
            [
                _years(
                    "recent",
                    _DATASET,
                    _SINCE,
                    PeriodThreshold(op="ne", year=2024, rolling=_OFFSET[0]),
                )
            ],
            ("recent",),
            id="an-ne-that-follows-the-newest-data",
        ),
        pytest.param(
            [_years("recent", _DATASET, _NOT_2022)], ("recent",), id="an-ne-over-an-open-end"
        ),
        pytest.param(
            [_CAP, _years("recent", "cap", _NOT_2022)],
            ("recent",),
            id="an-ne-over-a-fixed-cap-of-an-open-end",
        ),
        pytest.param(
            [
                _years("recent", _DATASET, PeriodThreshold(op="ne", year=2025)),
                _years("newest", "recent", _eq(2024, 0)),
            ],
            ("recent",),
            id="the-newest-year-over-an-ne-on-a-later-year",
        ),
    ],
)
def test_an_ne_over_years_newer_data_moves_is_refused(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    steps: list[EvidenceRequirement],
    refused: tuple[str, ...],
) -> None:
    """A year an `ne` removes from years newer data moves can become the
    newest year, or the one a later bound following the newest data names,
    which then holds no row, so an `ne` removes a fixed year from fixed years."""
    with pytest.raises(ValidationError) as raised:
        _plan(decision(), evidence_requirements=[requirement(period_end_open=True), *steps])

    error = _rolling_error(raised.value)
    assert error.issue is RollingIssue.EXCLUSION_ROLLS
    assert error.requirement_ids == refused


def test_a_first_year_after_the_newest_year_is_refused() -> None:
    """`gt` at offset 0 keeps only the years after the newest year at least
    half the series hold, and none when every series ends on it."""
    assert PeriodThreshold(op="gt", year=2023, rolling=_OFFSET[-1])

    with pytest.raises(ValidationError):
        PeriodThreshold(op="gt", year=2024, rolling=_OFFSET[0])


def test_an_ne_that_follows_the_newest_data_is_refused_over_fixed_years_too(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    """The refusal names the `ne`, before the rule that every rolling year
    reads a dataset retrieved with `period_end_open` asks for one."""
    fixed = _years("recent", _DATASET, _NOT_2022)
    assert _plan(decision(), evidence_requirements=[requirement(), fixed])
    rolling_ne = _years("recent", _DATASET, PeriodThreshold(op="ne", year=2024, rolling=_OFFSET[0]))

    with pytest.raises(ValidationError) as raised:
        _plan(decision(), evidence_requirements=[requirement(), rolling_ne])

    assert _rolling_error(raised.value).issue is RollingIssue.EXCLUSION_ROLLS


def test_a_rolling_period_over_a_dataset_with_a_fixed_end_is_refused(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    """The check reaches the dataset through the requirements between, and
    names the dataset to open, which is what the planner changes."""
    totals = derived_requirement()
    growth = _rolling_growth(totals.id)
    opened = requirement(period_end_open=True)
    assert _plan(decision(), evidence_requirements=[opened, totals, growth])

    with pytest.raises(ValidationError) as refused:
        _plan(decision(), evidence_requirements=[requirement(), totals, growth])

    error = _rolling_error(refused.value)
    assert error.issue is RollingIssue.NEEDS_OPEN_PERIOD
    assert error.fault is OpenPeriodFault.DATASET_CLOSED
    assert error.requirement_ids == (opened.id,)


def test_a_rolling_threshold_needs_an_open_dataset_too(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    latest = PeriodThreshold(op="eq", year=2024, rolling=RelativePeriod(offset=0))
    newest = FilterRequirement(
        id="newest", decision_ids=["measure"], input_id="annual_debt_service", predicates=(latest,)
    )
    assert _plan(decision(), evidence_requirements=[requirement(period_end_open=True), newest])

    with pytest.raises(ValidationError) as refused:
        _plan(decision(), evidence_requirements=[requirement(), newest])

    assert _rolling_error(refused.value).fault is OpenPeriodFault.DATASET_CLOSED


@pytest.mark.parametrize(
    ("requirement_", "from_start"),
    [
        (_rolling_growth("d"), True),
        (_rolling_growth("d", rolling=RollingWindow(window_end=RelativePeriod(offset=0))), False),
        (_rolling_growth("d", rolling=None), False),
        (_windowed_rank(), True),
        *(
            (
                FilterRequirement(
                    id="f",
                    input_id="d",
                    predicates=(
                        PeriodThreshold(op=op, year=2024, rolling=RelativePeriod(offset=offset)),
                    ),
                ),
                from_start,
            )
            for op, offset, from_start in [
                ("gte", 0, True),
                ("gt", -1, True),
                ("eq", 0, True),
                ("lte", 0, False),
            ]
        ),
    ],
)
def test_a_requirement_rolls_from_its_start_when_its_earliest_year_is_relative(
    requirement_: EvidenceRequirement, from_start: bool
) -> None:
    assert rolls_from_its_start(requirement_) is from_start


def test_an_open_period_window_states_its_start_year_alone() -> None:
    period = Period(start=2015, end=2024)

    assert period_window(period) == {"kind": "range", "start_year": 2015, "end_year": 2024}
    assert period_window(period, end_open=True) == {"kind": "range", "start_year": 2015}


def _answer(window: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any]:
    """A Search spec block over `window` resolving `resolved`."""
    return {"definition": {"identity": {"requested_window": window}}, "resolved": resolved}


_ALL = {"kind": "all"}


@pytest.mark.parametrize(
    ("block", "history"),
    [
        (
            _answer(_ALL, {"download": {"rows": 3120, "window": {"start_year": 1973}}}),
            FullHistory(first_year=1973, rows=3120),
        ),
        # The mint's own window wins over the served members' date range.
        (
            _answer(
                _ALL,
                {
                    "download": {"rows": None, "window": {"start_year": 1973, "end_year": 2024}},
                    "stat_vars": {"data_date_range": {"start": "1990", "end": "2024"}},
                },
            ),
            FullHistory(first_year=1973, rows=None),
        ),
        (
            _answer(_ALL, {"stat_vars": {"data_date_range": {"start": "1990", "end": "2024"}}}),
            FullHistory(first_year=1990, rows=None),
        ),
        (_answer(_ALL, {}), None),
        # An answer over a window reads the years inside it, not the dataset's first.
        (
            _answer(
                {"kind": "range", "start_year": 2020, "end_year": 2024},
                {"download": {"rows": 40, "window": {"start_year": 2020, "end_year": 2024}}},
            ),
            None,
        ),
        (
            _answer(
                {"kind": "range", "start_year": 2015},
                {"download": {"rows": 40, "window": {"start_year": 2015, "end_year": 2024}}},
            ),
            None,
        ),
        ({"resolved": {"download": {"window": {"start_year": 1973}}}}, None),
    ],
)
def test_only_an_answer_over_no_window_states_a_datasets_first_year(
    block: dict[str, Any], history: FullHistory | None
) -> None:
    assert full_history(block) == history


@pytest.mark.parametrize(
    ("window", "end_open", "first_year", "retrieved"),
    [
        ({"kind": "range", "start_year": 2015, "end_year": 2024}, False, None, True),
        ({"kind": "range", "start_year": 2015}, False, None, False),
        ({"kind": "range", "start_year": 2015}, True, None, True),
        # A whole-history read starts at the first year the dataset holds, and
        # every other open read at its stated start.
        ({"kind": "range", "start_year": 1970}, True, 1970, True),
        ({"kind": "range", "start_year": 1970}, True, None, False),
        ({"kind": "range", "start_year": 2015}, True, 1970, False),
        ({"kind": "range", "start_year": 2016}, True, None, False),
        ({"kind": "range", "start_year": 2015, "end_year": 2024}, True, None, False),
        ({"kind": "all"}, True, None, False),
        (None, True, None, False),
    ],
)
def test_a_requirement_is_retrieved_over_its_period_or_an_open_range_from_where_it_reads(
    requirement: RequirementFactory,
    window: object,
    end_open: bool,
    first_year: int | None,
    retrieved: bool,
) -> None:
    stated = requirement(period=Period(start=2015, end=2024), period_end_open=end_open)

    assert retrieves_period(window, stated, first_year=first_year) is retrieved


def test_an_operation_stating_other_rolling_than_its_requirement_is_a_mismatch() -> None:
    growth = _rolling_growth("d")
    carried = GrowthOperation(
        id="g", requirement_id="growth", input="d", window_start=2019, window_end=2024
    )
    assert mismatches(growth, carried, {"d": "d"}) == ["rolling"]
    assert mismatches(growth, carried.model_copy(update={"rolling": _LAST_FIVE}), {"d": "d"}) == []

    ranked = _windowed_rank()
    ordering = RankOperation(
        id="r",
        requirement_id="ranked",
        input="d",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        reduction=Reduction.SUM,
        window_start=2019,
        window_end=2024,
    )
    assert mismatches(ranked, ordering, {"d": "d"}) == ["rolling"]

    latest = PeriodThreshold(op="eq", year=2024, rolling=RelativePeriod(offset=0))
    newest = FilterRequirement(id="f", input_id="d", predicates=(latest,))
    fixed = FilterOperation(
        id="f", requirement_id="f", input="d", predicates=(PeriodThreshold(op="eq", year=2024),)
    )
    assert mismatches(newest, fixed, {"d": "d"}) == ["predicates"]


@dataclass(frozen=True)
class _Recorded:
    identity: dict[str, Any]
    groups: tuple[ResolvedGroup, ...] = ()


def test_a_copy_of_a_recorded_dataset_over_another_window_keeps_the_set_its_places_stand_for() -> (
    None
):
    # A recorded dataset over Africa's countries, copied over the
    # question's own years, still names Africa, so the copy's resubmit asks for it.
    fixture = Path(__file__).parent.parent / "fixtures" / "search_spec_region_members.json"
    africa = source_from_spec_block(json.loads(fixture.read_text())["spec"])
    assert africa.identity is not None
    window = {"kind": "range", "start_year": 2010, "end_year": 2020}

    pinned = windowed_pin(africa.spec_id, _Recorded(africa.identity, africa.groups), window)

    assert africa.groups
    assert pinned.groups == africa.groups


# --- The parties an ask breaks down by ---------------------------------------

_BY_CREDITOR_ASKS = ["Debt service by creditor for general government in 2031"]


def _creditor_plan(
    decision: DecisionFactory,
    across: tuple[str, ...],
    *,
    axis: str | None = None,
    group_by: tuple[GroupBy, ...] | None = None,
    declined: tuple[Decline, ...] = (),
    breakdowns: tuple[AskBreakdown, ...] = (AskBreakdown(ask=0, by=("creditor",)),),
    across_years: tuple[int, ...] = (),
) -> Plan:
    """A by-creditor plan: debt service across `across`, answered by a rank on `axis` or an
    aggregate grouping by `group_by` when either is given, else by the dataset itself."""
    requirements: list[EvidenceRequirement] = [
        DatasetRequirement(
            id="debt_service",
            decision_ids=["measure"],
            measure="general government debt service",
            across=across,
            period=Period(start=2031, end=2031),
        )
    ]
    if axis is not None:
        requirements.append(
            RankRequirement(
                id="ranked",
                decision_ids=["measure"],
                input_id="debt_service",
                direction=RankDirection.HIGHEST_FIRST,
                among=RankAmong.POPULATION,
                axis=axis,
                reduction=Reduction.SUM,
            )
        )
    if group_by is not None:
        requirements.append(
            AggregateRequirement(
                id="summed",
                decision_ids=["measure"],
                input_id="debt_service",
                reduction=Reduction.SUM,
                group_by=group_by,
            )
        )
    return Plan(
        question="general government debt service by creditor for 2031",
        decisions=[decision(fixes=DecisionFixes(among=RankAmong.POPULATION))],
        evidence_requirements=requirements,
        asks=_BY_CREDITOR_ASKS,
        ask_breakdowns=breakdowns,
        asks_across_years=across_years,
        answered=[AskAnswer(ask=0, outputs=(requirements[-1].id,))],
        declined=declined,
    )


@pytest.mark.parametrize(
    ("across", "axis", "group_by", "declined", "dropped"),
    [
        pytest.param(("recipient",), None, None, (), True, id="across-recipient-only"),
        pytest.param(("creditor",), None, None, (), False, id="across-creditor"),
        pytest.param(("lenders",), None, None, (), False, id="across-a-creditor-synonym"),
        pytest.param(("recipient", "creditor"), "creditor", None, (), False, id="ranked-creditors"),
        pytest.param(
            ("recipient", "creditor"), "recipient", None, (), True, id="ranked-recipients"
        ),
        pytest.param(
            ("creditor",), None, (GroupBy.PERIOD,), (), True, id="aggregate-sums-creditors-away"
        ),
        pytest.param(
            ("recipient",),
            None,
            None,
            (Decline(ask="by creditor", code=UnsupportedOperation.BREAKDOWN_NOT_KEPT, ask_ref=0),),
            False,
            id="breakdown-declined",
        ),
        pytest.param(
            ("recipient",),
            None,
            None,
            (Decline(ask="by creditor", code=UnsupportedOperation.UNSUPPORTED_ASK, ask_ref=0),),
            True,
            id="declined-under-another-code",
        ),
    ],
)
def test_an_answer_keeps_open_each_party_its_ask_breaks_down_by(
    decision: DecisionFactory,
    across: tuple[str, ...],
    axis: str | None,
    group_by: tuple[GroupBy, ...] | None,
    declined: tuple[Decline, ...],
    dropped: bool,
) -> None:
    def build() -> Plan:
        return _creditor_plan(decision, across, axis=axis, group_by=group_by, declined=declined)

    if not dropped:
        assert build().ask_breakdowns == (AskBreakdown(ask=0, by=("creditor",)),)
        return
    with pytest.raises(ValidationError) as refused:
        build()
    causes = [error.get("ctx", {}).get("error") for error in refused.value.errors()]
    assert [cause.dropped for cause in causes if isinstance(cause, AskRoleDroppedError)] == [
        {0: ("creditor",)}
    ]


_AID_ASK = "Each donor's aid to each recipient"
_DONOR_AND_RECIPIENT = (AskBreakdown(ask=0, by=("donor", "recipient")),)


@pytest.mark.parametrize(
    ("asks", "breakdowns", "outputs", "dropped"),
    [
        pytest.param(
            [_AID_ASK],
            _DONOR_AND_RECIPIENT,
            {0: [("donor", "recipient")]},
            None,
            id="both-parties-in-one-output",
        ),
        pytest.param(
            [_AID_ASK],
            _DONOR_AND_RECIPIENT,
            {0: [("donor",), ("recipient",)]},
            ("recipient",),
            id="a-party-per-output",
        ),
        pytest.param(
            [_AID_ASK],
            _DONOR_AND_RECIPIENT,
            {0: [("recipient",), ("donor",)]},
            ("donor",),
            id="a-party-per-output-recipients-first",
        ),
        pytest.param(
            [_AID_ASK],
            _DONOR_AND_RECIPIENT,
            {0: [("country",), ("donor",)]},
            ("recipient",),
            id="measured-against-the-output-keeping-the-most",
        ),
        pytest.param(
            [_AID_ASK], _DONOR_AND_RECIPIENT, {0: [("donor",)]}, ("recipient",), id="one-of-two"
        ),
        pytest.param(
            ["Aid by donor", "Aid by recipient"],
            (AskBreakdown(ask=0, by=("donor",)), AskBreakdown(ask=1, by=("recipient",))),
            {0: [("donor",)], 1: [("recipient",)]},
            None,
            id="an-ask-per-party",
        ),
        pytest.param(
            ["Aid by donor"],
            (AskBreakdown(ask=0, by=("donor",)),),
            {0: [("creditor",)]},
            None,
            id="another-word-for-the-side",
        ),
        pytest.param(
            ["Debt by creditor and debtor"],
            (AskBreakdown(ask=0, by=("creditor", "debtor")),),
            {0: [("lender",), ("borrower",)]},
            ("debtor",),
            id="a-party-per-output-in-other-words",
        ),
    ],
)
def test_an_answer_keeps_the_parties_of_one_ask_together_in_one_output(
    decision: DecisionFactory,
    asks: list[str],
    breakdowns: tuple[AskBreakdown, ...],
    outputs: dict[int, list[tuple[str, ...]]],
    dropped: tuple[str, ...] | None,
) -> None:
    """`outputs` gives each ask the `across` of each dataset answering it."""
    requirements: list[EvidenceRequirement] = []
    answered: list[AskAnswer] = []
    for ask, each_across in outputs.items():
        datasets = [
            DatasetRequirement(
                id=f"aid_{ask}_{index}",
                decision_ids=["measure"],
                measure="aid disbursed",
                across=across,
                period=Period(start=2031, end=2031),
            )
            for index, across in enumerate(each_across)
        ]
        requirements += datasets
        answered.append(AskAnswer(ask=ask, outputs=tuple(dataset.id for dataset in datasets)))

    def build() -> Plan:
        return Plan(
            question="each donor's aid to each recipient in 2031",
            decisions=[decision(fixes=DecisionFixes(among=RankAmong.POPULATION))],
            evidence_requirements=requirements,
            asks=asks,
            ask_breakdowns=breakdowns,
            answered=answered,
        )

    if dropped is None:
        assert build().ask_breakdowns == breakdowns
        return
    with pytest.raises(ValidationError) as refused:
        build()
    causes = [error.get("ctx", {}).get("error") for error in refused.value.errors()]
    by = breakdowns[0].by
    assert [
        (cause.dropped, cause.together)
        for cause in causes
        if isinstance(cause, AskRoleDroppedError)
    ] == [({0: dropped}, {0: by} if len(by) > 1 else {})]


@pytest.mark.parametrize(
    ("by", "taken"),
    [
        (("debtor", "creditor"), True),
        (("recipient", "debtor"), False),
        (("creditor", "recipient", "debtor"), False),
    ],
)
def test_an_ask_breakdown_takes_one_word_for_each_side_of_the_flow(
    by: tuple[PartyRole, ...], taken: bool
) -> None:
    if taken:
        assert AskBreakdown(ask=0, by=by).by == by
        return
    with pytest.raises(ValidationError):
        AskBreakdown(ask=0, by=by)


@pytest.mark.parametrize(
    "breakdowns",
    [
        (AskBreakdown(ask=1, by=("creditor",)),),
        (AskBreakdown(ask=0, by=("creditor",)), AskBreakdown(ask=0, by=("debtor",))),
    ],
    ids=["unlisted-ask", "ask-twice"],
)
def test_a_breakdown_names_a_listed_ask_once(
    decision: DecisionFactory, breakdowns: tuple[AskBreakdown, ...]
) -> None:
    with pytest.raises(ValidationError, match="breakdown"):
        _creditor_plan(decision, ("creditor",), breakdowns=breakdowns)


def test_a_breakdown_names_at_least_one_party() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        AskBreakdown(ask=0, by=())


def test_an_amendment_under_the_same_asks_keeps_their_breakdowns(
    decision: DecisionFactory,
) -> None:
    plan = _creditor_plan(decision, ("creditor",))

    assert plan.amend(question="q").ask_breakdowns == plan.ask_breakdowns
    assert plan.amend(asks=_BY_CREDITOR_ASKS, answered=plan.answered).ask_breakdowns == ()


def test_an_ask_across_years_enters_the_revision_once_set_and_survives_an_amendment(
    decision: DecisionFactory,
) -> None:
    plan = _creditor_plan(decision, ("creditor",))
    across = _creditor_plan(decision, ("creditor",), across_years=(0,))

    assert "asks_across_years" not in revision_content(plan)
    assert revision_content(across)["asks_across_years"] == [0]
    assert plan_revision(across) != plan_revision(plan)
    assert across.amend(question="q").asks_across_years == (0,)
    assert across.amend(asks=_BY_CREDITOR_ASKS, answered=across.answered).asks_across_years == ()
    amended = across.amend(
        asks=_BY_CREDITOR_ASKS,
        asks_across_years=(0,),
        ask_breakdowns=across.ask_breakdowns,
        answered=across.answered,
    )
    assert amended.asks_across_years == (0,)


@pytest.mark.parametrize("numbers", [(1,), (0, 0)], ids=["unlisted-ask", "ask-twice"])
def test_an_ask_across_years_names_a_listed_ask_once(
    decision: DecisionFactory, numbers: tuple[int, ...]
) -> None:
    with pytest.raises(ValidationError, match="across years"):
        _creditor_plan(decision, ("creditor",), across_years=numbers)
