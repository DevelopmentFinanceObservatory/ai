"""Tests for `harness.plan_display`: turning a `Plan` into wire-ready rows.

Builds plans directly from `planner.plan`/`planner.operations`/`planner.workflow`
rather than through the planner or the compiler -- this module has no
dependency on either, and neither do these tests.
"""

from __future__ import annotations

import typing
from collections.abc import Callable
from typing import Any

import pytest
from harness.plan_display import (
    _label,
    operation_rows,
    requirement_rows,
    shown_decisions,
    shown_plan,
    step_rows,
)
from planner.operations import (
    AggregateOperation,
    CompareOperation,
    FilterOperation,
    GrowthOperation,
    JoinOperation,
    Operation,
    QuotientOfOperation,
    RankOperation,
    RetrieveOperation,
    ShareOfOperation,
    Side,
    SumOfOperation,
    Translation,
)
from planner.plan import (
    AggregateRequirement,
    ComparisonRequirement,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    DecisionScope,
    Entity,
    EvidenceRequirement,
    FilterRequirement,
    FirstYear,
    GrowthRequirement,
    JoinRequirement,
    Period,
    Plan,
    QuotientRequirement,
    RankRequirement,
    RequirementSide,
    ResolvedBy,
    ScopeOptionState,
    ShareOfRequirement,
    SumOfRequirement,
    scope_choice_slot,
)
from planner.records import (
    Breakdown,
    BreakdownDimension,
    DenominatorScope,
    EntityIn,
    EvidenceIssue,
    EvidenceIssueCode,
    GroupBy,
    GrowthChange,
    PeriodThreshold,
    PlaceLevel,
    PlaceType,
    Predicate,
    RankAmong,
    RankDirection,
    Reduction,
    RelativePeriod,
    RollingWindow,
    Severity,
    StatVarReading,
    ValueThreshold,
)
from planner.workflow import (
    ArtifactKind,
    Binding,
    EvidenceBinding,
    OpenEntityAxis,
    ReadingSetBy,
    ResolvedEntity,
    SelectedSource,
    ToolId,
    Workflow,
    WorkflowStep,
)

from .split_plan import SPEC_DEBT, stored_split_plan

# The `decision` fixture, from `services/harness/tests/conftest.py`. Annotated
# structurally rather than imported, the same discipline `tests/planner/test_plan.py`
# uses, since a test module importing a conftest by name is ambiguous when the
# workspace root has one too.
DecisionFactory = Callable[..., Decision]

_SPEC_A = "spec_" + "a" * 32
_SPEC_B = "spec_" + "b" * 32


def _population_decision(decision: DecisionFactory, **overrides: object) -> Decision:
    """A `measure` decision that also fixes `among='population'`, for a population rank."""
    return decision(fixes=DecisionFixes(among=RankAmong.POPULATION), **overrides)


# --- A plan stating one requirement of every kind, wired into a valid DAG -----
#
# Mirrors `tests/planner/test_plan.py::_every_kind_plan`: `dataset_a` feeds
# `totals_a` (aggregate), which feeds `growth_a` (growth) and `rank_a` (rank);
# `dataset_b` feeds `filtered_b` (filter); `joined` (join) reads `filtered_b`
# and `dataset_a`; `share_a` (share_of), `quotient_a` (quotient_of),
# `comparison_a` (comparison) and `sum_a` (sum_of) each read `totals_a` and
# `filtered_b`. Every requirement id below is also its operation id and its
# workflow step id, so the same id names one thing throughout a row.

_EVERY_KIND_REQUIREMENT_ID: dict[type, str] = {
    DatasetRequirement: "dataset_a",
    AggregateRequirement: "totals_a",
    GrowthRequirement: "growth_a",
    ComparisonRequirement: "comparison_a",
    SumOfRequirement: "sum_a",
    RankRequirement: "rank_a",
    ShareOfRequirement: "share_a",
    QuotientRequirement: "quotient_a",
    JoinRequirement: "joined",
    FilterRequirement: "filtered_b",
}


def _every_kind_requirements() -> list[EvidenceRequirement]:
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
        id="joined", decision_ids=["measure"], left_id="filtered_b", right_id="dataset_a"
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
    return [
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
    ]


def _every_kind_plan(decision: DecisionFactory) -> Plan:
    """The DAG above, as a `DRAFT` plan -- enough for `requirement_rows`."""
    return Plan(
        question="How dependent is Kenya on Chinese lending?",
        decisions=[_population_decision(decision)],
        evidence_requirements=_every_kind_requirements(),
    )


def _every_kind_translation() -> Translation:
    return Translation(
        operations=[
            RetrieveOperation(id="dataset_a", requirement_id="dataset_a", spec_id=_SPEC_A),
            RetrieveOperation(id="dataset_b", requirement_id="dataset_b", spec_id=_SPEC_B),
            AggregateOperation(
                id="totals_a",
                requirement_id="totals_a",
                input="dataset_a",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
            GrowthOperation(
                id="growth_a",
                requirement_id="growth_a",
                input="totals_a",
                window_start=2016,
                window_end=2020,
            ),
            FilterOperation(
                id="filtered_b",
                requirement_id="filtered_b",
                input="dataset_b",
                predicates=(ValueThreshold(op="gt", value=0),),
            ),
            RankOperation(
                id="rank_a",
                requirement_id="rank_a",
                input="dataset_a",
                direction=RankDirection.HIGHEST_FIRST,
                among=RankAmong.POPULATION,
                reduction=Reduction.MEAN,
            ),
            JoinOperation(
                id="joined", requirement_id="joined", left="filtered_b", right="dataset_a"
            ),
            ShareOfOperation(
                id="share_a",
                requirement_id="share_a",
                numerator=Side(input="totals_a"),
                denominator=Side(input="filtered_b"),
            ),
            QuotientOfOperation(
                id="quotient_a",
                requirement_id="quotient_a",
                numerator=Side(input="totals_a"),
                denominator=Side(input="filtered_b"),
            ),
            CompareOperation(
                id="comparison_a",
                requirement_id="comparison_a",
                left=Side(input="totals_a"),
                right=Side(input="filtered_b"),
            ),
            SumOfOperation(
                id="sum_a",
                requirement_id="sum_a",
                left=Side(input="totals_a"),
                right=Side(input="filtered_b"),
            ),
        ],
        terminal=[
            "growth_a",
            "rank_a",
            "joined",
            "share_a",
            "quotient_a",
            "comparison_a",
            "sum_a",
        ],
    )


def _every_kind_workflow() -> Workflow:
    def _step(
        step_id: str,
        *,
        tool: str,
        server: typing.Literal["data-mcp", "analysis-mcp"] = "analysis-mcp",
        from_steps: tuple[str, ...] = (),
        args: dict[str, object] | None = None,
    ) -> WorkflowStep:
        return WorkflowStep(
            id=step_id,
            operation_id=step_id,
            tool=ToolId(server=server, name=tool),
            args=args or {},
            bindings={
                f"frame_ref_{i}": Binding(from_step=from_step, output=ArtifactKind.FRAME_REF)
                for i, from_step in enumerate(from_steps)
            },
            input_schema_hash="hash",
        )

    return Workflow(
        steps=[
            _step("dataset_a", tool="get_data", server="data-mcp", args={"spec_id": _SPEC_A}),
            _step("dataset_b", tool="get_data", server="data-mcp", args={"spec_id": _SPEC_B}),
            _step("totals_a", tool="aggregate", from_steps=("dataset_a",)),
            _step("growth_a", tool="growth", from_steps=("totals_a",)),
            _step("filtered_b", tool="filter", from_steps=("dataset_b",)),
            _step("rank_a", tool="rank", from_steps=("dataset_a",)),
            _step("joined", tool="join", from_steps=("filtered_b", "dataset_a")),
            _step("share_a", tool="share_of", from_steps=("totals_a", "filtered_b")),
            _step("quotient_a", tool="quotient_of", from_steps=("totals_a", "filtered_b")),
            _step("comparison_a", tool="compare", from_steps=("totals_a", "filtered_b")),
            _step("sum_a", tool="sum_of", from_steps=("totals_a", "filtered_b")),
        ],
        sources=[
            SelectedSource(spec_id=_SPEC_A, sentence="external debt service paid", has_data=True),
            SelectedSource(spec_id=_SPEC_B, sentence="government revenue", has_data=True),
        ],
        evidence_bindings=[
            EvidenceBinding(requirement_id="dataset_a", step_id="dataset_a", spec_id=_SPEC_A),
            EvidenceBinding(requirement_id="dataset_b", step_id="dataset_b", spec_id=_SPEC_B),
        ],
        terminal_steps=[
            "growth_a",
            "rank_a",
            "joined",
            "share_a",
            "quotient_a",
            "comparison_a",
            "sum_a",
        ],
    )


def _every_kind_compiled_plan(decision: DecisionFactory) -> Plan:
    """The DAG above, compiled -- enough for `step_rows` and `operation_rows`."""
    return _every_kind_plan(decision).with_translation(
        translation=_every_kind_translation(), workflow=_every_kind_workflow()
    )


# --- The union members, read at runtime ---------------------------------------
#
# `EvidenceRequirement` and `Operation` are each `Annotated[<union>, Field(...)]`;
# `get_args` peels the annotation, then the union, to the classes themselves.
# A kind added to either union and left out of `_EVERY_KIND_REQUIREMENT_ID`
# above fails these tests with a `KeyError` rather than passing having never
# been asked for a sentence.


def _union_members(annotated: object) -> list[type]:
    union = typing.get_args(annotated)[0]
    return list(typing.get_args(union))


_REQUIREMENT_KINDS = _union_members(EvidenceRequirement)
_OPERATION_KINDS = _union_members(Operation)


@pytest.mark.parametrize("requirement_cls", _REQUIREMENT_KINDS, ids=lambda c: c.__name__)
def test_every_evidence_requirement_kind_has_a_sentence(
    requirement_cls: type, decision: DecisionFactory
) -> None:
    plan = _every_kind_plan(decision)
    requirement_id = _EVERY_KIND_REQUIREMENT_ID[requirement_cls]

    rows = requirement_rows(plan)

    row = next(row for row in rows if row["id"] == requirement_id)
    assert isinstance(row["sentence"], str)
    assert row["sentence"].strip() != ""
    assert row["sentence"] != row["kind"]


@pytest.mark.parametrize("operation_cls", _OPERATION_KINDS, ids=lambda c: c.__name__)
def test_every_operation_kind_has_a_sentence(
    operation_cls: type, decision: DecisionFactory
) -> None:
    plan = _every_kind_compiled_plan(decision)
    assert plan.translation is not None
    operation_id = next(
        operation.id
        for operation in plan.translation.operations
        if isinstance(operation, operation_cls)
    )

    rows = operation_rows(plan)

    row = next(row for row in rows if row["id"] == operation_id)
    assert isinstance(row["sentence"], str)
    assert row["sentence"].strip() != ""


def test_requirement_rows_carries_id_kind_decision_ids_and_sentence(
    decision: DecisionFactory,
) -> None:
    plan = _every_kind_plan(decision)

    rows = requirement_rows(plan)

    dataset_row = next(row for row in rows if row["id"] == "dataset_a")
    assert dataset_row["kind"] == "dataset"
    assert dataset_row["decision_ids"] == ["measure"]
    assert "external debt service paid" in dataset_row["sentence"]
    assert "Kenya" in dataset_row["sentence"]
    sum_row = next(row for row in rows if row["id"] == "sum_a")
    assert sum_row["kind"] == "sum_of"
    assert sum_row["sentence"].endswith(
        "for each year both report; years one side lacks are left out."
    )


def _scope_decision(
    pin: str,
    axis: typing.Literal["sector", "measure_basis"],
    *,
    resolved_by: ResolvedBy = ResolvedBy.PLANNER_DEFAULT,
    other: typing.Literal["available", "unavailable"] = "available",
) -> Decision:
    """The scope decision `propose_plan` writes for `axis` of a pin of `pin`:
    two options, the first chosen, the other `other`."""
    ids = {
        "sector": ("sector:120+130", "sector:120"),
        "measure_basis": ("measure_basis:gross", "measure_basis:net"),
    }[axis]
    return Decision(
        slot=scope_choice_slot(pin, axis),
        description=f"How {axis} is counted.",
        options=(f"{ids[0]} label", f"{ids[1]} label"),
        chosen=f"{ids[0]} label",
        resolved_by=resolved_by,
        rationale="Search's default for this question.",
        fixes=DecisionFixes(source_spec_id=pin),
        scope=DecisionScope(
            choice_id=f"{pin}:{axis}",
            axis=axis,
            options=(
                ScopeOptionState(option_id=ids[0], status="available"),
                ScopeOptionState(
                    option_id=ids[1],
                    status=other,
                    code="conflicts_with" if other == "unavailable" else None,
                ),
            ),
        ),
    )


def test_sources_counted_alike_share_one_scope_row_and_others_keep_their_own(
    decision: DecisionFactory,
) -> None:
    """Two pins whose sector choices are the same share one row, the user's;
    basis choices whose options are not equally available keep a row each.
    Each requirement cites the row standing for its own decision."""
    sector_a = _scope_decision(_SPEC_A, "sector")
    sector_b = _scope_decision(_SPEC_B, "sector", resolved_by=ResolvedBy.USER)
    basis_a = _scope_decision(_SPEC_A, "measure_basis")
    basis_b = _scope_decision(_SPEC_B, "measure_basis", other="unavailable")

    def dataset(id: str, pin: str, *cited: Decision) -> DatasetRequirement:
        return DatasetRequirement(
            id=id,
            decision_ids=["measure", *(d.slot for d in cited)],
            measure="health aid",
            entities=[Entity(name="Africa", role="recipient")],
            period=Period(start=2015, end=2024),
            source_spec_id=pin,
        )

    plan = Plan(
        question="Health aid to Africa from two donors",
        decisions=[decision(), sector_a, basis_a, sector_b, basis_b],
        evidence_requirements=[
            dataset("aid_a", _SPEC_A, sector_a, basis_a),
            dataset("aid_b", _SPEC_B, sector_b, basis_b),
        ],
    )

    assert [d.slot for d in shown_decisions(plan)] == [
        "measure",
        basis_a.slot,
        sector_b.slot,
        basis_b.slot,
    ]
    rows = {row["id"]: row["decision_ids"] for row in requirement_rows(plan)}
    assert rows == {
        "aid_a": ["measure", sector_b.slot, basis_a.slot],
        "aid_b": ["measure", sector_b.slot, basis_b.slot],
    }


def _one_requirement_compiled_plan(
    decision: DecisionFactory,
    *,
    definition: str | None,
    breakdown: Breakdown | None = None,
    **source_fields: Any,
) -> Plan:
    """A `DRAFT` plan reading one dataset requirement, compiled, whose source carries `definition`
    and any other `source_fields`, and whose requirement states `breakdown`.

    `totals_a` (an aggregate) reads `dataset_a`, so the plan carries a
    calculation and its own source alongside the one requirement this test
    cares about.
    """
    dataset_a = DatasetRequirement(
        id="dataset_a",
        decision_ids=["measure"],
        measure="external debt service paid",
        entities=[Entity(name="Kenya", role="debtor")],
        period=Period(start=2015, end=2024),
        breakdown=breakdown,
    )
    totals_a = AggregateRequirement(
        id="totals_a",
        decision_ids=["measure"],
        input_id="dataset_a",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    plan = Plan(
        question="How much does Kenya pay to service its debt?",
        decisions=[decision()],
        evidence_requirements=[dataset_a, totals_a],
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="dataset_a", requirement_id="dataset_a", spec_id=_SPEC_A),
            AggregateOperation(
                id="totals_a",
                requirement_id="totals_a",
                input="dataset_a",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
        ],
        terminal=["totals_a"],
    )
    workflow = Workflow(
        steps=[
            WorkflowStep(
                id="dataset_a",
                operation_id="dataset_a",
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": _SPEC_A},
                input_schema_hash="hash",
            ),
            WorkflowStep(
                id="totals_a",
                operation_id="totals_a",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                bindings={
                    "frame_ref": Binding(from_step="dataset_a", output=ArtifactKind.FRAME_REF)
                },
                input_schema_hash="hash",
            ),
        ],
        sources=[
            SelectedSource(
                spec_id=_SPEC_A,
                sentence=_SOURCE_SENTENCE,
                has_data=True,
                definition=definition,
                **source_fields,
            )
        ],
        evidence_bindings=[
            EvidenceBinding(requirement_id="dataset_a", step_id="dataset_a", spec_id=_SPEC_A)
        ],
        terminal_steps=["totals_a"],
    )
    return plan.with_translation(translation=translation, workflow=workflow)


# What Search says the selected source holds, which the compiled card states.
_SOURCE_SENTENCE = "External debt service paid by Kenya, 2015 to 2024"


def test_a_selected_sources_definition_stays_out_of_the_dataset_sentence(
    decision: DecisionFactory,
) -> None:
    """A StatVar's `describe` definition is its constraint encoding, which a
    user cannot read, so the readback states Search's sentence for the source alone."""
    plan = _one_requirement_compiled_plan(
        decision, definition="md=PerCapita,mq=Nominal,mp=amount,pt=EconomicActivity"
    )

    rows = requirement_rows(plan)

    dataset_row = next(row for row in rows if row["id"] == "dataset_a")
    assert dataset_row["sentence"] == f"{_SOURCE_SENTENCE}."


def test_operation_rows_carries_id_kind_requirement_id_and_sentence(
    decision: DecisionFactory,
) -> None:
    plan = _every_kind_compiled_plan(decision)

    rows = operation_rows(plan)

    totals_row = next(row for row in rows if row["id"] == "totals_a")
    assert totals_row["kind"] == "aggregate"
    assert totals_row["requirement_id"] == "totals_a"
    assert "external debt service paid" in totals_row["sentence"]


# --- A minimal retrieve-then-aggregate plan ------------------------------------


def _retrieve_then_aggregate_plan(
    decision: DecisionFactory, group_by: tuple[GroupBy, ...] = (GroupBy.PERIOD,)
) -> Plan:
    dataset = DatasetRequirement(
        id="annual_debt_service",
        decision_ids=["measure"],
        measure="external debt service paid",
        entities=[Entity(name="Kenya", role="debtor")],
        period=Period(start=2015, end=2024),
    )
    totals = AggregateRequirement(
        id="annual_totals",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        reduction=Reduction.SUM,
        group_by=group_by,
    )
    draft = Plan(
        question="How much has Kenya paid in debt service each year?",
        decisions=[decision()],
        evidence_requirements=[dataset, totals],
    )
    translation = Translation(
        operations=[
            RetrieveOperation(
                id="annual_debt_service", requirement_id="annual_debt_service", spec_id=_SPEC_A
            ),
            AggregateOperation(
                id="annual_totals",
                requirement_id="annual_totals",
                input="annual_debt_service",
                reduction=Reduction.SUM,
                group_by=group_by,
            ),
        ],
        terminal=["annual_totals"],
    )
    workflow = Workflow(
        steps=[
            WorkflowStep(
                id="annual_debt_service",
                operation_id="annual_debt_service",
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": _SPEC_A},
                input_schema_hash="hash",
            ),
            WorkflowStep(
                id="annual_totals",
                operation_id="annual_totals",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                args={"aggregation": "sum", "group_by": ["time_period"]},
                bindings={
                    "frame_ref": Binding(
                        from_step="annual_debt_service", output=ArtifactKind.FRAME_REF
                    )
                },
                input_schema_hash="hash",
            ),
        ],
        sources=[
            SelectedSource(spec_id=_SPEC_A, sentence="external debt service paid", has_data=True)
        ],
        evidence_bindings=[
            EvidenceBinding(
                requirement_id="annual_debt_service", step_id="annual_debt_service", spec_id=_SPEC_A
            )
        ],
        terminal_steps=["annual_totals"],
    )
    return draft.with_translation(translation=translation, workflow=workflow)


def test_steps_that_read_alike_are_numbered_in_the_order_the_card_lists_them(
    decision: DecisionFactory,
) -> None:
    # The workflow runs `second_totals` before `first_totals`, so the card
    # lists it first and it takes "(1)".
    plan = _retrieve_then_aggregate_plan(decision)
    assert plan.translation is not None and plan.workflow is not None
    dataset, totals = plan.evidence_requirements
    total_op = plan.translation.operations[1]
    total_step = plan.workflow.steps[1]
    ids = ["second_totals", "first_totals"]
    draft = Plan(
        question=plan.question,
        decisions=plan.decisions,
        evidence_requirements=[
            dataset,
            *(totals.model_copy(update={"id": rid}) for rid in reversed(ids)),
        ],
    )
    compiled = draft.with_translation(
        translation=plan.translation.model_copy(
            update={
                "operations": [
                    plan.translation.operations[0],
                    *(
                        total_op.model_copy(update={"id": rid, "requirement_id": rid})
                        for rid in ids
                    ),
                ],
                "terminal": ids,
            }
        ),
        workflow=plan.workflow.model_copy(
            update={
                "steps": [
                    plan.workflow.steps[0],
                    *(
                        total_step.model_copy(update={"id": rid, "operation_id": rid})
                        for rid in ids
                    ),
                ],
                "terminal_steps": ids,
            }
        ),
    )

    rows = step_rows(compiled)

    assert [(row["id"], row["label"]) for row in rows] == [
        ("annual_debt_service", "Fetch external debt service paid for Kenya"),
        ("second_totals", "Total external debt service paid by year (1)"),
        ("first_totals", "Total external debt service paid by year (2)"),
    ]


@pytest.mark.parametrize(
    ("group_by", "label", "sentence"),
    [
        pytest.param(
            (GroupBy.PERIOD,),
            "Total external debt service paid by year",
            "The total of external debt service paid, grouped by year.",
            id="by-year",
        ),
        pytest.param(
            (),
            "Total external debt service paid as one figure over all the years and parties read",
            "The total of external debt service paid, as one figure over all the years and "
            "parties read.",
            id="one-figure",
        ),
    ],
)
def test_a_retrieve_then_aggregate_plan_produces_two_labelled_steps(
    decision: DecisionFactory, group_by: tuple[GroupBy, ...], label: str, sentence: str
) -> None:
    plan = _retrieve_then_aggregate_plan(decision, group_by)

    rows = step_rows(plan)

    assert [row["id"] for row in rows] == ["annual_debt_service", "annual_totals"]
    labels = [row["label"] for row in rows]
    assert labels[0] == "Fetch external debt service paid for Kenya"
    assert labels[1] == label
    sentences = {row["id"]: row["sentence"] for row in requirement_rows(plan)}
    assert sentences["annual_totals"] == sentence
    # Never a bare tool name.
    assert "get_data" not in labels[0]
    assert "aggregate" not in labels[1]


def test_a_steps_args_are_the_workflow_steps_args_dict_unchanged(
    decision: DecisionFactory,
) -> None:
    plan = _retrieve_then_aggregate_plan(decision)

    rows = step_rows(plan)

    retrieve_row = next(row for row in rows if row["id"] == "annual_debt_service")
    aggregate_row = next(row for row in rows if row["id"] == "annual_totals")
    assert retrieve_row["args"] == {"spec_id": _SPEC_A}
    assert aggregate_row["args"] == {"aggregation": "sum", "group_by": ["time_period"]}


# --- The four-step rank chain: retrieve -> rank -> filter -> growth -----------
#
# Mirrors `tests/planner/test_plan.py::test_a_growth_over_a_rank_needs_per_period_a_filter_and_absolute`.
# A per-period rank with a limit cuts each period, so no growth may read it;
# with `grow=False` the chain ends at the filter.


def _rank_chain_plan(
    decision: DecisionFactory,
    *,
    grow: bool = True,
    role: str = "debtor",
    **rank_fields: typing.Any,
) -> Plan:
    dataset = DatasetRequirement(
        id="annual_debt_service",
        decision_ids=["measure"],
        measure="external debt service paid",
        entities=[Entity(name="Kenya", role=role)],
        period=Period(start=2015, end=2024),
    )
    rank = RankRequirement(
        id="ranked",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        **({"direction": RankDirection.HIGHEST_FIRST} | rank_fields),
        among=RankAmong.POPULATION,
        per_period=True,
    )
    narrowed = FilterRequirement(
        id="named_entities",
        decision_ids=["measure"],
        input_id="ranked",
        predicates=(EntityIn(names=["Kenya"], role=role),),
    )
    growth = GrowthRequirement(
        id="growth",
        decision_ids=["measure"],
        input_id="named_entities",
        window_start=2016,
        window_end=2020,
        change=GrowthChange.ABSOLUTE,
    )
    steps = 4 if grow else 3
    terminal = ["growth" if grow else "named_entities"]
    draft = Plan(
        question="How has Kenya's rank among debtors moved?",
        decisions=[_population_decision(decision)],
        evidence_requirements=[dataset, rank, narrowed, growth][:steps],
    )
    translation = Translation(
        operations=[
            RetrieveOperation(
                id="annual_debt_service", requirement_id="annual_debt_service", spec_id=_SPEC_A
            ),
            RankOperation(
                id="ranked",
                requirement_id="ranked",
                input="annual_debt_service",
                **({"direction": RankDirection.HIGHEST_FIRST} | rank_fields),
                among=RankAmong.POPULATION,
                per_period=True,
            ),
            FilterOperation(
                id="named_entities",
                requirement_id="named_entities",
                input="ranked",
                predicates=(EntityIn(names=["Kenya"], role=role),),
            ),
            GrowthOperation(
                id="growth",
                requirement_id="growth",
                input="named_entities",
                window_start=2016,
                window_end=2020,
                change=GrowthChange.ABSOLUTE,
            ),
        ][:steps],
        terminal=terminal,
    )
    workflow = Workflow(
        steps=[
            WorkflowStep(
                id="annual_debt_service",
                operation_id="annual_debt_service",
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": _SPEC_A},
                input_schema_hash="hash",
            ),
            WorkflowStep(
                id="ranked",
                operation_id="ranked",
                tool=ToolId(server="analysis-mcp", name="rank"),
                args={"direction": "highest_first", "per_period": True},
                bindings={
                    "frame_ref": Binding(
                        from_step="annual_debt_service", output=ArtifactKind.FRAME_REF
                    )
                },
                input_schema_hash="hash",
            ),
            WorkflowStep(
                id="named_entities",
                operation_id="named_entities",
                tool=ToolId(server="analysis-mcp", name="filter"),
                args={"names": ["Kenya"], "role": "debtor"},
                bindings={"frame_ref": Binding(from_step="ranked", output=ArtifactKind.FRAME_REF)},
                input_schema_hash="hash",
            ),
            WorkflowStep(
                id="growth",
                operation_id="growth",
                tool=ToolId(server="analysis-mcp", name="growth"),
                args={"window_start": 2016, "window_end": 2020, "change": "absolute"},
                bindings={
                    "frame_ref": Binding(from_step="named_entities", output=ArtifactKind.FRAME_REF)
                },
                input_schema_hash="hash",
            ),
        ][:steps],
        sources=[
            SelectedSource(spec_id=_SPEC_A, sentence="external debt service paid", has_data=True)
        ],
        evidence_bindings=[
            EvidenceBinding(
                requirement_id="annual_debt_service", step_id="annual_debt_service", spec_id=_SPEC_A
            )
        ],
        terminal_steps=terminal,
    )
    return draft.with_translation(translation=translation, workflow=workflow)


def test_the_four_step_rank_chain_labels_distinguish_every_step(decision: DecisionFactory) -> None:
    plan = _rank_chain_plan(decision)

    rows = step_rows(plan)

    labels = [row["label"] for row in rows]
    assert labels == [
        "Fetch external debt service paid for Kenya",
        "Rank debtors by external debt service paid",
        "Narrow to Kenya",
        "Growth in external debt service paid, 2016-2020",
    ]
    # Every label distinguishes the step it names: none repeats another.
    assert len(set(labels)) == len(labels)


@pytest.mark.parametrize(
    ("role", "direction", "limit", "label"),
    [
        (
            "debtor",
            RankDirection.HIGHEST_FIRST,
            10,
            "Rank the top 10 debtors by external debt service paid",
        ),
        (
            "debtor",
            RankDirection.LOWEST_FIRST,
            2,
            "Rank the bottom 2 debtors by external debt service paid",
        ),
        (
            "debtor",
            RankDirection.HIGHEST_FIRST,
            1,
            "Rank the top debtor by external debt service paid",
        ),
        (
            "country",
            RankDirection.HIGHEST_FIRST,
            10,
            "Rank the top 10 countries by external debt service paid",
        ),
    ],
)
def test_a_limited_rank_label_names_the_end_and_the_count_it_keeps(
    decision: DecisionFactory, role: str, direction: RankDirection, limit: int, label: str
) -> None:
    plan = _rank_chain_plan(decision, grow=False, role=role, direction=direction, limit=limit)

    rank_row = next(row for row in step_rows(plan) if row["id"] == "ranked")

    assert rank_row["label"] == label


def test_a_limited_per_period_rank_sentence_keeps_the_limit_in_each_period(
    decision: DecisionFactory,
) -> None:
    plan = _rank_chain_plan(decision, grow=False, limit=10)

    sentence = next(row for row in requirement_rows(plan) if row["id"] == "ranked")["sentence"]

    assert sentence.endswith(
        "; each period is ranked on its own, and its kept entities carry their measured "
        "values; keeps the top 10 debtors in each period, more when 10th place is shared."
    )
    assert "rank position" not in sentence


def test_a_rank_sentence_states_no_limit_when_none_is_set(decision: DecisionFactory) -> None:
    plan = _rank_chain_plan(decision)

    sentence = next(row for row in requirement_rows(plan) if row["id"] == "ranked")["sentence"]

    assert "keeps" not in sentence
    assert "place is shared" not in sentence


# --- Rank and share_of sentences state the population/series-shape claims -----


def test_a_rank_sentence_states_its_population_and_whether_it_is_per_period(
    decision: DecisionFactory,
) -> None:
    dataset = DatasetRequirement(
        id="dataset_a",
        decision_ids=["measure"],
        measure="external debt service paid",
        entities=[Entity(name="Kenya", role="debtor")],
        period=Period(start=2024, end=2024),
    )
    narrowed = FilterRequirement(
        id="named_entities",
        decision_ids=["measure"],
        input_id="dataset_a",
        predicates=(EntityIn(names=["Kenya"], role="debtor"),),
    )
    named_rank = RankRequirement(
        id="rank_named",
        decision_ids=["measure"],
        input_id="named_entities",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.NAMED,
        per_period=False,
    )
    plan = Plan(
        question="q",
        decisions=[decision(fixes=DecisionFixes(among=RankAmong.NAMED))],
        evidence_requirements=[dataset, narrowed, named_rank],
    )

    rows = requirement_rows(plan)

    sentence = next(row for row in rows if row["id"] == "rank_named")["sentence"]
    assert "ranks exactly the entities named" in sentence
    assert "the window is collapsed into one ranking" in sentence

    one_year = named_rank.model_copy(update={"window_start": 2020, "window_end": 2020})
    one_year_plan = plan.model_copy(update={"evidence_requirements": [dataset, narrowed, one_year]})
    one_year_sentence = next(
        row for row in requirement_rows(one_year_plan) if row["id"] == "rank_named"
    )["sentence"]
    assert one_year_sentence.endswith("; ranks 2020.")

    bottom_two = named_rank.model_copy(update={"direction": RankDirection.LOWEST_FIRST, "limit": 2})
    limited = Plan(
        question="q",
        decisions=[decision(fixes=DecisionFixes(among=RankAmong.NAMED))],
        evidence_requirements=[dataset, narrowed, bottom_two],
    )
    limited_sentence = next(row for row in requirement_rows(limited) if row["id"] == "rank_named")[
        "sentence"
    ]
    assert limited_sentence.endswith(
        "; the window is collapsed into one ranking; keeps the bottom 2 debtors, "
        "more when 2nd place is shared."
    )


@pytest.mark.parametrize(
    ("numerator_measure", "numerator_entities", "denominator_measure", "axis", "expected"),
    [
        pytest.param(
            "lending",
            [Entity(name="China", role="creditor")],
            "GDP",
            "creditor",
            "Share of lending per creditor, each divided by one total of GDP.",
            id="lending-over-gdp",
        ),
        pytest.param(
            "ODA",
            [Entity(name="UK", role="donor"), Entity(name="Kenya", role="recipient")],
            "GNI",
            "donor",
            "Share of ODA per donor, each divided by one total of GNI.",
            id="oda-over-gni",
        ),
    ],
)
def test_a_pooled_share_over_another_dataset_names_its_measure_not_the_shares_axis(
    decision: DecisionFactory,
    numerator_measure: str,
    numerator_entities: list[Entity],
    denominator_measure: str,
    axis: str,
    expected: str,
) -> None:
    # "Each donor's ODA to Kenya as a share of Kenya's GNI": the GNI dataset
    # holds no donors, so a sentence pooling "across all donors" is false.
    dataset_a = DatasetRequirement(
        id="dataset_a",
        decision_ids=["measure"],
        measure=numerator_measure,
        entities=numerator_entities,
        period=Period(start=2015, end=2024),
    )
    dataset_b = DatasetRequirement(
        id="dataset_b",
        decision_ids=["measure"],
        measure=denominator_measure,
        entities=[Entity(name="Kenya", role="debtor" if axis == "creditor" else "recipient")],
        period=Period(start=2015, end=2024),
    )
    keyed_share = ShareOfRequirement(
        id="share_keyed",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="dataset_a"),
        denominator=RequirementSide(input_id="dataset_b"),
        axis=axis,
        denominator_scope=DenominatorScope.POOLED,
    )
    plan = Plan(
        question="q",
        decisions=[decision(fixes=DecisionFixes(denominator_scope=DenominatorScope.POOLED))],
        evidence_requirements=[dataset_a, dataset_b, keyed_share],
    )

    rows = requirement_rows(plan)

    sentence = next(row for row in rows if row["id"] == "share_keyed")["sentence"]
    assert sentence == expected


@pytest.mark.parametrize(
    ("numerator_role", "axis", "denominator_axis", "label", "sentence"),
    [
        pytest.param(
            "country",
            None,
            None,
            "Divide GDP by population",
            "GDP divided by population, at each period both report.",
            id="one-figure",
        ),
        pytest.param(
            "recipient",
            "recipient",
            "country",
            "Divide GDP by population per recipient",
            "GDP per recipient, each divided by its own population, at each period both report.",
            id="keyed-by-counterpart",
        ),
        pytest.param(
            "recipient",
            "recipient",
            None,
            "Divide GDP by population per recipient",
            "GDP per recipient, each divided by population, at each period both report.",
            id="keyed-over-one-total",
        ),
    ],
)
def test_a_quotient_names_both_measures_and_whose_denominator_each_entity_divides_by(
    decision: DecisionFactory,
    numerator_role: str,
    axis: str | None,
    denominator_axis: str | None,
    label: str,
    sentence: str,
) -> None:
    plan = Plan(
        question="q",
        decisions=[decision()],
        evidence_requirements=[
            DatasetRequirement(
                id="gdp",
                decision_ids=["measure"],
                measure="GDP",
                entities=[Entity(name="Kenya", role=numerator_role)],
                period=Period(start=2015, end=2024),
            ),
            DatasetRequirement(
                id="population",
                decision_ids=["measure"],
                measure="population",
                entities=[Entity(name="Kenya", role="country")],
                period=Period(start=2015, end=2024),
            ),
            QuotientRequirement(
                id="quotient",
                decision_ids=["measure"],
                numerator=RequirementSide(input_id="gdp"),
                denominator=RequirementSide(input_id="population"),
                axis=axis,
                denominator_axis=denominator_axis,
            ),
        ],
    )

    quotient = plan.requirement("quotient")
    row = next(row for row in requirement_rows(plan) if row["id"] == "quotient")

    assert _label(quotient, plan) == label
    assert row["sentence"] == sentence


def _pooled_recipient_share_sentence(
    decision: DecisionFactory,
    denominator_steps: list[EvidenceRequirement],
    *,
    denominator_entity: str | None = None,
) -> str:
    """The card sentence for the UK's ODA per recipient, pooled over the last of `denominator_steps`.

    The dataset names six recipients for the steps' filters to keep. A rank
    over the population cites the `population` decision.
    """
    dataset = DatasetRequirement(
        id="dataset",
        decision_ids=["measure"],
        measure="ODA",
        entities=[
            Entity(name="UK", role="donor"),
            *(Entity(name=name, role="recipient") for name in _SIX),
        ],
        period=Period(start=2015, end=2024),
    )
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="dataset"),
        denominator=RequirementSide(input_id=denominator_steps[-1].id, entity=denominator_entity),
        axis="recipient",
        denominator_scope=DenominatorScope.POOLED,
    )
    decisions = [decision(fixes=DecisionFixes(denominator_scope=DenominatorScope.POOLED))]
    if any("population" in step.decision_ids for step in denominator_steps):
        decisions.append(
            decision(slot="population", fixes=DecisionFixes(among=RankAmong.POPULATION))
        )
    plan = Plan(
        question="q",
        decisions=decisions,
        evidence_requirements=[dataset, *denominator_steps, share],
    )
    return next(row for row in requirement_rows(plan) if row["id"] == "share")["sentence"]


def _filter(id_: str, input_id: str, *predicates: Predicate) -> FilterRequirement:
    return FilterRequirement(
        id=id_, decision_ids=["measure"], input_id=input_id, predicates=predicates
    )


def _total(input_id: str, *group_by: GroupBy, id_: str = "total") -> AggregateRequirement:
    return AggregateRequirement(
        id=id_,
        decision_ids=["measure"],
        input_id=input_id,
        reduction=Reduction.SUM,
        group_by=group_by or (GroupBy.PERIOD,),
    )


def _top(
    id_: str,
    input_id: str,
    limit: int,
    among: RankAmong = RankAmong.POPULATION,
    axis: str = "recipient",
) -> RankRequirement:
    return RankRequirement(
        id=id_,
        decision_ids=["measure", "population"] if among is RankAmong.POPULATION else ["measure"],
        input_id=input_id,
        direction=RankDirection.HIGHEST_FIRST,
        among=among,
        axis=axis,
        reduction=Reduction.SUM,
        per_period=True,
        limit=limit,
    )


# --- A rank names the value its input holds, after every transform --------------


def _debt(id_: str = "debt", measure: str = "external debt service paid") -> DatasetRequirement:
    return DatasetRequirement(
        id=id_,
        decision_ids=["measure"],
        measure=measure,
        entities=[Entity(name="Kenya", role="debtor")],
        period=Period(start=2015, end=2024),
    )


def _growth(input_id: str, change: GrowthChange) -> GrowthRequirement:
    return GrowthRequirement(
        id="grew",
        decision_ids=["measure"],
        input_id=input_id,
        window_start=2016,
        window_end=2020,
        change=change,
    )


def _top_five(
    input_id: str, *, per_period: bool, reduction: Reduction | None = None
) -> RankRequirement:
    return RankRequirement(
        id="ranked",
        decision_ids=["measure"],
        input_id=input_id,
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        per_period=per_period,
        limit=5,
        reduction=reduction,
    )


def _per_debtor_share(numerator_id: str) -> ShareOfRequirement:
    return ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=numerator_id),
        denominator=RequirementSide(input_id="revenue"),
        axis="debtor",
        denominator_scope=DenominatorScope.POOLED,
    )


@pytest.mark.parametrize(
    ("requirements", "value"),
    [
        pytest.param(
            [_debt(), _growth("debt", GrowthChange.RELATIVE), _top_five("grew", per_period=False)],
            "the percentage change in external debt service paid from 2016 to 2020",
            id="relative-growth",
        ),
        pytest.param(
            [_debt(), _growth("debt", GrowthChange.ABSOLUTE), _top_five("grew", per_period=False)],
            "the change in external debt service paid from 2016 to 2020",
            id="absolute-growth",
        ),
        pytest.param(
            [
                _debt(),
                _total("debt", GroupBy.ENTITY, GroupBy.PERIOD),
                _top_five("total", per_period=True),
            ],
            "the total of external debt service paid",
            id="aggregate",
        ),
        pytest.param(
            [
                _debt(),
                _debt("revenue", "government revenue"),
                _total("debt", GroupBy.ENTITY, GroupBy.PERIOD),
                _per_debtor_share("total"),
                _top_five("share", per_period=True),
            ],
            "share of the total of external debt service paid",
            id="share-of-a-total",
        ),
        pytest.param(
            [
                _debt(),
                _debt("revenue", "government revenue"),
                _total("debt", GroupBy.ENTITY, GroupBy.PERIOD),
                QuotientRequirement(
                    id="quotient",
                    decision_ids=["measure"],
                    numerator=RequirementSide(input_id="total"),
                    denominator=RequirementSide(input_id="revenue"),
                    axis="debtor",
                    denominator_axis="debtor",
                ),
                _top_five("quotient", per_period=True),
            ],
            "the total of external debt service paid divided by government revenue",
            id="quotient-of-a-total",
        ),
        pytest.param(
            [
                _debt(),
                _filter("recent", "debt", PeriodThreshold(op="gte", year=2016)),
                _growth("recent", GrowthChange.RELATIVE),
                _top_five("grew", per_period=False),
            ],
            "the percentage change in external debt service paid from 2016 to 2020",
            id="growth-over-a-filter",
        ),
        pytest.param(
            [_debt(), _top_five("debt", per_period=False, reduction=Reduction.SUM)],
            "the total of external debt service paid",
            id="rank-summing-the-window",
        ),
        pytest.param(
            [_debt(), _top_five("debt", per_period=True, reduction=Reduction.SUM)],
            "the total of external debt service paid",
            id="rank-summing-each-period",
        ),
        pytest.param(
            [_debt(), _top_five("debt", per_period=False, reduction=Reduction.MEAN)],
            "the average of external debt service paid",
            id="rank-averaging-the-window",
        ),
        pytest.param(
            [_debt(), _top_five("debt", per_period=True, reduction=Reduction.MAX)],
            "the highest of external debt service paid",
            id="rank-taking-each-period-highest",
        ),
        pytest.param(
            [
                _debt(),
                _total("debt", GroupBy.ENTITY, GroupBy.PERIOD),
                _top_five("total", per_period=False, reduction=Reduction.MEAN),
            ],
            "the average of the total of external debt service paid",
            id="rank-averaging-a-total",
        ),
    ],
)
def test_a_rank_names_the_value_its_input_holds(
    decision: DecisionFactory, requirements: list[EvidenceRequirement], value: str
) -> None:
    """The card's label and sentence state what the rank orders by: the top
    5 by growth are not the top 5 by the amount, and a share of a total is
    not a share of the amount."""
    plan = Plan(
        question="q",
        decisions=[
            decision(
                fixes=DecisionFixes(
                    among=RankAmong.POPULATION, denominator_scope=DenominatorScope.POOLED
                )
            )
        ],
        evidence_requirements=requirements,
    )

    sentence = next(row for row in requirement_rows(plan) if row["id"] == "ranked")["sentence"]

    assert sentence.startswith(f"Ranks by {value}: ")
    assert _label(plan.requirement("ranked"), plan) == f"Rank the top 5 debtors by {value}"


_SIX = ["Kenya", "Ethiopia", "Chad", "Mali", "Niger", "Sudan"]


_TOP_5 = '"Rank the top 5 recipients by the total of ODA"'


@pytest.mark.parametrize(
    ("steps", "entity", "ending"),
    [
        pytest.param(
            [_total("dataset")],
            None,
            "each divided by one total of ODA.",
            id="no-narrowing",
        ),
        pytest.param(
            [_filter("recent", "dataset", PeriodThreshold(op="gte", year=2020)), _total("recent")],
            None,
            "each divided by one total of ODA.",
            id="year-filter",
        ),
        pytest.param(
            [_top("top", "dataset", 5)],
            None,
            f"each divided by one total of ODA over the rows {_TOP_5} keeps.",
            id="limit",
        ),
        pytest.param(
            [_top("top", "dataset", 5), _total("top")],
            None,
            f"each divided by one total of ODA over the rows {_TOP_5} keeps.",
            id="limit-then-total",
        ),
        pytest.param(
            [_top("top", "dataset", 5, axis="donor")],
            None,
            "each divided by one total of ODA over the rows "
            '"Rank the top 5 donors by the total of ODA" keeps.',
            id="limit-on-another-role",
        ),
        pytest.param(
            [_top("top", "dataset", 5, axis="donor"), _total("top")],
            None,
            "each divided by one total of ODA over the rows "
            '"Rank the top 5 donors by the total of ODA" keeps.',
            id="limit-on-another-role-then-total",
        ),
        pytest.param(
            [
                _filter("six", "dataset", EntityIn(names=_SIX, role="recipient")),
                _top("top", "six", 3, among=RankAmong.NAMED),
            ],
            None,
            "each divided by one total of ODA over the rows "
            '"Rank the top 3 recipients by the total of ODA" keeps.',
            id="filter-then-limit",
        ),
        pytest.param(
            [
                _filter(
                    "uk_to_two",
                    "dataset",
                    EntityIn(names=["UK"], role="donor"),
                    EntityIn(names=["Kenya", "Ethiopia"], role="recipient"),
                ),
                _total("uk_to_two"),
            ],
            None,
            'each divided by one total of ODA over the rows "Narrow to Ethiopia, Kenya, UK" keeps.',
            id="filters-on-two-roles",
        ),
        pytest.param(
            [
                _filter("named", "dataset", EntityIn(names=["Kenya"])),
                _filter("two", "named", EntityIn(names=["Kenya", "Chad"], role="recipient")),
            ],
            None,
            'each divided by one total of ODA over the rows "Narrow to Chad, Kenya" keeps.',
            id="roleless-filter",
        ),
        pytest.param(
            [_filter("big", "dataset", ValueThreshold(op="gt", value=1e9)), _total("big")],
            None,
            'each divided by one total of ODA over the rows "Filter ODA" keeps.',
            id="value-threshold",
        ),
        pytest.param(
            [
                _filter("two", "dataset", EntityIn(names=["Kenya", "Ethiopia"], role="recipient")),
                _total("two", GroupBy.ENTITY, GroupBy.PERIOD),
            ],
            "Kenya",
            'each divided by the total for Kenya in "Total ODA by year and by entity".',
            id="side-entity-after-filter",
        ),
        pytest.param(
            [_total("dataset", GroupBy.ENTITY, GroupBy.PERIOD)],
            "UK",
            'each divided by the total for UK in "Total ODA by year and by entity".',
            id="side-entity-on-another-role",
        ),
    ],
)
def test_a_pooled_share_names_the_last_step_before_its_denominator_that_drops_rows(
    decision: DecisionFactory,
    steps: list[EvidenceRequirement],
    entity: str | None,
    ending: str,
) -> None:
    sentence = _pooled_recipient_share_sentence(decision, steps, denominator_entity=entity)

    assert sentence == f"Share of ODA per recipient, {ending}"


def _two_share_plan(
    decision: DecisionFactory, steps: list[EvidenceRequirement], denominators: tuple[str, str]
) -> Plan:
    """ODA per recipient pooled twice, over `denominators[0]` (`share_a`) and `denominators[1]`."""
    dataset = DatasetRequirement(
        id="dataset",
        decision_ids=["measure"],
        measure="ODA",
        entities=[Entity(name="UK", role="donor"), Entity(name="Kenya", role="recipient")],
        period=Period(start=2015, end=2024),
    )
    shares = [
        ShareOfRequirement(
            id=share_id,
            decision_ids=["measure"],
            numerator=RequirementSide(input_id="dataset"),
            denominator=RequirementSide(input_id=denominator),
            axis="recipient",
            denominator_scope=DenominatorScope.POOLED,
        )
        for share_id, denominator in zip(("share_a", "share_b"), denominators, strict=True)
    ]
    return Plan(
        question="q",
        decisions=[
            decision(fixes=DecisionFixes(denominator_scope=DenominatorScope.POOLED)),
            decision(slot="population", fixes=DecisionFixes(among=RankAmong.POPULATION)),
        ],
        evidence_requirements=[dataset, *steps, *shares],
    )


def _share_sentences(plan: Plan) -> list[str]:
    sentences = {row["id"]: row["sentence"] for row in requirement_rows(plan)}
    return [sentences["share_a"], sentences["share_b"]]


def test_a_total_over_the_top_five_and_a_total_over_all_are_told_apart(
    decision: DecisionFactory,
) -> None:
    # "What share did each of the top five get, of their combined total and of
    # all ODA?" Both totals read "Total ODA by year".
    plan = _two_share_plan(
        decision,
        [
            _top("top", "dataset", 5),
            _total("top", id_="top_total"),
            _total("dataset", id_="all_total"),
        ],
        ("top_total", "all_total"),
    )

    assert _share_sentences(plan) == [
        f"Share of ODA per recipient, each divided by one total of ODA over the rows {_TOP_5} "
        "keeps.",
        "Share of ODA per recipient, each divided by one total of ODA.",
    ]


def test_a_share_quotes_the_one_step_whose_label_it_shares_by_its_place(
    decision: DecisionFactory,
) -> None:
    # Two top-5 ranks, one over every year and one over 2020 on, read alike;
    # each share quotes its own rank's numbered label.
    plan = _two_share_plan(
        decision,
        [
            _top("top_all", "dataset", 5),
            _filter("recent", "dataset", PeriodThreshold(op="gte", year=2020)),
            _top("top_recent", "recent", 5),
        ],
        ("top_all", "top_recent"),
    )

    assert _share_sentences(plan) == [
        "Share of ODA per recipient, each divided by one total of ODA over the rows "
        '"Rank the top 5 recipients by the total of ODA (1)" keeps.',
        "Share of ODA per recipient, each divided by one total of ODA over the rows "
        '"Rank the top 5 recipients by the total of ODA (2)" keeps.',
    ]


# --- A requirement covering every member of a party ---------------------------


_EVERY_COUNTRY = DatasetRequirement(
    id="health_spending",
    decision_ids=["measure"],
    measure="health spending per person",
    across=("country",),
    period=Period(start=2022, end=2022),
)
_COUNTRIES = OpenEntityAxis(column="observationAbout", member_count=193)
_DONORS = OpenEntityAxis(column="origin", member_count=55)
# Search's sentence for the health spending source, over every place it holds.
_HEALTH_SOURCE_SENTENCE = "Current health expenditure per capita in 193 places, 2022."


def _open_axis_rank_plan(
    decision: DecisionFactory,
    dataset: DatasetRequirement = _EVERY_COUNTRY,
    *,
    axis: str | None = None,
    reduction: Reduction | None = None,
    source: dict[str, Any] | None = None,
    level: PlaceType = PlaceType.COUNTRY,
    limit: int | None = None,
) -> Plan:
    """A compiled rank over the party `dataset` covers, its source stating `source`'s fields.

    By default "top countries by health spending per person": no place
    named, over a source that leaves its 193 places open, with a filter
    keeping `level` between the retrieval and the rank. A `dataset` that
    names its places, or covers a flow party, is ranked directly.
    """
    levelled = "country" in dataset.across
    ranks = "kept" if levelled else "health_spending"
    predicates = (PlaceLevel(level=level),)
    rank = RankRequirement(
        id="ranked",
        decision_ids=["measure"],
        input_id=ranks,
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis=axis,
        reduction=reduction,
        limit=limit,
    )
    keep = FilterRequirement(
        id="kept", decision_ids=["measure"], input_id="health_spending", predicates=predicates
    )
    draft = Plan(
        question="Which countries spend the most on health per person?",
        decisions=[_population_decision(decision)],
        evidence_requirements=[dataset, *([keep] if levelled else []), rank],
    )
    keep_operation = FilterOperation(
        id="kept", requirement_id="kept", input="health_spending", predicates=predicates
    )
    translation = Translation(
        operations=[
            RetrieveOperation(
                id="health_spending", requirement_id="health_spending", spec_id=_SPEC_A
            ),
            *([keep_operation] if levelled else []),
            RankOperation(
                id="ranked",
                requirement_id="ranked",
                input=ranks,
                direction=RankDirection.HIGHEST_FIRST,
                among=RankAmong.POPULATION,
                axis=axis,
                reduction=reduction,
                limit=limit,
            ),
        ],
        terminal=["ranked"],
    )
    keep_step = WorkflowStep(
        id="kept",
        operation_id="kept",
        tool=ToolId(server="analysis-mcp", name="filter"),
        bindings={"frame_ref": Binding(from_step="health_spending", output=ArtifactKind.FRAME_REF)},
        input_schema_hash="hash",
    )
    workflow = Workflow(
        steps=[
            WorkflowStep(
                id="health_spending",
                operation_id="health_spending",
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": _SPEC_A},
                input_schema_hash="hash",
            ),
            *([keep_step] if levelled else []),
            WorkflowStep(
                id="ranked",
                operation_id="ranked",
                tool=ToolId(server="analysis-mcp", name="rank"),
                bindings={"frame_ref": Binding(from_step=ranks, output=ArtifactKind.FRAME_REF)},
                input_schema_hash="hash",
            ),
        ],
        sources=[
            SelectedSource(
                spec_id=_SPEC_A,
                sentence=_HEALTH_SOURCE_SENTENCE,
                has_data=True,
                **({"open_entity_axes": (_COUNTRIES,)} if source is None else source),
            )
        ],
        evidence_bindings=[
            EvidenceBinding(
                requirement_id="health_spending", step_id="health_spending", spec_id=_SPEC_A
            )
        ],
        terminal_steps=["ranked"],
    )
    return draft.with_translation(translation=translation, workflow=workflow)


def test_an_open_axis_rank_states_the_party_it_covers_on_every_row(
    decision: DecisionFactory,
) -> None:
    plan = _open_axis_rank_plan(decision)

    labels = [row["label"] for row in step_rows(plan)]
    sentences = {row["id"]: row["sentence"] for row in requirement_rows(plan)}

    # The retrieval names no place, so it reads as every place with data. The
    # rank states no axis, so its word is the level the filter keeps, and it
    # states no count: the source's 193 counts every level of place.
    assert labels == [
        "Fetch health spending per person for every place with data",
        "Keep countries",
        "Rank countries by health spending per person",
    ]
    assert sentences == {
        "health_spending": _HEALTH_SOURCE_SENTENCE,
        "kept": "Keeps rows where the place is a country.",
        "ranked": (
            "Ranks by health spending per person: ranks every country the source holds; "
            "the window is collapsed into one ranking."
        ),
    }


@pytest.mark.parametrize(
    ("level", "label", "kept", "ranked"),
    [
        (PlaceType.CONTINENT, "Keep continents", "a continent", "the top 5 continents"),
        (PlaceType.CITY, "Keep cities", "a city", "the top 5 cities"),
        (
            PlaceType.ADMINISTRATIVE_AREA_1,
            "Keep first-level subdivisions",
            "a first-level subdivision",
            "the top 5 first-level subdivisions",
        ),
    ],
)
def test_the_card_names_the_level_of_place_a_filter_keeps(
    decision: DecisionFactory, level: PlaceType, label: str, kept: str, ranked: str
) -> None:
    plan = _open_axis_rank_plan(decision, level=level, limit=5)

    labels = [row["label"] for row in step_rows(plan)]
    sentences = {row["id"]: row["sentence"] for row in requirement_rows(plan)}

    assert labels[1:] == [label, f"Rank {ranked} by health spending per person"]
    assert sentences["kept"] == f"Keeps rows where the place is {kept}."
    assert f"keeps {ranked}" in sentences["ranked"]


def test_a_pooled_share_names_the_level_its_denominator_keeps(
    decision: DecisionFactory,
) -> None:
    """Nigeria's share of the GDP of every country: the denominator's retrieval names
    no place, so the card's own "Keep countries" step sets its population, and the
    share's sentence names the countries, as the rank's does."""
    nigeria = DatasetRequirement(
        id="nigeria",
        decision_ids=["measure"],
        measure="GDP",
        entities=[Entity(name="Nigeria", role="country")],
        period=Period(start=2023, end=2023),
    )
    every_place = DatasetRequirement(
        id="gdp",
        decision_ids=["measure"],
        measure="GDP",
        across=("country",),
        period=Period(start=2023, end=2023),
    )
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="nigeria"),
        denominator=RequirementSide(input_id="world"),
    )
    plan = Plan(
        question="What share of the world's GDP is Nigeria's?",
        decisions=[decision()],
        evidence_requirements=[
            nigeria,
            every_place,
            _filter("countries", "gdp", PlaceLevel(level=PlaceType.COUNTRY)),
            _total("countries", id_="world"),
            share,
        ],
    )

    sentences = {row["id"]: row["sentence"] for row in requirement_rows(plan)}

    assert sentences["share"] == (
        "Share of GDP as one figure overall, divided by GDP (every country)."
    )


def test_an_open_axis_rank_states_how_many_donors_the_source_holds(
    decision: DecisionFactory,
) -> None:
    # "Top donors of education aid to Kenya": the source binds Kenya and
    # leaves its 55 donors open on `origin`, and the card states the 55.
    dataset = DatasetRequirement(
        id="health_spending",
        decision_ids=["measure"],
        measure="education aid",
        entities=[Entity(name="Kenya", role="recipient")],
        across=("donors",),
        period=Period(start=2015, end=2023),
    )
    kenya = ResolvedEntity(role="destination", ref="country/KEN", label="Kenya")
    plan = _open_axis_rank_plan(
        decision,
        dataset,
        axis="donors",
        reduction=Reduction.SUM,
        source={
            "entities": (kenya,),
            "open_entity_axes": (_DONORS,),
            "measure_label": "gross disbursements (education)",
        },
    )

    labels = [row["label"] for row in step_rows(plan)]
    sentences = {row["id"]: row["sentence"] for row in requirement_rows(plan)}

    # The retrieval's label names the source's measure and its 55 donors, not
    # the planner's "education aid for every donor".
    assert labels[0] == "Fetch gross disbursements (education) for Kenya and all 55 donors"
    assert sentences["ranked"] == (
        "Ranks by the total of education aid: ranks all 55 donors the source holds; "
        "the window is collapsed into one ranking."
    )


def test_an_open_axis_rank_states_a_count_in_the_thousands_with_a_separator(
    decision: DecisionFactory,
) -> None:
    dataset = DatasetRequirement(
        id="health_spending",
        decision_ids=["measure"],
        measure="education aid",
        entities=[Entity(name="Kenya", role="recipient")],
        across=("donor",),
        period=Period(start=2022, end=2022),
    )
    kenya = ResolvedEntity(role="destination", ref="country/KEN", label="Kenya")
    donors = OpenEntityAxis(column="origin", member_count=1234)
    plan = _open_axis_rank_plan(
        decision,
        dataset,
        axis="donor",
        reduction=Reduction.SUM,
        source={"entities": (kenya,), "open_entity_axes": (donors,)},
    )

    sentences = {row["id"]: row["sentence"] for row in requirement_rows(plan)}

    assert "ranks all 1,234 donors the source holds" in sentences["ranked"]


def test_a_rank_over_one_of_two_open_parties_states_that_partys_count(
    decision: DecisionFactory,
) -> None:
    # "Every donor by every recipient": the rank's axis is the donors, so the
    # card states the 55 the source holds on `origin`, not the 140 recipients.
    dataset = DatasetRequirement(
        id="health_spending",
        decision_ids=["measure"],
        measure="education aid",
        across=("recipient", "donor"),
        period=Period(start=2022, end=2022),
    )
    recipients = OpenEntityAxis(column="destination", member_count=140)
    plan = _open_axis_rank_plan(
        decision,
        dataset,
        axis="donor",
        reduction=Reduction.SUM,
        source={"open_entity_axes": (recipients, _DONORS)},
    )

    labels = [row["label"] for row in step_rows(plan)]
    sentences = {row["id"]: row["sentence"] for row in requirement_rows(plan)}

    # A source stating no measure label leaves the label the requirement's measure.
    assert labels[0] == "Fetch education aid for all 140 recipients and all 55 donors"
    assert "ranks all 55 donors the source holds" in sentences["ranked"]


def test_a_rank_whose_source_does_not_answer_its_party_states_no_count(
    decision: DecisionFactory,
) -> None:
    # The card states a count only where `open_axes_for` matches the party to
    # an open axis of the selected source. A source leaving both its donors
    # and its recipients open holds no one population of countries.
    recipients = OpenEntityAxis(column="destination", member_count=140)
    plan = _open_axis_rank_plan(decision, source={"open_entity_axes": (recipients, _DONORS)})

    sentences = {row["id"]: row["sentence"] for row in requirement_rows(plan)}

    assert "ranks every country the source holds" in sentences["ranked"]
    assert "55" not in sentences["ranked"]


def test_a_named_entity_beside_an_open_party_reads_as_both(decision: DecisionFactory) -> None:
    # "Top donors of education aid to Kenya": Kenya is named, every donor is
    # not. The plan is a draft with no source selected, so no count is known.
    dataset = DatasetRequirement(
        id="aid_to_kenya",
        decision_ids=["measure"],
        measure="education aid",
        entities=[Entity(name="Kenya", role="recipient")],
        across=("donor",),
        period=Period(start=2020, end=2022),
    )
    rank = RankRequirement(
        id="ranked",
        decision_ids=["measure"],
        input_id="aid_to_kenya",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="donor",
        reduction=Reduction.SUM,
    )
    plan = Plan(
        question="Who are the top donors of education aid to Kenya?",
        decisions=[_population_decision(decision)],
        evidence_requirements=[dataset, rank],
    )

    sentences = {row["id"]: row["sentence"] for row in requirement_rows(plan)}

    assert sentences["aid_to_kenya"] == (
        "Retrieves education aid for Kenya and every donor, 2020-2022."
    )
    assert "ranks every donor the source holds" in sentences["ranked"]


@pytest.mark.parametrize("axis", ["countries", None])
def test_a_plural_party_reads_in_the_singular_after_every_and_for_one_kept(
    decision: DecisionFactory, axis: str | None
) -> None:
    # The planner wrote the party in the plural, and the rank's axis repeats
    # it or states nothing.
    dataset = DatasetRequirement(
        id="health_spending",
        decision_ids=["measure"],
        measure="health spending per person",
        across=("countries",),
        period=Period(start=2022, end=2022),
    )
    keep = FilterRequirement(
        id="kept",
        decision_ids=["measure"],
        input_id="health_spending",
        predicates=(PlaceLevel(level=PlaceType.COUNTRY),),
    )
    rank = RankRequirement(
        id="ranked",
        decision_ids=["measure"],
        input_id="kept",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis=axis,
        limit=1,
    )
    plan = Plan(
        question="Which country spends the most on health per person?",
        decisions=[_population_decision(decision)],
        evidence_requirements=[dataset, keep, rank],
    )

    sentences = {row["id"]: row["sentence"] for row in requirement_rows(plan)}

    assert sentences == {
        "health_spending": (
            "Retrieves health spending per person for every place with data, 2022-2022."
        ),
        "kept": "Keeps rows where the place is a country.",
        "ranked": (
            "Ranks by health spending per person: ranks every country the source holds; "
            "the window is collapsed into one ranking; keeps the top country, more when 1st "
            "place is shared."
        ),
    }


def test_a_rank_over_a_named_role_keeps_the_named_entities_as_its_focus(
    decision: DecisionFactory,
) -> None:
    # The same dataset ranked over Kenya's own role: the axis is not `across`.
    dataset = DatasetRequirement(
        id="aid_to_kenya",
        decision_ids=["measure"],
        measure="education aid",
        entities=[Entity(name="Kenya", role="recipient")],
        across=("donor",),
        period=Period(start=2020, end=2022),
    )
    rank = RankRequirement(
        id="ranked",
        decision_ids=["measure"],
        input_id="aid_to_kenya",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="recipient",
        reduction=Reduction.SUM,
    )
    plan = Plan(
        question="q",
        decisions=[_population_decision(decision)],
        evidence_requirements=[dataset, rank],
    )

    sentences = {row["id"]: row["sentence"] for row in requirement_rows(plan)}

    assert "with the named entities as the focus" in sentences["ranked"]


def test_the_args_on_a_step_are_the_workflow_steps_args_dict_unchanged(
    decision: DecisionFactory,
) -> None:
    plan = _rank_chain_plan(decision)

    rows = step_rows(plan)

    growth_row = next(row for row in rows if row["id"] == "growth")
    assert growth_row["args"] == {"window_start": 2016, "window_end": 2020, "change": "absolute"}


@pytest.mark.parametrize(
    ("source_fields", "clause"),
    [
        (
            {
                "served_count": None,
                "stat_var_member_count": 1554,
                "stat_var_reading": StatVarReading.SUM,
                "stat_var_reading_set_by": ReadingSetBy.RULE,
            },
            "summed over its 1,554 indicators by default",
        ),
        (
            {
                "served_count": 3,
                "stat_var_reading": StatVarReading.SUM,
                "stat_var_reading_set_by": ReadingSetBy.PLAN,
            },
            "summed over its 3 indicators",
        ),
        (
            {
                "served_count": 2,
                "stat_var_reading": StatVarReading.SEPARATE,
                "stat_var_reading_set_by": ReadingSetBy.PLAN,
            },
            "kept as 2 separate indicators",
        ),
        (
            {
                "served_count": None,
                "stat_var_member_count": 1,
                "stat_var_reading": StatVarReading.SUM,
                "stat_var_reading_set_by": ReadingSetBy.RULE,
            },
            "summed over its indicators by default",
        ),
        (
            {
                "served_count": None,
                "stat_var_reading": StatVarReading.SEPARATE,
                "stat_var_reading_set_by": ReadingSetBy.PLAN,
            },
            "kept as separate indicators",
        ),
        (
            {
                "served_count": None,
                "stat_var_member_count": 27,
                "stat_var_member_count_exact": False,
                "stat_var_reading": StatVarReading.SEPARATE,
                "stat_var_reading_set_by": ReadingSetBy.RULE,
            },
            "kept as separate indicators by default",
        ),
    ],
    ids=[
        "summed-by-the-rule",
        "summed-by-the-plan",
        "separate-by-the-plan",
        "one-confirmed-summed",
        "uncounted-separate",
        "floor-count-separate",
    ],
)
def test_the_dataset_sentence_states_how_a_family_is_read(
    decision: DecisionFactory, source_fields: dict[str, Any], clause: str
) -> None:
    plan = _one_requirement_compiled_plan(decision, definition=None, **source_fields)

    rows = requirement_rows(plan)

    dataset_row = next(row for row in rows if row["id"] == "dataset_a")
    assert dataset_row["sentence"] == (f"{_SOURCE_SENTENCE}. {clause[0].upper()}{clause[1:]}.")


@pytest.mark.parametrize(
    ("level", "rolled_up_to"),
    [(0, "the top level of its hierarchy"), (1, "level 1 below the top of its hierarchy")],
)
def test_a_breakdown_by_sector_reads_on_the_retrieval_the_total_and_the_rank(
    decision: DecisionFactory, level: int, rolled_up_to: str
) -> None:
    """ "Which sectors received the most aid in Kenya?": the card says the retrieval
    keeps the sector, the total is taken per sector, and the rank orders sectors."""
    aid = DatasetRequirement(
        id="aid",
        decision_ids=["measure"],
        measure="aid",
        entities=[Entity(name="Kenya", role="recipient")],
        across=("donor",),
        period=Period(start=2024, end=2024),
        stat_vars=StatVarReading.SEPARATE,
        breakdown=Breakdown(dimension=BreakdownDimension.SECTOR, level=level),
    )
    plan = Plan(
        question="q",
        decisions=[decision(fixes=DecisionFixes(among=RankAmong.POPULATION))],
        evidence_requirements=[
            aid,
            _total("aid", GroupBy.PERIOD, GroupBy.BREAKDOWN),
            RankRequirement(
                id="ranked",
                decision_ids=["measure"],
                input_id="total",
                direction=RankDirection.HIGHEST_FIRST,
                among=RankAmong.POPULATION,
                axis="sector",
                limit=10,
            ),
        ],
    )

    sentences = {row["id"]: row["sentence"] for row in requirement_rows(plan)}

    assert sentences["aid"] == (
        f"Retrieves aid for Kenya and every donor, 2024-2024, broken down by sector, each code "
        f"rolled up to {rolled_up_to}."
    )
    assert sentences["total"] == "The total of aid, grouped by year and by sector."
    one_figure = plan.model_copy(
        update={"evidence_requirements": [aid, _total("aid").model_copy(update={"group_by": ()})]}
    )
    assert {row["id"]: row["sentence"] for row in requirement_rows(one_figure)}["total"] == (
        "The total of aid, as one figure over all the years, parties and codes read."
    )
    assert sentences["ranked"].startswith(
        "Ranks by the total of aid: ranks every sector the source holds; "
    )
    assert _label(plan.requirement("ranked"), plan) == "Rank the top 10 sectors by the total of aid"


# A source naming health (120) and population (130), which share social
# infrastructure (100) at the top level.
_HEALTH_AND_POPULATION: dict[str, Any] = {
    "identity": {
        "slots": [
            {
                "property": "crsPurpose",
                "binding": {"state": "set", "refs": ["CRSPurposeCode120", "CRSPurposeCode130"]},
            }
        ]
    },
    "code_paths": {
        "CRSPurposeCode120": ("CRSPurposeCode100", "CRSPurposeCode120"),
        "CRSPurposeCode130": ("CRSPurposeCode100", "CRSPurposeCode130"),
    },
}


@pytest.mark.parametrize(
    ("named", "rolled_up_to"),
    [
        ({}, "the top level of its hierarchy"),
        (_HEALTH_AND_POPULATION, "level 1 below the top of its hierarchy"),
    ],
    ids=["no-sector-named", "health-and-population"],
)
def test_a_breakdown_states_the_level_it_reads_and_no_reading_of_the_family(
    decision: DecisionFactory, named: dict[str, Any], rolled_up_to: str
) -> None:
    """The plan states level 0; a source naming two sectors that share their top-level
    parent reads them where they part, and the card says so."""
    plan = _one_requirement_compiled_plan(
        decision,
        definition=None,
        breakdown=Breakdown(dimension=BreakdownDimension.SECTOR, level=0),
        served_count=None,
        stat_var_member_count=705,
        stat_var_reading=StatVarReading.SEPARATE,
        stat_var_reading_set_by=ReadingSetBy.PLAN,
        **named,
    )

    (dataset_row,) = [row for row in requirement_rows(plan) if row["id"] == "dataset_a"]

    assert dataset_row["sentence"] == (
        f"{_SOURCE_SENTENCE}. Broken down by sector, each code rolled up to {rolled_up_to}."
    )


def test_a_period_that_follows_the_newest_data_says_so_beside_its_years(
    decision: DecisionFactory,
) -> None:
    """The years stay as the plan read them, and each sentence says which of
    them move: the whole span, one end of it, or a threshold's year."""
    latest = RelativePeriod(offset=0)
    plan = Plan(
        question="How has Kenya's debt service grown?",
        decisions=[decision()],
        evidence_requirements=[
            DatasetRequirement(
                id="debt",
                decision_ids=["measure"],
                measure="external debt service paid",
                entities=[Entity(name="Kenya", role="debtor")],
                period=Period(start=2015, end=2024),
                period_end_open=True,
            ),
            GrowthRequirement(
                id="last_five",
                input_id="debt",
                window_start=2019,
                window_end=2024,
                rolling=RollingWindow(window_start=RelativePeriod(offset=-5), window_end=latest),
            ),
            GrowthRequirement(
                id="since",
                input_id="debt",
                window_start=2015,
                window_end=2024,
                rolling=RollingWindow(window_end=latest),
            ),
            GrowthRequirement(id="fixed", input_id="debt", window_start=2015, window_end=2020),
            FilterRequirement(
                id="newest",
                input_id="debt",
                predicates=(PeriodThreshold(op="eq", year=2024, rolling=latest),),
            ),
        ],
        # The dataset starts where the plan does, so reading it whole splits nothing.
        first_years=[FirstYear(requirement_id="debt", year=2015)],
    )

    sentences = {str(row["id"]): row["sentence"] for row in requirement_rows(plan)}

    measure = "external debt service paid"
    assert sentences == {
        "debt": f"Retrieves {measure} for Kenya, 2015-2024, following the newest year.",
        "last_five": (
            f"The change in {measure} from 2019 to 2024, following the newest year, read as "
            "a percentage change."
        ),
        "since": (
            f"The change in {measure} from 2015 to 2024, the end following the newest year, "
            "read as a percentage change."
        ),
        "fixed": f"The change in {measure} from 2015 to 2020, read as a percentage change.",
        "newest": "Keeps rows where the year is exactly 2024, following the newest year.",
    }
    assert (
        _label(plan.requirement("last_five"), plan)
        == f"Growth in {measure}, 2019-2024, following the newest year"
    )


# --- A plan whose dataset is read from its first year -------------------------------


def test_the_card_shows_one_row_per_step_the_model_wrote(decision: DecisionFactory) -> None:
    """`debt_history` is code's copy of `debt` for the rolling growth. The card names
    `debt` once, and the growth reads `debt`'s source."""
    plan = stored_split_plan(decision)

    steps = step_rows(plan)

    assert [row["id"] for row in steps] == ["debt", "total", "last_five"]
    assert [row["id"] for row in requirement_rows(plan)] == ["debt", "total", "last_five"]
    assert [row["id"] for row in operation_rows(plan)] == ["debt", "total", "last_five"]
    assert "debt_history" not in repr([steps, requirement_rows(plan), operation_rows(plan)])


def test_the_card_states_a_series_answer_over_the_years_the_plan_states(
    decision: DecisionFactory,
) -> None:
    """The copy reads from 2000. The total beside it keeps 2015-2024."""
    plan = stored_split_plan(decision)

    total = next(row for row in step_rows(plan) if row["id"] == "total")

    assert str(total["description"]).endswith("Result: one figure per year, 2015-2024.")


def test_the_shown_plan_is_the_authored_plan_with_the_compiled_parts_folded(
    decision: DecisionFactory,
) -> None:
    """Sources and bindings of the copy go together, and a warning on the copy names its origin once."""
    plan = stored_split_plan(decision)

    shown = shown_plan(plan)

    assert shown == shown_plan(shown)
    assert (shown.rolling_copies, shown.first_years) == ((), ())
    growth = shown.requirement("last_five")
    assert isinstance(growth, GrowthRequirement)
    assert growth.input_id == "debt"
    assert shown.workflow is not None
    assert [(b.requirement_id, b.spec_id) for b in shown.workflow.evidence_bindings] == [
        ("debt", SPEC_DEBT)
    ]
    assert [source.spec_id for source in shown.workflow.sources] == [SPEC_DEBT]
    assert [issue.requirement_id for issue in shown.workflow.warnings] == ["debt"]


def test_a_warning_on_a_copy_stands_in_for_its_origin_s_of_that_code(
    decision: DecisionFactory,
) -> None:
    """`debt` and its copy both read short of 2015-2024, and the copy says its
    retrieval asks for every year from 2000. The card shows one gap for
    `debt`, the copy's, in the words of the plan the model wrote, beside the
    origin's own warning of another code."""
    plan = stored_split_plan(decision)
    assert plan.workflow is not None

    def issue(code: EvidenceIssueCode, requirement_id: str, detail: str) -> EvidenceIssue:
        return EvidenceIssue(
            code=code, severity=Severity.WARNING, requirement_id=requirement_id, detail=detail
        )

    copy_gap = (
        "requirement 'debt_history': the source covers 2000-2023, the requirement asks for "
        "2015-2024, and its retrieval requests every year from 2000"
    )
    warned = plan.model_copy(
        update={
            "workflow": plan.workflow.model_copy(
                update={
                    "warnings": (
                        issue(
                            EvidenceIssueCode.WINDOW_SHORT,
                            "debt",
                            "the source covers 2016-2023, the requirement asks for 2015-2024",
                        ),
                        issue(EvidenceIssueCode.UNIT_UNVERIFIED, "debt", "no unit stated"),
                        issue(EvidenceIssueCode.WINDOW_SHORT, "debt_history", copy_gap),
                    )
                }
            )
        }
    )

    shown = shown_plan(warned).workflow

    assert shown is not None
    assert [(w.requirement_id, w.code, w.detail) for w in shown.warnings] == [
        ("debt", EvidenceIssueCode.UNIT_UNVERIFIED, "no unit stated"),
        (
            "debt",
            EvidenceIssueCode.WINDOW_SHORT,
            copy_gap.replace("'debt_history'", "'debt'"),
        ),
    ]


@pytest.mark.parametrize(
    ("axes", "measure"),
    [
        (
            ("sector", "measure_basis"),
            "net disbursements (official development assistance) for sector:120+130 label",
        ),
        (("measure_basis",), "net disbursements (official development assistance)"),
        ((), "gross ODA disbursements to health"),
    ],
    ids=["sector-and-basis", "basis", "no-scope-choice"],
)
def test_a_step_reading_a_scoped_dataset_names_its_measure_as_search_counts_it(
    decision: DecisionFactory,
    axes: tuple[typing.Literal["sector", "measure_basis"], ...],
    measure: str,
) -> None:
    """The planner wrote "gross ODA disbursements to health" for the pin it
    chose; a switch on the plan card re-pinned the dataset on net
    disbursements. A step reading a dataset whose basis or sector a scope
    decision settles names the source's measure and the sector chosen, so its
    label and sentence follow the pin; with no such decision they keep the
    planner's words."""
    compiled = _retrieve_then_aggregate_plan(decision)
    assert compiled.workflow is not None
    scoped = [_scope_decision(_SPEC_A, axis) for axis in axes]
    dataset, totals = compiled.evidence_requirements
    assert isinstance(dataset, DatasetRequirement)
    plan = compiled.model_copy(
        update={
            "decisions": [*compiled.decisions, *scoped],
            "evidence_requirements": [
                dataset.model_copy(
                    update={
                        "measure": "gross ODA disbursements to health",
                        "decision_ids": (*dataset.decision_ids, *(d.slot for d in scoped)),
                    }
                ),
                totals,
            ],
            "workflow": compiled.workflow.model_copy(
                update={
                    "sources": [
                        source.model_copy(
                            update={
                                "measure_label": "net disbursements (official development assistance)"
                            }
                        )
                        for source in compiled.workflow.sources
                    ]
                }
            ),
        }
    )

    labels = {row["id"]: row["label"] for row in step_rows(plan)}
    sentences = {row["id"]: row["sentence"] for row in requirement_rows(plan)}

    assert labels["annual_totals"] == f"Total {measure} by year"
    assert sentences["annual_totals"] == f"The total of {measure}, grouped by year."
