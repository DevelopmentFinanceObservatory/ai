"""Tests for `harness.plan_display`: turning a `Plan` into wire-ready rows.

Builds plans directly from `planner.plan`/`planner.operations`/`planner.workflow`
rather than through the planner or the compiler -- this module has no
dependency on either, and neither do these tests.
"""

from __future__ import annotations

import enum
import json
import os
import subprocess
import sys
import typing
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from compiler.sources import source_from_spec_block
from harness import plan_display
from harness.decision_pick import DecisionPick, decision_pick
from harness.plan_display import (
    _card_label,
    _concepts,
    _label,
    card_choice_text,
    card_decision_text,
    card_questions,
    card_rows,
    card_summary,
    card_title,
    choice_id,
    chosen_scope_decision,
    decision_choice_id,
    decision_option_id,
    listed_together,
    operation_rows,
    pinned_choice_id,
    plan_words,
    requirement_rows,
    share_wholes,
    shown_decisions,
    shown_plan,
    step_rows,
    switchable_decision,
)
from harness.scripted_plan import scripted_plan
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
    WholeOf,
    frame_shapes,
    scope_choice_slot,
    search_pin_slot,
    split_of,
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
from submission.service import SourceReadings, _card_names
from translator.mechanical import build_translation

from .share_plan import (
    ALL_DONORS,
    DAC_DONORS,
    NAMED_DONORS,
    UGANDA,
    US_DONOR,
    USA_RECEIVING,
    us_share_of_kenya_oda,
    us_share_per_recipient,
)
from .split_plan import SPEC_DEBT, SPEC_DEBT_HISTORY, split_draft, stored_split_plan

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
    Each requirement cites the row standing for its own decision. A row's id
    is the first requirement pinning one of its sources and the axis,
    whichever decision of the row is listed, and names that decision."""
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
    assert [choice_id(plan, d) for d in (sector_a, sector_b, basis_a, basis_b)] == [
        "aid_a:sector",
        "aid_a:sector",
        "aid_a:measure_basis",
        "aid_b:measure_basis",
    ]
    assert chosen_scope_decision(plan, "aid_a:sector") == sector_b
    # A card sent before these ids names the listed row by Search's own id.
    assert chosen_scope_decision(plan, f"{_SPEC_B}:sector") == sector_b
    assert chosen_scope_decision(plan, f"{_SPEC_A}:sector") is None


@pytest.mark.parametrize("merged", [False, True], ids=["apart", "merged"])
def test_a_share_sides_basis_rows_keep_their_ids_as_they_merge_and_split(
    decision: DecisionFactory, merged: bool
) -> None:
    """The numerator's and the denominator's basis decisions on different
    options are two rows, each with its own requirement's id. Switched to the
    same option they become one row, which takes the first requirement's id
    in plan order and stands for both, so a switch of it moves both; apart
    again, each row has its own id back."""
    numerator = _scope_decision(_SPEC_A, "measure_basis")
    denominator = _scope_decision(_SPEC_B, "measure_basis")
    if not merged:
        denominator = denominator.model_copy(update={"chosen": "measure_basis:net label"})

    def dataset(id: str, pin: str, cited: Decision) -> DatasetRequirement:
        return DatasetRequirement(
            id=id,
            decision_ids=["measure", cited.slot],
            measure="aid",
            entities=[Entity(name="Kenya", role="recipient")],
            period=Period(start=2015, end=2024),
            source_spec_id=pin,
        )

    plan = Plan(
        question="US health aid as a share of US aid",
        decisions=[decision(), numerator, denominator],
        evidence_requirements=[
            dataset("health_aid", _SPEC_A, numerator),
            dataset("total_aid", _SPEC_B, denominator),
        ],
    )

    rows = [d for d in shown_decisions(plan) if d.scope is not None]
    ids = [choice_id(plan, row) for row in rows]
    if merged:
        assert ids == ["health_aid:measure_basis"]
        (row,) = rows
        assert chosen_scope_decision(plan, "health_aid:measure_basis") == row
        assert listed_together(plan, row.slot) == (numerator, denominator)
    else:
        assert ids == ["health_aid:measure_basis", "total_aid:measure_basis"]
        assert [chosen_scope_decision(plan, id) for id in ids] == [numerator, denominator]


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


def test_steps_that_read_alike_are_numbered_in_the_order_the_plan_states_them(
    decision: DecisionFactory,
) -> None:
    # The plan states `first_totals` first, so it takes "(1)", though the workflow
    # runs `second_totals` first and the card lists it first.
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
        ("second_totals", "Total external debt service paid by year (2)"),
        ("first_totals", "Total external debt service paid by year (1)"),
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


def _among_decision(decision: DecisionFactory) -> DecisionFactory:
    """`decision` with the options a planner writes as `among`'s own values."""

    def _decision(**overrides: Any) -> Decision:
        written = {
            "description": "Whether to rank across all reporting donors or a restricted set.",
            "options": ["population", "named"],
            "chosen": "population",
        }
        return decision(**(written | overrides))

    return _decision


def _donor_rank_plan(decision: DecisionFactory) -> Plan:
    """The top donors to Kenya, ranked among every donor DAC2a holds, by a
    decision whose options the planner wrote as `among`'s values."""
    dataset = DatasetRequirement(
        id="health_spending",
        decision_ids=["measure"],
        measure="education aid",
        entities=[Entity(name="Kenya", role="recipient")],
        across=("donor",),
        period=Period(start=2022, end=2022),
    )
    kenya = ResolvedEntity(role="destination", ref="country/KEN", label="Kenya")
    donors = OpenEntityAxis(column="origin", member_count=32)
    return _open_axis_rank_plan(
        _among_decision(decision),
        dataset,
        axis="donor",
        reduction=Reduction.SUM,
        source={"entities": (kenya,), "open_entity_axes": (donors,), "published_tables": "DAC2a"},
    )


def test_a_rank_option_written_as_its_value_reads_as_who_the_rank_compares(
    decision: DecisionFactory,
) -> None:
    # The reported card: the planner wrote `among`'s values as the options and
    # names no donor. The card names the donors DAC2a holds, and a named rank,
    # which compiles only behind a filter naming its entities, as donors picked
    # by name. A chosen text naming a value the decision does not fix keeps
    # the planner's words.
    plan = _donor_rank_plan(decision)
    (among,) = plan.decisions
    contradicted = among.model_copy(update={"chosen": "named"})

    shown = [
        card_choice_text(o, plan, options=among.options, decision=among) for o in among.options
    ]

    assert shown == ["all 32 donors in DAC2a", "only donors picked by name"]
    assert card_choice_text("named", plan, options=(), decision=contradicted) == "named"


def _two_donor_ranks(decision: DecisionFactory, among: RankAmong, *kept: list[str]) -> Plan:
    """The top donors to Kenya in ODA and in private flows, both ranks citing
    one `among` decision: over every donor, or, given `kept`, over the donors
    each retrieval names, kept by a filter."""
    requirements: list[EvidenceRequirement] = []
    for index, flow in enumerate(("oda", "private")):
        requirements.append(
            DatasetRequirement(
                id=flow,
                decision_ids=["measure"],
                measure=flow,
                entities=[
                    Entity(name="Kenya", role="recipient"),
                    *(Entity(name=name, role="donor") for name in (kept[index] if kept else ())),
                ],
                across=() if kept else ("donor",),
                period=Period(start=2022, end=2022),
            )
        )
        ranked = flow
        if kept:
            ranked = f"{flow}_kept"
            requirements.append(_filter(ranked, flow, EntityIn(names=kept[index], role="donor")))
        requirements.append(
            RankRequirement(
                id=f"top_{flow}",
                decision_ids=["measure"],
                input_id=ranked,
                direction=RankDirection.HIGHEST_FIRST,
                among=among,
                axis="donor",
                reduction=Reduction.SUM,
            )
        )
    options = ["population", "named"] if among is RankAmong.POPULATION else ["named", "population"]
    return Plan(
        question="q",
        decisions=[decision(options=options, chosen=options[0], fixes=DecisionFixes(among=among))],
        evidence_requirements=requirements,
    )


def _compiled_donor_ranks(decision: DecisionFactory, sources: dict[str, tuple[str, int]]) -> Plan:
    """`_two_donor_ranks` over every donor, compiled: each flow retrieved from
    the source `sources` gives it, by its published tables and donor count."""
    draft = _two_donor_ranks(decision, RankAmong.POPULATION)
    specs = {"oda": _SPEC_A, "private": _SPEC_B}
    operations: list[Operation] = []
    steps: list[WorkflowStep] = []
    for flow, spec in specs.items():
        rank = typing.cast(RankRequirement, draft.requirement(f"top_{flow}"))
        operations += [
            RetrieveOperation(id=flow, requirement_id=flow, spec_id=spec),
            RankOperation(
                id=rank.id,
                requirement_id=rank.id,
                input=flow,
                direction=rank.direction,
                among=rank.among,
                axis=rank.axis,
                reduction=rank.reduction,
            ),
        ]
        steps += [
            WorkflowStep(
                id=flow,
                operation_id=flow,
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": spec},
                input_schema_hash="hash",
            ),
            WorkflowStep(
                id=rank.id,
                operation_id=rank.id,
                tool=ToolId(server="analysis-mcp", name="rank"),
                bindings={"frame_ref": Binding(from_step=flow, output=ArtifactKind.FRAME_REF)},
                input_schema_hash="hash",
            ),
        ]
    terminal = ["top_oda", "top_private"]
    workflow = Workflow(
        steps=steps,
        sources=[
            SelectedSource(
                spec_id=spec,
                sentence="s",
                has_data=True,
                published_tables=sources[flow][0],
                open_entity_axes=(OpenEntityAxis(column="origin", member_count=sources[flow][1]),),
            )
            for flow, spec in specs.items()
        ],
        evidence_bindings=[
            EvidenceBinding(requirement_id=flow, step_id=flow, spec_id=spec)
            for flow, spec in specs.items()
        ],
        terminal_steps=terminal,
    )
    return draft.with_translation(
        translation=Translation(operations=operations, terminal=terminal), workflow=workflow
    )


@pytest.mark.parametrize(
    ("sources", "population"),
    [
        pytest.param(
            {"oda": ("DAC2a", 32), "private": ("DAC2a", 32)},
            "all 32 donors in DAC2a",
            id="one-source",
        ),
        pytest.param(
            {"oda": ("DAC2a", 32), "private": ("CRS", 45)},
            "all donors each source holds",
            id="two-sources",
        ),
    ],
)
def test_ranks_citing_one_decision_are_named_by_what_they_all_share(
    decision: DecisionFactory, sources: dict[str, tuple[str, int]], population: str
) -> None:
    # Ranks reading different sources are not named by the first one's.
    plan = _compiled_donor_ranks(decision, sources)
    (among,) = plan.decisions

    shown = [
        card_choice_text(o, plan, options=among.options, decision=among) for o in among.options
    ]

    assert shown == [population, "only donors picked by name"]


@pytest.mark.parametrize(
    ("kept", "named"),
    [
        pytest.param(
            (["France", "Japan"], ["France", "Japan"]), "only France and Japan", id="alike"
        ),
        pytest.param(
            (["France", "Japan", "Italy", "Spain"],) * 2,
            "only the 4 donors the filter keeps",
            id="many",
        ),
        pytest.param(
            (["France", "Japan"], ["Italy"]), "only donors picked by name", id="different"
        ),
    ],
)
def test_a_named_rank_option_names_the_donors_its_filter_keeps(
    decision: DecisionFactory, kept: tuple[list[str], list[str]], named: str
) -> None:
    plan = _two_donor_ranks(decision, RankAmong.NAMED, *kept)
    (among,) = plan.decisions

    assert card_choice_text("named", plan, options=among.options, decision=among) == named


def _gni_share(decision: DecisionFactory, *denominators: str, narrowed: bool = False) -> Plan:
    """Each donor's ODA to Kenya over Kenya's GNI (and over Kenya's revenue
    for a second denominator), by a decision written as
    `denominator_scope`'s values. A `narrowed` denominator is read for Kenya
    and Uganda and kept to Kenya by a filter."""
    requirements: list[EvidenceRequirement] = [
        DatasetRequirement(
            id="oda",
            decision_ids=["measure"],
            measure="ODA",
            entities=[Entity(name="Kenya", role="recipient")],
            across=("donor",),
            period=Period(start=2022, end=2022),
        )
    ]
    for index, measure in enumerate(denominators):
        requirements += [
            DatasetRequirement(
                id=f"denominator_{index}",
                decision_ids=["measure"],
                measure=measure,
                entities=[
                    Entity(name="Kenya", role="country"),
                    *([Entity(name="Uganda", role="country")] if narrowed else []),
                ],
                period=Period(start=2022, end=2022),
            ),
            *(
                [
                    _filter(
                        f"kenya_{index}",
                        f"denominator_{index}",
                        EntityIn(names=["Kenya"], role="country"),
                    )
                ]
                if narrowed
                else []
            ),
            ShareOfRequirement(
                id=f"share_{index}",
                decision_ids=["measure"],
                numerator=RequirementSide(input_id="oda"),
                denominator=RequirementSide(
                    input_id=f"kenya_{index}" if narrowed else f"denominator_{index}"
                ),
                axis="donor",
                denominator_scope=DenominatorScope.POOLED,
            ),
        ]
    return Plan(
        question="q",
        decisions=[
            decision(
                options=["pooled", "per_axis"],
                chosen="pooled",
                fixes=DecisionFixes(denominator_scope=DenominatorScope.POOLED),
            )
        ],
        evidence_requirements=requirements,
    )


@pytest.mark.parametrize(
    ("denominators", "narrowed", "shown"),
    [
        pytest.param(
            ("GNI",),
            False,
            ["each divided by one total of GNI", "each divided by its own total of GNI"],
            id="one-share",
        ),
        pytest.param(
            ("GNI", "government revenue"),
            False,
            ["one shared total", "each its own total"],
            id="two-denominators",
        ),
        pytest.param(
            ("GNI",),
            True,
            ["one shared total", "each divided by its own total of GNI"],
            id="narrowed",
        ),
    ],
)
def test_a_share_option_written_as_its_scope_names_the_denominator(
    decision: DecisionFactory, denominators: tuple[str, ...], narrowed: bool, shown: list[str]
) -> None:
    # A pooled share over Kenya's GNI divides by GNI, not by a total over the
    # donors, and each option names that measure. Shares over different
    # denominators, and a pooled total over the rows a filter keeps, which
    # only a quoted step label names, read the value's fallback.
    plan = _gni_share(decision, *denominators, narrowed=narrowed)
    (scope,) = plan.decisions

    assert [
        card_choice_text(o, plan, options=scope.options, decision=scope) for o in scope.options
    ] == shown


def _oda_to_kenya(*kenya: Entity, across: tuple[str, ...] = ("donor",)) -> DatasetRequirement:
    """ODA from every donor (`across`) to `kenya`, the input of a total grouped by entity."""
    return DatasetRequirement(
        id="debt",
        decision_ids=["measure"],
        measure="ODA",
        entities=list(kenya),
        across=across,
        period=Period(start=2015, end=2024),
    )


@pytest.mark.parametrize(
    ("dataset", "shown"),
    [
        pytest.param(_debt(), ["by debtor", "by year"], id="one-party"),
        pytest.param(
            _oda_to_kenya(Entity(name="Kenya", role="recipient")),
            ["by recipient", "by year"],
            id="open-donors",
        ),
        pytest.param(
            _oda_to_kenya(Entity(name="Kenya")),
            ["by each place named", "by year"],
            id="roleless-place",
        ),
        pytest.param(
            _oda_to_kenya(
                Entity(name="Kenya", role="recipient"),
                Entity(name="France", role="donor"),
                across=(),
            ),
            ["by donor and recipient", "by year"],
            id="two-parties",
        ),
    ],
)
def test_a_group_by_option_written_as_entity_names_the_party(
    decision: DecisionFactory, dataset: DatasetRequirement, shown: list[str]
) -> None:
    # "entity" reads as what a total grouped by entity keeps: each party
    # named, and the places named with no role. An open `across` party is
    # summed away, so it names nothing.
    plan = Plan(
        question="q",
        decisions=[
            decision(
                options=["entity", "period"],
                chosen="entity",
                fixes=DecisionFixes(group_by=(GroupBy.ENTITY,)),
            )
        ],
        evidence_requirements=[dataset, _total("debt", GroupBy.ENTITY)],
    )
    (grouping,) = plan.decisions

    assert [
        card_choice_text(o, plan, options=grouping.options, decision=grouping)
        for o in grouping.options
    ] == shown


def _enums_in(annotation: object) -> set[type[enum.Enum]]:
    """Every enum `annotation` names, through unions, tuples and `Annotated`."""
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return {annotation}
    return {found for arg in typing.get_args(annotation) for found in _enums_in(arg)}


def test_every_value_a_decision_can_fix_has_card_words() -> None:
    # A new enum field on `DecisionFixes`, or a new value of one, fails here
    # until the card has words for it, with a fallback true in every plan
    # that never reads as the raw value.
    fixed = {
        name: kinds
        for name, field in DecisionFixes.model_fields.items()
        if (kinds := _enums_in(field.annotation))
    }
    values = {value for kind in plan_display.FIXED_ENUMS.values() for value in kind}
    fallbacks = {value: plan_display.FIXED_VALUE_WORDS[value].fallback for value in values}

    assert {name: {kind} for name, kind in plan_display.FIXED_ENUMS.items()} == fixed
    assert set(plan_display.FIXED_VALUE_WORDS) == values
    assert [
        value
        for value, words in fallbacks.items()
        if not words or words in {value.value, value.value.replace("_", " ")}
    ] == []
    for kind in plan_display.FIXED_ENUMS.values():
        assert len({fallbacks[value] for value in kind}) == len(kind), kind


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
        # The US health share of Kenya in 2022: the family serves 288, and the
        # answer sums the 13 that hold data.
        (
            {
                "served_count": 288,
                "stat_var_member_count": 13,
                "stat_var_reading": StatVarReading.SUM,
                "stat_var_reading_set_by": ReadingSetBy.PLAN,
            },
            "summed over its 13 indicators",
        ),
        # With the donors left open, Search's count also covers group totals,
        # which the retrieval leaves out.
        (
            {
                "served_count": 288,
                "stat_var_member_count": 13,
                "open_entity_axes": (_DONORS,),
                "stat_var_reading": StatVarReading.SUM,
                "stat_var_reading_set_by": ReadingSetBy.PLAN,
            },
            "summed over whichever of its 288 indicators hold data",
        ),
        (
            {
                "served_count": 3,
                "stat_var_reading": StatVarReading.SUM,
                "stat_var_reading_set_by": ReadingSetBy.PLAN,
            },
            "summed over whichever of its 3 indicators hold data",
        ),
        (
            {
                "served_count": 2,
                "stat_var_reading": StatVarReading.SEPARATE,
                "stat_var_reading_set_by": ReadingSetBy.PLAN,
            },
            "kept separate over whichever of its 2 indicators hold data",
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
        "confirmed-within-a-served-family",
        "confirmed-over-an-open-donor-axis",
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


def test_a_choice_row_names_its_input_as_the_card_labels_name_it(
    decision: DecisionFactory,
) -> None:
    """`debt` and a health aid retrieval each pin a source whose basis choice offers
    its options differently, so the card lists a Basis row for each, naming its
    input. `debt`'s copy for the rolling growth is no row of the card, so `debt`
    is named as its step label names it, with no party or number added."""
    stored = split_draft(decision)
    spec_health = "spec_" + "c" * 32
    debt_basis = _scope_decision(SPEC_DEBT, "measure_basis")
    health_basis = _scope_decision(spec_health, "measure_basis", other="unavailable")
    pins = {"debt": SPEC_DEBT, "debt_history": SPEC_DEBT_HISTORY}
    plan = Plan.model_validate(
        {
            **stored.model_dump(),
            "decisions": [*stored.decisions, debt_basis, health_basis],
            "evidence_requirements": [
                *(
                    requirement.model_copy(
                        update={
                            "decision_ids": (*requirement.decision_ids, debt_basis.slot),
                            "source_spec_id": pins[requirement.id],
                        }
                    )
                    if requirement.id in pins
                    else requirement
                    for requirement in stored.evidence_requirements
                ),
                DatasetRequirement(
                    id="health",
                    decision_ids=["measure", health_basis.slot],
                    measure="health aid",
                    entities=[Entity(name="Kenya", role="recipient")],
                    period=Period(start=2015, end=2024),
                    source_spec_id=spec_health,
                ),
            ],
        }
    )

    questions = card_questions(plan)

    assert (questions[debt_basis.slot], questions[health_basis.slot]) == (
        "Basis: external debt service paid",
        "Basis: health aid",
    )


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


# --- The whole a share divides by ---------------------------------------------


@pytest.mark.parametrize(
    ("numerator", "whole", "wholes"),
    [
        (US_DONOR, DAC_DONORS, {"share": "from official donors (DAC)"}),
        (US_DONOR, ALL_DONORS, {"share": "from every donor"}),
        # Official donors' share of every provider's ODA divides by every provider.
        (DAC_DONORS, ALL_DONORS, {"share": "from every donor"}),
        # A share of official donors' ODA divides by the same donors, which it names.
        (DAC_DONORS, DAC_DONORS, {}),
    ],
    ids=["dac_donors", "all_donors", "dac_of_all", "same_donors"],
)
def test_a_share_names_the_population_its_denominator_holds_and_its_numerator_does_not(
    decision: DecisionFactory,
    numerator: dict[str, Any],
    whole: dict[str, Any],
    wholes: dict[str, str],
) -> None:
    """The United States' share of ODA to Kenya divides by official donors
    (DAC) over one source and by every provider over another, and the
    source the denominator reads says which."""
    donor = "United States" if numerator is US_DONOR else "official donors (DAC)"
    plan = us_share_of_kenya_oda(decision, whole=whole, numerator=numerator, donor=donor)

    assert share_wholes(plan) == wholes


@pytest.mark.parametrize(
    ("whole", "parties"),
    [(DAC_DONORS, "Kenya and official donors (DAC)"), (ALL_DONORS, "Kenya and all 140 donors")],
    ids=["dac_donors", "all_donors"],
)
def test_the_approved_plan_names_the_parties_a_shares_denominator_source_holds(
    decision: DecisionFactory, whole: dict[str, Any], parties: str
) -> None:
    """The denominator asks for every donor; the plan says what its source
    holds, the official donors (DAC) its group stands for or the count of
    donors it leaves open, in the fetch step's label and the share's sentence."""
    plan = us_share_of_kenya_oda(decision, whole=whole, numerator=US_DONOR)

    rows = {row["id"]: row for row in step_rows(plan)}

    assert rows["all_oda"]["label"] == f"Fetch ODA gross disbursements for {parties}"
    assert rows["share"]["description"].startswith(
        "Share of ODA gross disbursements as one figure overall, divided by ODA gross "
        f"disbursements ({parties})."
    )


@pytest.mark.parametrize(
    ("whole_across", "label"),
    [
        (("donor",), "Fetch ODA gross disbursements for Kenya and official donors (DAC)"),
        # A fetch naming no donor leaves the set its source binds unsaid, as it did.
        ((), "Fetch ODA gross disbursements for Kenya"),
    ],
    ids=["across_donors", "kenya_only"],
)
def test_a_bound_set_stands_only_for_an_across_word_on_its_side(
    decision: DecisionFactory, whole_across: tuple[str, ...], label: str
) -> None:
    plan = us_share_of_kenya_oda(
        decision, whole=DAC_DONORS, numerator=US_DONOR, whole_across=whole_across
    )

    rows = {row["id"]: row for row in step_rows(plan)}

    assert rows["all_oda"]["label"] == label


@pytest.mark.parametrize(
    ("scope", "numerator_recipients", "whole"),
    [
        # Each recipient divides by its own total, so the recipients are no part of the whole.
        (DenominatorScope.PER_AXIS, 120, "from official donors (DAC)"),
        # Every recipient divides by one total across them, though the numerator holds as many.
        (DenominatorScope.POOLED, 150, "from every donor and to every recipient"),
    ],
    ids=["per_axis", "pooled"],
)
def test_a_keyed_shares_whole_follows_its_denominator_scope(
    decision: DecisionFactory, scope: DenominatorScope, numerator_recipients: int, whole: str
) -> None:
    plan = us_share_per_recipient(
        decision, scope=scope, numerator_recipients=numerator_recipients, whole_recipients=150
    )

    assert share_wholes(plan) == {"share": whole}


_LATER = (PeriodThreshold(op="gte", year=2020),)


@pytest.mark.parametrize(
    ("then", "wholes"),
    [
        (
            (
                FilterRequirement(id="later", input_id="share", predicates=_LATER),
                FilterOperation(
                    id="later", requirement_id="later", input="share", predicates=_LATER
                ),
            ),
            {"share": "from official donors (DAC)", "later": "from official donors (DAC)"},
        ),
        (
            (
                AggregateRequirement(
                    id="mean", input_id="share", reduction=Reduction.MEAN, group_by=()
                ),
                AggregateOperation(
                    id="mean",
                    requirement_id="mean",
                    input="share",
                    reduction=Reduction.MEAN,
                    group_by=(),
                ),
            ),
            {"share": "from official donors (DAC)", "mean": "from official donors (DAC)"},
        ),
        # A total of shares, or their change, is no share of the whole.
        (
            (
                AggregateRequirement(
                    id="total", input_id="share", reduction=Reduction.SUM, group_by=()
                ),
                AggregateOperation(
                    id="total",
                    requirement_id="total",
                    input="share",
                    reduction=Reduction.SUM,
                    group_by=(),
                ),
            ),
            {"share": "from official donors (DAC)"},
        ),
        (
            (
                GrowthRequirement(
                    id="change",
                    input_id="share",
                    window_start=2015,
                    window_end=2024,
                    change=GrowthChange.ABSOLUTE,
                ),
                GrowthOperation(
                    id="change",
                    requirement_id="change",
                    input="share",
                    window_start=2015,
                    window_end=2024,
                    change=GrowthChange.ABSOLUTE,
                ),
            ),
            {"share": "from official donors (DAC)"},
        ),
    ],
    ids=["filter", "mean", "total", "growth"],
)
def test_a_step_keeping_its_inputs_shares_keeps_their_whole(
    decision: DecisionFactory, then: tuple[EvidenceRequirement, Operation], wholes: dict[str, str]
) -> None:
    plan = us_share_of_kenya_oda(decision, whole=DAC_DONORS, numerator=US_DONOR, then=[then])

    assert share_wholes(plan) == wholes


_AT_LEAST_1BN = (ValueThreshold(op="gte", value=1e9),)
_POOLED_WHOLE = "from every donor and to every recipient"


def _narrowed(requirement: FilterRequirement | RankRequirement) -> tuple[Any, Any]:
    """`requirement` reading `all_oda`, beside the operation carrying it out."""
    if isinstance(requirement, FilterRequirement):
        operation: Operation = FilterOperation(
            id=requirement.id,
            requirement_id=requirement.id,
            input="all_oda",
            predicates=requirement.predicates,
        )
    else:
        operation = RankOperation(
            id=requirement.id,
            requirement_id=requirement.id,
            input="all_oda",
            direction=requirement.direction,
            among=requirement.among,
            axis=requirement.axis,
            reduction=requirement.reduction,
            per_period=requirement.per_period,
            limit=requirement.limit,
        )
    return requirement, operation


_TOP_TEN = RankRequirement(
    id="top",
    input_id="all_oda",
    direction=RankDirection.HIGHEST_FIRST,
    among=RankAmong.POPULATION,
    decision_ids=["population"],
    axis="donor",
    reduction=Reduction.SUM,
    per_period=True,
    limit=10,
)
# The share's sentence over a denominator narrowed by the step `label` names.
_OVER_ROWS = (
    "Share of ODA gross disbursements as one figure overall, divided by one total of ODA "
    'gross disbursements over the rows "{label}" keeps.'
)


@pytest.mark.parametrize(
    ("narrowing", "wholes", "sentence"),
    [
        # Donors giving at least $1bn are not every donor, and nothing typed names them.
        (
            FilterRequirement(id="big", input_id="all_oda", predicates=_AT_LEAST_1BN),
            {},
            _OVER_ROWS.format(label="Filter ODA gross disbursements"),
        ),
        (
            _TOP_TEN,
            {},
            _OVER_ROWS.format(
                label="Rank the top 10 donors by the total of ODA gross disbursements"
            ),
        ),
        # Keeping some years keeps every donor.
        (
            FilterRequirement(id="later", input_id="all_oda", predicates=_LATER),
            {"share": "from every donor"},
            "Share of ODA gross disbursements as one figure overall, divided by ODA gross "
            "disbursements (Kenya and all 140 donors).",
        ),
    ],
    ids=["value_filter", "top_ten", "period_filter"],
)
def test_a_denominator_narrowed_below_its_sources_population_names_no_whole(
    decision: DecisionFactory,
    narrowing: FilterRequirement | RankRequirement,
    wholes: dict[str, str],
    sentence: str,
) -> None:
    """Neither the card nor the plan's sentence names every donor for a
    total over the donors a filter or a limited rank keeps; the sentence
    names that step, whose own line states what it keeps."""
    plan = us_share_of_kenya_oda(
        decision, whole=ALL_DONORS, numerator=US_DONOR, narrowing=[_narrowed(narrowing)]
    )

    assert share_wholes(plan) == wholes
    assert {row["id"]: row["sentence"] for row in requirement_rows(plan)}["share"] == sentence


_UK_AND_US = ("United Kingdom", "United States")
_TWO_RECIPIENTS = {"whole_recipients": ("Kenya", "Uganda")}
_US_TO_TWO = {"entities": (*US_DONOR["entities"], UGANDA)}


@pytest.mark.parametrize(
    ("whole", "whole_donors", "options", "wholes"),
    [
        (NAMED_DONORS, _UK_AND_US, {}, {"share": "from United Kingdom and United States"}),
        # The United States narrowed out of the same frame.
        (
            NAMED_DONORS,
            _UK_AND_US,
            {"narrowed_from": "all_oda"},
            {"share": "from United Kingdom and United States"},
        ),
        # Narrowed out of the two donors' ODA to Kenya, it holds Kenya alone.
        (
            {"entities": (*NAMED_DONORS["entities"], UGANDA)},
            _UK_AND_US,
            {
                **_TWO_RECIPIENTS,
                "numerator": NAMED_DONORS,
                "us_donors": _UK_AND_US,
                "narrowed_from": "us",
            },
            {"share": "from United Kingdom and United States and to Kenya and Uganda"},
        ),
        # A frame holding the United States on both sides narrows to the donor
        # its requirement names.
        (
            {"entities": (*NAMED_DONORS["entities"], USA_RECEIVING)},
            _UK_AND_US,
            {"narrowed_from": "all_oda"},
            {"share": "from United Kingdom and United States"},
        ),
        # The denominator names the numerator's donor, by its words or by another
        # name the graph resolves to the same ref, beside recipients it does not.
        (_US_TO_TWO, ("United States",), _TWO_RECIPIENTS, {"share": "to Kenya and Uganda"}),
        (
            {**_US_TO_TWO, "entity_refs": {"usa": "country/USA"}},
            ("USA",),
            _TWO_RECIPIENTS,
            {"share": "to Kenya and Uganda"},
        ),
        # Donors giving at least $1bn of the two are not both of them.
        (
            NAMED_DONORS,
            _UK_AND_US,
            {
                "narrowing": [
                    _narrowed(
                        FilterRequirement(id="big", input_id="all_oda", predicates=_AT_LEAST_1BN)
                    )
                ]
            },
            {},
        ),
    ],
    ids=[
        "named_donors",
        "narrowed_out_of_the_frame",
        "narrowed_out_of_another_frame",
        "named_on_both_sides",
        "same_donor",
        "same_ref",
        "value_filter",
    ],
)
def test_a_share_of_named_donors_names_them_where_the_numerator_holds_others(
    decision: DecisionFactory,
    whole: dict[str, Any],
    whole_donors: tuple[str, ...],
    options: dict[str, Any],
    wholes: dict[str, str],
) -> None:
    """The United States' share of the United Kingdom's and the United
    States' ODA to Kenya divides by the two donors; Kenya, which both sides
    name, is no part of the whole."""
    plan = us_share_of_kenya_oda(
        decision,
        whole=whole,
        whole_across=(),
        whole_donors=whole_donors,
        **{"numerator": US_DONOR, **options},
    )

    assert share_wholes(plan) == wholes


@pytest.mark.parametrize(
    ("reduction", "wholes"),
    [
        (Reduction.SUM, {"share": "from every donor"}),
        # The mean or the largest donor's flow each year is no total of every donor's.
        (Reduction.MEAN, {}),
        (Reduction.MAX, {}),
    ],
    ids=["sum", "mean", "max"],
)
def test_a_denominator_reducing_by_no_total_names_no_whole(
    decision: DecisionFactory, reduction: Reduction, wholes: dict[str, str]
) -> None:
    plan = us_share_of_kenya_oda(
        decision, whole=ALL_DONORS, numerator=US_DONOR, whole_reduction=reduction
    )

    assert share_wholes(plan) == wholes


def test_a_denominator_narrowed_to_one_entity_names_that_entity(decision: DecisionFactory) -> None:
    """The United States' ODA to Kenya as a share of the United Kingdom's ODA
    to every recipient divides by the United Kingdom's row of a frame of two
    donors to all 150 recipients, and names the United Kingdom alone."""
    donors = (
        ResolvedEntity(role="origin", ref="country/GBR", label="United Kingdom"),
        ResolvedEntity(role="origin", ref="country/USA", label="United States"),
    )
    plan = us_share_of_kenya_oda(
        decision,
        whole={
            "entities": donors,
            "open_entity_axes": (OpenEntityAxis(column="destination", member_count=150),),
        },
        numerator=US_DONOR,
        whole_across=(),
        whole_donors=("United Kingdom", "United States"),
    )
    payload = plan.model_dump(mode="json")
    payload["asks"] = ["US ODA to Kenya as a percentage of UK ODA to every recipient each year"]
    payload["answered"] = [{"ask": 0, "outputs": ["share"]}]
    for requirement in payload["evidence_requirements"]:
        if requirement["id"] == "all_oda":
            requirement["entities"] = requirement["entities"][:2]
            requirement["across"] = ["recipient"]
        if requirement["id"] == "share":
            requirement["denominator"] = {"input_id": "all_oda", "entity": "United Kingdom"}
    payload["evidence_requirements"] = [
        item for item in payload["evidence_requirements"] if item["id"] != "all_by_year"
    ]
    payload["translation"]["operations"] = [
        {**item, "denominator": {"input": "all_oda", "entity": "United Kingdom"}}
        if item["id"] == "share"
        else item
        for item in payload["translation"]["operations"]
        if item["id"] != "all_by_year"
    ]
    payload["workflow"]["steps"] = [
        item for item in payload["workflow"]["steps"] if item["id"] != "all_by_year"
    ]
    share = next(item for item in payload["workflow"]["steps"] if item["id"] == "share")
    share["bindings"]["denominator"]["from_step"] = "all_oda"

    assert share_wholes(Plan.model_validate(payload)) == {
        "share": "to every recipient and from United Kingdom"
    }


@pytest.mark.parametrize(
    ("limit", "wholes"),
    [
        # Ranked each year with no limit, the values are each recipient's place.
        (None, {"share": _POOLED_WHOLE}),
        (5, {"share": _POOLED_WHOLE, "ranked": _POOLED_WHOLE}),
    ],
    ids=["positions", "top_five"],
)
def test_a_rank_writing_positions_carries_no_share_whole(
    decision: DecisionFactory, limit: int | None, wholes: dict[str, str]
) -> None:
    rank = RankRequirement(
        id="ranked",
        input_id="share",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        decision_ids=["population"],
        axis="recipient",
        per_period=True,
        limit=limit,
    )
    operation = RankOperation(
        id="ranked",
        requirement_id="ranked",
        input="share",
        direction=rank.direction,
        among=rank.among,
        axis="recipient",
        per_period=True,
        limit=limit,
    )
    plan = us_share_per_recipient(
        decision,
        scope=DenominatorScope.POOLED,
        numerator_recipients=150,
        whole_recipients=150,
        then=[(rank, operation)],
    )

    assert share_wholes(plan) == wholes


def _every_donor(
    count: int, *, dataset: str | None = "DAC2A", window: int = 2015
) -> dict[str, Any]:
    """A source of ODA to Kenya leaving its donors open, holding `count` of
    them, read from `dataset` over the years from `window`; `None` for a
    source stating no identity."""
    return {
        "entities": ALL_DONORS["entities"],
        "open_entity_axes": (OpenEntityAxis(column="origin", member_count=count),),
        "identity": (
            None
            if dataset is None
            else {"dataset": dataset, "requested_window": {"start": window, "end": 2024}}
        ),
    }


@pytest.mark.parametrize(
    ("numerator", "whole", "whole_words"),
    [
        (_every_donor(140, window=2020), _every_donor(140), None),
        (_every_donor(120), _every_donor(140), "from every donor"),
        # As many donors in another dataset may be other donors.
        (_every_donor(140, dataset="CRS"), _every_donor(140), "from every donor"),
        (_every_donor(140, dataset=None), _every_donor(140, dataset=None), "from every donor"),
    ],
    ids=["same_donors", "fewer_donors", "other_dataset", "unidentified"],
)
def test_a_numerator_population_is_the_denominators_by_its_identity_not_its_words(
    decision: DecisionFactory,
    numerator: dict[str, Any],
    whole: dict[str, Any],
    whole_words: str | None,
) -> None:
    """120 donors are not the 140 the denominator holds, though both read
    "every donor", and 140 donors are the same 140 only in the same dataset,
    over whatever years."""
    plan = us_share_of_kenya_oda(decision, whole=whole, numerator=numerator, every_donor=True)

    wholes = share_wholes(plan)

    assert wholes == ({} if whole_words is None else {"share": whole_words})


# Three plans a live harness proposed for US health aid, as the planner node's
# payloads carried them: one retrieval and its total; a health share of all
# US aid; and that share after the user switched its basis to net.
_RECORDED_PLANS = Path(__file__).parent / "fixtures" / "plan_card_recorded.json"


def _recorded_payload(name: str, *, search_labels: bool = True) -> dict[str, Any]:
    """The planner node's recorded payload `name`. The recording predates Search's
    `concept_label` and `published_tables`, so with `search_labels` each source
    carries them as Search sends them for these CRS sources: the health
    part's sector concept ("health" for ONE's grouping of CRS 120 and 130), the
    total's measure noun and the CRS table. Without, each source is as stored
    before them."""
    payload = json.loads(_RECORDED_PLANS.read_text())[name]
    if search_labels:
        sectors = {
            decision["slot"]
            for decision in payload["decisions"]
            if (decision.get("scope") or {}).get("axis") == "sector"
        }
        sectored = {
            requirement["source_spec_id"]
            for requirement in payload["evidence_requirements"]
            if sectors & set(requirement.get("decision_ids", ()))
        }
        for source in payload["workflow"]["sources"]:
            concept = (
                "health" if source["spec_id"] in sectored else "official development assistance"
            )
            source.update(concept_label=concept, published_tables=_CRS_TABLES)
    return payload


def _recorded_plan(name: str, *, search_labels: bool = True) -> Plan:
    """The recorded plan `name` (`_recorded_payload`)."""
    payload = _recorded_payload(name, search_labels=search_labels)
    return Plan.model_validate(
        {
            key: value
            for key, value in payload.items()
            if key not in ("summary", "revision", "moved")
        }
    )


# The tables Search names for a CRS source (`resolved.published_tables`).
_CRS_TABLES = "OECD Creditor Reporting System (CRS)"
_HEALTH = "health"
# The sector option the recorded plans hold, as Search labelled it without its codes.
_RECORDED_SECTOR = "health incl. population and reproductive health"
# The option label Search gave the recorded plan's sector choice "sector:120+130".
_RECORDED_SECTOR_LABEL = f"{_RECORDED_SECTOR} (CRS 120 + 130)"
_LONG_LABELS = [
    f"Fetch {_HEALTH} for United States",
    f"Total {_HEALTH}",
    "Fetch official development assistance for United States",
    "Total official development assistance",
    "Health as a share of official development assistance",
]


@pytest.mark.parametrize(
    ("name", "title", "summary", "labels"),
    [
        (
            "short",
            "Total health aid from the United States to Africa in 2024",
            f"Gross disbursements, current USD, for {_HEALTH}, from United States to Africa "
            "incl. regional and unspecified amounts, in 2024, from the OECD Creditor Reporting "
            "System (CRS).",
            [f"Fetch {_HEALTH} for United States and each member of Africa", f"Total {_HEALTH}"],
        ),
        (
            "long",
            "Health share of US ODA in 2023",
            f"Gross disbursements, current USD, in 2023, from the OECD Creditor Reporting "
            f"System (CRS): {_HEALTH} from United States to 129 recipients, as a share of the "
            "total from United States to 152 recipients.",
            _LONG_LABELS,
        ),
        (
            "long_switched_to_net",
            "Health share of US ODA in 2023",
            f"Net disbursements, current USD, in 2023, from the OECD Creditor Reporting "
            f"System (CRS): {_HEALTH} from United States to 129 recipients, as a share of the "
            "total from United States to 152 recipients.",
            _LONG_LABELS,
        ),
    ],
)
def test_the_card_states_a_recorded_plan_once_in_short_labels(
    name: str, title: str, summary: str, labels: list[str]
) -> None:
    """The card is titled by the ask and sums the plan up in one sentence;
    each step label names what it fetches or computes in the reader's words,
    leaving the basis and the sector codes to the summary and the choices,
    so a one-year total says no "by year" and a switch to net leaves every
    label as it was."""
    plan = _recorded_plan(name)

    rows = card_rows(plan)

    assert card_title(plan) == title
    assert card_summary(plan) == summary
    assert [row["label"] for row in rows] == labels


_POPULATION = "population policy, programme, and reproductive health"


@pytest.mark.parametrize(
    ("option", "concept", "named"),
    [
        pytest.param(0, "health", "health", id="grouped-sector"),
        pytest.param(2, _POPULATION, _POPULATION, id="single-code"),
        pytest.param(
            0, None, f"{_RECORDED_SECTOR} (CRS 120 + 130)", id="grouped-sector-before-labels"
        ),
        pytest.param(2, None, f"{_POPULATION}, CRS 130 alone", id="single-code-before-labels"),
    ],
)
def test_a_retrieval_is_named_by_the_concept_search_states_for_its_sector(
    option: int, concept: str | None, named: str
) -> None:
    """The recorded US health aid to Africa with its sector choice on ONE's health
    grouping (CRS 120 and 130) or on CRS 130 alone: the card names the retrieval
    by the concept Search states for the source, with no codes, and a source
    stored before Search stated one by the sector option as Search labelled it."""
    payload = _recorded_payload("short", search_labels=False)
    (sector,) = [d for d in payload["decisions"] if d["slot"].startswith("scope_sector_")]
    sector["chosen"] = sector["options"][option]
    (source,) = payload["workflow"]["sources"]
    if concept is not None:
        source.update(concept_label=concept, published_tables=_CRS_TABLES)
    plan = Plan.model_validate(
        {k: v for k, v in payload.items() if k not in ("summary", "revision", "moved")}
    )

    assert [row["label"] for row in card_rows(plan)] == [
        f"Fetch {named} for United States and each member of Africa",
        f"Total {named}",
    ]
    assert f"for {named}, from United States" in card_summary(plan)


def test_a_recorded_plan_from_before_searchs_labels_reads_the_labels_search_composed() -> None:
    """The recorded US health share as stored before Search sent `concept_label` and
    `published_tables`: the health part is named by its sector option as Search
    labelled it, codes and all, and the total by Search's measure label, each as
    written; the summary names the dataset by its provenance label."""
    plan = _recorded_plan("long", search_labels=False)

    rows = card_rows(plan)

    sector = f"{_RECORDED_SECTOR} (CRS 120 + 130)"
    total = "gross disbursements (official development assistance)"
    assert [row["label"] for row in rows] == [
        f"Fetch {sector} for United States",
        f"Total {sector}",
        f"Fetch {total} for United States",
        f"Total {total}",
        f"{sector[0].upper()}{sector[1:]} as a share of {total}",
    ]
    assert card_summary(plan) == (
        "Gross disbursements, current USD, in 2023, from the OECD Creditor Reporting "
        f"System (CRS): {sector} from United States to 129 recipients, as a share of the "
        "total from United States to 152 recipients."
    )


def test_a_declared_whole_is_named_by_the_axis_it_opens_and_never_the_planners_measure() -> None:
    """The recorded US health share with its total declared the health part over
    every sector, under the planner's measure "Total aid to health": the whole's
    labels name all sectors, the part's sector opened, and never that measure."""
    plan = _recorded_plan("long")
    plan = Plan.model_validate(
        {
            **plan.model_dump(),
            "evidence_requirements": [
                r.model_copy(
                    update={
                        "measure": "Total aid to health",
                        "whole_of": WholeOf(part="us_health_aid_raw", axis="sector"),
                    }
                )
                if r.id == "us_total_aid_raw"
                else r
                for r in plan.evidence_requirements
            ],
        }
    )

    labels = [row["label"] for row in card_rows(plan)]

    assert labels == [
        f"Fetch {_HEALTH} for United States",
        f"Total {_HEALTH}",
        "Fetch all sectors for United States",
        "Total all sectors",
        f"{_HEALTH[0].upper()}{_HEALTH[1:]} as a share of all sectors",
    ]


@pytest.mark.parametrize(
    ("whole", "population"),
    [(DAC_DONORS, "official donors (DAC)"), (ALL_DONORS, "140 donors")],
    ids=["dac_donors", "all_donors"],
)
def test_a_declared_donor_whole_is_named_by_its_parts_concept_and_the_donors_its_source_holds(
    decision: DecisionFactory, whole: dict[str, Any], population: str
) -> None:
    """The United States' share of ODA to Kenya with the total declared the US part
    over every donor, under the planner's measure "Total aid to Kenya": the whole
    counts the part's concept, so the card names it so, with the donors its source
    holds, the total of official donors (DAC) or the 140 left open."""
    plan = us_share_of_kenya_oda(decision, whole=whole, numerator=US_DONOR)
    plan = Plan.model_validate(
        {
            **plan.model_dump(),
            "evidence_requirements": [
                r.model_copy(
                    update={
                        "measure": "Total aid to Kenya",
                        "whole_of": WholeOf(part="us", axis="donor"),
                    }
                )
                if r.id == "all_oda"
                else r
                for r in plan.evidence_requirements
            ],
        }
    )

    labels = [row["label"] for row in card_rows(plan)]

    whole_name = f"ODA gross disbursements from {population} to Kenya"
    assert labels == [
        "Fetch ODA gross disbursements from United States to Kenya",
        f"Fetch {whole_name}",
        f"Total {whole_name} by year",
        f"ODA gross disbursements from United States to Kenya as a share of {whole_name}",
    ]


_ODA = "official development assistance"


def test_a_donor_whole_counted_on_another_series_is_named_by_its_own_concept(
    decision: DecisionFactory,
) -> None:
    """The United States' bilateral ODA to Kenya as a share of its whole over every
    donor, which Search counted on total ODA, the bilateral series not being
    published for official donors (DAC): the card names the whole by the
    concept Search states for its own source, total ODA, and the part by its."""
    plan = us_share_of_kenya_oda(
        decision,
        whole={**DAC_DONORS, "concept_label": _ODA},
        numerator={**US_DONOR, "concept_label": "bilateral ODA"},
    )
    plan = Plan.model_validate(
        {
            **plan.model_dump(),
            "evidence_requirements": [
                r.model_copy(update={"whole_of": WholeOf(part="us", axis="donor")})
                if r.id == "all_oda"
                else r
                for r in plan.evidence_requirements
            ],
        }
    )

    assert [row["label"] for row in card_rows(plan)] == [
        "Fetch bilateral ODA for United States and Kenya",
        f"Fetch {_ODA} for Kenya and official donors (DAC)",
        f"Total {_ODA} by year",
        f"Bilateral ODA as a share of {_ODA}",
    ]


@pytest.mark.parametrize(
    ("whole", "numerator", "concept"),
    [
        ({"measure_label": "total ODA"}, {"measure_label": "bilateral ODA"}, "total ODA"),
        ({}, {"measure_label": "bilateral ODA"}, "bilateral ODA"),
        ({"measure_label": "total ODA"}, {"concept_label": _ODA}, _ODA),
        ({"concept_label": _ODA}, {"concept_label": "bilateral ODA"}, _ODA),
    ],
    ids=["own_measure_label", "parts_measure_label", "parts_concept", "own_concept"],
)
def test_a_donor_whole_is_named_alike_on_the_card_and_its_stored_reading(
    decision: DecisionFactory,
    whole: dict[str, Any],
    numerator: dict[str, Any],
    concept: str,
) -> None:
    """A donor whole of the United States' ODA to Kenya, its source and its
    part's stating a concept or only a measure label: the card's fetch and the
    name its reading is stored under (`source_whole_name`, the "Counted as"
    line) name it by one concept, with or without `concept_label`."""
    plan = us_share_of_kenya_oda(
        decision, whole={**DAC_DONORS, **whole}, numerator={**US_DONOR, **numerator}
    )
    plan = Plan.model_validate(
        {
            **plan.model_dump(),
            "evidence_requirements": [
                r.model_copy(update={"whole_of": WholeOf(part="us", axis="donor")})
                if r.id == "all_oda"
                else r
                for r in plan.evidence_requirements
            ],
        }
    )
    assert plan.workflow is not None
    steps = {binding.requirement_id: binding.step_id for binding in plan.workflow.evidence_bindings}
    source, part = (plan.workflow.source_of(steps[key]) for key in ("all_oda", "us"))
    assert source is not None

    stored = plan_display.source_whole_name(
        "donor", source=source, part=part, sector=None, measure=None
    )

    # The card adds the parties each reads only to retrievals named alike.
    assert str(card_rows(plan)[1]["label"]).startswith(f"Fetch {concept} f")
    assert stored == f"{concept} from official donors (DAC) to Kenya"


def test_a_comparison_of_two_entities_of_one_retrieval_names_each_sides_entity(
    decision: DecisionFactory,
) -> None:
    """US grant equivalents minus the UK's, both read from one DAC1 fetch of the
    two donors: each side names the entity it narrows the fetch to, so the
    card label, the step label, the sentence and the plan's words tell the two
    sides apart."""
    plan = Plan(
        question="How much more grant-equivalent ODA did the US give than the UK in 2023?",
        decisions=[decision()],
        evidence_requirements=[
            DatasetRequirement(
                id="dac1",
                decision_ids=["measure"],
                measure="grant equivalents",
                entities=[
                    Entity(name="United States", role="donor"),
                    Entity(name="United Kingdom", role="donor"),
                ],
                period=Period(start=2023, end=2023),
            ),
            ComparisonRequirement(
                id="gap",
                decision_ids=["measure"],
                minuend=RequirementSide(input_id="dac1", entity="United States"),
                subtrahend=RequirementSide(input_id="dac1", entity="United Kingdom"),
            ),
        ],
    )
    gap = plan.requirement("gap")
    us, uk = "grant equivalents for United States", "grant equivalents for United Kingdom"

    card = _card_label(gap, plan, concepts=_concepts(plan).names, shapes=frame_shapes(plan))

    assert card == f"Compare {us} to {uk}"
    assert _label(gap, plan) == f"Compare {us} to {uk}"
    (row,) = [row for row in requirement_rows(plan) if row["id"] == "gap"]
    assert row["sentence"] == (
        f"Grant equivalents for United States minus {uk}, at every period the two share."
    )
    assert plan_words(plan).names["gap"] == f"{us} against {uk}"


@pytest.mark.parametrize(
    ("donors", "labels"),
    [
        pytest.param(
            ("United States", "United Kingdom"),
            [
                "Fetch grant equivalents from United States",
                "Fetch grant equivalents from United Kingdom",
            ],
            id="parties-differ",
        ),
        pytest.param(
            ("United States", "United States"),
            [
                "Fetch grant equivalents from United States (1)",
                "Fetch grant equivalents from United States (2)",
            ],
            id="parties-equal",
        ),
    ],
)
def test_retrievals_named_alike_name_their_parties_once(
    decision: DecisionFactory, donors: tuple[str, str], labels: list[str]
) -> None:
    """Two retrievals of one concept, as "compare US and UK health aid" plans
    them: the concept names each one's parties, so the retrieval's label
    adds them no second time."""
    plan = Plan(
        question="Compare US and UK grant equivalents in 2023.",
        decisions=[decision()],
        evidence_requirements=[
            DatasetRequirement(
                id=f"dac1_{number}",
                decision_ids=["measure"],
                measure="grant equivalents",
                entities=[Entity(name=donor, role="donor")],
                period=Period(start=2023, end=2023),
            )
            for number, donor in enumerate(donors)
        ],
    )
    concepts, shapes = _concepts(plan).names, frame_shapes(plan)

    shown = [
        _card_label(requirement, plan, concepts=concepts, shapes=shapes)
        for requirement in plan.evidence_requirements
    ]

    assert shown == labels


def test_a_card_answering_an_ask_of_only_whitespace_takes_the_plans_title() -> None:
    plan = _recorded_plan("short")
    plan = Plan.model_validate({**plan.model_dump(), "asks": [" \n "]})

    assert card_title(plan) == "Research plan"


@pytest.mark.parametrize("name", ["short", "long", "long_switched_to_net"])
def test_the_one_step_answering_the_ask_gives_the_answer(name: str) -> None:
    """Exactly one step gives the answer: the last step the ask's answer
    names, whose description ends with the result's scope."""
    plan = _recorded_plan(name)

    rows = card_rows(plan)

    answering = [row for row in rows if row["gives_answer"]]
    assert [row["id"] for row in answering] == [plan.answered[-1].outputs[-1]]
    assert str(answering[0]["description"]).endswith(
        "Result: one figure, 2024." if name == "short" else "Result: one figure, 2023."
    )


def test_a_retrievals_description_is_searchs_account_of_its_source() -> None:
    plan = _recorded_plan("long_switched_to_net")

    fetch = card_rows(plan)[2]

    assert fetch["description"] == (
        "United States of America's net ODA disbursements to 152 recipients in 2023. "
        "Current US dollars · OECD Creditor Reporting System (CRS)."
    )


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("crs_total", "crs total"),
        ("sector:120+130 health incl. population", f"{_RECORDED_SECTOR} (CRS 120 + 130)"),
        ("sector:130", "population policy, programme, and reproductive health, CRS 130 alone"),
        ("CRS total aid", "CRS total aid"),
        ("gross disbursements, current USD", "gross disbursements, current USD"),
    ],
)
def test_a_choice_shows_no_planner_slug_or_option_id(text: str, shown: str) -> None:
    assert card_choice_text(text, _recorded_plan("long"), options=(text,)) == shown


def test_options_one_label_would_read_alike_keep_the_planners_words() -> None:
    """Two options of one decision opening with the same scope option id would
    both read as that option's label alone, so each keeps the planner's words
    after the id, the chosen one included; an option no other reads alike
    with still reads as the label alone."""
    plan = _recorded_plan("long")
    numerator, total = (
        "sector:120+130 health for the numerator",
        "sector:120+130 health for the total",
    )
    offered = (numerator, total, "sector:130")

    shown = [card_choice_text(text, plan, options=offered) for text in (*offered, numerator)]

    assert shown == [
        f"{_RECORDED_SECTOR_LABEL} health for the numerator",
        f"{_RECORDED_SECTOR_LABEL} health for the total",
        "population policy, programme, and reproductive health, CRS 130 alone",
        f"{_RECORDED_SECTOR_LABEL} health for the numerator",
    ]


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        (
            "measure_basis:net because the user asked for net flows",
            "net disbursements, current USD because the user asked for net flows",
        ),
        (
            "sector:130",
            "population policy, programme, and reproductive health, CRS 130 alone",
        ),
        ("crs_total", "crs total"),
        (
            "sector:120+130 Health incl. population is the sector the user asked for",
            f"{_RECORDED_SECTOR_LABEL} Health incl. population is the sector the user asked for",
        ),
    ],
)
def test_a_decisions_reason_keeps_its_words_after_the_option_id(text: str, shown: str) -> None:
    """A reason, question or caveat opening with an option's id reads that id
    as the option's label and keeps the planner's words after it as written,
    including words that repeat the label."""
    payload = _recorded_payload("long")
    for decision in payload["decisions"]:
        if (decision.get("scope") or {}).get("axis") == "sector":
            decision["options"][0] = decision["chosen"] = _RECORDED_SECTOR_LABEL
    plan = Plan.model_validate(
        {
            key: value
            for key, value in payload.items()
            if key not in ("summary", "revision", "moved")
        }
    )
    assert card_decision_text(text, plan) == shown


_RELABELLED_NET = "net flows, current USD"


@pytest.mark.parametrize(
    ("rendered", "label"),
    [
        (None, "net disbursements, current USD"),
        ("scope_sector_spec_61d37f5bd032b8bba3d077187e627ded", "net disbursements, current USD"),
        ("scope_measure_basis_spec_a85874f71d3d96e35c368b9454e051ca", _RELABELLED_NET),
    ],
    ids=["no_decision", "another_decisions_option", "own_label"],
)
def test_an_option_id_two_decisions_label_apart_never_shows_raw(
    rendered: str | None, label: str
) -> None:
    """The recorded plan's two basis decisions, the second relabelling
    "measure_basis:net": a reason opening with that id reads the label of the
    decision it is of where that decision offers it, else of the first
    decision offering it in plan order, so the id never shows, including in
    the reason of the sector decision, which offers no basis option."""
    payload = _recorded_payload("long")
    for held in payload["decisions"]:
        if held["slot"] == "scope_measure_basis_spec_a85874f71d3d96e35c368b9454e051ca":
            held["options"][1] = _RELABELLED_NET
    plan = Plan.model_validate(
        {
            key: value
            for key, value in payload.items()
            if key not in ("summary", "revision", "moved")
        }
    )
    decision = next((held for held in plan.decisions if held.slot == rendered), None)

    text = "measure_basis:net because the user asked for net flows"
    assert card_decision_text(text, plan, decision=decision) == (
        f"{label} because the user asked for net flows"
    )
    assert (
        card_choice_text(
            "measure_basis:net", plan, decision=decision, options=("measure_basis:net",)
        )
        == label
    )


@pytest.mark.parametrize(
    ("ghana", "kenya", "names"),
    [
        (
            ("Ghana", 2023, 2023),
            ("Kenya", 2023, 2023),
            ("health aid to Ghana", "health aid to Kenya"),
        ),
        (
            ("Kenya", 2015, 2023),
            ("Kenya", 2023, 2023),
            ("health aid to Kenya in 2015-2023", "health aid to Kenya in 2023"),
        ),
        (
            ("Kenya", 2023, 2023),
            ("Kenya", 2023, 2023),
            ("health aid to Kenya (1)", "health aid to Kenya (2)"),
        ),
    ],
    ids=["entity", "period", "alike"],
)
def test_requirements_a_sentence_names_alike_are_told_apart_by_the_facet_that_differs(
    decision: Callable[..., Decision],
    ghana: tuple[str, int, int],
    kenya: tuple[str, int, int],
    names: tuple[str, str],
) -> None:
    """Two retrievals named alike read as the card names them: with the
    parties each reads, then the period where that differs, then a number
    where nothing does. A calculation reads by the names it reads."""

    def dataset(requirement_id: str, place: str, start: int, end: int) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure="health aid",
            entities=[Entity(name=place, role="recipient")],
            period=Period(start=start, end=end),
        )

    plan = Plan(
        question="Health aid to Ghana against Kenya",
        decisions=[decision()],
        evidence_requirements=[
            dataset("left", *ghana),
            dataset("right", *kenya),
            ComparisonRequirement(
                id="both",
                decision_ids=["measure"],
                minuend=RequirementSide(input_id="left"),
                subtrahend=RequirementSide(input_id="right"),
            ),
        ],
    )

    words = plan_words(plan)

    assert (words.names["left"], words.names["right"]) == names
    assert words.names["both"] == f"{names[0]} against {names[1]}"
    assert words.inputs["both"] == ("left", "right")


@pytest.mark.parametrize(
    ("measure", "axis", "held", "names"),
    [
        (
            "health",
            "provider",
            ("bilateral flows only", "bilateral plus imputed multilateral"),
            (
                "health from United States to Kenya, bilateral flows only",
                "health from United States to Kenya, bilateral plus imputed multilateral",
            ),
        ),
        (
            "official development assistance",
            "measure_basis",
            ("net", "grant equivalent"),
            (
                "official development assistance from United States to Kenya, net",
                "official development assistance from United States to Kenya, grant equivalent",
            ),
        ),
    ],
    ids=["crs_bilateral_vs_inclusive", "dac1_net_vs_grant_equivalent"],
)
def test_namesakes_holding_different_scope_options_are_told_apart_by_the_option(
    decision: DecisionFactory,
    measure: str,
    axis: typing.Literal["provider", "measure_basis"],
    held: tuple[str, str],
    names: tuple[str, str],
) -> None:
    """Two retrievals whose concept, parties and period match, pinned to
    sources holding different options on the provider or measure-basis axis,
    read by the option each holds, as the source labels it, before any
    number. The card's plan and the caveats `propose_plan` writes
    (`_card_names`) name them alike, and both report that each name states
    the option on that axis (`PlanWords.options`)."""
    pins = (_SPEC_A, _SPEC_B)
    scopes = {
        pin: Decision(
            slot=scope_choice_slot(pin, axis),
            description=f"How {axis} is counted.",
            options=held,
            chosen=chosen,
            resolved_by=ResolvedBy.PLANNER_DEFAULT,
            rationale="Search's default for this question.",
            fixes=DecisionFixes(source_spec_id=pin),
            scope=DecisionScope(
                choice_id=f"{pin}:{axis}",
                axis=axis,
                options=(
                    ScopeOptionState(option_id=f"{axis}:first", status="available"),
                    ScopeOptionState(option_id=f"{axis}:second", status="available"),
                ),
            ),
        )
        for pin, chosen in zip(pins, held, strict=True)
    }
    datasets = [
        DatasetRequirement(
            id=f"r{index}",
            decision_ids=["measure"],
            measure=measure,
            entities=[
                Entity(name="United States", role="donor"),
                Entity(name="Kenya", role="recipient"),
            ],
            period=Period(start=2023, end=2023),
            source_spec_id=pin,
        )
        for index, pin in enumerate(pins)
    ]
    plan = Plan(
        question=f"{measure} from the United States to Kenya, both ways",
        decisions=[decision(), *scopes.values()],
        evidence_requirements=[
            dataset.model_copy(update={"decision_ids": (*dataset.decision_ids, scopes[pin].slot)})
            for dataset, pin in zip(datasets, pins, strict=True)
        ],
    )

    caveat = _card_names(
        datasets,
        decisions=[decision()],
        scope_decisions={pin: [scope] for pin, scope in scopes.items()},
        sources=SourceReadings(lambda _: None),
    )
    words = plan_words(plan)

    assert (words.names["r0"], words.names["r1"]) == names
    assert (caveat.names["r0"], caveat.names["r1"]) == names
    assert words.options == caveat.options == {"r0": {axis}, "r1": {axis}}


@pytest.mark.parametrize(
    ("measure", "name"),
    [("health aid", "health aid to Kenya (1)"), ("aid for health", "aid for health")],
    ids=["unchanged", "measure_changed"],
)
def test_a_revision_names_what_the_card_holds_by_the_cards_names(
    decision: Callable[..., Decision], measure: str, name: str
) -> None:
    """A revision that did not compile names a retrieval the card shows by its
    name there while its concept, parties and period are the card's, and
    otherwise by its own, as it names a retrieval new to the revision."""

    def dataset(requirement_id: str, measure: str) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure=measure,
            entities=[Entity(name="Kenya", role="recipient")],
            period=Period(start=2023, end=2023),
        )

    prior = Plan(
        question="Health aid to Kenya",
        decisions=[decision()],
        evidence_requirements=[dataset("health", "health aid"), dataset("again", "health aid")],
    )
    revision = Plan(
        question="Health aid to Kenya",
        decisions=[decision()],
        evidence_requirements=[
            dataset("health", measure),
            dataset("education", "education aid"),
        ],
    )

    words = plan_words(revision, shown_as=prior)

    assert (words.names["health"], words.names["education"]) == (name, "education aid")


def test_a_revisions_changed_retrieval_is_told_apart_from_a_name_the_card_holds(
    decision: Callable[..., Decision],
) -> None:
    """A retrieval whose measure the revision changed to one the card already
    names, and one new to the revision named alike, read apart from the name
    the card holds, which its retrieval keeps, by numbers it does not hold."""

    def dataset(requirement_id: str, measure: str) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure=measure,
            entities=[Entity(name="Kenya", role="recipient")],
            period=Period(start=2023, end=2023),
        )

    prior = Plan(
        question="Health and education aid to Kenya",
        decisions=[decision()],
        evidence_requirements=[
            dataset("health", "health aid"),
            dataset("again", "health aid"),
            dataset("other", "education aid"),
        ],
    )
    revision = Plan(
        question="Health aid to Kenya",
        decisions=[decision()],
        evidence_requirements=[
            dataset("health", "health aid"),
            dataset("other", "health aid"),
            dataset("new", "health aid"),
        ],
    )

    words = plan_words(revision, shown_as=prior)

    assert [words.names[rid] for rid in ("health", "other", "new")] == [
        "health aid to Kenya (1)",
        "health aid to Kenya (2)",
        "health aid to Kenya (3)",
    ]


def test_a_revisions_retrieval_sharing_a_carried_retrievals_concept_keeps_its_parties(
    decision: Callable[..., Decision],
) -> None:
    """A retrieval the revision changed shares its concept with one the card
    holds by a name with parties added; it reads with its own parties too,
    since the concept alone reads as either."""

    def dataset(requirement_id: str, donor: str, start: int) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure="health aid",
            entities=[Entity(name=donor, role="donor"), Entity(name="Kenya", role="recipient")],
            period=Period(start=start, end=2023),
        )

    prior = Plan(
        question="Health aid to Kenya from the United States and France",
        decisions=[decision()],
        evidence_requirements=[
            dataset("us", "United States", 2023),
            dataset("france", "France", 2023),
        ],
    )
    revision = Plan(
        question="Health aid to Kenya from the United States and France",
        decisions=[decision()],
        evidence_requirements=[
            dataset("us", "United States", 2023),
            dataset("france", "France", 2020),
        ],
    )

    words = plan_words(revision, shown_as=prior)

    assert (words.names["us"], words.names["france"]) == (
        "health aid from United States to Kenya",
        "health aid from France to Kenya",
    )


def test_a_revisions_retrieval_changing_its_option_is_told_apart_from_a_numbered_carried_one(
    decision: DecisionFactory,
) -> None:
    """The card numbers two retrievals alike in every naming facet, "(1)" and
    "(2)"; a revision changes only the second's provider option. The first
    keeps its card name, and the second, a namesake of it by concept,
    parties and period whatever number the card's name holds, reads by the
    option it now holds."""
    pins = (_SPEC_A, _SPEC_B)
    held = ("bilateral flows only", "bilateral plus imputed multilateral")

    def scope(pin: str, chosen: str) -> Decision:
        return Decision(
            slot=scope_choice_slot(pin, "provider"),
            description="How provider is counted.",
            options=held,
            chosen=chosen,
            resolved_by=ResolvedBy.PLANNER_DEFAULT,
            rationale="Search's default for this question.",
            fixes=DecisionFixes(source_spec_id=pin),
            scope=DecisionScope(
                choice_id=f"{pin}:provider",
                axis="provider",
                options=(
                    ScopeOptionState(option_id="provider:first", status="available"),
                    ScopeOptionState(option_id="provider:second", status="available"),
                ),
            ),
        )

    def plan(second: str) -> Plan:
        scopes = [scope(pins[0], held[0]), scope(pins[1], second)]
        return Plan(
            question="Health aid from the United States to Kenya, twice",
            decisions=[decision(), *scopes],
            evidence_requirements=[
                DatasetRequirement(
                    id=f"r{index}",
                    decision_ids=["measure", held_scope.slot],
                    measure="health",
                    entities=[
                        Entity(name="United States", role="donor"),
                        Entity(name="Kenya", role="recipient"),
                    ],
                    period=Period(start=2023, end=2023),
                    source_spec_id=pin,
                )
                for index, (pin, held_scope) in enumerate(zip(pins, scopes, strict=True))
            ],
        )

    prior = plan(held[0])
    assert _concepts(prior).names == {
        "r0": "health from United States to Kenya (1)",
        "r1": "health from United States to Kenya (2)",
    }

    words = plan_words(plan(held[1]), shown_as=prior)

    assert (words.names["r0"], words.names["r1"]) == (
        "health from United States to Kenya (1)",
        "health from United States to Kenya, bilateral plus imputed multilateral",
    )
    assert words.options == {"r0": frozenset(), "r1": {"provider"}}


def test_a_revisions_new_retrieval_is_told_apart_from_the_name_a_carried_one_keeps(
    decision: DecisionFactory,
) -> None:
    """The card names health to Kenya, holding a provider option, apart from
    health to Uganda by its parties. A revision drops Uganda and adds health
    to Kenya with no provider decision. The carried retrieval keeps its card
    name; the new one, whose own facets read as that name, takes a number."""
    held = ("bilateral flows only", "bilateral plus imputed multilateral")
    provider = Decision(
        slot=scope_choice_slot(_SPEC_A, "provider"),
        description="How provider is counted.",
        options=held,
        chosen=held[0],
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="Search's default for this question.",
        fixes=DecisionFixes(source_spec_id=_SPEC_A),
        scope=DecisionScope(
            choice_id=f"{_SPEC_A}:provider",
            axis="provider",
            options=(
                ScopeOptionState(option_id="provider:first", status="available"),
                ScopeOptionState(option_id="provider:second", status="available"),
            ),
        ),
    )

    def dataset(requirement_id: str, recipient: str, *scopes: str) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure", *scopes],
            measure="health",
            entities=[Entity(name=recipient, role="recipient")],
            period=Period(start=2023, end=2023),
            source_spec_id=_SPEC_A if scopes else None,
        )

    kenya = dataset("kenya", "Kenya", provider.slot)
    prior = Plan(
        question="Health aid to Kenya and Uganda",
        decisions=[decision(), provider],
        evidence_requirements=[kenya, dataset("uganda", "Uganda")],
    )
    revision = Plan(
        question="Health aid to Kenya, twice",
        decisions=[decision(), provider],
        evidence_requirements=[kenya, dataset("again", "Kenya")],
    )
    assert _concepts(prior).names["kenya"] == "health to Kenya"

    words = plan_words(revision, shown_as=prior)

    assert (words.names["kenya"], words.names["again"]) == (
        "health to Kenya",
        "health to Kenya (1)",
    )


@pytest.mark.parametrize("helper", ["_concept", "_who"])
def test_a_revision_reads_each_retrievals_naming_once_per_plan(
    decision: Callable[..., Decision], monkeypatch: pytest.MonkeyPatch, helper: str
) -> None:
    """Naming a revision around the card's names reads each retrieval's concept
    and parties once for the card's plan and once for the revision, however
    many of them share a name."""

    def dataset(requirement_id: str, measure: str) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure=measure,
            entities=[Entity(name="Kenya", role="recipient")],
            period=Period(start=2023, end=2023),
        )

    prior = Plan(
        question="Health aid to Kenya",
        decisions=[decision()],
        evidence_requirements=[dataset("health", "health aid"), dataset("again", "health aid")],
    )
    revision = Plan(
        question="Health aid to Kenya",
        decisions=[decision()],
        evidence_requirements=[dataset("health", "health aid"), dataset("new", "health aid")],
    )
    real = getattr(plan_display, helper)
    reads: Counter[tuple[str, str]] = Counter()

    def counted(requirement: DatasetRequirement, plan: Plan, **kwargs: Any) -> str:
        reads[plan.question, requirement.id] += 1
        return real(requirement, plan, **kwargs)

    monkeypatch.setattr(plan_display, helper, counted)
    revision = revision.amend(question="Health aid to Kenya, revised")

    plan_words(revision, shown_as=prior)

    assert reads, helper
    assert max(reads.values()) == 1, reads


def test_a_revision_not_compiled_names_what_the_card_holds_by_the_cards_names() -> None:
    """The recorded plan, compiled, and an amendment of it that did not compile
    and so holds no workflow, here one rewording the question: each retrieval
    it holds unchanged reads the source the card's was compiled from, so it
    keeps the card's name."""
    prior = _recorded_plan("long")
    revision = prior.amend(question=f"{prior.question} Show it as a share.")
    assert revision.workflow is None

    words = plan_words(revision, shown_as=prior)

    card = _concepts(shown_plan(prior)).names
    assert card == {
        "us_health_aid_raw": "health",
        "us_total_aid_raw": "official development assistance",
    }
    assert {rid: words.names[rid] for rid in card} == card


@pytest.mark.parametrize(
    ("concept", "name"),
    [("health care", "health care"), (None, _RECORDED_SECTOR_LABEL)],
    ids=["concept_label", "no_concept_label"],
)
def test_a_retrieval_a_revision_repins_is_named_by_searchs_answer_for_its_pin(
    concept: str | None, name: str
) -> None:
    """A revision, not compiled, that pins the recorded plan's health retrieval
    to another dataset is named by the concept Search's answer for that pin
    states, and with none stated by the sector option's label."""
    prior = _recorded_plan("long")
    moved = f"spec_{'e' * 32}"
    revision = prior.model_copy(
        update={
            "translation": None,
            "workflow": None,
            "evidence_requirements": [
                r.model_copy(update={"source_spec_id": moved}) if r.id == "us_health_aid_raw" else r
                for r in prior.evidence_requirements
            ],
        }
    )
    block = {
        "spec_id": moved,
        "resolved": {"concept_label": concept} if concept is not None else {},
    }

    words = plan_words(revision, shown_as=prior, pinned={moved: source_from_spec_block(block)})

    assert words.names["us_health_aid_raw"] == name
    assert words.names["us_total_aid_raw"] == "official development assistance"


def test_namesakes_are_numbered_as_the_plan_states_them_before_and_after_compile(
    decision: DecisionFactory,
) -> None:
    """Two retrievals named alike, stated `z_aid` then `m_aid`, against the order of
    their ids. A rolling growth reads `z_aid` from its first year, so the split runs
    a copy, `z_aid_history`, beside it, and the compiler, taking operations by
    requirement id, runs `m_aid` before `z_aid`. The sentence a proposal writes over
    the plan as stated (`submission.service._card_names`), a revision's sentence over
    the plan not compiled, and the compiled card each number them as the plan states
    them."""
    retrieval = DatasetRequirement(
        id="z_aid",
        decision_ids=["measure"],
        measure="aid",
        entities=[Entity(name="Kenya", role="recipient")],
        period=Period(start=2015, end=2024),
        period_end_open=True,
    )
    draft = Plan(
        question="Aid to Kenya",
        decisions=[decision()],
        evidence_requirements=[
            retrieval,
            retrieval.model_copy(update={"id": "m_aid", "period_end_open": False}),
            GrowthRequirement(
                id="a_growth",
                input_id="z_aid",
                window_start=2019,
                window_end=2024,
                rolling=RollingWindow(
                    window_start=RelativePeriod(offset=-5), window_end=RelativePeriod(offset=0)
                ),
            ),
            SumOfRequirement(
                id="s_sum",
                decision_ids=["measure"],
                left=RequirementSide(input_id="m_aid"),
                right=RequirementSide(input_id="z_aid"),
            ),
        ],
        first_years=[FirstYear(requirement_id="z_aid", year=2000)],
    )
    split = split_of(draft)
    assert [copy.copy_id for copy in split.copies] == ["z_aid_history"]
    uncompiled = Plan.model_validate(
        {
            **draft.model_dump(),
            "evidence_requirements": split.requirements,
            "rolling_copies": split.copies,
        }
    )
    pins = {"z_aid": SPEC_DEBT, "z_aid_history": SPEC_DEBT_HISTORY, "m_aid": "spec_" + "c" * 32}
    translation = build_translation(uncompiled, pins)

    def retrieve(step_id: str) -> WorkflowStep:
        return WorkflowStep(
            id=step_id,
            operation_id=step_id,
            tool=ToolId(server="data-mcp", name="get_data"),
            args={"spec_id": pins[step_id]},
            input_schema_hash="hash",
        )

    def frame(step_id: str) -> Binding:
        return Binding(from_step=step_id, output=ArtifactKind.FRAME_REF)

    workflow = Workflow(
        # The compiler's order: requirements by id, each after the inputs it reads.
        steps=[
            retrieve("z_aid_history"),
            WorkflowStep(
                id="a_growth",
                operation_id="a_growth",
                tool=ToolId(server="analysis-mcp", name="growth"),
                bindings={"frame_ref": frame("z_aid_history")},
                input_schema_hash="hash",
            ),
            retrieve("m_aid"),
            retrieve("z_aid"),
            WorkflowStep(
                id="s_sum",
                operation_id="s_sum",
                tool=ToolId(server="analysis-mcp", name="sum_of"),
                bindings={"left_ref": frame("m_aid"), "right_ref": frame("z_aid")},
                input_schema_hash="hash",
            ),
        ],
        sources=[
            SelectedSource(spec_id=spec_id, sentence="aid", has_data=True)
            for spec_id in pins.values()
        ],
        evidence_bindings=[
            EvidenceBinding(requirement_id=rid, step_id=rid, spec_id=spec_id)
            for rid, spec_id in pins.items()
        ],
        terminal_steps=["a_growth", "s_sum"],
    )
    compiled = uncompiled.with_translation(translation=translation, workflow=workflow)

    caveat = _card_names(
        draft.evidence_requirements,
        decisions=draft.decisions,
        scope_decisions={},
        sources=SourceReadings(lambda _: None),
    ).names
    revision = plan_words(uncompiled).names
    card = plan_words(compiled).names

    retrievals = ("z_aid", "m_aid")
    assert [caveat[rid] for rid in retrievals] == ["aid to Kenya (1)", "aid to Kenya (2)"]
    assert [revision[rid] for rid in retrievals] == [caveat[rid] for rid in retrievals]
    assert [card[rid] for rid in retrievals] == [caveat[rid] for rid in retrievals]


# `_concepts` numbers the namesakes "aid" and "aid (1)" in card order, the "aid"
# pair first, so the numbers it gives depend on no hash order.
_NAMESAKES_BY_CARD_ORDER = ["aid (2)", "aid (3)", "aid (1) (1)", "aid (1) (2)"]


@pytest.mark.parametrize("seed", ["0", "1"])
def test_namesakes_are_numbered_in_card_order_whatever_the_hash_order(
    decision: DecisionFactory, seed: str
) -> None:
    """Two retrievals named "aid" and two named "aid (1)": numbering one pair
    can take a name the other holds, so the names depend on which pair is
    numbered first. Each pair is numbered in card order. Run in a fresh
    interpreter per hash seed, since the two seeds order a set of the two
    names each way. Each run takes about a second."""
    plan = Plan(
        question="Aid",
        decisions=[decision()],
        evidence_requirements=[
            DatasetRequirement(
                id=f"r{index}",
                decision_ids=["measure"],
                measure=measure,
                across=["recipient"],
                period=Period(start=2023, end=2023),
            )
            for index, measure in enumerate(["aid", "aid", "aid (1)", "aid (1)"])
        ],
    )
    script = (
        "import json, sys\n"
        "from harness.plan_display import _concepts\n"
        "from planner.plan import Plan\n"
        "plan = Plan.model_validate_json(sys.stdin.read())\n"
        "print(json.dumps(list(_concepts(plan).names.values())))\n"
    )

    ran = subprocess.run(
        [sys.executable, "-c", script],
        input=plan.model_dump_json(),
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": seed},
    )

    assert json.loads(ran.stdout) == _NAMESAKES_BY_CARD_ORDER


@pytest.mark.parametrize(
    "over",
    [
        AggregateRequirement(
            id="over_share", input_id="health_share", reduction=Reduction.MEAN, group_by=[]
        ),
        GrowthRequirement(
            id="over_share", input_id="health_share", window_start=2016, window_end=2020
        ),
        RankRequirement(
            id="over_share",
            input_id="health_share",
            direction=RankDirection.HIGHEST_FIRST,
            among=RankAmong.POPULATION,
            reduction=Reduction.MEAN,
        ),
        FilterRequirement(
            id="over_share", input_id="health_share", predicates=(ValueThreshold(op="gt", value=0),)
        ),
    ],
    ids=["aggregate", "growth", "rank", "filter"],
)
def test_a_step_over_a_share_keeps_the_shares_summary(over: EvidenceRequirement) -> None:
    """The answer is a total, growth, rank or filter over the recorded share: the
    summary still says what the share divides, "as a share of" its denominator and
    from which source."""
    plan = _recorded_plan("long")
    plan = plan.model_copy(
        update={
            "evidence_requirements": [
                *plan.evidence_requirements,
                over.model_copy(update={"decision_ids": ("measure",)}),
            ],
            "answered": [plan.answered[0].model_copy(update={"outputs": ("over_share",)})],
        }
    )

    assert card_summary(plan) == card_summary(_recorded_plan("long"))


@pytest.mark.parametrize(
    "over",
    [
        ComparisonRequirement(
            id="over_shares",
            minuend=RequirementSide(input_id="health_share"),
            subtrahend=RequirementSide(input_id="other_share"),
        ),
        SumOfRequirement(
            id="over_shares",
            left=RequirementSide(input_id="health_share"),
            right=RequirementSide(input_id="other_share"),
        ),
    ],
    ids=["comparison", "sum"],
)
def test_a_calculation_over_two_shares_has_no_summary(over: EvidenceRequirement) -> None:
    """The answer is a comparison or sum of two shares: naming either share's
    numerator alone would read as a plain amount, so the card shows its title only."""
    plan = _recorded_plan("long")
    share = plan.requirement("health_share")
    plan = plan.model_copy(
        update={
            "evidence_requirements": [
                *plan.evidence_requirements,
                share.model_copy(update={"id": "other_share"}),
                over.model_copy(update={"decision_ids": ("measure",)}),
            ],
            "answered": [plan.answered[0].model_copy(update={"outputs": ("over_shares",)})],
        }
    )

    assert card_summary(plan) is None


def test_a_share_whose_sides_read_one_retrieval_still_says_as_a_share_of() -> None:
    """Both sides of the share total the same retrieval: the summary keeps
    each side, joined by "as a share of", and does not read as one figure."""
    plan = _recorded_plan("long")
    plan = plan.model_copy(
        update={
            "evidence_requirements": [
                requirement.model_copy(update={"input_id": "us_health_aid_raw"})
                if requirement.id == "us_total_aid_sum"
                else requirement
                for requirement in plan.evidence_requirements
            ]
        }
    )

    side = f"{_HEALTH} from United States to 129 recipients"
    assert card_summary(plan) == (
        "Gross disbursements, current USD, in 2023, from the OECD Creditor Reporting "
        f"System (CRS): {side}, as a share of {side}."
    )


def _planner_decision(**update: Any) -> Decision:
    """A decision the planner wrote fixing the window's first year, with `update` set."""
    return Decision.model_validate(
        {
            "slot": "window",
            "description": "Which years the answer covers.",
            "options": ["since 2015", "since 2020"],
            "chosen": "since 2015",
            "resolved_by": "planner_default",
            "rationale": "The question asks for the last decade.",
            "fixes": {"window_start": 2015},
            **update,
        }
    )


@pytest.mark.parametrize(
    ("decision", "switchable"),
    [
        (_planner_decision(), True),
        (_planner_decision(fixes=None), False),
        (
            _planner_decision(slot=search_pin_slot(_SPEC_A), fixes={"source_spec_id": _SPEC_A}),
            False,
        ),
        (_scope_decision(_SPEC_A, "sector"), False),
        (_planner_decision(caveat="providers_differ"), False),
    ],
    ids=["fixes_fields", "rationale_only", "search_pin", "scope_choice", "caveat"],
)
def test_only_a_planner_decision_settling_fields_switches_on_the_card(
    decision: Decision, switchable: bool
) -> None:
    """The decisions `propose_plan` writes switch through Search or qualify
    the answer, and a rationale settles nothing a switch could change."""
    assert switchable_decision(decision) is switchable


def test_a_planner_decisions_row_id_never_names_a_scope_choice() -> None:
    """A requirement may be called `decision`, and a planner decision's slot
    may be a scope axis: the scope choice of that requirement on that axis
    and the planner decision's row still take different ids."""
    dumped = scripted_plan("kenya life expectancy", spec_id=None).model_dump()
    for requirement in dumped["evidence_requirements"]:
        if requirement["id"] == "life_expectancy":
            requirement.update(id="decision", source_spec_id=_SPEC_A)
        if requirement.get("input_id") == "life_expectancy":
            requirement["input_id"] = "decision"
    plan = Plan.model_validate(dumped | {"translation": None, "workflow": None})

    assert decision_choice_id(_planner_decision(slot="sector")) != pinned_choice_id(
        plan, {_SPEC_A}, "sector"
    )


def _fixing_measure(plan: Plan) -> Plan:
    """`plan` (the scripted Kenya life expectancy plan) with its measure
    decision fixing the measure its dataset reads."""
    dataset = next(r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement))
    return Plan.model_validate(
        {
            **plan.model_dump(),
            "decisions": [
                d.model_copy(update={"fixes": DecisionFixes(measure=dataset.measure)})
                for d in plan.decisions
            ],
        }
    )


def test_a_switch_of_a_planner_decision_picks_the_option_as_the_plan_states_it() -> None:
    """The row `decision/measure`'s option for a label picks that label, with
    the decision as the plan states it. A row or option the card does not
    offer picks nothing, and nor does the option the decision has chosen."""
    unfixed = scripted_plan("kenya life expectancy", spec_id=None)
    plan = _fixing_measure(unfixed)
    (measure,) = plan.decisions
    birth, healthy = measure.options
    assert (birth, healthy) == ("life expectancy at birth", "healthy life expectancy")
    assert measure.fixes is not None
    assert measure.chosen == birth

    picks = [
        decision_pick(plan, choice_id=choice, option_id=option, to_origin={})
        for choice, option in [
            ("decision/measure", decision_option_id(healthy)),
            ("decision/measure", decision_option_id(birth)),
            ("decision/measure", decision_option_id("life expectancy at 60")),
            ("decision/measure", "option_2"),
            ("decision/other", decision_option_id(healthy)),
            ("measure", decision_option_id(healthy)),
        ]
    ]

    assert picks == [
        DecisionPick.on_card(plan, slot="measure", chosen=healthy),
        None,
        None,
        None,
        None,
        None,
    ]
    assert (
        decision_pick(
            unfixed,
            choice_id="decision/measure",
            option_id=decision_option_id(healthy),
            to_origin={},
        )
        is None
    )
