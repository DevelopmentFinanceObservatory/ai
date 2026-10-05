"""A dataset requirement's `stat_vars` reading, as the plan itself checks it.

The plan has no source yet, so a requirement reading its family separately
carries an uncounted number of measures. A source binding one StatVar gives it
one, so no single-measure verb over it is refused when the plan is built; the
compiler counts the measures once the source is known and refuses a family of
several there (`tests/compiler/test_family_reading.py`).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from planner.plan import (
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    EvidenceRequirement,
    FilterRequirement,
    Plan,
    PlanStatus,
)
from planner.records import PeriodThreshold, StatVarReading

DecisionFactory = Callable[..., Decision]
RequirementFactory = Callable[..., DatasetRequirement]
AggregateFactory = Callable[..., AggregateRequirement]


def _plan(decisions: list[Decision], requirements: list[EvidenceRequirement]) -> Plan:
    return Plan(
        question="How much aid did Germany give each year?",
        decisions=decisions,
        evidence_requirements=requirements,
        status=PlanStatus.DRAFT,
    )


@pytest.mark.parametrize("reading", [None, *StatVarReading])
def test_an_aggregate_reads_a_requirement_whatever_its_reading(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: AggregateFactory,
    reading: StatVarReading | None,
) -> None:
    plan = _plan([decision()], [requirement(stat_vars=reading), derived_requirement()])

    assert plan.evidence_requirements[0].stat_vars is reading


def test_a_filter_reads_a_family_read_separately(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    from_2020 = FilterRequirement(
        id="from_2020",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        predicates=(PeriodThreshold(op="gte", year=2020),),
    )

    plan = _plan([decision()], [requirement(stat_vars=StatVarReading.SEPARATE), from_2020])

    assert plan.evidence_requirements[1] == from_2020


def test_a_requirement_contradicting_the_reading_its_decision_fixes_is_refused(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: AggregateFactory,
) -> None:
    total = decision(
        slot="read_as_total",
        description="Whether to read the family as its total or one series per StatVar.",
        options=["sum", "separate"],
        chosen="sum",
        rationale="The question asks for the total.",
        fixes=DecisionFixes(stat_vars=StatVarReading.SUM),
    )

    with pytest.raises(ValueError, match="states stat_vars None"):
        _plan(
            [decision(), total],
            [requirement(decision_ids=["measure", "read_as_total"]), derived_requirement()],
        )
