"""A dataset requirement's `stat_vars` reading, as the plan itself checks it.

The plan has no source yet, so a requirement reading its family separately
carries an uncounted number of measures. A source binding one StatVar gives it
one, so no single-measure verb over it is refused when the plan is built; the
compiler counts the measures once the source is known and refuses a family of
several there (`tests/compiler/test_family_reading.py`).

A requirement stating a `breakdown` sums the family within each code of one
classification, so it carries one measure and one more role, the
breakdown's, which an aggregate grouping by breakdown keeps alone.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from planner.plan import (
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    EvidenceRequirement,
    FilterRequirement,
    GrowthRequirement,
    Period,
    Plan,
    PlanStatus,
    RankRequirement,
    frame_shapes,
)
from planner.records import (
    Breakdown,
    BreakdownDimension,
    GroupBy,
    PeriodThreshold,
    RankAmong,
    RankDirection,
    Reduction,
    StatVarReading,
)
from planner.shapes import RoleDimension

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


# --- A breakdown by sector ------------------------------------------------------

_TOP_SECTORS = Breakdown(dimension=BreakdownDimension.SECTOR, level=0)


def _aid_to_kenya(**overrides: Any) -> DatasetRequirement:
    fields: dict[str, Any] = {
        "id": "aid_to_kenya",
        "decision_ids": ["measure"],
        "measure": "aid",
        "entities": [Entity(name="Kenya", role="recipient")],
        "across": "donor",
        "period": Period(start=2024, end=2024),
        "stat_vars": StatVarReading.SEPARATE,
        "breakdown": _TOP_SECTORS,
    }
    return DatasetRequirement(**(fields | overrides))


def _by_sector(**overrides: Any) -> AggregateRequirement:
    fields: dict[str, Any] = {
        "id": "by_sector",
        "decision_ids": ["measure"],
        "input_id": "aid_to_kenya",
        "reduction": Reduction.SUM,
        "group_by": (GroupBy.PERIOD, GroupBy.BREAKDOWN),
    }
    return AggregateRequirement(**(fields | overrides))


def _sector_rank(**overrides: Any) -> RankRequirement:
    fields: dict[str, Any] = {
        "id": "top_sectors",
        "decision_ids": ["measure", "every_sector"],
        "input_id": "by_sector",
        "direction": RankDirection.HIGHEST_FIRST,
        "among": RankAmong.POPULATION,
        "axis": "sector",
        "limit": 10,
    }
    return RankRequirement(**(fields | overrides))


def _every_sector(decision: DecisionFactory) -> Decision:
    return decision(
        slot="every_sector",
        description="Whether to rank every sector or named ones.",
        options=["every sector", "named sectors"],
        chosen="every sector",
        rationale="The question asks which sectors received the most.",
        fixes=DecisionFixes(among=RankAmong.POPULATION),
    )


def test_which_sectors_received_the_most_aid_in_kenya_ranks_each_sector_s_total(
    decision: DecisionFactory,
) -> None:
    """ "Which sectors received the most aid in Kenya in the latest year available?":
    the retrieval keeps the sector beside the recipient and every donor, and the
    aggregate by period and breakdown totals the donors within each sector."""
    plan = _plan(
        [decision(), _every_sector(decision)], [_aid_to_kenya(), _by_sector(), _sector_rank()]
    )

    shapes = frame_shapes(plan.evidence_requirements)
    sector = RoleDimension(role="sector", requirement_id="aid_to_kenya", breakdown=True)
    assert shapes["aid_to_kenya"].roles == (
        RoleDimension(role="recipient", requirement_id="aid_to_kenya"),
        RoleDimension(role="donor", requirement_id="aid_to_kenya", open=True),
        sector,
    )
    assert shapes["aid_to_kenya"].measures == 1
    assert shapes["by_sector"].roles == (sector,)
    assert shapes["top_sectors"].roles == (sector,)


_SECTOR_CHANGE = GrowthRequirement(
    id="sector_change",
    decision_ids=["measure"],
    input_id="by_sector",
    window_start=2019,
    window_end=2024,
)


@pytest.mark.parametrize(
    ("reader", "rank_decision"),
    [(_sector_rank(per_period=True), True), (_SECTOR_CHANGE, False)],
    ids=["each_year_ranked", "change_per_sector"],
)
def test_a_sector_ranking_sets_beside_another_year_by_a_per_period_rank_or_a_growth(
    decision: DecisionFactory, reader: EvidenceRequirement, rank_decision: bool
) -> None:
    """ "How does that compare with 2019?": a period covering both years, read by a
    rank of each year or by each sector's growth between them."""
    decisions = [decision(), _every_sector(decision)] if rank_decision else [decision()]
    since_2019 = _aid_to_kenya(period=Period(start=2019, end=2024))

    plan = _plan(decisions, [since_2019, _by_sector(), reader])

    (role,) = frame_shapes(plan.evidence_requirements)[reader.id].roles
    assert role.role == "sector"


@pytest.mark.parametrize(
    ("requirements", "refusal"),
    [
        (
            lambda: [_aid_to_kenya(stat_vars=StatVarReading.SUM), _by_sector()],
            "state stat_vars='separate'",
        ),
        (
            lambda: [
                _aid_to_kenya(entities=[Entity(name="Kenya", role="sector")]),
                _by_sector(),
            ],
            "no entity role or `across` may also be 'sector'",
        ),
        (
            lambda: [_aid_to_kenya(across="sector"), _by_sector()],
            "no entity role or `across` may also be 'sector'",
        ),
        (
            lambda: [_aid_to_kenya(breakdown=None), _by_sector()],
            "cannot group by breakdown over a frame that carries none",
        ),
    ],
    ids=["summed", "entity_role_named_sector", "across_sector", "aggregate_without_breakdown"],
)
def test_a_breakdown_the_plan_cannot_carry_out_is_refused(
    decision: DecisionFactory,
    requirements: Callable[[], list[EvidenceRequirement]],
    refusal: str,
) -> None:
    with pytest.raises(ValueError, match=refusal):
        _plan([decision()], requirements())
