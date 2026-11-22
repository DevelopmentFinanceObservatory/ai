"""A rank stating no reduction over a frame of several roles reads one value per entity only where each other role is one party.

"The top ten recipients of IDA's disbursements in 2022" retrieves IDA as the
donor beside every recipient. Each recipient holds one row in 2022, so the
rank needs no reduction. `planner.plan._rank_reduction_reason` admits the
rank where every role but its axis names one entity (`RoleDimension.single`),
and `propose_plan` reads the pinned source to check that the name binds one
party alone and that no other axis is open
(`submission.service._rank_roles_expand`). The blocks are the ones Search
returned for the planner entry eval's tasks.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from planner.plan import (
    ASKS_STATE_KEY,
    AnswerMode,
    AskAnswer,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    Period,
    PinSource,
    RankRequirement,
    load_plan,
    save_pin_sources,
)
from planner.records import RankAmong, RankDirection
from planner.specs import planner_tool_specs
from planner.tool_results import Outcome, ProblemCode

from ..decision_drafts import drafts
from ..planner_tools import call_planner_tool

_RECORDINGS = Path(__file__).parents[4] / "evals" / "planner_entry" / "recordings"

ASK = "the top recipients"

POPULATION = Decision(
    slot="population",
    description="Whether to rank every recipient or only the ones named.",
    options=["every recipient", "the recipients named"],
    chosen="every recipient",
    resolved_by="planner_default",
    rationale="The question asks for the top recipients among all of them.",
    fixes=DecisionFixes(among=RankAmong.POPULATION),
)


def _block(task: str) -> dict[str, Any]:
    """The spec block of the one search part the task's recording holds."""
    recording = json.loads((_RECORDINGS / f"{task}.json").read_text())
    (search,) = recording["searches"]
    (part,) = search["envelope"]["parts"]
    return part["spec"]


def _dataset(
    block: dict[str, Any],
    entities: tuple[Entity, ...],
    *,
    across: tuple[str, ...] = (),
    year: int,
) -> DatasetRequirement:
    return DatasetRequirement(
        id="flows",
        decision_ids=("measure",),
        measure="gross disbursements",
        entities=entities,
        across=across,
        period=Period(start=year, end=year),
        source_spec_id=block["spec_id"],
    )


def _propose(
    block: dict[str, Any],
    dataset: DatasetRequirement,
    axis: str,
    decision: Callable[..., Decision],
) -> tuple[dict[str, Any], Any]:
    state: dict[str, Any] = {ASKS_STATE_KEY: [ASK]}
    save_pin_sources(state, {block["spec_id"]: PinSource(block=block, settled=True)})
    ctx = SimpleNamespace(state=state, function_call_id=None, run_ended=False, records=[])
    rank = RankRequirement(
        id="ranked",
        decision_ids=("population",),
        input_id=dataset.id,
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis=axis,
        limit=10,
    )
    tools = {tool.name: tool for tool in planner_tool_specs()}
    result = call_planner_tool(
        tools["propose_plan"],
        question=ASK,
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision(), POPULATION]),
        evidence_requirements=[dataset, rank],
        answered=[AskAnswer(ask=0, outputs=("ranked",))],
        tool_context=ctx,
    )
    return result, ctx


def test_a_rank_beside_one_donor_over_one_year_needs_no_reduction(
    decision: Callable[..., Decision],
) -> None:
    """IDA binds one party on the donor column, and the recipients are the
    open axis the rank orders."""
    block = _block("rank-top-ida-recipients-2022")
    ida = Entity(name="International Development Association", role="donor")

    result, ctx = _propose(
        block, _dataset(block, (ida,), across=("recipient",), year=2022), "recipient", decision
    )

    assert result["outcome"] == Outcome.OK.value, result
    recorded = load_plan(ctx.state)
    assert recorded is not None
    assert recorded.requirement("ranked").reduction is None


@pytest.mark.parametrize(
    ("task", "entities", "across", "axis", "year", "held"),
    [
        pytest.param(
            "rank-dac-humanitarian-2023",
            (Entity(name="DAC countries", role="donor"),),
            ("recipient",),
            "recipient",
            2023,
            "binds 32 parties",
            id="a-donor-group-read-as-its-members",
        ),
        pytest.param(
            "aggregate-two-donors-yearly",
            (
                Entity(name="United Kingdom", role="donor"),
                Entity(name="Kenya", role="recipient"),
            ),
            (),
            "recipient",
            2022,
            "binds 2 parties on its column",
            id="a-source-binding-another-donor-beside-the-one-named",
        ),
        pytest.param(
            "rank-top-donors-malawi",
            (
                Entity(name="Malawi", role="recipient"),
                Entity(name="United Kingdom", role="donor"),
                Entity(name="United States", role="donor"),
            ),
            (),
            "donor",
            2022,
            "leaves an axis open",
            id="every-donor-left-open",
        ),
    ],
)
def test_a_rank_beside_a_role_holding_several_parties_states_a_reduction(
    decision: Callable[..., Decision],
    task: str,
    entities: tuple[Entity, ...],
    across: tuple[str, ...],
    axis: str,
    year: int,
    held: str,
) -> None:
    """One name on the other role, but Search reads "DAC countries" as its 32
    members, and the Malawi source holds every donor open beside the two the
    rank orders. Each ranked entity keeps several rows."""
    block = _block(task)

    result, ctx = _propose(
        block, _dataset(block, entities, across=across, year=year), axis, decision
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value, result
    assert held in result["message"]
    assert load_plan(ctx.state) is None
