"""A replan for the option the user picked on the plan card holds that pick.

The planner seam records the pick for the entry (`planner.plan.
DECISION_PICK_STATE_KEY`). `propose_plan` then refuses a plan whose decision
under the pick's slot states another option (`submission.service.
_picked_decision`), and records that decision as the user's, whatever the
requirements citing it fix. The pick lets go of another user decision
only when the new plan drops it with every requirement that cited it
(`submission.service._released_by_pick`).
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from google.adk.events import EventActions
from planner.agent import build_planner_tools
from planner.plan import (
    ASKS_STATE_KEY,
    AnswerMode,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    Period,
    RankRequirement,
    ResolvedBy,
    Unit,
    load_plan,
    save_decision_pick,
)
from planner.records import RankAmong, RankDirection, Reduction
from planner.tool_results import Outcome, ProblemCode

from ..decision_drafts import drafts
from ..session_events import response_event

DAC_TOTAL = "spec_" + "d" * 32
EVERY_DONOR = "spec_" + "e" * 32
YEAR = Period(start=2023, end=2023)
USD = Unit(currency="USD", prices="current")

DONOR_SCOPE = Decision(
    slot="donor_scope",
    description="Whether to count the DAC total or rank the donors.",
    options=["the DAC total", "the top 10 donors"],
    chosen="the DAC total",
    resolved_by=ResolvedBy.PLANNER_DEFAULT,
    rationale="The question names no donor.",
    fixes=DecisionFixes(
        entities=(
            Entity(name="DAC countries", role="donor"),
            Entity(name="Kenya", role="recipient"),
        )
    ),
)
# The user's own decision, which only the DAC total cites: a plan that ranks
# the donors leaves it with no citation.
USER_YEAR = Decision(
    slot="year",
    description="Which year the figure reads.",
    options=["2023", "2022"],
    chosen="2023",
    resolved_by=ResolvedBy.USER,
    rationale="The user asked for 2023.",
    fixes=DecisionFixes(period=YEAR),
)
TOP_TEN = DONOR_SCOPE.model_copy(
    update={
        "chosen": "the top 10 donors",
        "fixes": DecisionFixes(limit=10, among=RankAmong.POPULATION),
    }
)


def _dac_total() -> DatasetRequirement:
    return DatasetRequirement(
        id="dac",
        decision_ids=("measure", "donor_scope", "year"),
        measure="net ODA",
        entities=(
            Entity(name="DAC countries", role="donor"),
            Entity(name="Kenya", role="recipient"),
        ),
        period=YEAR,
        unit=USD,
        source_spec_id=DAC_TOTAL,
    )


def _ranked_donors() -> list[Any]:
    """The top 10 donors to Kenya over every donor, the plan the pick asks for."""
    return [
        DatasetRequirement(
            id="aid",
            decision_ids=("measure",),
            measure="net ODA",
            entities=(Entity(name="Kenya", role="recipient"),),
            across=("donor",),
            period=YEAR,
            unit=USD,
            source_spec_id=EVERY_DONOR,
        ),
        RankRequirement(
            id="top",
            decision_ids=("donor_scope",),
            input_id="aid",
            direction=RankDirection.HIGHEST_FIRST,
            among=RankAmong.POPULATION,
            axis="donor",
            reduction=Reduction.SUM,
            limit=10,
        ),
    ]


def _context() -> Any:
    """The part of ADK's `ToolContext` `propose_plan` reads, holding no asks."""
    return SimpleNamespace(
        state={ASKS_STATE_KEY: []}, actions=EventActions(), session=SimpleNamespace(events=[])
    )


def _propose_plan(ctx: Any, **arguments: Any) -> dict[str, Any]:
    tools = {tool.name: tool for tool in build_planner_tools(reads_user_words=False)}
    return tools["propose_plan"].func(**arguments, tool_context=ctx)


def _recorded_dac_total(ctx: Any, decision: Callable[..., Decision]) -> dict[str, Any]:
    """Record the DAC-total plan, with the response the session holds for it,
    so the user has said nothing since."""
    first = _propose_plan(
        ctx,
        question="how much aid does Kenya get",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision(), DONOR_SCOPE, USER_YEAR]),
        evidence_requirements=[_dac_total()],
    )
    assert first["outcome"] == Outcome.OK.value, first
    ctx.session.events = [response_event("propose_plan", first)]
    return first


def test_a_pick_that_changes_the_plans_shape_is_admitted_as_the_users(
    decision: Callable[..., Decision],
) -> None:
    """The pick moves the DAC total to a rank of the top 10 over every donor.
    The user's year decision, which only the DAC total cited, goes with it,
    and the picked decision is theirs."""
    ctx = _context()
    first = _recorded_dac_total(ctx, decision)
    save_decision_pick(ctx.state, ("donor_scope", "the top 10 donors"))

    result = _propose_plan(
        ctx,
        decisions=drafts([TOP_TEN]),
        evidence_requirements=_ranked_donors(),
        removed_evidence_requirements=["dac"],
        amends_version=first["version"],
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    picked = next(d for d in plan.decisions if d.slot == "donor_scope")
    assert (picked.chosen, picked.resolved_by) == ("the top 10 donors", ResolvedBy.USER)
    assert picked.fixes == DecisionFixes(limit=10, among=RankAmong.POPULATION)
    assert "year" not in {d.slot for d in plan.decisions}
    assert [r.id for r in plan.evidence_requirements] == ["aid", "top"]


@pytest.mark.parametrize(
    ("patch", "stated"),
    [
        pytest.param({"decisions": [DONOR_SCOPE]}, "the DAC total", id="restated"),
        pytest.param({"decisions": []}, "the DAC total", id="carried"),
        pytest.param({"removed_decisions": ["donor_scope"]}, None, id="removed"),
    ],
)
def test_a_plan_that_does_not_state_the_pick_is_refused(
    decision: Callable[..., Decision], patch: dict[str, Any], stated: str | None
) -> None:
    ctx = _context()
    first = _recorded_dac_total(ctx, decision)
    save_decision_pick(ctx.state, ("donor_scope", "the top 10 donors"))
    requirements = (
        [_dac_total().model_copy(update={"decision_ids": ("measure", "year")})]
        if "removed_decisions" in patch
        else [_dac_total()]
    )

    result = _propose_plan(
        ctx,
        decisions=drafts(patch.get("decisions", [])),
        removed_decisions=patch.get("removed_decisions", []),
        evidence_requirements=requirements,
        amends_version=first["version"],
    )

    assert result["outcome"] == Outcome.REFUSED.value, result
    assert result["code"] == ProblemCode.USER_PICK_NOT_HELD.value
    assert result["detail"] == {
        "slot": "donor_scope",
        "chosen": "the top 10 donors",
        "stated": stated,
    }
    recorded = load_plan(ctx.state)
    assert recorded is not None and recorded.version == first["version"]


def _year_kept(period: Period) -> list[Any]:
    """`_ranked_donors` with the donors read over `period` under the user's year decision."""
    aid, top = _ranked_donors()
    return [aid.model_copy(update={"decision_ids": ("measure", "year"), "period": period}), top]


@pytest.mark.parametrize(
    ("patch", "code"),
    [
        pytest.param(
            {
                "evidence_requirements": _year_kept(Period(start=2022, end=2022)),
                "removed_evidence_requirements": ["dac"],
            },
            ProblemCode.USER_DECISION_CONTRADICTED,
            id="still-cited-and-changed",
        ),
        pytest.param(
            {
                "evidence_requirements": [
                    _dac_total().model_copy(update={"decision_ids": ("measure",)}),
                    *_ranked_donors(),
                ]
            },
            ProblemCode.USER_DECISION_DROPPED,
            id="uncited-on-a-surviving-requirement",
        ),
    ],
)
def test_a_pick_releases_no_user_decision_the_new_plan_still_rests_on(
    decision: Callable[..., Decision], patch: dict[str, Any], code: ProblemCode
) -> None:
    """The pick speaks for its own slot. Another user decision the new plan
    still cites, or whose requirement it keeps, stands as the user made it."""
    ctx = _context()
    first = _recorded_dac_total(ctx, decision)
    save_decision_pick(ctx.state, ("donor_scope", "the top 10 donors"))

    result = _propose_plan(
        ctx,
        decisions=drafts([TOP_TEN]),
        amends_version=first["version"],
        **patch,
    )

    assert result["outcome"] == Outcome.REFUSED.value, result
    assert result["code"] == code.value
    assert result["detail"]["slot"] == "year"
