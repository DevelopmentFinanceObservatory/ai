"""A replan for the option the user picked on the plan card holds that pick.

The planner seam records the pick for the entry (`planner.plan.
DECISION_PICK_STATE_KEY`), with the decision as the card showed it.
`propose_plan` then refuses a plan whose decision under the pick's slot
states another option (`submission.service._picked_decision`) or whose
requirements citing it fix nothing or what they fixed on the card, and
records that decision as the user's. The pick lets go of another user
decision only when the picked option sets every field that decision fixed,
every requirement that cited it also cited the picked decision, and the new
plan drops them all (`submission.service._released_by_pick`).
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
    DecisionPick,
    Entity,
    Period,
    Plan,
    RankRequirement,
    ResolvedBy,
    Unit,
    load_plan,
    save_decision_pick,
)
from planner.records import RankAmong, RankDirection, Reduction
from planner.tool_results import Outcome, ProblemCode
from submission.service import _released_by_pick

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
# The pick of the top 10 donors on the card showing the DAC total.
PICK = DecisionPick(
    slot="donor_scope",
    chosen="the top 10 donors",
    origin_chosen="the DAC total",
    origin_fixes=DONOR_SCOPE.fixes,
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
    """The pick moves the DAC total to a rank of the top 10 over every donor,
    read over the user's year, and the picked decision is theirs."""
    ctx = _context()
    first = _recorded_dac_total(ctx, decision)
    save_decision_pick(ctx.state, PICK)

    result = _propose_plan(
        ctx,
        decisions=drafts([TOP_TEN]),
        evidence_requirements=_year_kept(YEAR),
        removed_evidence_requirements=["dac"],
        amends_version=first["version"],
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    picked = next(d for d in plan.decisions if d.slot == "donor_scope")
    assert (picked.chosen, picked.resolved_by) == ("the top 10 donors", ResolvedBy.USER)
    assert picked.fixes == DecisionFixes(limit=10, among=RankAmong.POPULATION)
    assert next(d for d in plan.decisions if d.slot == "year") == USER_YEAR
    assert [r.id for r in plan.evidence_requirements] == ["aid", "top"]


@pytest.mark.parametrize(
    ("patch", "stated"),
    [
        pytest.param({"decisions": [DONOR_SCOPE]}, "the DAC total", id="restated"),
        pytest.param({"decisions": []}, "the DAC total", id="carried"),
        pytest.param({"removed_decisions": ["donor_scope"]}, None, id="removed"),
        pytest.param(
            {"decisions": [DONOR_SCOPE.model_copy(update={"chosen": "the top 10 donors"})]},
            "the top 10 donors",
            id="relabelled-over-the-same-requirements",
        ),
    ],
)
def test_a_plan_that_does_not_state_the_pick_is_refused(
    decision: Callable[..., Decision], patch: dict[str, Any], stated: str | None
) -> None:
    ctx = _context()
    first = _recorded_dac_total(ctx, decision)
    save_decision_pick(ctx.state, PICK)
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


def test_a_restated_pick_its_requirements_fix_nothing_of_is_refused_saying_so(
    decision: Callable[..., Decision],
) -> None:
    """The plan states the picked option, cited only by the DAC total, which
    states neither a rank's limit nor what it ranks among: the decision fixes
    nothing, and the refusal says the requirements citing it fix none of
    what it names."""
    ctx = _context()
    first = _recorded_dac_total(ctx, decision)
    save_decision_pick(ctx.state, PICK)

    result = _propose_plan(
        ctx,
        decisions=drafts([TOP_TEN]),
        evidence_requirements=[_dac_total()],
        amends_version=first["version"],
    )

    assert result["code"] == ProblemCode.USER_PICK_NOT_HELD.value, result
    assert "fix none of what it names" in result["message"]


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
    save_decision_pick(ctx.state, PICK)

    result = _propose_plan(
        ctx,
        decisions=drafts([TOP_TEN]),
        amends_version=first["version"],
        **patch,
    )

    assert result["outcome"] == Outcome.REFUSED.value, result
    assert result["code"] == code.value
    assert result["detail"]["slot"] == "year"


def test_a_pick_releases_no_user_decision_of_a_requirement_it_did_not_change(
    decision: Callable[..., Decision],
) -> None:
    """The user's year rests on a requirement that does not cite the picked
    decision. A replan that renames that requirement and drops the year from
    it is refused, though no requirement that cited the year survives by id."""
    ctx = _context()
    gross = _dac_total().model_copy(
        update={"id": "gross", "decision_ids": ("measure", "year"), "measure": "gross ODA"}
    )
    first = _propose_plan(
        ctx,
        question="how much aid does Kenya get",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision(), DONOR_SCOPE, USER_YEAR]),
        evidence_requirements=[
            _dac_total().model_copy(update={"decision_ids": ("measure", "donor_scope")}),
            gross,
        ],
    )
    assert first["outcome"] == Outcome.OK.value, first
    ctx.session.events = [response_event("propose_plan", first)]
    save_decision_pick(ctx.state, PICK)

    result = _propose_plan(
        ctx,
        decisions=drafts([TOP_TEN]),
        evidence_requirements=[
            *_ranked_donors(),
            gross.model_copy(update={"id": "gross_renamed", "decision_ids": ("measure",)}),
        ],
        removed_evidence_requirements=["dac", "gross"],
        amends_version=first["version"],
    )

    assert result["outcome"] == Outcome.REFUSED.value, result
    assert result["code"] == ProblemCode.USER_DECISION_DROPPED.value
    assert result["detail"]["slot"] == "year"


def test_a_pick_releases_no_user_decision_no_requirement_cited() -> None:
    """A user decision no requirement of the plan on record cites rests on
    nothing the pick changed, so the pick does not let it go."""
    previous = Plan.model_construct(
        decisions=(DONOR_SCOPE, USER_YEAR),
        evidence_requirements=(
            _dac_total().model_copy(update={"decision_ids": ("measure", "donor_scope")}),
        ),
    )

    released = _released_by_pick(previous, _ranked_donors(), slot="donor_scope", picked=TOP_TEN)

    assert released == {"donor_scope"}


@pytest.mark.parametrize(
    ("fixes", "released"),
    [
        pytest.param(DecisionFixes(limit=5), {"donor_scope", "size"}, id="settled-by-the-pick"),
        pytest.param(
            DecisionFixes(limit=5, period=YEAR), {"donor_scope"}, id="fixing-more-than-the-pick"
        ),
        pytest.param(None, {"donor_scope"}, id="fixing-nothing"),
    ],
)
def test_a_pick_releases_a_user_decision_only_when_the_picked_option_sets_all_it_fixed(
    fixes: DecisionFixes | None, released: set[str]
) -> None:
    """Every requirement citing the user's decision also cites the picked one,
    and the new plan drops them all. The pick lets it go only when the
    picked option, as the proposal states it, sets every field it fixed."""
    size = USER_YEAR.model_copy(update={"slot": "size", "fixes": fixes})
    previous = Plan.model_construct(
        decisions=(DONOR_SCOPE, size),
        evidence_requirements=(
            _dac_total().model_copy(update={"decision_ids": ("measure", "donor_scope", "size")}),
        ),
    )

    assert (
        _released_by_pick(previous, _ranked_donors(), slot="donor_scope", picked=TOP_TEN)
        == released
    )


def test_a_pick_releases_no_user_decision_it_does_not_settle(
    decision: Callable[..., Decision],
) -> None:
    """The pick fixes a rank's limit and population; the user's year fixes a
    period. A replan that drops the DAC total, and with it the only citation
    of the year, while reading the donors over 2022 is refused: the pick
    does not settle the period the user decided."""
    ctx = _context()
    first = _recorded_dac_total(ctx, decision)
    save_decision_pick(ctx.state, PICK)
    aid, top = _ranked_donors()

    result = _propose_plan(
        ctx,
        decisions=drafts([TOP_TEN]),
        evidence_requirements=[
            aid.model_copy(update={"period": Period(start=2022, end=2022)}),
            top,
        ],
        removed_evidence_requirements=["dac"],
        amends_version=first["version"],
    )

    assert result["outcome"] == Outcome.REFUSED.value, result
    assert result["code"] == ProblemCode.USER_DECISION_DROPPED.value
    assert result["detail"]["slot"] == "year"


@pytest.mark.parametrize(
    "restated",
    [
        pytest.param(
            DONOR_SCOPE.model_copy(update={"chosen": "the top 10 donors", "fixes": None}),
            id="fixing-nothing",
        ),
        pytest.param(
            DONOR_SCOPE.model_copy(update={"chosen": "the top 10 donors"}),
            id="fixing-what-the-dac-total-fixed",
        ),
    ],
)
def test_a_later_proposal_of_the_pick_replan_holds_the_pick_against_the_plan_picked_on(
    decision: Callable[..., Decision], restated: Decision
) -> None:
    """A draft holding the pick is admitted. A later proposal of the same
    replan that returns the DAC total under the picked label is refused,
    though the plan it amends already states that label."""
    ctx = _context()
    first = _recorded_dac_total(ctx, decision)
    save_decision_pick(ctx.state, PICK)
    held = _propose_plan(
        ctx,
        decisions=drafts([TOP_TEN]),
        evidence_requirements=_year_kept(YEAR),
        removed_evidence_requirements=["dac"],
        amends_version=first["version"],
    )
    assert held["outcome"] == Outcome.OK.value, held
    ctx.session.events = [*ctx.session.events, response_event("propose_plan", held)]

    result = _propose_plan(
        ctx,
        decisions=drafts([restated]),
        evidence_requirements=[_dac_total()],
        removed_evidence_requirements=["aid", "top"],
        amends_version=held["version"],
    )

    assert result["outcome"] == Outcome.REFUSED.value, result
    assert result["code"] == ProblemCode.USER_PICK_NOT_HELD.value
    recorded = load_plan(ctx.state)
    assert recorded is not None and recorded.version == held["version"]
