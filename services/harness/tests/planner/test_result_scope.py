"""A result kept per member of a party nobody named rests on a decision saying so.

A dataset requirement's `across` reads every donor, and a growth read straight
off it is one growth per donor. `propose_plan` refuses such a plan with
`result_scope_unstated` (`planner.plan.unstated_result_scopes`) until an
aggregate sums the donors away or a decision states per-donor results, and
the plan card says how many figures each answering result gives
(`harness.plan_display.answer_scopes`, at the end of its step's description).
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from google.adk.events import EventActions
from harness.plan_display import answer_scopes, step_rows
from planner.agent import build_planner_tools
from planner.plan import (
    ASKS_ACROSS_YEARS_STATE_KEY,
    ASKS_STATE_KEY,
    AggregateRequirement,
    AnswerMode,
    AskAnswer,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    EvidenceRequirement,
    FilterRequirement,
    GrowthRequirement,
    Period,
    Plan,
    RankRequirement,
    RequirementSide,
    ShareOfRequirement,
    Unit,
    load_plan,
    unstated_result_scopes,
)
from planner.records import (
    DenominatorScope,
    EntityIn,
    GroupBy,
    GrowthChange,
    RankAmong,
    RankDirection,
    Reduction,
)
from planner.shapes import one_figure_remedy, pooling_remedy
from planner.tool_results import Outcome, ProblemCode

from ..decision_drafts import drafts
from ..test_plan_display import _retrieve_then_aggregate_plan

ASK = "how total aid to Ghana changed between 2015 and 2022"

PER_DONOR = Decision(
    slot="per_donor",
    description="Whether the growth is given for each donor or for all donors together.",
    options=["for each donor", "for all donors together"],
    chosen="for each donor",
    resolved_by="planner_default",
    rationale="The question compares donors.",
    fixes=DecisionFixes(across=("donor",)),
)


def _aid(
    decision_ids: tuple[str, ...] = ("measure",), recipients: tuple[str, ...] = ("Ghana",)
) -> DatasetRequirement:
    return DatasetRequirement(
        id="aid",
        decision_ids=decision_ids,
        measure="net ODA",
        entities=tuple(Entity(name=name, role="recipient") for name in recipients),
        across=("donor",),
        period=Period(start=2015, end=2022),
        unit=Unit(currency="USD", prices="current"),
    )


def _growth(input_id: str) -> GrowthRequirement:
    return GrowthRequirement(
        id="growth",
        input_id=input_id,
        window_start=2015,
        window_end=2022,
        change=GrowthChange.RELATIVE,
    )


_TOTAL = AggregateRequirement(
    id="total",
    input_id="aid",
    reduction=Reduction.SUM,
    group_by=(GroupBy.ENTITY, GroupBy.PERIOD),
)


POPULATION = Decision(
    slot="population",
    description="Whether to rank every donor or only the ones named.",
    options=["every donor", "the donors named"],
    chosen="every donor",
    resolved_by="planner_default",
    rationale="The question asks for the top donors among all of them.",
    fixes=DecisionFixes(among=RankAmong.POPULATION),
)


POOLED = Decision(
    slot="pooled",
    description="Whether each donor's share is of its own total or of all donors' total.",
    options=["of all donors' total", "of its own total"],
    chosen="of all donors' total",
    resolved_by="planner_default",
    rationale="The question asks how aid to Ghana splits between donors.",
    fixes=DecisionFixes(denominator_scope=DenominatorScope.POOLED),
)


def _kept(names: tuple[str, ...], role: str | None) -> FilterRequirement:
    return FilterRequirement(
        id="kept", input_id="aid", predicates=(EntityIn(names=names, role=role),)
    )


def _requirements(variant: str) -> list[EvidenceRequirement]:
    """A growth read straight off a per-donor dataset (`per-donor-unstated`), and the ways to settle it.

    The last requirement is the one that answers.
    """
    variants: dict[str, list[EvidenceRequirement]] = {
        "retrieved-per-donor": [_aid()],
        "recipient-filtered-roleless": [
            _aid(recipients=("Ghana", "Kenya")),
            _kept(("Ghana",), None),
            _growth("kept"),
        ],
        "summed": [_aid(), _TOTAL, _growth("total")],
        "per-donor-stated": [_aid(("measure", "per_donor")), _growth("aid")],
        "ranked-over-donors": [
            _aid(),
            RankRequirement(
                id="ranked",
                decision_ids=("population",),
                input_id="aid",
                direction=RankDirection.HIGHEST_FIRST,
                among=RankAmong.POPULATION,
                axis="donor",
                reduction=Reduction.SUM,
                per_period=True,
            ),
        ],
        "share-per-donor": [
            _aid(),
            _TOTAL,
            ShareOfRequirement(
                id="shares",
                decision_ids=("pooled",),
                numerator=RequirementSide(input_id="aid"),
                denominator=RequirementSide(input_id="total"),
                axis="donor",
                denominator_scope=DenominatorScope.POOLED,
            ),
        ],
    }
    return variants.get(variant, [_aid(), _growth("aid")])


def _decisions(variant: str, decision: Callable[..., Decision]) -> list[Decision]:
    extra = {
        "per-donor-stated": [PER_DONOR],
        "ranked-over-donors": [POPULATION],
        "share-per-donor": [POOLED],
    }
    return [decision(), *extra.get(variant, [])]


def _context() -> Any:
    """The part of ADK's `ToolContext` `propose_plan` reads: state, events and actions."""
    return SimpleNamespace(state={}, actions=EventActions(), session=SimpleNamespace(events=[]))


COMPUTED = AnswerMode.COMPUTED
SERIES = AnswerMode.SERIES


@pytest.mark.parametrize(
    ("variant", "mode", "asks", "refused"),
    [
        pytest.param("per-donor-unstated", COMPUTED, [ASK], True, id="answered-per-donor-unstated"),
        pytest.param("per-donor-unstated", COMPUTED, None, True, id="terminal-per-donor-unstated"),
        pytest.param("per-donor-unstated", SERIES, None, True, id="series-terminal-per-donor"),
        pytest.param("retrieved-per-donor", SERIES, None, False, id="series-retrieved-per-donor"),
        pytest.param("summed", COMPUTED, [ASK], False, id="summed-over-donors"),
        pytest.param("per-donor-stated", COMPUTED, [ASK], False, id="per-donor-stated"),
        pytest.param("ranked-over-donors", COMPUTED, [ASK], False, id="ranked-over-donors"),
        pytest.param("share-per-donor", COMPUTED, [ASK], False, id="share-per-donor"),
        pytest.param(
            "recipient-filtered-roleless", COMPUTED, [ASK], True, id="roleless-filter-two-roles"
        ),
    ],
)
def test_a_result_per_donor_rests_on_a_decision_stating_it(
    decision: Callable[..., Decision],
    variant: str,
    mode: AnswerMode,
    asks: list[str] | None,
    refused: bool,
) -> None:
    ctx = _context()
    if asks is not None:
        ctx.state[ASKS_STATE_KEY] = asks
    requirements = _requirements(variant)
    tools = {tool.name: tool for tool in build_planner_tools()}

    result = tools["propose_plan"].func(
        question=ASK,
        answer_mode=mode,
        decisions=drafts(_decisions(variant, decision)),
        evidence_requirements=requirements,
        answered=[AskAnswer(ask=0, outputs=(requirements[-1].id,))] if asks is not None else [],
        tool_context=ctx,
    )

    if not refused:
        assert result["outcome"] == Outcome.OK.value, result
        return
    assert result["outcome"] == Outcome.INVALID_INPUT.value
    assert result["code"] == ProblemCode.RESULT_SCOPE_UNSTATED.value
    assert result["detail"] == {
        "gaps": [
            {
                "requirement_id": "growth",
                "ask": 0 if asks is not None else None,
                "role": "donor",
                "dataset_requirement_id": "aid",
            }
        ]
    }
    assert load_plan(ctx.state) is None


@pytest.mark.parametrize("across_years", [False, True], ids=["unmarked", "marked"])
def test_a_result_per_donor_is_offered_the_donors_pooled_per_year(
    decision: Callable[..., Decision], across_years: bool
) -> None:
    """The refusal offers the donors pooled per year, and no figure over the
    years even for a marked ask: a growth, like every calculation that keeps
    a party open past the shape checks, reads the years apart."""
    ctx = _context()
    ctx.state[ASKS_STATE_KEY] = [ASK]
    ctx.state[ASKS_ACROSS_YEARS_STATE_KEY] = [0] if across_years else []
    requirements = _requirements("per-donor-unstated")
    tools = {tool.name: tool for tool in build_planner_tools()}

    result = tools["propose_plan"].func(
        question=ASK,
        answer_mode=COMPUTED,
        decisions=drafts(_decisions("per-donor-unstated", decision)),
        evidence_requirements=requirements,
        answered=[AskAnswer(ask=0, outputs=(requirements[-1].id,))],
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.RESULT_SCOPE_UNSTATED.value, result
    assert pooling_remedy(one_figure=False) in result["message"]
    assert one_figure_remedy() not in result["message"]


def test_one_refusal_names_every_result_kept_per_donor(
    decision: Callable[..., Decision],
) -> None:
    ctx = _context()
    tools = {tool.name: tool for tool in build_planner_tools()}
    change = _growth("aid").model_copy(update={"id": "change"})

    result = tools["propose_plan"].func(
        question=ASK,
        answer_mode=COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[_aid(), _growth("aid"), change],
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.RESULT_SCOPE_UNSTATED.value
    gaps = result["detail"]["gaps"]
    assert sorted(gap["requirement_id"] for gap in gaps) == ["change", "growth"]
    assert {gap["dataset_requirement_id"] for gap in gaps} == {"aid"}


def test_a_decision_stating_the_same_parties_in_another_order_states_the_scope(
    decision: Callable[..., Decision],
) -> None:
    """`across` names a set of parties, so the order a decision lists them in does not matter."""
    per_pair = PER_DONOR.model_copy(update={"fixes": DecisionFixes(across=("donor", "recipient"))})
    aid = _aid(("measure", "per_donor"), recipients=()).model_copy(
        update={"across": ("recipient", "donor")}
    )
    stated = Plan(
        question=ASK,
        answer_mode=COMPUTED,
        decisions=[decision(), per_pair],
        evidence_requirements=[aid, _growth("aid")],
    )
    unstated = stated.model_copy(
        update={
            "decisions": (decision(),),
            "evidence_requirements": (
                aid.model_copy(update={"decision_ids": ("measure",)}),
                _growth("aid"),
            ),
        }
    )

    assert [gap.role for gap in unstated_result_scopes(unstated)] == ["recipient", "donor"]
    assert unstated_result_scopes(stated) == []


@pytest.mark.parametrize(
    ("variant", "requirement_id", "phrase"),
    [
        ("per-donor-stated", "growth", "one figure per donor, 2015-2022"),
        ("summed", "growth", "one figure, 2015-2022"),
        ("summed", "aid", "one figure per donor per year, 2015-2022"),
    ],
)
def test_the_plan_card_says_how_many_figures_a_result_gives(
    decision: Callable[..., Decision], variant: str, requirement_id: str, phrase: str
) -> None:
    plan = Plan(
        question=ASK,
        answer_mode=AnswerMode.COMPUTED,
        decisions=tuple(_decisions(variant, decision)),
        evidence_requirements=tuple(_requirements(variant)),
        asks=(ASK,),
        answered=(AskAnswer(ask=0, outputs=(requirement_id,)),),
    )

    assert answer_scopes(plan)[requirement_id] == phrase


def test_the_step_whose_result_answers_states_its_scope_on_the_card(
    decision: Callable[..., Decision],
) -> None:
    rows = {row["id"]: row for row in step_rows(_retrieve_then_aggregate_plan(decision))}

    assert "Result:" not in str(rows["annual_debt_service"]["description"])
    assert str(rows["annual_totals"]["description"]).endswith(
        " Result: one figure per year, 2015-2024."
    )
