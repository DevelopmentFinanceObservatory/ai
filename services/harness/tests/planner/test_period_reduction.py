"""A step turns several years into one figure only for an ask the orchestrator marked across years.

A dataset requirement's `period` is the window of the search that found its
source, which for a question naming no years is every year the source holds.
`propose_plan` refuses an `aggregate` without `period` in its `group_by`, or a
`rank` with a `reduction`, over such years with `period_reduction_unstated`
(`planner.plan.unstated_period_reductions`) unless every answering result
computed from the step answers asks across years, and only those. In an entry
the user's amendment opened, a decision of theirs cited on the step admits it
too.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from data_mcp.definition_store import canonicalize, spec_id_for
from google.adk.events import EventActions
from harness.planner_seam import _reading_pin_sources
from harness.turn_records import RecordedIdentity
from planner.agent import build_planner_tools
from planner.plan import (
    AMENDMENT_ENTRY_STATE_KEY,
    ASKS_ACROSS_YEARS_STATE_KEY,
    ASKS_STATE_KEY,
    AggregateRequirement,
    AnswerMode,
    AskAnswer,
    Breakdown,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    EvidenceRequirement,
    FilterRequirement,
    GrowthRequirement,
    Period,
    PinSource,
    Plan,
    QuotientRequirement,
    RankRequirement,
    RequirementSide,
    ResolvedBy,
    ShareOfRequirement,
    SumOfRequirement,
    Unit,
    current_recorded_identity,
    load_plan,
    save_pin_sources,
    unstated_period_reductions,
)
from planner.records import (
    BreakdownDimension,
    DenominatorScope,
    EntityIn,
    GroupBy,
    GrowthChange,
    PeriodThreshold,
    RankAmong,
    RankDirection,
    Reduction,
    RelativePeriod,
    RollingWindow,
)
from planner.tool_results import Outcome, ProblemCode

from ..decision_drafts import drafts

ASK = "the donors that give the most aid to Kenya"
OTHER_ASK = "how aid to Kenya has moved"
SPEC = "spec_" + "a" * 32
EVERY_YEAR = Period(start=1990, end=2024)

POPULATION = Decision(
    slot="population",
    description="Whether to rank every donor or only the ones named.",
    options=["every donor", "the donors named"],
    chosen="every donor",
    resolved_by="planner_default",
    rationale="The question asks for the top donors among all of them.",
    fixes=DecisionFixes(among=RankAmong.POPULATION),
)
ONE_TOTAL = Decision(
    slot="one_total",
    description="Whether aid is totalled over every year or kept per year.",
    options=["one total over every year", "a total per year"],
    chosen="one total over every year",
    resolved_by="planner_default",
    rationale="The question asks how much aid Kenya has received in all.",
    fixes=DecisionFixes(group_by=(GroupBy.ENTITY,)),
)
ONE_RANKING = Decision(
    slot="one_ranking",
    description="Whether donors are ranked on every year together or on one year.",
    options=["every year together", "the newest year"],
    chosen="every year together",
    resolved_by="planner_default",
    rationale="The question asks who has given the most in all.",
    fixes=DecisionFixes(per_period=False),
)
ONE_YEAR = Decision(
    slot="one_year",
    description="Which year the total reads.",
    options=["the newest year", "every year"],
    chosen="the newest year",
    resolved_by="planner_default",
    rationale="The question names no year.",
    fixes=DecisionFixes(period=Period(start=2024, end=2024)),
)
YEARS_READ = ONE_YEAR.model_copy(
    update={"slot": "years_read", "fixes": DecisionFixes(period=EVERY_YEAR)}
)
SUMMED = Decision(
    slot="summed",
    description="Whether the years are summed or averaged.",
    options=["summed", "averaged"],
    chosen="summed",
    resolved_by="planner_default",
    rationale="Aid adds up over years.",
    fixes=DecisionFixes(reduction=Reduction.SUM),
)
USER_TOTAL = ONE_TOTAL.model_copy(update={"resolved_by": ResolvedBy.USER})
USER_RANKING = ONE_RANKING.model_copy(update={"resolved_by": ResolvedBy.USER})


def _aid(
    *,
    period: Period = EVERY_YEAR,
    pin: str | None = SPEC,
    decision_ids: tuple[str, ...] = ("measure",),
) -> DatasetRequirement:
    return DatasetRequirement(
        id="aid",
        decision_ids=decision_ids,
        measure="net ODA",
        entities=(Entity(name="Kenya", role="recipient"),),
        across=("donor",),
        period=period,
        unit=Unit(currency="USD", prices="current"),
        source_spec_id=pin,
    )


def _total(**overrides: Any) -> AggregateRequirement:
    fields: dict[str, Any] = {
        "id": "result",
        "input_id": "aid",
        "reduction": Reduction.SUM,
        "group_by": (GroupBy.ENTITY,),
    }
    return AggregateRequirement(**(fields | overrides))


def _ranked(**overrides: Any) -> RankRequirement:
    fields: dict[str, Any] = {
        "id": "result",
        "decision_ids": ("population",),
        "input_id": "aid",
        "direction": RankDirection.HIGHEST_FIRST,
        "among": RankAmong.POPULATION,
        "axis": "donor",
        "reduction": Reduction.SUM,
    }
    return RankRequirement(**(fields | overrides))


def _context(
    *,
    asks: Sequence[str] = (ASK,),
    across_years: Sequence[int] = (),
    amendment: bool | None = None,
    pins: dict[str, PinSource] | None = None,
) -> Any:
    """The part of ADK's `ToolContext` `propose_plan` reads.

    `across_years` numbers the asks the orchestrator marked, and
    `amendment` is what the seam recorded of the entry, `None` for a session
    it recorded nothing in.
    """
    state: dict[str, Any] = {
        ASKS_STATE_KEY: list(asks),
        ASKS_ACROSS_YEARS_STATE_KEY: list(across_years),
    }
    if amendment is not None:
        state[AMENDMENT_ENTRY_STATE_KEY] = amendment
    if pins is not None:
        save_pin_sources(state, pins)
    return SimpleNamespace(state=state, actions=EventActions(), session=SimpleNamespace(events=[]))


def _propose(
    ctx: Any,
    requirements: list[EvidenceRequirement],
    decisions: list[Decision],
    *,
    answered: Sequence[AskAnswer] | None = None,
) -> dict[str, Any]:
    """Propose `requirements`, the last of which answers ask 0 unless `answered` says otherwise.

    A plan on record in `ctx` is the one the call amends.
    """
    tools = {tool.name: tool for tool in build_planner_tools()}
    recorded = load_plan(ctx.state)
    return tools["propose_plan"].func(
        question=ASK,
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts(decisions),
        evidence_requirements=requirements,
        answered=(
            [AskAnswer(ask=0, outputs=(requirements[-1].id,))] if answered is None else answered
        ),
        amends_version=recorded.version if recorded is not None else None,
        tool_context=ctx,
    )


def _gap(
    *, step: str = "result", output: str = "result", asks: Sequence[int] = (0,), years: Period
) -> dict[str, Any]:
    return {
        "requirement_id": step,
        "output_id": output,
        "asks": list(asks),
        "years": {"start": years.start, "end": years.end},
        "from_first_year": False,
        "follows_newest": False,
        "until": None,
        "dataset_requirement_id": "aid",
    }


def _refused(result: dict[str, Any], *gaps: dict[str, Any]) -> None:
    assert result["outcome"] == Outcome.INVALID_INPUT.value, result
    assert result["code"] == ProblemCode.PERIOD_REDUCTION_UNSTATED.value, result
    assert result["detail"] == {"gaps": list(gaps)}


_STEPS = {
    "aggregate": (_total, [], USER_TOTAL, {"decision_ids": ("one_total",)}),
    "rank": (
        _ranked,
        [POPULATION],
        USER_RANKING,
        {"decision_ids": ("population", "one_ranking")},
    ),
}


@pytest.mark.parametrize("step", sorted(_STEPS))
def test_one_figure_over_years_answers_an_ask_across_years(
    decision: Callable[..., Decision], step: str
) -> None:
    build, cited, _, _ = _STEPS[step]
    requirements = [_aid(), build()]

    unmarked = _context()
    refused = _propose(unmarked, requirements, [decision(), *cited])
    marked = _context(across_years=[0])
    admitted = _propose(marked, requirements, [decision(), *cited])

    _refused(refused, _gap(years=EVERY_YEAR))
    assert "1990-2024" in refused["message"]
    assert load_plan(unmarked.state) is None
    assert admitted["outcome"] == Outcome.OK.value, admitted
    recorded = load_plan(marked.state)
    assert recorded is not None and recorded.asks_across_years == (0,)


@pytest.mark.parametrize("step", sorted(_STEPS))
def test_a_plan_with_no_asks_reduces_no_years(decision: Callable[..., Decision], step: str) -> None:
    build, cited, _, _ = _STEPS[step]

    result = _propose(_context(asks=()), [_aid(), build()], [decision(), *cited], answered=())

    _refused(result, _gap(asks=(), years=EVERY_YEAR))


def _top(input_id: str) -> RankRequirement:
    """A ranking of the one figure per donor `input_id` holds."""
    return RankRequirement(
        id="top",
        decision_ids=("population",),
        input_id=input_id,
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
    )


@pytest.mark.parametrize(
    ("answered", "asks"),
    [
        pytest.param(
            [AskAnswer(ask=0, outputs=("total",)), AskAnswer(ask=1, outputs=("top",))],
            (1,),
            id="a-later-result-answers-an-unmarked-ask",
        ),
        pytest.param(
            [AskAnswer(ask=0, outputs=("top",)), AskAnswer(ask=1, outputs=("top",))],
            (1,),
            id="one-result-answers-a-marked-and-an-unmarked-ask",
        ),
    ],
)
def test_every_answering_result_computed_from_the_step_answers_asks_across_years(
    decision: Callable[..., Decision], answered: list[AskAnswer], asks: tuple[int, ...]
) -> None:
    """Ask 0 is across years and ask 1 is not. One total over the years
    answers ask 0, and the ranking computed from it does not ride along."""
    requirements = [_aid(), _total(id="total"), _top("total")]
    ctx = _context(asks=(ASK, OTHER_ASK), across_years=[0])

    result = _propose(ctx, requirements, [decision(), POPULATION], answered=answered)

    _refused(result, _gap(step="total", output="top", asks=asks, years=EVERY_YEAR))


def test_one_refusal_names_every_reducing_step_and_the_result_it_feeds(
    decision: Callable[..., Decision],
) -> None:
    requirements = [_aid(), _total(id="total"), _top("total"), _ranked(id="ranked")]

    result = _propose(
        _context(),
        requirements,
        [decision(), POPULATION],
        answered=[AskAnswer(ask=0, outputs=("top", "ranked"))],
    )

    _refused(
        result,
        _gap(step="total", output="top", years=EVERY_YEAR),
        _gap(step="ranked", output="ranked", years=EVERY_YEAR),
    )


@pytest.mark.parametrize(
    ("requirements", "decisions"),
    [
        pytest.param(
            [_aid(), _total(group_by=(GroupBy.ENTITY, GroupBy.PERIOD))],
            [],
            id="aggregate-keeps-the-years",
        ),
        pytest.param(
            [
                _aid(period=Period(start=2024, end=2024), decision_ids=("measure", "one_year")),
                _total(),
            ],
            [ONE_YEAR],
            id="aggregate-reads-one-year",
        ),
        pytest.param(
            [_aid(), _ranked(per_period=True)],
            [POPULATION],
            id="rank-keeps-the-years",
        ),
        pytest.param(
            [_aid(), _ranked(window_start=2024, window_end=2024)],
            [POPULATION],
            id="rank-reads-one-year",
        ),
    ],
)
def test_each_remedy_the_refusal_names_is_admitted(
    decision: Callable[..., Decision],
    requirements: list[EvidenceRequirement],
    decisions: list[Decision],
) -> None:
    ctx = _context()

    result = _propose(ctx, requirements, [decision(), *decisions])

    assert result["outcome"] == Outcome.OK.value, result
    recorded = load_plan(ctx.state)
    assert recorded is not None
    assert recorded.evidence_requirements[-1] == requirements[-1]


@pytest.mark.parametrize(
    ("requirements", "decisions", "amendment"),
    [
        pytest.param(
            [_aid(), _total(decision_ids=("one_total",))], [ONE_TOTAL], False, id="aggregate"
        ),
        pytest.param(
            [
                _aid(),
                _ranked(
                    decision_ids=("population", "one_ranking"), window_start=1990, window_end=2024
                ),
            ],
            [POPULATION, ONE_RANKING],
            False,
            id="rank",
        ),
        pytest.param(
            [_aid(), _total(decision_ids=("one_total",))],
            [ONE_TOTAL],
            True,
            id="aggregate-in-an-amendment-entry",
        ),
        pytest.param(
            [_aid(decision_ids=("measure", "years_read")), _total()],
            [YEARS_READ.model_copy(update={"resolved_by": ResolvedBy.USER})],
            True,
            id="users-period-decision",
        ),
        pytest.param(
            [_aid(), _ranked(decision_ids=("population", "summed"))],
            [POPULATION, SUMMED.model_copy(update={"resolved_by": ResolvedBy.USER})],
            True,
            id="users-reduction-decision",
        ),
    ],
)
def test_a_planner_decision_or_one_naming_other_fields_states_no_figure_over_the_years(
    decision: Callable[..., Decision],
    requirements: list[EvidenceRequirement],
    decisions: list[Decision],
    amendment: bool,
) -> None:
    result = _propose(_context(amendment=amendment), requirements, [decision(), *decisions])

    _refused(result, _gap(years=EVERY_YEAR))


def test_a_search_that_named_the_years_admits_no_figure_over_them(
    decision: Callable[..., Decision],
) -> None:
    window = {"kind": "range", "start_year": 1990, "end_year": 2024}
    identity = {"stat_var": "NetODA", "requested_window": window}
    spec_id = spec_id_for(canonicalize(identity))
    pins = _reading_pin_sources(
        [{"status": "definite", "spec_id": spec_id, "requested_window_kind": "range"}],
        [spec_id],
        blocks={spec_id: {"spec_id": spec_id, "definition": {"identity": identity}}},
        earlier=(),
        warnings={},
    )

    result = _propose(_context(pins=pins), [_aid(pin=spec_id), _total()], [decision()])

    _refused(result, _gap(years=EVERY_YEAR))


def _yearly(ctx: Any, decision: Callable[..., Decision], step: str) -> None:
    """Record in `ctx` the plan a user is shown before they amend it: the step keeping its years."""
    kept = (
        _total(group_by=(GroupBy.ENTITY, GroupBy.PERIOD))
        if step == "aggregate"
        else _ranked(per_period=True)
    )
    first = _propose(ctx, [_aid(), kept], [decision(), POPULATION][: 1 + (step == "rank")])
    assert first["outcome"] == Outcome.OK.value, first


@pytest.mark.parametrize("step", sorted(_STEPS))
@pytest.mark.parametrize(
    "amendment",
    [
        pytest.param(True, id="amendment-entry"),
        pytest.param(False, id="another-entry"),
        pytest.param(None, id="session-seeded-with-the-plan"),
    ],
)
def test_the_users_decision_admits_the_reduction_in_the_entry_their_amendment_opened(
    decision: Callable[..., Decision], step: str, amendment: bool | None
) -> None:
    """The user amends a per-year plan at its card, asking for one figure. No
    ask changes, so the decision they made licenses it, in that entry only."""
    build, _, stated, cites = _STEPS[step]
    ctx = _context(amendment=amendment)
    _yearly(ctx, decision, step)

    result = _propose(ctx, [build(**cites)], [stated])

    if amendment:
        assert result["outcome"] == Outcome.OK.value, result
        recorded = load_plan(ctx.state)
        assert recorded is not None and recorded.version == 2
        return
    _refused(result, _gap(years=EVERY_YEAR))
    assert "resolved by `user`" not in result["message"]


@pytest.mark.parametrize("step", sorted(_STEPS))
def test_an_amendment_entrys_refusal_offers_the_users_decision(
    decision: Callable[..., Decision], step: str
) -> None:
    build, _, _, _ = _STEPS[step]
    ctx = _context(amendment=True)
    _yearly(ctx, decision, step)

    result = _propose(ctx, [build()], [])

    _refused(result, _gap(years=EVERY_YEAR))
    assert "resolved by `user`" in result["message"]


def _amended(ctx: Any, decision: Callable[..., Decision], step: str) -> None:
    """Record in `ctx` the plan the user's amendment made of `_yearly`, then leave that entry."""
    build, _, stated, cites = _STEPS[step]
    _yearly(ctx, decision, step)
    amended = _propose(ctx, [build(**cites)], [stated])
    assert amended["outcome"] == Outcome.OK.value, amended
    ctx.state[AMENDMENT_ENTRY_STATE_KEY] = False


@pytest.mark.parametrize("step", sorted(_STEPS))
@pytest.mark.parametrize(
    "period",
    [
        pytest.param(EVERY_YEAR, id="unchanged"),
        pytest.param(Period(start=2000, end=2024), id="over-fewer-years"),
    ],
)
def test_a_later_entry_keeps_the_users_reduction_whatever_its_years(
    decision: Callable[..., Decision], step: str, period: Period
) -> None:
    """A replan after a failed run may narrow the years. The total the user
    asked for at the card stays theirs."""
    build, _, stated, cites = _STEPS[step]
    ctx = _context(amendment=True)
    _amended(ctx, decision, step)

    result = _propose(ctx, [_aid(period=period), build(**cites)], [stated])

    assert result["outcome"] == Outcome.OK.value, result


@pytest.mark.parametrize("step", sorted(_STEPS))
def test_a_later_entry_keeps_no_licence_for_another_reduction(
    decision: Callable[..., Decision], step: str
) -> None:
    """A user who asked for one total did not ask for a maximum."""
    build, _, stated, cites = _STEPS[step]
    ctx = _context(amendment=True)
    _amended(ctx, decision, step)

    result = _propose(ctx, [_aid(), build(**cites, reduction=Reduction.MAX)], [stated])

    _refused(result, _gap(years=EVERY_YEAR))


def test_a_user_decision_that_reduced_no_years_licenses_none_later(
    decision: Callable[..., Decision],
) -> None:
    """The planner writes a `user` decision on a total of one year, which reduces
    nothing, then widens the year to every year in the next entry."""
    ctx = _context(amendment=False)
    step = _total(decision_ids=("one_total",))
    first = _propose(ctx, [_aid(period=ONE_YEAR_ONLY), step], [decision(), USER_TOTAL])
    assert first["outcome"] == Outcome.OK.value, first

    result = _propose(ctx, [_aid(), step], [USER_TOTAL])

    _refused(result, _gap(years=EVERY_YEAR))


def test_a_later_entry_keeps_the_users_ranking_over_another_window(
    decision: Callable[..., Decision],
) -> None:
    """The user's decision says the years are ranked together, and the window
    says which years, so a replan may move the window."""
    build, _, stated, cites = _STEPS["rank"]
    ctx = _context(amendment=True)
    _amended(ctx, decision, "rank")

    result = _propose(ctx, [build(**cites, window_start=2000, window_end=2024)], [stated])

    assert result["outcome"] == Outcome.OK.value, result


def _stated(decision: Callable[..., Decision], made: Decision, **fields: Any) -> Plan:
    return Plan(
        question=ASK,
        answer_mode=AnswerMode.SERIES,
        decisions=[decision(), made],
        evidence_requirements=[_aid(), _total(decision_ids=("one_total",))],
        asks=(ASK,),
        **fields,
    )


def test_a_later_entry_keeps_no_reduction_the_plan_it_amends_cited_another_decision_for(
    decision: Callable[..., Decision],
) -> None:
    """The plan on record cites the planner's own decision for its total. Restating
    that decision as the user's licenses nothing."""
    answered = [AskAnswer(ask=0, outputs=("result",))]
    previous = _stated(decision, ONE_TOTAL, answered=answered)
    plan = _stated(decision, USER_TOTAL, answered=answered)

    gaps = unstated_period_reductions(plan, previous=previous)

    assert [(gap.step_id, gap.asks) for gap in gaps] == [("result", (0,))]


def test_a_reduction_no_answering_result_is_computed_from_is_refused(
    decision: Callable[..., Decision],
) -> None:
    plan = _stated(
        decision,
        ONE_TOTAL,
        asks_across_years=(0,),
        answered=[AskAnswer(ask=0, outputs=("aid",))],
    )

    gaps = unstated_period_reductions(plan)

    assert [(gap.step_id, gap.output_id, gap.asks) for gap in gaps] == [("result", "result", ())]


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param([_total(group_by=(GroupBy.PERIOD,))], id="aggregate-by-period"),
        pytest.param(
            [
                _total(id="yearly", group_by=(GroupBy.ENTITY, GroupBy.PERIOD)),
                _ranked(input_id="yearly", axis=None, reduction=None, per_period=True),
            ],
            id="rank-each-year",
        ),
    ],
)
def test_a_step_that_keeps_the_years_reduces_none(
    decision: Callable[..., Decision], steps: list[EvidenceRequirement]
) -> None:
    plan = Plan(
        question=ASK,
        answer_mode=AnswerMode.COMPUTED,
        decisions=[decision(), POPULATION],
        evidence_requirements=[_aid(decision_ids=("measure", "population")), *steps],
    )

    assert unstated_period_reductions(plan) == []


POOLED = Decision(
    slot="pooled",
    description="Whether each donor's share is of its own total or of all donors' total.",
    options=["of all donors' total", "of its own total"],
    chosen="of all donors' total",
    resolved_by="planner_default",
    rationale="The question asks how aid splits between donors.",
    fixes=DecisionFixes(denominator_scope=DenominatorScope.POOLED),
)
ONE_YEAR_ONLY = Period(start=2024, end=2024)
_IN_2024 = FilterRequirement(
    id="kept", input_id="aid", predicates=(PeriodThreshold(op="eq", year=2024),)
)
_YEARLY = _total(id="yearly", group_by=(GroupBy.ENTITY, GroupBy.PERIOD))


def _series(
    requirement_id: str, *, measure: str = "net ODA", period: Period = EVERY_YEAR
) -> DatasetRequirement:
    """One donor's flow to Kenya, a single series."""
    return DatasetRequirement(
        id=requirement_id,
        decision_ids=("measure", "population", "pooled"),
        measure=measure,
        entities=(Entity(name="Kenya", role="recipient"), Entity(name="France", role="donor")),
        period=period,
        unit=Unit(currency="USD", prices="current"),
        source_spec_id=SPEC,
    )


def _side(input_id: str) -> RequirementSide:
    return RequirementSide(input_id=input_id)


def _both(period: Period = EVERY_YEAR) -> list[EvidenceRequirement]:
    """Two series over `period` and their sum, one row per year both report."""
    return [
        _series("bilateral", period=period),
        _series("imputed", measure="imputed multilateral aid", period=period),
        SumOfRequirement(
            id="both", decision_ids=("measure",), left=_side("bilateral"), right=_side("imputed")
        ),
    ]


def _shares(denominator: str) -> ShareOfRequirement:
    return ShareOfRequirement(
        id="shares",
        decision_ids=("pooled",),
        numerator=_side("aid"),
        denominator=_side(denominator),
        axis="donor",
        denominator_scope=DenominatorScope.POOLED,
    )


def _two_donors() -> DatasetRequirement:
    return _aid().model_copy(
        update={
            "entities": (
                Entity(name="Kenya", role="recipient"),
                Entity(name="France", role="donor"),
                Entity(name="Germany", role="donor"),
            ),
            "across": (),
        }
    )


@pytest.mark.parametrize(
    ("steps", "reducing"),
    [
        pytest.param(
            [_aid(), _YEARLY, _total(input_id="yearly")], ["result"], id="a-total-per-year"
        ),
        pytest.param(
            [
                _two_donors(),
                _ranked(id="ranks", per_period=True),
                _total(input_id="ranks", reduction=Reduction.MEAN),
            ],
            ["result"],
            id="a-ranking-per-year",
        ),
        pytest.param(
            [
                _aid(),
                FilterRequirement(
                    id="kept",
                    input_id="aid",
                    predicates=(EntityIn(names=("France",), role="donor"),),
                ),
                _total(input_id="kept"),
            ],
            ["result"],
            id="a-filter-keeping-the-years",
        ),
        pytest.param(
            [
                _aid(),
                _total(id="all", group_by=(GroupBy.PERIOD,)),
                _shares("all"),
                _ranked(input_id="shares", reduction=Reduction.MEAN),
            ],
            ["result"],
            id="a-share-per-year",
        ),
        pytest.param(
            [
                _series("aid"),
                _series("people", measure="population"),
                QuotientRequirement(
                    id="per_person", numerator=_side("aid"), denominator=_side("people")
                ),
                _total(input_id="per_person", reduction=Reduction.MEAN),
            ],
            ["result"],
            id="a-quotient-per-year",
        ),
        pytest.param([*_both(), _total(input_id="both")], ["result"], id="a-sum-per-year"),
        pytest.param(
            [_aid(), _total(id="total"), _total(input_id="total", reduction=Reduction.MAX)],
            ["total"],
            id="a-total-over-the-years",
        ),
        pytest.param(
            [
                _aid(),
                _total(id="total"),
                _total(id="spans", input_id="total", group_by=(GroupBy.ENTITY, GroupBy.PERIOD)),
                _total(input_id="spans", reduction=Reduction.MAX),
            ],
            ["total"],
            id="a-total-over-the-years-grouped-by-its-span",
        ),
        pytest.param(
            [_aid(), _total(id="peak", reduction=Reduction.MAX), _total(input_id="peak")],
            ["peak"],
            id="each-donors-peak-year",
        ),
        pytest.param(
            [_aid(), _ranked(id="top"), _ranked(input_id="top")],
            ["top"],
            id="a-ranking-over-the-years",
        ),
        pytest.param(
            [
                _aid(),
                GrowthRequirement(
                    id="growth",
                    input_id="aid",
                    window_start=1990,
                    window_end=2024,
                    change=GrowthChange.ABSOLUTE,
                ),
                _total(input_id="growth"),
            ],
            [],
            id="a-growth",
        ),
        pytest.param([_aid(), _IN_2024, _total(input_id="kept")], [], id="a-filter-to-one-year"),
        pytest.param(
            [
                _aid(),
                _IN_2024,
                _total(id="all", input_id="kept", group_by=(GroupBy.PERIOD,)),
                _shares("all"),
                _ranked(input_id="shares", reduction=Reduction.MEAN),
            ],
            [],
            id="a-share-of-one-year",
        ),
        pytest.param([*_both(ONE_YEAR_ONLY), _total(input_id="both")], [], id="a-sum-of-one-year"),
    ],
)
def test_a_step_reduces_the_years_its_input_holds_per_entity(
    decision: Callable[..., Decision], steps: list[EvidenceRequirement], reducing: list[str]
) -> None:
    """The last step reads the one before it. It reduces years only where
    that input holds a series of them: a step that already collapsed the
    years, picked one per entity, or was left one year holds none."""
    first = steps[0].model_copy(update={"decision_ids": ("measure", "population", "pooled")})
    plan = Plan(
        question=ASK,
        answer_mode=AnswerMode.COMPUTED,
        decisions=[decision(), POPULATION, POOLED],
        evidence_requirements=[first, *steps[1:]],
    )

    assert [gap.step_id for gap in unstated_period_reductions(plan)] == reducing


def test_a_total_of_two_sources_summed_year_by_year_answers_an_ask_across_years(
    decision: Callable[..., Decision],
) -> None:
    requirements = [*_both(), _total(input_id="both")]
    decisions = [decision(), POPULATION, POOLED]

    # Both sides pin one dataset, so adding them needs no pick from the user.
    pins = {SPEC: PinSource(provenance_ref="oecd/dac2a")}

    refused = _propose(_context(pins=pins), requirements, decisions)
    admitted = _propose(_context(across_years=[0], pins=pins), requirements, decisions)

    assert refused["code"] == ProblemCode.PERIOD_REDUCTION_UNSTATED.value, refused
    assert refused["detail"]["gaps"] == [
        _gap(years=EVERY_YEAR) | {"dataset_requirement_id": "bilateral"}
    ]
    assert admitted["outcome"] == Outcome.OK.value, admitted


_NEWEST = RelativePeriod(offset=0)


_EVERY_YEAR_HELD = {"stat_var": "NetODA", "requested_window": {"kind": "all"}}
_OPEN_SPEC = spec_id_for(canonicalize(_EVERY_YEAR_HELD))


@pytest.fixture
def aid_on_record() -> Iterator[None]:
    """The source `_open` pins on record, as `harness.studio` sets the lookup for a turn."""
    recorded = {_OPEN_SPEC: RecordedIdentity(identity=_EVERY_YEAR_HELD)}
    token = current_recorded_identity.set(recorded.get)
    yield
    current_recorded_identity.reset(token)


# Search's answer for `_OPEN_SPEC` over every year, shown to the planner.
_SHOWN_OPEN = {
    _OPEN_SPEC: PinSource(
        block={
            "spec_id": _OPEN_SPEC,
            "definition": {"identity": _EVERY_YEAR_HELD},
            "resolved": {
                "download": {"rows": 35, "window": {"start_year": 1990, "end_year": 2024}}
            },
        }
    )
}


def _open(**overrides: Any) -> DatasetRequirement:
    """Aid to Kenya retrieved with no end year, so newer years reach it."""
    return _aid(pin=_OPEN_SPEC, **overrides).model_copy(update={"period_end_open": True})


@pytest.mark.parametrize(
    ("requirements", "decisions", "years"),
    [
        pytest.param(
            [
                _open(period=ONE_YEAR_ONLY),
                _ranked(
                    window_start=2024,
                    window_end=2024,
                    rolling=RollingWindow(window_end=_NEWEST),
                ),
            ],
            [POPULATION],
            ONE_YEAR_ONLY,
            id="a-rank-whose-window-end-alone-follows",
        ),
        pytest.param(
            [_open(period=ONE_YEAR_ONLY), _total()], [], ONE_YEAR_ONLY, id="a-total-of-an-open-year"
        ),
        pytest.param(
            [
                _open(period=ONE_YEAR_ONLY),
                FilterRequirement(
                    id="kept",
                    input_id="aid",
                    predicates=(EntityIn(names=("France",), role="donor"),),
                ),
                _total(input_id="kept"),
            ],
            [],
            ONE_YEAR_ONLY,
            id="through-a-filter-on-donors",
        ),
        pytest.param(
            [
                _open(period=Period(start=2020, end=2024)),
                FilterRequirement(
                    id="kept",
                    input_id="aid",
                    predicates=(PeriodThreshold(op="gte", year=2024),),
                ),
                _total(input_id="kept"),
            ],
            [],
            ONE_YEAR_ONLY,
            id="through-a-fixed-first-year",
        ),
    ],
)
@pytest.mark.usefixtures("aid_on_record")
def test_one_year_that_newer_data_widens_is_several(
    decision: Callable[..., Decision],
    requirements: list[EvidenceRequirement],
    decisions: list[Decision],
    years: Period,
) -> None:
    """Each sums 2024 alone today and 2024 and every later year once newer data exists."""
    result = _propose(_context(), requirements, [decision(), *decisions])

    _refused(result, _gap(years=years) | {"follows_newest": True})
    assert "the years from 2024 to the newest" in result["message"]


@pytest.mark.parametrize(
    ("requirements", "decisions"),
    [
        pytest.param(
            [
                _open(period=ONE_YEAR_ONLY),
                _ranked(
                    window_start=2024,
                    window_end=2024,
                    rolling=RollingWindow(window_start=_NEWEST, window_end=_NEWEST),
                ),
            ],
            [POPULATION],
            id="a-rank-whose-window-ends-both-follow",
        ),
        pytest.param(
            [_open(period=ONE_YEAR_ONLY), _ranked(window_start=2024, window_end=2024)],
            [POPULATION],
            id="a-rank-of-a-fixed-year-of-an-open-period",
        ),
        pytest.param(
            [
                _open(period=ONE_YEAR_ONLY),
                FilterRequirement(
                    id="kept",
                    input_id="aid",
                    predicates=(PeriodThreshold(op="lte", year=2024),),
                ),
                _total(input_id="kept"),
            ],
            [],
            id="a-fixed-last-year-of-an-open-period",
        ),
        pytest.param(
            [
                _open(period=Period(start=2020, end=2024)),
                FilterRequirement(
                    id="kept",
                    input_id="aid",
                    predicates=(PeriodThreshold(op="eq", year=2024, rolling=_NEWEST),),
                ),
                _total(input_id="kept"),
            ],
            [],
            id="the-newest-year-as-it-moves",
        ),
    ],
)
@pytest.mark.usefixtures("aid_on_record")
def test_one_year_that_stays_one_year_reduces_none(
    decision: Callable[..., Decision],
    requirements: list[EvidenceRequirement],
    decisions: list[Decision],
) -> None:
    ctx = _context(pins=_SHOWN_OPEN)

    result = _propose(ctx, requirements, [decision(), *decisions])

    assert result["outcome"] == Outcome.OK.value, result


def test_a_result_no_ask_asks_for_is_listed_or_removed_and_the_asked_total_kept(
    decision: Callable[..., Decision],
) -> None:
    """Ask 0 is across years and 'result' answers it. 'top' ranks that total and
    answers nothing. The refusal keeps the total and moves the ranking."""
    requirements = [_aid(), _total(), _top("result")]
    decisions = [decision(), POPULATION]

    refused = _propose(
        _context(across_years=[0]),
        requirements,
        decisions,
        answered=[AskAnswer(ask=0, outputs=("result",))],
    )
    listed = _propose(
        _context(across_years=[0]),
        requirements,
        decisions,
        answered=[AskAnswer(ask=0, outputs=("result", "top"))],
    )

    _refused(refused, _gap(output="top", asks=(), years=EVERY_YEAR))
    assert "list 'top' in `answered` under an ask marked (across years)" in refused["message"]
    assert "add `period`" not in refused["message"]
    assert listed["outcome"] == Outcome.OK.value, listed


_SECTORS = Breakdown(dimension=BreakdownDimension.SECTOR, level=0)


@pytest.mark.parametrize(
    ("group_by", "kept"),
    [
        pytest.param((GroupBy.ENTITY,), (GroupBy.ENTITY, GroupBy.PERIOD), id="entity"),
        pytest.param((GroupBy.BREAKDOWN,), (GroupBy.BREAKDOWN, GroupBy.PERIOD), id="breakdown"),
        pytest.param(
            (GroupBy.ENTITY, GroupBy.BREAKDOWN),
            (GroupBy.ENTITY, GroupBy.PERIOD),
            id="entity-and-breakdown",
        ),
    ],
)
def test_the_aggregate_keeping_the_years_the_refusal_names_is_admitted(
    decision: Callable[..., Decision], group_by: tuple[GroupBy, ...], kept: tuple[GroupBy, ...]
) -> None:
    by_sector = _aid().model_copy(update={"breakdown": _SECTORS})
    words = " and ".join(f"`{column.value}`" for column in kept)

    refused = _propose(_context(), [by_sector, _total(group_by=group_by)], [decision()])
    admitted = _propose(_context(), [by_sector, _total(group_by=kept)], [decision()])

    assert refused["code"] == ProblemCode.PERIOD_REDUCTION_UNSTATED.value, refused
    named = "add `period` to the `group_by` of 'result'" if len(group_by) == 1 else f"by {words}"
    assert named in refused["message"]
    assert admitted["outcome"] == Outcome.OK.value, admitted


def _kenya() -> DatasetRequirement:
    """Aid to Kenya from every donor together, one role, named."""
    return _aid().model_copy(update={"across": ()})


def _every_donor() -> DatasetRequirement:
    """Aid from every donor to anyone, one role open."""
    return _aid().model_copy(update={"entities": ()})


@pytest.mark.parametrize(
    ("refused", "advised", "decisions"),
    [
        pytest.param(
            [_every_donor(), _ranked(reduction=None)],
            [_every_donor(), _ranked(reduction=None, window_start=2024, window_end=2024)],
            [POPULATION],
            id="a-rank-with-no-reduction",
        ),
        pytest.param(
            [_every_donor(), _total()],
            [_every_donor(), _ranked(window_start=2024, window_end=2024)],
            [],
            id="an-aggregate-keeping-no-role",
        ),
        pytest.param(
            [
                _kenya(),
                _ranked(id="top", axis=None, reduction=None, per_period=True, limit=5),
                _total(input_id="top"),
            ],
            [
                _kenya(),
                _ranked(axis=None, reduction=None, window_start=2024, window_end=2024, limit=5),
            ],
            [POPULATION],
            id="an-aggregate-over-a-cut-ranking",
        ),
    ],
)
def test_an_older_refusal_offers_one_year_before_a_reduction_and_its_plan_is_admitted(
    decision: Callable[..., Decision],
    refused: list[EvidenceRequirement],
    advised: list[EvidenceRequirement],
    decisions: list[Decision],
) -> None:
    """Each refusal's first remedy, the one an unmarked ask takes, is one year.
    The reduction is offered only for an ask marked (across years)."""
    first = _propose(_context(), refused, [decision(), *decisions])
    second = _propose(_context(), advised, [decision(), POPULATION])

    assert first["outcome"] == Outcome.INVALID_INPUT.value, first
    message = first["message"]
    assert "window_start and window_end" in message
    assert message.index("window_start and window_end") < message.index("(across years)")
    assert second["outcome"] == Outcome.OK.value, second


@pytest.mark.usefixtures("aid_on_record")
def test_a_users_ranking_of_one_year_licenses_no_window_that_later_follows_the_newest(
    decision: Callable[..., Decision],
) -> None:
    build, _, stated, cites = _STEPS["rank"]
    one_year = cites | {"window_start": 2024, "window_end": 2024}
    ctx = _context(amendment=False, pins=_SHOWN_OPEN)
    first = _propose(
        ctx, [_open(period=ONE_YEAR_ONLY), build(**one_year)], [decision(), POPULATION, stated]
    )
    assert first["outcome"] == Outcome.OK.value, first

    rolling = one_year | {"rolling": RollingWindow(window_end=_NEWEST)}
    result = _propose(ctx, [_open(period=ONE_YEAR_ONLY), build(**rolling)], [stated])

    _refused(result, _gap(years=ONE_YEAR_ONLY) | {"follows_newest": True})


@pytest.mark.parametrize(
    ("requirements", "answered", "gap", "advised", "advised_answers"),
    [
        pytest.param(
            [_aid(), _total()],
            [AskAnswer(ask=0, outputs=("result",)), AskAnswer(ask=1, outputs=("result",))],
            {"output": "result", "asks": (1,)},
            [_aid(), _total(), _YEARLY],
            [AskAnswer(ask=0, outputs=("result",)), AskAnswer(ask=1, outputs=("yearly",))],
            id="the-step-answers-a-marked-and-an-unmarked-ask",
        ),
        pytest.param(
            [_aid(), _total(id="total"), _top("total")],
            [AskAnswer(ask=0, outputs=("top",)), AskAnswer(ask=1, outputs=("top",))],
            {"step": "total", "output": "top", "asks": (1,)},
            [
                _aid(),
                _total(id="total"),
                _top("total"),
                _ranked(id="newest", window_start=2024, window_end=2024),
            ],
            [AskAnswer(ask=0, outputs=("top",)), AskAnswer(ask=1, outputs=("newest",))],
            id="a-later-result-answers-a-marked-and-an-unmarked-ask",
        ),
    ],
)
def test_an_unmarked_ask_beside_a_marked_one_gets_a_result_of_its_own(
    decision: Callable[..., Decision],
    requirements: list[EvidenceRequirement],
    answered: list[AskAnswer],
    gap: dict[str, Any],
    advised: list[EvidenceRequirement],
    advised_answers: list[AskAnswer],
) -> None:
    """Ask 0 is across years and ask 1 is not. The refusal keeps what answers ask
    0 and answers ask 1 apart, and the plan it describes is admitted."""

    def decisions(steps: list[EvidenceRequirement]) -> list[Decision]:
        ranks = any(isinstance(step, RankRequirement) for step in steps)
        return [decision(), POPULATION] if ranks else [decision()]

    refused = _propose(
        _context(asks=(ASK, OTHER_ASK), across_years=[0]),
        requirements,
        decisions(requirements),
        answered=answered,
    )
    admitted = _propose(
        _context(asks=(ASK, OTHER_ASK), across_years=[0]),
        advised,
        decisions(advised),
        answered=advised_answers,
    )

    _refused(refused, _gap(years=EVERY_YEAR, **gap))
    output = gap["output"]
    assert f"take {output!r} out of that ask's `answered`" in refused["message"]
    assert f"compute {output!r}" not in refused["message"]
    assert admitted["outcome"] == Outcome.OK.value, admitted


def test_an_amendment_about_something_else_licenses_no_total_an_older_decision_cited(
    decision: Callable[..., Decision],
) -> None:
    """The user chose one total for 2024, which reduces no years. A later
    amendment widens the years and restates nothing else: the decision and the
    aggregate are carried, and the user never asked for 35 years as one figure."""
    ctx = _context(amendment=False)
    step = _total(decision_ids=("one_total",))
    first = _propose(ctx, [_aid(period=ONE_YEAR_ONLY), step], [decision(), USER_TOTAL])
    assert first["outcome"] == Outcome.OK.value, first
    ctx.state[AMENDMENT_ENTRY_STATE_KEY] = True

    result = _propose(ctx, [_aid()], [], answered=())

    _refused(result, _gap(years=EVERY_YEAR))


@pytest.mark.usefixtures("aid_on_record")
@pytest.mark.parametrize(
    ("steps", "decisions"),
    [
        pytest.param(
            [_ranked(window_start=2024, window_end=2030)], [POPULATION], id="a-rank-window"
        ),
        pytest.param(
            [
                FilterRequirement(
                    id="kept",
                    input_id="aid",
                    predicates=(PeriodThreshold(op="lte", year=2030),),
                ),
                _total(input_id="kept"),
            ],
            [],
            id="a-filter-threshold",
        ),
        pytest.param(
            [
                FilterRequirement(
                    id="kept",
                    input_id="aid",
                    predicates=(PeriodThreshold(op="lt", year=2031),),
                ),
                _total(input_id="kept"),
            ],
            [],
            id="a-filter-threshold-before-a-year",
        ),
    ],
)
def test_a_fixed_last_year_newer_data_has_yet_to_reach_is_several_years(
    decision: Callable[..., Decision], steps: list[EvidenceRequirement], decisions: list[Decision]
) -> None:
    """The source holds 2024 alone and follows the newest data. A bound of 2030
    stops it there, six years on."""
    ctx = _context(pins=_SHOWN_OPEN)

    result = _propose(ctx, [_open(period=ONE_YEAR_ONLY), *steps], [decision(), *decisions])

    _refused(result, _gap(years=ONE_YEAR_ONLY) | {"follows_newest": True, "until": 2030})
    assert "the years from 2024 to the newest, as far as 2030" in result["message"]


def test_a_users_choice_of_years_licenses_no_ranking_over_them_together(
    decision: Callable[..., Decision],
) -> None:
    """The user's amendment chose 2015 to 2022. Which years a ranking covers
    and whether it combines them are separate statements."""
    years = Decision(
        slot="years",
        description="Which years the ranking covers.",
        options=["2015 to 2022", "every year"],
        chosen="2015 to 2022",
        resolved_by=ResolvedBy.USER,
        rationale="The user asked for these years.",
        fixes=DecisionFixes(window_start=2015, window_end=2022),
    )
    ranked = _ranked(decision_ids=("population", "years"), window_start=2015, window_end=2022)

    result = _propose(_context(amendment=True), [_aid(), ranked], [decision(), POPULATION, years])

    _refused(result, _gap(years=Period(start=2015, end=2022)))


@pytest.mark.parametrize(
    ("last", "until"),
    [pytest.param(2030, 2030, id="years-to-come"), pytest.param(2024, None, id="none-to-come")],
)
def test_a_sum_with_a_source_of_fixed_years_follows_the_newest_as_far_as_they_reach(
    decision: Callable[..., Decision], last: int, until: int | None
) -> None:
    """One side holds 2024 and follows the newest data; the other states 2020
    to `last`. Their sum holds the years both cover, 2024 today."""
    newest = _series("bilateral", period=ONE_YEAR_ONLY).model_copy(update={"period_end_open": True})
    fixed = _series(
        "imputed", measure="imputed multilateral aid", period=Period(start=2020, end=last)
    )
    both = SumOfRequirement(
        id="both", decision_ids=("measure",), left=_side("bilateral"), right=_side("imputed")
    )
    plan = Plan(
        question=ASK,
        answer_mode=AnswerMode.COMPUTED,
        decisions=[decision(), POPULATION, POOLED],
        evidence_requirements=[newest, fixed, both, _total(input_id="both")],
    )

    gaps = unstated_period_reductions(plan)

    assert [(gap.step_id, gap.held.follows_newest, gap.held.until) for gap in gaps] == (
        [("result", True, until)] if until else []
    )


def test_restating_an_older_decision_under_another_slot_does_not_make_it_the_amendments(
    decision: Callable[..., Decision],
) -> None:
    """The plan on record already made this choice, for a total of one year.
    Citing it again under a new slot in an amendment entry is the same choice."""
    ctx = _context(amendment=False)
    first = _propose(
        ctx,
        [_aid(period=ONE_YEAR_ONLY), _total(decision_ids=("one_total",))],
        [decision(), USER_TOTAL],
    )
    assert first["outcome"] == Outcome.OK.value, first
    ctx.state[AMENDMENT_ENTRY_STATE_KEY] = True
    again = USER_TOTAL.model_copy(update={"slot": "every_year"})

    result = _propose(ctx, [_aid(), _total(decision_ids=("every_year",))], [again])

    _refused(result, _gap(years=EVERY_YEAR))


_FIVE_BACK = RelativePeriod(offset=-5)


@pytest.mark.usefixtures("aid_on_record")
@pytest.mark.parametrize(
    ("steps", "decisions"),
    [
        pytest.param(
            [
                FilterRequirement(
                    id="kept",
                    input_id="aid",
                    predicates=(PeriodThreshold(op="gte", year=2019, rolling=_FIVE_BACK),),
                ),
                _total(input_id="kept"),
            ],
            [],
            id="a-filter-from-five-years-back",
        ),
        pytest.param(
            [
                _ranked(
                    window_start=2019,
                    window_end=2024,
                    rolling=RollingWindow(window_start=_FIVE_BACK, window_end=_NEWEST),
                )
            ],
            [POPULATION],
            id="a-window-rolling-at-both-ends-five-years-apart",
        ),
        pytest.param(
            [
                _ranked(
                    window_start=2019,
                    window_end=2030,
                    rolling=RollingWindow(window_start=_FIVE_BACK),
                )
            ],
            [POPULATION],
            id="a-window-rolling-at-its-start-to-a-fixed-year",
        ),
    ],
)
def test_a_start_trailing_the_newest_year_reaches_back_before_the_stated_period(
    decision: Callable[..., Decision], steps: list[EvidenceRequirement], decisions: list[Decision]
) -> None:
    """The plan states 2024 alone, and a start that follows the newest data has
    the dataset retrieved from its first year: 2019 to 2024 reach the step."""
    ctx = _context(pins=_SHOWN_OPEN)

    result = _propose(ctx, [_open(period=ONE_YEAR_ONLY), *steps], [decision(), *decisions])

    _refused(result, _gap(years=Period(start=2019, end=2024)))
    assert "reduces 2019-2024 to one figure" in result["message"]


@pytest.mark.usefixtures("aid_on_record")
def test_a_following_start_behind_a_fixed_one_keeps_no_year_from_becoming_several(
    decision: Callable[..., Decision],
) -> None:
    """From 2024 on, and from five years back of the newest: 2024 alone today,
    2024 and 2025 next year, since the fixed year is the later start until 2029."""
    kept = FilterRequirement(
        id="kept",
        input_id="aid",
        predicates=(
            PeriodThreshold(op="gte", year=2024),
            PeriodThreshold(op="gte", year=2019, rolling=_FIVE_BACK),
        ),
    )
    ctx = _context(pins=_SHOWN_OPEN)

    result = _propose(
        ctx, [_open(period=ONE_YEAR_ONLY), kept, _total(input_id="kept")], [decision()]
    )

    _refused(result, _gap(years=ONE_YEAR_ONLY) | {"follows_newest": True})


def _latest_year(**overrides: Any) -> RankRequirement:
    """A ranking of the newest year, both ends of its window following the newest data.

    Its start follows, so the dataset it reads is retrieved from the first
    year it holds.
    """
    fields: dict[str, Any] = {
        "id": "latest",
        "window_start": 2024,
        "window_end": 2024,
        "rolling": RollingWindow(window_start=_NEWEST, window_end=_NEWEST),
    }
    return _ranked(**(fields | overrides))


def _keeping(*predicates: PeriodThreshold) -> FilterRequirement:
    return FilterRequirement(id="kept", input_id="aid", predicates=predicates)


@pytest.mark.usefixtures("aid_on_record")
@pytest.mark.parametrize(
    "steps",
    [
        pytest.param([_total()], id="an-aggregate-with-no-bound"),
        pytest.param(
            [_keeping(PeriodThreshold(op="lte", year=2024)), _total(input_id="kept")],
            id="under-lte",
        ),
        pytest.param(
            [_keeping(PeriodThreshold(op="lt", year=2025)), _total(input_id="kept")],
            id="under-lt",
        ),
        pytest.param(
            [_keeping(PeriodThreshold(op="ne", year=2023)), _total(input_id="kept")],
            id="under-ne",
        ),
        pytest.param(
            [
                _total(id="all", group_by=(GroupBy.PERIOD,)),
                _shares("all"),
                _ranked(input_id="shares", reduction=Reduction.MEAN),
            ],
            id="over-a-two-sided-step",
        ),
    ],
)
def test_a_reader_of_a_dataset_another_branch_reads_from_its_first_year_holds_those_years(
    decision: Callable[..., Decision], steps: list[EvidenceRequirement]
) -> None:
    """The plan states 2024 alone. `latest` follows the newest data from its
    start, so 'aid' is retrieved from its first year for every step reading it."""
    ctx = _context(pins=_SHOWN_OPEN)
    aid = _open(period=ONE_YEAR_ONLY, decision_ids=("measure", "population", "pooled"))

    result = _propose(ctx, [aid, _latest_year(), *steps], [decision(), POPULATION, POOLED])

    assert result["code"] == ProblemCode.PERIOD_REDUCTION_UNSTATED.value, result
    (gap,) = result["detail"]["gaps"]
    assert (gap["requirement_id"], gap["from_first_year"]) == ("result", True)
    assert "every year 'aid' holds" in result["message"]


@pytest.mark.usefixtures("aid_on_record")
def test_a_one_year_rank_with_no_reduction_whose_end_alone_follows_is_refused(
    decision: Callable[..., Decision],
) -> None:
    """2024 alone today, 2024 and 2025 once 2025 arrives, when no entity has a
    single value to rank on."""
    ranked = _ranked(
        axis=None,
        reduction=None,
        window_start=2024,
        window_end=2024,
        rolling=RollingWindow(window_end=_NEWEST),
    )
    kenya = _open(period=ONE_YEAR_ONLY).model_copy(update={"across": ()})

    result = _propose(_context(pins=_SHOWN_OPEN), [kenya, ranked], [decision(), POPULATION])

    assert result["outcome"] == Outcome.INVALID_INPUT.value, result
    assert result["code"] == ProblemCode.INVALID_PLAN.value, result


@pytest.mark.usefixtures("aid_on_record")
def test_a_fixed_first_year_bounds_a_read_from_the_datasets_first_year(
    decision: Callable[..., Decision],
) -> None:
    """A rank window from 2020 over a dataset stated as 2024 and retrieved from
    its first year ranks 2020 to 2024."""
    ctx = _context(pins=_SHOWN_OPEN)
    steps = [_latest_year(), _ranked(window_start=2020, window_end=2024)]

    result = _propose(ctx, [_open(period=ONE_YEAR_ONLY), *steps], [decision(), POPULATION])

    _refused(result, _gap(years=Period(start=2020, end=2024)))


@pytest.mark.usefixtures("aid_on_record")
@pytest.mark.parametrize(
    "steps",
    [
        pytest.param(
            [_keeping(PeriodThreshold(op="eq", year=2024)), _total(input_id="kept")],
            id="the-filter-the-refusal-names",
        ),
        pytest.param([_ranked(window_start=2024, window_end=2024)], id="a-rank-of-one-year"),
        pytest.param(
            [_total(group_by=(GroupBy.ENTITY, GroupBy.PERIOD))], id="an-aggregate-by-period"
        ),
    ],
)
def test_a_reader_setting_its_own_year_or_keeping_the_years_is_admitted_beside_a_rolling_branch(
    decision: Callable[..., Decision], steps: list[EvidenceRequirement]
) -> None:
    ctx = _context(pins=_SHOWN_OPEN)

    result = _propose(
        ctx, [_open(period=ONE_YEAR_ONLY), _latest_year(), *steps], [decision(), POPULATION]
    )

    assert result["outcome"] == Outcome.OK.value, result


def _kenya_alone(*, end_open: bool) -> DatasetRequirement:
    """Aid to Kenya in 2024, one role, so a rank of one year needs no reduction."""
    base = _open(period=ONE_YEAR_ONLY) if end_open else _aid(period=ONE_YEAR_ONLY)
    return base.model_copy(update={"across": ()})


@pytest.mark.usefixtures("aid_on_record")
@pytest.mark.parametrize(
    ("end_open", "rolling"),
    [
        pytest.param(False, None, id="a-fixed-year-of-a-closed-dataset"),
        pytest.param(True, None, id="a-fixed-year-of-an-open-dataset"),
        pytest.param(
            True,
            RollingWindow(window_start=_NEWEST, window_end=_NEWEST),
            id="both-ends-following-at-one-offset",
        ),
    ],
)
def test_a_rank_with_no_reduction_over_a_window_that_stays_one_year_is_admitted(
    decision: Callable[..., Decision], end_open: bool, rolling: RollingWindow | None
) -> None:
    """The two windows the refusal of an end that alone follows names, and a closed year."""
    ranked = _ranked(axis=None, reduction=None, window_start=2024, window_end=2024, rolling=rolling)
    ctx = _context(pins=_SHOWN_OPEN)

    result = _propose(ctx, [_kenya_alone(end_open=end_open), ranked], [decision(), POPULATION])

    assert result["outcome"] == Outcome.OK.value, result


@pytest.mark.usefixtures("aid_on_record")
def test_a_rank_with_no_reduction_and_no_window_holds_every_year_a_rolling_branch_retrieves(
    decision: Callable[..., Decision],
) -> None:
    ctx = _context(pins=_SHOWN_OPEN)
    newest = _latest_year(axis=None, reduction=None)
    ranked = _ranked(axis=None, reduction=None)

    result = _propose(ctx, [_kenya_alone(end_open=True), newest, ranked], [decision(), POPULATION])

    assert result["code"] == ProblemCode.INVALID_PLAN.value, result
    assert "every year 'aid' holds" in result["message"]
