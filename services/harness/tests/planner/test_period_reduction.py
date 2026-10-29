"""A step turns several years into one figure only for an ask the orchestrator marked across years.

A dataset requirement's `period` is the window of the search that found its
source, which for a question naming no years is every year the source holds.
`propose_plan` refuses an `aggregate` without `period` in its `group_by`, or a
`rank` with a `reduction`, over such years with `period_reduction_unstated`
(`planner.plan.unstated_period_reductions`) unless every answering result
computed from the step answers asks across years, and only those, or the step
holds a licence. Where the planner reads the user's own words, in an entry
the user's amendment opened or in the standalone planner, a step it names in
`across_years` takes one, and keeps it while the step keeps its kind and
states no other reduction (`planner.plan.period_licences_held`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import partial
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
    MAX_PLAN_RECORD_CHARS,
    MAX_PLAN_REQUIREMENTS,
    AggregateRequirement,
    AnswerMode,
    AskAnswer,
    Breakdown,
    ComparisonRequirement,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Decline,
    Entity,
    EvidenceRequirement,
    FilterRequirement,
    FirstYear,
    GrowthRequirement,
    OneYearRead,
    Period,
    PeriodLicence,
    PinSource,
    Plan,
    QuotientRequirement,
    RankRequirement,
    RequirementSide,
    ResolvedBy,
    ShareOfRequirement,
    SumOfRequirement,
    Unit,
    UnsupportedOperation,
    WindowedPin,
    answering_results,
    authored,
    current_recorded_identity,
    frame_shapes,
    inputs_of,
    load_plan,
    one_year_followed,
    origin_of,
    period_licences_held,
    plan_revision,
    record_optional_decline,
    save_pin_sources,
    save_plan,
    split_of,
    unstated_period_reductions,
    whole_history_split,
    windowed_pins,
)
from planner.records import (
    MAX_ENTITIES,
    BreakdownDimension,
    DenominatorScope,
    EntityIn,
    GroupBy,
    GrowthChange,
    OpenPeriodFault,
    PeriodThreshold,
    RankAmong,
    RankDirection,
    Reduction,
    RelativePeriod,
    RollingIssue,
    RollingPlanError,
    RollingWindow,
    ValueThreshold,
)
from planner.shapes import VALUE_KEPT_SPAN_PAIRS
from planner.tool_results import Outcome, ProblemCode
from pydantic import ValidationError
from submission.service import OneYear, _window_pins

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
    declined: Sequence[Decline] = (),
    across_years: Sequence[str] = (),
    removed_decisions: Sequence[str] = (),
    reads_user_words: bool = False,
    restart: bool = False,
) -> dict[str, Any]:
    """Propose `requirements`, the last of which answers ask 0 unless `answered` says otherwise.

    A plan on record in `ctx` is the one the call amends, unless `restart`
    begins a plan of its own. The planner is the seam's, which reads the
    user's words only in an amendment entry, unless `reads_user_words` makes
    it the standalone planner's.
    """
    tools = {tool.name: tool for tool in build_planner_tools(reads_user_words=reads_user_words)}
    recorded = load_plan(ctx.state)
    return tools["propose_plan"].func(
        question=ASK,
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts(decisions),
        evidence_requirements=requirements,
        removed_decisions=list(removed_decisions),
        answered=(
            [AskAnswer(ask=0, outputs=(requirements[-1].id,))] if answered is None else answered
        ),
        declined=list(declined),
        across_years=list(across_years),
        amends_version=recorded.version if recorded is not None and not restart else None,
        starts_new_plan=restart,
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
        "follows_newest": False,
        "until": None,
        "dataset_requirement_id": "aid",
    }


def _refused(result: dict[str, Any], *gaps: dict[str, Any]) -> None:
    """Assert `result` refuses with `gaps`, read apart from the change each offers.

    The change, `one_year`, is what the remedy tests check, by following it
    (`test_the_one_year_each_refusal_offers_is_admitted`).
    """
    assert result["outcome"] == Outcome.INVALID_INPUT.value, result
    assert result["code"] == ProblemCode.PERIOD_REDUCTION_UNSTATED.value, result
    found = [
        {key: value for key, value in gap.items() if key not in ("keep_years", "one_year")}
        for gap in result["detail"]["gaps"]
    ]
    assert found == list(gaps)


_STEPS = {
    "aggregate": (_total, [], USER_TOTAL, {"decision_ids": ("one_total",)}),
    "rank": (
        _ranked,
        [POPULATION],
        USER_RANKING,
        {"decision_ids": ("population", "one_ranking")},
    ),
}


# Each step reducing the years, with the decisions it cites: `_STEPS`'
# two kinds, and an aggregate grouped by nothing, which reduces every
# donor and year to one figure.
_REDUCING = {
    **{step: (build, cited) for step, (build, cited, _, _) in _STEPS.items()},
    "one_figure": (partial(_total, group_by=()), []),
}


# The change keeping the years each of `_REDUCING`'s steps is offered.
_KEEPS_YEARS = {
    "aggregate": {"group_by": ["entity", "period"]},
    "one_figure": {"group_by": ["period"]},
    "rank": {"per_period": True},
}


@pytest.mark.parametrize("step", sorted(_REDUCING))
def test_one_figure_over_years_answers_an_ask_across_years(
    decision: Callable[..., Decision], step: str
) -> None:
    build, cited = _REDUCING[step]
    requirements = [_aid(), build()]

    unmarked = _context()
    refused = _propose(unmarked, requirements, [decision(), *cited])
    marked = _context(across_years=[0])
    admitted = _propose(marked, requirements, [decision(), *cited])

    _refused(refused, _gap(years=EVERY_YEAR))
    assert refused["detail"]["gaps"][0]["keep_years"] == _KEEPS_YEARS[step]
    assert "1990-2024" in refused["message"]
    assert load_plan(unmarked.state) is None
    assert admitted["outcome"] == Outcome.OK.value, admitted
    recorded = load_plan(marked.state)
    assert recorded is not None and recorded.asks_across_years == (0,)


@pytest.mark.parametrize(
    ("step", "cited"),
    [
        pytest.param(
            _total(group_by=(), decision_ids=("one_total",)),
            [ONE_TOTAL.model_copy(update={"fixes": DecisionFixes(group_by=())})],
            id="a-total-whose-group-a-decision-fixes",
        ),
        pytest.param(
            _ranked(decision_ids=("population", "one_ranking")),
            [POPULATION, ONE_RANKING],
            id="a-rank-a-decision-fixes-per-period-false",
        ),
    ],
)
def test_a_refusal_offers_no_keeping_of_the_years_a_cited_decision_forbids(
    decision: Callable[..., Decision], step: EvidenceRequirement, cited: list[Decision]
) -> None:
    """Keeping the years changes the field the decision fixes, so the plan it
    gives is refused and the refusal offers only the one year."""
    refused = _propose(_context(), [_aid(), step], [decision(), *cited])

    _refused(refused, _gap(years=EVERY_YEAR))
    (gap,) = refused["detail"]["gaps"]
    assert gap["keep_years"] is None
    assert gap["one_year"] is not None


def _declined(ask: int) -> Decline:
    return Decline(
        ask=(ASK, OTHER_ASK)[ask], code=UnsupportedOperation.NO_OPERATION_COVERS, ask_ref=ask
    )


@pytest.mark.parametrize(
    ("declined", "answers", "outcome"),
    [
        pytest.param((0, 1), False, (Outcome.INVALID_INPUT, "decline_unearned"), id="refused"),
        pytest.param((1,), True, (Outcome.OK, None), id="unrefused"),
    ],
)
def test_a_first_draft_declines_no_ask_a_reduction_refusal_named(
    decision: Callable[..., Decision],
    declined: tuple[int, ...],
    answers: bool,
    outcome: tuple[Outcome, str | None],
) -> None:
    """A first draft's permission to decline as `no_operation_covers`
    (`harness.planner_seam`) covers no ask a `period_reduction_unstated`
    refusal named in the entry, and still covers an ask it did not name."""
    ctx = _context(asks=(ASK, OTHER_ASK))
    record_optional_decline(ctx.state, UnsupportedOperation.NO_OPERATION_COVERS, terminal=True)
    _refused(
        _propose(ctx, [_aid(), _total()], [decision()], declined=[_declined(1)]),
        _gap(years=EVERY_YEAR),
    )

    result = _propose(
        ctx,
        [_aid(), _YEARLY] if answers else [],
        [decision()] if answers else [],
        answered=[AskAnswer(ask=0, outputs=("yearly",))] if answers else (),
        declined=[_declined(ask) for ask in declined],
    )

    assert (result["outcome"], result.get("code")) == (outcome[0].value, outcome[1]), result
    if not answers:
        assert result["detail"]["asks"] == [0]


@pytest.mark.parametrize("step", sorted(_REDUCING))
def test_a_plan_with_no_asks_reduces_no_years(decision: Callable[..., Decision], step: str) -> None:
    build, cited = _REDUCING[step]

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
        [
            {
                "text": "net ODA",
                "status": "definite",
                "spec_id": spec_id,
                "requested_window_kind": "range",
            }
        ],
        [spec_id],
        blocks={spec_id: {"spec_id": spec_id, "definition": {"identity": identity}}},
        earlier=(),
        warnings={},
    )

    result = _propose(_context(pins=pins), [_aid(pin=spec_id), _total()], [decision()])

    _refused(result, _gap(years=EVERY_YEAR))


def _yearly(ctx: Any, decision: Callable[..., Decision], step: str) -> None:
    """Record in `ctx` the plan a user is shown before amending it, its step keeping its years."""
    kept = (
        _total(group_by=(GroupBy.ENTITY, GroupBy.PERIOD))
        if step == "aggregate"
        else _ranked(per_period=True)
    )
    first = _propose(ctx, [_aid(), kept], [decision(), POPULATION][: 1 + (step == "rank")])
    assert first["outcome"] == Outcome.OK.value, first


def _licence(step: str, reduction: Reduction = Reduction.SUM) -> PeriodLicence:
    return PeriodLicence(step_id="result", step_kind=step, reduction=reduction)


@pytest.mark.parametrize("step", sorted(_STEPS))
@pytest.mark.parametrize(
    "amendment",
    [
        pytest.param(True, id="amendment-entry"),
        pytest.param(False, id="another-entry"),
        pytest.param(None, id="session-seeded-with-the-plan"),
    ],
)
def test_a_step_named_across_years_is_licensed_where_the_planner_reads_the_users_words(
    decision: Callable[..., Decision], step: str, amendment: bool | None
) -> None:
    """The user amends a per-year plan at its card, asking for one figure. No
    ask changes, so the planner names the step, in that entry only."""
    build, _, _, _ = _STEPS[step]
    ctx = _context(amendment=amendment)
    _yearly(ctx, decision, step)

    result = _propose(ctx, [build()], [], across_years=["result"])

    if amendment:
        assert result["outcome"] == Outcome.OK.value, result
        recorded = load_plan(ctx.state)
        assert recorded is not None and recorded.version == 2
        assert recorded.period_licences == (_licence(step),)
        return
    assert result["code"] == ProblemCode.INVALID_PLAN.value, result
    assert "(across years)" in result["message"]
    recorded = load_plan(ctx.state)
    assert recorded is not None and recorded.version == 1


@pytest.mark.parametrize("step", sorted(_STEPS))
@pytest.mark.parametrize(
    ("amendment", "offered"),
    [pytest.param(True, True, id="amendment-entry"), pytest.param(False, False, id="another")],
)
def test_a_refusal_offers_naming_the_step_only_where_the_planner_reads_the_users_words(
    decision: Callable[..., Decision], step: str, amendment: bool, offered: bool
) -> None:
    build, _, _, _ = _STEPS[step]
    ctx = _context(amendment=amendment)
    _yearly(ctx, decision, step)

    result = _propose(ctx, [build()], [])

    _refused(result, _gap(years=EVERY_YEAR))
    assert ("name 'result' in `across_years`" in result["message"]) is offered


def test_the_remedy_an_amendment_entrys_refusal_offers_is_admitted(
    decision: Callable[..., Decision],
) -> None:
    """The user chose one total for 2024, then asks for it over every year. The
    step cites their decision, so re-citing it changes nothing. The refusal
    offers naming the step, which is admitted, so the planner does not loop on
    the same refusal."""
    ctx = _context(amendment=False)
    step = _total(decision_ids=("one_total",))
    first = _propose(ctx, [_aid(period=ONE_YEAR_ONLY), step], [decision(), USER_TOTAL])
    assert first["outcome"] == Outcome.OK.value, first
    ctx.state[AMENDMENT_ENTRY_STATE_KEY] = True

    refused = _propose(ctx, [_aid()], [], answered=())
    admitted = _propose(ctx, [_aid()], [], answered=(), across_years=["result"])

    _refused(refused, _gap(years=EVERY_YEAR))
    assert "name 'result' in `across_years`" in refused["message"]
    assert "resolved by `user`" not in refused["message"]
    assert admitted["outcome"] == Outcome.OK.value, admitted


def _amended(ctx: Any, decision: Callable[..., Decision], step: str) -> None:
    """Record in `ctx` the plan the user's amendment made of `_yearly`, then leave that entry."""
    build, _, _, _ = _STEPS[step]
    _yearly(ctx, decision, step)
    amended = _propose(ctx, [build()], [], across_years=["result"])
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
@pytest.mark.parametrize(
    "named",
    [pytest.param([], id="unnamed"), pytest.param(["result"], id="named-again")],
)
def test_a_later_entry_keeps_the_users_reduction_whatever_its_years(
    decision: Callable[..., Decision], step: str, period: Period, named: list[str]
) -> None:
    """A replan after a failed run may narrow the years. The total the user
    asked for at the card stays theirs, and a replan naming it again restates
    the licence it holds."""
    build, _, _, _ = _STEPS[step]
    ctx = _context(amendment=True)
    _amended(ctx, decision, step)

    result = _propose(ctx, [_aid(period=period), build()], [], across_years=named)

    assert result["outcome"] == Outcome.OK.value, result
    recorded = load_plan(ctx.state)
    assert recorded is not None and recorded.period_licences == (_licence(step),)


_INTERVENING: dict[str, Callable[[], list[EvidenceRequirement]]] = {
    "aggregate-over-one-year": lambda: [_aid(period=ONE_YEAR_ONLY), _total()],
    "rank-over-one-year": lambda: [_aid(period=ONE_YEAR_ONLY), _ranked()],
    "rank-over-one-year-with-no-reduction": lambda: [
        _every_donor().model_copy(update={"period": ONE_YEAR_ONLY}),
        _ranked(reduction=None),
    ],
}


@pytest.mark.parametrize(
    "intervening",
    [pytest.param(name, id=name) for name in _INTERVENING],
)
def test_an_intervening_version_keeping_its_years_keeps_the_licence_through_storage(
    decision: Callable[..., Decision], intervening: str
) -> None:
    """A step licensed over every year, kept to one year in a replan, then
    restored in another keeps its licence. It holds its kind and states no
    other reduction, and each plan is stored and loaded between the entries.
    The restoring replan reads the brief, so only the licence admits it."""
    step = intervening.split("-", 1)[0]
    build, _, _, _ = _STEPS[step]
    ctx = _context(amendment=True)
    _amended(ctx, decision, step)

    def stored(ctx: Any) -> Any:
        recorded = load_plan(ctx.state)
        assert recorded is not None
        state = {**ctx.state}
        save_plan(state, recorded)
        return SimpleNamespace(state=state, actions=EventActions(), session=ctx.session)

    ctx = stored(ctx)
    narrowed = _propose(ctx, _INTERVENING[intervening](), [])
    ctx = stored(ctx)
    narrowed_plan = load_plan(ctx.state)
    restored = _propose(ctx, [_aid(), build()], [])

    assert narrowed["outcome"] == Outcome.OK.value, narrowed
    assert narrowed_plan is not None and narrowed_plan.period_licences == (_licence(step),)
    assert restored["outcome"] == Outcome.OK.value, restored
    recorded = load_plan(ctx.state)
    assert recorded is not None and recorded.version == 4
    assert recorded.period_licences == (_licence(step),)


@pytest.mark.parametrize("step", sorted(_STEPS))
def test_a_later_entry_keeps_no_licence_for_another_reduction(
    decision: Callable[..., Decision], step: str
) -> None:
    """A user who asked for one total did not ask for a maximum, and a replan
    reading the brief cannot name the maximum."""
    build, _, _, _ = _STEPS[step]
    ctx = _context(amendment=True)
    _amended(ctx, decision, step)

    unnamed = _propose(ctx, [_aid(), build(reduction=Reduction.MAX)], [])
    named = _propose(ctx, [_aid(), build(reduction=Reduction.MAX)], [], across_years=["result"])

    _refused(unnamed, _gap(years=EVERY_YEAR))
    assert named["code"] == ProblemCode.INVALID_PLAN.value, named
    assert "Take 'result' out of `across_years`" in named["message"]


@pytest.mark.parametrize(
    ("step", "intervening"),
    [
        pytest.param(
            "aggregate", _total(group_by=(GroupBy.ENTITY, GroupBy.PERIOD)), id="aggregate-by-period"
        ),
        pytest.param("rank", _ranked(per_period=True), id="rank-per-period"),
        pytest.param(
            "aggregate",
            _total(group_by=(GroupBy.ENTITY, GroupBy.PERIOD), reduction=Reduction.MEAN),
            id="aggregate-by-period-with-a-mean",
        ),
        pytest.param(
            "rank", _ranked(per_period=True, reduction=Reduction.MAX), id="rank-per-period-by-max"
        ),
        pytest.param(
            "rank", _total(group_by=(GroupBy.ENTITY, GroupBy.PERIOD)), id="a-step-of-another-kind"
        ),
    ],
)
def test_a_version_stating_another_reduction_or_kind_drops_the_licence(
    decision: Callable[..., Decision], step: str, intervening: EvidenceRequirement
) -> None:
    """The version between keeps its years, so it is admitted, but it states
    a figure for each year, or another figure, in place of the one the user
    asked for, and restoring that figure later needs the user's words
    again."""
    build, cited, _, _ = _STEPS[step]
    ctx = _context(amendment=True)
    _amended(ctx, decision, step)

    between = _propose(ctx, [intervening], [])
    restored = _propose(ctx, [_aid(), build()], cited)

    assert between["outcome"] == Outcome.OK.value, between
    recorded = load_plan(ctx.state)
    assert recorded is not None and recorded.period_licences == ()
    _refused(restored, _gap(years=EVERY_YEAR))


def test_a_reduction_corrected_within_the_amendment_entry_is_named_again(
    decision: Callable[..., Decision],
) -> None:
    """Round one of the entry records a sum. Round two makes it a mean. The
    licence goes with the sum, and naming the step again grants the mean."""
    ctx = _context(amendment=True)
    _yearly(ctx, decision, "aggregate")
    summed = _propose(ctx, [_total()], [], across_years=["result"])
    assert summed["outcome"] == Outcome.OK.value, summed

    unnamed = _propose(ctx, [_total(reduction=Reduction.MEAN)], [])
    named = _propose(ctx, [_total(reduction=Reduction.MEAN)], [], across_years=["result"])

    _refused(unnamed, _gap(years=EVERY_YEAR))
    assert named["outcome"] == Outcome.OK.value, named
    recorded = load_plan(ctx.state)
    assert recorded is not None
    assert recorded.period_licences == (_licence("aggregate", Reduction.MEAN),)


def test_a_new_result_named_in_the_amendment_entry_is_licensed_beside_one_already_licensed(
    decision: Callable[..., Decision],
) -> None:
    """Kenya already has one total. "Give Ghana as one total too" states the
    same decision wording and grouping for Ghana's step, or cites Kenya's."""
    ghana = _aid().model_copy(
        update={"id": "ghana_aid", "entities": (Entity(name="Ghana", role="recipient"),)}
    )
    kenya_total = _total(id="kenya_total", decision_ids=("one_total",))
    ghana_answers = [
        AskAnswer(ask=0, outputs=("kenya_total",)),
        AskAnswer(ask=1, outputs=("ghana_total",)),
    ]
    ctx = _context(amendment=True)
    first = _propose(
        ctx,
        [_aid(), kenya_total],
        [decision(), USER_TOTAL],
        answered=[AskAnswer(ask=0, outputs=("kenya_total",))],
        across_years=["kenya_total"],
    )
    assert first["outcome"] == Outcome.OK.value, first
    ctx.state[ASKS_STATE_KEY] = [ASK, OTHER_ASK]

    own = USER_TOTAL.model_copy(update={"slot": "ghana_one_total"})
    with_own = _propose(
        ctx,
        [ghana, _total(id="ghana_total", input_id="ghana_aid", decision_ids=("ghana_one_total",))],
        [own],
        answered=ghana_answers,
        across_years=["ghana_total"],
    )
    recorded = load_plan(ctx.state)
    citing_kenyas = _propose(
        ctx,
        [_total(id="ghana_total", input_id="ghana_aid", decision_ids=("one_total",))],
        [],
        removed_decisions=["ghana_one_total"],
        answered=ghana_answers,
        across_years=["ghana_total"],
    )

    assert with_own["outcome"] == Outcome.OK.value, with_own
    assert recorded is not None
    assert [licence.step_id for licence in recorded.period_licences] == [
        "kenya_total",
        "ghana_total",
    ]
    assert citing_kenyas["outcome"] == Outcome.OK.value, citing_kenyas


def test_a_later_entry_keeps_the_users_ranking_over_another_window(
    decision: Callable[..., Decision],
) -> None:
    """The licence says the years are ranked together, and the window says
    which years, so a replan may move the window."""
    build, _, _, _ = _STEPS["rank"]
    ctx = _context(amendment=True)
    _amended(ctx, decision, "rank")

    result = _propose(ctx, [build(window_start=2000, window_end=2024)], [])

    assert result["outcome"] == Outcome.OK.value, result


_REWORDED = USER_TOTAL.model_copy(
    update={
        "options": ["one figure over every year", "a figure per year"],
        "chosen": "one figure over every year",
    }
)


@pytest.mark.parametrize(
    "made",
    [
        pytest.param(
            Decision(
                slot="by_sector",
                description="Whether aid is split by sector.",
                options=["split by sector", "one figure for every sector"],
                chosen="split by sector",
                resolved_by=ResolvedBy.USER,
                rationale="The user asked for aid by sector.",
                fixes=DecisionFixes(group_by=(GroupBy.ENTITY, GroupBy.BREAKDOWN)),
            ),
            id="a-sector-split",
        ),
        pytest.param(_REWORDED, id="a-reworded-decision"),
        pytest.param(USER_TOTAL, id="a-users-decision"),
    ],
)
def test_a_users_decision_in_an_amendment_entry_licenses_no_reduction(
    decision: Callable[..., Decision], made: Decision
) -> None:
    """Which parties stay apart and whether the years stay apart share `group_by`,
    so a decision fixing it says nothing about the years. Only naming the step
    in `across_years` does."""
    by_sector = _aid().model_copy(update={"breakdown": _SECTORS})
    assert made.fixes is not None and made.fixes.group_by is not None
    step = _total(decision_ids=(made.slot,), group_by=made.fixes.group_by)

    result = _propose(_context(amendment=True), [by_sector, step], [decision(), made])

    _refused(result, _gap(years=EVERY_YEAR))


def test_a_stored_plan_citing_a_users_decision_carries_no_licence(
    decision: Callable[..., Decision],
) -> None:
    """A plan recorded without licences, or by a first draft citing a
    decision the planner marked as the user's, holds no licence to carry."""
    ctx = _context(amendment=False)
    answered = [AskAnswer(ask=0, outputs=("result",))]
    save_plan(ctx.state, _stated(decision, USER_TOTAL, answered=answered))

    result = _propose(ctx, [_aid(), _total(decision_ids=("one_total",))], [])

    _refused(result, _gap(years=EVERY_YEAR))


@pytest.mark.parametrize(
    ("named", "requirements"),
    [
        pytest.param(["absent"], [_aid(), _total()], id="a-step-the-plan-lacks"),
        pytest.param(
            ["result"],
            [_aid(), _total(group_by=(GroupBy.ENTITY, GroupBy.PERIOD))],
            id="a-step-keeping-the-years",
        ),
        pytest.param(["aid"], [_aid(), _total()], id="a-dataset-requirement"),
    ],
)
def test_naming_a_step_that_combines_no_years_is_refused(
    decision: Callable[..., Decision], named: list[str], requirements: list[EvidenceRequirement]
) -> None:
    ctx = _context(amendment=True)

    result = _propose(ctx, requirements, [decision()], across_years=named)

    assert result["code"] == ProblemCode.INVALID_PLAN.value, result
    assert repr(named[0]) in result["message"]
    assert load_plan(ctx.state) is None


@pytest.mark.parametrize("step", sorted(_STEPS))
def test_the_standalone_planner_licenses_a_step_it_names_in_a_plan_with_no_asks(
    decision: Callable[..., Decision], step: str
) -> None:
    """ "Total French aid to Guatemala between 2015 and 2022": the standalone
    planner reads the question itself and lists no asks."""
    build, cited, _, _ = _STEPS[step]
    years = Period(start=2015, end=2022)
    requirements = [_aid(period=years), build()]

    refused = _propose(
        _context(asks=()), requirements, [decision(), *cited], answered=(), reads_user_words=True
    )
    ctx = _context(asks=())
    admitted = _propose(
        ctx,
        requirements,
        [decision(), *cited],
        answered=(),
        across_years=["result"],
        reads_user_words=True,
    )

    _refused(refused, _gap(asks=(), years=years))
    assert "name 'result' in `across_years`" in refused["message"]
    assert admitted["outcome"] == Outcome.OK.value, admitted
    recorded = load_plan(ctx.state)
    assert recorded is not None and recorded.period_licences == (_licence(step),)


def test_a_restart_begins_with_no_licences(decision: Callable[..., Decision]) -> None:
    ctx = _context(asks=())
    first = _propose(
        ctx,
        [_aid(), _total()],
        [decision()],
        answered=(),
        across_years=["result"],
        reads_user_words=True,
    )
    assert first["outcome"] == Outcome.OK.value, first

    result = _propose(
        ctx, [_aid(), _total()], [decision()], answered=(), reads_user_words=True, restart=True
    )

    _refused(result, _gap(asks=(), years=EVERY_YEAR))


def _stated(decision: Callable[..., Decision], made: Decision, **fields: Any) -> Plan:
    return Plan(
        question=ASK,
        answer_mode=AnswerMode.SERIES,
        decisions=[decision(), made],
        evidence_requirements=[_aid(), _total(decision_ids=("one_total",))],
        asks=(ASK,),
        **fields,
    )


@pytest.mark.parametrize(
    ("licence", "licensed"),
    [
        pytest.param(_licence("aggregate"), True, id="its-own"),
        pytest.param(_licence("aggregate", Reduction.MEAN), False, id="another-reduction"),
        pytest.param(_licence("rank"), False, id="another-kind"),
        pytest.param(
            PeriodLicence(step_id="other", step_kind="aggregate", reduction=Reduction.SUM),
            False,
            id="another-step",
        ),
    ],
)
def test_a_licence_admits_only_the_step_of_its_id_kind_and_reduction(
    decision: Callable[..., Decision], licence: PeriodLicence, licensed: bool
) -> None:
    answered = [AskAnswer(ask=0, outputs=("result",))]
    plan = _stated(decision, ONE_TOTAL, answered=answered, period_licences=(licence,))

    gaps = unstated_period_reductions(plan)

    assert [gap.step_id for gap in gaps] == ([] if licensed else ["result"])


def test_an_amended_plan_holds_no_licence_until_propose_plan_carries_it(
    decision: Callable[..., Decision],
) -> None:
    answered = [AskAnswer(ask=0, outputs=("result",))]
    licensed = _stated(
        decision, ONE_TOTAL, answered=answered, period_licences=(_licence("aggregate"),)
    )
    unlicensed = _stated(decision, ONE_TOTAL, answered=answered)

    amended = licensed.amend()

    assert amended.period_licences == ()
    assert period_licences_held(amended, named=(), previous=licensed) == (_licence("aggregate"),)
    assert plan_revision(licensed) != plan_revision(unlicensed)
    assert plan_revision(unlicensed) == plan_revision(
        Plan.model_validate(unlicensed.model_dump(exclude={"period_licences"}))
    )


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
                    predicates=(EntityIn(names=("Kenya",), role="recipient"),),
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
    that input holds a series of them. A step that collapses the years,
    picks one per entity, or leaves one year holds none."""
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

    _refused(refused, _gap(years=EVERY_YEAR) | {"dataset_requirement_id": "bilateral"})
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
                    predicates=(EntityIn(names=("Kenya",), role="recipient"),),
                ),
                _total(input_id="kept"),
            ],
            [],
            ONE_YEAR_ONLY,
            id="through-a-filter-on-the-recipient",
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
        pytest.param(
            [
                _open(period=Period(start=2020, end=2024)),
                FilterRequirement(
                    id="kept",
                    input_id="aid",
                    predicates=(PeriodThreshold(op="gte", year=2024, rolling=_NEWEST),),
                ),
                _total(input_id="kept"),
            ],
            [],
            ONE_YEAR_ONLY,
            id="through-a-first-year-on-the-anchor",
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
    result = _propose(_context(pins=_SHOWN_OPEN), requirements, [decision(), *decisions])

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
    assert refused["detail"]["gaps"][0]["keep_years"] is None
    assert listed["outcome"] == Outcome.OK.value, listed


_SECTORS = Breakdown(dimension=BreakdownDimension.SECTOR, level=0)


@pytest.mark.parametrize(
    ("group_by", "kept"),
    [
        pytest.param((), (GroupBy.PERIOD,), id="nothing"),
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

    refused = _propose(_context(), [by_sector, _total(group_by=group_by)], [decision()])
    admitted = _propose(_context(), [by_sector, _total(group_by=kept)], [decision()])

    assert refused["code"] == ProblemCode.PERIOD_REDUCTION_UNSTATED.value, refused
    (gap,) = refused["detail"]["gaps"]
    assert gap["keep_years"] == {"group_by": [column.value for column in kept]}
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
    amendment widens the years and restates nothing else. The decision and the
    aggregate are carried, and the user did not ask for 35 years as one figure."""
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
    and whether it combines them are separate statements, and one year would
    rewrite the years the user chose, so the refusal offers none."""
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
    assert result["detail"]["gaps"][0]["one_year"] is None


@pytest.mark.parametrize(
    ("last", "until"),
    [pytest.param(2030, 2030, id="years-to-come"), pytest.param(2024, None, id="none-to-come")],
)
def test_a_sum_with_a_source_of_fixed_years_follows_the_newest_as_far_as_they_reach(
    decision: Callable[..., Decision], last: int, until: int | None
) -> None:
    """One side holds 2024 and follows the newest data. The other states 2020
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
    """The plan on record made this choice, for a total of one year.
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
    ("steps", "decisions", "follows_newest", "phrase"),
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
            True,
            "the years from 2019 to the newest",
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
            False,
            "2019-2024",
            id="a-window-rolling-at-both-ends-five-years-apart",
        ),
    ],
)
def test_a_start_trailing_the_newest_year_reaches_back_before_the_stated_period(
    decision: Callable[..., Decision],
    steps: list[EvidenceRequirement],
    decisions: list[Decision],
    follows_newest: bool,
    phrase: str,
) -> None:
    """The plan states 2024 alone, and a start that follows the newest data has
    the dataset retrieved from its first year, so 2019 to 2024 reach the step.
    The window's end follows the newest year half the series hold, as its
    start does. The filter sets no end, so it keeps each later year a series
    reports."""
    ctx = _context(pins=_SHOWN_OPEN)

    result = _propose(ctx, [_open(period=ONE_YEAR_ONLY), *steps], [decision(), *decisions])

    gap = _gap(years=Period(start=2019, end=2024))
    _refused(result, gap | {"follows_newest": follows_newest})
    assert f"reduces {phrase} to one figure" in result["message"]


@pytest.mark.usefixtures("aid_on_record")
def test_a_following_start_behind_a_fixed_one_keeps_no_year_from_becoming_several(
    decision: Callable[..., Decision],
) -> None:
    """From 2024 on, and from five years back of the newest, holds 2024 alone
    today and 2024 and 2025 next year, since the fixed year is the later start
    until 2029."""
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
    ("steps", "reduces"),
    [
        pytest.param([_total()], True, id="an-aggregate-with-no-bound"),
        pytest.param(
            [_keeping(PeriodThreshold(op="lte", year=2024)), _total(input_id="kept")],
            False,
            id="under-lte",
        ),
        pytest.param(
            [_keeping(PeriodThreshold(op="lt", year=2025)), _total(input_id="kept")],
            False,
            id="under-lt",
        ),
        pytest.param(
            [
                _total(id="all", group_by=(GroupBy.PERIOD,)),
                _shares("all"),
                _ranked(input_id="shares", reduction=Reduction.MEAN),
            ],
            True,
            id="over-a-two-sided-step",
        ),
    ],
)
def test_a_reader_beside_a_branch_reading_the_whole_history_holds_the_stated_years(
    decision: Callable[..., Decision], steps: list[EvidenceRequirement], reduces: bool
) -> None:
    """The plan states 2024 alone. `latest` follows the newest data from its
    start, so it reads 'aid' from its first year through a copy of its own,
    and every other step reads 2024 and each newer year: a total with no
    bound reduces them, and one capped at 2024 reads one year."""
    ctx = _context(pins=_SHOWN_OPEN)
    aid = _open(period=ONE_YEAR_ONLY, decision_ids=("measure", "population", "pooled"))

    result = _propose(ctx, [aid, _latest_year(), *steps], [decision(), POPULATION, POOLED])

    if not reduces:
        assert result["outcome"] == Outcome.OK.value, result
        return
    assert result["code"] == ProblemCode.PERIOD_REDUCTION_UNSTATED.value, result
    (gap,) = result["detail"]["gaps"]
    assert (gap["requirement_id"], gap["years"]) == ("result", {"start": 2024, "end": 2024})
    assert "the years from 2024 to the newest" in result["message"]


@pytest.mark.usefixtures("aid_on_record")
@pytest.mark.parametrize(
    ("period", "steps"),
    [
        pytest.param(
            ONE_YEAR_ONLY,
            [
                _ranked(
                    axis=None,
                    reduction=None,
                    window_start=2024,
                    window_end=2024,
                    rolling=RollingWindow(window_end=_NEWEST),
                )
            ],
            id="a-window-whose-end-alone-follows",
        ),
        pytest.param(
            Period(start=2020, end=2024),
            [
                _keeping(PeriodThreshold(op="gte", year=2024, rolling=_NEWEST)),
                _ranked(input_id="kept", axis=None, reduction=None),
            ],
            id="a-first-year-on-the-anchor-ahead-of-an-end-open-to-the-data",
        ),
    ],
)
def test_a_one_year_rank_with_no_reduction_that_newer_data_widens_is_refused(
    decision: Callable[..., Decision], period: Period, steps: list[EvidenceRequirement]
) -> None:
    """2024 alone today, 2024 and 2025 once 2025 arrives or a series reports
    past the year half of them hold, when no entity has a single value to
    rank on."""
    kenya = _open(period=period).model_copy(update={"across": ()})

    result = _propose(_context(pins=_SHOWN_OPEN), [kenya, *steps], [decision(), POPULATION])

    assert result["outcome"] == Outcome.INVALID_INPUT.value, result
    assert result["code"] == ProblemCode.INVALID_PLAN.value, result


@pytest.mark.usefixtures("aid_on_record")
def test_a_fixed_window_beside_a_branch_reading_the_whole_history_ranks_the_stated_years(
    decision: Callable[..., Decision],
) -> None:
    """A rank window over 2020-2024 beside `latest`, which reads 'aid' from its
    first year, ranks the 2024 the dataset states, one year, and reduces none."""
    ctx = _context(pins=_SHOWN_OPEN)
    steps = [_latest_year(), _ranked(window_start=2020, window_end=2024)]

    result = _propose(ctx, [_open(period=ONE_YEAR_ONLY), *steps], [decision(), POPULATION])

    assert result["outcome"] == Outcome.OK.value, result


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
def test_a_rank_with_no_reduction_and_no_window_beside_a_rolling_branch_holds_the_open_years(
    decision: Callable[..., Decision],
) -> None:
    ctx = _context(pins=_SHOWN_OPEN)
    newest = _latest_year(axis=None, reduction=None)
    ranked = _ranked(axis=None, reduction=None)

    result = _propose(ctx, [_kenya_alone(end_open=True), newest, ranked], [decision(), POPULATION])

    assert result["code"] == ProblemCode.INVALID_PLAN.value, result
    assert "the years from 2024 to the newest with no reduction" in result["message"]


def _kept_from(year: int, input_id: str = "aid") -> FilterRequirement:
    """The years from `year` on, a fixed first year, to which newer data adds later ones."""
    return FilterRequirement(
        id="kept", input_id=input_id, predicates=(PeriodThreshold(op="gte", year=year),)
    )


def _newest_rank(input_id: str) -> RankRequirement:
    """A ranking of 2024 with no reduction, its window's end alone following the newest data."""
    return _ranked(
        input_id=input_id,
        axis="recipient",
        reduction=None,
        window_start=2024,
        window_end=2024,
        rolling=RollingWindow(window_end=_NEWEST),
    )


_PER_YEAR = _total(id="all", input_id="kept", group_by=(GroupBy.PERIOD,))


@pytest.mark.usefixtures("aid_on_record")
@pytest.mark.parametrize(
    ("steps", "decisions"),
    [
        pytest.param([_newest_rank("kept")], [POPULATION], id="a-rank-of-the-filter"),
        pytest.param(
            [
                _PER_YEAR,
                ShareOfRequirement(
                    id="shares",
                    decision_ids=("pooled",),
                    numerator=_side("kept"),
                    denominator=_side("all"),
                    axis="recipient",
                    denominator_scope=DenominatorScope.POOLED,
                ),
                _newest_rank("shares"),
            ],
            [POPULATION, POOLED],
            id="a-rank-of-a-share-of-the-filter",
        ),
        pytest.param(
            [
                _PER_YEAR,
                QuotientRequirement(
                    id="per_total",
                    numerator=_side("kept"),
                    denominator=_side("all"),
                    axis="recipient",
                ),
                _newest_rank("per_total"),
            ],
            [POPULATION],
            id="a-rank-of-a-quotient-of-the-filter",
        ),
    ],
)
def test_a_rank_with_no_reduction_over_a_fixed_first_year_newer_data_widens_is_refused(
    decision: Callable[..., Decision], steps: list[EvidenceRequirement], decisions: list[Decision]
) -> None:
    """`year >= 2024` over an open 2024 holds 2024 today and 2024 and 2025 once
    2025 arrives, when no entity has a single value to rank on."""
    kenya = _open(period=ONE_YEAR_ONLY).model_copy(update={"across": ()})

    result = _propose(
        _context(pins=_SHOWN_OPEN), [kenya, _kept_from(2024), *steps], [decision(), *decisions]
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value, result
    assert "with no reduction" in result["message"]


def test_a_year_excluded_at_the_start_of_a_closed_period_leaves_one_year_to_total(
    decision: Callable[..., Decision],
) -> None:
    """`year != 2023` over 2023-2024 leaves 2024 alone, so the total reduces no years."""
    kept = _keeping(PeriodThreshold(op="ne", year=2023))

    result = _propose(
        _context(),
        [_aid(period=Period(start=2023, end=2024)), kept, _total(input_id="kept")],
        [decision()],
    )

    assert result["outcome"] == Outcome.OK.value, result


@pytest.mark.usefixtures("aid_on_record")
def test_a_fixed_first_year_over_a_limited_rank_per_year_of_open_data_keeps_its_cut(
    decision: Callable[..., Decision],
) -> None:
    """`year >= 2024` keeps 2024 today and every later year as it arrives, so the
    total by donor reads the years each donor made the top five in."""
    top = _ranked(id="top", per_period=True, limit=5)
    steps = [
        _two_donors().model_copy(
            update={
                "period": Period(start=2020, end=2024),
                "period_end_open": True,
                "source_spec_id": _OPEN_SPEC,
            }
        ),
        top,
        _kept_from(2024, "top"),
        _total(input_id="kept"),
    ]

    result = _propose(
        _context(pins=_SHOWN_OPEN, across_years=(0,)), steps, [decision(), POPULATION]
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value, result
    assert "per-period rank with a limit" in result["message"]


@pytest.mark.usefixtures("aid_on_record")
def test_a_rank_with_no_reduction_over_a_max_per_donor_of_the_newest_year_is_admitted(
    decision: Callable[..., Decision],
) -> None:
    """The newest year as it moves stays one year, so each donor's max is one row
    of it whatever arrives, and the rank needs no reduction."""
    newest = _keeping(PeriodThreshold(op="eq", year=2024, rolling=_NEWEST))
    best = _total(id="best", input_id="kept", reduction=Reduction.MAX)
    ranked = _ranked(input_id="best", axis="recipient", reduction=None)

    result = _propose(
        _context(pins=_SHOWN_OPEN),
        [_open(period=ONE_YEAR_ONLY), newest, best, ranked],
        [decision(), POPULATION],
    )

    assert result["outcome"] == Outcome.OK.value, result


@pytest.mark.usefixtures("aid_on_record")
def test_a_max_per_donor_of_an_open_year_picks_a_year_per_donor_once_newer_data_arrives(
    decision: Callable[..., Decision],
) -> None:
    """2024 alone today, and each donor's max falls in a year of its own once 2025
    arrives, which a rank refuses whatever its reduction."""
    best = _total(id="best", reduction=Reduction.MAX)
    ranked = _ranked(input_id="best", axis="recipient", reduction=None)

    result = _propose(
        _context(pins=_SHOWN_OPEN, across_years=(0,)),
        [_open(period=ONE_YEAR_ONLY), best, ranked],
        [decision(), POPULATION],
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value, result
    assert "max or min at the year it fell in" in result["message"]


def test_a_group_by_naming_a_column_twice_is_refused_as_an_invalid_plan(
    decision: Callable[..., Decision],
) -> None:
    """Grouping by `entity` twice is no grouping the refusal can add `period` to."""
    repeated = {**_total().model_dump(mode="json"), "group_by": ["entity", "entity"]}

    result = _propose(
        _context(),
        [_aid(), repeated],
        [decision()],
        answered=[AskAnswer(ask=0, outputs=("result",))],
    )

    assert result["outcome"] == Outcome.INVALID_INPUT.value, result
    assert result["code"] == ProblemCode.INVALID_PLAN.value, result
    assert "group_by" in result["message"]


def _step_of(requirements: list[EvidenceRequirement], step_id: str) -> tuple[int, Any]:
    (index,) = [i for i, step in enumerate(requirements) if step.id == step_id]
    return index, requirements[index]


def _follow(requirements: list[EvidenceRequirement], refused: dict[str, Any]) -> None:
    """Apply to `requirements` the one-year change `refused` offers, as the planner reads it.

    The step, the kind of change, the year and the offset come from the
    gap's `one_year`, and the edit from the sentence the refusal states it
    in, written here without `planner.plan.one_year_followed`. That the two
    agree is what this checks.
    """
    (gap,) = refused["detail"]["gaps"]
    change, step_id = gap["one_year"], gap["requirement_id"]
    year, offset = change["year"], change["offset"]
    rolling = None if offset is None else RelativePeriod(offset=offset)
    index, step = _step_of(requirements, step_id)
    if OneYear(change["change"]) is OneYear.WINDOW:
        rolls = "`rolling` unset" if offset is None else f"`rolling` naming both at offset {offset}"
        assert (
            f"set `window_start` and `window_end` on {step_id!r} to {year}, with {rolls}"
            in refused["message"]
        )
        window = (
            None if rolling is None else RollingWindow(window_start=rolling, window_end=rolling)
        )
        requirements[index] = step.model_copy(
            update={"window_start": year, "window_end": year, "rolling": window}
        )
        return
    rolls = "`rolling` unset" if offset is None else f"`rolling` at offset {offset}"
    assert (
        f"have {step_id!r} read its input {step.input_id!r} through a `filter` whose "
        f"`period_threshold` keeps {year} alone, with op `eq` and {rolls}" in refused["message"]
    )
    one_year = FilterRequirement(
        id="one_year",
        input_id=step.input_id,
        predicates=(PeriodThreshold(op="eq", year=year, rolling=rolling),),
    )
    requirements[index : index + 1] = [one_year, step.model_copy(update={"input_id": "one_year"})]


def _open_two_donors(period: Period) -> DatasetRequirement:
    """Aid to Kenya from France and Germany retrieved with no end year (`_open`)."""
    return _open(period=period).model_copy(
        update={"entities": _two_donors().entities, "across": ()}
    )


def _france() -> FilterRequirement:
    """Kenya's aid from France, a donor a filter keeps only where the requirement names it."""
    return FilterRequirement(
        id="france", input_id="aid", predicates=(EntityIn(names=("France",), role="donor"),)
    )


# Ask 1 answered by a series of its own, beside the total answering ask 0.
_BESIDE_A_SERIES = (ASK, OTHER_ASK)


def _unpinned_open() -> DatasetRequirement:
    """Aid to Kenya over 2020-2024 with no end year and no pin, which Search finds from 2020 on."""
    return _aid(pin=None, period=Period(start=2020, end=2024)).model_copy(
        update={"period_end_open": True}
    )


def _offered(
    requirements: list[EvidenceRequirement],
    decisions: list[Decision],
    change: OneYear,
    offset: int | None,
    *,
    id: str,
    sibling: str | None = None,
    year: int | None = None,
) -> Any:
    """A row whose refusal offers `change` at `offset`, `sibling` answering ask 1 beside it.

    `year` is the year offered, `None` for the last year the step holds today.
    """
    return pytest.param(requirements, decisions, change, offset, sibling, year, id=id)


# Each plan a `period_reduction_unstated` refusal offers one year to read, with
# the change, its offset, the sibling answering ask 1 beside the refused step
# and the year offered (`_offered`).
_ONE_YEAR_OFFERS = [
    _offered([_aid(), _ranked()], [POPULATION], OneYear.WINDOW, None, id="a-rank"),
    _offered(
        [
            _open(period=ONE_YEAR_ONLY),
            _ranked(window_start=2024, window_end=2024, rolling=RollingWindow(window_end=_NEWEST)),
        ],
        [POPULATION],
        OneYear.WINDOW,
        0,
        id="a-rank-whose-window-end-alone-follows",
    ),
    _offered(
        [
            _open(period=Period(start=2020, end=2024)),
            _keeping(PeriodThreshold(op="lte", year=2023, rolling=RelativePeriod(offset=-1))),
            _ranked(input_id="kept"),
        ],
        [POPULATION],
        OneYear.WINDOW,
        -1,
        id="a-rank-of-years-up-to-the-one-before-the-newest",
    ),
    _offered(
        [
            _open(period=Period(start=2020, end=2024)),
            _keeping(PeriodThreshold(op="lt", year=2024, rolling=_NEWEST)),
            _ranked(input_id="kept"),
        ],
        [POPULATION],
        OneYear.WINDOW,
        -1,
        id="a-rank-of-years-before-the-newest",
    ),
    _offered([_aid(), _total()], [], OneYear.FILTER, None, id="a-total-of-fixed-years"),
    _offered(
        [_aid(), _total(group_by=())],
        [],
        OneYear.FILTER,
        None,
        id="one-figure-of-fixed-years",
    ),
    _offered(
        [_open(period=ONE_YEAR_ONLY), _total(group_by=())],
        [],
        OneYear.FILTER,
        0,
        id="one-figure-of-an-open-end",
    ),
    _offered(
        [_unpinned_open(), _total()],
        [],
        OneYear.FILTER,
        None,
        id="a-total-of-an-open-end-with-no-pin",
    ),
    _offered(
        [_unpinned_open(), _ranked()],
        [POPULATION],
        OneYear.WINDOW,
        None,
        id="a-rank-of-an-open-end-with-no-pin",
    ),
    _offered(
        [_open(period=ONE_YEAR_ONLY), _total()],
        [],
        OneYear.FILTER,
        0,
        id="a-total-of-an-open-end",
    ),
    _offered(
        [_open(period=ONE_YEAR_ONLY), _kept_from(2024), _total(input_id="kept")],
        [],
        OneYear.FILTER,
        0,
        id="a-total-from-a-fixed-first-year-of-an-open-end",
    ),
    _offered(
        [_open(period=ONE_YEAR_ONLY), _latest_year(), _total()],
        [POPULATION],
        OneYear.FILTER,
        0,
        id="a-total-beside-a-branch-reading-from-the-first-year",
    ),
    _offered(
        [_open(period=ONE_YEAR_ONLY), _latest_year(), _ranked()],
        [POPULATION],
        OneYear.WINDOW,
        0,
        id="a-rank-beside-a-branch-reading-from-the-first-year",
    ),
    _offered(
        [
            _open_two_donors(ONE_YEAR_ONLY),
            _latest_year(),
            _france(),
            _total(input_id="france"),
        ],
        [POPULATION],
        OneYear.FILTER,
        0,
        id="a-total-of-one-donor-read-from-the-first-year",
    ),
    _offered(
        [*_both(), _total(input_id="both")],
        [POPULATION, POOLED],
        OneYear.FILTER,
        None,
        id="a-total-of-a-sum",
    ),
    _offered(
        [
            _open(period=Period(start=2020, end=2024)),
            _keeping(PeriodThreshold(op="gte", year=2024, rolling=_NEWEST)),
            _total(input_id="kept"),
        ],
        [],
        OneYear.FILTER,
        0,
        id="a-total-from-a-first-year-on-the-anchor",
    ),
    _offered(
        [
            _open(period=ONE_YEAR_ONLY),
            _keeping(PeriodThreshold(op="gte", year=2019, rolling=_FIVE_BACK)),
            _total(input_id="kept"),
        ],
        [],
        OneYear.FILTER,
        0,
        id="a-total-from-five-years-back",
    ),
    _offered(
        [
            _open(period=Period(start=2020, end=2024)),
            _keeping(
                PeriodThreshold(op="gte", year=2020),
                PeriodThreshold(op="lte", year=2023, rolling=RelativePeriod(offset=-1)),
            ),
            _total(input_id="kept"),
        ],
        [],
        OneYear.FILTER,
        -1,
        id="a-total-up-to-the-year-before-the-newest",
    ),
    _offered(
        [
            _open(period=Period(start=2020, end=2024)),
            _keeping(PeriodThreshold(op="lt", year=2024, rolling=_NEWEST)),
            _total(input_id="kept"),
        ],
        [],
        OneYear.FILTER,
        -1,
        id="a-total-of-the-years-before-the-newest",
    ),
    _offered(
        [
            _open(period=Period(start=2020, end=2024)),
            _keeping(PeriodThreshold(op="lte", year=2026)),
            _total(input_id="kept"),
        ],
        [],
        OneYear.FILTER,
        None,
        id="a-total-of-an-open-end-a-fixed-last-year-stops",
    ),
    _offered(
        [
            _open(period=Period(start=2020, end=2024)),
            _keeping(
                PeriodThreshold(op="lte", year=2026),
                PeriodThreshold(op="lte", year=2023, rolling=RelativePeriod(offset=-1)),
            ),
            _total(input_id="kept"),
        ],
        [],
        OneYear.FILTER,
        None,
        id="a-total-up-to-the-year-before-the-newest-a-fixed-last-year-stops",
    ),
    _offered(
        [
            _open(period=ONE_YEAR_ONLY),
            _ranked(
                id="yearly",
                per_period=True,
                window_start=2024,
                window_end=2024,
                rolling=RollingWindow(window_end=_NEWEST),
            ),
            _total(),
        ],
        [POPULATION],
        OneYear.FILTER,
        0,
        id="a-total-of-an-open-end-beside-a-rolling-branch",
    ),
    _offered(
        [
            _open(period=Period(start=2020, end=2024)),
            _ranked(
                id="yearly",
                per_period=True,
                window_start=2024,
                window_end=2024,
                rolling=RollingWindow(window_end=_NEWEST),
            ),
            _keeping(PeriodThreshold(op="lte", year=2023)),
            _total(input_id="kept"),
        ],
        [POPULATION],
        OneYear.FILTER,
        None,
        id="a-total-of-fixed-years-beside-a-rank-of-the-newest",
        sibling="yearly",
    ),
    _offered(
        [_aid(), _YEARLY, _total()],
        [],
        OneYear.FILTER,
        None,
        id="a-total-beside-a-series-of-the-same-dataset",
        sibling="yearly",
    ),
    _offered(
        [
            _aid(),
            _total(id="yearly", group_by=(GroupBy.PERIOD,)),
            GrowthRequirement(
                id="change",
                input_id="yearly",
                window_start=2015,
                window_end=2024,
                change=GrowthChange.ABSOLUTE,
            ),
            _total(),
        ],
        [],
        OneYear.FILTER,
        None,
        id="a-total-beside-a-growth-of-the-same-dataset",
        sibling="change",
    ),
    _offered(
        [
            _open(period=Period(start=2020, end=2024)),
            _keeping(PeriodThreshold(op="gte", year=2021, rolling=RelativePeriod(offset=-2))),
            _total(input_id="kept"),
        ],
        [],
        OneYear.FILTER,
        0,
        id="a-total-of-an-open-end-from-a-year-before-a-newest-year-the-data-passes",
        year=2023,
    ),
]


@pytest.mark.usefixtures("aid_on_record")
@pytest.mark.parametrize(
    ("requirements", "decisions", "change", "offset", "sibling", "year"), _ONE_YEAR_OFFERS
)
def test_the_one_year_each_refusal_offers_is_admitted(
    decision: Callable[..., Decision],
    requirements: list[EvidenceRequirement],
    decisions: list[Decision],
    change: OneYear,
    offset: int | None,
    sibling: str | None,
    year: int | None,
) -> None:
    """Each refusal offers a change to the refused step alone, on the last
    year its input holds. The year rolls at the offset newer data moves it
    by, under the newest year the step's other rolling years stand for,
    where the dataset's first year is on record, since the rolling year
    reads its whole history through copies of its own and every other step
    keeps its years. It is fixed where no first year is on record, for a
    dataset with no pin or one whose pin no answer over every year states. The plan
    that follows the change is admitted, and every other step, the
    dataset's `period` and the inputs between the step and the dataset
    included, is left as stated."""
    cited = [decision(), *decisions]
    asks, answered = (ASK,), None
    if sibling is not None:
        asks = _BESIDE_A_SERIES
        answered = [
            AskAnswer(ask=0, outputs=(requirements[-1].id,)),
            AskAnswer(ask=1, outputs=(sibling,)),
        ]
    # Both sides of a sum pin one dataset, so adding them needs no pick from the user.
    pins = {**_SHOWN_OPEN, SPEC: PinSource(provenance_ref="oecd/dac2a")}

    refused = _propose(_context(asks=asks, pins=pins), requirements, cited, answered=answered)
    (gap,) = refused["detail"]["gaps"]
    advised = list(requirements)
    _follow(advised, refused)
    read = OneYearRead(year=gap["one_year"]["year"], offset=gap["one_year"]["offset"])
    assert advised == one_year_followed(requirements, gap["requirement_id"], read)
    ctx = _context(asks=asks, pins=pins)
    admitted = _propose(ctx, advised, cited, answered=answered)

    offered = gap["years"]["end"] if year is None else year
    assert gap["one_year"] == {
        "change": change,
        "year": offered,
        "offset": offset,
        "together_with": [],
    }
    assert admitted["outcome"] == Outcome.OK.value, admitted
    stored = load_plan(ctx.state)
    assert stored is not None
    refused_step = gap["requirement_id"]
    # Admission pins each dataset to the source of the window it reads.
    unpinned = {"source_spec_id"}
    assert [
        step.model_dump(exclude=unpinned)
        for step in authored(stored).evidence_requirements
        if step.id not in (refused_step, "one_year")
    ] == [step.model_dump(exclude=unpinned) for step in requirements if step.id != refused_step]
    _, before = _step_of(requirements, refused_step)
    if change is OneYear.FILTER:
        assert authored(stored).requirement("one_year").input_id == before.input_id
        assert stored.requirement(refused_step).input_id == "one_year"
    else:
        assert stored.requirement(refused_step).input_id == before.input_id
    kept = [
        output
        for output in answering_results(stored)
        if output not in _computed_from(stored, refused_step)
    ]
    windowed, minted = _window_pins(
        _recorded(_model_plan(requirements, cited, asks=asks, answered=answered)), shown=pins
    )
    assert _windows_read(stored, windowed_pins(ctx.state), kept) == _windows_read(
        windowed, {copy.spec_id: copy for copy in minted}, kept
    )


def _windows_read(
    plan: Plan, minted: Mapping[str, WindowedPin], outputs: Iterable[str]
) -> dict[str, dict[str, object]]:
    """For each of `outputs`, what each retrieval it rests on reads, by the dataset requirement the model wrote.

    A retrieval pinning a copy `minted` holds reads that copy's
    `requested_window`, and one pinning its source as it stands reads the
    source's own window, named by its pin.
    """
    split = split_of(plan)
    by_id = {requirement.id: requirement for requirement in split.requirements}

    def datasets(step_id: str) -> set[str]:
        inputs = inputs_of(by_id[step_id])
        return {step_id} if not inputs else set().union(*map(datasets, inputs))

    read: dict[str, dict[str, object]] = {}
    for output in outputs:
        read[output] = {}
        for dataset_id in sorted(datasets(output)):
            dataset = by_id[dataset_id]
            assert isinstance(dataset, DatasetRequirement)
            pin = dataset.source_spec_id
            read[output][origin_of(split, dataset_id)] = (
                minted[pin].identity["requested_window"] if pin in minted else pin
            )
    return read


def _from_newest(id: str, offset: int) -> FilterRequirement:
    """2020 on over the max per donor of the open 2020-2024, as a first year `offset` before the newest year."""
    return FilterRequirement(
        id=id,
        input_id="all",
        predicates=(PeriodThreshold(op="gte", year=2020, rolling=RelativePeriod(offset=offset)),),
    )


@pytest.mark.usefixtures("aid_on_record")
@pytest.mark.parametrize(
    ("steps", "answered", "across_years"),
    [
        pytest.param(
            [
                _keeping(
                    PeriodThreshold(op="gt", year=2023),
                    PeriodThreshold(op="gte", year=2021, rolling=RelativePeriod(offset=-2)),
                ),
                _total(input_id="kept"),
            ],
            None,
            (),
            id="a-newest-year-the-total-does-not-hold",
        ),
    ],
)
def test_a_refusal_offers_no_one_year_where_the_newest_year_is_not_held(
    decision: Callable[..., Decision],
    steps: list[EvidenceRequirement],
    answered: list[AskAnswer] | None,
    across_years: tuple[str, ...],
) -> None:
    """From 2024 on, below a first year two before a newest year of 2023, the
    newest year is no year the total holds, and a fixed one is passed once
    the rolling first year moves past it, so the refusal offers only keeping
    the years. A max per donor that two results each keep from a rolling
    first year of their own, on newest years of 2024 and 2023, joins both
    groups once it rolls, so no rolling year agrees with both, and a fixed
    year is passed by either first year."""
    asks = (ASK,) if answered is None else _BESIDE_A_SERIES
    ctx = _context(asks=asks, pins=_SHOWN_OPEN)
    dataset = _open(period=Period(start=2020, end=2024))

    refused = _propose(
        ctx,
        [dataset, *steps],
        [decision()],
        answered=answered,
        across_years=list(across_years),
        reads_user_words=bool(across_years),
    )

    assert refused["code"] == ProblemCode.PERIOD_REDUCTION_UNSTATED.value, refused
    (gap,) = refused["detail"]["gaps"]
    assert gap["one_year"] is None
    assert "To read one year" not in refused["message"]


def _open_series(requirement_id: str, *, measure: str = "net ODA") -> DatasetRequirement:
    """One donor's flow to Kenya over 2020-2024 with no end year."""
    return _series(requirement_id, measure=measure, period=Period(start=2020, end=2024)).model_copy(
        update={"source_spec_id": _OPEN_SPEC, "period_end_open": True}
    )


def _totals_paired(
    verb: type[QuotientRequirement | SumOfRequirement], *, open_: bool, by_value: bool = False
) -> list:
    """The total of two series, over 2020-2024 open or fixed, divided or summed, side by side.

    With `by_value` the first total sums only the years a value filter kept.
    """
    series = (
        _open_series
        if open_
        else (lambda rid, **kw: _series(rid, period=Period(start=2020, end=2024), **kw))
    )
    sides = (_side("bilateral_total"), _side("imputed_total"))
    pair = (
        QuotientRequirement(id="pair", numerator=sides[0], denominator=sides[1])
        if verb is QuotientRequirement
        else SumOfRequirement(id="pair", decision_ids=("measure",), left=sides[0], right=sides[1])
    )
    kept = FilterRequirement(
        id="kept",
        decision_ids=("measure",),
        input_id="bilateral",
        predicates=(ValueThreshold(op="gt", value=84.357),),
    )
    return [
        series("bilateral"),
        series("imputed", measure="imputed multilateral aid"),
        *([kept] if by_value else []),
        _total(id="bilateral_total", input_id="kept" if by_value else "bilateral"),
        _total(id="imputed_total", input_id="imputed"),
        pair,
    ]


def _each_donors_part(
    verb: type[ShareOfRequirement | QuotientRequirement], *, by_value: bool = False
) -> list:
    """Each donor's total over 2020-2024 with no end year, against the total of those totals.

    With `by_value` each donor's total sums only the years a value filter kept.
    """
    sides = (_side("per_donor"), _side("overall"))
    part = (
        ShareOfRequirement(
            id="part",
            decision_ids=("pooled",),
            numerator=sides[0],
            denominator=sides[1],
            axis="donor",
            denominator_scope=DenominatorScope.POOLED,
        )
        if verb is ShareOfRequirement
        else QuotientRequirement(id="part", numerator=sides[0], denominator=sides[1])
    )
    donors = _two_donors().model_copy(
        update={
            "source_spec_id": _OPEN_SPEC,
            "period": Period(start=2020, end=2024),
            "period_end_open": True,
        }
    )
    kept = FilterRequirement(
        id="kept",
        decision_ids=("measure",),
        input_id="aid",
        predicates=(ValueThreshold(op="gt", value=84.357),),
    )
    return [
        donors,
        *([kept] if by_value else []),
        _total(id="per_donor", input_id="kept" if by_value else "aid"),
        _total(id="overall", input_id="per_donor", group_by=(GroupBy.PERIOD,)),
        part,
    ]


def _growth(**overrides: Any) -> GrowthRequirement:
    fields: dict[str, Any] = {
        "id": "change",
        "input_id": "aid",
        "window_start": 2020,
        "window_end": 2024,
        "change": GrowthChange.ABSOLUTE,
        "rolling": RollingWindow(window_end=_NEWEST),
    }
    return GrowthRequirement(**(fields | overrides))


_OVER_2020_TO_2024 = {"window_start": 2020, "window_end": 2024}
_ROLLING_END = {**_OVER_2020_TO_2024, "rolling": RollingWindow(window_end=_NEWEST)}


@pytest.mark.usefixtures("aid_on_record")
@pytest.mark.parametrize(
    ("requirements", "decisions", "admitted"),
    [
        pytest.param(
            [
                _open(period=Period(start=2020, end=2024)),
                _total(id="total"),
                _ranked(input_id="total", axis="recipient", reduction=None, **_OVER_2020_TO_2024),
            ],
            [POPULATION],
            False,
            id="a-fixed-window-over-a-total-of-open-years",
        ),
        pytest.param(
            [
                _open(period=Period(start=2020, end=2024)),
                _growth(),
                _ranked(input_id="change", **_OVER_2020_TO_2024),
            ],
            [POPULATION],
            False,
            id="a-fixed-window-over-a-growth-whose-end-rolls",
        ),
        pytest.param(
            [
                _open(period=Period(start=2020, end=2024)),
                _growth(),
                _ranked(input_id="change", **_ROLLING_END),
            ],
            [POPULATION],
            True,
            id="a-rolling-end-over-a-growth-whose-end-rolls",
        ),
        pytest.param(
            [
                _open(period=Period(start=2020, end=2024)),
                _ranked(id="per_donor", **_ROLLING_END),
                _ranked(input_id="per_donor", reduction=None, **_OVER_2020_TO_2024),
            ],
            [POPULATION],
            False,
            id="a-fixed-window-over-a-rank-whose-end-rolls",
        ),
        pytest.param(
            _totals_paired(QuotientRequirement, open_=True),
            [POPULATION, POOLED],
            False,
            id="a-quotient-of-totals-of-open-years",
        ),
        pytest.param(
            _totals_paired(SumOfRequirement, open_=True),
            [POPULATION, POOLED],
            False,
            id="a-sum-of-totals-of-open-years",
        ),
        pytest.param(
            _totals_paired(QuotientRequirement, open_=False),
            [POPULATION, POOLED],
            True,
            id="a-quotient-of-totals-of-fixed-years",
        ),
        pytest.param(
            _totals_paired(QuotientRequirement, open_=False, by_value=True),
            [POPULATION, POOLED],
            False,
            id="a-quotient-of-a-total-of-the-years-a-value-filter-kept",
        ),
        pytest.param(
            _totals_paired(SumOfRequirement, open_=False, by_value=True),
            [POPULATION, POOLED],
            False,
            id="a-sum-of-a-total-of-the-years-a-value-filter-kept",
        ),
        pytest.param(
            _each_donors_part(ShareOfRequirement),
            [POOLED],
            True,
            id="a-share-of-a-total-of-the-same-open-years",
        ),
        pytest.param(
            _each_donors_part(QuotientRequirement),
            [],
            True,
            id="a-quotient-of-a-total-of-the-same-open-years",
        ),
        pytest.param(
            _each_donors_part(QuotientRequirement, by_value=True),
            [],
            True,
            id="a-quotient-of-a-total-of-the-same-years-a-value-filter-kept",
        ),
    ],
)
def test_a_span_newer_data_moves_matches_no_fixed_window_or_span_that_stays(
    decision: Callable[..., Decision],
    requirements: list[EvidenceRequirement],
    decisions: list[Decision],
    admitted: bool,
) -> None:
    """analysis-mcp labels a total with every year it summed, a growth with
    its window and a rank with the years its window kept, and keeps a span
    row only under a window holding its label, or beside a side of the same
    label. A total of open years ends at the newest year the data reports,
    and a rolling window end follows the newest year, so a fixed window or a
    span over another dataset loses the label once newer data arrives. A
    window whose end follows the newest year as the span's does keeps it,
    and so does a total of that total, which keeps its label. A total of
    the years a value filter kept spans the years only the data leaves,
    such as 2016-2020 where France's 2015 value falls below the threshold,
    so it matches no span another step wrote."""
    # Both sides pin one dataset, so adding them needs no pick from the user.
    pins = {**_SHOWN_OPEN, SPEC: PinSource(provenance_ref="oecd/dac2a")}
    ctx = _context(across_years=[0], pins=pins)

    result = _propose(ctx, requirements, [decision(), *decisions])

    if admitted:
        assert result["outcome"] == Outcome.OK.value, result
    else:
        assert result["code"] == ProblemCode.INVALID_PLAN.value, result


def _keeping_from(
    requirement_id: str, input_id: str, *predicates: PeriodThreshold
) -> FilterRequirement:
    return FilterRequirement(
        id=requirement_id, decision_ids=("measure",), input_id=input_id, predicates=predicates
    )


@pytest.mark.parametrize(
    "trailing_first", [False, True], ids=["numerator-first", "denominator-first"]
)
def test_a_mean_over_a_pair_with_a_trailing_side_reduces_years_in_either_side_order(
    decision: Callable[..., Decision], trailing_first: bool
) -> None:
    """One side holds 2020 to the newest year. The other holds 2024 alone
    today, from 2024 or from six years back of the newest, whichever is later.
    Their quotient holds 2024 today, and 2024 and 2025 once 2025 arrives, so
    a mean of it by entity reduces years, which no ask across years licenses."""
    newest = RelativePeriod(offset=0)
    sides = [
        _open_series("bilateral"),
        _keeping_from("left", "bilateral", PeriodThreshold(op="lte", year=2024, rolling=newest)),
        _open_series("imputed", measure="imputed multilateral aid"),
        _keeping_from(
            "right",
            "imputed",
            PeriodThreshold(op="gte", year=2024),
            PeriodThreshold(op="gte", year=2018, rolling=RelativePeriod(offset=-6)),
            PeriodThreshold(op="lte", year=2024, rolling=newest),
        ),
    ]
    order = (_side("right"), _side("left")) if trailing_first else (_side("left"), _side("right"))
    pair = QuotientRequirement(id="pair", numerator=order[0], denominator=order[1])
    pins = {**_SHOWN_OPEN, SPEC: PinSource(provenance_ref="oecd/dac2a")}

    result = _propose(
        _context(pins=pins),
        [*sides, pair, _total(input_id="pair", reduction=Reduction.MEAN)],
        [decision(), POPULATION, POOLED],
    )

    assert result["code"] == ProblemCode.PERIOD_REDUCTION_UNSTATED.value, result
    assert [gap["requirement_id"] for gap in result["detail"]["gaps"]] == ["result"]


def _paired(kind: str, first: str, second: str) -> EvidenceRequirement:
    """`kind`'s requirement pairing `first` with `second`, in that order."""
    sides = (_side(first), _side(second))
    if kind == "quotient_of":
        return QuotientRequirement(id="pair", numerator=sides[0], denominator=sides[1])
    if kind == "share_of":
        return ShareOfRequirement(
            id="pair", decision_ids=("pooled",), numerator=sides[0], denominator=sides[1]
        )
    if kind == "compare":
        return ComparisonRequirement(
            id="pair", decision_ids=("measure",), minuend=sides[0], subtrahend=sides[1]
        )
    return SumOfRequirement(id="pair", decision_ids=("measure",), left=sides[0], right=sides[1])


_PAIRING_VERBS = ("quotient_of", "share_of", "compare", "sum_of")
_SIDE_ORDERS = pytest.mark.parametrize("swap", [False, True], ids=["as-stated", "swapped"])


@pytest.mark.parametrize("kind", _PAIRING_VERBS)
@pytest.mark.parametrize("other", ["imputed_total", "imputed"], ids=["a-total", "a-series"])
@_SIDE_ORDERS
def test_a_total_of_the_rows_a_value_filter_kept_pairs_with_no_other_steps_result(
    decision: Callable[..., Decision], kind: str, other: str, swap: bool
) -> None:
    """Over fixed 2020-2024, one total sums only the years a value filter
    kept, which analysis-mcp labels with the years those rows fall in. No
    other step's total or yearly series is known to carry that label, and no
    change to the pair computes the figure it states, so the refusal names
    none."""
    sides = ("bilateral_total", other)
    requirements = [
        *(
            r
            for r in _totals_paired(QuotientRequirement, open_=False, by_value=True)[:-1]
            if r.id != "imputed_total" or other == "imputed_total"
        ),
        _paired(kind, *(sides[::-1] if swap else sides)),
    ]
    pins = {**_SHOWN_OPEN, SPEC: PinSource(provenance_ref="oecd/dac2a")}

    refused = _propose(
        _context(across_years=[0], pins=pins), requirements, [decision(), POPULATION, POOLED]
    )

    assert refused["code"] == ProblemCode.INVALID_PLAN.value, refused
    assert VALUE_KEPT_SPAN_PAIRS in refused["message"], refused["message"]


@pytest.mark.parametrize("counterpart", ["kept", "kept_total"], ids=["a-series", "a-total"])
@_SIDE_ORDERS
def test_a_max_by_entity_pairs_with_no_series_a_value_filter_kept(
    decision: Callable[..., Decision], counterpart: str, swap: bool
) -> None:
    """Over fixed 2020-2024, each donor's max sits at the year it fell in, and
    a value filter can leave that year out of the other side, so the pair is
    refused in either order. A total of the kept rows is refused the same way,
    since reading it as a yearly series leaves the kept rows."""
    sides = ("peak", counterpart)
    requirements = [
        _series("bilateral", period=Period(start=2020, end=2024)),
        _series("imputed", measure="imputed multilateral aid", period=Period(start=2020, end=2024)),
        _total(id="peak", input_id="bilateral", reduction=Reduction.MAX),
        FilterRequirement(
            id="kept",
            decision_ids=("measure",),
            input_id="imputed",
            predicates=(ValueThreshold(op="gt", value=84.357),),
        ),
        *([_total(id="kept_total", input_id="kept")] if counterpart == "kept_total" else []),
        _paired("quotient_of", *(sides[::-1] if swap else sides)),
    ]
    pins = {**_SHOWN_OPEN, SPEC: PinSource(provenance_ref="oecd/dac2a")}

    refused = _propose(_context(pins=pins), requirements, [decision(), POPULATION, POOLED])

    assert refused["code"] == ProblemCode.INVALID_PLAN.value, refused
    assert "no plan pairs a max or min with the rows a value filter kept" in refused["message"]


def _ranks_compared() -> list:
    """France and Germany ranked by the sum and by the mean of their aid from 2020 to the newest
    year, France's rank by one less its rank by the other."""
    return [
        _open_two_donors(Period(start=2020, end=2024)),
        _ranked(id="by_sum", **_ROLLING_END),
        _ranked(id="by_mean", reduction=Reduction.MEAN, **_ROLLING_END),
        ComparisonRequirement(
            id="gap",
            decision_ids=("measure",),
            minuend=RequirementSide(input_id="by_sum", entity="France"),
            subtrahend=RequirementSide(input_id="by_mean", entity="France"),
        ),
    ]


@pytest.mark.usefixtures("aid_on_record")
@pytest.mark.parametrize(
    ("requirements", "decisions", "offers"),
    [
        pytest.param(
            _totals_paired(QuotientRequirement, open_=False),
            [POPULATION, POOLED],
            {"bilateral_total": (OneYear.FILTER, None), "imputed_total": (OneYear.FILTER, None)},
            id="a-quotient-of-totals-of-fixed-years",
        ),
        pytest.param(
            _ranks_compared(),
            [POPULATION],
            {"by_sum": (OneYear.WINDOW, 0), "by_mean": (OneYear.WINDOW, 0)},
            id="a-comparison-of-ranks-of-open-years",
        ),
    ],
)
def test_the_sides_of_a_pairing_are_offered_one_year_together(
    decision: Callable[..., Decision],
    requirements: list[EvidenceRequirement],
    decisions: list[Decision],
    offers: dict[str, tuple[OneYear, int | None]],
) -> None:
    """Each side of the pairing reduces 2020-2024, and either one read alone
    in 2024 matches no row of the other's span. Both read in 2024 match, so
    the refusal offers each its year together with the other, on the newest
    year where the sides' years follow it. The plan following both is
    admitted, and one following either alone is not."""
    pins = {**_SHOWN_OPEN, SPEC: PinSource(provenance_ref="oecd/dac2a")}
    cited = [decision(), *decisions]

    refused = _propose(_context(pins=pins), requirements, cited)

    assert refused["code"] == ProblemCode.PERIOD_REDUCTION_UNSTATED.value, refused
    gaps = refused["detail"]["gaps"]
    assert {gap["requirement_id"]: gap["one_year"] for gap in gaps} == {
        step_id: {
            "change": change,
            "year": 2024,
            "offset": offset,
            "together_with": [other for other in offers if other != step_id],
        }
        for step_id, (change, offset) in offers.items()
    }
    for step_id in offers:
        others = " and ".join(repr(other) for other in offers if other != step_id)
        assert f"together with the one-year change offered to {others}" in refused["message"]
    reads = {
        gap["requirement_id"]: OneYearRead(
            year=gap["one_year"]["year"], offset=gap["one_year"]["offset"]
        )
        for gap in gaps
    }
    followed = list(requirements)
    for step_id, read in reads.items():
        alone = one_year_followed(requirements, step_id, read)
        assert _propose(_context(pins=pins), alone, cited)["outcome"] != Outcome.OK.value
        followed = one_year_followed(followed, step_id, read)
    admitted = _propose(_context(pins=pins), followed, cited)
    assert admitted["outcome"] == Outcome.OK.value, admitted


# --- The whole-history split -------------------------------------------------
#
# A dataset read from its first year serves only the steps rolling from their
# start (`planner.plan.whole_history_split`), and every other reader keeps its
# stated years. These rows read the plan `propose_plan` validates before it
# records the split: its requirements as the model wrote them, and the first
# year on record of each dataset pinned to `_OPEN_SPEC`, which `_SHOWN_OPEN`
# states as 1990.


def _model_plan(
    requirements: Sequence[EvidenceRequirement],
    decisions: Sequence[Decision],
    *,
    asks: Sequence[str] = (ASK,),
    answered: Sequence[AskAnswer] | None = None,
    across_years: Sequence[int] = (),
    first_year: int | None = 1990,
) -> Plan:
    """`requirements` as a plan answering `asks`, the last requirement answering ask 0 unless `answered` says otherwise.

    Each dataset requirement stating `period_end_open` over `_OPEN_SPEC`
    has `first_year` on record, none where it is `None`.
    """
    first_years = (
        []
        if first_year is None
        else [
            FirstYear(requirement_id=requirement.id, year=first_year)
            for requirement in requirements
            if isinstance(requirement, DatasetRequirement)
            and requirement.period_end_open
            and requirement.source_spec_id == _OPEN_SPEC
        ]
    )
    return Plan(
        question=ASK,
        answer_mode=AnswerMode.COMPUTED,
        decisions=list(decisions),
        evidence_requirements=list(requirements),
        asks=list(asks),
        answered=(
            [AskAnswer(ask=0, outputs=(requirements[-1].id,))] if answered is None else answered
        ),
        asks_across_years=list(across_years),
        first_years=first_years,
    )


def _recorded(plan: Plan) -> Plan:
    """`plan` carrying its split, as `propose_plan` records it."""
    split = split_of(plan)
    return Plan.model_validate(
        {
            **plan.model_dump(),
            "evidence_requirements": split.requirements,
            "rolling_copies": split.copies,
        }
    )


def _reads(plan: Plan) -> dict[str, tuple[str, ...]]:
    """What each requirement of `plan` reads once it is split."""
    return {requirement.id: inputs_of(requirement) for requirement in split_of(plan).requirements}


_TOP = _ranked(
    id="top",
    window_start=2024,
    window_end=2024,
    rolling=RollingWindow(window_start=_NEWEST, window_end=_NEWEST),
)
_SINCE_2015 = Period(start=2015, end=2024)
_SERIES_AND_TOP = [AskAnswer(ask=0, outputs=("aid",)), AskAnswer(ask=1, outputs=("top",))]


@pytest.mark.parametrize(
    ("requirements", "answered", "copies", "reads", "starts"),
    [
        pytest.param(
            [_open(period=_SINCE_2015), _TOP],
            _SERIES_AND_TOP,
            [("aid_history", "aid")],
            {"aid": (), "aid_history": (), "top": ("aid_history",)},
            {"aid": 2015, "aid_history": 1990},
            id="a-series-beside-a-rank-rolling-from-its-start",
        ),
        pytest.param(
            [
                _open(period=_SINCE_2015),
                _YEARLY,
                _TOP.model_copy(update={"input_id": "yearly", "axis": "recipient"}),
            ],
            [AskAnswer(ask=0, outputs=("yearly",)), AskAnswer(ask=1, outputs=("top",))],
            [("aid_history", "aid"), ("yearly_history", "yearly")],
            {
                "aid": (),
                "aid_history": (),
                "yearly": ("aid",),
                "yearly_history": ("aid_history",),
                "top": ("yearly_history",),
            },
            {"aid": 2015, "aid_history": 1990, "yearly": 2015, "yearly_history": 1990},
            id="a-yearly-total-answering-and-read-by-the-rank",
        ),
        pytest.param(
            [_open(period=_SINCE_2015), _TOP],
            None,
            [],
            {"aid": (), "top": ("aid",)},
            {"aid": 1990},
            id="the-rank-alone",
        ),
        pytest.param(
            [_open(period=EVERY_YEAR), _TOP],
            _SERIES_AND_TOP,
            [],
            {"aid": (), "top": ("aid",)},
            {"aid": 1990},
            id="a-first-year-on-record-at-the-stated-start",
        ),
    ],
)
def test_only_the_readers_rolling_from_their_start_read_a_dataset_from_its_first_year(
    decision: Callable[..., Decision],
    requirements: list[EvidenceRequirement],
    answered: list[AskAnswer] | None,
    copies: list[tuple[str, str]],
    reads: dict[str, tuple[str, ...]],
    starts: dict[str, int],
) -> None:
    """`top` ranks the newest year, its start following the newest data, so it
    reads 'aid' from 1990, the first year on record, when 'aid' states a later
    start. A series answer or a total per year beside it keeps 2015-2024
    through the dataset the model wrote, and `top` reads copies. With nothing
    else reading 'aid', or with 1990 the stated start, nothing is copied."""
    plan = _model_plan(
        requirements,
        [decision(), POPULATION],
        asks=_BESIDE_A_SERIES if answered is not None else (ASK,),
        answered=answered,
    )

    split = split_of(plan)
    shapes = frame_shapes(plan)

    assert [(copy.copy_id, copy.origin_id) for copy in split.copies] == copies
    assert _reads(plan) == reads
    assert {step_id: shapes[step_id].period.start for step_id in starts} == starts


def test_a_recorded_split_is_its_own_split_and_reverses_to_the_model_plan(
    decision: Callable[..., Decision],
) -> None:
    """Over two datasets, each copied for a growth rolling from its start
    above their quotient, beside the quotient answering ask 0."""
    aid, imputed = _open_series("aid"), _open_series("imputed", measure="imputed multilateral aid")
    pair = QuotientRequirement(id="pair", numerator=_side("aid"), denominator=_side("imputed"))
    growth = _growth(
        input_id="pair",
        rolling=RollingWindow(window_start=RelativePeriod(offset=-4), window_end=_NEWEST),
    )
    plan = _model_plan(
        [aid, imputed, pair, growth],
        [decision(), POPULATION, POOLED],
        asks=_BESIDE_A_SERIES,
        answered=[AskAnswer(ask=0, outputs=("pair",)), AskAnswer(ask=1, outputs=("change",))],
    )

    recorded = _recorded(plan)

    assert [copy.copy_id for copy in recorded.rolling_copies] == [
        "aid_history",
        "pair_history",
        "imputed_history",
    ]
    assert _reads(recorded)["pair_history"] == ("aid_history", "imputed_history")
    assert split_of(recorded).requirements == recorded.evidence_requirements
    assert authored(recorded).evidence_requirements == plan.evidence_requirements


def test_a_rolling_copy_reads_the_source_admission_holds_for_the_requirement_it_copies(
    decision: Callable[..., Decision],
) -> None:
    """Admission holds the source of 'aid' by its id, as it holds a compile alignment
    made for it, which the planner was never shown. `top` reads 'aid_history', a copy
    of 'aid' over every year with an id of its own: it reads the source held for
    'aid', so both pin the dataset over their own open windows."""
    recorded = _recorded(
        _model_plan(
            [_open(period=_SINCE_2015), _TOP],
            [decision(), POPULATION],
            asks=_BESIDE_A_SERIES,
            answered=_SERIES_AND_TOP,
        )
    )

    windowed, minted = _window_pins(recorded, requirement_sources={"aid": _SHOWN_OPEN[_OPEN_SPEC]})

    pins = {
        requirement.id: requirement.source_spec_id
        for requirement in windowed.evidence_requirements
        if isinstance(requirement, DatasetRequirement)
    }
    windows = {copy.spec_id: copy.identity["requested_window"] for copy in minted}
    assert {requirement_id: windows.get(pin) for requirement_id, pin in pins.items()} == {
        "aid": {"kind": "range", "start_year": 2015},
        "aid_history": {"kind": "range", "start_year": 1990},
    }


def test_a_recorded_plan_whose_copies_are_not_its_split_is_refused(
    decision: Callable[..., Decision],
) -> None:
    recorded = _recorded(
        _model_plan(
            [_open(period=_SINCE_2015), _TOP],
            [decision(), POPULATION],
            asks=_BESIDE_A_SERIES,
            answered=_SERIES_AND_TOP,
        )
    )
    content = recorded.model_dump()
    content["evidence_requirements"][-1]["input_id"] = "aid"

    with pytest.raises(ValidationError, match="rolling copies are not the split"):
        Plan.model_validate(content)


@pytest.mark.parametrize(
    ("pin", "first_year", "spec_id"),
    [
        pytest.param(_OPEN_SPEC, None, _OPEN_SPEC, id="a-pin-with-no-first-year-on-record"),
        pytest.param(None, None, None, id="no-pin"),
    ],
)
def test_a_start_rolling_over_a_dataset_with_no_first_year_on_record_is_refused(
    decision: Callable[..., Decision], pin: str | None, first_year: int | None, spec_id: str | None
) -> None:
    """The rank reads the dataset from its first year, and none is on record,
    so the plan is refused before its years are read. With 1990 on record
    it is admitted."""
    dataset = _open(period=_SINCE_2015).model_copy(update={"source_spec_id": pin})

    with pytest.raises(ValidationError) as refused:
        _model_plan([dataset, _TOP], [decision(), POPULATION], first_year=first_year)

    error = refused.value.errors()[0]["ctx"]["error"]
    assert isinstance(error, RollingPlanError)
    assert (error.issue, error.fault, error.spec_id, error.requirement_ids) == (
        RollingIssue.NEEDS_OPEN_PERIOD,
        OpenPeriodFault.FIRST_YEAR_UNKNOWN,
        spec_id,
        ("aid",),
    )
    assert _model_plan([_open(period=_SINCE_2015), _TOP], [decision(), POPULATION])


def test_a_start_rolling_above_a_max_by_donor_reads_the_stated_years(
    decision: Callable[..., Decision],
) -> None:
    """The filter's first year follows the newest data, but it bounds the year
    each donor's max fell in, so 'aid' keeps 2015-2024 and the max picks over
    those years. The filter is refused for its own reason, that newer data
    moves its first year past the year a max fell in."""
    peak = _total(id="peak", reduction=Reduction.MAX)
    since = FilterRequirement(
        id="since",
        input_id="peak",
        predicates=(PeriodThreshold(op="gte", year=2010, rolling=RelativePeriod(offset=-14)),),
    )
    requirements = [_open(period=_SINCE_2015), peak, since]

    split = whole_history_split(
        requirements,
        answering={"since"},
        first_years=[FirstYear(requirement_id="aid", year=1990)],
        copies=(),
    )
    with pytest.raises(ValidationError, match="the first year the filter keeps, 2015-2024 today"):
        _model_plan(requirements, [decision()], across_years=(0,))

    assert (split.whole_history, split.copies) == ({}, ())


def test_a_rolling_window_over_a_total_by_donor_reads_the_stated_years(
    decision: Callable[..., Decision],
) -> None:
    """The rank's window starts 14 years back of the newest over each donor's
    total, a span of the stated 2015-2024, whose first year newer data moves
    out of the window, so it is refused by the span rules over those years."""
    summed = _total(id="summed")
    ranked = _ranked(
        input_id="summed",
        axis="recipient",
        reduction=None,
        window_start=2010,
        window_end=2024,
        rolling=RollingWindow(window_start=RelativePeriod(offset=-14), window_end=_NEWEST),
    )

    with pytest.raises(ValidationError, match="one span over 2015-2024"):
        _model_plan(
            [_open(period=_SINCE_2015), summed, ranked], [decision(), POPULATION], across_years=(0,)
        )


@pytest.mark.parametrize(
    "sibling",
    [
        pytest.param(
            [_keeping(PeriodThreshold(op="lte", year=2015)), _total(input_id="kept")],
            id="a-total-up-to-2015",
        ),
        pytest.param(
            [_keeping(PeriodThreshold(op="lt", year=2016)), _total(input_id="kept")],
            id="a-total-before-2016",
        ),
        pytest.param(
            [_keeping(PeriodThreshold(op="eq", year=2015)), _total(input_id="kept")],
            id="a-total-of-2015",
        ),
        pytest.param([_ranked(window_start=2010, window_end=2015)], id="a-rank-of-2010-2015"),
    ],
)
def test_a_sibling_of_a_rank_rolling_from_its_start_reads_the_stated_years(
    decision: Callable[..., Decision], sibling: list[EvidenceRequirement]
) -> None:
    """'aid' states 2020-2024 and `top` reads it from 1990, so a sibling
    reading years before 2020 reads none. 2015 is no year of what it reads."""
    with pytest.raises(ValidationError, match="2020-2024"):
        _model_plan(
            [_open(period=Period(start=2020, end=2024)), _TOP, *sibling],
            [decision(), POPULATION],
            asks=_BESIDE_A_SERIES,
            answered=[AskAnswer(ask=0, outputs=("top",)), AskAnswer(ask=1, outputs=("result",))],
            across_years=(1,),
        )


def test_a_window_rolling_from_its_start_over_years_up_to_2015_runs_out_of_years(
    decision: Callable[..., Decision],
) -> None:
    """The rank reads 'aid' from 1990 through a filter keeping the years up to
    2015, so the filter holds 1990-2015, and the rank's window, nine back of
    the newest at both ends, moves past 2015 once newer data arrives."""
    kept = _keeping(PeriodThreshold(op="lte", year=2015))
    nine_back = RelativePeriod(offset=-9)
    ranked = _ranked(
        input_id="kept",
        window_start=2015,
        window_end=2015,
        rolling=RollingWindow(window_start=nine_back, window_end=nine_back),
    )

    with pytest.raises(ValidationError) as refused:
        _model_plan(
            [_open(period=Period(start=2020, end=2024)), kept, ranked], [decision(), POPULATION]
        )

    error = refused.value.errors()[0]["ctx"]["error"]
    assert isinstance(error, RollingPlanError)
    assert (error.issue, error.requirement_ids) == (RollingIssue.YEARS_RUN_OUT, ("result",))
    assert "leaves no year" not in str(error)


def _computed_from(plan: Plan, step_id: str) -> set[str]:
    """The requirements of `plan`, as it runs, whose result rests on `step_id`, itself included."""
    reads = _reads(plan)
    found = {step_id}
    while grown := {node for node, inputs in reads.items() if found & set(inputs)} - found:
        found |= grown
    return found


@pytest.mark.usefixtures("aid_on_record")
@pytest.mark.parametrize(
    ("requirements", "decisions", "change", "offset", "sibling", "year"), _ONE_YEAR_OFFERS
)
def test_the_one_year_each_gap_offers_is_admitted_and_leaves_every_other_result_its_years(
    decision: Callable[..., Decision],
    requirements: list[EvidenceRequirement],
    decisions: list[Decision],
    change: OneYear,
    offset: int | None,
    sibling: str | None,
    year: int | None,
) -> None:
    """The year offered (`unstated_period_reductions`) is the one
    `test_the_one_year_each_refusal_offers_is_admitted` reads off the
    refusal. Followed on the plan the model wrote, the plan validates, and
    every result answering it that is not computed from the refused step
    keeps the years and the drift it read before."""
    asks, answered = (ASK,), None
    if sibling is not None:
        asks = _BESIDE_A_SERIES
        answered = [
            AskAnswer(ask=0, outputs=(requirements[-1].id,)),
            AskAnswer(ask=1, outputs=(sibling,)),
        ]
    plan = _model_plan(requirements, [decision(), *decisions], asks=asks, answered=answered)

    (gap,) = unstated_period_reductions(plan)
    read = gap.held.one_year
    assert read is not None
    followed = Plan.model_validate(
        {
            **plan.model_dump(),
            "evidence_requirements": one_year_followed(requirements, gap.step_id, read),
        }
    )

    assert (read.year, read.offset) == (gap.held.years.end if year is None else year, offset)
    before, after = frame_shapes(plan), frame_shapes(followed)
    kept = [
        output
        for output in answering_results(followed)
        if output not in _computed_from(followed, gap.step_id)
    ]
    assert [(before[o].period, before[o].drift) for o in kept] == [
        (after[o].period, after[o].drift) for o in kept
    ]


def test_a_rolling_one_year_reads_a_copy_and_leaves_a_yearly_series_its_years(
    decision: Callable[..., Decision],
) -> None:
    """'aid' states 2020-2024. The total answering ask 0 is offered the
    newest year as it moves. Followed, its filter reads a copy of 'aid'
    from 1990, and the total per year answering ask 1 keeps 2020 on."""
    plan = _model_plan(
        [_open(period=Period(start=2020, end=2024)), _YEARLY, _total()],
        [decision()],
        asks=_BESIDE_A_SERIES,
        answered=[AskAnswer(ask=0, outputs=("result",)), AskAnswer(ask=1, outputs=("yearly",))],
    )

    (gap,) = unstated_period_reductions(plan)
    assert gap.held.one_year == OneYearRead(year=2024, offset=0)
    followed = Plan.model_validate(
        {
            **plan.model_dump(),
            "evidence_requirements": one_year_followed(
                list(plan.evidence_requirements), gap.step_id, gap.held.one_year
            ),
        }
    )

    shapes = frame_shapes(followed)
    assert _reads(followed)["one_year"] == ("aid_history",)
    assert (shapes["yearly"].period.start, shapes["aid_history"].period.start) == (2020, 1990)


def test_a_total_over_a_dataset_first_held_in_its_stated_start_offers_no_earlier_year(
    decision: Callable[..., Decision],
) -> None:
    """The dataset's first year on record is 2020, its stated start, so the
    total's years are 2020-2024 and the year offered is the newest."""
    plan = _model_plan(
        [_open(period=Period(start=2020, end=2024)), _total()], [decision()], first_year=2020
    )

    (gap,) = unstated_period_reductions(plan)

    assert gap.held.years == Period(start=2020, end=2024)
    assert gap.held.one_year == OneYearRead(year=2024, offset=0)


def test_the_requirement_limit_counts_the_requirements_the_model_wrote(
    decision: Callable[..., Decision],
) -> None:
    """Twenty-four requirements, which the split takes to twenty-five, are
    admitted, and a twenty-fifth the model writes is refused."""
    donors = [
        FilterRequirement(
            id=f"donor_{n}",
            input_id="aid",
            predicates=(ValueThreshold(op="gt", value=float(n)),),
        )
        for n in range(MAX_PLAN_REQUIREMENTS - 2)
    ]
    requirements = [_open(period=_SINCE_2015), *donors, _TOP]

    plan = _model_plan(requirements, [decision(), POPULATION])

    assert len(plan.evidence_requirements) == MAX_PLAN_REQUIREMENTS
    assert len(_recorded(plan).evidence_requirements) == MAX_PLAN_REQUIREMENTS + 1
    extra = donors[0].model_copy(update={"id": "donor_extra"})
    with pytest.raises(ValidationError, match=f"over the {MAX_PLAN_REQUIREMENTS} a plan may take"):
        _model_plan([*requirements[:-1], extra, _TOP], [decision(), POPULATION])


def test_the_size_limit_measures_the_plan_the_model_wrote(
    decision: Callable[..., Decision],
) -> None:
    """A dataset naming many recipients takes over half the size limit, so its
    copy takes the recorded plan past it, and the plan is admitted on the
    size the model wrote."""
    recipients = tuple(
        Entity(name=f"Recipient {n:03d} " + "x" * 130, role="recipient")
        for n in range(MAX_ENTITIES)
    )
    aid = _open(period=_SINCE_2015).model_copy(update={"entities": recipients})

    plan = _model_plan(
        [aid, _TOP], [decision(), POPULATION], asks=_BESIDE_A_SERIES, answered=_SERIES_AND_TOP
    )

    written, recorded = (
        len(
            record.model_dump_json(
                include={"question", "answer_mode", "decisions", "evidence_requirements"}
            )
        )
        for record in (plan, _recorded(plan))
    )
    assert written <= MAX_PLAN_RECORD_CHARS < recorded
