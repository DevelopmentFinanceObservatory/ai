"""A replan for the option the user picked on the plan card holds that pick.

The planner seam records the pick for the entry (`planner.plan.
DECISION_PICK_STATE_KEY`), with the scope the card gave the replan
(`DecisionPick.on_card`, as `harness.decision_pick.decision_pick` reads it off the card). `propose_plan` admits a plan only when it keeps
every other requirement exactly as the card had it, ids and citations
included, keeps every decision of the user's and every one a kept
requirement cites unchanged, cited and uncontradicted, replaces the card's
requirements citing the picked decision with requirements that read
something else, and states the picked option fixing something
(`submission.service._picked_decision`). That decision is recorded as the
user's. Anything else is refused `USER_PICK_NOT_HELD`, whatever the session
records of the user's words, since a pick is none.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from google.adk.events import EventActions
from harness.decision_pick import decision_pick
from harness.plan_display import decision_option_id
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
# The DAC total's decision under the picked label, listing the period beside
# the entities, so its fixes differ from the card's though it reads the same.
RELABELLED_WIDER = DONOR_SCOPE.model_copy(
    update={
        "chosen": "the top 10 donors",
        "fixes": DONOR_SCOPE.fixes.model_copy(update={"period": YEAR}),
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


def _pick_top_ten(ctx: Any) -> None:
    """Record, for the entry, the pick of the top 10 donors on the card the plan in `ctx` shows."""
    card = load_plan(ctx.state)
    assert card is not None
    pick = decision_pick(
        card,
        choice_id="decision/donor_scope",
        option_id=decision_option_id("the top 10 donors"),
        to_origin={},
    )
    assert pick is not None
    save_decision_pick(ctx.state, pick)


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
    _pick_top_ten(ctx)

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
        pytest.param(
            {"decisions": [RELABELLED_WIDER]},
            "the top 10 donors",
            id="relabelled-listing-more-fields",
        ),
        pytest.param(
            {
                "decisions": [DONOR_SCOPE.model_copy(update={"chosen": "the top 10 donors"})],
                "renamed": True,
            },
            "the top 10 donors",
            id="relabelled-over-the-same-requirement-renamed",
        ),
    ],
)
def test_a_plan_that_does_not_state_the_pick_is_refused(
    decision: Callable[..., Decision], patch: dict[str, Any], stated: str | None
) -> None:
    """A plan carrying or restating the DAC total, or dropping the decision,
    is refused. So is the DAC total under the picked label, however many
    fields the decision lists and whatever the requirement is called: what
    the requirements citing it read is unchanged."""
    ctx = _context()
    first = _recorded_dac_total(ctx, decision)
    _pick_top_ten(ctx)
    requirements = (
        [_dac_total().model_copy(update={"decision_ids": ("measure", "year")})]
        if "removed_decisions" in patch
        else [_dac_total().model_copy(update={"id": "dac_total"})]
        if patch.get("renamed")
        else [_dac_total()]
    )

    result = _propose_plan(
        ctx,
        decisions=drafts(patch.get("decisions", [])),
        removed_decisions=patch.get("removed_decisions", []),
        evidence_requirements=requirements,
        removed_evidence_requirements=["dac"] if patch.get("renamed") else [],
        amends_version=first["version"],
    )

    assert result["outcome"] == Outcome.REFUSED.value, result
    assert result["code"] == ProblemCode.USER_PICK_NOT_HELD.value
    assert result["detail"] == {
        "slot": "donor_scope",
        "chosen": "the top 10 donors",
        "stated": stated,
        "kept": None,
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
    _pick_top_ten(ctx)

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
    "patch",
    [
        pytest.param(
            {
                "evidence_requirements": _year_kept(Period(start=2022, end=2022)),
                "removed_evidence_requirements": ["dac"],
            },
            id="still-cited-and-changed",
        ),
        pytest.param(
            {
                "evidence_requirements": [
                    _dac_total().model_copy(update={"decision_ids": ("measure",)}),
                    *_ranked_donors(),
                ]
            },
            id="uncited-on-a-surviving-requirement",
        ),
    ],
)
def test_a_pick_releases_no_user_decision_the_new_plan_still_rests_on(
    decision: Callable[..., Decision], patch: dict[str, Any]
) -> None:
    """The pick speaks for its own slot. Another user decision the new plan
    still cites, or whose requirement it keeps, stands as the user made it."""
    ctx = _context()
    first = _recorded_dac_total(ctx, decision)
    _pick_top_ten(ctx)

    result = _propose_plan(
        ctx,
        decisions=drafts([TOP_TEN]),
        amends_version=first["version"],
        **patch,
    )

    assert result["outcome"] == Outcome.REFUSED.value, result
    assert result["code"] == ProblemCode.USER_PICK_NOT_HELD.value
    assert result["detail"]["kept"] == "year"


def test_a_pick_releases_no_user_decision_of_a_requirement_it_did_not_change(
    decision: Callable[..., Decision],
) -> None:
    """The user's year rests on a requirement that does not cite the picked
    decision. A replan that renames that requirement and drops the year from
    it is refused for leaving that requirement off."""
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
    _pick_top_ten(ctx)

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
    assert result["code"] == ProblemCode.USER_PICK_NOT_HELD.value
    assert result["detail"]["kept"] == "gross"


@pytest.mark.parametrize(
    "picked",
    [
        pytest.param(TOP_TEN, id="fixing-the-rank"),
        pytest.param(
            TOP_TEN.model_copy(
                update={
                    "fixes": DecisionFixes(
                        limit=10, among=RankAmong.POPULATION, period=Period(start=2022, end=2022)
                    )
                }
            ),
            id="listing-the-period",
        ),
    ],
)
def test_a_pick_releases_no_other_user_decision(
    decision: Callable[..., Decision], picked: Decision
) -> None:
    """The user's year fixes a period. A replan that drops the DAC total, and
    with it the only citation of the year, while reading the donors over
    2022 under the picked decision is refused, whatever fields that decision
    lists: the pick speaks for its own slot alone."""
    ctx = _context()
    first = _recorded_dac_total(ctx, decision)
    _pick_top_ten(ctx)
    aid, top = _ranked_donors()

    result = _propose_plan(
        ctx,
        decisions=drafts([picked]),
        evidence_requirements=[
            aid.model_copy(
                update={
                    "period": Period(start=2022, end=2022),
                    "decision_ids": ("measure", "donor_scope"),
                }
            ),
            top,
        ],
        removed_evidence_requirements=["dac"],
        amends_version=first["version"],
    )

    assert result["outcome"] == Outcome.REFUSED.value, result
    assert result["code"] == ProblemCode.USER_PICK_NOT_HELD.value
    assert result["detail"]["kept"] == "year"


def test_a_pick_listing_entities_releases_no_user_decision_fixing_the_recipient(
    decision: Callable[..., Decision],
) -> None:
    """The user's own decision fixes who the DAC total is about. A replan
    for the pick that reads the donors to Ghana under the picked decision,
    listing its entities, is refused: the pick does not speak for the
    recipient."""
    ctx = _context()
    recipient = USER_YEAR.model_copy(
        update={
            "slot": "recipient",
            "description": "Which country the aid goes to.",
            "options": ["Kenya", "Ghana"],
            "chosen": "Kenya",
            "fixes": DONOR_SCOPE.fixes,
        }
    )
    first = _propose_plan(
        ctx,
        question="how much aid does Kenya get",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision(), DONOR_SCOPE, recipient]),
        evidence_requirements=[
            _dac_total().model_copy(
                update={"decision_ids": ("measure", "donor_scope", "recipient")}
            )
        ],
    )
    assert first["outcome"] == Outcome.OK.value, first
    ctx.session.events = [response_event("propose_plan", first)]
    _pick_top_ten(ctx)
    aid, top = _ranked_donors()
    ghana = (Entity(name="Ghana", role="recipient"),)

    result = _propose_plan(
        ctx,
        decisions=drafts(
            [
                TOP_TEN.model_copy(
                    update={
                        "fixes": DecisionFixes(limit=10, among=RankAmong.POPULATION, entities=ghana)
                    }
                )
            ]
        ),
        evidence_requirements=[
            aid.model_copy(update={"entities": ghana, "decision_ids": ("measure", "donor_scope")}),
            top,
        ],
        removed_evidence_requirements=["dac"],
        amends_version=first["version"],
    )

    assert result["outcome"] == Outcome.REFUSED.value, result
    assert result["code"] == ProblemCode.USER_PICK_NOT_HELD.value
    assert result["detail"]["kept"] == "recipient"


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
        pytest.param(RELABELLED_WIDER, id="listing-more-than-the-dac-total-fixed"),
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
    _pick_top_ten(ctx)
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


@pytest.mark.parametrize(
    "recorded",
    [
        pytest.param(lambda first: [], id="card-version-unrecorded"),
        pytest.param(
            lambda first: [
                response_event("propose_plan", first),
                response_event("propose_plan", {**first, "version": first["version"] + 1}),
            ],
            id="later-version-recorded-after-the-card",
        ),
    ],
)
def test_a_pick_holds_the_users_decisions_whatever_the_session_records(
    decision: Callable[..., Decision], recorded: Callable[[dict[str, Any]], list[Any]]
) -> None:
    """The card shows a plan the session holds no `propose_plan` response
    for, as after a scope switch revised it in code, or one a later version's
    response follows, as after a failed pick replan put it back. A replan
    reading the donors over 2022 and dropping the user's year is refused all
    the same: the pick is no word of the user's."""
    ctx = _context()
    first = _recorded_dac_total(ctx, decision)
    ctx.session.events = recorded(first)
    _pick_top_ten(ctx)
    aid, top = _ranked_donors()

    result = _propose_plan(
        ctx,
        decisions=drafts([TOP_TEN]),
        evidence_requirements=[
            aid.model_copy(
                update={
                    "period": Period(start=2022, end=2022),
                    "decision_ids": ("measure", "donor_scope"),
                }
            ),
            top,
        ],
        removed_evidence_requirements=["dac"],
        amends_version=first["version"],
    )

    assert result["code"] == ProblemCode.USER_PICK_NOT_HELD.value, result
    assert result["detail"]["kept"] == "year"
    recorded_plan = load_plan(ctx.state)
    assert recorded_plan is not None and recorded_plan.version == first["version"]


def _prior_total() -> DatasetRequirement:
    """The DAC total for 2021, which does not cite the picked decision."""
    return _dac_total().model_copy(
        update={
            "id": "dac_prior",
            "decision_ids": ("measure",),
            "period": Period(start=2021, end=2021),
        }
    )


@pytest.mark.parametrize(
    ("patch", "kept"),
    [
        pytest.param(
            {
                "decisions": [DONOR_SCOPE.model_copy(update={"chosen": "the top 10 donors"})],
                "evidence_requirements": [
                    _dac_total().model_copy(update={"decision_ids": ("measure", "year")}),
                    _prior_total().model_copy(update={"decision_ids": ("measure", "donor_scope")}),
                ],
            },
            "dac_prior",
            id="citation-moved-onto-a-kept-requirement",
        ),
        pytest.param(
            {
                "decisions": [TOP_TEN],
                "evidence_requirements": [
                    *_year_kept(YEAR),
                    _prior_total().model_copy(update={"period": Period(start=2020, end=2020)}),
                ],
                "removed_evidence_requirements": ["dac"],
            },
            "dac_prior",
            id="kept-requirement-edited",
        ),
        pytest.param(
            {
                "decisions": [TOP_TEN],
                "evidence_requirements": _year_kept(YEAR),
                "removed_evidence_requirements": ["dac", "dac_prior"],
            },
            "dac_prior",
            id="kept-requirement-removed",
        ),
    ],
)
def test_a_pick_replan_may_change_only_the_requirements_citing_the_picked_decision(
    decision: Callable[..., Decision], patch: dict[str, Any], kept: str
) -> None:
    """The card states the DAC total for 2023, citing the picked decision,
    and for 2021, not citing it. A replan that moves the picked decision's
    citation onto the 2021 total, or edits or removes that total, is refused
    naming it, and the refusal says what the replan may change."""
    ctx = _context()
    first = _propose_plan(
        ctx,
        question="how much aid does Kenya get",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision(), DONOR_SCOPE, USER_YEAR]),
        evidence_requirements=[_dac_total(), _prior_total()],
    )
    assert first["outcome"] == Outcome.OK.value, first
    _pick_top_ten(ctx)

    result = _propose_plan(
        ctx,
        decisions=drafts(patch["decisions"]),
        evidence_requirements=patch["evidence_requirements"],
        removed_evidence_requirements=patch.get("removed_evidence_requirements", []),
        amends_version=first["version"],
    )

    assert result["code"] == ProblemCode.USER_PICK_NOT_HELD.value, result
    assert result["detail"]["kept"] == kept
    assert "may change two things only" in result["message"]
    recorded = load_plan(ctx.state)
    assert recorded is not None and recorded.version == first["version"]


def test_a_pick_replan_keeping_the_rest_of_the_card_is_admitted(
    decision: Callable[..., Decision],
) -> None:
    """The DAC total for 2023 becomes the top 10 donors read over the user's
    year, and the 2021 total the card also states is carried as it was."""
    ctx = _context()
    first = _propose_plan(
        ctx,
        question="how much aid does Kenya get",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision(), DONOR_SCOPE, USER_YEAR]),
        evidence_requirements=[_dac_total(), _prior_total()],
    )
    assert first["outcome"] == Outcome.OK.value, first
    _pick_top_ten(ctx)

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
    assert sorted(r.id for r in plan.evidence_requirements) == ["aid", "dac_prior", "top"]
