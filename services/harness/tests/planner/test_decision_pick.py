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
from typing import Any, cast

import pytest
from analysis_mcp.server import mcp as analysis_mcp
from compiler.compile import compile_plan
from compiler.sources import source_from_spec_block
from compiler.surface import McpSurface
from data_mcp.server import mcp as data_mcp
from google.adk.events import EventActions
from harness.decision_pick import decision_pick
from harness.plan_display import decision_option_id
from harness.studio_state import _turn_identities
from planner.agent import build_planner_tools
from planner.plan import (
    ASKS_STATE_KEY,
    AggregateRequirement,
    AnswerMode,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    Period,
    RankRequirement,
    ResolvedBy,
    Unit,
    current_recorded_identity,
    load_plan,
    save_decision_pick,
    windowed_pins,
)
from planner.records import GroupBy, RankAmong, RankDirection, Reduction
from planner.tool_results import Outcome, ProblemCode
from translator.mechanical import build_translation

from ..decision_drafts import drafts
from ..session_events import response_event
from .test_agent import (
    _FRANCE_KENYA,
    _GROSS,
    _US_MULTILATERAL,
    _US_TOTAL,
    _dac_countries,
    _FakeToolContext,
    _share,
    _show_block,
    _show_blocks,
    _tools_by_name,
)

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
    ("patch", "stated", "kept"),
    [
        pytest.param({"decisions": [DONOR_SCOPE]}, "the DAC total", "dac", id="restated"),
        pytest.param({"decisions": []}, "the DAC total", "dac", id="carried"),
        pytest.param({"removed_decisions": ["donor_scope"]}, None, "dac", id="removed"),
        pytest.param(
            {"decisions": [DONOR_SCOPE.model_copy(update={"chosen": "the top 10 donors"})]},
            "the top 10 donors",
            "dac",
            id="relabelled-over-the-same-requirements",
        ),
        pytest.param(
            {"decisions": [RELABELLED_WIDER]},
            "the top 10 donors",
            "dac",
            id="relabelled-listing-more-fields",
        ),
        pytest.param(
            {
                "decisions": [DONOR_SCOPE.model_copy(update={"chosen": "the top 10 donors"})],
                "renamed": True,
            },
            "the top 10 donors",
            "dac_total",
            id="relabelled-over-the-same-requirement-renamed",
        ),
        pytest.param(
            {
                "decisions": [DONOR_SCOPE.model_copy(update={"chosen": "the top 10 donors"})],
                "extra": True,
            },
            "the top 10 donors",
            "dac",
            id="superset-of-the-card",
        ),
    ],
)
def test_a_plan_that_does_not_state_the_pick_is_refused(
    decision: Callable[..., Decision], patch: dict[str, Any], stated: str | None, kept: str
) -> None:
    """A plan carrying or restating the DAC total, or dropping the decision,
    is refused. So is the DAC total under the picked label, however many
    fields the decision lists, whatever the requirement is called, and
    whatever it adds beside it: a requirement still reads what the DAC total
    read, and the refusal names it."""
    ctx = _context()
    first = _recorded_dac_total(ctx, decision)
    _pick_top_ten(ctx)
    requirements = (
        [_dac_total().model_copy(update={"decision_ids": ("measure", "year")})]
        if "removed_decisions" in patch
        else [_dac_total().model_copy(update={"id": "dac_total"})]
        if patch.get("renamed")
        else [
            _dac_total(),
            _dac_total().model_copy(
                update={
                    "id": "dac_2020",
                    "period": Period(start=2020, end=2020),
                    "decision_ids": ("measure", "donor_scope"),
                }
            ),
        ]
        if patch.get("extra")
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
        "kept": kept,
    }
    recorded = load_plan(ctx.state)
    assert recorded is not None and recorded.version == first["version"]


def test_a_restated_pick_its_requirements_fix_nothing_of_is_refused_saying_so(
    decision: Callable[..., Decision],
) -> None:
    """The plan states the picked option, cited only by the donors to Kenya,
    which state neither a rank's limit nor what it ranks among: the decision
    fixes nothing, and the refusal says the requirements citing it fix none
    of what it names."""
    ctx = _context()
    first = _recorded_dac_total(ctx, decision)
    _pick_top_ten(ctx)
    aid, _ = _year_kept(YEAR)

    result = _propose_plan(
        ctx,
        decisions=drafts([TOP_TEN]),
        evidence_requirements=[aid],
        removed_evidence_requirements=["dac"],
        amends_version=first["version"],
    )

    assert result["code"] == ProblemCode.USER_PICK_NOT_HELD.value, result
    assert "fix none of what it names" in result["message"]


def _year_kept(period: Period) -> list[Any]:
    """`_ranked_donors` with the donors read over `period`, each requirement
    citing the picked decision and the user's year, as the DAC total did."""
    aid, top = _ranked_donors()
    return [
        aid.model_copy(
            update={"decision_ids": ("measure", "donor_scope", "year"), "period": period}
        ),
        top.model_copy(update={"decision_ids": ("donor_scope", "year")}),
    ]


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


def test_a_replacing_requirement_cites_the_users_decisions_the_replaced_ones_cited(
    decision: Callable[..., Decision],
) -> None:
    """The user's year is cited by the DAC total and by the gross total the
    pick leaves alone. A replan whose donors read 2022 without citing the
    year is refused, naming the donors, though the year still stands on the
    gross total."""
    ctx = _context()
    gross = _dac_total().model_copy(
        update={"id": "gross", "decision_ids": ("measure", "year"), "measure": "gross ODA"}
    )
    first = _propose_plan(
        ctx,
        question="how much aid does Kenya get",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision(), DONOR_SCOPE, USER_YEAR]),
        evidence_requirements=[_dac_total(), gross],
    )
    assert first["outcome"] == Outcome.OK.value, first
    _pick_top_ten(ctx)
    aid, top = _year_kept(Period(start=2022, end=2022))

    result = _propose_plan(
        ctx,
        decisions=drafts([TOP_TEN]),
        evidence_requirements=[
            aid.model_copy(update={"decision_ids": ("measure", "donor_scope")}),
            top,
        ],
        removed_evidence_requirements=["dac"],
        amends_version=first["version"],
    )

    assert result["code"] == ProblemCode.USER_PICK_NOT_HELD.value, result
    assert result["detail"]["kept"] == "aid"
    assert "leaves decision 'year' uncited on 'aid'" in result["message"]


def test_a_landed_pick_can_be_picked_back(decision: Callable[..., Decision]) -> None:
    """The top 10 donors replace the DAC total, and the user then picks the
    DAC total back on the new card. Every requirement the first replan added
    cites the pick, so the second replaces them all and is admitted."""
    ctx = _context()
    first = _recorded_dac_total(ctx, decision)
    _pick_top_ten(ctx)
    ranked = _propose_plan(
        ctx,
        decisions=drafts([TOP_TEN]),
        evidence_requirements=_year_kept(YEAR),
        removed_evidence_requirements=["dac"],
        amends_version=first["version"],
    )
    assert ranked["outcome"] == Outcome.OK.value, ranked
    card = load_plan(ctx.state)
    assert card is not None
    back = decision_pick(
        card,
        choice_id="decision/donor_scope",
        option_id=decision_option_id("the DAC total"),
        to_origin={},
    )
    assert back is not None
    save_decision_pick(ctx.state, back)

    result = _propose_plan(
        ctx,
        decisions=drafts([DONOR_SCOPE]),
        evidence_requirements=[_dac_total()],
        removed_evidence_requirements=["aid", "top"],
        amends_version=ranked["version"],
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [r.id for r in plan.evidence_requirements] == ["dac"]


def test_a_contradicted_decision_is_named_not_the_kept_requirement_it_sits_on(
    decision: Callable[..., Decision],
) -> None:
    """The planner's prices decision is cited by the DAC total and by the
    2021 total the pick leaves alone. Donors read at constant prices under it
    are refused naming the prices decision, which the replan contradicted,
    and not the 2021 total, which it never touched."""
    ctx = _context()
    prices = Decision(
        slot="prices",
        description="Which prices the figures are in.",
        options=["current", "constant"],
        chosen="current",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="Current prices are the default.",
        fixes=DecisionFixes(unit=USD),
    )
    first = _propose_plan(
        ctx,
        question="how much aid does Kenya get",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision(), DONOR_SCOPE, USER_YEAR, prices]),
        evidence_requirements=[
            _dac_total().model_copy(
                update={"decision_ids": ("measure", "donor_scope", "year", "prices")}
            ),
            _prior_total().model_copy(update={"decision_ids": ("measure", "prices")}),
        ],
    )
    assert first["outcome"] == Outcome.OK.value, first
    _pick_top_ten(ctx)
    aid, top = _year_kept(YEAR)
    constant = Unit(currency="USD", prices="constant", base_year=2022)

    result = _propose_plan(
        ctx,
        decisions=drafts([TOP_TEN]),
        evidence_requirements=[
            aid.model_copy(
                update={"decision_ids": (*aid.decision_ids, "prices"), "unit": constant}
            ),
            top,
        ],
        removed_evidence_requirements=["dac"],
        amends_version=first["version"],
    )

    assert result["code"] == ProblemCode.USER_PICK_NOT_HELD.value, result
    assert result["detail"]["kept"] == "prices"


def test_a_pick_replacing_a_shares_input_keeps_the_share_whatever_admission_derives_for_it() -> (
    None
):
    """The card shares US multilateral ODA over the US total, and admission
    gave the share a caveat that its inputs count different providers. A pick
    replacing the numerator, under its own id, with the US total leaves the
    share as the planner wrote it, and is admitted though admission drops
    the caveat it derived."""
    measure = Decision(
        slot="measure",
        description="What the share measures.",
        options=["a", "b"],
        chosen="a",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="r",
    )
    first = _dac_countries("first", _US_MULTILATERAL)
    part = Decision(
        slot="part",
        description="Which figure leads the share.",
        options=[first.measure, "another figure"],
        chosen=first.measure,
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="r",
        fixes=DecisionFixes(measure=first.measure),
    )
    ctx = _FakeToolContext()
    _show_blocks(ctx, _US_MULTILATERAL, _US_TOTAL)
    propose = _tools_by_name()["propose_plan"].func
    card = propose(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([measure, part]),
        evidence_requirements=[
            first.model_copy(update={"decision_ids": ("measure", "part")}),
            _dac_countries("second", _US_TOTAL),
            _share("first", "second"),
        ],
        tool_context=ctx,
    )
    assert card["outcome"] == Outcome.OK.value, card
    shown = load_plan(ctx.state)
    assert shown is not None
    assert any(d.caveat is not None for d in shown.decisions)
    pick = decision_pick(
        shown,
        choice_id="decision/part",
        option_id=decision_option_id("another figure"),
        to_origin={},
    )
    assert pick is not None
    save_decision_pick(ctx.state, pick)

    result = propose(
        amends_version=card["version"],
        decisions=drafts(
            [
                part.model_copy(
                    update={
                        "chosen": "another figure",
                        "fixes": DecisionFixes(measure="another figure"),
                    }
                )
            ]
        ),
        evidence_requirements=[
            _dac_countries("first", _US_TOTAL).model_copy(
                update={"measure": "another figure", "decision_ids": ("measure", "part")}
            )
        ],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result


# A ranking decision whose options differ only in how many donors it keeps.
RANKING = Decision(
    slot="donor_scope",
    description="How many of the largest donors to rank.",
    options=["the top 10 donors", "the top 5 donors"],
    chosen="the top 10 donors",
    resolved_by=ResolvedBy.PLANNER_DEFAULT,
    rationale="Ten donors cover most of the aid.",
    fixes=DecisionFixes(limit=10, among=RankAmong.POPULATION),
)


def test_a_pick_keeping_part_of_what_it_replaces_unchanged_is_admitted(
    decision: Callable[..., Decision],
) -> None:
    """The top 10 donors become the top 5: the donors the ranking reads stay
    as they were and only the ranking's limit moves. Part of what the card's
    requirements citing the pick read survives, and the replan is admitted."""
    ctx = _context()
    first = _propose_plan(
        ctx,
        question="who gives Kenya the most aid",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision(), RANKING, USER_YEAR]),
        evidence_requirements=_year_kept(YEAR),
    )
    assert first["outcome"] == Outcome.OK.value, first
    card = load_plan(ctx.state)
    assert card is not None
    pick = decision_pick(
        card,
        choice_id="decision/donor_scope",
        option_id=decision_option_id("the top 5 donors"),
        to_origin={},
    )
    assert pick is not None
    save_decision_pick(ctx.state, pick)
    aid, top = _year_kept(YEAR)

    result = _propose_plan(
        ctx,
        decisions=drafts(
            [
                RANKING.model_copy(
                    update={
                        "chosen": "the top 5 donors",
                        "fixes": DecisionFixes(limit=5, among=RankAmong.POPULATION),
                    }
                )
            ]
        ),
        evidence_requirements=[aid, top.model_copy(update={"limit": 5})],
        amends_version=first["version"],
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    picked = next(d for d in plan.decisions if d.slot == "donor_scope")
    assert (picked.chosen, picked.fixes) == (
        "the top 5 donors",
        DecisionFixes(limit=5, among=RankAmong.POPULATION),
    )
    assert next(r for r in plan.evidence_requirements if r.id == "aid") == aid


async def test_a_pick_over_a_windowed_pin_cites_the_users_year_and_still_compiles(
    decision: Callable[..., Decision],
) -> None:
    """France's aid to Kenya, pinned to Search's answer and recorded over the
    plan's years as a windowed copy, is summed by year under a planner
    decision. Picking the average replaces the sum with one citing the
    user's year as the sum did, though it states no period: the year and the pin settle as
    they were, and the plan still translates and compiles."""
    shape = Decision(
        slot="shape",
        description="How to combine each year's flows.",
        options=["the yearly sum", "the yearly average"],
        chosen="the yearly sum",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="A sum is the usual reading.",
        fixes=DecisionFixes(reduction=Reduction.SUM),
    )
    year = USER_YEAR.model_copy(
        update={
            "options": ["2015-2024", "2020-2024"],
            "chosen": "2015-2024",
            "fixes": DecisionFixes(period=Period(start=2015, end=2024)),
        }
    )
    aid = DatasetRequirement(
        id="aid",
        decision_ids=("measure", "year"),
        measure="official development assistance",
        entities=(Entity(name="France", role="donor"), Entity(name="Kenya", role="recipient")),
        period=Period(start=2015, end=2024),
        unit=USD,
        source_spec_id=_GROSS,
    )
    total = AggregateRequirement(
        id="total",
        decision_ids=("shape", "year"),
        input_id="aid",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    ctx = _FakeToolContext()
    _show_block(ctx, _FRANCE_KENYA)
    propose = _tools_by_name()["propose_plan"].func
    token = current_recorded_identity.set(_turn_identities({_GROSS: _FRANCE_KENYA}, ()).get)
    try:
        card = propose(
            question="how much aid does France give Kenya",
            answer_mode=AnswerMode.COMPUTED,
            decisions=drafts([decision(), shape, year]),
            evidence_requirements=[aid, total],
            tool_context=ctx,
        )
        assert card["outcome"] == Outcome.OK.value, card
        (copy,) = windowed_pins(ctx.state).values()
        shown = load_plan(ctx.state)
        assert shown is not None
        settled_year = next(d for d in shown.decisions if d.slot == "year")
        pick = decision_pick(
            shown,
            choice_id="decision/shape",
            option_id=decision_option_id("the yearly average"),
            to_origin={copy.spec_id: copy.windowed_from},
        )
        assert pick is not None
        assert pick.cited == ("shape", "year")
        save_decision_pick(ctx.state, pick)

        result = propose(
            amends_version=card["version"],
            decisions=drafts(
                [
                    shape.model_copy(
                        update={
                            "chosen": "the yearly average",
                            "fixes": DecisionFixes(reduction=Reduction.MEAN),
                        }
                    )
                ]
            ),
            evidence_requirements=[total.model_copy(update={"reduction": Reduction.MEAN})],
            tool_context=ctx,
        )
    finally:
        current_recorded_identity.reset(token)

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    decisions = {d.slot: d for d in plan.decisions}
    assert decisions["year"] == settled_year
    assert decisions["shape"].fixes == DecisionFixes(reduction=Reduction.MEAN)
    requirements = {r.id: r for r in plan.evidence_requirements}
    assert requirements["total"].decision_ids == ("shape", "year")
    assert requirements["aid"].source_spec_id == copy.spec_id  # type: ignore[union-attr]
    source = source_from_spec_block(cast(Any, {**_FRANCE_KENYA, "spec_id": copy.spec_id}))
    tools = await McpSurface(
        sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp}
    ).list_tools()
    compiled = compile_plan(
        plan=plan,
        translation=build_translation(plan, {"aid": copy.spec_id}),
        schemas={tool.name: tool for tool in tools},
        discovered={"aid": source},
    )
    assert compiled.workflow is not None, compiled.issues
