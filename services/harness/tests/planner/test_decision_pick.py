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
from google.adk.events import Event, EventActions
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.genai import types
from harness.decision_pick import decision_pick
from harness.plan_display import decision_option_id
from harness.studio_state import _turn_identities
from planner.agent import build_planner_tools
from planner.plan import (
    ASKS_STATE_KEY,
    DECISION_PICK_STATE_KEY,
    AggregateRequirement,
    AnswerMode,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    Period,
    RankRequirement,
    RequirementSide,
    ResolvedBy,
    ShareOfRequirement,
    Unit,
    current_recorded_identity,
    load_plan,
    save_decision_pick,
    windowed_pins,
)
from planner.records import GroupBy, RankAmong, RankDirection, Reduction
from planner.tool_results import Outcome, ProblemCode
from translator.mechanical import build_translation

from ..basis_alignment import HEALTH, TOTAL, provider_block
from ..decision_drafts import drafts
from ..session_events import response_event
from .test_agent import (
    _CONSTANT_GROSS,
    _FRANCE_KENYA,
    _GROSS,
    _US_MULTILATERAL,
    _US_TOTAL,
    _constant_block,
    _dac_countries,
    _FakeToolContext,
    _offering_constant,
    _prices_ask_events,
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
        alignments={},
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
        alignments={},
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
        alignments={},
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


def test_the_picked_share_proposed_with_no_pick_held_is_refused() -> None:
    """The plan the pick above records, US total ODA as a share of itself, proposed
    by the planner with no pick held: nothing the user chose put the sides on one
    series, so the share is refused for the planner to redraft."""
    part = Decision(
        slot="part",
        description="Which figure leads the share.",
        options=["US multilateral ODA", "another figure"],
        chosen="another figure",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="r",
        fixes=DecisionFixes(measure="another figure"),
    )
    measure = part.model_copy(
        update={"slot": "measure", "options": ["a", "b"], "chosen": "a", "fixes": None}
    )
    ctx = _FakeToolContext()
    _show_blocks(ctx, _US_TOTAL)

    result = _tools_by_name()["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([measure, part]),
        evidence_requirements=[
            _dac_countries("first", _US_TOTAL).model_copy(
                update={"measure": "another figure", "decision_ids": ("measure", "part")}
            ),
            _dac_countries("second", _US_TOTAL),
            _share("first", "second"),
        ],
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.SHARE_OF_ITSELF.value, result
    assert load_plan(ctx.state) is None


def test_a_pick_both_sides_cite_leaves_no_share_of_itself_admitted() -> None:
    """The card shares US multilateral ODA over the US total in 2019, both inputs
    citing the year. The user picks 2018, and the replan rewrites both inputs, pinning
    each on the US total. The pick moved only the year; the planner put the two sides
    on one series, so the share is refused."""
    measure = Decision(
        slot="measure",
        description="What the share measures.",
        options=["a", "b"],
        chosen="a",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="r",
    )
    year = Decision(
        slot="year",
        description="Which year the share reads.",
        options=["2019", "2018"],
        chosen="2019",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="r",
        fixes=DecisionFixes(period=Period(start=2019, end=2019)),
    )
    cited = {"decision_ids": ("measure", "year")}
    ctx = _FakeToolContext()
    _show_blocks(ctx, _US_MULTILATERAL, _US_TOTAL)
    propose = _tools_by_name()["propose_plan"].func
    card = propose(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([measure, year]),
        evidence_requirements=[
            _dac_countries("first", _US_MULTILATERAL).model_copy(update=cited),
            _dac_countries("second", _US_TOTAL).model_copy(update=cited),
            _share("first", "second"),
        ],
        tool_context=ctx,
    )
    assert card["outcome"] == Outcome.OK.value, card
    shown = load_plan(ctx.state)
    assert shown is not None
    pick = decision_pick(
        shown,
        choice_id="decision/year",
        option_id=decision_option_id("2018"),
        to_origin={},
        alignments={},
    )
    assert pick is not None
    save_decision_pick(ctx.state, pick)
    in_2018 = {**cited, "period": Period(start=2018, end=2018)}

    result = propose(
        amends_version=card["version"],
        decisions=drafts(
            [
                year.model_copy(
                    update={
                        "chosen": "2018",
                        "fixes": DecisionFixes(period=Period(start=2018, end=2018)),
                    }
                )
            ]
        ),
        evidence_requirements=[
            _dac_countries("first", _US_TOTAL).model_copy(update={**in_2018, "measure": "a"}),
            _dac_countries("second", _US_TOTAL).model_copy(update={**in_2018, "measure": "b"}),
        ],
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.SHARE_OF_ITSELF.value, result


def test_a_pick_on_another_decision_keeps_a_share_an_earlier_pick_made() -> None:
    """The user's pick put the first share's numerator on the US total, a share of
    itself admitted as theirs. A second pick, of the year a second share's inputs
    cite, keeps the first share as the card holds it, which the planner may not
    change, so the replan is admitted."""
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
    year = Decision(
        slot="year",
        description="Which year the second share reads.",
        options=["2019", "2018"],
        chosen="2019",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="r",
        fixes=DecisionFixes(period=Period(start=2019, end=2019)),
    )
    cites_year = {"decision_ids": ("measure", "year")}
    third = _dac_countries("third", _US_MULTILATERAL).model_copy(update=cites_year)
    fourth = _dac_countries("fourth", _US_TOTAL).model_copy(update=cites_year)
    ctx = _FakeToolContext()
    _show_blocks(ctx, _US_MULTILATERAL, _US_TOTAL)
    propose = _tools_by_name()["propose_plan"].func
    card = propose(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([measure, part, year]),
        evidence_requirements=[
            first.model_copy(update={"decision_ids": ("measure", "part")}),
            _dac_countries("second", _US_TOTAL),
            third,
            fourth,
            _share("first", "second"),
            _share("third", "fourth").model_copy(update={"id": "share_two"}),
        ],
        tool_context=ctx,
    )
    assert card["outcome"] == Outcome.OK.value, card
    picks = [("part", "another figure"), ("year", "2018")]
    replans = [
        (
            part.model_copy(
                update={
                    "chosen": "another figure",
                    "fixes": DecisionFixes(measure="another figure"),
                }
            ),
            [
                _dac_countries("first", _US_TOTAL).model_copy(
                    update={"measure": "another figure", "decision_ids": ("measure", "part")}
                )
            ],
        ),
        (
            year.model_copy(
                update={
                    "chosen": "2018",
                    "fixes": DecisionFixes(period=Period(start=2018, end=2018)),
                }
            ),
            [
                held.model_copy(update={"period": Period(start=2018, end=2018)})
                for held in (third, fourth)
            ],
        ),
    ]

    version = card["version"]
    for (slot, chosen), (picked, requirements) in zip(picks, replans, strict=True):
        shown = load_plan(ctx.state)
        assert shown is not None
        pick = decision_pick(
            shown,
            choice_id=f"decision/{slot}",
            option_id=decision_option_id(chosen),
            to_origin={},
            alignments={},
        )
        assert pick is not None
        save_decision_pick(ctx.state, pick)
        result = propose(
            amends_version=version,
            decisions=drafts([picked]),
            evidence_requirements=requirements,
            tool_context=ctx,
        )
        assert result["outcome"] == Outcome.OK.value, (slot, result)
        version = result["version"]


@pytest.mark.parametrize(
    ("reads", "refused"),
    [
        # The replan points the numerator's mean at the denominator's input: the share
        # step and its denominator are kept, but a step between them was rewritten
        # onto the denominator's series.
        pytest.param("second", True, id="rewired-onto-the-denominator"),
        pytest.param("first", False, id="kept-on-its-input"),
    ],
)
def test_a_pick_rewriting_a_step_beneath_a_share_onto_its_denominator_is_refused(
    reads: str, refused: bool
) -> None:
    """The card divides US multilateral ODA summed over its years by the US total's
    mean. The user picks the mean for the numerator too, and the replan rewrites the
    numerator's step. Read from the total, both sides are the total's mean, a share of
    itself the pick did not make, and it is refused; read from its own input, the
    share is admitted."""
    measure = Decision(
        slot="measure",
        description="What the share measures.",
        options=["a", "b"],
        chosen="a",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="r",
    )
    how = Decision(
        slot="how",
        description="How the numerator folds its years.",
        options=["sum", "mean"],
        chosen="sum",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="r",
        fixes=DecisionFixes(reduction=Reduction.SUM),
    )
    numerator = AggregateRequirement(
        id="num",
        decision_ids=("measure", "how"),
        input_id="first",
        reduction=Reduction.SUM,
        group_by=(),
    )
    ctx = _FakeToolContext()
    _show_blocks(ctx, _US_MULTILATERAL, _US_TOTAL)
    propose = _tools_by_name()["propose_plan"].func
    card = propose(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([measure, how]),
        evidence_requirements=[
            _dac_countries("first", _US_MULTILATERAL),
            _dac_countries("second", _US_TOTAL),
            numerator,
            AggregateRequirement(
                id="den",
                decision_ids=("measure",),
                input_id="second",
                reduction=Reduction.MEAN,
                group_by=(),
            ),
            _share("first", "second").model_copy(update={"id": "share_two"}),
            _share("num", "den").model_copy(update={"accept_provider_difference": True}),
        ],
        tool_context=ctx,
    )
    assert card["outcome"] == Outcome.OK.value, card
    shown = load_plan(ctx.state)
    assert shown is not None
    pick = decision_pick(
        shown,
        choice_id="decision/how",
        option_id=decision_option_id("mean"),
        to_origin={},
        alignments={},
    )
    assert pick is not None
    save_decision_pick(ctx.state, pick)

    result = propose(
        amends_version=card["version"],
        decisions=drafts(
            [
                how.model_copy(
                    update={"chosen": "mean", "fixes": DecisionFixes(reduction=Reduction.MEAN)}
                )
            ]
        ),
        evidence_requirements=[
            numerator.model_copy(update={"input_id": reads, "reduction": Reduction.MEAN})
        ],
        tool_context=ctx,
    )

    if refused:
        assert result["code"] == ProblemCode.SHARE_OF_ITSELF.value, result
    else:
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
        alignments={},
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
            alignments={},
        )
        assert pick is not None
        assert pick.cited_by_every == ("shape", "year")
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


@pytest.mark.parametrize(
    "pin", [pytest.param(TOTAL, id="onto-another-pin"), pytest.param(HEALTH, id="same-pin")]
)
def test_a_pick_asks_no_citation_of_a_scope_choice_admission_writes(pin: str) -> None:
    """US health aid is read as bilateral aid only, a scope the user's words
    set, so admission cites the user's scope choices from it. Picking the
    later years moves it to 2020-2024 on its own pin or on the US total's,
    citing only what the planner writes: admission carries or drops the
    scope choices with the pin, so the replan is admitted."""
    measure = Decision(
        slot="measure",
        description="What to measure.",
        options=["a", "b"],
        chosen="a",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="r",
    )
    years = Decision(
        slot="part",
        description="Which years.",
        options=["2015-2024", "2020-2024"],
        chosen="2015-2024",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="r",
        fixes=DecisionFixes(period=Period(start=2015, end=2024)),
    )

    def us_aid(requirement_id: str, spec_id: str, start: int) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=("measure", "part"),
            measure="US aid to health" if spec_id == HEALTH else "US total net ODA",
            entities=(Entity(name="United States", role="donor"),),
            period=Period(start=start, end=2024),
            source_spec_id=spec_id,
        )

    ctx = _FakeToolContext()
    _show_blocks(
        ctx,
        provider_block(HEALTH, narrow=True, active="provider:bilateral", provenance="explicit"),
        provider_block(TOTAL, narrow=False, active="provider:inclusive"),
    )
    propose = _tools_by_name()["propose_plan"].func
    card = propose(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([measure, years]),
        evidence_requirements=[us_aid("health", HEALTH, 2015)],
        tool_context=ctx,
    )
    assert card["outcome"] == Outcome.OK.value, card
    shown = load_plan(ctx.state)
    assert shown is not None
    assert any(d.scope is not None and d.resolved_by is ResolvedBy.USER for d in shown.decisions)
    pick = decision_pick(
        shown,
        choice_id="decision/part",
        option_id=decision_option_id("2020-2024"),
        to_origin={},
        alignments={},
    )
    assert pick is not None
    assert pick.cited_by_every == ("part",)
    save_decision_pick(ctx.state, pick)

    result = propose(
        amends_version=card["version"],
        decisions=drafts(
            [
                years.model_copy(
                    update={
                        "chosen": "2020-2024",
                        "fixes": DecisionFixes(period=Period(start=2020, end=2024)),
                    }
                )
            ]
        ),
        evidence_requirements=[us_aid("recent", pin, 2020)],
        removed_evidence_requirements=["health"],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result


@pytest.mark.parametrize(
    ("top", "listed"),
    [
        pytest.param({"reduction": Reduction.MEAN}, {}, id="ten-by-the-mean"),
        pytest.param({"direction": RankDirection.LOWEST_FIRST}, {}, id="ten-lowest-first"),
        pytest.param(
            {"direction": RankDirection.LOWEST_FIRST},
            {"direction": RankDirection.LOWEST_FIRST},
            id="ten-lowest-first-listing-the-direction",
        ),
    ],
)
def test_a_pick_whose_decision_settles_to_the_cards_value_is_refused(
    decision: Callable[..., Decision], top: dict[str, Any], listed: dict[str, Any]
) -> None:
    """The user picks the top 5 donors, and the replan ranks 10 another way.
    The requirements citing the decision still settle the limit and the
    population the card's option fixed, so the plan would rank 10 donors
    under the label of 5: refused, whatever other field the decision lists."""
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
        alignments={},
    )
    assert pick is not None
    save_decision_pick(ctx.state, pick)
    aid, ranked = _year_kept(YEAR)

    result = _propose_plan(
        ctx,
        decisions=drafts(
            [
                RANKING.model_copy(
                    update={
                        "chosen": "the top 5 donors",
                        "fixes": DecisionFixes(limit=5, among=RankAmong.POPULATION, **listed),
                    }
                )
            ]
        ),
        evidence_requirements=[aid, ranked.model_copy(update=top)],
        amends_version=first["version"],
    )

    assert result["code"] == ProblemCode.USER_PICK_NOT_HELD.value, result
    assert "fix what the card's option fixed" in result["message"]


def _two_years() -> tuple[Decision, Decision, list[DatasetRequirement]]:
    """A planner prices decision over the DAC total for the user's 2023 and
    for their base year 2015, each total citing its own year."""
    prices = Decision(
        slot="prices",
        description="Which prices the figures are in.",
        options=["current prices", "constant prices"],
        chosen="current prices",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="Current prices are the default.",
        fixes=DecisionFixes(unit=USD),
    )
    base = USER_YEAR.model_copy(
        update={
            "slot": "base",
            "options": ["2015", "2010"],
            "chosen": "2015",
            "fixes": DecisionFixes(period=Period(start=2015, end=2015)),
        }
    )
    latest = _dac_total().model_copy(update={"decision_ids": ("measure", "year", "prices")})
    earliest = latest.model_copy(
        update={
            "id": "dac15",
            "period": Period(start=2015, end=2015),
            "decision_ids": ("measure", "base", "prices"),
        }
    )
    return prices, base, [latest, earliest]


@pytest.mark.parametrize(
    ("cites_base", "outcome"),
    [
        pytest.param(True, Outcome.OK, id="each-its-own-year"),
        pytest.param(False, Outcome.REFUSED, id="base-year-dropped"),
    ],
)
def test_a_pick_over_figures_citing_different_user_years_keeps_each_year_cited(
    decision: Callable[..., Decision], cites_base: bool, outcome: Outcome
) -> None:
    """The user fixed 2023 and a 2015 base, each cited by one total. Picking
    constant prices restates both totals: each citing its own year is
    admitted, and one dropping the base year from every total is refused,
    naming the base year."""
    prices, base, totals = _two_years()
    ctx = _context()
    first = _propose_plan(
        ctx,
        question="how has aid to Kenya grown since 2015",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision(), prices, USER_YEAR, base]),
        evidence_requirements=totals,
    )
    assert first["outcome"] == Outcome.OK.value, first
    card = load_plan(ctx.state)
    assert card is not None
    pick = decision_pick(
        card,
        choice_id="decision/prices",
        option_id=decision_option_id("constant prices"),
        to_origin={},
        alignments={},
    )
    assert pick is not None
    save_decision_pick(ctx.state, pick)
    constant = Unit(currency="USD", prices="constant", base_year=2022)
    latest, earliest = (t.model_copy(update={"unit": constant}) for t in totals)
    if not cites_base:
        earliest = earliest.model_copy(update={"decision_ids": ("measure", "prices")})

    result = _propose_plan(
        ctx,
        decisions=drafts([prices.model_copy(update={"chosen": "constant prices"})]),
        evidence_requirements=[latest, earliest],
        amends_version=first["version"],
    )

    assert result["outcome"] == outcome.value, result
    if outcome is Outcome.REFUSED:
        assert result["detail"]["kept"] == "base"


@pytest.mark.parametrize(
    ("card_fixes", "picked_fixes"),
    [
        pytest.param(DONOR_SCOPE.fixes, DecisionFixes(measure="net ODA"), id="entities"),
        pytest.param(
            DecisionFixes(entities=DONOR_SCOPE.fixes.entities, unit=USD),
            DecisionFixes(measure="net ODA", unit=USD),
            id="entities-and-the-unit-both-options-share",
        ),
    ],
)
def test_a_pick_may_keep_what_it_replaces_as_the_input_of_a_new_figure(
    decision: Callable[..., Decision], card_fixes: DecisionFixes, picked_fixes: DecisionFixes
) -> None:
    """ "The DAC total" becomes "as a share of all aid to Kenya": the DAC
    total stays, unchanged, as the share's numerator, and the picked option
    fixes the measure both totals read. The plan no longer ends in the DAC
    total, and the entities the card's option fixed no longer settle, so the
    replan is admitted, though the unit both options fix stays as it was."""
    shown_as = DONOR_SCOPE.model_copy(
        update={"options": ["the DAC total", "as a share of all aid to Kenya"], "fixes": card_fixes}
    )
    ctx = _context()
    first = _propose_plan(
        ctx,
        question="how much aid does Kenya get",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision(), shown_as, USER_YEAR]),
        evidence_requirements=[_dac_total()],
    )
    assert first["outcome"] == Outcome.OK.value, first
    card = load_plan(ctx.state)
    assert card is not None
    pick = decision_pick(
        card,
        choice_id="decision/donor_scope",
        option_id=decision_option_id("as a share of all aid to Kenya"),
        to_origin={},
        alignments={},
    )
    assert pick is not None
    save_decision_pick(ctx.state, pick)
    every_donor = _dac_total().model_copy(
        update={
            "id": "all",
            "entities": (Entity(name="Kenya", role="recipient"),),
            "source_spec_id": EVERY_DONOR,
        }
    )
    share = ShareOfRequirement(
        id="share",
        decision_ids=("measure", "donor_scope", "year"),
        numerator=RequirementSide(input_id="dac"),
        denominator=RequirementSide(input_id="all"),
    )

    result = _propose_plan(
        ctx,
        decisions=drafts(
            [
                shown_as.model_copy(
                    update={"chosen": "as a share of all aid to Kenya", "fixes": picked_fixes}
                )
            ]
        ),
        evidence_requirements=[every_donor, share],
        amends_version=first["version"],
    )

    assert result["outcome"] == Outcome.OK.value, result


def test_a_pick_this_release_cannot_read_is_refused_and_nothing_is_recorded(
    decision: Callable[..., Decision],
) -> None:
    """The entry holds a pick an earlier release wrote, which states no fixes.
    Read as no pick it would let the replan change anything, so the replan
    is refused and the plan stays as the card showed it."""
    ctx = _context()
    first = _recorded_dac_total(ctx, decision)
    ctx.state[DECISION_PICK_STATE_KEY] = {
        "slot": "donor_scope",
        "chosen": "the top 10 donors",
        "origin_reading": "0" * 64,
        "cited": ["donor_scope"],
        "kept_requirements": {},
        "kept_decisions": {},
    }

    result = _propose_plan(
        ctx,
        decisions=drafts([TOP_TEN]),
        evidence_requirements=_year_kept(YEAR),
        removed_evidence_requirements=["dac"],
        amends_version=first["version"],
    )

    assert result["outcome"] == Outcome.REFUSED.value, result
    assert result["code"] == ProblemCode.USER_PICK_NOT_HELD.value
    recorded = load_plan(ctx.state)
    assert recorded is not None and recorded.version == first["version"]


@pytest.mark.parametrize(
    ("card_fixes", "picked_fixes"),
    [
        pytest.param(DONOR_SCOPE.fixes, DecisionFixes(measure="net ODA"), id="entities"),
        pytest.param(
            DecisionFixes(entities=DONOR_SCOPE.fixes.entities, unit=USD),
            DecisionFixes(measure="net ODA", unit=USD),
            id="entities-and-the-unit",
        ),
        pytest.param(
            DONOR_SCOPE.fixes,
            DecisionFixes(entities=DONOR_SCOPE.fixes.entities, measure="net ODA"),
            id="the-share-listing-entities",
        ),
    ],
)
def test_a_landed_shape_pick_can_be_picked_back(
    decision: Callable[..., Decision], card_fixes: DecisionFixes, picked_fixes: DecisionFixes
) -> None:
    """The share of all aid to Kenya lands for the DAC total, and the user
    picks the DAC total back. The replan drops the share and its denominator
    and keeps the DAC total, which reads the measure and unit the share's
    decision settled: a change of shape, admitted though those fields keep
    their values."""
    shown_as = DONOR_SCOPE.model_copy(
        update={"options": ["the DAC total", "as a share of all aid to Kenya"], "fixes": card_fixes}
    )
    ctx = _context()
    first = _propose_plan(
        ctx,
        question="how much aid does Kenya get",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision(), shown_as, USER_YEAR]),
        evidence_requirements=[_dac_total()],
    )
    assert first["outcome"] == Outcome.OK.value, first

    def pick(label: str) -> None:
        card = load_plan(ctx.state)
        assert card is not None
        picked = decision_pick(
            card,
            choice_id="decision/donor_scope",
            option_id=decision_option_id(label),
            to_origin={},
            alignments={},
        )
        assert picked is not None
        save_decision_pick(ctx.state, picked)

    pick("as a share of all aid to Kenya")
    shared = _propose_plan(
        ctx,
        decisions=drafts(
            [
                shown_as.model_copy(
                    update={"chosen": "as a share of all aid to Kenya", "fixes": picked_fixes}
                )
            ]
        ),
        evidence_requirements=[
            _dac_total().model_copy(
                update={
                    "id": "all",
                    "entities": (Entity(name="Kenya", role="recipient"),),
                    "source_spec_id": EVERY_DONOR,
                }
            ),
            ShareOfRequirement(
                id="share",
                decision_ids=("measure", "donor_scope", "year"),
                numerator=RequirementSide(input_id="dac"),
                denominator=RequirementSide(input_id="all"),
            ),
        ],
        amends_version=first["version"],
    )
    assert shared["outcome"] == Outcome.OK.value, shared
    pick("the DAC total")

    result = _propose_plan(
        ctx,
        decisions=drafts([shown_as]),
        evidence_requirements=[],
        removed_evidence_requirements=["all", "share"],
        amends_version=shared["version"],
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [r.id for r in plan.evidence_requirements] == ["dac"]


@pytest.mark.parametrize(
    ("pin", "unit", "refused"),
    [
        (_CONSTANT_GROSS, Unit(currency="USD", prices="constant", base_year=2023), None),
        (_GROSS, None, ProblemCode.USER_DECISION_DROPPED),
    ],
    ids=["held", "dropped"],
)
def test_a_pick_replan_keeps_the_users_prices_choice(
    pin: str, unit: Unit | None, refused: ProblemCode | None
) -> None:
    """The user chose constant prices for French aid to Kenya, then picked
    other years on the card. The prices decision is admission's to write and
    cite, so the replan need not cite it, and the pick does not release the
    choice: a replan in constant prices lands citing it, and one stating no
    unit is refused as dropping it."""
    measure = Decision(
        slot="measure",
        description="What to measure.",
        options=["a", "b"],
        chosen="a",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="r",
    )
    years = Decision(
        slot="years",
        description="Which years.",
        options=["2015-2022", "2019-2022"],
        chosen="2015-2022",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="r",
        fixes=DecisionFixes(period=Period(start=2015, end=2022)),
    )

    def aid(requirement_id: str, start: int, **update: Any) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=("measure", "years"),
            measure="French aid to Kenya",
            entities=(Entity(name="France", role="donor"), Entity(name="Kenya", role="recipient")),
            period=Period(start=start, end=2022),
            source_spec_id=_CONSTANT_GROSS,
            unit=Unit(currency="USD", prices="constant", base_year=2023),
        ).model_copy(update=update)

    ctx = _FakeToolContext()
    _show_blocks(ctx, _offering_constant(), _constant_block())
    ctx.session.events = _prices_ask_events(constant_spec_id=_CONSTANT_GROSS)
    propose = _tools_by_name()["propose_plan"].func
    card = propose(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([measure, years]),
        evidence_requirements=[aid("all", 2015)],
        tool_context=ctx,
    )
    assert card["outcome"] == Outcome.OK.value, card
    shown = load_plan(ctx.state)
    assert shown is not None
    priced = f"user_prices_{_GROSS}"
    assert priced in shown.requirement("all").decision_ids
    pick = decision_pick(
        shown,
        choice_id="decision/years",
        option_id=decision_option_id("2019-2022"),
        to_origin={},
        alignments={},
    )
    assert pick is not None
    save_decision_pick(ctx.state, pick)

    result = propose(
        amends_version=card["version"],
        decisions=drafts(
            [
                years.model_copy(
                    update={
                        "chosen": "2019-2022",
                        "fixes": DecisionFixes(period=Period(start=2019, end=2022)),
                    }
                )
            ]
        ),
        evidence_requirements=[aid("recent", 2019, source_spec_id=pin, unit=unit)],
        removed_evidence_requirements=["all"],
        tool_context=ctx,
    )

    if refused is not None:
        assert result["code"] == refused.value, result
        return
    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    assert priced in plan.requirement("recent").decision_ids


def _switch_event(choice_id: str, option_id: str) -> Any:
    """The approval card's answer switching row `choice_id` to option `option_id`."""
    part = create_request_input_response(
        "approve", {"result": {"switch": {"choice_id": choice_id, "option_id": option_id}}}
    )
    return Event(content=types.Content(role="user", parts=[part]))


@pytest.mark.parametrize(
    ("answered", "picked", "shown", "pin", "unit", "code"),
    [
        (
            "constant",
            "current prices",
            True,
            _GROSS,
            Unit(currency="USD", prices="current"),
            None,
        ),
        (
            "current",
            "constant prices",
            True,
            _CONSTANT_GROSS,
            Unit(currency="USD", prices="constant", base_year=2023),
            None,
        ),
        (
            "current",
            "constant prices",
            False,
            _GROSS,
            None,
            ProblemCode.PRICES_UNSERVED,
        ),
    ],
    ids=["to-current", "to-constant", "to-constant-unserved"],
)
def test_a_switch_of_the_prices_row_is_the_users_new_prices(
    answered: str, picked: str, shown: bool, pin: str, unit: Unit | None, code: ProblemCode | None
) -> None:
    """The user answered the prices question, then switched the card's prices
    row. The switch is a new answer for the part: the replan in the picked
    prices lands citing it, and constant prices with no constant series shown
    hand the turn back for a search."""
    measure = Decision(
        slot="measure",
        description="What to measure.",
        options=["a", "b"],
        chosen="a",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="r",
    )

    def aid(requirement_id: str, **update: Any) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=("measure",),
            measure="French aid to Kenya",
            entities=(Entity(name="France", role="donor"), Entity(name="Kenya", role="recipient")),
            period=Period(start=2015, end=2022),
            source_spec_id=_GROSS,
            unit=Unit(currency="USD", prices="current"),
        ).model_copy(update=update)

    ctx = _FakeToolContext()
    if shown:
        _show_blocks(ctx, _offering_constant(), _constant_block())
    else:
        _show_blocks(ctx, _FRANCE_KENYA)
    constant = answered == "constant"
    ctx.session.events = _prices_ask_events(
        constant_spec_id=_CONSTANT_GROSS if constant else None, pick=None if constant else "o1"
    )
    first = (
        aid(
            "first",
            source_spec_id=_CONSTANT_GROSS,
            unit=Unit(currency="USD", prices="constant", base_year=2023),
        )
        if constant
        else aid("first")
    )
    propose = _tools_by_name()["propose_plan"].func
    card = propose(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([measure]),
        evidence_requirements=[first],
        tool_context=ctx,
    )
    assert card["outcome"] == Outcome.OK.value, card
    plan = load_plan(ctx.state)
    assert plan is not None
    slot = f"user_prices_{_GROSS}"
    option = decision_option_id(picked)
    pick = decision_pick(
        plan, choice_id=f"decision/{slot}", option_id=option, to_origin={}, alignments={}
    )
    assert pick is not None
    save_decision_pick(ctx.state, pick)
    ctx.session.events = [
        *ctx.session.events,
        response_event("propose_plan", card),
        _switch_event(f"decision/{slot}", option),
    ]

    result = propose(
        amends_version=card["version"],
        evidence_requirements=[aid("second", source_spec_id=pin, unit=unit)],
        removed_evidence_requirements=["first"],
        tool_context=ctx,
    )

    if code is not None:
        assert result["code"] == code.value, result
        return
    assert result["outcome"] == Outcome.OK.value, result
    landed = load_plan(ctx.state)
    assert landed is not None
    assert slot in landed.requirement("second").decision_ids
    assert next(d for d in landed.decisions if d.slot == slot).chosen == picked


def test_no_pick_is_built_for_a_scope_choice_admission_writes() -> None:
    """A decision `propose_plan` writes other than the user's prices, a scope
    choice here, builds no pick from the card."""
    ctx = _FakeToolContext()
    _show_blocks(
        ctx,
        provider_block(HEALTH, narrow=True, active="provider:bilateral", provenance="explicit"),
    )
    card = _tools_by_name()["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts(
            [
                Decision(
                    slot="measure",
                    description="What to measure.",
                    options=["a", "b"],
                    chosen="a",
                    resolved_by=ResolvedBy.PLANNER_DEFAULT,
                    rationale="r",
                )
            ]
        ),
        evidence_requirements=[
            DatasetRequirement(
                id="health",
                decision_ids=("measure",),
                measure="US aid to health",
                entities=(Entity(name="United States", role="donor"),),
                period=Period(start=2015, end=2024),
                source_spec_id=HEALTH,
            )
        ],
        tool_context=ctx,
    )
    assert card["outcome"] == Outcome.OK.value, card
    plan = load_plan(ctx.state)
    assert plan is not None
    scoped = [d for d in plan.decisions if d.scope is not None and d.fixes is not None]
    assert scoped, [d.slot for d in plan.decisions]
    for decision in scoped:
        other = next(o for o in decision.options if o != decision.chosen)
        assert (
            decision_pick(
                plan,
                choice_id=f"decision/{decision.slot}",
                option_id=decision_option_id(other),
                to_origin={},
                alignments={},
            )
            is None
        )


def test_a_refused_prices_switch_leaves_the_card_s_prices_for_the_next_pick() -> None:
    """The user answered current prices with no constant series shown, then
    switched the card's prices row to constant: that replan ends on
    `prices_unserved`, and the card keeps current prices. A later switch of
    the years row is held to the card's current prices, so it lands."""
    measure = Decision(
        slot="measure",
        description="What to measure.",
        options=["a", "b"],
        chosen="a",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="r",
    )
    years = Decision(
        slot="years",
        description="Which years.",
        options=["2015-2022", "2019-2022"],
        chosen="2015-2022",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="r",
        fixes=DecisionFixes(period=Period(start=2015, end=2022)),
    )

    def aid(requirement_id: str, start: int, **update: Any) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=("measure", "years"),
            measure="French aid to Kenya",
            entities=(Entity(name="France", role="donor"), Entity(name="Kenya", role="recipient")),
            period=Period(start=start, end=2022),
            source_spec_id=_GROSS,
            unit=Unit(currency="USD", prices="current"),
        ).model_copy(update=update)

    ctx = _FakeToolContext()
    _show_blocks(ctx, _FRANCE_KENYA)
    ctx.session.events = _prices_ask_events(pick="o1")
    propose = _tools_by_name()["propose_plan"].func
    card = propose(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([measure, years]),
        evidence_requirements=[aid("all", 2015)],
        tool_context=ctx,
    )
    assert card["outcome"] == Outcome.OK.value, card
    plan = load_plan(ctx.state)
    assert plan is not None
    slot = f"user_prices_{_GROSS}"
    constant = decision_option_id("constant prices")
    pick = decision_pick(
        plan, choice_id=f"decision/{slot}", option_id=constant, to_origin={}, alignments={}
    )
    assert pick is not None
    save_decision_pick(ctx.state, pick)
    ctx.session.events = [
        *ctx.session.events,
        response_event("propose_plan", card),
        _switch_event(f"decision/{slot}", constant),
    ]
    switched = propose(
        amends_version=card["version"],
        evidence_requirements=[aid("all2", 2015, unit=None)],
        removed_evidence_requirements=["all"],
        tool_context=ctx,
    )
    assert switched["code"] == ProblemCode.PRICES_UNSERVED.value, switched
    later = decision_option_id("2019-2022")
    pick = decision_pick(
        plan, choice_id="decision/years", option_id=later, to_origin={}, alignments={}
    )
    assert pick is not None
    save_decision_pick(ctx.state, pick)
    ctx.session.events = [
        *ctx.session.events,
        response_event("propose_plan", switched),
        _switch_event("decision/years", later),
    ]

    result = propose(
        amends_version=card["version"],
        decisions=drafts(
            [
                years.model_copy(
                    update={
                        "chosen": "2019-2022",
                        "fixes": DecisionFixes(period=Period(start=2019, end=2022)),
                    }
                )
            ]
        ),
        evidence_requirements=[aid("recent", 2019)],
        removed_evidence_requirements=["all"],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
