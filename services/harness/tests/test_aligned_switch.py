"""A plan-card switch of a plan holding an input alignment moved onto another's pin.

The plan compares US total ODA the user stated gross with its average Search left on
net (`tests.planner.test_aligned_identity`). Alignment moved the average onto
gross, so both inputs read the gross pin's spec_id, each under its own basis row:
the total's the user's, the average's the planner's. A switch of either row moves
the input that row is for alone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from compiler.evidence import RetrievalEvidence, check_evidence
from compiler.sources import source_from_spec_block
from compiler.surface import ToolCallResult, ToolSurface
from data_mcp.definition_store import canonicalize, spec_id_for
from data_mcp.definition_writes import MAX_IDENTITY_BYTES, windowed_identity
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from harness.graph import switch as switch_module
from harness.graph.state import ORCHESTRATOR_SEARCHES_KEY
from harness.graph.switch import SwitchFailedError
from harness.plan_display import choice_id, shown_decisions
from harness.planner_seam import PlannerSeam
from harness.settings import Settings
from harness.turn_records import SearchReading
from planner.plan import (
    ASK_BREAKDOWNS_STATE_KEY,
    ASKS_STATE_KEY,
    AnswerMode,
    AskAnswer,
    AskBreakdown,
    DatasetRequirement,
    Decision,
    Entity,
    EvidenceRequirement,
    Period,
    PinSource,
    Plan,
    Reduction,
    RequirementSide,
    ResolvedBy,
    ShareOfRequirement,
    WholeOf,
    aligned_pins,
    alignments,
    definitions_on_record,
    identity_on_record,
    load_plan,
    period_window,
    pin_sources,
    save_pin_sources,
    windowed_pins,
)
from planner.records import EvidenceIssueCode
from planner.tool_results import Outcome
from planner.workflow import ResolvedGroup, ToolId
from submission.service import statement

from .basis_alignment import (
    HEALTH,
    TOTAL,
    TOTAL_GROSS_BILATERAL,
    ProviderCompiles,
    WholeCompiles,
    basis_block,
    provider_block,
    us_2023_block,
)
from .decision_drafts import drafts
from .planner.test_agent import (
    _COMPARE,
    DecisionFactory,
    _difference,
    _FakeToolContext,
    _reduced,
    _show_blocks,
    _tools_reading,
)
from .planner.test_aligned_identity import (
    _GROSS,
    _NET,
    _gross_compared_with_net_average,
    _gross_compiles,
    _left_on_net,
)

_GRANT_EQUIVALENT = f"spec_{'9' * 32}"


def _counted_on_grant_equivalents() -> dict[str, Any]:
    """`_GROSS`'s part counted in grant equivalents, the user's choice on the card,
    under the spec_id `_GRANT_EQUIVALENT`."""
    text = json.dumps(_GROSS).replace(_GROSS["spec_id"], _GRANT_EQUIVALENT)
    block = json.loads(text)
    scope = block["resolved"]["scope_choices"]
    (basis,) = [c for c in scope["choices"] if c["axis"] == "measure_basis"]
    basis["active"], basis["provenance"] = "measure_basis:grant_equivalent", "explicit"
    scope["entries"] = [
        {
            "axis": "measure_basis",
            "option_id": "measure_basis:grant_equivalent",
            "provenance": "explicit",
            "origin": {"kind": "choice", "choice_id": basis["id"]},
        }
    ]
    return block


def _counted_net() -> dict[str, Any]:
    """`_left_on_net`'s part with net disbursements chosen on the card: the spec_id
    `_NET`, as spec_ids are canonical over the identity, its basis the user's."""
    block = _left_on_net()
    scope = block["resolved"]["scope_choices"]
    (basis,) = [c for c in scope["choices"] if c["axis"] == "measure_basis"]
    basis["provenance"] = "explicit"
    scope["entries"] = [
        *scope["entries"],
        {
            "axis": "measure_basis",
            "option_id": "measure_basis:net",
            "provenance": "explicit",
            "origin": {"kind": "choice", "choice_id": basis["id"]},
        },
    ]
    return block


class _Resubmits:
    """A tool surface answering each resubmit of the DAC1 part by the basis its last
    entry counts: gross as `_GROSS`, grant equivalents as
    `_counted_on_grant_equivalents`, net as `_counted_net`. Every other tool is
    unreachable."""

    def __init__(self) -> None:
        self.posted: list[str] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        if tool.name != "resolve_definition":
            return ToolCallResult(is_error=True, structured_content={"code": "unreachable"})
        option = arguments["scope_entries"][-1]["option_id"]
        self.posted.append(option)
        block = {
            "measure_basis:gross": _GROSS,
            "measure_basis:grant_equivalent": _counted_on_grant_equivalents(),
            "measure_basis:net": _counted_net(),
        }[option]
        part = {"status": "definite", "spec": json.loads(json.dumps(block))}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


class _RevisingPlanner:
    """A `harness.graph.PlannerModel` whose revision runs `PlannerSeam.revise` over
    `surface` and records its arguments in `revisions` and the plan admission
    recorded in `admitted`, before translation answers."""

    def __init__(self, surface: ToolSurface) -> None:
        self._seam = PlannerSeam(settings=Settings(), surface=surface)
        self.revisions: list[dict[str, Any]] = []
        self.admitted: list[Plan] = []

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        self.revisions.append(kwargs)
        revised = await self._seam.revise(**kwargs)
        plan = load_plan(kwargs["ctx"].state)
        assert plan is not None
        self.admitted.append(plan)
        return revised


def _searched(block: Mapping[str, Any]) -> dict[str, Any]:
    """The turn's search reading of `block`, as `search` stores it."""
    return SearchReading.from_part(
        {
            "variable": "US total ODA",
            "status": "definite",
            "spec_id": block["spec_id"],
            "scope": block["resolved"]["scope_choices"],
            "asked_roles": ["donor"],
        }
    ).model_dump(mode="json")


async def _aligned_plan(decision: DecisionFactory) -> tuple[_FakeToolContext, Plan]:
    """The session of a turn that searched the gross and the net total and proposed
    the comparison, and the plan it holds."""
    ctx = _FakeToolContext()
    plan = await _gross_compared_with_net_average(ctx, decision)
    ctx.state[ORCHESTRATOR_SEARCHES_KEY] = [_searched(_GROSS), _searched(_left_on_net())]
    return ctx, plan


def _basis_row(plan: Plan, requirement_id: str) -> Decision:
    """The measure-basis decision `requirement_id` cites."""
    (row,) = [
        d
        for d in plan.decisions
        if d.slot in plan.requirement(requirement_id).decision_ids
        and d.scope is not None
        and d.scope.axis == "measure_basis"
    ]
    return row


async def _switch(
    ctx: _FakeToolContext, plan: Plan, row: Decision, option_id: str
) -> _RevisingPlanner:
    """Switch `row` of `plan` on the card to `option_id`, as the switch node does.
    Returns the planner the revision ran through."""
    surface = _Resubmits()
    planner = _RevisingPlanner(cast(ToolSurface, surface))
    # The revision's translation has no Search to discover through, so the switch
    # ends refused; what admission recorded is what these tests read.
    with suppress(SwitchFailedError):
        await switch_module._switched_plan(
            cast(Any, ctx),
            plan,
            choice_id=choice_id(plan, row),
            option_id=option_id,
            planner=cast(Any, planner),
            surface=cast(ToolSurface, surface),
        )
    return planner


@pytest.mark.parametrize("row", ["first", "second"], ids=["users-row", "aligned-row"])
async def test_a_switch_moves_only_the_requirements_its_row_is_for(
    decision: DecisionFactory, row: str
) -> None:
    """The total pins the gross source as the user's own, and the average was moved
    onto the same spec_id by alignment, each with its own basis row. A switch of
    either row to grant equivalents posts one resubmit and re-pins the requirement
    that row is for alone."""
    ctx, plan = await _aligned_plan(decision)

    planner = await _switch(ctx, plan, _basis_row(plan, row), "measure_basis:grant_equivalent")

    (revision,) = planner.revisions
    assert [r["id"] for r in revision["evidence_requirements"]] == [row]
    assert [r["source_spec_id"] for r in revision["evidence_requirements"]] == [_GRANT_EQUIVALENT]


async def test_a_switch_of_the_aligned_row_makes_the_input_the_users_own(
    decision: DecisionFactory,
) -> None:
    """The user switches the average's basis, which alignment moved to gross, to grant
    equivalents. The average becomes the user's own pin on grant equivalents, its
    basis row the user's, and the comparison keeps the contrast the user chose: the
    total stays on its gross pin under its own row."""
    ctx, plan = await _aligned_plan(decision)

    planner = await _switch(ctx, plan, _basis_row(plan, "second"), "measure_basis:grant_equivalent")

    (admitted,) = planner.admitted
    assert admitted.version > plan.version
    pins = {
        r.id: r.source_spec_id
        for r in admitted.evidence_requirements
        if isinstance(r, DatasetRequirement)
    }
    assert pins == {"first": _GROSS["spec_id"], "second": _GRANT_EQUIVALENT}
    rows = {
        rid: (_basis_row(admitted, rid).resolved_by, _basis_row(admitted, rid).chosen)
        for rid in ("first", "second")
    }
    assert {rid: resolved_by for rid, (resolved_by, _) in rows.items()} == {
        "first": ResolvedBy.USER,
        "second": ResolvedBy.USER,
    }
    assert rows["first"][1] != rows["second"][1]
    shown = {d.slot for d in shown_decisions(admitted)}
    assert {_basis_row(admitted, rid).slot for rid in ("first", "second")} <= shown


async def test_a_switch_of_the_aligned_row_back_to_its_own_basis_keeps_the_input_there(
    decision: DecisionFactory,
) -> None:
    """The user switches the average's basis, which alignment moved to gross, back to
    the net basis its own pin counts. The user's pick states net for the average, so
    alignment does not move it again: the average pins its own net source, its basis
    row the user's, and the comparison keeps the contrast the user chose."""
    ctx, plan = await _aligned_plan(decision)

    planner = await _switch(ctx, plan, _basis_row(plan, "second"), "measure_basis:net")

    (admitted,) = planner.admitted
    pins = {
        r.id: r.source_spec_id
        for r in admitted.evidence_requirements
        if isinstance(r, DatasetRequirement)
    }
    assert pins == {"first": _GROSS["spec_id"], "second": _NET}
    assert "second" not in alignments(ctx.state)
    row = _basis_row(admitted, "second")
    assert row.resolved_by is ResolvedBy.USER
    assert row.chosen != _basis_row(admitted, "first").chosen


@dataclass(frozen=True)
class _Recorded:
    """A definition a search in the conversation recorded (`planner.plan.RecordedDefinition`)."""

    identity: Mapping[str, Any]
    groups: tuple[ResolvedGroup, ...] = ()


async def test_a_switch_of_the_aligned_row_holds_when_another_input_pins_its_stated_source(
    decision: DecisionFactory,
) -> None:
    """The plan also reads the net average on its own, pinning the net source the
    average states directly, and both sources' identities are on record over 2023.
    The average's moved row still fixes the net pin the average states, and a switch
    of that row posts from the net reading and re-pins the average alone."""
    ctx = _FakeToolContext()
    save_pin_sources(
        ctx.state,
        {
            _GROSS["spec_id"]: PinSource(block=_GROSS, settled=True),
            _NET: PinSource(block=_left_on_net(), settled=True),
        },
    )
    question = (
        "How does US gross ODA in 2023 compare with its average, and what was the net average?"
    )
    ctx.state[ASKS_STATE_KEY] = [question]

    def dataset(requirement_id: str, pin: str) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure="US total ODA",
            entities=[Entity(name="United States of America", role="donor")],
            period=Period(start=2023, end=2023),
            source_spec_id=pin,
        )

    window = period_window(Period(start=2023, end=2023))
    with definitions_on_record(
        {
            spec_id: _Recorded({"requested_window": window, "spec_id": spec_id})
            for spec_id in (_GROSS["spec_id"], _NET)
        }
    ):
        result = await _tools_reading(cast(ToolSurface, _gross_compiles()))["propose_plan"].func(
            question=question,
            answer_mode=AnswerMode.COMPUTED,
            decisions=drafts([decision()]),
            answered=[AskAnswer(ask=0, outputs=(_COMPARE, "three"))],
            evidence_requirements=[
                dataset("first", _GROSS["spec_id"]),
                dataset("second", _NET),
                dataset("alone", _NET),
                _reduced("one", "first", Reduction.SUM),
                _reduced("two", "second", Reduction.MEAN),
                _reduced("three", "alone", Reduction.MEAN),
                _difference("one", "two"),
            ],
            tool_context=ctx,
        )
    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    row = _basis_row(plan, "second")
    assert row.fixes is not None
    assert row.fixes.source_spec_id == _NET
    ctx.state[ORCHESTRATOR_SEARCHES_KEY] = [_searched(_left_on_net())]

    planner = await _switch(ctx, plan, row, "measure_basis:grant_equivalent")

    (revision,) = planner.revisions
    assert [(r["id"], r["source_spec_id"]) for r in revision["evidence_requirements"]] == [
        ("second", _GRANT_EQUIVALENT)
    ]


_TOTAL_GROSS_INCLUSIVE = f"spec_{'8' * 32}"
_HEALTH_INCLUSIVE = f"spec_{'7' * 32}"


class _ProviderResubmits:
    """A tool surface answering each resubmit with the options it posts, each
    posted axis explicit, and recording each resubmit's spec_id and posted
    options in `posted`: of the US total, posting a provider scope and a basis,
    as `_TOTAL_GROSS_INCLUSIVE`; of US health aid, which Search counts gross,
    posting a provider scope, as `_HEALTH_INCLUSIVE`. A resubmit posting
    anything else is refused, as is every other tool."""

    def __init__(self) -> None:
        self.posted: list[tuple[str, dict[str, str]]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        if tool.name != "resolve_definition":
            return ToolCallResult(is_error=True, structured_content={"code": "unreachable"})
        posted = {e["axis"]: e["option_id"] for e in arguments["scope_entries"]}
        self.posted.append((arguments["spec_id"], posted))
        health = arguments["spec_id"].startswith(HEALTH)
        if "provider" not in posted or ("measure_basis" in posted) == health:
            return ToolCallResult(is_error=True, structured_content={"code": "unreachable"})
        block = provider_block(
            _HEALTH_INCLUSIVE if health else _TOTAL_GROSS_INCLUSIVE,
            narrow=health,
            active=posted["provider"],
            basis="measure_basis:gross" if health else posted["measure_basis"],
        )
        scope = block["resolved"]["scope_choices"]
        scope["id"] = scope["base"]["spec_id"] = arguments["spec_id"]
        scope["base"]["identity"] = arguments["identity"]
        scope["entries"] = arguments["scope_entries"]
        for choice in scope["choices"]:
            choice["id"] = f"{arguments['spec_id']}:{choice['axis']}"
            choice["provenance"] = "explicit"
        part = {"status": "definite", "spec": block}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


def _us_dataset(requirement_id: str, measure: str, pin: str) -> DatasetRequirement:
    """`measure` of the United States over 2015-2024, pinning `pin`."""
    return DatasetRequirement(
        id=requirement_id,
        decision_ids=["measure"],
        measure=measure,
        entities=[Entity(name="United States", role="donor")],
        period=Period(start=2015, end=2024),
        source_spec_id=pin,
    )


async def _bilateral_share(ctx: _FakeToolContext, decision: DecisionFactory) -> Plan:
    """Propose US health aid, which the user asked for bilateral, as a share of the
    US total, which Search counts in grant equivalents with imputed multilateral
    flows, and return the plan. Alignment compiles the total on gross and
    bilateral in one resubmit (`TOTAL_GROSS_BILATERAL`). The turn's searches are
    the two shown sources."""
    health = provider_block(HEALTH, narrow=True, active="provider:bilateral", provenance="explicit")
    total = provider_block(
        TOTAL, narrow=False, active="provider:inclusive", basis="measure_basis:grant_equivalent"
    )
    _show_blocks(ctx, health, total)
    question = "What share of US bilateral aid goes to health?"
    ctx.state[ASKS_STATE_KEY] = [question]
    result = await _tools_reading(cast(ToolSurface, ProviderCompiles()))["propose_plan"].func(
        question=question,
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        answered=[AskAnswer(ask=0, outputs=("share",))],
        evidence_requirements=[
            _us_dataset("health", "US aid to health", HEALTH),
            _us_dataset("total", "US total ODA", TOTAL),
            ShareOfRequirement(
                id="share",
                decision_ids=["measure"],
                numerator=RequirementSide(input_id="health"),
                denominator=RequirementSide(input_id="total"),
            ),
        ],
        tool_context=ctx,
    )
    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    assert set(alignments(ctx.state)) == {"total"}
    ctx.state[ORCHESTRATOR_SEARCHES_KEY] = [_searched(health), _searched(total)]
    return plan


def _provider_row(plan: Plan, requirement_id: str) -> Decision:
    """The provider decision `requirement_id` cites."""
    (row,) = [
        d
        for d in plan.decisions
        if d.slot in plan.requirement(requirement_id).decision_ids
        and d.scope is not None
        and d.scope.axis == "provider"
    ]
    return row


async def _switch_provider(
    ctx: _FakeToolContext, plan: Plan, row: Decision, surface: _ProviderResubmits
) -> _RevisingPlanner:
    """Switch `row` of `plan` on the card to imputed multilateral flows, Search
    answering through `surface`. Returns the planner the revision ran through."""
    planner = _RevisingPlanner(cast(ToolSurface, surface))
    with suppress(SwitchFailedError):
        await switch_module._switched_plan(
            cast(Any, ctx),
            plan,
            choice_id=choice_id(plan, row),
            option_id="provider:inclusive",
            planner=cast(Any, planner),
            surface=cast(ToolSurface, surface),
        )
    return planner


async def test_a_switch_of_one_axis_a_compile_moved_leaves_the_other_the_planners(
    decision: DecisionFactory,
) -> None:
    """The user switches the total's provider row, which alignment moved to
    bilateral, back to imputed multilateral flows: the total's provider is now the
    user's, and its basis, which the user never stated, stays the planner's."""
    ctx = _FakeToolContext()
    plan = await _bilateral_share(ctx, decision)

    planner = await _switch_provider(ctx, plan, _provider_row(plan, "total"), _ProviderResubmits())

    (admitted,) = planner.admitted
    assert admitted.requirement("total").source_spec_id == _TOTAL_GROSS_INCLUSIVE
    by_axis = {
        d.scope.axis: d.resolved_by
        for d in admitted.decisions
        if d.slot in admitted.requirement("total").decision_ids and d.scope is not None
    }
    assert by_axis == {
        "measure_basis": ResolvedBy.PLANNER_DEFAULT,
        "provider": ResolvedBy.USER,
    }


class _AcceptingPlanner:
    """A `harness.graph.PlannerModel` whose revision records its arguments in
    `revisions` and offers the plan it was given back, so the switch completes
    and stores what it switched."""

    def __init__(self) -> None:
        self.revisions: list[dict[str, Any]] = []

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        self.revisions.append(kwargs)
        return {"plan": kwargs["prior_plan"], "summary": "revised"}


async def _pinning_the_compile(
    ctx: _FakeToolContext, decision: DecisionFactory, *measures: str
) -> Plan:
    """`_bilateral_share` amended with health's share of each of `measures`, a
    requirement `bilateral<i>` pinning `TOTAL_GROSS_BILATERAL`, the compile
    alignment made for the share's total, which the planner was never shown.
    Returns the plan on record."""
    plan = await _bilateral_share(ctx, decision)
    ids = [f"bilateral{index}" if index else "bilateral" for index in range(len(measures))]
    amended = await _tools_reading(cast(ToolSurface, ProviderCompiles()))["propose_plan"].func(
        answered=[AskAnswer(ask=0, outputs=("share", *(f"of_{rid}" for rid in ids)))],
        evidence_requirements=[
            requirement
            for rid, measure in zip(ids, measures, strict=True)
            for requirement in (
                _us_dataset(rid, measure, TOTAL_GROSS_BILATERAL),
                ShareOfRequirement(
                    id=f"of_{rid}",
                    decision_ids=["measure"],
                    numerator=RequirementSide(input_id="health"),
                    denominator=RequirementSide(input_id=rid),
                ),
            )
        ],
        amends_version=plan.version,
        tool_context=ctx,
    )
    assert amended["outcome"] == Outcome.OK.value, amended
    plan = load_plan(ctx.state)
    assert plan is not None
    return plan


@pytest.mark.parametrize(
    ("measures", "stored"),
    [
        (("US bilateral ODA", "US bilateral ODA"), "US bilateral ODA"),
        (("US bilateral ODA", "US gross bilateral aid"), None),
    ],
    ids=["same-words", "different-words"],
)
async def test_a_compile_two_requirements_pin_switches_only_under_words_they_share(
    decision: DecisionFactory, measures: tuple[str, str], stored: str | None
) -> None:
    """Two requirements pin the compile directly, and the switch re-pins both and
    stores one reading of the switched dataset. Worded alike, it is stored under
    their words. Worded apart, the switch is refused before any resubmit: one
    reading would name one of them wrongly."""
    ctx = _FakeToolContext()
    plan = await _pinning_the_compile(ctx, decision, *measures)
    surface = _ProviderResubmits()
    planner = _AcceptingPlanner()
    switching = switch_module._switched_plan(
        cast(Any, ctx),
        plan,
        choice_id=choice_id(plan, _provider_row(plan, "bilateral")),
        option_id="provider:inclusive",
        planner=cast(Any, planner),
        surface=cast(ToolSurface, surface),
    )

    if stored is None:
        with pytest.raises(SwitchFailedError) as refused:
            await switching
        assert refused.value.code == switch_module.SCOPE_CHOICE_NOT_APPLIED
        assert (surface.posted, planner.revisions) == ([], [])
        return
    await switching
    texts = [
        r["text"]
        for r in ctx.state[ORCHESTRATOR_SEARCHES_KEY]
        if r["spec_id"] == _TOTAL_GROSS_INCLUSIVE
    ]
    assert texts == [stored]


@pytest.mark.parametrize("unread", [(), (TOTAL,), (TOTAL, HEALTH)], ids=["read", "total", "both"])
async def test_a_switch_of_a_row_of_a_compile_pinned_directly_posts_from_the_compile(
    decision: DecisionFactory, unread: tuple[str, ...]
) -> None:
    """An amendment adds health's share of the gross bilateral total, pinning the compile
    alignment made for the share's total, which the planner was never shown. The card
    lists that requirement's provider row, read from the compile's block, with
    health's. A switch of the row posts the compile's scope, whether or not the total
    it was compiled from has a stored reading, and health's from the block it was
    shown with when it has none, re-pins each requirement onto its own
    resubmit, and holds the switched compile under the words and parties of the
    requirement pinning it, with none of the total's provenance."""
    ctx = _FakeToolContext()
    plan = await _pinning_the_compile(ctx, decision, "US bilateral ODA")
    shown = pin_sources(ctx.state)
    shown[TOTAL] = shown[TOTAL].model_copy(update={"provenance_ref": "oecd-dac1"})
    save_pin_sources(ctx.state, shown)
    ctx.state[ORCHESTRATOR_SEARCHES_KEY] = [
        r for r in ctx.state[ORCHESTRATOR_SEARCHES_KEY] if r["spec_id"] not in unread
    ]
    surface = _ProviderResubmits()
    planner = _AcceptingPlanner()

    await switch_module._switched_plan(
        cast(Any, ctx),
        plan,
        choice_id=choice_id(plan, _provider_row(plan, "bilateral")),
        option_id="provider:inclusive",
        planner=cast(Any, planner),
        surface=cast(ToolSurface, surface),
    )

    assert surface.posted == [
        (
            f"{HEALTH}_base",
            {
                "sector": "sector:120",
                "region": "region:members_only",
                "provider": "provider:inclusive",
            },
        ),
        (
            f"{TOTAL}_base",
            {"measure_basis": "measure_basis:gross", "provider": "provider:inclusive"},
        ),
    ]
    (revision,) = planner.revisions
    assert {r["id"]: r["source_spec_id"] for r in revision["evidence_requirements"]} == {
        "health": _HEALTH_INCLUSIVE,
        "bilateral": _TOTAL_GROSS_INCLUSIVE,
    }
    (stored,) = [
        SearchReading.model_validate(r)
        for r in ctx.state[ORCHESTRATOR_SEARCHES_KEY]
        if r["spec_id"] == _TOTAL_GROSS_INCLUSIVE
    ]
    assert (stored.text, stored.asked_roles, stored.provenance_ref) == (
        "US bilateral ODA",
        ("donor",),
        None,
    )
    held = pin_sources(ctx.state)[_TOTAL_GROSS_INCLUSIVE]
    assert (held.provenance_ref, held.asked_roles) == (None, ("donor",))


class _Resolves:
    """A tool surface answering each resubmit as Search does, recording each call's
    arguments in `calls`; every other call is refused. The answer is the block, among
    `blocks`, of the dataset whose base it posts from, with each posted option active
    and explicit. Its identity is the base identity posted, its window kept, with a
    slot binding each posted option, and its spec_id is the one that identity hashes
    to, so its block puts it on record (`planner.plan.shown_identity`)."""

    def __init__(self, *blocks: Mapping[str, Any]) -> None:
        self.blocks = {b["resolved"]["scope_choices"]["base"]["spec_id"]: b for b in blocks}
        self.calls: list[Mapping[str, Any]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        if tool.name != "resolve_definition" or "scope_entries" not in arguments:
            return ToolCallResult(is_error=True, structured_content={"code": "unreachable"})
        self.calls.append(arguments)
        posted = {e["axis"]: e["option_id"] for e in arguments["scope_entries"]}
        identity = json.loads(json.dumps(arguments["identity"]))
        identity["slots"] = [
            *identity["slots"],
            *(
                {"property": f"scope_{axis}", "binding": {"state": "value", "ref": option}}
                for axis, option in sorted(posted.items())
            ),
        ]
        spec_id = spec_id_for(canonicalize(identity))
        block = json.loads(json.dumps(self.blocks[arguments["spec_id"]]))
        block["spec_id"] = spec_id
        block["definition"] = {**block["definition"], "identity": identity}
        scope = block["resolved"]["scope_choices"]
        scope["base"]["identity"] = arguments["identity"]
        scope["compiled"] = {**scope["compiled"], "spec_id": spec_id, "identity": identity}
        scope["entries"] = arguments["scope_entries"]
        for choice in scope["choices"]:
            if choice["axis"] in posted:
                choice["active"], choice["provenance"] = posted[choice["axis"]], "explicit"
        part = {"status": "definite", "spec": block}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


@dataclass(frozen=True)
class _ClosedShare:
    """A share over one year whose inputs Search found over another window, and
    the session that proposed it: `aligned` is the input admission compiled, through
    `surface`, from the base identity Search gave."""

    ctx: _FakeToolContext
    surface: _Resolves | WholeCompiles
    recorded: dict[str, _Recorded]
    period: Period
    aligned: str
    requirements: tuple[EvidenceRequirement, ...]
    question: str


def _over_latest(*blocks: dict[str, Any]) -> dict[str, _Recorded]:
    """Set each of `blocks`' base identity over the latest year, as Search answers a
    question naming none, and return each base identity as the search recorded it."""
    for block in blocks:
        block["resolved"]["scope_choices"]["base"]["identity"]["requested_window"] = {
            "kind": "latest"
        }
    return {
        block["spec_id"]: _Recorded(block["resolved"]["scope_choices"]["base"]["identity"])
        for block in blocks
    }


def _closed_share(kind: str) -> _ClosedShare:
    """The closed share `kind` names. `provider-and-basis`: US bilateral health aid
    over the US total, inclusive and in grant equivalents, both over the latest year
    and read over 2024; alignment compiles the total gross and bilateral in one
    resubmit. `basis`: US health aid counted gross over the total in grant
    equivalents, compiled gross. `whole`: US health aid Search found over 2023, read
    over 2022, as a share of its whole over every sector, compiled from it
    (`planner.agent.compile_whole`)."""
    ctx = _FakeToolContext()
    if kind == "whole":
        part = us_2023_block("health")
        _show_blocks(ctx, part)
        period = Period(start=2022, end=2022)
        health = DatasetRequirement(
            id="health",
            decision_ids=["measure"],
            measure="US aid to health",
            entities=[Entity(name="United States", role="donor")],
            period=period,
            source_spec_id=part["spec_id"],
        )
        whole = health.model_copy(
            update={
                "id": "whole",
                "measure": "US total ODA",
                "source_spec_id": None,
                "whole_of": WholeOf(part="health", axis="sector"),
            }
        )
        return _ClosedShare(
            ctx=ctx,
            surface=WholeCompiles(),
            recorded={part["spec_id"]: _Recorded(part["definition"]["identity"])},
            period=period,
            aligned="whole",
            requirements=(health, whole, _share_of("whole")),
            question="What share of US aid went to health in 2022?",
        )
    if kind == "basis":
        health = basis_block(HEALTH, narrow=True, active="measure_basis:gross")
        total = basis_block(TOTAL, narrow=False, active="measure_basis:grant_equivalent")
    else:
        health = provider_block(
            HEALTH, narrow=True, active="provider:bilateral", provenance="explicit"
        )
        total = provider_block(
            TOTAL, narrow=False, active="provider:inclusive", basis="measure_basis:grant_equivalent"
        )
    recorded = _over_latest(health, total)
    _show_blocks(ctx, health, total)
    period = Period(start=2024, end=2024)

    def dataset(requirement_id: str, measure: str, pin: str) -> DatasetRequirement:
        return _us_dataset(requirement_id, measure, pin).model_copy(update={"period": period})

    return _ClosedShare(
        ctx=ctx,
        surface=_Resolves(health, total),
        recorded=recorded,
        period=period,
        aligned="total",
        requirements=(
            dataset("health", "US aid to health", HEALTH),
            dataset("total", "US total ODA", TOTAL),
            _share_of("total"),
        ),
        question="What share of US bilateral aid goes to health in 2024?",
    )


def _share_of(denominator: str) -> ShareOfRequirement:
    """Health's share of `denominator`."""
    return ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="health"),
        denominator=RequirementSide(input_id=denominator),
    )


async def _propose(
    share: _ClosedShare, decision: DecisionFactory, *, amends: Plan | None = None
) -> Plan:
    """Propose `share`'s requirements, as an amendment of `amends` when given, with
    its searches' definitions on record, and return the plan admission recorded."""
    share.ctx.state[ASKS_STATE_KEY] = [share.question]
    propose = _tools_reading(cast(ToolSurface, share.surface))["propose_plan"].func
    stated: dict[str, Any] = (
        {"amends_version": amends.version}
        if amends is not None
        else {
            "question": share.question,
            "answer_mode": AnswerMode.COMPUTED,
            "decisions": drafts([decision()]),
        }
    )
    with definitions_on_record(share.recorded):
        result = await propose(
            **stated,
            answered=[AskAnswer(ask=0, outputs=("share",))],
            evidence_requirements=list(share.requirements),
            tool_context=share.ctx,
        )
    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(share.ctx.state)
    assert plan is not None
    return plan


def _dataset_pins(plan: Plan) -> dict[str, str | None]:
    """Each dataset requirement of `plan` with its pin."""
    return {
        r.id: r.source_spec_id
        for r in plan.evidence_requirements
        if isinstance(r, DatasetRequirement)
    }


@pytest.mark.parametrize("kind", ["provider-and-basis", "basis", "whole"])
async def test_an_aligned_input_keeps_the_requirements_period_when_the_search_was_over_another(
    decision: DecisionFactory, kind: str
) -> None:
    """Search found both inputs of a share over another window than the one the
    share reads. Alignment compiles one input from the base identity Search gave,
    whose window is Search's; the compile is held in the session and in no
    search record. Every dataset the admitted plan pins is moved to a copy
    over the share's period (`submission.service._window_pins`), and the compiled
    input's copy is the compile's identity, its bindings kept, over that period."""
    share = _closed_share(kind)

    plan = await _propose(share, decision)

    assert set(alignments(share.ctx.state)) == {share.aligned}
    (compiled,) = share.surface.calls
    window = period_window(share.period)
    assert compiled["identity"]["requested_window"] != window
    copies = windowed_pins(share.ctx.state)
    pins = _dataset_pins(plan)
    windows = {
        rid: copies[pin].identity["requested_window"] if pin in copies else None
        for rid, pin in pins.items()
    }
    assert windows == dict.fromkeys(pins, window)
    (held,) = aligned_pins(share.ctx.state)
    aligned_copy = copies[pins[share.aligned] or ""]
    assert aligned_copy.windowed_from == held.spec_id
    assert aligned_copy.identity == {
        **held.block["definition"]["identity"],
        "requested_window": window,
    }


async def test_a_restated_closed_share_keeps_its_compile_and_its_copies(
    decision: DecisionFactory,
) -> None:
    """The planner restates the share the first proposal admitted, pinning the
    datasets it was shown. The plan as the planner states it reads those pins;
    admission takes the same compile with no second resubmit, pins the same copies
    over 2024, and records the same copies and alignment."""
    share = _closed_share("provider-and-basis")
    first = await _propose(share, decision)
    copies = windowed_pins(share.ctx.state)
    aligned = alignments(share.ctx.state)
    assert _dataset_pins(first)["total"] in copies

    stated = statement(
        first,
        to_origin={copy.spec_id: copy.windowed_from for copy in copies.values()},
        alignments=aligned,
    )
    again = await _propose(share, decision, amends=first)

    assert _dataset_pins(stated) == {"health": HEALTH, "total": TOTAL}
    assert len(share.surface.calls) == 1
    assert again.version > first.version
    assert _dataset_pins(again) == _dataset_pins(first)
    assert windowed_pins(share.ctx.state) == copies
    assert alignments(share.ctx.state) == aligned


async def test_a_switch_on_a_closed_share_moves_both_inputs_to_copies_over_its_period(
    decision: DecisionFactory,
) -> None:
    """The user switches health's provider row of the 2024 share to imputed
    multilateral flows. The switch resubmits health from the dataset Search found
    over the latest year, and the revision compiles the total gross again from the
    total Search found, so neither has a search record. Both requirements the
    revision admits read copies over 2024."""
    share = _closed_share("provider-and-basis")
    plan = await _propose(share, decision)
    shown = pin_sources(share.ctx.state)
    share.ctx.state[ORCHESTRATOR_SEARCHES_KEY] = [
        _searched(shown[HEALTH].block),
        _searched(shown[TOTAL].block),
    ]
    planner = _RevisingPlanner(cast(ToolSurface, share.surface))

    with definitions_on_record(share.recorded), suppress(SwitchFailedError):
        await switch_module._switched_plan(
            cast(Any, share.ctx),
            plan,
            choice_id=choice_id(plan, _provider_row(plan, "health")),
            option_id="provider:inclusive",
            planner=cast(Any, planner),
            surface=cast(ToolSurface, share.surface),
        )

    switched, recompiled = share.surface.calls[1:]
    assert (switched["spec_id"], recompiled["spec_id"]) == (f"{HEALTH}_base", f"{TOTAL}_base")
    (admitted,) = planner.admitted
    copies = windowed_pins(share.ctx.state)
    pins = _dataset_pins(admitted)
    window = period_window(share.period)
    assert {
        rid: copies[pin].identity["requested_window"] if pin in copies else None
        for rid, pin in pins.items()
    } == dict.fromkeys(pins, window)


@pytest.mark.parametrize("padding", [0, MAX_IDENTITY_BYTES], ids=["under-cap", "past-cap"])
def test_a_shown_block_puts_its_identity_on_record_only_within_the_size_cap(
    padding: int,
) -> None:
    """A shown block whose identity hashes to its spec_id puts that identity on
    record, so a compile held only as a block is windowed. Past `MAX_IDENTITY_BYTES`
    as compact JSON it puts none, since data-mcp's `resolve_definition` would refuse
    a copy of it (`caller_definition`)."""
    block = json.loads(json.dumps(us_2023_block("health")))
    identity = block["definition"]["identity"]
    if padding:
        identity["padding"] = "x" * padding
    spec_id = spec_id_for(canonicalize(identity))
    block["spec_id"] = spec_id

    recorded = identity_on_record(spec_id, shown={spec_id: PinSource(block=block, settled=True)})

    assert (recorded.identity if recorded is not None else None) == (None if padding else identity)


# Dev Search's answers for the share of Kenya's ODA that goes to health, by name:
# the CRS health part counted gross, the DAC2a total counted net, both over the
# latest year, and the total compiled gross (the fixture's `comment` key says where
# the data came from).
_KENYA = json.loads(
    (Path(__file__).parent / "fixtures" / "share_kenya_health_2024.json").read_text()
)

# The plan for that question as `propose_plan` receives it: health aid summed over
# providers as a share of the net total, both over 2024.
_KENYA_PLAN: dict[str, Any] = {
    "question": (
        "What share of the ODA Kenya receives goes to health in the latest year with data "
        "(2024)? ODA to health from OECD CRS and total ODA from OECD DAC2a or CRS."
    ),
    "across_years": [],
    "amends_version": 0,
    "answered": [{"ask": 0, "outputs": ["share_health"]}],
    "evidence_requirements": [
        {
            "kind": "dataset",
            "source_spec_id": _KENYA["health"]["spec"]["spec_id"],
            "coverage": {"on_missing": "report_gaps"},
            "measure": "Gross ODA disbursements to Kenya for health",
            "id": "health_oda",
            "frequency": "annual",
            "entities": [{"role": "recipient", "name": "Kenya"}],
            "period": {"end": 2024, "start": 2024},
        },
        {
            "group_by": ["entity", "period"],
            "decision_ids": ["sum_health_providers"],
            "input_id": "health_oda",
            "kind": "aggregate",
            "id": "aggregate_health",
            "reduction": "sum",
        },
        {
            "source_spec_id": _KENYA["total"]["spec"]["spec_id"],
            "entities": [{"role": "recipient", "name": "Kenya"}],
            "period": {"start": 2024, "end": 2024},
            "id": "total_oda",
            "kind": "dataset",
            "measure": "Net disbursements to Kenya for official development assistance",
            "coverage": {"on_missing": "report_gaps"},
            "frequency": "annual",
        },
        {
            "kind": "share_of",
            "decision_ids": [],
            "denominator": {"entity": "Kenya", "input_id": "total_oda"},
            "numerator": {"input_id": "aggregate_health", "entity": "Kenya"},
            "accept_provider_difference": True,
            "id": "share_health",
        },
    ],
    "decisions": [
        {
            "slot": "sum_health_providers",
            "chosen": "aggregate_health_across_providers",
            "description": "Sum health ODA across providers for Kenya",
            "fixes": ["reduction", "group_by"],
            "rationale": (
                "Kenya receives health aid from multiple bilateral providers, which must be "
                "summed across providers to compute the national total health aid."
            ),
            "options": ["aggregate_health_across_providers", "keep_providers_separate"],
            "resolved_by": "planner_default",
        }
    ],
    "answer_mode": "computed",
}


class _KenyaCompiles:
    """A tool surface answering the resubmit of `_KENYA`'s total counted gross with
    the compile dev Search gave, recording each call's arguments in `calls`. The
    answer is looked up under the spec_id the posted identity hashes to; every
    other call is refused."""

    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        self.calls.append(arguments)
        posted = (
            spec_id_for(canonicalize(arguments["identity"])),
            tuple(e["option_id"] for e in arguments.get("scope_entries") or ()),
        )
        compiled = {(_KENYA["total"]["spec"]["spec_id"], ("measure_basis:gross",)): "total_gross"}
        if tool.name != "resolve_definition" or posted not in compiled:
            return ToolCallResult(is_error=True, structured_content={"code": "unreachable"})
        part = json.loads(json.dumps(_KENYA[compiled[posted]]))
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


async def test_the_kenya_share_reads_its_compiled_total_over_the_year_it_asks_for() -> None:
    """Dev Search found Kenya's health aid counted gross and its ODA total counted
    net, both over the latest year, and the planner proposed health's share of the
    total over 2024. Admission compiles the total gross, and Search's compile keeps
    the latest-year window. The admitted plan pins the compile's copy over 2024, and
    no dataset it pins is refused by the compiler for a window short of the year."""
    blocks = {name: _KENYA[name]["spec"] for name in ("health", "total")}
    ctx = _FakeToolContext()
    ctx.state[ASKS_STATE_KEY] = ["Share of ODA received by Kenya that goes to health"]
    ctx.state[ASK_BREAKDOWNS_STATE_KEY] = [
        AskBreakdown(ask=0, over=("donor",)).model_dump(mode="json")
    ]
    _show_blocks(ctx, *blocks.values())
    recorded = {
        block["spec_id"]: _Recorded(
            block["definition"]["identity"], source_from_spec_block(block).groups
        )
        for block in blocks.values()
    }
    surface = _KenyaCompiles()
    propose = _tools_reading(cast(ToolSurface, surface))["propose_plan"].func

    with definitions_on_record(recorded):
        result = await propose(**_KENYA_PLAN, tool_context=ctx)

    assert result["outcome"] == Outcome.OK.value, result
    (compiled,) = surface.calls
    assert compiled["spec_id"] == blocks["total"]["spec_id"]
    plan = load_plan(ctx.state)
    assert plan is not None
    gross = _KENYA["total_gross"]["spec"]
    assert gross["definition"]["identity"]["requested_window"] == {"kind": "latest"}
    window = period_window(Period(start=2024, end=2024))
    # The copy is spec_c42ddfe77c8918ecf475449ebac6d270, which dev Search answers
    # `definite` when it is posted on its own.
    copy, _ = windowed_identity(gross["definition"]["identity"], window)
    pins = _dataset_pins(plan)
    assert pins["total_oda"] == copy
    copies = windowed_pins(ctx.state)
    assert copies[copy].windowed_from == gross["spec_id"]
    identities = {
        **{
            block["spec_id"]: block["definition"]["identity"] for block in (*blocks.values(), gross)
        },
        **{pin: held.identity for pin, held in copies.items()},
    }
    issues = [
        issue
        for requirement in plan.evidence_requirements
        if isinstance(requirement, DatasetRequirement) and requirement.source_spec_id
        for issue in check_evidence(
            requirement,
            RetrievalEvidence(
                origin="discovery",
                spec_id=requirement.source_spec_id,
                has_data=True,
                identity=dict(identities[requirement.source_spec_id]),
            ),
        )
    ]
    assert EvidenceIssueCode.SCOPE_WINDOW_SHORT not in {issue.code for issue in issues}
