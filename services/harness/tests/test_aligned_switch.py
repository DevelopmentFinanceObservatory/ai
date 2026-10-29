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
from typing import Any, cast

import pytest
from compiler.surface import ToolCallResult, ToolSurface
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from harness.graph import switch as switch_module
from harness.graph.state import ORCHESTRATOR_SEARCHES_KEY
from harness.graph.switch import SwitchFailedError
from harness.plan_display import choice_id, shown_decisions
from harness.planner_seam import PlannerSeam
from harness.settings import Settings
from harness.turn_records import SearchReading
from planner.plan import (
    ASKS_STATE_KEY,
    AnswerMode,
    AskAnswer,
    DatasetRequirement,
    Decision,
    Entity,
    Period,
    PinSource,
    Plan,
    Reduction,
    RequirementSide,
    ResolvedBy,
    ShareOfRequirement,
    alignments,
    definitions_on_record,
    load_plan,
    period_window,
    save_pin_sources,
)
from planner.tool_results import Outcome
from planner.workflow import ResolvedGroup, ToolId

from .basis_alignment import (
    HEALTH,
    TOTAL,
    TOTAL_GROSS_BILATERAL,
    ProviderCompiles,
    provider_block,
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
    posted axis explicit, an axis it does not post counted as Search's default: of
    the US total as `_TOTAL_GROSS_INCLUSIVE`, of US health aid as
    `_HEALTH_INCLUSIVE`. Every other tool is unreachable."""

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        if tool.name != "resolve_definition":
            return ToolCallResult(is_error=True, structured_content={"code": "unreachable"})
        posted = {e["axis"]: e["option_id"] for e in arguments["scope_entries"]}
        health = arguments["spec_id"].startswith(HEALTH)
        block = provider_block(
            _HEALTH_INCLUSIVE if health else _TOTAL_GROSS_INCLUSIVE,
            narrow=health,
            active=posted.get("provider", "provider:inclusive"),
            basis=posted.get("measure_basis", "measure_basis:gross"),
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


async def _switch_provider(ctx: _FakeToolContext, plan: Plan, row: Decision) -> _RevisingPlanner:
    """Switch `row` of `plan` on the card to imputed multilateral flows, Search
    answering as `_ProviderResubmits`. Returns the planner the revision ran through."""
    surface = _ProviderResubmits()
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

    planner = await _switch_provider(ctx, plan, _provider_row(plan, "total"))

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


async def test_a_switch_of_a_row_of_a_compile_pinned_directly_posts_from_the_compile(
    decision: DecisionFactory,
) -> None:
    """An amendment adds health's share of the gross bilateral total, pinning the compile
    alignment made for the share's total, which the planner was never shown. The card
    lists that requirement's provider row, read from the compile's block, with
    health's. A switch of the row posts the compile's scope and re-pins each
    requirement onto its own resubmit."""
    ctx = _FakeToolContext()
    plan = await _bilateral_share(ctx, decision)
    amended = await _tools_reading(cast(ToolSurface, ProviderCompiles()))["propose_plan"].func(
        answered=[AskAnswer(ask=0, outputs=("share", "of_bilateral"))],
        evidence_requirements=[
            _us_dataset("bilateral", "US bilateral ODA", TOTAL_GROSS_BILATERAL),
            ShareOfRequirement(
                id="of_bilateral",
                decision_ids=["measure"],
                numerator=RequirementSide(input_id="health"),
                denominator=RequirementSide(input_id="bilateral"),
            ),
        ],
        amends_version=plan.version,
        tool_context=ctx,
    )
    assert amended["outcome"] == Outcome.OK.value, amended
    plan = load_plan(ctx.state)
    assert plan is not None

    planner = await _switch_provider(ctx, plan, _provider_row(plan, "bilateral"))

    (revision,) = planner.revisions
    assert {r["id"]: r["source_spec_id"] for r in revision["evidence_requirements"]} == {
        "health": _HEALTH_INCLUSIVE,
        "bilateral": _TOTAL_GROSS_INCLUSIVE,
    }
