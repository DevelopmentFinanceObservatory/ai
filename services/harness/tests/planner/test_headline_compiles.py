"""Two calculations over provider and basis rows, each proposed on Search's answers,
discovered with no fresh search and compiled against the MCP servers' schemas: the
share of US ODA going to multilaterals, its sides kept on their own provider rows,
and a gross total compared with a net average, the net side aligned on gross."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from analysis_mcp.server import mcp as analysis_mcp
from compiler.compile import compile_plan
from compiler.sources import source_from_spec_block
from compiler.surface import McpSurface, ToolCallResult, ToolSurface
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.server import mcp as data_mcp
from harness.plan_display import choice_id, shown_decisions
from harness.scope_choices import planner_axes
from planner.plan import (
    ASKS_STATE_KEY,
    AnswerMode,
    AskAnswer,
    DatasetRequirement,
    Entity,
    Period,
    PinSource,
    Plan,
    Reduction,
    ResolvedBy,
    load_plan,
    pin_sources,
    save_pin_sources,
)
from planner.tool_results import Outcome
from planner.workflow import ToolId, Workflow
from translator import discovery
from translator.discovery import BLOCKING_WARNING_CODES, build_discovery_tools
from translator.mechanical import build_translation

from ..basis_alignment import us_2023_block
from ..decision_drafts import drafts
from .test_agent import (
    _COMPARE,
    _SHARE,
    _SUM_BLOCKS,
    _US_MULTILATERAL,
    _US_TOTAL,
    DecisionFactory,
    _dac_countries,
    _difference,
    _FakeToolContext,
    _pins,
    _reduced,
    _SearchRefused,
    _share,
    _show_blocks,
    _tools_reading,
)


async def _compiled(ctx: _FakeToolContext) -> Workflow:
    """The plan `ctx` holds, each input discovered from its pin with no call to Search,
    translated and compiled against data-mcp's and analysis-mcp's schemas."""
    plan = load_plan(ctx.state)
    assert plan is not None
    refused = _SearchRefused()
    discover = build_discovery_tools(surface=cast(ToolSurface, refused))
    (discover_source,) = [t for t in discover if t.name == "discover_source"]
    pins = dict(_pins(ctx))
    for requirement_id in pins:
        found = await discover_source.func(requirement_id=requirement_id, tool_context=ctx)
        assert found["status"] == "definite", (requirement_id, found)
    assert refused.calls == []
    shown = pin_sources(ctx.state)
    discovered = {
        rid: source_from_spec_block(cast(Any, shown[spec_id].spec_block(spec_id)))
        for rid, spec_id in pins.items()
        if spec_id is not None
    }
    tools = await McpSurface(
        sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp}
    ).list_tools()
    compiled = compile_plan(
        plan=plan,
        translation=build_translation(plan, {rid: s.spec_id for rid, s in discovered.items()}),
        schemas={tool.name: tool for tool in tools},
        discovered=discovered,
    )
    assert compiled.workflow is not None, compiled.issues
    return compiled.workflow


def _retrievals(workflow: Workflow) -> dict[str, str]:
    """The spec_id each requirement's retrieval reads, by requirement id."""
    steps = {step.id: step for step in workflow.steps}
    read = {}
    for binding in workflow.evidence_bindings:
        step = steps[binding.step_id]
        if step.tool.name == "get_data":
            assert step.args["spec_id"] == binding.spec_id, step
            read[binding.requirement_id] = binding.spec_id
    return read


def _tool_names(workflow: Workflow) -> list[str]:
    return [f"{step.tool.server}.{step.tool.name}" for step in workflow.steps]


async def test_the_share_of_us_oda_going_to_multilaterals_compiles_on_each_rows_own_scope(
    decision: DecisionFactory,
) -> None:
    """ "What share of US ODA goes to multilaterals": DAC1 multilateral ODA, read in
    the searched words, over the DAC1 total Search counts by default, both on one
    part, as dfo-apis's offline engine answered them. Counting the total on the
    multilateral row would divide one series by itself, so each side keeps its own
    provider row with nothing compiled, and the plan compiles to two retrievals,
    each on its own pin, and a share of one over the other."""
    blocks = {"first": _US_MULTILATERAL, "second": _US_TOTAL}
    ctx = _FakeToolContext()
    _show_blocks(ctx, *blocks.values())

    result = await _tools_reading(cast(ToolSurface, _SearchRefused()))["propose_plan"].func(
        question="What share of US ODA goes to multilaterals?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            *(_dac_countries(rid, block) for rid, block in blocks.items()),
            _share("first", "second"),
        ],
        tool_context=ctx,
    )
    assert result["outcome"] == Outcome.OK.value, result

    workflow = await _compiled(ctx)

    multilateral, total = (block["spec_id"] for block in blocks.values())
    assert multilateral != total
    assert multilateral == _SUM_BLOCKS["dac_countries_multilateral"]["spec_id"]
    assert _retrievals(workflow) == {"first": multilateral, "second": total}
    assert _tool_names(workflow) == [
        "data-mcp.get_data",
        "data-mcp.get_data",
        "analysis-mcp.share_of",
    ]
    assert workflow.terminal_steps == (_SHARE,)


def _stated_gross() -> dict[str, Any]:
    """US DAC1 total ODA in 2023 counted gross, as dev Search answered it, its gross
    entry read from the asked words ("gross ODA") in place of the choice it was
    recorded from."""
    block = json.loads(json.dumps(us_2023_block("dac1_total_gross")))
    (entry,) = block["resolved"]["scope_choices"]["entries"]
    entry["origin"] = {"kind": "text", "text": "gross", "start": 3, "end": 8}
    return block


_GROSS = _stated_gross()
_NET = f"spec_{'7' * 32}"


def _left_on_net() -> dict[str, Any]:
    """`_GROSS`'s part counted net by default, under the spec_id `_NET`: the measure
    basis no one stated, as Search answers an ask that names none."""
    text = json.dumps(_GROSS)
    for old, new in ((_GROSS["spec_id"], _NET), ('"ref": "Gross"', '"ref": "Net"')):
        assert old in text, old
        text = text.replace(old, new)
    block = json.loads(text)
    scope = block["resolved"]["scope_choices"]
    (basis,) = [c for c in scope["choices"] if c["axis"] == "measure_basis"]
    basis["active"], basis["provenance"] = "measure_basis:net", "default"
    scope["entries"] = [e for e in scope["entries"] if e["axis"] != "measure_basis"]
    return block


class _GrossCompiles:
    """A tool surface whose `resolve_definition` answers `_GROSS` to a resubmit of its
    part on gross disbursements, as dev Search answered that post, recording each
    call's arguments in `calls`."""

    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        assert tool.name == "resolve_definition", tool
        self.calls.append(arguments)
        scope = _GROSS["resolved"]["scope_choices"]
        assert arguments["spec_id"] == scope["base"]["spec_id"]
        assert arguments["identity"] == scope["base"]["identity"]
        assert [e["option_id"] for e in arguments["scope_entries"]] == ["measure_basis:gross"]
        part = {"status": "definite", "spec": json.loads(json.dumps(_GROSS))}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


async def _gross_compared_with_net_average(
    ctx: _FakeToolContext,
    decision: DecisionFactory,
    *,
    warnings: tuple[str, ...] = (),
    settled: bool = True,
) -> None:
    """Propose US total ODA in 2023, its total stated gross, compared with its average
    left on net, both on one DAC1 part, the gross source shown with the search
    warnings `warnings`, settled or only offered as a candidate (`settled`). The
    sides differ by their operation, a sum against a mean, so the net input is
    aligned on gross, which Search compiles to the gross pin."""
    save_pin_sources(
        ctx.state,
        {
            _GROSS["spec_id"]: PinSource(block=_GROSS, settled=settled, warnings=warnings),
            _NET: PinSource(block=_left_on_net(), settled=True),
        },
    )
    question = "How does US gross ODA in 2023 compare with its average?"
    ctx.state[ASKS_STATE_KEY] = [question]
    surface = _GrossCompiles()

    def dataset(requirement_id: str, pin: str) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure="US total ODA",
            entities=[Entity(name="United States of America", role="donor")],
            period=Period(start=2023, end=2023),
            source_spec_id=pin,
        )

    result = await _tools_reading(cast(ToolSurface, surface))["propose_plan"].func(
        question=question,
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        answered=[AskAnswer(ask=0, outputs=(_COMPARE,))],
        evidence_requirements=[
            dataset("first", _GROSS["spec_id"]),
            dataset("second", _NET),
            _reduced("one", "first", Reduction.SUM),
            _reduced("two", "second", Reduction.MEAN),
            _difference("one", "two"),
        ],
        tool_context=ctx,
    )
    assert result["outcome"] == Outcome.OK.value, result
    assert len(surface.calls) == 1
    assert [
        (r.id, r.source_spec_id, r.aligned_from)
        for r in cast(Plan, load_plan(ctx.state)).evidence_requirements
        if isinstance(r, DatasetRequirement)
    ] == [("first", _GROSS["spec_id"], None), ("second", _GROSS["spec_id"], _NET)]


async def test_a_gross_total_compared_with_a_net_average_compiles_both_on_gross(
    decision: DecisionFactory,
) -> None:
    """The proposal compiles the average's input on gross through the resubmit a
    switch posts, and the workflow reads both inputs from the gross pin. The user's
    gross pin keeps the source they were shown, its basis theirs."""
    ctx = _FakeToolContext()
    await _gross_compared_with_net_average(ctx, decision)
    assert planner_axes(pin_sources(ctx.state)[_GROSS["spec_id"]].block) == frozenset()

    workflow = await _compiled(ctx)

    assert _retrievals(workflow) == {"first": _GROSS["spec_id"], "second": _GROSS["spec_id"]}
    assert _tool_names(workflow) == [
        "data-mcp.get_data",
        "analysis-mcp.aggregate",
        "data-mcp.get_data",
        "analysis-mcp.aggregate",
        "analysis-mcp.compare",
    ]
    assert workflow.terminal_steps == (_COMPARE,)


async def test_each_input_on_the_gross_pin_is_discovered_from_its_own_source(
    decision: DecisionFactory,
) -> None:
    """Discovery reads the gross pin's block for each input from the source that
    put it there: the total from the search the user's words settled, its basis
    theirs, and the average from the compile alignment made of its net pin, its
    basis the planner's."""
    ctx = _FakeToolContext()
    await _gross_compared_with_net_average(ctx, decision)
    plan = load_plan(ctx.state)
    assert plan is not None

    blocks = {
        rid: discovery._resolve_pin_block(
            ctx, plan=plan, requirement=cast(DatasetRequirement, plan.requirement(rid))
        )
        for rid in ("first", "second")
    }

    assert {rid: planner_axes(block) for rid, block in blocks.items()} == {
        "first": frozenset(),
        "second": frozenset({"measure_basis"}),
    }


async def test_an_aligned_input_reads_its_compiles_warnings_not_the_shown_pins(
    decision: DecisionFactory,
) -> None:
    """The search that showed the gross pin warned that it read part of the ask
    (`translator.discovery.BLOCKING_WARNING_CODES`), so the total is searched again.
    The average reads the compile of its net pin, which carries no such warning, and
    is discovered from it with no search."""
    ctx = _FakeToolContext()
    await _gross_compared_with_net_average(
        ctx, decision, warnings=(sorted(BLOCKING_WARNING_CODES)[0],)
    )
    refused = _SearchRefused()
    (discover_source,) = [
        tool
        for tool in build_discovery_tools(surface=cast(ToolSurface, refused))
        if tool.name == "discover_source"
    ]

    found = await discover_source.func(requirement_id="second", tool_context=ctx)

    assert found["status"] == "definite", found
    assert refused.calls == []


async def test_the_card_shows_the_averages_basis_as_the_planners_and_the_totals_as_the_users(
    decision: DecisionFactory,
) -> None:
    """The total's gross basis is the user's, read off the search their words settled,
    and the average's is the planner's, read off the compile alignment made of its net
    pin. Each input cites its own basis decision, and the plan card lists two basis
    rows, each governing its own input."""
    ctx = _FakeToolContext()
    await _gross_compared_with_net_average(ctx, decision)
    plan = cast(Plan, load_plan(ctx.state))

    cited = {
        rid: [
            (d.resolved_by, d.scope.aligned_for)
            for d in plan.decisions
            if d.slot in plan.requirement(rid).decision_ids
            and d.scope is not None
            and d.scope.axis == "measure_basis"
        ]
        for rid in ("first", "second")
    }
    rows = {
        choice_id(plan, d): d.resolved_by
        for d in shown_decisions(plan)
        if d.scope is not None and d.scope.axis == "measure_basis"
    }

    assert cited == {
        "first": [(ResolvedBy.USER, None)],
        "second": [(ResolvedBy.PLANNER_DEFAULT, "second")],
    }
    assert rows == {
        "first:measure_basis": ResolvedBy.USER,
        "second:measure_basis": ResolvedBy.PLANNER_DEFAULT,
    }


async def test_a_candidate_pinned_beside_a_compile_on_its_spec_id_reads_as_shown(
    decision: DecisionFactory,
) -> None:
    """The gross source is only offered as a candidate when the planner pins the total
    to it, and the average's compile answers the same spec_id. The candidate stays as
    shown, the total reads it with no planner marks and cites no decision of the
    compile's, and the average reads its compile."""
    ctx = _FakeToolContext()
    await _gross_compared_with_net_average(ctx, decision, settled=False)
    plan = cast(Plan, load_plan(ctx.state))

    blocks = {
        rid: discovery._resolve_pin_block(
            ctx, plan=plan, requirement=cast(DatasetRequirement, plan.requirement(rid))
        )
        for rid in ("first", "second")
    }
    compiled_rows = {
        d.slot for d in plan.decisions if d.scope is not None and d.scope.aligned_for is not None
    }

    assert pin_sources(ctx.state)[_GROSS["spec_id"]] == PinSource(block=_GROSS)
    assert {rid: planner_axes(block) for rid, block in blocks.items()} == {
        "first": frozenset(),
        "second": frozenset({"measure_basis"}),
    }
    assert compiled_rows
    assert compiled_rows.isdisjoint(plan.requirement("first").decision_ids)
    assert compiled_rows <= set(plan.requirement("second").decision_ids)
