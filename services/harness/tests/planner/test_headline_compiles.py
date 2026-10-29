"""Calculations over Search's recorded answers, each proposed, discovered with no
fresh search and compiled against data-mcp's and analysis-mcp's schemas."""

from __future__ import annotations

from typing import Any, cast

from analysis_mcp.server import mcp as analysis_mcp
from compiler.compile import compile_plan
from compiler.sources import source_from_spec_block
from compiler.surface import McpSurface, ToolSurface
from data_mcp.server import mcp as data_mcp
from planner.plan import AnswerMode, load_plan, pin_sources
from planner.tool_results import Outcome
from planner.workflow import Workflow
from translator.discovery import build_discovery_tools
from translator.mechanical import build_translation

from ..decision_drafts import drafts
from .test_agent import (
    _SHARE,
    _SUM_BLOCKS,
    _US_MULTILATERAL,
    _US_TOTAL,
    DecisionFactory,
    _dac_countries,
    _FakeToolContext,
    _pins,
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
