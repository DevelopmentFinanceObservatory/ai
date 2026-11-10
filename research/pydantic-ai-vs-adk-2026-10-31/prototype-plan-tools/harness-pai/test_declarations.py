"""Pass line 1: the tools the Pydantic AI model is offered are the tools arm B's model is.

Compared against the declaration objects `harness-ab`'s seam builds: field by field, and
as compact JSON characters (`measure.chars`), which `harness-ab/NOTES.md` reports.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mcp_planner import PROPOSE_PLAN
from measure import chars
from outward import ASK_TOOL
from pai_plan_mcp import PlanDeps, plan_agent, plan_toolsets
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models import Model
from pydantic_ai.tools import ToolDefinition
from rig import Rig, Script, b_client, b_seam, say

# `harness-ab`'s skill tools are ADK tools the seam builds for both arms; this
# prototype leaves them to the planner's existing Pydantic AI skill tools.
SKILL_TOOLS = {"get_skill", "get_skill_file"}

type AgentFactory = Callable[[Model, str], Agent[PlanDeps, Any]]


def declared(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters_json_schema": tool.parameters_json_schema,
    }


async def faults(rig: Rig, factory: AgentFactory) -> list[str]:
    """How what the Pydantic AI model reads before its first answer differs from B's."""
    async with b_client(rig) as client:
        seam = await b_seam(client)
    instruction = str(seam._agent.instruction)
    expected = {
        tool.name: tool._get_declaration().model_dump(mode="json", exclude_none=True)
        for tool in seam._agent.tools
        if tool.name not in SKILL_TOOLS
    }
    script = Script([say("done")])
    await factory(script.model, instruction).run("brief", deps=PlanDeps("ps_none"))
    tools = {tool.name: tool for tool in script.offered[0]}

    out = []
    if set(tools) != set(expected):
        out.append(f"tools offered {sorted(tools)}, B offers {sorted(expected)}")
    for name, dump in expected.items():
        if name in tools and declared(tools[name]) != dump:
            out.append(f"{name}: {chars(declared(tools[name]))} chars, B's {chars(dump)}")
    if script.instructions[0] != instruction:
        out.append("the instruction differs from B's")
    print(
        "offered:",
        {name: chars(declared(tool)) for name, tool in tools.items()},
        "B:",
        {name: chars(dump) for name, dump in expected.items()},
    )
    return out


async def test_the_model_is_offered_what_b_offers(rig: Rig) -> None:
    def factory(model: Model, instruction: str) -> Agent[PlanDeps, Any]:
        _, view = plan_toolsets(rig.source)
        return plan_agent(model, view, instruction=instruction)

    assert await faults(rig, factory) == []


async def test_the_ask_is_a_deferred_tool_and_the_proposal_is_not(rig: Rig) -> None:
    script = Script([say("done")])
    _, view = plan_toolsets(rig.source)
    await plan_agent(script.model, view, instruction="").run("brief", deps=PlanDeps("ps_none"))
    kinds = {tool.name: tool.kind for tool in script.offered[0]}
    assert kinds == {PROPOSE_PLAN: "function", ASK_TOOL: "external"}


async def test_control_the_whole_server_is_offered_when_nothing_prepares_it(rig: Rig) -> None:
    def factory(model: Model, instruction: str) -> Agent[PlanDeps, Any]:
        return Agent(
            model, deps_type=PlanDeps, toolsets=[MCPToolset(rig.source)], instructions=instruction
        )

    found = await faults(rig, factory)
    print("control, unprepared toolset:", found)
    assert found
    assert "run_plan" in found[0] and "start_plan" in found[0]
