"""The function declarations every agent sends its model, read as Gemini reads them.

Gemini reads nothing beside a `$ref`, so a description or default sitting next
to one never reaches the model. Every `$ref` resolves in `$defs`, and no
definition goes unreferenced.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from compiler.surface import FixtureSurface
from executor.ledger import InMemoryRunLedger
from google.adk.agents import LlmAgent
from google.adk.tools.base_tool import BaseTool
from harness.direct_recipes import DirectRecipes
from harness.orchestrator_tools import build_orchestrator_tools
from harness.planner_seam import PlannerSeam
from harness.settings import Settings
from harness.wiring import build_planner
from orchestrator.agent import create_orchestrator_agent
from planner.agent import create_planner_agent
from planner.declaration import _referred
from planner.settings import AgentSettings, SchemaDelivery

from .orchestrator_fakes import no_executor, no_resolver

# Every LLM agent the harness builds, by the builder that builds it: the
# planner as the server builds it, under each `SchemaDelivery`, the planner its
# own eval set scores, eager and compact, and the orchestrator, with and
# without its direct recipes.
_AGENTS: dict[str, Callable[[], LlmAgent]] = {
    "served_planner": lambda: (
        build_planner(settings=Settings(), surface=FixtureSurface(schemas=[]))._agent
    ),
    "served_compact_planner": lambda: (
        PlannerSeam(
            settings=Settings(),
            surface=FixtureSurface(schemas=[]),
            agent_settings=AgentSettings(planner_schema_delivery=SchemaDelivery.COMPACT),
        )._agent
    ),
    "standalone_planner": lambda: create_planner_agent(model_name="gemini-test"),
    "standalone_compact_planner": lambda: create_planner_agent(
        model_name="gemini-test", schema_delivery=SchemaDelivery.COMPACT
    ),
    "orchestrator": lambda: create_orchestrator_agent(
        model_name="gemini-test",
        tools=build_orchestrator_tools(resolver=no_resolver, executor=no_executor),
    ),
    "orchestrator_with_direct_recipes": lambda: create_orchestrator_agent(
        model_name="gemini-test",
        tools=build_orchestrator_tools(
            resolver=no_resolver,
            executor=no_executor,
            direct=DirectRecipes(
                recipes=("growth",),
                run_ledger=InMemoryRunLedger(),
                publisher=None,
                tool_surface=FixtureSurface(schemas=[]),
                translation_surface=FixtureSurface(schemas=[]),
            ),
        ),
    ),
}


def _refs_beside_other_keys(node: Any, path: str) -> list[str]:
    """The path of every object in `node` that holds a `$ref` and another key."""
    if isinstance(node, dict):
        found = [path] if "$ref" in node and len(node) > 1 else []
        return found + [
            hit
            for key, value in node.items()
            for hit in _refs_beside_other_keys(value, f"{path}/{key}")
        ]
    if isinstance(node, list):
        return [
            hit
            for index, item in enumerate(node)
            for hit in _refs_beside_other_keys(item, f"{path}/{index}")
        ]
    return []


@pytest.mark.parametrize("agent_name", sorted(_AGENTS))
def test_no_declaration_puts_a_key_beside_a_ref(agent_name: str) -> None:
    agent = _AGENTS[agent_name]()
    tools = [tool for tool in agent.tools if isinstance(tool, BaseTool)]
    declarations = [tool._get_declaration() for tool in tools]

    assert len(tools) == len(agent.tools)
    assert all(declarations)
    assert [
        hit
        for declaration in declarations
        if declaration is not None
        for hit in _refs_beside_other_keys(
            declaration.model_dump(exclude_none=True), declaration.name or ""
        )
    ] == []

    for declaration in declarations:
        assert declaration is not None
        schema = declaration.parameters_json_schema or {}
        assert _referred(schema) == set(schema.get("$defs", {})), declaration.name
