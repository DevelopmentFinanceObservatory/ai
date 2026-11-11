"""The function declarations the harness's models are sent, read as Gemini reads them.

Gemini reads nothing beside a `$ref`, so a description or default sitting next
to one never reaches the model. Every `$ref` resolves in `$defs`, and no
definition goes unreferenced.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest
from compiler.surface import FixtureSurface
from google.adk.tools.base_tool import BaseTool
from harness.orchestrator_tools import build_orchestrator_tools
from harness.settings import Settings
from harness.wiring import build_planner
from orchestrator.agent import create_orchestrator_agent
from planner.declaration import _referred
from planner.tool_spec import ToolSpec

from .orchestrator_fakes import no_executor, no_resolver
from .planner_tools import planner_tool_declaration

# The tools of every model the harness runs, by the builder that builds them: the
# planner as the server builds it, and the orchestrator.
_TOOLS: dict[str, Callable[[], Sequence[Any]]] = {
    "served_planner": lambda: (
        build_planner(settings=Settings(), surface=FixtureSurface(schemas=[]))._specs
    ),
    "orchestrator": lambda: (
        create_orchestrator_agent(
            model_name="gemini-test",
            tools=build_orchestrator_tools(resolver=no_resolver, executor=no_executor),
        ).tools
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


@pytest.mark.parametrize("model", sorted(_TOOLS))
def test_no_declaration_puts_a_key_beside_a_ref(model: str) -> None:
    tools = _TOOLS[model]()
    assert all(isinstance(tool, ToolSpec | BaseTool) for tool in tools)
    declarations = [planner_tool_declaration(tool) for tool in tools]

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
