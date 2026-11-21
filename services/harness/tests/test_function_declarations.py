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
from google.genai import types
from harness.orchestrator_tools import build_orchestrator_tools
from harness.settings import Settings
from harness.wiring import build_planner
from orchestrator.declarations import tool_declaration
from planner.declaration import _referred
from planner.tool_spec import function_declaration


async def _unused(**kwargs: Any) -> Any:
    raise AssertionError("no call expected")


# The declarations of every model the harness runs, by the builder that builds
# its tools: the planner as the server builds it, and the orchestrator.
_DECLARATIONS: dict[str, Callable[[], Sequence[types.FunctionDeclaration]]] = {
    "served_planner": lambda: [
        function_declaration(spec)
        for spec in build_planner(settings=Settings(), surface=FixtureSurface(schemas=[]))._specs
    ],
    "orchestrator": lambda: [
        tool_declaration(tool)
        for tool in build_orchestrator_tools(resolver=_unused, executor=_unused)
    ],
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


@pytest.mark.parametrize("model", sorted(_DECLARATIONS))
def test_no_declaration_puts_a_key_beside_a_ref(model: str) -> None:
    declarations = _DECLARATIONS[model]()

    assert declarations
    assert [
        hit
        for declaration in declarations
        for hit in _refs_beside_other_keys(
            declaration.model_dump(exclude_none=True), declaration.name or ""
        )
    ] == []

    for declaration in declarations:
        schema = declaration.parameters_json_schema or {}
        assert _referred(schema) == set(schema.get("$defs", {})), declaration.name
