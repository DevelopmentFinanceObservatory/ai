"""Test helpers that reach a planner tool through the ADK tool that serves it."""

from __future__ import annotations

from typing import Any

from google.adk.tools.base_tool import BaseTool
from google.genai import types
from planner.agent import SpecTool
from planner.tool_spec import ToolSpec


def call_planner_tool(tool: ToolSpec | SpecTool, /, *, tool_context: Any, **arguments: Any) -> Any:
    """Return what the planner tool answers `arguments` with (a coroutine when its body is async).

    The call goes through `SpecTool.invoke`, as a served call does. The
    arguments are converted and checked before the body runs. Every keyword
    must name one of the spec's parameters, so a misspelt one fails here and
    the conversion cannot drop it.
    """
    served = tool if isinstance(tool, SpecTool) else SpecTool(tool)
    unknown = set(arguments) - {parameter.name for parameter in served.spec.parameters}
    assert not unknown, f"{served.spec.name} has no parameter {sorted(unknown)}"
    return served.invoke(arguments, tool_context)


def planner_tool_declaration(tool: ToolSpec | BaseTool) -> types.FunctionDeclaration | None:
    """Return the function declaration the model is shown for the tool.

    A spec is read as `SpecTool` serves it, and every ADK tool, the
    orchestrator's `FunctionTool`s included, through ADK's own declaration.
    """
    served = SpecTool(tool) if isinstance(tool, ToolSpec) else tool
    return served._get_declaration()
