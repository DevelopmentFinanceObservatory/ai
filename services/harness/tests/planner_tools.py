"""Test helpers that reach a planner tool through the ADK tool that serves it."""

from __future__ import annotations

from typing import Any

from google.adk.tools.base_tool import BaseTool
from google.genai import types
from planner.agent import SpecTool
from planner.tool_spec import ToolSpec, function_declaration


def call_planner_tool(tool: ToolSpec | SpecTool, /, *, tool_context: Any, **arguments: Any) -> Any:
    """Return what the planner tool answers `arguments` with (a coroutine when its body is async).

    The call goes through `SpecTool.invoke`, as a served call does: the
    arguments are converted and checked before the body runs.
    """
    served = tool if isinstance(tool, SpecTool) else SpecTool(tool)
    return served.invoke(arguments, tool_context)


def planner_tool_declaration(tool: ToolSpec | BaseTool) -> types.FunctionDeclaration | None:
    """Return the function declaration the model is shown for the tool.

    A tool that is not a planner spec, such as the orchestrator's
    `FunctionTool`s, is read through ADK's own declaration.
    """
    if isinstance(tool, SpecTool):
        return function_declaration(tool.spec)
    if isinstance(tool, ToolSpec):
        return function_declaration(tool)
    return tool._get_declaration()
