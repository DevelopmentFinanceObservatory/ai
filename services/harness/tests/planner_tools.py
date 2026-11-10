"""Test helpers that reach a planner tool through the one tool class that wraps it."""

from __future__ import annotations

from typing import Any

from google.adk.tools import FunctionTool
from google.genai import types


def call_planner_tool(tool: FunctionTool, /, *, tool_context: Any, **arguments: Any) -> Any:
    """Return what the planner tool's function gives for `arguments` (a coroutine when it is async)."""
    return tool.func(tool_context=tool_context, **arguments)


def planner_tool_declaration(tool: FunctionTool) -> types.FunctionDeclaration | None:
    """Return the function declaration the model is shown for the planner tool."""
    return tool._get_declaration()
