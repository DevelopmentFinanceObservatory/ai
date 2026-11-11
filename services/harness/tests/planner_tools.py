"""Test helpers that reach a planner tool through its `ToolSpec`."""

from __future__ import annotations

from typing import Any

from google.adk.tools.base_tool import BaseTool
from google.genai import types
from planner.session import SessionCall
from planner.tool_spec import ToolSpec, function_declaration, run_tool, serve_tool


def _end_run(tool_context: Any) -> Any:
    def end_run() -> None:
        tool_context.actions.skip_summarization = True

    return end_run


def call_planner_tool(spec: ToolSpec, /, *, tool_context: Any, **arguments: Any) -> Any:
    """Return the tool's answer to `arguments`, a coroutine when the body is async.

    The call goes through `run_tool`, as a served call does. The arguments are
    converted and checked before the body runs. Every keyword must name one of
    the spec's parameters, so a misspelt one fails here and the conversion
    cannot drop it.
    """
    unknown = set(arguments) - {parameter.name for parameter in spec.parameters}
    assert not unknown, f"{spec.name} has no parameter {sorted(unknown)}"
    return run_tool(
        spec, arguments, call=SessionCall.of(tool_context), end_run=_end_run(tool_context)
    )


async def serve_planner_tool(spec: ToolSpec, *, args: dict[str, Any], tool_context: Any) -> Any:
    """Return the awaited answer to a call with `args`, as `serve_tool` serves it."""
    return await serve_tool(
        spec, args, call=SessionCall.of(tool_context), end_run=_end_run(tool_context)
    )


def planner_tool_declaration(tool: ToolSpec | BaseTool) -> types.FunctionDeclaration | None:
    """Return the function declaration the model is shown for the tool.

    A spec is declared as the planner's request declares it, and every ADK
    tool, the orchestrator's `FunctionTool`s included, through ADK's own
    declaration.
    """
    if isinstance(tool, ToolSpec):
        return function_declaration(tool)
    return tool._get_declaration()
