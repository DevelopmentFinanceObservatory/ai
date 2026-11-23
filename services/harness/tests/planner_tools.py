"""Test helpers that reach a planner tool through its `ToolSpec`.

A tool call runs on a context holding the turn's `state`, its transcript
`records`, the served call's `function_call_id` and `run_ended`, which the
tool's `end_run` sets (`planner.tool_spec.EndRun`).
"""

from __future__ import annotations

from typing import Any

from google.genai import types
from planner.session import SessionCall
from planner.tool_spec import EndRun, ToolSpec, function_declaration, run_tool, serve_tool


def session_call(tool_context: Any) -> SessionCall:
    """The `SessionCall` of a tool call on `tool_context`."""
    return SessionCall(tool_context.records, tool_context.state, tool_context.function_call_id)


def end_run_on(tool_context: Any) -> EndRun:
    """The `EndRun` of a tool call on `tool_context`: it sets `run_ended`."""

    def end_run() -> None:
        tool_context.run_ended = True

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
        spec, arguments, call=session_call(tool_context), end_run=end_run_on(tool_context)
    )


async def serve_planner_tool(spec: ToolSpec, *, args: dict[str, Any], tool_context: Any) -> Any:
    """Return the awaited answer to a call with `args`, as `serve_tool` serves it."""
    return await serve_tool(
        spec, args, call=session_call(tool_context), end_run=end_run_on(tool_context)
    )


def planner_tool_declaration(tool: ToolSpec) -> types.FunctionDeclaration:
    """Return the function declaration the model is shown for the tool, as the
    planner's request declares it."""
    return function_declaration(tool)
