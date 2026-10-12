"""The orchestrator's answer to a tool that raises.

ADK hands an `Exception` raised inside a tool to the agent's
`on_tool_error_callback`. A dict the callback returns becomes the tool's
result, and `None` lets the exception fail the turn. `answering_tool_failures`
builds that callback, so a tool that breaks costs the model one tool result
and the user still gets an answer from what did work. Pauses and cancellation
are `BaseException`s and never reach it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection
from typing import Any

from common.telemetry import current_trace_id
from google.adk.tools import BaseTool, ToolContext
from harness.orchestrator_tools import _ANSWER_NOW

logger = logging.getLogger(__name__)

type ToolErrorCallback = Callable[[BaseTool, dict[str, Any], ToolContext, Exception], dict | None]


def answering_tool_failures(tool_names: Collection[str]) -> ToolErrorCallback:
    """An `on_tool_error_callback` that logs a failure of one of `tool_names`
    at ERROR, with its traceback, and answers the model with the tool's name
    and the exception's class.

    The exception's message stays out of the answer, since it can carry
    internal URLs or data. A name outside `tool_names` is a tool the model
    made up. The callback returns `None` for it, so ADK's own answer, which
    lists the tools the model can call, stands.
    """

    def answer(
        tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, error: Exception
    ) -> dict | None:
        if tool.name not in tool_names:
            return None
        failure = type(error).__name__
        logger.error(
            "orchestrator tool %s raised %s session_id=%s invocation_id=%s trace_id=%s",
            tool.name,
            failure,
            tool_context.session.id,
            tool_context.invocation_id,
            current_trace_id(),
            exc_info=error,
        )
        return {
            "error": (
                f"{tool.name} failed inside DFO ({failure}). Do not repeat this call. {_ANSWER_NOW}"
            )
        }

    return answer
