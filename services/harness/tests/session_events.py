"""Fake ADK session events, as the planner's tools read them off `tool_context.session`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def response_event(name: str, response: dict[str, Any]) -> Any:
    """A fake session event for the `FunctionResponse` a tool `name` answered with.

    The shape `planner.call.answered_since_referral` reads: the planner's own
    tool responses (`propose_plan`, whose `translation` carries a referral) and the user's answers
    (`adk_request_input`) are all `FunctionResponse` parts on session events.
    """
    part = SimpleNamespace(
        function_call=None,
        function_response=SimpleNamespace(id=f"{name}-1", name=name, response=response),
    )
    return SimpleNamespace(content=SimpleNamespace(parts=[part]))
