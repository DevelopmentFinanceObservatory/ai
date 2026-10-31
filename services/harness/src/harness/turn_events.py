"""What the harness reads from a turn's ADK events.

Imports nothing of the harness's own, so the planner, the orchestrator, the
submission tools, the evals and the tests all import it without a cycle.
"""

from __future__ import annotations

from collections.abc import Sequence

from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import request_input
from google.genai import types

# The function name of ADK's `request_input` tool: every pause for the user's
# input, a planner-side ask and the workflow's own interrupts alike, is a call
# under this name and its answer a function response under it.
REQUEST_INPUT_FUNCTION_CALL_NAME = request_input.name


def _turn_question(callback_context: CallbackContext) -> types.Content | None:
    """The message the user opened the turn with, or None when the turn has none.

    It is the latest event of the invocation the user wrote outside any
    workflow node that holds text. ADK shows it to every node of the
    workflow. A user's answer to a pause is a user event of the same
    invocation too, but a function response with no text.
    """
    return next(
        (
            event.content
            for event in reversed(callback_context.session.events)
            if event.invocation_id == callback_context.invocation_id
            and event.author == "user"
            and not (event.node_info and event.node_info.path)
            and event.content is not None
            and any(part.text for part in event.content.parts or ())
        ),
        None,
    )


def after_turn_question(
    callback_context: CallbackContext, contents: Sequence[types.Content]
) -> int:
    """The position just past the turn's question (`_turn_question`) in
    `contents`, its latest copy there, or 0 when `contents` lack it."""
    question = _turn_question(callback_context)
    if question is None:
        return 0
    return next(
        (len(contents) - i for i, content in enumerate(reversed(contents)) if content == question),
        0,
    )
