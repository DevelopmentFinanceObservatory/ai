"""The names a turn's pauses and the planner's ask go by.

Imports nothing of the harness's own, so the planner, the orchestrator, the
submission tools, the evals and the tests all import it without a cycle.
"""

from __future__ import annotations

from google.adk.tools import request_input

# The function name of ADK's `request_input` tool: every pause for the user's
# input, a planner-side ask and the workflow's own interrupts alike, is a call
# under this name and its answer a function response under it.
REQUEST_INPUT_FUNCTION_CALL_NAME = request_input.name

# The planner's own ask, the tool it calls to put a question to the user
# (`ask_user_question`, `planner.specs`). The tool answers every call at once,
# refusing or accepting it. The graph's planner node pauses on an accepted
# ask, and the planner reads the user's answer to that pause as the answer to
# the call (`planner.session.session_steps`).
PLANNER_ASK = "ask_user_question"
