"""Env-driven settings for the planner agent."""

from __future__ import annotations

from common.settings import BaseServiceSettings
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.genai import types

# The concrete version `common.settings.BaseServiceSettings.model_name`'s floating
# "-latest" alias resolves to today. Pinned rather than aliased because an
# eval score has to be attributable to one model: under the alias, a score
# that moves between two runs could mean a changed instruction or a repointed
# alias, and the run itself gives no way to tell which.
_PINNED_AGENT_MODEL = "gemini-3.8-flash"


class AgentSettings(BaseServiceSettings):
    """The model the planner agent runs on.

    Separate from `harness.settings.Settings`, which configures the service
    around it -- the MCP URLs, the session backend -- and whose inherited
    `model_name` still drives the harness's own root agent. Reading a model
    string from here costs a caller none of that surface, which is what lets
    the eval agents under `evals/` read the same field the service does.

    `PLANNER_MODEL_NAME` overrides the pin. That override is how a
    different model gets scored (see `evals/README.md`).
    """

    planner_model_name: str = _PINNED_AGENT_MODEL


# `gemini-3.8-flash` takes a `thinking_level` of `LOW`, `MEDIUM` or `HIGH`
# (`MINIMAL` is a flash-lite-only value this model rejects) and otherwise
# runs at an unset, much larger default. The planner runs every generation at
# `LOW`: measured against the real planner brief, it reproduces the unset
# default's validity outcome on both a well-formed and a malformed draft, at
# roughly 1/14 to 1/46 the wall time and near-zero reasoning tokens.
# `include_thoughts=True` puts the model's own thought summary on the
# response, so a trace shows what a generation reasoned rather than only a
# token count. `drop_thought_parts`, registered as the planner's
# `after_model_callback`, keeps that summary out of the model's later
# requests and the user-facing stream once the trace has it.
AGENT_THINKING_CONFIG = types.ThinkingConfig(
    thinking_level=types.ThinkingLevel.LOW, include_thoughts=True
)


def drop_thought_parts(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    """Strip a generation's thought-summary parts before they become session history.

    ADK traces the raw response on the `generate_content` span
    (`google.adk.telemetry.tracing.trace_inference_result`) before any
    `after_model_callback` runs, so the trace already has whatever
    `AGENT_THINKING_CONFIG`'s `include_thoughts=True` put on the response,
    the thought summary written as a plain text part with no thought flag.
    This callback only controls what the session event ADK builds from it
    afterward.

    Left in, a thought-summary part would ride into every later request of
    the same turn, since the planner replays its own turn's prior events
    back to the model, and into the harness's SSE stream and its digest,
    which read the same events. A part carrying `thought_signature` is kept
    regardless of `thought`: Gemini requires that signature on a later
    request's function-call or final text part and answers one without it
    with a 400 (see `harness.planner_seam._SKIP_THOUGHT_SIGNATURE`).
    `_nl_planning._remove_thought_from_request` only strips `thought` for a
    non-`BuiltInPlanner`, so this callback is what keeps a thought part out
    of replay for the `BuiltInPlanner` the planner runs.

    `after_model_callback` runs a list in order and stops at the first
    truthy return, so a caller composing this alongside another callback
    that still needs the response has to list that one first, see
    `harness.planner_seam.PlannerSeam.__init__`. Returns `None` (response
    unchanged) when no part is dropped, and also when every part would be
    dropped, rather than returning a response with an empty `parts` list.
    """
    content = llm_response.content
    if content is None or not content.parts:
        return None
    parts = [part for part in content.parts if not part.thought or part.thought_signature]
    if not parts or len(parts) == len(content.parts):
        return None
    return llm_response.model_copy(update={"content": content.model_copy(update={"parts": parts})})
