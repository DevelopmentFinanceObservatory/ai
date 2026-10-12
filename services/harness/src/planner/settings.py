"""Env-driven settings for the planner agent."""

from __future__ import annotations

from common.settings import BaseServiceSettings
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

# The concrete version `common.settings.BaseServiceSettings.model_name`'s floating
# "-latest" alias resolves to today. Pinned rather than aliased because an
# eval score has to be attributable to one model: under the alias, a score
# that moves between two runs could mean a changed instruction or a repointed
# alias, and the run itself gives no way to tell which.
_PINNED_AGENT_MODEL = "gemini-3.8-flash"


class AgentSettings(BaseServiceSettings):
    """The model the planner agent runs on, and its entry deadlines.

    Separate from `harness.settings.Settings`, which configures the service
    around them -- the MCP URLs, the session backend -- and whose inherited
    `model_name` still drives the harness's own root agent. Reading a model
    string from here costs a caller none of that surface, which is what lets
    the eval agents under `evals/` read the same field the service does.

    `PLANNER_MODEL_NAME` overrides the pin. That override is how a
    different model gets scored (see `evals/README.md`).

    `PLANNER_ENTRY_DEADLINE_SECONDS` bounds one entry into the planner
    (`harness.planner_seam.PlannerSeam.__call__`), wall-clock, from the
    call's start. A correction round inside the same entry shares it; a
    resume after a user answer gets a fresh one, since that is its own
    `__call__`. 70s sets the time a discovery call inside the entry may
    take against the services beneath it. That call gets the deadline less
    `harness.planner_seam._NEXT_MODEL_CALL_MARGIN_S` (10s) less the
    generations before it, about 6s for the first, so 54s, which is not
    below dfo-apis' `request_timeout_s` (45s) for a search. A
    `resolve_definition` re-reads a family's members, which dfo-apis allows
    `resubmit_full_read_timeout_s` (90s), data-mcp `RESUBMIT_CALL_BUDGET_S`
    (100s), and the harness capability row `call_timeout_s` (110s). One cut
    at the entry's 54s keeps running for the entry's next proposal
    (`translator.discovery.CutCalls`).

    `PLANNER_ENTRY_DEADLINE_SECONDS_SAMPLED` is the deadline an entry drawn
    for `THINKING_SAMPLE_RATE` sampling gets instead: a sampled turn runs at
    the model's own default thinking level, deliberately slower than `LOW`
    by design, not by accident, so bounding it at the ordinary deadline would
    refuse a turn for having been sampled rather than for taking too long.
    240s clears a measured default-thinking generation of 79.92s (20,348
    reasoning tokens) with room for a multi-generation entry (a referral plus
    a correction round).

    `THINKING_SAMPLE_RATE` is the share of turns `harness.planner_seam.
    PlannerSeam` runs the planner at the model's default thinking level
    instead of `AGENT_THINKING_CONFIG`'s `LOW` (see `DEFAULT_THINKING_CONFIG`
    and `override_thinking_on_sampled_turns`). `0` disables sampling.
    """

    planner_model_name: str = _PINNED_AGENT_MODEL
    planner_entry_deadline_seconds: float = 70.0
    planner_entry_deadline_seconds_sampled: float = 240.0
    thinking_sample_rate: float = 0.05


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

# Session-state slot `harness.planner_seam.PlannerSeam.__call__` writes once
# per turn with this turn's sampling draw (a `bool`), and
# `override_thinking_on_sampled_turns` reads on every planner generation.
# Absent (a standalone agent run outside the seam) reads as `False`,
# so `AGENT_THINKING_CONFIG`'s `LOW` stands, unoverridden.
THINKING_SAMPLE_STATE_KEY = "planner:thinking_sampled"

# `AGENT_THINKING_CONFIG` with `thinking_level` left unset, the model's own
# default level, the much larger and slower budget its own comment measures
# `LOW` against. `override_thinking_on_sampled_turns` substitutes this on
# every generation of a sampled turn, so the unsampled `LOW` default stays
# measured against a live sample of what the larger budget would have done.
DEFAULT_THINKING_CONFIG = types.ThinkingConfig(include_thoughts=True)


def override_thinking_on_sampled_turns(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Replace this generation's `thinking_config` with `DEFAULT_THINKING_CONFIG`
    on a turn `THINKING_SAMPLE_STATE_KEY` marks sampled.

    `BuiltInPlanner.apply_thinking_config` runs ahead of every
    `before_model_callback` and always sets
    `llm_request.config.thinking_config` to `AGENT_THINKING_CONFIG` first, so
    `llm_request.config` already exists by the time this runs and this only
    has to overwrite the one field. Mutates `llm_request` in place and
    returns `None`, so composing this ahead of another `before_model_callback`
    never short-circuits the list, see
    `harness.planner_seam.PlannerSeam.__init__`.
    """
    if callback_context.session.state.get(THINKING_SAMPLE_STATE_KEY, False):
        assert llm_request.config is not None
        llm_request.config.thinking_config = DEFAULT_THINKING_CONFIG


def drop_thought_parts(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    """Strip a generation's thought-summary parts before they become session history.

    ADK traces the raw response (`google.adk.telemetry.tracing.trace_call_llm`)
    before any `after_model_callback` runs, so the trace already has whatever
    `AGENT_THINKING_CONFIG`'s `include_thoughts=True` put on the response;
    this callback only controls what the session event ADK builds from it
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
