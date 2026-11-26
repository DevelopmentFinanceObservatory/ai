"""Env-driven settings for the planner agent."""

from __future__ import annotations

from common.settings import BaseServiceSettings
from google.genai import types
from harness.settings import ThinkingLevelName
from pydantic import field_validator

# The name the planner runs under: the name of its agent and its spans, and
# the agent its usage is logged under.
AGENT_NAME = "dfo_planner"

# The planner's model, pinned to a concrete version because an eval score has
# to be attributable to one model: under a floating alias, a score that
# moves between two runs could mean a changed instruction or a repointed alias,
# and the run itself gives no way to tell which. It names the same model as the
# harness's `MODEL_NAME` default (`harness.settings.Settings.model_name`), since
# Studio refuses a planner and orchestrator on different surfaces
# (`harness.surface.resolve_surface`).
_PINNED_AGENT_MODEL = "gemini-4-hydrogen"

# The planner's reasoning effort when `PLANNER_THINKING_LEVEL` is unset. See
# `AGENT_THINKING_CONFIG` for the measurement behind `LOW`.
DEFAULT_PLANNER_THINKING_LEVEL: ThinkingLevelName = "LOW"


class AgentSettings(BaseServiceSettings):
    """The model the planner agent runs on.

    Separate from `harness.settings.Settings`, which configures the service
    around it -- the MCP URLs, the session backend -- and whose
    `model_name` drives the harness's own root agent. Reading a model
    string from here costs a caller none of that surface, which is what lets
    the eval agents under `evals/` read the same field the service does.

    `PLANNER_MODEL_NAME` overrides the pin. That override is how a
    different model gets scored (see `evals/README.md`).
    `PLANNER_THINKING_LEVEL` sets the planner's reasoning effort, read
    case-insensitively; unset or blank means `DEFAULT_PLANNER_THINKING_LEVEL`.
    """

    planner_model_name: str = _PINNED_AGENT_MODEL
    planner_thinking_level: ThinkingLevelName = DEFAULT_PLANNER_THINKING_LEVEL

    @field_validator("planner_thinking_level", mode="before")
    @classmethod
    def _normalise_thinking_level(cls, value: object) -> object:
        # `low`, ` Low ` and `LOW` all name one level. A blank value is the default.
        if isinstance(value, str):
            return value.strip().upper() or DEFAULT_PLANNER_THINKING_LEVEL
        return value


def agent_thinking_config(level: ThinkingLevelName) -> types.ThinkingConfig:
    """The planner's thinking config at `level`, with thought summaries on."""
    return types.ThinkingConfig(thinking_level=types.ThinkingLevel(level), include_thoughts=True)


# `gemini-3.8-flash` takes a `thinking_level` of `LOW`, `MEDIUM` or `HIGH`
# (`MINIMAL` is a flash-lite-only value this model rejects) and otherwise
# runs at an unset, much larger default. The planner runs every generation at
# `PLANNER_THINKING_LEVEL`, read once at import and `LOW` by default. Measured on
# `gemini-3.8-flash` against the real planner brief, `LOW` reproduces the unset
# default's validity outcome on both a well-formed and a malformed draft, at
# roughly 1/14 to 1/46 the wall time and near-zero reasoning tokens.
# `include_thoughts=True` puts the model's own thought summary on the
# response, so a trace shows what a generation reasoned rather than only a
# token count. `harness.gemini_agent.without_unsigned_thoughts` keeps that
# summary out of the model's later requests and the turn's transcript.
AGENT_THINKING_CONFIG = agent_thinking_config(AgentSettings().planner_thinking_level)
