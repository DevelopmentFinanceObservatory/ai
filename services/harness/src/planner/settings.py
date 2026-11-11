"""Env-driven settings for the planner agent."""

from __future__ import annotations

from collections.abc import Sequence

from common.settings import BaseServiceSettings
from google.genai import types

# The name the planner runs under: the author and branch of its events in the
# session (`harness.planner_events`) and the name of its agent's spans.
AGENT_NAME = "dfo_planner"

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
# token count. `without_unsigned_thoughts` keeps that summary out of the
# model's later requests and the session's events
# (`planner.pydantic_agent._without_unsigned_thoughts`).
AGENT_THINKING_CONFIG = types.ThinkingConfig(
    thinking_level=types.ThinkingLevel.LOW, include_thoughts=True
)


def without_unsigned_thoughts(parts: Sequence[types.Part]) -> list[types.Part] | None:
    """`parts`, one model response's, less every thought-summary part that
    carries no thought signature, or None when that would drop no part or
    every part.

    A part with a `thought_signature` is kept whether or not it is a thought,
    since Gemini requires the signature on a later request. A response made
    only of unsigned thoughts is kept whole, so no response is left empty.
    """
    kept = [part for part in parts if not part.thought or part.thought_signature]
    if not kept or len(kept) == len(parts):
        return None
    return kept
