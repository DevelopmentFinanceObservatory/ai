"""Backs the harness-pair-v1 eval set with the planner and its translator sub-agent.

Score it with `just eval-pair`, then read the results with `just eval-web` --
`evals/README.md` covers why scoring belongs to the CLI. Both need a
`GOOGLE_API_KEY` or Vertex AI credentials in the repo-root `.env`, which ADK
loads before importing this module.

What this set scores is the seam the other two cannot reach. `harness-planner-v2`
builds the planner alone, so `dfo_translator` is not among its tools and no case
there can observe the planner reaching for it; `harness-translator-v1` scores the
translator alone, driven by a turn the harness writes rather than by a planner.
Only the pair shows whether the planner has its plan translated before it
invites approval, and what it does when a decision comes back referred.

Built the way `harness.agents.create_planner_with_translator` builds it, with the
same canned discovery `harness-translator-v1` uses (`harness_translator_v1.
discovery_stub`; this package's `__init__` puts `evals/agents` on the path, which
`adk web` does for an agent folder and `adk eval` does not). The planner is
built with no surface, so it carries no skill tools and its instruction does not
name them; the canned discovery answers no skill read. The translator's own
turns do not reach the planner's
context, but they are all in the recorded trace, authored by `dfo_translator`.
A judge reads the trace, so a property about the planner's own calls has to say
whose calls it means.
"""

from __future__ import annotations

from harness_translator_v1.discovery_stub import EvalSurface
from planner.agent import create_planner_agent
from planner.settings import AgentSettings
from translator.agent import create_translator_agent

# The same fields the deployed pair is built from, so this scores the models the
# service runs; `PLANNER_MODEL_NAME` and `TRANSLATOR_MODEL_NAME` override them
# here exactly as they do there.
_AGENT_SETTINGS = AgentSettings()

root_agent = create_planner_agent(
    model_name=_AGENT_SETTINGS.planner_model_name,
    translator=create_translator_agent(
        model_name=_AGENT_SETTINGS.translator_model_name,
        surface=EvalSurface(),
        mode="single_turn",
    ),
)
