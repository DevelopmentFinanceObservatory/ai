"""Backs the harness-pair-v1 eval set with the planner and its translation.

Score it with `just eval-pair`, then read the results with `just eval-web` --
`evals/README.md` covers why scoring belongs to the CLI. Both need a
`GOOGLE_API_KEY` or Vertex AI credentials in the repo-root `.env`, which ADK
loads before importing this module.

What this set scores is the seam the planner-only set cannot reach.
`harness-planner-v2` builds the planner alone, so its `propose_plan` runs no
translation and no case there can observe the planner reading a verdict. Only
this set shows whether the planner reads the `translation` its accepted
proposal comes back with before it invites approval, and what it does when a
requirement comes back referred.

Built the way `harness.agents.create_planner_with_translator` builds it, with
the canned discovery in `harness_translator_v1.discovery_stub` (this package's
`__init__` puts `evals/agents` on the path, which `adk web` does for an agent
folder and `adk eval` does not). The planner is built with no surface, so it
carries no skill tools and its instruction does not name them; the canned
discovery answers no skill read.
"""

from __future__ import annotations

from harness_translator_v1.discovery_stub import EvalSurface
from planner.agent import create_planner_agent
from planner.settings import AgentSettings
from translator.tool import build_translate

# The same field the deployed planner is built from, so this scores the model
# the service runs; `PLANNER_MODEL_NAME` overrides it here exactly as it does
# there.
_AGENT_SETTINGS = AgentSettings()

root_agent = create_planner_agent(
    model_name=_AGENT_SETTINGS.planner_model_name,
    translate=build_translate(surface=EvalSurface()),
)
