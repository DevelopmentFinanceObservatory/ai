"""Backs the harness-planner-v2 eval set with the same planner agent the harness runs.

Score it with `just eval-planner`, then read the results with `just eval-web` --
`evals/README.md` covers why scoring belongs to the CLI. Both need a Gemini
developer API key, `GOOGLE_API_KEY`, in the repo-root `.env`, which ADK loads
before importing this module. The planner refuses to build when ADK's Google
variant is Vertex AI.

The planner is built with no translator, so the cases score
its plan-lifecycle tools alone.
"""

from __future__ import annotations

from planner.agent import create_planner_agent
from planner.settings import AgentSettings

# The same field the deployed planner is built from, so this scores the model
# the service runs; `PLANNER_MODEL_NAME` overrides it here exactly as it does
# there.
# Evaluating a model the env can't change would mean evaluating one the
# harness may not be running.
_AGENT_SETTINGS = AgentSettings()

root_agent = create_planner_agent(model_name=_AGENT_SETTINGS.planner_model_name)
