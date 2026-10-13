"""Backs the harness-planner-v2 eval set with the same planner agent the harness runs.

Score it with `just eval-planner`, then read the results with `just eval-web` --
`evals/README.md` covers why scoring belongs to the CLI. Both need a
`GOOGLE_API_KEY` or Vertex AI credentials in the repo-root `.env`, which ADK
loads before importing this module.

The planner is built with no translator, so the cases score
its plan-lifecycle tools alone.
"""

from __future__ import annotations

from planner.agent import create_planner_agent
from planner.settings import AgentSettings

# The same fields the deployed planner is built from, so this scores the model
# and schema delivery the service runs; `PLANNER_MODEL_NAME` and
# `PLANNER_SCHEMA_DELIVERY` override them here exactly as they do there.
# Evaluating a model the env can't change would mean evaluating one the
# harness may not be running.
_AGENT_SETTINGS = AgentSettings()

root_agent = create_planner_agent(
    model_name=_AGENT_SETTINGS.planner_model_name,
    schema_delivery=_AGENT_SETTINGS.planner_schema_delivery,
)
