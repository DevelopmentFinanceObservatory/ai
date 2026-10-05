"""Backs the harness-translator-v1 eval set with a real agent for ADK to load.

Score it with `just eval-translator`, then read the results with `just eval-web` --
`evals/README.md` covers why scoring belongs to the CLI. Both need a
`GOOGLE_API_KEY` or Vertex AI credentials in the repo-root `.env`, which ADK
loads before importing this module.

The surface below reads tool schemas from the real data-mcp and analysis-mcp,
so proposals compile against the same pinned capability table the service
uses, and sends discovery to a canned data-mcp whose `search` answers are
fixed per ask (see `discovery_stub`). A case's outcome therefore turns on the
translator, not on live data.
"""

from __future__ import annotations

from planner.settings import AgentSettings
from translator.agent import create_translator_agent

from .discovery_stub import EvalSurface

# The same field the deployed translator is built from, so this scores the model
# the service runs; `TRANSLATOR_MODEL_NAME` overrides it here exactly as it does
# there.
_MODEL_NAME = AgentSettings().translator_model_name

root_agent = create_translator_agent(model_name=_MODEL_NAME, surface=EvalSurface())
