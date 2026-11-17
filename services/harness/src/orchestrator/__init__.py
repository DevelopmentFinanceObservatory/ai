"""The orchestrator: one agent that answers a studio turn by calling tools.

It decides for itself what a question needs: find data, draw charts, ask
the user, or hand a multi-step question to the planner.
`orchestrator.agent.orchestrator_agent` builds it and
`orchestrator.run.run_orchestrator` runs it.
"""

from __future__ import annotations

__version__ = "0.1.0"

# The name the orchestrator runs under: its agent's name on the traces, and the
# author of its events in a stored conversation, which Studio's translator
# reads them by.
AGENT_NAME = "dfo_orchestrator"

__all__ = ["AGENT_NAME", "__version__"]
