"""The orchestrator: one agent that answers a studio turn by calling tools.

It decides for itself what a question needs: find data, draw charts, ask
the user, or hand a multi-step question to the planner.
`orchestrator.agent.create_orchestrator_agent` builds it.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
