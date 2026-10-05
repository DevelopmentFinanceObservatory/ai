"""The executor: runs an approved plan's workflow and checks what comes back.

Code, not an agent. `executor.run.execute_approved_plan` is its one public
entry point: it resolves each step's bindings from the results it holds,
validates the full arguments against the live tool schema, calls the tool,
checks the result against the kind of value the step was compiled to
produce, and checks each retrieval against the requirement it is evidence
for. A run is accepted only when every step succeeds and every requirement is
met; otherwise it stops with a typed outcome the planner can act on.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
