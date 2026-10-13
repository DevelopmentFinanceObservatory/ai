"""The executor: runs an approved plan's workflow and checks what comes back.

Code, not an agent. `executor.run` has three public entry points:
`execute_authorised`, the shared execution path for a run admitted under a
typed authorisation; `execute_approved_plan`, which runs an approved plan
through it; and `resume_run`, which carries on a run a restart left
unfinished. A run resolves each step's bindings from the results it holds,
validates the full arguments against the live tool schema, calls the tool,
checks the result against the kind of value the step was compiled to
produce, and checks each retrieval against the requirement it is evidence
for. A run is accepted only when every step succeeds and every requirement is
met; otherwise it stops with a typed outcome the planner can act on. A
defect that stops an admitted run's computation records the run failed and
raises `RunAbandonedError`, which names the run.
"""

from __future__ import annotations

from executor.run import RunAbandonedError

__version__ = "0.1.0"

__all__ = ["RunAbandonedError", "__version__"]
