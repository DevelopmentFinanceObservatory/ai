"""Translation, run as code inside the planner's `propose_plan`.

Finds the data a plan's requirements need, builds the operations that carry
them out, and has the compiler turn them into the workflow the user approves;
or refers one requirement back to the planner with the reason it cannot be
carried out. `propose_plan` returns that verdict under `translation`.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
