"""Translation, run as code behind the planner's `translate` tool.

Finds the data a plan's requirements need, builds the operations that carry
them out, and has the compiler turn them into the workflow the user approves;
or refers one requirement back to the planner with the reason it cannot be
carried out.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
