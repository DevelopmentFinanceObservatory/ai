"""The compiler: turns a plan's translation into the workflow that runs it.

Code, not an agent. `compiler.compile` maps each operation the translator
proposed to a tool through the hand-maintained capability table, writes the
tool's arguments from the requirement and the source discovery selected,
binds each step to the outputs it reads, and checks the whole against the
live tool surface and the plan's requirements before anyone approves it.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
