"""The `harness-server` console script.

`multiprocessing`'s `spawn` start method runs the parent's `__main__` module
again in every worker process, and under `harness-server` that module is the
console script, which imports this one. presentation-mcp's chart worker is
such a process, so this module imports `harness.server` only when `main`
runs. Imported at the top, it would load the whole harness into the worker:
about 230 MB of resident memory in place of about 60.
"""

from __future__ import annotations


def main() -> None:
    """Serve the harness (`harness.server.main`)."""
    from harness.server import main as serve  # noqa: PLC0415 - kept out of spawned workers

    serve()
