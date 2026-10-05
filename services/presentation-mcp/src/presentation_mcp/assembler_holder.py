"""The process-wide `Assembler` the tools call.

Mirrors `data_mcp.client_holder`, with one difference: there is no lazy
default. The lifespan is the only writer, calling `set_assembler` once
`release()` has proven the platform is reachable and admissible, and only
then registering the tools that call `get_assembler`. `get_assembler`
raising when nothing has been set is fail-closed and unreachable in normal
operation -- no tool exists for a caller to invoke before that window opens
or after shutdown clears it.
"""

from __future__ import annotations

from presentation_mcp.platform_contract import Assembler


class _AssemblerHolder:
    def __init__(self) -> None:
        self._assembler: Assembler | None = None

    def set(self, assembler: Assembler | None) -> None:
        self._assembler = assembler

    def get(self) -> Assembler:
        if self._assembler is None:
            msg = "no assembler is wired; the lifespan sets one before any tool is registered"
            raise RuntimeError(msg)
        return self._assembler


_holder = _AssemblerHolder()


def get_assembler() -> Assembler:
    """Return the process-wide assembler wired by the lifespan.

    Raises `RuntimeError` if called before the lifespan has wired one, or
    after shutdown has cleared it.
    """
    return _holder.get()


def set_assembler(assembler: Assembler | None) -> None:
    """Set (or clear) the process-wide assembler. Called only by the lifespan."""
    _holder.set(assembler)
