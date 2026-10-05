"""A generic lazy-singleton holder: `get()` builds `default_factory()` on first use if nothing
has been `set()` yet; `aclose()` releases whatever is held and clears it, so a later `get()`
rebuilds rather than handing back a closed instance -- the shape every per-service holder in
this workspace already hand-copies (`data_mcp.client_holder`, `data_mcp.definition_store.holder`,
`presentation_mcp.assembler_holder`), each with an independent copy of this class's logic.
New holders should build on this one instead of adding a fourth hand-copy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class _Closeable(Protocol):
    async def aclose(self) -> None: ...


class Holder[T: _Closeable]:
    """Module-level holder for one process-wide `T`, built lazily from `default_factory`
    on first `get()` if nothing was `set()` first."""

    def __init__(self, *, default_factory: Callable[[], T]) -> None:
        self._default_factory = default_factory
        self._value: T | None = None

    def set(self, value: T | None) -> None:
        self._value = value

    def get(self) -> T:
        if self._value is None:
            self._value = self._default_factory()
        return self._value

    def is_set(self) -> bool:
        """Whether a value has already been `set()`, without lazily building the
        default -- `get()` would build and keep one, which is exactly what a caller
        checking this wants to avoid."""
        return self._value is not None

    async def aclose(self) -> None:
        # Checked directly rather than routed through `get()`, so shutdown never lazily
        # builds a value no caller ever used. Cleared after closing, so a later `get()`
        # rebuilds instead of handing back the closed instance.
        if self._value is not None:
            await self._value.aclose()
            self._value = None
