"""The process-wide definition store shared by every data-mcp tool.

Built lazily on first use, mirroring `data_mcp.client_holder`: the deployed
topology mounts data-mcp's FastMCP app in-process and never calls
`data_mcp.server.main()`, so nothing would otherwise wire a store. A real
deployment's `server.py` lifespan calls `set_store()` with a store built from
config (GCS-backed when a bucket is configured, in-memory otherwise) before
any tool call can reach `get_store()`; tests do the same to pre-empt the lazy
default. `aclose_store()` is called from `server.py`'s lifespan on shutdown.

The lazy default itself is an in-memory store wrapped in
`CachingDefinitionStore`, so a caller that never wires a store still gets a
working one -- just not one any other process instance can see.
"""

from __future__ import annotations

from data_mcp.definition_store.caching import CachingDefinitionStore
from data_mcp.definition_store.memory import InMemoryDefinitionStore
from data_mcp.definition_store.protocol import DefinitionStore


class _StoreHolder:
    """Module-level holder for the definition store tools call."""

    def __init__(self) -> None:
        self._store: DefinitionStore | None = None

    def set(self, store: DefinitionStore | None) -> None:
        self._store = store

    def get(self) -> DefinitionStore:
        if self._store is None:
            self._store = CachingDefinitionStore(InMemoryDefinitionStore())
        return self._store

    async def aclose(self) -> None:
        # Checked directly rather than routed through `get()`, so shutdown
        # never lazily builds a store no call ever used. Cleared after
        # closing, so a later `get()` (a second server lifespan) rebuilds
        # instead of handing out the closed store.
        if self._store is not None:
            await self._store.aclose()
            self._store = None


_holder = _StoreHolder()


def get_store() -> DefinitionStore:
    """Return the process-wide definition store, building an in-memory default lazily."""
    return _holder.get()


def set_store(store: DefinitionStore | None) -> None:
    """Set (or clear) the definition store tools call, pre-empting the lazy default."""
    _holder.set(store)


async def aclose_store() -> None:
    """Close the definition store, if one was ever set or lazily built.

    Called from `server.py`'s lifespan on shutdown.
    """
    await _holder.aclose()
