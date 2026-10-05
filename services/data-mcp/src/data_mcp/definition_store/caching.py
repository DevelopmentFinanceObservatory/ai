"""`CachingDefinitionStore`: a bounded, positive-only cache decorator over any `DefinitionStore`."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

from data_mcp.definition_store.protocol import DefinitionStore, StoredDefinition

DEFAULT_MAX_ENTRIES = 4096


class CachingDefinitionStore:
    """Caches `inner.get()` hits; never caches a miss.

    A miss is a registration race, not a stable fact -- caching it would keep
    serving `None` for an id that lands in `inner` moments later. A hit is
    immutable by construction: the store holds the identity alone, and
    `spec_id` is its hash, so caching it can never go stale.

    Bounded by `max_entries`, evicting the least-recently-used entry once
    full. Entries are immutable, so two concurrent cold-key reads for the
    same key may both call through to `inner` rather than coordinating
    through a lock; both still return the same, correct value.
    """

    def __init__(self, inner: DefinitionStore, *, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._inner = inner
        self._max_entries = max_entries
        self._cache: OrderedDict[tuple[str, int], StoredDefinition] = OrderedDict()

    async def put_many(self, definitions: Sequence[StoredDefinition]) -> None:
        await self._inner.put_many(definitions)

    async def get(self, spec_id: str, *, schema_version: int) -> StoredDefinition | None:
        key = (spec_id, schema_version)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        found = await self._inner.get(spec_id, schema_version=schema_version)
        if found is not None:
            self._cache[key] = found
            self._cache.move_to_end(key)
            if len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
        return found

    async def aclose(self) -> None:
        await self._inner.aclose()
