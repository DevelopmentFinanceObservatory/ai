"""Tests for `CachingDefinitionStore`: positive-only caching, its LRU bound, and
concurrent cold-key reads -- exercised against `InMemoryDefinitionStore`, with
no mock at all, since the decorator shape means correctness holds regardless
of what `inner` is.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from data_mcp.definition_store.caching import CachingDefinitionStore
from data_mcp.definition_store.memory import InMemoryDefinitionStore
from data_mcp.definition_store.protocol import DefinitionStore, StoredDefinition


def _definition(n: int) -> StoredDefinition:
    return StoredDefinition(
        spec_id=f"spec_{n:032d}", schema_version=1, canonical_bytes=f'{{"n":{n}}}'.encode()
    )


class _CountingStore:
    """Wraps a `DefinitionStore`, counting `get()` calls and yielding control
    mid-call so concurrent callers can genuinely interleave."""

    def __init__(self, inner: DefinitionStore) -> None:
        self._inner = inner
        self.get_calls = 0

    async def put_many(self, definitions: Sequence[StoredDefinition]) -> None:
        await self._inner.put_many(definitions)

    async def get(self, spec_id: str, *, schema_version: int) -> StoredDefinition | None:
        self.get_calls += 1
        await asyncio.sleep(0)
        return await self._inner.get(spec_id, schema_version=schema_version)

    async def aclose(self) -> None:
        await self._inner.aclose()


async def test_a_hit_is_cached_and_does_not_reach_inner_again() -> None:
    counting = _CountingStore(InMemoryDefinitionStore())
    store = CachingDefinitionStore(counting)
    definition = _definition(1)
    await store.put_many([definition])

    first = await store.get(definition.spec_id, schema_version=1)
    second = await store.get(definition.spec_id, schema_version=1)

    assert first == definition
    assert second == definition
    assert counting.get_calls == 1


async def test_a_miss_is_never_cached() -> None:
    counting = _CountingStore(InMemoryDefinitionStore())
    store = CachingDefinitionStore(counting)
    spec_id = "spec_" + "0" * 32

    first = await store.get(spec_id, schema_version=1)
    assert first is None
    assert counting.get_calls == 1

    # The id lands in the backend moments later -- a cached None must not
    # shadow it.
    await counting.put_many(
        [StoredDefinition(spec_id=spec_id, schema_version=1, canonical_bytes=b"{}")]
    )
    second = await store.get(spec_id, schema_version=1)

    assert second is not None
    assert counting.get_calls == 2


async def test_concurrent_cold_key_reads_both_return_the_correct_value() -> None:
    counting = _CountingStore(InMemoryDefinitionStore())
    store = CachingDefinitionStore(counting)
    definition = _definition(2)
    await store.put_many([definition])

    first, second = await asyncio.gather(
        store.get(definition.spec_id, schema_version=1),
        store.get(definition.spec_id, schema_version=1),
    )

    assert first == definition
    assert second == definition
    # No lock coordinates concurrent misses -- both may reach `inner`.
    assert counting.get_calls in (1, 2)


async def test_the_cache_is_bounded_and_evicts_least_recently_used() -> None:
    store = CachingDefinitionStore(InMemoryDefinitionStore(), max_entries=2)
    definitions = [_definition(n) for n in range(3)]
    await store.put_many(definitions)

    # Warm entries 0 and 1, in order, then insert a third -- 0 is now the
    # least recently used and should be the one evicted.
    await store.get(definitions[0].spec_id, schema_version=1)
    await store.get(definitions[1].spec_id, schema_version=1)
    await store.get(definitions[2].spec_id, schema_version=1)

    assert len(store._cache) == 2
    assert (definitions[0].spec_id, 1) not in store._cache
    assert (definitions[1].spec_id, 1) in store._cache
    assert (definitions[2].spec_id, 1) in store._cache


async def test_a_hit_stays_cached_across_a_later_write_of_its_key() -> None:
    """A hit is immutable, since the spec_id is its identity's hash, so a later
    write of the same key, from this instance or another, leaves it cached."""
    counting = _CountingStore(InMemoryDefinitionStore())
    store = CachingDefinitionStore(counting)
    definition = _definition(4)
    await store.put_many([definition])
    assert await store.get(definition.spec_id, schema_version=1) == definition

    await store.put_many([definition])

    assert await store.get(definition.spec_id, schema_version=1) == definition
    assert counting.get_calls == 1


async def test_put_many_and_aclose_delegate_to_inner() -> None:
    inner = InMemoryDefinitionStore()
    store = CachingDefinitionStore(inner)
    definition = _definition(3)

    await store.put_many([definition])
    assert await inner.get(definition.spec_id, schema_version=1) == definition

    await store.aclose()  # does not raise; InMemoryDefinitionStore.aclose is a no-op
