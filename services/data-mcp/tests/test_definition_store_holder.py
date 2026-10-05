"""Tests for the process-wide definition store holder: lazy default, explicit
wiring, and rebuild-after-close -- mirrors `test_client_holder.py`.
"""

from __future__ import annotations

from typing import Any

import pytest
from data_mcp.definition_store import holder
from data_mcp.definition_store.caching import CachingDefinitionStore
from data_mcp.definition_store.memory import InMemoryDefinitionStore
from data_mcp.definition_store.protocol import StoredDefinition


@pytest.fixture(autouse=True)
async def _reset_store() -> Any:
    """Every test wires its own store state; nothing should leak to the next test."""
    yield
    holder.set_store(None)


async def test_get_store_lazily_builds_a_caching_wrapped_in_memory_store() -> None:
    store = holder.get_store()

    assert isinstance(store, CachingDefinitionStore)
    assert isinstance(store._inner, InMemoryDefinitionStore)
    # Cached, not rebuilt, on a second call.
    assert holder.get_store() is store


async def test_set_store_preempts_the_lazy_default() -> None:
    explicit = CachingDefinitionStore(InMemoryDefinitionStore())

    holder.set_store(explicit)

    assert holder.get_store() is explicit


async def test_a_store_wired_through_the_holder_actually_holds_state() -> None:
    store = holder.get_store()
    definition = StoredDefinition(
        spec_id="spec_" + "1" * 32, schema_version=1, canonical_bytes=b"{}"
    )

    await store.put_many([definition])
    result = await holder.get_store().get(definition.spec_id, schema_version=1)

    assert result == definition


async def test_aclose_store_closes_and_clears_so_a_later_get_rebuilds() -> None:
    first = holder.get_store()

    await holder.aclose_store()

    second = holder.get_store()
    assert second is not first


async def test_aclose_store_without_a_prior_get_is_a_no_op() -> None:
    await holder.aclose_store()  # does not raise even though nothing was ever built
