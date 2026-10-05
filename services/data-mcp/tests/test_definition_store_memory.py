"""Tests for `InMemoryDefinitionStore`: round-trip, miss, independence across instances, and first-write-wins."""

from __future__ import annotations

from data_mcp.definition_store.memory import InMemoryDefinitionStore
from data_mcp.definition_store.protocol import StoredDefinition

_DEFINITION = StoredDefinition(
    spec_id="spec_" + "a" * 32, schema_version=1, canonical_bytes=b'{"identity_version":"spec1"}'
)


async def test_get_on_an_empty_store_is_a_genuine_miss() -> None:
    store = InMemoryDefinitionStore()

    result = await store.get(_DEFINITION.spec_id, schema_version=_DEFINITION.schema_version)

    assert result is None


async def test_put_many_then_get_round_trips_the_stored_definition() -> None:
    store = InMemoryDefinitionStore()

    await store.put_many([_DEFINITION])
    result = await store.get(_DEFINITION.spec_id, schema_version=_DEFINITION.schema_version)

    assert result == _DEFINITION


async def test_get_is_still_a_miss_under_a_different_schema_version() -> None:
    store = InMemoryDefinitionStore()

    await store.put_many([_DEFINITION])
    result = await store.get(_DEFINITION.spec_id, schema_version=_DEFINITION.schema_version + 1)

    assert result is None


async def test_aclose_is_a_no_op_and_leaves_stored_definitions_readable() -> None:
    store = InMemoryDefinitionStore()
    await store.put_many([_DEFINITION])

    await store.aclose()

    result = await store.get(_DEFINITION.spec_id, schema_version=_DEFINITION.schema_version)
    assert result == _DEFINITION


async def test_the_first_write_of_a_key_wins() -> None:
    """A later write of a held `(spec_id, schema_version)` is dropped, as on GCS.

    The two definitions carry different bytes only to tell them apart: a real
    second write carries the same identity, since the spec_id is its hash.
    """
    store = InMemoryDefinitionStore()
    later = StoredDefinition(
        spec_id=_DEFINITION.spec_id,
        schema_version=_DEFINITION.schema_version,
        canonical_bytes=b'{"identity_version":"later"}',
    )

    await store.put_many([_DEFINITION])
    await store.put_many([later])

    assert await store.get(_DEFINITION.spec_id, schema_version=1) == _DEFINITION


async def test_the_first_entry_of_a_key_in_one_batch_wins() -> None:
    store = InMemoryDefinitionStore()
    later = StoredDefinition(
        spec_id=_DEFINITION.spec_id,
        schema_version=_DEFINITION.schema_version,
        canonical_bytes=b'{"identity_version":"later"}',
    )

    await store.put_many([_DEFINITION, later])

    assert await store.get(_DEFINITION.spec_id, schema_version=1) == _DEFINITION
