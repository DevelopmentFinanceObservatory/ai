"""`common.materialisation.holder`: confirms it wires `Holder[MaterialisationStore]`
correctly -- not a re-test of `Holder` itself (see `test_holder.py` for that)."""

from __future__ import annotations

from common.materialisation import InMemoryMaterialisationStore
from common.materialisation.holder import aclose_store, get_store, set_store


async def test_get_store_with_nothing_set_lazily_builds_an_in_memory_store() -> None:
    store = get_store()

    assert isinstance(store, InMemoryMaterialisationStore)
    assert get_store() is store


async def test_set_store_then_get_store_returns_the_set_store() -> None:
    fake = InMemoryMaterialisationStore()

    set_store(fake)

    assert get_store() is fake


async def test_aclose_store_closes_the_set_store_and_clears_the_holder() -> None:
    fake = InMemoryMaterialisationStore()
    set_store(fake)

    await aclose_store()

    rebuilt = get_store()
    assert rebuilt is not fake
    assert isinstance(rebuilt, InMemoryMaterialisationStore)


async def test_aclose_store_twice_in_a_row_is_a_no_op() -> None:
    await aclose_store()
    await aclose_store()
