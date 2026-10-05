"""The process-wide materialisation store shared by every analysis-mcp and
data-mcp tool. Built on `common.holder.Holder`, giving every consumer the
same `get_store`/`set_store`/`aclose_store` module-function surface."""

from __future__ import annotations

from common.holder import Holder
from common.materialisation.memory_store import InMemoryMaterialisationStore
from common.materialisation.store import MaterialisationStore

_holder: Holder[MaterialisationStore] = Holder(default_factory=InMemoryMaterialisationStore)


def get_store() -> MaterialisationStore:
    """Return the process-wide materialisation store, building an unseeded default lazily."""
    return _holder.get()


def set_store(store: MaterialisationStore | None) -> None:
    """Set (or clear) the materialisation store every caller resolves through."""
    _holder.set(store)


def store_is_set() -> bool:
    """Whether a store is already installed, without lazily building the default one
    `get_store()` would. `factory.install_store_if_unset` is the only caller: it needs
    to tell "nothing installed yet" apart from "the lazy default already exists" before
    deciding whether to install its own."""
    return _holder.is_set()


async def aclose_store() -> None:
    """Close the materialisation store, if one was ever set or lazily built."""
    await _holder.aclose()
