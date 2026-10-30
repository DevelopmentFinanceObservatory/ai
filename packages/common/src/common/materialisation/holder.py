"""The process-wide materialisation store shared by every analysis-mcp and
data-mcp tool. Built on `common.holder.Holder`, giving every consumer the
same `get_store`/`set_store`/`aclose_store` module-function surface.
`frame_mint_record` reads a frame's mint record from that store, and `dataset_pair` maps
the frame to the dataset pair it names."""

from __future__ import annotations

import logging

from pydantic import ValidationError

from common.holder import Holder
from common.materialisation.frame_ref import FrameRef
from common.materialisation.memory_store import InMemoryMaterialisationStore
from common.materialisation.mint_record import MintRecord
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


async def frame_mint_record(
    spec_id: str, content_version: str, *, logger: logging.Logger
) -> MintRecord | None:
    """The mint record the process's store keeps for the frame `{spec_id, content_version}`,
    or `None`.

    `None` for an `ops_` pair, a value that is not a pair the store could hold, a pair with
    no record, and a lookup that fails for any reason, which is logged on `logger`, the
    caller's own. The record is written best effort, so a caller treats `None` as "send the
    pair as given". The store is read only for a `spec_` pair, so a call with any other
    pair builds no default store.
    """
    if not spec_id.startswith("spec_"):
        return None
    try:
        ref = FrameRef(spec_id=spec_id, content_version=content_version)
    except ValidationError:
        return None
    try:
        return await get_store().get_mint_record(ref)
    except Exception:
        # Broad on purpose: the store's failure modes belong to its backend (the
        # storage client, its auth and its transport), and the lookup only ever
        # improves a call.
        logger.warning(
            "mint record of %s/%s unreadable; sending the pair as given",
            spec_id,
            content_version,
            exc_info=True,
        )
        return None


async def dataset_pair(
    spec_id: str, content_version: str, *, logger: logging.Logger
) -> tuple[str, str]:
    """The pair apis holds for the frame `{spec_id, content_version}`.

    `get_data` reports the pair its frame is held under in the process's store. A fetch with
    `columns`, a `groupby` or a party-total cut is held under a pair of its own, which apis
    never stored. The frame's record (`frame_mint_record`) names the pair apis does hold
    (`MintRecord.dataset`): the rendering's own dataset version, or the version apis mints
    for a collapse or a cut. That pair is returned when the record names one; any other
    pair is returned as given.
    """
    record = await frame_mint_record(spec_id, content_version, logger=logger)
    if record is None:
        return spec_id, content_version
    return record.dataset.spec_id, record.dataset.content_version
