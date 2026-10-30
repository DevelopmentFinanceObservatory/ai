"""The rendering a `spec_` frame holds, as its mint record states it.

A frame recovered at its Spec's newest version, or exported from the dataset version
its record names, must hold the rows the caller's frame holds. Those rows depend on
the frame's rendering: a `groupby` sums them and `without_party_totals` cuts them.
`get_data` records the rendering of every frame it reads, default arguments included
(`data_mcp.frame_materialisation.materialise_and_register`), so a pair with no record
is one whose rendering nobody can state: a mint that saw no frame, a record write that
ran out of deadline, a record the bucket's lifecycle rule deleted, or a pair `get_data`
never named.

`frame_rendering` is the one answer the recovering tools share:

- `FrameRendering`: the record stands, and it says how a root is read.
- `RenderingUnknown`: no record is held, the record does not decode, or it holds an
  option no root is read with. Each reads the same on every retry, and only `get_data`
  mends it.
- `RecordUnread`: the store failed the read, which a retry may mend.

A pair dfo-apis resolved a bare `spec_id` to is a dataset version read with the default
enrichment, which nobody fetched with `get_data`. `record_bare_read` writes its record,
so a later call naming that pair reads what a Spec root reads: the enrichment columns the
version's roles take, which only dfo-apis can resolve (`root_rendering`).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pydantic import ValidationError

from common.materialisation.frame_ref import FrameRef
from common.materialisation.holder import get_store
from common.materialisation.mint_record import MintRecord
from common.materialisation.store import FrameIntegrityError

# The mint option that asks apis to add the enrichment the version's own role columns take
# (apis' `MintOptions.enrichment`). apis answers as a mint naming the columns it resolved to,
# and the manifest's artifact lists them in its `rendering`, so a frame minted with it is
# named by them (`data_mcp.frame_materialisation.resolved_options`).
ENRICHMENT = "enrichment"
DEFAULT_ENRICHMENT = "default"

# The mint options a root is read with: the frame-layout Parquet, its enrichment
# columns, the party-total cut and the groupby of a breakdown or a sum (dfo-apis
# `analysis_roots.dataset_root`). `enrichment` is that of a bare read's record, which
# states no columns.
ROOT_READ_OPTIONS = frozenset(
    {"format", "bundle", "layout", "columns", ENRICHMENT, "without_party_totals", "groupby"}
)


# The `options` of a mint that sums nothing and selects no columns. Its frame is
# read from the version's canonical frame
# (`data_mcp.frame_materialisation.CANONICAL_FRAME_RENDERING`), and the requested
# rendering itself is never fetched. A mint cannot ask for the canonical frame alone,
# since apis' `MintOptions.format` is `csv`, `xlsx` or `parquet` and every mint renders
# one of them, so this asks for the cheapest: a bare parquet file in the frame layout,
# written with no projection and no graph name read. A mint with `columns` adds them
# to these options, and its frame is that rendering
# (`data_mcp.frame_materialisation._select_requested_artifact`).
DEFAULT_MINT_OPTIONS: Mapping[str, Any] = MappingProxyType(
    {"format": "parquet", "bundle": False, "layout": "frame"}
)

# How long a reply waits on the store for a best-effort record write.
_RECORD_WRITE_TIMEOUT_S = 2.0


def root_rendering(record: MintRecord) -> dict[str, Any]:
    """The rendering of a root read as `record`'s options state. A rendering with no
    `columns` takes the default enrichment (dfo-apis `RootRendering.columns`)."""
    rendering: dict[str, Any] = {
        "without_party_totals": record.options.get("without_party_totals") is True,
    }
    if record.options.get(ENRICHMENT) != DEFAULT_ENRICHMENT:
        # Sorted and listed once, as dfo-apis keeps a root's columns
        # (`analysis_roots.mint_options`), so the recipe's normal form is the one
        # dfo-apis computes.
        rendering["columns"] = sorted(set(record.options.get("columns") or ()))
    if (groupby := record.options.get("groupby")) is not None:
        rendering["groupby"] = groupby
    return rendering


def unread_options(record: MintRecord) -> list[str]:
    """The options of `record` no root is read with, sorted."""
    return sorted(set(record.options) - ROOT_READ_OPTIONS)


@dataclass(frozen=True, kw_only=True)
class FrameRendering:
    """The rendering `record` states for its frame."""

    record: MintRecord

    @property
    def rendering(self) -> dict[str, Any]:
        """The rendering a root of this frame is read with."""
        return root_rendering(self.record)

    @property
    def summed(self) -> bool:
        """Whether the frame's rows are a `groupby` sum."""
        return "groupby" in self.rendering

    @property
    def default_rows(self) -> bool:
        """Whether the frame holds the rows of a Spec read with no rendering. Its
        columns do not count: they add cells to the same rows."""
        return not self.summed and not self.rendering["without_party_totals"]


@dataclass(frozen=True, kw_only=True)
class RenderingUnknown:
    """No record states the frame's rendering. `options` are those of a record no root
    is read with, empty when no readable record is held."""

    options: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class RecordUnread:
    """The store failed a read of the frame's mint record."""


async def frame_rendering(
    ref: FrameRef, *, logger: logging.Logger
) -> FrameRendering | RenderingUnknown | RecordUnread:
    """The rendering of the `spec_` frame `ref`. See the module docstring. A failed read
    is logged on `logger`, the caller's own."""
    try:
        record = await get_store().get_mint_record(ref)
    except FrameIntegrityError as exc:
        logger.warning(
            "mint record of %s/%s undecodable: %s", ref.spec_id, ref.content_version, exc.reason
        )
        return RenderingUnknown()
    except Exception:
        # Broad on purpose, as `holder.frame_mint_record` reads it: the store's failures
        # belong to its backend.
        logger.warning(
            "mint record of %s/%s unreadable", ref.spec_id, ref.content_version, exc_info=True
        )
        return RecordUnread()
    if record is None:
        return RenderingUnknown()
    if unread := unread_options(record):
        return RenderingUnknown(options=tuple(unread))
    return FrameRendering(record=record)


async def write_mint_record(record: MintRecord, *, logger: logging.Logger) -> bool:
    """Write `record`, reporting whether the store took it within
    `_RECORD_WRITE_TIMEOUT_S`, for a reply that stands either way. A failed write is
    logged on `logger`, the caller's own."""
    try:
        async with asyncio.timeout(_RECORD_WRITE_TIMEOUT_S):
            await get_store().put_mint_record(record)
    except Exception:
        # Broad on purpose, as `frame_rendering` reads it: the store's failures belong
        # to its backend.
        logger.warning(
            "mint record of %s/%s unstored",
            record.frame.spec_id,
            record.frame.content_version,
            exc_info=True,
        )
        return False
    return True


async def record_bare_read(spec_id: str, content_version: str, *, logger: logging.Logger) -> None:
    """Write the mint record of the pair dfo-apis resolved a bare `spec_id` to, read with
    the default enrichment. A value that is no pair writes nothing. The store keeps the
    first record under a pair, so a record `get_data` wrote stands. No frame was read, so
    the record has no `stat_var_count`."""
    try:
        pair = FrameRef(spec_id=spec_id, content_version=content_version)
    except ValidationError:
        return
    options = {**DEFAULT_MINT_OPTIONS, ENRICHMENT: DEFAULT_ENRICHMENT}
    await write_mint_record(MintRecord(frame=pair, dataset=pair, options=options), logger=logger)
