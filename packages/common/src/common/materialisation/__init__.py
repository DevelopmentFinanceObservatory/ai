"""The materialisation seam shared by data-mcp and analysis-mcp: one
content-addressed store, keyed by the pair every surface in this platform
calls a frame reference."""

from __future__ import annotations

from common.materialisation.anchor_record import (
    AnchorPopulation,
    AnchorRecord,
    AnchorRecordIntegrityError,
    LeftOutSeries,
    decode_anchor_record,
    encode_anchor_record,
)
from common.materialisation.content_version import (
    ContentVersionDeriverUnregistered,
    get_content_version_deriver,
    register_content_version_deriver,
)
from common.materialisation.factory import build_store, install_store_if_unset
from common.materialisation.frame_codec import (
    frame_from_verified_parquet_bytes,
    frame_to_parquet_bytes,
)
from common.materialisation.frame_ref import FrameRef
from common.materialisation.gcs_store import GcsMaterialisationStore
from common.materialisation.holder import (
    aclose_store,
    dataset_pair,
    frame_mint_record,
    get_store,
    set_store,
    store_is_set,
)
from common.materialisation.memory_store import InMemoryMaterialisationStore
from common.materialisation.mint_record import (
    MintRecord,
    MintRecordUnrecoverableError,
    decode_mint_record,
    encode_mint_record,
    groupby_axes,
    normal_groupby,
)
from common.materialisation.rendering import (
    DEFAULT_MINT_OPTIONS,
    FrameRendering,
    RecordUnread,
    RenderingUnknown,
    frame_rendering,
    record_bare_read,
    root_rendering,
    unread_options,
)
from common.materialisation.store import (
    FrameExpired,
    FrameIntegrityError,
    FrameListing,
    FrameTooLarge,
    HeldFrame,
    MaterialisationStore,
)
from common.materialisation.trace import MaterialisationTrace, StoredFrameTrace
from common.materialisation.trace_codec import (
    TraceCodecUnknown,
    decode_trace,
    encode_trace,
    register_trace_codec,
)

__all__ = [
    "DEFAULT_MINT_OPTIONS",
    "AnchorPopulation",
    "AnchorRecord",
    "AnchorRecordIntegrityError",
    "ContentVersionDeriverUnregistered",
    "FrameExpired",
    "FrameIntegrityError",
    "FrameListing",
    "FrameRef",
    "FrameRendering",
    "FrameTooLarge",
    "GcsMaterialisationStore",
    "HeldFrame",
    "InMemoryMaterialisationStore",
    "LeftOutSeries",
    "MaterialisationStore",
    "MaterialisationTrace",
    "MintRecord",
    "MintRecordUnrecoverableError",
    "RecordUnread",
    "RenderingUnknown",
    "StoredFrameTrace",
    "TraceCodecUnknown",
    "aclose_store",
    "build_store",
    "dataset_pair",
    "decode_anchor_record",
    "decode_mint_record",
    "decode_trace",
    "encode_anchor_record",
    "encode_mint_record",
    "encode_trace",
    "frame_from_verified_parquet_bytes",
    "frame_mint_record",
    "frame_rendering",
    "frame_to_parquet_bytes",
    "get_content_version_deriver",
    "get_store",
    "groupby_axes",
    "install_store_if_unset",
    "normal_groupby",
    "record_bare_read",
    "register_content_version_deriver",
    "register_trace_codec",
    "root_rendering",
    "set_store",
    "store_is_set",
    "unread_options",
]
