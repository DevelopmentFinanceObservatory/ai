"""The materialisation seam shared by data-mcp and analysis-mcp: one
content-addressed store, keyed by the pair every surface in this platform
calls a frame reference."""

from __future__ import annotations

from common.materialisation.content_version import (
    ContentVersionDeriverUnregistered,
    get_content_version_deriver,
    register_content_version_deriver,
)
from common.materialisation.factory import build_store, install_store_if_unset
from common.materialisation.frame_codec import (
    frame_from_parquet_bytes,
    frame_from_verified_parquet_bytes,
    frame_to_parquet_bytes,
)
from common.materialisation.frame_ref import FrameRef
from common.materialisation.gcs_store import GcsMaterialisationStore
from common.materialisation.holder import aclose_store, get_store, set_store, store_is_set
from common.materialisation.memory_store import InMemoryMaterialisationStore
from common.materialisation.store import (
    FrameExpired,
    FrameIntegrityError,
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
    "ContentVersionDeriverUnregistered",
    "FrameExpired",
    "FrameIntegrityError",
    "FrameRef",
    "FrameTooLarge",
    "GcsMaterialisationStore",
    "HeldFrame",
    "InMemoryMaterialisationStore",
    "MaterialisationStore",
    "MaterialisationTrace",
    "StoredFrameTrace",
    "TraceCodecUnknown",
    "aclose_store",
    "build_store",
    "decode_trace",
    "encode_trace",
    "frame_from_parquet_bytes",
    "frame_from_verified_parquet_bytes",
    "frame_to_parquet_bytes",
    "get_content_version_deriver",
    "get_store",
    "install_store_if_unset",
    "register_content_version_deriver",
    "register_trace_codec",
    "set_store",
    "store_is_set",
]
