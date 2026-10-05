"""The pair every surface in this platform calls a frame reference."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FrameRef(BaseModel):
    """Names one materialised or computed frame: `{spec_id, content_version}`.

    `spec_id` is either a Datasets snapshot's `spec_` id or a toolkit
    result's `ops_` op_spec_hash; `content_version` is either
    frame-grammar's `cv_` family or the toolkit's own `ocv_` family. Both
    members must come from the same family, but this model does not enforce
    that pairing -- an internally inconsistent pair simply never matches a
    stored key, so a caller resolving it reaches the ordinary
    `frame_not_held` refusal, never a schema error.

    Mutable: `MaterialisationStore` keys its entries by the plain tuple
    `(ref.spec_id, ref.content_version)`, never by this object itself, so
    hashability isn't load-bearing -- matching the same unfrozen shape
    `analysis_mcp.op_spec.DatasetSnapshotRef` already uses.
    """

    model_config = ConfigDict(extra="forbid")

    spec_id: str = Field(pattern=r"^(?:spec|ops)_[0-9a-f]{32}$")
    content_version: str = Field(pattern=r"^(?:cv|ocv)_[0-9a-f]{32}$")
