"""The pair every surface in this platform calls a frame reference."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# The pairing of `spec_id` and `content_version` is not enforced: an
# internally inconsistent pair never matches a stored key, so a caller
# resolving it reaches the ordinary `frame_not_held` refusal, not a schema
# error. `MaterialisationStore` keys its entries by the plain tuple
# `(ref.spec_id, ref.content_version)`, not by this object, so hashability
# is not needed and the model stays unfrozen, like
# `analysis_mcp.op_spec.DatasetSnapshotRef`.
class FrameRef(BaseModel):
    """Names one frame by the `{spec_id, content_version}` pair a tool reply gives.

    A data or analysis tool's reply gives both values. `spec_id` is a dataset's
    `spec_` id or an analysis result's `ops_` id, and `content_version` is the
    matching `cv_` or `ocv_` value.
    """

    model_config = ConfigDict(extra="forbid")

    spec_id: str = Field(pattern=r"^(?:spec|ops)_[0-9a-f]{32}$")
    content_version: str = Field(pattern=r"^(?:cv|ocv)_[0-9a-f]{32}$")
