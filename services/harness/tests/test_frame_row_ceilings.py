"""The frame row ceiling the harness's frame-holding pieces share: data-mcp bounds a
fetched artifact, presentation-mcp a frame it publishes, and the durable store what it
writes and serves back. `common.settings.BaseServiceSettings.frame_rows_in_flight_max`
and `scripts/bench_frames.py` are derived from this one figure."""

from __future__ import annotations

from common.materialisation import gcs_store
from data_mcp.settings import Settings as DataMcpSettings
from presentation_mcp.settings import Settings as PresentationMcpSettings


def test_data_mcp_presentation_mcp_and_the_store_share_one_row_ceiling() -> None:
    ceilings = {
        "data-mcp max_frame_rows": DataMcpSettings.model_fields["max_frame_rows"].default,
        "presentation-mcp max_frame_rows": PresentationMcpSettings.model_fields[
            "max_frame_rows"
        ].default,
        "gcs_store._DEFAULT_MAX_FRAME_ROWS": gcs_store._DEFAULT_MAX_FRAME_ROWS,
    }

    assert len(set(ceilings.values())) == 1, ceilings
