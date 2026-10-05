"""The frame row ceiling the harness's three frame-holding pieces share: data-mcp bounds a
fetched artifact, analysis-mcp a resolved input, and the durable store what it writes and
serves back. `common.row_budget`'s default and `scripts/bench_frames.py` are derived from
this one figure."""

from __future__ import annotations

from analysis_mcp.settings import Settings as AnalysisMcpSettings
from common.materialisation import gcs_store
from data_mcp.settings import Settings as DataMcpSettings


def test_data_mcp_analysis_mcp_and_the_store_share_one_row_ceiling() -> None:
    ceilings = {
        "data-mcp max_frame_rows": DataMcpSettings.model_fields["max_frame_rows"].default,
        "analysis-mcp max_frame_rows": AnalysisMcpSettings.model_fields["max_frame_rows"].default,
        "gcs_store._DEFAULT_MAX_FRAME_ROWS": gcs_store._DEFAULT_MAX_FRAME_ROWS,
    }

    assert len(set(ceilings.values())) == 1, ceilings
