"""`frame_to_parquet_bytes`/`frame_from_parquet_bytes` preserve
`derive_frame_content_version`'s own output, not just a frame's canonical
bytes -- the identity a durable store's read path must hand back
unchanged to the next hop that resolves this frame as its own input.
"""

from __future__ import annotations

from analysis_mcp.frame_ref_resolution import derive_frame_content_version
from common.materialisation.frame_codec import frame_from_parquet_bytes, frame_to_parquet_bytes
from dfo_frame_grammar import Frame

_UPSTREAM_TRACE = {"graph_revision": "rev1", "source_revisions": [], "facets": []}


def _mixed_unit_frame() -> Frame:
    """Two role columns (`reporter`, `recipient`) and one row apiece with
    `unit=None`, a bare unit ref, and a converted `UnitRequest` unit."""
    return Frame.build(
        role_keys=["reporter", "recipient"],
        rows=[
            {
                "stat_var": "dcid:OdaDisbursedAmount",
                "reporter": "dcid:USA",
                "recipient": "dcid:BGD",
                "time_period": "2020",
                "value": 100.5,
                "unit": None,
            },
            {
                "stat_var": "dcid:OdaDisbursedAmount",
                "reporter": "dcid:GBR",
                "recipient": "dcid:HTI",
                "time_period": "2020",
                "value": 200.25,
                "unit": "dcid:USDollar",
            },
            {
                "stat_var": "dcid:OdaDisbursedAmount",
                "reporter": "dcid:FRA",
                "recipient": "dcid:BGD",
                "time_period": "2021",
                "value": 300.75,
                "unit": {"currency": "USD", "prices": "constant", "base_year": 2015},
            },
        ],
    )


def test_derive_frame_content_version_matches_after_a_parquet_round_trip() -> None:
    frame = _mixed_unit_frame()
    round_tripped = frame_from_parquet_bytes(frame_to_parquet_bytes(frame))

    original = derive_frame_content_version(frame=frame, upstream_trace=_UPSTREAM_TRACE)
    after_round_trip = derive_frame_content_version(
        frame=round_tripped, upstream_trace=_UPSTREAM_TRACE
    )

    assert after_round_trip == original
