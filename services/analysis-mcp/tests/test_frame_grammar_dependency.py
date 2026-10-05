"""Pins the dfo-spec-kit-frame-grammar git dependency's presence and import surface."""

from __future__ import annotations

from dfo_frame_grammar import Frame


def test_frame_build_round_trip() -> None:
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:Test",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": 1,
            }
        ],
    )

    assert frame.columns == (
        "stat_var",
        "reporter",
        "time_period",
        "value",
        "unit",
        "measurementMethod",
        "provenance",
        "status",
    )
    assert frame.rows == (("dcid:Test", "dcid:USA", "2020", 1.0, None, None, None, None),)
