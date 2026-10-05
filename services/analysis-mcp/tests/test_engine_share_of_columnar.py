"""`run_share_of` on `frame.table`: which overflowing sum it reports when
more than one (key, period) overflows."""

from __future__ import annotations

import pytest
from analysis_mcp.engine.share_of import ShareOfNonFiniteResultError, run_share_of
from dfo_frame_grammar import Frame

_NEAR_MAX = 1.0e308


def test_first_overflow_is_found_key_by_key_in_first_row_order() -> None:
    """Keys are summed in order of their first row, and each key's periods in
    order of theirs. Sector B's first row comes first, so its overflowing
    2011 sum is reported, although sector A's overflowing 2010 group starts
    on an earlier row than sector B's 2011 group."""
    frame = Frame.build(
        role_keys=["donor", "sector"],
        rows=[
            {"stat_var": "dcid:X", "donor": d, "sector": s, "time_period": t, "value": v}
            for d, s, t, v in [
                ("dcid:D1", "dcid:SB", "2010", 1.0),
                ("dcid:D2", "dcid:SA", "2010", _NEAR_MAX),
                ("dcid:D2", "dcid:SB", "2011", _NEAR_MAX),
                ("dcid:D3", "dcid:SA", "2010", _NEAR_MAX),
                ("dcid:D3", "dcid:SB", "2011", _NEAR_MAX),
            ]
        ],
    )
    assert [row[1:3] for row in frame.rows][:3] == [
        ("dcid:D1", "dcid:SB"),
        ("dcid:D2", "dcid:SA"),
        ("dcid:D2", "dcid:SB"),
    ]

    with pytest.raises(ShareOfNonFiniteResultError) as exc_info:
        run_share_of(
            frame, frame, numerator_entity={}, denominator_entity={}, numerator_by=["sector"]
        )

    assert exc_info.value.time_period == "2011"
    assert exc_info.value.reason == "overflow"
