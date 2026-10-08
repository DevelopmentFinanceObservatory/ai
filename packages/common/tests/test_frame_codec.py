"""Round-trip tests for `frame_to_parquet_bytes` and two of its decoders: a
`Frame`'s own canonical bytes survive a trip through Parquet unchanged, the
verified decoder wraps only the canonical schema, and the validating decoder
refuses what `Frame.build` refuses.
"""

from __future__ import annotations

import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from common.materialisation.frame_codec import (
    frame_from_parquet_bytes,
    frame_from_verified_parquet_bytes,
    frame_to_parquet_bytes,
)
from dfo_frame_grammar import Frame, FrameError, canonical_frame_bytes


def _parquet(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


def _mixed_unit_frame() -> Frame:
    """Three rows over two role columns (`reporter`, `recipient`), one
    with `unit=None`, one with a bare unit ref, one with a converted
    `UnitRequest` unit -- every shape `frame_codec`'s own unit column has
    to tell apart."""
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
                "measurementMethod": "dcid:M1",
                "provenance": "dcid:P1",
            },
        ],
    )


def test_frame_round_trips_through_parquet_bytes_with_mixed_units() -> None:
    frame = _mixed_unit_frame()

    round_tripped = frame_from_parquet_bytes(frame_to_parquet_bytes(frame))

    assert canonical_frame_bytes(round_tripped) == canonical_frame_bytes(frame)


def test_frame_round_trips_at_fifty_thousand_rows() -> None:
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:OdaDisbursedAmount",
                "reporter": f"dcid:Reporter{i:06d}",
                "time_period": "2020",
                "value": float(i),
                "unit": "dcid:USDollar",
            }
            for i in range(50_000)
        ],
    )

    round_tripped = frame_from_parquet_bytes(frame_to_parquet_bytes(frame))

    assert canonical_frame_bytes(round_tripped) == canonical_frame_bytes(frame)


def test_verified_decoder_returns_the_written_frame() -> None:
    frame = _mixed_unit_frame()

    decoded = frame_from_verified_parquet_bytes(frame_to_parquet_bytes(frame, metadata={"k": b"v"}))

    assert decoded == frame
    assert decoded.table.schema.metadata is None


def test_verified_decoder_refuses_the_earlier_layout() -> None:
    earlier = pa.table(
        {
            "stat_var": ["dcid:OdaDisbursedAmount"],
            "time_period": ["2020"],
            "value": [1.0],
            "unit": [json.dumps("dcid:USDollar")],
        }
    )

    with pytest.raises(FrameError) as caught:
        frame_from_verified_parquet_bytes(_parquet(earlier))

    assert caught.value.category == "schema_violation"


def test_validating_decoder_refuses_a_duplicate_grain_the_verified_decoder_trusts() -> None:
    table = _mixed_unit_frame().table
    doubled = _parquet(pa.concat_tables([table.slice(0, 1), table.slice(0, 1)]))

    with pytest.raises(FrameError) as caught:
        frame_from_parquet_bytes(doubled)

    assert caught.value.category == "duplicate_row_key"
    assert frame_from_verified_parquet_bytes(doubled).num_rows == 2


@pytest.mark.parametrize(
    "unit",
    [
        pytest.param({"currency": "USD", "prices": "current", "extra": 1}, id="unknown-member"),
        pytest.param({"currency": "USD", "prices": "constant", "base_year": "2015"}, id="str-year"),
        pytest.param(5, id="number"),
        pytest.param("", id="empty-ref"),
        pytest.param({"numerator": "dcid:USDollar"}, id="composed-without-a-denominator"),
    ],
)
def test_earlier_layout_unit_refuses_as_frame_build_does(unit: object) -> None:
    row = {"stat_var": "dcid:OdaDisbursedAmount", "time_period": "2020", "value": 1.0}
    earlier = pa.table({**{key: [cell] for key, cell in row.items()}, "unit": [json.dumps(unit)]})
    with pytest.raises(FrameError) as expected:
        Frame.build(role_keys=[], rows=[{**row, "unit": unit}])

    with pytest.raises(FrameError) as caught:
        frame_from_parquet_bytes(_parquet(earlier))

    assert caught.value.category == expected.value.category
