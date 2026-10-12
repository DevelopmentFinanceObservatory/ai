"""Record-and-replay for the Parquet frame codec: every frame in
`fixtures/frame_codec_differential.jsonl` goes through
`frame_to_parquet_bytes` then `frame_from_verified_parquet_bytes`, and the
decoded frame's `frame_sha256` must equal the recorded one.

The fixture pins the decoded frame, not the Parquet bytes, so a new on-disk
layout passes as long as it round-trips every frame to the same bytes.

Cases are seeded (`_SEED`) and cover roleless and multi-role frames, string,
converted (`UnitRequest`) and absent units in one column, facets, non-ASCII
refs, extreme doubles and an empty frame. Record the fixture again with
`uv run --package common python packages/common/tests/test_frame_codec_differential.py`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
from common.materialisation.frame_codec import (
    frame_from_verified_parquet_bytes,
    frame_to_parquet_bytes,
)
from dfo_frame_grammar import STATUS_COLUMN, Frame, canonical_frame_bytes, frame_sha256

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "frame_codec_differential.jsonl"
_SEED = 20260930

_UNITS: tuple[object, ...] = (
    "dcid:USDollar",
    "dcid:Percent",
    None,
    {"currency": "USD", "prices": "current"},
    {"currency": "EUR", "prices": "constant", "base_year": 2020},
    {"currency": "GBP", "prices": "constant", "base_year": -9007199254740991},
)
_EXTREMES = (
    5e-324,
    2.2250738585072014e-308,
    1.7976931348623157e308,
    -1.7976931348623157e308,
    -0.0,
    0.1 + 0.2,
    1 / 3,
    9007199254740993,
    -123456789.98765432,
)
_REFS = ("dcid:Kenya", "dcid:Côte_d'Ivoire", "dcid:Café", "dcid:日本", "dcid:Ωmega")


def _spec(frame: Frame) -> dict[str, Any]:
    document = json.loads(canonical_frame_bytes(frame))
    status = frame.columns.index(STATUS_COLUMN)
    return {
        "role_keys": list(frame.columns[1 : frame.columns.index("time_period")]),
        "columns": [column for column in frame.columns if column != STATUS_COLUMN],
        "rows": [row[:status] + row[status + 1 :] for row in document["rows"]],
    }


def _frame(spec: dict[str, Any]) -> Frame:
    return Frame.build(
        role_keys=spec["role_keys"],
        rows=[dict(zip(spec["columns"], row, strict=True)) for row in spec["rows"]],
    )


def _random_frame(rng: random.Random, *, roles: tuple[str, ...], row_count: int) -> Frame:
    rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    while len(rows) < row_count:
        row: dict[str, object] = {
            "stat_var": rng.choice(("dcid:StatVarA", "dcid:StatVarB")),
            **{role: rng.choice(_REFS) for role in roles},
            "time_period": rng.choice(("2019", "2020", "2021-03", "2022-12-31")),
            "value": rng.choice(_EXTREMES)
            if rng.random() < 0.2
            else rng.uniform(-1.0, 1.0) * 10.0 ** rng.randint(-8, 14),
            "unit": rng.choice(_UNITS),
            "measurementMethod": rng.choice((None, None, "dcid:MethodA")),
            "provenance": rng.choice((None, "dcid:Provenance1", "dcid:Provenance2")),
        }
        grain = tuple(
            json.dumps(cell, sort_keys=True) for key, cell in row.items() if key != "value"
        )
        if grain in seen:
            continue
        seen.add(grain)
        rows.append(row)
    return Frame.build(role_keys=roles, rows=rows)


def generate_cases() -> list[dict[str, Any]]:
    """The seeded input frames, in canonical JSON form."""
    rng = random.Random(_SEED)
    frames = [
        Frame.build(role_keys=["donor"], rows=[]),
        Frame.build(role_keys=[], rows=[]),
        Frame.build(
            role_keys=[],
            rows=[
                {"stat_var": "dcid:StatVarA", "time_period": "2020", "value": value}
                for value in (-0.0,)
            ],
        ),
    ]
    shapes = [(), ("donor",), ("donor", "recipient"), ("zone", "Área", "donor")]
    frames.extend(
        _random_frame(rng, roles=shapes[index % len(shapes)], row_count=rng.randint(1, 40))
        for index in range(24)
    )
    return [
        {"id": f"codec-{index:03d}", "frame": _spec(frame)} for index, frame in enumerate(frames)
    ]


def _roundtrip_sha256(frame: Frame) -> str:
    return frame_sha256(frame_from_verified_parquet_bytes(frame_to_parquet_bytes(frame)))


def _load() -> list[dict[str, Any]]:
    """The recorded cases; empty before the first recording, which
    `test_generator_reproduces_fixture_inputs` then fails on."""
    if not _FIXTURE_PATH.exists():
        return []
    return [json.loads(line) for line in _FIXTURE_PATH.read_text().splitlines() if line]


RECORDS = _load()


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
def test_codec_roundtrip_matches_recorded_frame(record: dict[str, Any]) -> None:
    frame = _frame(record["frame"])
    assert frame_sha256(frame) == record["input_frame_sha256"]
    assert _roundtrip_sha256(frame) == record["roundtrip_frame_sha256"]


def test_generator_reproduces_fixture_inputs() -> None:
    recorded = [{"id": record["id"], "frame": record["frame"]} for record in RECORDS]
    assert json.dumps(generate_cases()) == json.dumps(recorded)


def _record() -> None:
    cases = generate_cases()
    lines = []
    for case in cases:
        frame = _frame(case["frame"])
        record = {
            **case,
            "input_frame_sha256": frame_sha256(frame),
            "roundtrip_frame_sha256": _roundtrip_sha256(frame),
        }
        lines.append(json.dumps(record, separators=(",", ":")) + "\n")
    _FIXTURE_PATH.parent.mkdir(exist_ok=True)
    _FIXTURE_PATH.write_text("".join(lines))


if __name__ == "__main__":
    _record()
