"""Record-and-replay for the publish body: every frame in
`fixtures/frame_to_wire_differential.jsonl` goes through
`frame_publish.publish_body`, and the bytes it returns must equal httpx's
`json=` encoding of the publish body holding the recorded `role_keys` and
`rows`: row order, key order within each row, cell types (`1` against
`1.0`), float digits and the unit request objects.

Cases are seeded (`_SEED`). Record the fixture again with
`uv run --package presentation_mcp python services/presentation-mcp/tests/test_frame_to_wire_differential.py`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
from dfo_frame_grammar import STATUS_COLUMN, Frame, canonical_frame_bytes
from presentation_mcp import frame_publish
from presentation_mcp.frame_publish import publish_body

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "frame_to_wire_differential.jsonl"
_SEED = 20260930
_SPEC_ID = "ops_" + "a" * 32
_CONTENT_VERSION = "ocv_" + "b" * 32
_INPUT_CONTENT_VERSIONS = ("cv_" + "c" * 32, "cv_" + "d" * 32)

_UNITS: tuple[object, ...] = (
    "dcid:USDollar",
    None,
    {"currency": "USD", "prices": "current"},
    {"currency": "EUR", "prices": "constant", "base_year": 2020},
)
_REFS = ("dcid:Kenya", "dcid:Côte_d'Ivoire", "dcid:日本", "dcid:Ωmega", "dcid:Chad")


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
    rows: dict[str, dict[str, object]] = {}
    while len(rows) < row_count:
        row: dict[str, object] = {
            "stat_var": rng.choice(("dcid:StatVarA", "dcid:StatVarB")),
            **{role: rng.choice(_REFS) for role in roles},
            "time_period": rng.choice(("2019", "2020", "2021-03", "2022-12-31")),
            "value": rng.choice((0.0, 7.0, 1e21, 1e-7, -2.5, 0.1 + 0.2))
            if rng.random() < 0.3
            else rng.uniform(-1.0, 1.0) * 10.0 ** rng.randint(-8, 14),
            "unit": rng.choice(_UNITS),
            "measurementMethod": rng.choice((None, "dcid:MethodA")),
            "provenance": rng.choice((None, "dcid:Provenance1")),
        }
        grain = json.dumps({key: cell for key, cell in row.items() if key != "value"})
        rows.setdefault(grain, row)
    return Frame.build(role_keys=roles, rows=list(rows.values()))


def generate_cases() -> list[dict[str, Any]]:
    """The seeded input frames, in canonical JSON form."""
    rng = random.Random(_SEED)
    frames = [Frame.build(role_keys=["donor"], rows=[])]
    shapes = [(), ("donor",), ("recipient", "donor"), ("zone", "Área")]
    frames.extend(
        _random_frame(rng, roles=shapes[index % len(shapes)], row_count=rng.randint(1, 30))
        for index in range(16)
    )
    return [
        {"id": f"wire-{index:03d}", "frame": _spec(frame)} for index, frame in enumerate(frames)
    ]


def _body(frame: Frame) -> bytes:
    return publish_body(
        frame,
        spec_id=_SPEC_ID,
        content_version=_CONTENT_VERSION,
        input_content_versions=_INPUT_CONTENT_VERSIONS,
    )


def _httpx_body(wire: dict[str, Any]) -> bytes:
    """The publish body holding `wire`, encoded as httpx2 encodes a `json=` body."""
    document = {
        "spec_id": _SPEC_ID,
        "content_version": _CONTENT_VERSION,
        "role_keys": wire["role_keys"],
        "rows": wire["rows"],
        "input_content_versions": list(_INPUT_CONTENT_VERSIONS),
    }
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _load() -> list[dict[str, Any]]:
    """The recorded cases; empty before the first recording, which
    `test_generator_reproduces_fixture_inputs` then fails on."""
    if not _FIXTURE_PATH.exists():
        return []
    return [json.loads(line) for line in _FIXTURE_PATH.read_text().splitlines() if line]


RECORDS = _load()


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
def test_wire_body_matches_recorded_json(record: dict[str, Any]) -> None:
    assert _body(_frame(record["frame"])) == _httpx_body(record["wire"])


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
def test_wire_body_matches_recorded_json_across_slices(
    record: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three-row slices split every multi-row case, so this replays the
    joins between slices that the default slice size never reaches here."""
    monkeypatch.setattr(frame_publish, "_SLICE_ROWS", 3)
    assert _body(_frame(record["frame"])) == _httpx_body(record["wire"])


def test_generator_reproduces_fixture_inputs() -> None:
    recorded = [{"id": record["id"], "frame": record["frame"]} for record in RECORDS]
    assert json.dumps(generate_cases()) == json.dumps(recorded)


def _wire(body: bytes) -> dict[str, Any]:
    document = json.loads(body)
    return {"role_keys": document["role_keys"], "rows": document["rows"]}


def _record() -> None:
    lines = [
        json.dumps({**case, "wire": _wire(_body(_frame(case["frame"])))}, separators=(",", ":"))
        + "\n"
        for case in generate_cases()
    ]
    _FIXTURE_PATH.write_text("".join(lines))


if __name__ == "__main__":
    _record()
