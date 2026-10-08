"""Record-and-replay for the publish body: every frame in
`fixtures/frame_to_wire_differential.jsonl` goes through
`frame_publish.publish_body`, and the bytes it returns must equal httpx's
`json=` encoding of the publish body holding the recorded `role_keys` and
`rows`: row order, key order within each row, cell types (`1` against
`1.0`), float digits and the unit objects. A case that names composed
StatVars records the `composed_stat_vars` member too.

Cases are seeded (`_SEED`). One draws its units from `_COMPOSED_UNITS`, and
the cases after it carry `_COMPOSED_STAT_VARS` lists. Record the fixture again with
`uv run --package presentation_mcp python services/presentation-mcp/tests/test_frame_to_wire_differential.py`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
from common.materialisation import ComposedStatVar
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
# Drawn after every `_UNITS` case, so those cases keep their seeded rows.
_COMPOSED_UNITS: tuple[object, ...] = (
    {"numerator": "dcid:USDollar", "denominator": {"stat_var": "dcid:Count_Person"}},
    {
        "numerator": {"currency": "EUR", "prices": "constant", "base_year": 2020},
        "denominator": "dcid:SquareKilometer",
    },
    {"numerator": {"stat_var": "dcid:Count_Person"}, "denominator": "dcid:SquareKilometer"},
    "dcid:USDollar",
    None,
)
# Drawn after every other case. The second list nests: its sum names the
# quotient before it.
_COMPOSED_STAT_VARS: tuple[list[dict[str, Any]], ...] = (
    [
        {
            "stat_var": "dcid:StatVarA",
            "relation": "quotient",
            "first": ["dcid:Amount_EconomicActivity_GrossDomesticProduction_Nominal"],
            "second": ["dcid:Count_Person"],
        }
    ],
    [
        {
            "stat_var": "dcid:StatVarA",
            "relation": "quotient",
            "first": ["dcid:Dépense_Santé"],
            "second": ["dcid:Count_Person"],
        },
        {
            "stat_var": "dcid:StatVarB",
            "relation": "sum",
            "first": ["dcid:StatVarA", "dcid:人口"],
            "second": ["dcid:StatVarA"],
        },
    ],
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


def _random_frame(
    rng: random.Random,
    *,
    roles: tuple[str, ...],
    row_count: int,
    units: tuple[object, ...] = _UNITS,
) -> Frame:
    rows: dict[str, dict[str, object]] = {}
    while len(rows) < row_count:
        row: dict[str, object] = {
            "stat_var": rng.choice(("dcid:StatVarA", "dcid:StatVarB")),
            **{role: rng.choice(_REFS) for role in roles},
            "time_period": rng.choice(("2019", "2020", "2021-03", "2022-12-31")),
            "value": rng.choice((0.0, 7.0, 1e21, 1e-7, -2.5, 0.1 + 0.2))
            if rng.random() < 0.3
            else rng.uniform(-1.0, 1.0) * 10.0 ** rng.randint(-8, 14),
            "unit": rng.choice(units),
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
    frames.append(_random_frame(rng, roles=("donor",), row_count=12, units=_COMPOSED_UNITS))
    cases: list[dict[str, Any]] = [{"frame": _spec(frame)} for frame in frames]
    cases.extend(
        {
            "frame": _spec(_random_frame(rng, roles=("donor",), row_count=6)),
            "composed_stat_vars": entries,
        }
        for entries in _COMPOSED_STAT_VARS
    )
    return [{"id": f"wire-{index:03d}", **case} for index, case in enumerate(cases)]


def _body(case: dict[str, Any]) -> bytes:
    return publish_body(
        _frame(case["frame"]),
        spec_id=_SPEC_ID,
        content_version=_CONTENT_VERSION,
        input_content_versions=_INPUT_CONTENT_VERSIONS,
        composed_stat_vars=[
            ComposedStatVar(**entry) for entry in case.get("composed_stat_vars", ())
        ],
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
    if "composed_stat_vars" in wire:
        document["composed_stat_vars"] = wire["composed_stat_vars"]
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
    assert _body(record) == _httpx_body(record["wire"])


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
def test_wire_body_matches_recorded_json_across_slices(
    record: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three-row slices split every multi-row case, so this replays the
    joins between slices that the default slice size never reaches here."""
    monkeypatch.setattr(frame_publish, "_SLICE_ROWS", 3)
    assert _body(record) == _httpx_body(record["wire"])


def test_generator_reproduces_fixture_inputs() -> None:
    recorded = [
        {key: value for key, value in record.items() if key != "wire"} for record in RECORDS
    ]
    assert json.dumps(generate_cases()) == json.dumps(recorded)


def _entry(index: int, *, sides: int = 1, ref: str = "dcid:Count_Person") -> ComposedStatVar:
    return ComposedStatVar(
        stat_var=f"dcid:StatVar{index}", relation="sum", first=(ref,) * sides, second=(ref,)
    )


@pytest.mark.parametrize(
    "entries",
    [
        pytest.param([_entry(index) for index in range(16)], id="16-entries"),
        pytest.param([_entry(0, sides=63)], id="64-side-refs"),
        pytest.param([_entry(0, ref="dcid:" + "x" * 195)], id="200-character-ref"),
    ],
)
def test_a_list_at_a_door_bound_is_sent(entries: list[ComposedStatVar]) -> None:
    case = {**RECORDS[1], "composed_stat_vars": [entry.model_dump() for entry in entries]}

    document = json.loads(_body(case))

    assert len(document["composed_stat_vars"]) == len(entries)


@pytest.mark.parametrize(
    "entries",
    [
        pytest.param([], id="no-entries"),
        pytest.param([_entry(index) for index in range(17)], id="17-entries"),
        pytest.param([_entry(0, sides=64)], id="65-side-refs"),
        pytest.param([_entry(0, ref="dcid:" + "x" * 196)], id="201-character-ref"),
        pytest.param(
            [ComposedStatVar(stat_var="dcid:" + "x" * 196, relation="sum", first=(), second=())],
            id="201-character-stat-var",
        ),
        pytest.param([_entry(0, sides=0)], id="empty-side"),
        # 395 characters as written; NFC composes each "e" and its accent, leaving 200.
        pytest.param([_entry(0, ref="dcid:" + "e\u0301" * 195)], id="over-200-raw-200-after-nfc"),
        # 105 characters as written; NFC splits each letter in two, leaving 205.
        pytest.param([_entry(0, ref="dcid:" + "\u0958" * 100)], id="under-200-raw-205-after-nfc"),
        pytest.param([_entry(0, ref="dcid:Count\x00Person")], id="nul-in-a-ref"),
        pytest.param([_entry(0, ref="Count_Person")], id="ref-without-the-dcid-prefix"),
        pytest.param([_entry(0, ref="dcid:")], id="ref-of-the-prefix-alone"),
        pytest.param([_entry(0), _entry(0)], id="repeated-stat-var"),
        pytest.param([_entry(0, ref="dcid:StatVar0")], id="side-names-its-own-entry"),
        pytest.param([_entry(0, ref="dcid:StatVar1"), _entry(1)], id="side-names-a-later-entry"),
    ],
)
def test_a_list_the_door_would_refuse_leaves_the_member_out(
    entries: list[ComposedStatVar], caplog: pytest.LogCaptureFixture
) -> None:
    """The body is then the one a frame naming no composed StatVar publishes,
    and only a list that breaks a bound is logged, without any ref it holds."""
    record = RECORDS[1]
    case = {**record, "composed_stat_vars": [entry.model_dump() for entry in entries]}

    with caplog.at_level("INFO", logger=frame_publish.__name__):
        body = _body(case)

    assert body == _httpx_body(record["wire"])
    assert len(caplog.records) == bool(entries)
    for logged in caplog.records:
        assert logged.args[:2] == (_SPEC_ID, _CONTENT_VERSION)
        assert not any(ref in logged.getMessage() for entry in entries for ref in entry.first)


def _wire(body: bytes) -> dict[str, Any]:
    document = json.loads(body)
    return {
        key: document[key] for key in ("role_keys", "rows", "composed_stat_vars") if key in document
    }


def _record() -> None:
    lines = [
        json.dumps({**case, "wire": _wire(_body(case))}, separators=(",", ":")) + "\n"
        for case in generate_cases()
    ]
    _FIXTURE_PATH.write_text("".join(lines))


if __name__ == "__main__":
    _record()
