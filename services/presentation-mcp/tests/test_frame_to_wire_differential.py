"""Record-and-replay for the publish body: every frame in
`fixtures/frame_to_wire_differential.jsonl` goes through
`frame_publish.publish_body`, and the bytes it returns must equal httpx's
`json=` encoding of the publish body holding the recorded `role_keys` and
`rows`: row order, key order within each row, cell types (`1` against
`1.0`), float digits and the unit objects. A case that carries steps
records the `steps` member too.

Cases are seeded (`_SEED`). One draws its units from `_COMPOSED_UNITS`, and
the cases after it carry `_STEPS` lists. Record the fixture again with
`uv run --package presentation_mcp python services/presentation-mcp/tests/test_frame_to_wire_differential.py`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
import rfc8785
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
# Drawn after every other case. The second list chains a quotient onto two
# retrievals and carries labels outside ASCII.
_STEPS: tuple[list[dict[str, Any]], ...] = (
    [
        {
            "id": "s1",
            "op": "retrieve",
            "input": {"id": "spec_" + "1" * 32, "content_version": "cv_" + "2" * 32},
            "options": {"layout": "frame"},
        },
    ],
    [
        {
            "id": "s1",
            "op": "retrieve",
            "input": {
                "id": "spec_" + "1" * 32,
                "content_version": "cv_" + "2" * 32,
                "label": "Dépense de santé",
            },
            "options": {},
        },
        {
            "id": "s2",
            "op": "retrieve",
            "input": {"id": "spec_" + "3" * 32, "content_version": "cv_" + "4" * 32},
            "options": {},
        },
        {
            "id": "s3",
            "op": "quotient",
            "inputs": {"numerator": "s1", "denominator": "s2"},
            "policy": "quotient_of_intersect_periods",
            "params": {"numerator_by": ["recipient"]},
            "kept": {"entities": 38, "periods": 15},
            "left_out": [{"reason": "zero_denominator", "count": 2, "of": "entities"}],
            "unit": {"numerator": {"ref": "USDollar"}, "denominator": {"stat_var": "人口"}},
            "period_coverage": [{"period": "2010", "entities": 32}],
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
            "steps": steps,
        }
        for steps in _STEPS
    )
    return [{"id": f"wire-{index:03d}", **case} for index, case in enumerate(cases)]


def _body(case: dict[str, Any]) -> bytes:
    return publish_body(
        _frame(case["frame"]),
        spec_id=_SPEC_ID,
        content_version=_CONTENT_VERSION,
        input_content_versions=_INPUT_CONTENT_VERSIONS,
        steps=case.get("steps"),
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
    if "steps" in wire:
        document["steps"] = wire["steps"]
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


def _retrieve(index: int, *, note: str = "", scale: object = None) -> dict[str, Any]:
    options: dict[str, Any] = {"note": note}
    if scale is not None:
        options["scale"] = scale
    return {
        "id": f"s{index + 1}",
        "op": "retrieve",
        "input": {"id": "spec_" + "1" * 32, "content_version": "cv_" + "2" * 32},
        "options": options,
    }


def _one_step_of(size: int, *, char: str = "x", scale: object = None) -> list[dict[str, Any]]:
    """One retrieve step that takes `size` bytes as JCS UTF-8 JSON, its note
    padded with `char`."""
    base = len(rfc8785.dumps([_retrieve(0, scale=scale)]))
    padding, remainder = divmod(size - base, len(char.encode()))
    assert remainder == 0
    return [_retrieve(0, note=char * padding, scale=scale)]


def test_the_step_bounds_mirror_the_kit() -> None:
    kit = pytest.importorskip("dfo_descriptions")
    assert (frame_publish._MAX_SERVED_STEPS, frame_publish._MAX_STEPS_BYTES) == (
        kit.MAX_STEPS,
        kit.MAX_STEPS_BYTES,
    )


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param([_retrieve(index) for index in range(62)], id="62-steps"),
        pytest.param(_one_step_of(65_536), id="65536-bytes"),
        # JCS spells 1.0 as `1`, two bytes under Python's `1.0`.
        pytest.param(_one_step_of(65_536, scale=1.0), id="65536-bytes-holding-1.0"),
    ],
)
def test_a_step_list_at_a_door_bound_is_sent(steps: list[dict[str, Any]]) -> None:
    document = json.loads(_body({**RECORDS[1], "steps": steps}))

    assert document["steps"] == steps


@pytest.mark.parametrize(
    "steps",
    [
        # The kit's 64 less the two steps a card drawn from the frame adds.
        pytest.param([_retrieve(index) for index in range(63)], id="63-steps"),
        pytest.param(_one_step_of(65_537), id="65537-bytes"),
        # JCS spells 1e16 as `10000000000000000`, twelve bytes over Python's `1e+16`.
        pytest.param(_one_step_of(65_537, scale=1e16), id="65537-bytes-holding-1e16"),
        # The kit refuses an integer outside ±(2**53 - 1) as `not_i_json`.
        pytest.param([_retrieve(0, scale=2**53)], id="an-integer-past-i-json"),
        # Under 65,536 characters, over 65,536 bytes.
        pytest.param(_one_step_of(65_538, char="é"), id="65538-bytes-of-two-byte-characters"),
    ],
)
def test_a_step_list_over_a_door_bound_refuses_locally(steps: list[dict[str, Any]]) -> None:
    with pytest.raises(frame_publish.PublishBodyNotBuildable) as refused:
        _body({**RECORDS[1], "steps": steps})

    assert refused.value.refuse_locally is True


def _covered(index: int, member: str, entries: int) -> dict[str, Any]:
    """A rank or a share reading step `s<index>`, holding `entries` entries
    of `member` (`entity_coverage` naming provider-recipient pairs, or
    `period_coverage`)."""
    if member == "entity_coverage":
        coverage = [
            {
                "entity": {
                    "ref": f"financingProvider=country/P{entry:04d}, financingRecipient=country/KEN"
                },
                "periods": 15,
                "of": 15,
            }
            for entry in range(entries)
        ]
    else:
        coverage = [{"period": str(entry), "entities": 2} for entry in range(entries)]
    return {
        "id": f"s{index + 1}",
        "op": "rank" if member == "entity_coverage" else "share",
        "input": f"s{index}",
        "params": {"limit": entries},
        "kept": {"entities": entries},
        "left_out": [],
        member: coverage,
    }


@pytest.mark.parametrize(
    ("steps", "shed"),
    [
        # 300 pairs take about 32 KB and 400 about 43 KB: shedding the larger
        # alone fits, so the smaller keeps its coverage.
        pytest.param(
            [
                _retrieve(0),
                _covered(1, "entity_coverage", 300),
                _covered(2, "entity_coverage", 400),
            ],
            {2},
            id="two-multi-role-ranks-shed-the-larger",
        ),
        pytest.param(
            [
                _retrieve(0),
                _covered(1, "entity_coverage", 200),
                _covered(2, "entity_coverage", 300),
            ],
            set(),
            id="under-the-bound-keeps-all-coverage",
        ),
        # A rank's 11 KB of coverage is shed before a share's 15 KB of period
        # coverage, and shedding it alone fits.
        pytest.param(
            [
                _retrieve(0, note="x" * 45_000),
                _covered(1, "period_coverage", 500),
                _covered(2, "entity_coverage", 100),
            ],
            {2},
            id="entity-coverage-before-period-coverage",
        ),
    ],
)
def test_a_step_list_over_the_byte_bound_sheds_the_largest_coverage_first(
    steps: list[dict[str, Any]], shed: set[int]
) -> None:
    document = json.loads(_body({**RECORDS[1], "steps": steps}))

    assert document["steps"] == [
        {
            key: value
            for key, value in step.items()
            if not (index in shed and key.endswith("_coverage"))
        }
        for index, step in enumerate(steps)
    ]


def test_a_step_list_over_the_byte_bound_after_shedding_its_coverage_refuses_locally() -> None:
    steps = [*_one_step_of(65_537), _covered(1, "entity_coverage", 10)]

    with pytest.raises(frame_publish.PublishBodyNotBuildable) as refused:
        _body({**RECORDS[1], "steps": steps})

    assert refused.value.refuse_locally is True


def _wire(body: bytes) -> dict[str, Any]:
    document = json.loads(body)
    return {key: document[key] for key in ("role_keys", "rows", "steps") if key in document}


def _record() -> None:
    lines = [
        json.dumps({**case, "wire": _wire(_body(case))}, separators=(",", ":")) + "\n"
        for case in generate_cases()
    ]
    _FIXTURE_PATH.write_text("".join(lines))


if __name__ == "__main__":
    _record()
