"""Record-and-replay for the artifact decode step: every table in
`fixtures/frame_materialisation_differential.jsonl` is written as a Parquet
artifact shaped like the apis service's `get_data` output, decoded by
`frame_materialisation._frame_from_parquet_bytes`, and the outcome must equal
the recorded one: the frame's `frame_sha256`, a `FrameError` category, or the
type of any other exception.

apis writes every column as a string except `value` (float64), and a
column whose cells are all null (`status` always, often `unit` and the facet
axes) as a nullable string column. The fixture adds the other Arrow types the
decoder accepts (`date32`, timestamps, decimals, integers, dictionaries, a unit
struct) and malformed tables it must refuse. Record the fixture again with
`uv run --package data_mcp python services/data-mcp/tests/test_frame_materialisation_differential.py`.
"""

from __future__ import annotations

import datetime
import decimal
import io
import json
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from data_mcp.frame_materialisation import _frame_from_parquet_bytes
from dfo_frame_grammar import FrameError, frame_sha256

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "frame_materialisation_differential.jsonl"
_SEED = 20260930

_UNIT_STRUCT = pa.struct(
    [
        pa.field("currency", pa.string()),
        pa.field("prices", pa.string()),
        pa.field("base_year", pa.int64()),
    ]
)
_TYPES: dict[str, pa.DataType] = {
    "string": pa.string(),
    "large_string": pa.large_string(),
    "dictionary_string": pa.dictionary(pa.int32(), pa.string()),
    "float64": pa.float64(),
    "float32": pa.float32(),
    "int64": pa.int64(),
    "decimal128": pa.decimal128(24, 6),
    "date32": pa.date32(),
    "timestamp_us": pa.timestamp("us"),
    "null": pa.null(),
    "unit_struct": _UNIT_STRUCT,
}

type Column = dict[str, Any]


def _cell(kind: str, value: object) -> object:
    if value is None:
        return None
    if kind == "decimal128":
        return decimal.Decimal(str(value))
    if kind == "date32":
        return datetime.date.fromisoformat(str(value))
    if kind == "timestamp_us":
        return datetime.datetime.fromisoformat(str(value))
    if kind in ("float64", "float32") and isinstance(value, str):
        return float(value)
    return value


def _table(columns: list[Column]) -> pa.Table:
    return pa.table(
        {
            column["name"]: pa.array(
                [_cell(column["type"], value) for value in column["values"]],
                type=_TYPES[column["type"]],
            )
            for column in columns
        }
    )


def _parquet(columns: list[Column]) -> bytes:
    sink = io.BytesIO()
    pq.write_table(_table(columns), sink)
    return sink.getvalue()


def _column(name: str, kind: str, values: list[object]) -> Column:
    return {"name": name, "type": kind, "values": values}


def _dfo_apis_columns(
    rng: random.Random, *, row_count: int, roles: tuple[str, ...], unit: object = "dcid:USDollar"
) -> list[Column]:
    """A `get_data` artifact's columns: stat_var, roles, time_period, value,
    unit, measurementMethod, provenance, status, one row per distinct cell."""
    cells = sorted(
        {
            (
                tuple(f"dcid:Entity{rng.randrange(40):02d}" for _ in roles),
                str(2000 + rng.randrange(25)),
            )
            for _ in range(row_count)
        }
    )
    values = [rng.uniform(-1.0, 1.0) * 10.0 ** rng.randint(-6, 13) for _ in cells]
    return [
        _column("stat_var", "string", ["dcid:SomeStatVar"] * len(cells)),
        *(
            _column(role, "string", [entities[index] for entities, _ in cells])
            for index, role in enumerate(roles)
        ),
        _column("time_period", "string", [period for _, period in cells]),
        _column("value", "float64", values),
        _column("unit", "string", [unit] * len(cells)),
        _column("measurementMethod", "string", [None] * len(cells)),
        _column("provenance", "string", ["dcid:SomeProvenance"] * len(cells)),
        _column("status", "string", [None] * len(cells)),
    ]


def _replace(columns: list[Column], name: str, column: Column | None) -> list[Column]:
    """`columns` with `name` swapped for `column`, or dropped when `None`."""
    return [
        (column if existing["name"] == name else existing)
        for existing in columns
        if existing["name"] != name or column is not None
    ]


def generate_cases() -> list[dict[str, Any]]:
    """Seeded artifact tables, as column descriptions."""
    rng = random.Random(_SEED)
    cases: list[tuple[str, list[Column]]] = []
    for index, roles in enumerate(
        [("observationAbout",), ("donor", "recipient"), (), ("recipient", "donor", "sector")]
    ):
        cases.append((f"dfo-apis-{index}", _dfo_apis_columns(rng, row_count=40, roles=roles)))
    cases.append(
        ("dfo-apis-large", _dfo_apis_columns(rng, row_count=400, roles=("observationAbout",)))
    )
    base = _dfo_apis_columns(rng, row_count=6, roles=("observationAbout",))
    size = len(base[0]["values"])
    periods = base[2]["values"]
    values = base[3]["values"]
    cases += [
        ("empty", [_column(column["name"], column["type"], []) for column in base]),
        ("unit-null", _replace(base, "unit", _column("unit", "string", [None] * size))),
        (
            "unit-mixed",
            _replace(
                base,
                "unit",
                _column(
                    "unit",
                    "string",
                    [("dcid:Percent", None, "dcid:USDollar")[i % 3] for i in range(size)],
                ),
            ),
        ),
        ("status-null-type", _replace(base, "status", _column("status", "null", [None] * size))),
        ("no-status", _replace(base, "status", None)),
        ("no-facets", _replace(_replace(base, "measurementMethod", None), "provenance", None)),
        ("no-unit", _replace(base, "unit", None)),
        (
            "large-string",
            [
                _column(
                    c["name"], "large_string" if c["type"] == "string" else c["type"], c["values"]
                )
                for c in base
            ],
        ),
        (
            "dictionary",
            [
                _column(
                    c["name"],
                    "dictionary_string" if c["type"] == "string" else c["type"],
                    c["values"],
                )
                for c in base
            ],
        ),
        (
            "value-float32",
            _replace(
                base, "value", _column("value", "float32", [0.1, 1 / 3, -2.5, 1e30, 7.0, 0.0])
            ),
        ),
        (
            "value-int64",
            _replace(base, "value", _column("value", "int64", [1, -2, 9007199254740993, 0, 5, 6])),
        ),
        (
            "value-decimal",
            _replace(
                base,
                "value",
                _column(
                    "value",
                    "decimal128",
                    ["0.1", "-12345.678901", "1e-6", "0", "3.3", "99999999999.999999"],
                ),
            ),
        ),
        (
            "value-signed-zero",
            _replace(
                base, "value", _column("value", "float64", [-0.0, 0.0, -0.0, 5e-324, -5e-324, 1.0])
            ),
        ),
        ("value-nan", _replace(base, "value", _column("value", "float64", ["nan", *values[1:]]))),
        ("value-inf", _replace(base, "value", _column("value", "float64", ["-inf", *values[1:]]))),
        ("value-null", _replace(base, "value", _column("value", "float64", [None, *values[1:]]))),
        (
            "period-date32",
            _replace(
                base,
                "time_period",
                _column("time_period", "date32", [f"{p}-06-30" for p in periods]),
            ),
        ),
        (
            "period-timestamp",
            _replace(
                base,
                "time_period",
                _column("time_period", "timestamp_us", [f"{p}-01-02T03:04:05" for p in periods]),
            ),
        ),
        (
            "period-month-day",
            _replace(
                base,
                "time_period",
                _column(
                    "time_period",
                    "string",
                    [f"{p}-0{i % 9 + 1}" if i % 2 else f"{p}-12-31" for i, p in enumerate(periods)],
                ),
            ),
        ),
        (
            "period-invalid",
            _replace(
                base, "time_period", _column("time_period", "string", ["2020-13", *periods[1:]])
            ),
        ),
        ("period-missing", _replace(base, "time_period", None)),
        (
            "stat-var-unprefixed",
            _replace(base, "stat_var", _column("stat_var", "string", ["SomeStatVar"] * size)),
        ),
        (
            "non-nfc",
            _replace(
                base,
                "observationAbout",
                _column("observationAbout", "string", [f"dcid:Café{i}" for i in range(size)]),
            ),
        ),
        (
            "duplicate-grain",
            _replace(
                _replace(
                    base,
                    "observationAbout",
                    _column("observationAbout", "string", ["dcid:Same"] * size),
                ),
                "time_period",
                _column("time_period", "string", ["2020"] * size),
            ),
        ),
        (
            "unit-struct-constant",
            _replace(
                base,
                "unit",
                _column(
                    "unit",
                    "unit_struct",
                    [{"currency": "EUR", "prices": "constant", "base_year": 2020}] * size,
                ),
            ),
        ),
        (
            "unit-struct-current",
            _replace(
                base,
                "unit",
                _column(
                    "unit",
                    "unit_struct",
                    [{"currency": "USD", "prices": "current", "base_year": None}] * size,
                ),
            ),
        ),
        ("unit-struct-null", _replace(base, "unit", _column("unit", "unit_struct", [None] * size))),
    ]
    return [{"id": name, "columns": columns} for name, columns in cases]


def _outcome(columns: list[Column]) -> dict[str, object]:
    try:
        frame = _frame_from_parquet_bytes(_parquet(columns))
    except FrameError as exc:
        return {"frame_error": exc.category}
    except Exception as exc:
        return {"error": type(exc).__name__}
    return {"frame_sha256": frame_sha256(frame), "columns": list(frame.columns)}


def _load() -> list[dict[str, Any]]:
    """The recorded cases; empty before the first recording, which
    `test_generator_reproduces_fixture_inputs` then fails on."""
    if not _FIXTURE_PATH.exists():
        return []
    return [json.loads(line) for line in _FIXTURE_PATH.read_text().splitlines() if line]


RECORDS = _load()


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
def test_decode_matches_recorded_outcome(record: dict[str, Any]) -> None:
    assert _outcome(record["columns"]) == record["outcome"]


def test_generator_reproduces_fixture_inputs() -> None:
    recorded = [{"id": record["id"], "columns": record["columns"]} for record in RECORDS]
    assert json.dumps(generate_cases()) == json.dumps(recorded)


def _record() -> None:
    lines = [
        json.dumps({**case, "outcome": _outcome(case["columns"])}, separators=(",", ":")) + "\n"
        for case in generate_cases()
    ]
    _FIXTURE_PATH.write_text("".join(lines))


if __name__ == "__main__":
    _record()
