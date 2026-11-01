"""`harness.derived_frames`: how the harness reads a frame dfo-apis' runner stored."""

from __future__ import annotations

import io
import logging
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rfc8785
from dfo_frame_grammar import ComposedUnit, StatVarPart, unit_cell_to_json
from harness import orchestrator_tools
from harness.derived_frames import DerivedFile, holds_rank_positions

_ROOT = {"root": 0}


def _step(verb: str, child: dict[str, Any] = _ROOT, **parameters: Any) -> dict[str, Any]:
    return {"op_spec": {"verb": verb, "parameters": parameters}, "inputs": {"frame_ref": child}}


@pytest.mark.parametrize(
    ("tree", "positions"),
    [
        (_step("rank", per_period=True), True),
        (_step("rank", per_period=True, limit=3), False),
        (_step("rank", per_period=False), False),
        (_step("growth", _step("rank", per_period=True), change="absolute"), True),
        (_step("aggregate", group_by=["donor"]), False),
    ],
    ids=["per-period", "limited", "collapsed", "under-a-growth", "no-rank"],
)
def test_a_per_period_rank_with_no_limit_anywhere_in_the_tree_holds_positions(
    tree: dict[str, Any], positions: bool
) -> None:
    assert holds_rank_positions(tree) is positions


_PER_CAPITA = ComposedUnit(numerator="dcid:USDollar", denominator="dcid:Person")
_PER_PERSON_COUNTED = ComposedUnit(
    numerator="dcid:USDollar", denominator=StatVarPart(stat_var="dcid:Count_Person")
)


def _quotient_file(units: list[str]) -> DerivedFile:
    """A `quotient_of` result as the runner writes it: one row per unit, each unit
    cell as text, a composed unit as the JCS text of its object
    (`materialise.formats._cell_text` in dfo-apis)."""
    table = pa.table(
        {
            "stat_var": pa.array(["dcid:Amount_Aid_PerCapita"] * len(units)),
            "observationAbout": pa.array([f"country/C{index}" for index in range(len(units))]),
            "time_period": pa.array(["2022"] * len(units)),
            "value": pa.array([12.5] * len(units)),
            "unit": pa.array(units),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return DerivedFile(data=buf.getvalue(), num_rows=len(units), rank_positions=False)


def test_a_quotient_file_decodes_its_composed_units() -> None:
    texts = [
        rfc8785.dumps(unit_cell_to_json(unit)).decode()
        for unit in (_PER_CAPITA, _PER_PERSON_COUNTED)
    ]

    rows = _quotient_file(texts).rows()

    assert [row["unit"] for row in rows] == [_PER_CAPITA, _PER_PERSON_COUNTED]


async def test_a_derived_file_that_does_not_decode_is_logged_with_its_frame_pair(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """make_charts leaves its select as written for an undecodable frame, and logs
    the failure as an error naming the frame."""
    ref = {"spec_id": "ops_" + "a" * 32, "content_version": "ocv_" + "b" * 32}

    async def malformed(_: Any) -> DerivedFile:
        return _quotient_file(['{"numerator":"dcid:USDollar"}'])

    monkeypatch.setattr(orchestrator_tools, "derived_file", malformed)

    with caplog.at_level(logging.ERROR, logger=orchestrator_tools.logger.name):
        outcome = await orchestrator_tools._derived_frame_work(ref, lambda held: held.rows)

    assert outcome is None
    [record] = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert record.args == (ref["spec_id"], ref["content_version"])
    assert record.exc_info is not None
