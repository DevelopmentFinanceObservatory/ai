"""A converted frame as dfo-apis serves it, read by data-mcp and pinned by an
analysis-mcp verb.

dfo-apis writes a `get_data` Parquet artifact's `unit` column as text
(`materialise.formats._cell_text` in dfo-apis): a unit ref as its own text, a
converted unit as the JCS text of its `{currency, prices, base_year}` object.
`data_mcp.frame_materialisation._frame_from_parquet_bytes` decodes each cell
into the frame's own unit, so a converted-unit pin on `rank`, `aggregate` or
`share_of` compares with the cell by currency, prices and base year.

The EUR artifact mirrors the frame a France-to-Kenya-and-Senegal gross
disbursements `get_data` returned from dfo-apis: six rows, `unit` the text
`{"currency":"EUR","prices":"current"}`, and the 2024 values 460681.0254
(Kenya) and 153000.0342 (Senegal).
"""

from __future__ import annotations

import io
import secrets
from collections.abc import AsyncIterator, Mapping
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from analysis_mcp import server as analysis_server
from analysis_mcp.trace import Trace
from common.materialisation import (
    FrameRef,
    InMemoryMaterialisationStore,
    MaterialisationTrace,
    set_store,
)
from common.unit_refs import PERCENT_UNIT
from data_mcp.frame_materialisation import _frame_from_parquet_bytes
from dfo_frame_grammar import Frame
from dfo_frame_grammar.grammar import UnitRequest
from fastmcp import Client

EUR_CURRENT = {"currency": "EUR", "prices": "current"}
USD_CONSTANT_2024 = {"currency": "USD", "prices": "constant", "base_year": 2024}

# The text dfo-apis writes for each converted unit: JCS, keys in code-point order.
_UNIT_TEXT = {
    "EUR_CURRENT": '{"currency":"EUR","prices":"current"}',
    "USD_CONSTANT_2024": '{"base_year":2024,"currency":"USD","prices":"constant"}',
}

_STAT_VAR = "dcid:GrossDisbursements_Converted"
_KENYA = "country/KEN"
_SENEGAL = "country/SEN"
_VALUES = {
    (_KENYA, "2022"): 255864.18,
    (_KENYA, "2023"): 783333.35,
    (_KENYA, "2024"): 460681.0254,
    (_SENEGAL, "2022"): 98100.5,
    (_SENEGAL, "2023"): 120400.25,
    (_SENEGAL, "2024"): 153000.0342,
}
_KENYA_TOTAL = 255864.18 + 783333.35 + 460681.0254
_SENEGAL_TOTAL = 98100.5 + 120400.25 + 153000.0342


def _artifact(unit_text: str) -> bytes:
    """A `get_data(place_types=true)` Parquet artifact in dfo-apis's layout: canonical
    column order, every grain column a string, `unit` the text `unit_text` on every
    row, an all-null `status` column, and each party's kind, which a sum across the
    recipients reads."""
    keys = sorted(_VALUES)
    rows = len(keys)
    table = pa.table(
        {
            "stat_var": pa.array([_STAT_VAR] * rows),
            "financingProvider": pa.array(["country/FRA"] * rows),
            "financingRecipient": pa.array([recipient for recipient, _ in keys]),
            "time_period": pa.array([year for _, year in keys]),
            "value": pa.array([_VALUES[key] for key in keys], type=pa.float64()),
            "unit": pa.array([unit_text] * rows, type=pa.string()),
            "measurementMethod": pa.nulls(rows, type=pa.string()),
            "provenance": pa.array(["dcid:dc/base/OECD_CRS"] * rows),
            "status": pa.nulls(rows, type=pa.string()),
            "financingProvider_party_kind": pa.array(["party"] * rows),
            "financingRecipient_party_kind": pa.array(["party"] * rows),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


@pytest.fixture
async def store() -> AsyncIterator[InMemoryMaterialisationStore]:
    store = InMemoryMaterialisationStore()
    set_store(store)
    yield store
    set_store(None)


async def _register(store: InMemoryMaterialisationStore, artifact: bytes) -> dict[str, str]:
    """Decode `artifact` through data-mcp's own reader and hold it the way
    `materialise_and_register` does, under a fresh frame reference."""
    token = secrets.token_hex(16)
    ref = FrameRef(spec_id=f"spec_{token}", content_version=f"cv_{token}")
    trace = MaterialisationTrace(
        upstream_trace={"graph_revision": "g1", "facets": []},
        spec_id=ref.spec_id,
        content_version=ref.content_version,
    )
    await store.put(ref, frame=_frame_from_parquet_bytes(artifact), trace=trace, writer="test")
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


async def _call(tool: str, arguments: Mapping[str, object]) -> dict[str, Any]:
    async with Client(analysis_server.mcp) as client:
        result = await client.call_tool(tool, dict(arguments), raise_on_error=False)
    assert result.structured_content is not None
    return result.structured_content


async def _held(store: InMemoryMaterialisationStore, result: dict[str, Any]) -> tuple[Frame, Any]:
    """The result frame and its stored trace in the JSON form a durable store writes."""
    assert "code" not in result, result
    held = await store.get(
        FrameRef(spec_id=result["spec_id"], content_version=result["content_version"])
    )
    assert held is not None
    frame, trace = held
    assert isinstance(trace, Trace)
    return frame, trace.model_dump(mode="json")


def _verb_arguments(verb: str, frame_ref: dict[str, str], pin: object) -> dict[str, object]:
    if verb == "rank":
        return {
            "frame_ref": frame_ref,
            "rank_by": ["financingRecipient"],
            "aggregation": "sum",
            "unit": pin,
        }
    if verb == "aggregate":
        return {
            "frame_ref": frame_ref,
            "aggregation": "sum",
            "group_by": ["financingRecipient"],
            "unit": pin,
        }
    return {
        "frame_ref": frame_ref,
        "numerator_entity": {"financingRecipient": _KENYA},
        "numerator_unit": pin,
        "denominator_unit": pin,
    }


def _resolved_input_unit(verb: str, trace: dict[str, Any]) -> object:
    """The unit `verb` resolved its input in. A share states its own result in
    per cent and records each side's resolved unit in the policy detail."""
    policy = trace["applied_policy"]
    if verb == "share_of":
        assert policy["resolved_unit"] == PERCENT_UNIT
        return policy["detail"]["numerator_resolved_unit"]
    return policy["resolved_unit"]


def test_a_converted_unit_text_cell_decodes_to_its_request() -> None:
    frame = _frame_from_parquet_bytes(_artifact(_UNIT_TEXT["USD_CONSTANT_2024"]))

    units = {row[frame.columns.index("unit")] for row in frame.rows}
    assert units == {UnitRequest(currency="USD", prices="constant", base_year=2024)}


def test_a_ref_unit_text_cell_decodes_to_the_ref() -> None:
    frame = _frame_from_parquet_bytes(_artifact("CurrencyUSD"))

    units = {row[frame.columns.index("unit")] for row in frame.rows}
    assert units == {"CurrencyUSD"}


@pytest.mark.parametrize("verb", ["rank", "aggregate", "share_of"])
@pytest.mark.parametrize(
    ("unit_key", "pin"),
    [("EUR_CURRENT", EUR_CURRENT), ("USD_CONSTANT_2024", USD_CONSTANT_2024)],
)
async def test_a_converted_pin_resolves_a_converted_frame(
    verb: str, unit_key: str, pin: dict[str, object], store: InMemoryMaterialisationStore
) -> None:
    frame_ref = await _register(store, _artifact(_UNIT_TEXT[unit_key]))

    frame, trace = await _held(store, await _call(verb, _verb_arguments(verb, frame_ref, pin)))

    assert _resolved_input_unit(verb, trace) == pin
    values = frame.table.column("value").to_pylist()
    if verb == "share_of":
        years = frame.table.column("time_period").to_pylist()
        assert dict(zip(years, values, strict=True)) == {
            year: pytest.approx(
                100 * _VALUES[_KENYA, year] / (_VALUES[_KENYA, year] + _VALUES[_SENEGAL, year])
            )
            for year in ("2022", "2023", "2024")
        }
    else:
        recipients = frame.table.column("financingRecipient").to_pylist()
        assert dict(zip(recipients, values, strict=True)) == {
            _KENYA: pytest.approx(_KENYA_TOTAL),
            _SENEGAL: pytest.approx(_SENEGAL_TOTAL),
        }


@pytest.mark.parametrize("verb", ["rank", "aggregate", "share_of"])
async def test_a_mismatched_converted_pin_still_refuses_unit_heterogeneous(
    verb: str, store: InMemoryMaterialisationStore
) -> None:
    frame_ref = await _register(store, _artifact(_UNIT_TEXT["EUR_CURRENT"]))

    refusal = await _call(verb, _verb_arguments(verb, frame_ref, USD_CONSTANT_2024))

    assert refusal["code"] == "unit_heterogeneous"
    assert refusal["fix"]["units"] == [EUR_CURRENT]
    assert (
        "unit='converted to constant 2024 prices, US dollars' is not one of the units this "
        "frame declares: ['converted to current prices, euros']"
    ) in refusal["message"]


@pytest.mark.parametrize("verb", ["rank", "aggregate", "share_of"])
async def test_a_ref_unit_frame_resolves_its_ref_pin(
    verb: str, store: InMemoryMaterialisationStore
) -> None:
    frame_ref = await _register(store, _artifact("CurrencyUSD"))

    _frame, trace = await _held(
        store, await _call(verb, _verb_arguments(verb, frame_ref, "CurrencyUSD"))
    )

    assert _resolved_input_unit(verb, trace) == "CurrencyUSD"


async def test_a_converted_pin_on_a_ref_unit_frame_refuses_unit_heterogeneous(
    store: InMemoryMaterialisationStore,
) -> None:
    frame_ref = await _register(store, _artifact("CurrencyUSD"))

    refusal = await _call("rank", _verb_arguments("rank", frame_ref, EUR_CURRENT))

    assert refusal["code"] == "unit_heterogeneous"
    assert refusal["fix"]["units"] == ["CurrencyUSD"]
