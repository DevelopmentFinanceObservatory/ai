"""A converted unit as a verb's unit pin, called through the MCP server.

Each frame here mixes two converted units, so every verb refuses it
unpinned as `unit_heterogeneous`. The refusal's `fix["units"]` lists each
unit in the form the pin parameter accepts, and passing one back verbatim
resolves the call in one further call (issue #276).
"""

from __future__ import annotations

import secrets
from typing import Any

import pytest
from analysis_mcp import server as server_module
from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.identity import op_spec_hash_for
from analysis_mcp.op_spec import AggregateOpSpec
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, InMemoryMaterialisationStore
from dfo_frame_grammar import Frame
from fastmcp import Client

EUR_CONSTANT = {"currency": "EUR", "prices": "constant", "base_year": 2020}
USD_CURRENT = {"currency": "USD", "prices": "current"}
STAT_VAR = "dcid:ConvertedStatVar"
OTHER_STAT_VAR = "dcid:OtherConvertedStatVar"


def _minimal_trace() -> Trace:
    op_spec = AggregateOpSpec(
        frame_ref=FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32),
        aggregation="sum",
    )
    return Trace(
        upstream_trace={
            "graph_revision": "rev_test",
            "source_revisions": [{"source": "test", "revision": "1"}],
            "facets": [],
            "conversion": None,
        },
        frame_content_version="cv_" + "d" * 32,
        op_spec_hash=op_spec_hash_for(op_spec),
        execution_record=ExecutionRecord(content="{}"),
        applied_policy=AppliedPolicy(),
        content_version="ocv_" + "c" * 32,
        input_content_versions=["cv_" + "b" * 32],
        execution_identity=current_execution_identity(),
    )


async def _seed_mixed(
    store: InMemoryMaterialisationStore, *, stat_var: str = STAT_VAR
) -> dict[str, str]:
    """A two-donor, two-year frame holding every row once in each of
    `EUR_CONSTANT` and `USD_CURRENT`, the shape `get_data` returns when a
    definition's members convert into different units."""
    frame = Frame.build(
        role_keys=["donor"],
        rows=[
            {
                "stat_var": stat_var,
                "donor": donor,
                "time_period": year,
                "value": value * scale,
                "unit": unit,
            }
            for unit, scale in ((EUR_CONSTANT, 1.0), (USD_CURRENT, 2.0))
            for donor, base in (("dcid:DonorA", 10.0), ("dcid:DonorB", 20.0))
            for year, value in (("2020", base), ("2021", base * 1.5))
        ],
    )
    token = secrets.token_hex(16)
    ref = FrameRef(spec_id=f"spec_{token}", content_version=f"cv_{token}")
    await store.put(ref, frame=frame, trace=_minimal_trace(), writer="test")
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


@pytest.fixture
async def store(monkeypatch: pytest.MonkeyPatch) -> InMemoryMaterialisationStore:
    store = InMemoryMaterialisationStore()

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)
    return store


async def _call(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async with Client(server_module.mcp) as client:
        result = await client.call_tool(tool, arguments, raise_on_error=False)
    assert result.structured_content is not None
    return result.structured_content


async def _arguments(verb: str, store: InMemoryMaterialisationStore) -> dict[str, Any]:
    """Arguments for `verb` over mixed frames, without any unit pin."""
    frame_ref = await _seed_mixed(store)
    if verb == "aggregate":
        return {"frame_ref": frame_ref, "aggregation": "sum", "group_by": ["donor"]}
    if verb == "rank":
        return {"frame_ref": frame_ref, "rank_by": ["donor"], "aggregation": "sum"}
    if verb == "growth":
        return {
            "frame_ref": frame_ref,
            "window_start": "2020",
            "window_end": "2021",
            "entity_by": ["donor"],
        }
    if verb == "compare":
        return {
            "frame_ref": frame_ref,
            "left_entity": {"donor": "dcid:DonorA"},
            "right_entity": {"donor": "dcid:DonorB"},
        }
    if verb == "share_of":
        return {"frame_ref": frame_ref, "numerator_entity": {"donor": "dcid:DonorA"}}
    return {
        "frame_ref": frame_ref,
        "right_frame_ref": await _seed_mixed(store, stat_var=OTHER_STAT_VAR),
    }


# Each verb's unit-pin parameters, as its `unit_heterogeneous` refusal names them.
PIN_PARAMETERS = {
    "aggregate": ("unit",),
    "rank": ("unit",),
    "growth": ("unit",),
    "compare": ("left_unit", "right_unit"),
    "share_of": ("numerator_unit", "denominator_unit"),
    "join": ("left_unit", "right_unit"),
}


@pytest.mark.parametrize("verb", PIN_PARAMETERS)
async def test_a_mixed_converted_frame_resolves_by_pinning_a_refusal_candidate(
    verb: str, store: InMemoryMaterialisationStore
) -> None:
    arguments = await _arguments(verb, store)

    refusal = await _call(verb, arguments)

    assert refusal["code"] == "unit_heterogeneous"
    assert refusal["fix"]["units"] == [EUR_CONSTANT, USD_CURRENT]

    await _assert_pin_resolves(
        verb, arguments, refusal["fix"]["units"][1], store=store, expected=USD_CURRENT
    )


@pytest.mark.parametrize("verb", PIN_PARAMETERS)
async def test_a_refusal_candidate_id_passed_back_as_the_pin_resolves_the_frame(
    verb: str, store: InMemoryMaterialisationStore
) -> None:
    arguments = await _arguments(verb, store)

    refusal = await _call(verb, arguments)

    assert refusal["code"] == "unit_heterogeneous"
    candidate = refusal["fix"]["disambiguation"]["candidates"][0]
    assert candidate["id"] == EUR_CONSTANT
    await _assert_pin_resolves(verb, arguments, candidate["id"], store=store, expected=EUR_CONSTANT)


async def _assert_pin_resolves(
    verb: str,
    arguments: dict[str, Any],
    pin: object,
    *,
    store: InMemoryMaterialisationStore,
    expected: dict[str, object],
) -> None:
    """Retry `verb` with every unit-pin parameter set to `pin`, and check the
    call succeeds and its stored trace resolved to `expected`."""
    result = await _call(verb, {**arguments, **dict.fromkeys(PIN_PARAMETERS[verb], pin)})

    assert "code" not in result, result
    held = await store.get(
        FrameRef(spec_id=result["spec_id"], content_version=result["content_version"])
    )
    assert held is not None
    _frame, trace = held
    assert isinstance(trace, Trace)
    assert trace.model_dump(mode="json")["applied_policy"]["resolved_unit"] == expected


async def test_a_converted_pin_absent_from_the_frame_refuses_with_pinnable_candidates(
    store: InMemoryMaterialisationStore,
) -> None:
    absent = {"currency": "GBP", "prices": "current"}

    refusal = await _call("aggregate", {**await _arguments("aggregate", store), "unit": absent})

    assert refusal["code"] == "unit_heterogeneous"
    assert refusal["fix"]["pinned_unit"] == absent
    assert refusal["fix"]["units"] == [EUR_CONSTANT, USD_CURRENT]
    assert refusal["message"] == (
        "unit='converted to current prices, pounds sterling' is not one of the units this "
        "frame declares: ['converted to constant 2020 prices, euros', 'converted to current "
        "prices, US dollars']. Retry with `unit` set to one of these values: "
        '{"currency": "EUR", "prices": "constant", "base_year": 2020}, '
        '{"currency": "USD", "prices": "current"}.'
    )


async def test_a_malformed_converted_pin_is_refused_before_any_engine_runs(
    store: InMemoryMaterialisationStore,
) -> None:
    arguments = await _arguments("aggregate", store)

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            "aggregate",
            {**arguments, "unit": {"currency": "EUR", "prices": "constant"}},
            raise_on_error=False,
        )

    assert result.is_error is True
