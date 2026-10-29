"""A converted or composed unit as a verb's unit pin, called through the MCP
server.

Each frame here mixes two converted units or two composed units, so every
verb refuses it unpinned as `unit_heterogeneous`. The refusal's
`fix["units"]` lists each unit in the form the pin parameter accepts, and
passing one back verbatim resolves the call in one further call (issue #276).
"""

from __future__ import annotations

from typing import Any

import pytest
from analysis_mcp import server as server_module
from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.identity import op_spec_hash_for
from analysis_mcp.op_spec import AggregateOpSpec
from analysis_mcp.runner_client import ANALYSIS_PATH
from analysis_mcp.summary import VerbSummary
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from analysis_mcp.units import unit_name
from common.materialisation import FrameRef, InMemoryMaterialisationStore
from dfo_frame_grammar import Frame
from dfo_frame_grammar.grammar import unit_cell_from_json
from fastmcp import Client
from runner_replay import ReplayedRunner, frame_digest, request_key

EUR_CONSTANT = {"currency": "EUR", "prices": "constant", "base_year": 2020}
USD_CURRENT = {"currency": "USD", "prices": "current"}
EUR_PER_KM = {"numerator": EUR_CONSTANT, "denominator": "dcid:SquareKilometer"}
USD_PER_PERSON = {"numerator": "dcid:USDollar", "denominator": {"stat_var": "dcid:Count_Person"}}
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


@pytest.fixture(
    params=[(EUR_CONSTANT, USD_CURRENT), (EUR_PER_KM, USD_PER_PERSON)],
    ids=["converted", "composed"],
)
def units(request: pytest.FixtureRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    """The two units a mixed frame holds, in the order a refusal lists them."""
    return request.param


async def _seed_mixed(
    store: InMemoryMaterialisationStore,
    units: tuple[dict[str, Any], dict[str, Any]],
    *,
    stat_var: str = STAT_VAR,
) -> dict[str, str]:
    """A two-donor, two-year frame holding every row once in each of
    `units`, the shape `get_data` returns when a definition's members
    convert into different units."""
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
            for unit, scale in zip(units, (1.0, 2.0), strict=True)
            for donor, base in (("dcid:DonorA", 10.0), ("dcid:DonorB", 20.0))
            for year, value in (("2020", base), ("2021", base * 1.5))
        ],
    )
    token = frame_digest(frame)
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


# The key `_call` keeps a call's digest under, beside its structured content.
_DIGEST = "_digest"


async def _call(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async with Client(server_module.mcp) as client:
        result = await client.call_tool(tool, arguments, raise_on_error=False)
    assert result.structured_content is not None
    return {**result.structured_content, _DIGEST: result.content[0].text}


def digest_of(result: dict[str, Any]) -> str:
    return result[_DIGEST]


def _pair(result: dict[str, Any]) -> dict[str, str]:
    return {"spec_id": result["spec_id"], "content_version": result["content_version"]}


_DONOR_A = {"donor": "dcid:DonorA"}
_DONOR_B = {"donor": "dcid:DonorB"}
# The entity bindings of each verb that reads two slices of one frame.
_ENTITY_ARGUMENTS = {
    "compare": {"left_entity": _DONOR_A, "right_entity": _DONOR_B},
    "sum_of": {"left_entity": _DONOR_A, "right_entity": _DONOR_B},
    "share_of": {"numerator_entity": _DONOR_A},
    "quotient_of": {"numerator_entity": _DONOR_A, "denominator_entity": _DONOR_B},
}


async def _arguments(
    verb: str,
    store: InMemoryMaterialisationStore,
    units: tuple[dict[str, Any], dict[str, Any]] = (EUR_CONSTANT, USD_CURRENT),
) -> dict[str, Any]:
    """Arguments for `verb` over frames mixing `units`, without any unit pin."""
    frame_ref = await _seed_mixed(store, units)
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
    if verb in _ENTITY_ARGUMENTS:
        return {"frame_ref": frame_ref, **_ENTITY_ARGUMENTS[verb]}
    return {
        "frame_ref": frame_ref,
        "right_frame_ref": await _seed_mixed(store, units, stat_var=OTHER_STAT_VAR),
    }


# Each verb's unit-pin parameters, as its `unit_heterogeneous` refusal names them.
# The field of each verb's summary facts that names the unit a pin resolved.
_RESOLVED_UNIT = {
    "aggregate": "resolved_unit",
    "rank": "resolved_unit",
    "growth": "resolved_unit",
    "compare": "left_resolved_unit",
    "sum_of": "unit",
    "share_of": "numerator_resolved_unit",
    "quotient_of": "unit",
    "join": "left_resolved_unit",
}

PIN_PARAMETERS = {
    "aggregate": ("unit",),
    "rank": ("unit",),
    "growth": ("unit",),
    "compare": ("left_unit", "right_unit"),
    "sum_of": ("left_unit", "right_unit"),
    "share_of": ("numerator_unit", "denominator_unit"),
    "quotient_of": ("numerator_unit", "denominator_unit"),
    "join": ("left_unit", "right_unit"),
}


@pytest.mark.parametrize("verb", PIN_PARAMETERS)
async def test_a_mixed_frame_resolves_by_pinning_a_refusal_candidate(
    verb: str,
    store: InMemoryMaterialisationStore,
    units: tuple[dict[str, Any], dict[str, Any]],
    runner: ReplayedRunner,
) -> None:
    arguments = await _arguments(verb, store, units)

    refusal = await _call(verb, arguments)

    assert refusal["code"] == "unit_heterogeneous"
    assert refusal["fix"]["units"] == list(units)

    await _assert_pin_resolves(
        verb, arguments, refusal["fix"]["units"][1], runner=runner, expected=units[1]
    )


@pytest.mark.parametrize("verb", PIN_PARAMETERS)
async def test_a_refusal_candidate_id_passed_back_as_the_pin_resolves_the_frame(
    verb: str,
    store: InMemoryMaterialisationStore,
    units: tuple[dict[str, Any], dict[str, Any]],
    runner: ReplayedRunner,
) -> None:
    arguments = await _arguments(verb, store, units)

    refusal = await _call(verb, arguments)

    assert refusal["code"] == "unit_heterogeneous"
    candidate = refusal["fix"]["disambiguation"]["candidates"][0]
    assert candidate["id"] == units[0]
    await _assert_pin_resolves(verb, arguments, candidate["id"], runner=runner, expected=units[0])


async def _assert_pin_resolves(
    verb: str,
    arguments: dict[str, Any],
    pin: object,
    *,
    runner: ReplayedRunner,
    expected: dict[str, object],
) -> None:
    """Retry `verb` with every unit-pin parameter set to `pin`, and check the
    call succeeds and the runner's summary resolved to `expected`. `quotient_of`
    writes the two pinned sides as one composed unit, and refuses a pin that
    is composed already, and `share_of` resolves to per cent with the pin
    recorded as its numerator's unit. A verb that sums refuses a composed
    pin too:
    `sum_of` on its left side, `aggregate` and `rank` until they take a
    mean, and `share_of` on its two-donor denominator until that side reads
    one donor."""
    arguments = {**arguments, **dict.fromkeys(PIN_PARAMETERS[verb], pin)}
    result = await _call(verb, arguments)

    if "numerator" in expected and verb in ("aggregate", "rank", "sum_of"):
        assert result["code"] == "composed_unit_sum_unsupported"
        assert result["retry_mode"] == "correct_and_retry"
        side = {"side": "left"} if verb == "sum_of" else {}
        assert result["fix"] == {"verb": verb, "unit": expected, **side}
        if verb == "sum_of":
            return
        result = await _call(verb, {**arguments, "aggregation": "mean"})

    if "numerator" in expected and verb == "share_of":
        assert result["code"] == "composed_unit_sum_unsupported"
        assert result["fix"] == {"verb": verb, "unit": expected, "side": "denominator"}
        result = await _call(verb, {**arguments, "denominator_entity": _DONOR_A})

    if verb == "quotient_of":
        if "numerator" in expected:
            assert result["code"] == "quotient_of_side_unit_unsupported"
            assert result["fix"]["reason"] == "composed"
            return
        expected = {"numerator": expected, "denominator": expected}
    assert "code" not in result, result
    # The reply names the frame the runner stored for this call, and the unit
    # its summary resolved, a share's as its numerator's, is the pinned one.
    assert (result["spec_id"], result["content_version"]) in runner.stored.values()
    reply = runner.replies[request_key(ANALYSIS_PATH, runner.requests[-1])]
    summary = VerbSummary.parse(reply["body"]["summary"])
    resolved = getattr(summary.result, _RESOLVED_UNIT[verb])
    assert resolved == unit_cell_from_json(expected)
    assert unit_name(resolved) in digest_of(result)


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


async def test_every_unit_pin_advertises_a_composed_unit_as_a_refusal_s_object() -> None:
    """Each pin's input schema states the converted object's members beside
    the ref string. A composed unit is copied from a refusal, so the schema
    admits an object for it and spells out none of its members."""
    async with Client(server_module.mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    for verb, parameters in PIN_PARAMETERS.items():
        for parameter in parameters:
            forms = tools[verb].input_schema["properties"][parameter]["anyOf"]
            converted, composed = [form for form in forms if form["type"] == "object"]
            assert converted["required"] == ["currency", "prices"]
            assert converted["additionalProperties"] is False
            assert composed.keys() == {"type", "description"}
            assert "unit_heterogeneous" in composed["description"]


async def test_a_malformed_composed_pin_is_refused_before_any_engine_runs(
    store: InMemoryMaterialisationStore,
) -> None:
    arguments = await _arguments("aggregate", store, (EUR_PER_KM, USD_PER_PERSON))

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            "aggregate",
            {**arguments, "unit": {"numerator": "dcid:USDollar", "denominator": EUR_PER_KM}},
            raise_on_error=False,
        )

    assert result.is_error is True


async def test_a_composed_pin_absent_from_the_frame_refuses_with_pinnable_candidates(
    store: InMemoryMaterialisationStore,
) -> None:
    arguments = await _arguments("aggregate", store, (EUR_PER_KM, USD_PER_PERSON))
    absent = {"numerator": "dcid:USDollar", "denominator": "dcid:SquareKilometer"}

    refusal = await _call("aggregate", {**arguments, "unit": absent})

    assert refusal["code"] == "unit_heterogeneous"
    assert refusal["fix"]["pinned_unit"] == absent
    assert refusal["fix"]["units"] == [EUR_PER_KM, USD_PER_PERSON]


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
