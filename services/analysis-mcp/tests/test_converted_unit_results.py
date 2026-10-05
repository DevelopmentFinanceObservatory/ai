"""Verbs over a converted-unit frame (every `unit` cell a `UnitRequest`):
each tool returns a result, its digest names the unit in `get_data`'s
price-basis words, and its stored trace carries the unit's request object."""

from __future__ import annotations

import secrets
from typing import Any

import pytest
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.compare_tool import compare
from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.filter_tool import filter_verb
from analysis_mcp.growth_tool import growth
from analysis_mcp.identity import op_spec_hash_for
from analysis_mcp.join_tool import join
from analysis_mcp.op_spec import AggregateOpSpec
from analysis_mcp.rank_tool import rank
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, InMemoryMaterialisationStore, set_store
from dfo_frame_grammar import Frame
from fastmcp.tools import ToolResult

REQUEST = {"currency": "EUR", "prices": "constant", "base_year": 2020}
LABEL = "converted to constant 2020 prices, euros"
STAT_VAR = "dcid:ConvertedStatVar"
OTHER_STAT_VAR = "dcid:OtherConvertedStatVar"


@pytest.fixture
async def store():
    store = InMemoryMaterialisationStore()
    set_store(store)
    yield store
    set_store(None)


def _minimal_trace() -> Trace:
    op_spec = AggregateOpSpec(
        frame_ref=FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32),
        group_by=(),
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


async def _seed(store: InMemoryMaterialisationStore, *, stat_var: str = STAT_VAR) -> dict[str, str]:
    """A two-donor, two-year frame whose every row carries `REQUEST` as its
    unit, the way `tests/differential/cases.py` builds a converted case."""
    frame = Frame.build(
        role_keys=["donor"],
        rows=[
            {
                "stat_var": stat_var,
                "donor": donor,
                "time_period": year,
                "value": value,
                "unit": REQUEST,
            }
            for donor, base in (("dcid:DonorA", 10.0), ("dcid:DonorB", 20.0))
            for year, value in (("2020", base), ("2021", base * 1.5))
        ],
    )
    token = secrets.token_hex(16)
    ref = FrameRef(spec_id=f"spec_{token}", content_version=f"cv_{token}")
    await store.put(ref, frame=frame, trace=_minimal_trace(), writer="test")
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


async def _stored_trace_json(store: InMemoryMaterialisationStore, result: ToolResult) -> Any:
    """The result's stored trace, in the JSON form the durable store writes."""
    assert result.is_error is False, result.content
    ref = FrameRef(
        spec_id=result.structured_content["spec_id"],
        content_version=result.structured_content["content_version"],
    )
    held = await store.get(ref)
    assert held is not None
    _frame, trace = held
    assert isinstance(trace, Trace)
    return trace.model_dump(mode="json")


def _digest(result: ToolResult) -> str:
    return result.content[0].text


async def test_aggregate_names_the_converted_unit(store) -> None:
    result = await aggregate(frame_ref=await _seed(store), aggregation="sum", group_by=("donor",))

    trace = await _stored_trace_json(store, result)
    assert f"unit: {LABEL}\n" in _digest(result)
    assert trace["applied_policy"]["resolved_unit"] == REQUEST


async def test_rank_collapse_names_the_converted_unit(store) -> None:
    result = await rank(frame_ref=await _seed(store), rank_by=("donor",), aggregation="sum")

    trace = await _stored_trace_json(store, result)
    assert f"unit: {LABEL}\n" in _digest(result)
    assert trace["applied_policy"]["resolved_unit"] == REQUEST


async def test_growth_absolute_names_the_converted_unit(store) -> None:
    result = await growth(
        frame_ref=await _seed(store),
        window_start="2020",
        window_end="2021",
        entity_by=("donor",),
        change="absolute",
    )

    trace = await _stored_trace_json(store, result)
    assert f"unit: absolute change ({LABEL})" in _digest(result)
    assert trace["applied_policy"]["resolved_unit"] == REQUEST


async def test_compare_names_both_converted_units(store) -> None:
    result = await compare(
        frame_ref=await _seed(store),
        left_entity={"donor": "dcid:DonorA"},
        right_entity={"donor": "dcid:DonorB"},
    )

    trace = await _stored_trace_json(store, result)
    assert f"left unit: {LABEL}, right unit: {LABEL}" in _digest(result)
    assert trace["applied_policy"]["resolved_unit"] == REQUEST
    assert trace["applied_policy"]["detail"]["right_resolved_unit"] == REQUEST


async def test_join_names_both_converted_units(store) -> None:
    result = await join(
        frame_ref=await _seed(store), right_frame_ref=await _seed(store, stat_var=OTHER_STAT_VAR)
    )

    trace = await _stored_trace_json(store, result)
    assert f"left unit: {LABEL}, right unit: {LABEL}" in _digest(result)
    assert trace["applied_policy"]["resolved_unit"] == REQUEST
    assert trace["applied_policy"]["detail"]["right_resolved_unit"] == REQUEST


async def test_filter_names_the_converted_unit_on_each_kept_row(store) -> None:
    result = await filter_verb(
        frame_ref=await _seed(store),
        predicates=({"kind": "ordered_threshold", "column": "value", "op": "gte", "value": 20},),
    )

    await _stored_trace_json(store, result)
    kept_rows = [line for line in _digest(result).splitlines() if line.startswith("  stat_var=")]
    assert len(kept_rows) == 2
    assert all(line.endswith(f"[{LABEL}]") for line in kept_rows)
