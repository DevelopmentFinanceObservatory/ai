"""Verbs over a frame whose every `unit` cell is one converted unit (a
`UnitRequest`) or one composed unit (a `ComposedUnit`): each tool returns a
result, its digest names the unit in text (`get_data`'s price-basis words,
or "<numerator> per <denominator>"), its stored trace carries the unit's
object, and its output frame carries the unit where it carries a ref."""

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
from analysis_mcp.quotient_of_tool import quotient_of
from analysis_mcp.rank_tool import rank
from analysis_mcp.sum_of_tool import sum_of
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, InMemoryMaterialisationStore, set_store
from dfo_frame_grammar import Frame, unit_cell_to_json
from fastmcp.tools import ToolResult

REF = "dcid:USDollar"
# Each unit object a frame here holds in every row, with the text a digest
# names it by.
UNITS = {
    "converted": (
        {"currency": "EUR", "prices": "constant", "base_year": 2020},
        "converted to constant 2020 prices, euros",
    ),
    "ref_per_stat_var": (
        {"numerator": "USDollar", "denominator": {"stat_var": "dcid:Count_Person"}},
        "USDollar per Count_Person",
    ),
    "converted_per_stat_var": (
        {
            "numerator": {"currency": "EUR", "prices": "constant", "base_year": 2020},
            "denominator": {"stat_var": "dcid:Count_Person"},
        },
        "constant 2020 prices, euros per Count_Person",
    ),
}
STAT_VAR = "dcid:ConvertedStatVar"
OTHER_STAT_VAR = "dcid:OtherConvertedStatVar"


@pytest.fixture
async def store():
    store = InMemoryMaterialisationStore()
    set_store(store)
    yield store
    set_store(None)


@pytest.fixture(params=UNITS)
def unit(request: pytest.FixtureRequest) -> tuple[Any, str]:
    """A unit object and its text."""
    return UNITS[request.param]


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


async def _seed(
    store: InMemoryMaterialisationStore, unit: object, *, stat_var: str = STAT_VAR
) -> dict[str, str]:
    """A two-donor, two-year frame whose every row carries `unit`, the way
    `tests/differential/cases.py` builds a converted case."""
    frame = Frame.build(
        role_keys=["donor"],
        rows=[
            {
                "stat_var": stat_var,
                "donor": donor,
                "time_period": year,
                "value": value,
                "unit": unit,
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


async def test_aggregate_names_the_unit(store, unit) -> None:
    cell, label = unit
    result = await aggregate(
        frame_ref=await _seed(store, cell), aggregation="mean", group_by=("donor",)
    )

    trace = await _stored_trace_json(store, result)
    assert f"unit: {label}\n" in _digest(result)
    assert trace["applied_policy"]["resolved_unit"] == cell


async def test_rank_collapse_names_the_unit(store, unit) -> None:
    cell, label = unit
    result = await rank(frame_ref=await _seed(store, cell), rank_by=("donor",), aggregation="mean")

    trace = await _stored_trace_json(store, result)
    assert f"unit: {label}\n" in _digest(result)
    assert trace["applied_policy"]["resolved_unit"] == cell


async def test_growth_absolute_names_the_unit(store, unit) -> None:
    cell, label = unit
    result = await growth(
        frame_ref=await _seed(store, cell),
        window_start="2020",
        window_end="2021",
        entity_by=("donor",),
        change="absolute",
    )

    trace = await _stored_trace_json(store, result)
    assert f"unit: absolute change ({label})" in _digest(result)
    assert trace["applied_policy"]["resolved_unit"] == cell


async def test_compare_names_both_units(store, unit) -> None:
    cell, label = unit
    result = await compare(
        frame_ref=await _seed(store, cell),
        left_entity={"donor": "dcid:DonorA"},
        right_entity={"donor": "dcid:DonorB"},
    )

    trace = await _stored_trace_json(store, result)
    assert f"left unit: {label}, right unit: {label}" in _digest(result)
    assert trace["applied_policy"]["resolved_unit"] == cell
    assert trace["applied_policy"]["detail"]["right_resolved_unit"] == cell


async def test_join_names_both_units(store, unit) -> None:
    cell, label = unit
    result = await join(
        frame_ref=await _seed(store, cell),
        right_frame_ref=await _seed(store, cell, stat_var=OTHER_STAT_VAR),
    )

    trace = await _stored_trace_json(store, result)
    assert f"left unit: {label}, right unit: {label}" in _digest(result)
    assert trace["applied_policy"]["resolved_unit"] == cell
    assert trace["applied_policy"]["detail"]["right_resolved_unit"] == cell


async def test_filter_names_the_unit_on_each_kept_row(store, unit) -> None:
    cell, label = unit
    result = await filter_verb(
        frame_ref=await _seed(store, cell),
        predicates=({"kind": "ordered_threshold", "column": "value", "op": "gte", "value": 20},),
    )

    await _stored_trace_json(store, result)
    kept_rows = [line for line in _digest(result).splitlines() if line.startswith("  stat_var=")]
    assert len(kept_rows) == 2
    assert all(line.endswith(f"[{label}]") for line in kept_rows)


async def test_sum_of_names_the_unit_and_writes_it(store) -> None:
    cell, label = UNITS["converted"]
    result = await sum_of(
        frame_ref=await _seed(store, cell),
        left_entity={"donor": "dcid:DonorA"},
        right_entity={"donor": "dcid:DonorB"},
    )

    trace = await _stored_trace_json(store, result)
    assert label in _digest(result)
    assert trace["applied_policy"]["resolved_unit"] == cell
    assert set(map(str, await _output_units(store, result))) == {str(cell)}


_SUMS = {
    "aggregate": lambda ref: aggregate(frame_ref=ref, aggregation="sum", group_by=("donor",)),
    "rank": lambda ref: rank(frame_ref=ref, rank_by=("donor",), aggregation="sum"),
    "sum_of": lambda ref: sum_of(
        frame_ref=ref, left_entity={"donor": "dcid:DonorA"}, right_entity={"donor": "dcid:DonorB"}
    ),
}


@pytest.mark.parametrize("verb", _SUMS)
@pytest.mark.parametrize("name", ["ref_per_stat_var", "converted_per_stat_var"])
async def test_a_sum_over_a_composed_unit_refuses_and_stores_nothing(
    store, verb: str, name: str
) -> None:
    cell, _label = UNITS[name]
    ref = await _seed(store, cell)
    held = len(store._entries)

    result = await _SUMS[verb](ref)

    assert result.is_error is True
    assert result.structured_content["code"] == "composed_unit_sum_unsupported"
    assert result.structured_content["fix"]["verb"] == verb
    assert result.structured_content["fix"]["unit"] == cell
    assert len(store._entries) == held


@pytest.mark.parametrize("verb", _SUMS)
async def test_a_sum_over_a_plain_or_converted_unit_computes(store, verb: str) -> None:
    for cell in (REF, UNITS["converted"][0]):
        result = await _SUMS[verb](await _seed(store, cell))

        assert result.is_error is False
        assert (await _stored_trace_json(store, result))["applied_policy"]["resolved_unit"] == cell


_SIDES = {"left_entity": {"donor": "dcid:DonorA"}, "right_entity": {"donor": "dcid:DonorB"}}
_WINDOW = {"window_start": "2020", "window_end": "2021", "entity_by": ("donor",)}

# Each verb that reads a composed unit as a call over one frame, `join` over
# that frame and a second. `sum_of` refuses one.
CALLS = {
    "aggregate": lambda ref, _other: aggregate(
        frame_ref=ref, aggregation="mean", group_by=("donor",)
    ),
    "rank": lambda ref, _other: rank(frame_ref=ref, rank_by=("donor",), aggregation="mean"),
    "growth_relative": lambda ref, _other: growth(frame_ref=ref, **_WINDOW),
    "growth_absolute": lambda ref, _other: growth(frame_ref=ref, change="absolute", **_WINDOW),
    "compare": lambda ref, _other: compare(frame_ref=ref, **_SIDES),
    "join": lambda ref, other: join(frame_ref=ref, right_frame_ref=other),
    "filter": lambda ref, _other: filter_verb(
        frame_ref=ref,
        predicates=({"kind": "ordered_threshold", "column": "value", "op": "gte", "value": 20},),
    ),
}


async def _run(verb: str, store: InMemoryMaterialisationStore, unit: object) -> ToolResult:
    return await CALLS[verb](
        await _seed(store, unit), await _seed(store, unit, stat_var=OTHER_STAT_VAR)
    )


async def _output_units(store: InMemoryMaterialisationStore, result: ToolResult) -> list[object]:
    """The `unit` cell of each row of the result's stored frame, as JSON."""
    assert result.is_error is False, result.content
    held = await store.get(
        FrameRef(
            spec_id=result.structured_content["spec_id"],
            content_version=result.structured_content["content_version"],
        )
    )
    assert held is not None
    frame, _trace = held
    column = frame.columns.index("unit")
    return [unit_cell_to_json(row[column]) for row in frame.rows]


@pytest.mark.parametrize("verb", CALLS)
async def test_an_output_frame_carries_a_unit_object_where_it_carries_a_ref(
    verb: str, store, unit
) -> None:
    """Each verb writes into its output the unit it writes for a ref-unit
    input: the input's unit, or none where the verb's result has none."""
    cell, _label = unit

    over_ref = await _output_units(store, await _run(verb, store, REF))
    over_object = await _output_units(store, await _run(verb, store, cell))

    assert over_ref
    assert over_object == [cell if unit == REF else unit for unit in over_ref]


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected", "label"),
    [
        pytest.param(
            "converted",
            None,
            {
                "numerator": UNITS["converted"][0],
                "denominator": {"stat_var": OTHER_STAT_VAR},
            },
            "constant 2020 prices, euros per OtherConvertedStatVar",
            id="converted-over-unit-less",
        ),
        pytest.param(
            "converted",
            REF,
            {"numerator": UNITS["converted"][0], "denominator": REF},
            "constant 2020 prices, euros per dcid:USDollar",
            id="converted-over-ref",
        ),
    ],
)
async def test_quotient_of_writes_the_converted_unit_as_its_numerator_part(
    store, numerator: str, denominator: str | None, expected: object, label: str
) -> None:
    result = await quotient_of(
        frame_ref=await _seed(store, UNITS[numerator][0]),
        denominator_frame_ref=await _seed(store, denominator, stat_var=OTHER_STAT_VAR),
    )

    trace = await _stored_trace_json(store, result)
    assert f"in {label}\n" in _digest(result)
    assert trace["applied_policy"]["resolved_unit"] == expected
    assert await _output_units(store, result) == [expected, expected]


@pytest.mark.parametrize("name", ["ref_per_stat_var", "converted_per_stat_var"])
async def test_quotient_of_refuses_a_composed_side(store, name: str) -> None:
    cell, _label = UNITS[name]

    result = await quotient_of(
        frame_ref=await _seed(store, cell),
        denominator_frame_ref=await _seed(store, None, stat_var=OTHER_STAT_VAR),
    )

    assert result.is_error is True
    assert result.structured_content["code"] == "quotient_of_side_unit_unsupported"
    assert result.structured_content["fix"]["reason"] == "composed"
    assert result.structured_content["fix"]["unit"] == cell
