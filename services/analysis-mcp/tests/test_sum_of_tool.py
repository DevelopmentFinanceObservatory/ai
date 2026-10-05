"""The `sum_of` tool end to end: the two-dataset sum through the registered
tool, the refusals it shares with `compare`, and replay from the stored
execution record.
"""

from __future__ import annotations

import json
import secrets

import pytest
from analysis_mcp import server as server_module
from analysis_mcp.compare_tool import compare
from analysis_mcp.engine.sum_of import run_sum_of
from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.fixtures import seed_fixture_store
from analysis_mcp.identity import op_spec_hash_for, recompute_op_spec_hash
from analysis_mcp.op_spec import AggregateOpSpec
from analysis_mcp.sum_of_tool import sum_of
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, InMemoryMaterialisationStore, set_store
from dfo_frame_grammar import Frame
from fastmcp import Client


def _format(value: float) -> str:
    """`value` as the digest writes a whole number."""
    return str(int(value)) if value == int(value) else repr(value)


@pytest.fixture
async def seeded():
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)
    set_store(store)
    yield store, handles
    set_store(None)


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


async def _seed(store: InMemoryMaterialisationStore, rows: list[dict[str, object]]) -> dict:
    frame = Frame.build(role_keys=["reporter"], rows=rows)
    digest = secrets.token_hex(16)
    ref = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    await store.put(ref, frame=frame, trace=_minimal_trace(), writer="test")
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


def _row(*, year: str = "2020", value: float = 1.0, **extra: object) -> dict[str, object]:
    return {
        "stat_var": "dcid:X",
        "reporter": "dcid:USA",
        "time_period": year,
        "value": value,
        **extra,
    }


async def test_two_datasets_sum_through_the_registered_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """France's bilateral (2010-2024, reported) plus imputed multilateral
    (2011-2023, modelled) aid to Guatemala, through the live FastMCP
    `Client`. The result is one series over 2011-2023 under the operation's
    own ref, with both coverages, the left-out years, and the facet warnings
    on the digest."""
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)
    bilateral, imputed = handles.sum_pair

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            "sum_of", {"frame_ref": bilateral, "right_frame_ref": imputed}
        )

    assert result.is_error is False
    structured = result.structured_content
    assert structured["rows"] == 13
    assert structured["role_keys"] == []
    assert structured["stat_var_count"] == 1
    digest = result.content[0].text
    assert f"stat_var dcid:{structured['spec_id']}: sum of " in digest
    assert "sum over 2011 to 2023" in digest
    assert "left offered 2010 to 2024, right 2011 to 2023" in digest
    assert "coverage left 86.7%, right 100.0%" in digest
    # Each year only the bilateral side reports is named with that side's own
    # value, as its stored frame holds it.
    left_frame, _ = await store.get(FrameRef(**bilateral))
    columns = left_frame.table.to_pydict()
    reported = dict(zip(columns["time_period"], columns["value"], strict=True))
    assert (
        "2 period(s) left out of the sum:\n"
        "  2010: not in the sum; only the left side (dcid:OdaBilateralGrossDisbursement) "
        f"reports it: {_format(reported['2010'])}\n"
        "  2024: not in the sum; only the left side (dcid:OdaBilateralGrossDisbursement) "
        f"reports it: {_format(reported['2024'])}"
    ) in digest
    # Ahead of the summed periods, so a reader keeping only the digest's start
    # (the Studio hand-back keeps 800 characters of each step) still has it.
    assert digest.index("left out of the sum") < digest.index("summed period(s)")
    assert "2011: 4250000 + 1500000 = 5750000" in digest
    assert "facet 'provenance' differs between the sides" in digest
    assert "facet 'measurementMethod' differs between the sides" in digest

    stored_frame, stored_trace = await store.get(
        FrameRef(spec_id=structured["spec_id"], content_version=structured["content_version"])
    )
    assert set(stored_frame.table.column("stat_var").to_pylist()) == {
        f"dcid:{structured['spec_id']}"
    }
    assert sorted(stored_trace.input_content_versions) == sorted(
        [bilateral["content_version"], imputed["content_version"]]
    )
    assert stored_trace.applied_policy.detail["left_out"] == [
        {"time_period": year, "side": "left", "value": reported[year]} for year in ("2010", "2024")
    ]
    assert stored_trace.applied_policy.detail["left_coverage_fraction"] == 13 / 15


async def test_replay_from_execution_record_reproduces_the_stored_frame(seeded) -> None:
    store, handles = seeded
    bilateral, imputed = handles.sum_pair

    result = await sum_of(frame_ref=bilateral, right_frame_ref=imputed)
    stored_frame, stored_trace = await store.get(
        FrameRef(
            spec_id=result.structured_content["spec_id"],
            content_version=result.structured_content["content_version"],
        )
    )

    canonical_bytes = stored_trace.execution_record.content.encode()
    assert recompute_op_spec_hash(canonical_bytes) == stored_trace.op_spec_hash
    parameters = json.loads(stored_trace.execution_record.content)["parameters"]
    left_frame, _ = await store.get(FrameRef(**bilateral))
    right_frame, _ = await store.get(FrameRef(**imputed))
    replayed = run_sum_of(
        left_frame,
        right_frame,
        stat_var=f"dcid:{stored_trace.op_spec_hash}",
        left_entity=parameters["left_entity"],
        right_entity=parameters["right_entity"],
    )
    assert replayed.frame == stored_frame


async def test_sum_of_warns_on_the_facet_difference_compare_refuses(seeded) -> None:
    _store, handles = seeded
    bilateral, imputed = handles.sum_pair

    summed = await sum_of(frame_ref=bilateral, right_frame_ref=imputed)
    compared = await compare(frame_ref=bilateral, right_frame_ref=imputed)

    assert compared.structured_content["code"] == "cross_frame_facet_mismatch"
    assert summed.is_error is False


@pytest.mark.parametrize(
    ("left_rows", "right_rows", "code", "fix"),
    [
        (
            [_row(year="2020", unit="dcid:USDollar")],
            [_row(year="2023", unit="dcid:USDollar")],
            "comparison_periods_disjoint",
            {"left_periods": ["2020"], "right_periods": ["2023"]},
        ),
        (
            [_row(unit="dcid:USDollar")],
            [_row(unit="dcid:Percent")],
            "cross_frame_unit_mismatch",
            {"left": "dcid:USDollar", "right": "dcid:Percent"},
        ),
        (
            [_row()],
            [_row(unit="dcid:USDollar")],
            "cross_frame_unit_unbound",
            {"unbound_side": "left", "bound_side": "right", "bound_unit": "dcid:USDollar"},
        ),
        (
            [_row(value=1.7e308, unit="dcid:USDollar")],
            [_row(value=1.7e308, unit="dcid:USDollar")],
            "non_finite_result",
            {"group_key": {"time_period": "2020"}, "aggregation": "sum"},
        ),
        (
            [_row(unit="dcid:USDollar"), _row(unit="dcid:USDollar", reporter="dcid:GBR")],
            [_row(unit="dcid:USDollar")],
            "role_ambiguous",
            {"parameter": "left_entity"},
        ),
    ],
    ids=["periods disjoint", "unit mismatch", "unit unbound", "non-finite", "ambiguous slice"],
)
async def test_refusals(
    seeded,
    left_rows: list[dict[str, object]],
    right_rows: list[dict[str, object]],
    code: str,
    fix: dict[str, object],
) -> None:
    store, _handles = seeded
    left = await _seed(store, left_rows)
    right = await _seed(store, right_rows)

    result = await sum_of(frame_ref=left, right_frame_ref=right)

    assert result.is_error is True
    assert result.structured_content["code"] == code
    assert fix.items() <= result.structured_content["fix"].items()


async def test_unknown_entity_column_refuses_role_ambiguous(seeded) -> None:
    _store, handles = seeded
    bilateral, imputed = handles.sum_pair

    result = await sum_of(
        frame_ref=bilateral, right_frame_ref=imputed, left_entity={"reporter": "dcid:FRA"}
    )

    assert result.structured_content["code"] == "role_ambiguous"
    assert result.structured_content["fix"]["available"] == ["donor", "recipient"]
